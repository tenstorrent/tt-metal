# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The DFlash drafter on device (ttnn), **replicated** across the mesh.

Replaces the host PyTorch drafter: the target's taps never leave the mesh, and the draft logits come
off the target's own resident LM head.

Architecture, per speculative step:

1. :meth:`project_taps` — all-gather the target's fractured residual taps, then a **local** ``fc``
   and ``hidden_norm``. This is the K/V source, shared by all 5 layers. That gather is the drafter's
   only collective.
2. :meth:`forward` — Q comes from the noise block only; K/V span ``[context, noise]``. Four causal
   sliding-window layers (2048) then one **bidirectional** full-attention layer — that last one is
   what lets the 15 masked slots see each other, so it must stay unmasked.
3. The final norm feeds the target's LM head.

Why replicated and not tensor-parallel
--------------------------------------
See :mod:`.weights` for the measurement. Short version: the drafter's arithmetic is trivial (~55
GFLOP for a 16-token block) and a TP version was bound by collectives — 22 CCL ops per step, 45% of
wall clock, 0.70x the host PyTorch drafter. Replicating makes every projection local and leaves one
tap all-gather.

Two further departures from the target, both deliberate:

* **KV history is plain growing tensors, concatenated per step**, not a paged cache. The target's
  paged path can only write a multi-token run at a bucket-aligned offset (see
  :class:`~...reference.dflash.targets.TtTarget`) and speculation advances 1-16 tokens a step, so a
  contiguous history sidesteps the problem entirely. It is cheap at drafter shapes: 8 KV heads x 128
  dim is 2 KB per token per layer, so a 4096-token context is 8 MB per layer per tensor.
* **RoPE tables are resident on device** and sliced per step, rather than recomputed on host.
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt.attention.rope_tp import apply_partial_rope_prefill
from models.demos.blackhole.qwen36.tt.dflash.config import DFlashDrafterConfig
from models.demos.blackhole.qwen36.tt.dflash.weights import DrafterWeights, load_drafter_weights

_DRAM = ttnn.DRAM_MEMORY_CONFIG
_L1 = ttnn.L1_MEMORY_CONFIG

#: Additive-mask "invisible" value for the fixed-capacity path. NOT -inf, deliberately.
#:
#: The growing-history path uses -inf safely because its masks are causal over a short span, so no
#: 32-column tile is ever entirely masked. The fixed-capacity path masks the whole unwritten tail of
#: a C-row buffer, which hands the kernel many ALL-MASKED blocks; a flash-style softmax takes a
#: per-block max, and for such a block that max is -inf, so it evaluates exp(-inf - -inf). A large
#: finite sentinel makes the block max finite and the weights underflow to zero instead.
_MASK_NEG = -1e9

#: Default depth of the resident RoPE tables. The drafter's positions track the target's, so this
#: caps the sequence a drafter instance can serve; it is host trig once at construction.
DEFAULT_MAX_SEQ_LEN = 4096


class TtDFlashDrafter:
    """Device DFlash drafter, replicated on every device. Borrows the target's embedding + LM head."""

    def __init__(
        self,
        mesh_device,
        cfg: DFlashDrafterConfig,
        state_dict,
        *,
        tt_ccl=None,
        cache_path=None,
        max_seq_len=DEFAULT_MAX_SEQ_LEN,
        ctx_capacity=None,
        **weight_dtypes,
    ):
        """``weight_dtypes`` is passed through to :func:`~.weights.load_drafter_weights`
        (``proj_dtype`` / ``mlp_dtype`` / ``mlp_down_dtype``), so a caller can build two drafters
        that differ only in weight precision. That is what
        ``tests/reference/test_dflash_acceptance.py`` uses to price the bf4 MLP against a bf8 one
        under a single target load; the defaults are the shipped choice.
        """
        self.device = mesh_device
        self.cfg = cfg
        self.tp = mesh_device.get_num_devices()
        self.multi = self.tp > 1
        # Needed only for the tap all-gather; a replicated drafter has no other collective.
        if tt_ccl is None and self.multi:
            from models.tt_transformers.tt.ccl import TT_CCL

            tt_ccl = TT_CCL(mesh_device)
        self.tt_ccl = tt_ccl
        # Replicated: every device runs all heads, not a fracture of them.
        self.nh = cfg.num_attention_heads
        self.nkv = cfg.num_key_value_heads
        self.hd = cfg.head_dim
        self.scale = self.hd**-0.5
        # nlp_create_qkv_heads' tied mode rejects a fused width that also parses as untied, i.e. it
        # needs 2*nkv*hd NOT to divide into (nkv + 2*nkv) sections -- head_dim not a multiple of 3.
        # Asserted at construction so a config change fails here, not mid-step inside _kv_heads.
        assert (2 * self.nkv * self.hd) % (3 * self.nkv) != 0, (
            f"head_dim {self.hd} is a multiple of 3, so the fused [k|v] block is ambiguous to "
            "nlp_create_qkv_heads(kv_tied=True); _kv_heads would need two split projections again"
        )
        self.max_seq_len = max_seq_len
        # Fixed-capacity KV mode (opt-in). None keeps the growing-concat history, which is the
        # reference this mode is validated against; an int C makes every per-step shape constant,
        # which is what a trace capture requires. See _alloc_ctx_buffers.
        assert ctx_capacity is None or (
            ctx_capacity % 32 == 0 and ctx_capacity > 0
        ), f"ctx_capacity must be a positive multiple of the 32-row tile height, got {ctx_capacity}"
        self._cap = ctx_capacity
        self.weights: DrafterWeights = load_drafter_weights(
            mesh_device, state_dict, cfg, cache_path=cache_path, **weight_dtypes
        )
        self.compute_cfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        self._cos, self._sin = self._build_rope_tables()
        # Contiguous per-layer K/V history for the committed context; see the module docstring.
        self._ctx_k: list[ttnn.Tensor | None] = [None] * cfg.num_hidden_layers
        self._ctx_v: list[ttnn.Tensor | None] = [None] * cfg.num_hidden_layers
        self._ctx_len = 0
        # Staged step inputs; None until alloc_step_buffers(). See there.
        self._sb = None
        self._staged_new_ctx = 0
        if self._cap is not None:
            self._alloc_ctx_buffers()
        # Projection matmul program configs, keyed (m_tiles, K, N); see _proj_pc.
        self._mm_pc = {}

    # ---- properties the speculative loop reads ---------------------------------------------

    @property
    def block_size(self) -> int:
        return self.cfg.block_size

    @property
    def mask_token_id(self) -> int:
        return self.cfg.mask_token_id

    @property
    def target_layer_ids(self):
        return list(self.cfg.target_layer_ids)

    @property
    def context_len(self) -> int:
        return self._ctx_len

    # ---- state -----------------------------------------------------------------------------

    def reset(self) -> None:
        """Drop the drafter's KV history and start a new sequence."""
        for store in (self._ctx_k, self._ctx_v):
            for i, t in enumerate(store):
                if t is not None:
                    ttnn.deallocate(t)
                store[i] = None
        self._ctx_len = 0
        if self._cap is not None:
            self._alloc_ctx_buffers()

    def _replicate(self, t, layout=ttnn.TILE_LAYOUT):
        """Upload a host tensor, replicated across the mesh (the drafter runs every head everywhere)."""
        return ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=layout,
            device=self.device,
            memory_config=_DRAM,
            **(dict(mesh_mapper=ttnn.ReplicateTensorToMesh(self.device)) if self.multi else {}),
        )

    def _alloc_ctx_buffers(self) -> None:
        """Allocate the PERSISTENT per-layer K/V history, once, at a fixed [1, nkv, C, hd].

        This is the change that makes the drafter capturable. The default path rebinds
        ``self._ctx_k[i]`` to a freshly concatenated tensor every step, so both its shape and its
        address move; a trace bakes both. Here the buffer is allocated once and only ever written
        THROUGH -- ``ttnn.experimental.slice_write`` into the live tensor -- so the address a capture
        records stays valid for every replay.

        Rows past ``_ctx_len`` are zero and are masked out rather than trusted (see _fixed_masks):
        a zero K row is not a no-op under softmax, it is a key that scores 0 against every query.
        """
        zeros = torch.zeros(1, self.nkv, self._cap, self.hd, dtype=torch.bfloat16)
        for i in range(self.cfg.num_hidden_layers):
            self._ctx_k[i] = self._replicate(zeros)
            self._ctx_v[i] = self._replicate(zeros)

    # ---- staged step inputs (what a capture reads instead of freshly-built tensors) ----------

    def alloc_step_buffers(self, q_len=None, ctx_pad=16):
        """Allocate the PERSISTENT per-step inputs a traced drafter step reads.

        A capture bakes buffer addresses, so every tensor a step consumes has to live at a fixed
        address and be refilled in place before each replay. Today ``forward`` builds its rope
        slices and masks fresh every step (``ttnn.from_torch`` -- a host upload, illegal inside a
        capture) and takes the tap projection as whatever tensor ``project_taps`` just returned.
        These buffers are the fixed-address stand-ins; :meth:`stage_step` and :meth:`stage_taps`
        refill them, both OUTSIDE the capture.

        Requires fixed-capacity KV (``ctx_capacity``), since the mask width is ``C + ctx_pad +
        q_len`` and C has to exist for that to be a constant.
        """
        assert self._cap is not None, "staged step buffers need ctx_capacity; see _alloc_ctx_buffers"
        q_len = self.cfg.block_size if q_len is None else q_len
        kv_len = self._cap + ctx_pad + q_len
        self._sb = {
            "q_len": q_len,
            "ctx_pad": ctx_pad,
            "cos": self._replicate(torch.zeros(1, 1, ctx_pad + q_len, self.hd, dtype=torch.bfloat16)),
            "sin": self._replicate(torch.zeros(1, 1, ctx_pad + q_len, self.hd, dtype=torch.bfloat16)),
            "q_cos": self._replicate(torch.zeros(1, 1, q_len, self.hd, dtype=torch.bfloat16)),
            "q_sin": self._replicate(torch.zeros(1, 1, q_len, self.hd, dtype=torch.bfloat16)),
            "mask": self._replicate(torch.zeros(1, 1, q_len, kv_len, dtype=torch.bfloat16)),
            "mask_full": self._replicate(torch.zeros(1, 1, q_len, kv_len, dtype=torch.bfloat16)),
            "ctx": self._replicate(torch.zeros(1, 1, ctx_pad, self.cfg.hidden_size, dtype=torch.bfloat16)),
            # The block's token ids, for the noise embedding. This one is the reason a capture is
            # impossible without staging at all: TtTarget.embed_device does ttnn.from_torch(ids)
            # every step, and a host upload inside a capture is illegal outright -- not merely
            # baked in, but rejected.
            "tok": ttnn.from_torch(
                torch.zeros(1, q_len, dtype=torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                memory_config=_DRAM,
                **(dict(mesh_mapper=ttnn.ReplicateTensorToMesh(self.device)) if self.multi else {}),
            ),
        }
        return self._sb

    def _stage_host(self, t, dst, layout=ttnn.TILE_LAYOUT):
        """DMA one host tensor into a persistent device buffer. Outside capture."""
        host = ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=layout,
            device=None,
            **(dict(mesh_mapper=ttnn.ReplicateTensorToMesh(self.device)) if self.multi else {}),
        )
        ttnn.copy_host_to_device_tensor(host, dst)

    def stage_step(self, start: int, new_ctx: int):
        """Refill the rope and mask buffers for this step. **Outside** trace capture.

        This is where the step's variability goes. ``start`` and ``new_ctx`` change every step and
        both used to reach the device as op ATTRIBUTES -- a ``ttnn.slice`` offset for rope, a tensor
        shape for the mask -- which is exactly what bakes at capture. Here they become tensor
        CONTENTS instead, and one capture then serves every (start, new_ctx) the loop produces.
        """
        sb = self._sb
        q_len, ctx_pad = sb["q_len"], sb["ctx_pad"]
        span = torch.arange(start - ctx_pad, start + q_len).clamp_min(0)
        blk = torch.arange(start, start + q_len)
        self._stage_host(self._cos_host[:, :, span, :], sb["cos"])
        self._stage_host(self._sin_host[:, :, span, :], sb["sin"])
        self._stage_host(self._cos_host[:, :, blk, :], sb["q_cos"])
        self._stage_host(self._sin_host[:, :, blk, :], sb["q_sin"])
        mask, mask_full = self._fixed_mask_tensors(q_len, new_ctx, ctx_pad)
        self._stage_host(mask, sb["mask"])
        self._stage_host(mask_full, sb["mask_full"])

    def stage_tokens(self, ids):
        """DMA the block's token ids into the fixed buffer. **Outside** trace capture.

        The embedding then runs on device from this buffer, so the only thing crossing PCIe per step
        is ``q_len`` ints instead of an upload a capture would refuse outright.
        """
        sb = self._sb
        q_len = sb["q_len"]
        assert ids.shape[-1] <= q_len, f"staged token buffer holds {q_len} ids, got {ids.shape[-1]}"
        host = torch.zeros(1, q_len, dtype=torch.int32)
        host[:, : ids.shape[-1]] = ids.to(torch.int32).cpu().reshape(1, -1)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                host,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=None,
                **(dict(mesh_mapper=ttnn.ReplicateTensorToMesh(self.device)) if self.multi else {}),
            ),
            sb["tok"],
        )
        return sb["tok"]

    def stage_taps(self, kv_source, new_ctx: int):
        """Copy this step's tap projection into the fixed context buffer, LEFT-padded.

        Device-to-device: ``project_taps`` produces a fresh tensor at a fresh address every step,
        and a replay has to read one address. Left-padding matches the position map the masks are
        built for -- row ``i`` of the context region is absolute position ``start - ctx_pad + i``.
        """
        sb = self._sb
        ctx_pad = sb["ctx_pad"]
        assert 0 <= new_ctx <= ctx_pad, f"staged context holds {ctx_pad} rows, got {new_ctx}"
        self._staged_new_ctx = new_ctx
        # Clear, then write the real rows into the tail. The clear matters: rows the mask marks
        # invalid are still PROJECTED, and a previous step's values left in those rows would be
        # projected too -- masked out of the attention, but not out of the K/V the step commits.
        self._stage_host(torch.zeros(1, 1, ctx_pad, self.cfg.hidden_size, dtype=torch.bfloat16), sb["ctx"])
        if new_ctx:
            ttnn.experimental.slice_write(
                kv_source,
                sb["ctx"],
                (0, 0, ctx_pad - new_ctx, 0),
                (1, 1, ctx_pad, self.cfg.hidden_size),
                (1, 1, 1, 1),
            )
        return sb["ctx"]

    def _fixed_mask_tensors(self, q_len: int, new_ctx: int, ctx_pad: int):
        """The two additive masks the fixed-capacity path needs, at a constant ``[1, 1, q_len, C+32]``.

        Key columns are laid out ``[ history(C) | padded context(16) | block(q_len) ]``:

        * ``history`` is the persistent buffer; column ``j`` is absolute position ``j`` and is real
          only while ``j < _ctx_len``.
        * ``padded context`` is this step's newly accepted rows, LEFT-padded to a constant 16 so the
          fused kv_proj's M stops varying. Left-padding is what makes the position map uniform:
          row ``i`` of the 32-row kv_src is absolute position ``start - 16 + i``, contiguous across
          both regions. The pad rows duplicate positions that are ALREADY in the history buffer, so
          masking them is not an approximation -- those positions are still attended, once, via
          ``history``.
        * ``block`` is the drafted slots at ``[start, start + q_len)``.

        Two masks because the layers disagree: the four sliding layers want validity AND causality
        AND the window, while the bidirectional layer takes no mask at all today. Handing it a
        validity-only mask reproduces that exactly -- today every row it sees is real, so "no mask"
        and "mask nothing real" are the same function.
        """
        C, start = self._cap, self._ctx_len + new_ctx
        kv_len = C + ctx_pad + q_len
        k_pos = torch.empty(kv_len, dtype=torch.long)
        k_real = torch.zeros(kv_len, dtype=torch.bool)
        k_pos[:C] = torch.arange(C)
        k_real[:C] = torch.arange(C) < self._ctx_len
        k_pos[C : C + ctx_pad] = torch.arange(start - ctx_pad, start)
        k_real[C : C + ctx_pad] = torch.arange(ctx_pad) >= (ctx_pad - new_ctx)
        k_pos[C + ctx_pad :] = torch.arange(start, start + q_len)
        k_real[C + ctx_pad :] = True

        q_pos = torch.arange(start, start + q_len).unsqueeze(1)
        k_pos, k_real = k_pos.unsqueeze(0), k_real.unsqueeze(0)
        causal = (k_pos <= q_pos) & ((q_pos - k_pos) < self.cfg.sliding_window)

        def _host(visible):
            return torch.where(visible, 0.0, _MASK_NEG).reshape(1, 1, q_len, kv_len).to(torch.bfloat16)

        return _host(k_real & causal), _host(k_real.expand(q_len, kv_len))

    # ---- setup -----------------------------------------------------------------------------

    def _build_rope_tables(self):
        """Host trig ONCE at construction; every step then slices these on device.

        The drafter cannot borrow the target's tables: it rotates the full 128-dim head at
        ``rope_theta = 1e7``, where the target rotates 64 of 256 dims at its own theta.
        """
        rope_dim = self.hd
        inv_freq = 1.0 / (self.cfg.rope_theta ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
        freqs = torch.outer(torch.arange(self.max_seq_len).float(), inv_freq)
        # HF half-split layout: [freqs, freqs], which is what rotary_embedding_hf consumes.
        emb = torch.cat([freqs, freqs], dim=-1)

        def _upload(t):
            return ttnn.from_torch(
                t.reshape(1, 1, self.max_seq_len, rope_dim).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,  # ROW_MAJOR so a non-tile-aligned row slice is legal
                device=self.device,
                memory_config=_DRAM,
                **(dict(mesh_mapper=ttnn.ReplicateTensorToMesh(self.device)) if self.multi else {}),
            )

        # Keep the HOST trig too: the fixed-capacity path gathers rows from it (see _fixed_rope)
        # rather than slicing the device table, because a ttnn.slice offset is an operation
        # attribute and would bake at capture. bfloat16 and the same cast as the upload, so the
        # gathered rows are bit-identical to the sliced ones.
        self._cos_host = emb.cos().reshape(1, 1, self.max_seq_len, rope_dim).to(torch.bfloat16)
        self._sin_host = emb.sin().reshape(1, 1, self.max_seq_len, rope_dim).to(torch.bfloat16)
        return _upload(emb.cos()), _upload(emb.sin())

    def _fixed_rope(self, start: int, q_len: int, ctx_pad: int):
        """cos/sin for the fixed 32-row kv_src span and for the block's own queries.

        The kv_src span is ``[start - 16, start + q_len)`` -- the left-padded context then the
        block. ``start`` can be smaller than 16 on the first speculative step (the prompt may be
        shorter), which would ask for negative positions; those rows are pad and are masked out, so
        the index is clamped to 0 and they take position 0's rotation. Clamping cannot touch a real
        row: the real context occupies the LAST ``new_ctx`` of the 16, at positions
        ``[start - new_ctx, start)``, all non-negative.
        """
        span = torch.arange(start - ctx_pad, start + q_len).clamp_min(0)
        blk = torch.arange(start, start + q_len)
        assert start + q_len <= self.max_seq_len, (
            f"positions [{start}, {start + q_len}) exceed the drafter's {self.max_seq_len}-row RoPE "
            "tables; raise max_seq_len"
        )
        g = lambda tbl, idx: self._replicate(tbl[:, :, idx, :])  # noqa: E731
        return (
            g(self._cos_host, span),
            g(self._sin_host, span),
            g(self._cos_host, blk),
            g(self._sin_host, blk),
        )

    def _rope_slice(self, pos0: int, span: int):
        """cos/sin for absolute positions ``[pos0, pos0 + span)``, sliced on device."""
        assert pos0 + span <= self.max_seq_len, (
            f"positions [{pos0}, {pos0 + span}) exceed the drafter's {self.max_seq_len}-row RoPE "
            "tables; raise max_seq_len"
        )

        def _slice(tbl):
            r = ttnn.slice(tbl, [0, 0, pos0, 0], [1, 1, pos0 + span, self.hd])
            return ttnn.to_layout(r, ttnn.TILE_LAYOUT)

        return _slice(self._cos), _slice(self._sin)

    # ---- pieces ----------------------------------------------------------------------------

    def _proj_pc(self, m, weight, fused_activation=None):
        """Cached 1D (mcast_in0) program config for a projection matmul at this step's M.

        The drafter's projections all run at M = one tile row, where ttnn's **auto** config sits far
        off the DRAM roofline -- catastrophically so for the fused ``kv_proj``, whose N = 2048 is 64
        tiles. MEASURED by ``tests/perf/test_dflash_attn_matmul_sweep.py`` (full 8x8 grid, the
        drafter's own dtypes and compute config), against auto:

            q_proj   M=16 K=5120 N=4096   120.0 -> 108.7 us   (-9.4 %)
            kv_proj  M=32 K=5120 N=2048   116.5 ->  53.5 us   (-54.0 %, 39 us roofline)
            o_proj   M=16 K=4096 N=5120   111.1 -> 111.0 us   (auto is already optimal -- o_proj is
                                                               deliberately left on auto)

        PCC is unchanged (0.999967 either way). Three families LOST and should not be retried here:
        the DRAM-sharded weight layout (+11 % q, +24 % o, and its in0 reshard is a second op),
        ``create_prefill_mlp_matmul_program_config`` (+23 % to +37 %, it is a 2D config for M in the
        thousands), and ``COMPUTE_HIFI2_NO_FP32_ACC``, which was slower AND cost PCC.
        ``create_prefill_kpass1_matmul_program_config`` is illegal at M = 32.

        M is not constant -- ``q_len`` is the block and ``kv_seq`` is that plus the newly accepted
        context -- so configs are cached per ``(m_tiles, K, N, fused_activation)`` instead of built
        once.

        ``fused_activation`` is used by exactly one caller, ``_layer_mlp``'s ``gate_proj``, and NOT
        for speed -- see there.
        """
        k, n = weight.shape[-2], weight.shape[-1]
        key = (-(-m // 32), k, n, fused_activation)
        pc = self._mm_pc.get(key)
        if pc is None:
            # Full grid: the sweep put nc=64 first at both shapes, and the factory shapes it
            # wide-first (8 cols) which is what shortens the in0 multicast column.
            pc = self._mm_pc[key] = tpc.create_matmul_1d_decode_progcfg(
                m, k, n, num_cores=64, fused_activation=fused_activation, grid_w=8
            )
        return pc

    def _rms(self, x, weight, memory_config=None):
        """Standard RMSNorm. The gain is used as-is — the drafter is Qwen3, not zero-centered Qwen3.5."""
        return ttnn.rms_norm(x, weight=weight, epsilon=self.cfg.rms_norm_eps, memory_config=memory_config)

    def project_taps(self, taps: list[ttnn.Tensor]) -> ttnn.Tensor:
        """``hidden_norm(fc(gather(concat(taps))))`` — the K/V source, computed once per step.

        ``taps`` are the target's residual streams at ``cfg.target_layer_ids``, in that order, each
        ``[1, 1, S, hidden/tp]`` fractured on the hidden dim. Concatenating them locally and
        all-gathering gives every device the full 25,600-wide feature in **device-major** column
        order, which is why ``fc``'s rows were permuted to match at load time
        (:func:`~.weights.reorder_fc_rows`). This gather is the drafter's only collective.
        """
        assert len(taps) == len(self.cfg.target_layer_ids), (
            f"got {len(taps)} taps, expected {len(self.cfg.target_layer_ids)} "
            f"({list(self.cfg.target_layer_ids)}) — order matters, fc was trained on it"
        )
        joined = ttnn.concat(list(taps), dim=-1, memory_config=_DRAM) if len(taps) > 1 else taps[0]
        if self.multi:
            from models.demos.blackhole.qwen36.tt import tp_common as tpc

            # The repo's full-mesh (cluster_axis=None) gather. NOT ccl.tt_all_gather: that takes
            # cluster_axis literally and on a (1, N) mesh axis 0 holds one device, which
            # all_gather_async rejects ("num_devices > 1, but has 1").
            gathered = tpc.tuned_vocab_all_gather(
                joined,
                self.device,
                self.tt_ccl,
                dim=3,
                topology=ttnn.Topology.Linear,
                num_workers_per_link=2,
                chunks_per_sync=10,
            )
        else:
            gathered = joined
        projected = ttnn.linear(gathered, self.weights.fc, compute_kernel_config=self.compute_cfg, memory_config=_DRAM)
        if gathered is not joined:
            ttnn.deallocate(gathered)
        out = self._rms(projected, self.weights.hidden_norm, memory_config=_DRAM)
        ttnn.deallocate(projected)
        return out

    def _heads(self, proj, n_heads, seq):
        """``[1, 1, S, n*hd]`` -> ``[1, n, S, hd]``. Q only — K/V go through :meth:`_kv_heads`."""
        x = ttnn.reshape(proj, (1, seq, n_heads, self.hd))
        x = ttnn.transpose(x, 1, 2)
        return ttnn.to_memory_config(x, _DRAM)

    def _kv_heads(self, kv):
        """Split the fused ``kv_proj`` output ``[1, 1, S, 2*nkv*hd]`` into head-major K and V.

        One op for both, using ``nlp_create_qkv_heads``' **tied-KV** mode. That mode is meant for a
        ``[q | kv]`` block whose single K/V section serves both K and V, so handing it our ``[k | v]``
        block with ``num_heads == num_kv_heads == nkv`` makes the "q" section K and the tied section
        V — it returns ``(K, V, V)`` and the third output is V again, dropped here.

        VERIFIED BIT-EXACT against the ``reshape`` + ``transpose`` form it replaces: K, V and the
        duplicate all ``torch.equal``, at NKV=8 / HD=128 / S=32 (the drafter's own geometry).

        MEASURED (T3K, ctx64, 5 layers): 251 us against 287 us for the 2 ``ReshapeView`` + 2
        ``Transpose`` it replaces, on top of the ``kv_proj`` fusion's own -273 us of matmul. The
        margin is thin because this op's work split gives a 32-row input a single core (the same
        reason ``nlp_concat_heads`` LOSES on the way out -- see :meth:`_layer_attention`); it wins
        only because it does K and V in one launch instead of four.
        """
        k, v, v_dup = ttnn.experimental.nlp_create_qkv_heads(
            kv,
            num_heads=self.nkv,
            num_kv_heads=self.nkv,
            transpose_k_heads=False,
            kv_tied=True,
            memory_config=_DRAM,
        )
        ttnn.deallocate(v_dup)
        return k, v

    def _sliding_mask(self, q_len, kv_len):
        """Additive ``[1, 1, q_len, kv_len]`` mask for the causal sliding-window layers.

        Built once per step, not once per layer: every layer's KV history is the same length (they
        commit in lockstep), so all four sliding layers see the same ``(q_len, kv_len)``. The
        bidirectional layer takes no mask at all.
        """
        q_pos = torch.arange(kv_len - q_len, kv_len).unsqueeze(1)
        k_pos = torch.arange(kv_len).unsqueeze(0)
        visible = (k_pos <= q_pos) & ((q_pos - k_pos) < self.cfg.sliding_window)
        mask = torch.where(visible, 0.0, float("-inf")).reshape(1, 1, q_len, kv_len)
        return ttnn.from_torch(
            mask.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=_DRAM,
            **(dict(mesh_mapper=ttnn.ReplicateTensorToMesh(self.device)) if self.multi else {}),
        )

    def _layer_attention(
        self,
        layer_idx,
        lw,
        hidden,
        ctx_rm,
        cos,
        sin,
        q_cos,
        q_sin,
        mask,
        *,
        hist_len,
        mask_full=None,
        real_ctx=None,
        ctx_pad=16,
    ):
        """One layer's attention. Q from ``hidden`` (the block); K/V from ``[ctx_rm, hidden]``.

        ``ctx_rm`` arrives **ROW_MAJOR** and ``q_cos``/``q_sin`` arrive already sliced, both because
        they are identical across the five layers — see :meth:`forward`.

        ``ctx_kv`` is the shared :meth:`project_taps` output — note it goes to ``kv_proj``
        **raw**, without this layer's ``input_layernorm``. Only the noise branch is normed. That
        asymmetry is the drafter's, not an oversight (see the reference's ``Qwen3DFlashAttention``),
        and norming it here silently costs acceptance.

        ``hist_len`` is how many positions this layer's KV history already holds. Every row of
        ``ctx_kv`` is accepted context by construction, so all of them are committed; the block's own
        K/V is scratch and is dropped when this call returns.
        """
        q_len = hidden.shape[-2]
        new_ctx = ctx_rm.shape[-2] if ctx_rm is not None else 0

        # Row-axis concat of two part-tile row counts (16 ctx + 16 block) cannot happen in TILE, so
        # it goes through ROW_MAJOR. Doing that explicitly, with ``ctx_rm`` converted ONCE per step
        # in forward(), leaves only `hidden`'s untilize per layer -- ttnn's implicit conversion
        # untilized BOTH operands every layer (MEASURED: 2 x 11 us per layer, 5 of the 10
        # UntilizeWithUnpadding ops in the step were re-doing identical work).
        if new_ctx:
            hidden_rm = ttnn.to_layout(hidden, ttnn.ROW_MAJOR_LAYOUT)
            kv_src = ttnn.concat([ctx_rm, hidden_rm], dim=-2, memory_config=_DRAM)
            ttnn.deallocate(hidden_rm)
            kv_src = ttnn.to_layout(kv_src, ttnn.TILE_LAYOUT)
        else:
            kv_src = hidden
        kv_seq = kv_src.shape[-2]

        # Explicit 1D progcfgs, and L1 outputs: both are measured wins at these shapes (see
        # _proj_pc). The L1 output is legal for both consumers -- _heads reshapes q and then hands
        # it back to DRAM itself, and nlp_create_qkv_heads only requires an INTERLEAVED input --
        # and the tensors are 256 KB (q) and 128 KB (kv).
        q = ttnn.linear(
            hidden,
            lw.q_proj,
            compute_kernel_config=self.compute_cfg,
            program_config=self._proj_pc(q_len, lw.q_proj),
            memory_config=_L1,
        )
        # K and V read the same rows, so they are one fused matmul and one tied head-split.
        kv = ttnn.linear(
            kv_src,
            lw.kv_proj,
            compute_kernel_config=self.compute_cfg,
            program_config=self._proj_pc(kv_seq, lw.kv_proj),
            memory_config=_L1,
        )
        if new_ctx:
            ttnn.deallocate(kv_src)

        q = self._heads(q, self.nh, q_len)
        k, v = self._kv_heads(kv)
        ttnn.deallocate(kv)

        q = self._rms(q, lw.q_norm)
        k = self._rms(k, lw.k_norm)

        # RoPE: K takes every position it spans; Q takes only the trailing block positions. That
        # split is what pins the drafted block to its absolute positions. Q's cos/sin are sliced
        # once per step in forward(), not here -- see there for why that mattered.
        k = apply_partial_rope_prefill(k, cos, sin, self.nkv, self.hd)
        q = apply_partial_rope_prefill(q, q_cos, q_sin, self.nh, self.hd)

        hist_k, hist_v = self._ctx_k[layer_idx], self._ctx_v[layer_idx]
        if self._cap is not None:
            # Fixed-capacity: the buffer is persistent, so commit by writing THROUGH it rather than
            # by rebinding to a fresh concat. The real rows are the LAST `real_ctx` of the 16-row
            # padded context region (left-padded; see _fixed_masks), and they land at `hist_len` --
            # the length BEFORE this step, which is also why the mask still marks them invalid in
            # the history region and valid in the context region. Written once, counted once.
            if real_ctx:
                end = hist_len + real_ctx
                for src, dst in ((k, hist_k), (v, hist_v)):
                    rows = ttnn.slice(
                        src, (0, 0, ctx_pad - real_ctx, 0), (1, self.nkv, ctx_pad, self.hd), memory_config=_DRAM
                    )
                    ttnn.experimental.slice_write(
                        rows, dst, (0, 0, hist_len, 0), (1, self.nkv, end, self.hd), (1, 1, 1, 1)
                    )
                    ttnn.deallocate(rows)
            full_k = ttnn.concat([hist_k, k], dim=-2, memory_config=_DRAM)
            full_v = ttnn.concat([hist_v, v], dim=-2, memory_config=_DRAM)
            ttnn.deallocate(k)
            ttnn.deallocate(v)
        else:
            # Prepend this layer's committed history, then commit the newly accepted rows.
            full_k = ttnn.concat([hist_k, k], dim=-2, memory_config=_DRAM) if hist_k is not None else k
            full_v = ttnn.concat([hist_v, v], dim=-2, memory_config=_DRAM) if hist_v is not None else v

            if new_ctx:
                keep = hist_len + new_ctx
                new_hist_k = ttnn.slice(full_k, (0, 0, 0, 0), (1, self.nkv, keep, self.hd), memory_config=_DRAM)
                new_hist_v = ttnn.slice(full_v, (0, 0, 0, 0), (1, self.nkv, keep, self.hd), memory_config=_DRAM)
                if hist_k is not None:
                    ttnn.deallocate(hist_k)
                    ttnn.deallocate(hist_v)
                self._ctx_k[layer_idx], self._ctx_v[layer_idx] = new_hist_k, new_hist_v

        attn = ttnn.transformer.scaled_dot_product_attention(
            q,
            full_k,
            full_v,
            attn_mask=mask if self.cfg.is_sliding(layer_idx) else mask_full,
            is_causal=False,  # Q is the block only, so SDPA's own causal alignment would be wrong.
            scale=self.scale,
            memory_config=_DRAM,
        )
        for t in (q, full_k, full_v):
            ttnn.deallocate(t)

        # NOT nlp_concat_heads, though it is bit-identical here (verified torch.equal at q_len=16)
        # and one op instead of two: its work split is num_blocks = B * S/TILE_HEIGHT, so a 16-slot
        # block is ONE block and it runs on ONE core -- 63 us/layer against 49 us (42 reshape + 7
        # transpose) for this pair on 64 cores (MEASURED, T3K ctx64: 314 us vs 245 us over 5 layers,
        # +69 us). It only pays at prefill seq lengths; the drafter never has more than a tile row
        # of queries.
        attn = ttnn.transpose(attn, 1, 2)
        attn = ttnn.reshape(attn, (1, 1, q_len, self.nh * self.hd))
        # Replicated, so o_proj is a plain local matmul — no all-reduce. Left on the AUTO program
        # config deliberately: the sweep in _proj_pc found nothing that beats it at M=16 K=4096
        # N=5120 (every explicit config tied or lost), unlike q_proj and kv_proj.
        out = ttnn.linear(attn, lw.o_proj, compute_kernel_config=self.compute_cfg, memory_config=_DRAM)
        ttnn.deallocate(attn)
        return out

    def _layer_mlp(self, lw, x):
        """SwiGLU. All three matmuls stay on the **auto** program config, deliberately.

        ``up_proj`` and ``down_proj`` stay on the **auto** program config; ``gate_proj`` carries an
                explicit one solely to fuse its SiLU into the packer (below).

                SWEPT and nothing beat auto on TIME — the opposite of ``_layer_attention``, where the same
                families won big (see :meth:`_proj_pc`). ``tests/perf/test_dflash_mlp_matmul_sweep.py``,
                M=16 and the drafter's own dtypes, best explicit candidate of ~14 per shape:

                    gate  M=16 K=5120  N=17408   auto 490 us   best 1D 511 us   +4.3 %
                    up    M=16 K=5120  N=17408   auto 469 us   best 1D 495 us   +5.5 %
                    down  M=16 K=17408 N=5120    auto 447 us   best 1D 454 us   +1.7 %

                Three results worth keeping:

                * **DRAM-sharded is a trap here.** On ``down`` it is +207 % (1,371 us against 447 us), and on
                  ``gate`` it is not even legal -- the DRAM-sharded matmul rejects a fused activation
                  (``!parameters.user_fused_activation.has_value``). tt-perf-report prints "try a DRAM-sharded
                  program config" for these ops; at M = one tile row that advice is wrong.
                * **Fusing gate|up into one matmul does NOT pay**, unlike the same trick on ``kv_proj``:
                  975 us fused against 490 + 469 = 959 us split, and that is *before* the slice + silu the
                  fused form needs (SILU cannot be packer-fused across the ``up`` half).
                * ``activation="silu"`` here does **not** fuse into the packer on the auto path -- it emits a
                  separate ~18 us ``Unary`` (visible as 5 ops / 94 us in the profile). An explicit progcfg
                  does genuinely fuse it, and still loses overall (511 us as one op vs 472 + 18).

                What DID pay here is not a program config at all but the **weight dtype**: ``gate_proj`` and
                ``up_proj`` are bf4 (-43 %/-45 %, these are pure weight streaming so half the bytes is half
                the time), while ``down_proj`` stays bf8 because bf4 on it fails the drafter's PCC gate. The
                measurements and the accuracy trade are in :data:`~.weights.MLP_DOWN_DTYPE`.

                THE SILU IS FUSED INTO THE PACKER, and that is a DISPATCH-COUNT change, not a speed one.
                ``activation="silu"`` on the auto config does **not** fuse -- it emits a separate
                ``UnaryDeviceOperation`` (18 us x 5 layers, 94 us/step in the profile). An explicit progcfg
                with ``fused_activation=SILU`` genuinely fuses it. MEASURED over the whole SwiGLU block
                (``tests/perf/test_dflash_swiglu_fusion_sweep.py``, BFP4 weights):

                    current       561.5 us   4 ops   matmul 521 + mul 23 + unary 18
                    packer_fused  563.0 us   3 ops   matmul 540 + mul 23           <- shipped
                    manual_silu   578.6 us   4 ops   the control: explicit progcfg, silu NOT fused

                So the fusion itself is free and the +19 us is the explicit progcfg -- ``manual_silu``
                isolates that, being +17 us with no fusion at all. Device time is a wash (+1.5 us, the
                difference of two real numbers); what it buys is **5 fewer dispatches per step**, which is
                worth having in an untraced module at ~13 % device utilization, on the same reasoning that
                kept the tied K/V head-split in :meth:`_kv_heads`.

                ``ttnn.swiglu`` was also measured and is NOT a fused kernel: it decomposes into
                Slice + Unary + BinaryNg (5 ops, 623.5 us, +62 us) and returns the tile-PADDED height
                (M=32 for a 16-row input), which a 16-slot block's residual add cannot consume.
        """
        gate = ttnn.linear(
            x,
            lw.gate_proj,
            compute_kernel_config=self.compute_cfg,
            program_config=self._proj_pc(x.shape[-2], lw.gate_proj, fused_activation=ttnn.UnaryOpType.SILU),
            memory_config=_DRAM,
        )
        up = ttnn.linear(x, lw.up_proj, compute_kernel_config=self.compute_cfg, memory_config=_DRAM)
        hidden = ttnn.mul(gate, up, memory_config=_DRAM)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        # Replicated, so down_proj is a plain local matmul — no all-reduce.
        out = ttnn.linear(hidden, lw.down_proj, compute_kernel_config=self.compute_cfg, memory_config=_DRAM)
        ttnn.deallocate(hidden)
        return out

    # ---- forward ---------------------------------------------------------------------------

    def forward(self, kv_source: ttnn.Tensor, noise: ttnn.Tensor, start: int, *, staged: bool = False):
        """Run the drafter over one block.

        Args:
            kv_source: :meth:`project_taps` output, ``[1, 1, n_new, hidden]`` replicated — the newly
                accepted context rows, at absolute positions ``[start - n_new, start)``.
            noise: the block's input embeddings, ``[1, 1, q_len, hidden]`` replicated.
            start: absolute position of the block's first slot.

        Returns:
            ``[1, 1, q_len, hidden]`` — the final-normed hidden states for the block.
        """
        q_len = noise.shape[-2]
        # When staged, the context went to stage_taps rather than to this call, so the row count
        # comes from there and kv_source may be None.
        new_ctx = self._staged_new_ctx if staged else (kv_source.shape[-2] if kv_source is not None else 0)
        hist_len = self._ctx_len
        assert hist_len + new_ctx == start, (
            f"drafter context is at {hist_len} + {new_ctx} new rows but the block starts at {start}; "
            "the loop must hand over every accepted token's taps exactly once"
        )

        fixed = self._cap is not None
        if fixed:
            assert hist_len + new_ctx <= self._cap, (
                f"drafter context would reach {hist_len + new_ctx} rows, past its {self._cap}-row "
                "fixed capacity; raise ctx_capacity or fall back to the growing-history path"
            )
            # Context pads UP to a multiple of 16. In the steady state new_ctx is the accept count
            # (1..block_size) so this is a constant 16 and every shape is constant -- which is what a
            # capture needs. The ONE step that exceeds it is the first, where dflash_generate hands
            # over the whole prompt's taps at once; that step is a one-off outside the traced steady
            # state, so letting its shape grow costs nothing and beats refusing the prompt.
            ctx_pad = max(16, -(-new_ctx // 16) * 16)
            # Everything per-step is a CONSTANT shape here: 16 padded context rows + q_len block
            # rows for kv_src, a [1,1,q_len,C+16+q_len] mask pair, and a persistent [1,nkv,C,hd]
            # history. Only tensor CONTENTS vary, which is what a capture can tolerate.
            sb = self._sb if staged else None
            if staged:
                # Everything this step consumes already sits at a fixed address, refilled in place
                # by stage_step / stage_taps / stage_tokens. Nothing is built here, which is the
                # point: a replay re-runs recorded ops against recorded addresses and can do neither.
                assert sb is not None, "staged=True needs alloc_step_buffers()"
                assert q_len == sb["q_len"], f"staged for q_len {sb['q_len']}, got {q_len}"
                assert ctx_pad == sb["ctx_pad"], (
                    f"staged for ctx_pad {sb['ctx_pad']}, got {ctx_pad}; a step accepting more than "
                    f"{sb['ctx_pad']} rows (the prompt step) must run unstaged"
                )
                cos, sin = sb["cos"], sb["sin"]
                q_cos, q_sin = sb["q_cos"], sb["q_sin"]
                mask, mask_full = sb["mask"], sb["mask_full"]
                # stage_taps already left the context left-padded to ctx_pad at a fixed address.
                ctx_rm = ttnn.to_layout(sb["ctx"], ttnn.ROW_MAJOR_LAYOUT)
            else:
                cos, sin, q_cos, q_sin = self._fixed_rope(start, q_len, ctx_pad)
                mask, mask_full = (self._replicate(t) for t in self._fixed_mask_tensors(q_len, new_ctx, ctx_pad))
                ctx_rm = ttnn.to_layout(kv_source, ttnn.ROW_MAJOR_LAYOUT) if new_ctx else None
            if not staged and new_ctx < ctx_pad:
                pad = self._replicate(
                    # noise, not kv_source: the very first step has no accepted context at all
                    # (kv_source is None) and both carry the same hidden width.
                    torch.zeros(1, 1, ctx_pad - new_ctx, noise.shape[-1], dtype=torch.bfloat16),
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
                parts = [pad, ctx_rm] if ctx_rm is not None else [pad]
                ctx_rm = ttnn.concat(parts, dim=-2, memory_config=_DRAM) if len(parts) > 1 else pad
            x = noise
            for i, lw in enumerate(self.weights.layers):
                residual = x
                normed = self._rms(x, lw.input_layernorm, memory_config=_DRAM)
                attn = self._layer_attention(
                    i,
                    lw,
                    normed,
                    ctx_rm,
                    cos,
                    sin,
                    q_cos,
                    q_sin,
                    mask,
                    hist_len=hist_len,
                    mask_full=mask_full,
                    real_ctx=new_ctx,
                    ctx_pad=ctx_pad,
                )
                ttnn.deallocate(normed)
                x = ttnn.add(residual, attn, memory_config=_DRAM)
                ttnn.deallocate(attn)
                if residual is not noise:
                    ttnn.deallocate(residual)

                residual = x
                normed = self._rms(x, lw.post_attention_layernorm, memory_config=_DRAM)
                mlp = self._layer_mlp(lw, normed)
                ttnn.deallocate(normed)
                x = ttnn.add(residual, mlp, memory_config=_DRAM)
                ttnn.deallocate(mlp)
                ttnn.deallocate(residual)

            # A staged step owns only ctx_rm (its own layout conversion); the rest are persistent
            # buffers the NEXT step refills in place. Freeing them would invalidate exactly the
            # addresses a capture records.
            transient = (ctx_rm,) if staged else (cos, sin, q_cos, q_sin, mask, mask_full, ctx_rm)
            for t in transient:
                if t is not None:
                    ttnn.deallocate(t)
            self._ctx_len = hist_len + new_ctx
            out = self._rms(x, self.weights.norm, memory_config=_DRAM)
            if x is not noise:
                ttnn.deallocate(x)
            return out

        # cos/sin cover [start - new_ctx, start + q_len): the new context rows then the block, which
        # is exactly the span K sees.
        cos, sin = self._rope_slice(start - new_ctx, new_ctx + q_len)
        # Q's own cos/sin are the trailing q_len positions, [start, start + q_len). Sliced HERE,
        # once, straight off the resident ROW_MAJOR tables -- which is what those tables are
        # ROW_MAJOR for (see _build_rope_tables): a non-tile-aligned row slice is legal there and
        # costs a Slice plus one Tilize. _layer_attention used to slice the already-TILIZED cos/sin
        # instead, and once per layer, so each layer paid untilize + slice + retilize for an
        # identical result. MEASURED: 130 us and 30 ops per step became ~9 us and 4 ops.
        q_cos, q_sin = self._rope_slice(start, q_len)
        # One ROW_MAJOR copy of the context for all five layers' K/V concat (see _layer_attention).
        ctx_rm = ttnn.to_layout(kv_source, ttnn.ROW_MAJOR_LAYOUT) if new_ctx else None
        # Every layer's KV history is the same length, so one mask serves all the sliding layers.
        mask = self._sliding_mask(q_len, hist_len + new_ctx + q_len)

        x = noise
        for i, lw in enumerate(self.weights.layers):
            residual = x
            normed = self._rms(x, lw.input_layernorm, memory_config=_DRAM)
            attn = self._layer_attention(i, lw, normed, ctx_rm, cos, sin, q_cos, q_sin, mask, hist_len=hist_len)
            ttnn.deallocate(normed)
            x = ttnn.add(residual, attn, memory_config=_DRAM)
            ttnn.deallocate(attn)
            if residual is not noise:
                ttnn.deallocate(residual)

            residual = x
            normed = self._rms(x, lw.post_attention_layernorm, memory_config=_DRAM)
            mlp = self._layer_mlp(lw, normed)
            ttnn.deallocate(normed)
            x = ttnn.add(residual, mlp, memory_config=_DRAM)
            ttnn.deallocate(mlp)
            ttnn.deallocate(residual)

        for t in (cos, sin, q_cos, q_sin, mask, ctx_rm):
            if t is not None:
                ttnn.deallocate(t)
        self._ctx_len = hist_len + new_ctx

        out = self._rms(x, self.weights.norm, memory_config=_DRAM)
        if x is not noise:
            ttnn.deallocate(x)
        return out
