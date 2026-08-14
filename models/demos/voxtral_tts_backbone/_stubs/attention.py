# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN port of `attention` (`MistralAttention`) for
`/localdev/lserbedzija/hf_models/voxtral-tts-backbone`.

GQA self-attention: 32 query heads / 8 KV heads, head_dim 128, hidden 3072,
no biases, RoPE applied to Q and K, additive causal mask.

The canonical `models/tt_transformers/tt/attention.py::Attention` is not usable
as-is from this per-component harness: it is built around `ModelArgs` (a full
checkpoint config + tokenizer + mesh/CCL plumbing) and its forward takes
`(x, current_pos, rot_mats, user_id, mode, page_table, ...)` with its own KV
cache, not the `(hidden_states, position_embeddings, attention_mask)` triple the
HF golden defines. So the forward below is written directly against the same
ttnn primitives that module uses.

`__init__`/`build` stage weights with torch (allowed — the weights come from an
HF checkpoint). `__call__` is pure ttnn: `models/common/native_probe.py` counts
what actually executes and a single torch op in the forward would (correctly)
disqualify this as a host reimplementation.
"""
from __future__ import annotations

import torch
import ttnn

from models.demos.voxtral_tts_backbone._stubs.decode_matmul import build_plan


def _is_mesh_device(device) -> bool:
    try:
        if isinstance(device, ttnn.MeshDevice):
            return True
    except AttributeError:
        pass
    return hasattr(device, "get_device_ids") or hasattr(device, "get_devices")


def _replicate_mapper(device):
    if not _is_mesh_device(device):
        return None
    try:
        return ttnn.ReplicateTensorToMesh(device)
    except (AttributeError, TypeError):
        return None


def _stage(weight: torch.Tensor, device, dtype=ttnn.bfloat16):
    """Stage an `nn.Linear` weight (out, in) as a ttnn matmul operand (in, out).

    `dtype` is the STORED format. Decode is DRAM-bandwidth bound on these
    weights -- a projection's cost is the bytes it streams for one token -- so
    the stored format, not the math fidelity, is the lever that moves it.
    """
    host = weight.detach().to(torch.float32).transpose(0, 1).contiguous().to(torch.bfloat16)
    mapper = _replicate_mapper(device)
    if mapper is not None:
        return ttnn.from_torch(
            host,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=mapper,
        )
    return ttnn.from_torch(host, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


_TILE = 32


def _kv_write_memory_config(n_kv_heads: int, head_dim: int):
    """L1 shard config for the single-token cache write operand.

    `ttnn.experimental.paged_update_cache` — the only cache write that takes its
    row index as a DEVICE tensor, which is what keeps a traced decode step free
    of host scalars — wants a sharded `[1, batch, n_kv_heads, head_dim]` input.
    Batch is 1 here, so one core holds the whole (tile-padded) operand.
    """
    rows = max(_TILE, ((int(n_kv_heads) + _TILE - 1) // _TILE) * _TILE)
    try:
        return ttnn.create_sharded_memory_config(
            shape=(rows, int(head_dim)),
            core_grid=ttnn.CoreGrid(y=1, x=1),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
    except Exception:  # noqa: BLE001 - fall back to the int-index cache write
        return None


def _kv_second_core_memory_config(n_kv_heads: int, head_dim: int):
    """That same single-core operand shard, on a DIFFERENT core.

    `paged_fused_update_cache` writes K and V in ONE launch by running them on
    DISJOINT cores, and refuses operands that share one ("input_tensor1 and
    input_tensor2 must not overlap"). Head creation is batch-parallel, so at
    batch 1 it puts every one of its outputs on core (0,0) -- K and V included.
    Moving just one of them one core over is what makes the fused write legal,
    and a single 2 KB shard move is far cheaper than the launch it saves.
    """
    rows = max(_TILE, ((int(n_kv_heads) + _TILE - 1) // _TILE) * _TILE)
    try:
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))}),
                (rows, int(head_dim)),
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
    except Exception:  # noqa: BLE001 - fall back to two separate cache writes
        return None


#: Cores the decode attention may spend on ONE (batch, kv-head) pair.
#
# Handed no program config, `scaled_dot_product_attention_decode` sets this to
# the WHOLE grid, which at batch 1 with 8 KV heads works out to 16 cores per
# head and 128 active. Sixteen ways is far too fine for this cache: a 256-deep
# history is 16 positions per core, half a tile of work, and the kernel then
# pays a ceil(log2(16)) = 4-round tree reduction ACROSS those cores to put the
# head back together. The reduction, not the read, is what the op costs.
#
# Swept on this cache depth, as the op's device time in the profiled slice:
# default(16) 5.29 / 4 -> 3.25 / 2 -> 3.04 / 1 -> 2.99 ms. It is monotone, which
# is the tell that the reduction is the whole cost -- but the last step is worth
# 0.05 ms and whole-model came out marginally BETTER at 2 than at 1 (231.845 vs
# 231.919 ms), so take 2: one reduction round, 16 active cores, and it still has
# somewhere to go when a deeper cache makes the read matter again.
_SDPA_CORES_PER_HEAD = 2


def _sdpa_decode_program_config(device):
    """Decode-attention parallelism, or None to keep the op's own default."""
    try:
        grid = device.compute_with_storage_grid_size()
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid,
            q_chunk_size=0,
            k_chunk_size=0,
            max_cores_per_head_batch=_SDPA_CORES_PER_HEAD,
        )
    except Exception:  # noqa: BLE001 - keep the op's own core allocation
        return None


class TtAttention:
    def __init__(self, device, weights, dims, scaling):
        self.device = device
        self.wq, self.wk, self.wv, self.wo, self.wqkv = weights
        self.n_heads, self.n_kv_heads, self.head_dim = dims
        self.scaling = scaling
        self.kv_write_memory_config = _kv_write_memory_config(self.n_kv_heads, self.head_dim)
        self.kv_second_core_memory_config = _kv_second_core_memory_config(self.n_kv_heads, self.head_dim)
        #: Cleared if the one-launch K+V cache write refuses these operands.
        self._fused_cache_write = self.kv_second_core_memory_config is not None and hasattr(
            ttnn.experimental, "paged_fused_update_cache"
        )
        # K and V share a shape, so one plan serves both. 16 cores, not the 32
        # the "largest even split" default picks: the same k-reduction-shape
        # effect measured on down_proj applies here. Per call at bf8_b:
        # default 0.0528 / 32c 0.0379 / 16c 0.0251 / 8c 0.0314 / 4c 0.0535 ms.
        self.kv_plan = build_plan(device, int(self.wk.shape[-2]), int(self.wk.shape[-1]), max_cores=48)
        self.o_plan = build_plan(device, int(self.wo.shape[-2]), int(self.wo.shape[-1]))
        self.q_plan = build_plan(device, int(self.wq.shape[-2]), int(self.wq.shape[-1]))
        # 48 cores for the fused QKV, re-swept after the projections were fused:
        # 16 -> 17.47 / 32 -> 15.98 / 48 -> 13.61 / 96 -> 35.71 ms in the profiled
        # slice. The cap is not a workaround, it is the tuned value -- 32 was
        # right for the shape this plan had before Q/K/V became one 3072->6144
        # read, and the wider N moved the optimum. The cliff at 96 is the
        # k-reduction pathology down_proj documents: in0_block_w falls to 1, so
        # each core walks 96 sequential single-tile k-blocks for 2 output tiles.
        self.qkv_plan = build_plan(device, int(self.wqkv.shape[-2]), int(self.wqkv.shape[-1]), max_cores=48)
        #: Cleared the first time the fused rotation refuses this model's
        #: operands, so the explicit chain takes over for the rest of the run
        #: instead of raising inside a captured trace.
        self._fused_rope = True
        #: Cleared the first time the decode-native head layout refuses this
        #: model's operands, so the generic layout takes over for the rest of
        #: the run instead of raising inside a captured trace.
        self._decode_heads = hasattr(ttnn.experimental, "nlp_create_qkv_heads_decode")
        #: Cleared if head creation refuses the projection's own output shard,
        #: which costs only the interleaved conversion it was there to avoid.
        self._presharded_heads = True
        self._q_end = int(self.wq.shape[-1])
        self._k_end = self._q_end + int(self.wk.shape[-1])
        self._v_end = self._k_end + int(self.wv.shape[-1])
        # HiFi4 + fp32 accumulate: the projections and the SDPA feed a 0.99 PCC
        # gate, and bf16 LoFi accumulation is what usually costs those digits.
        try:
            self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
                device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )
        except Exception:  # noqa: BLE001 - accuracy tuning is best-effort
            self.compute_kernel_config = None

        self.sdpa_decode_program_config = _sdpa_decode_program_config(device)

    # ---------------------------------------------------------------- build
    @classmethod
    def build(cls, device, torch_module):
        if torch_module is None:
            raise RuntimeError("attention build needs the HF MistralAttention module to read weights from")
        cfg = getattr(torch_module, "config", None)
        head_dim = getattr(torch_module, "head_dim", None) or getattr(cfg, "head_dim", None)
        n_heads = getattr(cfg, "num_attention_heads", None)
        n_kv_heads = getattr(cfg, "num_key_value_heads", None) or n_heads
        q_w = torch_module.q_proj.weight
        k_w = torch_module.k_proj.weight
        v_w = torch_module.v_proj.weight
        o_w = torch_module.o_proj.weight
        if head_dim is None:
            head_dim = q_w.shape[0] // int(n_heads)
        head_dim = int(head_dim)
        # Trust the weights over the config for the head counts: a per-component
        # harness may hand us a submodule whose config was edited.
        n_heads = int(q_w.shape[0]) // head_dim
        n_kv_heads = int(k_w.shape[0]) // head_dim
        scaling = getattr(torch_module, "scaling", None) or head_dim**-0.5
        # STORED as bfloat8_b: the four projections are ~63 MB of the ~233 MB a
        # layer streams per decoded token, and decode is bound on those bytes.
        # Q/K/V feed RoPE and the attention scores, which 02 §13 flags as the
        # tensors not to push below bf8b -- bf8b is that floor, not past it.
        weights = (
            _stage(q_w, device, dtype=ttnn.bfloat8_b),
            _stage(k_w, device, dtype=ttnn.bfloat8_b),
            _stage(v_w, device, dtype=ttnn.bfloat8_b),
            _stage(o_w, device, dtype=ttnn.bfloat8_b),
            # Q, K and V read the same activation, so their weights are also
            # staged CONCATENATED (03 s1: always fuse QKV): one 3072x6144 read
            # and one matmul per token instead of three, and one shard instead
            # of two. The separate weights stay for prefill and for the
            # per-component test, which pinned the unfused body.
            _stage(torch.cat([q_w, k_w, v_w], dim=0), device, dtype=ttnn.bfloat8_b),
        )
        return cls(device, weights, (n_heads, n_kv_heads, head_dim), float(scaling))

    # -------------------------------------------------------------- forward
    def _split_heads(self, proj, n_heads):
        """(1, s, n*hd) -> (1, n, s, hd)"""
        shape = list(proj.shape)
        batch, seq = shape[0], shape[-2]
        heads = ttnn.reshape(proj, (batch, seq, n_heads, self.head_dim))
        return ttnn.permute(heads, (0, 2, 1, 3))

    def _rotate_half(self, x):
        half = self.head_dim // 2
        shape = list(x.shape)
        first = ttnn.slice(x, [0, 0, 0, 0], [shape[0], shape[1], shape[2], half])
        second = ttnn.slice(x, [0, 0, 0, half], [shape[0], shape[1], shape[2], self.head_dim])
        return ttnn.concat([ttnn.neg(second), first], dim=-1)

    def _q_proj(self, hidden_states, mm):
        """Q projection for the shapes the shared-shard path does not cover."""
        if self.q_plan is not None and self.q_plan.matches(hidden_states):
            return self.q_plan(hidden_states, self.wq, self.compute_kernel_config)
        return ttnn.linear(hidden_states, self.wq, **mm)

    def _kv_proj(self, hidden_states, weight, mm):
        """K/V projection, on the full grid for the decode shape.

        These are the NARROWEST projections in the block (3072 -> 1024, only 32
        output tiles), which is exactly the case `ttnn.linear` routes worst: at
        decode it reached 118 GB/s, less than a quarter of the device's DRAM
        bandwidth and by far the lowest of any projection here.
        """
        if self.kv_plan is not None and self.kv_plan.matches(hidden_states):
            return self.kv_plan(hidden_states, weight, self.compute_kernel_config)
        return ttnn.linear(hidden_states, weight, **mm)

    def _apply_rope(self, x, cos, sin):
        """`x*cos + rotate_half(x)*sin` — HF's rotation, as ONE dispatch.

        Written out (the fallback below) this is SEVEN ops for a single
        elementwise rotation: two slices, a neg and a concat to build
        rotate_half, then two multiplies and an add. It runs twice per layer (Q
        and K), so at 26 layers it is ~360 launches per decoded token against
        tensors of a few KB — which is exactly why the roofline tags this
        model's BinaryNg/Reshape/Concat/Slice ops `bound_by=dispatch`. Their
        cost is launch overhead, and the only lever that touches launch overhead
        is issuing fewer launches.

        `rotary_embedding_hf` is that same HF-style rotation as one kernel, and
        our operands are ALREADY the layout it documents for prefill mode:
        `_split_heads` gives [1, heads, s, hd] and `_as_broadcastable` gives
        [1, 1, s, hd]; a decode row is that shape with s=1. head_dim 128 is a
        multiple of 2*TILE, which is the op's alignment requirement.
        """
        if self._fused_rope:
            try:
                return ttnn.experimental.rotary_embedding_hf(x, cos, sin, is_decode_mode=False)
            except Exception:  # noqa: BLE001 - unsupported shape/dtype: explicit chain
                self._fused_rope = False
        return ttnn.add(ttnn.multiply(x, cos), ttnn.multiply(self._rotate_half(x), sin))

    def _as_broadcastable(self, t):
        """cos/sin arrive as (1, s, hd); SDPA-shaped tensors are (1, n, s, hd)."""
        shape = list(t.shape)
        if len(shape) == 4:
            return t
        if len(shape) == 3:
            return ttnn.reshape(t, (shape[0], 1, shape[1], shape[2]))
        raise RuntimeError(f"unexpected rotary embedding rank {len(shape)} (shape={shape})")

    def __call__(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        kv_cache=None,
        cache_fill=False,
        cache_pos=None,
        cache_pos_tensor=None,
        **_ignored,
    ):
        """Pure-ttnn GQA self-attention.

        `kv_cache`/`cache_fill`/`cache_pos`/`cache_pos_tensor` are OPTIONAL and
        default to None/False: with them absent this is bit-for-bit the
        graduated body the per-component PCC test pinned. They exist so the
        e2e pipeline can drive ONE attention body in both modes instead of
        forking a second one:

          * `kv_cache=(k_cache, v_cache), cache_fill=True`  -> prefill. The
            rotated K/V are written into the resident cache with
            `ttnn.fill_cache` and the unchanged full-sequence SDPA runs.
          * `kv_cache=(k_cache, v_cache)` plus `cache_pos=p` (a python row
            index) or `cache_pos_tensor=<int32 [1] device tensor>` (the resident
            index, which is what a traced step needs) -> one decode step. The
            rotated K/V for the single new token are written at that row and
            attention reads the resident cache via
            `ttnn.transformer.scaled_dot_product_attention_decode`, so the step
            is O(1) and never recomputes the prompt.

        Both branches are ttnn-only; no torch op is reachable from here.
        """
        if position_embeddings is None:
            raise RuntimeError(
                "attention forward needs position_embeddings=(cos, sin); the harness must pass the "
                "same pair it gave the HF reference"
            )
        cos, sin = position_embeddings
        cos = self._as_broadcastable(cos)
        sin = self._as_broadcastable(sin)

        mm = {"compute_kernel_config": self.compute_kernel_config} if self.compute_kernel_config else {}

        # Q, K and V all read the SAME activation, so each DISTINCT shard among
        # their plans is opened once and reused. Q wants 32 cores and K/V want
        # 16 (each measured separately), so that is two conversions rather than
        # three -- and one if a future retune brings them back onto one grid.
        if self._decode_heads and self._decode_native_applies(hidden_states, cos, kv_cache, cache_pos_tensor):
            try:
                return self._decode_native(hidden_states, cos, sin, kv_cache, cache_pos_tensor)
            except Exception:  # noqa: BLE001 - op/shape refused: generic path below
                self._decode_heads = False

        rope_applied = False
        if self.qkv_plan is not None and self.qkv_plan.matches(hidden_states):
            # ONE fused projection, then split. The three weights are contiguous
            # in one tensor, so this is a single wide read instead of three.
            fused = self.qkv_plan(hidden_states, self.wqkv, self.compute_kernel_config)
            rows = int(fused.shape[-2])
            # Q and K are ADJACENT in the fused tensor and RoPE is applied
            # per-head with the SAME cos/sin, so rotating the 40 q+kv heads as
            # one tensor is arithmetically identical to two rotations -- and it
            # is one dispatch instead of two. That matters because rotary is
            # bound_by=dispatch here: 2 per layer x 26 layers = 52 launches per
            # decoded token against a few KB each, so the count IS the cost.
            # Slicing q and k apart afterwards costs 2 ops but saves 1 slice and
            # a whole reshape+permute pair up front, so the block is 2 ops
            # lighter overall on top of halving the rotary count.
            qk = self._split_heads(
                ttnn.slice(fused, [0, 0, 0], [1, rows, self._k_end]), self.n_heads + self.n_kv_heads
            )
            qk = self._apply_rope(qk, cos, sin)
            query = ttnn.slice(qk, [0, 0, 0, 0], [1, self.n_heads, rows, self.head_dim])
            key = ttnn.slice(
                qk, [0, self.n_heads, 0, 0], [1, self.n_heads + self.n_kv_heads, rows, self.head_dim]
            )
            value = self._split_heads(
                ttnn.slice(fused, [0, 0, self._k_end], [1, rows, self._v_end]), self.n_kv_heads
            )
            rope_applied = True
        elif self.q_plan is not None and self.q_plan.matches(hidden_states) and self.kv_plan is not None:
            ckc = self.compute_kernel_config
            q_shard = self.q_plan.shard_input(hidden_states)
            kv_shard = q_shard if self.q_plan.shares_input_with(self.kv_plan) else self.kv_plan.shard_input(
                hidden_states
            )
            query = self._split_heads(self.q_plan.run_presharded(q_shard, self.wq, ckc), self.n_heads)
            key = self._split_heads(self.kv_plan.run_presharded(kv_shard, self.wk, ckc), self.n_kv_heads)
            value = self._split_heads(self.kv_plan.run_presharded(kv_shard, self.wv, ckc), self.n_kv_heads)
        else:
            query = self._split_heads(self._q_proj(hidden_states, mm), self.n_heads)
            key = self._split_heads(self._kv_proj(hidden_states, self.wk, mm), self.n_kv_heads)
            value = self._split_heads(self._kv_proj(hidden_states, self.wv, mm), self.n_kv_heads)

        if not rope_applied:
            query = self._apply_rope(query, cos, sin)
            key = self._apply_rope(key, cos, sin)

        if kv_cache is not None and (cache_pos is not None or cache_pos_tensor is not None):
            attn = self._attend_from_cache(query, key, value, kv_cache, cache_pos, cache_pos_tensor)
        else:
            if kv_cache is not None and cache_fill:
                k_cache, v_cache = kv_cache
                # [1, n_kv, s, hd] straight into the [1, n_kv, C, hd] cache;
                # rows [0:s) are written exactly, the rest stay as they were.
                ttnn.fill_cache(k_cache, key, 0)
                ttnn.fill_cache(v_cache, value, 0)
            # ttnn's SDPA is GQA-aware (nqh vs nkv), so no explicit repeat_kv.
            # The mask the harness hands us IS the causal mask the HF golden used,
            # so honour it instead of asking the kernel for its own.
            attn = ttnn.transformer.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                is_causal=attention_mask is None,
                scale=self.scaling,
                **mm,
            )
            attn = ttnn.permute(attn, (0, 2, 1, 3))

        shape = list(attn.shape)
        merged = (shape[0], shape[1], self.n_heads * self.head_dim)
        if self.o_plan is not None and self.o_plan.is_decode_row(attn):
            # The head-merge reshape is o_proj's only producer, so it emits the
            # merged tensor ALREADY in o_proj's shard instead of handing over an
            # interleaved one that then needs converting.
            merged_attn = ttnn.reshape(attn, merged, memory_config=self.o_plan.input_memory_config)
            return self.o_plan.run_presharded(merged_attn, self.wo, self.compute_kernel_config)
        attn = ttnn.reshape(attn, merged)
        if self.o_plan is not None and self.o_plan.matches(attn):
            return self.o_plan(attn, self.wo, self.compute_kernel_config)
        return ttnn.linear(attn, self.wo, **mm)

    # ------------------------------------------------- decode-native layout
    def _decode_native_applies(self, hidden_states, cos, kv_cache, cache_pos_tensor) -> bool:
        """True when this call is the traced decode step this path is built for.

        Needs the fused projection (so there IS a single qkv tensor to split),
        the resident cache with a DEVICE row index (the traced decode contract),
        and a cos table already replicated to at least `n_heads` rows -- see
        `_decode_native` for why the rotation reads it row-wise here.
        """
        return (
            self.qkv_plan is not None
            and self.o_plan is not None
            and kv_cache is not None
            and cache_pos_tensor is not None
            and self.qkv_plan.matches(hidden_states)
            and int(cos.shape[-2]) >= self.n_heads
        )

    def _decode_native(self, hidden_states, cos, sin, kv_cache, cache_pos_tensor):
        """One decode token, in the layout the decode kernels already want.

        The generic path shapes Q/K/V as `[1, heads, seq, hd]` because that is
        what a PREFILL SDPA reads. Nothing in a decode step reads it:
        `paged_update_cache` wants `[1, batch, kv_heads, hd]` and
        `scaled_dot_product_attention_decode` wants `[1, batch, q_heads, hd]`.
        Getting from one to the other costs a reshape, a permute and a slice per
        tensor, two more permutes for the cache writes and one back after the
        attention -- about nine pure-layout ops per layer, each launching a
        kernel to move a few KB. At 26 layers that is the single largest
        non-matmul cost in the step, and the profiler tags every one of those ops
        `bound_by=dispatch`: their cost IS the launch.

        `nlp_create_qkv_heads_decode` produces the decode layout DIRECTLY from
        the fused projection, L1-sharded one batch entry per core, so the whole
        chain collapses to one op -- and its outputs are then already exactly
        what the cache write and the decode SDPA take, with no permute at either
        end. `paged_fused_update_cache` writes K and V in a single launch on top
        of that.

        The rotation stays in PREFILL mode. In this layout the heads sit on the
        row axis, so the op walks cos/sin row-wise -- which is why the caller
        replicates the single position's cos/sin up the tile ONCE PER TOKEN
        (`_decode_native_applies` refuses the path otherwise). Every head rotates
        by the same angle, so every row it reads is the same row: identical
        arithmetic to broadcasting it, and the rotation now touches 4 tiles
        instead of 160 (31 of every 32 rows in the old layout were tile padding).
        """
        k_cache, v_cache = kv_cache
        query, key, value = self._create_heads(hidden_states)
        query = self._apply_rope(query, cos, sin)
        key = self._apply_rope(key, cos, sin)
        self._write_kv(k_cache, key, v_cache, value, cache_pos_tensor)
        attn = ttnn.transformer.scaled_dot_product_attention_decode(
            query,
            k_cache,
            v_cache,
            scale=self.scaling,
            cur_pos_tensor=cache_pos_tensor,
            program_config=self.sdpa_decode_program_config,
        )
        # Already [1, 1, q_heads, hd]: merging the heads is the last two dims,
        # emitted straight into o_proj's shard as the generic path also does.
        merged = ttnn.reshape(
            attn, (1, 1, self.n_heads * self.head_dim), memory_config=self.o_plan.input_memory_config
        )
        return self.o_plan.run_presharded(merged, self.wo, self.compute_kernel_config)

    def _write_kv(self, k_cache, key, v_cache, value, cache_pos_tensor):
        """Write this token's K and V into the resident caches.

        K and V arrive from head creation ALREADY in the sharded layout the
        cache write takes, so neither needs the permute + reshard the generic
        path pays to get there. What they do NOT arrive as is disjoint: batch 1
        puts both on core (0,0), and the one-launch fused write refuses
        overlapping operands. Moving V one core over buys the fused write --
        two launches of ~8 us each, writing 2 KB, become one plus a 2 KB shard
        move, and at 26 layers those launches are what the write costs, not the
        bytes.
        """
        if self._fused_cache_write:
            try:
                ttnn.experimental.paged_fused_update_cache(
                    k_cache,
                    key,
                    v_cache,
                    ttnn.to_memory_config(value, self.kv_second_core_memory_config),
                    update_idxs_tensor=cache_pos_tensor,
                )
                return
            except Exception:  # noqa: BLE001 - refused: two separate writes
                self._fused_cache_write = False
        ttnn.experimental.paged_update_cache(k_cache, key, update_idxs_tensor=cache_pos_tensor)
        ttnn.experimental.paged_update_cache(v_cache, value, update_idxs_tensor=cache_pos_tensor)

    def _create_heads(self, hidden_states):
        """The fused projection, split into decode-layout Q/K/V.

        Fed the WIDTH-SHARDED matmul result rather than an interleaved copy of
        it. Head creation is batch-parallel, so at batch 1 its outputs live on
        ONE core -- and handed an interleaved operand that one core has to pull
        the whole fused row (192 tiles) out of DRAM by itself, which is what
        makes this op cost ~33us for 12 tiles of output. The projection has just
        written that row across 32 cores' L1, and the op's sharded program
        factory reads a width-sharded operand from exactly there, so feeding it
        directly turns a single-core DRAM read into a fan-in over L1 and drops
        the interleaved conversion on the way.

        The shard is only legal for the op when it is WIDTH_SHARDED with one
        full tile-row per core, ROW_MAJOR -- which is what the plan builds -- so
        a refusal falls back to the interleaved feed for the rest of the run
        rather than raising inside a captured trace.
        """
        shard = self.qkv_plan.shard_input(hidden_states)
        if self._presharded_heads:
            try:
                fused = self.qkv_plan.run_presharded_raw(shard, self.wqkv, self.compute_kernel_config)
                return self._split_qkv(fused)
            except Exception:  # noqa: BLE001 - shard refused: interleaved feed
                self._presharded_heads = False
        return self._split_qkv(self.qkv_plan.run_presharded(shard, self.wqkv, self.compute_kernel_config))

    def _split_qkv(self, fused):
        """[1, tile, qkv] -> decode-layout q/k/v.

        The reshape only adds the leading batch axis the head-creation op reads
        `num_users` from; the padded form keeps the tile row the projection
        actually wrote, so it is a view, not a copy.
        """
        fused = ttnn.reshape(fused, (1, 1, 1, self._v_end), (1, 1, _TILE, self._v_end))
        return ttnn.experimental.nlp_create_qkv_heads_decode(
            fused, num_heads=self.n_heads, num_kv_heads=self.n_kv_heads
        )

    def _write_cache_row(self, cache, tensor, cache_pos, cache_pos_tensor):
        """Write ONE token's K or V into the resident cache.

        With a device index tensor the write goes through
        `ttnn.experimental.paged_update_cache`, which reads the row number out of
        a `[1]` int32 tensor — so a captured trace advances by itself instead of
        being pinned to whatever row was current at capture time. It wants the
        `[1, batch, n_kv, hd]` layout, sharded, hence the permute + reshard.
        With a python `cache_pos` the plain `ttnn.update_cache` takes the
        `[1, n_kv, 1, hd]` layout `_split_heads` already produced.
        """
        if cache_pos_tensor is not None and self.kv_write_memory_config is not None:
            row = ttnn.permute(tensor, (2, 0, 1, 3))
            ttnn.experimental.paged_update_cache(
                cache,
                ttnn.to_memory_config(row, self.kv_write_memory_config),
                update_idxs_tensor=cache_pos_tensor,
            )
        else:
            ttnn.update_cache(cache, tensor, cache_pos)

    def _attend_from_cache(self, query, key, value, kv_cache, cache_pos, cache_pos_tensor):
        """One decode step against the resident cache -> (1, 1, n_heads, hd).

        The decode SDPA kernel wants `[1, batch, n_q_heads, hd]`, hence the
        (2, 0, 1, 3) permute in and the (1, 0, 2, 3) permute back out.
        """
        k_cache, v_cache = kv_cache
        self._write_cache_row(k_cache, key, cache_pos, cache_pos_tensor)
        self._write_cache_row(v_cache, value, cache_pos, cache_pos_tensor)

        q_decode = ttnn.permute(query, (2, 0, 1, 3))
        if cache_pos_tensor is not None:
            pos_kwargs = {"cur_pos_tensor": cache_pos_tensor}
        else:
            pos_kwargs = {"cur_pos": [cache_pos]}
        attn = ttnn.transformer.scaled_dot_product_attention_decode(
            q_decode,
            k_cache,
            v_cache,
            scale=self.scaling,
            program_config=self.sdpa_decode_program_config,
            **pos_kwargs,
        )
        return ttnn.permute(attn, (1, 0, 2, 3))


# Module-level `build` — primary test entry point.
def build(device, torch_module=None):
    return TtAttention.build(device, torch_module)


# Legacy slug-named shim.
def attention(device, torch_module=None):
    return TtAttention.build(device, torch_module)
