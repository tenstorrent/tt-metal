# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4 sliding-only attention (``layer_types[i] == "sliding_attention"``), chunked prefill.

The reference builds all three attention kinds from one class: a sliding layer is
``DeepseekV4Attention`` with ``compressor = None`` and ``rope_layer_type = "main"`` (plain theta=10000
instead of the YaRN-scaled table HCA/CSA share with their compressor). Everything else -- the Q/KV
stems, the per-head sink, the grouped output projection -- is identical, so those come from
:class:`TtHCA` unchanged and only the attention core differs. Flash has two such layers (0 and 1); Pro
has none.

The attention core is where a sliding layer stops looking like HCA. HCA all-gathers the K/V and then
attends over ``[carry | chunk | compressed]``, so every chip sees ``Sk`` = the whole slab. A 128-token
window never reaches further back than the 128 rows before a chip's own first query, so each chip
attends over ``[halo | its own rows]`` instead -- ``Sk`` drops from 5248 to 768 and the SDPA walks 6
K-chunks per Q-chunk instead of 41.

``halo`` is the 128 rows preceding this chip's block, which is a DIFFERENT global offset per chip. That
comes from ``ttnn.slice`` with sequence-parallel-sharded index tensors: the bounds live in device
tensors, so only their shape is in the program and each chip reads its own offset -- verified on 8x4 as
one program serving all eight offsets.

Two things this deliberately does NOT do, both measured or reasoned rather than assumed:

* It keeps HCA's full K/V all-gather. The attention only needs each chip's own rows plus a 128-row
  halo, so gathering just the per-chip tails would cut that CCL ~5x. But a ragged chunk's carry is the
  global ``[real_len - 128, real_len)`` block, which a tails-only gather does not contain, and the
  gathered slab is what makes ``_carry_index`` work. The compute win is independent of this and is the
  larger one.
* It masks rather than passing ``sliding_window_size``. The op's band comes from LOCAL Q/K indices, so
  it needs Q front-padded by the window to line up, and then chunk 0's chip 0 has no real halo -- 128
  zero KEY rows, which a zero-value key still adds to the softmax denominator, and
  ``compute_streaming.hpp``'s ``static_assert`` forbids combining a mask with the window to -inf them.
  A mask handles that as data (one SP-sharded tensor, chip 0 differing only on the first chunk) and
  keeps one program. It costs the loop narrowing: 6 K-chunks instead of 2, ~150 us per chunk per layer
  against a ~200 ms whole-model chunk.
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import DeepseekV4RotaryEmbedding
from models.demos.deepseek_v3_d_p.tt.mla.compressor import TtCompressorUtils
from models.demos.deepseek_v3_d_p.tt.mla.heavily_compressed_attention import TtHCA


class TtSWAState:
    """Chunked-prefill state, owned by the caller and passed to ``TtSWA.forward``.

    One tensor, and it never grows: a sliding layer's whole history is the 128 tokens before the next
    chunk starts. ``kv_actual`` says how many real tokens precede this chunk, which is what selects the
    first-chunk mask variant and the rotary offset."""

    def __init__(self, carry, max_seq_len):
        self.carry = carry  # [B, 1, sliding_window, head_dim]
        self.max_seq_len = int(max_seq_len)
        self.kv_actual = 0


class TtSWA(LightweightModule):
    def __init__(
        self,
        device,
        *,
        q_a_proj_weight: torch.Tensor,
        q_a_norm_weight: torch.Tensor,
        q_b_proj_weight: torch.Tensor,
        kv_proj_weight: torch.Tensor,
        kv_norm_weight: torch.Tensor,
        sinks: torch.Tensor,
        o_a_proj_weight: torch.Tensor,
        o_b_proj_weight: torch.Tensor,
        rotary_emb,
        num_heads: int,
        head_dim: int,
        rope_head_dim: int,
        sliding_window: int,
        o_groups: int,
        rms_norm_eps: float = 1e-6,
        sp_axis: int = 0,
        tp_axis: int = 1,
        topology=ttnn.Topology.Linear,
        dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ):
        # The stems, the sink packing and the grouped o_proj are the reference's shared half, so they
        # come from a TtHCA built with compressor=None rather than copied. Only _attention and the
        # state differ, and both live here.
        self._hca = TtHCA(
            device,
            compressor=None,
            q_a_proj_weight=q_a_proj_weight,
            q_a_norm_weight=q_a_norm_weight,
            q_b_proj_weight=q_b_proj_weight,
            kv_proj_weight=kv_proj_weight,
            kv_norm_weight=kv_norm_weight,
            sinks=sinks,
            o_a_proj_weight=o_a_proj_weight,
            o_b_proj_weight=o_b_proj_weight,
            rotary_emb=rotary_emb,
            num_heads=num_heads,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            sliding_window=sliding_window,
            o_groups=o_groups,
            rms_norm_eps=rms_norm_eps,
            sp_axis=sp_axis,
            tp_axis=tp_axis,
            topology=topology,
            dtype=dtype,
            weights_dtype=weights_dtype,
            memory_config=memory_config,
        )
        self.device = device
        self.dtype = dtype
        self.memory_config = memory_config
        self.head_dim = int(head_dim)
        self.rope_head_dim = int(rope_head_dim)
        self.sliding_window = int(sliding_window)
        self.scaling = self.head_dim**-0.5
        self.sp_axis, self.tp_axis = sp_axis, tp_axis
        self.sp_factor = self._hca.sp_factor
        self.tp_factor = self._hca.tp_factor
        self.ops: TtCompressorUtils = self._hca.ops

        # Everything below comes from alloc_state, which every caller has to run before forward.
        self._mask = None  # persistent additive mask; forward overwrites only the halo columns
        self._head_cols = None  # the halo columns, in a normal and a first-chunk variant
        self._halo_bounds = None  # SP-sharded slice bounds: this chip's 128 rows of history
        self._slab_rope = None
        self._slab_index = None
        self._carry_index = None

    @classmethod
    def from_reference(cls, device, reference, config, **kwargs) -> "TtSWA":
        assert getattr(reference, "compressor", None) is None, (
            "TtSWA expects a sliding-only reference layer (compressor is None); a layer with a "
            "compressor is HCA or CSA"
        )
        # A sliding layer has no compressor to borrow a rotary_emb from -- and unlike HCA it does not
        # need one, since its rope is the plain theta=10000 variant the config alone determines.
        rotary_emb = getattr(reference, "rotary_emb", None) or DeepseekV4RotaryEmbedding(config)
        return cls(
            device,
            q_a_proj_weight=reference.q_a_proj.weight,
            q_a_norm_weight=reference.q_a_norm.weight,
            q_b_proj_weight=reference.q_b_proj.weight,
            kv_proj_weight=reference.kv_proj.weight,
            kv_norm_weight=reference.kv_norm.weight,
            sinks=reference.sinks,
            o_a_proj_weight=reference.o_a_proj.weight,
            o_b_proj_weight=reference.o_b_proj.weight,
            rotary_emb=rotary_emb,
            num_heads=config.num_attention_heads,
            head_dim=config.head_dim,
            rope_head_dim=config.qk_rope_head_dim,
            sliding_window=config.sliding_window,
            o_groups=config.o_groups,
            rms_norm_eps=config.rms_norm_eps,
            **kwargs,
        )

    def alloc_state(self, max_seq_len: int, batch: int = 1, chunk_tokens: int | None = None) -> TtSWAState:
        """Size the state and build every host tensor forward will need, so forward builds none.

        ``chunk_tokens`` is the slab width forward will be called with; the mask and the slice bounds
        are sized from it and are the same for every chunk."""
        assert batch == 1, f"SWA state is single-user for now, got batch={batch}"
        chunk = int(chunk_tokens or max_seq_len)
        sw, sp = self.sliding_window, self.sp_factor
        assert chunk % (sw * sp) == 0, (
            f"the slab is {chunk} wide, which is not a multiple of sliding_window * sp_factor "
            f"({sw} * {sp} = {sw * sp}); every chip's share has to be a whole number of windows"
        )
        seq_local = chunk // sp

        self._build_masks(seq_local)
        self._build_halo_bounds(chunk, seq_local)
        self._build_carry_index(chunk)
        # A row per TOKEN, and the "main" rope variant: a sliding layer does not share the compressor's
        # YaRN-scaled table (reference DeepseekV4Attention.rope_layer_type).
        self._slab_rope = self.ops.build_rope_table(int(max_seq_len) + chunk, 1, layer_type="main")
        self._slab_index = self.ops.rope_index_base(seq_local)
        return TtSWAState(
            carry=self.ops.from_torch(torch.zeros(batch, 1, sw, self.head_dim)),
            max_seq_len=max_seq_len,
        )

    def _build_masks(self, seq_local: int):
        """The additive mask over the key layout ``[halo | own rows]``, both chip-local.

        Key column ``j`` holds the token at global position ``base - sw + j`` and query row ``i`` the one
        at ``base + i``, where ``base`` is this chip's first token. The causal window
        ``base + i - sw < pos <= base + i`` then reduces to ``i < j <= i + sw`` -- the chip offset
        cancels, so ONE replicated mask serves every chip.

        The halo columns need a second version for the first chunk, where the carry holds no tokens yet.
        Only chip 0 reads the carry -- chips 1..sp-1 take their halo from a predecessor inside this same
        chunk -- so that version differs per chip, and ``log(nez(chip_index))`` turns the chip index into
        the 0 / -inf it needs without a host tensor.

        Built from index vectors so no large host tensor is created. They are float32 because
        bfloat16 stops being exact above 256, and the column index runs to 768."""
        sw = self.sliding_window
        width = sw + seq_local

        ic = self.ops.from_torch(torch.arange(seq_local).float().view(1, 1, seq_local, 1), dtype=ttnn.float32)
        jc = self.ops.from_torch(torch.arange(width).float().view(1, 1, 1, width), dtype=ttnn.float32)
        band = ttnn.typecast(ttnn.log(ttnn.multiply(ttnn.gt(jc, ic), ttnn.le(jc, ttnn.add(ic, float(sw))))), self.dtype)
        self._mask = band

        head = ttnn.slice(band, [0, 0, 0, 0], [1, 1, seq_local, sw])
        chip = self.ops.from_torch(
            torch.arange(self.sp_factor).float().view(1, 1, self.sp_factor, 1),
            self.ops.mesh_mapper(sp_dim=2),
            dtype=ttnn.float32,
        )
        not_chip_0 = ttnn.typecast(ttnn.log(ttnn.nez(chip)), self.dtype)  # 0 on chips 1.., -inf on chip 0
        self._head_cols = {False: head, True: ttnn.add(head, not_chip_0)}

    def _build_halo_bounds(self, chunk: int, seq_local: int):
        """Slice bounds for this chip's halo: rows ``[d*seq_local, d*seq_local + sw)`` of
        ``[carry | gathered chunk]``.

        The offset differs per chip, so the bounds are SP-sharded device tensors -- ``ttnn.slice`` reads
        them at runtime, which keeps only their shape in the program and lets one program serve all
        ``sp`` offsets. They must be 1-D, hence the flat layout."""
        sw, sp = self.sliding_window, self.sp_factor
        starts = torch.tensor([v for d in range(sp) for v in (0, 0, d * seq_local, 0)], dtype=torch.int32)
        ends = torch.tensor(
            [v for d in range(sp) for v in (1, 1, d * seq_local + sw, self.head_dim)], dtype=torch.int32
        )
        mapper = self.ops.mesh_mapper(sp_dim=0)
        self._halo_bounds = tuple(
            self.ops.from_torch(t, mapper, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT) for t in (starts, ends)
        )
        # ttnn.slice sizes its output as in_rows / num_devices, so this is a divisor and not a mesh size.
        self._halo_num_devices = (sw + chunk) // sw

    def _build_carry_index(self, chunk: int):
        """start/end index tensors for the carry slice, one pair per real_len a chunk can have.

        The carry is the GLOBAL ``[real_len - sw, real_len)`` block, not the last rows of the padded
        slab, and those differ as soon as ``real_len < chunk``. The bounds are the same on every chip
        (the gathered slab is replicated), so unlike the halo these are not sharded."""
        sw = self.sliding_window

        def idx(vals):
            return self.ops.from_torch(
                torch.tensor(vals, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
            )

        self._carry_index = {
            real_len: (idx([0, 0, real_len - sw, 0]), idx([1, 1, real_len, self.head_dim]))
            for real_len in range(sw, int(chunk) + 1, sw)
        }

    def _attention(self, q, sliding_kv, cos, sin, carry, kv_actual: int, real_len: int):
        """``q`` [B, H/tp, S/sp, head_dim] and this chip's own ``sliding_kv`` [B, 1, S/sp, head_dim]
        -> ``(attn, next_carry)``.

        The gather is over the WHOLE chunk even though attention reads 768 rows of it, because the
        carry slice needs the global slab; see the module docstring."""
        batch, num_heads_local, seq_local, _ = q.shape
        sw = self.sliding_window

        if self.sp_factor > 1:
            gathered = ttnn.experimental.all_gather_async(
                sliding_kv,
                dim=2,
                multi_device_global_semaphore=self._hca.tt_ccl.get_and_cycle_ag_semaphore_handles(
                    cluster_axis=self.sp_axis
                ),
                barrier_semaphore=self._hca.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=self.sp_axis),
                num_links=self._hca.ccl_num_links,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=self._hca.tp_ccl_topology,
                cluster_axis=self.sp_axis,
            )
        else:
            gathered = sliding_kv

        # [carry | chunk] is the global key sequence this chunk can see; each chip takes the 128 rows
        # that precede its own block out of it.
        hist = ttnn.concat([carry, gathered], dim=2)
        start, end = self._halo_bounds
        halo = ttnn.slice(hist, start, end, slice_dim=2, num_devices=self._halo_num_devices)
        kv = ttnn.concat([halo, sliding_kv], dim=2)

        # Only the halo columns move between chunks, and always over the same range, so the mask is
        # built once and this overwrites that range in place.
        head_cols = self._head_cols[kv_actual == 0]
        ttnn.experimental.slice_write(
            head_cols, self._mask, start=[0, 0, 0, 0], end=[1, 1, seq_local, sw], step=[1, 1, 1, 1]
        )

        attn = ttnn.transformer.scaled_dot_product_attention(
            q,
            kv,
            kv,
            attn_mask=self._mask,
            is_causal=False,
            scale=self.scaling,
            attention_sink=self._hca.sinks_sdpa,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=self.device.compute_with_storage_grid_size(),
                q_chunk_size=128,
                k_chunk_size=128,
                exp_approx_mode=False,
            ),
        )

        # The next chunk's window reaches back into this one, so the carry is this chunk's last REAL
        # keys. Taking the start from a device tensor keeps only its shape in the program, so one
        # program serves every real_len.
        c_start, c_end = self._carry_index[real_len]
        next_carry = ttnn.slice(gathered, c_start, c_end, slice_dim=2, num_devices=gathered.shape[2] // sw)

        nope_dim = self.head_dim - self.rope_head_dim
        nope = ttnn.slice(attn, [0, 0, 0, 0], [batch, num_heads_local, seq_local, nope_dim])
        rope = ttnn.slice(attn, [0, 0, 0, nope_dim], [batch, num_heads_local, seq_local, self.head_dim])
        # Undoing V's RoPE is the same rotation with the sign of sin flipped, so cos is reused.
        rope = ttnn.experimental.rotary_embedding_llama(
            rope, cos, ttnn.neg(sin), self._hca.trans_mat, is_decode_mode=False
        )
        return ttnn.concat([nope, rope], dim=-1), next_carry

    def forward(self, hidden_states, seq_len_actual: int | None = None, state: TtSWAState = None):
        """[B, 1, S/sp, hidden/tp] -> the same, one chunk. ``seq_len_actual`` is the chunk's real token
        count; the rest of the slab is padding the mask never lets a real query read."""
        assert state is not None, "TtSWA.forward needs the state alloc_state returned"
        seq_len = hidden_states.shape[2] * self.sp_factor
        real_len = int(seq_len if seq_len_actual is None else seq_len_actual)
        sw = self.sliding_window
        assert real_len % sw == 0, (
            f"a chunk must end on a window boundary, got {real_len} with sliding_window {sw}: the next "
            f"chunk's first query needs the {sw} tokens before it, which is what the carry holds"
        )

        slab_index = self.ops.rope_index(self._slab_index, state.kv_actual)
        cos, sin = self.ops.rope_gather(self._slab_rope, slab_index)
        q = self._hca._q_stem(hidden_states, cos, sin)
        sliding_kv = self._hca._kv_stem(hidden_states, cos, sin)

        attn, next_carry = self._attention(
            q, sliding_kv, cos, sin, carry=state.carry, kv_actual=state.kv_actual, real_len=real_len
        )

        state.kv_actual += real_len
        state.carry = next_carry
        return self._hca._o_proj(attn)
