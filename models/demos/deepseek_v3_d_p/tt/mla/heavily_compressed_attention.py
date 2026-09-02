# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4 Heavily Compressed Attention (TTNN prefill).
Mirrors ``DeepseekV4Attention`` in ``reference/deepseek_v4/modeling_deepseek_v4.py``."""

from __future__ import annotations

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.tt.mla.compressor import TtCompressorUtils, TtHCACompressor, rope_table_tokens
from models.demos.deepseek_v3_d_p.tt.mla.rope import get_rot_transformation_mat
from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache


def _cache_write_rows(chunk_entries: int) -> int:
    """How many cache rows one write covers: this chunk's new entries plus the 0..31 rows already sitting in
    the last tile. The cache is written a whole tile at a time, so this rounds up to whole tiles -- 64 rows
    for a 4096-token chunk, 96 for 5120."""
    return -(-(ttnn.TILE_SIZE - 1 + chunk_entries) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE


class TtHCAState:
    """Chunked-prefill state, owned by the caller and passed to ``TtHCA.forward``.

    The device tensors keep a FIXED shape for the whole prefill and only their contents advance --
    that is what lets one compiled program serve every chunk. The counters say how much is real;
    the attention mask -infs the rest."""

    def __init__(self, compressed_kv, sliding_carry, tail, max_seq_len):
        self.compressed_kv = compressed_kv  # [B, 1, compressed_capacity, head_dim]
        self.sliding_carry = sliding_carry  # [B, 1, sliding_window, head_dim]
        # The cache's last tile is usually only partly filled. Its entries sit here, right-aligned, so the
        # next write can place them and the new entries with a single shift.
        self.tail = tail  # [B, 1, TILE_SIZE, head_dim]
        self.max_seq_len = int(max_seq_len)
        self.entry_count = 0
        self.kv_actual = 0


class TtHCA(LightweightModule):
    """HCA block: query/kv stems + compressor + attention core + grouped output projection.

    Block I/O is ``[B, 1, S/sp, hidden/tp]``, so layers chain without a reshard."""

    def __init__(
        self,
        device,
        *,
        compressor: TtHCACompressor,
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
        self.device = device
        self.dtype = dtype
        self.weights_dtype = weights_dtype
        self.memory_config = memory_config
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.rope_head_dim = int(rope_head_dim)
        self.sliding_window = int(sliding_window)
        self.scaling = self.head_dim**-0.5
        self.rotary_emb = rotary_emb
        self.rms_norm_eps = float(rms_norm_eps)
        self.compressor = compressor

        self.sp_axis, self.tp_axis = sp_axis, tp_axis
        self.sp_factor = device.shape[sp_axis]
        self.tp_factor = device.shape[tp_axis]
        self.tp_ccl_topology = topology
        self.tt_ccl = get_tt_ccl(device) if (self.sp_factor > 1 or self.tp_factor > 1) else None
        self.ccl_num_links = 2 if is_blackhole() else 1
        self.ops = TtCompressorUtils(
            device,
            rotary_emb=rotary_emb,
            sp_axis=sp_axis,
            tp_axis=tp_axis,
            dtype=dtype,
            weights_dtype=weights_dtype,
            memory_config=memory_config,
        )

        # Pre-divided by scale: SDPA scales BOTH QK and the sink internally, the reference scales only
        # QK -- dividing here cancels the kernel's extra multiply. TP-sharded to match the query heads.
        sinks_host = sinks.detach().reshape(1, self.num_heads, 1, 1) / self.scaling
        self.sinks_sdpa = self.ops.from_torch(sinks_host, mesh_mapper=self.ops.mesh_mapper(tp_dim=1))

        self.wq_a = self.ops.to_tt_linear_weight(q_a_proj_weight, tp_shard_dim=2)
        self.wq_b = self.ops.to_tt_linear_weight(q_b_proj_weight, tp_shard_dim=3)
        self.q_a_norm_weight = self.ops.from_torch(q_a_norm_weight.detach().reshape(1, 1, 1, -1))
        self.q_b_norm_weight = self.ops.from_torch(torch.ones(1, 1, 1, self.head_dim))
        self.wkv = self.ops.to_tt_linear_weight(kv_proj_weight, tp_shard_dim=2)
        self.kv_norm_weight = self.ops.from_torch(kv_norm_weight.detach().reshape(1, 1, 1, self.head_dim))

        # o_a_proj is block-diagonal over o_groups. Groups partition the heads, so a TP chip owns whole
        # groups: keep it as ONE batched weight sharded on the group axis and run a single batched
        # matmul -- each chip applies only its own groups, no collective
        self.o_groups = int(o_groups)
        in_per_group = self.num_heads * self.head_dim // self.o_groups
        o_a_grouped = o_a_proj_weight.detach().view(self.o_groups, -1, in_per_group).transpose(1, 2).unsqueeze(0)
        self.wo_a = self.ops.from_torch(
            o_a_grouped, mesh_mapper=self.ops.mesh_mapper(tp_dim=1), dtype=self.weights_dtype
        )
        self.wo_b = self.ops.to_tt_linear_weight(o_b_proj_weight, tp_shard_dim=2)

        self.trans_mat = self.ops.from_torch(get_rot_transformation_mat())
        # Everything below comes from alloc_state, which every caller has to run before forward.
        self._shift_cache = {}
        self._take_cache = {}
        self._carry_index = {}
        self._slab_rope = None
        self._slab_index = None
        self._mask = None  # persistent additive mask; forward overwrites only the moving columns
        self._mask_col = None
        self._carry_cols = None
        self._kv_pad = None  # zero rows that bring Sk up to a tile multiple

    def _build_masks(self, seq_global: int, cap: int):
        """The persistent additive mask over the key layout ``[carry | chunk | compressed | pad]``. Carry,
        chunk and pad columns are the same for every chunk and are written here; forward overwrites only
        the compressed columns, always at the same offset, so nothing recompiles.

        The carry columns come in two versions because the first chunk has no history: its carry holds
        zeros for positions that do not exist yet, so those columns must be -inf. Nothing else differs --
        in ``allowed = (j >= 0) & (j <= i) & (i - j < sw)`` the offset kv_actual cancels out of the last
        two conditions.

        Built from four small index vectors, so no large host tensor is created. They are float32 for the
        reason ``_build_mask_consts`` gives."""
        carry, sw = self.sliding_window, self.sliding_window
        raw = carry + seq_global
        sk_pad = -(-(raw + cap) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        sp_mapper = self.ops.mesh_mapper(sp_dim=2)

        ic = self.ops.from_torch(
            torch.arange(seq_global).float().view(1, 1, seq_global, 1), sp_mapper, dtype=ttnn.float32
        )
        ic_lo = self.ops.from_torch(
            (torch.arange(seq_global) - sw).float().view(1, 1, seq_global, 1), sp_mapper, dtype=ttnn.float32
        )
        jc = self.ops.from_torch((torch.arange(raw) - carry).float().view(1, 1, 1, raw), dtype=ttnn.float32)

        # j <= i  and  i - j < sw, both with kv_actual cancelled
        sliding = ttnn.typecast(ttnn.log(ttnn.multiply(ttnn.le(jc, ic), ttnn.gt(jc, ic_lo))), self.dtype)

        zero_seq = ttnn.multiply(ic, 0.0)
        blank = ttnn.typecast(
            ttnn.add(zero_seq, self.ops.from_torch(torch.zeros(1, 1, 1, cap), dtype=ttnn.float32)), self.dtype
        )
        parts = [sliding, blank]
        pad_w = sk_pad - raw - cap
        if pad_w:
            parts.append(
                ttnn.typecast(
                    ttnn.log(ttnn.add(zero_seq, self.ops.from_torch(torch.zeros(1, 1, 1, pad_w), dtype=ttnn.float32))),
                    self.dtype,
                )
            )
        self._mask = ttnn.concat(parts, dim=3)
        self._mask_col = raw
        # SDPA pads a non-tile-multiple Sk with zeros, and a supplied mask reads those columns as 0, which
        # means "attend". So the kv side gets real zero rows and the mask -infs their columns.
        sk = raw + cap
        self._kv_pad = self.ops.from_torch(torch.zeros(1, 1, sk_pad - sk, self.head_dim)) if sk_pad > sk else None

        # Both versions of the [seq, carry] slab are kept and forward writes the right one. Keeping a
        # second full mask instead cost 1.1 ms on device for the same information.
        self._carry_cols = {
            False: ttnn.slice(sliding, [0, 0, 0, 0], [1, 1, sliding.shape[2], carry]),
            True: ttnn.typecast(
                ttnn.log(ttnn.multiply(zero_seq, self.ops.from_torch(torch.zeros(1, 1, 1, carry), dtype=ttnn.float32))),
                self.dtype,
            ),
        }

    def alloc_state(self, max_seq_len: int, batch: int = 1, chunk_tokens: int | None = None) -> TtHCAState:
        """Size the state once for the longest context this layer will serve, so its shape is fixed for
        every chunk. ``max_seq_len`` is the longest context to serve, ``chunk_tokens`` the slab width
        forward will be called with (defaults to one chunk).
        Contents start zeroed and nothing is read before it is written -- the mask -infs everything past
        ``entry_count`` / ``kv_actual``, including chunk 0's empty carry.

        Every host tensor this layer will ever need is built here, which is what leaves forward with none.
        So the caller has to own the state: a prefill of one chunk allocates the same way a long one
        does."""
        entries = -(-int(max_seq_len) // self.compressor.compress_rate)
        capacity = -(-entries // ttnn.TILE_SIZE) * ttnn.TILE_SIZE  # cache writes land on tile boundaries
        # A write always rewrites whole tiles, so the last one can reach past the entries themselves.
        # ``chunk_tokens`` sizes that headroom.
        chunk = chunk_tokens or max_seq_len
        align = self.compressor.compress_rate * self.sp_factor
        assert chunk % align == 0, (
            f"the slab is {chunk} wide, which is not a multiple of compress_rate * sp_factor "
            f"({self.compressor.compress_rate} * {self.sp_factor} = {align}); every chip's share of it has "
            f"to end on a compression-window boundary. prepare_input rounds a raw prompt up to that."
        )
        width = -(-int(chunk) // self.compressor.compress_rate)
        capacity += _cache_write_rows(width)
        # Every one-hot the write can ever need: with chunks of differing real length r_e reaches every
        # value, so the whole TILE_SIZE set is built.
        self._build_tail_tile_matrices(width)
        self._build_carry_index(chunk)
        self._build_masks(chunk, capacity)
        self.compressor.alloc_tables(max_seq_len, chunk, capacity)

        # One rope table per state, so forward only gathers. This one has a row per TOKEN; the compressor
        # builds its own, with a row per ENTRY.
        self._slab_rope = self.ops.build_rope_table(rope_table_tokens(max_seq_len, chunk), 1)
        self._slab_index = self.ops.rope_index_base(chunk // self.sp_factor)
        # ``tail`` is not migrated, so it stays interleaved.
        return TtHCAState(
            compressed_kv=self._alloc_cache(batch, capacity),
            sliding_carry=self._alloc_cache(batch, self.sliding_window),
            tail=self.ops.from_torch(torch.zeros(batch, 1, ttnn.TILE_SIZE, self.head_dim)),
            max_seq_len=max_seq_len,
        )

    def _alloc_cache(self, batch: int, rows: int):
        """``[batch, 1, rows, head_dim]``, ND-sharded so the migration address table can name it.

        ``seq_len`` is pre-multiplied by sp_factor to cancel the division init_kvpe_cache does for its
        SP-sharded cache: HCA replicates both caches, so every chip holds all ``rows``."""
        return init_kvpe_cache(
            kvpe_cache_head_dim=self.head_dim,
            mesh_device=self.device,
            seq_len=rows * self.sp_factor,
            mesh_shape=list(self.device.shape),
            sp_axis=self.sp_axis,
            num_kvpe_cache_layers=1,
            num_users=batch,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
        )

    @classmethod
    def from_reference(cls, device, reference, config, **kwargs) -> "TtHCA":
        # Forward the mesh/CCL config so the compressor rides the same SP/TP axes as the block.
        compressor_keys = ("sp_axis", "tp_axis", "topology", "dtype", "weights_dtype", "memory_config")
        compressor = TtHCACompressor.from_reference(
            device, reference.compressor, config, **{k: kwargs[k] for k in compressor_keys if k in kwargs}
        )
        return cls(
            device,
            compressor=compressor,
            q_a_proj_weight=reference.q_a_proj.weight,
            q_a_norm_weight=reference.q_a_norm.weight,
            q_b_proj_weight=reference.q_b_proj.weight,
            kv_proj_weight=reference.kv_proj.weight,
            kv_norm_weight=reference.kv_norm.weight,
            sinks=reference.sinks,
            o_a_proj_weight=reference.o_a_proj.weight,
            o_b_proj_weight=reference.o_b_proj.weight,
            rotary_emb=reference.compressor.rotary_emb,
            num_heads=config.num_attention_heads,
            head_dim=config.head_dim,
            rope_head_dim=config.qk_rope_head_dim,
            sliding_window=config.sliding_window,
            o_groups=config.o_groups,
            rms_norm_eps=config.rms_norm_eps,
            **kwargs,
        )

    def _q_stem(self, hidden_states, cos, sin):
        """[B, 1, S/sp, hidden/tp] -> q [B, num_heads/tp, S/sp, head_dim]. ``cos``/``sin`` cover the padded
        slab and are built once per call: both stems and the output un-rope want the same rotation, and
        building it again costs ~2.9 ms of host time."""
        input_shape = tuple(hidden_states.shape)
        if len(input_shape) != 4 or input_shape[1] != 1:
            raise ValueError(f"Expected hidden_states shape [B, 1, S, hidden], got {input_shape}")
        batch, seq_len = input_shape[0], input_shape[2]
        num_heads_local = self.num_heads // self.tp_factor

        q = ttnn.linear(hidden_states, self.wq_a, memory_config=self.memory_config)

        # Row-parallel -> partial sums; all-reduce rebuilds the full q_lora, replicated across TP.
        if self.tp_factor > 1:
            q = ttnn.experimental.reduce_scatter_minimal_async(
                q,
                persistent_output_buffers=None,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis=self.tp_axis),
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=self.tp_axis),
                num_links=self.ccl_num_links,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=self.tp_ccl_topology,
                cluster_axis=self.tp_axis,
            )
            q = ttnn.experimental.all_gather_async(
                q,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=self.tp_axis),
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=self.tp_axis),
                num_links=self.ccl_num_links,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=self.tp_ccl_topology,
                cluster_axis=self.tp_axis,
            )

        q = ttnn.rms_norm(q, weight=self.q_a_norm_weight, epsilon=self.rms_norm_eps)
        q = ttnn.linear(q, self.wq_b, memory_config=self.memory_config)

        q, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            q,
            num_heads=num_heads_local,
            num_kv_heads=0,
            transpose_k_heads=False,
            memory_config=self.memory_config,
        )
        q = ttnn.rms_norm(q, weight=self.q_b_norm_weight, epsilon=self.rms_norm_eps)

        nope_dim = self.head_dim - self.rope_head_dim
        nope = ttnn.slice(q, [0, 0, 0, 0], [batch, num_heads_local, seq_len, nope_dim])
        rope = ttnn.slice(q, [0, 0, 0, nope_dim], [batch, num_heads_local, seq_len, self.head_dim])
        rope = ttnn.experimental.rotary_embedding_llama(rope, cos, sin, self.trans_mat, is_decode_mode=False)
        return ttnn.concat([nope, rope], dim=-1)

    def _kv_stem(self, hidden_states, cos, sin):
        """[B, 1, S/sp, hidden/tp] -> single-head sliding_kv [B, 1, S/sp, head_dim], TP-replicated.
        K == V in V4. Returns the full S; the sliding-window truncation is a chunked-prefill concern."""
        input_shape = tuple(hidden_states.shape)
        if len(input_shape) != 4 or input_shape[1] != 1:
            raise ValueError(f"Expected hidden_states shape [B, 1, S, hidden], got {input_shape}")
        batch, seq_len = input_shape[0], input_shape[2]

        kv = ttnn.linear(hidden_states, self.wkv, memory_config=self.memory_config)

        # kv_proj is contraction(row)-parallel like wq_a -> partial-sum single-head KV; TP all-reduce
        # (reduce_scatter + all_gather) rebuilds the full head_dim, replicated across TP.
        if self.tp_factor > 1:
            kv = ttnn.experimental.reduce_scatter_minimal_async(
                kv,
                persistent_output_buffers=None,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis=self.tp_axis),
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=self.tp_axis),
                num_links=self.ccl_num_links,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=self.tp_ccl_topology,
                cluster_axis=self.tp_axis,
            )
            kv = ttnn.experimental.all_gather_async(
                kv,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=self.tp_axis),
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=self.tp_axis),
                num_links=self.ccl_num_links,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=self.tp_ccl_topology,
                cluster_axis=self.tp_axis,
            )

        kv = ttnn.rms_norm(kv, weight=self.kv_norm_weight, epsilon=self.rms_norm_eps)

        nope_dim = self.head_dim - self.rope_head_dim
        nope = ttnn.slice(kv, [0, 0, 0, 0], [batch, 1, seq_len, nope_dim])
        rope = ttnn.slice(kv, [0, 0, 0, nope_dim], [batch, 1, seq_len, self.head_dim])
        rope = ttnn.experimental.rotary_embedding_llama(rope, cos, sin, self.trans_mat, is_decode_mode=False)
        return ttnn.concat([nope, rope], dim=-1)

    def _attention(
        self,
        q,
        sliding_kv,
        compressed_kv,
        mask_block,
        cos,
        sin,
        carry,
        kv_actual: int,
        real_len: int,
    ):
        """SP-gathers sliding_kv to full S, concats ``[carry | sliding | compressed | pad]``, runs
        per-chip SDPA, then undoes V's RoPE. ``carry`` is the previous chunk's raw KV tail, zeros on the
        first chunk; ``kv_actual`` the global position of this chunk's first query.

        Returns ``(attn, next_carry)``. The carry can only be taken here: before the gather those rows
        live on the last SP chip alone."""
        batch, seq_local = q.shape[0], q.shape[2]
        seq_len = seq_local * self.sp_factor  # global query/main-key length
        num_heads_local = self.num_heads // self.tp_factor

        # Per-chip SDPA needs every key on every chip; compressed_kv is already replicated.
        if self.sp_factor > 1:
            sliding_kv = ttnn.experimental.all_gather_async(
                sliding_kv,
                dim=2,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=self.sp_axis),
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=self.sp_axis),
                num_links=self.ccl_num_links,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=self.tp_ccl_topology,
                cluster_axis=self.sp_axis,
            )

        # The next chunk's sliding window reaches back into this one, so the carry has to be this chunk's
        # last REAL keys -- rows [real_len - sliding_window, real_len) -- not the last rows of the padded
        # slab, which differ as soon as real_len < seq_len. A plain slice cannot do it: its start value is
        # part of the program, so every new real_len would compile another
        # one. Taking the start from a device tensor keeps only its shape in the program, so one program
        # serves every offset -- measured on 8x4 as +1 program on the first call and +0 after.
        #
        # Taken on the last chunk too, whose carry nobody reads: skipping it saves one op out of ~100.
        start, end = self._carry_index[self._carry_key(real_len)]
        next_carry = ttnn.slice(sliding_kv, start, end, slice_dim=2, num_devices=seq_len // self.sliding_window)

        # Pad Sk to a multiple of 32 by hand: SDPA would pad it with zeros, and the mask reads its own pad
        # columns as "attend", which would pollute the softmax. The mask -infs the columns added here.
        # Both caches are ND-sharded, and concat refuses a mix of sharded and interleaved inputs.
        parts = [
            ttnn.to_memory_config(carry, ttnn.DRAM_MEMORY_CONFIG),
            sliding_kv,
            ttnn.to_memory_config(compressed_kv, ttnn.DRAM_MEMORY_CONFIG),
        ]
        if self._kv_pad is not None:
            parts.append(self._kv_pad)
        kv = ttnn.concat(parts, dim=2)

        # Only the compressed columns move between chunks, and always over the same column range, so the
        # mask is built once and this overwrites that range in place. The offset is part of the program,
        # but it never changes, so one program serves every chunk.
        mask = self._mask
        rows = mask_block.shape[2]
        carry_cols = self._carry_cols[kv_actual == 0]
        ttnn.experimental.slice_write(
            carry_cols, mask, start=[0, 0, 0, 0], end=[batch, 1, rows, carry_cols.shape[3]], step=[1, 1, 1, 1]
        )
        ttnn.experimental.slice_write(
            mask_block,
            mask,
            start=[0, 0, 0, self._mask_col],
            end=[batch, 1, rows, self._mask_col + mask_block.shape[3]],
            step=[1, 1, 1, 1],
        )

        attn = ttnn.transformer.scaled_dot_product_attention(
            q,
            kv,
            kv,
            attn_mask=mask,
            is_causal=False,
            scale=self.scaling,
            attention_sink=self.sinks_sdpa,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=self.device.compute_with_storage_grid_size(),
                q_chunk_size=128,
                k_chunk_size=128,
                exp_approx_mode=False,
            ),
        )

        nope_dim = self.head_dim - self.rope_head_dim
        nope = ttnn.slice(attn, [0, 0, 0, 0], [batch, num_heads_local, seq_local, nope_dim])
        rope = ttnn.slice(attn, [0, 0, 0, nope_dim], [batch, num_heads_local, seq_local, self.head_dim])
        # Undoing V's RoPE is the same rotation with the sign of sin flipped, so cos is reused and the
        # negation is one op on device instead of another host build.
        rope = ttnn.experimental.rotary_embedding_llama(rope, cos, ttnn.neg(sin), self.trans_mat, is_decode_mode=False)
        return ttnn.concat([nope, rope], dim=-1), next_carry

    def _write_compressed(self, state, new_entries, n_new):
        """Append this call's entries to the cache at row ``state.entry_count``.

        One entry is one row, so the offset advances by ``chunk/compress_rate`` per chunk -- a multiple of
        TILE_SIZE only for chunks of 4096 tokens. ``fill_cache_for_user_`` needs that alignment
        which a 5120-token chunk does not give it, so the write goes through ``_write_tail_tile`` instead.

        The whole padded width is written, not just the real entries, so the width is the same for every
        chunk; the mask -infs everything past ``total_entries`` anyway."""
        width = new_entries.shape[2]
        tile_start = (state.entry_count // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        write_end = tile_start + _cache_write_rows(width)
        assert write_end <= state.compressed_kv.shape[2], (
            f"compressed cache full: writing rows [{tile_start}, {write_end}) exceeds capacity "
            f"{state.compressed_kv.shape[2]}; allocate the state with a larger max_seq_len"
        )
        self._write_tail_tile(state, new_entries, width, n_new)

    def _write_tail_tile(self, state, new_entries, width, n_new):
        """fill_cache leaves update_idx out of its program but needs it tile-aligned, so this writes at the
        tile boundary below entry_count and supplies that whole tile -- the entries already in it come from
        ``state.tail``.

        The tail being right-aligned is what makes one shift enough: it and ``new_entries`` then sit in
        ``src`` under the same rule, so placing them is a single shift by ``TILE_SIZE - r_e``.

        The shift is a matmul against a one-hot matrix and not a slice: a slice's offset becomes part of
        its program, a matmul carries it as data."""
        tile = ttnn.TILE_SIZE
        f, r_e = divmod(state.entry_count, tile)
        shift, take = self._tail_tile_matrices(r_e, width, n_new)

        src = ttnn.concat([state.tail, new_entries], dim=2)  # [B, 1, tile + width, head_dim]
        merged = ttnn.matmul(shift, src, memory_config=self.memory_config)
        ttnn.kv_cache.fill_cache_for_user_(state.compressed_kv, merged, 0, update_idx=f * tile)
        state.tail = ttnn.matmul(take, merged, memory_config=self.memory_config)

    def _build_tail_tile_matrices(self, width):
        """Every one-hot pair the write can need, for one slab width. Called from alloc_state and never
        from forward, because a pair costs ~0.9 ms of host time to build. The whole set is 768 KB a chip."""
        tile, buf = ttnn.TILE_SIZE, _cache_write_rows(width)
        for r_e in range(tile):
            # merged row i takes src row i + (tile - r_e); rows past the entries stay zero, so nothing
            # reads past src.
            rows = torch.arange(r_e + width)
            shift = torch.zeros(1, 1, buf, tile + width)
            shift[0, 0, rows, rows + (tile - r_e)] = 1.0
            self._shift_cache[(r_e, width)] = self.ops.from_torch(shift)

        # The take matrix is keyed on r_e + n_new and not r_e + width: the whole padded width is written,
        # but entry_count only advances by the real entries. Rows before the tile's first live entry stay
        # zero, so the next chunk's shift skips them by construction instead of having to mask them.
        for s in range(tile + width):
            rows = torch.arange(tile - s % tile, tile)
            take = torch.zeros(1, 1, tile, buf)
            take[0, 0, rows, rows + s - tile] = 1.0
            self._take_cache[s] = self.ops.from_torch(take)

    def _carry_key(self, real_len):
        """The carry index is tabulated per whole compression window. A ragged chunk rounds down, which is
        safe because only the final chunk may be ragged and nothing reads its carry."""
        rate = self.compressor.compress_rate
        return max(rate, (int(real_len) // rate) * rate)

    def _build_carry_index(self, chunk_tokens):
        """start/end index tensors for the carry slice, one pair per real_len a chunk can have. Built here
        because forward must build no host tensors; each pair is 8 uint32s."""
        rate, sw = self.compressor.compress_rate, self.sliding_window
        assert sw % ttnn.TILE_SIZE == 0 and rate % sw == 0, (
            f"the carry slice needs a tile-aligned start and one whole window per step: sliding_window "
            f"{sw} must be a multiple of {ttnn.TILE_SIZE} and divide compress_rate {rate}"
        )

        def idx(vals):
            return self.ops.from_torch(
                torch.tensor(vals, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
            )

        for real_len in range(rate, int(chunk_tokens) + 1, rate):
            self._carry_index[real_len] = (idx([0, 0, real_len - sw, 0]), idx([1, 1, real_len, self.head_dim]))

    def _tail_tile_matrices(self, r_e, width, n_new):
        """Lookup only. A miss means alloc_state was given a different chunk width, and building one here
        would put host work back into forward."""
        shift = self._shift_cache.get((r_e, width))
        take = self._take_cache.get(r_e + n_new)
        assert shift is not None and take is not None, (
            f"no tail-tile one-hot for (r_e={r_e}, width={width}, n_new={n_new}); alloc_state built widths "
            f"{sorted({w for _, w in self._shift_cache})} -- pass chunk_tokens matching the slab width"
        )
        return shift, take

    def _o_proj(self, attn):
        """[B, num_heads/tp, S/sp, head_dim] -> [B, 1, S/sp, hidden/tp], the block's own input layout."""
        batch, _, seq_len, _ = attn.shape
        in_per_group = self.num_heads * self.head_dim // self.o_groups
        groups_local = self.o_groups // self.tp_factor

        x = ttnn.reshape(attn, [groups_local, attn.shape[1] // groups_local, seq_len, self.head_dim])
        x = ttnn.experimental.nlp_concat_heads(x, memory_config=self.memory_config)
        x = ttnn.reshape(x, [batch, groups_local, seq_len, in_per_group])

        grouped = ttnn.linear(x, self.wo_a, memory_config=self.memory_config)  # [B, groups_local, S, o_lora_rank]
        o_lora_rank = grouped.shape[-1]
        grouped = ttnn.concat(
            [ttnn.slice(grouped, [0, g, 0, 0], [batch, g + 1, seq_len, o_lora_rank]) for g in range(groups_local)],
            dim=-1,
        )  # [B, 1, S, groups_local*o_lora_rank]

        out = ttnn.linear(grouped, self.wo_b, memory_config=self.memory_config)  # partial-sum [B,1,S,hidden]

        # Reduce-scatter, not a full all-reduce: it both sums the partials and slices to hidden/tp,
        # which is already the layout the next block wants.
        if self.tp_factor > 1:
            out = ttnn.experimental.reduce_scatter_minimal_async(
                out,
                persistent_output_buffers=None,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis=self.tp_axis),
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=self.tp_axis),
                num_links=self.ccl_num_links,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=self.tp_ccl_topology,
                cluster_axis=self.tp_axis,
            )
        return out

    def forward(
        self,
        hidden_states,
        seq_len_actual: int | None = None,
        *,
        state: TtHCAState,
    ):
        """One chunk: [B, 1, S_pad/sp, hidden/tp] in and out; the caller keeps the first S_real rows.

        ``seq_len_actual`` is the chunk's real pre-pad length. Where the chunk sits in the sequence comes
        from ``state``, which ``alloc_state`` builds and this advances in place -- a prefill of one chunk
        passes a state too, so there is no second path through here."""
        batch = hidden_states.shape[0]
        seq_pad_global = hidden_states.shape[2] * self.sp_factor
        compress_rate = self.compressor.compress_rate
        real_len = seq_pad_global if seq_len_actual is None else seq_len_actual

        assert batch == 1, f"HCA prefill expects batch 1, got {batch}"

        assert real_len >= compress_rate, (
            f"HCA prefill needs at least one full compression window: got seq_len {real_len} < "
            f"compress_rate {compress_rate}"
        )

        n_new = real_len // compress_rate
        total_entries = state.entry_count + n_new
        assert state.kv_actual + real_len <= state.max_seq_len, (
            f"context longer than the state was allocated for: {state.kv_actual + real_len} tokens > "
            f"max_seq_len {state.max_seq_len}"
        )
        # Checked on tokens and not on entry_count, where a dropped partial window is invisible: 4097
        # tokens still gives 32 entries, and the next chunk would start at the wrong position.
        #
        # A non-final chunk ending mid-window would strand those leftover tokens -- they never join a
        # compression window -- and push the next chunk off the 128-token grid the compressor assumes.
        assert state.kv_actual % compress_rate == 0, (
            f"cannot append after a chunk with {state.kv_actual % compress_rate} leftover tokens; only "
            f"the final chunk may be ragged, non-final chunks must be a multiple of {compress_rate}"
        )

        # One rotation for the whole padded slab, shared by both stems and the output un-rope.
        slab_index = self.ops.rope_index(self._slab_index, state.kv_actual)
        cos, sin = self.ops.rope_gather(self._slab_rope, slab_index)
        q = self._q_stem(hidden_states, cos, sin)
        sliding_kv = self._kv_stem(hidden_states, cos, sin)
        new_entries, mask_block = self.compressor(
            hidden_states,
            seq_len_actual=seq_len_actual,
            first_window_position=state.entry_count * compress_rate,
        )
        # Attention then reads the WHOLE cache every chunk, so its shape stays constant and the mask
        # -infs everything past total_entries.
        self._write_compressed(state, new_entries, n_new)

        attn, next_carry = self._attention(
            q,
            sliding_kv,
            state.compressed_kv,
            mask_block,
            cos,
            sin,
            carry=state.sliding_carry,
            kv_actual=state.kv_actual,
            real_len=real_len,
        )

        state.entry_count = total_entries
        state.kv_actual += real_len
        # In place, not a rebind: the migration table is built once from this tensor's address.
        ttnn.kv_cache.fill_cache_for_user_(state.sliding_carry, next_carry, 0, update_idx=0)
        return self._o_proj(attn)
