# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Shared prefill core for DeepSeek-V4 attention blocks.

Both V4 compressed-attention variants -- HCA (``compress_rate`` 128) and CSA (``compress_rate`` 4 plus a
Lightning Indexer) -- run the same attention body as ``DeepseekV4Attention``: query/kv stems, one masked
SDPA with attention sinks over ``[carry | sliding | compressed | pad]``, V's RoPE undone on the output,
then the grouped output projection. They differ only in how the compressed entries reach the cache and in
what the compressed mask columns say, which is what the subclasses own."""

from __future__ import annotations

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.tt.mla.compressor import TtCompressorUtils
from models.demos.deepseek_v3_d_p.tt.mla.rope import get_rot_transformation_mat
from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl, resolve_per_axis_topology


class TtV4AttentionBase(LightweightModule):
    """V4 attention core. Block I/O is ``[B, 1, S/sp, hidden/tp]``, so layers chain without a reshard."""

    def __init__(
        self,
        device,
        *,
        compressor,
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

        self.is_mesh = hasattr(device, "shape")
        self.sp_axis, self.tp_axis = sp_axis, tp_axis
        self.sp_factor = device.shape[sp_axis] if self.is_mesh else 1
        self.tp_factor = device.shape[tp_axis] if self.is_mesh else 1
        # q/kv/o-proj collectives ride the TP axis; the sliding_kv all-gather rides SP. See
        # resolve_per_axis_topology for why one topology cannot serve both.
        self.sp_ccl_topology, self.tp_ccl_topology = resolve_per_axis_topology(topology, sp_axis, tp_axis)
        self.tt_ccl = get_tt_ccl(device) if (self.is_mesh and (self.sp_factor > 1 or self.tp_factor > 1)) else None
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
        self._carry_index = {}
        self._slab_rope = None
        self._slab_index = None
        self._mask = None  # persistent additive mask; forward overwrites only the moving columns
        self._mask_col = None
        self._carry_cols = None
        self._kv_pad = None  # zero rows that bring Sk up to a tile multiple

    @property
    def chunk_align(self) -> int:
        """Token granularity every non-final chunk must be a multiple of, and the step the carry index is
        tabulated at. Set by the variant: what it takes for both the compressed-cache write offset and the
        carry slice start to land where their ops need them."""
        raise NotImplementedError

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

    def _q_lora(self, hidden_states):
        """[B, 1, S/sp, hidden/tp] -> q_lora [B, 1, S/sp, q_lora_rank], TP-replicated. This is the
        reference's ``q_residual``: the indexer scores from it, so CSA needs it on its own."""
        input_shape = tuple(hidden_states.shape)
        if len(input_shape) != 4 or input_shape[1] != 1:
            raise ValueError(f"Expected hidden_states shape [B, 1, S, hidden], got {input_shape}")

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

        return ttnn.rms_norm(q, weight=self.q_a_norm_weight, epsilon=self.rms_norm_eps)

    def _q_heads(self, q_lora, cos, sin):
        """q_lora -> q [B, num_heads/tp, S/sp, head_dim]. ``cos``/``sin`` cover the padded slab and are
        built once per call: both stems and the output un-rope want the same rotation, and building it
        again costs ~2.9 ms of host time."""
        batch, seq_len = q_lora.shape[0], q_lora.shape[2]
        num_heads_local = self.num_heads // self.tp_factor

        q = ttnn.linear(q_lora, self.wq_b, memory_config=self.memory_config)

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

    def _q_stem(self, hidden_states, cos, sin):
        """[B, 1, S/sp, hidden/tp] -> q [B, num_heads/tp, S/sp, head_dim]."""
        return self._q_heads(self._q_lora(hidden_states), cos, sin)

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
                topology=self.sp_ccl_topology,
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
        parts = [carry, sliding_kv, compressed_kv]
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

    def _carry_key(self, real_len):
        """The carry index is tabulated per whole ``chunk_align`` step. A ragged chunk rounds down, which
        is safe because only the final chunk may be ragged and nothing reads its carry."""
        align = self.chunk_align
        return max(align, (int(real_len) // align) * align)

    def _build_carry_index(self, chunk_tokens):
        """start/end index tensors for the carry slice, one pair per real_len a chunk can have. Built here
        because forward must build no host tensors; each pair is 8 uint32s."""
        align, sw = self.chunk_align, self.sliding_window
        assert sw % ttnn.TILE_SIZE == 0 and align % sw == 0, (
            f"the carry slice needs a tile-aligned start and one whole window per step: sliding_window "
            f"{sw} must be a multiple of {ttnn.TILE_SIZE} and divide the chunk alignment {align}"
        )

        def idx(vals):
            return self.ops.from_torch(
                torch.tensor(vals, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
            )

        for real_len in range(align, int(chunk_tokens) + 1, align):
            self._carry_index[real_len] = (idx([0, 0, real_len - sw, 0]), idx([1, 1, real_len, self.head_dim]))

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
