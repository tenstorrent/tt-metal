# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4 sliding-only attention (``layer_types[i] == "sliding_attention"``), chunked prefill.

Each chip attends over ``[halo | its own rows]``, where the halo is the ``sliding_window`` rows
preceding its first query: that is as far back as the window reaches.
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import DeepseekV4RotaryEmbedding
from models.demos.deepseek_v3_d_p.tt.mla.compressor import TtCompressorUtils
from models.demos.deepseek_v3_d_p.tt.mla.rope import get_rot_transformation_mat
from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache

#: A slice bound on a tiled tensor moves a tile row at a time.
TILE_HEIGHT = 32


class TtSWAState:
    """Chunked-prefill state. One tensor, and it never grows: the whole history a sliding layer needs
    is the ``sliding_window`` tokens before the next chunk starts."""

    def __init__(self, carry, max_seq_len):
        self.carry = carry
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
        self.device = device
        self.dtype = dtype
        self.weights_dtype = weights_dtype
        self.memory_config = memory_config
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.rope_head_dim = int(rope_head_dim)
        self.sliding_window = int(sliding_window)
        self.o_groups = int(o_groups)
        self.scaling = self.head_dim**-0.5
        self.rotary_emb = rotary_emb
        self.rms_norm_eps = float(rms_norm_eps)

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

        # SDPA scales the sink along with QK, so the division here cancels that.
        sinks_host = sinks.detach().reshape(1, self.num_heads, 1, 1) / self.scaling
        self.sinks_sdpa = self.ops.from_torch(sinks_host, mesh_mapper=self.ops.mesh_mapper(tp_dim=1))

        self.wq_a = self.ops.to_tt_linear_weight(q_a_proj_weight, tp_shard_dim=2)
        self.wq_b = self.ops.to_tt_linear_weight(q_b_proj_weight, tp_shard_dim=3)
        self.q_a_norm_weight = self.ops.from_torch(q_a_norm_weight.detach().reshape(1, 1, 1, -1))
        self.q_b_norm_weight = self.ops.from_torch(torch.ones(1, 1, 1, self.head_dim))
        self.wkv = self.ops.to_tt_linear_weight(kv_proj_weight, tp_shard_dim=2)
        self.kv_norm_weight = self.ops.from_torch(kv_norm_weight.detach().reshape(1, 1, 1, self.head_dim))

        # Block-diagonal over o_groups, and a TP chip owns whole groups, so one batched matmul serves it.
        in_per_group = self.num_heads * self.head_dim // self.o_groups
        o_a_grouped = o_a_proj_weight.detach().view(self.o_groups, -1, in_per_group).transpose(1, 2).unsqueeze(0)
        self.wo_a = self.ops.from_torch(
            o_a_grouped, mesh_mapper=self.ops.mesh_mapper(tp_dim=1), dtype=self.weights_dtype
        )
        self.wo_b = self.ops.to_tt_linear_weight(o_b_proj_weight, tp_shard_dim=2)

        self.trans_mat = self.ops.from_torch(get_rot_transformation_mat())

        # Built by alloc_state, which every caller runs before forward.
        self._masks = None
        self._halo_bounds = None
        self._halo_num_devices = None
        self._slab_rope = None
        self._slab_index = None
        self._carry_index = None

    @classmethod
    def from_reference(cls, device, reference, config, **kwargs) -> "TtSWA":
        assert getattr(reference, "compressor", None) is None, (
            "TtSWA expects a sliding-only reference layer (compressor is None); a layer with a "
            "compressor is HCA or CSA"
        )
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
        """Build every host tensor forward will need, so forward builds none."""
        assert batch == 1, f"SWA state is single-user for now, got batch={batch}"
        chunk = int(chunk_tokens or max_seq_len)
        sw, sp = self.sliding_window, self.sp_factor
        assert chunk % (TILE_HEIGHT * sp) == 0, (
            f"the slab is {chunk} wide, which is not a multiple of tile height * sp_factor "
            f"({TILE_HEIGHT} * {sp} = {TILE_HEIGHT * sp}); a chip's share must be whole tiles"
        )
        assert chunk % sw == 0, (
            f"the slab is {chunk} wide, which is not a multiple of sliding_window {sw}; the halo and "
            f"carry slices size their output as (sw + chunk) / num_devices"
        )
        seq_local = chunk // sp

        self._build_masks(seq_local)
        self._build_halo_bounds(chunk, seq_local)
        self._build_carry_index(chunk)
        self._slab_rope = self.ops.build_rope_table(int(max_seq_len) + chunk, 1, layer_type="main")
        self._slab_index = self.ops.rope_index_base(seq_local)
        return TtSWAState(
            carry=self._alloc_carry(batch, sw),
            max_seq_len=max_seq_len,
        )

    def _alloc_carry(self, batch: int, rows: int):
        """ND-sharded so the migration address table can name it. ``seq_len`` is pre-multiplied by
        sp_factor to cancel the division init_kvpe_cache does: the carry is replicated."""
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

    def _build_masks(self, seq_local: int):
        """Both masks over the key layout ``[halo | own rows]``, built here and never touched again."""
        sw = self.sliding_window
        width = sw + seq_local

        ic = self.ops.from_torch(torch.arange(seq_local).float().view(1, 1, seq_local, 1), dtype=ttnn.float32)
        jc = self.ops.from_torch(torch.arange(width).float().view(1, 1, 1, width), dtype=ttnn.float32)
        band = ttnn.typecast(ttnn.log(ttnn.multiply(ttnn.gt(jc, ic), ttnn.le(jc, ttnn.add(ic, float(sw))))), self.dtype)

        chip = self.ops.from_torch(
            torch.arange(self.sp_factor).float().view(1, 1, self.sp_factor, 1),
            self.ops.mesh_mapper(sp_dim=2),
            dtype=ttnn.float32,
        )
        keep = ttnn.nez(ttnn.add(ttnn.nez(chip), ttnn.ge(jc, float(sw))))
        empty_carry = ttnn.typecast(ttnn.log(keep), self.dtype)
        # Only chunk 0 differs: chip 0's halo is the zeroed carry, and a zero key still counts in softmax.
        self._masks = {False: band, True: ttnn.add(band, empty_carry)}

    def _build_halo_bounds(self, chunk: int, seq_local: int):
        """Which part of the gathered KV each chip takes as its halo."""
        sw, sp = self.sliding_window, self.sp_factor
        starts = torch.tensor([v for d in range(sp) for v in (0, 0, d * seq_local, 0)], dtype=torch.int32)
        ends = torch.tensor(
            [v for d in range(sp) for v in (1, 1, d * seq_local + sw, self.head_dim)], dtype=torch.int32
        )
        mapper = self.ops.mesh_mapper(sp_dim=0)
        self._halo_bounds = tuple(
            self.ops.from_torch(t, mapper, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT) for t in (starts, ends)
        )
        self._halo_num_devices = (sw + chunk) // sw

    def _carry_key(self, real_len: int):
        return (int(real_len) // TILE_HEIGHT) * TILE_HEIGHT

    def _build_carry_index(self, chunk: int):
        """Which part of ``[carry | chunk]`` becomes the carry for the next chunk, one pair per real_len."""
        sw = self.sliding_window

        def idx(vals):
            return self.ops.from_torch(
                torch.tensor(vals, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
            )

        self._carry_index = {
            real_len: (idx([0, 0, real_len, 0]), idx([1, 1, real_len + sw, self.head_dim]))
            for real_len in range(0, int(chunk) + 1, TILE_HEIGHT)
        }

    def _attention(self, q, sliding_kv, cos, sin, carry, kv_actual: int, real_len: int):
        """``q`` [B, H/tp, S/sp, head_dim] + this chip's ``sliding_kv`` -> ``(attn, next_carry)``.

        The gather covers the whole chunk because the carry slice needs the global slab."""
        batch, num_heads_local, seq_local, _ = q.shape
        sw = self.sliding_window

        if self.sp_factor > 1:
            gathered = ttnn.experimental.all_gather_async(
                sliding_kv,
                dim=2,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=self.sp_axis),
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=self.sp_axis),
                num_links=self.ccl_num_links,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=self.tp_ccl_topology,
                cluster_axis=self.sp_axis,
            )
        else:
            gathered = sliding_kv

        # The carry is ND-sharded and concat will not mix sharded with interleaved.
        hist = ttnn.concat([ttnn.to_memory_config(carry, ttnn.DRAM_MEMORY_CONFIG), gathered], dim=2)
        start, end = self._halo_bounds
        halo = ttnn.slice(hist, start, end, slice_dim=2, num_devices=self._halo_num_devices)
        kv = ttnn.concat([halo, sliding_kv], dim=2)

        attn = ttnn.transformer.scaled_dot_product_attention(
            q,
            kv,
            kv,
            attn_mask=self._masks[kv_actual == 0],
            is_causal=False,
            scale=self.scaling,
            attention_sink=self.sinks_sdpa,
            # With a mask the reader streams a [q_chunk, k_chunk] tile, so wider sizes exhaust L1.
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=self.device.compute_with_storage_grid_size(),
                q_chunk_size=128,
                k_chunk_size=128,
                exp_approx_mode=False,
            ),
        )

        # Device-tensor bounds keep only their shape in the program, so one program serves every real_len.
        c_start, c_end = self._carry_index[self._carry_key(real_len)]
        next_carry = ttnn.slice(hist, c_start, c_end, slice_dim=2, num_devices=self._halo_num_devices)

        nope_dim = self.head_dim - self.rope_head_dim
        nope = ttnn.slice(attn, [0, 0, 0, 0], [batch, num_heads_local, seq_local, nope_dim])
        rope = ttnn.slice(attn, [0, 0, 0, nope_dim], [batch, num_heads_local, seq_local, self.head_dim])
        # V's rope is undone by the same rotation with sin's sign flipped.
        rope = ttnn.experimental.rotary_embedding_llama(rope, cos, ttnn.neg(sin), self.trans_mat, is_decode_mode=False)
        return ttnn.concat([nope, rope], dim=-1), next_carry

    def _q_stem(self, hidden_states, cos, sin):
        """[B, 1, S/sp, hidden/tp] -> q [B, num_heads/tp, S/sp, head_dim]."""
        input_shape = tuple(hidden_states.shape)
        if len(input_shape) != 4 or input_shape[1] != 1:
            raise ValueError(f"Expected hidden_states shape [B, 1, S, hidden], got {input_shape}")
        batch, seq_len = input_shape[0], input_shape[2]
        num_heads_local = self.num_heads // self.tp_factor

        q = ttnn.linear(hidden_states, self.wq_a, memory_config=self.memory_config)

        # Row-parallel, so partial sums; the all-reduce rebuilds q_lora replicated across TP.
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
        K == V in V4, and the sliding-window truncation is the caller's concern."""
        input_shape = tuple(hidden_states.shape)
        if len(input_shape) != 4 or input_shape[1] != 1:
            raise ValueError(f"Expected hidden_states shape [B, 1, S, hidden], got {input_shape}")
        batch, seq_len = input_shape[0], input_shape[2]

        kv = ttnn.linear(hidden_states, self.wkv, memory_config=self.memory_config)

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

    def _o_proj(self, attn):
        """[B, num_heads/tp, S/sp, head_dim] -> [B, 1, S/sp, hidden/tp], the block's input layout."""
        batch, _, seq_len, _ = attn.shape
        in_per_group = self.num_heads * self.head_dim // self.o_groups
        groups_local = self.o_groups // self.tp_factor

        x = ttnn.reshape(attn, [groups_local, attn.shape[1] // groups_local, seq_len, self.head_dim])
        x = ttnn.experimental.nlp_concat_heads(x, memory_config=self.memory_config)
        x = ttnn.reshape(x, [batch, groups_local, seq_len, in_per_group])

        grouped = ttnn.linear(x, self.wo_a, memory_config=self.memory_config)
        o_lora_rank = grouped.shape[-1]
        grouped = ttnn.concat(
            [ttnn.slice(grouped, [0, g, 0, 0], [batch, g + 1, seq_len, o_lora_rank]) for g in range(groups_local)],
            dim=-1,
        )

        out = ttnn.linear(grouped, self.wo_b, memory_config=self.memory_config)

        # Reduce-scatter sums the partials and slices to hidden/tp in one op.
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

    def forward(self, hidden_states, seq_len_actual: int | None = None, state: TtSWAState = None):
        """[B, 1, S/sp, hidden/tp] -> the same, one chunk. ``seq_len_actual`` is the real token count;
        the rest of the slab is padding no real query reads."""
        assert state is not None, "TtSWA.forward needs the state alloc_state returned"
        seq_len = hidden_states.shape[2] * self.sp_factor
        real_len = int(seq_len if seq_len_actual is None else seq_len_actual)
        sw = self.sliding_window
        # On the ACCUMULATED length: a final chunk may end mid-tile, a non-final one may not.
        assert state.kv_actual % TILE_HEIGHT == 0, (
            f"cannot append after a chunk with {state.kv_actual % TILE_HEIGHT} leftover tokens; only "
            f"the final chunk may end mid-tile, non-final chunks must be a multiple of {TILE_HEIGHT}"
        )

        slab_index = self.ops.rope_index(self._slab_index, state.kv_actual)
        cos, sin = self.ops.rope_gather(self._slab_rope, slab_index)
        q = self._q_stem(hidden_states, cos, sin)
        sliding_kv = self._kv_stem(hidden_states, cos, sin)

        attn, next_carry = self._attention(
            q, sliding_kv, cos, sin, carry=state.carry, kv_actual=state.kv_actual, real_len=real_len
        )

        state.kv_actual += real_len
        # The migration table records this address once, so the write is in place.
        ttnn.kv_cache.fill_cache_for_user_(state.carry, next_carry, 0, update_idx=0)
        return self._o_proj(attn)
