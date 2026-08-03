# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4 Heavily Compressed Attention (TTNN prefill).
Mirrors ``DeepseekV4Attention`` in ``reference/deepseek_v4/modeling_deepseek_v4.py``."""

from __future__ import annotations

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.tt.mla.rope import get_rot_transformation_mat
from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl


def hca_block_bias(position_ids: torch.Tensor, compressed_len: int, compress_rate: int) -> torch.Tensor:
    """Per-query causal mask: query ``t`` may attend entry ``w`` only if ``t >= (w+1)*compress_rate``."""
    batch, seq_len = position_ids.shape
    entry_indices = torch.arange(compressed_len)
    causal_threshold = (position_ids + 1) // compress_rate
    block_bias = torch.zeros(batch, 1, seq_len, compressed_len)
    return block_bias.masked_fill(
        entry_indices.view(1, 1, 1, -1) >= causal_threshold.unsqueeze(1).unsqueeze(-1),
        float("-inf"),
    )


class _TtHCABase(LightweightModule):
    """Helpers shared by the compressor and the block. Subclasses must set ``device`` / ``dtype`` /
    ``weights_dtype`` / ``memory_config`` / ``rotary_emb`` and the mesh attributes before calling these."""

    @staticmethod
    def prepare_input(hidden: torch.Tensor, sp_factor: int, compress_rate: int):
        """Pad seq up to a multiple of ``compress_rate * sp_factor`` so each SP shard holds whole
        compression windows. Must run before sharding: appending per-chip would land mid-sequence.
        Pad rows are causally masked downstream and trimmed via the returned ``seq_len_actual``."""
        seq_len_actual = hidden.shape[1]
        align = compress_rate * sp_factor
        pad = (-seq_len_actual) % align
        if pad:
            hidden = torch.nn.functional.pad(hidden, (0, 0, 0, pad))
        return hidden, seq_len_actual

    def _to_tt_linear_weight(self, weight: torch.Tensor, tp_shard_dim: int | None = None):
        # tp_shard_dim indexes the transposed 4D weight [1, 1, in, out]: 2 = contraction (in), 3 = output.
        # Handed over at its source dtype: from_torch quantizes straight to weights_dtype, and a bf16
        # stop on the way to bfloat8_b would pre-round away what the block exponent could still keep.
        torch_weight = weight.detach().transpose(-2, -1).contiguous().unsqueeze(0).unsqueeze(0)
        mesh_mapper = None
        if self.is_mesh:
            if tp_shard_dim is not None and self.tp_factor > 1:
                dims = [None, None]
                dims[self.tp_axis] = tp_shard_dim
                mesh_mapper = ttnn.ShardTensor2dMesh(self.device, mesh_shape=tuple(self.device.shape), dims=dims)
            else:
                mesh_mapper = ttnn.ReplicateTensorToMesh(self.device)
        return ttnn.from_torch(
            torch_weight,
            device=self.device,
            dtype=self.weights_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self.memory_config,
            mesh_mapper=mesh_mapper,
        )

    def _from_torch(self, x: torch.Tensor, mesh_mapper=None, dtype=None):
        if self.is_mesh and mesh_mapper is None:
            mesh_mapper = ttnn.ReplicateTensorToMesh(self.device)
        return ttnn.from_torch(
            x,
            device=self.device,
            dtype=dtype or self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self.memory_config,
            mesh_mapper=mesh_mapper,
        )

    def _cos_sin(self, positions: torch.Tensor, negate_sin: bool = False):
        """``negate_sin`` gives the conjugate rotation used to undo RoPE on the attention output."""
        positions = positions[:1].to(torch.long)
        cos, sin = self.rotary_emb(torch.zeros(1), position_ids=positions, layer_type="compress")
        if negate_sin:
            sin = -sin
        cos = cos.repeat_interleave(2, dim=-1).unsqueeze(1)
        sin = sin.repeat_interleave(2, dim=-1).unsqueeze(1)
        mesh_mapper = None
        if self.is_mesh and self.sp_factor > 1:
            dims = [None, None]
            dims[self.sp_axis] = 2
            mesh_mapper = ttnn.ShardTensor2dMesh(self.device, mesh_shape=tuple(self.device.shape), dims=dims)
        return self._from_torch(cos, mesh_mapper=mesh_mapper), self._from_torch(sin, mesh_mapper=mesh_mapper)


class TtHCACompressor(_TtHCABase):
    def __init__(
        self,
        device,
        *,
        kv_proj_weight: torch.Tensor,
        gate_proj_weight: torch.Tensor,
        position_bias: torch.Tensor,
        kv_norm_weight: torch.Tensor,
        head_dim: int,
        compress_rate: int,
        rope_head_dim: int,
        rotary_emb,
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
        self.head_dim = int(head_dim)
        self.compress_rate = int(compress_rate)
        self.rope_head_dim = int(rope_head_dim)
        self.rotary_emb = rotary_emb
        self.rms_norm_eps = float(rms_norm_eps)

        self.is_mesh = hasattr(device, "shape")
        self.sp_axis, self.tp_axis = sp_axis, tp_axis
        self.sp_factor = device.shape[sp_axis] if self.is_mesh else 1
        self.tp_factor = device.shape[tp_axis] if self.is_mesh else 1
        self.ccl_topology = topology
        self.tt_ccl = get_tt_ccl(device) if (self.is_mesh and (self.sp_factor > 1 or self.tp_factor > 1)) else None
        self.ccl_num_links = 2 if is_blackhole() else 1

        self.wkv = self._to_tt_linear_weight(kv_proj_weight, tp_shard_dim=2)
        self.wgate = self._to_tt_linear_weight(gate_proj_weight, tp_shard_dim=2)
        self.position_bias = self._from_torch(position_bias.detach().reshape(1, 1, self.compress_rate, self.head_dim))
        self.kv_norm_weight = self._from_torch(kv_norm_weight.detach().reshape(1, 1, 1, self.head_dim))
        self.trans_mat = self._from_torch(get_rot_transformation_mat())

    @classmethod
    def from_reference(cls, device, reference, config, **kwargs) -> "TtHCACompressor":
        return cls(
            device,
            kv_proj_weight=reference.kv_proj.weight,
            gate_proj_weight=reference.gate_proj.weight,
            position_bias=reference.position_bias,
            kv_norm_weight=reference.kv_norm.weight,
            head_dim=config.head_dim,
            compress_rate=config.compress_rates["heavily_compressed_attention"],
            rope_head_dim=config.qk_rope_head_dim,
            rotary_emb=reference.rotary_emb,
            rms_norm_eps=config.rms_norm_eps,
            **kwargs,
        )

    def forward(
        self,
        hidden_states,
        position_ids: torch.Tensor,
        seq_len_actual: int | None = None,
        first_window_position: int = 0,
        total_entries: int | None = None,
    ):
        """[B, 1, S_pad/sp, hidden/tp] -> compressed_kv [B, 1, S_pad/compress_rate, head_dim] (replicated),
        plus a host block_bias [B, 1, S_real, entries].

        ``seq_len_actual`` is the real pre-pad length (``None`` = unpadded single-shot).
        ``first_window_position`` is the global token position of this call's first window; without it a
        later chunk would rotate its entries as if the sequence restarted.

        The output width follows the padded tensor, not the real length -- trailing pad-derived entries
        are left in so the shape stays fixed across chunks. The caller tracks how many are real and the
        mask -infs the rest."""
        input_shape = tuple(hidden_states.shape)
        if len(input_shape) != 4 or input_shape[1] != 1:
            raise ValueError(f"Expected hidden_states shape [B, 1, S, hidden], got {input_shape}")
        batch, seq_len = input_shape[0], input_shape[2]
        if seq_len_actual is None:
            seq_len_actual = seq_len * self.sp_factor

        kv = ttnn.linear(hidden_states, self.wkv, memory_config=self.memory_config)
        gate = ttnn.linear(hidden_states, self.wgate, memory_config=self.memory_config)

        # Row-parallel weights leave a partial sum on every chip -> TP all-reduce (RS + AG).
        if self.tp_factor > 1:
            for name in ("kv", "gate"):
                t = kv if name == "kv" else gate
                t = ttnn.experimental.reduce_scatter_minimal_async(
                    t,
                    persistent_output_buffers=None,
                    dim=3,
                    multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(
                        cluster_axis=self.tp_axis
                    ),
                    barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=self.tp_axis),
                    num_links=self.ccl_num_links,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    topology=self.ccl_topology,
                    cluster_axis=self.tp_axis,
                )
                t = ttnn.experimental.all_gather_async(
                    t,
                    dim=3,
                    multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(
                        cluster_axis=self.tp_axis
                    ),
                    barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=self.tp_axis),
                    num_links=self.ccl_num_links,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    topology=self.ccl_topology,
                    cluster_axis=self.tp_axis,
                )
                if name == "kv":
                    kv = t
                else:
                    gate = t

        usable = (seq_len // self.compress_rate) * self.compress_rate
        n_windows = usable // self.compress_rate  # windows this chip owns
        t_real = seq_len_actual // self.compress_rate
        if n_windows > 0:
            gate = ttnn.slice(gate, [0, 0, 0, 0], [batch, 1, usable, self.head_dim])
            gate = ttnn.reshape(gate, [batch, n_windows, self.compress_rate, self.head_dim])
            gate = ttnn.add(gate, self.position_bias)
            weights = ttnn.softmax(gate, dim=2, numeric_stable=True)

            kv = ttnn.slice(kv, [0, 0, 0, 0], [batch, 1, usable, self.head_dim])
            kv = ttnn.reshape(kv, [batch, n_windows, self.compress_rate, self.head_dim])
            pooled = ttnn.sum(ttnn.multiply(kv, weights), dim=2)

            compressed = ttnn.reshape(pooled, [batch, 1, n_windows, self.head_dim])
            compressed = ttnn.rms_norm(compressed, weight=self.kv_norm_weight, epsilon=self.rms_norm_eps)

            nope_dim = self.head_dim - self.rope_head_dim
            nope = ttnn.slice(compressed, [0, 0, 0, 0], [batch, 1, n_windows, nope_dim])
            rope = ttnn.slice(compressed, [0, 0, 0, nope_dim], [batch, 1, n_windows, self.head_dim])
            positions = (
                torch.arange(n_windows * self.sp_factor) * self.compress_rate + first_window_position
            ).unsqueeze(0)
            cos, sin = self._cos_sin(positions)
            rope = ttnn.experimental.rotary_embedding_llama(rope, cos, sin, self.trans_mat, is_decode_mode=False)
            compressed_kv = ttnn.concat([nope, rope], dim=-1)

            if self.sp_factor > 1:
                compressed_kv = ttnn.experimental.all_gather_async(
                    compressed_kv,
                    dim=2,
                    multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(
                        cluster_axis=self.sp_axis
                    ),
                    barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=self.sp_axis),
                    num_links=self.ccl_num_links,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    topology=self.ccl_topology,
                    cluster_axis=self.sp_axis,
                )
        else:
            compressed_kv = self._from_torch(torch.zeros(batch, 1, 0, self.head_dim))

        # Must span every entry the queries can see -- under chunked prefill that includes earlier
        # chunks', not just this call's. total_entries=None -> single-shot, where the two coincide.
        bias_entries = t_real if total_entries is None else total_entries
        block_bias = None
        if seq_len_actual > 1 and bias_entries > 0:
            block_bias = hca_block_bias(position_ids, bias_entries, self.compress_rate)

        return compressed_kv, block_bias


class TtHCAState:
    """Chunked-prefill state, owned by the caller and passed to ``TtHCA.forward``.

    Both device tensors keep a FIXED shape for the whole prefill and only their contents advance --
    that is what lets one compiled program serve every chunk. The counters say how much is real;
    the attention mask -infs the rest."""

    def __init__(self, compressed_kv, sliding_carry):
        self.compressed_kv = compressed_kv  # [B, 1, compressed_capacity, head_dim]
        self.sliding_carry = sliding_carry  # [B, 1, sliding_window, head_dim]
        self.entry_count = 0
        self.kv_actual = 0


class TtHCA(_TtHCABase):
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

        self.is_mesh = hasattr(device, "shape")
        self.sp_axis, self.tp_axis = sp_axis, tp_axis
        self.sp_factor = device.shape[sp_axis] if self.is_mesh else 1
        self.tp_factor = device.shape[tp_axis] if self.is_mesh else 1
        self.tp_ccl_topology = topology
        self.tt_ccl = get_tt_ccl(device) if (self.is_mesh and (self.sp_factor > 1 or self.tp_factor > 1)) else None
        self.ccl_num_links = 2 if is_blackhole() else 1

        # Pre-divided by scale: SDPA scales BOTH QK and the sink internally, the reference scales only
        # QK -- dividing here cancels the kernel's extra multiply. TP-sharded to match the query heads.
        sinks_host = sinks.detach().reshape(1, self.num_heads, 1, 1) / self.scaling
        if self.is_mesh and self.tp_factor > 1:
            sink_dims = [None, None]
            sink_dims[self.tp_axis] = 1
            sink_mapper = ttnn.ShardTensor2dMesh(device, mesh_shape=tuple(device.shape), dims=sink_dims)
            self.sinks_sdpa = self._from_torch(sinks_host, mesh_mapper=sink_mapper)
        else:
            self.sinks_sdpa = self._from_torch(sinks_host)

        self.wq_a = self._to_tt_linear_weight(q_a_proj_weight, tp_shard_dim=2)
        self.wq_b = self._to_tt_linear_weight(q_b_proj_weight, tp_shard_dim=3)
        self.q_a_norm_weight = self._from_torch(q_a_norm_weight.detach().reshape(1, 1, 1, -1))
        self.q_b_norm_weight = self._from_torch(torch.ones(1, 1, 1, self.head_dim))
        self.wkv = self._to_tt_linear_weight(kv_proj_weight, tp_shard_dim=2)
        self.kv_norm_weight = self._from_torch(kv_norm_weight.detach().reshape(1, 1, 1, self.head_dim))

        # o_a_proj is block-diagonal over o_groups. Groups partition the heads, so a TP chip owns whole
        # groups: keep it as ONE batched weight sharded on the group axis and run a single batched
        # matmul -- each chip applies only its own groups, no collective
        self.o_groups = int(o_groups)
        in_per_group = self.num_heads * self.head_dim // self.o_groups
        o_a_grouped = o_a_proj_weight.detach().view(self.o_groups, -1, in_per_group).transpose(1, 2).unsqueeze(0)
        o_a_mapper = None
        if self.is_mesh and self.tp_factor > 1:
            o_a_dims = [None, None]
            o_a_dims[self.tp_axis] = 1
            o_a_mapper = ttnn.ShardTensor2dMesh(device, mesh_shape=tuple(device.shape), dims=o_a_dims)

        self.wo_a = self._from_torch(o_a_grouped, mesh_mapper=o_a_mapper, dtype=self.weights_dtype)
        self.wo_b = self._to_tt_linear_weight(o_b_proj_weight, tp_shard_dim=2)

        self.trans_mat = self._from_torch(get_rot_transformation_mat())

    def alloc_state(self, max_seq_len: int, batch: int = 1) -> TtHCAState:
        """Size the state once for the longest context this layer will serve, so its shape is fixed for
        every chunk. ``max_seq_len`` is the same serving-capacity knob the runner resolves for the KV
        cache. Contents start zeroed and nothing is read before it is written -- the mask -infs
        everything past ``entry_count`` / ``kv_actual``, including chunk 0's empty carry."""
        entries = -(-int(max_seq_len) // self.compressor.compress_rate)
        capacity = -(-entries // ttnn.TILE_SIZE) * ttnn.TILE_SIZE  # cache writes land on tile boundaries
        return TtHCAState(
            compressed_kv=self._from_torch(torch.zeros(batch, 1, capacity, self.head_dim)),
            sliding_carry=self._from_torch(torch.zeros(batch, 1, self.sliding_window, self.head_dim)),
        )

    @classmethod
    def from_reference(cls, device, reference, config, **kwargs) -> "TtHCA":
        # Forward the mesh/CCL config so the compressor rides the same SP/TP axes as the block.
        compressor_keys = ("sp_axis", "tp_axis", "topology", "dtype", "weights_dtype")
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

    def _q_stem(self, hidden_states, position_ids: torch.Tensor):
        """[B, 1, S/sp, hidden/tp] -> q [B, num_heads/tp, S/sp, head_dim]. ``position_ids`` is full-length."""
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
        cos, sin = self._cos_sin(position_ids)
        rope = ttnn.experimental.rotary_embedding_llama(rope, cos, sin, self.trans_mat, is_decode_mode=False)
        return ttnn.concat([nope, rope], dim=-1)

    def _kv_stem(self, hidden_states, position_ids: torch.Tensor):
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
        cos, sin = self._cos_sin(position_ids)
        rope = ttnn.experimental.rotary_embedding_llama(rope, cos, sin, self.trans_mat, is_decode_mode=False)
        return ttnn.concat([nope, rope], dim=-1)

    def _attn_mask(
        self,
        batch: int,
        seq_len: int,
        block_bias: torch.Tensor,
        sk_pad: int,
        kv_actual: int = 0,
        carry_len: int = 0,
    ):
        """Additive mask [B, 1, S, sk_pad] over the key layout ``[carry | chunk | compressed | pad]``:
        sliding-causal on the raw keys, block_bias on the compressed ones, -inf on the rest.

        ``seq_len`` and ``kv_actual`` are GLOBAL (query length of this chunk, and the position of its
        first query); ``carry_len`` is how many raw keys precede it. Built on host and SP-sharded on
        query rows to match q; head-independent, so TP-replicated."""
        t_len = block_bias.shape[-1]
        raw = carry_len + seq_len
        i = torch.arange(seq_len).view(-1, 1) + kv_actual
        j = torch.arange(raw).view(1, -1) + (kv_actual - carry_len)
        # j >= 0 drops the carry columns of chunk 0, whose positions do not exist yet.
        allowed = (j >= 0) & (j <= i) & (i - j < self.sliding_window)
        full = torch.full((batch, 1, seq_len, sk_pad), float("-inf"))
        full[..., :raw] = torch.zeros(seq_len, raw).masked_fill(~allowed, float("-inf")).view(1, 1, seq_len, raw)
        full[:, :, : block_bias.shape[2], raw : raw + t_len] = block_bias.to(torch.float32)
        mesh_mapper = None
        if self.is_mesh and self.sp_factor > 1:
            dims = [None, None]
            dims[self.sp_axis] = 2
            mesh_mapper = ttnn.ShardTensor2dMesh(self.device, mesh_shape=tuple(self.device.shape), dims=dims)
        return self._from_torch(full, mesh_mapper=mesh_mapper)

    def _attention(
        self,
        q,
        sliding_kv,
        compressed_kv,
        block_bias: torch.Tensor,
        position_ids: torch.Tensor,
        carry=None,
        kv_actual: int = 0,
    ):
        """SP-gathers sliding_kv to full S, concats ``[carry | sliding | compressed | pad]``, runs
        per-chip SDPA, then undoes V's RoPE. ``carry`` is the previous chunk's raw KV tail (None in
        single-shot); ``kv_actual`` the global position of this chunk's first query.

        Returns ``(attn, next_carry)``. The carry can only be taken here: before the gather those rows
        live on the last SP chip alone."""
        batch, seq_local = q.shape[0], q.shape[2]
        seq_len = seq_local * self.sp_factor  # global query/main-key length
        num_heads_local = self.num_heads // self.tp_factor
        carry_len = 0 if carry is None else carry.shape[2]

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

        next_carry = ttnn.slice(
            sliding_kv,
            [0, 0, seq_len - self.sliding_window, 0],
            [batch, 1, seq_len, self.head_dim],
        )

        # Pad Sk to a multiple of 32 explicitly: SDPA tile-pads a non-aligned Sk with zeros, and a
        # provided mask's pad columns default to 0 (= attend), which pollutes the softmax. _attn_mask
        # -infs the columns we add here.
        sk = carry_len + seq_len + compressed_kv.shape[2]
        sk_pad = ((sk + 31) // 32) * 32
        parts = [sliding_kv, compressed_kv] if carry is None else [carry, sliding_kv, compressed_kv]
        if sk_pad > sk:
            parts.append(self._from_torch(torch.zeros(batch, 1, sk_pad - sk, self.head_dim)))
        kv = ttnn.concat(parts, dim=2)
        mask = self._attn_mask(batch, seq_len, block_bias, sk_pad, kv_actual=kv_actual, carry_len=carry_len)

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
        cos, sin = self._cos_sin(position_ids, negate_sin=True)
        rope = ttnn.experimental.rotary_embedding_llama(rope, cos, sin, self.trans_mat, is_decode_mode=False)
        return ttnn.concat([nope, rope], dim=-1), next_carry

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
        position_ids: torch.Tensor,
        seq_len_actual: int | None = None,
        state: TtHCAState | None = None,
    ):
        """One chunk: [B, 1, S_pad/sp, hidden/tp] in and out; the caller keeps the first S_real rows.

        ``position_ids`` holds this chunk's GLOBAL positions and ``seq_len_actual`` its real pre-pad
        length. ``state`` is advanced in place. ``state=None`` is single-shot, which is just a one-chunk
        prefill -- a throwaway state is allocated here so there is no separate compute path for it."""
        batch = hidden_states.shape[0]
        seq_pad_global = hidden_states.shape[2] * self.sp_factor
        compress_rate = self.compressor.compress_rate
        real_len = seq_pad_global if seq_len_actual is None else seq_len_actual

        assert batch == 1, f"HCA prefill expects batch 1, got {batch}"

        assert real_len >= compress_rate, (
            f"HCA prefill needs at least one full compression window: got seq_len {real_len} < "
            f"compress_rate {compress_rate}"
        )

        if state is None:
            state = self.alloc_state(real_len, batch=batch)

        n_new = real_len // compress_rate
        total_entries = state.entry_count + n_new
        capacity = state.compressed_kv.shape[2]
        assert total_entries <= capacity, (
            f"compressed cache full: {total_entries} entries > capacity {capacity}; allocate the state "
            f"with a larger max_seq_len"
        )
        # Checked on tokens, not entry_count: a dropped partial window is invisible there (4097 tokens
        # still lands on 32 entries) and the next chunk would start at the wrong window position.
        chunk_align = compress_rate * ttnn.TILE_SIZE
        assert state.kv_actual % chunk_align == 0, (
            f"cannot append after a chunk with {state.kv_actual % chunk_align} leftover tokens; only the "
            f"final chunk may be ragged, non-final chunks must be a multiple of {chunk_align}"
        )

        # The rope op matches cos/sin to the tensor seq, so the stems need one position per PADDED row.
        # The compressor keeps the real ones -- they drive block_bias.
        pos_padded = position_ids
        if position_ids.shape[1] < seq_pad_global:
            tail = torch.arange(1, seq_pad_global - position_ids.shape[1] + 1).unsqueeze(0)
            pos_padded = torch.cat([position_ids, position_ids[:, -1:] + tail], dim=1)

        q = self._q_stem(hidden_states, pos_padded)
        sliding_kv = self._kv_stem(hidden_states, pos_padded)
        new_entries, block_bias = self.compressor(
            hidden_states,
            position_ids,
            seq_len_actual=seq_len_actual,
            first_window_position=state.entry_count * compress_rate,
            total_entries=total_entries,
        )
        # Attention then reads the WHOLE cache every chunk, so its shape stays constant and the mask
        # -infs everything past total_entries.
        ttnn.kv_cache.fill_cache_for_user_(state.compressed_kv, new_entries, 0, update_idx=state.entry_count)

        attn, next_carry = self._attention(
            q,
            sliding_kv,
            state.compressed_kv,
            block_bias,
            pos_padded,
            carry=state.sliding_carry,
            kv_actual=state.kv_actual,
        )

        state.entry_count = total_entries
        state.kv_actual += real_len
        state.sliding_carry = next_carry
        return self._o_proj(attn)
