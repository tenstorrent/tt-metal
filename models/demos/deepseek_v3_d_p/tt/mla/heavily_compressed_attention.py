# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4 Heavily Compressed Attention (TTNN prefill). Mirrors ``DeepseekV4Attention``
(reference ``modeling_deepseek_v4.py``, paper §2.3.2). ``TtHCACompressor`` softmax-pools
every complete window of ``compress_rate`` source tokens into one compressed KV entry;
``TtHCA`` is the block that composes it with the query/kv stems (and, as they land, the
attention core + output projection)."""

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
    """Shared TTNN helpers for the HCA compressor/block: weight tilize and interleaved
    cos/sin from the reference compress rotary. Not instantiated directly; subclasses set
    ``device`` / ``dtype`` / ``memory_config`` / ``rotary_emb`` and the mesh attributes
    (``is_mesh`` / ``sp_axis`` / ``tp_axis`` / ``sp_factor`` / ``tp_factor``) before calling these."""

    @staticmethod
    def prepare_input(hidden: torch.Tensor, sp_factor: int, compress_rate: int):
        """Host-side, pre-shard: pad the seq dim up to a multiple of ``compress_rate * sp_factor`` so
        each SP shard holds whole compression windows (pad can't be added to an already-sharded seq —
        per-chip append would land mid-sequence). Returns ``(padded_hidden, seq_len_actual)``; the pad
        rows are causally masked downstream and trimmed out of the compressed output via seq_len_actual."""
        seq_len_actual = hidden.shape[1]
        align = compress_rate * sp_factor
        pad = (-seq_len_actual) % align
        if pad:
            hidden = torch.nn.functional.pad(hidden, (0, 0, 0, pad))
        return hidden, seq_len_actual

    def _to_tt_linear_weight(self, weight: torch.Tensor, tp_shard_dim: int | None = None):
        # tp_shard_dim indexes the transposed 4D weight [1, 1, in, out]: 2 = contraction (in), 3 = output.
        torch_weight = weight.detach().transpose(-2, -1).contiguous().unsqueeze(0).unsqueeze(0).to(torch.bfloat16)
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
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self.memory_config,
            mesh_mapper=mesh_mapper,
        )

    def _from_torch(self, x: torch.Tensor, mesh_mapper=None):
        if self.is_mesh and mesh_mapper is None:
            mesh_mapper = ttnn.ReplicateTensorToMesh(self.device)
        return ttnn.from_torch(
            x.to(torch.bfloat16),
            device=self.device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self.memory_config,
            mesh_mapper=mesh_mapper,
        )

    def _cos_sin(self, positions: torch.Tensor, negate_sin: bool = False):
        """Interleaved cos/sin [1, 1, N, rope_head_dim] from the reference compress rotary.
        ``negate_sin`` gives the conjugate rotation used for undo-RoPE (rope with -sin). On a mesh
        the seq axis is SP-sharded to match the SP-sharded query rows (natural-order, non-balanced)."""
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
        ccl_num_links: int | None = None,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ):
        self.device = device
        self.dtype = dtype
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
        # The compressor rides BOTH axes: TP all-reduce for wkv/wgate, SP all-gather for the pooled
        # compressed KV -> needs the CCL object whenever either factor > 1.
        self.ccl_topology = topology
        self.tt_ccl = get_tt_ccl(device) if (self.is_mesh and (self.sp_factor > 1 or self.tp_factor > 1)) else None
        self.ccl_num_links = (2 if is_blackhole() else 1) if ccl_num_links is None else ccl_num_links

        # wkv/wgate: contraction(row)-parallel like the sliding wkv -> partial-sum single-head KV/gate,
        # TP all-reduced in forward to a TP-replicated full head_dim.
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

    def forward(self, hidden_states, position_ids: torch.Tensor, seq_len_actual: int | None = None):
        """``hidden_states``: TTNN [B, 1, S_pad/sp, hidden/tp] (seq host-padded to a multiple of
        compress_rate*sp via ``prepare_input``). ``position_ids``: torch [B, S_real] (real positions).
        ``seq_len_actual``: real pre-pad length; only ``T_real = S_real // compress_rate`` compressed
        entries survive (pad-derived windows are trimmed). ``None`` -> no padding (single-shot: use the
        full seq). Returns ``compressed_kv`` [B, 1, T_real, head_dim] (TP-replicated, SP-gathered) and
        host ``block_bias`` [B, 1, S_real, T_real] (or None)."""
        input_shape = tuple(hidden_states.shape)
        if len(input_shape) != 4 or input_shape[1] != 1:
            raise ValueError(f"Expected hidden_states shape [B, 1, S, hidden], got {input_shape}")
        batch, seq_len = input_shape[0], input_shape[2]  # seq_len = per-chip S_pad/sp
        if seq_len_actual is None:
            seq_len_actual = seq_len * self.sp_factor

        kv = ttnn.linear(hidden_states, self.wkv, memory_config=self.memory_config)
        gate = ttnn.linear(hidden_states, self.wgate, memory_config=self.memory_config)

        # wkv/wgate are contraction(row)-parallel -> partial-sum single-head KV/gate; TP all-reduce
        # (reduce_scatter + all_gather) each to the full head_dim, replicated across TP.
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
        n_windows = usable // self.compress_rate  # windows this chip owns (S_pad/sp is rate-aligned on mesh)
        t_real = seq_len_actual // self.compress_rate
        if n_windows > 0:
            # device: +position_bias then softmax over the window axis (per channel), weighted sum
            gate = ttnn.slice(gate, [0, 0, 0, 0], [batch, 1, usable, self.head_dim])
            gate = ttnn.reshape(gate, [batch, n_windows, self.compress_rate, self.head_dim])
            gate = ttnn.add(gate, self.position_bias)
            weights = ttnn.softmax(gate, dim=2, numeric_stable=True)

            kv = ttnn.slice(kv, [0, 0, 0, 0], [batch, 1, usable, self.head_dim])
            kv = ttnn.reshape(kv, [batch, n_windows, self.compress_rate, self.head_dim])
            pooled = ttnn.sum(ttnn.multiply(kv, weights), dim=2)

            compressed = ttnn.reshape(pooled, [batch, 1, n_windows, self.head_dim])
            compressed = ttnn.rms_norm(compressed, weight=self.kv_norm_weight, epsilon=self.rms_norm_eps)

            # RoPE (device) on the trailing rope_head_dim channels only (op caps head_dim <= 256).
            # Positions are GLOBAL window indices: chip r owns global windows [r*n_windows, (r+1)*n_windows),
            # so _cos_sin SP-shards the full arange(n_windows*sp) to match each chip's window slice.
            nope_dim = self.head_dim - self.rope_head_dim
            nope = ttnn.slice(compressed, [0, 0, 0, 0], [batch, 1, n_windows, nope_dim])
            rope = ttnn.slice(compressed, [0, 0, 0, nope_dim], [batch, 1, n_windows, self.head_dim])
            positions = (torch.arange(n_windows * self.sp_factor) * self.compress_rate).unsqueeze(0)
            cos, sin = self._cos_sin(positions)
            rope = ttnn.experimental.rotary_embedding_llama(rope, cos, sin, self.trans_mat, is_decode_mode=False)
            compressed_kv = ttnn.concat([nope, rope], dim=-1)

            # SP all-gather the per-chip windows -> full T_pad global, replicated; then trim the
            # pad-derived tail so only the T_real real compressed entries survive.
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
            if t_real < compressed_kv.shape[2]:
                compressed_kv = ttnn.slice(compressed_kv, [0, 0, 0, 0], [batch, 1, t_real, self.head_dim])
        else:
            compressed_kv = self._from_torch(torch.zeros(batch, 1, 0, self.head_dim))

        block_bias = None
        if seq_len_actual > 1 and t_real > 0:
            block_bias = hca_block_bias(position_ids, t_real, self.compress_rate)

        return compressed_kv, block_bias


class TtHCA(_TtHCABase):
    """DeepSeek-V4 Heavily Compressed Attention block (TTNN prefill), mirrors
    ``DeepseekV4Attention``. Brought up stage by stage: query/kv stems now; attention
    core + output projection to follow. Composes ``TtHCACompressor`` for the long-range
    compressed-KV branch. ``_q_stem`` / ``_kv_stem`` mirror the reference query path
    (L817-820) and sliding KV path (L822-823); the full ``forward`` assembles them with
    the compressor + attention once those land."""

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
        ccl_num_links: int | None = None,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ):
        self.device = device
        self.dtype = dtype
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
        self.tt_ccl = get_tt_ccl(device) if (self.is_mesh and self.tp_factor > 1) else None
        self.ccl_num_links = (2 if is_blackhole() else 1) if ccl_num_links is None else ccl_num_links

        # sink pre-divided by scale: SDPA scales BOTH QK and the sink by `scale` internally,
        # but the reference scales only QK -> divide the sink so the kernel's ×scale cancels.
        self.sinks_sdpa = self._from_torch(sinks.detach().reshape(1, self.num_heads, 1, 1) / self.scaling)

        # q_a_proj: TP-shard the contraction (hidden) so each chip matmuls its hidden shard to a
        # partial q_lora (all-reduced in _q_stem). q_b_proj: TP-shard the output (heads) so each
        # chip owns num_heads/tp heads.
        self.wq_a = self._to_tt_linear_weight(q_a_proj_weight, tp_shard_dim=2)
        self.wq_b = self._to_tt_linear_weight(q_b_proj_weight, tp_shard_dim=3)
        self.q_a_norm_weight = self._from_torch(q_a_norm_weight.detach().reshape(1, 1, 1, -1))
        self.q_b_norm_weight = self._from_torch(torch.ones(1, 1, 1, self.head_dim))
        # kv_proj: same contraction(row)-parallel scheme as wq_a. The single-head KV is all-reduced in
        # _kv_stem -> TP-replicated (every TP chip's query heads attend the same KV).
        self.wkv = self._to_tt_linear_weight(kv_proj_weight, tp_shard_dim=2)
        self.kv_norm_weight = self._from_torch(kv_norm_weight.detach().reshape(1, 1, 1, self.head_dim))

        # Grouped output projection: o_a_proj is block-diagonal (o_groups independent
        # (num_heads*head_dim/o_groups) -> o_lora_rank blocks); o_b_proj mixes to hidden.
        self.o_groups = int(o_groups)
        in_per_group = self.num_heads * self.head_dim // self.o_groups
        o_a_grouped = o_a_proj_weight.detach().view(self.o_groups, -1, in_per_group)
        self.wo_a = [self._to_tt_linear_weight(o_a_grouped[g]) for g in range(self.o_groups)]
        self.wo_b = self._to_tt_linear_weight(o_b_proj_weight)

        self.trans_mat = self._from_torch(get_rot_transformation_mat())

    @classmethod
    def from_reference(cls, device, reference, config, **kwargs) -> "TtHCA":
        # Forward the mesh/CCL config so the compressor rides the same SP/TP axes as the block.
        compressor_keys = ("sp_axis", "tp_axis", "topology", "ccl_num_links")
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
        """Query path (reference L817-820). ``hidden_states``: TTNN [B, 1, S/sp, hidden/tp];
        ``position_ids``: torch [B, S] (full). Returns ``q`` TTNN [B, num_heads/tp, S/sp, head_dim]."""
        input_shape = tuple(hidden_states.shape)
        if len(input_shape) != 4 or input_shape[1] != 1:
            raise ValueError(f"Expected hidden_states shape [B, 1, S, hidden], got {input_shape}")
        batch, seq_len = input_shape[0], input_shape[2]
        num_heads_local = self.num_heads // self.tp_factor

        q = ttnn.linear(hidden_states, self.wq_a, memory_config=self.memory_config)

        # q_a_proj is contraction(row)-parallel: hidden is TP-sharded on columns and wq_a on the
        # contraction, so each chip holds a full-shape but partial-sum q_lora -> TP all-reduce
        # (reduce_scatter + all_gather) rebuilds the full q_lora latent, replicated across TP.
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

        q = ttnn.reshape(q, [batch, seq_len, num_heads_local, self.head_dim])
        q = ttnn.permute(q, (0, 2, 1, 3))
        q = ttnn.rms_norm(q, weight=self.q_b_norm_weight, epsilon=self.rms_norm_eps)

        nope_dim = self.head_dim - self.rope_head_dim
        nope = ttnn.slice(q, [0, 0, 0, 0], [batch, num_heads_local, seq_len, nope_dim])
        rope = ttnn.slice(q, [0, 0, 0, nope_dim], [batch, num_heads_local, seq_len, self.head_dim])
        cos, sin = self._cos_sin(position_ids)
        rope = ttnn.experimental.rotary_embedding_llama(rope, cos, sin, self.trans_mat, is_decode_mode=False)
        return ttnn.concat([nope, rope], dim=-1)

    def _kv_stem(self, hidden_states, position_ids: torch.Tensor):
        """Sliding KV path (reference L822-823, K == V). ``hidden_states``: TTNN [B, 1, S/sp, hidden/tp].
        Returns single-head ``sliding_kv`` TTNN [B, 1, S/sp, head_dim], TP-replicated + SP-sharded
        (full S in stateless single-shot; sliding-window truncation is chunked-prefill only)."""
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

    def _attn_mask(self, batch: int, seq_len: int, block_bias: torch.Tensor, sk_pad: int):
        """Combined additive mask [B, 1, S, sk_pad] (TILE): sliding-window causal (width
        ``sliding_window``) over the S main keys, block_bias over the T compressed keys, then
        -inf over the [S+T, sk_pad) tile-padding so SDPA ignores the zero-padded KV columns.
        Built on host, uploaded to device."""
        t_len = block_bias.shape[-1]
        i = torch.arange(seq_len).view(seq_len, 1)
        j = torch.arange(seq_len).view(1, seq_len)
        allowed = (j <= i) & (i - j < self.sliding_window)
        main = torch.zeros(seq_len, seq_len).masked_fill(~allowed, float("-inf"))
        full = torch.full((batch, 1, seq_len, sk_pad), float("-inf"))
        full[..., :seq_len] = main.view(1, 1, seq_len, seq_len)
        full[..., seq_len : seq_len + t_len] = block_bias.to(torch.float32)
        return self._from_torch(full)

    def _attention(self, q, sliding_kv, compressed_kv, block_bias: torch.Tensor, position_ids: torch.Tensor):
        """Attention core (reference L833/843/718-746/869). Inputs: ``q`` [B,64,S,512],
        ``sliding_kv`` [B,1,S,512], ``compressed_kv`` [B,1,T,512], ``block_bias`` host
        torch [B,1,S,T], ``position_ids`` torch [B,S]. Concats KV, runs SDPA with the
        combined mask + per-head sink, then undoes V's RoPE. Returns [B,64,S,512]."""
        batch, seq_len = q.shape[0], q.shape[2]
        # Pad the concatenated KV seq (S + T) up to a multiple of 32: SDPA tile-pads a
        # non-aligned Sk with ZEROS and a provided mask's pad columns default to 0 (= attend),
        # polluting the softmax -- pad explicitly and mark those columns -inf in _attn_mask.
        sk = seq_len + compressed_kv.shape[2]
        sk_pad = ((sk + 31) // 32) * 32
        parts = [sliding_kv, compressed_kv]
        if sk_pad > sk:
            parts.append(self._from_torch(torch.zeros(batch, 1, sk_pad - sk, self.head_dim)))
        kv = ttnn.concat(parts, dim=2)
        mask = self._attn_mask(batch, seq_len, block_bias, sk_pad)

        attn = ttnn.transformer.scaled_dot_product_attention(
            q,
            kv,
            kv,
            attn_mask=mask,
            is_causal=False,
            scale=self.scaling,
            attention_sink=self.sinks_sdpa,
        )

        nope_dim = self.head_dim - self.rope_head_dim
        nope = ttnn.slice(attn, [0, 0, 0, 0], [batch, self.num_heads, seq_len, nope_dim])
        rope = ttnn.slice(attn, [0, 0, 0, nope_dim], [batch, self.num_heads, seq_len, self.head_dim])
        cos, sin = self._cos_sin(position_ids, negate_sin=True)
        rope = ttnn.experimental.rotary_embedding_llama(rope, cos, sin, self.trans_mat, is_decode_mode=False)
        return ttnn.concat([nope, rope], dim=-1)

    def _o_proj(self, attn):
        """Grouped output projection (reference L871-873). ``attn`` [B, num_heads, S, head_dim]
        -> reshape heads into o_groups blocks -> per-group o_a_proj (block-diagonal) -> concat
        -> o_b_proj. Returns [B, 1, S, hidden]."""
        batch, _, seq_len, _ = attn.shape
        x = ttnn.permute(attn, (0, 2, 1, 3))  # [B, S, num_heads, head_dim]
        x = ttnn.reshape(x, [batch, 1, seq_len, self.num_heads * self.head_dim])  # [B, 1, S, num_heads*head_dim]

        in_per_group = self.num_heads * self.head_dim // self.o_groups
        groups = []
        for g in range(self.o_groups):
            xg = ttnn.slice(x, [0, 0, 0, g * in_per_group], [batch, 1, seq_len, (g + 1) * in_per_group])
            groups.append(ttnn.linear(xg, self.wo_a[g], memory_config=self.memory_config))
        grouped = ttnn.concat(groups, dim=-1)  # [B, 1, S, o_groups * o_lora_rank]
        return ttnn.linear(grouped, self.wo_b, memory_config=self.memory_config)  # [B, 1, S, hidden]

    def forward(self, hidden_states, position_ids: torch.Tensor):
        """Full HCA block (prefill, single-shot), mirrors ``DeepseekV4Attention.forward``.
        ``hidden_states`` TTNN [B, 1, S, hidden]; ``position_ids`` torch [B, S]. Returns
        [B, 1, S, hidden]: query/kv stems -> compressor -> attention core -> output proj."""
        q = self._q_stem(hidden_states, position_ids)
        sliding_kv = self._kv_stem(hidden_states, position_ids)
        compressed_kv, block_bias = self.compressor(hidden_states, position_ids)
        attn = self._attention(q, sliding_kv, compressed_kv, block_bias, position_ids)
        return self._o_proj(attn)
