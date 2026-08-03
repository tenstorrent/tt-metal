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
            dtype=self.weights_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self.memory_config,
            mesh_mapper=mesh_mapper,
        )

    def _from_torch(self, x: torch.Tensor, mesh_mapper=None, dtype=None):
        if self.is_mesh and mesh_mapper is None:
            mesh_mapper = ttnn.ReplicateTensorToMesh(self.device)
        return ttnn.from_torch(
            x.to(torch.bfloat16),
            device=self.device,
            dtype=dtype or self.dtype,
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
        dtype=ttnn.bfloat16,
        # Matmul weights only. Norms / position_bias / sinks / trans_mat stay bf16 -- they never
        # enter a matmul, so quantizing them costs accuracy and saves nothing (mirrors mla.py).
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
        # The compressor rides BOTH axes: TP all-reduce for wkv/wgate, SP all-gather for the pooled
        # compressed KV -> needs the CCL object whenever either factor > 1.
        self.ccl_topology = topology
        self.tt_ccl = get_tt_ccl(device) if (self.is_mesh and (self.sp_factor > 1 or self.tp_factor > 1)) else None
        self.ccl_num_links = 2 if is_blackhole() else 1  # Blackhole trains 2 fabric routing planes, others 1

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

    def forward(
        self,
        hidden_states,
        position_ids: torch.Tensor,
        seq_len_actual: int | None = None,
        first_window_position: int = 0,
        total_entries: int | None = None,
    ):
        """``hidden_states``: TTNN [B, 1, S_pad/sp, hidden/tp] (seq host-padded to a multiple of
        compress_rate*sp via ``prepare_input``). ``position_ids``: torch [B, S_real] (real positions).
        ``seq_len_actual``: real pre-pad length; only ``T_real = S_real // compress_rate`` compressed
        entries survive (pad-derived windows are trimmed). ``None`` -> no padding (single-shot: use the
        full seq). ``first_window_position``: global token position of this call's FIRST window
        (``entry_count * compress_rate`` under chunked prefill) — without it a later chunk would rotate
        its entries as if the sequence restarted.

        Returns ``compressed_kv`` [B, 1, S_pad/compress_rate, head_dim] (TP-replicated, SP-gathered) —
        one entry per padded window, so the width follows the tensor, not the real length. Trailing
        pad-derived entries are left in: the caller knows only the first ``S_real/compress_rate`` are
        real and the mask drops the rest. Also returns host ``block_bias`` [B, 1, S_real, entries]."""
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
            positions = (
                torch.arange(n_windows * self.sp_factor) * self.compress_rate + first_window_position
            ).unsqueeze(0)
            cos, sin = self._cos_sin(positions)
            rope = ttnn.experimental.rotary_embedding_llama(rope, cos, sin, self.trans_mat, is_decode_mode=False)
            compressed_kv = ttnn.concat([nope, rope], dim=-1)

            # SP all-gather the per-chip windows -> the full padded window count, replicated. The block is
            # deliberately NOT trimmed to the real entry count: its width then depends only on the (fixed)
            # chunk width, so a short final chunk presents the same shape as a full one and adds no
            # program. How many of these entries are real is carried by the caller's entry_count and the
            # attention mask -infs the rest; only a FINAL chunk may be short, so nothing reads them later.
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

        # block_bias must span every compressed entry the queries can see -- under chunked prefill that
        # includes earlier chunks' entries, not just this call's (matching the reference, whose bias is
        # built over the running cache). total_entries=None -> single-shot, where the two coincide.
        bias_entries = t_real if total_entries is None else total_entries
        block_bias = None
        if seq_len_actual > 1 and bias_entries > 0:
            block_bias = hca_block_bias(position_ids, bias_entries, self.compress_rate)

        return compressed_kv, block_bias


class TtHCAState:
    """Per-layer chunked-prefill state, carried across chunks and handed to ``TtHCA.forward``
    (owned by the caller, like MLA's ``kvpe_cache`` — not hidden inside the module).

    Both device tensors keep a FIXED shape for the whole prefill; only their contents advance. That is
    what lets one compiled program serve every chunk. ``entry_count`` / ``kv_actual`` say how much of
    each is real — the attention mask reads them to ``-inf`` the rest.
    """

    def __init__(self, compressed_kv, sliding_carry):
        self.compressed_kv = compressed_kv  # [B, 1, compressed_capacity, head_dim]; first entry_count real
        self.sliding_carry = sliding_carry  # [B, 1, sliding_window, head_dim]; the raw KV a new chunk looks back into
        self.entry_count = 0  # compressed entries emitted so far
        self.kv_actual = 0  # real tokens consumed so far


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
        dtype=ttnn.bfloat16,
        # Matmul weights only. Norms / position_bias / sinks / trans_mat stay bf16 -- they never
        # enter a matmul, so quantizing them costs accuracy and saves nothing (mirrors mla.py).
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
        # Rides both axes: TP all-reduce (q/kv stems) and SP all-gather (sliding KV in _attention),
        # so the CCL object is needed whenever either factor > 1.
        self.tt_ccl = get_tt_ccl(device) if (self.is_mesh and (self.sp_factor > 1 or self.tp_factor > 1)) else None
        self.ccl_num_links = 2 if is_blackhole() else 1  # Blackhole trains 2 fabric routing planes, others 1

        # sink pre-divided by scale: SDPA scales BOTH QK and the sink by `scale` internally, but the
        # reference scales only QK -> divide the sink so the kernel's ×scale cancels. Per-head, so it
        # is TP-sharded on heads (dim 1) to match the head-sharded q; every other axis is broadcast.
        sinks_host = sinks.detach().reshape(1, self.num_heads, 1, 1) / self.scaling
        if self.is_mesh and self.tp_factor > 1:
            sink_dims = [None, None]
            sink_dims[self.tp_axis] = 1
            sink_mapper = ttnn.ShardTensor2dMesh(device, mesh_shape=tuple(device.shape), dims=sink_dims)
            self.sinks_sdpa = self._from_torch(sinks_host, mesh_mapper=sink_mapper)
        else:
            self.sinks_sdpa = self._from_torch(sinks_host)

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
        # Groups partition the heads, so a TP chip owns whole groups: keep o_a as ONE batched
        # [1, o_groups, in_per_group, o_lora_rank] weight sharded on the group axis (dim 1) and run a
        # single batched matmul -- each chip applies only its own groups, no collective.
        self.o_groups = int(o_groups)
        in_per_group = self.num_heads * self.head_dim // self.o_groups
        o_a_grouped = o_a_proj_weight.detach().view(self.o_groups, -1, in_per_group).transpose(1, 2).unsqueeze(0)
        o_a_mapper = None
        if self.is_mesh and self.tp_factor > 1:
            o_a_dims = [None, None]
            o_a_dims[self.tp_axis] = 1
            o_a_mapper = ttnn.ShardTensor2dMesh(device, mesh_shape=tuple(device.shape), dims=o_a_dims)
        # Goes through _from_torch (not _to_tt_linear_weight) because it is pre-grouped, but it IS a
        # matmul weight -> same dtype as the rest.
        self.wo_a = self._from_torch(o_a_grouped, mesh_mapper=o_a_mapper, dtype=self.weights_dtype)
        # o_b_proj contracts over all o_groups*o_lora_rank columns while a chip holds only its own
        # groups' slice -> contraction(row)-parallel, reduce-scattered in _o_proj.
        self.wo_b = self._to_tt_linear_weight(o_b_proj_weight, tp_shard_dim=2)

        self.trans_mat = self._from_torch(get_rot_transformation_mat())

    def alloc_state(self, max_seq_len: int, batch: int = 1) -> TtHCAState:
        """Allocate this layer's chunked-prefill state, sized once for the longest context it will serve
        (``max_seq_len`` — the same serving-capacity knob the runner resolves for the KV cache, so the
        capacity lives with the cache rather than with the layer). Both tensors then keep that shape for
        every chunk, which is what lets one compiled program serve the whole prefill.

        Contents start zeroed and nothing is read before it is written: the mask keys off
        ``entry_count`` / ``kv_actual`` and -infs everything past them, including chunk 0's empty carry."""
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

        # [B, 1, S, heads*head_dim] -> [B, heads, S, head_dim]. Done as one fused op rather than
        # reshape+permute: the intermediate [B, S, heads, head_dim] puts heads (16) on dim -2, which
        # TILE pads to 32 -- the pair cost 1,152 us/chunk where this costs a fraction. num_kv_heads=0
        # selects the Q-only form (same call MLA makes at mla.py:860).
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

    def _attn_mask(
        self,
        batch: int,
        seq_len: int,
        block_bias: torch.Tensor,
        sk_pad: int,
        kv_actual: int = 0,
        carry_len: int = 0,
    ):
        """Combined additive mask [B, 1, S, sk_pad] (TILE) over the key layout
        ``[carry | chunk | compressed | tile-pad]``: sliding-window causal across the raw keys,
        block_bias across the compressed ones, -inf over the rest.

        ``seq_len`` is the GLOBAL query length of this chunk; ``kv_actual`` the global position of its
        first query; ``carry_len`` how many raw keys precede it (0 in single-shot and in chunk 0).
        Built on host from global positions; on a mesh it is SP-sharded on the query rows (dim 2) to
        match the SP-sharded q — head-independent so TP-replicated, and full over the key columns
        since every query attends all keys."""
        t_len = block_bias.shape[-1]
        raw = carry_len + seq_len  # carry and chunk keys are CONTIGUOUS in global positions
        i = torch.arange(seq_len).view(-1, 1) + kv_actual  # global query positions
        j = torch.arange(raw).view(1, -1) + (kv_actual - carry_len)  # global key positions
        # Same sliding-causal rule as ever, just read in global positions; j >= 0 drops the carry
        # columns of the very first chunk, where those positions do not exist yet.
        allowed = (j >= 0) & (j <= i) & (i - j < self.sliding_window)
        full = torch.full((batch, 1, seq_len, sk_pad), float("-inf"))
        full[..., :raw] = torch.zeros(seq_len, raw).masked_fill(~allowed, float("-inf")).view(1, 1, seq_len, raw)
        # block_bias covers the real query rows; padded query rows (if any) keep -inf and are discarded.
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
        """Attention core (reference L833/843/718-746/869). Inputs: ``q`` [B, num_heads/tp, S/sp, 512],
        ``sliding_kv`` [B,1,S/sp,512] (SP-sharded), ``compressed_kv`` [B,1,T,512] (replicated),
        ``block_bias`` host torch [B,1,S,T], ``position_ids`` torch [B,S]. ``carry`` [B,1,sliding_window,512]
        is the previous chunk's raw KV tail (chunked prefill; None in single-shot), ``kv_actual`` the global
        position of this chunk's first query. SP-gathers the sliding KV to full S, concats
        ``[carry | sliding | compressed | pad]``, runs single-device SDPA per chip (q SP+TP-sharded, KV
        replicated, mask SP-sharded, sink TP-sharded), then undoes V's RoPE.

        Returns ``(attn, next_carry)`` — attn [B, num_heads/tp, S/sp, 512], and the last ``sliding_window``
        rows of the gathered sliding KV, which is what the NEXT chunk needs to look back into. The carry
        can only be taken here: before the gather those rows live on the last SP chip alone."""
        batch, seq_local = q.shape[0], q.shape[2]
        seq_len = seq_local * self.sp_factor  # global query/main-key length
        num_heads_local = self.num_heads // self.tp_factor
        carry_len = 0 if carry is None else carry.shape[2]

        # Every query attends all keys, so the single-device SDPA needs the full sliding KV on each chip:
        # gather the SP-sharded sliding KV to full S (compressed KV is already replicated).
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

        # The next chunk's look-back: the tail of THIS chunk's raw KV, now that it is gathered.
        next_carry = ttnn.slice(
            sliding_kv,
            [0, 0, seq_len - self.sliding_window, 0],
            [batch, 1, seq_len, self.head_dim],
        )

        # Pad the concatenated KV seq (carry + S + T) up to a multiple of 32: SDPA tile-pads a
        # non-aligned Sk with ZEROS and a provided mask's pad columns default to 0 (= attend),
        # polluting the softmax -- pad explicitly and mark those columns -inf in _attn_mask.
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
                # Measured on 8x4, chunk 4096 (Sq/chip 512, Sk 4256). SDPA device time:
                #   q (at k=32):  32 -> 3,096 us | 64 -> 2,185 | 128 -> 1,205 | 256 -> 2,318 | 512 -> L1 OOM
                #   k (at q=128): 32 -> 1,205    | 128 ->  994 | 160 -> 1,048 | 192, 256 -> L1 OOM
                # 128/128 is 3.1x the 32/32 default. Both curves are V-shaped, so the optimum is not
                # simply the largest that fits in L1 -- 160 fits and is slower than 128.
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
        """Grouped output projection (reference L871-873). ``attn`` [B, num_heads/tp, S/sp, head_dim]
        -> heads regrouped into the chip's own o_groups/tp blocks -> batched o_a_proj (block-diagonal,
        purely local: groups partition heads) -> o_b_proj (contraction-parallel) -> TP reduce-scatter.
        Returns [B, 1, S/sp, hidden/tp] — the same layout the block takes as input."""
        batch, _, seq_len, _ = attn.shape
        in_per_group = self.num_heads * self.head_dim // self.o_groups
        groups_local = self.o_groups // self.tp_factor

        # Regroup with LEADING-axis reshapes only. The obvious reshape/permute route puts groups_local
        # (2) or heads (16) on dim -2, which TILE pads to 32 -- [B, S, 2, 4096] holds 8.4 MB of data in
        # 134 MB of tiles, and the five ops it took cost 2,594 us/chunk. Here dims -2/-1 never change,
        # so the tiling is untouched and both reshapes are metadata-only (~3 us).
        #
        # nlp_concat_heads reads dim 0 as batch and dim 1 as heads, so presenting attn as
        # [groups, heads_per_group, ...] makes it emit one in_per_group-wide row per group in a single
        # call. That also bounds its L1 circular buffers by in_per_group (a model constant) instead of
        # heads_local*head_dim (a mesh function): concat-heads CBs are ~2 * width * 32 * 2 B, so the
        # all-heads form needs 4.2 MB at tp=1 / 2.1 MB at tp=2 against a 1.5 MB budget. MLA calls it on
        # all heads (mla.py:1016) and gets away with it only because its v_head_dim is 128, not 512.
        x = ttnn.reshape(attn, [groups_local, attn.shape[1] // groups_local, seq_len, self.head_dim])
        x = ttnn.experimental.nlp_concat_heads(x, memory_config=self.memory_config)  # [groups, 1, S, in_per_group]
        # Group axis must be dim 1 for the batched matmul, and dim 0 must be 1 for the weight to broadcast.
        x = ttnn.reshape(x, [batch, groups_local, seq_len, in_per_group])

        grouped = ttnn.linear(x, self.wo_a, memory_config=self.memory_config)  # [B, groups_local, S, o_lora_rank]
        o_lora_rank = grouped.shape[-1]
        grouped = ttnn.concat(
            [ttnn.slice(grouped, [0, g, 0, 0], [batch, g + 1, seq_len, o_lora_rank]) for g in range(groups_local)],
            dim=-1,
        )  # [B, 1, S, groups_local*o_lora_rank]

        out = ttnn.linear(grouped, self.wo_b, memory_config=self.memory_config)  # partial-sum [B,1,S,hidden]

        # Contraction-parallel o_b -> partial sums. Reduce-scatter (NOT a full all-reduce) both sums them
        # and slices the result to hidden/tp, which is exactly the layout the next block expects.
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
        """One HCA chunk, mirrors ``DeepseekV4Attention.forward``.

        ``hidden_states`` TTNN [B, 1, S_pad/sp, hidden/tp] (seq host-padded to compress_rate*sp via
        ``prepare_input``); ``position_ids`` torch [B, S_real] holding this chunk's GLOBAL positions;
        ``seq_len_actual`` its real pre-pad length. ``state`` carries the compressed-KV cache and the raw
        look-back across chunks and is advanced in place; ``None`` means single-shot, which is simply a
        one-chunk prefill — a state sized to this call is made here so the compute path is identical
        either way (no separate single-shot branch).

        Returns [B, 1, S_pad/sp, hidden/tp]; the caller keeps the first S_real rows."""
        batch = hidden_states.shape[0]
        seq_pad_global = hidden_states.shape[2] * self.sp_factor
        compress_rate = self.compressor.compress_rate
        real_len = seq_pad_global if seq_len_actual is None else seq_len_actual

        # Prefill serves one request per call (concurrent users get their own state), matching MLA --
        # which assumes the same but only says so in a comment, so a batched input there fails silently.
        assert batch == 1, f"HCA prefill expects batch 1, got {batch}"

        # Below one full compression window there is no compressed KV at all (block_bias would be None
        # and the attention core has no compressed columns to mask). Fail here rather than deeper in.
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
        # The cache write copies whole tiles, so a chunk may only start on a tile boundary. Chunk 0 always
        # does (offset 0); later chunks require chunk_len % (compress_rate * TILE_SIZE) == 0.
        assert state.entry_count % ttnn.TILE_SIZE == 0, (
            f"compressed cache write offset {state.entry_count} is not tile-aligned; chunk length must be "
            f"a multiple of compress_rate * TILE_SIZE ({compress_rate * ttnn.TILE_SIZE})"
        )

        # The stems/undo-RoPE need one position per PADDED row (the rope op matches cos/sin to the tensor
        # seq), so extend the real positions; the compressor keeps the real ones (they drive block_bias).
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
        # Publish this chunk's entries into the fixed-capacity cache; attention then reads the whole
        # cache every chunk (constant shape) and the mask -infs everything past total_entries.
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
