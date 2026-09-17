# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Reusable TT compressor infrastructure and the HCA compressor."""

from __future__ import annotations

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.tt.mla.rope import get_rot_transformation_mat
from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl


def rope_table_tokens(max_seq_len: int, chunk_tokens: int) -> int:
    """Cover the context rounded to whole chunks plus one padded slab."""
    return -(-int(max_seq_len) // chunk_tokens) * chunk_tokens + chunk_tokens


class TtCompressorUtils:
    """Mesh, tensor, and indexed-RoPE operations used by compressors and their consumers."""

    def __init__(
        self,
        device,
        *,
        rotary_emb,
        sp_axis: int,
        tp_axis: int,
        dtype,
        weights_dtype,
        memory_config,
    ):
        self.device = device
        self.rotary_emb = rotary_emb
        self.dtype = dtype
        self.weights_dtype = weights_dtype
        self.memory_config = memory_config
        self.is_mesh = hasattr(device, "shape")
        self.sp_axis, self.tp_axis = sp_axis, tp_axis
        self.sp_factor = device.shape[sp_axis] if self.is_mesh else 1
        self.tp_factor = device.shape[tp_axis] if self.is_mesh else 1

    def to_tt_linear_weight(self, weight: torch.Tensor, tp_shard_dim: int | None = None):
        torch_weight = weight.detach().transpose(-2, -1).contiguous().unsqueeze(0).unsqueeze(0)
        return self.from_torch(
            torch_weight, mesh_mapper=self.mesh_mapper(tp_dim=tp_shard_dim), dtype=self.weights_dtype
        )

    def from_torch(self, x: torch.Tensor, mesh_mapper=None, dtype=None, layout=ttnn.TILE_LAYOUT, on_device=True):
        """Convert a host tensor, replicating it across a mesh by default."""
        if self.is_mesh and mesh_mapper is None:
            mesh_mapper = ttnn.ReplicateTensorToMesh(self.device)
        tiled = on_device and layout == ttnn.TILE_LAYOUT
        return ttnn.from_torch(
            x,
            device=self.device if on_device else None,
            dtype=dtype or self.dtype,
            layout=layout,
            memory_config=self.memory_config if tiled else None,
            mesh_mapper=mesh_mapper,
        )

    def mesh_mapper(self, sp_dim: int | None = None, tp_dim: int | None = None):
        """Map host dimensions onto the configured SP and TP mesh axes."""
        if not self.is_mesh:
            return None
        dims = [None, None]
        if sp_dim is not None and self.sp_factor > 1:
            dims[self.sp_axis] = sp_dim
        if tp_dim is not None and self.tp_factor > 1:
            dims[self.tp_axis] = tp_dim
        if dims == [None, None]:
            return ttnn.ReplicateTensorToMesh(self.device)
        return ttnn.ShardTensor2dMesh(self.device, mesh_shape=tuple(self.device.shape), dims=dims)

    def scalar_buffer(self, dtype, shape=(1, 1, 1, 1), layout=ttnn.TILE_LAYOUT):
        """Allocate a device scalar buffer that can be overwritten without reallocating."""
        return self.from_torch(torch.zeros(*shape, dtype=torch.int32), dtype=dtype, layout=layout)

    def push_scalar(self, buf, value):
        """Overwrite a device buffer with one host scalar."""
        host_dtype = torch.float32 if buf.dtype == ttnn.float32 else torch.int32
        host = self.from_torch(
            torch.full(tuple(buf.shape), value, dtype=host_dtype),
            dtype=buf.dtype,
            layout=buf.layout,
            on_device=False,
        )
        ttnn.copy_host_to_device_tensor(host, buf)
        return buf

    def build_rope_table(self, count: int, stride: int, layer_type: str = "compress"):
        """Build replicated cos/sin tables for compressed or token positions.

        ``layer_type`` picks the reference's rope variant: "compress" is the YaRN-scaled table the
        HCA/CSA layers share with their compressor, "main" the plain theta=10000 one a sliding-only
        layer uses (``DeepseekV4Attention.rope_layer_type``)."""
        positions = (torch.arange(count) * stride).unsqueeze(0)
        cos, sin = self.rotary_emb(torch.zeros(1), position_ids=positions.to(torch.long), layer_type=layer_type)
        return tuple(self.from_torch(t.repeat_interleave(2, dim=-1)) for t in (cos, sin))

    def rope_index_base(self, rows: int):
        """Build the constant and mutable halves of an indexed-RoPE gather index."""
        const = self.from_torch(
            torch.arange(self.sp_factor * rows, dtype=torch.int32).view(self.sp_factor, rows),
            mesh_mapper=self.mesh_mapper(sp_dim=0),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        return const, self.scalar_buffer(ttnn.uint32, shape=(1, 1), layout=ttnn.ROW_MAJOR_LAYOUT)

    def rope_index(self, index_base, base: int):
        """Build this slab's gather index from its constant and scalar halves."""
        const, buf = index_base
        return ttnn.add(const, self.push_scalar(buf, base))

    def rope_gather(self, table, index):
        """Gather indexed rows from a replicated RoPE table."""
        out = []
        for tensor in table:
            gathered = ttnn.embedding(index, tensor, layout=ttnn.TILE_LAYOUT, dtype=self.dtype)
            out.append(ttnn.reshape(gathered, [1, 1, gathered.shape[-2], gathered.shape[-1]]))
        return tuple(out)


class TtCompressorBase(LightweightModule):
    """Common device, mesh, and input setup for TT compressor implementations."""

    def __init__(
        self,
        device,
        *,
        rotary_emb,
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
        self.rotary_emb = rotary_emb

        self.is_mesh = hasattr(device, "shape")
        self.sp_axis, self.tp_axis = sp_axis, tp_axis
        self.sp_factor = device.shape[sp_axis] if self.is_mesh else 1
        self.tp_factor = device.shape[tp_axis] if self.is_mesh else 1
        self.ccl_topology = topology
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

    @staticmethod
    def prepare_input(hidden: torch.Tensor, sp_factor: int, compress_rate: int):
        """Pad sequence rows so every SP shard owns whole compression windows."""
        seq_len_actual = hidden.shape[1]
        align = compress_rate * sp_factor
        pad = (-seq_len_actual) % align
        if pad:
            hidden = torch.nn.functional.pad(hidden, (0, 0, 0, pad))
        return hidden, seq_len_actual


class TtHCACompressor(TtCompressorBase):
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
        super().__init__(
            device,
            rotary_emb=rotary_emb,
            sp_axis=sp_axis,
            tp_axis=tp_axis,
            topology=topology,
            dtype=dtype,
            weights_dtype=weights_dtype,
            memory_config=memory_config,
        )
        self.head_dim = int(head_dim)
        self.compress_rate = int(compress_rate)
        self.rope_head_dim = int(rope_head_dim)
        self.rms_norm_eps = float(rms_norm_eps)

        self.wkv = self.ops.to_tt_linear_weight(kv_proj_weight, tp_shard_dim=2)
        self.wgate = self.ops.to_tt_linear_weight(gate_proj_weight, tp_shard_dim=2)
        self.position_bias = self.ops.from_torch(
            position_bias.detach().reshape(1, 1, self.compress_rate, self.head_dim)
        )
        self.kv_norm_weight = self.ops.from_torch(kv_norm_weight.detach().reshape(1, 1, 1, self.head_dim))
        self.trans_mat = self.ops.from_torch(get_rot_transformation_mat())
        self._entry_rope = None
        self._entry_index = None
        self._mask_consts = None

    def alloc_tables(self, max_seq_len: int, chunk_tokens: int, mask_width: int):
        """Build the indexed-RoPE and mask constants needed by forward."""
        tokens = rope_table_tokens(max_seq_len, chunk_tokens)
        self._entry_rope = self.ops.build_rope_table(-(-tokens // self.compress_rate), self.compress_rate)
        self._entry_index = self.ops.rope_index_base(chunk_tokens // self.sp_factor // self.compress_rate)
        self._mask_consts = self._build_mask_consts(chunk_tokens, mask_width)

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

    def _build_mask_consts(self, seq_global: int, width: int):
        """Build the constant vectors and mutable scalars for the compressed mask."""
        sp_mapper = self.ops.mesh_mapper(sp_dim=2)
        rate = self.compress_rate
        return {
            "seq": seq_global,
            "thr": self.ops.from_torch(
                ((torch.arange(seq_global) + 1) // rate).float().view(1, 1, seq_global, 1),
                sp_mapper,
                dtype=ttnn.float32,
            ),
            "ic": self.ops.from_torch(
                torch.arange(seq_global).float().view(1, 1, seq_global, 1),
                sp_mapper,
                dtype=ttnn.float32,
            ),
            "w": self.ops.from_torch(torch.arange(width).float().view(1, 1, 1, width), dtype=ttnn.float32),
            "ec": self.ops.scalar_buffer(ttnn.float32),
            "rl": self.ops.scalar_buffer(ttnn.float32),
        }

    def _mask_block(self, seq: int, first_window_position: int, seq_len_actual: int):
        """Build additive compressed-cache mask columns on device."""
        rate = self.compress_rate
        seq_global = seq * self.sp_factor
        consts = self._mask_consts
        assert consts is not None and consts["seq"] == seq_global, (
            f"mask constants cover {None if consts is None else consts['seq']} query rows but this call has "
            f"{seq_global}; alloc_tables has to be given the slab forward is called with"
        )
        within = ttnn.lt(
            consts["w"],
            ttnn.add(
                consts["thr"],
                self.ops.push_scalar(consts["ec"], first_window_position // rate),
            ),
        )
        live = ttnn.lt(consts["ic"], self.ops.push_scalar(consts["rl"], seq_len_actual))
        return ttnn.typecast(ttnn.log(ttnn.multiply(within, live)), self.dtype)

    def forward(
        self,
        hidden_states,
        seq_len_actual: int | None = None,
        first_window_position: int = 0,
    ):
        """Compress an SP/TP-sharded hidden-state slab and build its mask columns."""
        input_shape = tuple(hidden_states.shape)
        if len(input_shape) != 4 or input_shape[1] != 1:
            raise ValueError(f"Expected hidden_states shape [B, 1, S, hidden], got {input_shape}")
        batch, seq_len = input_shape[0], input_shape[2]
        if seq_len_actual is None:
            seq_len_actual = seq_len * self.sp_factor

        kv = ttnn.linear(hidden_states, self.wkv, memory_config=self.memory_config)
        gate = ttnn.linear(hidden_states, self.wgate, memory_config=self.memory_config)

        if self.tp_factor > 1:
            for name in ("kv", "gate"):
                tensor = kv if name == "kv" else gate
                tensor = ttnn.experimental.reduce_scatter_minimal_async(
                    tensor,
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
                tensor = ttnn.experimental.all_gather_async(
                    tensor,
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
                    kv = tensor
                else:
                    gate = tensor

        n_windows = seq_len // self.compress_rate
        t_real = seq_len_actual // self.compress_rate
        assert n_windows > 0, (
            f"each chip needs at least one whole compression window: {seq_len} rows is under "
            f"compress_rate {self.compress_rate}; run prepare_input on the hidden states first"
        )
        gate = ttnn.reshape(gate, [batch, n_windows, self.compress_rate, self.head_dim])
        gate = ttnn.add(gate, self.position_bias)
        weights = ttnn.softmax(gate, dim=2, numeric_stable=True)

        kv = ttnn.reshape(kv, [batch, n_windows, self.compress_rate, self.head_dim])
        pooled = ttnn.sum(ttnn.multiply(kv, weights), dim=2)

        compressed = ttnn.reshape(pooled, [batch, 1, n_windows, self.head_dim])
        compressed = ttnn.rms_norm(compressed, weight=self.kv_norm_weight, epsilon=self.rms_norm_eps)

        nope_dim = self.head_dim - self.rope_head_dim
        nope = ttnn.slice(compressed, [0, 0, 0, 0], [batch, 1, n_windows, nope_dim])
        rope = ttnn.slice(compressed, [0, 0, 0, nope_dim], [batch, 1, n_windows, self.head_dim])
        idx = self.ops.rope_index(self._entry_index, first_window_position // self.compress_rate)
        cos, sin = self.ops.rope_gather(self._entry_rope, idx)
        rope = ttnn.experimental.rotary_embedding_llama(rope, cos, sin, self.trans_mat, is_decode_mode=False)
        compressed_kv = ttnn.concat([nope, rope], dim=-1)

        if self.sp_factor > 1:
            compressed_kv = ttnn.experimental.all_gather_async(
                compressed_kv,
                dim=2,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=self.sp_axis),
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=self.sp_axis),
                num_links=self.ccl_num_links,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=self.ccl_topology,
                cluster_axis=self.sp_axis,
            )

        mask_block = None
        if seq_len_actual > 1 and t_real > 0:
            mask_block = self._mask_block(seq_len, first_window_position, seq_len_actual)

        return compressed_kv, mask_block
