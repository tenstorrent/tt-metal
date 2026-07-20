# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Four-chip tensor-parallel Falcon3-10B decoder layer.

The selected single-chip :class:`OptimizedDecoder` is the numerical and local
program-config baseline. This module restores the compiler-proven TP=4 algebra:
QKV and gate/up are column parallel, O and down are row parallel, attention and
KV cache operate on rank-local heads, and the two row-parallel partials are
sum-reduced to a replicated, stack-compatible residual stream.

The target is deliberately fixed to the full 1x4 Blackhole p300c mesh available
to this stage. Supporting smaller or different mesh shapes is out of scope.
"""

from __future__ import annotations

import math
from typing import Mapping

import torch

import ttnn
from models.autoports.tiiuae_falcon3_10b_base.tt.functional_decoder import (
    EMITTED_BATCH,
    IR_REPRESENTATIVE_LAYER,
    _config_value,
    _resolve_layer_tensor,
)
from models.autoports.tiiuae_falcon3_10b_base.tt.optimized_decoder import (
    PRECISION_POLICIES,
    OptimizedDecoder,
    _advisor_matmul_program_config,
    _advisor_norm_memory_config,
    _compute_config,
    _core_grid_for_tiles,
    _dram_matmul_program_config,
    _dram_sharded_memory_config,
    _largest_divisor,
    _matmul_output_memory_config,
    _prefill_matmul_program_config,
    _sharded_memory_config,
    _width_sharded_output_memory_config,
)
from models.common.lightweightmodule import LightweightModule

TARGET_MESH_SHAPE = (1, 4)
TENSOR_PARALLEL_SIZE = 4
TENSOR_PARALLEL_AXIS = 1
DEFAULT_PAGE_BLOCK_SIZE = 32
LOGICAL_LOCAL_INTERMEDIATE = 23040 // TENSOR_PARALLEL_SIZE
# Eight-bank DRAM width sharding requires N % (32 * 8) == 0.
PADDED_LOCAL_INTERMEDIATE = 6144


def _mesh_weight(
    host: torch.Tensor,
    *,
    dtype,
    mesh_device,
    shard_dim: int | None,
    local_shape: tuple[int, int] | None = None,
    memory_config: ttnn.MemoryConfig | None = None,
) -> ttnn.Tensor:
    """Materialize a replicated or one-dimension tensor-parallel weight."""
    host = host.detach().to(torch.bfloat16).contiguous()
    mapper = (
        ttnn.ReplicateTensorToMesh(mesh_device)
        if shard_dim is None
        else ttnn.ShardTensorToMesh(mesh_device, dim=shard_dim)
    )
    if memory_config is None:
        if local_shape is None:
            raise ValueError("local_shape is required for DRAM-sharded weights")
        memory_config = _dram_sharded_memory_config(mesh_device, *local_shape)
    return ttnn.from_torch(
        host,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=memory_config,
        mesh_mapper=mapper,
    )


def _replicated_tensor(host, *, dtype, layout, mesh_device, memory_config):
    return ttnn.from_torch(
        host.contiguous(),
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=memory_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _rank_grouped_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, tp: int) -> torch.Tensor:
    """Pack ``[Q_rank,K_rank,V_rank]`` contiguously for every mesh rank."""
    q_chunks = q.transpose(-2, -1).chunk(tp, dim=-1)
    k_chunks = k.transpose(-2, -1).chunk(tp, dim=-1)
    v_chunks = v.transpose(-2, -1).chunk(tp, dim=-1)
    return torch.cat(
        [piece for rank in range(tp) for piece in (q_chunks[rank], k_chunks[rank], v_chunks[rank])], dim=-1
    )


def _rank_padded_column(weight: torch.Tensor, tp: int, padded_width: int) -> torch.Tensor:
    """Transpose, TP-column-shard, and pad each rank chunk independently."""
    chunks = weight.transpose(-2, -1).chunk(tp, dim=-1)
    pieces = [torch.nn.functional.pad(chunk, (0, padded_width - int(chunk.shape[-1]))) for chunk in chunks]
    return torch.cat(pieces, dim=-1)


def _rank_padded_row(weight: torch.Tensor, tp: int, padded_height: int) -> torch.Tensor:
    """Transpose, TP-row-shard, and append zero K rows to each rank chunk."""
    chunks = weight.transpose(-2, -1).chunk(tp, dim=-2)
    pieces = [torch.nn.functional.pad(chunk, (0, 0, 0, padded_height - int(chunk.shape[-2]))) for chunk in chunks]
    return torch.cat(pieces, dim=-2)


def _rank_packed_pair(first: torch.Tensor, second: torch.Tensor, tp: int) -> torch.Tensor:
    """Pack two already rank-grouped column projections within every TP rank."""
    first_chunks = first.chunk(tp, dim=-1)
    second_chunks = second.chunk(tp, dim=-1)
    return torch.cat([piece for rank in range(tp) for piece in (first_chunks[rank], second_chunks[rank])], dim=-1)


class DecodeAllReduceResources:
    """Mesh-wide persistent resources shared by every layer during decode.

    The async all-reduce needs an intermediate tensor four times wider than a
    rank-local partial and global semaphores on the worker subdevice.  One pool
    is sufficient for a sequential decoder stack; callers constructing several
    layers should pass the first layer's pool to the remaining layers.
    """

    def __init__(
        self,
        *,
        mesh_device,
        batch: int,
        hidden_size: int,
        residual_memory_config,
        residual_num_cores: int,
    ):
        self.mesh_device = mesh_device
        self.batch = batch
        self.hidden_size = hidden_size
        self.residual_memory_config = residual_memory_config
        rows = 32 * math.ceil(batch / 32)
        grid_size = mesh_device.compute_with_storage_grid_size()
        self.all_cores = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1),
                )
            }
        )
        self.sub_device_id = ttnn.SubDeviceId(0)
        self.sub_device_manager = mesh_device.create_sub_device_manager([ttnn.SubDevice([self.all_cores])], 0)
        self.activate()
        self.semaphores = [ttnn.create_global_semaphore(mesh_device, self.all_cores, 0) for _ in range(2)]
        intermediate_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                residual_memory_config.shard_spec.grid,
                (rows, TENSOR_PARALLEL_SIZE * hidden_size // residual_num_cores),
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        self.persistent_buffer = ttnn.zeros(
            (1, 1, rows, TENSOR_PARALLEL_SIZE * hidden_size),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=intermediate_memory_config,
        )
        self.semaphore_index = 0
        self.closed = False

    def activate(self) -> None:
        self.mesh_device.load_sub_device_manager(self.sub_device_manager)
        self.mesh_device.set_sub_device_stall_group([self.sub_device_id])

    def validate(self, *, mesh_device, batch: int, hidden_size: int, residual_memory_config) -> None:
        if mesh_device is not self.mesh_device:
            raise ValueError("decode all-reduce resources belong to a different mesh device")
        if (batch, hidden_size) != (self.batch, self.hidden_size):
            raise ValueError("decode all-reduce resources do not match this decoder shape")
        if residual_memory_config != self.residual_memory_config:
            raise ValueError("decode all-reduce resources do not match the residual layout")
        if self.closed:
            raise RuntimeError("decode all-reduce resources are already closed")

    def all_reduce(self, partial, *, cluster_axis: int, topology, num_links: int, memory_config):
        semaphore = self.semaphores[self.semaphore_index]
        self.semaphore_index = (self.semaphore_index + 1) % len(self.semaphores)
        return ttnn.experimental.all_reduce_async(
            partial,
            self.persistent_buffer,
            cluster_axis=cluster_axis,
            mesh_device=self.mesh_device,
            multi_device_global_semaphore=semaphore,
            memory_config=memory_config,
            dtype=ttnn.bfloat16,
            topology=topology,
            num_links=num_links,
            subdevice_id=self.sub_device_id,
        )

    def close(self) -> None:
        if self.closed:
            return
        self.persistent_buffer.deallocate(True)
        self.mesh_device.reset_sub_device_stall_group()
        self.mesh_device.clear_loaded_sub_device_manager()
        self.closed = True


class MultichipDecoder(LightweightModule):
    """Falcon3-10B dense decoder specialized for the full 1x4 p300c mesh."""

    single_chip_baseline_cls = OptimizedDecoder

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Mapping[str, torch.Tensor],
        *,
        hf_config,
        layer_idx: int = IR_REPRESENTATIVE_LAYER,
        mesh_device,
        batch: int = EMITTED_BATCH,
        max_cache_len: int = 32768,
        precision_policy: str = "all_bfp4_lofi",
        decode_matmul_mode: str = "dram_sharded",
        use_packed_mlp: bool = False,
        packed_mlp_unpack_mode: str = "dram",
        decode_rope_mode: str = "dedicated",
        decode_output_mode: str = "direct_dram",
        use_persistent_decode_all_reduce: bool = True,
        decode_all_reduce_resources: DecodeAllReduceResources | None = None,
        qkv_target_cores: int = 4,
        o_target_cores: int = 2,
        gate_up_target_cores: int = 24,
        down_target_cores: int | None = None,
        prefill_grid_x: int = 11,
        prefill_in0_block_w: int = 8,
        ccl_dtype=ttnn.bfloat16,
        num_links: int = 2,
        topology=ttnn.Topology.Ring,
        page_block_size: int = DEFAULT_PAGE_BLOCK_SIZE,
        rope_cache: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
    ) -> "MultichipDecoder":
        mesh_shape = tuple(int(value) for value in mesh_device.shape)
        if mesh_shape != TARGET_MESH_SHAPE:
            raise ValueError(f"MultichipDecoder requires mesh {TARGET_MESH_SHAPE}, got {mesh_shape}")
        if mesh_device.get_num_devices() != TENSOR_PARALLEL_SIZE:
            raise ValueError(f"MultichipDecoder requires exactly {TENSOR_PARALLEL_SIZE} devices")
        if precision_policy not in PRECISION_POLICIES:
            raise ValueError(f"Unknown precision policy {precision_policy!r}")
        decode_matmul_modes = (
            "dram_sharded",
            "shard_advisor",
            "final_shard_advisor",
            "advisor_o",
            "advisor_o_48core",
        )
        if decode_matmul_mode not in decode_matmul_modes:
            raise ValueError(f"decode_matmul_mode must be one of {decode_matmul_modes}")
        if packed_mlp_unpack_mode not in ("dram", "l1_sharded"):
            raise ValueError("packed_mlp_unpack_mode must be 'dram' or 'l1_sharded'")
        if decode_rope_mode not in ("explicit", "dedicated"):
            raise ValueError("decode_rope_mode must be 'explicit' or 'dedicated'")
        if decode_output_mode not in ("l1_staged", "direct_dram"):
            raise ValueError("decode_output_mode must be 'l1_staged' or 'direct_dram'")
        if qkv_target_cores not in (2, 4, 8, 16):
            raise ValueError("qkv_target_cores must be 2, 4, 8, or 16")
        if o_target_cores not in (2, 4, 6, 8, 12, 16, 24, 48):
            raise ValueError("o_target_cores must be 2, 4, 6, 8, 12, 16, 24, or 48")
        if gate_up_target_cores not in (8, 12, 16, 24, 32, 48):
            raise ValueError("gate_up_target_cores must be 8, 12, 16, 24, 32, or 48")
        if down_target_cores is None:
            # BF16/HiFi4 needs the wider grid to keep the down-projection
            # circular buffers below Blackhole's 1.5 MiB/core L1 limit.  The
            # selected BFP4/LoFi production policy is faster on eight cores.
            down_target_cores = 24 if PRECISION_POLICIES[precision_policy]["mlp_down"] == ttnn.bfloat16 else 8
        if down_target_cores not in (4, 8, 12, 24):
            raise ValueError("down_target_cores must be 4, 8, 12, or 24")
        if prefill_grid_x not in (8, 11):
            raise ValueError("prefill_grid_x must be 8 or 11")
        if prefill_in0_block_w not in (4, 8, 12, 24):
            raise ValueError("prefill_in0_block_w must be 4, 8, 12, or 24")
        if ccl_dtype not in (ttnn.bfloat16, ttnn.bfloat8_b):
            raise ValueError("ccl_dtype must be bfloat16 or bfloat8_b")
        if num_links not in (1, 2):
            raise ValueError("num_links must be 1 or 2 on the target p300c mesh")
        if topology not in (ttnn.Topology.Ring, ttnn.Topology.Linear):
            raise ValueError("topology must be Ring or Linear")
        if page_block_size <= 0 or page_block_size % 32:
            raise ValueError("page_block_size must be a positive multiple of 32")
        if batch <= 0:
            raise ValueError("batch must be positive")

        hidden_size = int(_config_value(hf_config, "hidden_size"))
        num_layers = int(_config_value(hf_config, "num_hidden_layers"))
        num_heads = int(_config_value(hf_config, "num_attention_heads"))
        num_kv_heads = int(_config_value(hf_config, "num_key_value_heads"))
        head_dim = int(_config_value(hf_config, "head_dim"))
        intermediate_size = int(_config_value(hf_config, "intermediate_size"))
        rms_norm_eps = float(_config_value(hf_config, "rms_norm_eps"))
        max_position_embeddings = int(_config_value(hf_config, "max_position_embeddings"))
        expected = (3072, 40, 12, 4, 256, 23040)
        actual = (hidden_size, num_layers, num_heads, num_kv_heads, head_dim, intermediate_size)
        if actual != expected:
            raise ValueError(f"HF config does not match Falcon3-10B IR: got {actual}, expected {expected}")
        if str(_config_value(hf_config, "hidden_act")) != "silu":
            raise ValueError("Falcon3 multichip decoder requires SwiGLU/SILU")
        if not math.isclose(rms_norm_eps, 1e-6, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Expected rms_norm_eps=1e-6, got {rms_norm_eps}")
        if max_cache_len <= 0:
            raise ValueError("max_cache_len must be positive")
        if max_cache_len > max_position_embeddings:
            raise ValueError(f"max_cache_len={max_cache_len} exceeds HF context {max_position_embeddings}")
        if not 0 <= layer_idx < num_layers:
            raise ValueError(f"layer_idx={layer_idx} is outside [0,{num_layers})")
        if num_heads % TENSOR_PARALLEL_SIZE or num_kv_heads % TENSOR_PARALLEL_SIZE:
            raise ValueError("query and KV heads must divide the target mesh")
        if hidden_size % TENSOR_PARALLEL_SIZE or intermediate_size % TENSOR_PARALLEL_SIZE:
            raise ValueError("hidden and intermediate dimensions must divide the target mesh")
        if bool(getattr(hf_config, "attention_bias", False)) or bool(getattr(hf_config, "mlp_bias", False)):
            raise ValueError("The supplied Falcon3 graph has no attention or MLP biases")

        q = _resolve_layer_tensor(state_dict, layer_idx, "self_attn.q_proj.weight")
        k = _resolve_layer_tensor(state_dict, layer_idx, "self_attn.k_proj.weight")
        v = _resolve_layer_tensor(state_dict, layer_idx, "self_attn.v_proj.weight")
        o = _resolve_layer_tensor(state_dict, layer_idx, "self_attn.o_proj.weight")
        gate = _resolve_layer_tensor(state_dict, layer_idx, "mlp.gate_proj.weight")
        up = _resolve_layer_tensor(state_dict, layer_idx, "mlp.up_proj.weight")
        down = _resolve_layer_tensor(state_dict, layer_idx, "mlp.down_proj.weight")
        input_norm = _resolve_layer_tensor(state_dict, layer_idx, "input_layernorm.weight")
        post_norm = _resolve_layer_tensor(state_dict, layer_idx, "post_attention_layernorm.weight")

        local_heads = num_heads // TENSOR_PARALLEL_SIZE
        local_kv_heads = num_kv_heads // TENSOR_PARALLEL_SIZE
        local_qkv_size = (local_heads + 2 * local_kv_heads) * head_dim
        local_hidden = hidden_size // TENSOR_PARALLEL_SIZE
        policy = dict(PRECISION_POLICIES[precision_policy])
        prefill_memcfg = ttnn.DRAM_MEMORY_CONFIG
        qkv_host = _rank_grouped_qkv(q, k, v, TENSOR_PARALLEL_SIZE)
        gate_host = _rank_padded_column(gate, TENSOR_PARALLEL_SIZE, PADDED_LOCAL_INTERMEDIATE)
        up_host = _rank_padded_column(up, TENSOR_PARALLEL_SIZE, PADDED_LOCAL_INTERMEDIATE)
        down_host = _rank_padded_row(down, TENSOR_PARALLEL_SIZE, PADDED_LOCAL_INTERMEDIATE)

        self = object.__new__(cls)
        self.mesh_device = mesh_device
        self.layer_idx = layer_idx
        self.batch = batch
        self.max_cache_len = max_cache_len
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.local_num_heads = local_heads
        self.local_num_kv_heads = local_kv_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.local_hidden_size = local_hidden
        self.logical_local_intermediate_size = LOGICAL_LOCAL_INTERMEDIATE
        self.local_intermediate_size = PADDED_LOCAL_INTERMEDIATE
        self.local_qkv_size = local_qkv_size
        self.rms_norm_eps = rms_norm_eps
        self.scale = 1.0 / math.sqrt(head_dim)
        self.precision_policy_name = precision_policy
        self.precision_policy = policy
        self.decode_matmul_mode = decode_matmul_mode
        self.use_packed_mlp = use_packed_mlp
        self.packed_mlp_unpack_mode = packed_mlp_unpack_mode
        self.decode_rope_mode = decode_rope_mode
        self.decode_output_mode = decode_output_mode
        self.use_persistent_decode_all_reduce = use_persistent_decode_all_reduce
        self.decode_all_reduce_resources = decode_all_reduce_resources
        self.owns_decode_all_reduce_resources = False
        self.tensor_parallel_size = TENSOR_PARALLEL_SIZE
        self.tensor_parallel_axis = TENSOR_PARALLEL_AXIS
        self.topology = topology
        self.num_links = num_links
        self.ccl_dtype = ccl_dtype
        self.page_block_size = page_block_size
        self.qkv_target_cores = qkv_target_cores
        self.o_target_cores = o_target_cores
        self.gate_up_target_cores = gate_up_target_cores
        self.down_target_cores = down_target_cores
        self.prefill_grid_x = prefill_grid_x
        self.prefill_in0_block_w = prefill_in0_block_w

        advisor_all = decode_matmul_mode in ("shard_advisor", "final_shard_advisor")
        advisor_o = advisor_all or decode_matmul_mode in ("advisor_o", "advisor_o_48core")
        if advisor_all and use_packed_mlp:
            raise ValueError("the exact shard-advisor graph describes separate gate/up projections")
        self.qkv_weight = _mesh_weight(
            qkv_host, dtype=policy["attention"], mesh_device=mesh_device, shard_dim=-1, memory_config=prefill_memcfg
        )
        self.qkv_decode_weight = (
            _mesh_weight(
                qkv_host,
                dtype=policy["attention"],
                mesh_device=mesh_device,
                shard_dim=-1,
                local_shape=(hidden_size, local_qkv_size),
            )
            if not advisor_all
            else None
        )
        self.o_weight = _mesh_weight(
            o.transpose(-2, -1),
            dtype=policy["attention"],
            mesh_device=mesh_device,
            shard_dim=-2,
            memory_config=prefill_memcfg,
        )
        self.o_decode_weight = (
            _mesh_weight(
                o.transpose(-2, -1),
                dtype=policy["attention"],
                mesh_device=mesh_device,
                shard_dim=-2,
                local_shape=(local_hidden, hidden_size),
            )
            if not advisor_o
            else None
        )
        if use_packed_mlp:
            gate_up_host = _rank_packed_pair(gate_host, up_host, TENSOR_PARALLEL_SIZE)
            self.gate_up_weight = _mesh_weight(
                gate_up_host,
                dtype=policy["mlp_gate_up"],
                mesh_device=mesh_device,
                shard_dim=-1,
                memory_config=prefill_memcfg,
            )
            self.gate_up_decode_weight = _mesh_weight(
                gate_up_host,
                dtype=policy["mlp_gate_up"],
                mesh_device=mesh_device,
                shard_dim=-1,
                local_shape=(hidden_size, 2 * PADDED_LOCAL_INTERMEDIATE),
            )
            self.gate_weight = None
            self.up_weight = None
            self.gate_decode_weight = None
            self.up_decode_weight = None
        else:
            self.gate_weight = _mesh_weight(
                gate_host,
                dtype=policy["mlp_gate_up"],
                mesh_device=mesh_device,
                shard_dim=-1,
                memory_config=prefill_memcfg,
            )
            self.up_weight = _mesh_weight(
                up_host,
                dtype=policy["mlp_gate_up"],
                mesh_device=mesh_device,
                shard_dim=-1,
                memory_config=prefill_memcfg,
            )
            self.gate_decode_weight = (
                _mesh_weight(
                    gate_host,
                    dtype=policy["mlp_gate_up"],
                    mesh_device=mesh_device,
                    shard_dim=-1,
                    local_shape=(hidden_size, PADDED_LOCAL_INTERMEDIATE),
                )
                if not advisor_all
                else None
            )
            self.up_decode_weight = (
                _mesh_weight(
                    up_host,
                    dtype=policy["mlp_gate_up"],
                    mesh_device=mesh_device,
                    shard_dim=-1,
                    local_shape=(hidden_size, PADDED_LOCAL_INTERMEDIATE),
                )
                if not advisor_all
                else None
            )
            self.gate_up_weight = None
            self.gate_up_decode_weight = None
        self.down_weight = _mesh_weight(
            down_host, dtype=policy["mlp_down"], mesh_device=mesh_device, shard_dim=-2, memory_config=prefill_memcfg
        )
        self.down_decode_weight = (
            _mesh_weight(
                down_host,
                dtype=policy["mlp_down"],
                mesh_device=mesh_device,
                shard_dim=-2,
                local_shape=(PADDED_LOCAL_INTERMEDIATE, hidden_size),
            )
            if not advisor_all
            else None
        )
        self.input_norm_weight = _mesh_weight(
            input_norm, dtype=ttnn.bfloat16, mesh_device=mesh_device, shard_dim=None, memory_config=prefill_memcfg
        )
        self.post_attention_norm_weight = _mesh_weight(
            post_norm, dtype=ttnn.bfloat16, mesh_device=mesh_device, shard_dim=None, memory_config=prefill_memcfg
        )

        rope_parameters = getattr(hf_config, "rope_parameters", None) or getattr(hf_config, "rope_scaling", None) or {}
        rope_theta = getattr(hf_config, "rope_theta", None) or rope_parameters.get("rope_theta")
        if float(rope_theta or 0) != 1000042.0 or rope_parameters.get("rope_type", "default") != "default":
            raise ValueError("Falcon3 multichip decoder requires the emitted default RoPE contract")
        if rope_cache is None:
            positions = torch.arange(max_cache_len, dtype=torch.float32)
            inv_freq = 1.0 / (float(rope_theta) ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
            angles = torch.cat([torch.outer(positions, inv_freq)] * 2, dim=-1)
            self.cos_cache = _replicated_tensor(
                angles.cos(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.sin_cache = _replicated_tensor(
                angles.sin(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.owns_rope_cache = True
        else:
            if len(rope_cache) != 2:
                raise ValueError("rope_cache must contain exactly (cos_cache, sin_cache)")
            self.cos_cache, self.sin_cache = rope_cache
            self.owns_rope_cache = False

        grid_size = mesh_device.compute_with_storage_grid_size()
        grid_x = min(batch, int(grid_size.x))
        while grid_x > 0 and (batch % grid_x or batch // grid_x > int(grid_size.y)):
            grid_x -= 1
        if grid_x == 0:
            raise ValueError(f"batch={batch} cannot form a decode head grid within {grid_size}")
        physical_rows = 32 * math.ceil(batch / 32)
        self.decode_head_memory_config = ttnn.create_sharded_memory_config(
            # ``shape`` is the full tensor shape for this helper; it derives
            # the per-core height from ``core_grid``.  Keep the physical
            # tensor tile-aligned while retaining ``batch`` as the logical
            # user count.
            shape=(physical_rows, head_dim),
            core_grid=ttnn.CoreGrid(x=grid_x, y=batch // grid_x),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        self.attention_compute_config = _compute_config(mesh_device, policy["attention_fidelity"])
        self.mlp_compute_config = _compute_config(mesh_device, policy["mlp_fidelity"])
        self.norm_compute_config = _compute_config(mesh_device, ttnn.MathFidelity.HiFi2)
        self.kv_cache_dtype = policy["kv_cache"]
        self._build_configs()
        if self.decode_all_reduce_resources is not None:
            self.decode_all_reduce_resources.validate(
                mesh_device=self.mesh_device,
                batch=self.batch,
                hidden_size=self.hidden_size,
                residual_memory_config=self.residual_memory_config,
            )
        return self

    def _build_configs(self) -> None:
        grid_size = self.mesh_device.compute_with_storage_grid_size()
        max_x, max_y = min(8, int(grid_size.x)), min(10, int(grid_size.y))
        padded_rows = 32 * math.ceil(self.batch / 32)
        hidden_tiles = self.hidden_size // 32
        local_hidden_tiles = self.local_hidden_size // 32
        qkv_tiles = self.local_qkv_size // 32
        local_mlp_tiles = self.local_intermediate_size // 32

        self.residual_grid = _core_grid_for_tiles(hidden_tiles, hidden_tiles, target_cores=32, max_x=max_x, max_y=max_y)
        self.residual_num_cores = self.residual_grid.num_cores
        self.qkv_grid = _core_grid_for_tiles(
            hidden_tiles, qkv_tiles, target_cores=self.qkv_target_cores, max_x=max_x, max_y=max_y
        )
        self.o_grid = _core_grid_for_tiles(
            local_hidden_tiles, hidden_tiles, target_cores=self.o_target_cores, max_x=max_x, max_y=max_y
        )
        self.gate_up_grid = _core_grid_for_tiles(
            hidden_tiles, local_mlp_tiles, target_cores=self.gate_up_target_cores, max_x=max_x, max_y=max_y
        )
        self.packed_gate_up_grid = _core_grid_for_tiles(
            hidden_tiles, 2 * local_mlp_tiles, target_cores=48, max_x=max_x, max_y=max_y
        )
        self.down_grid = _core_grid_for_tiles(
            local_mlp_tiles, hidden_tiles, target_cores=self.down_target_cores, max_x=max_x, max_y=max_y
        )

        self.residual_memory_config = _sharded_memory_config(padded_rows, self.hidden_size, self.residual_grid)
        self.norm_memory_config = self.residual_memory_config
        self.qkv_input_memory_config = _sharded_memory_config(padded_rows, self.hidden_size, self.qkv_grid)
        self.qkv_output_memory_config = _matmul_output_memory_config(
            padded_rows, self.local_qkv_size, self.qkv_grid, self.mesh_device
        )
        self.o_input_memory_config = _sharded_memory_config(padded_rows, self.local_hidden_size, self.o_grid)
        self.o_output_memory_config = _matmul_output_memory_config(
            padded_rows, self.hidden_size, self.o_grid, self.mesh_device
        )
        self.mlp_input_memory_config = _sharded_memory_config(padded_rows, self.hidden_size, self.gate_up_grid)
        if self.use_packed_mlp:
            self.mlp_input_memory_config = _sharded_memory_config(
                padded_rows, self.hidden_size, self.packed_gate_up_grid
            )
        self.mlp_output_memory_config = _matmul_output_memory_config(
            padded_rows, self.local_intermediate_size, self.gate_up_grid, self.mesh_device
        )
        self.packed_mlp_output_memory_config = _matmul_output_memory_config(
            padded_rows, 2 * self.local_intermediate_size, self.packed_gate_up_grid, self.mesh_device
        )
        self.down_input_memory_config = _sharded_memory_config(
            padded_rows, self.local_intermediate_size, self.down_grid
        )
        self.down_output_memory_config = _matmul_output_memory_config(
            padded_rows, self.hidden_size, self.down_grid, self.mesh_device
        )

        norm_block_w = hidden_tiles // self.residual_grid.num_cores
        self.norm_program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[self.residual_grid.x, self.residual_grid.y],
            subblock_w=_largest_divisor(norm_block_w, limit=4),
            block_h=padded_rows // 32,
            block_w=norm_block_w,
            inplace=False,
        )
        self.qkv_decode_program_config = _dram_matmul_program_config(
            padded_rows, self.hidden_size, self.local_qkv_size, self.qkv_grid
        )
        self.o_decode_program_config = _dram_matmul_program_config(
            padded_rows, self.local_hidden_size, self.hidden_size, self.o_grid
        )
        self.gate_decode_program_config = _dram_matmul_program_config(
            padded_rows, self.hidden_size, self.local_intermediate_size, self.gate_up_grid
        )
        self.packed_gate_decode_program_config = _dram_matmul_program_config(
            padded_rows,
            self.hidden_size,
            2 * self.local_intermediate_size,
            self.packed_gate_up_grid,
            in0_block_w=2,
        )
        self.down_decode_program_config = _dram_matmul_program_config(
            padded_rows, self.local_intermediate_size, self.hidden_size, self.down_grid
        )
        if self.decode_matmul_mode == "shard_advisor":
            # Exact TP-local choices from optimized_multichip_decoder/
            # shard_advise/final_ir.mlir. Keep the legacy residual contract
            # around this candidate so the first A/B isolates the advisor's
            # local matmul/layout family from inter-layer distribution changes.
            self.qkv_decode_program_config = _advisor_matmul_program_config(
                grid=(11, 4), in0_block_w=8, per_core_n=1, out_subblock_w=1
            )
            self.o_decode_program_config = _advisor_matmul_program_config(
                grid=(11, 9), in0_block_w=8, per_core_n=1, out_subblock_w=1
            )
            self.gate_decode_program_config = _advisor_matmul_program_config(
                grid=(11, 9), in0_block_w=2, per_core_n=2, out_subblock_w=2
            )
            self.down_decode_program_config = _advisor_matmul_program_config(
                grid=(11, 9), in0_block_w=8, per_core_n=1, out_subblock_w=1
            )
            self.qkv_input_memory_config = ttnn.L1_MEMORY_CONFIG
            self.qkv_output_memory_config = _width_sharded_output_memory_config(
                padded_rows, self.local_qkv_size, 40, self.mesh_device
            )
            self.o_input_memory_config = ttnn.DRAM_MEMORY_CONFIG
            self.o_output_memory_config = _width_sharded_output_memory_config(
                padded_rows, self.hidden_size, 96, self.mesh_device
            )
            self.mlp_input_memory_config = _width_sharded_output_memory_config(
                padded_rows, self.hidden_size, 48, self.mesh_device
            )
            self.mlp_output_memory_config = _width_sharded_output_memory_config(
                padded_rows, self.local_intermediate_size, 96, self.mesh_device
            )
            self.down_input_memory_config = ttnn.L1_MEMORY_CONFIG
            self.down_output_memory_config = _width_sharded_output_memory_config(
                padded_rows, self.hidden_size, 96, self.mesh_device
            )
        elif self.decode_matmul_mode == "final_shard_advisor":
            # Apply every feasible choice from
            # shard_advise/final_graph_corrected.  The
            # advisor is single-device, so the two row-parallel reductions are
            # restored with a persistent TP=4 all-reduce in the selected
            # 96-core residual layout.  The concat op is the only unfixable
            # choice in report.json; keep its required production-legal
            # height-sharded input while applying the corrected RoPE layouts.
            self.residual_num_cores = 96
            self.residual_memory_config = _width_sharded_output_memory_config(
                padded_rows, self.hidden_size, self.residual_num_cores, self.mesh_device
            )
            self.norm_memory_config = _advisor_norm_memory_config(padded_rows, self.hidden_size)
            self.norm_program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=[11, 1],
                subblock_w=3,
                block_h=padded_rows // 32,
                block_w=9,
                inplace=False,
            )
            self.qkv_decode_program_config = _advisor_matmul_program_config(
                grid=(11, 4), in0_block_w=8, per_core_n=1, out_subblock_w=1
            )
            self.o_decode_program_config = _advisor_matmul_program_config(
                grid=(11, 9), in0_block_w=8, per_core_n=1, out_subblock_w=1
            )
            self.gate_decode_program_config = _advisor_matmul_program_config(
                grid=(11, 9), in0_block_w=2, per_core_n=2, out_subblock_w=2
            )
            self.down_decode_program_config = _advisor_matmul_program_config(
                grid=(11, 9), in0_block_w=8, per_core_n=1, out_subblock_w=1
            )
            self.qkv_input_memory_config = ttnn.L1_MEMORY_CONFIG
            self.qkv_output_memory_config = _width_sharded_output_memory_config(
                padded_rows, self.local_qkv_size, 40, self.mesh_device
            )
            self.o_input_memory_config = ttnn.DRAM_MEMORY_CONFIG
            self.o_output_memory_config = self.residual_memory_config
            self.mlp_input_memory_config = _width_sharded_output_memory_config(
                padded_rows, self.hidden_size, 48, self.mesh_device
            )
            self.mlp_output_memory_config = _width_sharded_output_memory_config(
                padded_rows, self.local_intermediate_size, 96, self.mesh_device
            )
            self.down_input_memory_config = ttnn.L1_MEMORY_CONFIG
            self.down_output_memory_config = self.residual_memory_config
            self.advisor_query_transpose_memory_config = ttnn.create_sharded_memory_config(
                shape=(32, 32),
                core_grid=ttnn.CoreGrid(x=8, y=3),
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self.advisor_query_rope_memory_config = ttnn.create_sharded_memory_config(
                shape=(32, self.head_dim),
                core_grid=ttnn.CoreGrid(x=3, y=1),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self.advisor_query_output_memory_config = ttnn.create_sharded_memory_config(
                shape=(128, 32),
                core_grid=ttnn.CoreGrid(x=8, y=8),
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self.advisor_key_transpose_memory_config = ttnn.create_sharded_memory_config(
                shape=(32, 32),
                core_grid=ttnn.CoreGrid(x=8, y=1),
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self.advisor_key_rope_memory_config = ttnn.create_sharded_memory_config(
                shape=(32, self.head_dim),
                core_grid=ttnn.CoreGrid(x=1, y=1),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self.advisor_key_output_memory_config = ttnn.create_sharded_memory_config(
                shape=(32 * self.batch, self.head_dim),
                core_grid=ttnn.CoreGrid(x=1, y=1),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
        elif self.decode_matmul_mode in ("advisor_o", "advisor_o_48core"):
            o_cores = 96 if self.decode_matmul_mode == "advisor_o" else 48
            o_grid = (11, 9) if o_cores == 96 else (8, 6)
            o_per_core_n = 1 if o_cores == 96 else 2
            self.o_decode_program_config = _advisor_matmul_program_config(
                grid=o_grid,
                in0_block_w=8,
                per_core_n=o_per_core_n,
                out_subblock_w=o_per_core_n,
            )
            self.o_input_memory_config = ttnn.DRAM_MEMORY_CONFIG
            self.o_output_memory_config = (
                _width_sharded_output_memory_config(padded_rows, self.hidden_size, o_cores, self.mesh_device)
                if o_cores == 96
                else _sharded_memory_config(padded_rows, self.hidden_size, ttnn.CoreGrid(x=8, y=6))
            )
        self.decode_sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8), exp_approx_mode=False, q_chunk_size=0, k_chunk_size=0
        )

    def allocate_kv_cache(
        self, max_cache_len: int | None = None, *, paged: bool = False, num_blocks: int | None = None
    ):
        cache_len = self.max_cache_len if max_cache_len is None else int(max_cache_len)
        if cache_len <= 0:
            raise ValueError("max_cache_len must be positive")
        if paged:
            pages_per_user = math.ceil(cache_len / self.page_block_size)
            required_blocks = self.batch * pages_per_user
            physical_blocks = required_blocks if num_blocks is None else int(num_blocks)
            if physical_blocks < required_blocks:
                raise ValueError(f"paged cache needs at least {required_blocks} blocks, got {physical_blocks}")
            shape = (physical_blocks, self.local_num_kv_heads, self.page_block_size, self.head_dim)
        else:
            shape = (self.batch, self.local_num_kv_heads, cache_len, self.head_dim)
        return tuple(
            ttnn.zeros(
                shape,
                dtype=self.kv_cache_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for _ in range(2)
        )

    def _validate_hidden(self, hidden_states, *, decode: bool):
        shape = tuple(int(value) for value in hidden_states.shape)
        if len(shape) != 4 or shape[0] != 1 or shape[3] != self.hidden_size:
            raise ValueError(f"hidden_states must be [1,batch,seq,{self.hidden_size}], got {shape}")
        batch, seq_len = shape[1], shape[2]
        if batch != self.batch:
            raise ValueError(f"runtime batch {batch} does not match configured batch {self.batch}")
        if decode and seq_len != 1:
            raise ValueError("decode requires one logical token")
        if not decode and not 0 < seq_len <= self.max_cache_len:
            raise ValueError(f"prefill seq_len must be in [1,{self.max_cache_len}], got {seq_len}")
        return batch, seq_len

    def _validate_cache_position(self, cache_position):
        if tuple(int(value) for value in cache_position.shape) != (self.batch,) or cache_position.dtype != ttnn.int32:
            raise ValueError(f"cache_position must be int32 with shape [{self.batch}]")

    def _validate_caches(self, key_cache, value_cache, *, page_table=None):
        k_shape = tuple(int(value) for value in key_cache.shape)
        v_shape = tuple(int(value) for value in value_cache.shape)
        if k_shape != v_shape or len(k_shape) != 4:
            raise ValueError(f"K/V cache shapes must match, got {k_shape}/{v_shape}")
        if k_shape[1] != self.local_num_kv_heads or k_shape[3] != self.head_dim:
            raise ValueError(
                f"each rank cache must own {self.local_num_kv_heads} KV heads of width {self.head_dim}, got {k_shape}"
            )
        if page_table is None:
            if k_shape[0] != self.batch:
                raise ValueError(f"contiguous cache first dimension must be batch={self.batch}")
            return
        pt_shape = tuple(int(value) for value in page_table.shape)
        if len(pt_shape) != 2 or pt_shape[0] != self.batch:
            raise ValueError(f"page_table must be [{self.batch},pages], got {pt_shape}")
        if k_shape[2] != self.page_block_size:
            raise ValueError(f"paged cache block dimension must be {self.page_block_size}")
        if pt_shape[1] * self.page_block_size < self.max_cache_len:
            raise ValueError("page table does not cover configured cache length")

    def _rotary_slice(self, start: int, length: int):
        end = start + length
        if start < 0 or end > self.max_cache_len:
            raise ValueError(f"rotary range [{start},{end}) exceeds cache length {self.max_cache_len}")
        cos_2d = ttnn.slice(self.cos_cache, [start, 0], [end, self.head_dim])
        sin_2d = ttnn.slice(self.sin_cache, [start, 0], [end, self.head_dim])
        cos = ttnn.reshape(cos_2d, (1, 1, length, self.head_dim))
        sin = ttnn.reshape(sin_2d, (1, 1, length, self.head_dim))
        ttnn.deallocate(cos_2d, False)
        ttnn.deallocate(sin_2d, False)
        return cos, sin

    def _decode_rotary_positions(self, rotary_position):
        """Gather one RoPE row per user from an independent device tensor."""
        converted = rotary_position.dtype != ttnn.uint32
        position_u32 = ttnn.typecast(rotary_position, ttnn.uint32) if converted else rotary_position
        position_u32_2d = ttnn.reshape(position_u32, (1, self.batch))
        cos_3d = ttnn.embedding(position_u32_2d, self.cos_cache, layout=ttnn.TILE_LAYOUT)
        sin_3d = ttnn.embedding(position_u32_2d, self.sin_cache, layout=ttnn.TILE_LAYOUT)
        ttnn.deallocate(position_u32_2d, False)
        if converted:
            ttnn.deallocate(position_u32, True)
        cos = ttnn.unsqueeze_to_4D(cos_3d)
        sin = ttnn.unsqueeze_to_4D(sin_3d)
        ttnn.deallocate(cos_3d, False)
        ttnn.deallocate(sin_3d, False)
        return cos, sin

    def _rotate_half_decode(self, tensor):
        shape = tuple(int(value) for value in tensor.shape)
        first = ttnn.slice(tensor, [0, 0, 0, 0], [shape[0], shape[1], shape[2], self.head_dim // 2])
        second = ttnn.slice(tensor, [0, 0, 0, self.head_dim // 2], [shape[0], shape[1], shape[2], self.head_dim])
        negative_second = ttnn.neg(second)
        rotated = ttnn.concat([negative_second, first], dim=-1)
        for item in (first, second, negative_second):
            ttnn.deallocate(item, True)
        return rotated

    def _apply_decode_rope_per_user(self, tensor, cos, sin):
        heads = int(tensor.shape[2])
        cos_by_user = ttnn.transpose(cos, 1, 2)
        sin_by_user = ttnn.transpose(sin, 1, 2)
        if int(cos_by_user.shape[1]) != self.batch:
            cos_padded, sin_padded = cos_by_user, sin_by_user
            cos_by_user = ttnn.slice(cos_padded, [0, 0, 0, 0], [1, self.batch, 1, self.head_dim])
            sin_by_user = ttnn.slice(sin_padded, [0, 0, 0, 0], [1, self.batch, 1, self.head_dim])
            ttnn.deallocate(cos_padded, True)
            ttnn.deallocate(sin_padded, True)
        cos_heads = ttnn.repeat(cos_by_user, ttnn.Shape([1, 1, heads, 1]))
        sin_heads = ttnn.repeat(sin_by_user, ttnn.Shape([1, 1, heads, 1]))
        rotated = self._rotate_half_decode(tensor)
        cosine_term = ttnn.mul(tensor, cos_heads)
        sine_term = ttnn.mul(rotated, sin_heads)
        output = ttnn.add(cosine_term, sine_term)
        for item in (cos_by_user, sin_by_user, cos_heads, sin_heads, rotated, cosine_term, sine_term):
            ttnn.deallocate(item, True)
        return output

    def _apply_decode_rope_dedicated(self, tensor, cos, sin):
        """Use the HF rotate-half op with batch represented as sequence rows."""
        if self.decode_matmul_mode == "final_shard_advisor" and self.batch == 32:
            query = int(tensor.shape[2]) == self.local_num_heads
            transpose_memory_config = (
                self.advisor_query_transpose_memory_config if query else self.advisor_key_transpose_memory_config
            )
            rope_memory_config = self.advisor_query_rope_memory_config if query else self.advisor_key_rope_memory_config
            output_memory_config = (
                self.advisor_query_output_memory_config if query else self.advisor_key_output_memory_config
            )
            transposed = ttnn.transpose(tensor, 1, 2, memory_config=transpose_memory_config)
            rope_input = self._move_owned(transposed, rope_memory_config)
            cos_l1 = ttnn.to_memory_config(cos, self.advisor_key_rope_memory_config)
            sin_l1 = ttnn.to_memory_config(sin, self.advisor_key_rope_memory_config)
            rotated = ttnn.experimental.rotary_embedding(
                rope_input,
                cos_l1,
                sin_l1,
                None,
                memory_config=rope_memory_config,
                compute_kernel_config=self.attention_compute_config,
            )
            ttnn.deallocate(rope_input, True)
            ttnn.deallocate(cos_l1, True)
            ttnn.deallocate(sin_l1, True)
            output = ttnn.transpose(rotated, 1, 2, memory_config=output_memory_config)
            ttnn.deallocate(rotated, True)
            return output
        transposed = ttnn.transpose(tensor, 1, 2)
        rotated = ttnn.experimental.rotary_embedding(
            transposed,
            cos,
            sin,
            None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.attention_compute_config,
        )
        ttnn.deallocate(transposed, True)
        output = ttnn.transpose(rotated, 1, 2)
        ttnn.deallocate(rotated, True)
        return output

    def _move_owned(self, tensor, memory_config):
        if tensor.memory_config() == memory_config:
            return tensor
        moved = ttnn.to_memory_config(tensor, memory_config)
        ttnn.deallocate(tensor, True)
        return moved

    def _all_reduce_partial(self, partial, *, memory_config):
        ccl_input = partial
        if ccl_input.dtype != self.ccl_dtype:
            ccl_input = ttnn.typecast(ccl_input, self.ccl_dtype)
            ttnn.deallocate(partial, True)
        if (
            self.decode_all_reduce_resources is not None
            and memory_config == self.residual_memory_config
            and ccl_input.dtype == ttnn.bfloat16
        ):
            reduced = self.decode_all_reduce_resources.all_reduce(
                ccl_input,
                cluster_axis=self.tensor_parallel_axis,
                num_links=self.num_links,
                topology=self.topology,
                memory_config=memory_config,
            )
        else:
            reduced = ttnn.all_reduce(
                ccl_input,
                cluster_axis=self.tensor_parallel_axis,
                num_links=self.num_links,
                topology=self.topology,
                memory_config=memory_config,
            )
        ttnn.deallocate(ccl_input, True)
        if reduced.dtype != ttnn.bfloat16:
            bf16 = ttnn.typecast(reduced, ttnn.bfloat16)
            ttnn.deallocate(reduced, True)
            return bf16
        return reduced

    def _unpad_prefill_sequence(self, tensor, *, batch, heads, seq_len):
        if int(tensor.shape[2]) == seq_len:
            return tensor
        padded = tensor
        tensor = ttnn.slice(padded, [0, 0, 0, 0], [batch, heads, seq_len, self.head_dim])
        ttnn.deallocate(padded, True)
        return tensor

    def _decode_norm(self, residual, weight):
        norm_input = residual
        converted = False
        if norm_input.memory_config() != self.norm_memory_config:
            norm_input = ttnn.to_memory_config(norm_input, self.norm_memory_config)
            converted = True
        output = ttnn.rms_norm(
            norm_input,
            epsilon=self.rms_norm_eps,
            weight=weight,
            program_config=self.norm_program_config,
            compute_kernel_config=self.norm_compute_config,
            memory_config=self.norm_memory_config,
        )
        if converted:
            ttnn.deallocate(norm_input, True)
        return output

    def _prepare_decode_heads(self, tensor, num_heads, *, memory_config):
        interleaved = ttnn.to_memory_config(tensor, ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(tensor, True)
        expected_shape = (1, self.batch, num_heads, self.head_dim)
        if tuple(int(value) for value in interleaved.shape) == expected_shape:
            return self._move_owned(interleaved, memory_config)
        unpadded = ttnn.slice(interleaved, [0, 0, 0, 0], list(expected_shape))
        ttnn.deallocate(interleaved, True)
        prepared = ttnn.to_memory_config(unpadded, memory_config)
        ttnn.deallocate(unpadded, True)
        return prepared

    def _fill_prefill_cache(
        self,
        cache,
        fill,
        *,
        user_id,
        page_table,
        valid_len: int | None = None,
        chunk_start_idx: int = 0,
        chunk_page_table=None,
    ):
        if valid_len is not None:
            if valid_len <= 0:
                return
            if valid_len < int(fill.shape[2]):
                fill = ttnn.slice(
                    fill,
                    [0, 0, 0, 0],
                    [1, self.local_num_kv_heads, valid_len, self.head_dim],
                )
        casted = fill.dtype != cache.dtype
        if casted:
            typed = ttnn.typecast(fill, cache.dtype)
            ttnn.deallocate(fill, False)
            fill = typed
        if page_table is None:
            if chunk_start_idx:
                raise ValueError("chunked prefill requires a paged KV cache")
            ttnn.fill_cache(cache, fill, batch_idx=user_id)
        else:
            fill_page_table = chunk_page_table if chunk_page_table is not None else page_table
            ttnn.experimental.paged_fill_cache(cache, fill, fill_page_table, batch_idx=user_id)
        ttnn.deallocate(fill, casted and cache.dtype != ttnn.bfloat16)

    def _prefill_linear_chunked(self, tensor, weight, *, k, n, compute_kernel_config):
        rows = int(tensor.shape[-2])

        def run(chunk):
            chunk_rows = int(chunk.shape[-2])
            return ttnn.matmul(
                chunk,
                weight,
                dtype=ttnn.bfloat16,
                program_config=_prefill_matmul_program_config(
                    self.mesh_device,
                    chunk_rows,
                    k,
                    n,
                    grid_x_limit=self.prefill_grid_x,
                    in0_block_w=self.prefill_in0_block_w,
                ),
                compute_kernel_config=compute_kernel_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        if rows <= 1024:
            return run(tensor)
        chunks = []
        for start in range(0, rows, 1024):
            end = min(start + 1024, rows)
            chunk = ttnn.slice(tensor, [0, 0, start, 0], [1, 1, end, k], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            chunks.append(run(chunk))
            ttnn.deallocate(chunk, True)
        output = ttnn.concat(chunks, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for chunk in chunks:
            ttnn.deallocate(chunk, True)
        return output

    def _prefill_attention(
        self,
        residual,
        *,
        batch,
        seq_len,
        key_cache,
        value_cache,
        page_table,
        prompt_lens=None,
        chunk_start_idx: int = 0,
        chunk_page_table=None,
    ):
        m = batch * seq_len
        normed = ttnn.rms_norm(
            residual,
            epsilon=self.rms_norm_eps,
            weight=self.input_norm_weight,
            compute_kernel_config=self.norm_compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        fused_qkv = self._prefill_linear_chunked(
            normed,
            self.qkv_weight,
            k=self.hidden_size,
            n=self.local_qkv_size,
            compute_kernel_config=self.attention_compute_config,
        )
        ttnn.deallocate(normed, True)
        fused_qkv = ttnn.reshape(fused_qkv, (batch, seq_len, self.local_qkv_size))
        query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
            fused_qkv,
            None,
            num_heads=self.local_num_heads,
            num_kv_heads=self.local_num_kv_heads,
            transpose_key=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(fused_qkv, True)
        cos, sin = self._rotary_slice(chunk_start_idx, seq_len)
        key_rotated = ttnn.experimental.rotary_embedding(key, cos, sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        query_rotated = ttnn.experimental.rotary_embedding(query, cos, sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for item in (key, query):
            ttnn.deallocate(item, True)
        ttnn.deallocate(cos, False)
        ttnn.deallocate(sin, False)
        key_rotated = self._unpad_prefill_sequence(
            key_rotated, batch=batch, heads=self.local_num_kv_heads, seq_len=seq_len
        )
        query_rotated = self._unpad_prefill_sequence(
            query_rotated, batch=batch, heads=self.local_num_heads, seq_len=seq_len
        )
        for user_id in range(batch):
            valid_len = seq_len
            if prompt_lens is not None:
                valid_len = min(seq_len, max(0, int(prompt_lens[user_id]) - chunk_start_idx))
                if valid_len == 0:
                    continue
            value_user = ttnn.slice(
                value, [user_id, 0, 0, 0], [user_id + 1, self.local_num_kv_heads, seq_len, self.head_dim]
            )
            key_user = ttnn.slice(
                key_rotated, [user_id, 0, 0, 0], [user_id + 1, self.local_num_kv_heads, seq_len, self.head_dim]
            )
            fill_kwargs = dict(
                user_id=user_id,
                page_table=page_table,
                valid_len=valid_len,
                chunk_start_idx=chunk_start_idx,
                chunk_page_table=chunk_page_table,
            )
            self._fill_prefill_cache(value_cache, value_user, **fill_kwargs)
            self._fill_prefill_cache(key_cache, key_user, **fill_kwargs)
        sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8), exp_approx_mode=False, q_chunk_size=64, k_chunk_size=64
        )
        if chunk_start_idx:
            attention = ttnn.transformer.chunked_scaled_dot_product_attention(
                input_tensor_q=query_rotated,
                input_tensor_k=key_cache,
                input_tensor_v=value_cache,
                page_table_tensor=page_table,
                chunk_start_idx=chunk_start_idx,
                compute_kernel_config=self.attention_compute_config,
                program_config=sdpa_program_config,
            )
        else:
            attention = ttnn.transformer.scaled_dot_product_attention(
                query_rotated,
                key_rotated,
                value,
                is_causal=True,
                scale=self.scale,
                compute_kernel_config=self.attention_compute_config,
                program_config=sdpa_program_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        for item in (query_rotated, key_rotated, value):
            ttnn.deallocate(item, True)
        attention_heads = attention
        attention = ttnn.transformer.concatenate_heads(attention_heads, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attention_heads, True)
        concatenated = attention
        attention = ttnn.reshape(concatenated, (1, 1, m, self.local_hidden_size))
        ttnn.deallocate(concatenated, False)
        projected = self._prefill_linear_chunked(
            attention,
            self.o_weight,
            k=self.local_hidden_size,
            n=self.hidden_size,
            compute_kernel_config=self.attention_compute_config,
        )
        ttnn.deallocate(attention, True)
        projected = self._all_reduce_partial(projected, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        output = ttnn.add(residual, projected, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(projected, True)
        return output

    def _decode_qkv(self, residual, *, rotary_position):
        normed = self._decode_norm(residual, self.input_norm_weight)
        qkv_input = self._move_owned(normed, self.qkv_input_memory_config)
        fused_qkv = ttnn.matmul(
            qkv_input,
            self.qkv_decode_weight if self.qkv_decode_weight is not None else self.qkv_weight,
            dtype=ttnn.bfloat16,
            program_config=self.qkv_decode_program_config,
            compute_kernel_config=self.attention_compute_config,
            memory_config=self.qkv_output_memory_config,
        )
        ttnn.deallocate(qkv_input, True)
        fused_qkv = ttnn.sharded_to_interleaved(fused_qkv, ttnn.L1_MEMORY_CONFIG, ttnn.bfloat16)
        query, key, value = ttnn.experimental.nlp_create_qkv_heads_decode(
            fused_qkv,
            num_heads=self.local_num_heads,
            num_kv_heads=self.local_num_kv_heads,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )
        ttnn.deallocate(fused_qkv, True)
        if int(value.shape[1]) != self.batch:
            value = self._prepare_decode_heads(
                value, self.local_num_kv_heads, memory_config=self.decode_head_memory_config
            )
        cos, sin = self._decode_rotary_positions(rotary_position)
        if self.batch == 1:
            cos_l1 = ttnn.to_memory_config(cos, ttnn.L1_MEMORY_CONFIG)
            sin_l1 = ttnn.to_memory_config(sin, ttnn.L1_MEMORY_CONFIG)
            key_rotated = ttnn.experimental.rotary_embedding(
                key, cos_l1, sin_l1, 0, memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
            )
            query_rotated = ttnn.experimental.rotary_embedding(
                query, cos_l1, sin_l1, 0, memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
            )
            ttnn.deallocate(key, True)
            ttnn.deallocate(query, True)
            ttnn.deallocate(cos_l1, True)
            ttnn.deallocate(sin_l1, True)
            key_rotated = self._prepare_decode_heads(
                key_rotated, self.local_num_kv_heads, memory_config=self.decode_head_memory_config
            )
            query_rotated = self._prepare_decode_heads(
                query_rotated, self.local_num_heads, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        else:
            advisor_rope = (
                self.decode_matmul_mode == "final_shard_advisor"
                and self.decode_rope_mode == "dedicated"
                and self.batch == 32
            )
            if not advisor_rope:
                key = self._prepare_decode_heads(key, self.local_num_kv_heads, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                query = self._prepare_decode_heads(query, self.local_num_heads, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            # The dedicated op is fastest when the logical user rows already
            # fill a physical tile.  For smaller batches its output exposes
            # the logical row count, which cannot be converted to the
            # tile-height head-sharded cache layout.  Keep padding ownership
            # inside the decoder by using the established per-user form for
            # that case; the public input remains unpadded.
            use_dedicated_rope = self.decode_rope_mode == "dedicated" and self.batch % 32 == 0
            apply_rope = self._apply_decode_rope_dedicated if use_dedicated_rope else self._apply_decode_rope_per_user
            key_rotated = apply_rope(key, cos, sin)
            query_rotated = apply_rope(query, cos, sin)
            ttnn.deallocate(key, True)
            ttnn.deallocate(query, True)
            key_rotated = self._move_owned(key_rotated, self.decode_head_memory_config)
            if advisor_rope:
                query_rotated = self._move_owned(query_rotated, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(cos, True)
        ttnn.deallocate(sin, True)
        return query_rotated, key_rotated, value

    def _decode_attention(
        self,
        residual,
        *,
        key_cache,
        value_cache,
        cache_position,
        rotary_position,
        position_index,
        page_table,
    ):
        query, key, value = self._decode_qkv(residual, rotary_position=rotary_position)
        ttnn.experimental.paged_update_cache(
            value_cache, value, update_idxs_tensor=cache_position, share_cache=False, page_table=page_table
        )
        ttnn.experimental.paged_update_cache(
            key_cache, key, update_idxs_tensor=cache_position, share_cache=False, page_table=page_table
        )
        ttnn.deallocate(value, True)
        ttnn.deallocate(key, True)
        if page_table is None:
            attention = ttnn.transformer.scaled_dot_product_attention_decode(
                query,
                key_cache,
                value_cache,
                cur_pos_tensor=cache_position,
                is_causal=True,
                scale=self.scale,
                program_config=self.decode_sdpa_program_config,
                compute_kernel_config=self.attention_compute_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            attention = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                query,
                key_cache,
                value_cache,
                page_table_tensor=page_table,
                cur_pos_tensor=cache_position,
                scale=self.scale,
                program_config=self.decode_sdpa_program_config,
                compute_kernel_config=self.attention_compute_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        ttnn.deallocate(query, True)
        attention_sharded = ttnn.to_memory_config(attention, self.decode_head_memory_config)
        ttnn.deallocate(attention, True)
        concatenated = ttnn.experimental.nlp_concat_heads_decode(attention_sharded, num_heads=self.local_num_heads)
        ttnn.deallocate(attention_sharded, True)
        projected_input = self._move_owned(concatenated, self.o_input_memory_config)
        projected = ttnn.matmul(
            projected_input,
            self.o_decode_weight if self.o_decode_weight is not None else self.o_weight,
            dtype=ttnn.bfloat16,
            program_config=self.o_decode_program_config,
            compute_kernel_config=self.attention_compute_config,
            memory_config=self.o_output_memory_config,
        )
        ttnn.deallocate(projected_input, True)
        projected = self._move_owned(projected, self.residual_memory_config)
        projected = self._all_reduce_partial(projected, memory_config=self.residual_memory_config)
        output = ttnn.add(residual, projected, memory_config=self.residual_memory_config)
        ttnn.deallocate(residual, True)
        ttnn.deallocate(projected, True)
        return output

    def _prefill_mlp_chunk(self, residual):
        m = int(residual.shape[-2])
        normed = ttnn.rms_norm(
            residual,
            epsilon=self.rms_norm_eps,
            weight=self.post_attention_norm_weight,
            compute_kernel_config=self.norm_compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if self.use_packed_mlp:
            packed = ttnn.matmul(
                normed,
                self.gate_up_weight,
                dtype=ttnn.bfloat16,
                program_config=_prefill_matmul_program_config(
                    self.mesh_device,
                    m,
                    self.hidden_size,
                    2 * self.local_intermediate_size,
                    grid_x_limit=self.prefill_grid_x,
                    in0_block_w=self.prefill_in0_block_w,
                ),
                compute_kernel_config=self.mlp_compute_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            gate = ttnn.slice(packed, [0, 0, 0, 0], [1, 1, m, self.local_intermediate_size])
            up = ttnn.slice(
                packed,
                [0, 0, 0, self.local_intermediate_size],
                [1, 1, m, 2 * self.local_intermediate_size],
            )
            ttnn.deallocate(packed, True)
        else:
            program = _prefill_matmul_program_config(
                self.mesh_device,
                m,
                self.hidden_size,
                self.local_intermediate_size,
                grid_x_limit=self.prefill_grid_x,
                in0_block_w=self.prefill_in0_block_w,
            )
            gate = ttnn.matmul(
                normed,
                self.gate_weight,
                dtype=ttnn.bfloat16,
                program_config=program,
                compute_kernel_config=self.mlp_compute_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            up = ttnn.matmul(
                normed,
                self.up_weight,
                dtype=ttnn.bfloat16,
                program_config=program,
                compute_kernel_config=self.mlp_compute_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        ttnn.deallocate(normed, True)
        gated = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(gate, True)
        ttnn.deallocate(up, True)
        down = ttnn.matmul(
            gated,
            self.down_weight,
            dtype=ttnn.bfloat16,
            program_config=_prefill_matmul_program_config(
                self.mesh_device,
                m,
                self.local_intermediate_size,
                self.hidden_size,
                grid_x_limit=self.prefill_grid_x,
                in0_block_w=self.prefill_in0_block_w,
            ),
            compute_kernel_config=self.mlp_compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(gated, True)
        down = self._all_reduce_partial(down, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        output = ttnn.add(residual, down, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(residual, True)
        ttnn.deallocate(down, True)
        return output

    def _prefill_mlp(self, residual):
        rows = int(residual.shape[-2])
        if rows <= 1024:
            return self._prefill_mlp_chunk(residual)
        chunks = []
        for start in range(0, rows, 1024):
            end = min(start + 1024, rows)
            chunk = ttnn.slice(
                residual, [0, 0, start, 0], [1, 1, end, self.hidden_size], memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            chunks.append(self._prefill_mlp_chunk(chunk))
        ttnn.deallocate(residual, True)
        output = ttnn.concat(chunks, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for chunk in chunks:
            ttnn.deallocate(chunk, True)
        return output

    def _decode_mlp(self, residual):
        normed = self._decode_norm(residual, self.post_attention_norm_weight)
        mlp_input = self._move_owned(normed, self.mlp_input_memory_config)
        if self.use_packed_mlp:
            packed = ttnn.matmul(
                mlp_input,
                self.gate_up_decode_weight,
                dtype=ttnn.bfloat16,
                program_config=self.packed_gate_decode_program_config,
                compute_kernel_config=self.mlp_compute_config,
                memory_config=self.packed_mlp_output_memory_config,
            )
            if self.packed_mlp_unpack_mode == "l1_sharded":
                gate = ttnn.slice(
                    packed,
                    [0, 0, 0, 0],
                    [1, 1, self.batch, self.local_intermediate_size],
                    memory_config=self.mlp_output_memory_config,
                )
                up = ttnn.slice(
                    packed,
                    [0, 0, 0, self.local_intermediate_size],
                    [1, 1, self.batch, 2 * self.local_intermediate_size],
                    memory_config=self.mlp_output_memory_config,
                )
                ttnn.deallocate(packed, True)
            else:
                packed_dram = ttnn.to_memory_config(packed, ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(packed, True)
                gate = ttnn.slice(packed_dram, [0, 0, 0, 0], [1, 1, self.batch, self.local_intermediate_size])
                up = ttnn.slice(
                    packed_dram,
                    [0, 0, 0, self.local_intermediate_size],
                    [1, 1, self.batch, 2 * self.local_intermediate_size],
                )
                ttnn.deallocate(packed_dram, True)
        else:
            gate = ttnn.matmul(
                mlp_input,
                self.gate_decode_weight if self.gate_decode_weight is not None else self.gate_weight,
                dtype=ttnn.bfloat16,
                program_config=self.gate_decode_program_config,
                compute_kernel_config=self.mlp_compute_config,
                memory_config=self.mlp_output_memory_config,
            )
            up = ttnn.matmul(
                mlp_input,
                self.up_decode_weight if self.up_decode_weight is not None else self.up_weight,
                dtype=ttnn.bfloat16,
                program_config=self.gate_decode_program_config,
                compute_kernel_config=self.mlp_compute_config,
                memory_config=self.mlp_output_memory_config,
            )
        ttnn.deallocate(mlp_input, True)
        gated = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=ttnn.bfloat16,
            memory_config=(
                ttnn.DRAM_MEMORY_CONFIG
                if self.use_packed_mlp and self.packed_mlp_unpack_mode == "dram"
                else self.mlp_output_memory_config
            ),
        )
        ttnn.deallocate(gate, True)
        ttnn.deallocate(up, True)
        down_input = self._move_owned(gated, self.down_input_memory_config)
        down = ttnn.matmul(
            down_input,
            self.down_decode_weight if self.down_decode_weight is not None else self.down_weight,
            dtype=ttnn.bfloat16,
            program_config=self.down_decode_program_config,
            compute_kernel_config=self.mlp_compute_config,
            memory_config=self.down_output_memory_config,
        )
        ttnn.deallocate(down_input, True)
        down = self._move_owned(down, self.residual_memory_config)
        down = self._all_reduce_partial(down, memory_config=self.residual_memory_config)
        output = ttnn.add(residual, down, memory_config=self.residual_memory_config)
        ttnn.deallocate(residual, True)
        ttnn.deallocate(down, True)
        return output

    def prefill_forward(
        self,
        hidden_states,
        *,
        key_cache,
        value_cache,
        page_table=None,
        prompt_lens=None,
        chunk_start_idx: int = 0,
        chunk_page_table=None,
    ):
        batch, seq_len = self._validate_hidden(hidden_states, decode=False)
        self._validate_caches(key_cache, value_cache, page_table=page_table)
        if prompt_lens is not None and len(prompt_lens) != batch:
            raise ValueError(f"prompt_lens must contain {batch} entries")
        if chunk_start_idx < 0 or chunk_start_idx + seq_len > self.max_cache_len:
            raise ValueError("prefill chunk lies outside the supported context")
        required_pages = math.ceil((chunk_start_idx + seq_len) / self.page_block_size)
        if page_table is not None and required_pages > int(page_table.shape[1]):
            raise ValueError("page_table does not cover the logical prefill length")
        if chunk_start_idx and (page_table is None or chunk_page_table is None):
            raise ValueError("chunked prefill requires full and chunk page tables")
        residual = ttnn.reshape(hidden_states, (1, 1, batch * seq_len, self.hidden_size))
        residual = self._prefill_attention(
            residual,
            batch=batch,
            seq_len=seq_len,
            key_cache=key_cache,
            value_cache=value_cache,
            page_table=page_table,
            prompt_lens=prompt_lens,
            chunk_start_idx=chunk_start_idx,
            chunk_page_table=chunk_page_table,
        )
        residual = self._prefill_mlp(residual)
        output = ttnn.reshape(residual, (1, batch, seq_len, self.hidden_size))
        ttnn.deallocate(residual, False)
        return output

    def _validate_decode_runtime(
        self,
        *,
        key_cache,
        value_cache,
        cache_position,
        rotary_position,
        position_index,
        page_table,
    ):
        self._validate_caches(key_cache, value_cache, page_table=page_table)
        self._validate_cache_position(cache_position)
        if tuple(int(value) for value in rotary_position.shape) != (self.batch,) or rotary_position.dtype not in (
            ttnn.int32,
            ttnn.uint32,
        ):
            raise ValueError(f"rotary_position must be int32/uint32 with shape [{self.batch}]")
        if not 0 <= position_index < self.max_cache_len:
            raise ValueError(f"position_index={position_index} is outside configured cache")

    def _ensure_decode_all_reduce_resources(self) -> None:
        if not self.use_persistent_decode_all_reduce:
            return
        if self.ccl_dtype != ttnn.bfloat16:
            return
        if self.decode_all_reduce_resources is None:
            self.decode_all_reduce_resources = DecodeAllReduceResources(
                mesh_device=self.mesh_device,
                batch=self.batch,
                hidden_size=self.hidden_size,
                residual_memory_config=self.residual_memory_config,
                residual_num_cores=self.residual_num_cores,
            )
            self.owns_decode_all_reduce_resources = True
        else:
            self.decode_all_reduce_resources.validate(
                mesh_device=self.mesh_device,
                batch=self.batch,
                hidden_size=self.hidden_size,
                residual_memory_config=self.residual_memory_config,
            )
            self.decode_all_reduce_resources.activate()

    def decode_forward_to_residual(
        self,
        hidden_states,
        *,
        key_cache,
        value_cache,
        cache_position,
        position_index: int,
        page_table=None,
        rotary_position=None,
    ):
        """Decode a public input and keep the replicated residual width-sharded in L1."""
        batch, _ = self._validate_hidden(hidden_states, decode=True)
        rotary_position = cache_position if rotary_position is None else rotary_position
        self._validate_decode_runtime(
            key_cache=key_cache,
            value_cache=value_cache,
            cache_position=cache_position,
            rotary_position=rotary_position,
            position_index=position_index,
            page_table=page_table,
        )
        self._ensure_decode_all_reduce_resources()
        residual = ttnn.reshape(hidden_states, (1, 1, batch, self.hidden_size))
        padded_rows = 32 * math.ceil(batch / 32)
        if batch != padded_rows:
            logical_residual = residual
            residual = ttnn.pad(
                logical_residual,
                padding=[(0, 0), (0, 0), (0, padded_rows - batch), (0, 0)],
                value=0.0,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(logical_residual, False)
        residual = ttnn.to_memory_config(residual, self.residual_memory_config)
        residual = self._decode_attention(
            residual,
            key_cache=key_cache,
            value_cache=value_cache,
            cache_position=cache_position,
            rotary_position=rotary_position,
            position_index=position_index,
            page_table=page_table,
        )
        residual = self._decode_mlp(residual)
        return residual

    def decode_forward_from_residual(
        self,
        residual,
        *,
        key_cache,
        value_cache,
        cache_position,
        position_index: int,
        page_table=None,
        rotary_position=None,
    ):
        """Decode from the stack-native replicated width-sharded L1 residual contract."""
        padded_rows = 32 * math.ceil(self.batch / 32)
        expected_shape = (1, 1, padded_rows, self.hidden_size)
        shape = tuple(int(value) for value in residual.shape)
        if shape != expected_shape:
            raise ValueError(f"stack residual must have shape {expected_shape}, got {shape}")
        if residual.memory_config() != self.residual_memory_config:
            raise ValueError("stack residual must preserve the decoder's residual_memory_config")
        rotary_position = cache_position if rotary_position is None else rotary_position
        self._validate_decode_runtime(
            key_cache=key_cache,
            value_cache=value_cache,
            cache_position=cache_position,
            rotary_position=rotary_position,
            position_index=position_index,
            page_table=page_table,
        )
        self._ensure_decode_all_reduce_resources()
        residual = self._decode_attention(
            residual,
            key_cache=key_cache,
            value_cache=value_cache,
            cache_position=cache_position,
            rotary_position=rotary_position,
            position_index=position_index,
            page_table=page_table,
        )
        return self._decode_mlp(residual)

    def materialize_decode_output(self, residual):
        """Consume a stack residual and restore the public DRAM output contract."""
        batch = self.batch
        direct_dram = self.decode_output_mode == "direct_dram" and int(residual.shape[-2]) == batch
        if int(residual.shape[-2]) != batch:
            padded = residual
            residual = ttnn.slice(padded, [0, 0, 0, 0], [1, 1, batch, self.hidden_size])
            ttnn.deallocate(padded, False)
        if direct_dram:
            output_dram = ttnn.to_memory_config(residual, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(residual, True)
            output = ttnn.reshape(output_dram, (1, batch, 1, self.hidden_size))
            ttnn.deallocate(output_dram, False)
            return output
        residual_interleaved = self._move_owned(residual, ttnn.L1_MEMORY_CONFIG)
        output_l1 = ttnn.reshape(residual_interleaved, (1, batch, 1, self.hidden_size))
        ttnn.deallocate(residual_interleaved, False)
        output = ttnn.to_memory_config(output_l1, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(output_l1, True)
        return output

    def decode_forward(
        self,
        hidden_states,
        *,
        key_cache,
        value_cache,
        cache_position,
        position_index: int,
        page_table=None,
        rotary_position=None,
    ):
        residual = self.decode_forward_to_residual(
            hidden_states,
            key_cache=key_cache,
            value_cache=value_cache,
            cache_position=cache_position,
            position_index=position_index,
            page_table=page_table,
            rotary_position=rotary_position,
        )
        return self.materialize_decode_output(residual)
