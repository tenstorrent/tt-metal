# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Four-chip tensor-parallel Falcon3 decoder layer.

The selected single-chip :class:`OptimizedDecoder` is the numerical and local
program-config baseline.  This module restores the compiler-proven column/row
tensor parallel algebra on the full Blackhole mesh available to this stage:
QKV and gate/up are column parallel, O and down are row parallel, and the two
row-parallel partials are reduced to a replicated residual stream.

The target is deliberately fixed to a 1x4 Blackhole mesh.  Its fabric and CCL
operations use the compiler-proven Ring topology on mesh axis 1.  Supporting
smaller or other mesh shapes is outside the stage contract.
"""

from __future__ import annotations

import math
from typing import Mapping

import torch

import ttnn
from models.autoports.tiiuae_falcon3_7b_base.tt.functional_decoder import (
    EMITTED_BATCH,
    _config_value,
    _resolve_layer_tensor,
)
from models.autoports.tiiuae_falcon3_7b_base.tt.optimized_decoder import (
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
from models.common.modules.tt_ccl import default_topology, get_num_links

TARGET_MESH_SHAPE = (1, 4)
TENSOR_PARALLEL_SIZE = 4
TENSOR_PARALLEL_AXIS = 1
DEFAULT_PAGE_BLOCK_SIZE = 32

MULTICHIP_PRECISION_POLICIES = {
    **PRECISION_POLICIES,
    # Candidate required to isolate MLP math fidelity while keeping every
    # projection weight at BFP4 and attention on the selected LoFi contract.
    "all_bfp4_mlp_hifi2": {
        "attention": ttnn.bfloat4_b,
        "mlp_gate_up": ttnn.bfloat4_b,
        "mlp_down": ttnn.bfloat4_b,
        "kv_cache": ttnn.bfloat8_b,
        "attention_fidelity": ttnn.MathFidelity.LoFi,
        "mlp_fidelity": ttnn.MathFidelity.HiFi2,
    },
}


def _mesh_weight(
    host: torch.Tensor,
    *,
    dtype,
    mesh_device,
    shard_dim: int | None,
    local_shape: tuple[int, int] | None = None,
    memory_config=None,
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


def _replicated_tensor(host: torch.Tensor, *, dtype, layout, mesh_device, memory_config) -> ttnn.Tensor:
    return ttnn.from_torch(
        host.contiguous(),
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=memory_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _rank_grouped_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tp: int,
    *,
    padded_local_width: int | None = None,
) -> torch.Tensor:
    """Pack and optionally right-pad `[Q_rank,K_rank,V_rank]` per mesh rank."""
    q_chunks = q.transpose(-2, -1).chunk(tp, dim=-1)
    k_chunks = k.transpose(-2, -1).chunk(tp, dim=-1)
    v_chunks = v.transpose(-2, -1).chunk(tp, dim=-1)
    rank_groups = [torch.cat((q_chunks[rank], k_chunks[rank], v_chunks[rank]), dim=-1) for rank in range(tp)]
    if padded_local_width is not None:
        logical_width = int(rank_groups[0].shape[-1])
        if padded_local_width < logical_width:
            raise ValueError(f"padded_local_width={padded_local_width} is smaller than logical QKV {logical_width}")
        if padded_local_width > logical_width:
            rank_groups = [
                torch.cat((group, group.new_zeros(group.shape[0], padded_local_width - logical_width)), dim=-1)
                for group in rank_groups
            ]
    return torch.cat(rank_groups, dim=-1)


def _rank_padded_shards(host: torch.Tensor, *, tp: int, dim: int, padded_local_size: int) -> torch.Tensor:
    """Pad each future mesh shard independently before concatenating it."""
    shards = list(host.chunk(tp, dim=dim))
    logical_local_size = int(shards[0].shape[dim])
    if padded_local_size < logical_local_size:
        raise ValueError(f"padded_local_size={padded_local_size} is smaller than logical shard {logical_local_size}")
    if padded_local_size == logical_local_size:
        return host
    padded = []
    for shard in shards:
        pad_shape = list(shard.shape)
        pad_shape[dim] = padded_local_size - logical_local_size
        padded.append(torch.cat((shard, shard.new_zeros(pad_shape)), dim=dim))
    return torch.cat(padded, dim=dim)


def _rank_grouped_pair(first: torch.Tensor, second: torch.Tensor, *, tp: int) -> torch.Tensor:
    """Pack two already rank-grouped column-parallel tensors per mesh rank."""
    first_shards = first.chunk(tp, dim=1)
    second_shards = second.chunk(tp, dim=1)
    return torch.cat([torch.cat((first_shards[rank], second_shards[rank]), dim=1) for rank in range(tp)], dim=1)


class MultichipDecoder(LightweightModule):
    """Falcon3 dense decoder specialized for the available 1x4 Blackhole mesh."""

    single_chip_baseline_cls = OptimizedDecoder

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Mapping[str, torch.Tensor],
        *,
        hf_config,
        layer_idx: int,
        mesh_device,
        batch: int = EMITTED_BATCH,
        max_cache_len: int | None = None,
        precision_policy: str = "all_bfp4_lofi",
        decode_matmul_mode: str = "dram_sharded",
        advisor_residual_mode: str = "legacy_32core",
        use_packed_mlp: bool = False,
        return_sharded_decode_output: bool = True,
        qkv_target_cores: int = 4,
        o_target_cores: int = 4,
        gate_up_target_cores: int = 24,
        down_target_cores: int = 8,
        ccl_dtype=ttnn.bfloat8_b,
        ccl_mode: str = "persistent_async",
        num_links: int | None = None,
        attention_activation_dtype=ttnn.bfloat8_b,
        mlp_activation_dtype=ttnn.bfloat8_b,
        prefill_grid_x_limit: int = 11,
        prefill_l1_inputs: bool = False,
        page_block_size: int = DEFAULT_PAGE_BLOCK_SIZE,
        rope_cache: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
        persistent_ccl_resources: tuple[object, object, object, object] | None = None,
    ) -> "MultichipDecoder":
        if tuple(int(v) for v in mesh_device.shape) != TARGET_MESH_SHAPE:
            raise ValueError(f"MultichipDecoder requires mesh {TARGET_MESH_SHAPE}, got {mesh_device.shape}")
        if mesh_device.get_num_devices() != TENSOR_PARALLEL_SIZE:
            raise ValueError(f"MultichipDecoder requires exactly {TENSOR_PARALLEL_SIZE} devices")
        visible_devices = int(ttnn.get_num_devices())
        if visible_devices != TENSOR_PARALLEL_SIZE:
            raise ValueError(
                f"MultichipDecoder requires exactly {TENSOR_PARALLEL_SIZE} host-visible devices, got {visible_devices}"
            )
        if precision_policy not in MULTICHIP_PRECISION_POLICIES:
            raise ValueError(f"Unknown precision policy {precision_policy!r}")
        if decode_matmul_mode not in ("dram_sharded", "shard_advisor"):
            raise ValueError("decode_matmul_mode must be 'dram_sharded' or 'shard_advisor'")
        if advisor_residual_mode not in ("legacy_32core", "report"):
            raise ValueError("advisor_residual_mode must be 'legacy_32core' or 'report'")
        if use_packed_mlp and decode_matmul_mode != "dram_sharded":
            raise ValueError("use_packed_mlp is currently a DRAM-sharded decode candidate")
        if qkv_target_cores not in (4, 8, 16):
            raise ValueError("qkv_target_cores must be 4, 8, or 16")
        if o_target_cores not in (4, 8, 12, 16, 24, 48):
            raise ValueError("o_target_cores must be 4, 8, 12, 16, 24, or 48")
        if gate_up_target_cores not in (8, 12, 24, 48):
            raise ValueError("gate_up_target_cores must be 8, 12, 24, or 48")
        if down_target_cores not in (8, 12, 24, 48):
            raise ValueError("down_target_cores must be 8, 12, 24, or 48")
        if ccl_dtype not in (ttnn.bfloat16, ttnn.bfloat8_b):
            raise ValueError("ccl_dtype must be bfloat16 or bfloat8_b")
        if ccl_mode not in ("standard", "persistent_async"):
            raise ValueError("ccl_mode must be 'standard' or 'persistent_async'")
        if attention_activation_dtype not in (ttnn.bfloat16, ttnn.bfloat8_b):
            raise ValueError("attention_activation_dtype must be bfloat16 or bfloat8_b")
        if mlp_activation_dtype not in (ttnn.bfloat16, ttnn.bfloat8_b):
            raise ValueError("mlp_activation_dtype must be bfloat16 or bfloat8_b")
        if prefill_grid_x_limit not in (10, 11):
            raise ValueError("prefill_grid_x_limit must be 10 or 11")
        detected_num_links = get_num_links(mesh_device, TENSOR_PARALLEL_AXIS)
        if num_links is None:
            num_links = detected_num_links
        if num_links not in (1, 2) or num_links > detected_num_links:
            raise ValueError(
                f"num_links must be 1..{detected_num_links} on the target four-chip Blackhole mesh, got {num_links}"
            )
        detected_default_topology = default_topology(mesh_device)
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
        max_cache_len = max_position_embeddings if max_cache_len is None else int(max_cache_len)
        if max_cache_len <= 0:
            raise ValueError("max_cache_len must be positive")
        expected = (3072, 28, 12, 4, 256, 23040)
        actual = (hidden_size, num_layers, num_heads, num_kv_heads, head_dim, intermediate_size)
        if actual != expected:
            raise ValueError(f"HF config does not match Falcon3-7B IR: got {actual}, expected {expected}")
        if str(_config_value(hf_config, "hidden_act")) != "silu":
            raise ValueError("Falcon3 multichip decoder requires SwiGLU/SILU")
        if not math.isclose(rms_norm_eps, 1e-6, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Expected rms_norm_eps=1e-6, got {rms_norm_eps}")
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
        local_intermediate = intermediate_size // TENSOR_PARALLEL_SIZE
        dram_grid = mesh_device.dram_grid_size()
        dram_banks = int(dram_grid.x) * int(dram_grid.y)
        dram_width_granularity = 32 * dram_banks
        qkv_decode_granularity = math.lcm(dram_width_granularity, 32 * qkv_target_cores)
        mlp_decode_granularity = math.lcm(
            dram_width_granularity,
            32 * gate_up_target_cores,
            32 * down_target_cores,
        )
        local_qkv_decode_size = qkv_decode_granularity * math.ceil(local_qkv_size / qkv_decode_granularity)
        local_intermediate_decode_size = mlp_decode_granularity * math.ceil(local_intermediate / mlp_decode_granularity)
        qkv_host = _rank_grouped_qkv(q, k, v, TENSOR_PARALLEL_SIZE)
        qkv_decode_host = _rank_grouped_qkv(
            q,
            k,
            v,
            TENSOR_PARALLEL_SIZE,
            padded_local_width=local_qkv_decode_size,
        )
        gate_host = gate.transpose(-2, -1)
        up_host = up.transpose(-2, -1)
        down_host = down.transpose(-2, -1)
        gate_decode_host = _rank_padded_shards(
            gate_host,
            tp=TENSOR_PARALLEL_SIZE,
            dim=1,
            padded_local_size=local_intermediate_decode_size,
        )
        up_decode_host = _rank_padded_shards(
            up_host,
            tp=TENSOR_PARALLEL_SIZE,
            dim=1,
            padded_local_size=local_intermediate_decode_size,
        )
        down_decode_host = _rank_padded_shards(
            down_host,
            tp=TENSOR_PARALLEL_SIZE,
            dim=0,
            padded_local_size=local_intermediate_decode_size,
        )
        gate_up_decode_host = _rank_grouped_pair(gate_decode_host, up_decode_host, tp=TENSOR_PARALLEL_SIZE)

        policy = dict(MULTICHIP_PRECISION_POLICIES[precision_policy])
        prefill_memcfg = ttnn.DRAM_MEMORY_CONFIG
        self = cls()
        self.mesh_device = mesh_device
        self.layer_idx = layer_idx
        self.batch = batch
        self.max_cache_len = max_cache_len
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.local_num_heads = local_heads
        self.local_num_kv_heads = local_kv_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.local_hidden_size = local_hidden
        self.local_intermediate_size = local_intermediate
        self.local_intermediate_decode_size = local_intermediate_decode_size
        self.mlp_decode_padding = local_intermediate_decode_size - local_intermediate
        self.local_qkv_size = local_qkv_size
        self.local_qkv_decode_size = local_qkv_decode_size
        self.qkv_decode_padding = local_qkv_decode_size - local_qkv_size
        self.dram_banks = dram_banks
        self.rms_norm_eps = rms_norm_eps
        self.scale = 1.0 / math.sqrt(head_dim)
        self.precision_policy_name = precision_policy
        self.precision_policy = policy
        self.decode_matmul_mode = decode_matmul_mode
        self.advisor_residual_mode = advisor_residual_mode
        self.use_packed_mlp = use_packed_mlp
        self.return_sharded_decode_output = return_sharded_decode_output
        self.tensor_parallel_size = TENSOR_PARALLEL_SIZE
        self.tensor_parallel_axis = TENSOR_PARALLEL_AXIS
        # The generic helper conservatively returns Linear for non-T3K meshes.
        # This fixed target has a physical 1-D ring and the compiler graph also
        # requires Ring, so retain the detected value as provenance and use Ring.
        self.detected_default_topology = detected_default_topology
        self.topology = ttnn.Topology.Ring
        self.visible_device_count = visible_devices
        self.detected_num_links = detected_num_links
        self.num_links = num_links
        self.ccl_dtype = ccl_dtype
        self.ccl_mode = ccl_mode
        self.attention_activation_dtype = attention_activation_dtype
        self.mlp_activation_dtype = mlp_activation_dtype
        self.prefill_grid_x_limit = prefill_grid_x_limit
        self.prefill_l1_inputs = prefill_l1_inputs
        self.page_block_size = page_block_size
        self.qkv_target_cores = qkv_target_cores
        self.o_target_cores = o_target_cores
        self.gate_up_target_cores = gate_up_target_cores
        self.down_target_cores = down_target_cores

        # Prefill and decode retain distinct physical copies, matching the
        # single-chip AutoFix result.  Only each rank's TP slice is resident.
        self.qkv_weight = _mesh_weight(
            qkv_host,
            dtype=policy["attention"],
            mesh_device=mesh_device,
            shard_dim=1,
            memory_config=prefill_memcfg,
        )
        self.qkv_decode_weight = _mesh_weight(
            qkv_decode_host,
            dtype=policy["attention"],
            mesh_device=mesh_device,
            shard_dim=1,
            local_shape=(hidden_size, local_qkv_decode_size),
            memory_config=ttnn.DRAM_MEMORY_CONFIG if decode_matmul_mode == "shard_advisor" else None,
        )
        self.o_weight = _mesh_weight(
            o.transpose(-2, -1),
            dtype=policy["attention"],
            mesh_device=mesh_device,
            shard_dim=0,
            memory_config=prefill_memcfg,
        )
        self.o_decode_weight = _mesh_weight(
            o.transpose(-2, -1),
            dtype=policy["attention"],
            mesh_device=mesh_device,
            shard_dim=0,
            local_shape=(local_hidden, hidden_size),
            memory_config=ttnn.DRAM_MEMORY_CONFIG if decode_matmul_mode == "shard_advisor" else None,
        )
        self.gate_weight = _mesh_weight(
            gate_host,
            dtype=policy["mlp_gate_up"],
            mesh_device=mesh_device,
            shard_dim=1,
            memory_config=prefill_memcfg,
        )
        self.up_weight = _mesh_weight(
            up_host,
            dtype=policy["mlp_gate_up"],
            mesh_device=mesh_device,
            shard_dim=1,
            memory_config=prefill_memcfg,
        )
        if use_packed_mlp:
            self.gate_up_decode_weight = _mesh_weight(
                gate_up_decode_host,
                dtype=policy["mlp_gate_up"],
                mesh_device=mesh_device,
                shard_dim=1,
                local_shape=(hidden_size, 2 * local_intermediate_decode_size),
            )
            self.gate_decode_weight = None
            self.up_decode_weight = None
        else:
            self.gate_up_decode_weight = None
            self.gate_decode_weight = _mesh_weight(
                gate_decode_host,
                dtype=policy["mlp_gate_up"],
                mesh_device=mesh_device,
                shard_dim=1,
                local_shape=(hidden_size, local_intermediate_decode_size),
                memory_config=ttnn.DRAM_MEMORY_CONFIG if decode_matmul_mode == "shard_advisor" else None,
            )
            self.up_decode_weight = _mesh_weight(
                up_decode_host,
                dtype=policy["mlp_gate_up"],
                mesh_device=mesh_device,
                shard_dim=1,
                local_shape=(hidden_size, local_intermediate_decode_size),
                memory_config=ttnn.DRAM_MEMORY_CONFIG if decode_matmul_mode == "shard_advisor" else None,
            )
        self.down_weight = _mesh_weight(
            down_host,
            dtype=policy["mlp_down"],
            mesh_device=mesh_device,
            shard_dim=0,
            memory_config=prefill_memcfg,
        )
        self.down_decode_weight = _mesh_weight(
            down_decode_host,
            dtype=policy["mlp_down"],
            mesh_device=mesh_device,
            shard_dim=0,
            local_shape=(local_intermediate_decode_size, hidden_size),
            memory_config=ttnn.DRAM_MEMORY_CONFIG if decode_matmul_mode == "shard_advisor" else None,
        )
        self.input_norm_weight = _mesh_weight(
            input_norm,
            dtype=ttnn.bfloat16,
            mesh_device=mesh_device,
            shard_dim=None,
            memory_config=prefill_memcfg,
        )
        self.post_attention_norm_weight = _mesh_weight(
            post_norm,
            dtype=ttnn.bfloat16,
            mesh_device=mesh_device,
            shard_dim=None,
            memory_config=prefill_memcfg,
        )

        rope_parameters = getattr(hf_config, "rope_parameters", None) or {}
        rope_theta = getattr(hf_config, "rope_theta", None) or rope_parameters.get("rope_theta")
        if float(rope_theta) != 1000042.0 or rope_parameters.get("rope_type", "default") != "default":
            raise ValueError("Falcon3 multichip decoder requires the emitted default RoPE contract")
        self.owns_rope_cache = rope_cache is None
        if rope_cache is None:
            positions = torch.arange(max_cache_len, dtype=torch.float32)
            inv_freq = 1.0 / (float(rope_theta) ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
            angles = torch.outer(positions, inv_freq)
            angles = torch.cat([angles, angles], dim=-1)
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
        else:
            if len(rope_cache) != 2:
                raise ValueError("rope_cache must contain exactly (cos_cache, sin_cache)")
            self.cos_cache, self.sin_cache = rope_cache
            for name, cache in (("cos_cache", self.cos_cache), ("sin_cache", self.sin_cache)):
                shape = tuple(int(value) for value in cache.shape)
                if len(shape) != 2 or shape[0] < max_cache_len or shape[1] != head_dim:
                    raise ValueError(f"Shared {name} shape {shape} must cover at least ({max_cache_len}, {head_dim})")
                if cache.get_dtype() != ttnn.bfloat16 or cache.get_layout() != ttnn.TILE_LAYOUT:
                    raise ValueError(f"Shared {name} must be BF16 TILE_LAYOUT")
                if len(ttnn.get_device_tensors(cache)) != TENSOR_PARALLEL_SIZE:
                    raise ValueError(f"Shared {name} must be resident on the target four-device mesh")

        grid_size = mesh_device.compute_with_storage_grid_size()
        grid_x = min(batch, int(grid_size.x))
        while grid_x > 0 and (batch % grid_x != 0 or batch // grid_x > int(grid_size.y)):
            grid_x -= 1
        if grid_x == 0:
            raise ValueError(f"batch={batch} cannot form a decode head grid within {grid_size}")
        self.decode_head_memory_config = ttnn.create_sharded_memory_config(
            shape=(32, head_dim),
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
        self._allocate_persistent_ccl_buffers(persistent_ccl_resources)
        return self

    def _build_configs(self) -> None:
        grid_size = self.mesh_device.compute_with_storage_grid_size()
        max_x = min(8, int(grid_size.x))
        max_y = min(10, int(grid_size.y))
        padded_rows = 32 * math.ceil(self.batch / 32)
        hidden_tiles = self.hidden_size // 32
        local_hidden_tiles = self.local_hidden_size // 32
        qkv_tiles = self.local_qkv_decode_size // 32
        local_mlp_tiles = self.local_intermediate_decode_size // 32

        self.residual_grid = _core_grid_for_tiles(hidden_tiles, hidden_tiles, target_cores=32, max_x=max_x, max_y=max_y)
        self.qkv_grid = _core_grid_for_tiles(
            hidden_tiles, qkv_tiles, target_cores=self.qkv_target_cores, max_x=max_x, max_y=max_y
        )
        self.o_grid = _core_grid_for_tiles(
            local_hidden_tiles, hidden_tiles, target_cores=self.o_target_cores, max_x=max_x, max_y=max_y
        )
        self.gate_up_grid = _core_grid_for_tiles(
            hidden_tiles,
            local_mlp_tiles,
            target_cores=self.gate_up_target_cores,
            max_x=max_x,
            max_y=max_y,
        )
        self.down_grid = _core_grid_for_tiles(
            local_mlp_tiles,
            hidden_tiles,
            target_cores=self.down_target_cores,
            max_x=max_x,
            max_y=max_y,
        )
        self.packed_mlp_grid = _core_grid_for_tiles(
            hidden_tiles,
            2 * local_mlp_tiles,
            target_cores=self.gate_up_target_cores,
            max_x=max_x,
            max_y=max_y,
        )

        self.residual_memory_config = _sharded_memory_config(padded_rows, self.hidden_size, self.residual_grid)
        self.qkv_input_memory_config = _sharded_memory_config(padded_rows, self.hidden_size, self.qkv_grid)
        self.qkv_output_memory_config = _matmul_output_memory_config(
            padded_rows, self.local_qkv_decode_size, self.qkv_grid, self.mesh_device
        )
        self.o_input_memory_config = _sharded_memory_config(padded_rows, self.local_hidden_size, self.o_grid)
        self.o_output_memory_config = _matmul_output_memory_config(
            padded_rows, self.hidden_size, self.o_grid, self.mesh_device
        )
        self.mlp_input_memory_config = _sharded_memory_config(padded_rows, self.hidden_size, self.gate_up_grid)
        self.mlp_output_memory_config = _matmul_output_memory_config(
            padded_rows, self.local_intermediate_decode_size, self.gate_up_grid, self.mesh_device
        )
        self.down_input_memory_config = _sharded_memory_config(
            padded_rows, self.local_intermediate_decode_size, self.down_grid
        )
        self.down_output_memory_config = _matmul_output_memory_config(
            padded_rows, self.hidden_size, self.down_grid, self.mesh_device
        )
        self.packed_mlp_output_memory_config = _matmul_output_memory_config(
            padded_rows, 2 * self.local_intermediate_decode_size, self.packed_mlp_grid, self.mesh_device
        )
        self.norm_memory_config = self.residual_memory_config

        norm_block_w = hidden_tiles // self.residual_grid.num_cores
        self.norm_program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[self.residual_grid.x, self.residual_grid.y],
            subblock_w=_largest_divisor(norm_block_w, limit=4),
            block_h=padded_rows // 32,
            block_w=norm_block_w,
            inplace=False,
        )
        self.qkv_decode_program_config = _dram_matmul_program_config(
            padded_rows, self.hidden_size, self.local_qkv_decode_size, self.qkv_grid
        )
        self.o_decode_program_config = _dram_matmul_program_config(
            padded_rows, self.local_hidden_size, self.hidden_size, self.o_grid
        )
        self.gate_decode_program_config = _dram_matmul_program_config(
            padded_rows, self.hidden_size, self.local_intermediate_decode_size, self.gate_up_grid
        )
        self.down_decode_program_config = _dram_matmul_program_config(
            padded_rows, self.local_intermediate_decode_size, self.hidden_size, self.down_grid
        )
        self.packed_gate_decode_program_config = _dram_matmul_program_config(
            padded_rows,
            self.hidden_size,
            2 * self.local_intermediate_decode_size,
            self.packed_mlp_grid,
            in0_block_w=1,
        )
        if self.decode_matmul_mode == "shard_advisor":
            # Exact batch-32 local TP-rank proposal from this stage's
            # shard_advise/final_ir.mlir. CCL remains explicit around the
            # two row-parallel partials and is measured with these layouts.
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
                padded_rows, self.local_qkv_decode_size, 40, self.mesh_device
            )
            self.o_input_memory_config = ttnn.L1_MEMORY_CONFIG
            self.o_output_memory_config = _width_sharded_output_memory_config(
                padded_rows, self.hidden_size, 96, self.mesh_device
            )
            self.mlp_input_memory_config = _width_sharded_output_memory_config(
                padded_rows, self.hidden_size, 48, self.mesh_device
            )
            self.mlp_output_memory_config = _width_sharded_output_memory_config(
                padded_rows, self.local_intermediate_decode_size, 96, self.mesh_device
            )
            self.down_input_memory_config = ttnn.L1_MEMORY_CONFIG
            self.down_output_memory_config = _width_sharded_output_memory_config(
                padded_rows, self.hidden_size, 96, self.mesh_device
            )
            if self.advisor_residual_mode == "report":
                self.residual_memory_config = _width_sharded_output_memory_config(
                    padded_rows, self.hidden_size, 96, self.mesh_device
                )
                self.norm_memory_config = _advisor_norm_memory_config(padded_rows, self.hidden_size)
                norm_block_w = math.ceil(hidden_tiles / 11)
                self.norm_program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
                    compute_with_storage_grid_size=[11, 1],
                    subblock_w=_largest_divisor(norm_block_w, limit=4),
                    block_h=padded_rows // 32,
                    block_w=norm_block_w,
                    inplace=False,
                )
        self.decode_sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            exp_approx_mode=False,
            q_chunk_size=0,
            k_chunk_size=0,
        )

    def allocate_kv_cache(
        self,
        max_cache_len: int | None = None,
        *,
        paged: bool = False,
        num_blocks: int | None = None,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
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
        return (
            ttnn.zeros(
                shape,
                dtype=self.kv_cache_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            ttnn.zeros(
                shape,
                dtype=self.kv_cache_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
        )

    def _allocate_persistent_ccl_buffers(
        self,
        persistent_ccl_resources: tuple[object, object, object, object] | None,
    ) -> None:
        self.attention_ccl_buffer = None
        self.mlp_ccl_buffer = None
        self.attention_ccl_semaphore = None
        self.mlp_ccl_semaphore = None
        self.owns_persistent_ccl_resources = False
        self.persistent_ccl_resources = None
        if self.ccl_mode != "persistent_async":
            if persistent_ccl_resources is not None:
                raise ValueError("persistent_ccl_resources require ccl_mode='persistent_async'")
            return
        padded_rows = 32 * math.ceil(self.batch / 32)
        output_shard_spec = self.residual_memory_config.shard_spec
        if output_shard_spec is None:
            raise ValueError("persistent async all-reduce requires an L1-sharded residual")
        core_grid = output_shard_spec.grid
        output_shard_width = int(output_shard_spec.shape[1])
        buffer_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                core_grid,
                [padded_rows, output_shard_width * self.tensor_parallel_size],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        if persistent_ccl_resources is not None:
            if len(persistent_ccl_resources) != 4:
                raise ValueError("persistent_ccl_resources must contain two buffers and two semaphores")
            attention_buffer, mlp_buffer, attention_semaphore, mlp_semaphore = persistent_ccl_resources
            # ShardTensor2dMesh exposes the per-rank tensor shape here, not
            # the pre-distribution host shape's mesh-axis extent.
            expected_shape = (1, 1, padded_rows, self.hidden_size * self.tensor_parallel_size)
            for name, buffer in (("attention", attention_buffer), ("MLP", mlp_buffer)):
                if tuple(int(value) for value in buffer.shape) != expected_shape:
                    raise ValueError(
                        f"shared {name} CCL buffer has shape {tuple(buffer.shape)}, expected {expected_shape}"
                    )
                if buffer.dtype != self.ccl_dtype or buffer.memory_config() != buffer_memory_config:
                    raise ValueError(f"shared {name} CCL buffer does not match the decoder CCL dtype/layout contract")
            self.attention_ccl_buffer = attention_buffer
            self.mlp_ccl_buffer = mlp_buffer
            self.attention_ccl_semaphore = attention_semaphore
            self.mlp_ccl_semaphore = mlp_semaphore
            self.persistent_ccl_resources = persistent_ccl_resources
            return
        buffer_host = torch.zeros(
            (TARGET_MESH_SHAPE[0], TARGET_MESH_SHAPE[1], padded_rows, self.hidden_size * self.tensor_parallel_size),
            dtype=torch.bfloat16,
        )
        mapper = ttnn.ShardTensor2dMesh(
            self.mesh_device,
            dims=(0, 1),
            mesh_shape=ttnn.MeshShape(*TARGET_MESH_SHAPE),
        )
        self.attention_ccl_buffer = ttnn.from_torch(
            buffer_host,
            dtype=self.ccl_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=buffer_memory_config,
            mesh_mapper=mapper,
        )
        self.mlp_ccl_buffer = ttnn.from_torch(
            buffer_host,
            dtype=self.ccl_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=buffer_memory_config,
            mesh_mapper=mapper,
        )
        self.attention_ccl_semaphore = ttnn.create_global_semaphore(self.mesh_device, core_grid, 0)
        self.mlp_ccl_semaphore = ttnn.create_global_semaphore(self.mesh_device, core_grid, 0)
        self.owns_persistent_ccl_resources = True
        self.persistent_ccl_resources = (
            self.attention_ccl_buffer,
            self.mlp_ccl_buffer,
            self.attention_ccl_semaphore,
            self.mlp_ccl_semaphore,
        )

    def _validate_hidden(self, hidden_states, *, decode: bool) -> tuple[int, int]:
        shape = tuple(int(v) for v in hidden_states.shape)
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

    def _validate_cache_position(self, cache_position) -> None:
        if tuple(int(v) for v in cache_position.shape) != (self.batch,) or cache_position.dtype != ttnn.int32:
            raise ValueError(f"cache_position must be int32[{self.batch}]")

    def _validate_caches(self, key_cache, value_cache, *, page_table) -> None:
        k_shape = tuple(int(v) for v in key_cache.shape)
        v_shape = tuple(int(v) for v in value_cache.shape)
        if k_shape != v_shape or len(k_shape) != 4:
            raise ValueError(f"K/V cache shapes must match, got {k_shape}/{v_shape}")
        if k_shape[1] != self.local_num_kv_heads or k_shape[3] != self.head_dim:
            raise ValueError(
                f"each rank cache must own {self.local_num_kv_heads} KV heads of width {self.head_dim}, got {k_shape}"
            )
        if page_table is None:
            if k_shape[0] != self.batch:
                raise ValueError(f"contiguous cache first dimension must be batch={self.batch}")
        else:
            pt_shape = tuple(int(v) for v in page_table.shape)
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
        """Gather one RoPE row per user without a host scalar dependency."""
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
        """NeoX rotate-half used by per-user decode RoPE."""
        shape = tuple(int(v) for v in tensor.shape)
        first = ttnn.slice(tensor, [0, 0, 0, 0], [shape[0], shape[1], shape[2], self.head_dim // 2])
        second = ttnn.slice(
            tensor,
            [0, 0, 0, self.head_dim // 2],
            [shape[0], shape[1], shape[2], self.head_dim],
        )
        negative_second = ttnn.neg(second)
        rotated = ttnn.concat([negative_second, first], dim=-1)
        ttnn.deallocate(first, True)
        ttnn.deallocate(second, True)
        ttnn.deallocate(negative_second, True)
        return rotated

    def _apply_decode_rope_per_user(self, tensor, cos, sin):
        """Apply independent device-resident RoPE positions to each batch user."""
        heads = int(tensor.shape[2])
        cos_by_user = ttnn.transpose(cos, 1, 2)
        sin_by_user = ttnn.transpose(sin, 1, 2)
        if int(cos_by_user.shape[1]) != self.batch:
            cos_padded = cos_by_user
            sin_padded = sin_by_user
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
        ttnn.deallocate(cos_by_user, True)
        ttnn.deallocate(sin_by_user, True)
        ttnn.deallocate(cos_heads, True)
        ttnn.deallocate(sin_heads, True)
        ttnn.deallocate(rotated, True)
        ttnn.deallocate(cosine_term, True)
        ttnn.deallocate(sine_term, True)
        return output

    def _move_owned(self, tensor, memory_config):
        if tensor.memory_config() == memory_config:
            return tensor
        moved = ttnn.to_memory_config(tensor, memory_config)
        ttnn.deallocate(tensor, True)
        return moved

    def _all_reduce_partial(self, partial, *, memory_config, ccl_role: str | None = None):
        effective_ccl_dtype = self.ccl_dtype if ccl_role is not None else ttnn.bfloat16
        ccl_input = partial
        if ccl_input.dtype != effective_ccl_dtype:
            ccl_input = ttnn.typecast(ccl_input, effective_ccl_dtype)
            ttnn.deallocate(partial, True)
        if self.ccl_mode == "persistent_async" and ccl_role is not None:
            if ccl_role == "attention":
                buffer = self.attention_ccl_buffer
                semaphore = self.attention_ccl_semaphore
            elif ccl_role == "mlp":
                buffer = self.mlp_ccl_buffer
                semaphore = self.mlp_ccl_semaphore
            else:
                raise ValueError(f"unknown persistent CCL role {ccl_role!r}")
            reduced = ttnn.experimental.all_reduce_async(
                ccl_input,
                buffer,
                cluster_axis=self.tensor_parallel_axis,
                mesh_device=self.mesh_device,
                multi_device_global_semaphore=semaphore,
                dtype=effective_ccl_dtype,
                memory_config=memory_config,
                topology=self.topology,
                num_links=self.num_links,
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

    def _unpad_prefill_sequence(self, tensor, *, batch: int, heads: int, seq_len: int):
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

    def _prepare_decode_heads(self, tensor, num_heads: int, *, memory_config):
        interleaved = ttnn.to_memory_config(tensor, ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(tensor, True)
        expected_shape = (1, self.batch, num_heads, self.head_dim)
        if tuple(int(v) for v in interleaved.shape) == expected_shape:
            return self._move_owned(interleaved, memory_config)
        unpadded = ttnn.slice(interleaved, [0, 0, 0, 0], [1, self.batch, num_heads, self.head_dim])
        ttnn.deallocate(interleaved, True)
        prepared = ttnn.to_memory_config(unpadded, memory_config)
        ttnn.deallocate(unpadded, True)
        return prepared

    def _fill_prefill_cache(
        self,
        cache,
        fill,
        *,
        user_id: int,
        page_table,
        valid_len: int | None = None,
        chunk_start_idx: int = 0,
        chunk_page_table=None,
    ) -> None:
        if valid_len is not None:
            if valid_len <= 0:
                return
            if valid_len < int(fill.shape[2]):
                fill = ttnn.slice(
                    fill,
                    [0, 0, 0, 0],
                    [1, self.local_num_kv_heads, valid_len, self.head_dim],
                )
        if fill.dtype != cache.dtype:
            typed = ttnn.typecast(fill, cache.dtype)
            # A batch-one full-user slice aliases the parent K/V allocation,
            # which SDPA still consumes below.  Release only the slice handle.
            ttnn.deallocate(fill, False)
            fill = typed
        if page_table is None:
            if chunk_start_idx:
                raise ValueError("chunked prefill requires a paged KV cache")
            ttnn.fill_cache(cache, fill, batch_idx=user_id)
        else:
            fill_page_table = chunk_page_table if chunk_page_table is not None else page_table
            ttnn.experimental.paged_fill_cache(cache, fill, fill_page_table, batch_idx=user_id)
        ttnn.deallocate(fill, fill.dtype == cache.dtype and cache.dtype != ttnn.bfloat16)

    def _prefill_linear_chunked(self, tensor, weight, *, k: int, n: int, compute_kernel_config):
        """Bound prefill matmul L1 usage while preserving the logical row count."""
        rows = int(tensor.shape[-2])

        def run(chunk):
            chunk_rows = int(chunk.shape[-2])
            matmul_input = chunk
            if self.prefill_l1_inputs and chunk_rows <= 32:
                matmul_input = ttnn.to_memory_config(chunk, ttnn.L1_MEMORY_CONFIG)
            output = ttnn.matmul(
                matmul_input,
                weight,
                dtype=ttnn.bfloat16,
                program_config=_prefill_matmul_program_config(
                    self.mesh_device,
                    chunk_rows,
                    k,
                    n,
                    grid_x_limit=self.prefill_grid_x_limit,
                    in0_block_w=8,
                ),
                compute_kernel_config=compute_kernel_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            if matmul_input is not chunk:
                ttnn.deallocate(matmul_input, True)
            return output

        if rows <= 1024:
            return run(tensor)
        chunks = []
        for start in range(0, rows, 1024):
            end = min(start + 1024, rows)
            chunk = ttnn.slice(
                tensor,
                [0, 0, start, 0],
                [1, 1, end, k],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
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
        if self.prefill_l1_inputs and m <= 32:
            dram_normed = normed
            normed = ttnn.to_memory_config(dram_normed, ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(dram_normed, True)
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
        ttnn.deallocate(key, True)
        ttnn.deallocate(query, True)
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
                key_rotated,
                [user_id, 0, 0, 0],
                [user_id + 1, self.local_num_kv_heads, seq_len, self.head_dim],
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
        ttnn.deallocate(query_rotated, True)
        ttnn.deallocate(key_rotated, True)
        ttnn.deallocate(value, True)
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
            self.qkv_decode_weight,
            dtype=self.attention_activation_dtype,
            program_config=self.qkv_decode_program_config,
            compute_kernel_config=self.attention_compute_config,
            memory_config=self.qkv_output_memory_config,
        )
        ttnn.deallocate(qkv_input, True)
        padded_qkv = ttnn.sharded_to_interleaved(fused_qkv, ttnn.L1_MEMORY_CONFIG, ttnn.bfloat16)
        ttnn.deallocate(fused_qkv, True)
        if self.qkv_decode_padding:
            fused_qkv = ttnn.slice(
                padded_qkv,
                [0, 0, 0, 0],
                [1, 1, int(padded_qkv.shape[-2]), self.local_qkv_size],
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.deallocate(padded_qkv, True)
        else:
            fused_qkv = padded_qkv
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
            key = self._prepare_decode_heads(key, self.local_num_kv_heads, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            query = self._prepare_decode_heads(query, self.local_num_heads, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            key_rotated = self._apply_decode_rope_per_user(key, cos, sin)
            query_rotated = self._apply_decode_rope_per_user(query, cos, sin)
            ttnn.deallocate(key, True)
            ttnn.deallocate(query, True)
            key_rotated = self._move_owned(key_rotated, self.decode_head_memory_config)
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
        position_index: int,
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
            self.o_decode_weight,
            dtype=self.attention_activation_dtype,
            program_config=self.o_decode_program_config,
            compute_kernel_config=self.attention_compute_config,
            memory_config=self.o_output_memory_config,
        )
        ttnn.deallocate(projected_input, True)
        projected = self._move_owned(projected, self.residual_memory_config)
        projected = self._all_reduce_partial(projected, memory_config=self.residual_memory_config, ccl_role="attention")
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
        program = _prefill_matmul_program_config(
            self.mesh_device,
            m,
            self.hidden_size,
            self.local_intermediate_size,
            grid_x_limit=self.prefill_grid_x_limit,
            in0_block_w=8,
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
        if self.prefill_l1_inputs and m <= 32:
            dram_gated = gated
            gated = ttnn.to_memory_config(dram_gated, ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(dram_gated, True)
        down = ttnn.matmul(
            gated,
            self.down_weight,
            dtype=ttnn.bfloat16,
            program_config=_prefill_matmul_program_config(
                self.mesh_device,
                m,
                self.local_intermediate_size,
                self.hidden_size,
                grid_x_limit=self.prefill_grid_x_limit,
                in0_block_w=_largest_divisor(self.local_intermediate_size // 32, limit=8),
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
                dtype=self.mlp_activation_dtype,
                program_config=self.packed_gate_decode_program_config,
                compute_kernel_config=self.mlp_compute_config,
                memory_config=self.packed_mlp_output_memory_config,
            )
            ttnn.deallocate(mlp_input, True)
            packed_dram = ttnn.to_memory_config(packed, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(packed, True)
            padded_rows = int(packed_dram.shape[-2])
            gate = ttnn.slice(
                packed_dram,
                [0, 0, 0, 0],
                [1, 1, padded_rows, self.local_intermediate_decode_size],
            )
            up = ttnn.slice(
                packed_dram,
                [0, 0, 0, self.local_intermediate_decode_size],
                [1, 1, padded_rows, 2 * self.local_intermediate_decode_size],
            )
            ttnn.deallocate(packed_dram, True)
        else:
            gate = ttnn.matmul(
                mlp_input,
                self.gate_decode_weight,
                dtype=self.mlp_activation_dtype,
                program_config=self.gate_decode_program_config,
                compute_kernel_config=self.mlp_compute_config,
                memory_config=self.mlp_output_memory_config,
            )
            up = ttnn.matmul(
                mlp_input,
                self.up_decode_weight,
                dtype=self.mlp_activation_dtype,
                program_config=self.gate_decode_program_config,
                compute_kernel_config=self.mlp_compute_config,
                memory_config=self.mlp_output_memory_config,
            )
            ttnn.deallocate(mlp_input, True)
        gated = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=self.mlp_activation_dtype,
            memory_config=(ttnn.DRAM_MEMORY_CONFIG if self.use_packed_mlp else self.mlp_output_memory_config),
        )
        ttnn.deallocate(gate, True)
        ttnn.deallocate(up, True)
        down_input = self._move_owned(gated, self.down_input_memory_config)
        down = ttnn.matmul(
            down_input,
            self.down_decode_weight,
            dtype=self.mlp_activation_dtype,
            program_config=self.down_decode_program_config,
            compute_kernel_config=self.mlp_compute_config,
            memory_config=self.down_output_memory_config,
        )
        ttnn.deallocate(down_input, True)
        down = self._move_owned(down, self.residual_memory_config)
        down = self._all_reduce_partial(down, memory_config=self.residual_memory_config, ccl_role="mlp")
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
        batch, _ = self._validate_hidden(hidden_states, decode=True)
        self._validate_caches(key_cache, value_cache, page_table=page_table)
        self._validate_cache_position(cache_position)
        if rotary_position is None:
            rotary_position = cache_position
        elif tuple(int(v) for v in rotary_position.shape) != (self.batch,) or rotary_position.dtype not in (
            ttnn.int32,
            ttnn.uint32,
        ):
            raise ValueError(f"rotary_position must be int32/uint32[{self.batch}]")
        if not 0 <= position_index < self.max_cache_len:
            raise ValueError(f"position_index={position_index} is outside configured cache")
        residual = ttnn.reshape(hidden_states, (1, 1, batch, self.hidden_size))
        padded_rows = 32 * math.ceil(batch / 32)
        if batch not in (1, padded_rows):
            logical_residual = residual
            residual = ttnn.pad(
                logical_residual,
                padding=[(0, 0), (0, 0), (0, padded_rows - batch), (0, 0)],
                value=0.0,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(logical_residual, False)
        if residual.memory_config() != self.residual_memory_config:
            input_alias = residual
            residual = ttnn.to_memory_config(input_alias, self.residual_memory_config)
            ttnn.deallocate(input_alias, False)
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
        if int(residual.shape[-2]) != batch:
            padded = residual
            residual = ttnn.slice(
                padded,
                [0, 0, 0, 0],
                [1, 1, batch, self.hidden_size],
                memory_config=self.residual_memory_config,
            )
            ttnn.deallocate(padded, False)
        if self.return_sharded_decode_output:
            output = ttnn.reshape(residual, (1, batch, 1, self.hidden_size))
            ttnn.deallocate(residual, False)
            return output
        residual_interleaved = self._move_owned(residual, ttnn.L1_MEMORY_CONFIG)
        output_l1 = ttnn.reshape(residual_interleaved, (1, batch, 1, self.hidden_size))
        ttnn.deallocate(residual_interleaved, False)
        output = ttnn.to_memory_config(output_l1, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(output_l1, True)
        return output
