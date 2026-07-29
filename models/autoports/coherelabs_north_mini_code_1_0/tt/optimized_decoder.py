# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Single-device optimized decoder for CohereLabs/North-Mini-Code-1.0.

The public prefill, decode, paged-cache, and trace contracts are inherited from
``FunctionalDecoder``.  The measured math path is not inherited: this class
overrides weight materialization, normalization, attention, dense MLP, routed
MoE, prefill, and decode with explicitly configured optimized TTNN operations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import ttnn
from models.autoports.coherelabs_north_mini_code_1_0.tt.functional_decoder import (
    ADVERTISED_CONTEXT,
    DEFAULT_PAGE_SIZE,
    MODEL_ID,
    FunctionalDecoder,
    _load_expert_weights,
    _require_tensor,
    _rope_output_permutation,
)


@dataclass(frozen=True)
class OptimizationConfig:
    """Named cumulative optimization contract used by sweeps and final tests."""

    attention_weight_dtype: object = ttnn.bfloat8_b
    dense_gate_up_dtype: object = ttnn.bfloat4_b
    dense_down_dtype: object = ttnn.bfloat4_b
    prefill_dense_gate_up_dtype: object = ttnn.bfloat8_b
    prefill_dense_down_dtype: object = ttnn.bfloat8_b
    expert_gate_up_dtype: object = ttnn.bfloat8_b
    expert_down_dtype: object = ttnn.bfloat8_b
    dense_expert_gate_up_dtype: object = ttnn.bfloat8_b
    dense_expert_down_dtype: object = ttnn.bfloat8_b
    expert_activation_dtype: object = ttnn.bfloat16
    router_dtype: object = ttnn.bfloat16
    router_fidelity: object = ttnn.MathFidelity.HiFi2
    kv_cache_dtype: object = ttnn.bfloat16
    attention_fidelity: object = ttnn.MathFidelity.LoFi
    dense_gate_up_fidelity: object = ttnn.MathFidelity.LoFi
    dense_down_fidelity: object = ttnn.MathFidelity.LoFi
    prefill_dense_gate_up_fidelity: object = ttnn.MathFidelity.HiFi2
    prefill_dense_down_fidelity: object = ttnn.MathFidelity.HiFi2
    expert_gate_up_fidelity: object = ttnn.MathFidelity.LoFi
    expert_down_fidelity: object = ttnn.MathFidelity.LoFi
    decode_qkv_cores: int = 16
    decode_o_cores: int = 16
    decode_dense_gate_up_cores: int = 16
    decode_dense_down_cores: int = 16
    decode_qkv_in0_block_w: int = 4
    decode_o_in0_block_w: int = 8
    decode_dense_gate_up_in0_block_w: int = 4
    decode_dense_down_in0_block_w: int = 6
    serving_decode_qkv_cores: int = 16
    serving_decode_o_cores: int = 16
    serving_decode_dense_gate_up_cores: int = 16
    serving_decode_dense_down_cores: int = 16
    serving_decode_qkv_in0_block_w: int = 4
    serving_decode_o_in0_block_w: int = 8
    serving_decode_dense_gate_up_in0_block_w: int = 4
    serving_decode_dense_down_in0_block_w: int = 6
    decode_residual_cores: int = 16
    serving_decode_residual_cores: int = 16
    direct_o_input: bool = False
    direct_down_input: bool = False
    sparse_gate_up_grid: tuple[int, int] = (8, 3)
    sparse_down_grid: tuple[int, int] = (8, 8)
    sparse_gate_up_in0_block_w: int = 16
    sparse_down_in0_block_w: int = 12
    moe_chunk_size: int = 4
    sparse_intermediate_dram: bool = False
    dense_expert_batch_threshold: int = 32
    dense_expert_chunk_size: int = 1024
    dense_expert_cores: int = 100
    dense_expert_gate_up_in0_block_w: int = 0
    dense_expert_gate_up_per_core_m: int = 0
    dense_expert_gate_up_per_core_n: int = 0
    dense_expert_gate_up_subblock_h: int = 0
    dense_expert_gate_up_subblock_w: int = 0
    dense_expert_down_in0_block_w: int = 0
    dense_expert_down_per_core_m: int = 0
    dense_expert_down_per_core_n: int = 0
    dense_expert_down_subblock_h: int = 0
    dense_expert_down_subblock_w: int = 0
    packed_dense_experts: bool = False
    packed_dense_gate_up: bool = False
    fused_kv_update: bool = True
    serving_fused_kv_update: bool = False
    explicit_sdpa_program: bool = False


def _compute_config(fidelity, *, fp32_dest_acc=False):
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc,
        packer_l1_acc=True,
    )


def _rect_grid(num_cores: int, max_x: int = 11, max_y: int = 10) -> ttnn.CoreGrid:
    for y in range(min(max_y, num_cores), 0, -1):
        if num_cores % y == 0 and num_cores // y <= max_x:
            return ttnn.CoreGrid(x=num_cores // y, y=y)
    raise ValueError(f"cannot form a rectangular grid for {num_cores} cores")


def _dram_grid(mesh_device) -> tuple[ttnn.CoreRangeSet, int]:
    size = mesh_device.dram_grid_size()
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(size.x - 1, size.y - 1))})
    return grid, size.x * size.y


def _dram_weight_memory_config(mesh_device, k: int, n: int):
    grid, cores = _dram_grid(mesh_device)
    padded_n = math.ceil(n / (ttnn.TILE_SIZE * cores)) * ttnn.TILE_SIZE * cores
    shard = ttnn.ShardSpec(grid, (k, padded_n // cores), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard)


def _l1_width_memory_config(rows: int, width: int, cores: int):
    grid = _rect_grid(cores)
    return ttnn.create_sharded_memory_config(
        (rows, width // cores),
        grid,
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _decode_dram_program(m: int, n: int, cores: int, in0_block_w: int, fused_activation=None):
    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w,
        per_core_M=math.ceil(m / ttnn.TILE_SIZE),
        per_core_N=math.ceil(n / (ttnn.TILE_SIZE * cores)),
        fused_activation=fused_activation,
    )


def _prefill_program(m: int, k: int, n: int, fused_activation=None):
    grid_x, grid_y = (10, 10) if m >= 1024 else (8, 8)
    k_tiles = math.ceil(k / ttnn.TILE_SIZE)
    in0_block_w = max(divisor for divisor in range(1, k_tiles // grid_y + 1) if k_tiles % divisor == 0)
    per_core_n = math.ceil(n / (ttnn.TILE_SIZE * grid_x))
    out_subblock_w = next(divisor for divisor in (4, 3, 2, 1) if per_core_n % divisor == 0 and divisor <= 4)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        per_core_M=math.ceil(m / (ttnn.TILE_SIZE * grid_y)),
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=fused_activation,
        fuse_batch=True,
    )


def _prefill_physical_m(batch: int, seq_len: int) -> int:
    """Rows seen by fused-batch tiled matmuls after per-user sequence padding."""
    return batch * math.ceil(seq_len / ttnn.TILE_SIZE) * ttnn.TILE_SIZE


def _sparse_program(grid: tuple[int, int], n: int, in0_block_w: int, fused_activation=None):
    cores = grid[0] * grid[1]
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(*grid),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=1,
        out_block_h=1,
        out_block_w=1,
        per_core_M=1,
        per_core_N=math.ceil(n / (ttnn.TILE_SIZE * cores)),
        fuse_batch=False,
        fused_activation=fused_activation,
        mcast_in0=True,
    )


def _dense_expert_program(cfg: OptimizationConfig, *, down: bool):
    prefix = "dense_expert_down" if down else "dense_expert_gate_up"
    in0_block_w = getattr(cfg, f"{prefix}_in0_block_w")
    if not in0_block_w:
        return None
    grid = _rect_grid(cfg.dense_expert_cores)
    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
        in0_block_w=in0_block_w,
        out_subblock_h=getattr(cfg, f"{prefix}_subblock_h"),
        out_subblock_w=getattr(cfg, f"{prefix}_subblock_w"),
        per_core_M=getattr(cfg, f"{prefix}_per_core_m"),
        per_core_N=getattr(cfg, f"{prefix}_per_core_n"),
    )


def _to_device(
    source,
    mesh_device,
    *,
    dtype,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    layout=ttnn.TILE_LAYOUT,
):
    return ttnn.from_torch(
        source.contiguous(),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=dtype,
        layout=layout,
        memory_config=memory_config,
    )


class OptimizedDecoder(FunctionalDecoder):
    """Optimized North-Mini decoder with phase-specific on-device paths."""

    def __init__(self, *, optimization_config: OptimizationConfig, **kwargs):
        super().__init__(**kwargs)
        self.requested_optimization_config = optimization_config
        if self.batch == 32:
            optimization_config = replace(
                optimization_config,
                decode_qkv_cores=optimization_config.serving_decode_qkv_cores,
                decode_o_cores=optimization_config.serving_decode_o_cores,
                decode_dense_gate_up_cores=optimization_config.serving_decode_dense_gate_up_cores,
                decode_dense_down_cores=optimization_config.serving_decode_dense_down_cores,
                decode_qkv_in0_block_w=optimization_config.serving_decode_qkv_in0_block_w,
                decode_o_in0_block_w=optimization_config.serving_decode_o_in0_block_w,
                decode_dense_gate_up_in0_block_w=optimization_config.serving_decode_dense_gate_up_in0_block_w,
                decode_dense_down_in0_block_w=optimization_config.serving_decode_dense_down_in0_block_w,
                decode_residual_cores=optimization_config.serving_decode_residual_cores,
                fused_kv_update=optimization_config.serving_fused_kv_update,
            )
        self.optimization_config = optimization_config
        cfg = optimization_config
        padded_rows = ttnn.TILE_SIZE
        storage = self.mesh_device.compute_with_storage_grid_size()

        # ``nlp_create_qkv_heads_decode`` and decode RoPE assign one batch lane
        # to each core in row-wise sub-core order.  A mathematically equivalent
        # rectangular 32-core set changes that ordering on Blackhole and
        # corrupts lanes >= 8, so preserve the producer's exact core sequence.
        decode_rope_grid = ttnn.num_cores_to_corerangeset(
            min(self.batch, storage.x * storage.y),
            storage,
            row_wise=True,
        )
        self.decode_rope_memory_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, self.head_dim),
            core_grid=decode_rope_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        residual_grid = (
            ttnn.CoreGrid(x=cfg.decode_residual_cores, y=1)
            if cfg.decode_residual_cores <= 11
            else _rect_grid(cfg.decode_residual_cores)
        )
        self.residual_memory_config = ttnn.create_sharded_memory_config(
            (padded_rows, self.hidden_size // cfg.decode_residual_cores),
            residual_grid,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        block_w = self.hidden_size // cfg.decode_residual_cores // ttnn.TILE_SIZE
        subblock_w = next(divisor for divisor in (4, 3, 2, 1) if block_w % divisor == 0)
        self.decode_norm_program = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[residual_grid.x, residual_grid.y],
            subblock_w=subblock_w,
            block_h=1,
            block_w=block_w,
            inplace=False,
        )
        self.decode_sdpa_program = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            exp_approx_mode=False,
            q_chunk_size=0,
            k_chunk_size=0,
        )
        all_workers = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(storage.x - 1, storage.y - 1),
                )
            }
        )
        k_grid = ttnn.num_cores_to_corerangeset_in_subcoregrids(
            ttnn.CoreCoord(0, 0), self.batch, all_workers, row_wise=True
        )
        v_start = ttnn.CoreCoord(self.batch % storage.x, self.batch // storage.x)
        v_grid = ttnn.num_cores_to_corerangeset_in_subcoregrids(v_start, self.batch, all_workers, row_wise=True)
        self.decode_fused_k_memory_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, self.head_dim),
            core_grid=k_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.decode_fused_v_memory_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, self.head_dim),
            core_grid=v_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.attention_compute = _compute_config(cfg.attention_fidelity)
        self.dense_gate_up_compute = _compute_config(cfg.dense_gate_up_fidelity)
        self.dense_down_compute = _compute_config(cfg.dense_down_fidelity)
        self.prefill_dense_gate_up_compute = _compute_config(cfg.prefill_dense_gate_up_fidelity)
        self.prefill_dense_down_compute = _compute_config(cfg.prefill_dense_down_fidelity)
        self.expert_gate_up_compute = _compute_config(cfg.expert_gate_up_fidelity)
        self.expert_down_compute = _compute_config(cfg.expert_down_fidelity)
        self.router_compute = _compute_config(cfg.router_fidelity, fp32_dest_acc=True)

    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx,
        mesh_device,
        batch=1,
        max_cache_len=ADVERTISED_CONTEXT,
        page_size=DEFAULT_PAGE_SIZE,
        optimization_config: OptimizationConfig | None = None,
        **_kwargs,
    ):
        import torch

        cfg = optimization_config or OptimizationConfig()
        if not isinstance(mesh_device, ttnn.MeshDevice) or tuple(mesh_device.shape) != (1, 1):
            raise ValueError("OptimizedDecoder requires a single-device 1x1 MeshDevice")
        if batch < 1 or batch > 32:
            raise ValueError(f"optimized decode batch must be in [1, 32], got {batch}")
        if not 1 <= max_cache_len <= int(hf_config.max_position_embeddings):
            raise ValueError("max_cache_len is outside the model context contract")
        if page_size < ttnn.TILE_SIZE or page_size % ttnn.TILE_SIZE:
            raise ValueError("page_size must be a positive tile multiple")

        hidden_size = int(hf_config.hidden_size)
        num_heads = int(hf_config.num_attention_heads)
        num_kv_heads = int(hf_config.num_key_value_heads)
        head_dim = int(hf_config.head_dim)
        q = _require_tensor(state_dict, layer_idx, "self_attn.q_proj.weight")
        k = _require_tensor(state_dict, layer_idx, "self_attn.k_proj.weight")
        v = _require_tensor(state_dict, layer_idx, "self_attn.v_proj.weight")
        o = _require_tensor(state_dict, layer_idx, "self_attn.o_proj.weight")
        norm = _require_tensor(state_dict, layer_idx, "input_layernorm.weight")
        q = q.index_select(0, _rope_output_permutation(num_heads, head_dim))
        k = k.index_select(0, _rope_output_permutation(num_kv_heads, head_dim))
        qkv = torch.cat((q, k, v), dim=0).transpose(-2, -1)

        weights = {
            "qkv": _to_device(
                qkv,
                mesh_device,
                dtype=cfg.attention_weight_dtype,
                memory_config=_dram_weight_memory_config(mesh_device, hidden_size, qkv.shape[-1]),
            ),
            "o": _to_device(
                o.transpose(-2, -1),
                mesh_device,
                dtype=cfg.attention_weight_dtype,
                memory_config=_dram_weight_memory_config(mesh_device, o.shape[-1], o.shape[-2]),
            ),
            "norm": _to_device(
                norm.reshape(1, 1, 1, hidden_size),
                mesh_device,
                dtype=ttnn.bfloat16,
            ),
        }

        mlp_type = hf_config.mlp_layer_types[layer_idx]
        if mlp_type == "dense":
            gate = _require_tensor(state_dict, layer_idx, "mlp.gate_proj.weight").transpose(-2, -1)
            up = _require_tensor(state_dict, layer_idx, "mlp.up_proj.weight").transpose(-2, -1)
            down = _require_tensor(state_dict, layer_idx, "mlp.down_proj.weight").transpose(-2, -1)
            if cfg.packed_dense_gate_up:
                packed = torch.cat((gate, up), dim=-1)
                weights["gate_up"] = _to_device(
                    packed,
                    mesh_device,
                    dtype=cfg.dense_gate_up_dtype,
                    memory_config=_dram_weight_memory_config(mesh_device, hidden_size, packed.shape[-1]),
                )
            else:
                for name, source in (("gate_proj", gate), ("up_proj", up)):
                    weights[name] = _to_device(
                        source,
                        mesh_device,
                        dtype=cfg.dense_gate_up_dtype,
                        memory_config=_dram_weight_memory_config(mesh_device, hidden_size, source.shape[-1]),
                    )
            weights["down_proj"] = _to_device(
                down,
                mesh_device,
                dtype=cfg.dense_down_dtype,
                memory_config=_dram_weight_memory_config(mesh_device, down.shape[-2], hidden_size),
            )
            weights["prefill_gate_proj"] = _to_device(
                gate,
                mesh_device,
                dtype=cfg.prefill_dense_gate_up_dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            weights["prefill_up_proj"] = _to_device(
                up,
                mesh_device,
                dtype=cfg.prefill_dense_gate_up_dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            weights["prefill_down_proj"] = _to_device(
                down,
                mesh_device,
                dtype=cfg.prefill_dense_down_dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        elif mlp_type == "sparse":
            gate, up, down = _load_expert_weights(
                state_dict, layer_idx, int(hf_config.num_experts), int(hf_config.intermediate_size)
            )
            weights["router"] = _to_device(
                _require_tensor(state_dict, layer_idx, "mlp.gate.weight").transpose(-2, -1),
                mesh_device,
                dtype=cfg.router_dtype,
            )
            weights["expert_gate"] = _to_device(gate, mesh_device, dtype=cfg.expert_gate_up_dtype)
            weights["expert_up"] = _to_device(up, mesh_device, dtype=cfg.expert_gate_up_dtype)
            weights["expert_down"] = _to_device(down, mesh_device, dtype=cfg.expert_down_dtype)
            if cfg.dense_expert_gate_up_dtype == cfg.expert_gate_up_dtype:
                weights["dense_expert_gate"] = weights["expert_gate"]
                weights["dense_expert_up"] = weights["expert_up"]
            else:
                weights["dense_expert_gate"] = _to_device(gate, mesh_device, dtype=cfg.dense_expert_gate_up_dtype)
                weights["dense_expert_up"] = _to_device(up, mesh_device, dtype=cfg.dense_expert_gate_up_dtype)
            if cfg.dense_expert_down_dtype == cfg.expert_down_dtype:
                weights["dense_expert_down"] = weights["expert_down"]
            else:
                weights["dense_expert_down"] = _to_device(down, mesh_device, dtype=cfg.dense_expert_down_dtype)
            if cfg.packed_dense_experts:
                weights["dense_expert_gate_up"] = _to_device(
                    torch.cat((gate, up), dim=-1),
                    mesh_device,
                    dtype=cfg.dense_expert_gate_up_dtype,
                )
        else:
            raise ValueError(f"unsupported North-Mini MLP layer kind {mlp_type!r}")

        return cls(
            hf_config=hf_config,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            batch=batch,
            max_cache_len=max_cache_len,
            page_size=page_size,
            weights=weights,
            optimization_config=cfg,
        )

    def create_paged_kv_cache(self, *, num_blocks: int | None = None):
        min_blocks = self.batch * math.ceil(self.max_cache_len / self.page_size)
        num_blocks = min_blocks if num_blocks is None else int(num_blocks)
        if num_blocks < min_blocks:
            raise ValueError(f"num_blocks={num_blocks} cannot cover required {min_blocks} blocks")
        shape = (num_blocks, self.num_kv_heads, self.page_size, self.head_dim)
        kwargs = dict(
            dtype=self.optimization_config.kv_cache_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return ttnn.zeros(shape, **kwargs), ttnn.zeros(shape, **kwargs)

    def _normalize_decode(self, hidden_states):
        x = ttnn.reshape(hidden_states, (1, 1, self.batch, self.hidden_size))
        x = ttnn.to_memory_config(x, self.residual_memory_config)
        normalized = ttnn.rms_norm(
            x,
            epsilon=self.eps,
            weight=self.weights["norm"],
            program_config=self.decode_norm_program,
            memory_config=self.residual_memory_config,
            compute_kernel_config=self.attention_compute,
        )
        return x, normalized

    def _qkv_prefill(
        self,
        normalized,
        seq_len,
        position_cos,
        position_sin,
        *,
        batch_size=None,
        apply_rope=True,
    ):
        batch = batch_size or self.batch
        fused = ttnn.linear(
            normalized,
            self.weights["qkv"],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=_prefill_program(
                _prefill_physical_m(batch, seq_len),
                self.hidden_size,
                self.num_heads * self.head_dim + 2 * self.num_kv_heads * self.head_dim,
            ),
            compute_kernel_config=self.attention_compute,
        )
        fused = ttnn.reshape(fused, (batch, seq_len, -1))
        query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
            fused,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            transpose_key=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if self.use_rope and apply_rope:
            query = ttnn.experimental.rotary_embedding(
                query, position_cos, position_sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            key = ttnn.experimental.rotary_embedding(
                key, position_cos, position_sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            query = ttnn.slice(query, (0, 0, 0, 0), (batch, self.num_heads, seq_len, self.head_dim))
            key = ttnn.slice(key, (0, 0, 0, 0), (batch, self.num_kv_heads, seq_len, self.head_dim))
        return query, key, value

    def _attention_prefill(
        self,
        normalized,
        *,
        key_cache,
        value_cache,
        page_table,
        position_cos,
        position_sin,
        seq_len,
    ):
        if self.batch > 1 and seq_len % ttnn.TILE_SIZE:
            return self._attention_prefill_token_packed(
                normalized,
                key_cache=key_cache,
                value_cache=value_cache,
                page_table=page_table,
                position_cos=position_cos,
                position_sin=position_sin,
                seq_len=seq_len,
            )
        query, key, value = self._qkv_prefill(normalized, seq_len, position_cos, position_sin)
        cache_dtype = self.optimization_config.kv_cache_dtype
        key_fill = ttnn.typecast(key, cache_dtype) if cache_dtype != ttnn.bfloat16 else key
        value_fill = ttnn.typecast(value, cache_dtype) if cache_dtype != ttnn.bfloat16 else value
        for user in range(self.batch):
            key_user = ttnn.slice(key_fill, (user, 0, 0, 0), (user + 1, self.num_kv_heads, seq_len, self.head_dim))
            value_user = ttnn.slice(value_fill, (user, 0, 0, 0), (user + 1, self.num_kv_heads, seq_len, self.head_dim))
            ttnn.experimental.paged_fill_cache(key_cache, key_user, page_table, batch_idx=user)
            ttnn.experimental.paged_fill_cache(value_cache, value_user, page_table, batch_idx=user)
        attended = ttnn.transformer.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
            scale=self.scale,
            sliding_window_size=self.sliding_window,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attended = ttnn.transformer.concatenate_heads(attended, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        projected = ttnn.linear(
            attended,
            self.weights["o"],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=_prefill_program(
                _prefill_physical_m(self.batch, seq_len),
                self.num_heads * self.head_dim,
                self.hidden_size,
            ),
            compute_kernel_config=self.attention_compute,
        )
        if self.batch > 1:
            projected = ttnn.reshape(projected, (1, self.batch, seq_len, self.hidden_size))
        return projected

    def _attention_prefill_token_packed(
        self,
        normalized,
        *,
        key_cache,
        value_cache,
        page_table,
        position_cos,
        position_sin,
        seq_len,
    ):
        """Pack users as tokens for matmuls, then restore independent SDPA batches."""
        total_tokens = self.batch * seq_len
        cache_dtype = self.optimization_config.kv_cache_dtype
        packed = ttnn.to_layout(normalized, ttnn.ROW_MAJOR_LAYOUT)
        packed = ttnn.reshape(packed, (1, 1, total_tokens, self.hidden_size))
        packed = ttnn.to_layout(packed, ttnn.TILE_LAYOUT)
        # The large-prefill 10x10 program produces non-finite values for this
        # packed non-aligned shape at serving batch. Keep the public logical
        # sequence unrestricted and use the already-correct <=512-row program.
        qkv_chunks = []
        for start in range(0, total_tokens, 512):
            end = min(start + 512, total_tokens)
            chunk = ttnn.slice(packed, (0, 0, start, 0), (1, 1, end, self.hidden_size))
            qkv_chunks.append(
                self._qkv_prefill(
                    chunk,
                    end - start,
                    None,
                    None,
                    batch_size=1,
                    apply_rope=False,
                )
            )
        query, key, value = tuple(
            chunks[0] if len(chunks) == 1 else ttnn.concat(chunks, dim=2) for chunks in zip(*qkv_chunks)
        )

        def tokens_to_users(tensor, heads):
            tensor = ttnn.to_layout(tensor, ttnn.ROW_MAJOR_LAYOUT)
            tensor = ttnn.permute(tensor, (0, 2, 1, 3))
            tensor = ttnn.reshape(tensor, (self.batch, seq_len, heads, self.head_dim))
            tensor = ttnn.permute(tensor, (0, 2, 1, 3))
            return ttnn.to_layout(tensor, ttnn.TILE_LAYOUT)

        query = tokens_to_users(query, self.num_heads)
        key = tokens_to_users(key, self.num_kv_heads)
        value = tokens_to_users(value, self.num_kv_heads)
        if self.use_rope:
            query = ttnn.experimental.rotary_embedding(
                query,
                position_cos,
                position_sin,
                None,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            key = ttnn.experimental.rotary_embedding(
                key,
                position_cos,
                position_sin,
                None,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            query = ttnn.slice(
                query,
                (0, 0, 0, 0),
                (self.batch, self.num_heads, seq_len, self.head_dim),
            )
            key = ttnn.slice(
                key,
                (0, 0, 0, 0),
                (self.batch, self.num_kv_heads, seq_len, self.head_dim),
            )
        key_fill = ttnn.typecast(key, cache_dtype) if cache_dtype != ttnn.bfloat16 else key
        value_fill = ttnn.typecast(value, cache_dtype) if cache_dtype != ttnn.bfloat16 else value
        for user in range(self.batch):
            key_user = ttnn.slice(
                key_fill,
                (user, 0, 0, 0),
                (user + 1, self.num_kv_heads, seq_len, self.head_dim),
            )
            value_user = ttnn.slice(
                value_fill,
                (user, 0, 0, 0),
                (user + 1, self.num_kv_heads, seq_len, self.head_dim),
            )
            ttnn.experimental.paged_fill_cache(key_cache, key_user, page_table, batch_idx=user)
            ttnn.experimental.paged_fill_cache(value_cache, value_user, page_table, batch_idx=user)
        attended = ttnn.transformer.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
            scale=self.scale,
            sliding_window_size=self.sliding_window,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attended = ttnn.transformer.concatenate_heads(attended, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attended = ttnn.to_layout(attended, ttnn.ROW_MAJOR_LAYOUT)
        attended = ttnn.reshape(attended, (1, 1, total_tokens, self.num_heads * self.head_dim))
        attended = ttnn.to_layout(attended, ttnn.TILE_LAYOUT)
        projected_chunks = []
        for start in range(0, total_tokens, 512):
            end = min(start + 512, total_tokens)
            chunk = ttnn.slice(
                attended,
                (0, 0, start, 0),
                (1, 1, end, self.num_heads * self.head_dim),
            )
            projected_chunks.append(
                ttnn.linear(
                    chunk,
                    self.weights["o"],
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    program_config=_prefill_program(
                        end - start,
                        self.num_heads * self.head_dim,
                        self.hidden_size,
                    ),
                    compute_kernel_config=self.attention_compute,
                )
            )
        projected = projected_chunks[0] if len(projected_chunks) == 1 else ttnn.concat(projected_chunks, dim=2)
        projected = ttnn.to_layout(projected, ttnn.ROW_MAJOR_LAYOUT)
        projected = ttnn.reshape(projected, (1, self.batch, seq_len, self.hidden_size))
        return ttnn.to_layout(projected, ttnn.TILE_LAYOUT)

    def _qkv_decode(self, normalized, position_cos, position_sin):
        cfg = self.optimization_config
        qkv_width = self.num_heads * self.head_dim + 2 * self.num_kv_heads * self.head_dim
        fused_sharded = ttnn.linear(
            normalized,
            self.weights["qkv"],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=_decode_dram_program(
                ttnn.TILE_SIZE,
                qkv_width,
                cfg.decode_qkv_cores,
                cfg.decode_qkv_in0_block_w,
            ),
            compute_kernel_config=self.attention_compute,
        )
        fused = ttnn.sharded_to_interleaved(fused_sharded, ttnn.L1_MEMORY_CONFIG, ttnn.bfloat16)
        fused = ttnn.reshape(fused, (1, 1, self.batch, qkv_width))
        query, key, value = ttnn.experimental.nlp_create_qkv_heads_decode(
            fused,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            overlap_qk_coregrid=not self.optimization_config.fused_kv_update,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )
        if self.use_rope:
            position_cos = ttnn.interleaved_to_sharded(position_cos, self.decode_rope_memory_config)
            position_sin = ttnn.interleaved_to_sharded(position_sin, self.decode_rope_memory_config)
            query = ttnn.experimental.rotary_embedding_hf(
                query,
                position_cos,
                position_sin,
                is_decode_mode=True,
                memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
            )
            key = ttnn.experimental.rotary_embedding_hf(
                key,
                position_cos,
                position_sin,
                is_decode_mode=True,
                memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
            )
        return query, key, value

    def _attention_decode(
        self,
        normalized,
        *,
        key_cache,
        value_cache,
        page_table,
        current_positions,
        position_cos,
        position_sin,
    ):
        cfg = self.optimization_config
        query, key, value = self._qkv_decode(normalized, position_cos, position_sin)
        if cfg.fused_kv_update:
            key = ttnn.to_memory_config(key, self.decode_fused_k_memory_config)
            value = ttnn.to_memory_config(value, self.decode_fused_v_memory_config)
            ttnn.experimental.paged_fused_update_cache(
                key_cache,
                key,
                value_cache,
                value,
                update_idxs_tensor=current_positions,
                page_table=page_table,
            )
        else:
            ttnn.experimental.paged_update_cache(
                key_cache, key, update_idxs_tensor=current_positions, page_table=page_table
            )
            ttnn.experimental.paged_update_cache(
                value_cache, value, update_idxs_tensor=current_positions, page_table=page_table
            )
        attended = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            query,
            key_cache,
            value_cache,
            page_table_tensor=page_table,
            cur_pos_tensor=current_positions,
            scale=self.scale,
            sliding_window_size=self.sliding_window,
            program_config=self.decode_sdpa_program if cfg.explicit_sdpa_program else None,
            compute_kernel_config=self.attention_compute,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attended = ttnn.to_memory_config(attended, self.decode_concat_memory_config)
        attended = ttnn.experimental.nlp_concat_heads_decode(
            attended,
            num_heads=self.num_heads,
            sub_core_grids=self.decode_sub_core_grids,
        )
        if not cfg.direct_o_input:
            attended = ttnn.to_memory_config(
                attended,
                _l1_width_memory_config(ttnn.TILE_SIZE, self.num_heads * self.head_dim, cfg.decode_o_cores),
            )
        projected = ttnn.linear(
            attended,
            self.weights["o"],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=_decode_dram_program(
                ttnn.TILE_SIZE,
                self.hidden_size,
                cfg.decode_o_cores,
                cfg.decode_o_in0_block_w,
            ),
            compute_kernel_config=self.attention_compute,
        )
        projected = ttnn.slice(projected, (0, 0, 0, 0), (1, 1, self.batch, self.hidden_size))
        return ttnn.to_memory_config(projected, self.residual_memory_config)

    def _dense_mlp_decode(self, normalized):
        cfg = self.optimization_config
        if cfg.packed_dense_gate_up:
            packed = ttnn.linear(
                normalized,
                self.weights["gate_up"],
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                program_config=_decode_dram_program(
                    ttnn.TILE_SIZE,
                    2 * self.intermediate_size,
                    cfg.decode_dense_gate_up_cores,
                    cfg.decode_dense_gate_up_in0_block_w,
                ),
                compute_kernel_config=self.dense_gate_up_compute,
            )
            gate, up = ttnn.split(packed, self.intermediate_size, dim=-1)
        else:
            gate = ttnn.linear(
                normalized,
                self.weights["gate_proj"],
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                program_config=_decode_dram_program(
                    ttnn.TILE_SIZE,
                    self.intermediate_size,
                    cfg.decode_dense_gate_up_cores,
                    cfg.decode_dense_gate_up_in0_block_w,
                ),
                compute_kernel_config=self.dense_gate_up_compute,
            )
            up = ttnn.linear(
                normalized,
                self.weights["up_proj"],
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                program_config=_decode_dram_program(
                    ttnn.TILE_SIZE,
                    self.intermediate_size,
                    cfg.decode_dense_gate_up_cores,
                    cfg.decode_dense_gate_up_in0_block_w,
                ),
                compute_kernel_config=self.dense_gate_up_compute,
            )
        activated = ttnn.multiply(ttnn.silu(gate), up)
        if not cfg.direct_down_input:
            activated = ttnn.to_memory_config(
                activated,
                _l1_width_memory_config(ttnn.TILE_SIZE, self.intermediate_size, cfg.decode_dense_down_cores),
            )
        output = ttnn.linear(
            activated,
            self.weights["down_proj"],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=_decode_dram_program(
                ttnn.TILE_SIZE,
                self.hidden_size,
                cfg.decode_dense_down_cores,
                cfg.decode_dense_down_in0_block_w,
            ),
            compute_kernel_config=self.dense_down_compute,
        )
        return ttnn.to_memory_config(output, self.residual_memory_config)

    def _dense_mlp_prefill(self, normalized, seq_len):
        cfg = self.optimization_config
        m = _prefill_physical_m(self.batch, seq_len)
        gate = ttnn.linear(
            normalized,
            self.weights["prefill_gate_proj"],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=_prefill_program(m, self.hidden_size, self.intermediate_size),
            compute_kernel_config=self.prefill_dense_gate_up_compute,
        )
        up = ttnn.linear(
            normalized,
            self.weights["prefill_up_proj"],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=_prefill_program(m, self.hidden_size, self.intermediate_size),
            compute_kernel_config=self.prefill_dense_gate_up_compute,
        )
        return ttnn.linear(
            ttnn.multiply(ttnn.silu(gate), up),
            self.weights["prefill_down_proj"],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=_prefill_program(m, self.intermediate_size, self.hidden_size),
            compute_kernel_config=self.prefill_dense_down_compute,
        )

    def _sparse_moe_chunk(self, normalized, token_count):
        cfg = self.optimization_config
        projection_memory = ttnn.DRAM_MEMORY_CONFIG if cfg.sparse_intermediate_dram else ttnn.L1_MEMORY_CONFIG
        activation_memory = projection_memory
        down_memory = projection_memory
        flat = ttnn.reshape(normalized, (1, token_count, 1, self.hidden_size))
        logits = ttnn.linear(
            flat,
            self.weights["router"],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            core_grid=_rect_grid(4),
            compute_kernel_config=self.router_compute,
        )
        top_values, top_indices = ttnn.topk(logits, k=self.top_k, dim=-1, sorted=True)
        top_values = ttnn.sigmoid(top_values)
        routing = ttnn.scatter(ttnn.zeros_like(logits), dim=-1, index=top_indices, src=top_values)
        sparsity = ttnn.to_layout(routing, ttnn.ROW_MAJOR_LAYOUT)
        expert_gate = ttnn.reshape(
            self.weights["expert_gate"],
            (1, self.num_experts, self.hidden_size, self.intermediate_size),
        )
        expert_up = ttnn.reshape(
            self.weights["expert_up"],
            (1, self.num_experts, self.hidden_size, self.intermediate_size),
        )
        expert_down = ttnn.reshape(
            self.weights["expert_down"],
            (1, self.num_experts, self.intermediate_size, self.hidden_size),
        )

        gate = ttnn.sparse_matmul(
            flat,
            expert_gate,
            sparsity=sparsity,
            nnz=None,
            memory_config=projection_memory,
            program_config=_sparse_program(
                cfg.sparse_gate_up_grid,
                self.intermediate_size,
                cfg.sparse_gate_up_in0_block_w,
            ),
            compute_kernel_config=self.expert_gate_up_compute,
            dtype=cfg.expert_activation_dtype,
        )
        up = ttnn.sparse_matmul(
            flat,
            expert_up,
            sparsity=sparsity,
            nnz=None,
            memory_config=projection_memory,
            program_config=_sparse_program(
                cfg.sparse_gate_up_grid, self.intermediate_size, cfg.sparse_gate_up_in0_block_w
            ),
            compute_kernel_config=self.expert_gate_up_compute,
            dtype=cfg.expert_activation_dtype,
        )
        gate = ttnn.reshape(gate, (token_count, self.num_experts, 1, self.intermediate_size))
        up = ttnn.reshape(up, (token_count, self.num_experts, 1, self.intermediate_size))
        gate = ttnn.silu(gate, memory_config=activation_memory)
        activated = ttnn.multiply(gate, up, dtype=cfg.expert_activation_dtype, memory_config=activation_memory)
        down = ttnn.sparse_matmul(
            activated,
            expert_down,
            sparsity=sparsity,
            nnz=None,
            is_input_a_sparse=True,
            is_input_b_sparse=False,
            memory_config=down_memory,
            program_config=_sparse_program(cfg.sparse_down_grid, self.hidden_size, cfg.sparse_down_in0_block_w),
            compute_kernel_config=self.expert_down_compute,
            dtype=cfg.expert_activation_dtype,
        )
        routing = ttnn.reshape(routing, (token_count, self.num_experts, 1, 1))
        down = ttnn.multiply(down, routing, dtype=cfg.expert_activation_dtype, memory_config=down_memory)
        reduced = ttnn.experimental.fast_reduce_nc(down, dims=[1], memory_config=down_memory)
        # Keep tokens on the tiled H axis. Concatenating sub-tile token counts
        # on N would insert each chunk's physical padding between logical rows.
        reduced = ttnn.slice(reduced, (0, 0, 0, 0), (token_count, 1, 1, self.hidden_size))
        return ttnn.reshape(reduced, (1, 1, token_count, self.hidden_size))

    def _dense_expert_moe_chunk(self, normalized, token_count):
        """Batched expert path used where sparse output padding is counterproductive."""
        cfg = self.optimization_config
        expert_grid_kwargs = (
            {"core_grid": _rect_grid(cfg.dense_expert_cores)}
            if cfg.dense_expert_cores and not cfg.dense_expert_gate_up_in0_block_w
            else {}
        )
        gate_program = _dense_expert_program(cfg, down=False)
        down_program = _dense_expert_program(cfg, down=True)
        flat = ttnn.reshape(normalized, (token_count, self.hidden_size))
        logits = ttnn.linear(
            flat,
            self.weights["router"],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            core_grid=_rect_grid(8),
            compute_kernel_config=self.router_compute,
        )
        top_values, top_indices = ttnn.topk(logits, k=self.top_k, dim=-1, sorted=True)
        top_values = ttnn.sigmoid(top_values)
        routing = ttnn.scatter(ttnn.zeros_like(logits), dim=-1, index=top_indices, src=top_values)

        expert_input = ttnn.reshape(flat, (1, token_count, self.hidden_size))
        expert_input = ttnn.repeat(expert_input, ttnn.Shape((self.num_experts, 1, 1)))
        expert_gate = ttnn.reshape(
            self.weights["dense_expert_gate"],
            (self.num_experts, self.hidden_size, self.intermediate_size),
        )
        expert_up = ttnn.reshape(
            self.weights["dense_expert_up"],
            (self.num_experts, self.hidden_size, self.intermediate_size),
        )
        expert_down = ttnn.reshape(
            self.weights["dense_expert_down"],
            (self.num_experts, self.intermediate_size, self.hidden_size),
        )
        if cfg.packed_dense_experts:
            packed = ttnn.matmul(
                expert_input,
                ttnn.reshape(
                    self.weights["dense_expert_gate_up"],
                    (self.num_experts, self.hidden_size, 2 * self.intermediate_size),
                ),
                dtype=cfg.expert_activation_dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.expert_gate_up_compute,
                program_config=gate_program,
                **expert_grid_kwargs,
            )
            gate, up = ttnn.split(packed, self.intermediate_size, dim=-1)
        else:
            gate = ttnn.linear(
                expert_input,
                expert_gate,
                dtype=cfg.expert_activation_dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.expert_gate_up_compute,
                program_config=gate_program,
                **expert_grid_kwargs,
            )
            up = ttnn.matmul(
                expert_input,
                expert_up,
                dtype=cfg.expert_activation_dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.expert_gate_up_compute,
                program_config=gate_program,
                **expert_grid_kwargs,
            )
        gate = ttnn.silu(gate, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        activated = ttnn.multiply(
            gate,
            up,
            dtype=self.optimization_config.expert_activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        expert_output = ttnn.matmul(
            activated,
            expert_down,
            dtype=self.optimization_config.expert_activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.expert_down_compute,
            program_config=down_program,
            **({} if down_program else expert_grid_kwargs),
        )
        routing = ttnn.permute(routing, (1, 0))
        routing = ttnn.reshape(routing, (self.num_experts, token_count, 1))
        expert_output = ttnn.multiply(
            expert_output,
            routing,
            dtype=self.optimization_config.expert_activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return ttnn.sum(
            expert_output,
            dim=0,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.expert_down_compute,
        )

    def _sparse_moe(self, normalized, seq_len):
        total_tokens = self.batch * seq_len
        cfg = self.optimization_config
        pack_multi_user = self.batch > 1 and seq_len % ttnn.TILE_SIZE

        def restore_batch_shape(result):
            if pack_multi_user:
                result = ttnn.to_layout(result, ttnn.ROW_MAJOR_LAYOUT)
                result = ttnn.reshape(result, (1, self.batch, seq_len, self.hidden_size))
                return ttnn.to_layout(result, ttnn.TILE_LAYOUT)
            return ttnn.reshape(result, (1, self.batch, seq_len, self.hidden_size))

        if pack_multi_user:
            # A view cannot discard each batch row's physical sequence pad.
            # Pack logical tokens on-device before flattening non-aligned
            # multi-user prefill inputs.
            flat = ttnn.to_layout(normalized, ttnn.ROW_MAJOR_LAYOUT)
            flat = ttnn.reshape(flat, (1, 1, total_tokens, self.hidden_size))
            flat = ttnn.to_layout(flat, ttnn.TILE_LAYOUT)
        else:
            flat = ttnn.reshape(normalized, (1, 1, total_tokens, self.hidden_size))
        if total_tokens >= cfg.dense_expert_batch_threshold:
            chunks = ttnn.split(flat, cfg.dense_expert_chunk_size, dim=2)
            outputs = [self._dense_expert_moe_chunk(chunk, chunk.shape[2]) for chunk in chunks]
            result = outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=0)
            return restore_batch_shape(result)
        chunk_size = cfg.moe_chunk_size
        if total_tokens <= chunk_size:
            result = self._sparse_moe_chunk(flat, total_tokens)
        else:
            outputs = []
            for start in range(0, total_tokens, chunk_size):
                end = min(start + chunk_size, total_tokens)
                logical_chunk = end - start
                chunk = ttnn.slice(flat, (0, 0, start, 0), (1, 1, end, self.hidden_size))
                if logical_chunk < chunk_size:
                    chunk = ttnn.pad(
                        chunk,
                        padding=((0, 0), (0, 0), (0, chunk_size - logical_chunk), (0, 0)),
                        value=0.0,
                    )
                output = self._sparse_moe_chunk(chunk, chunk_size)
                if logical_chunk < chunk_size:
                    output = ttnn.slice(output, (0, 0, 0, 0), (1, 1, logical_chunk, self.hidden_size))
                outputs.append(output)
            result = ttnn.concat(outputs, dim=2)
        return restore_batch_shape(result)

    def prefill_forward(
        self,
        hidden_states,
        *,
        key_cache,
        value_cache,
        page_table,
        position_cos=None,
        position_sin=None,
    ):
        seq_len = self._validate_hidden(hidden_states, decode=False)
        if self.use_rope and (position_cos is None or position_sin is None):
            raise ValueError("this layer kind requires position_cos and position_sin")
        normalized = ttnn.rms_norm(
            hidden_states,
            epsilon=self.eps,
            weight=self.weights["norm"],
            compute_kernel_config=self.attention_compute,
        )
        attention = self._attention_prefill(
            normalized,
            key_cache=key_cache,
            value_cache=value_cache,
            page_table=page_table,
            position_cos=position_cos,
            position_sin=position_sin,
            seq_len=seq_len,
        )
        mlp = (
            self._dense_mlp_prefill(normalized, seq_len)
            if self.mlp_type == "dense"
            else self._sparse_moe(normalized, seq_len)
        )
        return ttnn.add(ttnn.add(hidden_states, attention), mlp)

    def decode_forward(
        self,
        hidden_states,
        *,
        key_cache,
        value_cache,
        page_table,
        current_positions,
        position_cos=None,
        position_sin=None,
    ):
        self._validate_hidden(hidden_states, decode=True)
        if tuple(current_positions.shape) != (self.batch,):
            raise ValueError(f"current_positions must have shape ({self.batch},)")
        if self.use_rope and (position_cos is None or position_sin is None):
            raise ValueError("this layer kind requires position_cos and position_sin")
        residual, normalized = self._normalize_decode(hidden_states)
        attention = self._attention_decode(
            normalized,
            key_cache=key_cache,
            value_cache=value_cache,
            page_table=page_table,
            current_positions=current_positions,
            position_cos=position_cos,
            position_sin=position_sin,
        )
        if self.mlp_type == "dense":
            mlp = self._dense_mlp_decode(normalized)
        else:
            normalized_interleaved = ttnn.sharded_to_interleaved(normalized, ttnn.L1_MEMORY_CONFIG, ttnn.bfloat16)
            normalized_public = ttnn.reshape(normalized_interleaved, (1, self.batch, 1, self.hidden_size))
            mlp = self._sparse_moe(normalized_public, 1)
            mlp = ttnn.reshape(mlp, (1, 1, self.batch, self.hidden_size))
            mlp = ttnn.to_memory_config(mlp, self.residual_memory_config)
        output = ttnn.add(ttnn.add(residual, attention), mlp, memory_config=self.residual_memory_config)
        return ttnn.reshape(output, (1, self.batch, 1, self.hidden_size))


__all__ = ["MODEL_ID", "OptimizationConfig", "OptimizedDecoder"]
