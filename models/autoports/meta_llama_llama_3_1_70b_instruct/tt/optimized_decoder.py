# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Single-device optimized Llama 3.1 decoder layer.

The public tensor/cache contract is inherited from ``FunctionalDecoder``.  The
runtime path is independent: decode uses the shard-advisor-seeded 1-D
projection chain, dedicated
head/RoPE/SDPA operations, and a fused SwiGLU elementwise step.  Prefill remains
DRAM interleaved, uses explicit 2-D matmul configurations, and retains
composite SDPA with its measured implicit program selection.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace

import torch
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

import ttnn
from models.autoports.meta_llama_llama_3_1_70b_instruct.tt.functional_decoder import (
    EMITTED_BATCH,
    EMITTED_CACHE_LENGTH,
    EMITTED_PREFILL_SEQUENCE,
    FunctionalDecoder,
    _config_value,
    _state_tensor,
)

TILE_SIZE = 32


@dataclass(frozen=True)
class OptimizationConfig:
    """Cumulative optimized-decoder policy and experiment controls.

    The defaults are the measured winner after crossing precision, fidelity,
    matmul geometry, SDPA, and prefill-grid candidates: BF16 activations/norms,
    BFP4/LoFi projection weights and the current-pass shard-advisor 1-D decode
    topology. Eight recurrent real-weight decode positions verify that the
    lower-precision attention cache remains above the functional PCC bar.
    """

    attention_weight_dtype: object = ttnn.bfloat4_b
    gate_up_weight_dtype: object = ttnn.bfloat4_b
    down_weight_dtype: object = ttnn.bfloat4_b
    attention_math_fidelity: object = ttnn.MathFidelity.LoFi
    gate_up_math_fidelity: object = ttnn.MathFidelity.LoFi
    down_math_fidelity: object = ttnn.MathFidelity.LoFi
    residual_cores: int = 32
    qkv_cores: int = 32
    output_cores: int = 32
    gate_up_cores: int = 32
    down_cores: int = 32
    qkv_in0_block_w: int | None = None
    output_in0_block_w: int | None = None
    gate_up_in0_block_w: int | None = 4
    down_in0_block_w: int | None = 7
    sdpa_grid: tuple[int, int] = (8, 8)
    explicit_sdpa_program_config: bool = False
    explicit_sdpa_compute_kernel: bool = False
    prefill_grid: tuple[int, int] = (11, 10)
    prefill_gate_up_in0_block_w: int = 4
    packed_gate_up: bool = False
    packed_decode_gate_up: bool = False
    decode_matmul_strategy: str = "advisor_1d"
    fused_cache_update: bool = False
    advisor_gate_up_grid: tuple[int, int] = (11, 10)
    advisor_gate_up_in0_block_w: int = 2
    advisor_gate_up_per_core_n: int = 9
    advisor_gate_up_out_subblock_w: int = 3
    advisor_down_grid: tuple[int, int] = (11, 6)
    advisor_down_in0_block_w: int = 4
    advisor_down_per_core_n: int = 4
    advisor_down_out_subblock_w: int = 4
    advisor_exact_residual_chain: bool = False

    def with_changes(self, **changes) -> "OptimizationConfig":
        return replace(self, **changes)


def _rectangular_core_grid(num_cores: int, device_grid) -> ttnn.CoreGrid:
    if num_cores < 1 or num_cores > device_grid.x * device_grid.y:
        raise ValueError(f"num_cores={num_cores} does not fit device grid {device_grid}")
    for y in range(min(device_grid.y, num_cores), 0, -1):
        if num_cores % y == 0:
            x = num_cores // y
            if x <= device_grid.x:
                return ttnn.CoreGrid(x=x, y=y)
    raise ValueError(f"num_cores={num_cores} cannot form a rectangular grid inside {device_grid}")


def _largest_divisor(value: int, maximum: int = 16) -> int:
    for candidate in range(min(value, maximum), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


def _validate_matmul_geometry(*, role: str, k: int, n: int, cores: int, in0_block_w: int | None) -> int:
    k_tiles = k // TILE_SIZE
    n_tiles = n // TILE_SIZE
    if k % TILE_SIZE or n % TILE_SIZE:
        raise ValueError(f"{role}: K={k} and N={n} must be tile aligned")
    if k_tiles % cores or n_tiles % cores:
        raise ValueError(f"{role}: {cores} cores must divide both K tiles={k_tiles} and N tiles={n_tiles}")
    shard_k_tiles = k_tiles // cores
    selected = in0_block_w if in0_block_w is not None else _largest_divisor(shard_k_tiles)
    if selected < 1 or shard_k_tiles % selected:
        raise ValueError(f"{role}: in0_block_w={selected} must divide input shard K tiles={shard_k_tiles}")
    return selected


def _width_sharded_l1(*, rows: int, width: int, cores: int, device_grid) -> ttnn.MemoryConfig:
    grid = _rectangular_core_grid(cores, device_grid)
    if width % cores:
        raise ValueError(f"width={width} must be divisible by cores={cores}")
    return ttnn.create_sharded_memory_config(
        (rows, width // cores),
        grid,
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _advisor_width_sharded_l1(*, rows: int, width: int, cores: int, grid: tuple[int, int]) -> ttnn.MemoryConfig:
    """Create the possibly non-rectangular width shards emitted by shard-advise."""

    grid_x, grid_y = grid
    core_ranges = ttnn.num_cores_to_corerangeset(
        cores,
        ttnn.CoreCoord(grid_x, grid_y),
        row_wise=True,
    )
    shard_width = math.ceil(width / cores / TILE_SIZE) * TILE_SIZE
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core_ranges, (rows, shard_width), ttnn.ShardOrientation.ROW_MAJOR),
    )


def _advisor_block_sharded_l1(*, rows: int, width: int, cores: int, grid: tuple[int, int]) -> ttnn.MemoryConfig:
    """Create the padded single-row block shard emitted for the input norm."""

    grid_x, grid_y = grid
    core_ranges = ttnn.num_cores_to_corerangeset(
        cores,
        ttnn.CoreCoord(grid_x, grid_y),
        row_wise=True,
    )
    shard_width = math.ceil(width / cores / TILE_SIZE) * TILE_SIZE
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core_ranges, (rows, shard_width), ttnn.ShardOrientation.ROW_MAJOR),
    )


def _dram_weight_memory_config(mesh_device, *, k: int, n: int) -> ttnn.MemoryConfig:
    dram_grid_size = mesh_device.dram_grid_size()
    dram_cores = dram_grid_size.x * dram_grid_size.y
    padded_n = math.ceil(n / (TILE_SIZE * dram_cores)) * TILE_SIZE * dram_cores
    if padded_n != n:
        raise ValueError(f"DRAM-sharded weight N={n} would require padding to {padded_n}")
    dram_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1),
            )
        }
    )
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(dram_grid, (k, n // dram_cores), ttnn.ShardOrientation.ROW_MAJOR),
    )


def _to_device_tensor(
    tensor: torch.Tensor,
    mesh_device,
    *,
    layout=ttnn.TILE_LAYOUT,
    dtype=ttnn.bfloat16,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
):
    return ttnn.from_torch(
        tensor.contiguous(),
        device=mesh_device,
        layout=layout,
        dtype=dtype,
        memory_config=memory_config,
    )


def _compute_kernel_config(fidelity):
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def _dram_matmul_program_config(*, role: str, m: int, k: int, n: int, cores: int, in0_block_w: int | None):
    selected_in0 = _validate_matmul_geometry(
        role=role,
        k=k,
        n=n,
        cores=cores,
        in0_block_w=in0_block_w,
    )
    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=selected_in0,
        per_core_M=math.ceil(m / TILE_SIZE),
        per_core_N=n // TILE_SIZE // cores,
        fused_activation=None,
    )


def _advisor_matmul_program_config(*, grid: tuple[int, int], in0_block_w: int, per_core_n: int, out_subblock_w: int):
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        per_core_M=1,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


def _advisor_output_cores(*, n: int, per_core_n: int, grid: tuple[int, int]) -> int:
    if n % TILE_SIZE:
        raise ValueError(f"advisor output N={n} must be tile aligned")
    if per_core_n < 1:
        raise ValueError(f"advisor per_core_N must be positive, got {per_core_n}")
    cores = math.ceil(n / TILE_SIZE / per_core_n)
    if cores > grid[0] * grid[1]:
        raise ValueError(
            f"advisor output needs {cores} cores for N={n}, per_core_N={per_core_n}, "
            f"but grid={grid} has only {grid[0] * grid[1]}"
        )
    return cores


def _out_subblock_w(per_core_n: int) -> int:
    for width in (4, 3, 2, 1):
        if per_core_n % width == 0:
            return width
    return 1


class OptimizedDecoder(FunctionalDecoder):
    """Optimized dense decoder with the functional decoder's public contract."""

    def __init__(
        self,
        *,
        optimization_config: OptimizationConfig,
        prefill_qkv_weight,
        prefill_output_weight,
        prefill_gate_weight,
        prefill_up_weight,
        prefill_down_weight,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.optimization_config = optimization_config
        self.prefill_qkv_weight = prefill_qkv_weight
        self.prefill_output_weight = prefill_output_weight
        self.prefill_gate_weight = prefill_gate_weight
        self.prefill_up_weight = prefill_up_weight
        self.prefill_down_weight = prefill_down_weight
        if optimization_config.decode_matmul_strategy not in {"dram_sharded", "advisor_1d"}:
            raise ValueError(
                "decode_matmul_strategy must be 'dram_sharded' or 'advisor_1d', got "
                f"{optimization_config.decode_matmul_strategy!r}"
            )
        self.use_advisor_1d = optimization_config.decode_matmul_strategy == "advisor_1d"
        self.use_advisor_exact_chain = self.use_advisor_1d and optimization_config.advisor_exact_residual_chain
        self.decode_packed_gate_up = optimization_config.packed_gate_up or optimization_config.packed_decode_gate_up
        if optimization_config.advisor_exact_residual_chain and not self.use_advisor_1d:
            raise ValueError("advisor_exact_residual_chain requires decode_matmul_strategy='advisor_1d'")
        advisor_gate_up_per_core_n = optimization_config.advisor_gate_up_per_core_n
        if advisor_gate_up_per_core_n % optimization_config.advisor_gate_up_out_subblock_w:
            raise ValueError(
                "advisor_gate_up_out_subblock_w must divide the effective " f"per_core_N={advisor_gate_up_per_core_n}"
            )
        if optimization_config.advisor_down_per_core_n % optimization_config.advisor_down_out_subblock_w:
            raise ValueError(
                "advisor_down_out_subblock_w must divide " f"per_core_N={optimization_config.advisor_down_per_core_n}"
            )
        if optimization_config.fused_cache_update:
            raise ValueError(
                "fused_cache_update is unsupported for this 64Q/8KV decode topology: "
                "the required non-overlapping, head-aligned QKV split triggers a "
                "Blackhole nlp_create_qkv_heads_decode NoC-sanitizer failure"
            )
        device_grid = self.mesh_device.compute_with_storage_grid_size()
        padded_rows = TILE_SIZE * math.ceil(self.batch / TILE_SIZE)

        self.residual_mem_config = _width_sharded_l1(
            rows=padded_rows,
            width=self.hidden_size,
            cores=optimization_config.residual_cores,
            device_grid=device_grid,
        )
        self.qkv_input_mem_config = _width_sharded_l1(
            rows=padded_rows,
            width=self.hidden_size,
            cores=optimization_config.qkv_cores,
            device_grid=device_grid,
        )
        self.output_input_mem_config = _width_sharded_l1(
            rows=padded_rows,
            width=self.hidden_size,
            cores=optimization_config.output_cores,
            device_grid=device_grid,
        )
        self.gate_up_input_mem_config = _width_sharded_l1(
            rows=padded_rows,
            width=self.hidden_size,
            cores=optimization_config.gate_up_cores,
            device_grid=device_grid,
        )
        self.down_input_mem_config = _width_sharded_l1(
            rows=padded_rows,
            width=self.intermediate_size,
            cores=optimization_config.down_cores,
            device_grid=device_grid,
        )

        self.norm_program_config = self._make_norm_program_config(
            optimization_config.residual_cores,
            padded_rows,
        )
        if self.use_advisor_1d:
            # final_ir.mlir recommendations for the 70B dense decode graph. The
            # gate/up out-subblock remains configurable so register-constrained
            # compute policies can use a smaller divisor of per_core_N=9.
            self.qkv_decode_program_config = _advisor_matmul_program_config(
                grid=(11, 10), in0_block_w=2, per_core_n=3, out_subblock_w=3
            )
            self.output_decode_program_config = _advisor_matmul_program_config(
                grid=(11, 8), in0_block_w=2, per_core_n=3, out_subblock_w=3
            )
            self.gate_up_decode_program_config = _advisor_matmul_program_config(
                grid=optimization_config.advisor_gate_up_grid,
                in0_block_w=optimization_config.advisor_gate_up_in0_block_w,
                per_core_n=advisor_gate_up_per_core_n,
                out_subblock_w=optimization_config.advisor_gate_up_out_subblock_w,
            )
            self.down_decode_program_config = _advisor_matmul_program_config(
                grid=optimization_config.advisor_down_grid,
                in0_block_w=optimization_config.advisor_down_in0_block_w,
                per_core_n=optimization_config.advisor_down_per_core_n,
                out_subblock_w=optimization_config.advisor_down_out_subblock_w,
            )
            self.advisor_qkv_output_mem_config = _advisor_width_sharded_l1(
                rows=padded_rows, width=10240, cores=107, grid=(11, 10)
            )
            self.advisor_gate_up_output_mem_config = _advisor_width_sharded_l1(
                rows=padded_rows,
                width=self.intermediate_size * (2 if self.decode_packed_gate_up else 1),
                cores=_advisor_output_cores(
                    n=self.intermediate_size * (2 if self.decode_packed_gate_up else 1),
                    per_core_n=advisor_gate_up_per_core_n,
                    grid=optimization_config.advisor_gate_up_grid,
                ),
                grid=optimization_config.advisor_gate_up_grid,
            )
            self.advisor_residual_output_mem_config = _advisor_width_sharded_l1(
                rows=padded_rows, width=self.hidden_size, cores=86, grid=(11, 8)
            )
            self.advisor_down_output_mem_config = _advisor_width_sharded_l1(
                rows=padded_rows,
                width=self.hidden_size,
                cores=_advisor_output_cores(
                    n=self.hidden_size,
                    per_core_n=optimization_config.advisor_down_per_core_n,
                    grid=optimization_config.advisor_down_grid,
                ),
                grid=optimization_config.advisor_down_grid,
            )
            self.advisor_input_norm_mem_config = _advisor_block_sharded_l1(
                rows=padded_rows, width=self.hidden_size, cores=11, grid=(11, 1)
            )
            self.advisor_qkv_input_mem_config = _advisor_width_sharded_l1(
                rows=padded_rows, width=self.hidden_size, cores=64, grid=(11, 6)
            )
            self.advisor_post_norm_mem_config = _advisor_block_sharded_l1(
                rows=padded_rows, width=self.hidden_size, cores=11, grid=(11, 1)
            )
            self.advisor_gate_up_input_mem_config = _advisor_width_sharded_l1(
                rows=padded_rows, width=self.hidden_size, cores=64, grid=(11, 6)
            )
            self.advisor_down_input_mem_config = _advisor_width_sharded_l1(
                rows=padded_rows, width=self.intermediate_size, cores=75, grid=(11, 7)
            )
            self.advisor_input_norm_program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=[11, 1],
                subblock_w=4,
                block_h=padded_rows // TILE_SIZE,
                block_w=24,
                inplace=False,
            )
            self.advisor_post_norm_program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=[11, 1],
                subblock_w=4,
                block_h=padded_rows // TILE_SIZE,
                block_w=24,
                inplace=False,
            )
        else:
            self.qkv_decode_program_config = _dram_matmul_program_config(
                role="qkv",
                m=padded_rows,
                k=self.hidden_size,
                n=(self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
                cores=optimization_config.qkv_cores,
                in0_block_w=optimization_config.qkv_in0_block_w,
            )
            self.output_decode_program_config = _dram_matmul_program_config(
                role="output",
                m=padded_rows,
                k=self.hidden_size,
                n=self.hidden_size,
                cores=optimization_config.output_cores,
                in0_block_w=optimization_config.output_in0_block_w,
            )
            gate_up_n = self.intermediate_size * (2 if self.decode_packed_gate_up else 1)
            self.gate_up_decode_program_config = _dram_matmul_program_config(
                role="gate_up",
                m=padded_rows,
                k=self.hidden_size,
                n=gate_up_n,
                cores=optimization_config.gate_up_cores,
                in0_block_w=optimization_config.gate_up_in0_block_w,
            )
            self.down_decode_program_config = _dram_matmul_program_config(
                role="down",
                m=padded_rows,
                k=self.intermediate_size,
                n=self.hidden_size,
                cores=optimization_config.down_cores,
                in0_block_w=optimization_config.down_in0_block_w,
            )

        self.attention_compute_kernel = _compute_kernel_config(optimization_config.attention_math_fidelity)
        self.gate_up_compute_kernel = _compute_kernel_config(optimization_config.gate_up_math_fidelity)
        self.down_compute_kernel = _compute_kernel_config(optimization_config.down_math_fidelity)
        self.norm_compute_kernel = _compute_kernel_config(ttnn.MathFidelity.HiFi4)
        self.sdpa_compute_kernel = _compute_kernel_config(ttnn.MathFidelity.HiFi2)
        self.decode_sdpa_compute_kernel = (
            self.sdpa_compute_kernel if optimization_config.explicit_sdpa_compute_kernel else None
        )
        self.decode_sdpa_program_config = (
            ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=optimization_config.sdpa_grid,
                exp_approx_mode=False,
                q_chunk_size=0,
                k_chunk_size=0,
            )
            if optimization_config.explicit_sdpa_program_config
            else None
        )
        self.prefill_sdpa_program_config = (
            ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=optimization_config.sdpa_grid,
                exp_approx_mode=False,
                q_chunk_size=64,
                k_chunk_size=64,
            )
            if optimization_config.explicit_sdpa_program_config
            else None
        )
        self._prefill_program_configs: dict[tuple[str, int], object] = {}

    def _make_norm_program_config(self, cores: int, padded_rows: int):
        grid = _rectangular_core_grid(cores, self.mesh_device.compute_with_storage_grid_size())
        block_w = self.hidden_size // cores // TILE_SIZE
        subblock_w = _out_subblock_w(block_w)
        return ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[grid.x, grid.y],
            subblock_w=subblock_w,
            block_h=padded_rows // TILE_SIZE,
            block_w=block_w,
            inplace=False,
        )

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Mapping[str, torch.Tensor],
        *,
        hf_config,
        layer_idx: int,
        mesh_device,
        batch: int = EMITTED_BATCH,
        max_cache_len: int = EMITTED_CACHE_LENGTH,
        optimization_config: OptimizationConfig | None = None,
        **_kwargs,
    ) -> "OptimizedDecoder":
        num_devices = mesh_device.get_num_devices() if hasattr(mesh_device, "get_num_devices") else 1
        if num_devices != 1:
            raise ValueError(f"OptimizedDecoder requires a 1x1 mesh, got {num_devices} devices")
        if batch < 1 or batch > EMITTED_BATCH:
            raise ValueError(f"batch must be in [1, {EMITTED_BATCH}], got {batch}")
        if max_cache_len < 1:
            raise ValueError(f"max_cache_len must be positive, got {max_cache_len}")

        policy = optimization_config or OptimizationConfig()
        hidden_size = int(_config_value(hf_config, "hidden_size"))
        num_heads = int(_config_value(hf_config, "num_attention_heads"))
        num_kv_heads = int(_config_value(hf_config, "num_key_value_heads"))
        head_dim = int(_config_value(hf_config, "head_dim") or hidden_size // num_heads)
        intermediate_size = int(_config_value(hf_config, "intermediate_size"))
        rms_norm_eps = float(_config_value(hf_config, "rms_norm_eps"))

        q = _state_tensor(state_dict, layer_idx, "self_attn.q_proj.weight").to(torch.bfloat16)
        k = _state_tensor(state_dict, layer_idx, "self_attn.k_proj.weight").to(torch.bfloat16)
        v = _state_tensor(state_dict, layer_idx, "self_attn.v_proj.weight").to(torch.bfloat16)
        output = _state_tensor(state_dict, layer_idx, "self_attn.o_proj.weight").to(torch.bfloat16)
        gate = _state_tensor(state_dict, layer_idx, "mlp.gate_proj.weight").to(torch.bfloat16)
        up = _state_tensor(state_dict, layer_idx, "mlp.up_proj.weight").to(torch.bfloat16)
        down = _state_tensor(state_dict, layer_idx, "mlp.down_proj.weight").to(torch.bfloat16)
        input_norm = _state_tensor(state_dict, layer_idx, "input_layernorm.weight").to(torch.bfloat16)
        post_attention_norm = _state_tensor(
            state_dict,
            layer_idx,
            "post_attention_layernorm.weight",
        ).to(torch.bfloat16)

        qkv = torch.cat((q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)), dim=-1)
        gate_t = gate.transpose(0, 1)
        up_t = up.transpose(0, 1)
        decode_packed_gate_up = policy.packed_gate_up or policy.packed_decode_gate_up
        gate_up = torch.cat((gate_t, up_t), dim=-1) if decode_packed_gate_up else None

        rotary = LlamaRotaryEmbedding(hf_config)
        positions = torch.arange(max_cache_len, dtype=torch.long).unsqueeze(0)
        rope_probe = torch.zeros((1, 1, max_cache_len, head_dim), dtype=torch.bfloat16)
        cos, sin = rotary(rope_probe, positions)
        cos = cos.to(torch.bfloat16).unsqueeze(1)
        sin = sin.to(torch.bfloat16).unsqueeze(1)

        qkv_n = (num_heads + 2 * num_kv_heads) * head_dim
        advisor_1d = policy.decode_matmul_strategy == "advisor_1d"
        qkv_mem = (
            ttnn.DRAM_MEMORY_CONFIG if advisor_1d else _dram_weight_memory_config(mesh_device, k=hidden_size, n=qkv_n)
        )
        output_mem = (
            ttnn.DRAM_MEMORY_CONFIG
            if advisor_1d
            else _dram_weight_memory_config(mesh_device, k=hidden_size, n=hidden_size)
        )
        decode_gate_up_n = intermediate_size * (2 if decode_packed_gate_up else 1)
        gate_up_mem = (
            ttnn.DRAM_MEMORY_CONFIG
            if advisor_1d
            else _dram_weight_memory_config(mesh_device, k=hidden_size, n=decode_gate_up_n)
        )
        down_mem = (
            ttnn.DRAM_MEMORY_CONFIG
            if advisor_1d
            else _dram_weight_memory_config(mesh_device, k=intermediate_size, n=hidden_size)
        )

        qkv_weight = _to_device_tensor(
            qkv,
            mesh_device,
            dtype=policy.attention_weight_dtype,
            memory_config=qkv_mem,
        )
        output_weight = _to_device_tensor(
            output.transpose(0, 1),
            mesh_device,
            dtype=policy.attention_weight_dtype,
            memory_config=output_mem,
        )
        gate_weight = _to_device_tensor(
            gate_up if decode_packed_gate_up else gate_t,
            mesh_device,
            dtype=policy.gate_up_weight_dtype,
            memory_config=gate_up_mem,
        )
        up_weight = (
            None
            if decode_packed_gate_up
            else _to_device_tensor(
                up_t,
                mesh_device,
                dtype=policy.gate_up_weight_dtype,
                memory_config=gate_up_mem,
            )
        )
        down_weight = _to_device_tensor(
            down.transpose(0, 1),
            mesh_device,
            dtype=policy.down_weight_dtype,
            memory_config=down_mem,
        )

        # Blackhole DRAM-sharded matmuls silently corrupt results when a 2-D
        # prefill program's per_core_N does not match the eight-bank weight
        # shard width. Keep the fast sharded copies for decode and independent
        # interleaved copies for the larger 11x10 prefill programs. Advisor 1-D
        # weights are already interleaved and can be shared across both phases.
        if advisor_1d:
            prefill_qkv_weight = qkv_weight
            prefill_output_weight = output_weight
            if policy.packed_decode_gate_up and not policy.packed_gate_up:
                prefill_gate_weight = _to_device_tensor(gate_t, mesh_device, dtype=policy.gate_up_weight_dtype)
                prefill_up_weight = _to_device_tensor(up_t, mesh_device, dtype=policy.gate_up_weight_dtype)
            else:
                prefill_gate_weight = gate_weight
                prefill_up_weight = up_weight
            prefill_down_weight = down_weight
        else:
            prefill_qkv_weight = _to_device_tensor(qkv, mesh_device, dtype=policy.attention_weight_dtype)
            prefill_output_weight = _to_device_tensor(
                output.transpose(0, 1), mesh_device, dtype=policy.attention_weight_dtype
            )
            prefill_gate_weight = _to_device_tensor(
                gate_up if policy.packed_gate_up else gate_t,
                mesh_device,
                dtype=policy.gate_up_weight_dtype,
            )
            prefill_up_weight = (
                None
                if policy.packed_gate_up
                else _to_device_tensor(up_t, mesh_device, dtype=policy.gate_up_weight_dtype)
            )
            prefill_down_weight = _to_device_tensor(down.transpose(0, 1), mesh_device, dtype=policy.down_weight_dtype)

        return cls(
            optimization_config=policy,
            prefill_qkv_weight=prefill_qkv_weight,
            prefill_output_weight=prefill_output_weight,
            prefill_gate_weight=prefill_gate_weight,
            prefill_up_weight=prefill_up_weight,
            prefill_down_weight=prefill_down_weight,
            mesh_device=mesh_device,
            layer_idx=layer_idx,
            batch=batch,
            max_cache_len=max_cache_len,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            rms_norm_eps=rms_norm_eps,
            input_norm=_to_device_tensor(input_norm, mesh_device),
            post_attention_norm=_to_device_tensor(post_attention_norm, mesh_device),
            qkv_weight=qkv_weight,
            output_weight=output_weight,
            gate_weight=gate_weight,
            up_weight=up_weight,
            down_weight=down_weight,
            rotary_cos=_to_device_tensor(cos, mesh_device),
            rotary_sin=_to_device_tensor(sin, mesh_device),
            position_indices=_to_device_tensor(
                torch.arange(max_cache_len, dtype=torch.int32).unsqueeze(1).expand(-1, batch),
                mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.int32,
            ),
            mask_positions=_to_device_tensor(
                torch.arange(max_cache_len, dtype=torch.int32).reshape(1, 1, 1, max_cache_len),
                mesh_device,
                dtype=ttnn.int32,
            ),
            mask_zero=_to_device_tensor(torch.zeros((1, 1, 1, 1), dtype=torch.bfloat16), mesh_device),
            mask_negative_infinity=_to_device_tensor(
                torch.full((1, 1, 1, 1), float("-inf"), dtype=torch.bfloat16),
                mesh_device,
            ),
        )

    def _prefill_program_config(self, role: str, seq_len: int, *, k: int, n: int, fidelity):
        key = (role, seq_len)
        if key in self._prefill_program_configs:
            return self._prefill_program_configs[key]
        grid_x, grid_y = self.optimization_config.prefill_grid
        per_core_n = math.ceil(n / TILE_SIZE / grid_x)
        maximum_in0_block_w = (
            self.optimization_config.prefill_gate_up_in0_block_w if role in {"gate", "up", "gate_up_packed"} else 8
        )
        config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(grid_x, grid_y),
            in0_block_w=_largest_divisor(k // TILE_SIZE, maximum=maximum_in0_block_w),
            out_subblock_h=1,
            out_subblock_w=_out_subblock_w(per_core_n),
            # Fuse-batch still pads every user's M dimension independently to
            # a tile, so size the grid from batch * ceil(seq/32), not seq alone.
            per_core_M=max(1, math.ceil(self.batch * math.ceil(seq_len / TILE_SIZE) / grid_y)),
            per_core_N=per_core_n,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        )
        self._prefill_program_configs[key] = config
        return config

    def _prefill_linear(self, x, weight, *, role: str, seq_len: int, k: int, n: int, compute_kernel):
        return ttnn.linear(
            x,
            weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self._prefill_program_config(
                role,
                seq_len,
                k=k,
                n=n,
                fidelity=compute_kernel,
            ),
            compute_kernel_config=compute_kernel,
        )

    def _mlp_prefill(self, hidden_states, seq_len: int):
        if self.optimization_config.packed_gate_up:
            gate_up = self._prefill_linear(
                hidden_states,
                self.prefill_gate_weight,
                role="gate_up_packed",
                seq_len=seq_len,
                k=self.hidden_size,
                n=2 * self.intermediate_size,
                compute_kernel=self.gate_up_compute_kernel,
            )
            gate = ttnn.slice(
                gate_up,
                [0, 0, 0, 0],
                [1, self.batch, seq_len, self.intermediate_size],
                [1, 1, 1, 1],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            up = ttnn.slice(
                gate_up,
                [0, 0, 0, self.intermediate_size],
                [1, self.batch, seq_len, 2 * self.intermediate_size],
                [1, 1, 1, 1],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            gate = self._prefill_linear(
                hidden_states,
                self.prefill_gate_weight,
                role="gate",
                seq_len=seq_len,
                k=self.hidden_size,
                n=self.intermediate_size,
                compute_kernel=self.gate_up_compute_kernel,
            )
            up = self._prefill_linear(
                hidden_states,
                self.prefill_up_weight,
                role="up",
                seq_len=seq_len,
                k=self.hidden_size,
                n=self.intermediate_size,
                compute_kernel=self.gate_up_compute_kernel,
            )
        gated = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return self._prefill_linear(
            gated,
            self.prefill_down_weight,
            role="down",
            seq_len=seq_len,
            k=self.intermediate_size,
            n=self.hidden_size,
            compute_kernel=self.down_compute_kernel,
        )

    def _mlp_decode(self, hidden_states):
        if self.use_advisor_exact_chain:
            mlp_input_mem_config = self.advisor_gate_up_input_mem_config
        elif self.use_advisor_1d:
            mlp_input_mem_config = ttnn.L1_MEMORY_CONFIG
        else:
            mlp_input_mem_config = self.gate_up_input_mem_config
        projection_output_mem_config = (
            self.advisor_gate_up_output_mem_config if self.use_advisor_1d else ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
        )
        mlp_input = ttnn.to_memory_config(hidden_states, mlp_input_mem_config)
        if self.decode_packed_gate_up:
            gate_up = ttnn.linear(
                mlp_input,
                self.gate_weight,
                dtype=ttnn.bfloat16,
                memory_config=projection_output_mem_config,
                program_config=self.gate_up_decode_program_config,
                compute_kernel_config=self.gate_up_compute_kernel,
            )
            gate_up = ttnn.to_memory_config(gate_up, ttnn.L1_MEMORY_CONFIG)
            gate = ttnn.slice(
                gate_up,
                [0, 0, 0, 0],
                [1, 1, self.batch, self.intermediate_size],
                [1, 1, 1, 1],
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            up = ttnn.slice(
                gate_up,
                [0, 0, 0, self.intermediate_size],
                [1, 1, self.batch, 2 * self.intermediate_size],
                [1, 1, 1, 1],
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            if not self.use_advisor_1d:
                gate = ttnn.to_memory_config(gate, self.down_input_mem_config)
                up = ttnn.to_memory_config(up, self.down_input_mem_config)
        else:
            gate = ttnn.linear(
                mlp_input,
                self.gate_weight,
                dtype=ttnn.bfloat16,
                memory_config=projection_output_mem_config,
                program_config=self.gate_up_decode_program_config,
                compute_kernel_config=self.gate_up_compute_kernel,
            )
            up = ttnn.linear(
                mlp_input,
                self.up_weight,
                dtype=ttnn.bfloat16,
                memory_config=projection_output_mem_config,
                program_config=self.gate_up_decode_program_config,
                compute_kernel_config=self.gate_up_compute_kernel,
            )
        gated_output_mem_config = (
            ttnn.L1_MEMORY_CONFIG
            if self.decode_packed_gate_up and self.use_advisor_1d
            else projection_output_mem_config
        )
        gated = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=ttnn.bfloat16,
            memory_config=gated_output_mem_config,
        )
        if self.use_advisor_exact_chain:
            gated_mem_config = self.advisor_down_input_mem_config
        elif self.use_advisor_1d:
            gated_mem_config = ttnn.L1_MEMORY_CONFIG
        else:
            gated_mem_config = self.down_input_mem_config
        gated = ttnn.to_memory_config(gated, gated_mem_config)
        down = ttnn.linear(
            gated,
            self.down_weight,
            dtype=ttnn.bfloat16,
            memory_config=(
                self.advisor_down_output_mem_config if self.use_advisor_1d else ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
            ),
            program_config=self.down_decode_program_config,
            compute_kernel_config=self.down_compute_kernel,
        )
        return down if self.use_advisor_exact_chain else ttnn.to_memory_config(down, self.residual_mem_config)

    def prefill_forward(self, hidden_states, key_cache, value_cache):
        seq_len = self._validate_hidden_states(hidden_states)
        self._validate_caches(key_cache, value_cache)

        residual = hidden_states
        normed = ttnn.rms_norm(
            hidden_states,
            epsilon=self.rms_norm_eps,
            weight=self.input_norm,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.norm_compute_kernel,
        )
        fused_qkv = self._prefill_linear(
            normed,
            self.prefill_qkv_weight,
            role="qkv",
            seq_len=seq_len,
            k=self.hidden_size,
            n=(self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
            compute_kernel=self.attention_compute_kernel,
        )
        fused_qkv = ttnn.reshape(
            fused_qkv,
            [self.batch, seq_len, (self.num_heads + 2 * self.num_kv_heads) * self.head_dim],
        )
        query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
            fused_qkv,
            None,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            transpose_key=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        cos = ttnn.slice(self.rotary_cos, [0, 0, 0, 0], [1, 1, seq_len, self.head_dim], [1, 1, 1, 1])
        sin = ttnn.slice(self.rotary_sin, [0, 0, 0, 0], [1, 1, seq_len, self.head_dim], [1, 1, 1, 1])
        query = ttnn.experimental.rotary_embedding(
            query,
            cos,
            sin,
            None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        key = ttnn.experimental.rotary_embedding(
            key,
            cos,
            sin,
            None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        query = ttnn.slice(
            query,
            [0, 0, 0, 0],
            [self.batch, self.num_heads, seq_len, self.head_dim],
            [1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        key = ttnn.slice(
            key,
            [0, 0, 0, 0],
            [self.batch, self.num_kv_heads, seq_len, self.head_dim],
            [1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        key_fill = key if key.dtype == key_cache.dtype else ttnn.typecast(key, key_cache.dtype)
        value_fill = value if value.dtype == value_cache.dtype else ttnn.typecast(value, value_cache.dtype)
        for user_id in range(self.batch):
            key_user = ttnn.slice(
                key_fill,
                [user_id, 0, 0, 0],
                [user_id + 1, self.num_kv_heads, seq_len, self.head_dim],
                [1, 1, 1, 1],
            )
            value_user = ttnn.slice(
                value_fill,
                [user_id, 0, 0, 0],
                [user_id + 1, self.num_kv_heads, seq_len, self.head_dim],
                [1, 1, 1, 1],
            )
            ttnn.fill_cache(key_cache, key_user, user_id)
            ttnn.fill_cache(value_cache, value_user, user_id)

        attention = ttnn.transformer.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
            scale=self.scale,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.prefill_sdpa_program_config,
            compute_kernel_config=self.sdpa_compute_kernel,
        )
        attention = ttnn.transformer.concatenate_heads(attention, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attention = ttnn.reshape(attention, [1, self.batch, seq_len, self.hidden_size])
        attention = self._prefill_linear(
            attention,
            self.prefill_output_weight,
            role="output",
            seq_len=seq_len,
            k=self.hidden_size,
            n=self.hidden_size,
            compute_kernel=self.attention_compute_kernel,
        )
        hidden_states = ttnn.add(
            residual,
            attention,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        residual = hidden_states
        hidden_states = ttnn.rms_norm(
            hidden_states,
            epsilon=self.rms_norm_eps,
            weight=self.post_attention_norm,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.norm_compute_kernel,
        )
        hidden_states = self._mlp_prefill(hidden_states, seq_len)
        return ttnn.add(
            residual,
            hidden_states,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def decode_forward(self, hidden_states, key_cache, value_cache, *, current_pos: int):
        self._validate_hidden_states(hidden_states, expected_seq_len=1)
        self._validate_caches(key_cache, value_cache)
        if current_pos < 0 or current_pos >= self.max_cache_len:
            raise ValueError(f"current_pos must be in [0, {self.max_cache_len}), got {current_pos}")

        hidden_states = ttnn.reshape(hidden_states, [1, 1, self.batch, self.hidden_size])
        hidden_states = ttnn.to_memory_config(
            hidden_states,
            ttnn.L1_MEMORY_CONFIG if self.use_advisor_exact_chain else self.residual_mem_config,
        )
        residual = hidden_states
        input_norm_mem_config = (
            self.advisor_input_norm_mem_config if self.use_advisor_exact_chain else self.residual_mem_config
        )
        input_norm_program_config = (
            self.advisor_input_norm_program_config if self.use_advisor_exact_chain else self.norm_program_config
        )
        hidden_states = ttnn.to_memory_config(hidden_states, input_norm_mem_config)
        normed = ttnn.rms_norm(
            hidden_states,
            epsilon=self.rms_norm_eps,
            weight=self.input_norm,
            program_config=input_norm_program_config,
            memory_config=input_norm_mem_config,
            compute_kernel_config=self.norm_compute_kernel,
        )
        if self.use_advisor_exact_chain:
            qkv_input_mem_config = self.advisor_qkv_input_mem_config
        elif self.use_advisor_1d:
            qkv_input_mem_config = ttnn.L1_MEMORY_CONFIG
        else:
            qkv_input_mem_config = self.qkv_input_mem_config
        normed = ttnn.to_memory_config(normed, qkv_input_mem_config)
        fused_qkv = ttnn.linear(
            normed,
            self.qkv_weight,
            dtype=ttnn.bfloat16,
            memory_config=(
                self.advisor_qkv_output_mem_config if self.use_advisor_1d else ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
            ),
            program_config=self.qkv_decode_program_config,
            compute_kernel_config=self.attention_compute_kernel,
        )
        fused_qkv = ttnn.sharded_to_interleaved(fused_qkv, ttnn.L1_MEMORY_CONFIG, ttnn.bfloat16)
        query, key, value = ttnn.experimental.nlp_create_qkv_heads_decode(
            fused_qkv,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            overlap_qk_coregrid=True,
            memory_config=self.decode_attention_mem_config,
        )

        cos = ttnn.slice(
            self.rotary_cos,
            [0, 0, current_pos, 0],
            [1, 1, current_pos + 1, self.head_dim],
            [1, 1, 1, 1],
        )
        sin = ttnn.slice(
            self.rotary_sin,
            [0, 0, current_pos, 0],
            [1, 1, current_pos + 1, self.head_dim],
            [1, 1, 1, 1],
        )
        query = ttnn.experimental.rotary_embedding(
            query,
            cos,
            sin,
            0,
            memory_config=self.decode_attention_mem_config,
        )
        key = ttnn.experimental.rotary_embedding(
            key,
            cos,
            sin,
            0,
            memory_config=self.decode_cache_mem_config,
        )

        update_indices = ttnn.slice(
            self.position_indices,
            [current_pos, 0],
            [current_pos + 1, self.batch],
            [1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        update_indices = ttnn.reshape(update_indices, [self.batch])
        ttnn.experimental.paged_update_cache(
            key_cache,
            key,
            update_idxs_tensor=update_indices,
            share_cache=False,
            page_table=None,
        )
        ttnn.experimental.paged_update_cache(
            value_cache,
            value,
            update_idxs_tensor=update_indices,
            share_cache=False,
            page_table=None,
        )

        query = ttnn.to_memory_config(query, ttnn.DRAM_MEMORY_CONFIG)
        mask_position = ttnn.slice(
            self.position_indices,
            [current_pos, 0],
            [current_pos + 1, 1],
            [1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        mask_position = ttnn.reshape(mask_position, [1, 1, 1, 1], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        valid_cache_positions = ttnn.ge(
            mask_position,
            self.mask_positions,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attention_mask = ttnn.where(
            valid_cache_positions,
            self.mask_zero,
            self.mask_negative_infinity,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attention_mask = ttnn.repeat(
            attention_mask,
            ttnn.Shape([1, 1, self.padded_num_heads, 1]),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attention = ttnn.transformer.scaled_dot_product_attention_decode(
            query,
            key_cache,
            value_cache,
            is_causal=False,
            attn_mask=attention_mask,
            cur_pos_tensor=None,
            scale=self.scale,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.decode_sdpa_program_config,
            compute_kernel_config=self.decode_sdpa_compute_kernel,
        )
        attention = ttnn.to_memory_config(attention, self.decode_attention_mem_config)
        attention = ttnn.experimental.nlp_concat_heads_decode(
            attention,
            num_heads=self.num_heads,
            sub_core_grids=self.decode_compute_core_grid,
        )
        attention = ttnn.slice(
            attention,
            [0, 0, 0, 0],
            [1, 1, self.batch, self.hidden_size],
            [1, 1, 1, 1],
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        attention = ttnn.to_memory_config(
            attention,
            ttnn.L1_MEMORY_CONFIG if self.use_advisor_1d else self.output_input_mem_config,
        )
        attention = ttnn.linear(
            attention,
            self.output_weight,
            dtype=ttnn.bfloat16,
            memory_config=(
                self.advisor_residual_output_mem_config if self.use_advisor_1d else ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
            ),
            program_config=self.output_decode_program_config,
            compute_kernel_config=self.attention_compute_kernel,
        )
        if not self.use_advisor_exact_chain:
            attention = ttnn.to_memory_config(attention, self.residual_mem_config)
        hidden_states = ttnn.add(
            residual,
            attention,
            dtype=ttnn.bfloat16,
            memory_config=(
                self.advisor_residual_output_mem_config if self.use_advisor_exact_chain else self.residual_mem_config
            ),
        )

        residual = hidden_states
        post_norm_mem_config = (
            self.advisor_post_norm_mem_config if self.use_advisor_exact_chain else self.residual_mem_config
        )
        post_norm_program_config = (
            self.advisor_post_norm_program_config if self.use_advisor_exact_chain else self.norm_program_config
        )
        hidden_states = ttnn.to_memory_config(hidden_states, post_norm_mem_config)
        hidden_states = ttnn.rms_norm(
            hidden_states,
            epsilon=self.rms_norm_eps,
            weight=self.post_attention_norm,
            program_config=post_norm_program_config,
            memory_config=post_norm_mem_config,
            compute_kernel_config=self.norm_compute_kernel,
        )
        hidden_states = self._mlp_decode(hidden_states)
        hidden_states = ttnn.add(
            residual,
            hidden_states,
            dtype=ttnn.bfloat16,
            memory_config=(
                self.advisor_residual_output_mem_config if self.use_advisor_exact_chain else self.residual_mem_config
            ),
        )
        if self.use_advisor_exact_chain:
            hidden_states = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG)
            hidden_states = ttnn.reshape(hidden_states, [1, self.batch, 1, self.hidden_size])
            return ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.reshape(hidden_states, [1, self.batch, 1, self.hidden_size])


__all__ = [
    "EMITTED_BATCH",
    "EMITTED_CACHE_LENGTH",
    "EMITTED_PREFILL_SEQUENCE",
    "OptimizationConfig",
    "OptimizedDecoder",
]
