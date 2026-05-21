import math
from dataclasses import dataclass
from typing import Callable

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc


TILE = 32
NUM_WARMUP = 0
NUM_ITERS = 1


@dataclass(frozen=True)
class MatmulSweepSpec:
    m: int
    k: int
    n: int
    in0_dtype: ttnn.DataType
    in1_dtype: ttnn.DataType
    out_dtype: ttnn.DataType
    torch_in0_dtype: torch.dtype
    torch_in1_dtype: torch.dtype
    math_fidelity: ttnn.MathFidelity
    pcc: float = 0.95


def out_subblock_w(per_core_n: int) -> int:
    for w in range(min(8, per_core_n), 0, -1):
        if per_core_n % w == 0:
            return w
    return 1


def compute_kernel(spec: MatmulSweepSpec):
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=spec.math_fidelity,
        math_approx_mode=spec.math_fidelity == ttnn.MathFidelity.LoFi,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def core_range_set_for_num_cores(num_cores: int, grid_x: int = 8):
    return ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(core_id % grid_x, core_id // grid_x),
                ttnn.CoreCoord(core_id % grid_x, core_id // grid_x),
            )
            for core_id in range(num_cores)
        }
    )


def dram_core_range_set(device, num_banks: int | None = None):
    dram_grid = device.dram_grid_size()
    if num_banks is not None:
        return ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(bank_id % int(dram_grid.x), bank_id // int(dram_grid.x)),
                    ttnn.CoreCoord(bank_id % int(dram_grid.x), bank_id // int(dram_grid.x)),
                )
                for bank_id in range(num_banks)
            }
        )
    return ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(int(dram_grid.x) - 1, int(dram_grid.y) - 1),
            )
        }
    )


def dram_width_in1_mem(device, spec: MatmulSweepSpec, num_banks: int | None = None):
    dram_grid = device.dram_grid_size()
    banks = num_banks or int(dram_grid.x) * int(dram_grid.y)
    n_padded = math.ceil(spec.n / (TILE * banks)) * TILE * banks
    return ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.DRAM,
        shard_spec=ttnn.ShardSpec(
            dram_core_range_set(device, num_banks),
            [spec.k, n_padded // banks],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )


def dram_sharded_in1_mem(device, spec: MatmulSweepSpec, in1_layout: str):
    if in1_layout == "interleaved":
        return ttnn.DRAM_MEMORY_CONFIG
    dram_grid = device.dram_grid_size()
    dram_x = int(dram_grid.x)
    dram_y = int(dram_grid.y)
    num_banks = dram_x * dram_y
    if in1_layout == "width":
        n_padded = math.ceil(spec.n / (TILE * num_banks)) * TILE * num_banks
        shard_shape = [spec.k, n_padded // num_banks]
        layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED
    elif in1_layout == "height":
        k_padded = math.ceil(spec.k / (TILE * num_banks)) * TILE * num_banks
        shard_shape = [k_padded // num_banks, spec.n]
        layout = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    elif in1_layout == "block":
        n_padded = math.ceil(spec.n / (TILE * dram_x)) * TILE * dram_x
        k_padded = math.ceil(spec.k / (TILE * dram_y)) * TILE * dram_y
        shard_shape = [k_padded // dram_y, n_padded // dram_x]
        layout = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    else:
        raise ValueError(f"unknown in1_layout: {in1_layout}")
    return ttnn.MemoryConfig(
        memory_layout=layout,
        buffer_type=ttnn.BufferType.DRAM,
        shard_spec=ttnn.ShardSpec(dram_core_range_set(device), shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
    )


def cfg_width_explicit(spec: MatmulSweepSpec, grid_x: int, grid_y: int, per_core_n: int, in0_block_w: int):
    def _builder(device):
        in0_mem = ttnn.create_sharded_memory_config(
            (1, 1, spec.m, spec.k),
            core_grid=ttnn.CoreGrid(y=grid_y, x=grid_x),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
        out_mem = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            buffer_type=ttnn.BufferType.L1,
        )
        prog = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=in0_block_w,
            per_core_M=spec.m // TILE,
            per_core_N=per_core_n,
            fused_activation=None,
        )
        return in0_mem, dram_width_in1_mem(device, spec), out_mem, prog

    return _builder


def cfg_height_explicit(
    spec: MatmulSweepSpec, grid_x: int, grid_y: int, per_core_n: int, in0_block_w: int, in1_layout: str
):
    def _builder(device):
        in0_mem = ttnn.create_sharded_memory_config(
            (1, 1, spec.m, spec.k),
            core_grid=ttnn.CoreGrid(y=grid_y, x=grid_x),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
        out_mem = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            buffer_type=ttnn.BufferType.L1,
        )
        prog = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=(grid_x, grid_y),
            in0_block_w=in0_block_w,
            out_subblock_h=1,
            out_subblock_w=out_subblock_w(per_core_n),
            per_core_M=max(1, (spec.m // TILE) // max(1, grid_x * grid_y)),
            per_core_N=per_core_n,
        )
        return in0_mem, dram_sharded_in1_mem(device, spec, in1_layout), out_mem, prog

    return _builder


def cfg_block_explicit(
    spec: MatmulSweepSpec, grid_x: int, grid_y: int, per_core_n: int, in0_block_w: int, in1_layout: str
):
    def _builder(device):
        per_core_m = max(1, (spec.m // TILE) // max(1, grid_y))
        in0_mem = ttnn.create_sharded_memory_config(
            (1, 1, spec.m, spec.k),
            core_grid=ttnn.CoreGrid(y=grid_y, x=grid_x),
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
        out_mem = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            buffer_type=ttnn.BufferType.L1,
        )
        prog = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(grid_x, grid_y),
            in0_block_w=in0_block_w,
            out_subblock_h=1,
            out_subblock_w=out_subblock_w(per_core_n),
            out_block_h=per_core_m,
            out_block_w=per_core_n,
            per_core_M=per_core_m,
            per_core_N=per_core_n,
            transpose_mcast=False,
            fused_activation=None,
        )
        return in0_mem, dram_sharded_in1_mem(device, spec, in1_layout), out_mem, prog

    return _builder


def cfg_mcast1d(spec: MatmulSweepSpec, grid_x: int, grid_y: int, per_core_n: int, out_sw: int, in0_block_w: int):
    def _builder(device):
        prog = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(grid_x, grid_y),
            in0_block_w=in0_block_w,
            out_subblock_h=1,
            out_subblock_w=out_sw,
            per_core_M=1,
            per_core_N=per_core_n,
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=False,
        )
        return ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG, prog

    return _builder


def cfg_auto_default(spec: MatmulSweepSpec):
    def _builder(device):
        return ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, None

    return _builder


def cfg_dram_sharded(
    spec: MatmulSweepSpec,
    num_compute_cores: int,
    per_core_n: int,
    in0_block_w: int,
    num_dram_banks: int | None = None,
):
    def _builder(device):
        in0_grid = core_range_set_for_num_cores(num_compute_cores)
        in0_mem = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            buffer_type=ttnn.BufferType.L1,
            shard_spec=ttnn.ShardSpec(
                in0_grid,
                [spec.m, spec.k // num_compute_cores],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        out_grid = core_range_set_for_num_cores(num_compute_cores)
        out_mem = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            buffer_type=ttnn.BufferType.L1,
            shard_spec=ttnn.ShardSpec(
                out_grid,
                [spec.m, TILE * per_core_n],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        prog = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=in0_block_w,
            per_core_M=spec.m // TILE,
            per_core_N=per_core_n,
            fused_activation=None,
        )
        return in0_mem, dram_width_in1_mem(device, spec, num_dram_banks), out_mem, prog

    return _builder


def cfg_exact_gate_up(spec: MatmulSweepSpec):
    def _builder(device):
        in0_grid = core_range_set_for_num_cores(16)
        out_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))})
        in0_mem = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            buffer_type=ttnn.BufferType.L1,
            shard_spec=ttnn.ShardSpec(in0_grid, [spec.m, spec.k // 16], ttnn.ShardOrientation.ROW_MAJOR),
        )
        out_mem = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            buffer_type=ttnn.BufferType.L1,
            shard_spec=ttnn.ShardSpec(out_grid, [spec.m, TILE * 10], ttnn.ShardOrientation.ROW_MAJOR),
        )
        prog = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 7),
            in0_block_w=3,
            out_subblock_h=1,
            out_subblock_w=5,
            per_core_M=1,
            per_core_N=10,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        )
        return in0_mem, dram_width_in1_mem(device, spec), out_mem, prog

    return _builder


def build_configs(
    spec,
    width_combos,
    height_combos,
    block_combos,
    mcast_combos=(),
    dram_configs=(),
    exact_gate=False,
    include_auto_default=False,
):
    configs: list[tuple[str, Callable]] = []
    if include_auto_default:
        configs.append(("auto_mcast1d_default", cfg_auto_default(spec)))
    for dram_config in dram_configs:
        name, cores, pcn, ibw, *rest = dram_config
        num_dram_banks = rest[0] if rest else None
        configs.append((name, cfg_dram_sharded(spec, cores, pcn, ibw, num_dram_banks)))
    if exact_gate:
        configs.append(("exact_trial_in0l1_16cores_in1dram_outl1_8x7_pcn10", cfg_exact_gate_up(spec)))
    for gx, gy, pcn, ibw in width_combos:
        configs.append((f"width_sharded_{gx}x{gy}_pcn{pcn}_ibw{ibw}", cfg_width_explicit(spec, gx, gy, pcn, ibw)))
    in1_variants = ("interleaved", "width", "height", "block")
    block_in1_variants = ("interleaved", "width", "block")
    for gx, gy, pcn, ibw in height_combos:
        for in1 in in1_variants:
            configs.append(
                (
                    f"height_sharded_{gx}x{gy}_pcn{pcn}_ibw{ibw}_in1{in1}",
                    cfg_height_explicit(spec, gx, gy, pcn, ibw, in1),
                )
            )
    for gx, gy, pcn, ibw in block_combos:
        for in1 in block_in1_variants:
            configs.append(
                (
                    f"block_sharded_{gx}x{gy}_pcn{pcn}_ibw{ibw}_in1{in1}",
                    cfg_block_explicit(spec, gx, gy, pcn, ibw, in1),
                )
            )
    for gx, gy, pcn, osw, ibw in mcast_combos:
        configs.append((f"mcast1d_{gx}x{gy}_pcn{pcn}_osw{osw}", cfg_mcast1d(spec, gx, gy, pcn, osw, ibw)))
    return configs


def run_matmul_sweep_test(
    device,
    spec: MatmulSweepSpec,
    config_name: str,
    cfg_builder: Callable,
    skip_configs,
    xfail_configs,
    allow_pcc_failure_configs=(),
):
    if config_name in skip_configs:
        pytest.skip(skip_configs[config_name])
    if config_name in xfail_configs and config_name == "exact_trial_in0l1_16cores_in1dram_outl1_8x7_pcn10":
        pytest.xfail(xfail_configs[config_name])

    torch.manual_seed(0)
    torch_a = torch.randn((1, 1, spec.m, spec.k), dtype=torch.bfloat16) * 0.1
    torch_b = torch.randn((1, 1, spec.k, spec.n), dtype=torch.bfloat16) * 0.1
    torch_ref = torch.matmul(torch_a.to(torch.float32), torch_b.to(torch.float32))

    in0_mem, in1_mem, out_mem, prog = cfg_builder(device)
    input_a = ttnn.from_torch(
        torch_a,
        dtype=spec.in0_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in0_mem,
    )
    input_b = ttnn.from_torch(
        torch_b,
        dtype=spec.in1_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in1_mem,
    )

    def _run():
        kwargs = {
            "memory_config": out_mem,
            "dtype": spec.out_dtype,
            "compute_kernel_config": compute_kernel(spec),
        }
        if prog is not None:
            kwargs["program_config"] = prog
        return ttnn.matmul(input_a, input_b, **kwargs)

    last_out = None
    try:
        for _ in range(NUM_WARMUP):
            out = _run()
            ttnn.synchronize_device(device)
            ttnn.deallocate(out)
        for _ in range(NUM_ITERS):
            if last_out is not None:
                ttnn.deallocate(last_out)
            last_out = _run()
        ttnn.synchronize_device(device)
        out_torch = ttnn.to_torch(last_out).to(torch.float32)
        passed, pcc = comp_pcc(torch_ref, out_torch, spec.pcc)
        if config_name in xfail_configs:
            pytest.xfail(xfail_configs[config_name])
        if not passed and config_name not in allow_pcc_failure_configs:
            assert passed, f"{config_name} PCC {float(pcc):.4f} below target {spec.pcc}"
    finally:
        if last_out is not None:
            ttnn.deallocate(last_out)
        ttnn.deallocate(input_a)
        ttnn.deallocate(input_b)
