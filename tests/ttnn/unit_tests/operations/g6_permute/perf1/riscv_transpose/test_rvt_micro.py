# riscv_transpose microbench: one Tensix core, each selected RISC transposes one p44 unit (64 pages x 32
# bf16 values -> 32 rows x 64 values) of its own L1 region `reps` times, one zone RVT-UNIT per unit.
# Timing only (the values are whatever L1 holds); correctness is gated in test_rvt_op.py.
# Read the zones with ttnn/ttnn/operations/g6_permute/perf_experiments/riscv_transpose/rvt_report.py micro.
from pathlib import Path

import pytest
import torch
import ttnn

KDIR = Path(__file__).resolve().parents[7] / "ttnn/ttnn/operations/g6_permute/perf_experiments/riscv_transpose/kernels"

PAGES, VALUES = 64, 32
REPS = 8
IMPLS = {
    "base": 0,
    "line": 1,
    "half": 2,
    "blk": 3,
    "blkv": 4,
    "blks": 5,
    "st_seq": 8,
    "st_row": 9,
    "ld_seq": 10,
    "sth_row": 11,
    "st_pair": 12,
    "st_alt2": 13,
    "ld_col": 14,
}
OptLevel = ttnn._ttnn.program_spec.KernelBuildOptLevel
OPT = {"O2": OptLevel.O2, "O3": OptLevel.O3, "Os": OptLevel.Os}

# id: (riscs, impl, batch, opt[, dst_stride, src_stride]). riscs: B = BRISC, N = NCRISC, 0/1/2 = TRISC.
VARIANTS = {
    "B_blkv8x2": ("B", "blkv", (8 << 8) | 2, "O2"),
    "B_blks8x2": ("B", "blks", (8 << 8) | 2, "O2"),
    "B_blkv8x4": ("B", "blkv", (8 << 8) | 4, "O2"),
    "B_blks8x4": ("B", "blks", (8 << 8) | 4, "O2"),
    "B_blkv8x1": ("B", "blkv", (8 << 8) | 1, "O2"),
    "B_blkv16x1": ("B", "blkv", (16 << 8) | 1, "O2"),
    "B_blkv8x3": ("B", "blkv", (8 << 8) | 3, "O2"),
    "B_blkv16x2": ("B", "blkv", (16 << 8) | 2, "O2"),
    "1_blkv8x2": ("1", "blkv", (8 << 8) | 2, "O2"),
    "1_blkv8x4": ("1", "blkv", (8 << 8) | 4, "O2"),
    "BN_blkv8x2": ("BN", "blkv", (8 << 8) | 2, "O2"),
    "B0_blkv8x2": ("B0", "blkv", (8 << 8) | 2, "O2"),
    "BN012_blkv8x2": ("BN012", "blkv", (8 << 8) | 2, "O2"),
    "BN012_blks8x2": ("BN012", "blks", (8 << 8) | 2, "O2"),
    "BN012_blkv8x4": ("BN012", "blkv", (8 << 8) | 4, "O2"),
    "BN012_blkv16x1": ("BN012", "blkv", (16 << 8) | 1, "O2"),
    "BN012_blkv8x1": ("BN012", "blkv", (8 << 8) | 1, "O2"),
    "N012_blkv8x2": ("N012", "blkv", (8 << 8) | 2, "O2"),
    "B12_blkv8x2": ("B12", "blkv", (8 << 8) | 2, "O2"),
}


def _geom(variant):
    v = VARIANTS[variant]
    dst = v[4] if len(v) > 4 else 128
    src = v[5] if len(v) > 5 else 64
    region = (PAGES * src + VALUES * dst + 15) // 16 * 16
    return dst, src, region


def _ct(variant, region_id, impl, batch):
    dst, src, region = _geom(variant)
    return [region_id, region, REPS, PAGES, VALUES, src, dst, IMPLS[impl], batch]


def build(variant):
    riscs, impl, batch, opt = VARIANTS[variant][:4]
    REGION = _geom(variant)[2]
    core = ttnn.CoreCoord(0, 0)
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    cbs = [
        ttnn.CBDescriptor(
            total_size=5 * REGION,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=REGION)],
        )
    ]
    kernels = []
    for letter, proc, cfg in (
        ("B", ttnn.DataMovementProcessor.RISCV_0, ttnn.WriterConfigDescriptor),
        ("N", ttnn.DataMovementProcessor.RISCV_1, ttnn.ReaderConfigDescriptor),
    ):
        if letter in riscs:
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=str(KDIR / "micro_dm.cpp"),
                    core_ranges=cores,
                    compile_time_args=_ct(variant, 0 if letter == "B" else 1, impl, batch),
                    opt_level=OPT[opt],
                    config=cfg(),
                )
            )
    mask = sum(1 << t for t in range(3) if str(t) in riscs)
    if mask:
        ct = _ct(variant, mask, impl, batch)
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KDIR / "micro_trisc.cpp"),
                core_ranges=cores,
                compile_time_args=ct,
                opt_level=OPT[opt],
                config=ttnn.ComputeConfigDescriptor(),
            )
        )
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_micro(variant, device):
    desc = build(variant)
    dummy = ttnn.from_torch(torch.zeros(32, 32, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    first = ttnn._ttnn.get_device_operation_id()
    for _ in range(3):
        ttnn.generic_op([dummy, dummy], desc)
        ttnn.synchronize_device(device)
    print(f"\nRVT_MICRO {variant} first_op={first} last_op={ttnn._ttnn.get_device_operation_id() - 1}")
