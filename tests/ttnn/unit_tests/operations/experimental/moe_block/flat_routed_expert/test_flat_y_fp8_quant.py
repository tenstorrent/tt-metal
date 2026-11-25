# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""FP8_Y_PLAN.md M2: the flat expert's fp8 y quantize sequence (flat_routed_expert/device/kernels/y_fp8_quant.hpp) on
one core, bit exact against a torch golden.

The test kernel (kernels/y_fp8_quant_compute.cpp) runs it as the down compute does for GLM: fp32 DEST in half sync,
each row tile in passes of 3 tiles (a 96-value block a pass), quantized on the PACK thread's SFPU, pack-untilized into
an e4m3 CB; each row's block scales into a 16-byte cell (fp32 scale a block, the rest 0).

Golden: per (row, block of 96): s = 2^ceil(log2(fp32(max(amax, 1e-4) * fp32(1/448)))), q = e4m3_rne(y / s)
(torch.float8_e4m3fn: round to nearest even, e4m3 subnormals included)."""

import pytest
import torch
from loguru import logger

import ttnn

TILE = 32
COLUMNS = 6  # tiles a row tile (GLM's down core)
PASS = 3  # tiles a pass (a block)
BLOCK = PASS * TILE
KERNELS = "tests/ttnn/unit_tests/operations/experimental/moe_block/flat_routed_expert/kernels"


def golden(y, block=BLOCK):
    """y [rows, COLUMNS * 32] fp32 -> (codes uint8 [rows, cols], scales fp32 [rows, blocks])."""
    rows = y.shape[0]
    b = y.reshape(rows, -1, block)
    amax = b.abs().amax(-1)
    v = torch.clamp(amax, min=torch.tensor(1e-4, dtype=torch.float32)) * torch.tensor(1 / 448, dtype=torch.float32)
    bits = v.view(torch.int32)
    scale = ((bits + 0x7FFFFF) & 0x7F800000).view(torch.float32)
    q = b / scale[..., None]
    codes = q.to(torch.float8_e4m3fn).view(torch.uint8).reshape(rows, -1)
    return codes, scale


def make_input(row_tiles, seed=0):
    g = torch.Generator().manual_seed(seed)
    rows = row_tiles * TILE
    y = torch.randn(rows, COLUMNS * TILE, generator=g)
    # per (row, block) magnitudes over a wide range: 2^-30 .. 2^20
    mag = torch.pow(2.0, torch.randint(-30, 21, (rows, COLUMNS // PASS), generator=g).float())
    y = (y.reshape(rows, -1, BLOCK) * mag[..., None]).reshape(rows, -1)
    # special rows: all zero, one outlier, amax exactly 448 x 2^k, amax just above / below, tiny (under the 1e-4 clamp)
    y[0] = 0
    y[1] = torch.randn(COLUMNS * TILE, generator=g) * 1e-3
    y[1, 5] = 1e4
    y[2, :BLOCK] = torch.linspace(-448, 448, BLOCK)
    y[2, BLOCK:] = torch.linspace(-448 * 4, 448 * 2, BLOCK)
    y[3] = y[2] * (1 + 2**-20)
    y[4] = y[2] * (1 - 2**-20)
    y[5] = torch.randn(COLUMNS * TILE, generator=g) * 1e-7
    # every e4m3 rounding tie: values k + 0.5 ulp
    y[6, :BLOCK] = (torch.arange(BLOCK).float() + 0.5) * (448 / 128)
    return y


def run(device, y, quantize=True, unpack_to_dest=True, pass_columns=PASS, out_bf16=False, block_tiles=PASS):
    rows = y.shape[0]
    row_tiles = rows // TILE
    t_in = ttnn.from_torch(
        y, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    t_y = ttnn.from_torch(
        torch.zeros(rows, COLUMNS * TILE * (2 if out_bf16 else 1), dtype=torch.uint8),
        dtype=ttnn.uint8,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cell_words = 4 if COLUMNS // block_tiles <= 4 else 8  # (a 16- or 32-byte cell a row)
    t_s = ttnn.from_torch(
        torch.full((row_tiles, 32 * cell_words), 0x7FC00001, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # the out CB's row stride: an odd pass width into e4m3 writes a tile past the pass (y_fp8_quant.hpp)
    stride = COLUMNS + (pass_columns if pass_columns % 2 and not out_bf16 else 0)
    core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])

    def cb(idx, fmt, page, pages):
        return ttnn.CBDescriptor(
            total_size=pages * page,
            core_ranges=core,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt, page_size=page)],
        )

    cbs = [
        cb(0, ttnn.float32, 4096, 2 * COLUMNS),
        cb(16, ttnn.bfloat16 if out_bf16 else ttnn.DataType.FP8_E4M3, 2048 if out_bf16 else 1024, 2 * stride),
        cb(17, ttnn.float32, 32 * cell_words * 4, 2),
    ]
    reader_rt = ttnn.RuntimeArgs()
    reader_rt[0][0] = [t_in.buffer_address(), row_tiles * COLUMNS, 0]
    writer_rt = ttnn.RuntimeArgs()
    writer_rt[0][0] = [t_y.buffer_address(), t_s.buffer_address()]
    compute_config = ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True, dst_full_sync_en=False)
    # (the fp32 input straight into DEST: through SrcA it would lose mantissa bits)
    if unpack_to_dest:
        compute_config.unpack_to_dest_mode = [ttnn.UnpackToDestMode.UnpackToDestFp32] + [
            ttnn.UnpackToDestMode.Default
        ] * 63
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source="ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
            core_ranges=core,
            compile_time_args=ttnn.TensorAccessorArgs(t_in).get_compile_time_args(),
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{KERNELS}/y_fp8_quant_writer.cpp",
            core_ranges=core,
            compile_time_args=[
                row_tiles,
                stride,
                COLUMNS * TILE * (2 if out_bf16 else 1),
                stride * TILE * (2 if out_bf16 else 1),
                32 * cell_words * 4,
            ]
            + ttnn.TensorAccessorArgs(t_y).get_compile_time_args()
            + ttnn.TensorAccessorArgs(t_s).get_compile_time_args(),
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{KERNELS}/y_fp8_quant_compute.cpp",
            core_ranges=core,
            compile_time_args=[
                row_tiles,
                COLUMNS,
                pass_columns,
                int(quantize),
                int(out_bf16),
                stride,
                block_tiles,
                cell_words,
            ],
            runtime_args=[],
            config=compute_config,
        ),
    ]
    ttnn.generic_op([t_in, t_y, t_s], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    codes = ttnn.to_torch(t_y).to(torch.uint8)
    cells = ttnn.to_torch(t_s).to(torch.int32).reshape(rows, cell_words)
    return codes, cells


@pytest.mark.parametrize("block_tiles", [3, 1], ids=["1x96", "1x32"])
@pytest.mark.parametrize("row_tiles", [1, 4])
def test_y_fp8_quant(device, row_tiles, block_tiles):
    if ttnn.get_arch_name() != "blackhole":
        pytest.skip("Blackhole only")
    y = make_input(row_tiles)
    block = block_tiles * TILE
    want_codes, want_scales = golden(y, block)
    codes, cells = run(device, y, block_tiles=block_tiles)
    nb = COLUMNS // block_tiles

    got_scales = cells[:, :nb].view(torch.float32)
    bad_s = (got_scales.view(torch.int32) != want_scales.view(torch.int32)).nonzero()
    assert (cells[:, nb:] == 0).all(), "the cells' unused words are not 0"
    assert (
        len(bad_s) == 0
    ), f"{len(bad_s)} scales differ, first {bad_s[:4].tolist()}: got {got_scales[tuple(bad_s[0])]} want {want_scales[tuple(bad_s[0])]}"

    diff = (codes != want_codes).nonzero()
    if len(diff):
        r, c = diff[0].tolist()
        q = (y[r, c] / want_scales[r, c // block]).item()
        logger.warning(
            f"{len(diff)} codes differ; first ({r}, {c}): y {y[r, c].item()} q {q} got 0x{codes[r, c]:02x} want 0x{want_codes[r, c]:02x}"
        )
        # the codes that differ: by how much (as e4m3 values)
        f8 = lambda t: t.view(torch.float8_e4m3fn).float()  # noqa: E731
        for r, c in diff[:16].tolist():
            logger.info(
                f"  ({r},{c}) q {(y[r, c] / want_scales[r, c // block]).item():.6g} got {f8(codes[r, c:c+1]).item()} want {f8(want_codes[r, c:c+1]).item()}"
            )
    logger.info(
        f"M2 1x{block} row_tiles {row_tiles}: {codes.numel()} codes, {len(diff)} differ; {got_scales.numel()} scales exact"
    )
    assert len(diff) == 0
