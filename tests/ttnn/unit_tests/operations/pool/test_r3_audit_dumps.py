# Shadow (not checked in): byte dumps for the cross-family audit (ops/blackhole/AUDIT_RESULTS_AND_DRAINS.md), run on a
# farm of main and a farm of the PR head; tools/audit_cmp.py compares the dumps.
#   T2 (#59143): the tilizeA_B block unpack of max_pool2d and avg_pool2d: every bf16 pattern through 1, 2, 4, 9 and 16
#      rows per window, 16 and 32 channels, 16-bit DEST, and 32-bit DEST for avg_pool2d (max_pool2d takes bf16 only).
#   T3 (#59143): fast tilize of a bf16 input (ttnn.tilize), every bf16 pattern at 1 to 8 and 16 tiles per row.
#   U1 (#59149): the plain pack untilize (generic op): every bf16 pattern at 1 to 4 tiles per row with a 16-bit DEST;
#      Float32, Int32 and UInt32 with a 32-bit DEST at 1, 2, 3, 4, 6, 8 tiles (blocks of up to 4), datums numbered by
#      position plus seeded random bit patterns.
# AUDIT_DUMP_DIR / AUDIT_TAG select where each run writes its outputs.
import math
import os

import pytest
import torch
import ttnn

DUMP = os.environ.get("AUDIT_DUMP_DIR")
TAG = os.environ.get("AUDIT_TAG", "untagged")
TILE = 32


def _save(name, t):
    if DUMP:
        os.makedirs(f"{DUMP}/{TAG}", exist_ok=True)
        torch.save(t.contiguous().clone(), f"{DUMP}/{TAG}/{name}.pt")


def _bf16_patterns(n):
    return (torch.arange(n, dtype=torch.int32) % 65536).to(torch.int16).view(torch.bfloat16)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("dest32", [False, True], ids=["dest16", "dest32"])
@pytest.mark.parametrize("channels", [16, 32])
@pytest.mark.parametrize("kernel", [(1, 1), (1, 2), (2, 2), (3, 3), (4, 4)], ids=["k1", "k2", "k4", "k9", "k16"])
@pytest.mark.parametrize("pool", ["max", "avg"])
def test_t2_pool(device, pool, kernel, channels, dest32):
    kh, kw = kernel
    windows = math.ceil(65536 / (kh * kw * channels))
    out_w = 32
    out_h = math.ceil(windows / out_w)
    h, w = out_h * kh, out_w * kw
    x = _bf16_patterns(h * w * channels).reshape(1, 1, h * w, channels)
    if dest32 and pool == "max":
        pytest.skip("max_pool2d takes a bf16 input only, so its DEST is 16-bit")
    t = ttnn.from_torch(x, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    args = dict(input_tensor=t, batch_size=1, input_h=h, input_w=w, channels=channels, kernel_size=kernel,
                stride=kernel, padding=[0, 0, 0, 0], dtype=ttnn.bfloat16, output_layout=ttnn.ROW_MAJOR_LAYOUT)
    if pool == "max":
        out = ttnn.max_pool2d(**args, dilation=(1, 1))
    else:
        cc = ttnn.init_device_compute_kernel_config(device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                    fp32_dest_acc_en=dest32)
        out = ttnn.avg_pool2d(**args, compute_kernel_config=cc)
    res = ttnn.to_torch(out).view(torch.int16)
    _save(f"t2_{pool}_k{kh * kw}_c{channels}_{'dest32' if dest32 else 'dest16'}", res)


@pytest.mark.parametrize("width", [1, 2, 3, 4, 5, 6, 7, 8, 16])
def test_t3_fast_tilize(device, width):
    rows = math.ceil(65536 / (width * TILE * TILE)) * TILE
    x = _bf16_patterns(rows * width * TILE).reshape(1, 1, rows, width * TILE)
    t = ttnn.from_torch(x, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    y = ttnn.tilize(t)
    res = ttnn.to_torch(y).view(torch.int16)
    _save(f"t3_w{width}", res)
    # Golden: the same bits (a layout change); reported, not asserted (DEST may flush denormals on both trees).
    print(f"t3_w{width}: {(res != x.view(torch.int16)).sum().item()} datums differ from the input bits")


READER = """
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(1);
    constexpr auto src_args = TensorAccessorArgs<2>();
    const auto src = TensorAccessor(src_args, src_addr);
    const uint32_t tile_bytes = get_tile_size(cb_in);
    for (uint32_t t = 0; t < num_tiles; ++t) {
        cb_reserve_back(cb_in, 1);
        noc_async_read(src.get_noc_addr(t), get_write_ptr(cb_in), tile_bytes);
        noc_async_read_barrier();
        cb_push_back(cb_in, 1);
    }
}
"""

WRITER = """
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t wait_pages = get_compile_time_arg_val(1);
    constexpr uint32_t num_rows = get_compile_time_arg_val(2);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(3);
    constexpr auto dst_args = TensorAccessorArgs<4>();
    const auto dst = TensorAccessor(dst_args, dst_addr);
    cb_wait_front(cb_out, wait_pages);
    const uint32_t l1_addr = get_read_ptr(cb_out);
    for (uint32_t r = 0; r < num_rows; ++r) {
        noc_async_write(l1_addr + r * row_bytes, dst.get_noc_addr(r), row_bytes);
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out, wait_pages);
}
"""

PACK_COMPUTE = """
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack_untilize.h"

void kernel_main() {
    constexpr uint32_t width = get_compile_time_arg_val(0);
    constexpr uint32_t num_rows = get_compile_time_arg_val(1);
    constexpr uint32_t block = get_compile_time_arg_val(2);
    compute_kernel_hw_startup(0, 16);
    pack_untilize_init<block, width>(0, 16);
    for (uint32_t r = 0; r < num_rows; ++r) {
        cb_reserve_back(16, width);
        for (uint32_t b = 0; b < width / block; ++b) {
            cb_wait_front(0, block);
            pack_untilize_block<block, width>(0, 1, 16, b);
            cb_pop_front(0, block);
        }
        cb_push_back(16, width);
    }
    pack_untilize_uninit(16);
}
"""


def _core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def _cb(cb_id, dtype, num_pages):
    page_size = ttnn.tile_size(dtype)
    return ttnn.CBDescriptor(
        total_size=page_size * num_pages,
        core_ranges=_core(),
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=cb_id, data_format=dtype, page_size=page_size)],
    )


def _pack_untilize(device, values, dtype, width, block, dest32):
    num_rows, row_datums = values.shape
    num_tiles = (num_rows // TILE) * width
    tt_in = ttnn.from_torch(values, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_out = ttnn.from_torch(torch.zeros_like(values), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    reader_rt = ttnn.RuntimeArgs()
    reader_rt[0][0] = [tt_in.buffer_address()]
    writer_rt = ttnn.RuntimeArgs()
    writer_rt[0][0] = [tt_out.buffer_address()]
    cc = ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=dest32)
    if dest32:
        modes = [ttnn.UnpackToDestMode.Default] * 64
        modes[0] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cc.unpack_to_dest_mode = modes
    datum_bytes = 4 if dest32 else 2
    kernels = [
        ttnn.KernelDescriptor(kernel_source=READER, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                              core_ranges=_core(),
                              compile_time_args=[0, num_tiles, *ttnn.TensorAccessorArgs(tt_in).get_compile_time_args()],
                              runtime_args=reader_rt, config=ttnn.ReaderConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=PACK_COMPUTE, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                              core_ranges=_core(), compile_time_args=[width, num_rows // TILE, block], config=cc),
        ttnn.KernelDescriptor(kernel_source=WRITER, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                              core_ranges=_core(),
                              compile_time_args=[16, num_tiles, num_rows, row_datums * datum_bytes,
                                                 *ttnn.TensorAccessorArgs(tt_out).get_compile_time_args()],
                              runtime_args=writer_rt, config=ttnn.WriterConfigDescriptor()),
    ]
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[],
                                     cbs=[_cb(0, dtype, num_tiles), _cb(16, dtype, num_tiles)])
    ttnn.generic_op([tt_in, tt_out], program)
    return ttnn.to_torch(tt_out)


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_u1_pack_untilize_bf16(device, width):
    rows = math.ceil(65536 / (width * TILE * TILE)) * TILE
    x = _bf16_patterns(rows * width * TILE).reshape(rows, width * TILE)
    res = _pack_untilize(device, x, ttnn.bfloat16, width, width, False).view(torch.int16)
    _save(f"u1_bf16_w{width}", res)
    print(f"u1_bf16_w{width}: {(res != x.view(torch.int16)).sum().item()} datums differ from the input bits")


@pytest.mark.parametrize("width", [1, 2, 3, 4, 6, 8])
@pytest.mark.parametrize("dtype", ["float32", "int32", "uint32"])
def test_u1_pack_untilize_32bit(device, dtype, width):
    rows = 4 * TILE
    n = rows * width * TILE
    gen = torch.Generator().manual_seed(width)
    # The upper half of the datums carry their own position (placement), the lower half random bit patterns.
    pos = torch.arange(n, dtype=torch.int64) * 2654435761 % (1 << 31)
    rnd = torch.randint(0, 1 << 31, (n,), generator=gen, dtype=torch.int64)
    bits = torch.where(torch.arange(n) % 2 == 0, pos, rnd).to(torch.int32).reshape(rows, width * TILE)
    if dtype == "float32":
        x, tt_dtype = bits.view(torch.float32), ttnn.float32
        x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
    elif dtype == "int32":
        x, tt_dtype = bits, ttnn.int32
    else:
        x, tt_dtype = bits.abs(), ttnn.uint32
    block = max(b for b in range(1, 5) if width % b == 0)
    res = _pack_untilize(device, x, tt_dtype, width, block, True)
    res_bits = res.view(torch.int32) if res.dtype in (torch.float32, torch.int32) else res.to(torch.int32)
    _save(f"u1_{dtype}_w{width}", res_bits)
    x_bits = x.view(torch.int32) if x.dtype in (torch.float32, torch.int32) else x.to(torch.int32)
    print(f"u1_{dtype}_w{width}: {(res_bits != x_bits).sum().item()} datums differ from the input bits")


# U2 (#59149): an Fp8_e4m3 untilize of a full-width block of an odd tile count, one tile row. The reader fills the
# 32 * width bytes after the output block (the next CB page) with 0xAA before the input arrives; the writer returns the
# 32 output rows and those bytes as a 33rd row. Main's per-row Last flushed 32 zero bytes after the last row there.
U2_READER = """
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t block_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t tail_bytes = get_compile_time_arg_val(4);
    constexpr auto src_args = TensorAccessorArgs<5>();
    const auto src = TensorAccessor(src_args, src_addr);
    volatile tt_l1_ptr uint8_t* tail = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(get_write_ptr(cb_out) + block_bytes);
    for (uint32_t i = 0; i < tail_bytes; ++i) {
        tail[i] = 0xAA;
    }
    const uint32_t tile_bytes = get_tile_size(cb_in);
    for (uint32_t t = 0; t < num_tiles; ++t) {
        cb_reserve_back(cb_in, 1);
        noc_async_read(src.get_noc_addr(t), get_write_ptr(cb_in), tile_bytes);
        noc_async_read_barrier();
        cb_push_back(cb_in, 1);
    }
}
"""

U2_COMPUTE = """
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack_untilize.h"

void kernel_main() {
    constexpr uint32_t width = get_compile_time_arg_val(0);
    compute_kernel_hw_startup(0, 16);
    pack_untilize_init<width, width>(0, 16);
    cb_reserve_back(16, width);
    cb_wait_front(0, width);
    pack_untilize_block<width, width>(0, 1, 16, 0);
    cb_pop_front(0, width);
    cb_push_back(16, width);
    pack_untilize_uninit(16);
}
"""


@pytest.mark.parametrize("width", [1, 3, 5, 7])
def test_u2_fp8_past_block(device, width):
    dest32 = True  # an Fp8_e4m3 CB needs a 32-bit DEST
    gen = torch.Generator().manual_seed(width)
    exp = torch.randint(1, 15, (TILE, width * TILE), generator=gen)
    man = torch.randint(0, 8, (TILE, width * TILE), generator=gen)
    sign = torch.randint(0, 2, (TILE, width * TILE), generator=gen)
    values = (sign * 128 + exp * 8 + man).to(torch.uint8).view(torch.float8_e4m3fn).to(torch.bfloat16)
    tt_in = ttnn.from_torch(values, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    row_bytes = width * TILE
    tt_out = ttnn.from_torch(torch.zeros(TILE + 1, row_bytes, dtype=torch.uint8), dtype=ttnn.uint8,
                             layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    fp8_tile = ttnn.tile_size(ttnn.fp8_e4m3)
    reader_rt = ttnn.RuntimeArgs()
    reader_rt[0][0] = [tt_in.buffer_address()]
    writer_rt = ttnn.RuntimeArgs()
    writer_rt[0][0] = [tt_out.buffer_address()]
    cc = ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=dest32)
    kernels = [
        ttnn.KernelDescriptor(kernel_source=U2_READER, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                              core_ranges=_core(),
                              compile_time_args=[0, width, 16, width * fp8_tile, row_bytes,
                                                 *ttnn.TensorAccessorArgs(tt_in).get_compile_time_args()],
                              runtime_args=reader_rt, config=ttnn.ReaderConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=U2_COMPUTE, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                              core_ranges=_core(), compile_time_args=[width], config=cc),
        ttnn.KernelDescriptor(kernel_source=WRITER, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                              core_ranges=_core(),
                              compile_time_args=[16, width, TILE + 1, row_bytes,
                                                 *ttnn.TensorAccessorArgs(tt_out).get_compile_time_args()],
                              runtime_args=writer_rt, config=ttnn.WriterConfigDescriptor()),
    ]
    program = ttnn.ProgramDescriptor(
        kernels=kernels, semaphores=[],
        cbs=[_cb(0, ttnn.bfloat16, width), _cb(16, ttnn.fp8_e4m3, width + 1)],
    )
    ttnn.generic_op([tt_in, tt_out], program)
    res = ttnn.to_torch(tt_out)
    _save(f"u2_w{width}", res)
    expected = values.to(torch.float8_e4m3fn).view(torch.uint8)
    print(f"u2_w{width}: rows {(res[:TILE] != expected).sum().item()} bytes differ; after the block "
          f"{(res[TILE] == 0xAA).sum().item()} of {row_bytes} bytes still 0xAA, {(res[TILE] == 0).sum().item()} zero")


# U4 (#59149): pack_untilize_dest with narrow_row and 24 datums per row, one bf16 tile with distinct datums. Row r of
# the output must be row r of the tile, columns 0 to 23.
U4_COMPUTE = """
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack_untilize.h"

void kernel_main() {
    constexpr uint32_t row_datums = get_compile_time_arg_val(0);
    compute_kernel_hw_startup(0, 16);
    copy_tile_to_dst_init_short(0);
    pack_untilize_dest_init<1, 1, true, row_datums>(16);
    cb_wait_front(0, 1);
    cb_reserve_back(16, 1);
    tile_regs_acquire();
    copy_tile(0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_untilize_dest<1, 1, false, true, row_datums>(16);
    tile_regs_release();
    cb_push_back(16, 1);
    cb_pop_front(0, 1);
    pack_untilize_uninit(16);
}
"""


@pytest.mark.parametrize("row_datums", [8, 16, 17, 20, 24, 31])
def test_u4_narrow_row(device, row_datums):
    tile = (torch.arange(TILE * TILE, dtype=torch.int32) + 0x3F80).to(torch.int16).view(torch.bfloat16)
    tile = tile.reshape(TILE, TILE)
    tt_in = ttnn.from_torch(tile, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out_datums = TILE * row_datums
    tt_out = ttnn.from_torch(torch.zeros(1, out_datums, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                             layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    reader_rt = ttnn.RuntimeArgs()
    reader_rt[0][0] = [tt_in.buffer_address()]
    writer_rt = ttnn.RuntimeArgs()
    writer_rt[0][0] = [tt_out.buffer_address()]
    kernels = [
        ttnn.KernelDescriptor(kernel_source=READER, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                              core_ranges=_core(),
                              compile_time_args=[0, 1, *ttnn.TensorAccessorArgs(tt_in).get_compile_time_args()],
                              runtime_args=reader_rt, config=ttnn.ReaderConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=U4_COMPUTE, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                              core_ranges=_core(), compile_time_args=[row_datums], config=ttnn.ComputeConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=WRITER, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                              core_ranges=_core(),
                              compile_time_args=[16, 1, 1, out_datums * 2,
                                                 *ttnn.TensorAccessorArgs(tt_out).get_compile_time_args()],
                              runtime_args=writer_rt, config=ttnn.WriterConfigDescriptor()),
    ]
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[],
                                     cbs=[_cb(0, ttnn.bfloat16, 1), _cb(16, ttnn.bfloat16, 1)])
    ttnn.generic_op([tt_in, tt_out], program)
    res = ttnn.to_torch(tt_out).view(torch.int16).reshape(TILE, row_datums)
    _save(f"u4_r{row_datums}", res)
    expected = tile.view(torch.int16)[:, :row_datums]
    wrong = (res != expected)
    print(f"u4_r{row_datums}: {wrong.sum().item()} of {res.numel()} datums wrong, in columns "
          f"{sorted(set(wrong.nonzero()[:, 1].tolist()))[:4]}...")


# U1 / U4 narrow rows (#59149): the narrow pack at every width of 1 to 31 datums per row, bf16, Float32 and Fp8_e4m3
# outputs, one tile with distinct datums (Fp8_e4m3: codes distinct within each row). The expected output is the tile's
# first row_datums columns, row after row with no gap. Main steps rows in whole 16-byte units, so a row whose byte size is
# not a multiple of 16 lands at the wrong offset, and rows over 16 datums read the wrong face.
@pytest.mark.parametrize("row_datums", list(range(1, 32)))
@pytest.mark.parametrize("dtype", ["bf16", "fp32", "fp8", "bf16d32"])
def test_u5_narrow_rows(device, dtype, row_datums):
    idx = torch.arange(TILE * TILE, dtype=torch.int32).reshape(TILE, TILE)
    if dtype == "fp8":
        # Normal Fp8_e4m3 codes 0x08 to 0x77 and their negatives: 224 values, distinct along a row and shifted per row.
        code = ((idx % TILE) * 7 + (idx // TILE) * 3) % 224
        code = torch.where(code < 112, code + 8, code - 112 + 8 + 128)
        tile = code.to(torch.uint8).view(torch.float8_e4m3fn).to(torch.bfloat16)
        out_tt, out_t, dsize, dest32 = ttnn.fp8_e4m3, torch.uint8, 1, True
    else:
        tile = (idx + 0x3F80).to(torch.int16).view(torch.bfloat16)
        out_tt, out_t, dsize, dest32 = (ttnn.float32, torch.float32, 4, True) if dtype == "fp32" else (
            ttnn.bfloat16, torch.bfloat16, 2, dtype == "bf16d32")
    tt_in = ttnn.from_torch(tile, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out_bytes = TILE * row_datums * dsize
    holder_dtype = ttnn.uint8 if dtype == "fp8" else out_tt
    tt_out = ttnn.from_torch(torch.zeros(1, TILE * row_datums, dtype=out_t), dtype=holder_dtype,
                             layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    reader_rt = ttnn.RuntimeArgs()
    reader_rt[0][0] = [tt_in.buffer_address()]
    writer_rt = ttnn.RuntimeArgs()
    writer_rt[0][0] = [tt_out.buffer_address()]
    kernels = [
        ttnn.KernelDescriptor(kernel_source=READER, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                              core_ranges=_core(),
                              compile_time_args=[0, 1, *ttnn.TensorAccessorArgs(tt_in).get_compile_time_args()],
                              runtime_args=reader_rt, config=ttnn.ReaderConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=U4_COMPUTE, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                              core_ranges=_core(), compile_time_args=[row_datums],
                              config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=dest32)),
        ttnn.KernelDescriptor(kernel_source=WRITER, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                              core_ranges=_core(),
                              compile_time_args=[16, 1, 1, out_bytes,
                                                 *ttnn.TensorAccessorArgs(tt_out).get_compile_time_args()],
                              runtime_args=writer_rt, config=ttnn.WriterConfigDescriptor()),
    ]
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[],
                                     cbs=[_cb(0, ttnn.bfloat16, 1), _cb(16, out_tt, 1)])
    ttnn.generic_op([tt_in, tt_out], program)
    res = ttnn.to_torch(tt_out)
    if dtype == "fp8":
        got = res.to(torch.uint8).reshape(TILE, row_datums).to(torch.int32)
        exp = tile.to(torch.float8_e4m3fn).view(torch.uint8)[:, :row_datums].to(torch.int32)
    elif dtype == "fp32":
        got = res.view(torch.int32).reshape(TILE, row_datums)
        exp = tile.float().view(torch.int32)[:, :row_datums]
    else:
        got = res.view(torch.int16).reshape(TILE, row_datums).to(torch.int32)
        exp = tile.view(torch.int16)[:, :row_datums].to(torch.int32)
    _save(f"u5_{dtype}_r{row_datums}", got)
    print(f"u5_{dtype}_r{row_datums}: {(got != exp).sum().item()} of {got.numel()} datums differ from the golden")


# The height-sharded row-major transpose with H % 32 = 24 and a shard of 24 x 128 (3072 elements, a multiple of 1024):
# the sharded transpose factory packs its rows with narrow_row at 24 datums.
def test_u4_transpose_24x128(device):
    torch.manual_seed(0)
    x = torch.randn(1, 1, 24, 128).bfloat16()
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                            ttnn.ShardSpec(grid, [24, 128], ttnn.ShardOrientation.ROW_MAJOR))
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem)
    y = ttnn.to_torch(ttnn.transpose(t, -2, -1))
    _save("u4_transpose_24x128", y.view(torch.int16))
    print(f"u4_transpose_24x128: {(y != x.transpose(-2, -1)).sum().item()} of {y.numel()} datums differ from torch")


# U6: the same transpose at H % 32 = 5, 12, 21, 24, 29 in bf16 (16-bit DEST) and Float32 (32-bit DEST,
# Fp32Mode::Lossless); W makes the shard a multiple of 1024 elements, so the sharded row-major factory runs.
@pytest.mark.parametrize("h", [5, 12, 21, 24, 29])
@pytest.mark.parametrize("dtype", ["bf16", "fp32"])
def test_u6_transpose_narrow(device, dtype, h):
    torch.manual_seed(0)
    tdt, ttdt = (torch.float32, ttnn.float32) if dtype == "fp32" else (torch.bfloat16, ttnn.bfloat16)
    w = max(128, 1024 // math.gcd(h, 1024))
    x = torch.randn(1, 1, h, w).to(tdt)
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                            ttnn.ShardSpec(grid, [h, w], ttnn.ShardOrientation.ROW_MAJOR))
    t = ttnn.from_torch(x, dtype=ttdt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem)
    y = ttnn.to_torch(ttnn.transpose(t, -2, -1))
    _save(f"u6_transpose_{dtype}_{h}", y)
    print(f"u6_transpose_{dtype}_{h}: {(y != x.transpose(-2, -1)).sum().item()} of {y.numel()} datums differ from torch")


# U7 (#59149, U1's block splits): the plain pack untilize in blocks narrower than the row, against main. bf16 with a
# 16-bit DEST, every bf16 pattern; Float32 and Int32 with a 32-bit DEST, datums numbered by position plus seeded random
# bit patterns (as test_u1_pack_untilize_32bit).
@pytest.mark.parametrize("width, block", [(2, 1), (3, 1), (4, 1), (4, 2), (6, 2), (6, 3), (8, 1), (8, 2), (8, 4),
                                          (10, 5), (12, 3), (12, 4), (12, 6), (14, 7), (16, 4), (16, 8), (24, 3),
                                          (24, 6), (24, 8)])
def test_u7_split_bf16(device, width, block):
    rows = math.ceil(65536 / (width * TILE * TILE)) * TILE
    x = _bf16_patterns(rows * width * TILE).reshape(rows, width * TILE)
    res = _pack_untilize(device, x, ttnn.bfloat16, width, block, False).view(torch.int16)
    _save(f"u7_bf16_w{width}_b{block}", res)
    print(f"u7_bf16_w{width}_b{block}: {(res != x.view(torch.int16)).sum().item()} datums differ from the input bits")


@pytest.mark.parametrize("width, block", [(2, 1), (4, 1), (4, 2), (6, 1), (6, 2), (8, 1), (8, 2), (12, 3), (12, 4),
                                          (16, 4)])
@pytest.mark.parametrize("dtype", ["float32", "int32"])
def test_u7_split_32bit(device, dtype, width, block):
    rows = 4 * TILE
    n = rows * width * TILE
    gen = torch.Generator().manual_seed(100 * width + block)
    pos = torch.arange(n, dtype=torch.int64) * 2654435761 % (1 << 31)
    rnd = torch.randint(0, 1 << 31, (n,), generator=gen, dtype=torch.int64)
    bits = torch.where(torch.arange(n) % 2 == 0, pos, rnd).to(torch.int32).reshape(rows, width * TILE)
    if dtype == "float32":
        x = bits.view(torch.float32)
        x, tt_dtype = torch.where(torch.isfinite(x), x, torch.zeros_like(x)), ttnn.float32
    else:
        x, tt_dtype = bits, ttnn.int32
    res = _pack_untilize(device, x, tt_dtype, width, block, True)
    res_bits = res.view(torch.int32)
    _save(f"u7_{dtype}_w{width}_b{block}", res_bits)
    print(f"u7_{dtype}_w{width}_b{block}: {(res_bits != x.view(torch.int32)).sum().item()} datums differ from the input bits")


# U8 (#59149, U1's dense form): the DeepSeek V3 B1 SDPA tail with a row-major output, which packs its L block with the
# dense pack untilize (experimental/sdpa.h:636; the same call as sdpa_reduce_worker.hpp:252 at 16 tiles). 8x32 tiles,
# bf16, 16-bit DEST, one block of 2, 4, 8 or 16 tiles (16 is the production width). m1 = m2 = 0, so both exp factors are
# 1 and l2 = 0: the output is l1 through the op's own math. l1 runs through every bf16 pattern over the calls.
@pytest.mark.parametrize("width", [2, 4, 8, 16])
def test_u8_dense_sdpa_tail(device, width):
    from models.demos.deepseek_v3_b1.micro_ops.sdpa_tail.op import SdpaTailSingleCore

    tile = ttnn.Tile([8, 32])
    l_shape, ms_shape = (8, width * 32), (8, 32)
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    l_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                              ttnn.ShardSpec(grid, l_shape, ttnn.ShardOrientation.ROW_MAJOR))
    ms_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                               ttnn.ShardSpec(grid, ms_shape, ttnn.ShardOrientation.ROW_MAJOR))
    ms = torch.zeros(ms_shape, dtype=torch.bfloat16)
    ms[:, 1] = 1.0
    per_call = l_shape[0] * l_shape[1]
    outs = []
    for c in range(65536 // per_call):
        l1 = _bf16_patterns(65536)[c * per_call:(c + 1) * per_call].reshape(l_shape)
        t = lambda v, lay=ttnn.TILE_LAYOUT, tl=tile, mem=l_mem: ttnn.from_torch(
            v, dtype=ttnn.bfloat16, layout=lay, device=device, memory_config=mem, tile=tl)
        l_out, _ = SdpaTailSingleCore.op(t(l1), t(torch.zeros(l_shape, dtype=torch.bfloat16)), t(ms, mem=ms_mem),
                                         t(ms, mem=ms_mem), t(torch.zeros(l_shape, dtype=torch.bfloat16),
                                                               ttnn.ROW_MAJOR_LAYOUT, None),
                                         t(torch.zeros(ms_shape, dtype=torch.bfloat16), mem=ms_mem),
                                         scale=1.0, block_size=width, num_blocks=1, final_reduction=False, dense=True)
        outs.append(ttnn.to_torch(l_out).view(torch.int16))
    res = torch.stack(outs)
    _save(f"u8_dense_w{width}", res)
    x = _bf16_patterns(65536).reshape(res.shape)
    print(f"u8_dense_w{width}: {(res != x.view(torch.int16)).sum().item()} of {res.numel()} datums differ from l1")
