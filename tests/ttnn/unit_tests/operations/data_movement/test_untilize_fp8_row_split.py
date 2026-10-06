# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Pack untilize to Fp8_e4m3 of rows that the even block split leaves in one-tile blocks (tt-metal#59140).

Each row of a one-tile block is its own 32-datum L1 stream, and an Fp8_e4m3 stream reaches L1 in 64-byte units, so a
one-tile block writes 32 bytes of zeros over the next block. The untilize helper and the DeepSeek prefill dispatch and
combine compute kernels split such rows into blocks of two or more tiles; the compute API refuses one-tile blocks of a
wider row at kernel build. The widths are the ones each path used to split into one-tile blocks: no divisor between 2
and 8 for the dispatch and combine kernels, none between 2 and the DEST limit for the helper.
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole

pytestmark = pytest.mark.skipif(not is_blackhole(), reason="Fp8_e4m3 pack untilize is Blackhole only")

TILE = 32
NO_CB = 32
CB_IN = 0
CB_PRE = 1
CB_OUT = 16
ROWS = 32
SENTINEL = 0xFFFFFFFF
DEEPSEEK_KERNELS = "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill"

# Reads the input tiles one at a time; around them, optionally pushes a page of a small uint32 CB carrying a value (the
# dispatch kernel's route signal or the combine kernel's token count) and a final page (the dispatch sentinel).
READER = """
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t cb_pre = get_compile_time_arg_val(2);
    constexpr uint32_t pre_pages = get_compile_time_arg_val(3);
    constexpr uint32_t pre_value = get_compile_time_arg_val(4);
    constexpr uint32_t has_post = get_compile_time_arg_val(5);
    constexpr uint32_t post_value = get_compile_time_arg_val(6);
    constexpr auto src_args = TensorAccessorArgs<7>();
    const auto src = TensorAccessor(src_args, src_addr);
    const uint32_t tile_bytes = get_tile_size(cb_in);

    if constexpr (pre_pages > 0) {
        cb_reserve_back(cb_pre, pre_pages);
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_pre))[0] = pre_value;
        cb_push_back(cb_pre, pre_pages);
    }
    for (uint32_t t = 0; t < num_tiles; ++t) {
        cb_reserve_back(cb_in, 1);
        noc_async_read(src.get_noc_addr(t), get_write_ptr(cb_in), tile_bytes);
        noc_async_read_barrier();
        cb_push_back(cb_in, 1);
    }
    if constexpr (has_post) {
        cb_reserve_back(cb_pre, 1);
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_pre))[0] = post_value;
        cb_push_back(cb_pre, 1);
    }
}
"""

# Waits for the untilized rows and writes them to the row-major output, one row per page.
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

HELPER_COMPUTE = """
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t width = get_compile_time_arg_val(0);
    compute_kernel_hw_startup(0, 16);
    compute_kernel_lib::untilize<width, 0, 16>(1);
}
"""

# The init sequence of the DeepSeek prefill dispatch and combine compute kernels, with the block width given directly.
API_INIT_COMPUTE = """
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack_untilize.h"

void kernel_main() {
    constexpr uint32_t block_ct_dim = get_compile_time_arg_val(0);
    constexpr uint32_t full_ct_dim = get_compile_time_arg_val(1);
    compute_kernel_hw_startup(0, 16);
    pack_untilize_init<block_ct_dim, full_ct_dim>(0, 16);
    pack_untilize_uninit(16);
}
"""


def _core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def _cb(cb_id, dtype, page_size, num_pages):
    return ttnn.CBDescriptor(
        total_size=page_size * num_pages,
        core_ranges=_core(),
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=cb_id, data_format=dtype, page_size=page_size)],
    )


def _block_ct_dim(full_ct_dim):
    # The dispatch and combine factories' block: the largest divisor of the row up to 8.
    return next(b for b in range(8, 0, -1) if full_ct_dim % b == 0)


def _fp8_values(full_ct_dim, seed):
    # Normal Fp8_e4m3 values, exact in bf16, so the expected output does not depend on the rounding of the conversion.
    gen = torch.Generator().manual_seed(seed)
    exponent = torch.randint(1, 15, (ROWS, full_ct_dim * TILE), generator=gen)
    mantissa = torch.randint(0, 8, (ROWS, full_ct_dim * TILE), generator=gen)
    sign = torch.randint(0, 2, (ROWS, full_ct_dim * TILE), generator=gen)
    bits = (sign << 7) | (exponent << 3) | mantissa
    return bits.to(torch.uint8).view(torch.float8_e4m3fn).to(torch.bfloat16)


def _run(device, compute, full_ct_dim, out_dtype, out_page_size, out_pages, in_pages, pre_cb=None, seed=0):
    values = _fp8_values(full_ct_dim, seed)
    tt_in = ttnn.from_torch(values, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    fp8 = out_dtype == ttnn.fp8_e4m3
    # The output CB holds Fp8_e4m3; its bytes are read back through a uint8 tensor of the same rows.
    tt_out = ttnn.from_torch(
        torch.zeros(ROWS, full_ct_dim * TILE, dtype=torch.uint8 if fp8 else torch.bfloat16),
        dtype=ttnn.uint8 if fp8 else ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    row_bytes = full_ct_dim * TILE * (1 if fp8 else 2)

    # pre_cb: (pages, value, post value or None)
    pre_pages, pre_value, post_value = pre_cb if pre_cb else (0, 0, None)
    reader_rt = ttnn.RuntimeArgs()
    reader_rt[0][0] = [tt_in.buffer_address()]
    reader = ttnn.KernelDescriptor(
        kernel_source=READER,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_core(),
        compile_time_args=[
            CB_IN,
            full_ct_dim,
            CB_PRE if pre_cb else NO_CB,
            pre_pages,
            pre_value,
            int(post_value is not None),
            post_value or 0,
            *ttnn.TensorAccessorArgs(tt_in).get_compile_time_args(),
        ],
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer_rt = ttnn.RuntimeArgs()
    writer_rt[0][0] = [tt_out.buffer_address()]
    writer = ttnn.KernelDescriptor(
        kernel_source=WRITER,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_core(),
        compile_time_args=[
            CB_OUT,
            out_pages,
            ROWS,
            row_bytes,
            *ttnn.TensorAccessorArgs(tt_out).get_compile_time_args(),
        ],
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    cbs = [
        _cb(CB_IN, ttnn.bfloat16, ttnn.tile_size(ttnn.bfloat16), in_pages),
        _cb(CB_OUT, out_dtype, out_page_size, out_pages),
    ]
    if pre_cb:
        cbs.append(_cb(CB_PRE, ttnn.uint32, 16, max(pre_pages, 2)))
    program = ttnn.ProgramDescriptor(kernels=[reader, compute, writer], semaphores=[], cbs=cbs)
    ttnn.generic_op([tt_in, tt_out], program)

    actual = ttnn.to_torch(tt_out)
    if fp8:
        expected = values.to(torch.float8_e4m3fn).view(torch.uint8)
    else:
        expected, actual = values.view(torch.int16), actual.view(torch.int16)
    mismatched = (expected != actual).nonzero()
    assert mismatched.numel() == 0, f"{mismatched.shape[0]} of {expected.numel()} datums differ, first {mismatched[:4]}"


def _helper(device, out_dtype, width, full_sync):
    compute = ttnn.KernelDescriptor(
        kernel_source=HELPER_COMPUTE,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_core(),
        compile_time_args=[width],
        config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True, dst_full_sync_en=full_sync),
    )
    _run(device, compute, width, out_dtype, ttnn.tile_size(out_dtype), width, width)


def _deepseek_compute(kernel, compile_time_args, runtime_args=None):
    rt = ttnn.RuntimeArgs()
    rt[0][0] = runtime_args or []
    return ttnn.KernelDescriptor(
        kernel_source=f"{DEEPSEEK_KERNELS}/{kernel}",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=_core(),
        compile_time_args=compile_time_args,
        runtime_args=rt,
        # The factories' configuration for an Fp8_e4m3 output.
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, dst_full_sync_en=True
        ),
    )


@pytest.mark.parametrize("full_ct_dim", [11, 13, 17])
def test_fp8_one_tile_blocks_refused(device, expect_error, full_ct_dim):
    # A direct caller of the compute API that asks for one-tile blocks of a wider row is refused at kernel build.
    compute = ttnn.KernelDescriptor(
        kernel_source=API_INIT_COMPUTE,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_core(),
        compile_time_args=[1, full_ct_dim],
        config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True),
    )
    tensors = [
        ttnn.from_torch(torch.zeros(TILE, TILE, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
        for _ in range(2)
    ]
    program = ttnn.ProgramDescriptor(
        kernels=[compute],
        semaphores=[],
        cbs=[
            _cb(CB_IN, ttnn.bfloat16, ttnn.tile_size(ttnn.bfloat16), 1),
            _cb(CB_OUT, ttnn.fp8_e4m3, 1024, full_ct_dim),
        ],
    )
    with expect_error(RuntimeError, "one-tile blocks of a wider row"):
        ttnn.generic_op(tensors, program)
        ttnn.synchronize_device(device)


@pytest.mark.parametrize(
    "width, full_sync",
    [
        (5, False),  # DEST limit 4: blocks of 3 and 2
        (7, False),  # 4 and 3
        (11, False),  # 4, 4 and 3
        (13, False),  # 4, 4, 3 and 2
        (11, True),  # DEST limit 8: 8 and 3
        (13, True),  # 8 and 5
        (17, True),  # 8, 5 and 4
    ],
)
def test_helper_fp8_row_split(device, width, full_sync):
    _helper(device, ttnn.fp8_e4m3, width, full_sync)


@pytest.mark.parametrize(
    "out_dtype, width",
    [
        (ttnn.fp8_e4m3, 6),  # blocks of 3, the even split
        (ttnn.fp8_e4m3, 8),  # blocks of 4
        (ttnn.bfloat16, 11),  # other formats keep one-tile blocks
    ],
)
def test_helper_even_split_unchanged(device, out_dtype, width):
    _helper(device, out_dtype, width, False)


@pytest.mark.parametrize("full_ct_dim", [11, 13, 17, 12, 224])
def test_dispatch_untilize_fp8(device, full_ct_dim):
    # The dispatch compute kernel for one batch of 32 tokens: a route signal, the row's tiles, then the sentinel.
    block = _block_ct_dim(full_ct_dim)
    compute = _deepseek_compute(
        "dispatch/device/kernels/compute/untilize_dispatch.cpp",
        [CB_PRE, CB_OUT, CB_IN, full_ct_dim * TILE, ROWS, block],
    )
    _run(
        device,
        compute,
        full_ct_dim,
        ttnn.fp8_e4m3,
        full_ct_dim * TILE,
        ROWS,
        2 * block,
        pre_cb=(1, 0, SENTINEL),
        seed=full_ct_dim,
    )


@pytest.mark.parametrize("full_ct_dim", [11, 13, 17, 12, 224])
def test_combine_untilize_fp8(device, full_ct_dim):
    # The combine compute kernel for one expert of 32 tokens on one untilizer core: the token count, then the tiles.
    block = _block_ct_dim(full_ct_dim)
    counter_pages = 1
    compute = _deepseek_compute(
        "combine/device/kernels/compute/untilize_combine.cpp",
        [CB_OUT, CB_IN, CB_PRE, counter_pages, 1, 0, ROWS, ROWS, full_ct_dim, block, counter_pages + 1],
        runtime_args=[0, 1, 0, 1, 0, 1],
    )
    _run(
        device,
        compute,
        full_ct_dim,
        ttnn.fp8_e4m3,
        full_ct_dim * TILE,
        ROWS,
        2 * block,
        pre_cb=(counter_pages + 1, ROWS, None),
        seed=full_ct_dim,
    )
