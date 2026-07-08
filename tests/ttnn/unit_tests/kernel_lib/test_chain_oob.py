# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
G6 — out-of-bounds / boundary tests for eltwise_chain.

Coverage spec: ttnn/cpp/ttnn/kernel_lib/docs/eltwise_helper_test_coverage.html (group G6).

The DST-capacity cases ("use more DST than we have"). The helper static_asserts every
element's DEST slot against DEST_AUTO_LIMIT, which itself depends on the compute config:

    sync mode      bf16 (fp32_acc off)    fp32 (fp32_acc on)
    half-sync             8                       4
    full-sync           16                       8

So a slot that is legal in one config is over-limit in another. Each test pins
(slot, dst_full_sync_en, fp32_dest_acc_en) and asserts either:
  - a legal slot compiles + runs and reproduces the input (identity copy), or
  - an over-limit slot FAILS to compile with "DEST slot exceeds DEST_AUTO_LIMIT".

The compile-time guard surfaces as an exception from ttnn.generic_op (the kernel JIT build
fails); we catch it and assert the assert message is present, so the test proves the guard
actually fired rather than the build failing for some unrelated reason.
"""

import torch
import pytest
import ttnn
from loguru import logger
from tests.ttnn.utils_for_testing import comp_pcc
import tests.ttnn.unit_tests.kernel_lib.chain_test_lib as lib

KERNEL = "ttnn/cpp/ttnn/kernel_lib/tests/oob/dst_slot.cpp"
DST_OVERFLOW_MSG = "DEST slot exceeds"


def _run_identity_copy(device, slot, num_tiles, fp32_dest_acc_en, dst_full_sync_en):
    """Build + run the dst_slot identity-copy chain. Returns (golden, output) torch tensors."""
    shape = [1, 1, 32, 32 * num_tiles]
    dt = ttnn.bfloat16
    core_grid = lib.single_core_grid()

    torch_in, tt_in = lib.make_input(shape, dt, device, seed=101)
    tt_out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    cbs = [
        lib.cb_descriptor(0, dt, 2, core_grid),
        lib.cb_descriptor(16, dt, 2, core_grid),
    ]
    reader = lib.build_reader_kernel([tt_in], num_tiles, core_grid)
    writer = lib.build_writer_1out_kernel(tt_out, num_tiles, core_grid)
    compute = lib.build_compute_kernel(
        KERNEL,
        compile_time_args=[num_tiles, slot],
        core_grid=core_grid,
        fp32_dest_acc_en=fp32_dest_acc_en,
        dst_full_sync_en=dst_full_sync_en,
    )
    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    output = ttnn.generic_op([tt_in, tt_out], program)
    return torch_in.to(torch.float32), ttnn.to_torch(output).to(torch.float32)


# =============================================================================
# Positive twin — a legal slot compiles, runs, and copies the input exactly.
# =============================================================================
@pytest.mark.parametrize("slot", [0, 3])
@pytest.mark.parametrize(
    "fp32_dest_acc_en, dst_full_sync_en",
    [(False, False), (True, False)],  # half-sync bf16 (limit 8) and half-sync fp32 (limit 4)
)
def test_dst_slot_legal_identity(device, slot, fp32_dest_acc_en, dst_full_sync_en):
    """OOB positive twin: D0/D3 are within every limit; identity copy must reproduce the input."""
    golden, out = _run_identity_copy(device, slot, 4, fp32_dest_acc_en, dst_full_sync_en)
    pcc_ok, msg = comp_pcc(golden, out, lib.pcc_threshold([ttnn.bfloat16]))
    logger.info(f"legal slot={slot} fp32_acc={fp32_dest_acc_en} sync_full={dst_full_sync_en} | {msg}")
    assert pcc_ok, msg


# =============================================================================
# OOB-01 — slot 8 over the half-sync bf16 limit (8). Must fail to compile.
# =============================================================================
def test_oob01_dst_overflow_halfsync_bf16(device, expect_error):
    """D8 is not < 8 (half-sync bf16 limit) -> the per-element static_assert must fire."""
    with expect_error(Exception, DST_OVERFLOW_MSG):
        _run_identity_copy(device, slot=8, num_tiles=4, fp32_dest_acc_en=False, dst_full_sync_en=False)
    logger.info("OOB-01: D8 over half-sync bf16 limit correctly rejected at compile time")


# =============================================================================
# OOB-02 — cross-mode boundary. Slot 5: legal half-sync bf16 (limit 8),
#          over-limit once fp32_dest_acc shrinks the limit to 4.
# =============================================================================
def test_oob02_dst_slot5_legal_bf16(device):
    """Same slot, bf16 half-sync (limit 8): 5 < 8 -> compiles + runs correctly."""
    golden, out = _run_identity_copy(device, slot=5, num_tiles=4, fp32_dest_acc_en=False, dst_full_sync_en=False)
    pcc_ok, msg = comp_pcc(golden, out, lib.pcc_threshold([ttnn.bfloat16]))
    logger.info(f"OOB-02 bf16: slot=5 legal | {msg}")
    assert pcc_ok, msg


def test_oob02_dst_slot5_overflow_fp32(device, expect_error):
    """Same slot under fp32_dest_acc (limit 4): 5 is not < 4 -> must fail to compile."""
    with expect_error(Exception, DST_OVERFLOW_MSG):
        _run_identity_copy(device, slot=5, num_tiles=4, fp32_dest_acc_en=True, dst_full_sync_en=False)
    logger.info("OOB-02 fp32: slot=5 over fp32 half-sync limit (4) correctly rejected")


# =============================================================================
# OOB-03 — runtime block_size clamp. block_size is a runtime EltwiseShape field, so it
# can't be static_asserted; the chain clamps it down to chain_max_block_v at runtime
# (eltwise_chain.inl:2024-2033) so it can NEVER overflow DEST. An over-large block_size
# must therefore still produce the exact identity copy (clamp only changes loop structure).
#
# The chain here is a Bulk + Block reader (block-capable), so block_size is honored. Bulk
# stages the whole window upfront -> the input CB must hold all n pages.
# =============================================================================
BLOCK_KERNEL = "ttnn/cpp/ttnn/kernel_lib/tests/oob/block_clamp.cpp"


@pytest.mark.parametrize("block_size", [1, 4, 1000])
def test_oob03_block_size_clamp_identity(device, block_size):
    """chain_lane_width=1 here -> chain_max_block_v=8 (half-sync bf16). block_size=1000 must clamp
    to 8 and still copy the input exactly; {1,4} are within range. All three agree with the input."""
    n = 16
    dt = ttnn.bfloat16
    shape = [1, 1, 32, 32 * n]
    core_grid = lib.single_core_grid()

    torch_in, tt_in = lib.make_input(shape, dt, device, seed=303)
    tt_out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    # Bulk reader/writer stage the full window upfront -> size both CBs for all n tiles.
    cbs = [lib.cb_descriptor(0, dt, n, core_grid), lib.cb_descriptor(16, dt, n, core_grid)]
    reader = lib.build_reader_kernel([tt_in], n, core_grid)
    writer = lib.build_writer_1out_kernel(tt_out, n, core_grid)
    compute = lib.build_compute_kernel(BLOCK_KERNEL, [n, block_size], core_grid)

    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    output = ttnn.generic_op([tt_in, tt_out], program)
    golden = torch_in.to(torch.float32)
    out = ttnn.to_torch(output).to(torch.float32)
    pcc_ok, msg = comp_pcc(golden, out, lib.pcc_threshold([dt]))
    logger.info(f"OOB-03 block_size={block_size} (clamps to <=8) | {msg}")
    assert pcc_ok, msg
