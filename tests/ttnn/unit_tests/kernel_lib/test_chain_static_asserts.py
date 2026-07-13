# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
G2 — compile-time guard (negative) tests for eltwise_chain.

Coverage spec: ttnn/cpp/ttnn/kernel_lib/docs/eltwise_helper_test_coverage.html (group G2).

The helper carries static_asserts that REFUSE illegal chain configurations at compile time.
A guard that has rotted into an always-true no-op is invisible until someone ships the footgun
it was meant to stop. These tests prove each guard still fires: an illegal kernel must fail to
build with the expected assert message (caught as an exception from ttnn.generic_op's JIT build).

No device work happens — the build fails before launch — but generic_op needs a device handle
to drive the build, so the `device` fixture is still requested.
"""

import ttnn
from loguru import logger
import tests.ttnn.unit_tests.kernel_lib.chain_test_lib as lib

KERNEL_DIR = "ttnn/cpp/ttnn/kernel_lib/tests/static_asserts"


def _expect_build_failure(device, expect_error, kernel_name, expected_msg):
    """Build a 1-in/1-out program around an illegal compute kernel; assert the JIT build fails
    with expected_msg in the error. CBs/tensors are well-formed so the ONLY failure source is the
    compute kernel's static_assert."""
    n = 4
    dt = ttnn.bfloat16
    shape = [1, 1, 32, 32 * n]
    core_grid = lib.single_core_grid()

    _, tt_in = lib.make_input(shape, dt, device, seed=1)
    tt_out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    cbs = [lib.cb_descriptor(0, dt, 2, core_grid), lib.cb_descriptor(16, dt, 2, core_grid)]
    reader = lib.build_reader_kernel([tt_in], n, core_grid)
    writer = lib.build_writer_1out_kernel(tt_out, n, core_grid)
    compute = lib.build_compute_kernel(f"{KERNEL_DIR}/{kernel_name}", [n], core_grid)
    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)

    with expect_error(Exception, expected_msg):
        ttnn.generic_op([tt_in, tt_out], program)
    logger.info(f"{kernel_name}: correctly rejected at compile time ('{expected_msg}')")


def test_sa03_block_streaming_illegal(device, expect_error):
    """SA-03: Block index + Streaming lifecycle is the absolute-index footgun — must not compile."""
    _expect_build_failure(device, expect_error, "copytile_block_streaming.cpp", "is illegal for Block")


def test_sa02_packtile_collision_illegal(device, expect_error):
    """SA-02: two PackTile on the same (CB, DEST slot) — must not compile."""
    _expect_build_failure(device, expect_error, "packtile_collision.cpp", "two PackTile elements collide")


def test_sa04_row_streaming_illegal(device, expect_error):
    """SA-04: Row/Col broadcast index + Streaming policy — must not compile."""
    _expect_build_failure(device, expect_error, "row_streaming.cpp", "non-streaming policy")


def test_sa07_tile_offset_streaming_illegal(device, expect_error):
    """SA-07: TileOffset::Set + Streaming (needs Bulk-family / CallerManaged) — must not compile."""
    _expect_build_failure(device, expect_error, "tile_offset_streaming.cpp", "TileOffset::Set requires")


def test_sa08_cba_cbb_index_mismatch_illegal(device, expect_error):
    """SA-08: BinaryFpu same CB for both operands with mismatched indices — must not compile."""
    _expect_build_failure(device, expect_error, "cba_cbb_index_mismatch.cpp", "AIndex and BIndex must match")


def test_sa09_setupowner_caller_not_hoistable_illegal(device, expect_error):
    """SA-09: SetupOwner::Caller on a chain whose setup isn't fully boot-hoistable (non-uniform SFPU)
    — the caller can't own a once-before-the-loop setup that doesn't exist, so it must not compile."""
    _expect_build_failure(device, expect_error, "setupowner_caller_not_hoistable.cpp", "boot-hoistable")


def test_sa10_setupowner_caller_reconfig_illegal(device, expect_error):
    """SA-10: SetupOwner::Caller with a live (non-None) reconfig knob — under Caller the chain emits
    no reconfig, so the knob is inert/deceptive; must not compile (force the caller to declare None)."""
    _expect_build_failure(device, expect_error, "setupowner_caller_reconfig.cpp", "non-None reconfig knob")


def test_sa11_l1_accumulation_wrong_lifecycle_illegal(device, expect_error):
    """SA-11: L1 accumulation cannot use a general-purpose output lifecycle."""
    _expect_build_failure(
        device,
        expect_error,
        "l1_accumulation_wrong_lifecycle.cpp",
        "L1 accumulation requires OutputLifecycle::L1Accumulation",
    )


def test_sa12_l1_lifecycle_without_accumulation_illegal(device, expect_error):
    """SA-12: L1-purpose output lifecycles require an accumulating PackTile."""
    _expect_build_failure(
        device,
        expect_error,
        "l1_lifecycle_without_accumulation.cpp",
        "those lifecycles require L1 accumulation",
    )


def test_sa13_l1_accumulation_multiple_output_cbs_illegal(device, expect_error):
    """SA-13: The packer-global accumulation region may target only one output CB."""
    _expect_build_failure(
        device,
        expect_error,
        "l1_accumulation_multiple_output_cbs.cpp",
        "L1 accumulation supports only one output CB",
    )


def test_sa14_l1_accumulation_mixed_pack_modes_illegal(device, expect_error):
    """SA-14: Ordinary packs cannot inherit chain-wide L1 accumulation mode."""
    _expect_build_failure(
        device,
        expect_error,
        "l1_accumulation_mixed_pack_modes.cpp",
        "cannot mix accumulating and ordinary PackTile elements",
    )


def test_sa15_l1_accumulation_mixed_accumulation_modes_illegal(device, expect_error):
    """SA-15: Preloaded and seed-first writers cannot share one accumulation region."""
    _expect_build_failure(
        device,
        expect_error,
        "l1_accumulation_mixed_accumulation_modes.cpp",
        "must all use the same L1 accumulation mode",
    )


def test_sa16_l1_accumulation_multiple_lifecycle_owners_illegal(device, expect_error):
    """SA-16: Only one writer may reserve and publish the shared accumulator tile."""
    _expect_build_failure(
        device,
        expect_error,
        "l1_accumulation_multiple_lifecycle_owners.cpp",
        "only one PackTile may own the L1-accumulation",
    )
