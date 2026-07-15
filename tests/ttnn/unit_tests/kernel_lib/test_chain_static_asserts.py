# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Compile-time guard (negative) tests for eltwise_chain.

The helper's static_asserts REFUSE illegal chain configs at compile time. A guard that rots into an
always-true no-op is invisible until someone ships the footgun. Each test proves a guard still fires:
the illegal kernel must fail to build with the expected assert message (an exception from
ttnn.generic_op's JIT build). No device work happens, but generic_op needs a device handle to build,
so the `device` fixture is still requested.
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


def test_block_streaming_illegal(device, expect_error):
    """Block index + Streaming lifecycle is the absolute-index footgun — must not compile."""
    _expect_build_failure(device, expect_error, "copytile_block_streaming.cpp", "is illegal for Block")


def test_packtile_collision_illegal(device, expect_error):
    """Two PackTile on the same (CB, DEST slot) — must not compile."""
    _expect_build_failure(device, expect_error, "packtile_collision.cpp", "two PackTile elements collide")


def test_row_streaming_illegal(device, expect_error):
    """Row/Col broadcast index + Streaming policy — must not compile."""
    _expect_build_failure(device, expect_error, "row_streaming.cpp", "non-streaming policy")


def test_tile_offset_streaming_illegal(device, expect_error):
    """TileOffset::Set + Streaming (needs Bulk-family / CallerManaged) — must not compile."""
    _expect_build_failure(device, expect_error, "tile_offset_streaming.cpp", "TileOffset::Set requires")


def test_cba_cbb_index_mismatch_illegal(device, expect_error):
    """BinaryFpu same CB for both operands with mismatched indices — must not compile."""
    _expect_build_failure(device, expect_error, "cba_cbb_index_mismatch.cpp", "AIndex and BIndex must match")


def test_setupowner_caller_not_hoistable_illegal(device, expect_error):
    """SetupOwner::Caller on a chain whose setup isn't fully boot-hoistable (non-uniform SFPU)
    — the caller can't own a once-before-the-loop setup that doesn't exist, so it must not compile."""
    _expect_build_failure(device, expect_error, "setupowner_caller_not_hoistable.cpp", "boot-hoistable")


def test_setupowner_caller_reconfig_illegal(device, expect_error):
    """SetupOwner::Caller with a live (non-None) reconfig knob — under Caller the chain emits
    no reconfig, so the knob is inert/deceptive; must not compile (force the caller to declare None)."""
    _expect_build_failure(device, expect_error, "setupowner_caller_reconfig.cpp", "non-None reconfig knob")


def test_l1_accumulation_wrong_lifecycle_illegal(device, expect_error):
    """L1 accumulation cannot use a general-purpose streaming output lifecycle."""
    _expect_build_failure(
        device,
        expect_error,
        "l1_accumulation_wrong_lifecycle.cpp",
        "L1 accumulation requires OutputLifecycle::L1Accumulation",
    )


def test_l1_lifecycle_without_accumulation_illegal(device, expect_error):
    """L1-purpose output lifecycles cannot silently run with accumulation disabled."""
    _expect_build_failure(
        device,
        expect_error,
        "l1_lifecycle_without_accumulation.cpp",
        "those lifecycles require L1 accumulation",
    )


def test_l1_accumulation_multiple_output_cbs_illegal(device, expect_error):
    """The packer-global accumulation bracket may target only one output CB."""
    _expect_build_failure(
        device,
        expect_error,
        "l1_accumulation_multiple_output_cbs.cpp",
        "L1 accumulation supports only one output CB",
    )


def test_l1_accumulation_mixed_pack_modes_illegal(device, expect_error):
    """Ordinary packs cannot inherit the chain-wide L1-accumulation mode."""
    _expect_build_failure(
        device,
        expect_error,
        "l1_accumulation_mixed_pack_modes.cpp",
        "cannot mix accumulating and ordinary PackTile elements",
    )


def test_l1_accumulation_mixed_accumulation_modes_illegal(device, expect_error):
    """Preloaded and seed-first packs cannot share one packer-global accumulation region."""
    _expect_build_failure(
        device,
        expect_error,
        "l1_accumulation_mixed_accumulation_modes.cpp",
        "must all use the same L1 accumulation mode",
    )


def test_l1_accumulation_multiple_lifecycle_owners_illegal(device, expect_error):
    """Only one pack may reserve and publish the shared accumulator tile."""
    _expect_build_failure(
        device,
        expect_error,
        "l1_accumulation_multiple_lifecycle_owners.cpp",
        "only one PackTile may own the L1-accumulation",
    )


def test_dest_accumulation_wrong_lifecycle_illegal(device, expect_error):
    """DEST accumulation cannot use an ordinary streaming output lifecycle."""
    _expect_build_failure(
        device,
        expect_error,
        "dest_accumulation_wrong_lifecycle.cpp",
        "DEST accumulation requires OutputLifecycle::DestAccumulation",
    )


def test_dest_lifecycle_without_accumulation_illegal(device, expect_error):
    """DEST-purpose output lifecycles cannot silently run without an accumulating BinaryFpu."""
    _expect_build_failure(
        device,
        expect_error,
        "dest_lifecycle_without_accumulation.cpp",
        "those lifecycles require an accumulating BinaryFpu",
    )


def test_dest_accumulation_pack_mismatch_illegal(device, expect_error):
    """The single final pack must read the sticky accumulator slot."""
    _expect_build_failure(
        device,
        expect_error,
        "dest_accumulation_pack_mismatch.cpp",
        "PackTile must pack the sticky DEST slot",
    )


def test_relu_l1_accumulation_illegal(device, expect_error):
    """Packer ReLU on an L1-accumulating pack — clamp-vs-accumulate ordering unverified; forbidden."""
    _expect_build_failure(device, expect_error, "relu_l1_accumulation.cpp", "packer ReLU combined with L1 accumulation")


def test_relu_dest_accumulation_illegal(device, expect_error):
    """Packer ReLU on a DEST-accumulation chain — the set/reset aren't wired on that walk; forbidden."""
    _expect_build_failure(
        device, expect_error, "relu_dest_accumulation.cpp", "packer ReLU combined with DEST accumulation"
    )


def test_relu_setupowner_caller_illegal(device, expect_error):
    """SetupOwner::Caller with a packer-ReLU knob — the chain emits no setup under Caller, so the ReLU
    is inert; must not compile (all reconfigs are None, so ReLU is the sole trigger)."""
    _expect_build_failure(device, expect_error, "relu_setupowner_caller.cpp", "non-None reconfig knob")
