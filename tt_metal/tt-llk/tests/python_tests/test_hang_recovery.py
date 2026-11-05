# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# A kernel hung on purpose must not take the core with it: the next launch in the same process has to finish.

import pytest
from conftest import skip_for_quasar
from helpers.chip_architecture import ChipArchitecture
from helpers.test_config import BuildMode, TestConfig
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.debug_tensix import TensixDebug
from ttexalens.tt_exalens_lib import check_context


def _clear_valid_flags():
    # The stall kernel only hangs while SrcA/SrcB are not valid, which earlier tests may have left set
    device = check_context().devices[0]
    debug_tensix = TensixDebug(
        OnChipCoordinate.create(TestConfig.TENSIX_LOCATION, device=device)
    )
    debug_tensix.inject_instruction(
        debug_tensix.device.instructions.TT_OP_CLEARDVALID(0b11, 0), 0
    )


def _launch(source, timeout=2):
    cfg = TestConfig(f"sources/hang_{source}_test.cpp")
    cfg.generate_variant_hash()
    cfg.build_elfs()
    cfg.run_elf_files()
    cfg.wait_for_tensix_operations_finished(timeout=timeout)


@skip_for_quasar
@pytest.mark.parametrize("order", [("stall", "spin"), ("spin", "stall")])
def test_hang_recovery(order):
    # Blackhole: recovery after the stall still leaves state that fails a later test_matmul (#64513 CI, #64272).
    if (
        TestConfig.CHIP_ARCH == ChipArchitecture.BLACKHOLE
        and not TestConfig.TEST_TARGET.run_simulator
    ):
        pytest.skip("BH recovery still being iterated (#64272)")
    if (
        TestConfig.BUILD_MODE == BuildMode.PRODUCE
        or TestConfig.TEST_TARGET.run_simulator
    ):
        pytest.skip()
    for kind in order:
        if kind == "stall":
            _clear_valid_flags()
        with pytest.raises(  # allow-pytest.raises: no expect_error in LLK suite
            TimeoutError
        ):
            _launch(kind)
        # Two launches: one right after the hang, one more to show it stays recovered
        _launch("ok")
        _launch("ok")
