# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# A kernel hung on purpose must not take the core with it: the next launch in the same process has to finish, and a
# later kernel has to compute on its own operands.

import pytest
from conftest import skip_for_quasar
from helpers.chip_architecture import ChipArchitecture
from helpers.device import _SRCA_SRCB
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, MathFidelity
from helpers.test_config import BuildMode, TestConfig
from test_matmul import test_matmul as run_matmul
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.debug_tensix import TensixDebug
from ttexalens.tt_exalens_lib import check_context

# CLEARDVALID mode: bit 0 is Reset, bit 1 is KeepReadingSameSrc.
_KEEP_READING_SAME_SRC = 0b10


def _clear_valid_flags():
    # The stall kernel only hangs while SrcA/SrcB are not valid, which earlier tests may have left set.
    # KeepReadingSameSrc keeps the Matrix Unit on the bank the Unpackers write next.
    device = check_context().devices[0]
    debug_tensix = TensixDebug(
        OnChipCoordinate.create(TestConfig.TENSIX_LOCATION, device=device)
    )
    debug_tensix.inject_instruction(
        debug_tensix.device.instructions.TT_OP_CLEARDVALID(
            _SRCA_SRCB, _KEEP_READING_SAME_SRC
        ),
        0,  # thread 0
    )


def _matmul_after_recovery():
    # Fails when the hangs left the Unpackers and the Matrix Unit on different SrcA/SrcB banks: the matmul then
    # multiplies what an earlier launch unpacked.
    run_matmul(
        MathFidelity.LoFi,
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
        DestAccumulation.No,
        1,
        1,
        1,
    )


def _launch(source: str, timeout=2):
    cfg = TestConfig(source)
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
    if TestConfig.TEST_TARGET.run_simulator:
        pytest.skip()
    if TestConfig.BUILD_MODE == BuildMode.PRODUCE:
        _matmul_after_recovery()  # compiles the matmul; TestConfig.run ends the test here
    for kind in order:
        if kind == "stall":
            _clear_valid_flags()
            source = "sources/hang_stall_test.cpp"
        else:
            source = "sources/hang_spin_test.cpp"
        with pytest.raises(  # allow-pytest.raises: no expect_error in LLK suite
            TimeoutError
        ):
            _launch(source)
        # Two launches: one right after the hang, one more to show it stays recovered
        _launch("sources/hang_ok_test.cpp")
        _launch("sources/hang_ok_test.cpp")
    _matmul_after_recovery()
