# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.chip_architecture import ChipArchitecture
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation
from helpers.param_config import input_output_formats
from helpers.test_config import TestConfig


# We force device print to be enabled during this test,
# and turn it back off when it completes.
@pytest.fixture(scope="module", autouse=True)
def _force_device_print_enabled():
    prev = TestConfig.DEVICE_PRINT_ENABLED
    TestConfig.DEVICE_PRINT_ENABLED = True
    yield
    TestConfig.DEVICE_PRINT_ENABLED = prev


def test_device_print():
    formats = input_output_formats([DataFormat.Int32])[0]

    configuration = TestConfig(
        "sources/device_print_test.cpp", formats, dest_acc=DestAccumulation.Yes
    )
    outcome = configuration.run()
    lines = outcome.device_print_lines

    full = "".join(lines)
    assert full, "No device print output received"

    # Unpack: multi-size args (reordering takes place)
    assert "unpack: i8=-1 u8=255 i16=-100 u16=65535" in full
    assert "_unpack" in full

    # Enum resolution from debug info:
    #   regular enum -> single name
    #   flag enum -> "name1 | name2"
    #   '#' spec -> fully qualified "type::name | type::name"
    #   unknown value -> "(type)value"
    assert "unpack: enum=Green" in full
    assert "unpack: flag=R | X" in full
    assert "unpack: flag_full=Perm::R | Perm::W" in full
    assert "unpack: flag_unk=(Perm)24" in full

    # Name resolution follows DWARF declaration order, not numeric order.
    # Rev is declared { Z=4, Y=2, X=1 }, so 7 -> "Z | Y | X", not "X | Y | Z".
    assert "unpack: flag_rev=Z | Y | X" in full

    # Math: one print per type category
    assert "math: i32=-1 u32=65536" in full
    assert "math: float=1.0" in full
    assert "math: bool=true false" in full
    assert "math: ptr=0xdeadbeef" in full
    assert "_math" in full  # CTSTR
    assert "math: hex=00000abc" in full
    assert "math: pad=    test" in full

    # We print 2048 iterations (weighing 8 bytes each) to force a drain.
    # Whether this test hits the stall path depends on the buffer size.
    missing = [i for i in range(2048) if f"w={i}" not in full]
    assert not missing, (
        f"Missing {len(missing)} of 2048 wrap iterations; "
        f"first 10 missing: {missing[:10]}"
    )

    # Pack
    assert "pack: i64=-1000000" in full
    assert "_pack" in full

    # SFPU is only built on Quasar
    if TestConfig.CHIP_ARCH == ChipArchitecture.QUASAR:
        assert "sfpu: u8=3 i8=-1" in full
        assert "_sfpu" in full
