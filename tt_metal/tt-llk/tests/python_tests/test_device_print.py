# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from helpers.device_print import run_with_device_print
from helpers.format_config import DataFormat
from helpers.param_config import input_output_formats
from helpers.test_config import DevicePrintBuild, TestConfig


def test_device_print():
    formats = input_output_formats([DataFormat.Int32])[0]

    configuration = TestConfig(
        "sources/device_print_test.cpp",
        formats,
        device_print_build=DevicePrintBuild.Yes,
    )

    _, lines = run_with_device_print(configuration)

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
    assert "w=0\n" in full, "Missing first wrap iteration"
    assert "w=2047\n" in full, "Missing last wrap iteration; possible message loss"

    # Pack
    assert "pack: i64=-1000000" in full
    assert "_pack" in full
