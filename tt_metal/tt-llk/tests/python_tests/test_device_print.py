# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from helpers.device_print import run_with_device_print
from helpers.format_config import DataFormat
from helpers.param_config import input_output_formats
from helpers.test_config import TestConfig


def test_device_print():
    formats = input_output_formats([DataFormat.Int32])[0]

    configuration = TestConfig("sources/device_print_test.cpp", formats)

    _, lines = run_with_device_print(configuration)

    full = "".join(lines)
    assert full, "No device-print output received"

    # Unpack: multi-size args (reordering takes place)
    assert "unpack: i8=-1 u8=255 i16=-100 u16=65535" in full
    assert "_unpack" in full

    # Math: one print per type category
    assert "math: i32=-1 u32=65536" in full
    assert "math: float=1.0" in full
    assert "math: bool=true false" in full
    assert "math: ptr=0xdeadbeef" in full
    assert "_math" in full  # CTSTR
    assert "math: hex=00000abc" in full
    assert "math: pad=    test" in full

    # We print 160 iterations (weighing 8 bytes each) to force a drain.
    # This test depends on the buffer being under 1280 or so bytes.
    assert "w=0\n" in full, "Missing first wrap iteration"
    assert "w=159\n" in full, "Missing last wrap iteration; possible message loss"

    # Pack
    assert "pack: i64=-1000000" in full
    assert "_pack" in full
