# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host only (PLAN.md section 19.14): the stage loop of kernels/attn_order.hpp, which X4's merge cores run under
HY4A_FIXED_ORDER, compiled on the host and run against every order of arrival of a stage's partners (tests/
host_order.cpp says what is checked): the merge kernel gets exactly its count of partials in every pattern, and in
the fixed order the sequence of partners delivered does not depend on the order of arrival.

    python -m pytest models/experimental/hy4_preview/fused/layer_attn/tests/test_order_host.py
"""

import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).parent


@pytest.fixture(scope="module")
def binary(tmp_path_factory):
    exe = tmp_path_factory.mktemp("order") / "host_order"
    subprocess.run(
        ["g++", "-std=c++17", "-O2", "-Wall", "-Werror", "-o", str(exe), str(HERE / "host_order.cpp")], check=True
    )
    return exe


# the column stage of a chip in mesh row 0, 3, 7 (7 partners), the row stage in column 0, 2 (3 partners), a column
# pair, one partner, and the second sub-pass's column stage at 16 rows (the mask's bits shifted by 8)
@pytest.mark.parametrize(
    "mask, shift", [(0xFE, 0), (0xF7, 0), (0x7F, 0), (0xE, 0), (0xB, 0), (0x2, 0), (0x1, 0), (0xFE, 8), (0xB, 8)]
)
def test_every_arrival_order(binary, mask, shift):
    out = subprocess.run([str(binary), hex(mask), str(shift)], capture_output=True, text=True)
    assert out.returncode == 0, (out.returncode, out.stdout, out.stderr)
    cases, longest = (int(v) for v in out.stdout.split())
    partners = bin(mask).count("1")
    print(
        f"HY4_MK8A order host, mask {mask:#x} << {shift}: {cases} stages ({partners} partners: every order of arrival x 3 gap "
        f"patterns x every subset that never lands x both orders of taking), the longest {longest} polls: counts and sequences as wanted"
    )
