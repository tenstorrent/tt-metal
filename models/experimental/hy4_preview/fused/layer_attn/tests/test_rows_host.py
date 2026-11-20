# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host only: the row mask and the two sub-passes of program P2 at 16 token rows. tests/host_rows.cpp includes the
kernels' own header (kernels/attn_rows.hpp) and runs the parties of X4 against each other for all 65,536 masks (what
it checks is in its head comment); here it is compiled and run for 7, 3 and 1 sources, and the header's functions
are compared with a model written independently in Python.

    python -m pytest models/experimental/hy4_preview/fused/layer_attn/tests/test_rows_host.py
"""

import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).parent


@pytest.fixture(scope="module")
def binary(tmp_path_factory):
    exe = tmp_path_factory.mktemp("rows") / "host_rows"
    subprocess.run(
        ["g++", "-std=c++17", "-O2", "-Wall", "-Werror", "-o", str(exe), str(HERE / "host_rows.cpp")], check=True
    )
    return exe


def model(mask, sources):
    """(waits, partials) from the description: 8 p + i is live when its bit is set; a sub-pass has a payload when
    one of its rows is live."""
    live = [[(mask >> (8 * p + i)) & 1 for i in range(8)] for p in range(2)]
    payloads = sum(1 for p in range(2) if any(live[p]))
    rows = sum(map(sum, live))
    # coordinator: 2 group waits per source chip; relay: one per arrival; merge core: local + one per source per live
    # row; row op: 2; post hub: 1
    waits = 2 * sources + payloads * sources + rows * (1 + sources) + 2 + 1
    return waits, rows * sources


@pytest.mark.parametrize("sources", [7, 3, 1])
def test_rows_all_masks(binary, sources):
    out = subprocess.run([str(binary), str(sources)], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    words = out.stdout.split()
    waits = partials = 0
    for mask in range(1 << 16):
        w, p = model(mask, sources)
        waits += w
        partials += p
    assert words[0] == "masks" and int(words[1]) == 1 << 16, out.stdout
    assert int(words[3]) == waits and int(words[5]) == partials, (out.stdout, waits, partials)
    print(f"HY4_MK8A rows host, {sources} sources: {out.stdout.strip()}")
