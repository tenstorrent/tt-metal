# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-run parity check of the decode step between a short-context and a long-context build (parses two demo logs; skipped unless DSV41_CTX_PARITY_LOGS="<small build log>,<1M build log>").

Run the same ISL on a build for ~64k context and on the 1M build (sessions ``isl64k_c70k_b4`` and ``isl64k_b4`` of demo/text_demo.py, plain, ``run_isl.sh <tag> <session> 0 plain``) and compare:
the ``Decode: <ms> ms/token`` line of the 1M build must stay within DSV41_CTX_PARITY_TOL (default 10 %) of the small build (measured after the bucket ladder: 47.0 vs 50.2 ms, 6.8 %; before: 64.8 ms, 38 %).
"""

import os
import re

import pytest

LOGS = os.environ.get("DSV41_CTX_PARITY_LOGS")


def _decode_ms(path):
    m = re.findall(r"Decode: ([0-9.]+) ms/token", open(path).read())
    assert m, f"no 'Decode:' line in {path}"
    return float(m[-1])


@pytest.mark.skipif(not LOGS, reason="set DSV41_CTX_PARITY_LOGS=<small build log>,<1M build log>")
def test_decode_step_does_not_depend_on_the_built_context():
    small, big = (_decode_ms(p) for p in LOGS.split(","))
    tol = float(os.environ.get("DSV41_CTX_PARITY_TOL", "0.10"))
    assert big <= small * (1 + tol), f"1M build decodes at {big} ms/token vs {small} on the short build"
