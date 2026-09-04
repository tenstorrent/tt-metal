# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Drift guard for the Phase 2a C++ fast path.

The C++ `canonical_headers()` in perf_counter_metrics.cpp and the Python
`PERF_COUNTER_CSV_HEADERS` in perf_counter_analysis.py emit the SAME columns of
cpp_device_perf_report.csv; either one drifting silently corrupts the report
(shifted/mislabeled counter columns). This test reconstructs the C++ list from
source and pins it byte-for-byte against the Python list -- editing one side
without the other fails here instead of on-device.
"""

import re
from pathlib import Path

from tracy.perf_counter_analysis import PERF_COUNTER_CSV_HEADERS


def _repo_root() -> Path:
    # tests/ttnn/tracy/<this> -> repo root is three parents up.
    return Path(__file__).resolve().parents[3]


def _cpp_canonical_headers() -> list:
    """Rebuild canonical_headers() by replaying the same push_back/quad/array
    construction the C++ uses, so the check tracks intent (order + spelling),
    not a copied literal snapshot."""
    src = (_repo_root() / "tt_metal/impl/profiler/perf_counter_metrics.cpp").read_text()

    start = src.index("const std::vector<std::string>& canonical_headers()")
    body = src[start : src.index("return headers;", start)]

    # `const char* NAME[] = { "a", "b", ... };` base-name arrays, looped later.
    arrays = {}
    for m in re.finditer(r"const char\*\s*(\w+)\[\]\s*=\s*\{(.*?)\};", body, re.S):
        arrays[m.group(1)] = re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(2))

    headers = []

    def quad(base, suffix):
        for stat in (" Min", " Median", " Max", " Avg"):
            headers.append(base + stat + suffix)

    # Walk emissions in source order: explicit push_back, standalone quad("..",".."),
    # and `for (const char* b : ARR) quad(b, "SUFFIX")` expanded over the array.
    lines = body.splitlines()
    for i, line in enumerate(lines):
        push = re.search(r'h\.push_back\("((?:[^"\\]|\\.)*)"\)', line)
        loop = re.search(r"for\s*\(const char\*\s*\w+\s*:\s*(\w+)\)", line)
        lit_quad = re.search(r'quad\("((?:[^"\\]|\\.)*)"\s*,\s*"((?:[^"\\]|\\.)*)"\)', line)
        if push:
            headers.append(push.group(1))
        elif loop:
            suffix = next(
                (
                    s.group(1)
                    for j in range(i, min(i + 4, len(lines)))
                    for s in [re.search(r'quad\(b\s*,\s*"((?:[^"\\]|\\.)*)"\)', lines[j])]
                    if s
                ),
                None,
            )
            assert suffix is not None, f"could not resolve quad suffix for loop over {loop.group(1)!r}"
            for base in arrays[loop.group(1)]:
                quad(base, suffix)
        elif lit_quad:
            quad(lit_quad.group(1), lit_quad.group(2))

    return headers


def test_cpp_and_python_counter_headers_match():
    cpp = _cpp_canonical_headers()
    py = list(PERF_COUNTER_CSV_HEADERS)

    # Sanity: the reconstruction actually parsed something (guards a silent
    # regex break that would otherwise make any Python list "match" an empty one).
    assert len(cpp) > 300, f"C++ header reconstruction looks broken: only {len(cpp)} columns parsed"

    if cpp != py:
        for idx, (c, p) in enumerate(zip(cpp, py)):
            if c != p:
                raise AssertionError(
                    f"counter header drift at column {idx}: C++ {c!r} != Python {p!r}. "
                    f"perf_counter_metrics.cpp and perf_counter_analysis.py must stay in lockstep."
                )
        raise AssertionError(
            f"counter header count drift: C++ has {len(cpp)} columns, Python has {len(py)} "
            f"(first {min(len(cpp), len(py))} agree)."
        )
