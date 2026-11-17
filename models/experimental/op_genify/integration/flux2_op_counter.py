# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""pytest plugin: count which matmul ran per test (genify_matmul vs. the stock ops tt_dit calls).

Proves the TT_GENIFY_MATMUL switch in models/tt_dit/layers/linear.py really routes a test's matmuls
(and shows any stock fallback). Load it with
    scripts/run_safe_pytest.sh <test> -p models.experimental.op_genify.integration.flux2_op_counter
A per-test line and a total are printed in the terminal summary.
"""

import collections
import functools

import pytest

_current: collections.Counter = collections.Counter()
_per_test: dict[str, dict[str, int]] = {}


def _wrap(owner, name, label):
    fn = getattr(owner, name, None)
    if fn is None:
        return

    @functools.wraps(fn)
    def counted(*args, **kwargs):
        _current[label] += 1
        return fn(*args, **kwargs)

    setattr(owner, name, counted)


def pytest_configure(config):
    import ttnn
    from models.experimental.op_genify import genify_matmul_adapter

    _wrap(genify_matmul_adapter, "matmul", "genify_matmul")
    _wrap(ttnn, "linear", "ttnn.linear")
    _wrap(ttnn, "matmul", "ttnn.matmul")
    _wrap(ttnn.experimental, "minimal_matmul", "minimal_matmul")
    _wrap(ttnn.experimental, "minimal_matmul_split", "minimal_matmul_split")
    _wrap(ttnn.experimental, "dit_minimal_matmul_addcmul_fused", "dit_minimal_matmul_addcmul_fused")
    _wrap(ttnn.experimental, "all_gather_minimal_matmul_async", "all_gather_minimal_matmul_async")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    _current.clear()
    yield
    _per_test[item.nodeid] = dict(_current)


def pytest_terminal_summary(terminalreporter):
    total = collections.Counter()
    terminalreporter.write_sep("=", "matmul calls per test (flux2_op_counter)")
    for nodeid, counts in _per_test.items():
        total.update(counts)
        terminalreporter.write_line(f"{nodeid.split('::')[-1]}: {dict(sorted(counts.items()))}")
    terminalreporter.write_line(f"TOTAL: {dict(sorted(total.items()))}")
