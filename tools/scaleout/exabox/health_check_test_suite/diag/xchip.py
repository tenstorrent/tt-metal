# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""pytest plugin for the bus-bar didt load (-p diag.xchip): at each determinism check a chip whose output differs
from the largest group of identical outputs is logged as 'Device N output differs from the other chips' and fails
the test. It wraps OpTestBase's determinism step, so no metal code changes."""

import concurrent.futures

_device_ids: list = []


def cross_chip_outliers(outputs, equal) -> list[int]:
    """Indices of outputs not equal to the largest group of equal outputs."""
    groups: list[list[int]] = []
    for i, t in enumerate(outputs):
        for g in groups:
            if equal(outputs[g[0]], t):
                g.append(i)
                break
        else:
            groups.append([i])
    if len(groups) < 2:
        return []
    keep = max(groups, key=len)
    return sorted(i for g in groups if g is not keep for i in g)


def _pool(torch, logger):
    class Pool(concurrent.futures.ThreadPoolExecutor):
        # The determinism step is the one map(torch.equal, reference_outputs, outputs) call.
        def map(self, fn, *iterables, **kw):
            if fn is not torch.equal or len(iterables) != 2:
                return super().map(fn, *iterables, **kw)
            refs, outputs = list(iterables[0]), list(iterables[1])
            bad = set(cross_chip_outliers(outputs, torch.equal))
            for i in sorted(bad):
                dev = _device_ids[i] if i < len(_device_ids) else i
                logger.warning(f"Device {dev} output differs from the other chips")
            return (False if i in bad else m for i, m in enumerate(super().map(fn, refs, outputs, **kw)))

    return Pool


def pytest_configure(config):
    import torch
    from tt_py_test_utils_common import op_test_base as m

    m.ThreadPoolExecutor = _pool(torch, m.logger)
    run = m.OpTestBase.run_op_test

    def run_op_test(self):
        _device_ids[:] = list(self.device_ids)
        return run(self)

    m.OpTestBase.run_op_test = run_op_test
