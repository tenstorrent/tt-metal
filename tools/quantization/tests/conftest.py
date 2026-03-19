# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

_results = []


@pytest.fixture
def record_pcc_result():
    """Fixture that returns a callable for tests to register PCC results."""

    def _record(*, test, fidelity, iters, pcc_before, pcc_after):
        improved = pcc_after > pcc_before
        _results.append(
            {
                "test": test,
                "fidelity": fidelity,
                "iters": iters,
                "pcc_before": pcc_before,
                "pcc_after": pcc_after,
                "delta": pcc_after - pcc_before,
                "improved": improved,
            }
        )

    return _record


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not _results:
        return
    terminalreporter.section("Quantization Rounding Correction Summary")
    for r in _results:
        icon = "\u2705" if r["improved"] else "\u274c"
        terminalreporter.write_line(f"  {icon} {r['test']} | fidelity={r['fidelity']} | iters={r['iters']}")
        terminalreporter.write_line(f"       PCC before: {r['pcc_before']:.6f}")
        terminalreporter.write_line(f"       PCC after:  {r['pcc_after']:.6f}")
        terminalreporter.write_line(f"       Delta:      {r['delta']:+.6f}")
