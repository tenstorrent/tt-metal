# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Select candidate kernel sources while keeping the evaluator/runtime shared."""

import os
from pathlib import Path
import json
import time

import pytest

_session_start = None
_collection_end = None
_reports = []


def permute_packed_bits(packed, permutation):
    result = packed.clone().zero_()
    for output_bit, source_bit in enumerate(permutation):
        result |= ((packed >> source_bit) & 1) << output_bit
    return result


def pytest_sessionstart(session):
    global _session_start, _collection_end
    _session_start, _collection_end = time.time(), None
    _reports.clear()


def pytest_collection_finish(session):
    global _collection_end
    _collection_end = time.time()


def pytest_runtest_logreport(report):
    _reports.append(dict(nodeid=report.nodeid, phase=report.when, outcome=report.outcome, seconds=report.duration))


def pytest_sessionfinish(session, exitstatus):
    destination = os.environ.get("NVFP4_PYTEST_METRICS")
    if destination:
        payload = dict(
            session_start_unix=_session_start,
            session_end_unix=time.time(),
            collection_seconds=None if _collection_end is None else _collection_end - _session_start,
            collected=session.testscollected,
            exitstatus=int(exitstatus),
            reports=_reports,
        )
        Path(destination).write_text(json.dumps(payload, indent=2) + "\n")


@pytest.fixture(autouse=True)
def nvfp4_candidate_kernel_directory(monkeypatch):
    candidate = os.environ.get("NVFP4_KERNEL_ROOT")
    if candidate:
        # Kernel lookup checks CWD before TT_METAL_KERNEL_PATH. The safe runner
        # initially enters the runtime tree; enter the candidate before fixtures
        # create programs so that the runtime cannot silently use baseline code.
        monkeypatch.chdir(Path(candidate).resolve(strict=True))
    permutation = json.loads(os.environ.get("NVFP4_INPUT_BIT_PERMUTATION", "null"))
    if permutation is not None:
        import ttnn

        original_pack = ttnn.experimental.nvfp4.pack

        def pack_with_encoding(*args, **kwargs):
            packed, scales = original_pack(*args, **kwargs)
            if kwargs.get("experimental_fast", False):
                packed = permute_packed_bits(packed, permutation)
            return packed, scales

        monkeypatch.setattr(ttnn.experimental.nvfp4, "pack", pack_with_encoding)
