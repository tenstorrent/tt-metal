# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""X.10 (host only): a test_profile host cProfile (<task>_host<suffix>.pstats, BRINGUP_PROFILE_HOST=1) split into
the main thread's host time a chunk: ttnn op launches (FastOperation.__call__, CCLs included), the engram waits
(take_engram_rows -> _wait_job), begin_chunk (tails, row allocation, job submission), frees (ccl.dealloc / model.free),
the final sync (the device draining), and the rest (Python glue). cProfile sees the main thread only: the engram
worker's hash / gather / upload are in the idle map (tests/perf/test_host_idle_map.py), not here.

    PYTHONPATH=$PWD python -m models.demos.deepseek_v41_flash_d_p.tests.perf.host_split <pstats> [runs]
"""

from __future__ import annotations

import pstats
import sys


def _pick(st, file_tail: str, name: str):
    return [v for (f, _, n), v in st.items() if n == name and f.endswith(file_tail)]


def split(path: str, runs: int = 5) -> dict:
    st = pstats.Stats(path).stats
    cum = lambda rows: sum(v[3] for v in rows) * 1e3 / runs  # noqa: E731
    calls = lambda rows: round(sum(v[1] for v in rows) / runs)  # noqa: E731
    total = cum(_pick(st, "profile.py", "run"))
    ops = _pick(st, "decorators.py", "__call__")
    out = {
        "chunk_cprofile_ms": total,
        "op_launches_ms": cum(ops),
        "op_launch_calls": calls(ops),
        "engram_wait_ms": cum(_pick(st, "model.py", "_wait_job")),
        "begin_chunk_ms": cum(_pick(st, "model.py", "begin_chunk")),
        "embed_ms": cum(_pick(st, "model.py", "embed")),
        "frees_ms": cum(_pick(st, "ccl.py", "dealloc")),
        "final_sync_ms": cum(_pick(st, "model.py", "sync")),
    }
    # ops launched inside begin_chunk / embed / dealloc are counted once (as launches), not again in their parents
    seen = sum(v for k, v in out.items() if k.endswith("_ms") and k != "chunk_cprofile_ms")
    out["python_other_ms"] = total - seen
    top = sorted(((v[2] * 1e3 / runs, f"{f.split('/')[-1]}:{ln} {n}") for (f, ln, n), v in st.items()), reverse=True)
    out["top_self"] = [(round(ms, 2), fn) for ms, fn in top[:15]]
    return out


if __name__ == "__main__":
    r = split(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 5)
    for k, v in r.items():
        print(f"{k}: {v if not isinstance(v, float) else round(v, 2)}")
