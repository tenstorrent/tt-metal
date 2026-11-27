# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Mark CI failures that a re-run on the same commit cannot fix, so auto-retry skips them.

Opt-in: nothing here has any effect unless the job sets CI_FAILURE_RECORDS to a directory. Each
deterministic failure then writes one file there, and the job's classify-failure step turns the
directory into a ``ci-no-retry`` annotation that the auto-retry workflow reads. See
.github/actions/classify-failure/README.md.

One deterministic failure makes the whole job non-retryable: by policy it repeats on a re-run, so the
job cannot go green whatever else failed alongside it. Raise these only for a verdict on a complete
run -- a perf value outside its baseline band. A missing measurement, a crash before the gate, or a
misrouted host is not one.

Stdlib only, so standalone gate scripts and bare worker hosts can use it. Shell scripts can call
``python3 -m models.common.ci_failure "<reason>"`` (or this file directly), or write a one-line .txt
into $CI_FAILURE_RECORDS themselves.
"""

import os
import socket
import sys
import uuid

RECORDS_ENV = "CI_FAILURE_RECORDS"


class DeterministicFailure(AssertionError):
    """A test failure that will repeat on a re-run of the same commit.

    Subclasses AssertionError, so pytest reports it, -x stops on it and pytest.raises(AssertionError)
    catches it exactly as before. Under pytest the root conftest records it automatically.
    """


class PerfGateFailure(DeterministicFailure):
    """A measured perf value outside its configured baseline band."""


def assert_perf_gate(ok, message):
    """``assert ok, message`` for a perf verdict: raises PerfGateFailure, so auto-retry skips the job."""
    if not ok:
        raise PerfGateFailure(message)


def record_deterministic_failure(reason):
    """Record one deterministic failure for this job. No-op unless CI_FAILURE_RECORDS is set.

    Never raises: losing a record only costs a re-run, which is what happens without this module.
    """
    root = os.environ.get(RECORDS_ENV)
    if not root:
        return
    name = f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    tmp = os.path.join(root, f".{name}.tmp")
    try:
        os.makedirs(root, exist_ok=True)
        # Written aside and renamed into place: classify-failure only reads *.txt, so it never sees a
        # half-written record. One file per record, so concurrent writers (ranks, workers) never collide.
        with open(tmp, "w") as fh:
            fh.write(" ".join(str(reason).split())[:500] or "deterministic failure (no reason given)")
            fh.write("\n")
        os.replace(tmp, os.path.join(root, f"{name}.txt"))
    except OSError as exc:
        print(f"ci_failure: deterministic failure not recorded ({exc})", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(f"usage: {sys.argv[0]} <reason>")
    record_deterministic_failure(" ".join(sys.argv[1:]))
