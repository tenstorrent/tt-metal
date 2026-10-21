#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Add host call-boundary timestamps to the existing asynchronous prefill owner.

No device synchronization is added. GPT-OSS's existing asynchronous completion
queue joins after all decoder-layer events, before prefill_chunk returns.
These intervals describe host-observed prefill service, not device-only cycles.
"""
import functools
import inspect
import json
import os
import resource
import time
from pathlib import Path

from models.demos.common.prefill.runners import prefill_runner as runner


def main():
    path = Path(os.environ["PREFILL_PERF_EVENTS"])
    path.parent.mkdir(parents=True, exist_ok=True)
    original = runner._compute_and_send
    signature = inspect.signature(original)

    @functools.wraps(original)
    def timed(*args, **kwargs):
        bound = signature.bind(*args, **kwargs).arguments
        start = original(*args, **kwargs)
        end = time.time()
        meta = bound["meta"]
        record = dict(
            kind="prefill_chunk",
            rank=bound["rank"],
            chunk_id=bound["c"],
            slot=meta["slot_id"],
            from_token=meta["actual_start"],
            to_token=meta["actual_end"],
            start_epoch=start,
            end_epoch=end,
            duration_s=end - start,
            boundary="host compute call; existing asynchronous layer completion drain",
        )
        with path.open("a") as output:
            output.write(json.dumps(record) + "\n")
        return start

    runner._compute_and_send = timed
    _, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    resource.setrlimit(resource.RLIMIT_NPROC, (hard, hard))
    runner.main()


if __name__ == "__main__":
    main()
