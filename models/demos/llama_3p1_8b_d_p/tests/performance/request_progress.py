# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Untimed benchmark progress; these records never establish acceptance."""

import faulthandler
import json
import math
from datetime import datetime, timezone
from pathlib import Path


def parse_warmup_stack_seconds(value):
    seconds = 0.0 if value is None else float(value)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Warmup stack timeout must be finite and nonnegative")
    return seconds


def emit_progress(path, event, **identity):
    record = dict(
        utc=datetime.now(timezone.utc).isoformat(),
        event=event,
        scope="phase_progress_not_acceptance",
        execution_verified=False,
        full_model_accepted=False,
        **identity,
    )
    with Path(path).open("a") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")


def run_observed_request(run, *, emit, warmup_stack_seconds=0, stack_handler=faulthandler, **kwargs):
    seconds = parse_warmup_stack_seconds(warmup_stack_seconds)
    identity = {key: kwargs[key] for key in ("slot", "phase", "repetition", "tokens")}
    on_sample = kwargs.pop("on_sample", None)

    def forward_complete(sample):
        if on_sample is not None:
            on_sample(sample)
        # run_request invokes this after prompt_end and before readback_start.
        emit("forward_complete", **identity, completed_chunks=len(sample["chunks"]))

    emit("request_begin", **identity)
    armed = identity["phase"] == "warmup" and seconds > 0
    if armed:
        stack_handler.dump_traceback_later(seconds, repeat=False, exit=False)
    try:
        result = run(**kwargs, on_sample=forward_complete)
        # The original helper has finished readback, release and synchronization.
        emit("readback_release_complete", **identity, output_checks=len(result[1]))
        return result
    finally:
        if armed:
            stack_handler.cancel_dump_traceback_later()
