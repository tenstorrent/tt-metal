# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Pure-CPU timing definitions and validation for the configured-length eager prefill benchmark."""

import math
import statistics

TOKENS = 2048
SLOTS = (0, 1)
MEASURED = 3


class CleanupError(RuntimeError):
    """One or more owned-resource cleanup actions failed after all actions were attempted."""

    def __init__(self, failures):
        self.failures = list(failures)
        super().__init__(f"{len(self.failures)} owned-resource cleanup action(s) failed")


def _failure(action, error):
    return dict(action=action, exception_type=type(error).__name__, message=str(error))


def _release_owned(retained, release):
    values = list(retained)
    retained.clear()
    failures = []
    for index, value in enumerate(values):
        try:
            release(value)
        except BaseException as error:
            failures.append(_failure(f"retained[{index}]", error))
    return failures


def release_owned_resources(retained, release):
    """Release every retained value once, then fail after all attempts if any release failed."""
    failures = _release_owned(retained, release)
    if failures:
        raise CleanupError(failures)


def finalize_owned_resources(*, retained, release, actions, report, persist, primary_error):
    """Attempt all final cleanup and report persistence without masking an active primary error."""
    failures = list(primary_error.failures) if isinstance(primary_error, CleanupError) else []
    failures.extend(_release_owned(retained, release))
    for name, action in actions:
        try:
            action()
        except BaseException as error:
            failures.append(_failure(name, error))
    if failures:
        report["cleanup_errors"] = failures
    try:
        persist()
    except BaseException as error:
        failures.append(_failure("report.persist", error))
        report["cleanup_errors"] = failures
    if failures and primary_error is None:
        raise CleanupError(failures)
    return failures


def chunk_plan(tokens):
    if type(tokens) is not int or not 0 < tokens <= 131072 or tokens % 1024:
        raise ValueError("tokens must be a positive multiple of 1024 through 131072")
    return tuple((start, start + 1024) for start in range(0, tokens, 1024))


def request_schedule():
    return [(slot, "warmup", 0) for slot in SLOTS] + [
        (slot, "measured", rep) for rep in range(MEASURED) for slot in SLOTS
    ]


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def summary(values):
    if not values or not all(positive(value) for value in values):
        raise ValueError("Expected finite positive seconds or rates")
    return dict(
        count=len(values), minimum=min(values), median=statistics.median(values), maximum=max(values), samples=values
    )


def request_sample(
    *, slot, phase, repetition, prompt_start, prompt_end, chunks, programs_before, programs_after, tokens=TOKENS
):
    plan = chunk_plan(tokens)
    if type(slot) is not int or slot not in SLOTS or phase not in ("warmup", "measured"):
        raise ValueError("Invalid slot or phase")
    expected_repetitions = (0,) if phase == "warmup" else range(MEASURED)
    if type(repetition) is not int or repetition not in expected_repetitions:
        raise ValueError("Invalid repetition")
    if [tuple((row["start"], row["end"])) for row in chunks] != list(plan):
        raise ValueError("Every ordered 1024-token chunk is required exactly once")
    if not all(type(value) in (int, float) and math.isfinite(value) for value in (prompt_start, prompt_end)):
        raise ValueError("Invalid prompt clock")
    previous = prompt_start
    normalized = []
    for chunk in chunks:
        values = [chunk[key] for key in ("chunk_start", "forward_start", "forward_end")]
        if not all(type(value) in (int, float) and math.isfinite(value) for value in values):
            raise ValueError("Invalid chunk clock")
        begin, forward_begin, end = values
        if not previous <= begin <= forward_begin < end <= prompt_end:
            raise ValueError("Chunk timing intervals overlap or escape prompt window")
        previous = end
        normalized.append(
            dict(
                start=chunk["start"],
                end=chunk["end"],
                tokens=1024,
                prompt_chunk_wall_seconds=end - begin,
                forward_wall_seconds=end - forward_begin,
                upload_and_sync_wall_seconds=forward_begin - begin,
            )
        )
    elapsed = prompt_end - prompt_start
    if not positive(elapsed):
        raise ValueError("Prompt wall denominator must be positive")
    forward = sum(row["forward_wall_seconds"] for row in normalized)
    if any(type(value) is not int or value < 0 for value in (programs_before, programs_after)):
        raise ValueError("Invalid program-cache counts")
    return dict(
        slot=slot,
        phase=phase,
        repetition=repetition,
        tokens=tokens,
        prompt_wall_seconds=elapsed,
        forward_wall_seconds=forward,
        tokens_per_second_per_user=tokens / elapsed,
        prompt_unattributed_wall_seconds=elapsed - sum(row["prompt_chunk_wall_seconds"] for row in normalized),
        programs_before=programs_before,
        programs_after=programs_after,
        chunks=normalized,
        clock_timestamps=dict(prompt_start=prompt_start, prompt_end=prompt_end, chunks=chunks),
    )


def validate_sample(sample):
    rebuilt = request_sample(
        tokens=sample["tokens"],
        slot=sample["slot"],
        phase=sample["phase"],
        repetition=sample["repetition"],
        prompt_start=sample["clock_timestamps"]["prompt_start"],
        prompt_end=sample["clock_timestamps"]["prompt_end"],
        chunks=sample["clock_timestamps"]["chunks"],
        programs_before=sample["programs_before"],
        programs_after=sample["programs_after"],
    )
    if rebuilt != sample:
        raise ValueError("Reported timing or throughput differs from raw clocks")
    if sample["phase"] == "measured" and sample["programs_before"] != sample["programs_after"]:
        raise ValueError("Program cache grew during a measured request")


def aggregate(samples):
    expected_order = request_schedule()
    capacities = {sample["tokens"] for sample in samples}
    if len(capacities) != 1:
        raise ValueError("Every request must use the same logical capacity")
    tokens = next(iter(capacities))
    plan = chunk_plan(tokens)
    actual = [(s["slot"], s["phase"], s["repetition"]) for s in samples]
    if actual != expected_order:
        raise ValueError("Need one warmup per slot followed by three alternating measured requests per slot")
    for sample in samples:
        validate_sample(sample)
    groups = []
    for slot in SLOTS:
        rows = [row for row in samples if row["slot"] == slot and row["phase"] == "measured"]
        groups.append(
            dict(
                slot=slot,
                measured_requests=MEASURED,
                prompt_wall_seconds=summary([row["prompt_wall_seconds"] for row in rows]),
                forward_wall_seconds=summary([row["forward_wall_seconds"] for row in rows]),
                tokens_per_second_per_user=summary([row["tokens_per_second_per_user"] for row in rows]),
                chunks=[
                    dict(
                        start=start,
                        end=end,
                        prompt_chunk_wall_seconds=summary(
                            [row["chunks"][index]["prompt_chunk_wall_seconds"] for row in rows]
                        ),
                        forward_wall_seconds=summary([row["chunks"][index]["forward_wall_seconds"] for row in rows]),
                        upload_and_sync_wall_seconds=summary(
                            [row["chunks"][index]["upload_and_sync_wall_seconds"] for row in rows]
                        ),
                    )
                    for index, (start, end) in enumerate(plan)
                ],
            )
        )
    return dict(
        per_slot=groups,
        warmups=[row for row in samples if row["phase"] == "warmup"],
        measurement="Synchronized eager host wall; forward-only is the sum of every model-call interval",
        throughput=f"{tokens} tokens divided by each single-user prompt wall; two slots execute sequentially",
        cold_compile="First slot0 warmup includes fresh-JIT compile effects; no isolated compile-time claim",
    )


def run_request(
    *,
    tokens,
    slot,
    phase,
    repetition,
    upload,
    forward,
    synchronize,
    program_count,
    inspect,
    release,
    clock,
    retained,
    on_sample=None,
):
    """Time the original eager request sequence; inspect/release only after prompt completion.

    The caller retains cleanup responsibility if upload, forward or inspection raises.
    No callback is permitted to read model outputs inside upload or forward.
    """
    plan = chunk_plan(tokens)
    if retained:
        raise ValueError("Each request must start without retained outputs")
    synchronize()
    programs_before = program_count()
    prompt_start = clock()
    chunks = []
    outputs = []
    for start, end in plan:
        chunk_start = clock()
        token = upload(start, end)
        retained.append(token)
        synchronize()
        forward_start = clock()
        output = forward(token, start, end)
        try:
            synchronize()
        except BaseException:
            retained.append(output)
            raise
        forward_end = clock()
        retained.append(output)
        outputs.append(output)
        chunks.append(
            dict(start=start, end=end, chunk_start=chunk_start, forward_start=forward_start, forward_end=forward_end)
        )
    prompt_end = clock()
    programs_after = program_count()
    sample = request_sample(
        tokens=tokens,
        slot=slot,
        phase=phase,
        repetition=repetition,
        prompt_start=prompt_start,
        prompt_end=prompt_end,
        chunks=chunks,
        programs_before=programs_before,
        programs_after=programs_after,
    )
    if on_sample is not None:
        on_sample(sample)
    readback_start = clock()
    checks = []
    for output, (start, end) in zip(outputs, plan):
        checks.extend(inspect(output, start, end))
    readback_seconds = clock() - readback_start
    release_owned_resources(retained, release)
    synchronize()
    return sample, checks, readback_seconds
