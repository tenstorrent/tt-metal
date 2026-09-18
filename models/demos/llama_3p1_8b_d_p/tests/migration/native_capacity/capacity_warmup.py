"""Warm production geometry before native clients exist, without serving acknowledgments."""

import hashlib
import json

from capacity_execution import require


def token_digest(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def warmup_geometry(runtime, cache, doc, fixtures, check, completed):
    """Use compile's owning-input path; leave serving IDs and completion sink untouched."""
    runtime._check_ready()
    require(
        runtime.compiled and runtime._layer_completion_sink is None,
        "Warmup must precede installing the native completion sink",
    )
    before_id = runtime._last_request_id
    rows = []
    runtime._active = True
    try:
        for phase in doc["phases"]:
            real = fixtures[phase["fixture"]]
            # Valid vocabulary IDs, with every token distinct from the real book request.
            tokens = [(token + 1) % 128256 for token in real]
            require(all(a != b for a, b in zip(real, tokens)), "Warmup prompt not distinct")
            warmup_hash, real_hash = token_digest(tokens), token_digest(real)
            for call in phase["compute_calls"]:
                check()
                chunk = runtime.make_chunk_input(tokens[call["begin"] : call["end"]], actual_start=call["begin"])
                try:
                    runtime._run(chunk, cache, call["slot"], call["begin"], call["end"], None, None)
                finally:
                    chunk.deallocate(True)
                check()
                row = dict(
                    slot=call["slot"],
                    begin=call["begin"],
                    end=call["end"],
                    prompt_sha256=warmup_hash,
                    real_prompt_sha256=real_hash,
                    native_acks=0,
                )
                rows.append(row)
                completed(row)
    except BaseException:
        runtime._failed = True
        raise
    finally:
        runtime._active = False
    require(
        runtime._last_request_id == before_id and runtime._layer_completion_sink is None,
        "Warmup changed serving request state",
    )
    return dict(
        full32_calls=len(rows),
        calls=rows,
        native_acks=0,
        serving_request_id_before=before_id,
        serving_request_id_after=runtime._last_request_id,
    )


def warmup_barrier(role, run_warmup, publish, wait_peer, timeout):
    """Neither endpoint may construct a native client before the source's complete warmup."""
    if role == "source":
        receipt = run_warmup()
        publish("warmup-finished", warmup=receipt)
        wait_peer("warmup-observed", timeout=timeout)
    else:
        receipt = wait_peer("warmup-finished", timeout=timeout)["warmup"]
        publish("warmup-observed", full32_calls=receipt["full32_calls"])
    return receipt
