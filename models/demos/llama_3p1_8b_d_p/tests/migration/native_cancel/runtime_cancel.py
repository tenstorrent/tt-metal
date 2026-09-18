"""Real runtime/ack seam for the bounded two-epoch cancellation fixture.

The selected page is all valid: [0,32) has no padding in either call. A changed
hash per config/layer rejects a wholly skipped write; it is not a numerical
golden and does not establish that every value was recomputed.
"""

import hashlib
import json
import time

GROUPS = {f"{config}:{layer}" for config in range(16) for layer in range(32)}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def tokens_hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def prompt_pair(fixture, restart_fixture):
    slots = fixture.get("slots")
    require(
        isinstance(slots, list) and len(slots) == 2 and all(isinstance(x, list) and len(x) == 2048 for x in slots),
        "two frozen2K prompts required",
    )
    require(all(type(x) is int and 0 <= x < 128256 for row in slots for x in row), "invalid frozen token ID")
    tested = restart_fixture.get("tokens", {})
    require(tested.get("A") == slots[0][:1033], "restart evidence uses another original prompt")
    fresh = tested.get("C")
    require(
        isinstance(fresh, list) and len(fresh) == 65 and all(type(x) is int and 0 <= x < 128256 for x in fresh),
        "wrong accepted restart fixture",
    )
    require(slots[0][:32] != fresh[:32], "restart requires a distinct valid input page")
    return slots[0], fresh[:32]


def validate_real_chunk(row, *, epoch, nonce, ids):
    request = 0 if epoch == "a" else 1
    end = 1024 if epoch == "a" else 32
    require(epoch in ("a", "b") and len(ids) == end, "wrong real chunk")
    require(
        row.get("run_nonce") == nonce and row.get("epoch") == epoch and row.get("request_id") == request,
        "stale real chunk identity",
    )
    require((row.get("slot"), row.get("begin"), row.get("end")) == (0, 0, end), "wrong real range")
    require(row.get("token_ids_sha256") == tokens_hash(ids), "wrong real chunk input")
    require(
        row.get("real_h2d") is True
        and row.get("borrowed_input_preserved") is True
        and row.get("metadata_rows") == [[0, 0, end]] * 32,
        "H2D contract incomplete",
    )
    before, after = row.get("source_before", {}), row.get("source_after", {})
    require(set(before) == set(after) == GROUPS, "selected source page inventory differs")
    require(all(before[key] != after[key] for key in GROUPS), "missing valid source write")
    sync, captured = row.get("synchronized_ns"), row.get("captured_ns")
    require(
        type(sync) is int and type(captured) is int and 0 < sync < captured,
        "source snapshot was not captured after synchronization",
    )
    acks = row.get("acks", [])
    require(
        [(x.get("layer"), x.get("request_id")) for x in acks] == [(layer, request) for layer in range(32)],
        "exact32 layer acknowledgments required",
    )
    times = [x.get("published_ns") for x in acks]
    require(
        all(type(x) is int for x in times) and captured < times[0] and all(a < b for a, b in zip(times, times[1:])),
        "source snapshot must precede every published ack",
    )
    require(row.get("sync_count") == 1, "exactly one runtime synchronization required")
    return row


def produce_real_chunk(
    runtime, cache, *, epoch, nonce, request_id, ids, receive, check_borrowed, capture, push, clock=time.monotonic_ns
):
    end = 1024 if epoch == "a" else 32
    require(
        epoch in ("a", "b") and request_id == (0 if epoch == "a" else 1) and len(ids) == end, "wrong real fixture call"
    )
    require(runtime.config.num_layers == 32, "exact32 runtime layers required")
    row = dict(
        run_nonce=nonce,
        epoch=epoch,
        request_id=request_id,
        slot=0,
        begin=0,
        end=end,
        token_ids_sha256=tokens_hash(ids),
        source_before=capture("prewrite"),
        source_after={},
        acks=[],
        real_h2d=True,
        borrowed_input_preserved=False,
        sync_count=0,
    )
    require(set(row["source_before"]) == GROUPS, "prewrite page inventory differs")
    packet = receive(ids, end)
    row["metadata_rows"] = packet["metadata_rows"]
    require(row["metadata_rows"] == [[0, 0, end]] * 32, "H2D metadata differs")
    original_sync = runtime._synchronize

    def synchronized(mesh):
        original_sync(mesh)
        row["sync_count"] += 1
        row["synchronized_ns"] = clock()

    def completed(layer, request):
        require(
            request == request_id and layer == len(row["acks"]) and 0 <= layer < 32,
            "extra or reordered runtime acknowledgment",
        )
        require(row["sync_count"] == 1, "runtime acknowledgment preceded synchronization")
        if layer == 0:
            row["source_after"] = capture("postwrite")
            row["captured_ns"] = clock()
            require(set(row["source_after"]) == GROUPS, "postwrite page inventory differs")
            require(
                all(row["source_before"][key] != row["source_after"][key] for key in GROUPS),
                "missing valid source write",
            )
        push(layer, request)
        row["acks"].append(dict(layer=layer, request_id=request, published_ns=clock()))

    runtime._synchronize = synchronized
    runtime.set_layer_completion_sink(completed)
    try:
        runtime.prefill_chunk(
            packet["tokens"],
            cache,
            slot_id=0,
            actual_start=0,
            actual_end=end,
            request_id=request_id,
            metadata_msg=packet["metadata"],
        )
        check_borrowed(packet)
        row["borrowed_input_preserved"] = True
        return validate_real_chunk(row, epoch=epoch, nonce=nonce, ids=ids)
    finally:
        runtime._synchronize = original_sync
        # A failed production call poisons the runtime. Its public setter correctly
        # refuses work; do not mask that primary error while retaining native owners.
        if not runtime._failed:
            runtime.set_layer_completion_sink(None)
