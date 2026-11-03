# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Device-free batching logic of the rerank server worker (pattern from Aniruddha Tupe's strands-decider server,
tt-metal atupe/strands-decider-fastapi models/demos/strands_agent/serve/batcher.py).

An item is ``{"id": int, "ids": [token id list per pair], "t_submit": float}``: one request chunk."""

import queue
import time


def collect_batch(get, max_wait_s, max_pairs, carry=None, clock=time.monotonic):
    """Block for one item (or start from ``carry``), then take more while the batch stays within max_pairs pairs.

    Before max_wait_s after the first item, block for new items; after it (or when max_wait_s is 0), take only
    what is already queued. An item that would push the batch over max_pairs is returned as the next carry, so the
    cap is hard (a single item larger than max_pairs still runs alone). get(timeout) returns an item or raises
    queue.Empty (None blocks, 0 does not block); an item None means stop. Returns (items, stop, carry)."""
    first = carry if carry is not None else get(None)
    if first is None:
        return [], True, None
    items = [first]
    count = len(first["ids"])
    deadline = clock() + max_wait_s
    while count < max_pairs:
        remaining = deadline - clock()
        try:
            item = get(remaining if remaining > 0 else 0)
        except queue.Empty:
            break
        if item is None:
            return items, True, None
        if count + len(item["ids"]) > max_pairs:
            return items, False, item
        items.append(item)
        count += len(item["ids"])
    return items, False, None


def run_batch(score_ids, items, batch_id):
    """Score every item's pairs with one score_ids call; a device error fails every item of the batch."""
    t_batch_start = time.time()
    flat = [ids for item in items for ids in item["ids"]]
    try:
        t1 = time.perf_counter()
        logits = score_ids(flat)
        device_ms = (time.perf_counter() - t1) * 1000
    except Exception as e:
        return [{"id": item["id"], "ok": False, "status": 500, "detail": repr(e)} for item in items]
    if len(logits) != len(flat):
        detail = f"score_ids returned {len(logits)} logits for {len(flat)} pairs"
        return [{"id": item["id"], "ok": False, "status": 500, "detail": detail} for item in items]
    timing = {
        "batch_id": batch_id,
        "batch_tasks": len(items),
        "batch_pairs": len(flat),
        "device_ms": round(device_ms, 3),
        "t_batch_start": t_batch_start,
    }
    results, pos = [], 0
    for item in items:
        n = len(item["ids"])
        results.append(
            {"id": item["id"], "ok": True, "logits": [float(x) for x in logits[pos : pos + n]], "timing": timing}
        )
        pos += n
    return results
