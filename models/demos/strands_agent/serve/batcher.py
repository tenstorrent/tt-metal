# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-free batching logic of the server worker."""

import queue
import time

from models.demos.strands_agent.tt.answers import build_answer


def collect_batch(get, max_wait_s, max_questions, clock=time.monotonic):
    """Block for one item, then take more until max_questions questions.

    Before max_wait_s after the first item, block for new items; after it (or when max_wait_s is 0), take only
    what is already queued. get(timeout) returns an item or raises queue.Empty (None blocks, 0 does not block);
    an item None means stop. Returns (items, stop)."""
    first = get(None)
    if first is None:
        return [], True
    items = [first]
    count = len(first["qs"])
    deadline = clock() + max_wait_s
    while count < max_questions:
        remaining = deadline - clock()
        try:
            item = get(remaining if remaining > 0 else 0)
        except queue.Empty:
            break
        if item is None:
            return items, True
        items.append(item)
        count += len(item["qs"])
    return items, False


def run_batch(decider, items, batch_id):
    """Answer the items (pre-encoded questions) with one device call; a device error fails every item."""
    t_batch_start = time.time()
    flat = [q for item in items for q in item["qs"]]
    try:
        t1 = time.perf_counter()
        logits = iter(decider.question_logits_batch(flat))
        device_ms = (time.perf_counter() - t1) * 1000
        cfg = decider.config
        timing = {
            "batch_id": batch_id,
            "batch_requests": len(items),
            "batch_questions": len(flat),
            "device_ms": round(device_ms, 3),
            "t_batch_start": t_batch_start,
        }
        results = []
        for item in items:
            qs = item["qs"]
            answers = {
                q.name: build_answer(
                    q, next(logits)[0], cfg.temperature_by_kind, cfg.temperature, cfg.ordinal_smoothing
                )
                for q in qs
            }
            usage = {"input_tokens": sum(len(q.input_ids) for q in qs), "output_tokens": len(qs)}
            results.append(
                {"id": item["id"], "ok": True, "result": {"answers": answers, "usage": usage}, "timing": timing}
            )
        return results
    except Exception as e:
        return [{"id": item["id"], "ok": False, "status": 500, "detail": repr(e)} for item in items]
