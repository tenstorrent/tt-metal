"""Device-free batching logic of the embedding server's chip worker (unit-tested in test_emb_serve.py).

Pattern from Aniruddha's strands-decider server (atupe/strands-decider-fastapi, serve/batcher.py): the worker blocks
for one task, then takes more until a cap on work units. Here a unit is one text (one row of the forward)."""

import queue
import time


def collect_batch(get, max_wait_s, max_texts, clock=time.monotonic):
    """Block for one task, then take more until max_texts texts.

    Before max_wait_s after the first task, block for new tasks; after it (or when max_wait_s is 0), take only what is
    already queued. get(timeout) returns a task or raises queue.Empty (None blocks, 0 does not block); a task None
    means stop. A task is (req_id, payload); every payload is one text. Returns (tasks, stop)."""
    first = get(None)
    if first is None:
        return [], True
    tasks = [first]
    deadline = clock() + max_wait_s
    while len(tasks) < max_texts:
        remaining = deadline - clock()
        try:
            task = get(remaining if remaining > 0 else 0)
        except queue.Empty:
            break
        if task is None:
            return tasks, True
        tasks.append(task)
    return tasks, False


def right_pad(id_lists, pad_id=0):
    """B lists of token ids -> (input_ids, attention_mask) as nested lists [B, L], right-padded to the longest."""
    width = max(len(ids) for ids in id_lists)
    input_ids = [list(ids) + [pad_id] * (width - len(ids)) for ids in id_lists]
    mask = [[1] * len(ids) + [0] * (width - len(ids)) for ids in id_lists]
    return input_ids, mask
