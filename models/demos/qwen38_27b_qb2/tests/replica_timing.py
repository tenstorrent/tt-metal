# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Independent host observations of already-enqueued replica work."""

import time
from concurrent.futures import ThreadPoolExecutor


def completion_times(items, synchronize, order, *, clock=time.perf_counter):
    """Timestamp each replica in its own waiter, before joining any waiter.

    ``synchronize`` must release the GIL while blocked, as the pinned TTNN
    synchronize_device binding does. These remain host completion upper bounds;
    they are not device timestamps. Recording after serial waits would charge
    an earlier slow replica's time to every faster replica observed afterward.
    """
    if not items or sorted(order) != list(range(len(items))):
        raise ValueError("Completion order must be a permutation of nonempty replicas")

    def observe(index):
        synchronize(items[index])
        return index, clock()

    result = [None] * len(items)
    with ThreadPoolExecutor(max_workers=len(items)) as workers:
        futures = [workers.submit(observe, index) for index in order]
        for future in futures:
            index, completed = future.result()
            result[index] = completed
    return result
