# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Keep torch's intra-op thread pool to half the CPUs this process may run on.

Most host-side torch work in a pipeline call is a chain of small OpenMP parallel regions (strided
copies, pads, casts, concatenations), and each region waits for all of its threads. When torch has as
many threads as the process has CPUs and some of those CPUs are busy with other processes, every
region waits for the threads stuck behind them. Leaving half the CPUs free lets the scheduler keep
the threads off busy CPUs, whatever the entry point and whether or not the process runs on a subset
of the machine's CPUs.
"""

from __future__ import annotations

import contextlib
import os
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Iterator


def half_of_affinity_mask() -> int:
    """Half the CPUs this process may run on. Uses the affinity mask, which ``os.cpu_count()`` ignores."""
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:  # no sched_getaffinity on this platform
        cpus = os.cpu_count() or 1
    return max(1, cpus // 2)


@contextlib.contextmanager
def cap_torch_threads_to_half_the_cpus() -> Iterator[None]:
    """Run the block with at most half of this process's CPUs as torch threads, then restore the count.

    Only ever lowers the count, so a caller that chose fewer threads (a single-threaded server worker)
    is left alone. The mask is read on every entry. Also usable as a decorator:
    ``@cap_torch_threads_to_half_the_cpus()``.

    torch's thread count is process-wide, so capped blocks running at the same time on different Python
    threads can restore each other's count.
    """
    prev = torch.get_num_threads()
    cap = half_of_affinity_mask()
    if prev <= cap:
        yield
        return
    torch.set_num_threads(cap)
    try:
        yield
    finally:
        torch.set_num_threads(prev)
