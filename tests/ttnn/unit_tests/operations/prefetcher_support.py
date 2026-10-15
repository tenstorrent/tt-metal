# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Leaf helpers for tensor-prefetcher tests: the support skip gate and the pipes it delivers into.

Kept free of model/tracy/nightly-test imports so a test that only needs these does not pull in
prefetcher_common's dependency chain at collection time.
"""

import pytest
import ttnn


def require_tensor_prefetcher(device):
    """Skip unless the tensor prefetcher is usable on this device.

    Two things make it unusable, and the skip names both because the fix differs: no programmable
    DRAM cores, or the streaming profiler holding the same DRISCs.
    """
    if not ttnn.experimental.is_tensor_prefetcher_supported(device):
        pytest.skip(
            "tensor prefetcher unsupported here: either programmable DRAM cores are unavailable "
            "(need Blackhole and firmware >= 19.12.0.0), or TT_METAL_STREAMING_PROFILER=1 is set and its "
            "relay holds the DRISCs the prefetcher needs"
        )


def make_tensor_prefetcher_pipes(device, bank_to_receivers, ring_size, single_sender=False):
    """A PrefetcherPipeSpace over exactly the receivers in ``bank_to_receivers`` and the Tensor-prefetcher
    pipes carved from it, every ring ``ring_size`` bytes.

    ``single_sender`` gives each bank one pipe, as a K-row-major weight needs, rather than splitting a
    bank's receivers across its two DRISC senders. Returns (space, pipes); the pipes keep the space
    alive, so a caller needs the space only to carve more pipes from it. Call it before any Program runs
    on the receivers: pipe rings come from the persistent L1 arena, which refuses a core a live Program
    holds.
    """
    receiver_domain = ttnn.CoreRangeSet(
        {core_range for _, receivers in bank_to_receivers for core_range in receivers.ranges()}
    )
    space = ttnn.experimental.create_prefetcher_pipe_space(
        device,
        sender_cores=ttnn.CoreRangeSet(set()),
        receiver_domain=receiver_domain,
        ring_size=ring_size,
        max_receivers_per_pipe=max(receivers.num_cores() for _, receivers in bank_to_receivers),
        # Every bank has two DRISC senders.
        num_dram_senders=2 * len(bank_to_receivers),
    )
    pipes = ttnn.experimental.create_prefetcher_pipes_for_tensor_prefetcher(
        space, bank_to_receivers, support_multi_receiver_shards=single_sender
    )
    return space, pipes
