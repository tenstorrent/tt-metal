#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    dump_circular_buffers

Description:
    Circular buffer state of every running Tensix core, one row per CB of the current program.

    The launch message's local_cb_mask says which CBs exist, and their geometry is the host-written
    config in the kernel config buffer. Kind is local for a CB whose producer and consumer are on this
    core, or global sender / global receiver for this core's end of a GlobalCircularBuffer. A receiver
    usually also has a local CB over the same memory; that is a separate queue, not a duplicate.

    Pushed/Popped count pages pushed by the producer and popped by the consumer, and Occupancy is the
    difference. Local counters are stream registers, reset for every program and compared modulo 2^16
    like the kernels do. Global counters live in L1 for the buffer's lifetime, and a global sender
    shows its slowest receiver.

    CB content needs --include-cb-content.


Owner:
    onenezicTT
"""

from circular_buffer_provider import run as get_circular_buffer_provider
from run_checks import run as get_run_checks
from triage import ScriptConfig, run_script
from ttexalens.context import Context

script_config = ScriptConfig(
    depends=["run_checks", "circular_buffer_provider"],
)


def run(args, context: Context):
    run_checks = get_run_checks(args, context)
    provider = get_circular_buffer_provider(args, context)
    return run_checks.run_per_block_check(provider.get_cached_circular_buffers, block_filter="tensix")


if __name__ == "__main__":
    run_script()
