# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0

"""Device open for Inkling-2-Mini decode: a bigger kernel-config ring.

The dispatcher writes each program's kernel binaries and runtime args into a ring in worker L1 (from the kernel
config base up to the allocator's L1 base, 69 KB by default). A decode step is ~690 programs; when the programs in
flight fill the ring (the 6-kernel message-gate program, the 83-core MoE program), the next launch waits for an older
program to finish and only then gets its binaries: ~5-7 us gaps before those programs. ``worker_l1_size`` moves the
allocator's L1 base up, which makes the ring that much bigger. Only dispatch changes; every program and its output
are the same.

    dev = open_device(trace_region_size=512 << 20)   # ttnn.open_device(device_id=0, ...) + RING_KB more ring
"""

import ttnn

RING_KB = 64  # extra ring (L1 taken from the allocator on every worker core); 0 = the default ring
L1_SIZE = 1536 * 1024  # Blackhole worker L1
DEFAULT_L1_BASE = 112576  # Blackhole: the allocator's L1 base with the default 69 KB ring


def worker_l1_size(ring_kb=RING_KB):
    """open_device's worker_l1_size for ``ring_kb`` extra ring (0 = the default)."""
    return L1_SIZE - DEFAULT_L1_BASE - (ring_kb << 10) if ring_kb else 0


def open_device(device_id=0, ring_kb=RING_KB, **kw):
    dev = ttnn.open_device(device_id=device_id, worker_l1_size=worker_l1_size(ring_kb), **kw)
    if ring_kb:
        base = ttnn._ttnn.device.GetAllocatorBaseAddress(dev, ttnn.BufferType.L1)
        assert base == DEFAULT_L1_BASE + (ring_kb << 10), f"L1 base {base}: not the Blackhole layout this assumes"
    return dev
