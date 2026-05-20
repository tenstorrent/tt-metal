# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Integration test for ttnn.dram_prefetcher in DRAM-core mode (DRAM-sender global_cb).

Exercises the new DramPrefetcherDramCoreProgramFactory path:
- Op accepts a DRAM-sender global_cb without raising.
- The factory builds the program and the DRISC kernel JIT-compiles successfully.

End-to-end data flow with a matmul receiver is out of scope for this prototype:
the DRISC kernel's `remote_cb_sender_barrier` waits for the receiver to ack pages,
and there's no receiver kernel running here (the matmul-side attachment is the
caller's responsibility). To verify the data flow end-to-end use the C++ standalone
test `test_dram_core_prefetch` which adds both kernels to a single Program.
"""

import os
import pytest
import torch
import ttnn


def _is_blackhole(device) -> bool:
    arch = getattr(device, "arch", lambda: None)()
    if arch is None:
        return True
    return "BLACKHOLE" in str(arch).upper()


def _dram_programmable_enabled() -> bool:
    return os.environ.get("TT_METAL_ENABLE_BLACKHOLE_DRAM_PROGRAMMABLE_CORES", "0") == "1"


def test_dram_prefetcher_dram_core_factory_builds(device):
    if not _is_blackhole(device):
        pytest.skip("DRAM-core mode requires Blackhole")
    if not _dram_programmable_enabled():
        pytest.skip("TT_METAL_ENABLE_BLACKHOLE_DRAM_PROGRAMMABLE_CORES not set")

    # One DRAM-sender bank pushing one tile worth of data to one worker receiver.
    bank_id = 0
    receiver_core = ttnn.CoreCoord(0, 0)
    receivers = ttnn.CoreRangeSet({ttnn.CoreRange(receiver_core, receiver_core)})

    K, N = 32, 32
    pt = torch.randn(K, N)
    dram_core_range_set = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(dram_core_range_set, [K, N], ttnn.ShardOrientation.ROW_MAJOR),
    )
    data_tensor = ttnn.as_tensor(
        pt,
        device=device,
        dtype=ttnn.bfloat16,
        memory_config=mem_config,
        layout=ttnn.TILE_LAYOUT,
    )

    # The op contract still expects a trailing tensor_addrs tensor; the DRAM-core path ignores it.
    addrs_dummy = torch.zeros(1, 1)
    addrs_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
            [1, 1],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    addrs = ttnn.as_tensor(addrs_dummy, device=device, dtype=ttnn.uint32, memory_config=addrs_mem_config)

    gcb = ttnn.create_global_circular_buffer_with_dram_senders(device, [(bank_id, receivers)], size=4096)

    # The op accepts the arguments and the factory builds a Program / JIT-compiles the DRISC
    # kernel. In slow dispatch (required for DRAM-core kernels) the launch synchronously waits
    # for the kernel to complete — and without a receiver kernel to ack the pushed pages, the
    # DRISC sender hangs in `remote_cb_sender_barrier`. We expect (and catch) the device
    # operation timeout that fires from TT_METAL_OPERATION_TIMEOUT_SECONDS. Any other host-side
    # raise indicates a real bug in the new factory path.
    raised_message = ""
    try:
        ttnn.dram_prefetcher(
            [data_tensor, addrs],
            num_layers=1,
            global_cb=gcb,
        )
    except RuntimeError as e:
        raised_message = str(e)

    if "Timeout" not in raised_message and raised_message != "":
        pytest.fail(f"dram_prefetcher(DRAM-sender global_cb) raised non-timeout error: {raised_message}")
