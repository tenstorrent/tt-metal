# SPDX-License-Identifier: Apache-2.0
"""Host-only validation of a fixed pool against launch allocator accounting."""

import math


def validate_pool_memory(*, allocator, device_ids, cache_specs, batch, context):
    banks = allocator["num_banks"]
    free = banks * allocator["total_bytes_free_per_bank"]
    contiguous = banks * allocator["largest_contiguous_bytes_free_per_bank"]
    # Ten shared buffer pairs in the full model; count each storage ID once.
    storage = {}
    for storage_id, shape, tile_bytes in cache_specs:
        blocks, heads, page, dim = shape
        tiles = blocks * heads * math.ceil(page / 32) * math.ceil(dim / 32)
        size = 2 * math.ceil(tiles / banks) * banks * tile_bytes
        if storage_id in storage and storage[storage_id] != size:
            raise ValueError("Shared KV storage has inconsistent physical sizes")
        storage[storage_id] = size
    pool = sum(storage.values())
    # Existing ledger: prepared state peak at B32 and analytical transient
    # allowance from optimized_vllm/memory_comparison.json. Trace region is
    # already excluded from total allocator bytes and is not subtracted again.
    page_maps = len(cache_specs) * batch * math.ceil(context / 128) * 4 * 4
    reserve = page_maps + 147472896 + 805306368
    fits = pool + reserve <= min(free, contiguous)
    return {
        "pool_sizing": "fixed",
        "pool_sizing_reason": "Stock plugin fixes its pool before loading weights; this launch validates its existing pool against measured shared-mesh allocator free space, not adaptive sizing.",
        "measurement_basis": "One shared mesh allocator measurement applied uniformly to participating devices; these are not independent physical allocator samples. KV uses replicated ttnn.zeros on the mesh with equal shape/dtype on every device.",
        "per_device": [
            {
                "device_id": int(i),
                "free_bytes": free,
                "contiguous_free_bytes": contiguous,
                "pool_bytes": pool,
                "reserve_bytes": reserve,
            }
            for i in device_ids
        ],
        "page_map_bytes": page_maps,
        "prepared_reserve_bytes": 147472896,
        "transient_reserve_bytes": 805306368,
        "reserve_evidence": "doc/agentic_serving.json ledger; doc/optimized_vllm/memory_comparison.json; transient reserve is analytical, not a full-stack measured peak",
        "trace_reserve_already_excluded": True,
        "fits": fits,
        "remaining_bytes_after_reserves": min(free, contiguous) - pool - reserve,
    }
