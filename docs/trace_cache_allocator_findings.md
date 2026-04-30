# Trace Cache Allocator Findings

This note summarizes the allocator comparison using:

- `/home/jbauman/trace_alloc_9.json`
- `/home/jbauman/trace_alloc_300.json`

The comparison used `scripts/compare_trace_allocators.py` with the corrected trace-cache2 model:

- Nonbinary data is a unique per-trace-node allocation and is transferred every time.
- Binary data is a reusable per-program allocation.
- Binary and nonbinary allocations share the same worker buffer, so either can evict older data from the other category after an appropriate stall.
- Badness is `sum(1 / 2^stall_distance)`, where `stall_distance = trace_idx - stall_idx - 1`.
- Stalls at least `launch_msg_buffer_num_entries - 1` trace nodes back contribute zero badness.

## No-Fail Behavior

The trace-cache2 model now has a no-fail fallback:

1. Try the normal eviction heuristic.
2. If that fails, search for a contiguous span that can satisfy the allocation while excluding allocations from the current trace node.
3. If no such span exists, stall on the newest older allocation, reset allocator state, and restart allocation for the current trace node.

After this change, reuse-window variants `1..16` completed on both captures without allocation failure.

## Results

### `trace_alloc_9.json`

| Allocator | Variant | Badness | Stalls | Dist0 Stalls | Transfer |
| --- | --- | ---: | ---: | ---: | ---: |
| SimpleTraceAllocator | replay match | 125.844 | 1377 | 21 | 21.93 MiB |
| trace-cache2 | rw=2 | 256.094 | 1284 | 1 | 18.82 MiB |
| trace-cache2 | rw=3 | 167.594 | 1659 | 0 | 20.94 MiB |
| trace-cache2 | rw=6 | 133.688 | 1719 | 5 | 21.92 MiB |

Best trace-cache2 badness on this capture was `rw=6`, but it remained worse than Simple by `+7.844` badness while transferring approximately the same amount of data. It used `4209` normal evictions, `0` fallback span evictions, and `2` reset fallbacks.

### `trace_alloc_300.json`

The Simple replay did not match captured addresses for this file, so Simple badness used captured stall metadata while transfer bytes came from the replay model.

| Allocator | Variant | Badness | Stalls | Dist0 Stalls | Transfer |
| --- | --- | ---: | ---: | ---: | ---: |
| SimpleTraceAllocator | captured stalls | 50.094 | 809 | 2 | 11.81 MiB |
| trace-cache2 | rw=2 | 243.125 | 886 | 0 | 11.73 MiB |
| trace-cache2 | rw=6 | 62.125 | 671 | 36 | 13.68 MiB |
| trace-cache2 | rw=11 | 52.906 | 621 | 9 | 13.37 MiB |
| trace-cache2 | rw=12 | 52.906 | 587 | 9 | 13.21 MiB |

Best trace-cache2 badness on this capture was `rw=11` or `rw=12`, but both remained worse than Simple and transferred more data. The `rw=6` run used `2262` normal evictions, `0` fallback span evictions, and `0` reset fallbacks.

## Combined Comparison

Across both captures:

| Allocator | Variant | Total Badness | Transfer |
| --- | --- | ---: | ---: |
| SimpleTraceAllocator | baseline | 175.938 | 33.74 MiB |
| trace-cache2 | rw=2 | 499.219 | 30.55 MiB |
| trace-cache2 | rw=6 | 195.812 | 35.60 MiB |
| trace-cache2 | rw=12 | 225.594 | 35.13 MiB |

`rw=6` was the best combined trace-cache2 variant by badness, but it was still worse than Simple by `+19.875` badness and transferred about `+1.86 MiB` more data.

## Conclusion

With the corrected binary/nonbinary model and no-fail fallback, trace-cache2 no longer has hard allocation failures in these captures. However, the tested reuse-window variants do not outperform SimpleTraceAllocator on the requested badness metric. The best combined variant, `rw=6`, is close on `trace_alloc_9.json` but still loses overall and transfers more total data.
