# Batched complete-prefix native qualification: passed

On October 11, 2026, the sealed `prefix-qualification-20261011-v1` bundle passed
all three native hardware gates on TP4 physical IDs `[0, 4, 12, 8]`:

| Gate | Result |
| --- | --- |
| Packed physical transfer | Exact bytes on all ranks; neighbours unchanged; corrupt restore unpublished |
| Four-layer continuation | Exact suffix logits, 32 teacher-forced traced decode steps, active-frontier restore |
| Full64 continuation | Exact suffix logits and 32 traced steps; active-frontier restore preserves trace, tensor addresses and neighbouring recurrent state |

The full-model case uses 4096 prefix tokens, 32 suffix tokens and a checkpoint
at the actual consumed frontier 4160 after decoding. It retains BFP8 weights/KV,
BF16 activations/conv and FP32 recurrent state. Its 282.8125 MiB complete-state
checkpoint includes all attention, GDN and convolution state.

| Full64 operation | Earlier serial | Batched | Change |
| --- | ---: | ---: | ---: |
| Capture | 2.20667 s | 1.25948 s | 42.9% shorter |
| Restore | 3.54193 s | 1.84508 s | 47.9% shorter |
| Recompute 4K prefix | 1.18545 s | 1.18886 s | Separate measured prefill baselines |

**Restore still costs 55.2% more than recomputing this 4K prefix.** This is not
an end-to-end serving speedup. Both native runs deliberately randomize physical
page order: the batched run still makes 16,896 individual window transfers.
It uses 770 capture and 1538 restore fences, with at most 1,044,480 bytes of
host staging. Restore also reads destination bytes before writing them.
The 4-layer case took 75.4 ms capture, 111.4 ms restore and 561.2 ms prefill;
its ratio cannot be extrapolated to the full model.

Fresh serving allocations may allow larger contiguous windows. That could
reduce individual submissions substantially, but actual HTTP/cache speedup
remains unmeasured. The random-page case is an isolation/fragmentation stress
test, not a claim about the serving allocator's typical page layout. These file
reads can hit the OS page cache; this is not a physical SSD bandwidth benchmark.
The exact same precision policy is present in both serial and batched receipts.

The persistent unit was:
`qwen38-prefix-native-prefix-qualification-20261011-v1.service`.
Its recorded invocation is `94fd4d00f8f54c998d39c5d416a93244`, initial PID 998629.
Final status: MainPID 0, Result success, ExecMainStatus 0, inactive/dead; the native
queue records completed/passed/cleanup_completed=true and the dirty marker was
absent. Full64's device test took 333.57 s, dominated by its cached-weight loading.
No recovery/reset was needed.

`native-terminal.json` records a fresh source/seal/receipt validation after
completion. Every stage receipt is bound by hash in `native/queue.json`.
`collection.json` binds the collected source bytes and compressed log bytes;
original logs and checkpoint payloads remain in the host task directory.
Admission evidence first verified the parent profiler's successful terminal
status and its final clean queue before this unit was started.

**HTTP was not launched.** New user priority returned the Galaxy to profiling
larger matmul/SDPA costs after this native run completed. The prepared HTTP
recipe remains gated and unstarted; serving remaps, preemption, restart, actual
warm TTFT and agent throughput are still unqualified. No feature default or
release merge, AgentX or GPQA was enabled. See the
[sealed preparation](../prefix-serving-preparation-v1/README.md) and
[operator recipe](../../experiments/PREFIX-QUALIFICATION-BUNDLE.md).
