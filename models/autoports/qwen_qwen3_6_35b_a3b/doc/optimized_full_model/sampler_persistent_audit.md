# Persistent split-sampler CCL experiment

Status: **hardware-rejected, not promoted**. The watched raw gather isolates an unflushed one-tile scatter tail in native broadcast; see [exact rejection, repro and narrow unapplied repair proposal](sampler_persistent_broadcast_rejection.md). The analysis below records the original ownership/barrier proposal; the factory's data-transfer tail defect prevents using it unchanged. Production generator/shared sampling/native CCL are unchanged. Decoder persistent scratch remains enabled independently. The measured selected sampler is approximately 0.488 ms; that alone is not grounds to omit this optimization.

The exact restriction is a scatter worker whose positive output-page count is
not divisible by `packet_size / output_page_size`. Here each rank contributes one
2048-byte BF16 tile to a 4096-byte/two-page packet. `FabricScatterWriter::send`
only emits full packets and has no tail flush before source-CB release. The
[watched v3 raw-gather repro](persistent_sampler_l1_watched_v3.diagnostics.json)
records 3072/4096 mismatches per rank before sampling or trace capture: local32
columns are correct, remote96 remain poison. The
[v3 log](logs/persistent_sampler_l1_watched_v3.log) preserves the failing gate.

The unapplied narrow C++ repair would set the broadcast writer's packet size to
2048 only when `num_output_pages == 1 && output_page_size == 2048`, selecting its
existing unicast implementation; all other geometry would remain unchanged.
There is no per-op packet-size API. A global 2048-byte fabric payload conflicts
with the selected FP32 collective pages. Width64 padding at one link is a legal
but untried workaround that doubles payload and requires four pad chunks removed
from each gathered tensor. Neither workaround nor the proposed native repair is
claimed enabled or faster. The linked rejection note gives exact source anchors,
commands, costs, and requalification gates; [anomaly_resolution.md](anomaly_resolution.md)
tracks the failure sequence separately from selected-path correctness.

## Ownership and exact selected shapes

`SamplerCCL.line_all_gather` currently ignores `buffer_key`. The selected split path computes local top-32 on each of four vocabulary shards. Values are BF16 TILE `[1,1,32,32]`, indices UINT16 TILE with the same shape; each gather produces interleaved DRAM `[1,1,32,128]`. Two owned outputs therefore occupy **16,384 bytes per rank**, excluding CCL semaphore/workload storage. No full-vocabulary gather is introduced.

In `models/common/sampling/tt_sampling.py:TTSampling.forward`, gathered BF16 values alias `topk_values_gathered_bf16_interleaved` for this DRAM sampling contract. The final unconditional `ttnn.deallocate(topk_values_gathered_bf16_interleaved)` force-frees that allocation even if another object retains it. Keeping a Python reference alone is insufficient. Gathered indices are widened to a fresh INT32 tensor on the selected route; the original gathered UINT16 tensor has no explicit forced deallocation there. The tie-break helper returns a fresh add result, so its final deallocation is safe.

Minimal production hook proposed after qualification: optional CCL ownership/release method at the gathered-values release site, preserving ordinary deallocation for CCLs without this hook. A model CCL owns buffers through every live trace and releases them only after traces retire. Match allocation address plus dtype/memory configuration, not Python object identity. Restrict this first implementation to the proven split DRAM route; do not silently extend ownership to force-argmax, full-row, or alternate memory-layout branches.

The isolated `tt/sampler_persistent_probe.py` binds a checked private copy of `TTSampling.forward` to **one candidate instance**, changing only that release site. It does not modify the shared class or globally patch `ttnn.deallocate`. SeedManager retains its original sampler object. The private source and shared-source hashes are recorded. Production integration should use an explicit optional hook, not this experimental source-copy mechanism.

## Reuse ordering and rank skew

The ordinary minimal-default all-gather writer receives `barrier_semaphore.has_value() && !using_persistent_buffers` from `all_gather_async_default_program_factory.cpp`. Supplying an output buffer disables its startup barrier. Same-command-queue ordering protects local reuse, but does not prevent an early rank from remotely overwriting a persistent gather output while a slower rank still consumes the prior output in tie-break/sampling. Merely alternating two buffers bounds the reuse interval; it is not an arbitrary-skew proof.

The current minimal-default reader atomically subtracts the invocation's ready credits instead of clearing all credits. Thus the inherited shared-sampler warning about clearing ready credits describes an older failure mechanism; the missing cross-rank reuse fence remains a separate concern. This experiment does not rely on removing that warning.

Existing `use_broadcast=True` selects `AllGatherViaBroadcastFactory`. Its writer performs an **unconditional init barrier before output writes**, including persistent-output calls, and a final ready barrier after writes. Both semaphores are private to the cached workload. For the exact TILE input, page extents are `[1,1,1,1]`; the native contiguous-output-slice requirement is satisfied for gather dimension 3.

The ordering argument is independent of balanced rank timing:

1. A rank enters the next values-gather init barrier only after its previous local sampler consumers finish on the same CQ.
2. No rank writes the reused output until all ranks enter that init barrier. Hence every rank's previous consumer precedes every rank's new output writes.
3. The final ready barrier guarantees all ranks previously passed/reset the init barrier before any can complete the workload and enter its next invocation. Conversely the next init barrier prevents next-generation ready credits from overtaking an unfinished prior final-barrier reset.

This relies on normal ordered single-CQ execution, the native fabric ordering/flush contracts, and distinct cached workloads for values and indices. Concurrent independent CQs sharing these buffers are unsupported. The hardware gate still must check the concrete descriptors, program cache, buffer liveness, and repeated replay; source reasoning alone is not a pass.

The broadcast factory creates its own semaphores and synchronizes devices on a **program-cache miss**. All datatype/link/shape configurations must therefore be warmed before any trace capture. The harness warms both BF16 and UINT16 collectives, all copies, and every per-rank token/seed read layout before its first capture. No program-cache tracking exemption is used. Candidate `--num-links 1` and `2` are distinct process runs; the baseline remains its selected two-link implementation.

## Gate and timing scope

`tests/run_stage7_sampler_persistent_probe.py` uses the real Qwen terminal to obtain one full-logit fixture plus three constructed fixtures. All 32 rows change across four ranks, including negative logits and cross-shard ties. It compares current split sampling with the persistent-broadcast candidate for greedy, top-k 8, top-k 32/top-p 0.75, and mixed per-row parameters. Both traces include the actual `_sampling_forward`, `tt_out_tok` output, active-row merge, and device seed advance.

Each acceptance burst queues 64 steps without per-token host synchronization/readback. Prior full-logit inputs are poisoned with NaNs, then replaced in full; candidate persistent values/indices are also poisoned before every invocation. Device snapshots preserve every step's tokens and seeds for **all four ranks**. Checks require exact baseline/candidate tokens and seeds, rank agreement, valid global top-k membership, greedy lowest-ID maxima, and an independent exact device-seed increment oracle. This is queued reuse stress, not an artificially injected rank-delay test; arbitrary-skew safety comes from the audited barrier protocol plus hardware validation.

Clean timing measures five bursts of 100 nonblocking full-sampler trace replays with boundary synchronization. Input updates, poisoning, and history snapshots are excluded from the timing region. These are terminal sampler numbers, not full-model decode throughput. The selected baseline and candidate are measured in the same process; preserve the ordinary final full-model timing gate after any promotion.

Run serially under the parent's device lock, from checkout root:

```bash
TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 PYTHONPATH=. python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/tests/run_stage7_sampler_persistent_probe.py --mode acceptance --num-links 1 --output models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_full_model/persistent_sampler_acceptance_l1.json
TT_METAL_WATCHER=1 TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 PYTHONPATH=. python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/tests/run_stage7_sampler_persistent_probe.py --mode acceptance --num-links 1 --output models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_full_model/persistent_sampler_watcher_l1.json
TT_METAL_WATCHER=0 TT_METAL_DEVICE_PROFILER=0 TTNN_OP_PROFILER=0 TT_METAL_PROFILER_TRACE_TRACKING=0 TT_METAL_TRACE_ALLOC_TRACKING=0 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 PYTHONPATH=. python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/tests/run_stage7_sampler_persistent_probe.py --mode timing --num-links 1 --output models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_full_model/persistent_sampler_timing_l1.json
```

Repeat acceptance/timing with `--num-links 2` and distinct `l2` output paths to discriminate link overhead without changing numerical policy. Keep watched and clean timing in separate processes. Existing production remains the fallback until all relevant gates pass; a failing broadcast experiment is evidence to repair, not authorization to enable unsafe default-AG persistent buffers.
