# Weighted norm tail: direct interleaved input reader

Source-only candidate. No device exactness, latency benefit, or promotion is claimed. Existing frozen source/evidence and production are unchanged.

## Current contract and measured cost

Frozen44f87 `v6_transport_conv_norm_decay_lookup_integration.py.snapshot` has recurrent coreFP32 TILE `[1,1,32,128]`. The decoder first casts core toBF16, reshapes to `[1,32,128]`, widens toFP32 interleavedL1, squares inFP32 interleavedL1, and computes the **native** last-axis sum into FP32 TILE `[1,32,1]`, padded `[1,32,32]`, interleavedL1. These producers are explicit source requests, not assumptions. `projection_z` is the slice of ordered QZ output; the valid decode profile confirms BF16 interleavedL1 at the compact consumer. The test therefore uses L1 Z for complete-adapter measurements; older DRAM-only microbenchmark input placement is not assumed to be the active decoder layout.

`WeightedNormTail.__call__` currently copies core into a core0 HEIGHT shard `[32,128]` and sums into a core0 HEIGHT shard `[32,32]`. The consumer binds those resident tensor buffers directly as CB0/1. Profile `v6_profile_c71_single_prefill_s64/decode_op_sequence.csv` steps123/124 measured0.583/0.323µs (first measured replay), total0.906µs kernel time. Step125 weighted tail is9.336µs and step126 compact lookup38.190µs. These kernel times neither equal wall latency nor predict the eventual benefit of removing two dispatches.

## Minimal change

A new BRISC reader uses existing `TensorAccessorArgs`/`TensorAccessor`, `Noc.async_read`, and `DataflowBuffer` APIs (`tt_metal/hw/inc/api/dataflow/noc.h`, `dataflow_buffer.h`, and `api/tensor/noc_traits.h`). It reads four physical FP32 core tiles and one sum tile, raw bytes, into exactly sized local CB0/1. It reserves4/1pages, issues all reads, waits for the NOC barrier, and publishes4/1pages. No tile transpose, scalar arithmetic, data conversion, or logical padding interpretation is added.

The compute source differs only by removing the two resident-input `reserve_back/push_back` lines, since the reader now owns publication. Existing `wait_front(4/1)`, direct-FP32 unpack, tile ordering, native rsqrt, explicit multiply/add stores, normalized and weighted BF16 rounding, gamma buffer, FP32 output pack, and input pops remain byte-identical. Source proof reinserting those lines restores the entire original compute string exactly. Gamma stays the existing shared core0 shard, and output/debug buffers stay the original shared shards; no writer kernel is necessary. The compactSiLU source/table/rounding/directWIDTH output are entirely unchanged.

The initial native square and sum are deliberately retained with the identical input placement/program selection. Simply requesting sharded producer outputs could choose a different reduction schedule or add conversion internally; the direct reader is a smaller exact boundary to test first. No producer-placement performance conclusion is inferred.

Input scratch is20,480bytes/core, replacing exactly20,480bytes of transient sharded input tensors. It adds no persistent device allocation, table, state, or KV-cache bytes. The compute footprint and compact table29,952bytes/core remain unchanged. Reader instruction/runtime metadata exists; final integrated capacity must still be verified, rather than inferred from operator count. This component covers the existing single-token/cache-boundary gated-tail shape; no multirow prefill algorithm is altered by this preparation.

## Frozen probe and gates

`tests/v6_weighted_norm_interleaved_probe.py` SHA256 `f1dc330d9f59e15dac3fc95ef795eb625a76526062bbc6d1648337dc5dbc2283`. Pinned autoflake2.3.1/isort5.13.2/Black23.10.1 were applied before source freeze. `v6_weighted_norm_interleaved_cpu_setup.json` passed without importingTTNN/openinghardware; it records pinned44 base, exact compute inverse proof, fixture provenance, payload hash, and adjacent reader/compute/executed-source snapshots. The first CPU-only invocation caught a missing fixture-helper metadata argument before creating evidence; that argument was added, then formatting/setup rerun successfully. No device failure or measured-source rewrite occurred.

Fixtures (16): position128, position129, position130, position137, position181, position3271, random0, random1, random2, random3, random4, random5, random6, random7, zero_1.0, zero_-1.0. Actual captured cores retain their originalFP32 values before BF16 rounding; random cases span magnitudes and explicit positive/negative zero complete-adapter controls are added. Two additional direct-tail L1/DRAM transport control tiles cover finite subnormals, signed zeros and BF16 midpoints. Debug checks compare exact normalizedFP32, inverse, weightedFP32 carrier and finalweighted output; completeadapter compares exact BF16 WIDTH output against the frozen44 weighted-tail+compactlookup baseline. No tolerance or gate is relaxed.

Optional completeadapter timings include coreBF16 boundary/reshape/FP32 widening, native square/sum, weightedtail, compactlookup, every runtime placement and directWIDTH output. Setup, upload/readback, and following output projection are excluded equally. Candidate is timed first, then baseline, with3warmups,5 eager samples and5×20 trace replays; every replaybatch output is checked exactly. Profiling/watcher/allocation tracking remain disabled for timing. A timing regression rejects this path regardless of reduced dispatch count.

```bash
env -u TT_VISIBLE_DEVICES -u TT_METAL_WATCHER -u TT_METAL_DEVICE_PROFILER \
  -u TT_METAL_TRACE_ALLOC_TRACKING -u TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE PYTHONPATH=. \
  python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/tests/v6_weighted_norm_interleaved_probe.py \
  --run-device --timing --samples 5 --replays 20 \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_weighted_norm_interleaved_device.json \
  > models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_weighted_norm_interleaved_device.log 2>&1
```

Parent owns serialized hardware access. `TT_VISIBLE_DEVICES` must be unset and physicalchip0 is explicit. Use fresh output paths. Passing this component would authorize a separate static integration with whole-decoder correctness/performance/capacity qualification; it is not itself acceptance.
