# Greedy sampler profile and decision

## Common-path comparison

The required TP4 local vocabulary width is 65,536. Common `Sampling1D` TopK measured approximately 10.625 ms, while the reduced model trace measured approximately 5.15 ms, so the sampler dominated token-out. Partitioned common candidates improved shape coverage but failed semantic greedy tie-breaking on real BF16 softcapped Gemma logits: equal maxima at global tokens 177 and 192 returned 192. Common force-argmax also failed exact shard-boundary IDs. `SamplingGenerator` was rejected because it rounds batch one to batch 32 and its mutable/internal trace ownership does not match the fixed-slot generator.

Native 32/64-width TILE candidate all-gather asserted in `minimal_default_writer`. Broadcast-backed BF16 corrupted candidate values, and the row-major composite hung in trace replay. FP32 broadcast was exact but did not resolve common TopK latency and tie semantics. Experimental changes were reverted.

## Selected custom path

`Gemma4GreedyTP4Sampler` uses an eight-core local BF16 tile maxloc per device, a tiny two-link Linear TP all-gather of `(score, global_token)` pairs, and a final exact pair reducer. Equal scores choose the lower global token. The result is written to the persistent token tensor consumed by the next model trace replay.

Boundary evidence: `[0,32767,32768,65535,65536,262143]`; equal-score tie `177` versus `192` resolves to `177`; batch two and three trace replays pass. Watcher drove fixes for aligned DRAM output and the physical 16-byte pair-page stride.

## Final source-current performance

Reduced workload: real optimized layers 0 and 5, prompt 149, six generated tokens, batch one. This capture is after the sampler alignment, physical-stride, persistent-gather-output, mixed-prefill, and teardown fixes. Setup profiler traffic was flushed before four steady replays; no profiler markers were dropped.

```text
local winner median           299.464 us (299.342-299.586)
final reducer median            0.4205 us (0.418-0.424)
sampler share/device ops        9.68%
steady end-to-end               3.484 ms/token
sampler share/steady e2e         ~8.6%
steady decode                 287.018 t/s/u
token host refreshes          0
full-logit readbacks          0
sampled-token readbacks       1 (final only)
```

`tt-perf-report` attributes 56.25% of reduced device time to the local vocab-sharded LM head and 9.68% to the custom sampler local winner, so canonical sampling is not the dominant operation. The earlier 17:38 profile predates alignment/stride fixes and is retained only as historical evidence.

Final source CSV SHA-256: `cefa4861ae9713bc1d83c117dab38760939997a0a305ee71a03483f1c7d528a3`. Compact counters: `reduced_token_out_final_perf.json`. Official filtered/summary output and advice disposition: `perf/`. The multi-gigabyte raw profiler trees remain local-only and are not staged.
