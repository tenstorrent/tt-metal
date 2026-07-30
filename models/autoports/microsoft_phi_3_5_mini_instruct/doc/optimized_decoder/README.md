# Phi-3.5 Mini optimized decoder

This stage adds an independently dispatched `OptimizedDecoder`; neither prefill nor decode calls the functional implementation at runtime. It preserves packed QKV, adopts measured split gate/up, and preserves paged KV-cache semantics, LongRoPE selection, trace replay, non-aligned logical sequence lengths, and the 131,072-token context contract.

## Final policy

- Blackhole, one device, dense layer kind (the checkpoint has one meaningful decoder-layer topology).
- Decode weights: BFP4/LoFi, DRAM width-sharded; `per_core_M=1` programs partition outputs over a 16-core activation grid with block widths QKV/O/gate/up/down = 6/6/6/6/16. Most final profiler projection rows record 80 active worker cores; tail rows report 79, 74, or 69. “8/16/32” below refers to the output partition and activation/residual grid, not this reported internal worker count.
- Decode residual/norm/MLP activations: L1 width-sharded. SDPA uses an explicit 8x8 program.
- KV cache: paged BFP8, page size 32. Prefill casts K/V once before paged fill; decode updates consume BF16 K/V into the BFP8 cache.
- Prefill: phase-specific interleaved BFP4 weights. Logical lengths are internally padded/chunked; there is no public alignment restriction.

## Operation-topology audit

| Area | Current topology | Candidate | Action | Evidence |
|---|---|---|---|---|
| Attention projections | one packed same-input QKV matmul | three BFP4 split Q/K/V matmuls plus adapted DRAM concat | keep packed | split is correct at PCC 0.999996 but costs 0.555/0.719 ms versus 0.481/0.658 ms |
| MLP input projections | one packed gate/up matmul | two BFP4 gate/up matmuls | adopt split | split repeats at 0.481/0.658 ms versus packed 0.487/0.662 ms |
| Decode weights | interleaved BF16 | DRAM-sharded BFP8/BFP4 across 8/16/32 cores | choose BFP4, 16 cores | candidate table below; 16 wins both required batches |
| Residual/norm/MLP chain | DRAM/interleaved boundaries | L1 width-sharded chain | adopted | layer norms are about 7 us in the final profile |
| Attention | primitive decode attention/default config | paged SDPA composite, explicit 8x8 | adopted | explicit is faster at batch 1 and 32 |
| KV cache | paged BF16 | paged BFP8 | adopted | preserves PCC and improves batch 32 by about 45 us |
| Prefill matmuls | implicit BF16 configs | BFP4/LoFi and explicit large 2D programs | BFP4 adopted; explicit programs rejected | final phase-interleaved 1.824 ms versus 1.762 ms functional; the earlier explicit candidate was 1.942 ms and did not recover the regression |
| Head/RoPE boundaries | TTNN head transforms and RoPE layout conversions | remove conversions | rejected as required | head width is 96/48 and RoPE slice width 48 is not tile-shard compatible; profile contains no host fallback |
| DRAM-matmul/norm boundary | round-robin DRAM-matmul output versus rectangular norm grid | common shard | retain one device reshard | DRAM-sharded matmul factory computes round-robin output while sharded RMSNorm requires its rectangular grid; final mean reshard is 1.55 us at b1 and 1.33 us at b32 |

There are no runtime `torch`, `from_torch`, `to_torch`, collectives, or host fallbacks in the measured methods. The recurring conversion map is: input interleaved→rectangular width-shard for RMSNorm; round-robin matmul output→rectangular width-shard at the norm/residual boundary; Q/K sharded→interleaved for 48-wide RoPE slices and back to head layout; SDPA output→concat-head input sharding; and head transforms' tile padding/unpadding. Each is a device operation at an incompatible supported-op boundary.

## Correctness

Acceptance is the functional stage bar, PCC >= 0.995.

| Path | PCC / result |
|---|---:|
| synthetic prefill, seq 33 | 0.999993 |
| synthetic prefill, seq 65 | 0.999994 |
| real prefill, seq 33 | 0.999924 |
| real decode after paged prefill | 0.999946 |
| real traced decode, batch 1 | 0.999996 |
| real traced decode, batch 32 | 0.999993 |
| LongRoPE traced decode, position 4096 | 0.999996 |
| nonzero prefill, seq 32769, last token | 0.996934 |
| real decode, logical context 131072 | 0.999939 |
| zero prefill, seq 131071 and 131072 | exact zero, correct shapes |

Both batch sizes replay the captured trace ten times with bitwise-identical outputs. The real prefill-to-decode test proves that BFP8 paged-cache contents are consumed correctly.

## Performance

Same process shape, real layer-0 weights, warmed seq-128 prefill, and 20 traced decode replays:

| Path | Functional | Optimized | Delta |
|---|---:|---:|---:|
| prefill b1, warmed | 1.762 ms | 1.824 ms | +3.5% |
| prefill b32, warmed | 37.714 ms | 30.658 ms | -18.7% |
| decode b1, traced mean | 1.050 ms | 0.481 ms | -54.1% |
| decode b32, traced mean | 1.269 ms | 0.658 ms | -48.2% |

The primary batch-1 decode target beats the best correct same-harness baseline and batch-32 prefill/decode improve materially. Batch-1 prefill has a 3.5% material regression; its host-dispatched head/RoPE sequence dominates despite faster BFP4 matmuls. Explicit large prefill programs did not recover it and were rejected.

Selected sweep results (mean traced milliseconds):

| Candidate | b1 | b32 | Result |
|---|---:|---:|---|
| BFP4/LoFi, 8 cores | 0.490 | 0.672 | reject |
| BFP4/LoFi, 16-core output grid, packed gate/up | 0.487 | 0.662 | topology control |
| BFP4/LoFi, 16-core output grid, split gate/up | 0.481 | 0.658 | choose |
| BFP4/LoFi, 32 cores | 0.509 | 0.674 | reject |
| BFP8 attention/down, BFP4 gate/up, 16 cores | 0.508 | 0.685 | reject |
| BFP4 attention, BFP8 down, 16 cores | 0.496 | 0.671 | reject |
| BFP8 policy, HiFi2 | 0.665 | 0.840 | reject |
| default SDPA | 0.489 | 0.699 | reject |
| explicit 8x8 SDPA | 0.486 | 0.661 | choose |
| BF16 KV cache | 0.485 | 0.706 | reject; batch-32 regression |
| split Q/K/V, adapted DRAM concat | 0.555 | 0.719 | reject; PCC 0.999996 |

All practical output-grid candidates were exercised. QKV/O/gate/up have 96 K tiles and down has 256; 8, 16, and 32 permit common role programs. The 8-core BFP8 trial first exceeded L1, then passed after crossing with final BFP4—so it was rejected on performance, not on a first API error. Batch 1 and 32 were measured independently even though batch 1 tile-pads to physical M=32 and therefore also satisfies `per_core_M=1`.

Precision-locked per-role program search (whole-layer traced mean, milliseconds):

| Changed role | Block width | b1 | b32 | Result |
|---|---:|---:|---:|---|
| none (QKV/O/gate/up/down) | 6/6/6/6/16 | 0.481 | 0.658 | choose |
| QKV | 3 | 0.491 | 0.667 | reject |
| O | 3 | 0.492 | 0.666 | reject |
| split gate/up | 3 | 0.490 | 0.666 | reject |
| down | 8 | 0.490 | 0.665 | reject |

The selected values are the largest legal K-tile divisors for the 16-way output partition; the table covers the next smaller divisor independently for every role. The DRAM-sharded program family does not expose output subblock controls (`tt-perf-report`: “No output subblock size found”). Interleaved weights are the functional control, and BFP8/BFP4 plus 8/16/32 cover the other legal memory/precision geometry families.

## Profiler conclusion and roofline

Final Tracy CSV: `tracy/final/ops_perf_results.csv.gz` (gzip-compressed). Its signposts cover prefill and decode at both batch 1 and 32. The profile verifies BFP4/LoFi matmuls and no host fallback. Most projection rows report 80 active worker cores, with partial tail rows at 79/74/69. Steady decode means are QKV 55.5 us, O 22.3 us, gate 49.6 us, up 49.6 us, and down 47.6 us. `tt-perf-report` flags the projections as bandwidth-bound and suggests higher fidelity for accuracy; measured HiFi2 was slower and final PCC exceeds the bar. It finds no valid output subblock recommendation for this DRAM-sharded family.

One layer reads about 56.6 MB of BFP4 weights plus about 0.8 MB of BFP8 K/V at context 128. At the report's 512 GB/s Blackhole bandwidth reference, the bandwidth-only lower bound is about 0.112 ms. The traced end-to-end 0.481 ms includes SDPA, norms, transforms, and trace synchronization; the five steady projection kernels total about 0.225 ms.

## Commands

```bash
pytest -q -s models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_optimized_decoder.py
pytest -q -s models/autoports/microsoft_phi_3_5_mini_instruct/tests/optimized_decoder_perf.py
PHI_DECODER_IMPL=functional pytest -q -s models/autoports/microsoft_phi_3_5_mini_instruct/tests/optimized_decoder_perf.py
python -m tracy -r -p -v -m pytest -q -s models/autoports/microsoft_phi_3_5_mini_instruct/tests/optimized_decoder_perf.py
TT_METAL_WATCHER=10 pytest -q -s models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_optimized_decoder.py
```

This stage is decoder-only. Multichip, full-model, LM-head, MoE, CCL, and vLLM work are intentionally out of scope.
