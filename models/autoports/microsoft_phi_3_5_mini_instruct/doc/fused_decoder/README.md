# Phi-3.5 Mini fused decoder

This stage provides the graph-fused, single-device layer in
`tt/fused_decoder.py`. It preserves the functional decoder's BF16 public
tensor contract, paged BF16 KV cache, 32-token pages, LongRoPE selection,
trace-safe on-device positions, non-aligned prefill lengths, and 131072-token
maximum context.

The model has one meaningful decoder-layer kind: a dense Phi3 layer with
32 query heads, 32 KV heads, 96-wide heads, packed QKV, and packed gate/up
weights. Layer 0 with official cached weights is covered in addition to
deterministic statistic-matched synthetic weights.

## Fused graph

The final graph applies three measured topology decisions:

1. `silu(gate) -> multiply(up)` becomes one Binary-NG multiply with
   `input_tensor_a_activations=[SILU]`.
2. At serving batch 32, prefill's generic head concatenation becomes the
   dedicated `experimental.nlp_concat_heads` kernel. Batch 1 retains generic
   concatenation because the dedicated launch was slower there.
3. At batch 1, the two paged K/V cache writes become
   `paged_fused_update_cache` on disjoint K/V L1 shard grids. At serving batch
   32, the required V reshard cost exceeded the write-fusion saving, so the
   faster two-write topology is retained.

The fused decoder subclasses the functional decoder to share setup, contract
validation, LongRoPE, and the graph regions that were already optimal. Its
prefill, decode, and SwiGLU methods are overridden. The static fused-path test
asserts those overrides and the exact dedicated ops, preventing a functional
fallback from satisfying the stage.

## Correctness

Acceptance is PCC >= 0.995. The final fused suite covers:

- prefill at logical lengths 31, 32, 33, 63, 64, and 65;
- batch-2 paged-cache routing with a permuted page table;
- non-aligned 131071 prefill and exact 131072 prefill;
- a nonzero 32769-token last-token oracle (PCC 0.999865);
- synthetic decode (PCC 0.999998);
- official real layer-0 prefill (PCC 0.999991) and decode (PCC 0.999995);
- real-weight decode at logical context 131072 (PCC 0.999988);
- short- and long-RoPE controls, including traced position 4096;
- three bitwise-identical steady trace replays and three eager controls at
  batch 1 and batch 32.

The final 26-test integrated run passed (24 fused functional/structural cases
plus two dedicated-RoPE candidate probes). The watcher-separated run passed the
real-weight test plus traced batch-1/batch-32 determinism tests with no watcher
error. Runtime forwards contain no Torch conversion or host fallback.

The transient disjoint K/V shard grids used by the batch-1 fused update do not
change cache allocation, dtype, page layout, or DRAM capacity. Therefore
`doc/context_contract.json` remains unchanged at 131072 with no capability
reduction.

## Before and after performance

Measurements are one Blackhole p300c device, sequence/context 128. Prefill is
warmed eager execution. Decode is warmed trace replay. The clean Tracy run
uses one prefill sample and five decode replays; the final non-profiler A/B
uses six alternating-order runs with 100 samples per implementation per run.

| Path | Batch | Functional host ms | Fused host ms | Functional device-kernel ms | Fused device-kernel ms | Ops |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prefill | 1 | 2.076 | 1.896 | 1.510 | 1.504 | 51 -> 50 |
| Prefill | 32 | 37.756 | 37.375 | 37.315 | 36.978 | 177 -> 176 |
| Traced decode, 5 replays | 1 | 1.099/replay | 1.061/replay | 1.020/replay | 0.993/replay | 58 -> 57/replay |
| Traced decode, 5 replays | 32 | 1.226/replay | 1.224/replay | 1.130/replay | 1.126/replay | 55 -> 54/replay |

The final non-profiler check alternated implementation order across six
independent runs and took 100 warmed samples per implementation in each run.
The table reports the mean of the six run means and their between-run sample
standard deviation:

| Path | Batch | Functional mean ± SD | Fused mean ± SD | Paired wins |
| --- | ---: | ---: | ---: | ---: |
| Prefill | 1 | 1.590 ± 0.017 ms | 1.580 ± 0.009 ms | 5/6 |
| Prefill | 32 | 37.675 ± 0.015 ms | 37.328 ± 0.026 ms | 6/6 |
| Traced decode | 1 | 1.0503 ± 0.0003 ms | 1.0126 ± 0.0004 ms | 6/6 |
| Traced decode | 32 | 1.2161 ± 0.0013 ms | 1.2134 ± 0.0014 ms | 5/6 |

The paired mean deltas (functional minus fused) were +0.0102, +0.3467,
+0.0377, and +0.0027 ms respectively. The raw recoverable artifact is
`ab_alternating_final_6x100.log`; all six eight-case runs passed.

The final clean profiler CSV is `tracy/ops_final.csv`. Signpost-delimited
`tt-perf-report` tables and CSVs are:

- `tracy/{functional,fused}_prefill_b{1,32}_final.{txt,csv}`
- `tracy/{functional,fused}_decode_b{1,32}_final.{txt,csv}`

`tracy/profile_console_final_v4.log` is the clean five-decode-replay
collection and has zero profiler-DRAM-buffer-loss warnings. Duration-histogram
`overflow(...)` buckets are range statistics, not lost profiler markers.
Earlier failed/rejected collections are retained and classified in
`work_log.md`.

## Graph-fusion audit

### Measured operation sequence

| Region | Functional sequence | Movement | Final decision |
| --- | --- | --- | --- |
| Input | RMSNorm -> packed QKV linear -> split QKV/heads | DRAM throughout | Already dedicated/packed; retain |
| RoPE | slice halves -> neg/concat -> two multiply + add, for Q and K | decode sharded -> DRAM -> sharded | Required: head width 96 splits at 48, unsupported by the width-64 rotary kernel |
| Attention | SDPA/paged SDPA -> concat heads -> O linear | dedicated attention kernels; required decode head shard | Prefill dedicated concat retained; decode concat already dedicated |
| Cache prefill | two paged fills | DRAM paged cache | No paired fill op exists |
| Cache decode b1 | two paged updates | K/V L1 shards -> paged DRAM | Paired fused update retained; faster |
| Cache decode b32 | two paged updates | existing QKV-head shards -> paged DRAM | Paired update rejected: extra V reshard lost |
| MLP | RMSNorm -> packed gate/up linear -> two slices -> SiLU -> multiply -> down linear -> residual add | DRAM | SiLU folded into multiply |

### Dedicated fused ops

- Activation, RMSNorm, SDPA, split QKV/heads, decode head creation, prefill
  head concatenation, decode head concatenation, and paged cache update were
  assessed.
- RMSNorm, packed QKV split, SDPA, decode head creation/concatenation were
  already dedicated in the functional graph.
- Dedicated prefill head concat at batch 32 and batch-1 paired cache update
  were applied. The batch-1 concat candidate was measured slower and reverted.
- Phi's rotate-half width is 96 with 48-wide halves; both available rotary
  kernels require incompatible geometry/interleaving. The native HF op was
  run and rejected its 96 padded width; padding to 128 ran but changed the
  midpoint to 64 and failed the Phi semantic oracle. The llama transformation
  is a repeated 32x32 adjacent-pair rotation and cannot cross the 48 boundary.
  Exact executable evidence is `rope_candidate.log` and
  `tests/fused_decoder_rope_candidate.py`.
- There is no applicable MoE, gate/router, top-k, convolution, pooling,
  distributed norm, collective, or sampler graph in this single dense layer.

### Graph rewrites

- QKV is already one packed shared-LHS linear. Gate/up is already one packed
  shared-LHS linear; splitting it would add a matmul dispatch.
- There is no RepVGG convolution, spatial reduction, or redundant
  permute-reshape-permute identity.
- Long-prefill chunking accepts arbitrary logical lengths and internally pads
  only tail work; no public alignment restriction was introduced.

### Op merging

- SiLU was folded into the consuming multiply and was PCC/performance checked.
- There are no linear biases in Phi-3.5 Mini, so matmul+bias is inapplicable.
- Matmul activation fusion is not equivalent to SwiGLU and was not applied.
- Residual-add + RMSNorm cannot replace the existing pair without losing the
  unnormalized residual required by the later residual add or recomputing it.
- No adjacent transpose, slice, stable-softmax max subtraction, reduction,
  padding consumer, batchnorm, convolution scale, or convolution activation is
  present.

The final profiler-delimited paths contain no Torch/from_torch/to_torch or host
fallback. Decode RoPE's tilize/untilize and sharded/interleaved transitions are
the minimal functional path for a 48-wide rotate-half on TTNN; batch-1's one
extra V reshard is required by the paired update and is net faster. No
unnecessary layout conversion remains.

## Commands

```bash
pytest -q -s \
  models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_fused_decoder_full.py \
  models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_fused_decoder.py

pytest -q -s \
  models/autoports/microsoft_phi_3_5_mini_instruct/tests/fused_decoder_ab_perf.py

for round in 1 2 3 4 5 6; do
  if (( round % 2 )); then order=functional-first; else order=fused-first; fi
  PHI_FUSED_AB_ORDER=$order PHI_FUSED_DECODE_REPLAYS=100 pytest -q -s \
    models/autoports/microsoft_phi_3_5_mini_instruct/tests/fused_decoder_ab_perf.py
done

PHI_FUSED_PREFILL_REPLAYS=1 PHI_FUSED_DECODE_REPLAYS=5 \
python -m tracy -r -p -v -m pytest -q -s \
  models/autoports/microsoft_phi_3_5_mini_instruct/tests/fused_decoder_ab_perf.py

pytest -q -s \
  models/autoports/microsoft_phi_3_5_mini_instruct/tests/fused_decoder_rope_candidate.py

TT_METAL_WATCHER=10 \
TT_METAL_LOGS_PATH=$PWD/models/autoports/microsoft_phi_3_5_mini_instruct/doc/fused_decoder/watcher_final \
pytest -q -s \
  models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_fused_decoder_full.py \
  -k 'real_weight_paged_prefill_and_decode or decode_trace_replay_is_deterministic'
```
