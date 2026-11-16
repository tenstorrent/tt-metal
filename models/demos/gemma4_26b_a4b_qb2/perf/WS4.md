# WS4 · moe_decode: batch-1 routed experts (B, 16 Nov 2026)

Branch `ws4-moe-op`, gate commit `eeb249376e` (= `integration` d9ea6e68e1 + WS4). Precision `bfp8_release_selected`.
Status: **gate passed** (16 Nov 2026, 19:49-20:05 UTC).

## Result

| | integration (WS1) | WS4 | Target |
|---|---|---|---|
| Expert weight read per chip, Gemma BFP8 (`moe_decode_bench.txt`) | ~170 GB/s (indexed `sparse_matmul`) | **401 GB/s** (13.8 MB in 34.4 µs) | ≥ 400 GB/s |
| Expert weight read per chip, gpt-oss BFP8 | – | 389 GB/s (28.2 MB in 72.5 µs); BFP4: Gemma 315, gpt-oss 350 GB/s | |
| MoE block per layer in the model: the routed experts (`moe_decode` gate\|up + down, `ops_profile.txt`) | ~80 µs | **34.5 µs** (layer 0) / **34.6 µs** (layer 5), 2 programs | ≤ 35 µs |
| Router + routed experts per sliding layer | ~141 µs, 33 programs | 59.1 µs, 6 programs (router 24.4 µs incl. the expert-input `mul`) | (WS1 proof: ≤ 10 programs) |
| Batch-1 native step, 128 prompt (`decode_timing.json`) | 11.64 ms | **8.44 ms** (-27.5 %) | (no regression) |
| Batch-1 native step, 4K prompt | 12.1 ms | 8.91 ms (-26.6 %) | |
| Batch 8 / 16 / 32 step | 18.48 / 20.97 / 23.33 ms | 18.53 / 21.05 / 23.36 ms (+0.3 / +0.4 / +0.1 %) | ≤ 3 % slower |
| Unit tests vs torch (`tests/test_moe_decode.py`) | – | 5 passed | Gemma + gpt-oss, BFP8 / BFP4 |

"MoE block" is read as the routed-expert block that `moe_decode` replaces; the row below it adds the router.

## What `moe_decode` is

`tt/moe_decode/` (Python `ProgramDescriptor` ops + kernels, see its README): the batch-1 MoE decode with
tensor-parallel experts (every chip holds its slice of each expert's intermediate dimension, as before).

- **Router tail** (`router_topk_b1`, 1 core): top-k of the FP32 logits, softmax over the picks, per-expert scale,
  in integer arithmetic (the data-movement cores have no FPU), written as the `RouterTopk.op_packed` row. Replaces the
  25-program batch-1 gate path (and `RouterTopk.op_packed`, 49 µs at batch 1).
- **gate|up** (3 readers on each DRAM bank's row + reducers): each reader reads only its bank (K slice of its column
  pairs for each routed expert, one contiguous chunk per expert), `custom_mm` on 1x32 activation rows gives fp32
  partials, reducers sum the banks, add the gate/up bias, apply GeGLU or SwiGLU and the routing weight.
- **down** (one core per output column, on its bank's row): the k experts' h rows concatenated along K, one
  `custom_mm` sums the experts; the optional down bias rides in a zero-padding row of the weights.
- Weights: DRAM width-sharded, one shard per bank, uneven splits where the tiles do not divide (H 2880). Each bank
  serves the same bytes for any routing. Built from the checkpoint at load, replacing the tensor-parallel decode copy.

Batch-1 router + experts in the model: `mul`, FP32 `linear`, `router_topk_b1`, expert `mul`, gate|up, down (6
programs; WS1's ≤ 10 proof for batch 1 closes with this).

## Path to 401 GB/s (Gemma, BFP8, 13.8 MB of routed-expert weights per chip)

| Step | gate\|up + down |
|---|---|
| Interleaved tiles, one core per (slot, pair) and per column | 47.8 µs, 289 GB/s |
| Bank-sharded chunks, compute per chunk as it lands | 40.2 µs, 343 GB/s |
| One fused program (h pulled by the down cores) | 42.3 µs (DRAM-bound with cross-bank reads; dropped) |
| Bank-local gate\|up (3 readers per bank row + reducers) | 34.4 µs, 400 GB/s |
| Down weights before h; stateful packet reads, one VC per reader | 34.4 µs, 401 GB/s |

Measured and dropped: multicasting h from 48 cores (serialized, 93 µs); splitting a core's reads over both NOCs
(77 µs); 2 readers per bank (6 output tiles exceed DST with fp32 accumulation); FP32-centered bf16 logits into
`RouterTopk` (agreement 99.6 % but 61 µs/layer).

## Accuracy

`numerics.txt` (decoder vs HF layer, `bfp8_release_selected`): layer 0 prefill 0.99882, decode 0.99985; layer 5
prefill 0.99906, decode 0.99991 (integration: 0.99882 / 0.99984 / 0.99906 / 0.99991). `quick_check.json` vs
`results/WS1/quick_check.json`: PASS (min user PCC 0.99982, KV equal, inactive users untouched). `test_moe_decode.txt`:
5 passed (Gemma and gpt-oss shapes, BFP8 and BFP4, PCC ≥ 0.9999 per chip vs torch on the quantized weights;
`router_topk_b1` vs the HF router tail).

Agreement (`agreement.json`, run with `GREEDY_AGREEMENT_PROMPTS=8`, 8 prompts x 128 tokens = 1024 scored tokens, as
agreed with the lead; the 2-prompt form moved 0.4 % per near-tie flip): **1014 / 1024 = 99.02 %** teacher-forced vs integration's
greedy tokens (per prompt 97.7-100 %).

Against HF itself (`hf_agreement.txt`: HF bf16 on CPU, top-1 at every position of integration's greedy tokens, same
histories): integration's tokens match HF at 997 / 1024 (97.4 %), WS4's teacher-forced predictions at 1001 / 1024
(97.8 %; the gate run's predictions are token-identical to the run compared). WS4 is not further from the reference model than integration; the 1 % disagreement between them is near-tie
flips.

## Serving and GPQA

`serve_bench.sh perf,gpqa` (vLLM, capacity 16): all runs completed.

| | integration (WS1) | WS4 |
|---|---|---|
| 1 user, 128/128: TTFT, decode tok/s | 88.3 ms, 85.4 | 88.7 ms, **117.6** |
| 16 users, 128/128: TTFT, decode tok/s per user, total tok/s | 1563 ms, 59.5, 553 | 1586 ms, 59.5, 550 |
| GPQA Diamond first 10 (threshold 0.8) | 9 / 10 | **8 / 10** (passed) |

GPQA samples at temperature 1.0; 8 vs 9 of 10 is within its run-to-run spread.

## Changes in code owned by WS1 (each its own commit)

- `GeneralizedRouter.decode_packed` (router file): FP32 projection + `router_topk_b1` for batch 1.
- `OptimizedExperts` (earlier): mix-matmul output dtype and the same-dtype typecast guard (WS2).

## Known limits

- Batch 1 only; batch 2-32 stays on WS1's expert-parallel path.
- gpt-oss BFP8 read 389 GB/s in the gate run (402 GB/s in an earlier run); the 400 GB/s proof is the Gemma shape.
- Agreement is 99.02 %, just above the 99 % line (10 near-tie flips in 1024; HF top-1 agreement is higher than
  integration's, see above).
- 401 GB/s is at the target with little margin: the gate|up program streams at ~450 GB/s and spends ~2-3 µs in the
  reduction tail; the two programs run back to back (a fused program measured DRAM-bound at the same total).
- `custom_mm` is LoFi only (the previous expert matmuls were LoFi too).
- The router projection (13 µs, 4 cores, FP32 x BF16) is now the largest batch-1 MoE item.

## Proof files (`perf/results/WS4/`)

`decode_timing.json`, `numerics.txt`, `quick_check.json`, `agreement.json`, `hf_agreement.txt`, `test_moe_decode.txt`,
`moe_decode_bench.txt`, `ops_profile.txt`, `gpqa/summary.json`, `serve_perf_c1.json`, `serve_perf_c16.json`.
