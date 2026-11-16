# WS1 · Batched decode (A, 16 Nov 2026)

Branch `ws1-batched`, gate commit `00df3b5fa31` (= `integration` 9a2593673f + WS1). Precision `bfp8_release_selected`.
Status: **landed partial**. Every proof line is met except one: router + experts at **batch 1** is still 33 device programs
(target ≤ 10; batch 32 is 9). The batch-1 MoE block is what WS4's `moe_decode` replaces. B confirmed that op takes
`RouterTopk.op_packed` output directly, which makes batch-1 router + experts about 4 programs. The gap closes when WS4 lands.
The stage metric improved 21× (batch-32 step 501 → 23.3 ms), so it lands under the partial rule.

## Result

| | WS0 | `integration` (WS2) | WS1 | Proof |
|---|---|---|---|---|
| Batch-32 native step (`decode_timing.json`) | 501.0 ms | 372.3 ms | **23.3 ms** (42.9 tok/s/user, 1371 total) | ≤ 35 ms ✓ |
| Batch 8 / 16 step | 129 / 255 ms | 97.1 / 190.7 ms | 18.5 / 21.0 ms | |
| Batch-1 step, 128 / 4K prompt | 18.6 / 19.0 ms | 11.48 / 11.96 ms | 11.64 / 12.14 ms (85.9 / 82.4 tok/s) | ≤ 3 % slower ✓ (+1.4 %) |
| vLLM serving 128/128, 16 users (`serve_perf_c16.json`) | 3.9 tok/s/user | 5.2 tok/s/user | **59.5 tok/s/user**, 554 total, TTFT 1.6 s, 32/32 | ≥ 25 ✓ |
| vLLM serving, 1 user | 53.6 tok/s | 86.6 tok/s | 85.4 tok/s, TTFT 88 ms | |
| Batched vs serial min user PCC, B 2/3/8/32 mixed positions (`quick_check.json`) | | | 0.99981 / 0.99999 (L0 / L5) | ≥ 0.999 ✓ |
| KV writes batched vs serial | | | bit-identical | exact ✓ |
| Inactive slot (position -1) | | | cache pages bit-identical; others PCC 0.99981 | untouched ✓ |
| Router + experts programs, batch 32 (`ops_profile_b32.txt`) | 34 | 33 | **9** (4 + 5) | ≤ 10 ✓ |
| Router + experts programs, batch 1 (`ops_profile_b1.txt`) | 34 | 33 | 33 (25 + 8, unchanged) | ≤ 10 ✗ (WS4) |

The batch-1 step is 1.4 % slower than integration (11.48 → 11.64 ms). That is within the 3 % no-regression limit and
probably run-to-run noise: the batch-1 decode path is unchanged, and the quick check shows the same layer time
(B1_layers_ms 11.91 vs 11.81).

## What changed

Batch 2-32 decode now runs every layer once for all users. Previously each user went through attention, router and experts in turn.
`GEMMA4_26B_WS1_BATCHED=0` restores the per-user path. The batch-1 path is unchanged.

| Change | Effect (quick check, 2-layer, scaled to 30 layers) |
|---|---|
| One pass for all users: input norm in the WS2 sharded layout, QKV with the decode program for one row tile, `QkvHeadsNormRope` for B users, two `paged_update_cache`, one paged SDPA, `ConcatHeadsDecode`, one all-reduce | B32 371 → 31 ms |
| Experts expert-parallel for batched decode: the existing 32-whole-experts-per-chip prefill weights, union of the users' selections, weights in the decode precision policy (BFP8 down; prefill keeps BFP4) | (part of the above; BFP4 down gave PCC 0.995 vs serial) |
| Batched SDPA takes q interleaved in DRAM: the head op shards users over the 11-wide grid, SDPA maps them on 8x8 and misread users ≥ 8 when q was sharded | PCC at B16/B32 0.45 → 0.9998 |
| Router: scale, BF16 projection, `RouterTopk` (gemma4 custom op: this chip's local weight table and union sparsity in one program). Experts: gate_up `sparse_matmul`, `GegluDecode`, down `sparse_matmul`, `MoeCombine`. Expert-parallel weights packed `[up \| gate]` for `GegluDecode`; prefill slices to match and is bit-identical | router + experts ~24 → 9 programs; B32 31.2 → 28.0 ms |
| Model: inactive slots (position -1) stay in the batched pass; the cache update and SDPA skip them. RoPE rows are gathered once per step and layer type at every batch size | vLLM partial batches no longer loop per slot |

Measured and rejected: batch 1 through the expert-parallel path (7 programs instead of 33) was **20 % slower**
(B1_layers_ms 11.91 → 14.31). A token's 8 experts land unevenly on the 4 chips, so the busiest chip reads more than the
tensor-parallel split does.

## Where batch 32 spends its time (`ops_profile_b32.txt`, sliding layer, device µs)

gate_up `sparse_matmul` 270, down 139, `MoeCombine` 31, `GegluDecode` 17, `RouterTopk` 65, everything else ~230.
The two expert matmuls read the 32 local experts at about **490-500 GB/s per chip**, close to Blackhole DRAM peak.
Batch 32 is now bandwidth-bound on expert weights. A dense all-expert path (WS3 as planned) would read the same bytes, so further
batch-32 gains come from fewer bytes (BFP4 down/gate where accuracy holds), from overlapping the glue, or from `RouterTopk`
(65 µs: scalar top-k of 32 rows).

## Accuracy

`tests/test_device_numerics.py`: 4 passed (`numerics.txt`), PCC identical to integration (decode 0.99984 / 0.99991,
prefill 0.99882 / 0.99906).

Agreement (`agreement.json`, `perf/tools/greedy_agreement.py`, reference = `integration` 9a2593673f run from a worktree):
batch-1 teacher-forced **100 %** on both prompts. The new `b8` mode feeds the reference tokens through one 8-user batched decode
with the prompts at different positions: **99.6 %** (worst slot 99.2 %).

## Serving and GPQA

`perf/tools/serve_bench.sh qb2port <out> perf,gpqa`: GPQA-10 **9 / 10**; serving 1 user 2/2 requests, 16 users 32/32,
no failures.

## Tools

- `perf/tools/quick_check.py`: `--check-batch` takes a list, plus an inactive-slot check (slot 1 at -1: its pages must stay
  bit-identical and the other users must match serial).
- `perf/tools/greedy_agreement.py` (from B): mode `b<N>` adds batched teacher-forced agreement.
- `perf/tools/layer_profile.py` (from B): `--batch N`.
- `perf/tools/ops_scope.py` (from B): anchors for the batched path (RouterTopk router, experts through MoeCombine).

## Known limits

- Batch-1 router + experts is 33 programs. It closes with WS4's `moe_decode` (input: `RouterTopk.op_packed`).
- Batched decode covers 2-32 users on one row tile. Above 32 is out of scope; `max_batch_size` is 32.
- About 2 GB per chip of extra DRAM: BFP8 copies of the expert-parallel down weights (and gate on the 5 full layers),
  so batched decode follows the decode precision policy.
