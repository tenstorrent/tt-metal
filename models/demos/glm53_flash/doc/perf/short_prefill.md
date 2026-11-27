# Short-prefill cost on the TP32 stack (F17, 2026-11-13)

Question: why do short prefills run at ~2K tok/s (ISL 4K) while 32K runs at ~4.8K tok/s, and what does it cost AgentX? This matters because AgentX's uncached work is mostly per-turn suffixes after a cached prefix (`agentx_scaling_model.md`). This note is device-free. `chain_pf_1.sh` holds the measurements that settle the open terms.

## 1. Where a prefill's time goes

The generator chunks a prompt at the largest bucket (`StackBackend.prefill_chunk_tokens` = max(`GLM53_PREFILL_BUCKETS`) = 8,192 with the galaxy defaults: 2048, 8192). The last partial chunk runs in the smallest bucket that holds it (`TPPrefill.bucket_for`): ≤ 2,048 tokens → the 2,048 key, otherwise the 8,192 key. Both buckets run sequence-parallel (`mode`: c % 1,024 == 0): 64 and 256 rows per chip. One chunk is one trace replay of all 45 layers plus the head (`_prefill_chunk_tp`). A chunk past position 0 uses the sparse key (bucket, smallest captured window ≥ start + bucket).

### Measured (metal e2e TTFT, B 4 build on 782ac745e7d + M5; served b16 bench)

| source | ISL (per user) | OSL (per user) | concurrency | chunks run | TTFT | tok/s |
|---|---|---|---|---|---|---|
| metal `m5p_k3_128` | 127 | 32 | 1 | 1 × 256 (replicated bucket of the short-ISL defaults) | 189 ms | — |
| metal `m5_k3_4k_dense` | 4,095 | 128 | 1 | 1 × 8,192 at 0 (**4,095 valid of 8,192 rows**) | 1,594 ms | 2.6K |
| metal `m5_k3_32k` | 32,767 | 128 | 1 | 4 × 8,192 | 6,818 ms | 4.8K |
| metal `m5_k3_1m` | 1,048,447 | 128 | 1 | 128 × 8,192 | 282,055 ms | 3.7K |
| served `scale1` (b16, buckets 2048,8192) | 128 | 128 | 1 | 1 × 2,048 at 0 | 342 ms | — |
| served `scale1` | 4,095 | 128 | 1 | 1 × 8,192 at 0 | 1,823 ms | 2.2K |
| served `scale1` | 32,767 | 128 | 4 / 8 / 16 | 4 × 8,192 each, serialised | 14.4 / 36.9 / 88.5 s mean | 3.3 / 3.5 / 3.6K (in + out, total) |

### Derived (estimates from the rows above, not measured per chunk)

- **8,192 chunk after position 0, 8–32K context:** (6,818 − 1,594) / 3 ≈ **1.74 s**, i.e. 0.21 ms per row. Averaged over a 1M prompt, 2.20 s per chunk: the DSA context term (index scorer and top-k over a growing pool) adds ~0.5 s by ~1M.
- **8,192 chunk at position 0 plus request overhead:** ≈ 1.59 s. With ~0.15–0.2 s of head, sampling and host work, the chunk itself is ≈ 1.4 s.
- **Served overhead over metal:** ≈ 0.23 s per request (1,823 vs 1,594 ms at 4K c1).
- **2,048 chunk at position 0:** ≲ 0.1–0.2 s (the served ISL 128 TTFT of 342 ms minus the served overhead). If that holds, a 2,048 chunk costs ≤ 1/7 of an 8,192 chunk for 1/4 of the rows. The cost per row would then be strongly superlinear in chunk size; the in-chunk causal DSA scorer is quadratic at position 0. **This is the term the key bench must confirm.** Every chunk policy below depends on it.

So the ISL 4K penalty is mainly **padding**: half of the 8,192 rows are zero. The rest is a ~0.15–0.2 s fixed cost per request (head, sampler and host writes; served +0.23 s), which a 32K prompt amortises over 4 chunks.

### Padding on the AgentX trace

Source: the b16 c 8 export, 350 profiling requests. "Exact" means exact-position snapshots, the `chain_sched_1` / `chain_scale_2` ref leg: the cached prefix ends at the previous prompt's end.

| prefill policy | suffix ISL P50 / P90 / mean (per request) | OSL (per user) | concurrency | rows replayed | valid | padding |
|---|---|---|---|---|---|---|
| measured (aligned 8,192 snapshots) | 6,076 / 11,024 / 8,037 | 863 | 8 | 3,684,352 | 2,812,786 | 23.7% |
| exact snapshots, base (2048, 8192) | 1,284 / 4,613 / 2,371 | 863 | 8 | 1,536,000 | 777,724 | **49.4%** |
| exact + tail split ≤ 3 × 2,048 | same | 863 | 8 | 1,187,840 | 777,724 | 34.5% (−22.7% rows) |
| exact + tail split + a 1,024 bucket | same | 863 | 8 | 1,038,336 | 777,724 | 25.1% (−32.4%) |
| exact + tail split + 512 + 1,024 buckets | same | 863 | 8 | 990,720 | 777,724 | 21.5% (−35.5%) |

Under exact snapshots, 221 of 328 suffixes are ≤ 2,048 tokens (143 ≤ 1,024; 37 ≤ 256), and 90 are 2–8K. Each of those 2–8K suffixes replays a full 8,192-row chunk.

## 2. Fixes, ranked by AgentX gain

Gains use F14's saturated device (c 8 / c 16 ≈ 100% busy, c 4 95%) and assume cost ∝ rows at 0.21 ms per row. All are estimates until `chain_pf_1` measures.

| # | fix | ISL (per user) | OSL (per user) | gain at c 4 / c 8 / c 16 (total tok/s) | TTFT | status |
|---|---|---|---|---|---|---|
| 1 | **Tail split** (`GLM53_PF_TAIL_SPLIT=3`): a 2,049–6,144-token last chunk runs as ≤ 3 × 2,048 on the already-captured 2,048 keys | AgentX suffixes after exact snapshots | 863 | ~+2 / +3 / +4% (−0.21 s of device time per request) | P90 −0.6 to −0.9 s; P50 unchanged (P50 suffix ≤ 2,048) | **implemented** (wt_pf), chain_pf_1 A/B |
| 2 | **1,024-row bucket** (`GLM53_PREFILL_BUCKETS=1024,2048,8192`, no code if it builds) | ≤ 1,024-token suffixes (44% of requests) | 863 | ~+1–1.5% on top of #1 | small | `pf_bench_1k` tests build + cost; 32 rows per chip may hit a kernel limit |
| 3 | **2,048-only chunking** (`GLM53_PREFILL_BUCKETS=2048`, no code) | cold / first requests 65K–262K, long suffixes | 863 | decided by the bench: if 4·c(2048) < c(8192), cold prefill gets c(8192)/(4·c(2048)) faster (cold TTFT P99 31–113 s shrinks by the same factor) and decode stalls under interleave drop from ~1.7 s to ~0.2–0.45 s per chunk; if not, long prompts lose | P99 | chain_pf_1 32K / prefix rows |
| 4 | **Host overlap / served overhead** (~0.23 s per request served vs metal) | all | — | ~0 at saturation (decode fills the gap); TTFT only | −0.2 s at low load | not started |
| 5 | **Packed multi-user suffix prefill** (`GLM53_PF_PACKED`, exists) | many short suffixes at c 8 / c 16 | — | only if the bench shows a large fixed cost per chunk | — | blocked: fails the identical-prompt gate as served (ledger 2026-11-12) |
| 6 | **Interleave 1 : N** | — | — | 0 (trades TTFT for TPOT once saturated; F14) | — | — |

Exact snapshots (F14 #1, +8 / +12 / +20%) come first and are already in flight. Fixes #1–#3 act on the suffix they leave.

## 3. Implementation (wt_pf)

- **`tt/generator.py`, `Glm53Generator._tail_split`.** Opt-in through `GLM53_PF_TAIL_SPLIT=k`. It runs before the slice-boundary rule.
  - Splits the request's last chunk into ⌈n/b⌉ chunks of the smallest bucket b when that fits k and replays fewer rows than the natural bucket.
  - Skips the split when b is off the continuation grid, or when the last piece would end on a chip slice boundary.
  - Counters: `prefill_tail_splits`, `prefill_tail_rows_saved`.
  - Host-checked with ttnn import-blocked: 4,095 → 2,048 + 2,047; 4,150 at 32,768 → 2,048 + 2,048 + 54; 1,284 unchanged; 8,192 / 9,000 / 6,145 unchanged; pieces cover the request in order.
- **`tests/test_mk_e2e.py`.**
  - `GLM53_MK_PF_BENCH=N`: the key bench, after the e2e and its gates. Per bucket it times start 0 full and half-full, plus the last chunk of each sparse window, N replays with a device sync on each side, then fits fixed + per-row over the full chunks at start 0.
  - `GLM53_MK_PREFIX_TAIL`: the prefix row's suffix length (default 256 = unchanged).
  - `MK_PREFIX_ALL_IDS`: an ids hash on the prefix row, so base and fix runs can be compared.

## 4. Measurement: `_logs/mesh/chain_pf_1.sh` (not launched)

It waits on `chain_(mu5_1|w8_2|scale_2|ctx_1)`. B 4 build, OSL 128 unless noted.

| row | config | ISL (per user) | OSL (per user) | concurrency |
|---|---|---|---|---|
| `pf_bench_4k` | key bench, buckets 2048,8192, windows 32/64/96/128K, watchdog 300 s | 4,096 | 32 | 1 |
| `pf_bench_1k` | + a 1,024 bucket (non-fatal) | 4,096 | 32 | 1 |
| `pf_4k_x1_base` / `_split3` / `_b2048` | TTFT A/B, ids gate vs base | 4,096 | 128 | 1 |
| `pf_4k_x4_base` / `_split3` / `_b2048` | TTFT A/B (4 distinct users) | 4,096 | 128 | 4 |
| `pf_32k_x1_base` / `_b2048` | TTFT A/B (split3 plans 32,767 like base) | 32,768 | 128 | 1 |
| `pf_pre_32k4k_base` / `_split3` / `_b2048` | 32,768 cached (attach) + a ~4,150-token suffix: the AgentX shape; B == B_cold inside each row plus an ids gate vs base | ~36,900 | 128 | 1–2 |

## 5. Risks

- **ids may move.** Chunk boundaries move under split3 and b2048: KDA chunk-state hand-off points and the DSA in-chunk scorer rows change. Near-ties can flip, and the gate reports SAME / DIFFER. A DIFFER needs its first divergence and the top-2 gap before calling it a regression.
- **The 1,024 bucket** (32 rows per chip) may break a kernel assumption; `pf_bench_1k` fails at build time, not as a hang.
- **The bench writes garbage** into slot 0 after the gates; timing only.
- **Hangs.** No new kernels and no new fabric phases. split3 and b2048 replay keys that are already captured; the 1,024 bucket captures new keys of an existing program family.
- **Trace region.** b2048 drops the 8,192 keys (frees space); the 1,024 bucket adds 5 keys at boot (~+45 s capture).
