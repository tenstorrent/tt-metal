# pplx-embed-v1-4B on Blackhole P150 — optimisation log with negatives

Every experiment run on 2026-09-22/23, with the numbers. ISL=512 throughout.
E2E = `demo/_common.py` `run_perf`, 10 iterations, best-of, chip 0
(`TT_VISIBLE_DEVICES=0`). "Kernel" = `DEVICE KERNEL DURATION` from the profiler
(see §0 for why not FW duration). Code: `arg/pplx-embed-upstream`.

Positive results and the optimizations themselves are summarised in `POSITIVE_RESULTS.md`.

## Current best (shipping) vs H200

Measured in the demo's extended-trace mode (forward + pooling + I/O in one traced
replay; the default as of 2026-09-23 — see §17).

| batch | H200 | current best | × H200 | 3× target | gap |
|---|---|---|---|---|---|
| bs1  | 5.437 ms   | **17.6 ms** cold (18.3 sustained) | 3.24× | 16.3 ms  | −1.3 ms (−7%) |
| bs8  | 33.081 ms  | **115.3 ms** cold (121.0 sustained) | 3.49× | 99.2 ms  | −16 ms (−14%) |
| bs16 | 67.225 ms  | **221.0 ms** cold (228.2 sustained) | 3.29× | 201.7 ms | −19 ms (−9%) |
| bs32 | 139.150 ms | **425.5 ms** cold (451.1 sustained) | 3.06× | 417.5 ms | −8 ms (−2%) |

Previous: 25.2 / 155.6 / 288.7 / 543.5 (extended-trace fix), 25.9 / 156.4 / 290.9 / 557.8 (Generator fallback).

What got us here (positives, for context): MinimalMatmul subblock 1×8 (bs8 −12%,
bs16 −13%, bs32 −18%); head-split QKV/concat ported to 4B (bs1 −2.7 ms, bs32
−8.8 ms); `in0_block_w` cap 8→38 (bs1 −4.4 ms; FF2 35%→70% of roofline); fused
SwiGLU at bs8/16 (−1.6% / −5.7%); block-sharded LayerNorm at bs1 (−1.9 ms); SDPA
q/k chunk 512/256 (−2..4% batched). STS-B Spearman 0.8116→0.8125 unchanged.

## 0. Measurement corrections (these invalidated earlier conclusions)

| what | wrong reading | correct reading |
|---|---|---|
| `DEVICE FW DURATION` as op cost | bs1 per-iter FW sum **32.7 ms** vs e2e 25.9 → BinaryNg "7.8 ms / 30%" | FW starts before the previous op finishes (wait for GO). Kernel sum **21.0 ms**; BinaryNg **2.0 ms / 9.6%**; gaps 4.9 ms (19%) |
| eager `ttnn.*` microbenchmarks | add 512×2560 **150 µs**, I2S **85 µs** | dispatch-bound. Trace replay: add **8.8 µs**, I2S **5.1 µs** (model: 3.4) |
| `HARVESTING_STATE 0x0` after `tt-smi -r` = board broken | "needs sysadmin / CPLD" | not tensix harvesting; chip opens at 12×10. Real cause: 32-chip cluster open fails; `TT_VISIBLE_DEVICES=0` fixes |
| whole-CSV op shares | S2I 3.9–5% "at every batch"; Matmul 50% at bs32 | Generator warmup runs the LM head (10 `[32×2560]×[2560×16032/7648]` matmuls + S2Is per pass). Steady-state iterations have **0**. Filter to the traced window |
| rotary "worth 0.2–0.25 ms" (stub returned `clone`) | measured rotary − clone | profiler: **1.66 ms** bs1, **28.0 ms** bs32 |
| "FF1/FF3 are BFP8, 2.8 ms of roofline" | dead knob | `DecodersPrecision.performance` already sets FF1_FF3 = bfp4 |

## 1. Block-sharded activations (Sankar) — wash

Tuned-vs-tuned (both arms swept: grids 8×8→12×10, every `in0_block_w` divisor incl. K/gx,
every `out_subblock_h×w ≤ 8`, sharded and interleaved outputs). Traced-op µs at bs1 shapes:

| role (M×K×N) | L1-interleaved | block-sharded | Δ |
|---|---|---|---|
| QKV 512×2560×6144 | 102.7 (ibw 16) | 106.3 | +3.5% |
| FF13 512×2560×9728 | 155.7 (ibw 8) | 159.6 | +2.5% |
| FF2 512×9728×2560 | 155.5 (ibw 16) | 157.0 | +1.0% |
| WO 512×4096×2560 | 72.6 (ibw 32) | 73.7 | +1.5% |

Batched path (`minimal_matmul`, 120 cores, the shapes that are 93% of bs32 matmul time):

| role | DRAM | L1-int | block-sharded | Δ vs DRAM | bs32 share |
|---|---|---|---|---|---|
| FF13 512×2560×9728 | 152.1 | 150.0 | 148.1 | −2.6% | 49.3% |
| FF2 512×9728×2560 | 144.1 | 143.8 | 144.5 | +0.3% | 18.5% |
| QKV 2048×2560×6144 | 190.2 | 188.7 | 185.6 | −2.4% | 15.6% |
| WO 1024×4096×2560 | 82.1 | 80.1 | 79.1 | −3.7% | 9.9% |

Share-weighted **−2.0% of matmul time = <1% e2e**, smaller than the spread between shard
configs. The 2D mcast broadcasts in0 along each core row regardless of where it starts.
Where sharding *did* pay: LayerNorm (real gather to remove), −1.9 ms at bs1. A first
un-tuned run gave +11% — an artefact of `in0_block_w = full K` in the sharded arm.

## 2. L1-resident residual stream across the model (Sankar) — +3% at bs8, doesn't fit bs16/32

| batch | current | residual L1 + block-sharded LN | Δ |
|---|---|---|---|
| bs1 | 25.9 | 25.9 | 0 (already L1) |
| bs8 | 156.4 | **161.1** | **+3.0%** |
| bs16 | 290.9 | does not fit — 43 KB/core short (`buf@820992`, dataflow end `865408`) | — |
| bs32 | 557.8 | does not fit — 43 KB/core short, same addresses | — |

Decomposition at bs8, all at SDPA 256/128: DRAM **166.3** → residual-L1 (wide tensors
spilled) **163.7** (−1.6%) → + WO/FF2 outputs in L1 **161.1** (−3.1%). The cross-op
effect is real, but fitting needs SDPA 512/256→256/128 which alone costs **+6.3%**
(156.4→166.3): SDPA CBs at 512/256 are ~1.2 MB/core (q 256 KB, k/v 256 KB, 512 KB q×k,
outputs), leaving ~300 KB for every L1 tensor. Wide tensors can't be resident: Q and SDPA
out ~280 KB/core at bs8; MLP intermediate 353 KB/core (1.4 MB at bs32). Two static regions
compete: SDPA CBs on 8×10 (short 81.7 KB) then matmul dataflow buffers on 12×10 (short
89.4 KB). `QWEN_MM_BLOCK` moved the SDPA CB end by **0 bytes**; on the matmul region it
plateaued (89→53 KB after the first halving, then flat for 2,8,8 / 4,4,8 / 2,4,4 / 2,2,4).
Knob fix landed: `TT_PREFILL_<op>_L1=0` now forces DRAM (before, "0" fell through to L1).

## 3. SDPA grid / chunk — no e2e effect; small chunks hurt batch

SDPA runs on 64 (bs1) / 80 (batched) of 120 cores; at bs1 `q_chunk=512` → 32 work units.
Isolated sweep said 94.6→76.5 µs. **Did not survive e2e** — the microbench ran HiFi2, the
model runs LoFi; its "baseline" was slower than the model's real 75.2 µs.

| config | bs1 | bs8 | bs16 | bs32 |
|---|---|---|---|---|
| baseline | 25.9 | 157.1 | 290.9 | 557.8 |
| grid 12×10 | 26.0 | 156.3 | 291.2 | 555.3 |
| 10×10 + q_chunk 128 | 26.0 | **170.4 (+8.5%)** | **326.9 (+12.4%)** | **617.4 (+10.7%)** |

Height-sharded SDPA input: op rejects sharded operands (`sdpa_device_operation.cpp:44`);
SDPA already schedules `B·nh·ceil(Sq/q_chunk)` chunks itself — no gather to remove.

## 4. `prefill_len_cutoff` (MLP chunk = M) — no-op

| cutoff | bs1 | bs8 | bs16 | bs32 |
|---|---|---|---|---|
| **512** | 25.9 | 157.1 | 290.9 | 557.8 |
| 1024 | 25.9 | 156.2 | 290.9 | 557.3 |
| 2048 | 25.9 | 158.0 | 290.9 | 559.7 |
| 4096 | 25.9 | 157.9 | 290.9 | 558.2 |

Predicted by the real 4-D form: at 16384 rows FF13 = 106.8/106.5/106.6 ns/row for
Z=32/M=512, Z=8/M=2048, Z=4/M=4096 → **80.3–80.5% of LoFi peak**; FF2 96.6–97.2 ns/row →
**88.2–88.8%**. Only total rows matter. A `[1,1,M,K]` sweep had shown 34%→57% "scaling" —
that form starves the 120-core grid; the model doesn't use it.

## 5. Custom prefill matmul op (Sankar's `matmul_decode` analogue) — <6% ceiling

Weight prefetch pays only when weight bandwidth limits: bfp4→bfp8 (2× weight bytes) costs
**+12–15%** (FF13 130→150, FF2 125→143, WO 70→78 µs) and **0%** for QKV (197→192).
Measured DRAM ~416 GB/s; hottest matmul needs 177 GB/s. Matmuls at 80–89% of peak, ~33%
of bs32 device time → a *perfect* custom op is worth <6% e2e. The op also requires in0
tile heights ∈ {1,2,4,8} — a rewrite for 512-row activations, not a port.

## 6. Q→BFP8 typecast before SDPA at batch — neutral/negative, kept

K and V reach SDPA as bf16 (`skip_kv_cache_fill`), so the 590 µs/layer cast looked like
pure overhead (18 ms/iter at bs32). Skipped: bs8 **156.6** vs 156.4, bs16 **290.8** vs
290.9, bs32 **563.7 vs 557.8 (+1.1%)**. SDPA on bf16 Q costs what the cast saved.

## 7. `minimal_matmul` at bs1 on 120 cores — much worse

| config | bs1 |
|---|---|
| legacy 2D 8×8 (shipping) | **25.9** |
| minimal, subblock 1×8 | 45.6 |
| minimal, subblock 2×4 | 45.6 |
| minimal 1×8 + fused SwiGLU | 55.6 |

The bs1 legacy matmuls run at ~42% of *device* peak but on 64 cores — ~79% of those cores'
peak. The loss is the grid cap (Mt=16 → gy | 16), not the kernel.

## 8. SiLU moved into FF1's matmul epilogue at bs32 — +2.2%

`minimal_matmul(fused_activation=SILU)` + plain mul: **569.8 vs 557.8**. The SFPU epilogue
serialises with a matmul already at 80% of peak. (Standalone at bs1 shapes the SiLU is
~55% of the BinaryNg mul: 75.5 vs 34.1 µs traced.)

## 9. Head-split kernels, one barrier per Q/K/V unit — no gain, op at roofline

Bit-exact vs `nlp_create_qkv_heads`. bs1 `[1,1,512,6144]` bf16: **60.5→69.2 µs**;
B=8 bfp8: **137.4→135.9 µs** = 53 MB in 136 µs = **390 GB/s** (DRAM roofline). Reverted.
Model-side: GenericOp is 41.3 ms kernel at bs32 (582 µs/call ≈ 4× the B=8 number) — at
roofline; only removing the pass (fusion into the QKV matmul writer) helps.

## 10. Block-sharded residual add — not worth it

Traced: add 512×2560 L1-interleaved **8.8 µs**, block-sharded 8×8 **6.3 µs**, mixed
sharded/interleaved 8.9 µs, I2S 5.1, S2I 6.0. The model's adds already take 6.3–6.8 µs of
kernel time (the 51–109 µs in the profile was FW wait, §0). Sharding the residual would save
~2 µs × 2 adds × 36 layers ≈ 0.15 ms at bs1.

## 11. Earlier in the session (numbers from the first pass)

| experiment | result |
|---|---|
| legacy matmul for batched prefill | bs8 220.2 vs 189.3; bs32 868.9 vs 707.2 |
| MLP prefill chunking (smaller chunks) | 705.8 → 746.1 → 780.7 (bs32) |
| L1 batched activations (all variants) | "statically allocated dataflow buffers clash with L1 buffer" |
| bs1 core grid, 3 sweeps | flat 28.8–29.0 |
| head_groups for the head-split ops | flat |
| bs32 fused-SwiGLU recovery, 13 configs | best 567.5 vs 559.0 → fusion off at bs32 |
| sharded LayerNorm at batch | blocked by L1 (128 tiles/core) |
| tt-blaze | decode-oriented; bs1 already L1-resident, dispatch 4.5% |
| LM head running per iteration (7%) | retracted — warmup only; steady-state iterations have 0 calls |

## 13. Fused SwiGLU `minimal_matmul` at bs32 — structurally slower (chip 2, traced, `[1,32,512,2560]`)

| variant | µs | vs unfused total |
|---|---|---|
| unfused FF1 (N=9728, sb 1×8) ×2 | 1777.7 ×2 = 3555 | — |
| SwiGLU mul + SiLU | 1676.4 | — |
| **unfused total** | **5231.8** | — |
| fused blk 4,8,8 sb 1×4 (best) | 5753.9 | **+10.0%** |
| fused blk 8,8,8 sb 1×4 | 5811.8 | +11.1% |
| fused blk 8,8,8 sb 2×4 | 5872.9 | +12.3% |
| fused blk 8,8,8 sb 1×2 | 5949.1 | +13.7% |
| fused blk 8,4,8 / 8,8,4 / 8,8,16 / 2,8,8 | 5953 / 6200 / 6200 / 7640 | +14% … +46% |

The fused kernel keeps a gate and an up tile per output tile in DST, capping `subblock_w`
at 4; at that cap the fused matmul alone (5754) is 1.62× the unfused pair (3555). The
1676 µs mul it removes cannot cover that. Not fixable from the model side.

## 14. bs32 "gaps" (e2e − Σkernel = 60.6 ms) are tails inside ops, not dispatch

Sum of `OP TO OP LATENCY` over one steady-state iteration: **0.4 ms** (0.5 µs per op).
The remainder is within-op non-kernel time — e.g. FF13 traced 1778 µs vs 1659 µs kernel;
2432 blocks over 120 cores = 20.27 per core → ~5% tail. Per-op work-split tuning, not a
single lever.

## 15. RMSNorm compute-kernel config at bs32 — insensitive (chip 5, traced, `[1,1,16384,2560]` bfp8)

| fidelity | approx | fp32 acc | µs |
|---|---|---|---|
| HiFi2 (shipping) | 0 | 0 | 259.7 |
| HiFi2 | 1 | 0 | 259.6 |
| LoFi | 0 | 0 | 260.0 |
| LoFi | 1 | 0 | **256.0** (−1.4%) |
| any | any | 1 | 269–281 (+5–8%) |
| HiFi4 | 0/1 | 0 | 275 |

The interleaved LN kernel is structure-bound, not math-bound; fidelity is not a lever.

## 16. Why the bs1 legacy-grid bench was wrong, and two more bs1 accounting corrections

- **Weights.** Every standalone matmul bench here used DRAM-*interleaved* weights; the model
  uses **DRAM-width-sharded** weights (profile: `w=DRAM_WIDTH_SHARDED`). The model's 8×8
  legacy kernels (QKV 64.8 / FF2 102.6 / WO 45.2 µs) are faster than the bench's *best*
  wider grid (84.8 / 119.1 / 56.6). Reconstructing the model's exact `matmul_config`
  derivation still gave 139.5 (FF2) / 71.2 (WO) at 8×8 with interleaved weights. Any future
  matmul bench must use width-sharded weights or it says nothing about the model.
- **Launch tax is negligible.** Serial per-op time (FW end − previous FW end) minus kernel:
  **0.5–0.8 µs per op** for every op type at bs1 and bs32. Op-count reduction does not buy
  launch time; it only buys the fused ops' own kernel time.
- **Host bubble.** The first ops of each iteration (`UntilizeCodegen` 2.29 ms, `Embeddings`
  1.11 ms serial vs ~0 kernel) are the device waiting on the host between replays:
  **3.4 ms/iter at bs1 (13% of 25.9)**, 3.8 ms at bs32 (0.7%). Inputs are a few KB
  (tokens + chunk idx), so this is host round-trip latency (three blocking
  `copy_host_to_device_tensor`, trace issue, readback, sync), not bytes. bs1-specific lever.
- **q_norm compute config (chip 7, traced, `[32,32,512,128]` bf16):** shipping HiFi2/approx
  off ≈ 764–792 µs; **LoFi + approx, fp32-acc off = 691.6 µs (−11%)** → ~−3 ms/iter at
  bs32 (−0.5%). Accuracy (STS-B) must be re-validated before use.

## 17. POSITIVE — demo trace-key bug: timed iterations were on the Generator fallback

`trace_id_prefill.get(f"{seq_len}_0_{batch_size}")` never matched the Generator's 4-part
key, so the demo timed `prefill_forward_text` per iteration (eager slice+norm+to_layout,
4 H2D copies, blocking readback). Extended-trace mode: bs1 **25.2** (−2.7%), bs8 **155.6**
(−0.5%), bs16 **288.7** (−0.8%), bs32 **543.5** (−2.6%). Host cost per iteration in the
new mode at bs1 = 0.3 ms (device 25.05). `serve.py` was never affected. Landed as the
demo default (`--no-full-pipeline` opts out).

## 18. q_norm/k_norm RMSNorm at LoFi + approx (`QWEN_NORM_LOFI_APPROX=1`) — rejected on both axes

Traced q_norm 764 → 692 µs (−11%) did not survive e2e: bs32 full-pipeline **547.1 vs
543.5 (+0.7%)**, and STS-B Spearman **0.7974 vs 0.8125 (−0.015)**. Knob kept default-off.

## 19. POSITIVE — fused head-split + Q/K RMSNorm (`custom_ops/fused_qkv_heads_norm`, default on)

One generic_op with a compute kernel replaces `nlp_create_qkv_heads` + `q_norm` + `k_norm`.
Standalone PCC 1.00000 bf16 / 0.99900 bfp8; B=8 traced 275.1 → 193.8 µs (−29.6%). E2E
(full-pipeline): bs1 25.2→**25.0**, bs8 155.6→**144.7** (−7.0%), bs16 288.7→**276.8**
(−4.1%), bs32 543.5→**519.2** (−4.5%). STS-B 0.8125→**0.8135**.

## 20. POSITIVE — RoPE fused into the same op (`QWEN_FUSED_ROTARY`, default on)

Per head after gamma: `rot = x @ T` (single 32×32 tile-local rotation), `out = x·cos + rot·sin`.
Standalone PCC q 0.99997 / k 1.00000 vs `rotary_embedding_llama`; B=8 traced 670.1 → 334.5 µs
(−50.1%). E2E: bs1 25.0→**23.9** (−4.4%), bs8 144.7→**143.4**, bs16 276.8→**263.5** (−4.8%),
bs32 519.2→**495.0** (−4.7%). STS-B 0.8135→**0.8134**. Gotcha hit and fixed: upstream frees the
pre-rotary Q/K after rotary, which with an identity rotary are the SDPA inputs → guard one
deallocation each.

## 21. POSITIVE — fused op emits Q and K/V in bfp8 (`QWEN_FUSED_Q_BFP8=force`, `QWEN_FUSED_KV_BFP8=1`)

Q packed into its own output CB (17) in bfp8, K/V in bfp8, straight out of the fused
head-split+norm+RoPE op → the per-layer Q Typecast (469 µs at bs32 = 17 ms/iter) is gone
and SDPA reads half the operand bytes. Standalone B=8 traced: fused+Typecast(Q) 457.7 →
321.9 µs (−29.7%); fused+Typecast(Q,K,V) 519.6 → 312.0 µs (−39.9%).
E2E: bs1 23.9→**23.7**, bs8 143.4→**135.3** (−5.6%), bs16 263.5→**250.4** (−5.0%),
bs32 495.0→**474.3** (−4.2%). Q-only: 23.7 / 137.6 / 252.6 / 480.1.
STS-B (bs1, so it exercises the bfp8 operands): 0.8134 → 0.8164 (Q) → **0.8190** (Q+K/V).

Sub-result, bfp8 packing mode (B=8): `bfp8_pack_precise=False` (generic_op default) mean
|err| vs un-quantised 0.00654, PCC(fused8, typecast8) 0.99978; precise: 0.00601 (= the stock
Typecast's 0.00602), PCC 0.99980; kernel time identical (322.3 vs 322.2 µs). Precise is the
default (`QWEN_FUSED_BFP8_PRECISE=0` to disable).

## 22. POSITIVE — QKV projection writes bfp8 (`QWEN_QKV_OUT_BFP8=1`, default on)

Upstream pins the QKV matmul output to bf16 for the stock rotary's sake; the fused op is
now the only reader and takes bfp8. Projection writes and fused-op reads halve (201 → 100 MB
per layer at bs32). E2E: bs1 23.7 (unchanged), bs8 135.3→**127.5** (−5.8%), bs16
250.4→**240.9** (−3.8%), bs32 474.3→**456.7** (−3.7%). STS-B 0.8190→**0.8161** (day start
0.8134). Q/K/V are quantised once more before norm/RoPE — that is the whole accuracy delta.

## 23. POSITIVE (small) — generic_op core ranges: per-core `CoreRange`s cost ≈0.4 µs/core per launch

Standalone trace replay, bs1 head-split (kernel ≈20 µs): 55.9 µs/op at 120 cores, 30.6 at 64,
19.7 at 32; native `ttnn.add` same size 8.4. Cause: one single-core range per worker → kernel
binaries unicast per core per launch. Merged rectangles: 13.0 µs/op (120 cores); concat heads
7.9. Plus fewest-cores split (bs1 128 units → 64 cores × 2): norm+RoPE 60.3 → 51.7 µs/op.
E2E gain is much smaller than standalone because the dispatcher overlaps the next launch with
the previous (longer) op: bs1 23.7→**23.4**, bs8 127.5→**126.7**, bs16 240.9→**240.6**,
bs32 456.7→**455.8**. STS-B 0.8161 unchanged. Standalone back-to-back short ops overstate
launch cost; only the model timeline counts.

Where bs1 actually stands (profile ATTRIBUTES): all four bs1 matmuls run the legacy 2D-mcast
config on an **8×8 = 64-core grid** (`per_core_M=2`, LoFi): QKV 65 µs, WO 45, FF1/FF3 103 each,
FF2 102 → 15.2 ms of the 23.4 ms at ≈245 TFLOP/s, half of what the bs32 minimal_matmul reaches
(≈490). M = 16 tile-rows is the constraint: grid_y must divide 16 and grid_x must divide N
tiles (304/192/80), which pins 8×8 on a 12×10 part.

## 24. Wider legacy-matmul grids at bs1 — INVALID with DRAM width-sharded weights (NaN), not a win

All four bs1 matmuls run the legacy 2D-mcast kernel on 8×8 = 64 cores. Trying to widen:
`QWEN_QKV_GRID_X=12` (96 cores) gave bs1 23.4→23.0 and `QWEN_LEGACY_GRID_FF2/WO=10,8` 23.4→23.3 —
but STS-B came back **NaN**. Standalone with the model's DRAM-width-sharded bfp4 weights
(`check_mm_grids.py`): 8×8 correct; **10×8 and 12×8 return inf on every projection**; the same
grids on interleaved weights are correct. The kernel maps core column ↔ DRAM shard 1:1, so the
column count is pinned to the 8 DRAM banks, not to M. Interleaved weights on wider grids are
slower than sharded 8×8 (§16). Standalone (sharded, timing only): QKV 8×8 78.9 → 12×8 71.3 µs,
FF1 118 → 105, FF2 111 → 103, WO 52.5 → 50.4 — i.e. even if it were valid, only ≈ −10%.
Also failed: `MatmulMultiCoreReuseMultiCast1DProgramConfig` mcast_in0 on 76/102/120 cores
(TT_FATAL in program_spec), `transpose_mcast` (TT_FATAL), minimal_matmul at M=512 with blocks
(2,8,8)/(2,8,4)/(1,8,8): 110–197 µs for QKV vs 79 legacy. The `QWEN_QKV_GRID_X` per_core_N fix
stays in the code (correct sizing) but the knob must not be used with sharded weights.
The decode-style `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` (many workers per
DRAM bank, in0 L1-width-sharded + multicast — the natural 120-core design for this shape) hard-
fails validation at M=512: `M == 1 ... currently only supports` one tile row. A prefill version
of it is the custom op Sankar described; it is kernel work, not configuration.

## 25. Compute kernel v2 (DST-reuse, 6 passes/head instead of 9) — no speedup

`compute_qkv_heads_norm_v2.cpp` (`QWEN_FUSED_COMPUTE_V2=1`): PCC vs v1 ≥ 0.9995, timing
B=1 58.3→58.6 µs, B=8 236.1→235.7, B=32 837.4→833.3 (−0.5%). Pass count is not what costs
the fused op its +350 µs over the pure head-split at bs32 (521 vs 869 µs).

Follow-up: caching the cos/sin tiles across the 8 head-group units of a seq tile (reader pushes
once per seq tile, compute pops in lockstep) looked like a big win standalone — B=32 833.9→607.6
µs (−27%), B=8 235.7→203.3 — but same-chip A/B in the model (chips 4/6/7/8, sequential):
v1 23.3 / 126.5 / 240.2 / 454.4 vs v2+cache 23.2 / 126.7 / 239.6 / 455.1 — **nothing**. The
bench read cos/sin from DRAM; the model keeps its RoPE matrices in L1 (`QWEN_ROPE_PREFILL_L1=1`),
so there was no DRAM re-read to remove. STS-B with v2 0.8153 (vs 0.8161). Kept as a default-off
probe (`QWEN_FUSED_COMPUTE_V2=1`). Lesson (again): a standalone bench must replicate the memory
placement of *every* operand, and chip-to-chip e2e variation is ≈1.5% (bs16 on chip 8: 240.6,
on chip 10: 244.0 with an unrelated knob), so A/B only on the same chip, sequentially.

## 26. POSITIVE (small) — batched SDPA grid 12×10 + k-chunk 512 (`QWEN_SDPA_BATCHED_WIDE`, default on)

Standalone, model-exact config: B=32 932 → 833 (12×10) → 806 µs (k512); B=8 308 → 274 → 262.
bs1 fails with k512 (SDPA CBs clash with the L1-resident activations) so batch>1 only.
Same-chip sequential A/B: bs8 126.7→**126.2**, bs16 239.6→**239.6**, bs32 455.1→**450.6** (−1.0%).
bs8/bs16 gain far less than standalone predicted (the same placement lesson as §25 probably
applies to Q/K/V at bs8/16); bs32 matches (−4.5 ms ≈ 36 × 126 µs).

## 27. bs32 SwiGLU product `ttnn.mul` is already at roofline — nothing to gain from a custom eltwise kernel

`[1,32,512,9728]` bfp8: plain `ttnn.mul` 1247 µs vs 1181 µs roofline (508 MB at 430 GB/s) = 95%;
`add` 1268, `silu` 1272; `mul` with fused SiLU 1604 (the model fuses SiLU into the FF1 matmul, so
it runs the plain mul). The 60 ms/iter this product costs at bs32 (13%) can only go away by not
materialising it — silu(gate)·up as a matmul epilogue (FF3 reading gate, or FF2's in0 path) —
which is a matmul-kernel project, like the M=512 DRAM-sharded matmul for bs1 (§24).

## 28. POSITIVE — fused residual add + RMSNorm (`custom_ops/fused_add_rmsnorm`, bs16+)

Standalone bfp8 W=2560: M=16384 595.8 → 529.9 µs (−11%; roofline 415), M=4096 183.3 → 177.4,
M=512 52.9 → **62.8 (+19%)** — 16 rows use 16 cores, so bs1 keeps the stock ops. Same-chip
A/B: bs32 450.6→**443.5** (−1.6%), bs16 239.6→**234.8** (−2.0%), **bs8 126.2→128.3 (+1.7%)**
despite the standalone −3% at M=4096 (excluded via `_MIN_ROWS=8192`). Re-A/B after the bs8 matmul
block change (chip 7): off 122.5 vs on 122.2 (−0.2%) — neutral now, so bs8 stays on the stock ops.
Accuracy vs torch at M=16384: PCC 0.99890 fused vs 0.99879 stock; per-row max|err| 0.164 vs
0.134, mean|err| 0.0102 vs 0.0097 — same bfp8-limited regime. In the model
(`QWEN_FUSED_ADD_NORM_VERIFY=1`, bs16, real text): all 72 fused calls of a forward vs the stock
add + norm on the same live tensors — sum PCC ≥ 0.99988, norm PCC ≥ 0.9993.

Metric warning (cost ~1 h): comparing end-of-model hidden states between two runs is not an
equivalence test on this bfp8 pipeline. A known-benign kernel change (v2 head-split compute,
per-op PCC 0.9995, STS-B unchanged) gives per-token cos 0.87–0.97 after 36 layers, and the
batched post-processing (`process_hidden_states_after_prefill_trace_batched`) slices the
*flattened* [1,1,B·S,H] output, so its 32 rows are token positions 480–511 of user 0, not
users. STS-B (bs1) cannot exercise bs16+ paths; per-call verification on live tensors is the
usable check for batched-only changes.

## 29. minimal_matmul block sweep (model-faithful sharded bfp4 weights): bs8 wants different blocks; SiLU as a matmul activation is ruinous

Plain `minimal_matmul`, LoFi, bfp8 in/out, 12×10 grid, default blocks (8,8,8) sb 1×8 vs alternatives:
- **bs8 (M=4096)**: FF2 16,8,8 **−16.3%** (507→424 µs), QKV 8,4,8 **−15.0%** (377→320), WO 16,8,8
  **−11.6%** (234→207), FF1 8,8,16 −10.5% (moot: bs8 runs the fused SwiGLU kernel). **POSITIVE e2e:
  bs8 126.4 → 123.4 (−2.4%, chip 7); landed as the bs8 default.**
- **bs16 (M=8192)**: FF2/QKV/WO defaults are best (alternatives +1…+4%); FF1 8,8,16 −14.8% (moot, fused).
- **bs32 (M=16384)**: defaults best everywhere (16,8,8 within −0.3…−1.3%; others +3…+10%).
Fused SwiGLU kernel (`fuse_swiglu=True`, packed w13): bs16 blocks 4,8,8 sb 1×4 **2945 vs 3133 µs (−6.0%)**
for the current 8,8,8 1×8 → **POSITIVE e2e: bs16 237.0 → 231.8 (−2.2%, chip 8), landed as the bs16 default**; bs8 current config is already the best of 10; DRAM width-sharding the
packed weight is 1–4% *slower* than interleaved everywhere. At bs32 the fused kernel (best 5783 µs)
is still +16% vs the stock FF1 + FF3 + mul·SiLU (1659 + 1659 + 1676 = 4994) — §13 stands.
K_block sweep (plain matmuls, current blocks otherwise): every larger K step is slower —
bs32 FF1 K10 +1.8%, K16 +7.3%, K20 +7.8%, K40 +17%; FF2 K19 +2.9%, K38 +5.4%; QKV/WO +2…+15%;
bs16/bs8 the same direction (bs8 QKV K10 +17% — its K4 is the right call). The fused-SwiGLU
kernel is the exception: at bs16, 4,20,8 sb 1×4 = 2874 µs vs 2988 for 4,8,8 (−3.8%; **e2e 232.8 →
228.3, −1.9%, landed as the bs16 default**); at bs8 the shipped 8,8,8 1×8 stays best of 25 configs (N_block 16 fails validation).
Side finding: `fused_activation=SILU` on the FF1 `minimal_matmul` costs **1839 → 3285 µs (+79%)** at
bs32; SiLU inside the multiply costs +400 µs (1270 → 1604–1676) and a separate `ttnn.silu` 1272 — the
mul·SiLU the stock path uses is the cheapest place for it, and the SFPU SiLU is a large part of why
the fused-SwiGLU kernel loses at bs32. (A first version of this bench used FF1-with-SiLU as the
unfused baseline and wrongly showed the fused kernel −8% at bs32.)

## 30. Trace replay is not slower than eager at bs32; the profiler's Σkernel is not wall time

Same process, chip 6, bs32 full forward + post-processing: eager (host-dispatched, device-synced)
best **451.3 ms**, trace replay best **451.5 ms**. So there is no trace-specific overhead to
recover. The 40000-budget profile of the eager pass (401 ops, all with device data) reports
device wall 384.8 ms = Σ kernel 384.5 ms with zero inter-op gaps (FW−kernel 56.7 ms is the
wait-for-GO overlap) — 13% short of the 451 ms the same forward takes. Kernel durations are
derived from device cycles at the nominal AICLK. **Confirmed with tt-smi sampled every 4 s during a
150-iteration bs32 run: the loaded P150 runs at 1087–1143 MHz at 166–170 W / 66 °C** (idle chips
800 MHz / 37 W; chips holding an idle device 1350 MHz / 60 W). 384.5 ms × 1350/1110 ≈ 468 ms ≈ the
446–451 ms measured. Consequences: (1) every per-op "µs" and every roofline % in these notes that
came from the profiler is ~20% optimistic for sustained batched runs — compute-bound ops scale
with AICLK, DRAM does not, so the batched matmuls are closer to 70% of the *available* FPU rate
and the DRAM-bound ops closer to roofline than the tables say; (2) the chip is power-limited at
~170 W — anything that lowers power per FLOP (fewer active cores on DRAM-bound ops, bfp4 operands)
buys clock as well as bandwidth; (3) e2e numbers are unaffected. tt-smi device order matches
TT_VISIBLE_DEVICES here (index 6 was the loaded chip 6).

## 31. `ttnn.mul(..., fast_and_approximate_mode=True)` for the SwiGLU SiLU — no effect

bs32 `[1,32,512,9728]` bfp8: exact 1591.1 µs, approx 1599.1 (+0.5%), identical outputs (PCC and
mean|err| bit-for-bit the same); bs1 60.4 vs 60.2. The flag does not reach the SiLU path. Note the
bs1 number: the multiply is 27.4 µs plain vs 60.4 with SiLU → the SiLU costs 33 µs × 36 = 1.2 ms
per bs1 forward (5%) — worth moving if a cheaper home exists (see §32).

## 32. SiLU as the legacy 2D matmul's fused activation at bs1 — +10% worse

bs1 (M=512, sharded bfp4 w1, L1 activations, chip 1): FF1 plain 112.3 µs → with
`fused_activation=SILU` 173.2 µs (+61 µs), while the SiLU it removes from the multiply costs 33 µs
(60.4 → 27.4). Stock FF1+FF3+mul(SiLU) 272.6 µs vs FF1(SiLU)+FF3+plain mul 300.7 (+10.3%); PCC
0.99992 between the two. Together with §13/§29 (minimal_matmul: +79%) and the README's bs32 probe
(+2.2% e2e): the SFPU activation epilogue is expensive in every tt-metal matmul kernel here — the
multiply is the cheapest home for SiLU on this model, at every batch size.

## 33. `custom_ops/silu_mul`: the SwiGLU product as a generic_op — −13.7% at bs32, slower at bs1

`silu(a)·b` with a streamed per-unit reader, `copy_tile` → `silu_tile` on DST → dest-reuse FPU
multiply by b → pack (bfp8 in/out, precise pack). Standalone vs `ttnn.mul(a, b,
input_tensor_a_activations=[SILU])`:
- bs32 `[1,32,512,9728]`: **1373 vs 1591 µs (−13.7%)**; plain mul 1258 → the SFPU SiLU now costs
  115 µs instead of 333. Accuracy vs torch: PCC 0.99941 / mean|d| 0.0162 vs the stock 0.99897 /
  0.0244 — the stock activation path is *less* precise. `math_approx_mode` changes nothing.
- bs1 `[1,1,512,9728]` L1: 62.4 vs 59.2 (+5%) — 40 tiles per core, launch/latency-bound; stays stock.
- Variants: x·sigmoid_fast(x) via `mul_binary_tile` — slower (1456) and less accurate (PCC 0.9985);
  Blackhole `clamped_silu_glu_tile` — slower (1687) and wrong for |x| > 10 (max|d| 14; DeepSeek-V4
  clamp semantics), so not usable for this model.
**E2E A/B bs32 (chip 6): 443.8 → 438.1 (−1.3%), landed**; in-model verify (`QWEN_SILU_MUL_VERIFY=1`,
bs32, all 36 calls): PCC vs stock ≥ 0.9992, and vs torch the fused result is the closer one in 33/36 layers (`QWEN_SILU_MUL`, rows ≥ 8192 → bs32 only;
bs8/16 run the fused kernel).

## 34. Row-split fused add + RMSNorm for bs1 (`fused_add_rmsnorm_split`, R cores per tile-row)

At bs1 there are only 16 tile-rows, so the row-granular fused op used 16 of 120 cores and lost
(§28). The split variant gives each row R=5 cores (80 cores, 16 column tiles each): every core
adds and squares its slice, reduces it to a partial mean-square tile, writes that tile into slot k
of the other four cores' CB 8 over the NoC and bumps their semaphore; when a core's semaphore
reaches R it sums the partials, does rsqrt(+eps) and applies inv·gamma to its slice. One launch
replaces add → interleaved-to-sharded → block-sharded LayerNorm → sharded-to-interleaved.
Standalone M=512 (L1, bfp8): **35.4 µs vs 49.2 for add + interleaved rms_norm (−28%)**, vs 59.4 for
the row-granular fused op; PCC vs torch 0.999969 (stock 0.999966). Gotcha: the first attempt
launched the e2e A/B before the integration edit had applied (an anchor mismatch left the file
unchanged) — always check the edit's assertion output before launching.
**E2E (chip 4): bs1 23.2 → 24.4 ms (+5.2%) — NEGATIVE, default off** (`QWEN_FUSED_ADD_NORM_SPLIT=1`
to enable; STS-B with it 0.8140 vs 0.8161 — numerically fine). The standalone baseline was add +
*interleaved* rms_norm (49 µs); the model's bs1 path is
add → I2S → block-sharded LN on 80 cores → S2I (≈25–30 µs per pair incl. launches), and the split
op's ≈35 µs is fixed-cost bound: ~48 small NoC reads + barrier, six DST sessions on 16 tiles, a
semaphore round trip and the writes all serialize with nothing to overlap. It would need ≤ 20 µs
to pay; that means pipelining the reader against the compute and dropping the per-call gamma
reload — not a config change. Lesson: at bs1 every op is latency-bound, so "fewer ops" only wins
when the fused op's fixed cost is below the sum of the stock kernels', not just their bytes.

## 35. Fused-SwiGLU epilogue: batching gate/up pairs per DST session makes it *slower*

The `minimal_matmul` `fuse_swiglu` epilogue (`compute_metal2.cpp: swiglu_block`) does one gate/up
pair per DST session — acquire, `copy_init`, two copies, `silu_tile_init`, `silu_tile`,
`mul_binary_tile_init`, `mul_binary_tile`, pack, release — per output tile (≈2100 cycles/tile, vs
≈700 for the SiLU itself), which is the whole "1.62× per FLOP" loss of §13. Batching four pairs
per session with the inits hoisted (the obvious fix) measured **slower**: bs8 1512 → 1558 µs,
bs16 2874 → 3046, bs32 5783 → 6018, and less accurate (PCC vs torch 0.9984 vs 0.9992 unfused).
The one-pair form pipelines math against pack across sessions; four SiLUs in a row stall the
packer. Reverted. The epilogue's floor is the SFPU SiLU (~0.8 ms/layer at bs32) which cannot
overlap the FPU matmul on the same math thread — so the realistic ceiling for a "SwiGLU fusion
that keeps matmul efficiency" at bs32 is ≈ (product traffic 1.26 ms − exposed SiLU 0.8 ms) ≈
−0.4…−0.5 ms/layer ≈ −15 ms (3–4%), *if* the epilogue overhead beyond SiLU were removed —
not the −40…−55 ms projected earlier. bs8 note: unfused FF1(8,8,16) + FF3 + `silu_mul` is now
1475 µs vs fused 1512 (−2.4%) standalone, but **e2e (chip 7) 121.7 → 123.8 (+1.7%)** — the fused kernel
stays at bs8 (the unfused path pays two extra output writes and one more launch per layer).

## 12. Open / in progress

- **Fused QKV epilogue** (head-split + q/k RMSNorm + rotary in one model-local `generic_op` with a
  compute kernel): the only bs32 item whose numbers still support >20 ms — rotary 27 ms + q/k-norm
  30 ms + one head-split pass 41 ms are all DRAM-bound passes over the same Q/K/V tensors. Design read
  in progress (rotary per-tile math, attention prefill sequence, generic_op compute-kernel descriptor).
- **bs1 non-kernel device time**: full-pipeline mode is device-bound at 25.05 ms with a kernel sum of
  ~21 ms; a profile in this mode (chip 4) is locating the remaining ~4 ms.
- Everything else tried today is closed with numbers in §1–§18.

## 36. POSITIVE — bs1 SDPA q_chunk 256 (−3.9%) and the bs1 matmul block sweep (≈0)

Standalone SDPA `[1,32,512,128]` / K,V `[1,8,512,128]` bfp8 in L1 at the model's exact compute
config (LoFi, `fp32_dest_acc_en=False` → the *streaming* kernel, exp approx). The shipped bs1 config
(8×8, q512/k256) was **79.1 µs**; the earlier sweep (§3) had run HiFi2 and was discarded.

| grid / chunks (fp32 acc off) | µs | Δ vs shipped |
|---|---|---|
| 8×8 q512/k256 (shipped) | 79.1 | — |
| **8×8 q256/k256** | **55.0** | **−30%** |
| 8×10 q256/k256 | 55.4 | −30% |
| 8×8 q256/k128 | 58.9 | −26% |
| 8×10 q256/k512 | 66.7 | −16% |
| 12×10 q256/k256 | 67.6 | −15% |
| 8×8 q256/k512 | 71.5 | −10% |
| 8×8 q128/k128 | 72.2 | −9% |
| 8×8 q512/k256, fp32 acc **on** (legacy kernel) | 109.3 | +38% |

q256 turns 32 work units into 64, filling the 8×8 grid; 120 cores do not help at this size.
**e2e bs1 23.3 → 22.4 ms (−3.9%, chip 4, same-chip A/B); STS-B 0.8161 (unchanged).** Landed as a
bs1-only default (`apply_workload_env`: `QWEN_SDPA_Q_CHUNK=256`, `QWEN_SDPA_K_CHUNK=256`; opt out
`QWEN_SDPA_BS1_Q256=0`). Batched sizes keep 12×10 / k512.

**Legacy 2D-mcast block sweep (8×8, sharded bfp4 weights, L1 bfp8 activations, traced):**

| projection (K×N) | shipped | best | Δ |
|---|---|---|---|
| QKV 2560×6144 | in0_bw 10, sb 1×4: 71.0 | in0_bw 8, sb 1×6: 70.1 | −1.2% |
| WO 4096×2560 | in0_bw 16, sb 1×2: 52.3 | sb 1×5: 51.3 | −2.0% |
| FF1/FF3 2560×9728 | in0_bw 10, sb 1×2: 115.9 | in0_bw 8, sb 2×2: 111.7 | −3.7% |
| FF2 9728×2560 | in0_bw 38, sb 1×2: 113.1 | sb 1×5: 111.0 | −1.9% |

Sum ≈ −12 µs/layer ≈ −0.45 ms predicted; **e2e 22.4 → 22.3 ms (−0.1 ms, within noise)**. Kept as a
bs1-only default because every projection is individually faster (`QWEN_LEGACY_*_K<k>_N<n>` knobs
keyed by shape, guarded to skip shapes they do not divide; opt out `QWEN_LEGACY_BS1_BLOCKS=0`).

## 37. POSITIVE — DRAM-width-sharded in1 reader: one NoC read per block-row segment (−3.6% bs1)

`reader_bmm_tile_layout_in1_sender_writer_padding.cpp` (`IN1_DRAM_WIDTH_SHARDED`) issued one
576-byte (bfp4) NoC read per tile: 240 requests per 138 KB block for QKV, then a barrier. The tiles of
a block row that live in one bank are contiguous in DRAM and in the L1 block, so the row segment is
now a single multi-burst read (13.8 KB QKV, 21.9 KB FF1, 5.8 KB WO/FF2). Numerics identical (same
bytes to the same addresses; PCC vs torch unchanged to 5 digits).

| projection | per-tile reads | row-segment reads | Δ |
|---|---|---|---|
| QKV | 70.3 | 67.8 | −3.6% |
| WO | 51.3 | 49.0 | −4.5% |
| FF1 | 114.0 | 110.5 | −3.1% |
| FF2 | 111.6 | 102.8 | −7.9% |

**e2e bs1 22.3 → 21.5 ms (−3.6%, chip 4, kernel file swapped between runs).** Only the bs1 path uses
this kernel here (batched sizes run `minimal_matmul` on interleaved weights).

## 38. Where the bs1 matmul time goes — reader vs compute (diagnostic, chip 3, traced, 8×8)

| variant (QKV 512×2560×6144) | µs | reading |
|---|---|---|
| in1 DRAM width-sharded bfp4, LoFi (model) | 70.5 | — |
| same, **HiFi2** | 120.7 | +50 µs for 2× math passes → LoFi math ≈ 50 µs of the 70 |
| in1 DRAM **interleaved** bfp4 | 97.0 | tile-granular interleaved reads are the slowest option |
| in1 **L1** interleaved bfp4 | 101.1 | even from L1 — it is the request pattern, not DRAM |
| 12×8 grid, in1 interleaved (96 cores) | 68.3 | more cores buy nothing while the reader dominates |
| in0 bf16 instead of bfp8 | 71.1 | in0 volume is irrelevant |

FF1 and WO behave the same (HiFi2 +64%/+58%, interleaved +23%/+31%, 12×8 interleaved −4%/+12%).
So at LoFi the 64-core kernel is ≈70% FPU time and ≈30% weight delivery; §37 recovers part of the
delivery cost. **The 8×8 grid is now close to its LoFi compute time** (≈50 µs QKV); further bs1
matmul gains need more cores, which is blocked on the 8-bank ↔ 8-column mapping (see §39).

## 39. more cores for the bs1 matmuls: 1D multicast (NEGATIVE) and wider 2D grids (POSITIVE after a factory fix)

- **`MatmulMultiCoreReuseMultiCast1DProgramConfig(mcast_in0=True)`, per_core_M=16, every core
  reading its own in1 N-slice** (192 configs: grids 8×8 / 12×8 / 10×8 / 12×10 / 8×10, in0_block_w
  2–16, subblocks, interleaved weights; all width-sharded variants are rejected by validation):
  best **81.4 µs vs 70.4 (+16%)** for QKV and the same ~82 µs on every grid → bound by the single
  in0 sender (1.4 MB of L1-interleaved bfp8 read and multicast by one core ≈ 17 GB/s). Not a path.
- **2D grids with more columns than DRAM banks** (9–12 columns × 8 rows on the 8-bank width-sharded
  weights): every config returns inf/garbage (`finite=False`, or finite with PCC 0.1 for WO 10×8),
  confirming §16, and the timing is at best −6% (12×8 QKV 66.2, 10×8 WO 48.1, 12×8 FF2 103.4) with
  1.5× the weight bytes read. Cause found in `matmul_multicore_reuse_mcast_2d_program_factory.cpp`:
  the per-column bank walk sets `worker_core_stride = per_core_N_storage - storage_core_stride`,
  i.e. a column takes a whole bank stripe even when `per_core_N < per_core_N_storage`, so the L1
  block is overrun. Fix: cap at `per_core_N` (both factory variants). **With the fix every wide grid
  is numerically identical to 8×8 (PCC vs torch 0.99992 / 0.99989 / 0.99992 / 0.99974, same as 8×8)
  and much faster** (chip 3, traced, coalesced reader of §37, bfp4 width-sharded weights over 8 banks):

  | projection | 8×8 (shipped) | 12×8 | 12×10 | 10×8 | 11×8 |
  |---|---|---|---|---|---|
  | QKV 512×2560×6144 | 67.5 (1×4) | **49.8** (pn16, 1×4) | 48.5 | 59.2 | 59.5 |
  | WO 512×4096×2560 | 48.7 (1×2) | **40.1** (pn7, 2×1) | 39.7 | 43.6 | 43.9 |
  | FF1/FF3 512×2560×9728 | 110.9 (1×2) / 109.0 (2×2) | **79.5** (pn26, 2×2) | 78.6 | 95.1 | 84.1 |
  | FF2 512×9728×2560 | 101.8 (1×2) | **78.8** (pn7, 2×1) | 80.1 | 87.2 | 87.1 |

  12 columns read 1–2 bank segments per column; with per_core_N=7 the derived 1×1 subblock is slow
  (WO 56.9, FF2 109.0) — 2×1 is required. Sum: ≈ −112 µs/layer ≈ −4 ms at bs1.
  **e2e bs1 21.4 → 17.7 ms (−17%, chip 4, same-chip A/B `QWEN_LEGACY_BS1_WIDE=0` vs default);
  STS-B 0.8161 unchanged; bs8/16/32 unchanged (123.0 / 228.7 / 438.0).** Landed as the bs1 default
  (`QWEN_QKV_GRID_X=12`, `QWEN_LEGACY_GRID_{FF13,FF2,WO}=12,8`, `QWEN_LEGACY_TIGHT_PER_CORE_N=1`,
  2×1 subblocks for per_core_N 7). 12×10 (per_core_M=2 over 10 rows pads M to 640) is within 1 µs.
- **12×8 fine sweep (NEGATIVE e2e):** standalone QKV 2×4 49.3 vs 1×4 51.9 (−4.9%), WO 1×7 39.1 vs
  2×1 40.2 (−2.9%), FF1 in0_bw 5 79.0 vs 81.0 (−2.5%), FF2 1×7 75.6 vs 78.9 (−4.2%), ≈ −11 µs/layer;
  e2e 17.6 → 17.7 (chip 4, same-chip A/B) — within noise, not landed.

## 42. Gio's BGE-M3 P150 branch (`gtobarTT/bge_m3_p150_optimizations`) — what transfers

Reviewed 25 commits (B1 4.31 → 3.73 ms, B8 23.7 → 11.0, B16 39.0 → 20.5, B32 65.7 → 42 ms on a
13×10 p150a; Galaxy chips are 12×10, matching ours). Per item:

| BGE-M3 change | here |
|---|---|
| Streaming SDPA kernel (`fp32_dest_acc_en=False`) + LoFi | already ours (`compute_kernel_config_lofi`); the legacy kernel is +38% at bs1 (§36) |
| SDPA q256/k512 at B8/B16 on the streaming kernel | ours is 12×10/k512 batched (§: landed 2026-09-23); q256 at bs1 is §36 |
| QKV output / Q,K,V heads / LayerNorm output in L1 at B8/B16 | knobs exist; trace capture clashes by 42 KB/core with any subset (§40, §2) |
| Matmul outputs written in the LayerNorm shard layout at B1 (drops 48 I2S ops) | our I2S/S2I pairs cost 2.9 µs/chain at bs1 (≈0.2 ms total); launch tax is 0.5–0.8 µs/op (§16) — not worth an op-count project |
| In-model matmul sweeps, `out_block_h` splitting for L1 fit, 11×10/12×10/13×10 grids | our batched blocks were swept the same way (bs8/bs16 landings); the bs1 grid was blocked by the factory bug fixed in §39 |
| `no_padding` mask skip; mask cast once | n/a — GQA, no attention mask |
| Two command queues (`bge_m3_2cq` branch) | bs1 I/O is 0.2 ms of 17.7 (§41); nothing to overlap |
| Sustained-load AICLK drop 1350 → ~1170 MHz | matches our 1.09–1.14 GHz reading; bs16 drifts 217 → 235 ms within 10 iterations on a warm chip |

## 40. NEGATIVE — bs8/bs16 op outputs in L1 (Gio's BGE-M3 B8/B16 win does not transfer)

BGE-M3 gained 10–20% at B8/B16 from writing the QKV output, the Q/K/V heads and the LayerNorm output
to L1 (`gtobarTT/bge_m3_p150_optimizations`). Here the knobs exist (`TT_PREFILL_INTERMEDIATE_L1=1`,
per-op `TT_PREFILL_{QKV,HEADS,SDPA,CONCAT,FF13,FF2,WO}_L1`), and both the full set and the
attention-only set (QKV + heads + SDPA + concat, FF in DRAM) fail trace capture at bs8 and bs16 with
the same clash: `L1 buffer allocated at 1462656 and static circular buffer region ends at 1505408`
(42 KB short, program 285–293, the 12×10 SDPA/heads region). Same wall as §2. BGE-M3 has dim 1024
and 16 heads of 64; our 2560-dim, 32×128 heads at bs8 are 2.5× the bytes per core.

## 41. bs1 host path is already clean

`QWEN_ITER_TIMING=1` at bs1 (extended trace): h2d 0.04 ms, trace issue 0.02, readback enqueue 0.02,
**device sync 23.2**, to_torch 0.1 → the 3.4 ms host bubble of §16 is gone since the extended trace
landed; bs1 is device time only (kernels + ≈0.5–0.8 µs/op launch tax over ~600 ops).

## 43. NEGATIVE — legacy 2D matmul (sharded weights, coalesced reads) at bs8/bs16 shapes vs minimal_matmul

With §37/§39 the legacy kernel is worth re-checking where the batched path runs `minimal_matmul`
(chip 5, traced, bfp4 width-sharded weights, bfp8 DRAM activations, LoFi; legacy swept over
12×8 / 8×8 / 12×10 grids, every in0_block_w that fits L1 and subblocks up to 8 tiles):

| shape | minimal_matmul 12×10 (shipped blocks) | best legacy 2D | Δ |
|---|---|---|---|
| bs8 FF1 4096×2560×9728 | 615 µs (332 TFLOP/s) | 12×10 pm13 bw4 1×2: 631 | +2.6% |
| bs8 FF2 4096×9728×2560 | 431 (473) | 12×8 pm16 bw19 8×1: 523 | +21% |
| bs16 QKV 8192×2560×6144 | 595 (433) | 12×10 pm26 bw2 1×8: 731 | +23% |
| bs16 WO 8192×4096×2560 | 381 (451) | 12×8 pm32 bw8 8×1: 493 | +29% |

`minimal_matmul` stays for every batched projection; the legacy kernel's per_core_M ≥ 13 blocks
force in0_block_w down to 2–4 tiles to fit L1 and it loses its edge. (bs8 QKV/WO and bs16 FF1/FF2
rows are in `/tmp/bench_batched_legacy.log`; same picture.)

## 44. Gio's BGE-M3 branch, bs>1 items checked on this model (chips 7/8/10/11, same-chip A/B)

| BGE-M3 B8/B16/B32 change | ours | result |
|---|---|---|
| Q/K/V in bf8 for SDPA, streaming kernel, LoFi, q256/k512 | already shipped | — |
| QKV output / heads / SDPA out / concat in L1 | `TT_PREFILL_{QKV,HEADS,SDPA,CONCAT}_L1=1` | bs8, bs16: L1 clash (§40) |
| same with SDPA k-chunk 256 to shrink the SDPA CBs | `QWEN_SDPA_K_CHUNK=256` + the four knobs | bs8: still clashes, 28 KB short (`1073152` vs `1101952`) |
| WO / FF2 outputs in L1 (the residual add reads one operand from L1) | `TT_PREFILL_WO_L1=1 TT_PREFILL_FF2_L1=1` | bs8 122.8 → 122.3 (noise) |
| LayerNorm output in L1 (B8/B16: −4%) | new knob `TT_PREFILL_LN_L1=1` (`distributed_norm.py`) | bs8 121.5 → **123.2 (+1.4%)**; bs16 227.7 → 227.9 |
| LayerNorm with fp32 accumulation off (B16: −0.5 ms, PCC-gated) | new knob `QWEN_NORM_FP32_ACC=0` (`rmsnorm.py`) | bs8 123.4 → 122.5; bs32 434.3 → **439.6**; STS-B 0.8161 → 0.8159; bs16 229.4 → 216.9 sits inside bs16's run-to-run band (note below) |
| legacy 2D matmuls on 11×10–13×10 grids with `out_block_h` | measured (§43) | minimal_matmul faster |
| Q/K/V heads in L1 only (BGE-M3 B8: −1.0 ms) | `TT_PREFILL_HEADS_L1=1` (+ `QWEN_SDPA_K_CHUNK=256`) | bs8 k512: clash (`1490944`); bs8 k256: 121.4 → 120.5 (−0.7%, inside bs8's ±1 ms); bs16 k256: clash |
| `no_padding` mask skip, mask cast once, 2 command queues | no mask; I/O ≈ 0.2 ms | n/a |

## 45. NEGATIVE — minimal_matmul at bs1 (M=512) on 96/120 cores vs the legacy 12×8 kernel

Sweep on chip 3 (traced; grids 12×10 / 12×8; M/K/N blocks 4–16 / 5–10 / 4–8; subblocks 1×8, 2×4,
1×4; bfp4 weights interleaved or width-sharded; 180 configs per projection, all ran):

| projection | legacy 12×8 (§39) | best minimal_matmul | Δ |
|---|---|---|---|
| QKV | 49.8 µs | 12×8 blk 4,10,8 sb 1×8: 79.9 | +60% |
| WO | 40.1 | 12×8 blk 4,10,8 sb 1×8: 61.5 | +53% |
| FF1/FF3 | 79.5 | 12×8 blk 4,10,8 sb 1×8: 127.4 | +60% |
| FF2 | 78.8 | 12×8 blk 4,10,8 sb 1×8: 124.7 | +58% |

At M=512 each core owns at most 2 M-tiles per block, so `minimal_matmul`'s block reuse never
materialises; the legacy 2D multicast keeps the bs1 path (the reverse holds from M=4096 up, §43).
Also: fused heads-norm-RoPE 2-way Q split (`QWEN_HEADSPLIT_Q_SPLIT=2`, bit-identical output,
256 units of 12 tiles instead of 128 of 24): standalone 48.4 → 44.5 µs (−8%) at bs1, 185.0 → 183.3
at bs8; **e2e 17.7 → 17.6 (noise)**. Kept as an opt-in knob, default 1.

Our LayerNorm at bs8 is 90 µs per call, bfp8 in → bfp8 out, 120 cores, HiFi2 with fp32
accumulation (two passes over 10.6 MB plus the write ≈ 32 MB → ≈ 350 GB/s): DRAM-bound, so an L1
output should have helped as it did for BGE-M3 — it did not; the consumer (`minimal_matmul`)
streams in0 at full rate from DRAM anyway. **bs16 run-to-run band:** the shipped configuration
measured 216.8 / 228.7 / 229.1 / 229.4 / 227.7 / 230.5 ms across today's runs on chips 8 and 5,
and drifted 217 → 235 within one 10-iteration run as the chip warmed, so a single bs16 A/B cannot
resolve less than ≈ 5%. A bs16 A/A (same config twice, chip 7) gave 216.8 / 216.7, and the fp32-off
repeat on chip 5 gave 216.7 → 231.8: the band is between process launches, not within one.

## 46. bs>1 round (2026-09-24): SDPA 12×8, weight layout, multi-wave row-split add+RMSNorm

**Correction:** the batched model runs SDPA at **q512/k512 on 12×10** (profile attributes), not
q256/k512 — the q_chunk override path never reached the batched shapes. Standalone (bfp8 Q/K/V in
DRAM, LoFi, streaming kernel, traced):

| batch | 12×10 q512/k512 (model) | **12×8 q512/k512** | 10×10 q512/k512 | 12×10 q256/k512 |
|---|---|---|---|---|
| 8 | 272 µs | **234 (−14%)** | 249 | 411 |
| 16 | 470 | **458 (−2.5%)** | 506 | 683 |
| 32 | 856 | **812 (−5%)** | 888 | 1142 |

256 / 512 / 1024 work units take 3 / 5 / 9 waves on 96 cores as on 120, and 96 cores contend less
for DRAM. e2e: bs8 122.8 → 121.7 (−0.9%), bs16 230.6 → 229.3 (single pair; bs16 band), bs32 below.

**Weight layout for `minimal_matmul`** (bfp4, shipped blocks, traced): interleaved vs width-sharded
QKV/WO/FF1: M=4096 +1.5…+2.3% (sharded better), M=8192 −3.0/−1.7/−3.4%, M=16384 −3.4/−2.0/−1.6%;
FF2 prefers sharded at every M (+1…+2%). Knob `QWEN_WEIGHT_INTERLEAVED_K<k>_N<n>=1`
(`create_dram_sharded_mem_config`). e2e bs32 QKV+WO+W1/W3 interleaved: **441.9 → 436.9 (−1.1%)**;
bs16 single pair 216.7 → 231.4 is the bs16 band (3-pair alternating run below).

**Multi-wave row-split fused add+RMSNorm** (`fused_add_rmsnorm_split`, now any rows_t × R: unit u runs
on core u mod C in wave u div C, C a multiple of R, fixed peer groups, CB 8 slot per wave, monotonic
semaphore). Standalone, DRAM bfp8, PCC vs stock ≥ 0.9996 (same as the row-granular kernel):

| M | stock add + rms_norm | fused R=1 | R=2 | R=4 | **R=5** |
|---|---|---|---|---|---|
| 4096 | 184 µs | 178 (−3%) | 165 (−10%) | 144 (−22%) | **141 (−23.5%)** |
| 8192 | 320 | 299 (−7%) | 281 (−12%) | 250 (−22%) | **242 (−24%)** |
| 16384 | 597 | 530 (−11%) | 491 (−18%) | **459 (−23%)** | 463 (−22.5%) |

Predicted e2e: −3.1 ms bs8, −5.6 ms bs16, −10 ms bs32 (2 calls/layer). Knob `QWEN_FUSED_ADD_NORM_R`
(with `QWEN_FUSED_ADD_NORM_MIN_ROWS=4096` at bs8). R=10 at M=4096: 136.2 µs (−25.7%); R=20: 226.9 (+24%,
exchange-bound). **e2e: bs8 R=5 123.1 → 119.9 (−2.6%, chip 7); bs32 R=4 449.4 → 443.2 (−1.4%, chip 10).**
Per-call PCC vs the stock add + rms_norm in the model at bs8: sum ≥ 0.99988, norm ≥ 0.99980 (same
level as the row-granular kernel already shipping at bs16/bs32). bs32 SDPA 12×8: 435.1 → 436.4 e2e
(neutral; standalone −5%) — stays 12×10 at bs32.

**bs16, alternating multi-launch A/Bs (best-of-10 per launch, ms):**

| knob | A (shipped) launches | B launches | reading |
|---|---|---|---|
| weights interleaved (QKV/WO/W1/W3) | 216.7 / 231.1 / 216.7 | 228.6 / 216.7 / 216.6 | neutral: both arms visit both modes |
| SDPA 12×8 q512/k512 | 230.0 / 228.2 / 228.1 | 229.4 / 230.7 / 228.4 | neutral (all slow-mode) |
| add+norm row-split R=5 | 228.3 / 231.6 | 216.8 / 216.5 | B always fast-mode; gated on (standalone −24%) |

**Explained (tt-smi sampled at ~1 s during bs16 runs on chip 5):** the "modes" are the AICLK. Iterations
that read 216.7 ms ran at 1281–1350 MHz; a run that read 238–243 ms ran at 1112–1162 MHz throughout;
the earlier 217 → 232 flips happened ≈ 0.6 s into the load (iteration 3) when the power manager pulled
the clock down, and a run that starts on a warm chip never sees the fast iterations. Time scales as
1/AICLK (216.8 × 1300/1130 ≈ 249 vs 241 measured). No power or clock setting was touched
(`AICLK_LIMIT_MAX` 1350, board defaults). Consequence: "best of 10" is the cold number; sustained
(iterations 5–9) is bs8 123.8 (was 126.7), bs16 232 (was 235), bs32 452.6 (was 452.3), bs1 17.8 (18.0).

**bs32 warm-up bias:** a 10-iteration bs32 run reads 434.5 on iteration 0 and 452–453 on the rest
(power-limited clock), and "best" is the cold first iteration. In a sequential A/B the B arm starts
on a warmer chip, so bs32 A/Bs are biased *against* B; the two bs32 wins above (−5.0 and −6.2 ms)
were measured under that bias. Steady-state numbers are ≈ +4% over the reported best at bs32.

**Combined defaults, same chip:** bs32 old defaults 439.8 → new (add+norm R=4 + interleaved
QKV/WO/W1/W3) **428.1 (−2.7%, chip 6, B arm on the warmer chip)**; bs8 new defaults (SDPA 12×8 +
add+norm R=5) **118.5** (previous best 123.4, −4.0%); bs16 new defaults 216.7 / 226.5 across two
launches (the two modes; previous best 228.3 was a slow-mode run).

## 47. Accuracy through the batched paths (2026-09-24, shipped defaults after §46)

`eval_accuracy_tt.py` only ever exercised batch 1. New `demo/eval_accuracy_batched.py` runs the STS-B
test set (2758 texts) through the perf demo's exact batch-B configuration (`apply_workload_env(B,
512)`, eager `ttnn_prefill_forward`, final RMSNorm on host, masked mean over the real tokens; fixed
ISL 512, no bucketing, no attention mask — the same padding for every B, so the numbers compare
across batch sizes but sit below the bucketed bs1 script's 0.8161):

| batch path | STS-B Spearman (masked mean) | all finite |
|---|---|---|
| 1 | 0.8121 | yes |
| 8 | 0.8123 | yes |
| 16 | 0.8140 | yes |
| 32 | 0.8159 | yes |

Per-text cosine of the batched embeddings against the batch-1 embeddings is in the run log below
(`/tmp/embs_B*.pt`). Mean over the padded ISL ("fast" pooling without bucketing) scores 0.51–0.58 at
every B: with no attention mask and ~490 pad keys per query it is the padding, not the kernels.

## 48. Sustained-clock probes (2026-09-24): what still helps once the power manager has settled

Method: `sustained_run.sh` — one 30-iteration run per arm on the same chip, tt-smi sampled every
≈1.3 s, "sustained" = median of iterations 15–29, both arms back to back (the second arm starts
warmer, which biases against it by ≈ 1%). Telemetry during the sustained window: bs32 AICLK
1081–1125 MHz (six chips), power 160 W median with 176–185 W peaks, 58–64 °C; bs16 1170–1190 MHz;
bs8 1190–1200 MHz; cold iterations run at 1350 MHz.

**AICLK drops 21% at bs32 but the iteration slows only 4.4% (434 → 453)**, so only ≈ 21% of the
bs32 iteration scales with the core clock; the rest is bound by DRAM traffic, which the power
manager does not slow. Fewer active cores do not buy clock back: every "fewer cores" probe lost.

| batch / chip | probe | base sustained | probe sustained | Δ |
|---|---|---|---|---|
| bs32 / 10 | fused SwiGLU (`QWEN_FUSE_SWIGLU=1`, blk 4,8,8 / 1×4) | 469.3 | 482.3 | **+2.8%** |
| bs32 / 11 | FF1/FF3 N block 16 (`QWEN_MM_BLOCK_FF13=8,8,16`) | 447.4 | 457.2 | **+2.2%** |
| bs32 / 12 | fused heads op on 60 cores (`QWEN_HEADSPLIT_MAX_CORES=60`) | 455.1 | 505.7 | **+11%** |
| bs32 / 5 | SDPA 10×10 | 456.8 | 464.8 | +1.8% |
| bs32 / 1 | SDPA 8×8 | 448.1 | 455.5 | +1.7% |
| bs8 / 3 | fused heads op on 60 cores | 124.4 | 126.8 | +1.9% |
| bs16 / 9 | interleaved QKV/WO weights | 235.5 | 235.9 | 0 |
| bs16 / 2 | fused add+RMSNorm **off** (R=0) | 238.1 (on) | 243.1 (off) | the gate is worth **−2.1%** sustained |

Baselines with the shipped defaults, sustained: bs8 123.8–124.4, bs16 232.5–238.1, bs32 447–469
(chip spread ≈ 5% at bs32 under the power cap; same-chip pairs only).

What is left for the sustained number is DRAM traffic, ≈ 2.3 GB per layer at bs32 of which
≈ 1.2 GB is pure data movement (fused heads 214 MB, concat 142, SwiGLU product 510, add+norm 336):
1. the SwiGLU product computed in FF2's in0 path instead of FF1's epilogue (FF2 has one N block per
   core at N=2560, so the product would be computed exactly once per in0 block; removes the 170 MB
   write + read and the 1.37 ms op per layer at bs32 ≈ −30 ms) — proposed on #57627;
2. SDPA writing its output straight into the `[B, S, H·d]` tile layout (a tile-id remap in the
   writer) — removes the concat pass (142 MB/layer, ≈ 12 ms at bs32, 2.5 ms at bs8) — proposed on #57628;
3. the QKV `minimal_matmul` writing head-major `[B, H, S, d]` tiles (a tile-id remap in the writer)
   so the fused heads op only norms Q/K in place — removes ≈ 120 MB/layer (≈ 10 ms at bs32) — #57722.

## 49. POSITIVE — SDPA writes the concatenated-heads layout directly (`output_heads_concat`)

Sustained-regime item 2 of §48, implemented in the SDPA op (branch commit "sdpa: output_heads_concat"):
the writer lays head h's tiles at column tiles [h·vDHt, (h+1)·vDHt) of each row tile — a tile-id remap
(`TensorTileShape(B, 1, Sq, NQH·vDHt)`, row stride NQH·vDHt), no data reshuffle — so the concat pass
before the output projection disappears. Bit-identical to `nlp_concat_heads(sdpa(...))` at bs1/8/32
(unit test added). Standalone, traced: SDPA alone bs8 231.8 → 239.3 µs, bs32 863.7 → 920.9 (the strided
drain costs 3–7%), against the removed concat pass (~68 / ~350 µs). e2e, same chip:

| batch | cold A → B | sustained A → B (median it15–29) |
|---|---|---|
| 8 (chip 7 / 10) | 117.7 → **115.4** (−2.0%) | 126.7 → 126.8 (flat, 6 samples) |
| 16 (chip 8) | 226.2 → **221.9** (−1.9%) | — |
| 32 (chip 6 / 11) | 433.8 → **430.1** (−0.9%) | 445.8 → **440.3** (−1.2%) |

STS-B through the batch-8 path 0.8123, embeddings identical to the concat path (per-text cosine 1.0000).
Landed as the default for batch > 1 (`QWEN_SDPA_CONCAT_OUT=0` opts out); bs1 keeps its 4 µs model-local
concat because the base forward reshapes the bs1 SDPA output before concat. Remaining upside: a drain
that writes each head's tile run as one NoC transaction (recovers the 3–7% SDPA cost); items 1 (SwiGLU
epilogue) and 3 (head-major matmul output + SDPA head offsets) stay kernel asks (#57627, #57722).

## 50. Gio's three BGE-M3 items of 2026-09-24 (fused SDPA, new LayerNorm op, embeddings in L1) — sized here

The branch tip on GitHub (`gtobarTT/bge_m3_p150_optimizations`, 09-23 19:17) does not carry the three items;
what is pushed (L1 placements, B1 projection outputs in the LayerNorm shard layout, `no_padding` mask skip, SDPA
LoFi/streaming/k512) is §40/§42/§44. Sized against our 2026-09-24 profiles (chips 3/5 for the standalone runs):

| BGE-M3 item | here |
|---|---|
| SDPA writes the concatenated output and reads Q/K/V straight from the QKV projection | the first half is §49 (bs>1) and now bs1 too (below). The second half does not apply: between QKV and SDPA sits the fused head-split + Q/K RMSNorm + RoPE op (bs1 44 µs/layer = 1.6 ms, 9.5% of device time; bs8 177 µs = 6.4 ms; bs32 580 µs = 20.9 ms, 5.7%), which BGE-M3 does not have. A layout-only variant (the op rewrites Q/K in place in the QKV layout, SDPA reads V strided) saves at most V's pass, 1/6 of that op ≈ 3.5 ms at bs32, minus a strided-read penalty in SDPA of the size of the §49 drain (3–7% ≈ 1–2 ms) → ≤ 0.5%. Folding norm + RoPE into SDPA's reader is a kernel project, not a wiring change. |
| New LayerNorm op, bit-identical, 30% faster per call than stock (B8 38 vs 55 µs) | our `fused_add_rmsnorm_split` is already past that point. Stock `rms_norm` at [16384×2560] bfp8 (chip 5, traced): 299 µs = 298 GB/s (58% of 512); with `residual_input_tensor` 416 µs = 322 GB/s (63%). Ours in-model at bs32: 398 µs for four tensors (178 MB) = **448 GB/s (87%)**, i.e. +39% bandwidth over the stock fused-residual op and at the ceiling the best stock op reaches (bf16 add 449 GB/s = 88%). At bs8 ours is 126 µs = 354 GB/s (69%), bs16 223 µs = 400 GB/s (78%): the remaining headroom is ramp/fixed cost at small M, ≤ 1.8 ms at bs8 (1.6%), ≤ 1.7 ms at bs16 (0.9%), ≈ 0 at bs32. |
| Embedding outputs kept in L1 | bs1's embedding output is already in L1 (profile). At bs32 the embedding op (408 µs) and the first LayerNorm (402 µs) each move 84 MB; an L1 output saves one write + one read ≈ 0.33 ms = 0.08%. Not worth a knob. |

Achieved DRAM bandwidth of stock DM-bound ops at the bs32 shapes (chip 5, traced, DRAM interleaved; 512 GB/s peak):

| op | bfp8 [16384×2560] | bf16 [16384×2560] | bfp8 [16384×9728] | bf16 [16384×9728] |
|---|---|---|---|---|
| clone | 410 GB/s (80%) | 406 (79%) | 419 (82%) | 416 (81%) |
| add / mul | 402 / 400 (78%) | 436 / 435 (85%) | 406 / 405 (79%) | 449 / 447 (88%) |
| rms_norm / + residual | 298 / 322 (58 / 63%) | 372 / 394 (73 / 77%) | — | — |
| silu | — | — | 266 (52%) | 264 (51%) |

Stock `silu` is SFPU-bound at half the DRAM rate, so our `silu_mul` at bs32 (1362 µs = 373 GB/s, 73%) is
compute-bound as well; its DRAM floor is 1131 µs (≤ 8 ms at bs32 if the SFPU were free, which it is not) — the
fused SwiGLU epilogue (#57627) stays the fix. DRAM height-sharded tensors are rejected by this build's dataflow
buffer (`TT_THROW dataflow_buffer.cpp:2682`), so a big-page read layout could not be probed with stock ops.

**What did transfer — bs1, same chip (4), 10 iterations, best / median:**

| arm | best | median | note |
|---|---|---|---|
| baseline (×3) | 17.6 / 17.7 / 17.7 | 17.8 / 17.85 / 17.85 | |
| SDPA `output_heads_concat` at bs1 (`QWEN_SDPA_CONCAT_OUT_BS1=1`) | 17.5 | 17.6 | standalone SDPA 57.3 → 54.1 µs *and* the 4.6 µs model-local concat op is gone; bit-identical (unit test) |
| residual adds write the norm's 10×8 block-shard layout (`QWEN_BS1_RESID_SHARDED=1`) | 17.6 | 17.8 | the 72 I2S ops (2.4 µs + 0.6 µs gap each) become no-ops, but the 80-core sharded-output add gives most of it back: **neutral alone** |
| both | **17.5** | **17.5** | −0.3 ms on the median (−1.7%); STS-B 0.8161 unchanged |

*Correction (09-24, later):* only the even layers' I2S became no-ops (36 of 72): the wrapper's `supported(x, x)` gate
rejected the sharded residual the previous layer handed over, so odd layers ran the stock adds. That is part of why the item
measured neutral alone. Fixed to cover every layer (I2S 36 → 1, bs1 −0.07 ms e2e); see POSITIVE_RESULTS.

Standalone (chip 3, L1 bfp8 [512×2560]): add → I2S → sharded LN → S2I 28.9 µs as one traced chain; add with
block-sharded output 13.2 µs (vs 10.5 + 8.7), chain 25.3 µs; add with one sharded and one interleaved input
11.3 µs, bit-identical. The sharded LN cannot write an interleaved output (`TT_FATAL` in its validation), so the
S2I before each matmul stays (the 12×8 matmul grid cannot take the 10×8 shard either).
*Correction (09-25):* it can. With a block-sharded in0 the 2D multicast factory sizes its in0 senders from the
shard grid's width, not the compute grid's (`matmul_multicore_reuse_mcast_2d_program_factory.cpp`), so the 10 shard
columns multicast their K slices across all 12 compute columns; the validation only needs shard height = per_core_M,
in0_block_w | shard width (8, not the model's 10), ROW_MAJOR and fuse_batch. Standalone QKV 51.7 → 49.6 µs and
FF1+FF3 152.7 → 148.1 µs against S2I + interleaved (`perf_tools/bench_bs1_norm_shard_mm.py`); landed as
`QWEN_BS1_NORM_SHARDED_OUT=1` (POSITIVE_RESULTS). The claim above was never tested. Both landed as bs1
defaults; the neutral-alone item is kept because it is free with the concat change and removes 72 ops.

**Follow-up, same day — the residual item had only reached every other layer.** `_wrap_layer` ran the fused
add+norm kernels' `supported(x, x)` check before the bs1 branch; a block-sharded input fails it, so layers 1, 3,
5, … took the stock path and re-interleaved the residual (the re-profile still showed 36 I2S ops: 18 layers × 2).
That is why "alone" read neutral. With the bs1 branch evaluated first, all 72 I2S are gone: same chip, 10
iterations, best 17.3 / 17.4 (two runs), median 17.5, against 17.5 / 17.7 with the concat item alone;
30-iteration run cold **17.3**, sustained 17.7; STS-B 0.8161. Lesson: when a wrapper chain has a capability
check, an op that changes the residual's layout must be routed before it, and a per-op profile (op counts per
layer) is the quickest way to see a change that lands on only half the layers.

## 51. bs1 fused heads op (head split + Q/K RMSNorm + RoPE): what bounds it, and what the fixes ran into (2026-09-24)

8× p150b host, chip 0; standalone numbers are device kernel time of the op at the bs1 shapes (64 cores, 2 units per
core, a unit = 4 Q + 1 K head normalised + 1 V head copied), median of 12 calls.

**Ablation (scratch kernels, not in the repo).** Full op 39.5 µs; compute only (reader/writer skip the unit tiles)
36.6; data movement only (compute passes the CBs through) 11.3; handshakes only 8.5, of which 7.5 is the gamma read
(64 cores reading the same 8 tiles; 1.0 without it). So the op is compute-bound: the unit traffic overlaps except for
the first unit in and the last out (~3 µs). RoPE is 8.4 µs of compute (36.6 vs 28.2 norm-only). Compute also waits
~5 µs for gamma at its first head (compute-only 36.5 → 31.1 without the gamma read).

**Gamma in L1 interleaved instead of DRAM: nothing in the model.** Compute-only standalone 36.5 → 33.0, but the full
op 41.5 → 40.9 standalone and 41.5 → 41.3 in-model: in the full op the gamma read competes with every core's unit
reads, and where gamma lives does not change that. Per-core DRAM copies of gamma: 36.5 → 34.2 compute-only, not
pursued. Reverted.

**Compute v3 and the kernel-config buffer.** Batching the phases across heads first ran each phase over the Q heads
and then the K head (two template instantiations): standalone 41.8 → 30.6 µs, bit-identical once every phase moved
the CBs' full capacity (the first version sized the CBs for 4 heads, used 4 + 1 per unit, and the second unit's
indexed accesses ran past the CB ends: Q PCC 0.79, K inf; it only showed in non-resident mode, where the static-CB
neighbours differ). In the model it bought **nothing** (e2e 16.510 vs 16.511 ms): the compute binary grew 24.0 →
34.9 KB, and with SDPA's ~39 KB next the two programs no longer fit Blackhole's 69 KB per-core kernel-config buffer
(`bh_hal_tensix.cpp`), so the dispatcher could not stage SDPA while the heads op ran: +3 µs gap before the op and
+2 after it per layer ate the 7 µs saved. Runtime head counts and CB ids made the binary bigger (36.0 KB: the
inlined LLK calls no longer constant-fold); `#pragma GCC optimize("Os")` made it 11.9 KB but the op slower than v1
(44.6 µs). Running the unit's Q and K heads as one chunk (one instantiation, compile-time CB ids) gave 21.6 KB and
29.6 µs standalone; in-model 28.7 µs, the gap before the op back to 0.34 µs (after it: 2.1 µs, still open).
Lesson: on Blackhole a model-local kernel's binary size is part of its cost; check `TENSIX COMPUTE n MAX KERNEL
SIZE` and the op-to-op latency around the op, not only its kernel time.

## 52. bs1 SDPA on more than 64 cores: the unit count decides, and most of the op is fixed cost (2026-09-25)

8× p150b host, chip 0; standalone `perf_tools/bench_sdpa_bs1_wide.py` (traced, Q/K/V bfp8 in L1, `pack_gqa_heads`,
LoFi, exp approx, output unconcatenated so chunks may cross Q heads; each config checked against the 8×8 q256/k512
output).

**Why 120 cores does not come for free.** Packed, Q is 8 heads × 64 row tiles = 512 row tiles. The op gives each core a
contiguous run of Q chunks and one K/V chain per head, and a core's time is set by its row tiles plus a per-chunk fixed
cost. q256 on 8×8 is 64 chunks, one per core, 8 row tiles each, each grid row one head (row multicast of K/V).
Candidates, standalone µs:

| grid, q/k chunk | chunks | row tiles per busiest core | µs |
|---|--:|--:|--:|
| 8×8 q256/k512 (09-24 default) | 64 | 8 | 40.2 |
| **11×8 q192/k512 (shipped)** | 88 (11 per head, last one 4 tiles) | 6 | **36.6** (33.6 on a second run) |
| 11×8 q192/k256 | 88 | 6 | 37.3 |
| 12×9 q160/k512 | 104 (13 per head) | 5 | 44.0 |
| 12×10 q128/k512 | 128 | 8 (8 cores do 2 chunks) | 53.4 |
| 12×10 q64/k256 | 256 | 6 (3 chunks) | 55.4 |

- q128 on 120 cores cannot help: 128 chunks leave 8 cores with two, so the busiest core still has 8 row tiles.
- q160 gives 5 row tiles per core but a head's 13 chunks span two grid rows, so the chain multicast (all-or-nothing:
  same physical row, no gaps, uniform q counts) turns off for every head and K/V hops core to core down 13-core chains.
- q64 re-streams K/V and pays the per-chunk cost three times per core.

**Most of the op is fixed cost.** Fitting time = fixed + per-row × rows to q256 (8 rows, 40.2 µs) and q192 (6 rows,
36.6 µs) gives ~26 µs fixed and ~1.8 µs per row tile: cutting rows by 25% bought 9% (16% on the second run). With
one Q chunk and one K chunk per core nothing overlaps the K arrival (injector read of ~70 KB from L1-interleaved
banks + multicast), the Q read, the V arrival and the output drain; only the subblock streaming inside the chunk does.
The kernel already overlaps exp (SFPU, pack thread) with the matmuls (FPU, math thread), so the 64-core bound is
~max(12 µs FPU, ~21 µs SFPU), not their sum. The remaining headroom is in that fixed part (a device-profiler zone
split is the next measurement), not in the core count.

**Shipping q192 needed a writer change.** A 6-tile chunk crosses the 16-tile Q heads of its group, and the
concatenated `[1, 1, S, NQH·d]` output assumed it never did (host check `q_chunk | Sq`). `write_block_row_grouped` /
`write_block` take the head length and the chunk's first row in its head and move a wrapped row back one head length
and right one head (`head_wrap_tile_offset`); the check is gone. Without the concat output the model needs the 4.6 µs
concat op again, which cancels the gain.
