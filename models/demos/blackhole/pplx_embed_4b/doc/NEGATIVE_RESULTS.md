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

## 53. Heads-op compute v3 at bs8/16/32: bit-identical, but slower than v1 (2026-09-25)

`QWEN_FUSED_COMPUTE_V3=1` at the batched sizes (heads op on 114 cores, DRAM activations), 8× p150b host, same-chip
alternating A/B, 3 pairs × 20 iterations. STS-B through the batched path is unchanged (0.8123 / 0.8140 / 0.8159 at
batch 8 / 16 / 32), so the op stays bit-identical, but every pair was slower on best-of-run:

| batch | v1 best (ms) | v3 best (ms) | Δ |
|---|---|---|--:|
| 8 (chip 0) | 108.2 / 108.1 / 108.2 | 108.6 / 108.7 / 108.7 | +0.5 (+0.4%) |
| 16 (chip 1) | 202.7 / 202.8 / 202.8 | 205.0 / 205.0 / 204.7 | +2.2 (+1.1%) |
| 32 (chip 2) | 392.8 / 397.4 / 398.7 | 398.7 / 402.1 / 402.0 | +4.6 (+1.2%) |

Not profiled. At bs1 the per-phase set-up it removes is a large share of a 64-core, L1-resident op; at the batched
sizes the op streams from DRAM and that share is small. The kernel is now `compute_qkv_heads_norm_bs1.cpp` and stays
the bs1 default only. Resident heads-op constants (`QWEN_FUSED_RESIDENT_CONSTS=1`) also stay bs1-only: at bs8 and bs16
the per-core shards clash with the heads op's static CBs (`TT_THROW: Statically allocated circular buffers ... clash
with L1 buffers`) on the first warm-up forward.

## 54. bs16 fused add+RMSNorm: what bounds it, and why L1 interleaved beat a resident-sharded rewrite (2026-09-25)

8× p150b host; standalone numbers are traced wall time per call at the bs16 shape ([8192 × 2560] bfp8, R = 5,
120 cores, 11 waves), median of 5.

**Ablation (scratch kernels with compile-time skips, not in the repo).** Full op 241 µs; data movement only (no
compute) 231; read + write only (no compute, no partial exchange) 231; compute + exchange only (no DRAM traffic) 97;
read only 140, write only 169 (same bytes: writes are the slower direction, 265 vs 319 GB/s); handshakes only 25.
Double-buffering the streaming CBs and dropping the per-wave write barrier: 240-242 (no change). Dropping the partial
exchange made it slower (263): likely the exchange keeps each 5-core row group in lock step, which DRAM prefers
(not verified). Reference streaming ops on the same tensors: `ttnn.add` 390 GB/s, `ttnn.clone` 393 GB/s; the op's data
movement runs at 386 GB/s and the full op at 370. So the op is DRAM-bound at ~95% of what streaming ops reach here;
compute (97 µs) hides under the traffic. Only fewer bytes help.

**L1 placement.** b (WO / FF2 outputs) and the norm output are short-lived, so unlike the residual stream (§2) they
never share L1 with SDPA's CBs, and they fit at bs16 (186 KB/core) and bs32 (372 KB/core). With both in L1 interleaved
the op fell to 365.9 µs/call at bs32 (434.9 before), but end to end the gain vanished: the stock decoder moves the
attention output to the residual's DRAM config before the post-attention add (`decoder.py`, a workaround for the stock
`ttnn.add`), which became 36 L1 → DRAM copies (7.2 ms at bs32). The fused path reads b from anywhere, so the move is
skipped there: 36 copies → 0, op 255.0 µs/call, bs32 replay 370.5 → 357.4 ms. `minimal_matmul` addresses every in0 /
in1 / output tile through `TensorAccessor`, so it takes L1 (interleaved or sharded) inputs and outputs unchanged.

**Resident-sharded rewrite: correct, no faster.** The op's schedule (unit u = row · R + k on core u mod 120, wave
u div 120) is round-robin ND sharding with one unit per shard, so b and the output can be resident in that layout,
read and packed in place through CBs aliased to the core's shards (`cb_descriptor_from_sharded_tensor` works on ND
tensors). Built and verified bit-identical in every placement combination (and in STS-B), but it needed a ttnn view
relaxation for ND-sharded row regroups ([1, 8, 1024, W] ↔ [1, 1, 8192, W]), and QKV reading the unit-sharded in0 was
5-9% slower at its best block config (bs16 570 → 601 µs, bs32 1084 → 1165) while WO / FF2 / FF1+FF3 were neutral.
Against L1 interleaved, same chip, sustained_run.sh at bs16: cold 195.4 vs 196.6, sustained 224.7 vs 225.4 ms. Both
remove the same DRAM traffic; residency only adds the NoC reads / writes of b and the output, which the DRAM-bound op
did not miss. Dropped (tools kept: `perf_tools/bench_batched_mm_sharded_io.py`, `sweep_qkv_sharded_in0.py`; the sweep
also found bs32 QKV with in0 in DRAM ~5% faster at M16 K16 N4 sb1×4 than the shipped 8/8/8, not yet A/B'd in-model).

**Residual sums.** The post-attention sum (add 2's a) lives across the MLP only and fits at bs8 / bs16 (landed); at bs32
it is 72 KB/core short, also with FF2's output back in DRAM. The post-MLP sum is the next layer's input: the decoder
asserts it sits in the residual's config, and it would have to share L1 with SDPA (~66 KB/core free, §40), so it stays
in DRAM. One resident-mode bs16 run hung in warm-up (killed after 30 min) while a device profile ran on another chip;
it did not recur in later runs. Profiles taken while other jobs ran on the host came out broken (device-only report,
or "End marker found without a corresponding start marker"); profile with the host otherwise idle.

## 55. bs16 batched SDPA: contention, not compute; packing does not help; K/V reuse does (2026-09-25)

8× p150b host, chip 0; standalone traced time per call at the bs16 config (Q [16, 32, 512, 128], K/V [16, 8, 512, 128]
bfp8 in DRAM, non-causal, 12×10, q512 / k512, LoFi, streaming kernel, heads-concat output), `perf_tools/bench_sdpa_bs16_ablate.py`.

**Roofline miss.** In-model 485 µs (511-515 standalone) against a ~230 µs estimate (DRAM 89 MB at ~390 GB/s, softmax exp
on the SFPU ~178 µs, matmuls ~104 µs). The estimate assumed K/V read once per KV head; at q512 each Q head is one unit on
one core, the K/V chains never form, and the 4 Q heads sharing a KV head each read it: 71 MB of K/V (17.8 unique), 142 MB
per call.

**Device-profiler kernel spans** (one call; zones enabled by flipping `call_step`'s profiling tag in
`compute_streaming.hpp`, reverted): compute and reader average 346 µs per core, slowest core 482 µs; writer 278 / 379. The
op ends with its slowest core, and 512 units on 120 cores leave 32 cores with 5 units (balanced: ~412 µs). The zone
detail was truncated after ~7 steps per core (profiler buffer); the recorded part has softmax (subtract max + exp) as the
largest compute zone per step, then Q@Kᵀ. Profile a run with 1-2 units per core for complete zones.

**Core count barely matters.** 12×10 511, 8×10 512, 8×8 530, 12×8 473 µs (fewer cores faster): the time per unit grows
with the number of active cores (66 µs at 64 cores, 79 at 96, ~102 at 120), i.e. contention on DRAM, not compute (32
cores: ~60 µs per unit). `exp_approx_mode` has no effect (the streaming kernel always uses the approximate exp). Finer Q
chunks made it worse (q256 697, q128 954 µs): each chunk re-reads K/V.

**GQA packing (`pack_gqa_heads`, op supports B > 1) did not help:** 12×10 packed 647 µs (units cross packed-head
boundaries, chains mix q counts, multicast turns off, unicast forwarding serializes), 8×8 packed = unpacked (530).

**K/V reuse** (landed as SDPA `reuse_kv`, POSITIVE_RESULTS): the reader skips K/V for a unit with the same (batch, KV
head) as the previous one on its core, compute keeps them when the next unit shares them, and no chains are built.
Bit-identical to reuse off. 12×10 q512 512 → 425 µs; since finer chunks no longer re-read K/V, q256 382, **q128 355**,
q96 364, q64 402; 11×10 / 12×9 / 12×8 at q128 373 / 370 / 390; packed + reuse q128 359. bs8: 247.5 → 233.9 (12×8 q128;
less DRAM-bound), bs32: 911.6 → 620.5 (12×10 q128). In the model the cold gain matches the standalone one (bs16 −5.6 ms,
bs32 −9.8 ms over 36 calls) but the sustained gain is a third of it: less waiting on DRAM means more power per
iteration, and the power manager settles the clock 10-15 MHz lower.

## 56. bs16 fused heads op: compute-bound; the QKV output does not fit in L1 next to the QKV matmul (2026-09-28)

8× p150b host. Standalone traced time per call of the fused heads op (head split + Q/K RMSNorm + RoPE, v1 compute) at
the bs16 config (QKV [16, 1, 512, 6144] bfp8, bfp8 Q/K/V out in DRAM), `perf_tools/bench_heads_bs16_ablate.py` (scratch
kernel variants that skip the unit reads, the unit writes, or replace the compute with a tile copy). cos / sin / the
rotation tile in L1 as in the model (`QWEN_ROPE_PREFILL_L1=1`); `ABL_ROPE=DRAM` gives the second column:

| variant | DRAM in | L1 in | DRAM in, cos/sin in DRAM |
|---|---|---|---|
| full | 327.3 µs | 304.5 µs | 435.3 µs |
| compute only (no unit read / write) | 301.6 | | 309.8 |
| data movement only (copy compute) | 272.0 | 224.8 | 350.4 |
| read only | 183.4 | 232.5 | 250.9 |
| write only | 200.9 | | 267.1 |
| handshakes only (copy compute, no read / write) | 63.8 | | 112.8 |

At the model's placement the op is compute-bound (full 327 against compute-only 302); an L1 input saves only 23 µs. A
first pass of this bench read cos / sin from DRAM and concluded the op was bound by the DRAM input read (435 → 320 with
the input in L1): the cos / sin reads were what the L1 input relieved (entry 25's lesson, again).

**The v1 compute is per-phase overhead.** Per-phase wall-clock split of the v1 compute (`perf_tools/bench_heads_bs16_phases.py`,
per-TRISC accumulators, 114 cores, 18 units per core; a unit is one 32-row tile of one KV group: 4 Q + 1 K heads
normalised, V copied): ~412k cycles per core, ~4.5k per normalised head over 9 phases of 4 tiles each (~500 cycles per
phase). The math thread spends a third of its time in eps + rsqrt (one tile per head: fp32, non-approximate, all four
faces although only column 0 is used), and the rest is spread evenly over the phases. The compute kernels side by side
(`perf_tools/bench_heads_bs16_kernels.py`, cos / sin in L1, PCC against v1):

| compute | L1 in | DRAM in | vs v1 |
|---|---|---|---|
| v1 (batched default) | 304.7 µs | 327.6 µs | |
| v2 (dest-reuse, 6 passes per head, entry 25) | 298.5 | 324.1 | PCC Q 0.9988, K 0.9997 |
| v3 (each phase once per unit, over its 5 heads; bs1 default) | 214.2 | 372.3 | bit-identical |

Fewer passes (v2) do not help; paying each phase's reconfigure / init / CB handshakes once per unit instead of once per
head (v3) cuts the compute by 30%, but only when the input read does not compete with it (DRAM in: slower than v1,
entry 53).

**In the model the QKV output does not fit in L1 at bs16.** `TT_PREFILL_QKV_L1=1` at bs16 puts the 53.5 MB QKV output
in L1 (446 KB per core on 12×10) while the norm output is also L1 resident; the QKV `minimal_matmul`'s static CBs at the
default 8,8,8 blocks end at 595072, and the output lands at 517184: `Statically allocated dataflow buffers ... clash
with L1 buffers` (78 KB per core short). Smaller blocks make room, but the matmul slows down more than the heads op speeds
up (e2e best of 10, `ab_one.sh`, same chip): 8,8,8 DRAM 189.2 → 8,4,8 L1 189.0 ms (sustained iteration 9 223.4 →
225.7); 8,8,8 DRAM 189.5 → 4,8,8 L1 197.8; 4,4,8 DRAM 202.0 → 4,4,8 L1 200.3. bs8 fits (27 MB, 223 KB per core) and gains: QKV output in L1,
default on at bs8 (POSITIVE_RESULTS). At bs16 the clash turned out to be fragmentation (below); with the post-attention
sum in DRAM the QKV output fits at the default blocks and it lands at bs16 too.

**The QKV matmul is compute-bound, and its output write overlaps.** `perf_tools/bench_qkv_mm_bs16_ablate.py [batch]`
(QKV minimal_matmul at the model's config, from `perf_tools/capture_qkv_call.py`: bfp8 in0 in L1, **bfp4 weights DRAM
width-sharded [2560, 768] over the 8 banks**, bfp8 out, 12×10, 8,8,8 blocks, subblock 1,8, LoFi, packer L1 acc;
variants patch `matmul_dataflow_common_metal2.hpp`, the header the op's Metal 2.0 kernels include; output PCC 1.000 for
full and ~0 for every skip variant), µs per call:

| variant | bs16, L1 out | bs16, DRAM out | bs32, DRAM out |
|---|---|---|---|
| full | 559.5 | 559.3 | 1123.9 |
| no output write | 552.0 | 552.1 | 1123.0 |
| no in0 read | 531.9 | 542.6 | 1086.7 |
| no in1 read | 556.3 | 563.5 | 1089.9 |
| no reads | 520.5 | 529.5 | 1041.1 |
| compute only | 504.9 | 504.8 | 972.9 |

Compute is 87-90% of the call and the output write is hidden (−1 to −7 µs). A first pass of this bench used bf8
interleaved weights and measured 667 µs with a 110-126 µs output-write stall; that stall was an artifact of the wrong
weights (the commit message of the bs8 landing repeats it). Fused QKV + heads estimate from these numbers: the fused
op costs the matmul plus the heads epilogue on the same cores, and the epilogue at v3's cost on the matmul's grid (each
core owns 22 / 43 row tiles × 5 heads at bs16 / bs32; the V heads are copies, so rows y=8-9 carry no norm work unless
the weight columns are permuted to 4 norm + 1 V head per core) is ~205-257 µs at bs16 and ~372-465 µs at bs32. bs16
today (QKV in L1, v3) is 559 + 214 = 773 µs, fused 764-816: nothing to gain. bs32 today (DRAM, v1) is 1124 + 594 =
1718 µs, fused 1496-1589: −130 to −220 µs per layer (≈5-8 ms). bs32 heads op (`bench_heads_bs16_ablate.py 32`): v1
full 594.5, compute only 572.0, data movement only 530.4; v3 compute only 388.6.

**Why v3 loses with a DRAM input: the op sits at the DRAM floor, and the saturated bandwidth reaches the top grid rows
last.** Per-unit split (`bench_heads_bs16_phases.py`, `PH_MODE=unit`): with a DRAM input, v3's mean core is faster than
v1's (347k vs 418k cycles), but the cores of grid rows 0-2 wait on their reader (unpack wait-for-input 183k cycles on
the slowest, v1 ~20k) and end the op at 503k. Mean cycles by grid row, v3: 449 478 412 332 320 303 296 290 288 280k (v1
the same gradient, 436 → 398k, hidden by its slower compute; L1 input: flat). It follows the core's position, not its
data or its NoC: the same with the unit ranges handed out in reverse core order, with the reader on NoC1 and the writer
on NoC0 (and everything much slower: v1 L1-input 305 → 500 µs), and with cos / sin in DRAM. Unit counts weighted by
row (all 120 cores, rows 0-1 12 units, rows 7-9 20) even out the finish: v3 DRAM-input 372 → 327 µs, but that only
matches v1 (328). The op moves 107 MB per call (read + write) and its data movement alone takes 272 µs (~390 GB/s), so
with a DRAM input v1 is within ~55 µs of the floor and a faster compute cannot go below it. The experiment knobs (NoC
swap, reversed units, row weights) were removed; v3 follows the QKV-output-in-L1 knob (bs8 default).

**L1 at the bs16 QKV call: the room is there, fragmented.** `perf_tools/l1_map_first_layer.py 16` (live L1 buffers
after each op of one eager prefill; CB region from 111,744 B, 1,461,120 B limit): when the QKV matmul runs, the only
large L1 tensor is the norm output (181 KiB per core), but it sits at 963,264 B, low, because it was allocated while the
FF2 output and the post-attention sum were still live above it. The QKV matmul's CBs end at 595,072 B (472 KiB), so
~360 KiB per core is free below the norm output and ~364 KiB above it. A fused QKV + heads op fits: the heads op's own
CBs other than its input and output are ~90 KB (v1 layout) or ~230 KB (v3, a unit's heads), under the 360 KiB at the
default 8,8,8 blocks. The unfused QKV output in L1 (446 KiB) fits in neither gap alone although ~723 KiB are free:
`TT_PREFILL_QKV_L1=1`'s clash at bs16 is fragmentation, not capacity. With the post-attention sum in DRAM
(`QWEN_FUSED_ADD_NORM_SUM1_L1=0`) the norm output is allocated high and the QKV output fits below it at 8,8,8; with the
v3 heads compute, sustained_run.sh, 3 alternating rounds, chip 0: cold / sustained 189.2 / 222.0, 189.3 / 223.4, 189.4 /
223.3 → 186.0 / 219.3, 186.1 / 221.1, 186.0 / 220.6 ms (−1.7 / −1.2%): the QKV output's L1 gain outweighs the sum's.
Landed as the bs16 default (POSITIVE_RESULTS).

**bs32 in two half-batch chunks (landed, POSITIVE_RESULTS).** Splitting N instead (two `[2560, 3072]` weights, half the
KV groups each, no input split) makes each chunk slower than half the call: 676.5 µs per chunk at M=16384 (N=96 tiles
over 10 cores: N blocks of 8 + 2) against 1155.7 for the full call, +197 µs per layer. Splitting M keeps each chunk the
bs16 matmul (561.4 µs). At bs32 the norm output (363 KiB per core) sat at 777,216 B, low for the same reason as at bs16
(allocated while FF2's output, in L1, held the top slot), leaving ~182 KiB below it; moving FF2's output to DRAM would
fix the layout but costs +5.1 ms cold (366.1 → 371.2). Preallocating the norm output before FF2
(`QWEN_FUSED_ADD_NORM_PREALLOC=1`) puts it at the top (lowest buffer at the QKV call 1,149,312 B) with FF2's output in L1, neutral on its own
(367.1 vs 367.4 ms); at bs16 it lets the post-attention sum back into L1 beside the QKV output (−0.8 / −1.1 ms cold /
sustained). Chunked bs32: 369.1 / 433.3 → 362.2 / 430.4 ms cold / sustained, STS-B 0.8146.

## 57. Heads-op rsqrt over column 0's faces only: 16% less compute, slower op (2026-09-28)

After the row reduce only column 0 of the mean-square tile holds values, and the `bcast_cols` multiply that consumes
`rsqrt(ms + eps)` reads column 0 only, but `rsqrt_tile` is hard-coded to `VectorMode::RC` (all four faces, fp32,
non-approximate). The same SFPU call with `VectorMode::C` (faces 0 and 2) is bit-identical with half the rsqrt work;
v3 has it behind compile-time arg 9 (`QWEN_FUSED_RSQRT_COL=1`, default off).

Per-core compute drops as expected: v1 401.6k → 356.4k cycles per core (the math thread's eps+rsqrt phase 7681 →
5077 cycles per unit, nothing else moves; `bench_heads_bs16_phases.py`, `PH_KERNEL=`), v3 287.6k → 241.8k (−16%).
Standalone wall time, bs16 shapes, L1 input (`bench_heads_bs16_kernels.py 16 v1 v3 v3:<kernel>`, an unmodified copy of
v3 through the same override reproduces v3 exactly): v1 304.2 → 271.2 µs, but **v3 215.0 → 256.5 µs** (bs32 393.8 →
541.3); with a DRAM input v3 gets faster (372 → 335). v3's compute (~179 µs) now finishes well under the op's data
movement with an L1 input (~225 µs, dominated by the 53.5 MB of Q/K/V tile writes to DRAM), and the op lands above
that floor, as v3 did against a DRAM input (§56): once compute outruns the data movement, the traffic bunches and the
top grid rows fall behind. The binaries are not the cause (v3 math 6.2 → 7.2 KB; v1, larger still, gains). e2e,
`ab_one.sh`, one chip per batch: bs1 15.5 → 15.7 ms, bs8 98.5 → 98.2, bs16 185.1 → 185.8, bs32 359.5 → 362.6 (iteration
9 at bs32 434.5 → 441.8). The v3 heads op is bound by its output writes now, not its compute; further compute cuts
need the Q/K/V write side fixed first (their DRAM traffic, or keeping them in L1 for SDPA).

## 58. FF1 + FF3: the fused-SwiGLU epilogue was the cost; applied on the pack thread in the last K block (landed) (2026-09-28)

Configs from `perf_tools/capture_qkv_call.py` (`CAP_N=9728,19456`): bs8 / bs16 run the fused-SwiGLU `minimal_matmul`
on the packed gate/up weight ([2560, 19456] bfp4, DRAM interleaved; 8,8,8 1×8 / 4,20,8 1×4), bs32 ran FF1 and FF3
unfused (bfp4 width-sharded, 8,8,8 1×8) plus `silu_mul`; in0 bfp8 in L1, out bfp8 in DRAM, LoFi. `perf_tools/
bench_mm_ablate.py ff13 <batch>` (skip reads / writes / both, as for QKV in §56), µs per call:

| | full | compute only | TFLOP/s full / compute |
|---|---|---|---|
| bs8 fused | 1433.0 | 1404.7 | 285 / 291 |
| bs16 fused | 2704.9 | 2651.1 | 302 / 308 |
| bs32 FF1 | 1670.0 | 1478.9 | 489 / 552 |

bs32's plain FF1 is efficient with an L1 activation (tenstorrent/tt-metal#57626's 332 TFLOP/s was measured with
DRAM activations at M=4096: stale). The fused kernel is compute-bound at ~300 TFLOP/s; the same packed shape as a plain
matmul (`ff13plain`) computes in 745.2 / 1477.3 µs (bs8 / bs16, 548-552 TFLOP/s; data-movement-bound in full, 1071 /
2202, because it writes the 2× wider output). bs32 device profile (tracy, one replay, 340.5 ms of kernel time): FF1 +
FF3 119.5 ms (35%), `silu_mul` 49.1 ms (14%, 1364 µs per call at the DRAM floor), FF2 + WO 77.3, QKV 39.2, SDPA 21.0.

**Where the epilogue went** (`perf_tools/bench_swiglu_epilogue.py`, variants of `swiglu_block` patched in place,
bs16 / bs8 µs): base 2709 / 1435, copy + pack only 1613 / 788, SiLU only 2287 / 1199, multiply only 1964 / 1010, inits
hoisted (one pair per DST session kept) 2676 / 1417. So the SFPU SiLU (~675 µs at bs16) and the SFPU multiply (~350)
are the epilogue, the per-tile inits are not (§35's batching was right to find the packer, not the inits, the limit).
A single SFPU pass computing silu(gate) · up (`moe_compute` / `moe_gpt`'s `swiglu_sfpu.h` pattern, `_sfpu_sigmoid_`
and the bf16 roundings of silu_tile + mul_binary_tile) is 2307 / 1222 on the pack thread (2348 / 1231 on the math
thread; moe's bf16 exp + one reciprocal step 2292 / 1212, less precise); PCC vs base 1.00000.

**In the K loop** (landed in `compute_metal2.cpp`, `matmul_blocks_swiglu`): on each output block's last K block the math
thread accumulates the subblock from zero, adds its partial sums of K blocks 0..K-2 with one dest-reuse add (same
rounding order as the packer's L1 accumulation), and the pack thread applies the single-pass SwiGLU in its DST half and
packs straight into the half-width output, overlapping the math thread's next subblock; no epilogue pass re-reads the
intermediate. µs per call and PCC vs an fp32 torch SwiGLU of the same operands (first 64 rows): bs8 1435.0 → 1049.8
(0.98697 → 0.98697), bs16 2704.8 → 1958.7 (0.98690 → 0.98691), bs32 fused 5324.8 → 4128.7 at 8,8,8 1×8, 5165.5 →
3830.5 at 4,20,8 1×4. Reloading the partials into DST first and accumulating on top was as fast but less accurate
(PCC vs base 0.99989 / 0.99974, vs torch 0.98692 / 0.98670: the large partial swamps the bf16 DST accumulation).
ttnn nightly `test_minimal_matmul.py -k swiglu` (4, incl. the bias path, which keeps the epilogue) and
`test_minimal_matmul_split.py -k swiglu` (6) pass.

At bs32 the fused kernel (3831 µs per layer) now beats FF1 + FF3 + `silu_mul` (1670 × 2 + 1364 = 4704), and it is the
bs32 default (§13's "structurally slower" was the epilogue). e2e, sustained_run.sh, 3 alternating rounds per batch
(committed vs new kernel, chips 0 / 1 / 2 concurrently): bs8 98.6 / 113.7 → 85.0 / 103.7 ms (cold / sustained, −13.8 /
−8.8%), bs16 185.1 / 225.7 → 158.3 / 210.4 (−14.5 / −6.8%), bs32 364.1 / 436.3 → 329.4 / 411.0 (−9.5 / −5.8%, fused);
the settled clock drops 40-90 MHz (more power per iteration with less waiting). STS-B 0.8116 / 0.8150 / 0.8135 at
batch 8 / 16 / 32 (0.8114 / 0.8144 / 0.8146 before; bs32 changes path). The FF13 block knobs are shared with the
unfused FF1 / FF3: bs32's 4,20,8 1×4 applies only when fused (the unfused path at those blocks: 410.2 ms).

**Blocks re-swept for the new kernel** (`perf_tools/bench_ff13_sweep.py <batch>`, 255 configs per batch: M 4 / 8 / 16,
K 5-20, N 4 / 8 / 16, subblocks 1×2 … 4×2; the failures are L1 clashes): K_block 20 with a 1×8 subblock wins at every
batch; the fused kernel takes 1×8 (the old "capped at 1×4" note no longer holds). bs8 8,8,8 1×8 1058.4 → 8,20,8 1×8
996.0 µs, bs16 4,20,8 1×4 1964.8 → 1×8 1927.4, bs32 4,20,8 1×4 3848.9 → 1×8 3763.7. e2e, 3 alternating rounds:
cold / sustained bs8 85.0 / 104.1 → 82.5 / 101.4 ms, bs16 158.3 / 210.7 → 157.1 / 209.5, bs32 329.3 / 411.1 →
326.4 / 408.6; STS-B bs8 0.8114. Landed, applied only with the fused kernel (the block knobs also drive the unfused
FF1 / FF3). bs1 (`QWEN_FUSE_SWIGLU_BS1=1`, M=512, 110 configs): best 2,20,8 1×2 214.4 µs (the probe's old 2,8,8 1×4
259.2), but the legacy FF1 + FF3 + mul it replaces is cheaper e2e: 15.6 → 17.6 ms at the old config, 15.7 → 16.0 at
the best. bs1 stays unfused; the probe's defaults now point at the best config.

## 59. FF2 / WO: compute-bound at ~80% of the LoFi roofline; bs32's M_block-16 win is lost to the power cap (2026-09-28)

Model calls (`capture_qkv_call.py`, `CAP_N=2560`): FF2 in0 [M, 9728] and WO in0 [M, 4096] bfp8 in **DRAM**, bfp4 weights
DRAM width-sharded ([K, 320] per bank), bfp8 out in L1, LoFi; blocks bs8 16,8,8 1×8, bs16 / bs32 8,8,8 1×8.
`perf_tools/bench_mm_ablate.py ff2|wo <batch>` (the presets now read in0 from DRAM, `MM_IN0=` overrides), µs:

| | full | no output write | no in0 read | no in1 read | compute only | TFLOP/s full / compute |
|---|---|---|---|---|---|---|
| FF2 bs8 | 372.6 | 372.8 | 372.0 | 370.2 | 366.6 | 548 / 557 |
| FF2 bs16 | 734.6 | 730.5 | 733.0 | 723.7 | 718.7 | 555 / 568 |
| FF2 bs32 | 1505.6 | 1501.8 | 1503.6 | 1401.1 | 1389.8 | 542 / 587 |
| WO bs8 | 173.6 | 172.9 | 172.7 | 170.5 | 167.0 | 495 / 514 |
| WO bs16 | 333.0 | 330.2 | 329.3 | 324.8 | 320.7 | 516 / 536 |
| WO bs32 | 660.8 | 655.3 | 660.0 | 623.1 | 610.0 | 520 / 563 |

Both are compute-bound (92-98%); the DRAM in0 read and the output write are hidden. The LoFi roofline is 4096 FLOP per
cycle per core (8×16 × 16×16 per cycle, `tech_reports/GEMM_FLOPS`): 663.6 TFLOP/s on 120 cores at 1.35 GHz, ~496 at
the ~1.01 GHz bs16 / bs32 settle; so FF2 / WO run at 79-88% of the FPU. At bs32 the weight read is exposed (−105 /
−38 µs when skipped): each core re-reads its in1 slice once per M block, and bs32 has 6 M blocks per core.

Block sweep (`perf_tools/bench_mm_sweep.py ff2|wo <batch>`, 126-168 configs): bs8 and bs16 keep their blocks (best
within 0.1-1%). bs32 M_block 16 halves the re-reads: FF2 1510.0 → 1417.1 µs (16,4,8 1×8), WO 666.7 → 631.9 (16,8,8).
In the model FF2's 16,4,8 CBs (~716 KB) clash with L1 (the preallocated norm halves and FF2's L1 output leave
~665 KB, lowest buffer 777,216 B); 16,8,4 1×4 fits (1459.8 µs standalone). With WO 16,8,8, sustained_run.sh, 3
alternating rounds, bs32 chip 0: cold 323.0 → 320.9 ms, **sustained 399.6 → 402.6** (settled clock ~1035 → ~1017
MHz). More work per watt-second is not faster under the power cap: kept the 8,8,8 blocks.

## 60. Fused FF1+FF3: the pack-thread SwiGLU was half exposed; K_block 40 hides it (landed) (2026-09-29)

With the SwiGLU on the pack thread (§58) the fused kernel ran at 413-434 TFLOP/s against ~550 for the same packed shape
as a plain matmul. `perf_tools/bench_ff13_fused_ablate.py <batch>` patches `compute_metal2.cpp` (skip or double the
pack-thread SwiGLU call, skip the last K block's dest-reuse add of the partial sums) and optionally the dataflow header
(skip reads / writes, as `bench_mm_ablate.py`), at the model's blocks (bs8 8,20,8 1×8, bs16 / bs32 4,20,8 1×8), µs:

| | full | no SFPU | SFPU ×2 | no partials add | no SFPU, no add | compute only | compute only, no SFPU | compute only, no SFPU, no add |
|---|---|---|---|---|---|---|---|---|
| bs8 | 987.3 | 840.9 | 1329.5 | 979.6 | 810.1 | 929.6 | 771.0 | 734.0 |
| bs16 | 1931.2 | 1631.0 | 2611.7 | 1911.3 | 1560.7 | 1842.1 | 1515.8 | 1438.6 |
| bs32 | 3758.4 | 3104.3 | 5149.2 | 3720.3 | 2951.1 | 3644.6 | 2999.0 | 2842.2 |

One SwiGLU pass costs what a second one adds (342 / 680 / 1391 µs), and about half of it is exposed (skipping it saves
15-17%); the partial-sum add (1%) and the reads / writes (3-6%) are not the gap. Per output tile the SFPU pass is ~1400
cycles (680 µs over ~650 output tiles per core at bs16), but it only overlaps the last K block's math: K_block 20 × 2
(gate, up) × 16 cycles = 640 per output tile. The §58 re-sweep never reached a larger K block (K in 5..20).

K_block 40 doubles the window (`bench_ff13_sweep.py`, µs): bs8 8,40,8 1×8 858.6, 8,40,6 1×6 881.9 (was 999.3); bs16
4,80,4 1×4 1687.6, 4,40,8 1×8 1703.1 (1937.2); bs32 6,40,8 1×8 3254.8, 4,40,8 1×8 3276.5 (3763.4). Subblocks other
than 1×W lose (2×4, 2×2: the SwiGLU pairs gate / up along W). K_block 80 (one K block, no partials) fits only with
small M / N blocks and is not better.

In the model the larger CBs are L1-limited. bs16 4,40,8 fits (cold 156.9 → 148.9 ms). bs8 8,40,8 clashes (static CBs
end at 1,377,408 B, lowest L1 buffer 1,240,704: the post-attention residual sum in L1); 8,40,6 fits. bs32 4,40,8
clashed at 777,216 B: `l1_map_first_layer.py 32` showed the post-attention add's norm output (the FF13 input, 363 KB
per core) allocated below WO's L1 output, which is live during the add and freed right after, so FF13 ran with the top
363 KB of L1 empty and its CBs capped below the norm output. Two fixes, cold bs32 ms:

| | ms |
|---|---|
| default (4,20,8 1×8) | 322.6 |
| 4,40,4 1×4 (fits as is) | 316.7 |
| WO output to DRAM (`TT_PREFILL_WO_L1=0`) | 327.5 |
| WO output to DRAM + 4,40,8 | 308.4 |
| norm output preallocated before WO (`QWEN_FUSED_ADD_NORM_PREALLOC_FF=1`) | 323.6 |
| preallocated + 4,40,8 (landed) | 302.4 |

Trading WO's L1 output for DRAM pays (+4.2 ms for −19), but the preallocation (the FF2 hook's trick, §56, for the
post-attention norm) gets the room without the trade: the lowest L1 buffer during FF13 moves to 1,149,312 B. bs32
6,40,8 still misses by 4 KB. STS-B 0.8119 / 0.8147 / 0.8152 (bs8 / 16 / 32). sustained_run.sh, 3 alternating rounds per
batch, chip 0, cold / sustained ms: bs8 82.5 / 102.0 → 78.5 / 99.5, bs16 157.1 / 201.7 → 148.8 / 196.1, bs32 322.8 /
400.1 → 303.1 / 388.7.

## 61. bs1 fused SwiGLU with K_block 40: the SwiGLU is not what is exposed at M=512 (2026-09-29)

§60's K_block 40 hides the pack-thread SwiGLU at bs8 / 16 / 32, so bs1 (`QWEN_FUSE_SWIGLU_BS1=1`; §58: 214.4 µs, 15.7 →
16.0 ms e2e) was retried against the legacy FF1 + FF3 + mul it replaces (69.7 + 69.7 + 52.3 ≈ 192 µs per layer in the
7dbf479 device profile). `bench_ff13_sweep.py 1` over M 1 / 2 / 4, K 20 / 40 / 80, N 2 / 4 / 6 / 8 / 16, 1×W subblocks
(99 configs, LoFi without fp32 dest as bs1's FF1 / FF3): best per K block 4,20,8 1×2 214.6 µs, 2,40,6 1×2 228.8,
4,80,6 1×6 238.9 (12 K_block-80 configs clash with L1). Nothing reaches 192.

`bench_ff13_fused_ablate.py 1` (µs) shows why:

| blocks | full | no SFPU | SFPU ×2 | compute only | compute only, no SFPU, no add |
|---|---|---|---|---|---|
| 4,20,8 1×2 | 212.3 | 204.9 | 253.3 | 165.7 | 137.3 |
| 2,40,6 1×2 | 223.6 | 221.1 | 232.9 | 158.6 | 144.9 |

At K_block 20 the SwiGLU is 7 µs exposed (3%, against 15-17% batched): with 16 rows of tiles the per-core output is
small and the SFPU pass already fits under the last K block's math. The gap is data movement (reads / writes cost 47 µs,
22%); K_block 40 only halves the SwiGLU's 7 µs and loses more in the K loop. e2e, sustained_run.sh, 2 alternating
rounds, chip 1, cold / sustained ms: default 15.6-15.7 / 15.8, fused 4,20,8 1×2 16.1 / 16.2, fused 2,20,8 1×2 16.1 /
16.2-16.3. bs1 stays unfused. Note that `QWEN_FUSE_SWIGLU_BS1=1` alone is a no-op: the demo defaults
`QWEN_FUSE_SWIGLU=0` at bs1, so the packed weight is never built; both must be set.

## 62. Fused FF1+FF3: a SiLU sized for the bfp8 output, a third of the SFPU time (landed) (2026-09-29)

**What a free SiLU would buy.** At the K_block-40 blocks the pack-thread SwiGLU is mostly hidden:
`bench_ff13_fused_ablate.py <batch>` with `MM_BLOCKS=4,40,8,1,8`, µs:

| | full | no SFPU | SFPU ×2 | no partials add | no SFPU, no add | compute only, no SFPU, no add |
|---|---|---|---|---|---|---|
| bs16 | 1700.4 | 1641.9 | 2290.3 | 1630.0 | 1569.0 | 1454.8 |
| bs32 | 3259.6 | 3125.0 | 4524.9 | 3106.2 | 2966.0 | 2861.9 |

One pass is 590 / 1265 µs of pack-thread work and 58 / 135 of it exposed (3.4 / 4.1%); at K_block 40 the partial-sum
add (70 / 153) costs as much. e2e with the SFPU call removed (wrong output, `sustained_run.sh`, 2 alternating rounds,
chips 0 / 1): bs16 cold / sustained 148.8 / 195.1 → 146.7 / 188.7 ms, bs32 302.0 / 403.6 → 292.2 / 388.4, i.e. a
ceiling of −3.3 / −3.8% sustained, more than cold as the power-cap argument predicts.

**Nothing to borrow from tt-blaze.** `dram_streaming_swiglu`'s SILU modes call the stock `calculate_silu` /
`_sfpu_sigmoid_`; its fork's sigmoid / exp / recip headers are identical to ours. Its PACK-side SFPU, single init and
math / pack semaphores are what §58 already does. The one idea, `sdpa_exp_unclamped` (no upper clamp for inputs ≤ 0),
does not apply to a sigmoid's two-sided input.

**Blackhole's `sfpi::approx_exp` (SFPARECIP mode 2) is not an exp.** Probed through the fused kernel's output, it
returns ~sign(x)·e^|x| for |x| < 2 (0.4-0.7% median error) and saturates at 4.0 beyond; usable only after a range
reduction that costs what `exp_21f` does.

**Variants** (`perf_tools/bench_swiglu_variants.py 16 [1 4]`: the pass patched into `swiglu_sfpu.hpp`, error against
torch's fp32 SwiGLU of the device's own bf16 pre-activations, relative L2 at gate std ~1 / ~4; pass cost from
`SW_REPEAT=3`, (3 passes − 1) / 2):

| exp(−gate) / reciprocal | µs per call | pass cost µs | rel err ×1 / ×4 |
|---|---|---|---|
| silu_tile + mul (`exp_21f`, 1 Newton step, bf16 roundings) | 1701.5 | 637 | 0.01283 / 0.01229 |
| `exp_21f`, 1 Newton step, no roundings | 1700.9 | 578 | 0.01166 / 0.01105 |
| `exp_21f`, bare SFPARECIP | 1689.4 | 483 | 0.01190 / 0.01099 |
| Schraudolph, 1 Newton step | 1687.7 | 282 | 0.01383 / 0.01098 |
| Schraudolph, bare SFPARECIP | 1685.7 | 188 | 0.01391 / 0.01096 |
| Schraudolph centred, bare SFPARECIP | 1692.1 | 208 | 0.01228 / 0.01101 |
| same, loop unrolled 8× (landed) | 1686.8 | 184 | 0.01228 / 0.01101 |

Schraudolph (Neural Computation 11(4), 1999) is `_sfpu_exp_21f_bf16_` without its degree-2 mantissa polynomial:
`(x / ln2 + 127) · 2^23` reinterpreted as a float is 2^int · (1 + frac), within 6.1% of e^x, always over. Shifting
the bias by 0.043 (half of log2 1.061; a back-of-envelope choice, Schraudolph's own RMS-optimal shift is ~0.058)
centres the error at ~±3%. Every variant sits at the bfp8 output's ~1.2% floor; the bf16 roundings the old pass copied
from silu_tile + mul cost more accuracy than the cheaper exp does. With `fp32_dest_acc_en` the pass keeps the accurate
sigmoid.

**Where the exposed time is not.** The landed pass cuts the pack-thread work by 71% but the exposed time only from 58
to ~45 µs (bs16 1700 → 1687 against 1642 with no SFPU). Every binary SFPU call opens with `STALLWAIT(STALL_SFPU,
MATH)` (`_llk_math_eltwise_sfpu_start_`), redundant on the pack thread after the MATH_PACK semaphore wait; calling
the pass without it: 1703.2 → 1700.6 (bs16), 3271.3 → 3267.8 (bs32). Not the stall.

**Half of it is each output block's last subblock.** Skipping the pass on one subblock per output block
(`SW_SKIP=first|last|all`, 4 subblocks per block at 4,40,8 1×8, so each skip removes 25% of the work), µs:

| | full | skip first | skip last | skip all | exposed | saved by first / last |
|---|---|---|---|---|---|---|
| bs16 landed | 1688.0 | 1685.0 | 1668.2 | 1646.2 | 41.8 | 3.0 / 19.8 (7 / 47%) |
| bs16 silu_tile + mul | 1702.6 | 1699.7 | 1671.3 | 1646.2 | 56.4 | 2.9 / 31.3 (5 / 55%) |
| bs32 landed | 3225.8 | 3210.6 | 3182.3 | 3127.4 | 98.4 | 15.2 / 43.5 (15 / 44%) |
| bs32 silu_tile + mul | 3267.1 | 3246.9 | 3177.2 | 3128.6 | 138.5 | 20.2 / 89.9 (15 / 65%) |

The first subblock's pass hides behind the next subblock's math; the last one's, which should hide behind the next
output block's K block 0 in the other DST half, half does not (~0.45 µs per block at bs16, about half of one
subblock's pass). Why is open: the pack thread's next-block setup (intermediate reserve, L1-acc / format reconfig)
queued behind the tail, or a CB handoff at the block boundary; device-profiler zones on the math and pack threads
around the boundary would say which. (It was the output writer's deferred write, §63.)

**e2e**, sustained_run.sh, 3 alternating rounds per batch, chips 2 / 0 / 1 concurrently, medians of 3, cold /
sustained ms: bs8 78.5 / 101.2 → 78.3 / 99.4 (−0.3 / −1.8%), bs16 148.9 / 194.9 → 148.1 / 194.4 (−0.5 / −0.3%),
bs32 300.5 / 403.7 → 297.8 / 401.8 (−0.9 / −0.5%); the new pass is faster in all 9 pairs, cold and sustained. STS-B
0.8119 / 0.8147 / 0.8152 → 0.8110 / 0.8132 / 0.8156 (bs8 / 16 / 32; the batch paths alone spread 0.8121-0.8159),
per-text embedding cosine vs the old pass mean 0.995 (p1 0.955-0.962). ttnn nightly `test_minimal_matmul.py -k
swiglu` (4) and `test_minimal_matmul_split.py -k swiglu` (6) pass; their relative RMSE vs torch 0.0078-0.0085 before
and after.

## 63. minimal_matmul: the output writer held the next block's in1 behind the previous block's tail (landed) (2026-09-30)

Following §62's tail finding, bs16 fused FF13, `bench_swiglu_variants.py` with `SW_SKIP`, µs:

| blocks | full | skip first | skip last | skip all | exposed | saved by skip last |
|---|---|---|---|---|---|---|
| 4,40,4 1×4 (2 K blocks) | 1805.5 | 1803.3 | 1756.0 | 1737.8 | 67.7 | 49.5 (73%) |
| 4,80,4 1×4 (1 K block, no partials, no add) | 1664.9 | 1660.5 | 1651.4 | 1627.2 | 37.7 | 13.5 (36%) |
| 4,40,4, partial-sum packs skipped | 1814.4 | — | 1751.9 | 1737.9 | 76.5 | 62.5 |
| 4,40,8 1×8, partial-sum packs skipped | 1684.5 | — | 1660.1 | 1637.1 | 47.4 | 24.4 |
| 4,40,4, partial-sum add skipped | 1749.3 | 1742.3 | 1696.3 | 1691.1 | 58.2 | 53.0 (91%) |
| 4,40,8 1×8, partial-sum add skipped | 1616.1 | 1606.6 | 1594.3 | 1573.8 | 42.3 | 21.8 |

**Not the partial-sum packs** (K block 0's packs queued on the pack thread behind the previous block's SwiGLU): skipping
them leaves the tail as exposed. **Not the partial-sum add** either, though it is a cost of its own, 66-69 µs (4%) of
math-thread throughput at bs16. **K_block 80** (one K block: no packs, no add) only fits small blocks
(`bench_ff13_sweep.py`, 32 K-80 configs per batch): bs16 4,80,4 1×4 1664.7 vs 4,40,8 1×8 1685.2; bs8 best 4,80,6 965.8
vs 8,40,6 857.7; bs32 best 4,80,2 3386.1 vs 4,40,8 3227.8. Not taken.

**The cause.** The in1 reader (`dm_in1_sender_out_metal2.cpp`, likewise `dm_in0_sender_metal2.cpp` where it writes)
writes an output block during the next block's K loop, at `k_block_iter == defer_write_k_block`, and waits for that
output (`dfb_out.wait_front`) before reading (and forwarding) that K block's in1. The descriptor set
`defer_write_k_block = min(core.y * k_blocks_per_core, K_blocks - 1)`: with 2 K blocks, 0 on row 0, so those writers
stalled the next block's first K block behind the previous block's last subblock and its write; and injector cores never
deferred (`defer_write && !is_injector_core`), writing each block synchronously before injecting the next block's in1,
which holds every core down the forwarding chain. Kernel-patch variants (µs, bs16 / bs32 fused FF13 at 4,40,8 1×8,
bs8 at 8,40,6 1×6): base 1686.3 / 3236.3 / 872.2; injectors defer (at their `core.y` K block) 1693.3 / 3234.5 / 869.9;
defer point never K block 0 (injectors still synchronous) 1692.5 / 3228.2; both 1654.4 / 3132.1; everything deferred
to the last K block 1649.2 / 3129.0 / 871.0. With the SwiGLU skipped entirely too 1644.1 → 1616.2 (bs16), 3126.1 →
3060.8 (bs32): the synchronous write cost more than the SwiGLU tail.

**Landed:** the descriptor never defers to K block 0 when there is a later K block (`defer_write_k_block_for`; the
`core.y` stagger for large-K matmuls is kept), and injectors defer like the other writers. The model's other matmuls
(`bench_mm_ablate.py <preset> <batch>`, full variant, µs, base → fix):

| | bs8 | bs16 | bs32 |
|---|---|---|---|
| FF13 fused | 872.2 → 871.0 | 1686.3 → 1649.9 | 3236.3 → 3139.4 |
| QKV | 266.3 → 266.3 | 559.9 → 534.1 | 1125.2 → 1072.4 |
| FF2 | 373.1 → 373.0 | 732.2 → 724.3 | 1502.2 → 1474.1 |
| WO | 174.3 → 173.3 | 333.4 → 325.3 | 660.9 → 638.5 |

Outputs unchanged (the variant bench's error vs torch identical to 5 digits). ttnn nightly `test_minimal_matmul.py` +
`test_minimal_matmul_split.py`: 216 passed, 276 skipped (`test_performance` excluded: it shells out to tracy with the
system python). e2e, sustained_run.sh, 3 alternating rounds (rebuilt between arms), chips 2 / 0 / 1, medians, cold /
sustained ms: bs8 78.2 / 100.0 → 78.3 / 100.4 (the sustained +0.4% repeats in all 3 pairs, with the new arm first;
standalone bs8 is unchanged), bs16 148.0 / 194.1 → 145.3 / 192.8 (−1.8 / −0.7%), bs32 298.3 / 401.9 → 292.0 / 399.6
(−2.1 / −0.6%).

## 64. SDPA's DRAM round-trip: only K / V matter; K / V in L1 landed at bs8 / 16, bs32 does not fit (2026-09-30)

WIP_HANDOFF item 3. `bench_sdpa_bs16_ablate.py <bs> l1`, the shipped reuse_kv q128 config (12×8 at bs8, 12×10 at
bs16 / 32) with each operand moved from DRAM to L1 interleaved, µs per call, PCC 1.00000 vs the shipped config
throughout:

| placement | bs8 | bs16 | bs32 |
|---|---|---|---|
| all DRAM (shipped) | 237.0 | 355.8 | 624.4 |
| Q in L1 | 222.5 | 338.0 | 592.0 |
| **K / V in L1** | **201.2** | **308.7** | **563.0** |
| Q + K / V in L1 | 200.9 | 306.1 | clash |
| output in L1 | 225.9 | 350.1 | 604.6 |
| Q + K / V + output in L1 | 195.0 | 304.4 | OOM |

K / V are 74 / 148 / 297 KB per core (K + V) at bs8 / 16 / 32; Q is twice that and adds little on top. **Landed** at
bs8 / 16 as `QWEN_HEADS_KV_L1=1` (the heads op's `kv_memory_config`; POSITIVE_RESULTS). bs1 already holds Q / K / V in
L1 (`l1_map_first_layer.py 1`: 19 / 5 / 5 KB per core, SDPA reads them there).

**bs32 does not fit.** `l1_map_first_layer.py 32` with the knob: the first half-batch QKV matmul (the first op after
K / V are allocated) clashes, `L1 buffer allocated at 405120 and static dataflow buffer region ends at 595072`. Live at
that matmul, per core: the two preallocated half-batch norm outputs (2 × 181 KB, top of L1), K and V (2 × 145 KB) and
the matmul's L1 output (446 KB), ~1.1 MB against ~945 KB above the matmul's CBs: 190 KB short. Ruled out: K alone in L1
(still ~41 KB short); K / V allocated after chunk 0's matmul (they land where chunk 1's matmul CBs go); chunk 0's QKV
output to DRAM (its heads op then reads half the batch from DRAM, ~80 µs per layer, more than the 61 µs SDPA saves).

**Open: 4 QKV chunks at bs32** (quarter-batch norm outputs 4 × 90 KB + K / V 290 KB + a 223 KB QKV output ≈ 880 KB,
fits). The matmul side is roughly neutral since §63 (4 × 266 = 1064 µs vs 2 × 534 = 1068 µs per layer), so the gain
would be SDPA's 61 µs per layer minus two more heads-op launches. It needs `fused_add_rmsnorm_split` to write four
output tensors (its writer has two accessors) and `decoder_fusion.py` / `qkv_chunks.py` to allow `QWEN_QKV_CHUNKS=4`.
Not tried.

## 65. Batched SDPA: compute-bound with data movement close behind; the pack thread paced the softmax; row sums moved to the math thread (landed) (2026-09-30)

Tools: `sdpa_kernel_variants.py` (patched kernel trees), `bench_sdpa_floors.py` (traced, optional device-profiler
parse), `bench_sdpa_zones.py` (per-unit zones). Run them from a directory with no `ttnn/` tree: kernel lookup tries the
cwd before `TT_METAL_KERNEL_PATH`, and from the repo root every variant silently compiles the repo's kernels (the first
attempt here timed all variants equal to the control).

**Floors.** Traced replays under the device profiler, µs of 1.35 GHz device cycles, at the in-model placements (eager
back-to-back runs of the data-movement-only variant swing between ~484 and ~640 µs at bs32; traced ones do not):

| | control | compute only (no NoC reads / writes) | data movement only (compute stubbed) |
|---|---|---|---|
| bs8, K / V in L1 | 178.5 | 164.9 | 102.0 |
| bs16, K / V in L1 | 284.7 | 268.2 | 183.3 |
| bs32, all DRAM | 596.8 | 519.4 | 483.1 |

Compute is the larger floor everywhere; at bs32 data movement is within 7% of it, and the kernel runs 6-12% above the
larger floor because the two do not fully overlap. All DRAM at bs8 / 16: DM-only 145.0 / 257.1 against compute 164.9 /
268.2. The device-profile artifact's SDPA rows now use these floors as the roofline ("compute + DM").

**Where a unit goes** (one q128 chunk of one head against its KV head's 512 tokens; math-thread zones at the bs8 shape,
compute only, cycles): Q·Kᵀ 5,528 (2 × 128 tile matmuls), x − max + exp not hidden 5,175, P·V 5,934, normalize 2,887,
row max 466, other 312; 20,302 total, 15.0 µs. The matmuls run at 71% of the LoFi peak (8,192). Data waits add 1,249
cycles with K / V in L1 and 3,926 all DRAM (next Q chunk before the first Q·Kᵀ, V in the drain).

**Why MATH waits on PACK's exp.** Dest holds two 8-tile halves. Per column block of the second row group MATH does x −
max (half A) and a 2×4 Q·Kᵀ subblock (half B, ~690 cycles); PACK does exp on half A (500-540 cycles, SFPU from the pack
thread), packs it back in place and L1-accumulates the 8 row-sum packs (455-465 together), then packs half B. PACK is
the slower thread, so MATH blocks in `tile_regs_acquire` (its "SUB" zone reads 725-750 cycles for an 8-tile subtract).
Probes (compute only, wrong output): no row-sum packs 18,004 cycles per unit (−11%), no exp 16,517 (−19%, so almost
none of the exp overlapped anything).

**Row sums on the math thread (landed).** With one K chunk (no online-softmax correction), `sub_exp` skips the row-sum
packs and normalize computes each row group's sums from the exp'd scores still in `cb_qkt_im`:

| normalize's row sum | unit (cycles) | normalize |
|---|---|---|
| L1-accumulated by the pack thread (before) | 20,302 | 2,887 |
| `reduce_tile<SUM, REDUCE_ROW>`, 16 tiles per tile row | 20,007 | 4,852 |
| 1×1 matmul against `col_identity` per score tile | 20,059 | ~4,850 |
| **one matmul per score column over the row group** (`rt_dim` 2, `col_identity` unpacked once) | **18,766** | 3,615 |

Re-reading a score tile through unpack costs ~57 cycles either way (the main matmuls reach ~22 per tile-product by
reusing unpacked operands across an 8-tile subblock); batching the row group halves the calls and puts both rows'
reciprocals in one dest acquire. `matmul_block` on Blackhole has no MOP over K (`kt_dim` is only in0's row stride), so a
16-tile `col_identity` would not have streamed. The batched init has to be followed by a 1×1 `matmul_block_init`: the
next row group's V matmul only re-inits short and hung on the leftover `rt_dim` / `kt_dim` (every q_chunk ≥ 256 shape;
q128 never runs a V matmul after a normalize in the same unit, so the model shapes did not show it). Scope: Blackhole,
single K chunk, not ring / in-place V / attention sink; `SDPA_SUM_ON_PACK` restores the old path; the recip scratch CB
grows to the normalize row-group height (2 tiles).

Traced, device µs per call: bs8 179.3 → 168.7 (−5.9%), bs16 286.1 → 266.1 (−7.0%), bs32 600.4 → 566.2 (−5.7%). PCC vs
torch at the model config 0.999248 → 0.999234 (max error unchanged; causal, padded-Sk and two-K-chunk cases also
match). ttnn SDPA unit tests pass (reuse_kv 97, pack_gqa_heads 29, windowed 68, prefill 8, output_heads_concat 8).
STS-B through `eval_accuracy_batched.py` bs8 / 16 / 32: 0.8110 / 0.8132 / 0.8156 → 0.8120 / 0.8174 / 0.8148; per-text
cosine new vs old mean 0.9946 (min 0.86), inside the model's own spread (old kernel bs8 vs bs16: mean 0.9942, min 0.80).
End to end SDPA is ~8% of the replay: cold bs8 77.0 → 76.7 ms, bs16 144.0 → 143.1, bs32 within noise (same chip, 2
alternating rounds).

**No cheaper exp.** `exp_approx_mode` already runs the replay-buffer `SFPLOADMACRO` Schraudolph pipeline (load, MAD by
scale / ln 2 plus bias, round to int, shift into the exponent, store; `ckernel_sfpu_exp.h`, rated ~68 cycles a tile) and
measures ~64 cycles a tile (EXP zone 500-540 per 8 tiles, unchanged by the row-sum change, which cut PACK SUB_EXP from
455-465 to 121-168 and MATH's acquire wait from 725-750 to 265-420). Unlike the SwiGLU sigmoid (§62) there is no
heavier formulation to strip. Splitting the exp over MATH and PACK would have both threads drive the one SFPU (shared
LREG / addrmod state). SDPA compute stops here.

**What the data movement at bs32 is.** Floors on these kernels (traced, device µs): control 566.6, compute only 480.9,
data movement only 482.8 (178 MB, ~369 GB/s, ~90% of what streaming ops reach), reads only 335.2, writes only 247.4.
CB-wait zones at bs16 all DRAM (`sdpa_kernel_variants.py` + wait zones) put nearly all the waiting on each core's first
unit: 58,900 cycles (~44 µs) while all 120 cores pull their first head's K + V (16.7 MB) at once, against 2,950
cycles per core for every later KV-head switch together; the output CB never back-pressures (47 cycles). A next-head
K / V prefetch in the reader (the next head's slots reserved from a head's second Q chunk on, its tiles issued a Q
chunk's worth per Q chunk) only added contention: bs16 all DRAM 325.1 -> 371.1 µs, bs32 566.6 -> 619.7, first-unit wait
65,975 cycles; reverted. Streaming K into compute would not shorten the start either (V is needed ~6 µs later). The
lever is fewer K / V bytes in DRAM: K / V in L1 cut the first-unit wait to 19,300 cycles (§66).

## 66. bs32 QKV in four quarter-batch chunks: K / V fit in L1 (landed) (2026-09-30)

§64's open item. `fused_add_rmsnorm_split` writes its normalised output as 2 or 4 equal row parts (the writer has four
output accessors), `decoder_fusion.py` preallocates `QWEN_QKV_CHUNKS` parts, and the attention's chunk hooks (already
chunk-count generic) run QKV + the heads op per quarter batch into full-batch Q / K / V, with K / V in L1. Live per core
at the first chunk: 4 × 90 KB norm outputs (the same 362 KB as two halves), K / V 290 KB, a 223 KB QKV output (was
446): it fits. `test_qkv_chunks.py 4`: the 4-part add+norm and the 4-chunk heads op are bit-identical to one tensor /
one full-batch call.

Device profile (bs32 replay, ms):

| | 2 chunks (was) | 1 chunk + K / V L1 | 4 chunks | **4 chunks + K / V L1** |
|---|---|---|---|---|
| QKV matmuls | 37.72 (71 calls) | 38.73 (36) | 36.95 (141) | 36.92 |
| heads op | 15.26 (71) | 21.00 (36, DRAM input) | 16.74 (141) | 16.73 |
| SDPA | 19.77 | 17.99 | 19.74 | **17.97** |
| replay | 280.00 | 285.08 | 280.64 | **278.88** |

SDPA −50 µs per layer (549 -> 499), quarter QKV matmuls slightly faster than halves; the heads op pays ~10 µs of fixed
cost per call (115.3 µs per quarter vs 209.7 per half). sustained_run.sh, 2 alternating rounds per chip, cold /
sustained: chip 0 294.2 / 374.5 -> 293.5 / 373.8 ms, chip 1 296.7 / 382.4 -> 295.2 / 376.6. STS-B bs32 0.8148 ->
0.8155; not bit-identical, and not because of the chunking ops: `minimal_matmul` itself differs across M (the QKV
matmul at M = 4096 vs M = 8192 with the same config: 7.5% of elements on every row, max 0.094, bfp8 rounding), while K /
V in L1 alone and 2 chunks vs 1 are bit-identical and the eval is deterministic. Default at bs32
(`QWEN_QKV_CHUNKS=4`, `QWEN_HEADS_KV_L1=1`); `QWEN_QKV_CHUNKS=2` restores the old path.

## 67. NoC transaction ids for the custom ops: the heads op is compute-bound, not DRAM-bound; two bit-identical fixes, neutral e2e (landed) (2026-09-30)

Question: would transaction-id (trid) pipelined reads / writes (as in `all_gather/.../multicast_reader.cpp`,
`moe_compute/.../dm0.cpp`) pay in the model-local ops that looked data-movement-bound? 8× p150b host; standalone on
chip 5, traced; e2e `sustained_run.sh`, 2 alternating rounds per batch on chips 1 / 2 / 0 (bs8 / 16 / 32).

**Heads op at today's placement (`perf_tools/bench_heads_placement_ablate.py`: v3, QKV in L1, Q DRAM, K / V L1,
preallocated outputs), µs per call:**

| variant | bs16 | bs8 | bs32 quarter chunk (8 → 32) | bs16, Q in L1 |
|---|---|---|---|---|
| full | 216.2 | 120.5 | 120.0 | 215.2 |
| compute only (no unit read / write) | 210.7 | 115.6 | 117.0 | 209.2 |
| data movement only (copy compute) | 223.9 | 109.2 | 110.9 | 233.8 |
| read only | 238.3 | 107.8 | 110.9 | 237.0 |
| write only | 162.7 | 88.5 | 89.1 | 84.5 |
| handshakes only | 63.8 | 38.8 | 39.7 | 64.2 |

Compute-bound at every batch size: full is within 3-5 µs of compute only. Not DRAM-bound: Q's DRAM write is ~100 µs of
writer time (write only 162.7 vs 84.5 with Q in L1) and all of it hidden (full 216.2 vs 215.2). The DM floor is close
behind (read only ≈ compute at bs8 / bs32), so a faster compute would hit the reader next: it moves ~2 GB/s per core
from L1 (26 KB units, one barrier per unit), which is where trid-pipelined reads would matter. Page → bank math is not
the cost (constant-divisor multiply for the 120 L1 banks).
§56's "sits at the DRAM floor" applied to the old DRAM-input placement only.

**Heads op cos / sin double-buffered (landed, `QWEN_HEADS_ROT_DB=1` default in the op).** With v3 the cos / sin CBs
held one unit (`cache_rot` is v2 only): the reader's next-unit cos / sin read waited for compute to pop this unit's,
after the Q / K heads, and only the V copy covered it. Two tiles deep: bs16 214.7 / 214.8 → 208.7 / 209.2 µs, bs32
chunk 121.4 / 120.1 → 117.8 / 118.7 (compute only drops by the same 5-6 µs: the old floor included the bubble);
bit-identical Q / K / V at bs8 / 16; +16 KB CB per core, fits at bs8 / 16 / 32. E2e cold / sustained 76.9 / 99.1,
143.5 / 193.6, 291.1 / 377.1 ms: within noise of the baseline (~0.1-0.2 ms expected).

**Add+norm partial exchange on its own trid (landed, `QWEN_ADD_NORM_PART_TRID=1` default in the op).** The writer's
`noc_async_write_barrier()` before the semaphore increments also waited for the wave's sum slice to be acked by DRAM;
the partial writes now carry trid 1 (`noc_async_write_one_packet_with_trid`, then `noc_async_write_set_trid(0)`: the id
stays in the command buffer's packet tag) and the barrier is `noc_async_write_barrier_with_trid(1)`. Bit-identical at
bs8 / 16 / 32 (`perf_tools/bench_add_norm_placement.py`). Standalone 86.9 → 87.8, 130.5 → 128.0, 274.6 → 273.8 µs
(bs8 / 16 / 32). E2e cold / sustained, off → on: bs8 76.7 / 98.7, 76.7 / 99.7 → 76.7 / 99.9, 76.8 / 99.3; bs16 143.4 /
192.8, 143.2 / 194.9 → 143.3 / 194.7, 143.2 / 194.9; bs32 291.0 / 375.5, 291.7 / 382.0 → 291.6 / 379.0, 292.3 /
379.9: neutral. The exchange latency was not the wave's cost. Every CB of the op holds one wave (a, b, sum, out), so
the next wave's add waits for this wave's normalised writes to be acked: that serialisation is the next thing to try.

**The hang that preceded these runs.** Every e2e run (and the committed HEAD) hung in the first warmup prefill.
tt-triage (`tools/tt-triage.py --run=dump_callstacks`) put all cores in SDPA's `normalize_row_streaming`, pack thread in
`scratch_cb.reserve_back(sbh)`: the installed `_ttnncpp.so` (00:19) predated 0458a990466 (20:08), which sized that CB to
a row group in the program factory, so the JIT'd kernel reserved sbh tiles of a 1-tile CB. `./build_metal.sh` fixed it
(STS-B bs8 0.8120 after). PERF_GUIDE §4 now says to rebuild after host-side changes.

**Addendum: what bounds the add+norm, and double-buffering its CBs (negative).** `perf_tools/bench_add_norm_ablate.py`
at the model's placement (a DRAM; b / normalised out L1; sum L1 at bs8 / 16, DRAM at bs32), µs per call:

| variant | bs8 | bs16 | bs32 |
|---|---|---|---|
| full | 88.0 | 129.0 | 275.3 |
| no a / b reads | 73.4 | 114.2 | 233.4 |
| no sum / out writes | 83.9 | 125.2 | 217.1 |
| no reads, no writes | 67.3 | 98.7 | 159.8 |
| local exchange (no peer writes / semaphore) | 92.5 | 130.7 | 291.3 |
| compute + handshakes (no reads / writes / exchange) | 62.5 | 88.7 | 150.0 |
| data movement only (copy compute) | 79.6 | 118.3 | 253.2 |
| handshakes only (copy compute, nothing moved) | 39.5 | 51.3 | 83.4 |

Data-movement-bound at every batch size: the DM floor is the larger one and the op runs at 90-92% of it. At bs32 that
floor is the DRAM traffic (a read + sum write, 89 MB: ~350 GB/s, about what stock streaming ops reach here, §54); at
bs8 / 16 only a is in DRAM (22 / 44 µs at 512 GB/s), so the floor is per-wave latency, not bandwidth. The copy compute
is not free (handshakes only), so the DM floors are upper bounds. The exchange is not the cost. The profile artifact's
"L1 / compute" tag at bs8 / 16 (DRAM floor under half the call) is wrong for this op; "DRAM" at bs32 is right.

CBs 0 / 1 / 16 / 17 two waves deep (a scratch knob, reverted): bs8 87.8 → 89.4 µs; bs16 / 32 do not fit beside the L1
operands even standalone (static CBs clash at 423,808). The next wave's add already overlaps the writer, since compute
runs the stages in order anyway; what is exposed is the per-wave latency of the reads and writes.

## 68. bs1 SwiGLU product: the SiLU was the cost; the single-pass SwiGLU lands, sharding and a LUT sigmoid do not (2026-10-01)

At cdb9143 the bs1 SwiGLU product (`ttnn.mul(a, b, input_tensor_a_activations=[SILU])` on FF1 / FF3's [512, 9728]
bfp8 outputs, L1 interleaved) took 52.1 µs per call, 1.88 ms of the 15.27 ms replay, with no FPU work and every operand
in L1. Standalone at the model's placement (chip 5, traced, device kernel µs from the device profiler via
`perf_tools/device_kernel_us.py`; `perf_tools/bench_bs1_swiglu.py` for the FF1 + FF3 + product chain):

| variant | interleaved L1, 120 cores | block-sharded 12×8 (FF1 / FF3's grid), 96 cores |
|---|---|---|
| `ttnn.mul(silu(a), b)` (the model) | 52.1 | 56.9 |
| `ttnn.mul(a, b)`, no SiLU | 19.5 | 4.0 |
| `ttnn.silu(a)` | 41.3 | — |
| `silu_mul` mode 0 (precise `silu_tile`, dest-reuse multiply) | 56.0 | — |
| `silu_mul` mode 3 (`minimal_matmul`'s single-pass SwiGLU) | **30.2** | 33.6 |
| mode 3, output writes skipped / output CB aliased on a sharded output | — | 32.9 / 32.8 |
| mode 3 with the 3-segment LUT sigmoid (`calculate_sigmoid_appx`) | 18.6 | — |

The SiLU costs ~1,400 cycles per tile; the single-pass SwiGLU (one SFPU pass over the gate / up pair, Schraudolph exp
and a bare SFPARECIP sized for the bfp8 output, §62) ~770, which makes the product SFPU-bound: on the shards the reads
are gone (inputs stay resident: CBs 0 / 1 alias each core's shard) and the output write costs 0.7 µs, yet 52 tiles per
core on 96 cores (33.6) lose to 40.5 per core on 120 cores reading interleaved tiles (30.2). The sharded path also
needs FF1 / FF3 at 1×2 subblocks (`out_subblock_h == 1` for a sharded output with per_core_N 26): 69.8 → 71.2 µs each,
bit-identical. It stays in `silu_mul` (`supported_sharded`) but nothing uses it.

Accuracy (standalone, against an fp32 torch SwiGLU of the same bfp8 inputs): rel. RMSE stock 0.0176, mode 0 0.0111,
mode 3 0.0113 (PCC 0.99991 vs 0.99981 stock), LUT sigmoid 0.0785 (max |err| 1.09 vs 0.19): the LUT is at the read
floor but 7× less accurate, not usable; a finer piecewise sigmoid (`lut2`, 6 segments) is the untried middle, worth at
most ~10 µs per layer. In-model (`QWEN_SILU_MUL_VERIFY=1`, eager bs1, all 36 calls) mode 3 is closer to torch than the
stock op in 24 / 36 layers, min PCC 0.99974 against stock's 0.99972.

**Landed** (`QWEN_SILU_MUL_BS1=1`, default; the wrapper routes products below `QWEN_SILU_MUL_MIN_ROWS` = 8192 rows to
mode 3, larger ones keep mode 0): e2e chip 4, `ab_one.sh` 2 rounds, cold best 15.6 / 15.6 → 14.8 / 14.8 ms (−5.1%);
`sustained_run.sh` 2 alternating rounds, sustained 15.7 / 15.7 → 14.9 / 15.0 ms (bs1 holds 1350 MHz). STS-B bs1
bucketed 0.8159 (0.8161 before), fixed-ISL batch 1 0.8117 (0.8121). bs8 / 16 / 32 run the fused-SwiGLU matmul and never
reach the product.

## 69. bs1 matmuls: compute-kernel-bound on 96 cores; FF1 / FF3 on the 1D matmul over 120 cores with DRAM-streamed weights (landed) (2026-10-01)

The bs1 projections (legacy 2D multicast on 12×8, M = 16 tile rows over 8 grid rows, bfp4 weights DRAM width-sharded
over the 8 banks, LoFi) were 10.2 ms of the 15.27 ms replay at ~70% of the FPU on their 96 cores. Kernel-variant
ablation (`perf_tools/mm_legacy_variants.py` trees via `TT_METAL_KERNEL_PATH`, `perf_tools/bench_bs1_mm_ablate.py`,
device µs, chip 5):

| | full | compute only | no in1 (reads + multicast) | no in0 | no output writes | FPU ideal on 96 cores |
|---|---|---|---|---|---|---|
| QKV | 42.5 | 40.8 | 41.7 | 42.6 | 41.6 | 30.3 |
| WO | 31.6 | 31.0 | 31.2 | 31.5 | 31.2 | 20.2 |
| FF1 / FF3 | 69.6 | 68.5 | 68.8 | 70.2 | 69.7 | 48.0 |
| FF2 | 68.7 | 68.6 | 68.9 | 68.8 | 68.3 | 48.0 |

Data movement is entirely hidden; the compute kernel takes ~21 cycles per tile matmul against LoFi's 16. (The
`no_in1_read` guard alone did not take effect: the coalesced weight reads go through another call site.) Bigger
blocks barely help: compute only, 2 × 26 / 2 × 16 / 2 × 7 tiles per core run 20.8 / 20.2 / 21.1 cycles per tile
matmul, 8 × 8 with 4×2 subblocks 19.1, 8 × 16 18.6. In-model configs (block-sharded in0 pins QKV / FF1's
in0_block_w to its 8-tile shard): FF2 1×7 / in0_block_w 19 68.7 → 65.9, WO 1×7 31.8 → 31.3, the rest at their best:
≈ 3 µs per layer, the §39 fine-sweep result again (noise e2e), not pursued.

The lever is the 24 idle cores, which the 2D kernel cannot reach (16 M tile rows over 10 grid rows). With every core
owning all 16 M rows and an N slice (the 1D in0-multicast kernel, `MatmulMultiCoreReuseMultiCast1DProgramConfig`
`mcast_in0`), FF1 / FF3's 304 N tiles fill 102 cores at 3 per core:

| FF1, 1D on 12×10, 16 × 3 tiles per core, device µs | in0_block_w 2 | 4 | 8 | 16 |
|---|---|---|---|---|
| weights L1-resident, in0 L1 interleaved (one in0 sender) | 105.1 | 89.3 | 78.7 | — |
| weights L1-resident, in0 width-sharded over 10 cores | 100.7 | 78.0 | 61.3 | — |
| weights L1-resident, in0 width-sharded over 5 cores | — | — | 64.3 | 60.2 |
| weights DRAM interleaved (stock 1D, tile reads), in0 width-sharded over 10 cores | — | — | 91.8–105 | — |
| **weights DRAM width-sharded (patched factory), in0 width-sharded over 10 cores** | — | — | **61.9** | 63.1 (5 cores) |

4 N tiles per core (76 cores) is 79-85 µs. `minimal_matmul` at M = 512 is +60% (§45); the DRAM-sharded config
requires M = 1 tile; the gather_in0 ring keeps all of in0 per core (1.4 MB). Width-sharded in0 over 40 / 20 cores
returned wrong results (PCC 0.23 / 0.47) without an error.

**tt-metal change.** The 1D factory had no DRAM width-sharded in1: it treated any sharded in1 as an L1-local shard,
and its Metal 2.0 fork (the default factory) lacks the `IN1_DRAM_WIDTH_SHARDED` reader by design. Now
`MatmulMultiCoreReuseMultiCast1DProgramConfig(mcast_in0=True)` with a DRAM width-sharded in1 runs on the legacy
builder, whose mcast_in0 path sets `IN1_DRAM_WIDTH_SHARDED` and gives each core the (bytes, bank) segments covering its
N tiles, as the 2D factory's in1 senders do (the shared legacy in1 kernel already reads them); the program-cache
override no longer re-points cb_src1 at a DRAM in1. The weight stream is hidden (61.9 vs 61.3 L1-resident), PCC 0.99995.

**Landed** (`QWEN_BS1_FF13_1D=1`, default; `tt/mlp.py` `_wrap_ff13_1d`): at bs1 FF1 / FF3 read the MLP norm's 10×8
block-sharded output resharded once onto 10 cores (512 × 256 each, 1.6 µs device) and run the 1D config above.
E2e chip 4, `ab_one.sh` 2 rounds: cold 14.8 / 14.8 → 14.1 / 14.1 ms (−4.7%); `sustained_run.sh` 2 alternating rounds:
14.9 / 14.9 → 14.3 / 14.3 ms. STS-B unchanged (0.8159 bucketed, 0.8117 fixed ISL). QKV (192 N tiles: 2 per core
fills 96 cores) and WO / FF2 (80) gain nothing from 120 cores.

## 70. bs1 SDPA: compute-bound at one unit per core; more cores need ragged row groups and multi-row K / V delivery (diagnostic) (2026-10-01)

The bs1 call (profile attributes at 3164ba1): Q [1, 32, 512, 128], K / V [1, 8, 512, 128] bfp8 in L1, `pack_gqa_heads`,
`output_heads_concat`, q192 / k512 on 11×8 = 88 units of 6 Q row tiles, one per core, LoFi, exp approx; 27.1 µs
in-model. `perf_tools/bench_sdpa_bs1_floors.py` with the `sdpa_kernel_variants.py` trees, device µs (chip 5):

| q192 / k512, 11×8 | full | compute only | DM only | DM reads only / writes only |
|---|---|---|---|---|
| µs | 27.6 | 25.2 | 11.7 | 12.0 / 10.2 |

Compute-bound at 91% of the compute floor; §52's "mostly fixed data-movement cost" (a wall-clock fit) does not hold on
device time. (The compute-only output still matches torch: each core reads the same chunk every call and its CBs keep
the previous run's data, so the PCC check cannot prove that variant; its patch counts are asserted.)

**Where a unit goes** (`zones` / `zconly` trees, math thread, compute only, cycles): row groups of 2 rows; Q·Kᵀ + exp
7,709 / 4,651 / 4,798, row max 3 × ~236, P·V 4,591 (with the last exp) / 2,673 / 2,414, normalize 1,848 / 1,790 /
1,796; unit 32,856 (24.3 µs) of the 25.3 µs kernel, against 12,288 cycles of LoFi FPU work. Per row group this is
§65's bs8 cost (~5k cycles per 2-row group, exp at its rated speed, subtract and row sums already on the math thread),
so nothing new there. The pack thread's normalize zones read ~4,000 cycles because it packs the P·V output inside them
(its P·V zones read ~200); per-thread unit totals match (32,856 / 32,902). bs1-specific: the first row group carries
~3,000 cycles (~2.2 µs) of per-unit startup that one unit per core cannot amortise.

**More cores lose** (device µs, full / compute only / DM only where measured): 8×8 q256 33.6; 12×9 q160 36.8 /
35.1 / 26.0; 12×10 q160 36.8 / 35.1 / 26.0; 11×10 q160 37.9; 12×8 q160 46.2; 12×10 q128 43.7 / 40.1 / 31.5 (8 cores
take two units); 12×10 q96 45.7; k256 31.4 (q192) / 35.8 (q160). Two causes at q160, both structural:
`determine_largest_subblock_size(Sq_chunk_t = 5, 16, 8)` finds no 2- or 4-row subblock and picks 1×8, so a 5-row unit
runs five 1-row groups (compute only 35.1 against 25.2 for six rows in three groups), and a head's 13 chunks span two
grid rows, which turns the K / V chain multicast off (DM only 26.0 against 11.7). Fixing both (ragged 2 + 2 + 1 row
groups in the streaming compute; K / V delivery for heads spanning grid rows) bounds bs1 SDPA at ~23 µs: ≈ −0.17 ms
(1.2%) of bs1 for two tt-metal changes. Not pursued; bs1 stays on q192 / 11×8.
