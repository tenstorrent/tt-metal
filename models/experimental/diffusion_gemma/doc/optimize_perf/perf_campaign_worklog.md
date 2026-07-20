# DiffusionGemma perf-optimization campaign (#47465 → goal 100 tok/s)

> Chronological campaign log: intermediate “current” and “ceiling” statements
> are historical at their section date. The selected result is the 2026-07-10
> self-conditioning logits-L1 section below and
> `selfcond_logits_l1_e2e.json` (18.844 tokens/block/s at @48).

## Authority matrix (2026-07-17)

- **Pure prefill, 64K build:** use
  `context_window_prefill_only_chunkedlong_20260713_msl65536.json` at `233b88276ab`
  (16K 2950 tok/s, 32K 3022 tok/s, 64K 1842 tok/s). The artifact without `chunkedlong` is the
  superseded dense-fallback control.
- **Selected fixed-K=48 output-block throughput:** 18.844 tok/s from
  `selfcond_logits_l1_e2e.json` at `0472860c40c`; it is warmed trace replay, not pure prefill or
  first-request TTFT, and has not been rerun at current HEAD.
- **Live vLLM:** use `doc/vllm_integration/README.md` and dated live artifacts. Do not substitute
  raw frozen-prefix, early-halt, or standalone-session rows for current live-serving throughput.

Optimization unit = **traced** denoise step over the 256 canvas + commit (dg-08 methodology).
Config: Blackhole QB2 `(1×4)` TP, tuned true-sparse MoE (`DG_SPARSE_MOE_TUNED=1`, HiFi2).

## Baseline (2026-07-08, tuned, 30 layers)

| path | ms | note |
|---|---:|---|
| denoise step — **traced** | **233.4** | the ranking metric |
| denoise step — eager | 720.8 | trace = **3.1× faster** (dispatch overhead is 68% of eager) |
| commit — eager | 129.0 / token | not yet traced; per-token autoregressive |
| prefill TTFT (18-tok prompt) | 607.9 | |

Traced 233 ms ≈ device-FW sum (276 ms/dev projected), i.e. trace closes the eager dispatch gap;
the eager op-breakdown therefore maps directly onto the traced step.

## Op-topology audit (share of the traced denoise step)

From the 2L+6L→30L device-FW decomposition (`whole_gen_opprofile/`):

| bucket | share | ~ms of 233 | where |
|---|---:|---:|---|
| MoE + attention **Matmul** | 35% | ~82 | 5 matmuls in `sparse_moe.sparse_experts_forward` (dispatch, gate/up/down, combine) + attn proj |
| **layout / glue** | 28% | ~65 | `build_capacity_dispatch` typecast×4 / scatter / gather / slice / reshape; tilize↔untilize; sharded↔interleaved |
| **elementwise / reduce** | 22% | ~51 | BinaryNg, Unary, Reduce (activation, routing, entropy-accept) |
| LayerNorm | 6% | ~14 | |
| TP collectives | 4% | ~10 | AllGather / ReduceScatter |
| diffusion token-select (ArgMax) | 4% | ~10 | per-step, fixed |

Permute cumsum-artifact is **gone** (1.8%) with the capacity-dispatch MoE — the old #47465
`SparseMatmul+Permute` breakdown is obsolete.

## Prioritized levers (to be applied + measured one at a time, traced before/after)

1. **layout/glue in `sparse_moe.py` (28%)** — collapse redundant `typecast`s (idx→uint32 *and* →float32),
   fuse the scatter/gather dispatch-matrix build, avoid tilize↔untilize round-trips. Lowest risk, in-repo.
2. **elementwise fusion (22%)** — fuse activation + routing-weight multiplies (BinaryNg/Unary chains).
3. **Multiple Command Queues** — overlap input writes / output readback with compute (tt-enable-tracing skill).
4. **commit path** — commit is eager 129 ms/tok; trace + batch it (batched-decode work) — likely the largest
   full-generation lever if a block commits many tokens. Audit block-time split (denoise Σ vs commit) next.
5. **datatype sweep** — bf8 experts (DRAM 11.6→~5.8 GiB/chip, faster matmul) if fidelity holds (datatype-sweep skill).

Roofline: per denoise step re-reads all resident weights (13.1 GiB/chip, 88.6% MoE experts) over the full
256 canvas — weight traffic, not incremental KV, sets the floor. `100 tok/s` needs the block-time split first.

## Block-time split (2026-07-08) — commit dominates, not denoise

| phase | per 256-token block | note |
|---|---:|---|
| denoise | ~11.2 s | ≤48 steps × 233 ms traced |
| **commit** | **~31–35 s** | **256 sequential single-token decode-appends** (one full 30L forward each) |
| ⇒ block | ~43 s → ~6 tok/s | commit is ~75% of block time |

**So the #1 lever is the commit, not the denoise step.** After denoise the 256 canvas tokens are
known; populating their KV is algebraically a *causal prefill of 256 tokens at start_pos*, not 256
autoregressive decodes.

## Batched commit — 24.8× prize, but currently numerically broken

`verify_commit_batching.py --num-layers 30` (cloned KV caches, seq vs batched, per-layer PCC):

- **speedup = 24.82×** (commit 35.1 s → **1.41 s**). This alone would take the block ~43 s → ~13 s ⇒ ~6 → ~20 tok/s.
- **PCC FAIL: 232/240 K/V checks fail.** Signature: **layer 0 K/V = 0.99999 (exact), then error compounds every layer** (L1 K 0.89, L5 0.41, L11 0.04, L24 V = −0.10). Classic accumulation.
- Diagnosis: layer-0 *written K/V* is correct, so embedding + K/V projection are right; the divergence is in the **attention output** (hidden state passed layer→layer). Suspects in `commit_batched.py`: the `build_device_commit_causal_mask` prefix+canvas causal pattern, canvas-Q RoPE offset, or the read-back-then-SDPA numerics — not the KV write.

### Two paths forward (next iteration)
1. **Reuse the proven `chunked_prefill.py` (PCC 0.9997)** for commit — it already does correct causal +
   sliding-window prefill at a non-zero `start_pos` (`chunk_start_idx` + `paged_fill_cache`). Commit ≡ prefill
   the 256 canvas tokens at `start_pos`; correct-by-construction, sidesteps the `commit_batched.py` mask bug. **Preferred.**
2. Debug `commit_batched.py`'s attention layer-by-layer (compare batched vs sequential attention output at layer 0,
   isolate mask/RoPE/SDPA). Higher risk.

Do NOT enable batched commit until PCC ≥ 0.997 (KV correctness gates block-to-block coherence).

## CORRECTION (2026-07-08 session B) — the batched commit is NOT buggy; `verify_commit_batching` is an invalid gate

The "batched commit numerically broken / attention output is the bug" diagnosis above is a
**misdiagnosis** and is retracted. Isolated on-device probes settle it decisively:

**1. The batched attention is correct.** New isolation probe `probe_attn_only.py` = `verify_commit_batching`
with `layer.enable_moe_block=False` forced on **both** paths, so the layer tail (shared_mlp + per-row
RMSNorms) is identical and the *only* remaining difference is attention:

| probe | worst K/V PCC | speedup | verdict |
|---|---:|---:|---|
| attention-only, 4 layers | **0.9977** (L3) | 13.8× | attention PASSES |
| attention-only, 30 layers | 0.494 (L29); ≥0.99 through L8, 0.96 @ L11, 0.86 @ L15, 0.71 @ L19 | 16.7× | see note |

Layer-0 attention KV is 0.99999 (and prior single-layer `probe_commit_l0attn.py` = 0.99992). The 30-layer
decay to 0.49 is **not an attention bug**: with MoE off the batched (prefill masked-SDPA + chunked RoPE) and
sequential (decode flash-SDPA + per-user RoPE) are different **bf16** kernels, and deep-network residual
feedback amplifies their tiny per-layer differences. ⇒ **No two non-bit-identical commit implementations can
meet `--pcc 0.997` at `--num-layers 30`.** The 0.997/30L threshold measures bf16 chaos-amplification, not correctness.

**2. The full-model failure is the MoE, and the *sequential* MoE is the defective one.** With MoE on (baseline,
`DG_SPARSE_MOE_CAPACITY=32`): 4L worst 0.612 (L3 V), 30L 232/240 fail. The torch-oracle gate
`probe_moe_vs_torch.py` (re-run this session, bit-exact layer-0 input, routing agreement 0.9969):

| MoE output vs HF torch oracle | PCC |
|---|---:|
| **batched** (`sparse_experts_forward`) | **0.856** ← higher = correct kernel |
| sequential (`_commit_experts_decode_forward`, decode `sparse_matmul` nnz=8) | 0.579 |

**VERDICT (device): BATCHED matches torch; SEQUENTIAL is the buggy kernel** — reproducing the 2026-07-04
resolution in `commit_batching.md`. `_commit_experts_decode_forward` is a near-verbatim copy of the shared
`models/demos/gemma4/tt/experts/decode.py::decode_forward` (same `sparse_matmul` + reshape/transpose), so the
inaccuracy is the gemma4 decode sparse-MoE kernel, which the batched path deliberately avoids (it uses the
accurate prefill/capacity-dispatch MoE).

**Consequence:** `verify_commit_batching.py` compares the **correct** batched path against the **defective**
sequential reference, so its FAIL is expected and meaningless (already declared a non-gate in
`commit_batching.md`). The correct gate is `probe_moe_vs_torch.py`. **No change was made to `commit_batched.py`:
it is not defective, and forcing `verify` to pass would require corrupting the fast path to reproduce the
defective sequential MoE.** Ruled out along the way: swapping the batched MoE to the sequential decode MoE is
impossible (decode `sparse_matmul` requires batch=1 — `sparsity volume must == 128`); `DG_SPARSE_MOE_CAPACITY=256`
(drop-free) overflows L1 in the tuned MoE matmul (EC=32768).

**Recommendation:** gate commit correctness on `probe_moe_vs_torch.py` (batched↔torch) and a *single-layer*
attention probe, not on `verify_commit_batching.py` (batched↔sequential, 30L, 0.997). Separately, the defective
gemma4 decode sparse-MoE still governs the paged/vLLM sequential commit (#47488) and is worth fixing there.

## Log
- 2026-07-08: baseline captured (traced 233 ms/step); op audit from whole_gen_opprofile; levers prioritized.
- 2026-07-08: block-time split → commit (~31 s, 256 sequential decodes) dominates. Batched commit verified 24.8× faster but KV PCC fails (232/240, layer-0-exact then compounds). Next: route commit through chunked_prefill (preferred) or debug commit_batched attention.
- 2026-07-08 (session B): retracted the "attention bug" diagnosis. `probe_attn_only.py` (MoE off, both paths) → attention-only KV PCC 0.9977 @ 4L (PASS), 0.494 @ L29 (bf16 prefill-vs-decode compounding, not a bug). `probe_moe_vs_torch.py` re-run → batched 0.856 vs torch, sequential 0.579 → **sequential MoE is the defective kernel**. `verify_commit_batching.py` is an invalid non-gate (correct-batched vs defective-sequential; also unreachable at 0.997/30L for any non-bit-identical pair). No `commit_batched.py` fix; recommend gating on `probe_moe_vs_torch.py`.

## Multi-step trace batching @48 — measured, REJECTED as default (2026-07-08)

Direction: **maintain current precision, optimize speed** (no dtype/step changes). Evaluated the one
remaining precision-neutral in-repo @48 lever, multi-step trace batching (`DG_DENOISE_TRACED_MULTISTEP`),
via `sweep_at48.py` at 30L, 48 steps, 3 blocks (device, main build). Steady-block t/s:

| config | steps | t/s | block s | committed_sha | verdict |
|---|---|---:|---:|---|---|
| traced_tuned_s48 (single-step, current default) | 48 | 17.82 | 14.365 | a9f0d18709b07d1e | baseline |
| multistep_g12_s48 (bounded window G=12, 4 replays/block) | 48 | 17.87 | 14.328 | **a9f0d18709b07d1e** | **bit-exact**, +0.3% (noise) |
| multistep_wb_s48 (whole-block, 1 replay) | 48 | — | — | — | **crash**: TT_FATAL buffer region overflow (traced_denoise_multistep_block) |
| traced_tuned_s12 (ref, fewer steps) | 12 | 42.55 | 6.017 | 24393ba7aad6077c | — |

**Verdict:** at 48 steps the block is **compute-bound** (48 × 30-layer MoE); multi-step batching only
removes per-replay host dispatch, which is negligible at @48 (+0.3%, bit-exact sha match confirms it
changes nothing numerically). Whole-block window overflows the trace buffer. **NOT made default** — no
@48 benefit, and wb crashes. It stays opt-in (it helps only the low-step regime where dispatch matters).

## Campaign conclusion (current precision, model-faithful @48)

The precision-neutral in-repo speed levers are **exhausted**: matmul-geometry (landed, verified
exhausted), terminal trim (landed), traced denoise loop (landed, 2.72×), multi-step batching (measured,
no @48 gain). **Model-faithful @48 ceiling ≈ 17.8 t/s.** Benchmark 100 t/s is already cleared at low step
counts (@4=104, @12=42.5). Closing the gap to model-faithful 100 t/s requires OUT-OF-GATE work: fewer
denoise steps by design (blocked by #48291 sparse-MoE fidelity → early-halt can't fire) and/or an
upstream fused MoE kernel and/or lower precision (bf8) — all excluded by the "maintain precision / no
shared-gemma4 edits" constraints.

## 2026-07-08 — bfp8 MoE experts (dg-07 datatype sweep): measured, FAILS fidelity gate, NOT landed

The campaign conclusion above flagged "lower precision (bf8)" as an excluded lever. It has now been
**measured** (full detail: `doc/datatype_sweep/`). DG-local knob `DG_EXPERTS_BFP8=1`
(`tt/precision_build.py`) flips ONLY the MoE expert gate/up/down weights bf16 → `bfloat8_b`; the
decision path keeps its existing BF16 production logits/entropy arithmetic (injected Gumbel noise
may be FP32). No `models/demos/gemma4/` edits.

**Decision agreement (bf16 vs bfp8, deterministic 16-step trajectory, 30L):** committed clean-argmax
agreement **0.227** (bar ≥0.95), mean entropy PCC **0.631** (min 0.036), mean accept/renoise IoU
**0.501** (min 0.0). Step-0 pure-logits argmax agreement 0.949 (~5% flip) compounds to 0.227 committed
(~77% of committed tokens change). Sample text a wash (coherent opening then #48291 degeneration, both
dtypes). **Fails all three bars.**

**DRAM:** 13.268 → **7.830 GiB/chip** (−5.44 GiB, −41%). **Traced throughput:** @48 18.18→**19.83**
(+9.1%), @24 31.49→**33.99**, @12 54.58→**57.84** — only ~6–9% (step not weight-bound; MoE matmul
launch/overhead-limited). 100 t/s crossover ~4.1 steps (bfp8) vs ~3.8 (bf16): negligible shift, and
4 steps is far below quality-acceptable.

**Verdict:** bfp8 experts **REJECTED** on diffusion-decision fidelity. bf16 experts stay selected; the
model-faithful **@48 ≈ 17.8–18.2 t/s ceiling stands**. The ~8% bfp8 speed win is not worth ~77% commit
divergence given #48291 leaves no fidelity headroom. Knob landed OFF-by-default for reuse if #48291 is
resolved.

## 2026-07-08 — data-dependent early-halt (dg-08 lever 8): mechanism landed, no-op under #48291

Full detail + design: `doc/optimize_perf/early_halt.md`. DG-local flag `DG_DENOISE_EARLY_HALT=1`
(`DG_DENOISE_EARLY_HALT_WINDOW=K`; 1 = scheme A per-step, K>1 = scheme B chunked-halt) recovers the
eager StableAndConfident early-halt inside the traced loop **without tracing the whole variable-length
loop**: capture a fixed K-step window, replay one window at a time, read ONE tiny on-device halt scalar
(mean entropy + argmax-stability mismatch — `tt/denoise_loop.py::write_halt_scalars`) and branch
continue/stop on host. NOT the retired 5-tensor/step readback (`bench_loop_readback.py` = 27.76 ms/step).
No `models/demos/gemma4/` edits. **Default OFF** (fixed-48 traced unchanged).

**Eager halt oracle (30L, `probe_halt_gap.py`, 3 blocks):** every block runs the full 48 steps,
`halted=False`. The **stability gate fires** (blocks 1–2 have 14–18 argmax-stable steps) but the
**confidence gate never does**: mean entropy floors at ~0.14–0.51 nats, ~30–100× the 0.005 threshold.
Early-halt is a measured no-op — a #48291 logit-distribution consequence, not a mechanism gap.

**Correctness (`probe_early_halt.py`, 6L + 30L):**
- Guard 1 (no-halt ≡ fixed-48): scheme-A commits BYTE-IDENTICAL to fixed-48 traced at 6L (`3d744378…`)
  and **30L (`a9f0d18709b07d1e`, = the established `traced_tuned_s48` sha)**.
- Guard 2 (forced-halt ≡ eager), **demonstrated firing at 30L** (elevated threshold so the confidence
  gate no longer blocks): eager and scheme A both halt block 1 at step 2 (`[48,2]`, `halted=[F,T]`) with
  byte-identical commit; scheme B(K=4) halts block 1 at the step-4 window boundary (`[48,4]`), same
  committed tokens (argmax stable across steps 1–4). Per-step device `(mean_entropy, mismatch)` vs eager
  records agree to **1.2e-5** (entropy) / **exact 0** (mismatch, integer token-id compare in fp32) over
  all 48 steps; unstable block 0 runs the full budget and stays byte-identical to fixed-48.

**Overhead + break-even (30L, N=48, traced steady block):**

| config | block s | t/s | overhead | break-even step |
|---|---:|---:|---|---:|
| fixed-48 traced (baseline) | 14.069 | **18.20** | — (step_dev 0.260 s, commit 1.57 s) | — |
| scheme A no-halt / real | 14.351 / 14.349 | 17.84 | **5.87 ms/step** (48 syncs) | 46.9 |
| scheme B K=4 no-halt | 14.406 | 17.77 | 28.1 ms/window (12 syncs) | 46.7 |
| scheme B K=8 no-halt | 14.277 | 17.93 | 34.7 ms/window (6 syncs) | 47.2 |

Overhead is ~2% of the block because the denoise steps are already device-serialized (canvas
data-dependency), so a per-step sync adds only a host round-trip, not a pipeline stall. **Break-even
≈47/48 ⇒ any early halt wins**; the whole cost is the ~2% no-halt overhead on blocks that run the budget.

**Verdict:** mechanism correct + ready (flag OFF by default). Under #48291 it never fires, so enabling
it is a ~2% net loss today. Flip when #48291 lifts the entropy floor below 0.005 (early-halt then fires
and beats fixed-48 for any converged block) or a schedule cut lowers the step budget. This is the
in-repo mechanism half of `path_to_100tps.md` lever 8; the quality half stays gated on #48291.

## 2026-07-09 — L1-residency pass (dg-08): full-canvas RMSNorm is a NEW +15.8% @48 lever

Full detail: `doc/optimize_perf/l1_residency.md` + `l1_residency_summary.json`. **First on-device
measurement of the "layout/glue (28%)" lever this log flagged as unexecuted** (this campaign ran
device-free; the box was free this session — 4x Blackhole p300c, `(1,4)` TP=4). Two DG-local flags
added, both **default OFF** (default path byte-unchanged); zero `models/demos/gemma4/` edits.

**HIGH-4 `DG_NORM_FULLCANVAS` (`tt/denoise_forward.py`) — LANDED opt-in, the pass's win.** Collapses
the 8×(32-row slice → gemma4 fast-path norm → DRAM-concat) chunked RMSNorm into ONE 256-row
width-sharded `rms_norm` (`block_h=8`, reuses `norm.tt_weight`). Per-norm micro **9.8×** (1.32 →
0.134 ms), PCC 0.999998. **Traced e2e (30L, seed 0): @48 17.855 → 20.676 t/s (+15.8%); @12 49.841 →
61.476 t/s (+23.3%)** — ~41 ms/step saved. It SURVIVES trace (removes real `Slice`/`Concat`/norm-launch
ops, unlike the MoE DRAM writes below). NOT bit-identical (`committed_sha` differs; `block_h=8` vs 8×
`block_h=1` bf16 reduction-order compounds under #48291), so opt-in default OFF pending a dg-05
decision-fidelity check to flip default. **This refutes the earlier "norm de-chunking is not fresh
headroom" note** — that was framed against the 137 ms/layer dense state; at tuned MoE the chunked-norm
slice/concat glue is ~15% of the step and is NOT overlap-hidden. So the "@48 ≈ 17.8–18.2 t/s
precision-neutral ceiling" is really ~20.7 t/s (per-row-identical math; only bf16 reduction-order
differs) once the norm glue is removed.

**HIGH-1/HIGH-2/MED-5 `DG_MOE_L1` (`tt/sparse_moe.py`) — MEASURED WASH, rejected as default.** Pin the
23 MB gather / 23 MB down MoE activation outputs L1 instead of DRAM. Isolated micro: full MoE fwd
−3.2% (gather matmul −57%), **bit-identical** (PCC vs off 1.000025, vs dense 0.99955 unchanged). But
traced e2e @48 18.128 → 18.016 (−0.6%), @12 53.213 → 53.421 (+0.4%) — a wash within noise, bit-identical
output. The DRAM round-trips are **overlap-hidden** under trace (the ~1.5–1.74× FW overlap); L1 reclaims
nothing on the critical path. MED-5 (gate/up L1) is a no-op (`batched_experts` weight-bound, flat).
Flag kept opt-in for reuse if a fused gather-experts kernel removes the overlap.

**Reasoned-closed:** HIGH-3 residual-stream L1 (coupled — every consumer takes DRAM → net-zero without
the full stack); MED-6 attention L1 (the `diffusion_attention.py:400-411` DRAM force is a guarded
passthrough no-op; real L1-SDPA blocked by the flash CB clash; attn ~1 ms/6L); MED-7 masks (~2 MB,
sub-ms). Measured: all `InterleavedToSharded`+`ShardedToInterleaved` in denoise = <3% of the step, so
conversion round-trips were never the lever — the chunked-norm slice/concat was.

**Watcher:** clean on `DG_NORM_FULLCANVAS=1` with `TT_METAL_WATCHER_DISABLE_ETH=1` (the plain-watcher
ACTIVE_ETH kernel-config-buffer overflow is the known watcher+fabric limitation, not a norm defect).

**Stage review (independent, xhigh): clean-pass** @ commit `fbabe620f21`. No P1/P2. All 6 headline
claims re-derived from the raw `RESULT_*`/`E2E_RESULT` logs; zero gemma4 files in the commit; the
non-bit-identical HIGH-4 win correctly landed opt-in default-OFF under gate D. Follow-ups addressed
post-review: (1) closed the reviewer's no-weight-branch gap — `bench_norm_fullcanvas.py` now tests a
`with_scale=False` stub (PCC 1.00005, per-row-equivalent, 4.8×) and logs `RESULT_NORM_KIND`, which
confirms `moe.router.norm` is the ONLY denoise norm taking that branch; (2) fixed the bench docstring
overclaim + a units typo + the doc/summary commit-constant mismatch (now both ~1.57s → ~41 ms/step);
(3) documented the branch-level `git diff main` gate provenance (stale local main, not dg-08) and the
default-flip / watcher-soak gates in `l1_residency.md` §Stage-review follow-ups.

**DG_NORM_FULLCANVAS default-flip gate (dg-05 decision fidelity) — RUN, FAILED → STAYS OPT-IN**
(full detail: `norm_fullcanvas_flip_gate.md` + `norm_fullcanvas_flip_agreement.json`). Ran
`decision_agreement.py` chunked(default) vs full-canvas, 30L / 16 steps, injected noise, clean argmax,
production MoE, everything pinned except `DG_NORM_FULLCANVAS`: **committed clean-argmax match 0.145**
(bar ≥0.95 — WORSE than the rejected bfp8's 0.227), argmax agree 0.544, accept IoU 0.504, entropy PCC
0.659 (the last two ≈ identical to bfp8's 0.501 / 0.631). The 2e-6/norm bf16 reduction-order difference
chaos-amplifies under #48291 (no argmax cushion, 30L×16 steps) to flip ~85% of committed tokens vs the
current default. Both trajectories coherent-then-degenerate (neither validated more faithful — the flip
changes *which* equally-(un)faithful bf16 output, not *whether* it is faithful). Per the flip rule
(hold vs chunked within the #48291 bar) 0.145 ≪ 0.95 → **default stays OFF; no code change.** A flip
would require an absolute HF-vs-TT check + ideally #48291 resolved. The +15.8% @48 win remains available
opt-in.

## 2026-07-09 — self-conditioning embedding prechunk: bit-exact +3.03% @48, default ON

Full evidence: `doc/optimize_perf/selfcond_prechunk.md`. A repaired synchronized component probe found
the self-conditioning soft embedding at **25.863 ms** in a two-layer real-checkpoint step, versus only
4.361 ms for LM head and 1.595 ms for the self-conditioning MLP. Its online softmax uses 32 ordered
8K-vocabulary chunks and was copying the matching 8K rows from the 256K×2816 embedding table before
every matmul.

`tt/self_conditioning.py` now builds the same BF16 table as 32 persistent 8K-row tensors and directly
feeds each existing matmul. It removes 32 device `Slice` operations per denoise step without changing
chunk boundaries, values, matmul geometry, or reduction order. The monolithic allocation is replaced,
not retained, so embedding-weight bytes are unchanged. `DG_SELFCOND_PRECHUNK_EMBED=0` is the diagnostic
opt-out; the selected path is default ON.

**Measured gates (QB2 TP=4, 30L, traced, three 256-token blocks):**
- synchronized soft embedding: **25.863 → 18.210 ms (-29.6%)**; two-layer full step 82.334 → 73.575 ms;
- @48 steady block: **13.9946 → 13.6555 s**, **18.293 → 18.747 t/s (+2.48%)**;
- @12 steady block: **4.3943 → 4.3401 s**, **58.258 → 58.984 t/s (+1.25%)**;
- 48/12 slope: **266.675 → 258.761 ms/warmed traced step (-7.914 ms, -2.97%)**;
- output identity: exact established SHAs at both budgets (`a9f0d18709b07d1e` @48,
  `24393ba7aad6077c` @12), each over three blocks.

Final exact-provenance processes reserved `DG_TRACE_REGION_SIZE=10737418240` before mesh open and
recorded every generation phase; the harness now rejects every other reservation and non-canonical
workload before opening the mesh. Control → unset default is **14.0971 → 13.6817 s** and
**18.160 → 18.711 t/s @48 (+3.03%)**; @12 is **4.4479 → 4.3710 s** and **57.555 → 58.568 t/s
(+1.76%)**. The resulting slope is **268.033 → 258.631 ms/warmed traced step (-9.403 ms,
-3.51% latency)**. Explicit `=1` reproduces 13.6996 s / 18.687 t/s. Full prefill + three-block
generation improves **155.4222 → 153.3410 s (+1.36%)**, and TTFT improves 127.227 → 125.977 s.
The earlier 18.819 t/s row is superseded by this self-contained final run.

This is precision- and decision-preserving, unlike full-canvas RMSNorm: no decision-fidelity waiver or
#48291 interpretation is needed. Three prompt-correct qualitative control/default runs also matched
bit-for-bit; complete outputs show existing #48291 generation defects are neither improved nor
worsened. Full-depth eager record gates injected identical initial canvases and 48 renoise tensors
into control/default and compared every per-step argmax, sampled ids, entropy, accept mask, next
canvas, and explicit clean commit candidate: **48/48 steps exact across all fields** in both RUN-first
argmax and production chunked-Gumbel modes. Proportional tests pass (40/40), including
the new partial final chunk and persistent-chunk lifecycle assertions; a separate watcher run passed
on the traced 4-step path with a non-aligned 24-token prompt; the batch-local no-shared-edits gate
passes versus the clean starting HEAD `0472860c40c`.

The persistent embedding payload remains exactly 1408 MiB/chip, but allocation topology changes from
one tensor to 32 (+31 allocations). A fresh full-depth `max_seq_len=262144` final-default eager
production chunked-Gumbel smoke completed all 48 steps and one 256-token block with 0.733 GiB/chip
still free. Full-depth *traced* 256K
is not currently composable: with no reservation the trace overlaps the DRAM high-water mark, while
176–512 MiB reservations leave no contiguous 128 MiB token-entropy temporary. This exact capacity
limit and the independently passing traced 1024-allocation canvas-feedback path are now explicit in
`doc/context_contract.json`; the advertised eager 256K capability is preserved. The traced controller
explicitly rejects real Gumbel noise, so the ranking remains labeled RUN-first argmax and no traced
production-Gumbel throughput is claimed.

## 2026-07-10 — dynamic logits `ttnn.split`: measured wash, rejected and removed

After prechunking the static embedding rows, the matching dynamic logits still use 32 sequential
8K-row `ttnn.slice` operations. A fresh candidate replaced them with one
`ttnn.split(..., 8192, dim=-1)` while preserving BF16 values, matmul shapes, and reduction order.
The synchronized soft-embedding component was unchanged (**18.2129 → 18.2116 ms**). Canonical
full-30L traced @48 steady blocks regressed **13.6284 → 13.6445 s** and
**18.784 → 18.762 t/s (-0.12%)**, with exact three-block commit digest
`a9f0d18709b07d1e`. The candidate's lower complete-generation total was entirely first-block
trace-capture variance; warmed execution did not improve. The code and selector were removed.
Exact commands and rows are in `selfcond_logits_split_rejection.md` and
`selfcond_logits_split_rejection.json`.

## 2026-07-10 — self-conditioning logits L1 chain: exact +0.71% final default

After rejecting `ttnn.split`, synchronized placement probes kept the existing 32 dynamic logits
slices but changed their output memory. L1 for only the slices improved the soft embedding
18.213 -> 17.721 ms; retaining `slice -> subtract -> exp`, denominator reductions, and the ordered
denominator accumulator in L1 improved it to **16.038 ms (-11.94%)** and the full two-layer probe
73.524 -> **71.359 ms**. Chunk matmuls, ordered numerator accumulation, and the final divide remain
in DRAM. The candidate changes only memory placement.

Three fresh independent DRAM-control @48 processes measured 13.6284, 13.6161, and 13.6051 s per
steady block. Two explicit L1-chain processes measured 13.4969 and 13.5253 s. The medians improve
**13.6161 -> 13.5111 s (-0.77%)** and **18.801 -> 18.9475 t/s (+0.78%)**, with exact
`a9f0d18709b07d1e` three-block commits. A same-model sequential A/B contradicted those
independent-process rows (the second session regressed 13.6456 -> 13.7841 s), and one explicit @12
sample also regressed within variance. Both are retained as limitations rather than hidden.

After all gates passed, `DG_SELFCOND_LOGITS_L1` was defaulted to `chain` with `off` as a diagnostic
escape. After the later 32K experiment and stage-review fixes were complete, the required fresh
process with all self-conditioning selectors unset reproduced **13.5849 s / 18.844 t/s @48**. The
paired final @12 result is **4.3122 s / 59.366 t/s**, implying **257.575 ms/warmed traced step**.
Against the prior selected default this is **-0.71% block latency / +0.71% throughput**. Complete
prefill plus three blocks was 153.9791 s versus 153.3410 s previously (**+0.42% regression**), so no
full-generation win is claimed. An intervening default control measured 13.6321 s and an earlier
post-cleanup run measured 13.5120 s; the final review-followup run is the conservative headline.

Stage review found that TTNN output inheritance also retains the denominator reduction/accumulator
in L1. Making that inherited placement explicit with output-memory arguments regressed the required
run to 13.6380 s / 18.771 t/s, so those arguments were removed. The inherited placement is now
source-commented and covered by a memory-aware unit test; documentation names the actual boundary.

Correctness is exact across all six recorded fields and the final commit for 48/48 steps under both
RUN-first argmax and production chunked-Gumbel with identical injected noise. Three prompt-correct
production-Gumbel A/B outputs are exact. Full-depth 30L / 48-step / 256K capability passes with a
non-aligned prompt; the separate traced watcher smoke reports four clean attaches/detaches and zero
error signatures. Proportional tests pass (41/41).

An adjacent mode that also placed numerator matmuls/accumulation and the final divide in L1 saved
only 0.375 ms beyond the selected chain in the component probe. Its traced attempt was discarded
because an external hardware test started after preflight and overlapped the run; the speculative
mode was removed and no contaminated number is evidence.

Full commands, raw rows, final-default policy, and limitations:
`selfcond_logits_l1.md` / `selfcond_logits_l1_e2e.json`.

## 2026-07-10 — larger self-conditioning vocab chunks: faster, decision-inexact, removed

A fresh selector swept the ordered online-softmax grouping from 8K to 16K, 32K, and 64K
vocabulary rows. The 32K point reduced soft embedding **16.038 -> 15.322 ms (-4.46%)** and won the
canonical warmed @48 run **13.6321 -> 13.5042 s / 18.779 -> 18.957 t/s (+0.95%)**. However, the
three-block clean-commit digest changed from the established `a9f0d18709b07d1e` to
`f224bc72c06ce5a0` under the identical canonical prompt, seed, and fixed 48-step workload.

The candidate therefore fails the minimum diffusion-decision identity gate despite its speed.
Changing chunk width changes BF16 matmul/reduction grouping. The selector and speculative source
were removed; the established ordered 8192-vocabulary path remains selected. Exact commands and
rows: `selfcond_vocab_chunk_rejection.md` / `.json`.

---

**Prior early-halt stage review (not this prechunk batch; independent, xhigh): clean-pass** @ commit `b88f2c361f8` (+ follow-up doc/comment
clarifications). No required work. Confirmed: on-device single-scalar halt (not the retired 5-tensor
readback), host branch == eager StableAndConfident rule, trace-safe capture, Guard 1 byte-identical,
Guard 2 eager-faithful firing (scheme A exact; scheme B commit correct under convergence-stability, so
scheme A is the default), honest #48291 no-op framing, self-consistent overhead/break-even arithmetic,
and the dg-08 commit adds NOTHING to `models/demos/gemma4/`. (Reviewer noted a pre-existing 1-line
`experts/operations.py` dealloc from dg-04 #47464 — out of dg-08 scope, flagged to that owner.)

## 2026-07-15 — FUSED2 default-on (bit-identical) + raw-denoise-vs-serving reframing

Traced A/B on the full 30L model (`bench_lever_e2e.py --levers frozen,frozen_fused2 --budgets 48,12`,
canonical prompt, frozen-prefix capture-once so steady blocks are pure replay; `/tmp/dg_fused2_ab.json`):

| lever | @48 | @12 | committed_sha |
|---|---|---|---|
| frozen (baseline) | 49.63 t/s (block 5.158 s) | 104.41 t/s (block 2.452 s) | 304e8023.. / 7dd2e2d9.. |
| frozen + FUSED2 | **50.76 t/s** (+2.3%) | **109.81 t/s** (+5.2%) | 304e8023.. / 7dd2e2d9.. (identical) |

`@48` and `@12` in this table are maximum budgets, not realized fixed step counts: early halt was
enabled and the `@48` runs completed after roughly nine denoise steps. These rows isolate the FUSED2
A/B only; they are invalid as replacements for the fixed-K=48 18.844 tok/s selected headline.

`committed_sha` matches exactly → `DG_MOE_DISPATCH_FUSED2` is bit-identical end-to-end (adds to the
existing `verify_dispatch_fused.py` [0:EC] gate). It is denoise-only (`build_capacity_dispatch` is
called only by `sparse_experts_forward`; the ragged prefill path does not use it) and DG runs a single
fixed MoE shape (S=256/E=128/C=256/top_k=8), so **flipped default ON** (`sparse_moe.py:_dispatch_fused2_enabled`;
`DG_MOE_DISPATCH_FUSED2=0` to revert).

**Reframing (important for where the remaining headroom is):** the raw traced denoise is already fast —
**~50 t/s @48 / ~104 t/s @12** at short context. The vLLM live sweep's 11–17 t/s
(`doc/vllm_integration/live_context_*_20260715.json`) is ~3.5× slower per block at similar/longer
context, i.e. the serving slowness is dominated by (a) per-step attention scaling with the frozen-prefix
length (the diffusion recompute reads the whole [prefix+256] every step) and (b) the vLLM
plumbing/commit overhead per block — NOT the denoise step compute. Per-step micro-levers (FUSED2 +2–5%,
skip-double-topk untried, entropy-accept chain) are incremental; the larger serving wins are early-halt
(now default-on, fires ~9–17/48 on real content) and the dg-09 vLLM-overhead / long-context attention
surface.

## 2026-07-15 (cont.) — per-component profile: the step is backbone-forward-bound

`prof_step_breakdown.py --num-layers 2 --iters 5` (synchronized, tuned-sparse config; `/tmp/dg_prof_breakdown.log`):

| component | ms | share of ~484 ms 30L step |
|---|---:|---:|
| 30-layer forward (14.43 ms/layer × 30) | ~433 | **89%** |
| terminal (sampling/entropy/accept over 262k vocab) | 28.99 | 6% |
| soft-embedding (self-cond) | 16.04 | 3% |
| LM head | 4.36 | 1% |
| self-cond gated MLP | 1.69 | <1% |
| **projected 30L step** | **483.86** | — |

Within one layer (14.43 ms): **MoE experts 2.63 ms (18%, ~roofline)**; the other ~11.8 ms is
attention/SDPA + o_proj + **2 TP all-reduce (CCL)** + norms + shared MLP.

**Reconciliation / correction:** the true traced per-step is ~447 ms (probe/vLLM 350–457) ≈ this
eager projection; the earlier "~50 t/s @48 / 3.5× vLLM overhead" was a mis-read — `bench_lever_e2e`
left `DG_DENOISE_EARLY_HALT` default-on so its "@48" halted at ~9 steps (5.16 s block), not 48. There
is no vLLM plumbing overhead; full-48 is ~11–12 t/s, early-halt ~9 steps is ~49 t/s.

**Bottom line for dg-08:** the step is **backbone-forward-bound** (per-layer attention + TP CCL), not
MoE (roofline), terminal, or serving glue. Remaining DG-local per-step levers are small increments
(FUSED2 +2–5% landed; skip-double-topk untried; terminal 29 ms has some room). The dominant per-layer
attention+CCL cost needs multi-week upstream fused kernels (fused gather-experts-combine, fused
matmul+CCL) — not DG-local and not a gemma4 edit. The biggest practical serving lever is early-halt
(fewer steps; already default-on, fires ~9–17/48 on convergent real content).

## 2026-07-16 — effective-capacity correction + trace-safe compact ragged denoise MoE

The July-15 attribution above had a capacity mismatch.  The measured layer forward used the
zero-drop production default `capacity=canvas_length=256`, but its isolated `moe_layer` call
hard-coded `capacity=32`.  That old micro dropped 41–84% of routes on concentrated real routing and
triggered a different tuned geometry.  Same-process QB2 measurement with the effective capacity
fixed gives **9.11–9.21 ms/MoE layer**, not 2.63 ms.  The profiler now resolves
`DG_SPARSE_MOE_CAPACITY` exactly like production, records max expert load/drop count, asserts zero
drops, and separately times the selected router+MoE block.

A new opt-in `DG_DENOISE_COMPACT_RAGGED=1` path removes the dense routing scatter, second top-k,
`[S,E*C]` dispatch matrices, and uniform `E*C=32768` expert rows.  A DG-local generic-op kernel
packs compact top-k metadata on device into fixed buffers: 128 primary 32-row expert segments plus
64 fixed overflow segments.  The primary bank reuses the existing C=32 roofline-tuned batched
matmuls; overflow uses the ragged-prefill sparse-matmul contract; combine reuses
`embedding + fast_reduce_nc`.  There is no host metadata/readback and the operation graph/buffers
are static under trace.

Evidence on QB2 TP=4:

- metadata kernel: watcher-clean and elementwise exact for slot-token, inverse-map, scaled route
  weights, and segment sparsity;
- reduced full denoise trajectory: watcher-clean for two steps through attention, compact MoE,
  terminal, and commit;
- selected router+MoE component: **9.63 → 5.62 ms/layer (-41.7%)**;
- full 30L traced, frozen prefix, fixed @12: **5.6844 → 4.3903 s/block**,
  **45.035 → 58.311 tok/s (+29.5%)**;
- full 30L traced, frozen prefix, fixed @48: **19.7898 → 14.2743 s/block**,
  **12.936 → 17.934 tok/s (+38.6%)**; the block delta is **114.9 ms/denoise step**;
- trace replay is healthy through three 48-step blocks.

The candidate remains **opt-in**.  It changes matmul/reduction geometry, so committed SHA differs.
The canonical full-30L 8-step HF gate observed committed match **0.90625** and failed the unchanged
strict `>0.95` production gate (which is already documented as mis-specified relative to the
intrinsic bf16 floor, but has not received owner sign-off).  Output remained coherent in the traced
A/B.  Do not default-enable until the gate policy is resolved or a numerically identical compact
expert kernel lands.

### 2026-07-16 follow-up — compact fidelity repaired

Focused A/B localized the regression to two BF16 reduction changes:

1. compact primary expert gate/up used K-block 22, while K-block 8 plus down
   K-block 2 reproduces the zero-drop C=256 expert rows elementwise;
2. `fast_reduce_nc` rounds each route contribution before summation, while the
   baseline combine matmul fuses multiply-accumulate.

The selected compact mode now uses reduction-compatible expert geometry and a
parallel compact-to-dense compatibility boundary.  Two small device kernels
build only the baseline combine matrix (without dense router/dispatch) and
scatter compact expert tiles into the baseline `[E*C,H]` layout; the existing
combine matmul then retains its exact accumulation contract.

Correctness:

- reduced 1L/2-step dense-vs-compact trajectory: all six fields exactly 1.0;
- full traced @12 and @48: committed SHA is identical to baseline
  (`4660b41d83efb4a4` / `304e8023feff5100`);
- canonical strict 8-step HF committed agreement is restored to the exact
  pre-change values: seed 0 **0.99609375**, seed 1 **0.9140625**.

Performance of the exact mode:

- fixed @12: 5.7309→5.5579 s/block, 44.670→46.061 tok/s (**+3.1%**);
- fixed @48: 19.7827→18.8807 s/block, 12.941→13.559 tok/s (**+4.8%**),
  saving **18.8 ms/denoise step**.

`DG_DENOISE_COMPACT_RAGGED=1` remains opt-in, but now defaults internally to
the exact `dense_compat` combine.  `DG_COMPACT_COMBINE=fast_reduce` retains the
decision-inexact +38.6% speed ceiling for kernel-development experiments only.

## 2026-07-16 — native SDPA frequency + exact canvas-norm spike

**Native SDPA is already the live path.** Per-layer fallback instrumentation and
`probe_sdpa_fallback_frequency.py` measured:

- short canonical prompt: **0/30 fallback layers**, all 30 native;
- 2013-token logical / 2016-token cache prefix: **0/30 fallback layers**, all 30 native.

The old "manual GQA fallback dominates" statement came from the June bring-up
allocator state and is not actionable on the current branch. No SDPA fallback
optimization was added.

**Bit-exact canvas RMSNorm spike:** stock batch reshape is rejected because a
width-sharded 256-row tensor requires shard height 256, and all tested
full-canvas compute-fidelity configs remain non-identical. The experimental
descriptor-fusion path can fuse four exact `slice + block_h=1 RMSNorm` phases;
two four-phase programs reproduce the eight-chunk output byte-for-byte. An
eight-phase program exceeds Blackhole's kernel-config buffer
(121152 B requested vs 70656 B).

That exact fusion is not shippable yet:

- reusing a fused descriptor deadlocks in an inter-phase semaphore wait;
- forcing a fresh descriptor/semaphore set is exact but costs ~90 ms/norm in
  eager host construction;
- creating the fresh semaphore/output state inside Metal trace fails with
  `Writes are not supported during trace capture`.

The experimental selector was removed rather than leaving a trace-crashing
path. Capturing the ~41 ms/step norm prize now requires a DG-local static
canvas-norm program with reset-safe barriers and preallocated trace outputs (or
a fix to the experimental fusion runtime), not another stock TTNN config.

A smaller exact fallback replaced eight `ttnn.slice` calls with one
`ttnn.split`: the isolated norm improved **1.430→1.321 ms (-7.6%)** and was
byte-identical, but full traced fixed-12 regressed **5.4344→5.5790 s/block**
(47.107→45.886 tok/s, **-2.6%**). Trace overlap already hides the individual
slice programs; the split selector was removed.

## 2026-07-16 — two-stage vocabulary reduction spike: slower, removed

A Diffulex-inspired DG-local generic-op candidate reduced each
`[32 rows, 2048 vocab]` chunk to clean-argmax and entropy statistics
`(max, Σexp, Σexp·shift)` while the logits were resident in L1, then merged the
128 chunk records in a second device stage. Stage 1 was load-balanced over all
110 available compute cores. The candidate read each logit tile from DRAM once,
used fixed-shape buffers only, and captured/replayed successfully under Metal
trace.

The reduced traffic did not translate to lower Blackhole latency. On the real
`[1,1,256,262144]` RUN-first terminal (`DG_DEDUP_ARGMAX=1`, 24 traced replays):

- released full-width composition: **28.3808 ms/step**;
- load-balanced two-stage, 1024-wide chunks: **35.1693 ms/step**;
- load-balanced two-stage, 2048-wide chunks: **35.4838 ms/step**.

The best two-stage row is **+6.79 ms / +23.9% slower**. TTNN's existing
full-width reductions already spread each primitive efficiently over the
device; the custom path replaces DRAM passes with repeated per-chunk
reduce/SFPU setup plus a second statistics merge, and that extra compute/control
cost dominates.

Production-shape synthetic decision evidence was mixed: clean argmax was exact
for all 256 positions and the budget-0.1 accept mask happened to match exactly
(4 accepted positions), but entropy was not byte-identical
(max absolute delta **0.203125**, mean **0.039354**) because chunk partial sums
reassociate the BF16 reduction. Since the candidate is both slower and
decision-inexact, no full-model run can make it selectable. The selector,
Python wrapper, and six stage kernels were removed; the released terminal is
unchanged.
