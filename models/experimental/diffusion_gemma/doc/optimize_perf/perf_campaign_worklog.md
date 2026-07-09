# DiffusionGemma perf-optimization campaign (#47465 → goal 100 tok/s)

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
decision path (router/logits/entropy/argmax/accept) stays bf16/fp32. No `models/demos/gemma4/` edits.

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

---

**Stage review (independent, xhigh): clean-pass** @ commit `b88f2c361f8` (+ follow-up doc/comment
clarifications). No required work. Confirmed: on-device single-scalar halt (not the retired 5-tensor
readback), host branch == eager StableAndConfident rule, trace-safe capture, Guard 1 byte-identical,
Guard 2 eager-faithful firing (scheme A exact; scheme B commit correct under convergence-stability, so
scheme A is the default), honest #48291 no-op framing, self-consistent overhead/break-even arithmetic,
and the dg-08 commit adds NOTHING to `models/demos/gemma4/`. (Reviewer noted a pre-existing 1-line
`experts/operations.py` dealloc from dg-04 #47464 — out of dg-08 scope, flagged to that owner.)
