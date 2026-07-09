# DiffusionGemma — L1-residency pass on the denoise hot path (dg-08, #47465)

**Objective:** raise output tok/s by keeping the denoise-step activation/residual/norm/attention
intermediates **L1-resident** across op boundaries instead of round-tripping DRAM, spilling to DRAM
only when a tensor cannot fit. DiffusionGemma-local only; zero `models/demos/gemma4/` edits.

**This is the FIRST on-device measurement of these levers.** The prior campaign
(`path_to_100tps.md`, `perf_campaign_worklog.md`) ran device-free ("the QB2 box is owned by another
agent") and named "layout/glue in `sparse_moe.py` (28%)" as the top *unexecuted* in-repo lever, then
declared the precision-neutral in-repo ceiling at ~17.8–18.2 t/s @48. This pass had a live 4-chip
Blackhole box (P150x4, mesh `(1,4)`, TP=4, `ENABLE_TRACY=OFF`) and settles the layout/glue levers
with measured before/after — including the ranking metric the deliverable requires (**traced**
steady-state tok/s), not just isolated-op timing.

## TL;DR verdict

| lever | flag | isolated micro | **traced e2e @48** | landed |
|---|---|---|---|---|
| **HIGH-4 full-canvas RMSNorm** | `DG_NORM_FULLCANVAS` | **9.8× / norm**, PCC 0.999998 | **17.86 → 20.68 t/s (+15.8%)** | **opt-in (default off); recommend flip pending decision-fidelity** |
| HIGH-1 gather + HIGH-2 down L1 | `DG_MOE_L1` | MoE fwd −3.2% (gather −57%), bit-identical | 18.13→18.02 (−0.6%), 53.2→53.4 @12 (wash) | opt-in, default off |
| MED-5 gate/up L1 | `DG_MOE_L1=chain` | `batched_experts` flat | no-op by construction | no |
| HIGH-3 residual-stream L1 | — | coupled (every consumer takes DRAM) | not measured standalone | no |
| MED-6 attention L1 | — | DRAM-force is a guarded passthrough no-op | not pursued (flash-SDPA CB clash) | no |
| MED-7 mask L1 | — | ~2 MB masks, sub-ms | not pursued | no |

**Headline finding:** the objective's premise (DRAM round-trips are reclaimable headroom) is
**correct — but the reclaimable round-trips are the chunked-RMSNorm slice/concat glue, NOT the MoE
matmul activation round-trips.** Collapsing the 8×(32-row slice → norm → DRAM-concat) RMSNorm to one
256-row width-sharded norm lifts the model-faithful @48 step from **17.86 → 20.68 t/s (+15.8%)** and
@12 from **49.8 → 61.5 t/s (+23.3%)**. The MoE gather/down L1 levers, by contrast, are a measured
**wash** because their DRAM writes already overlap adjacent compute under trace. Both were only
distinguishable **on device**.

## Method / measurement (ENABLE_TRACY=OFF substitute)

`ENABLE_TRACY=OFF`, so no `tt-perf-report` op-CSV. Per the playbook the substitute is the
**traced Metal capture/replay path** (the ranking metric IS traced, only op-attribution CSV is
unavailable) plus **synchronized per-op/per-component device-time tables**
(`time.perf_counter` + `ttnn.synchronize_device`). Evidence class: `hardware-profiler-limited`;
trace capture/replay works. Harnesses (all under `doc/optimize_perf/`):
- `bench_moe_l1_residency.py` — isolated tuned-MoE per-component + full-fwd timing + PCC (2L).
- `bench_norm_fullcanvas.py` — per-norm chunked-vs-full-canvas timing + PCC (2L).
- `bench_moe_l1_e2e.py`, `bench_lever_e2e.py` — traced 30L steady-state tok/s per lever, with
  `committed_sha` (bit-for-bit output identity check) and coherence text head.

## Baselines reproduced (traced, model-faithful, 3-block steady, seed 0)

Baselines reproduce the campaign's established `committed_sha`, confirming the harness measures the
true model-faithful path (run-to-run block latency varies ~1.5%; the norm win is far above that):

| budget | t/s | steady block s | committed_sha | matches |
|---|---:|---:|---|---|
| @48 (model-faithful) | 17.86–18.13 | 14.12–14.34 | `a9f0d18709b07d1e` | worklog `traced_tuned_s48` |
| @12 | 49.8–53.2 | 4.81–5.14 | `24393ba7aad6077c` | worklog `traced_tuned_s12` |

## HIGH-4 — full-canvas RMSNorm (`tt/denoise_forward.py`, `DG_NORM_FULLCANVAS`) — the win

**Mechanism.** DiffusionGemma chunks the 256-row canvas into 8× 32-row slices
(`_chunked_norm_forward`/`_rms_norm_dram`) **specifically so each slice hits gemma4 RMSNorm's
width-sharded fast path** (`rms_norm.py::_forward_sharded`, `block_h=1`, 32-row-only); calling
`norm.forward` on the full 256 rows falls to the slow plain-interleaved path. That costs **7 extra
slices + 1 DRAM concat + 7 extra sharded-norm launches + 8 I2S/S2I round-trips per norm call**, and
there are ~6–8 norm calls/layer × 30 layers. `DG_NORM_FULLCANVAS=1` runs **one 256-row width-sharded
`rms_norm` (`block_h=8`)** reusing `norm.tt_weight` (reading the weight is data-use, not a gemma4
edit) and hands the L1 output straight back. RMSNorm is per-row independent of `block_h`, so the math
is per-row equivalent.

**Isolated micro** (`bench_norm_fullcanvas.py`, 2L, 40 iters). Covers BOTH `_chunked_norm_forward`
branches — the weighted gemma4 fast-path AND the `with_scale=False`/`tt_weight=None` branch
(`_rms_norm_dram` vs `_fullcanvas_norm(weight=None)`), exercised by a `_NoWeightNorm` stub. A model
scan (`RESULT_NORM_KIND`) confirms exactly one denoise-path norm takes the no-weight branch at >32
rows — **`moe.router.norm`** (`with_scale=False`); every layer norm (input/post-attn/pre-ff/post-ff,
moe post-ff) is weighted:

| norm | chunked ms | full-canvas ms | speedup | PCC |
|---|---:|---:|---:|---:|
| input_layernorm (weighted) | 1.31 | 0.15 | **8.6×** | 0.999994 |
| post_feedforward_layernorm (weighted) | 1.32 | 0.14 | **9.2×** | 1.000015 |
| **no-weight stub** (= `moe.router.norm` branch) | 0.68 | 0.14 | **4.8×** | 1.000050 |

(PCC values ≈1.0 both branches — per-row equivalent; the ~2e-6 delta is the bf16 reduction-order noted
below. Run-to-run the weighted speedup measured 8.6–9.8×.)

**Traced e2e** (`bench_lever_e2e.py`, 30L, `baseline` vs `norm`, seed 0):

| budget | baseline t/s | **norm t/s** | Δ t/s | baseline block s | norm block s | committed_sha |
|---|---:|---:|---:|---:|---:|---|
| @48 | 17.855 | **20.676** | **+15.8%** | 14.3376 | 12.3815 | a9f0d18709b07d1e → **ead6eaa16dee8a57** |
| @12 | 49.841 | **61.476** | **+23.3%** | 5.1364 | 4.1642 | 24393ba7aad6077c → **dbb6d6f142940846** |

Derived per-step (subtracting the ~1.57 s commit measured in `early_halt.md`): baseline ≈ 0.266 →
**norm ≈ 0.225 s/step — ~41 ms/step saved (~15%)**. Output stays coherent ("a diffusion language model
is a generative model that produces text by iteratively refining a sequence of random noise into
coherent language…"). The block-latency delta (−1.956 s/block @48) is ~13× the ~1.5% run-to-run
baseline noise (block 14.12–14.34 s across runs), so the win is well above the noise floor.

**Caveat — NOT bit-identical.** `committed_sha` differs (per-norm PCC 0.999998, not 1.0): `block_h=8`
vs 8×`block_h=1` uses a different **bf16 reduction/accumulation order** in the sharded LayerNorm
kernel; under #48291 (the model commits the clean argmax with no cushion) that ~2e-6 per-norm delta
compounds over 30L × 48 steps and flips some argmax decisions. This is the **same bf16
chaos-amplification** the campaign documented for batched commit (`commit_batching.md`: "no two
non-bit-identical bf16 kernels meet 0.997 at 30L") — a per-row-equivalent, precision-neutral
(same dtype/fidelity) numerical path, not a precision reduction.

**Landing decision.** The objective's correctness gate D requires **bit-identical** argmax/accept
before/after; HIGH-4 does not meet it, so it lands **opt-in, `DG_NORM_FULLCANVAS` default OFF** — the
default path is byte-unchanged. It is the largest in-repo denoise lever found since the true-sparse
MoE, and directly refutes the campaign's "norm de-chunking is not fresh headroom" note (that was
framed against the 137 ms/layer dense state; at tuned MoE the chunked-norm slice/concat glue is a real
~15% of the step and is NOT overlap-hidden). **Recommended follow-up to flip the default ON:** a dg-05
decision-fidelity check that `DG_NORM_FULLCANVAS` agrees with the torch/HF reference as well as the
baseline does (both ~0.84 under #48291) — i.e. that the flip changes *which* equally-valid bf16 output,
not *whether* it is faithful.

## HIGH-1 / HIGH-2 / MED-5 — MoE token-gather activation L1 (`tt/sparse_moe.py`, `DG_MOE_L1`) — wash

**Mechanism.** The token-gather MoE writes two large activation tensors to DRAM and re-reads them:
the gather matmul output `dispatched` `[1,1,EC=4096,H=2816]` = **23.1 MB** (re-read 46 MB by gate+up),
and the down matmul output `down` `[1,E,C,H]` = **23.1 MB** (re-read 23 MB by combine).
`DG_MOE_L1 ∈ {off,gather,down,both,chain,all}` pins those outputs `L1_MEMORY_CONFIG` instead of DRAM.
Pure output-placement change; default `off` = bit-identical DRAM path.

**Isolated micro** (`bench_moe_l1_residency.py`, tuned, 2L, 30 iters):

| component ms/layer | off (DRAM) | gather | down | both |
|---|---:|---:|---:|---:|
| gather_matmul | 0.101 | **0.043** | 0.098 | 0.043 |
| combine_matmul | 0.092 | 0.092 | **0.077** | 0.077 |
| batched_experts | 1.776 | 1.771 | 1.762 | 1.756 |
| **full MoE fwd** | **2.878** | 2.818 | 2.851 | **2.786 (−3.2%)** |
| PCC vs off / vs dense | — / 0.99955 | 1.000025 / 0.99955 | 1.000025 / 0.99955 | 1.000025 / 0.99955 |

L1 nearly halves the gather matmul (removes its 23 MB DRAM write); `batched_experts` (weight-bound at
M=1 tile, 62% of the MoE) does not move → **MED-5 (gate/up L1) is a no-op** and the MoE is
weight-traffic-bound, not activation-round-trip-bound.

**Traced e2e** (`bench_moe_l1_e2e.py`, `off` vs `both`):

| budget | off t/s | both t/s | Δ | committed_sha (off == both) |
|---|---:|---:|---:|---|
| @48 | 18.128 | 18.016 | **−0.6%** | `a9f0d18709b07d1e` (bit-identical) |
| @12 | 53.213 | 53.421 | **+0.4%** | `24393ba7aad6077c` (bit-identical) |

**Verdict: WASH, REJECTED as default, kept opt-in.** Straddles zero (±0.5% noise), bit-identical
output. The isolated ~2.8 ms/step MoE saving is **overlap-hidden** under trace (the profile's
~1.5–1.74× FW overlap — the matmul DRAM writes already overlap adjacent compute). A block-sharded L1
variant (the objective's expert-major re-lay) would add a *reshard* on top of a win that is already
overlap-hidden, so it cannot recover the gap. Flag retained (bit-identical, trace-safe) for reuse if a
fused gather-experts kernel ever removes the overlap that hides the win.

## HIGH-3 / MED-6 / MED-7 — not pursued (reasoned closure)

- **HIGH-3 residual-stream L1** is **coupled**: every consumer of the 256×2816 residual (input/post-
  attn/pre-ff norms, attention entry, MoE entry) currently takes a DRAM-interleaved input, so pinning
  the residual L1 alone just inserts a reshard at each boundary — net-zero unless the norms
  (HIGH-4), MoE entry (MED-5), and attention (MED-6) all consume L1 too. That is a full coherent-layer
  L1 rewrite; MED-5 is a measured no-op and MED-6 is blocked (below), so the residual-only lever has no
  standalone win. (This is the OPT-003 residual-contract rule: it pays off only as a stack.)
- **MED-6 attention L1**: the `to_memory_config(..., DRAM)` force at `diffusion_attention.py:400-411`
  is **guarded** (`if tt_q_dram is not tt_q`) — RoPE/concat already output DRAM-interleaved, so it is a
  passthrough no-op, not "anti-L1 overhead." A real L1-sharded SDPA is blocked by the flash-SDPA CB
  clash (documented), and attention is only ~1 ms/6L of the denoise step, so it is not worth the risk.
- **MED-7 mask L1**: the disp/comb/disp_t masks are ~2 MB and the ops are sub-ms; the disp_t transpose
  is inside the (already small) dispatch-build glue. No material headroom.

## Why the conversion round-trips are not the (MoE) lever (measured)

The whole-denoise `InterleavedToSharded` + `ShardedToInterleaved` device-FW is **~1.34 ms over 6
layers (≈4 ms/step over 30L, <3% of the step)** — and that is ALL conversions (attention + norm +
MoE), not just the MoE (`whole_gen_opprofile/phase_op_agg_6L.csv`). The MoE activation round-trips do
not appear as explicit conversion ops — the matmuls write/read DRAM-interleaved directly, folded into
matmul device time — and under trace that time overlaps adjacent compute, which is why HIGH-1/2 are a
wash. The reclaimable glue was the chunked-norm **`Slice` (2.5 ms/6L) + `Concat` (1.26 ms/6L) +
redundant `LayerNorm` launches**, which HIGH-4 removes.

## Roofline reconciliation (unchanged by this pass)

Per denoise step the model re-reads the full resident weight bank (13.27 GiB/chip, ~88.6% MoE
experts) over the 256 canvas — weight traffic, not incremental KV, sets the floor (all-128 experts
active at S=256; top-8 buys compute/data-movement, never weight bytes). The all-128 bf16 weight floor
is ~12.3 ms/step @1024 GB/s peak; the measured ~0.23–0.27 s/step is op-efficiency-bound above that,
dominated by the weight-bound expert matmul (~92% of the 256 GB/s roofline, immovable in-repo) and the
terminal argmax/entropy over the 262144 vocab (blocked in-repo by the 18-bit-index fp32-reduction
wall). HIGH-4 attacks neither of those; it removes the norm/slice/concat glue layered on top, which is
why it is a real +15.8% while the MoE-activation levers are a wash.

## Stage-review follow-ups / residual risk

- **gemma4 gate — this commit is clean; the branch-level `git diff main` is not (pre-existing, not
  dg-08).** `git diff-tree fbabe620f21 -- models/demos/gemma4/` is **empty** — the dg-08 commit touches
  only DiffusionGemma-local files. The literal `git diff main -- models/demos/gemma4/` reads non-empty
  only because local `main` is ~842 commits stale (bulk = merged upstream Gemma4 PRs) plus the
  separately-owned DiffusionGemma footprint edits (#47464 commit-decode, the 1-line experts dealloc =
  optimize-playbook ceiling #4 / plan.md R0.4/R-new). Both are cross-stage and out of dg-08 scope; the
  meaningful commit-scoped invariant (dg-08 adds nothing to gemma4) holds. Fast-forwarding local `main`
  would make the automated `git diff main` gate checkable again.
- **Default-flip gate for `DG_NORM_FULLCANVAS`.** Flipping the default ON must first clear a dg-05
  decision-fidelity check that the full-canvas norm agrees with the torch/HF reference as well as the
  chunked baseline does (both ~0.84 under #48291) — i.e. that the flip changes *which* equally-valid
  bf16 output, not *whether* it is faithful. The no-weight branch (`moe.router.norm`) is now
  isolated-PCC-verified per-row-equivalent (1.00005), so the only open item for the flip is the
  compounding non-bit-identity of the committed argmax, which is exactly what the fidelity check settles.
- **Watcher scope.** `DG_NORM_FULLCANVAS=1` was watcher-verified on a short smoke (4 steps / 1 block,
  `TT_METAL_WATCHER_DISABLE_ETH=1`): the full-canvas `rms_norm` + I2S/S2I kernels all execute, watcher
  attaches/detaches clean, zero violation strings in `generated/watcher/watcher.log`. A full @48 /
  multi-block watcher soak is deferred (acceptable for an opt-in lever; do it as part of the default-flip
  gate). Raw logs backing every number here live in the run scratchpad (not committed, per the
  artifact policy); the committed docs/JSON mirror them to the digit.
