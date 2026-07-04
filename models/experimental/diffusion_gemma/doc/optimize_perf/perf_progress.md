# DiffusionGemma decode-throughput optimization — running progress

QB2 (`bh-qbge-06`, P150x4, mesh `(1,4)`, TP=4), branch `diffusion-gemma-function`, `build_Release`.
Optimization unit = **denoise step over the 256-token canvas** (≤48 steps/block) + **commit**.
Goal: raise decode throughput from ~1.3–2.3 t/s toward 30 t/s. One verified lever at a time.
All decode numbers are **traced** (eager is not acceptable evidence).

## Metric definitions (reported per lever)

- `ms_per_denoise_step` — warmed traced device time for one denoise step over the full 256 canvas, 30 layers.
- `steps_per_block` — denoise steps actually run before halt (≤ 48; data-dependent early-halt).
- `ms_per_block` = `steps_per_block × ms_per_denoise_step + commit_ms`.
- `tokens_per_block_per_s` = `256 / (ms_per_block / 1000)`.

## Baseline (established, prior work — see README.md / work_log.md / ttft_ts_sweep.md)

| quantity | value | source |
|---|---|---|
| TTFT (full 30L prefill) | 152–211 s | ttft_ts_sweep.md |
| ms_per_denoise_step (traced, 30L, projected from L=1/2/4) | **≈ 4176 ms** | README headline |
| — per-layer denoise | 137.55 ms/layer | README |
| — fixed overhead (embed+LMhead+terminal+norm) | 49.24 ms | README |
| commit (256 single-token decode-appends, 30L) | ≈ 31.5 s/block | README |
| ms_per_block @ 48 steps | ≈ 231.9 s | README |
| tokens_per_block_per_s @ 48 steps | **≈ 1.10** | README |
| roofline floor (weight traffic, 256 GB/s) | ≈ 24–49 ms/step | work_log §4 |

Per-step is context-independent (MoE weight-traffic bound); measured ~4176 ms is ~85–170× the
bandwidth roofline → **op-count / dense-MoE-compute bound**, large headroom.

## Target arithmetic

30 t/s ⇒ `ms_per_block ≈ 8533 ms`. At 48 steps that means ~170 ms/step incl. commit — a ~27×
reduction from 231.9 s/block. The denoise steps (200 s @48) dominate the block; commit is 31.5 s.

---

## Lever 1 — Traced-decode baseline / wire trace into measured path — **WASHOUT (recorded, not landed)**

**Question:** is the measured decode path traced? If eager, does wiring the trace-safe loop win?

**Findings (fresh, this session):**
- The measured decode path is **eager**. `serving.decode_block → denoise_and_commit_block →
  denoise_block` runs the eager loop with **5 host readbacks per step** (`argmax`, `entropy`,
  `sampled`, `accept_mask`, `canvas` → `to_torch`) plus a `torch.equal` stability halt check.
  Nothing wraps `run_fixed_denoise_steps` (the existing trace-safe device-only loop) in
  `begin_trace_capture`.
- **Quantified the readback overhead** apples-to-apples on the same reduced model
  (`bench_loop_readback.py`, L=2, 8 steps, canvas 256):

  | path | ms/step (L=2) |
  |---|---|
  | A: production eager `denoise_block` (host readback/step) | **359.27** |
  | B: trace-safe device-only loop, **traced** | **331.52** |
  | **readback overhead (A−B)** | **27.76 ms/step** |

- The 27.76 ms/step readback is a ~fixed per-step host cost (5 tiny 256-elem tensors + torch
  work), roughly layer-count independent. Against the 30-layer step (~4176 ms) it is **~0.66%**.
  This reconciles the README's "traced ≈ eager (~3%)" for the no-readback microbench.
- The whole-run tracy `GAP_FRACTION=81%` is a **device-profiler artifact**, not real host gap:
  clean wall-clock shows only 27.76 ms/step host overhead.

**Decision — do NOT wire a fixed-step trace into the measured path:**
1. The saving (~0.66% at 30L) is negligible vs the op-cost-bound step.
2. Tracing forces a **fixed 48-step** budget (early-halt cannot shorten a static trace). The eager
   path halts early when the canvas is stable+confident; converting to fixed-48 would **remove
   early-halt** and likely *regress* real blocks that converge in far fewer than 48 steps.

Lever 1 is therefore an honest washout for the denoise step. The genuine wins are the per-step
device compute (levers 2–4, dense-128-expert MoE) and the commit (256 sequential decode-appends).
Tracing may still help the **commit's** 256-decode inner loop and is revisited under lever 5.

_Artifacts:_ `bench_loop_readback.py`, `artifacts/lever1_readback_L2.log`,
`artifacts/lever1_traced_baseline_L2.log`.

---

## Lever 2 — Operation-topology audit — _in progress_

Clean 1-layer eager Tracy capture (`lever2_tracy_L1.log`, per-step over 3 steps). **Caveat:** the
device-profiler dropped many device rows ("no device perf row" skips) so `SparseMatmulDeviceOperation`
reads **0 ms in the denoise region** (clearly wrong — it is 16.27 ms for 3 ops in the prefill
region). MoE compute time is therefore under-counted; treat absolute op ms as lower bounds.

**Per denoise step (1 layer), reliable signal:**

| op | ms/step (1L) | %fw | note |
|---|---|---|---|
| `PermuteDeviceOperation` (MoE expert-major reorder) | **86.9** | 70% | biggest op; `transpose(gate/up,1,3)` feeding down `sparse_matmul` |
| `ArgMaxDeviceOperation` (terminal, 2×/step) | 20.2 | 16% | over 262144 vocab; already ROW_MAJOR-optimized; **fixed per-step** |
| `BinaryNgDeviceOperation` | 10.2 | 8% | mostly MoE elementwise (routing-mul, geglu) |
| `SparseMatmulDeviceOperation` (MoE dense-128 compute) | ~60 (missing) | — | inferred: prefill 16ms/32-tok × 8 chunks |
| `MatmulDeviceOperation` (attention/shared_mlp) | 0.8 | <1% | |

**Findings feeding the levers:**

1. **The MoE dominates the per-layer step (~80%)** = Permute reorder (~87 ms) + sparse-matmul
   compute (~60 ms). BOTH scale with **num_experts computed (128)**. The dense-128 pattern
   (`prefill.py`: `nnz = num_experts × group_size`, all-ones sparsity, routing applied *after* to
   zero 120/128) is the structural waste. The Permute is intrinsic to that pattern
   (`moe_transpose_investigation.md`: in-place rewrite = 0%, cost just relocates) — so the only way
   to shrink it is to compute **fewer experts** (lever 3), not to rewrite the transpose.
2. **`sparse_matmul` sparsity granularity is per-GROUP (per 32-token tile), not per-token.** The
   256-canvas is 8 chunks × 32 tokens; each chunk's 32 tokens share one sparsity row, so they need
   ~all 128 experts. True per-token top-8 needs `group_size=1` (256 tiny tiles → padding erases the
   win) OR a **token-gather rewrite** (sort tokens by expert). The gemma4 **decode** path already
   uses `nnz=top_k=8` (true sparse) but only for seq_len=1.
3. **Commit = 256 sequential single-token decode-appends** (`generate.py:623`). Each pays the full
   30-layer weight traffic → weight cost paid 256× instead of amortized once. It also computes a
   full 2816×262144 **LM head per token and discards it** (`commit_canvas_tokens` deallocates the
   logits). Batching the commit into one 256-token causal prefill-append is a ~256× weight
   amortization; skipping the LM head is a smaller safe sub-win.
4. **ArgMax is fixed per-step** (~20 ms), so it is ~0.5% of the 30-layer step — not worth chasing.

### Clean MoE microbench (`bench_moe.py`, non-profiled wall-clock, 256 canvas)

| quantity | value |
|---|---|
| expert weight dtype (actual) | **BFLOAT16** (DiffusionGemma overrides gemma4's bfp8 default) |
| MoE forward (router+experts) | 138.46 ms/layer |
| — router only | 3.21 ms |
| — **experts only (dense-128)** | **137.60 ms** |
| — shared_mlp only | 2.33 ms |

**The dense-128 experts forward IS the per-layer denoise cost** (~137 ms/layer ≈ README per-layer;
router/shared_mlp/attention are single-digit ms). At 30 layers this is ~99% of the ~4176 ms step.

**Roofline reconciliation (per layer, per chip @256 GB/s):** expert weights (bf16) ≈ 415 MB →
**~1.6 ms weight roofline**; dense-128 compute ≈ 106 GFLOP → **~2 ms compute roofline**. Measured
137 ms is **~65–85× above both** → the experts forward is **data-movement / op-overhead bound**
(the expert-major transpose reorder ~87 ms + 8-chunk op overhead), NOT weight- or compute-bound.

**Consequence:** BFP8/BFP4 expert precision does **NOT** help (weight traffic is only 1.6 ms; the
bottleneck is the transpose reorder). The only lever that cuts 137 ms/layer is computing **fewer
experts** (transpose + compute both scale with num_experts=128) — i.e. true-sparse.

### Lever-3 prototype (`bench_sparse_moe.py`): per-token nnz=8 — **BLOCKED**

Routing is exactly top-8/token (min=max=8 non-zero → the `nnz=8` invariant holds). But feeding the
256-token canvas to the decode path (per-token `nnz=top_k=8`) fails:
`TT_FATAL: sparsity.logical_volume() == batch_length` (sparse_matmul_device_operation.cpp:154).
`sparse_matmul` sparsity granularity is per M-tile-group, not per-token; per-token top-8 would need
`group_size=256` (each token its own 32-tile) → 32× M-padding that erases the 16× expert win.
**True-sparse therefore requires a token-gather rewrite** (sort tokens by expert → contiguous
per-expert batches), whose gather/scatter overhead is the mission's flagged washout risk. This is a
large, higher-risk rewrite scoped as the next big lever.

_Baseline confirmed:_ dense-128 = 137.08 ms/call. Artifacts: `lever2_moe_bench.log`,
`lever3_sparse_proto.log`.

### Lever-4 quick wins — ALL washed out or blocked (hard device evidence)

| candidate | result | evidence |
|---|---|---|
| BFP8/BFP4 expert precision | **won't help** — weight traffic is only ~1.6 ms/layer; MoE is transpose-bound | roofline reconciliation above |
| per-token `nnz=8` sparse (decode path on 256) | **BLOCKED** — `sparse_matmul` sparsity per M-tile-group, not per-token | `lever3_sparse_proto.log` |
| larger prefill chunk (64/128/256) | **BLOCKED** — `num_blocks_total <= num_cores_available` core overflow | `lever4_chunk_sweep.log` |
| MoE intermediates DRAM→L1 | **washout 1.01×** (137.09 → 135.14 ms), PCC 1.0 | `lever4_moe_l1.log` |

The in-repo DRAM copy reproduces the shared path exactly (137.08 ms, PCC 1.0), confirming the
transpose is an intrinsic slow-reorder kernel (not a DRAM-placement artifact); L1 barely moves it.

## Conclusion & remaining levers

**The wall is the dense-128 experts forward: 137 ms/layer × 30 ≈ 4.1 s = ~99% of the denoise step.**
It is data-movement/transpose bound (~65–85× above weight & compute rooflines), and every quick,
low-risk lever (trace, precision, per-token sparsity, chunk size, L1 placement) has been ruled out
with device evidence. The two remaining wins are both larger rewrites:

1. **True-sparse token-gather MoE (biggest lever).** Sort the 256 canvas tokens by their top-8
   expert assignments into contiguous per-expert batches, run a dense matmul per expert on only its
   ~16 tokens (padded to a 32-tile), scatter back weighted. Cuts experts computed 128→~effective-32
   (after tile padding) ⇒ cuts BOTH the transpose reorder and compute ~4–8×. Risk (mission-flagged):
   the on-device gather/permute + scatter overhead may erase the win — must be prototyped and
   measured before the full rewrite. Needs `sort`/`gather`/`scatter`/permute over 256×8 assignments,
   variable per-expert token counts (capacity/pad handling), all TP-sharded and trace-safe.
2. **Commit batching (256 decodes → 1 causal prefill-append).** The commit re-encodes the finished
   256-token canvas as 256 sequential single-token decode-appends (`generate.py:623`), paying the
   full 30-layer weight traffic 256× (≈31.5 s/block). One causal 256-token prefill-append writes all
   KV in a single backbone pass (≈4.2 s, one denoise-step-equivalent) → ~7× on the commit, ~1.25×
   on the block at ~25 steps. Needs a DiffusionGemma-local causal prefill-append at offset `start_pos`
   (analogous to the existing `commit_decode.py`), verified by KV-cache PCC vs the 256-decode path.
   Also skip the discarded final-norm+LM-head in the commit (small safe sub-win, ~3% of commit).

Neither lever alone reaches 30 t/s (the step's 30× dense-MoE is fundamental); the token-gather MoE
is the only structural path to a materially cheaper per-step, and is the correct next investment.
