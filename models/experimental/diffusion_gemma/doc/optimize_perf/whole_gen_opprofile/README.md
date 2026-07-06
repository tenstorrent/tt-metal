# Whole-generation op-level profile — #47465 (CURRENT optimized DiffusionGemma)

Re-measurement of op-level device timing over the **whole generation** (prefill + the 48-step
denoise block + commit) on the current optimized model (`diffusion-gemma-function`, tip
`c890f7af7fb`). Supersedes the original #47465 profile, which timed **one denoise step on the
dense-128 path**. Since then the big optimizations landed: true-sparse token-gather MoE
(`tt/sparse_moe.py`), fused `transpose_b` QKᵀ (removed the Ktᵀ-materialization Permute that was
~97% of the old per-layer device-fw), traced serving loop, batched commit, OPT-004, dedup.

## Headline

Over the model-faithful whole generation (30 layers, canvas 256, **48 denoise steps/block**,
256 commit tokens/block, 1 block), by device-FW attribution:

| op-code | whole-gen % | what it is |
|---|---:|---|
| `SparseMatmulDeviceOperation` | **47.2%** | routed top-8 MoE gate/up/down (`tt/sparse_moe.py`) |
| `PermuteDeviceOperation` [6-D `{0;3;2;1;4;5}`] | **42.1%** | `cumsum(dim=2)` in `build_capacity_dispatch` — **FW-overlap artifact, ~1% real** (see CORRECTION) |
| `PermuteDeviceOperation` [4-D `{0;3;2;1}`] | 3.0% | attention head-layout permute |
| `TransposeDeviceOperation` | 2.0% | commit-decode path (`commit_decode.py`) |
| `BinaryNgDeviceOperation` | 0.9% | residual/RoPE/entropy elementwise |
| everything else (each) | <0.9% | CCLs, LayerNorm, Slice, ArgMax, SDPA, TopK, … |

**The whole generation is MoE-dominated by the expert matmuls (`SparseMatmul` 47%).** The 42%
`PermuteDeviceOperation` line above is a **sum-of-device-FW attribution artifact, NOT a real cost** —
see the CORRECTION below. The genuine bottleneck is the batched expert matmul: E=128 experts each
doing a skinny M=32 (1-tile) `[32,2816]@[2816,176]` matmul, weight-traffic-bound (all 128 experts
active at canvas 256), the shared-gemma4-MoE compute ceiling.

Phase split (device-FW): **DENOISE 94.02%**, COMMIT 5.74%, PREFILL 0.24%. The 48× denoise loop
utterly dominates, so the whole-generation op mix ≈ the denoise-step op mix.

## CORRECTION (2026-07-06, no-trace eager microbench — supersedes the 42% permute reading)

The 42.1% `PermuteDeviceOperation [6-D {0;3;2;1;4;5}]` was **mis-attributed and mis-weighted**.
Established by `ttnn.graph` op-capture (fast-runtime-mode OFF) + eager per-op timing at real shapes
(E=128, C=32, EC=4096, H=2816, S=256, TP=4 `(1,4)` mesh); probes in `~/dg-agent-runs/`
(`moe_graph_probe.py`, `dispatch_graph_probe.py`, `cumsum_microbench.py`).

1. **Wrong op.** It is NOT the "expert-group reshape". The `ttnn.reshape` `[1,1,EC,H]↔[1,E,C,H]`
   (sparse_moe.py L372/L378) is a **zero-copy metadata view** — graph capture shows *no* device op,
   eager time 0.006 / 0.003 ms. The 6-D permute is emitted by **`ttnn.cumsum(mask, dim=2)`**
   (L107, the exclusive-slot count in `build_capacity_dispatch`): cumsum along a non-last dim
   tilizes as `permute → cumsum → permute` = 4× `PermuteDeviceOperation` per call.

2. **Not 42% of real time.** Eager per-op: `cumsum(dim=2)` = **0.178 ms**, vs the full dispatch
   0.809 ms and **one** expert matmul 4.17 ms (3 big matmuls/layer). So the permute is **~1% of the
   MoE layer**, not 42%. The 42% is a **sum-of-device-FW overlap artifact** (the README's own caveat:
   FW windows overlap ~1.5–1.74×); the permute runs on data-movement cores and overlaps the matmul
   compute — **exactly the same trap as the old Ktᵀ "97%" permute that gave 0% when optimized.**

3. **A fix exists but is ~0.1% e2e.** Replacing `cumsum(dim=2)` with a lower-triangular matmul
   `L[S,S] @ mask[S,E]` (inclusive cumsum, **0 permutes**) is **0.0149 ms (92% faster), PCC 0.999995**
   (bit-exact — counts ≤256 are exact in bf16). But it saves ~0.16 ms/layer → ~0.24 s/block out of
   ~200 s = **~0.1% end-to-end** (and ~0% under traced overlap). Not landed: negligible win, and the
   dispatch feeds integer routing-column indices where a dtype slip is a correctness bug, not a perf
   bug. Kept here as a validated micro-lever if the dispatch is ever refactored.

**Bottom line:** no 42% permute win exists; it was a profiling artifact. The real lever remains the
weight-traffic-bound skinny-M expert matmul (`SparseMatmul`) — the shared-MoE ceiling.

## Method — reduced-layer 2-point fit (why not a single 30-layer capture)

The on-device profiler op buffer (`PROGRAM_SUPPORT_COUNT`, default 1000 → ~3.3k ops captured)
**cannot hold a 30-layer forward (~30k ops)**: a direct `--num-layers 30` Tracy run silently drops
device timing after the first contiguous ~3.3k ops (verified: prefill + <1 denoise step captured,
all 67k commit ops empty). This is the buffer overflow the `optimize` skill mandates avoiding.

So each repeating unit is profiled at two small layer counts that fit the buffer, with the buffer
raised via `TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT` (this sizes **DRAM**, not L1 — safe at low
layer counts), then linearly extrapolated to 30 layers and composed:

```
whole_gen = prefill(30L)·1  +  denoise_step(30L)·48  +  commit_token(30L)·256
```

The fit is well-conditioned: op **counts** scale exactly linearly (SparseMatmul ×3 from 2→6 layers,
ArgMax constant = per-step overhead), and at this short context (prompt+canvas ≈ 288 < sliding
window 1024) sliding- and full-attention layers attend to identical positions, so per-layer cost is
kind-independent. The 6-layer point still includes a full-attention layer (layer 5) as a check.

### Runs (Tracy build worktree `/home/zni/tt-metal-tracy`, `python -m tracy -r -p -v`, `--no-trace`)

| run | cmd | buffer | coverage |
|---|---|---|---|
| 2L | `prof_denoise_step.py --num-layers 2 --canvas-length 256 --iters 1 --commit-tokens 8` | 8000 | prefill/denoise 100%, commit 94.6% |
| 6L | `prof_denoise_step.py --num-layers 6 --canvas-length 256 --iters 1 --commit-tokens 4` | 22000 | 100% all phases |

`--commit-tokens` was added to `prof_denoise_step.py` so the commit phase (256 single-token
decode-appends, else >200k ops) fits the buffer; each commit token is an independent single-token
decode so per-token cost is canvas-independent and scaled ×256.

## Two device metrics (both from the same runs)

* **sum-of-device-FW per op-code → the OP MIX** (which ops dominate; standard tt-perf-report
  attribution). On the (1,4) mesh the per-op FW windows **overlap ~1.5–1.74×** (concurrent
  programs), so sum-FW > wall time and is used only for **relative** op share.
* **device-busy SPAN** (max FW-end − min FW-start, cycles / 1.35 GHz AICLK) → true per-phase device
  time. Validated to match the warmed wall-clock exactly (denoise 2L span 352 ms = wall 352.6 ms;
  6L span 916.9 ms = wall 917.5 ms; commit 6L span 110.3 ms = wall 110.8 ms).

## Important honesty notes

* **Model-faithful 48 steps.** The model runs the full 48 denoise steps; HF early-halt is a no-op
  under #48291, so steps are not reduced. Profiled at the real step count.
* **Eager, not serving speed.** `--no-trace` eager-under-Tracy timing exists only to attribute
  per-op device time; the supplementary span figures are profiler-inflated. The **actual** serving
  speed is the traced path (~17.92 t/s model-faithful, prior #47465 work), not the eager spans here.
* **Prefill span is cold** (JIT-compile-polluted, uncached — no warm-up before the measured
  region). Use prefill **device-FW** (~1 s one-time, negligible), not its span.

## Files

* `compose_whole_gen_opprofile.py` — reads the two raw Tracy CSVs, does the 2-point fit, composes
  the whole generation. Regenerate: run it with `--csv2 … --n2 2 --ncommit2 8 --csv6 … --n6 6
  --ncommit6 4`.
* `whole_gen_op_breakdown.txt` — the composed whole-generation table + per-phase op mix.
* `whole_gen_summary.json` — machine-readable summary.
* `phase_op_agg_2L.csv`, `phase_op_agg_6L.csv` — per-phase per-op aggregates (count, device-FW ms,
  device-kernel ms, op-to-op-gap ms; COMMIT normalized to **per commit token**). These are the small
  reproducible inputs; the multi-GB raw `ops_perf_results*.csv` are intentionally **not** committed
  (per the `optimize` skill artifact policy).

## Next optimization target (for #47465 follow-up)

The 6-D sparse-MoE token-gather **Permute** (23,100 ops / generation, 42% of device-FW) is now
co-dominant with the sparse matmul it feeds. Reducing/fusing that gather/scatter reshape in
`tt/sparse_moe.py` (or moving it into the sparse-matmul contract) is the single largest remaining
op-level lever, alongside the sparse-matmul geometry (OPT-004/OPT-014, LoFi already in use).
