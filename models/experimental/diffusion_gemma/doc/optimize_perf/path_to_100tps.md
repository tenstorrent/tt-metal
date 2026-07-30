# DiffusionGemma — path to 100 t/s (roadmap arithmetic, #47465)

Status: roadmap arithmetic and lever provenance only. Its starting-line state (the token-gather MoE) was DELETED 2026-07-29, so no number here is a current result.
Owns: the 100 t/s target arithmetic, the per-layer budget table, the weight-floor headroom argument, and the in-repo vs upstream lever boundary.
See also: [refuted list](../REFUTED.md) · [stage hub](README.md) · [campaign ledger](perf_progress.md) · [weight roofline](work_log.md)

Absorbs `path_to_30tps.md` (deleted 2026-07-30).

## The arithmetic

`t/s = 256 / (ms_per_block / 1000)`, so **100 t/s ⇔ `ms_per_block ≈ 2560 ms`** and 30 t/s ⇔ ~8533 ms.
With a commit costing about one denoise-step-equivalent,
`ms_per_block ≈ (steps + 1) × (fixed_overhead + 30 × per_layer_ms)`.

| steps | step_ms budget | per_layer budget (fixed = 49 ms) | per_layer budget (fixed trimmed to 25 ms) |
|---|---|---|---|
| 48 | 52 ms | **infeasible** (step < fixed) | 0.9 ms (below the @256 floor) |
| 24 | 102 ms | 1.77 ms | 2.57 ms |
| 20 | 122 ms | 2.43 ms | 3.23 ms |
| 16 | 151 ms | 3.40 ms | 4.19 ms |

**STRUCTURAL CONCLUSION: 100 t/s is arithmetically impossible at the full 48-step budget**, because
the implied per-step budget (52 ms) is barely above the measured fixed overhead alone (49 ms). 100 t/s
is a **short-step-regime** target: it needs ~16–20 steps AND a per-layer of ~3–4 ms AND the fixed
overhead roughly halved — none alone suffices, they multiply.

## It is not a physics wall

At 21 steps the block sits at `21 × 12.3 ms = 258 ms` of all-128 weight traffic @1024 GB/s against a
2560 ms target — **~10× headroom over the weight floor**. 100 t/s is **NOT weight-floor limited**; the
whole distance is implementation efficiency. The weight-byte model, the all-128 floor and the
coupon-collector fact that a 256-token canvas activates essentially every expert (so top-k never cuts
weight bytes) are canonical in [work_log.md](work_log.md).

Expert weights are TP-sharded on `moe_intermediate` (704 padded to 96×4 = 768/chip):
`128 × 3 × (2816 × 704) = 761 M params/layer = 415 MB/chip/layer`.

Distance to that floor, as measured on the two historical paths: dense-128 at 4175.7 ms/step was **85×** the 49 ms @256 all-128 floor, and the token-gather path at ~379 ms/step was **~7.7×** it — i.e. the campaign moved the step from deeply op-count bound toward weight bound, and stopped there.

## Why the batched-expert matmul could not be tuned out (structural, outlives the deleted path)

On the token-gather path the batched-experts matmul read that 415 MB bank at only **~46 GB/s**, ~18%
of @256 and ~4.6% of @1024; the 6.6× gap decomposed into **~7.3 ms of untuned matmul inefficiency**
plus **~3.6 ms of dispatch / gather / combine / all-reduce overhead** on a 1.62 ms weight floor.

**The structural reason that outlives the path:** `M = 1 tile` (32 rows) per expert means **zero
weight reuse across M**, so a per-expert batched matmul is DRAM-read-bound at minimum arithmetic
intensity no matter how it is tuned. And realizing gather and combine as dense matmuls
(`[EC,S] @ [S,H]` and `[S,EC] @ [EC,H]` with `EC = 4096`) moves a 23 MB output and a 2 MB one-hot mask
that **carry no model information** — the strongest argument for a fused kernel.

## In-repo vs upstream boundary

A **fused gather-experts-combine kernel** and a **per-token / down-layout `sparse_matmul` variant**
are the only out-of-gate levers; everything else that was ranked here was DiffusionGemma-local, and
the token-gather campaign bought 13× **without touching a kernel**. The in-reader-gather fused-kernel
infeasibility verdict is in the [refuted list](../REFUTED.md).

## Honest verdict

The in-repo-only ceiling was **~60–80 t/s in the favourable regime**. 100 t/s for all prompts at 48
steps needs **fewer denoise steps by design** (a quality decision, not a perf edit) **plus** an
upstream fused MoE kernel.

> **OPEN CONTRADICTION (unexplained):** this roadmap's step-reduction lever assumed early halt is
> available, the rest of the tree recorded that it never fires under #48291, and under the concat MoE
> it fires at `[9,17,2]`/48 and K=10–43. See [early_halt.md](early_halt.md). Not explained.

## Superseded arithmetic kept as provenance

- The 30 t/s stacked-lever route is **refuted**: dense 1.10 → +true-sparse 5.15 → +commit batching
  11.28 → +24 steps 18.83 → +2CQ/dedup 21.65 t/s stops short of 30, and two of its stacked levers ran
  on paths later deleted. Its verdict — purely mechanical, all-prompts, no-quality-tradeoff 30 t/s at
  the full 48-step budget is **not reachable in-repo**; the robust routes are fewer denoise steps by
  design or a kernel-level fused MoE outside the module — still stands.
- Levers this doc once ranked that have since been answered: bfp8 experts (**rejected**, committed
  clean-argmax agreement 0.227), OPT-004 block-size tuning (**exhausted**), the sparse commit (never
  built as `tt/commit_prefill.py` — the batched commit shipped as `tt/commit_batched.py` instead).
  One line each in the [refuted list](../REFUTED.md).
