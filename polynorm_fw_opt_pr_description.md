## PolyNorm FW kernel compute optimizations

Four compute-kernel micro-optimizations for the fused PolyNorm3 forward pass, inspired by the preweighting approach used in the backward kernel ([review comment](https://github.com/tenstorrent/tt-metal/pull/42598#discussion_r3102126865) by @vmelnykovTT on the frobenius normalize PR). Each commit is a self-contained, reviewable step; all 10 `*PolyNorm*` unit tests pass at every step.

The FW kernel contract and interface are unchanged — this is purely kernel-internal work.

---

## Changes

1. **Fold `w_k` into `inv_rms_k` once per row** (`cb_weighted_coeffs`, `c_11`)
   - Previously, every output tile redundantly read `cb_w0/1/2` and multiplied by the weight scalar inside the Pass-2 inner loop.
   - New: `prepare_weighted_coeffs_for_row()` precomputes `w2·inv_rms_x`, `w1·inv_rms_x2`, `w0·inv_rms_x3` into a 3-tile CB once per row.
   - `emit_output_for_row()` drops 3 `copy_scalar(cb_wK)` + 3 multiplies per output tile.

2. **Fuse 3× `reduce_sum_to_inv_rms` → 1 acquire** (`reduce_sum_pows_to_inv_rms_triplet`)
   - The three row-reductions now share a single `tile_regs_acquire/commit` cycle.
   - Saves 2 acquire/commit/pack cycles, 2 add/sqrt/recip init sequences and 2 redundant `copy_tile(cb_eps)` calls per row.
   - Validates that `reduce_init` + multiple `reduce_tile` into different `idst` within one acquire works as documented (matches `ttnn::layernorm_sharded` pattern).

3. **Horner's method in `emit_output_for_row()`**
   - Rewrites the Pass-2 polynomial evaluation from direct form
     `y = coeff2·x³ + coeff1·x² + coeff0·x + bias`
     to Horner form
     `y = x·(coeff0 + x·(coeff1 + x·coeff2)) + bias`.
   - Cuts per-output-tile multiplications from 5 to 3 and frees one DEST register (4 → 3 used).

4. **`recip_tile<false>` (non-legacy)**
   - Switches `recip_tile` from `legacy_compat=true` to `false` for improved accuracy and speed ([frobenius normalize review](https://github.com/tenstorrent/tt-metal/pull/42598#discussion_r3102126865)).

---

## Benchmark

`polynorm_fusion_benchmark` (new op-level benchmark merged with BW) — wall-clock, timed region is one `polynorm3(...) + backward()` step with synthetic upstream grad, single sync at end of measured loop. Defaults: `WARMUP=2`, `MEASURE=5`, `BATCHES=1,2,4,8,16`.

| Model | S | C |
|------|--:|--:|
| tinyllama | 2048 | 5632 |
| undisclosed-model | 4096 | 4096 |

### Forward-fusion impact (backward stays composite)

| Model | Batch | Composite ms | FW fused ms | Saved ms | Speedup | Reduction |
|------|------:|-------------:|---------:|---------:|--------:|----------:|
| tinyllama | 1 | 18.59 | 15.29 | 3.30 | 1.22x | 17.77% |
| tinyllama | 2 | 34.22 | 27.63 | 6.59 | 1.24x | 19.25% |
| tinyllama | 4 | 66.03 | 52.91 | 13.12 | 1.25x | 19.87% |
| tinyllama | 8 | 128.87 | 102.73 | 26.14 | 1.25x | 20.28% |
| tinyllama | 16 | 259.30 | 205.15 | 54.16 | 1.26x | 20.89% |
| undisclosed-model | 1 | 26.04 | 21.03 | 5.01 | 1.24x | 19.23% |
| undisclosed-model | 2 | 48.49 | 39.07 | 9.42 | 1.24x | 19.42% |
| undisclosed-model | 4 | 94.87 | 76.05 | 18.83 | 1.25x | 19.85% |
| undisclosed-model | 8 | 187.35 | 149.40 | 37.95 | 1.25x | 20.26% |
| undisclosed-model | 16 | 370.41 | 294.08 | 76.33 | 1.26x | 20.61% |

**Takeaway:** with the new op-level benchmark methodology, FW fusion alone gives **1.22x-1.26x** speedup (**17.77%-20.89%** step-time reduction) while keeping the same kernel contract/interface.

### Cumulative FW-opt add-up (FW path only, per optimization commit)

Temporarily omitted to avoid stale numbers. This section will be repopulated only with freshly rerun staged benchmarks.

### CI Status
_Auto-generated on every push. Badges update live. Click a badge to filter runs by this branch._

- [![](https://github.com/tenstorrent/tt-metal/actions/workflows/sanity-tests.yaml/badge.svg?branch=mdragula/polynorm_fw_opt)](https://github.com/tenstorrent/tt-metal/actions/workflows/sanity-tests.yaml?query=branch:mdragula/polynorm_fw_opt)
- [![](https://github.com/tenstorrent/tt-metal/actions/workflows/blackhole-post-commit.yaml/badge.svg?branch=mdragula/polynorm_fw_opt)](https://github.com/tenstorrent/tt-metal/actions/workflows/blackhole-post-commit.yaml?query=branch:mdragula/polynorm_fw_opt)
- [![](https://github.com/tenstorrent/tt-metal/actions/workflows/tt-metal-l2-nightly.yaml/badge.svg?branch=mdragula/polynorm_fw_opt)](https://github.com/tenstorrent/tt-metal/actions/workflows/tt-metal-l2-nightly.yaml?query=branch:mdragula/polynorm_fw_opt)
- [![](https://github.com/tenstorrent/tt-metal/actions/workflows/pipeline-select.yaml/badge.svg?branch=mdragula/polynorm_fw_opt)](https://github.com/tenstorrent/tt-metal/actions/workflows/pipeline-select.yaml?query=branch:mdragula/polynorm_fw_opt)
- [![](https://github.com/tenstorrent/tt-metal/actions/workflows/pipeline-select-t3k.yaml/badge.svg?branch=mdragula/polynorm_fw_opt)](https://github.com/tenstorrent/tt-metal/actions/workflows/pipeline-select-t3k.yaml?query=branch:mdragula/polynorm_fw_opt)
- [![](https://github.com/tenstorrent/tt-metal/actions/workflows/pipeline-select-galaxy.yaml/badge.svg?branch=mdragula/polynorm_fw_opt)](https://github.com/tenstorrent/tt-metal/actions/workflows/pipeline-select-galaxy.yaml?query=branch:mdragula/polynorm_fw_opt)
