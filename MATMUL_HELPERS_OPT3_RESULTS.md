# Matmul helpers `opt_3` migration results

Branch: `wransom/matmul_helpers_opt_3`
Started: 2026-04-23
Hardware: Wormhole n150 (single-chip)

## Recipe

For each explicit `MatmulMultiCoreReuseMultiCast{,1D}ProgramConfig` in a model:

1. **Try the sentence-BERT pattern**: reshape `(out_subblock_h, out_subblock_w)` to maximize
   `h*w ≤ DST_capacity` (8 for half-sync bf16) with `h|per_core_M, w|per_core_N`, using
   `row_major_output=True` if the chosen `(h, w)` needs it (`h>1 AND w<per_core_N`).
2. **L1 fit check**: `row_major_output=True` forces separate `out_cb` / `interm0_cb` regions,
   roughly adding `per_core_M * per_core_N * tile_size` bytes to the static-CB footprint. If
   the compile asserts on `Statically allocated circular buffers clash with L1 buffers`,
   **fall back to the legacy-compatible max** — `(1, w)` with `w = largest divisor of per_core_N
   that fits DST`, no rmo flag.
3. **Skip lever 3 (K-iteration / `in0_block_w`)** — sentence-BERT did not use it; keep recipe
   uniform for breadth. Track as future-opportunity per model.
4. **Skip flag-only adds** where existing subblock is already at max DST volume and no reshape
   is available — the rmo flag alone is a measured no-op (+0.05% on sentence-BERT).

### Keep/revert gate

- PCC must pass (hard gate — revert if it fails).
- Device-kernel or e2e perf median must improve **≥5%** over baseline.
- Same commit updates both config file(s) and perf-test thresholds.

---

## Results table

| Model | Scope | Baseline | After | Δ | Configs touched | Commit |
|---|---|---:|---:|---:|---|---|
| sentence-BERT | total matmul device time (Tracy) | 13,385,817 ns | 10,515,489 ns | **-21.4%** | ff2+qkv reshape (4,2)+rmo; ff1+self_out flag-only | `0a421bb6a56` + threshold `7a89d6a8f55` |
| **Falcon7b** | prefill seq=1024 device-kernel sps | 3120 | 3742 | **+19.9%** | mm_h_to_4h `(1,8)`, mm_4h_to_h `(1,6)` — both legacy-compatible h=1 | pending |
| BGE-large | — | — | — | — | — | pending |
| BGE-m3 | — | — | — | — | — | pending |
| DistilBERT | — | — | — | — | — | pending |

Additional Falcon7b metric: e2e inference (median of 9 post-edit / 6 baseline runs at seq=1024): **0.368s → 0.307s (-16.6%)**. High run-to-run variance (8-9% std/mean) but effect size > 2× noise. Device-kernel measurement (<1% variance) confirms the win.

---

## Future opportunities (not pursued — tracked for follow-on PRs)

### Falcon7b

1. **mm_4h_to_h upgrade to `(4, 2)` + `row_major_output=True`**
   - DST volume 6/8 → 8/8 (expected ~5% additional kernel-time reduction on this op)
   - Blocked by L1 overflow: `Statically allocated circular buffers ... clash with L1 buffers`
     at rmo's forced `out_cb`/`interm0_cb` split
   - Workaround: drop `in0_block_w` from 8 to 4 to free L1 (in1 CB halves: 156→78 KB). Would
     double K-iterations (72→144), unclear if net positive — would need measurement.

2. **FUSED_QKV_MM_OPTIMIZED_PROGCFG** (`per_core_M=8, per_core_N=21`)
   - Commented-out developer intent: `out_subblock_w=1, # 7,` → legacy-compatible `(1, 7)` volume 7/8
   - Only reachable from `seq_len == 2048` branch, which is currently **skipped** (`#40015`)
   - Ready to unlock whenever that regression is resolved

3. **QKT_OPTIMIZED_PROGCFG / QKTV_MM_OPTIMIZED_PROGCFG lambdas**
   - Both have `out_subblock_h=1, # subblock_h,` and `out_subblock_w=1, # subblock_w` — caller's
     subblock args are ignored
   - For seq=1024 (num_slices=4, tiles_per_shard=10): QKT `per_core_N = seq_len/32 = 32` allows
     `(1, 8)` volume 8 (vs current 1), or `(2, 4)` rmo
   - QKTV `per_core_N = 2` is narrow; limited upgrade room
   - Requires un-hardcoding the lambdas AND updating callers to pass sensible subblock values
     (currently hardcoded to subblock_w=1 except for seq=2048)

4. **K-iteration tuning (lever 3)** on mm_h_to_4h (`in0_block_w=3` over Kt=144 → 48 iters) and
   mm_4h_to_h (`in0_block_w=8` over Kt=576 → 72 iters). Per the helpers plan, lever 3 can
   contribute 30-40% of synthetic-baseline wins on shapes with small `in0_block_w`. Would need
   L1 fit check at compile time and a delta measurement per config.

5. **LM head 2D mcast** (`falcon_lm_head.py`): subblock already (2, 4) volume 8/8 → flag-only
   no-op, skipped per recipe.

6. **Decode-path configs**: `MatmulMultiCoreReuseProgramConfig` lambda at `model_config.py:260`
   used in decode mode. Not surveyed — decode shapes and DST budget may have their own upgrade
   candidates but weren't in the prefill perf-test path.

### Generic follow-ons (apply to any model)

- **Un-skip skipped perf tests** (e.g., `test_perf_wh_bare_metal` prefill_seq2048 / `#40015`)
  to validate configs that only activate at those shapes.
- **L1 fit pre-check** — a sketch-level helper that estimates `in0_block_w × per_core_N × tile_size +
  per_core_M × per_core_N × tile_size × (2 if rmo else 1) + ...` before touching the device
  would let us bound which `(h, w)+rmo` combos are safe without compile-trial-and-error.
- **Auto-upgrade path for manual configs** (Part 7 item 2 of the opt_2 review): a tuner pass
  that augments user-passed configs (flip rmo to True + reshape subblock if L1 fits) would
  remove the need for manual edits entirely. Designed-but-unbuilt; ~1-2 days of work.

---

## Scope notes

- WH n150 single-chip constrains what I can run here. Models requiring T3000/TG/Galaxy/multi-chip
  (DeepSeek-V3, Llama3-70b-galaxy, the `t3000/` and `tg/` model demos) cannot be validated on
  this hardware and are **blocked on HW access**.
- Blackhole-specific demos (`models/demos/blackhole/`) are also out of scope for this branch.
- `models/experimental/` is typically stale and deprioritized unless actively maintained.
- The ~75 file total in the opt_2 review plan includes all of the above; my effective n150-feasible
  scope is closer to ~5-15 models.
