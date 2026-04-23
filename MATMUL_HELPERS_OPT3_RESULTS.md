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

### Explicit-config manual migration

| Model | Scope | Baseline | After | Δ | Configs touched | Commit |
|---|---|---:|---:|---:|---|---|
| sentence-BERT | total matmul device time (Tracy) | 13,385,817 ns | 10,515,489 ns | **-21.4%** | ff2+qkv reshape (4,2)+rmo; ff1+self_out flag-only | `0a421bb6a56` + threshold `7a89d6a8f55` |
| Falcon7b | prefill seq=1024 device-kernel sps | 3120 | 3742 | **+19.9%** | mm_h_to_4h `(1,8)`, mm_4h_to_h `(1,6)` — both legacy-compatible h=1 | `0fbd77527c2` |
| BGE-large | — | pending | pending | pending | self_out `(1,4)→(2,4)` legacy — only candidate (rest capped by `fp32_dest_acc_en=True` DST=4) | pending; blocked on HF weights |
| demos/bert | device-kernel sps | 252 (declared) | pending | pending | qkv `(1,6)→(4,2)+rmo`; query_by_key (non-mcast) `(1,6)→(4,2)` | pending |

### Skipped (auto-config — benefits automatically from opt_2 auto-tuner, no manual edits)

Per your guidance: these still need a profile A/B to confirm the auto-tuner produced wins without
regressions, and to bump device/e2e perf thresholds that were set against pre-auto-tuner numbers.

| Model | Status | Next step |
|---|---|---|
| BGE-m3 | All mcast configs commented out → `ttnn.linear` auto-config | Profile on opt_3 vs pre-auto-tuner baseline; bump thresholds |
| DistilBERT | `ttnn.linear` only, no explicit configs | Same |
| bert_tiny (WH) | — | Survey + profile A/B |
| Mamba (WH) | — | Survey + profile A/B |
| Qwen3-embedding-8b (WH) | — | Survey + profile A/B |

### Skipped (no upgrade available — already at max DST volume given shape)

| Model | Reason |
|---|---|
| owl-ViT | All 4 configs have `per_core_M=1` → h=1 forced → max DST volume is `per_core_N`; current subblocks already at max. Flag-only no-op per recipe. |

### Blocked / maintainer-declared unsupported on WH

| Model | Reason |
|---|---|
| `demos/bert` | `@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")`. 6 configs with 2 upgrade candidates (qkv `(1,6)→(4,2)+rmo`, query_by_key `(1,6)→(4,2)`) that would match sentence-BERT's pattern, but no way to validate from n150. |

### Additional manual-config candidates surveyed (defer decision)

These have explicit configs and WH-feasibility possibility but need weights / targeted follow-up:

| Model | File | Configs | Notes |
|---|---|---:|---|
| ViT-WH | `demos/vision/classification/vit/wormhole/tt/ttnn_optimized_sharded_vit_wh.py` | 5 mcast + 2 non-mcast | `per_core_M=7` (prime) caps multi-row rmo to `(7, 1)` volume 7. Upgrades possible but each needs L1 fit check. Weights: `google/vit-base-patch16-224` (~300 MB). |
| ResNet50 (WH) | `demos/vision/classification/resnet50/ttnn_resnet/tt/ttnn_functional_resnet50.py` | 1 (1D mcast) | Single classifier-like matmul config. Limited surface area. |
| SDXL (WH) | `demos/vision/generative/stable_diffusion/wormhole/tt/` (4 files) | 4+ | Multiple files. Large weights (~8 GB). |
| Segformer | `demos/vision/segmentation/segformer/tt/ttnn_segformer_mlp.py` | TBD | Not counted |
| MobileNetV2 | `demos/vision/classification/mobilenetv2/tt/ttnn_mobilenetv2.py` | TBD | Not counted |
| SigLip (multimodal) | `demos/multimodal/siglip/tt/attention.py` | TBD | Not counted |
| Qwen 2.5/3 VL | `demos/qwen25_vl/`, `demos/qwen3_vl/` | TBD | Likely multi-chip, needs check |
| metal_BERT_large_11 | `demos/metal_BERT_large_11/` | TBD | Legacy BERT implementation |
| gemma4 | `demos/gemma4/tt/experts/decode.py` | TBD | Likely multi-chip |

### Deferred (weights require ~1.3 GB HF download, below priority)

- **BGE-large** (`models/demos/wormhole/bge_large_en/ttnn/common.py`): 4 explicit configs, one
  upgrade candidate (`self_out` `(1,4) → (2,4)` legacy-compatible, volume 4 → 8). Other three
  configs capped at volume 4 by `fp32_dest_acc_en=True` (DST=4). Estimated e2e upside on the
  self_out edit alone is ~3% — below the 5% bar unless combined with something else. **Revisit
  if MLPerf cache gets mounted or if full download is cheap.**

### Not attempted (multi-chip, not feasible on WH n150)

DeepSeek-V3 (`demos/deepseek_v3*/`), Llama3-70b-Galaxy (`demos/llama3_70b_galaxy/`), T3000 and TG
demos, GPT-OSS (`demos/gpt_oss/`, likely multi-chip — needs confirmation). **Blocked on hardware
access.** Config count surveys:
- DeepSeek-V3 / V3_d_p: 6+ files with explicit configs
- GPT-OSS: 22 configs across `attention/config.py`, `experts/config.py`, `experts_throughput/config.py`
- Llama3-70b-galaxy: 3+ files

### Additional Falcon7b metric

E2E inference (median of 9 post-edit / 6 baseline runs at seq=1024): **0.368s → 0.307s (-16.6%)**. High run-to-run variance (8-9% std/mean) but effect size > 2× noise. Device-kernel measurement (<1% variance) confirms the win.

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

### Cross-cutting — helpers / auto-tuner

- **DST size must be queried, not hard-coded.** DST register-file capacity is not a constant — it
  varies with data type, whether DST double-buffering is on, whether `fp32_dest_acc_en=True`
  (halves capacity), half-sync vs full-sync protocol, and potentially across devices/architectures.
  Today the auto-tuner uses hardcoded `DST=8` (half-sync bf16) or `DST=4` (fp32_dest_acc_en). This
  works for manual tuning where a human can infer the right cap, but **the helpers / auto-tuner
  must fully query DST capacity from the device + compute-kernel-config** for correctness under
  all modes. Concrete trigger: BGE-large's qkv / ff2 configs use `fp32_dest_acc_en=True` →
  effective DST=4; if the tuner assumed DST=8 for these it would emit an over-sized subblock and
  the program would fail to compile. Tracked as a follow-up for the helpers work itself, not per
  model.

- **Auto-config models need a profile A/B across the auto-tuner landing** to confirm the tuner
  actually delivers wins without regressions, and to bump device/e2e perf thresholds that were
  set against pre-auto-tuner numbers. Applies to BGE-m3, DistilBERT, and any other model calling
  `ttnn.linear` / `ttnn.matmul` without an explicit `program_config`. Methodology per model:
  checkout `main` (or the commit immediately before `8e9f28d664b`), run the model's device-perf
  test, record. Checkout `wransom/matmul_helpers_opt_2` (or later), rerun, compare. If faster,
  bump `expected_perf` in the device-perf test to the new measured value and commit.

  **Status on n150 today** (2026-04-23): none of the auto-config models with device-perf tests
  are directly runnable without weights downloads not currently cached:
  - `bert_tiny` (`mrm8488/bert-tiny-finetuned-squadv2`): ~50 MB but test is `@pytest.mark.skip`
    (#26288, "Seems to have changed in perf") — un-skip + download + bump threshold workstream.
  - `DistilBERT` (`distilbert-base-uncased-distilled-squad`): ~250 MB, also `@pytest.mark.skip`
    (#26285, same reason).
  - `Mamba` (`state-spaces/mamba-2.8b`): ~5 GB, test active but weights missing locally.
  - BGE-m3 and Qwen3-embedding-8b: no device-perf test exists in the tree.

  The skip markers on bert_tiny and DistilBERT are effectively your "thresholds are stale from
  auto-tuner" signal already present in the tree — those issues (#26288, #26285) are the
  natural home for the A/B-and-threshold-bump work.

---

## Scope notes

- WH n150 single-chip constrains what I can run here. Models requiring T3000/TG/Galaxy/multi-chip
  (DeepSeek-V3, Llama3-70b-galaxy, the `t3000/` and `tg/` model demos) cannot be validated on
  this hardware and are **blocked on HW access**.
- Blackhole-specific demos (`models/demos/blackhole/`) are also out of scope for this branch.
- `models/experimental/` is typically stale and deprioritized unless actively maintained.
- The ~75 file total in the opt_2 review plan includes all of the above; my effective n150-feasible
  scope is closer to ~5-15 models.
