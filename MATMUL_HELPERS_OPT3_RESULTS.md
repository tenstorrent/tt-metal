# Matmul helpers `opt_3` migration results

Branch: `wransom/matmul_helpers_opt_3`
Started: 2026-04-23
Hardware: Wormhole n150 (single-chip) + Blackhole p100a (added 2026-04-23)

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

| Model | Arch | Scope | Baseline | After | Δ | Configs touched | Commit |
|---|---|---|---:|---:|---:|---|---|
| sentence-BERT | WH | total matmul device time (Tracy) | 13,385,817 ns | 10,515,489 ns | **-21.4%** | ff2+qkv reshape (4,2)+rmo; ff1+self_out flag-only | `0a421bb6a56` + threshold `7a89d6a8f55` |
| Falcon7b | WH | prefill seq=1024 device-kernel sps | 3120 | 3742 | **+19.9%** | mm_h_to_4h `(1,8)`, mm_4h_to_h `(1,6)` — both legacy-compatible h=1 | `0fbd77527c2` |
| sentence-BERT | BH | device-kernel sps | 976.5 | 991.7 | +1.56% — below 5% bar, **reverted** | attempted ff2+qkv (4,2)+rmo pattern | (no commit; structural win was the auto-tuner fix, not the config change) |
| SDXL UNet 1024x1024 | BH | device-kernel duration | 79.04 ms | 79.25 ms | +0.26% slower — **reverted** | 11 legacy-compatible subblock upgrades across FF2/TM/ATTN_QKV/ATTN_OUT/RESNET_CONV | (no commit) |
| ResNet50 | BH | device-kernel sps (batch 32) | 12975.5 | — | n/a | single explicit mcast config at `per_core_M=1, per_core_N=1` — already at max; no manual migration target | (no commit) |
| BGE-large | WH | — | pending | pending | pending | self_out `(1,4)→(2,4)` legacy — only candidate (rest capped by `fp32_dest_acc_en=True` DST=4) | pending; blocked on HF weights |
| demos/bert | WH | device-kernel sps | 252 (declared) | pending | pending | qkv `(1,6)→(4,2)+rmo`; query_by_key (non-mcast) `(1,6)→(4,2)` | blocked — `@skipif(is_wormhole_b0())` |

### Skipped (auto-config — benefits automatically from opt_2 auto-tuner, no manual edits)

Per your guidance: these still need a profile A/B to confirm the auto-tuner produced wins without
regressions, and to bump device/e2e perf thresholds that were set against pre-auto-tuner numbers.

| Model | Status | Measured on opt_3 (WH n150) | Threshold | Δ |
|---|---|---:|---:|---:|
| Falcon7b prefill seq=128 | auto-config (ttnn.linear fallback) | 2095.3 sps | 2115 | -0.93% (within ±3%, PASS) |
| Falcon7b decode seq=128 | auto-config (L1_SHARDED) | 636.3 sps | 647 | -1.65% (within ±3%, PASS) |
| Falcon7b decode seq=1024 | auto-config (L1_SHARDED) | 574.4 sps | 572 | +0.44% (within ±3%, PASS) |
| Falcon7b decode seq=2047 | auto-config (L1_SHARDED) | Tracy subprocess failure (pre-existing, not auto-tuner-related) | 548 | n/a |
| sentence-BERT | manual config (upstream) | 546.5 sps | 546.5 | 0% (threshold locked in from upstream work; stability confirmed on opt_3) |
| BGE-m3 | All mcast configs commented out → `ttnn.linear` | no device-perf test in tree | — | — |
| DistilBERT wall-clock (seq=384) | `ttnn.linear` only, no explicit configs | median 0.0326 s / 246 sps (7 runs: 0.0325/0.0330/0.0325/0.0380/0.0326/0.0325/0.0326) | expected 0.0338 s (set to actual × 1.05 in `dccb887fc5a`, so pre-auto-tuner actual was ~0.0322 s) | **+1.2% vs pre-auto-tuner actual — within noise, flat** |
| DistilBERT device-perf (seq=768) | same model | tracy subprocess blocked — inner pytest hung in setup `GetNumPCIeDevices()` even after `tt-smi -r`, 25-min wall clock before inner 300s pytest-timeout fired | 245 (stale) | skip is `@pytest.mark.skip #26285`; setup-hang is pre-existing infra issue, not auto-tuner regression |
| bert_tiny (WH) | auto-config | `@skip #26288` — weights not local (`mrm8488/bert-tiny-finetuned-squadv2`) | 6850 (stale) | pending |
| Mamba (WH) | auto-config | weights not local (`state-spaces/mamba-2.8b`, ~5 GB) | 1.634 ms/layer | pending |
| Qwen3-embedding-8b (WH) | — | no device-perf test in tree | — | — |

**Observations on the WH auto-config A/B data:**
- **Falcon7b** auto-config paths (prefill seq=128, decode seq=128, decode seq=1024): all three
  measure within ±2% of the pre-auto-tuner `expected_perf`. Auto-tuner is flat on Falcon7b's
  auto-config paths. Consistent with the cross-arch hypothesis below — pack-phase lever doesn't
  dominate for small shapes / decode-mode matmuls. No threshold bumps needed.
- **DistilBERT** wall-clock (`test_performance_distilbert_for_qa`, seq=384): 7 runs measured
  0.0325, 0.0330, 0.0325, 0.0380, 0.0326, 0.0325, 0.0326 s. Median 0.0326 s. **Important
  correction**: the `expected_inference_time=0.0338 s` threshold was set in
  `dccb887fc5a` (Oct 2025, pre-auto-tuner) using "5% margin above actual measurements" per
  the commit message — so the pre-auto-tuner ACTUAL was ~0.0322 s, not 0.0338 s. My median of
  0.0326 s is therefore **+1.2% vs the pre-auto-tuner actual, within the 1.5% intra-session
  noise — i.e. flat**, not a -3.6% win. The -3.6% against the threshold is just the buffer
  the threshold was set with. DistilBERT joins Falcon7b's auto-config paths on the "flat"
  side of the ledger — no auto-tuner regression, no measurable win either.
- **DistilBERT** device-perf (`test_distilbert_perf_device`, seq=768): test is
  `@pytest.mark.skip` (#26285). Attempted unskip+measure; hits an unrelated infrastructure
  hang in `ttnn._ttnn.device.GetNumPCIeDevices()` during tracy subprocess setup, even after
  `tt-smi -r`. Not an auto-tuner issue; the skip is doing the right thing until the infra
  is fixed. Threshold bump blocked on this separate issue.
- **sentence-BERT**: measures 546.5 sps exactly on threshold (set in `7a89d6a8f55`). The
  opt_2 auto-tuner + manual-config work upstream are stable on opt_3.

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

### Auto-config bug sweep on BH

Targeted PCC sweep on BH to catch more of the divisibility-class bug the
`63d9bc9e4da` fix addressed. Ran 25+ BH PCC tests across model regions most
likely to exercise batched-sharded matmul paths (attention, transformer
blocks, cross-attention, feedforward, GEGLU, ResNet blocks, VAE, full UNet).
All pass. No additional auto-tuner bugs in this class surfaced beyond the
one already fixed. LoRA + refiner tests skipped on missing weights.

Breakdown:
- SDXL base attention: 4/4 pass
- SDXL base UNet 1024x1024: 1/1 pass
- SDXL base transformerblock / feedforward / GEGLU: 8/8 pass
- SDXL base crossattn (up/down/mid) + resnetblock: 11/11 pass
- SDXL VAE (encode/decode): 2/2 pass (512x512 deliberately skipped on BH)

ResNet50 K-lever check (per explicit user ask): the single matmul config in
`ttnn_functional_resnet50.py` has `in0_block_w=2`, and the per-core K shard
is 2 tiles — so `in0_block_w` is already at max. No K-iteration lever
available for ResNet50 either. Combined with per_core_M=per_core_N=1 capping
the subblock lever, ResNet50 has no helper-based optimization target.

### Cross-arch hypothesis from the negative BH results

Two independent BH migrations (sentence-BERT, SDXL UNet 1024x1024) applied the same WH-pattern
subblock reshape with legacy-compatible `(h, w)` choices and got measurably ~0% delta (+1.56%
and -0.26% respectively). Both passed PCC, neither cleared the 5% bar. The WH equivalents of
these patterns landed -21.4% (sentence-BERT upstream) and +19.9% (Falcon7b).

The most plausible explanation is that BH's matmul kernel time on these shapes is not
dominated by pack-phase overhead in the way WH's is — so reshaping subblocks to increase DST
fill doesn't translate to wall-clock savings. Candidate structural differences: BH has more
cores per die, different NoC topology, potentially different LLK pipeline behavior that
already amortizes pack overhead.

**Action item**: before doing more BH-side manual migrations, empirically measure *where* the
time actually goes on one BH model (Tracy breakdown per op-kind / per-phase). Without that
decomposition we're guessing at which lever matters, and the data so far says this lever
doesn't.

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

### Cross-cutting — helpers / auto-tuner fixes landed

- **Auto-tuner divisibility bug: fixed in `63d9bc9e4da`.** For batched matmul with a height-sharded
  input A (multiple batch instances stacked along the per-core height axis), the auto-tuner was
  being fed `per_core_M = shard_height_in_tiles` but the downstream validator checks
  `out_subblock_h % M_per_batch_instance == 0`. When shard height > per-batch M (e.g. 24 vs 12),
  the tuner picked subblocks like (8, 1) that satisfied the per-core check but failed validation.
  Fix: clamp the tuner's `per_core_M` input at `min(per_core_M, M)` at both sharded-batched call
  sites (`create_matmul_program_config` and `get_matmul_program_config` non-mcast else branch).
  Found by the BH sentence-BERT PCC test, which failed at the attention context_layer bmm;
  fix makes it PASS. Regression check: `tests/ttnn/unit_tests/operations/matmul/test_matmul.py`
  557 passed / 136 skipped / 0 failed on Blackhole p100a. **This was the real BH unblock** — the
  subsequent BH sentence-BERT config edit (matching the WH pattern) only added +1.56% on top,
  so the meaningful win for BH sentence-BERT came from this fix, not from re-applying the
  manual-config recipe.

- **Build fix for fresh-tree gtest include.** `test_common_utils.cpp` uses `<gtest/gtest.h>` but
  the static lib target only linked TTNN::CPP and relied on transitive PCH inheritance that
  held on WH but broke on BH. Cherry-picked from `opt_2` as `6bb8052df6e`.

### Cross-cutting — helpers / auto-tuner follow-ups

- **DST size must be queried, not hard-coded.** DST register-file capacity is not a constant — it
  varies with data type, whether DST double-buffering is on, whether `fp32_dest_acc_en=True`
  (halves capacity), half-sync vs full-sync protocol, and potentially across devices/architectures.

  **Current state after investigation (2026-04-23):** the public API IS abstract — auto-tuner
  callers pass `DeviceComputeKernelConfig` and the tuner routes through
  `ttnn::get_dest_reg_count(config, tile_shape)` which handles `dst_full_sync_en`,
  `fp32_dest_acc_en`, and custom tile shapes correctly. Existing unit tests
  (`test_matmul_auto_tuner.cpp`) cover all 4 DST-capacity cases (16/8/4/8 tiles across sync ×
  fp32 combos). So the tuning surface is already queryable.

  **What's left (two follow-on cleanups, neither is correctness-breaking today):**
  1. `ttnn/.../compute_kernel_config.cpp` re-`#define`s `DATUMS_PER_ROW=16` and
     `DEST_REGISTER_FULL_SIZE=64*16` as host-side copies of constants that live in the
     arch-specific `tensix_types.h`. All three currently-supported arches (WH-b0, BH, Quasar)
     happen to agree on these values, so today's tree is numerically correct, but a future arch
     could silently break the host-side tuner. Attempted to replace the `#define`s with
     `#include "tensix_types.h"` directly; fails because `hw/inc/internal/tt-1xx/${ARCH}` isn't
     on `ttnn_op_core`'s include path, and `hostdevcommon/dprint_common.h`'s transitive include
     is gated behind `!defined(KERNEL_BUILD) && !defined(FW_BUILD)`. Proper fix is adding a
     host-side HAL API (e.g. `hal.get_dest_reg_count(compute_kernel_config, tile_shape)`) mirroring
     the existing `hal.get_arch_num_circular_buffers()` pattern, so host-side tuning can query
     per-arch constants at runtime instead of relying on the `#define` coincidence. Scope: ~1 day.
  2. `matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:151` still has
     `uint32_t max_subblock_w = fp32_dest_acc_en ? 4 : 8;` — hardcoded legacy pre-auto-tuner
     logic that assumes half-sync and bypasses `get_dest_reg_count`. Thread the full
     `compute_kernel_config` through the function signature and route through the query.

  Concrete trigger for why this matters: BGE-large's qkv / ff2 configs use
  `fp32_dest_acc_en=True` → effective DST=4. The abstract tuner handles this correctly today,
  but the DRAM-sharded factory path would quietly emit an over-sized subblock if a future
  config picks `dst_full_sync_en=True` with fp32 (should allow 8 tiles, current code caps at 4).

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
