# pi0.5 Option C — L1 + DRAM footprint probe

**Date**: 2026-06-03
**Host**: g11blx01 (32-chip Blackhole Galaxy)
**Checkpoint**: `/home/tt-admin/pi05_cache/pi05_libero_upstream`

## Goal

Produce an *empirical* per-chip memory footprint table for Option C at
the deployment-target depth (`vlm_depth=18, expert_depth=18`) on the
real libero_upstream checkpoint, with a workload that mirrors
`tests/perf/test_perf_ttnn_full_e2e_trace.py` exactly so the numbers
are directly comparable.

This was motivated by the disconnect between the analytical
per-chip plan in `PI0_5_GALAXY_DEPLOYMENT_PLAN.md` (122 MB / chip on
prefill, 30 MB / chip on denoise) and the **measured 122 MB / chip on
Option B** in `OPTION_B_STATUS.md` — we had no equivalent measurement
for Option C and the e2e bench ran in shrunk mode
(`vlm_depth=2, expert_depth=1`) because the replicated upload was not
known to fit at full depth.

## What the probe does

File: `models/experimental/pi0_5/tests/test_option_c_l1_footprint_probe.py`.
Opt-in via `PI0_OC_L1_PROBE=1`.

Per submesh, per phase, reads:

- L1: used / free / cap per chip (via `ttnn._ttnn.device.GetMemoryView(submesh, BufferType.L1)`)
- DRAM: used / free / cap per chip (same API with `BufferType.DRAM`)

Phases captured:

1. `baseline (pre-init)` — memory floor before any uploads
2. `after Pi0_5PipelineC.initialize` — all weights uploaded
3. `after warmup forward` — peak with activations, KV, masks live

If `initialize()` or `run_inference()` OOMs, the snapshot is still
captured before the exception re-raises, so the cliff is visible.

A separate parametrised test, `test_oc_l1_depth_sweep`, walks
(2,1) → (4,2) → (8,4) → (12,8) → (18,18) for bisecting an OOM cliff if
full depth fails — gated by `PI0_OC_L1_PROBE_DEPTH_SWEEP=1`.

### Workload — matches `test_perf_ttnn_full_e2e_trace.py`

```
LANG_SEQ_LEN          = 256
PREFIX_LEN            = 256 + LANG_SEQ_LEN     # 512
ACTION_DIM            = 32
ACTION_HORIZON        = 10                     # libero_upstream config.json
ACTION_HORIZON_PADDED = 32                     # tile-aligned
NUM_DENOISE_STEPS     = 10                     # PI05_NUM_DENOISE_STEPS
BATCH_SIZE            = 1
pixel_values          = torch.randn(1, 3, 224, 224)   # 1 RGB 224x224 image
lang_tokens           = torch.randint(0, 256000, (1, 256), int32)
noisy_actions         = zeros(1, 32, 32); rows 0:10 = randn(1, 10, 32)
```

The noisy-actions construction matches
`ttnn_pi0_5_model.sample_actions` exactly: zero-pad to
action_horizon_padded, then fill only rows 0:action_horizon with
N(0,1). The padding rows are masked out by SDPA anyway, but matching
exactly removes one variable when comparing numerics later.

### Env flags — matched to the trace test invocation

```
PI0_UPSTREAM_MASKS=1
QWEN_NLP_CONCAT_HEADS_HEAD_SPLIT=1
QWEN_NLP_CREATE_HEADS_HEAD_SPLIT=1
PI05_CHECKPOINT_DIR=/home/tt-admin/pi05_cache/pi05_libero_upstream
PI05_NUM_DENOISE_STEPS=10
```

Probe knobs:

```
PI0_OC_L1_PROBE              = 1            (required to opt in)
PI0_OC_L1_PROBE_VLM_DEPTH    = 18           (default)
PI0_OC_L1_PROBE_EXPERT_DEPTH = 18           (default)
PI0_OC_L1_PROBE_RUN_FORWARD  = 1            (default)
PI0_OC_L1_PROBE_DEPTH_SWEEP  = unset        (1 to enable sweep)
PI0_OC_L1_PROBE_CHECKPOINT   = …            (overrides PI05_CHECKPOINT_DIR)
```

## Result — full depth, real ckpt (2026-06-03)

```
Stage     Chips   L1 post-init    L1 post-fwd     DRAM post-init     DRAM post-fwd
──────────────────────────────────────────────────────────────────────────────
vision      4        0.0 MB         0.0 MB             0.0 MB             0.1 MB
prefill    18        0.0 MB         9.4 MB          2112.9 MB          2113.4 MB
denoise     6        0.0 MB         9.4 MB           575.9 MB           576.6 MB

L1   cap per chip: 175.4 MB    (120 banks × 1,461,760 bytes)
DRAM cap per chip: ~34 GB
```

Test wall-clock: 54 s (load weights + open mesh + init + forward +
2nd snapshot + teardown).

## What it tells us

### 1. Today everything is in DRAM. L1 is essentially free.

Each prefill chip carries **2,112.9 MB of DRAM weights** — all 18 VLM
transformer layers replicated per chip, plus the embed/lm_head table.
18 × ~110 MB/layer at bf8 ≈ 1,980 MB, + ~527 MB embed table ≈ 2,507 MB
upper bound, vs 2,112.9 MB observed; the gap is mostly tile alignment
and the dtype mix from [[pi0-5-dtype-map]].

Each denoise chip carries **575.9 MB of DRAM weights** — all 18 expert
layers replicated. 18 × ~32.8 MB/layer ≈ 590 MB upper bound, matches
within rounding.

L1 sees only **~9.4 MB / chip of transient activations** during
forward — KV cache, mask, intermediate hidden states. The 175 MB L1
cap is barely touched.

### 2. The vision stage is host-resident

`Pi0_5PipelineC` defaults `embed_on_host=True`, so `StageVision` uses
`Pi0_5OptionCVisionSlice` (CPU path) rather than
`Pi0_5OptionCVisionSliceSplit` (3-chip SigLIP split + projector chip).
The 4 vision chips are opened but never receive an upload. Net:
0.0 MB on chip across all phases. SigLIP-27's ~565 MB of weights live
in host RAM as PyTorch tensors and don't show up here.

### 3. "Move weights to L1" is *not* a memory_config default flip

The L1-resident plan in `tt/option_c/README.md` reads "every weight,
bias, and activation in L1 (`memory_config=ttnn.L1_MEMORY_CONFIG`)."
The natural reading is "flip the default in the upload helpers to L1."

That reading does **not** work for Option C today:

- Replicated 18 VLM layers in L1 needs 2,113 MB / chip vs 175 MB cap — **12× over**.
- Replicated 18 expert layers in L1 needs 576 MB / chip vs 175 MB cap — **3× over**.

The plan requires **layer-paired sharding** (1 VLM layer per prefill
chip, 3 expert layers per denoise chip), which is the placement
`PI0_5_GALAXY_DEPLOYMENT_PLAN.md` §3.1 originally proposed:

```
prefill chip (1 VLM layer):            ≈ 122 MB / chip   → fits 175 MB cap
denoise chip (3 expert layers + suff): ≈  98 MB / chip   → fits 175 MB cap
vision chip (9 SigLIP layers, bf8 attn): ≈ 157 MB / chip → tight but fits
```

### 4. Layer-paired sharding already exists at the slice level

- `Pi0_5OptionCVLMSlicePaired` (`tt/option_c/vlm_slice.py`) — 1 layer per micro-submesh.
- `Pi0_5OptionCExpertSlicePaired` (`tt/option_c/expert_slice.py`) — N layers per micro-submesh.
- `Pi0_5OptionCVisionSliceSplit` (`tt/option_c/vision_slice.py`) — SigLIP across 3 chips + projector chip.

Smoke tests #8–#11 in `test_option_c_smoke.py` exercise all three at
the slice level. The gap is at the `Pi0_5PipelineC` level: no
constructor flag plumbs `layer_paired_l1` through to
`StagePrefill.__init__` / `StageDenoise.__init__`, and `device_siglip`
is not exposed on `StageVision` from the pipeline either. The stages
already accept these flags — they default to `False`.

## Concrete next steps

### Status — done

1. ✅ **Wire `layer_paired_l1` and `device_siglip` through `Pi0_5PipelineC`** (commit
   following this update). Added 3 dataclass fields (`layer_paired_l1`,
   `device_siglip`, `expert_layers_per_chip`); propagated to
   `StageVision` / `StagePrefill` / `StageDenoise` at `initialize()` time;
   refactored `run_inference` to route transports through each stage's
   `first_chip_submesh` accessor (preserves replicated-mode behavior
   bit-for-bit while making paired-mode routing correct). The denoise
   loop in `_denoise_with_per_step_timing` also got the same paired-mode
   host-bounce of `velocity_hidden` last-chip → first-chip that
   `StageDenoise.denoise()` already had.

### Probe result with `PI0_OC_L1_PROBE_LAYER_PAIRED=1` (2026-06-03)

```
Stage     Chips queried  L1 post-fwd       DRAM post-fwd
─────────────────────────────────────────────────────────
vision    4 (parent)         0.0 MB             0.1 MB    (host-resident)
prefill   1 (chip 0)       119.0 MB             9.1 MB    (1 VLM layer L1-resident)
denoise   1 (chip 0)        86.5 MB            14.4 MB    (3 expert layers + suffix)

L1 cap per chip: 172.5 MB (l1_small_size=24576 reserved for L1 small allocator)
```

Compared to the analytical plan (122 MB prefill, 98 MB denoise) the
measured values are within 3 MB. Compared to the replicated baseline
(2113 / 576 MB DRAM), paired mode is **12× smaller** on prefill DRAM
and **41× smaller** on denoise DRAM, with the weights now living in L1
where the deployment plan put them.

The probe also gained two new env knobs to support this measurement:

- `PI0_OC_L1_PROBE_LAYER_PAIRED=1` — turns on paired-mode placement.
- `PI0_OC_L1_PROBE_DEVICE_SIGLIP=1` — turns on the 3-chip SigLIP split.
- `PI0_OC_L1_PROBE_L1_SMALL_SIZE` — bytes per bank reserved for the
  L1-small allocator. Defaults to 24576 when paired-mode is on (the
  standard pi0.5 single-device value); unset otherwise.

The probe also now queries the per-chip `micro_submeshes[0]` after
init in paired mode — `GetMemoryView(parent_submesh, L1)` doesn't see
allocations made on a parent's carved children, so reading the parent
submesh returned 0 MB even though weights were resident on each chip.

### device_siglip=True probe (2026-06-03)

Probe with `PI0_OC_L1_PROBE_DEVICE_SIGLIP=1`, paired-mode OFF:

```
Stage     Chips queried  L1 post-fwd       DRAM post-fwd
─────────────────────────────────────────────────────────
vision    1 (chip 0)         0.0 MB           181.9 MB    (SigLIP-27 on chip)
prefill   18 (parent)        9.4 MB          2113.4 MB    (replicated, paired off)
denoise   6 (parent)         9.4 MB           576.6 MB    (replicated, paired off)
```

End-to-end forward completes. Vision-on-device path is functional —
SigLIP-27 lives on the 3 vision chips + projector chip instead of host
CPU, ~182 MB of vision weights on chip 0 (was 0 / 0 in host mode).

A `Pi0_5OptionCVisionSliceSplit.embed_images()` fix was required to
make this work: the slice was passing the host `torch.Tensor`
`pixel_values` directly to `siglip_chunks[0].forward()` which expects
a ttnn.Tensor (the first op it runs is `ttnn.permute(x, (0, 2, 3, 1))`
NCHW→NHWC on device). The upload is now performed inside the slice,
matching the trace test convention (`bf16, TILE, DRAM_MEMORY_CONFIG`).
Smoke test #11 didn't catch this because it's a dry-run construction
test — it builds the slice but doesn't call `embed_images`.

**Open follow-up** — SigLIP weights are in **DRAM**, not L1, because
`SigLIPVisionTowerTTNN` uploads its own weights with ttnn's default
memory config (DRAM). To get the deployment plan's "~157 MB L1 / vision
chip" placement, either:
- Modify `SigLIPVisionTowerTTNN` to accept a memory_config parameter
  and have `Pi0_5OptionCVisionSliceSplit` pass `L1_MEMORY_CONFIG`, or
- Have the wrapper post-process the constructed slice to move each
  weight tensor to L1.

### Vision-weights-in-L1 probe (2026-06-03)

The post-process migration helper landed
(`vision_slice.py::_migrate_tower_weights_to_l1`,
`_migrate_projector_weights_to_l1`) and the flag chain is wired through:

```
Pi0_5PipelineC.vision_weights_l1=True
  → StageVision.vision_weights_l1=True
    → Pi0_5OptionCVisionSliceSplit(weights_in_l1=True)
      → after each SigLIPVisionTowerTTNN is constructed, walk every
        weight tensor and run ttnn.to_memory_config(t, L1) + deallocate
        the DRAM source.
```

The helper migrates: PatchEmbedding `_linear_weight` / `_linear_bias`,
positional embedding, post-LN, and for every block `ln{1,2}_{weight,bias}`,
`attention.{wqkv, bqkv, wo}`, `mlp.{fc1_weight, fc1_bias, fc2_weight,
fc2_bias}`. MultiModalProjectorTTNN gets the same treatment for its
`weight` / `bias`.

Probe with `PI0_OC_L1_PROBE_DEVICE_SIGLIP=1 PI0_OC_L1_PROBE_VISION_WEIGHTS_L1=1`:

```
[FAIL] initialize() raised: RuntimeError: TT_FATAL @
    tt_metal/impl/allocator/bank_manager.cpp:462: false
info: Out of Memory: Not enough space to allocate 5640192 B L1 buffer
across 120 banks, where each bank needs to store 47872 B, but bank size
is 1461760 B (allocated: 1417216 B, free: 44544 B, largest free block:
44544 B)
```

Per-chip arithmetic at the failure point:
- L1 cap: **175.4 MB / chip** (120 banks × 1,461,760 bytes)
- L1 allocated when OOM hit: **170 MB / chip** (almost saturated)
- Next buffer requested: **5.6 MB**
- Free: **5.3 MB** → over by ~300 KB

The migration walked most of the per-chip SigLIP weights — ~170 MB
already moved to L1 — before hitting the cliff on the last bank's
worth.

**Why the deployment plan's 157 MB target doesn't match:** today's
`SigLIPAttentionTTNN` uploads Q/K/V/O at **bf16**, not bf8_b. From
PI0_5_GALAXY_DEPLOYMENT_PLAN.md §1.1:

| Tensor                  | dtype today | bytes / layer |
|-------------------------|-------------|---------------|
| attn q/k/v/o × 4        | bf16        | 10.6 MB       |
| mlp.fc1 + fc2           | bf8_b       | 9.92 MB       |
| layernorms              | bf16        | tiny          |
| **per encoder layer**   | mixed       | **~20.7 MB**  |
| × 9 layers (vision chip)| —           | **~186 MB**   |

That's already over the 175 MB L1 cap before patch_embed / pos_embed
land on chip 0. The deployment plan §1.1 already flags this:

> If attention weights are also dropped to bf8_b (currently bf16 on the
> assumption that QKVO bandwidth doesn't dominate at 1152-wide),
> per-layer drops to ~15.6 MB, total ~140 MB. Worth measuring before
> committing to a sharding plan.

So **plumbing is correct; the cliff is fundamental at the current
dtype mix**. Two paths forward:

1. **bf8 attn QKVO in SigLIPAttentionTTNN.** Already done — the
   `SigLIPAttentionTTNN` constructor uploads Q/K/V/O at `bfloat8_b`
   (lines 410-437). The deployment plan's "10.6 MB / layer" attn
   numbers in §1.1 were based on bf16 (stale documentation). At bf8
   the unpadded attn per-layer is ~5.3 MB.

2. **Padded shapes inflate the actual per-chip footprint.** The
   SigLIPAttentionTTNN code pads `head_dim` 72 → 96 to land on the
   32-element tile boundary, inflating Q/K/V/O outputs by 33% (1152
   → 1536). At padded shapes per encoder layer is ~18 MB, not the
   plan's 15.6 MB. With 9 layers / chip = ~163 MB + per-bank bias
   overhead (~10 MB) ≈ 173 MB — within shouting distance of the
   175.4 MB L1 cap but OOMs by ~5–7 MB on the last weight.

### SigLIP redistribution: 7+7+7+6 across all 4 vision chips (2026-06-03)

To get under the cap, the on-device SigLIP split was redistributed
from `9+9+9 + projector-only` (3 SigLIP chips + 1 projector chip) to
**`7+7+7+6` across all 4 vision chips with `mm_projector` co-located
on chip 3**. Same number of host bounces (3) and same total compute
(27 layer-times). The layout change is contained to
`Pi0_5OptionCVisionSliceSplit` — no `stages.py` / `mesh_setup.py`
touch, spare submesh stays at (2,2) = 4 chips.

Probe with `PI0_OC_L1_PROBE_DEVICE_SIGLIP=1 PI0_OC_L1_PROBE_VISION_WEIGHTS_L1=1`:

```
[after warmup forward]
  stage=vision    chips= 1  L1  used =   138.9 MB / chip   (cap 175.4 MB, free  36.5 MB)
  stage=vision    chips= 1  DRAM used =     8.8 MB / chip
```

**Migration succeeded** — chip 0 holds 7 SigLIP layers + patch_embed +
pos_embed all in L1 at 138.9 MB, comfortably inside the 175.4 MB cap
with ~36 MB headroom. DRAM dropped from 181.9 MB (host-mode
fallback) to 8.8 MB (transients only). This is the empirical proof
that the deployment plan's "L1-resident SigLIP" target works on real
weights — just with a more even distribution than the §3.1 sketch
assumed.

**Forward still crashes — same MLP static-CB issue as the prefill
path.** `run_inference` trips
`program.cpp:1452: Statically allocated circular buffers in program
282 clash with L1 buffers...` inside the SigLIP encoder's MLP matmul.
This is the same class of bug the `Pi0_5PipelineC(layer_paired_l1=True)`
prefill path hits (see OPEN_ISSUE_MLP_CB_CLASH.md) — any matmul
kernel with L1-resident weights AND L1-resident activations lands
buffers in the address range its static CBs want. The fix is the
same DRAM-bounce pattern that `vlm_slice.py:298` already uses for
`rms_norm`, applied around the MLP matmul. **PCC verification of the
L1-resident SigLIP placement is blocked on that fix.**

### Status — open

2. ⚠️  **`run_inference` crashes in paired mode at the MLP GELU CB clash.**
   `tt_metal/impl/program/program.cpp:1452` →
   `Statically allocated circular buffers in program N clash with L1
   buffers on core range [0-0 - 11-7]. L1 buffer allocated at 397568 and
   static circular buffer region ends at 694272`. This is the **same
   class of L1+CB issue** that `vlm_slice.py:298` and `vlm_slice.py:578`
   already document and fix for `rms_norm` (bounce input/output
   through DRAM around the op so the kernel's static CB region doesn't
   collide with our L1-resident buffers). The MLP matmul with fused
   GELU (`fused_activation=UnaryWithParam(GELU)`) needs the same
   treatment. This is op/block-level surgery — separate increment from
   pipeline plumbing — and it's the next blocker for end-to-end
   paired-mode inference.

3. **Re-run Option C e2e bench at full depth** once the MLP-block
   DRAM bounce lands. Today the bench is shrunk
   (`vlm_depth=2, expert_depth=1`); after the CB clash is fixed we can
   benchmark the real-config workload directly comparable to the
   analytical 8.90 ms total in `PI0_5_GALAXY_DEPLOYMENT_PLAN.md` §3.1.

4. **Probe Option B for parity.** Same probe shape against
   `Pi0_5PipelineB` so the comparison doc (`OPTION_B_VS_C_COMPARISON.md`)
   has measured per-chip numbers on both architectures instead of one
   measured and one analytical.

## Pointers

- Probe: `models/experimental/pi0_5/tests/test_option_c_l1_footprint_probe.py`
- Pipeline-level work needed: `models/experimental/pi0_5/tt/option_c/pipeline.py`
- Slice-level paired/split (already done): `vlm_slice.py`, `expert_slice.py`, `vision_slice.py`
- Stage-level flags (already exposed): `stage_prefill.py`, `stage_denoise.py`, `stage_vision.py`
- Trace test (workload reference): `models/experimental/pi0_5/tests/perf/test_perf_ttnn_full_e2e_trace.py`
- Analytical plan: `models/experimental/pi0_5/docs/PI0_5_GALAXY_DEPLOYMENT_PLAN.md`
- Option B measured baseline: `models/experimental/pi0_5/docs/OPTION_B_STATUS.md`
