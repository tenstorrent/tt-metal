# pi0.5 Option C — L1-resident weights findings (2026-06-03)

A single readable timeline of the L1-placement investigation: what we
tried, what worked, what didn't, and why. Companion to the more
focused docs ([OPEN_ISSUE_MLP_CB_CLASH.md](./OPEN_ISSUE_MLP_CB_CLASH.md),
[OPTION_C_L1_FOOTPRINT_PROBE.md](./OPTION_C_L1_FOOTPRINT_PROBE.md),
[OPTION_C_TP_WITHIN_STAGE_PLAN.md](./OPTION_C_TP_WITHIN_STAGE_PLAN.md)) —
pick this up first when you come back to the L1 work.

---

## The goal

Move pi0.5 matmul weights from DRAM to L1 (per-chip SRAM, 175.4 MB cap)
to remove the DRAM-bandwidth bottleneck on per-token matmul reads. The
deployment plan §3.1 sketched this as the path to the analytical 8.9 ms
end-to-end latency target on a Blackhole Galaxy.

## Where Option C is today

`Pi0_5PipelineC` runs **end-to-end with all weights in DRAM**:

```
Stage      Chips   L1 used   DRAM used
vision       4      0.0 MB     0.0 MB  (host-resident via embed_on_host=True)
prefill     18      0.0 MB  2113   MB  (all 18 VLM layers replicated per chip)
denoise      6      0.0 MB   576   MB  (all 18 expert layers replicated per chip)

L1 cap per chip: 175.4 MB
DRAM cap per chip: ~34 GB
```

This is the working state — bench runs land at ~209 ms total (Option C)
vs ~220 ms (Option B), see
[OPTION_B_VS_C_COMPARISON.md](./OPTION_B_VS_C_COMPARISON.md). The L1
plan is **future state**, not today.

## What we built (works)

1. **L1 migration helper** (`tt/option_c/vision_slice.py`).
   `_migrate_tower_weights_to_l1(tower)` walks every weight tensor in a
   constructed `SigLIPVisionTowerTTNN` and moves it to L1 via
   `ttnn.to_memory_config(t, L1)` + `ttnn.deallocate(t)`.
   `_migrate_projector_weights_to_l1(proj)` does the same for the
   `MultiModalProjectorTTNN`.

2. **Flag chain through the pipeline** (commit `0f3a0b2fe94`).
   `Pi0_5PipelineC` accepts three dataclass flags:
   `layer_paired_l1`, `device_siglip`, `vision_weights_l1`. Each
   propagates to the corresponding stage constructor; all default to
   False (current behavior).

3. **L1 footprint probe**
   (`tests/test_option_c_l1_footprint_probe.py`). Real-checkpoint,
   trace-test-matched workload; reports per-chip L1+DRAM at three
   phases (pre-init / post-init / post-warmup-forward); env-knob driven
   so we can A/B placements.

4. **SigLIP redistribution** (commit `dde14013647`). Original layout
   was `9+9+9 SigLIP + 1 projector` across 4 vision chips. Per-chip
   load with all weights migrated overshot the 175 MB L1 cap by ~5 MB.
   Redistributed to **7+7+7+6 across all 4 chips with mm_projector
   co-located on chip 3**. Same number of host bounces (3), same total
   compute. Per-chip load drops to ~129 MB on the heaviest chip with
   ~46 MB headroom.

5. **Migration succeeds at full depth.** Probe with
   `PI0_OC_L1_PROBE_DEVICE_SIGLIP=1 PI0_OC_L1_PROBE_VISION_WEIGHTS_L1=1`:

   ```
   stage=vision  chip 0   L1 used = 138.9 MB   DRAM used = 8.8 MB
   ```

   Down from 181.9 MB DRAM (host-mode baseline). The plumbing works.

## What blocks the forward (the CB clash)

After migration succeeds, `run_inference()` crashes inside the SigLIP
MLP matmul kernel:

```
RuntimeError: TT_THROW @ tt_metal/impl/program/program.cpp:1452
Statically allocated circular buffers in program 282 clash with L1
buffers on core range [0-0 - 7-0]. L1 buffer allocated at 537600 and
static circular buffer region ends at 733696
```

The exact same class of bug shows up for the paired prefill
(`PI0_OC_L1_PROBE_LAYER_PAIRED=1`) — program 788, VLM MLP, same
pattern.

### Mechanism

Every ttnn matmul kernel reserves a static circular buffer (CB) region
at the low end of L1 for its accumulation tiles. The ttnn allocator
places L1-interleaved buffers (our weights) wherever there's space —
**it doesn't know about the kernel's CB region**. When weights are
L1-resident, some weight buffers land in the contested low-L1 range.
The kernel's startup validation (`validate_circular_buffer_region`)
sees a collision and aborts.

### Per-bank arithmetic

| Component | per-L1-bank size |
|---|---|
| L1 bank capacity (Blackhole) | 1.43 MB |
| Static CB region (SigLIP MLP with fused GELU) | ~0.73 MB (bottom) |
| Available L1 above CB region | **~0.70 MB** |
| Our SigLIP weight load per bank (9-layer/chip naive) | ~1.16 MB |
| Our SigLIP weight load per bank (7+7+7+6 redistributed) | ~1.16 MB on busiest chip |
| Threshold to clear: weight load ≤ available above CB | **0.70 MB / bank** |

**Weights physically don't fit above the CB region.** The allocator has
no choice but to place some inside it.

### Important non-obvious fact

The CB region size **depends on the matmul kernel's per-chip shape**.
The 0.73 MB number above is for the specific shape that SigLIP's MLP
matmul runs at on Option C. Different shapes → different CB sizes.
**TP=8 on Option B's MLP shrinks the per-chip MLP dim 8×, which shrinks
the CB region dramatically** — likely down to ~0.05-0.10 MB / bank.
That's why Option B's TP=8 path has never tripped this issue (it's
inherently smaller-shape) and why it's relevant to the next question.

## What we tried that didn't work

### 1. `l1_small_size` reservation

`open_galaxy_mesh(layout, l1_small_size=N)` reserves N bytes per bank
for the L1-small allocator. Static CBs do **not** live in L1_small —
they live in regular L1. Increasing l1_small_size doesn't push the CB
region up and counts 1:1 against L1 capacity. Standard pi0.5 value is
24576 (24 KB / bank); going higher just eats L1.

### 2. DRAM-bouncing matmul OUTPUTS

Mirrored Option B's `tp_block.py:215-224` pattern — added a
`_output_memory_config` flag on `SigLIPMLPTTNN` /
`SigLIPAttentionTTNN`, defaulted L1, set to DRAM after the migration
helper ran. **Same crash, same addresses.** The colliding L1 buffer is
a **weight** (or weight-adjacent intermediate), not the output. Moving
the output to DRAM doesn't help — the weight stays where the
allocator placed it.

This fix is **correct for the all_reduce / output-only case**
(Option B's actual use). It just doesn't fit the matmul case where
weights themselves collide. Reverted (commit `d60d268b6ef`) to keep
the codebase honest.

### 3. Per-tensor migration scope tuning

Skipping `patch_embed` + `pos_embed` + `post_ln` + LayerNorm migrations
(keeping them in DRAM) brought the chip-0 load from ~178 MB to ~173 MB
— still over the cap on the original 9+9+9 layout. The 7+7+7+6
redistribution was what actually cleared it for migration; the CB
clash is a separate gating.

## What might work (researched, not implemented)

Per
[OPTION_C_TP_WITHIN_STAGE_PLAN.md](./OPTION_C_TP_WITHIN_STAGE_PLAN.md)
— the synthesis of a multi-agent research workflow on TP within stage:

### Prefill: TP=2 via `(2,1)` col-pairs

(6,3) prefill submesh carves into 9 × `(2,1)` sub-meshes, 2 VLM layers
per sub-mesh. Per-chip weight load drops 110 → 55 MB (= 0.46 MB /
bank). **Clears the CB threshold.** Scope: medium (~6 files; 2 are
near-verbatim copies of Option B's `tp_block.py`).

### Denoise: TP within stage is *invariant*

On (6,1) = 6 chips with 18 expert layers, TP=N gives N×(1/N) =
constant per-chip weight load = 98.4 MB for any N. **TP alone cannot
fix denoise.** Real denoise fix is one of:
1. Shard the adaRMS modulation Dense `[1024, 6144]` along its 6144
   axis. Saves ~12 MB/layer × 3 layers/chip = 36 MB/chip → drops
   per-bank to ~0.57 MB → fits.
2. Drop adaRMS mod from bf16 → bf8. Saves ~18 MB/chip → same effect.
3. The MLP DRAM-bounce structural fix mirroring
   `vlm_slice.py:298`'s rms_norm pattern.

### SigLIP

Same 9-layer / chip pattern as prefill — fits up to ~110 MB / chip
naive, but per-bank load goes over 0.70 MB threshold. TP within the
4-chip vision submesh doesn't help (4 chips ÷ 4-way TP = 1 chip per
sub-mesh, no sharding). The 7+7+7+6 redistribution gets weights INTO
L1, but the forward still trips the CB clash. Likely needs the same
TP / kernel-shape lever as prefill, but vision doesn't have enough
chips to do TP=8.

## The non-pi0.5 paths

| Path | Scope | Resolves CB clash? |
|---|---|---|
| Prefill TP=2 (this codebase) | medium | yes — per-bank drops to 0.46 MB |
| Denoise modulation sharding + bf8 + MLP DRAM-bounce | small-medium per piece | yes — combined |
| **tt-blaze port** (different runtime) | large | yes — FusedOp compiler lays out CBs + weights together |
| Kernel engineering (shrink/relocate matmul static CB region) | small in pi0_5, large in `tt_metal` | yes — root cause |
| More chips per stage (8-chip SigLIP, 12-chip prefill) | medium — layout change + spare submesh redesign | yes — drops per-bank by chip count ratio |

## Code state today

| Concept | Code lives at | Active? |
|---|---|---|
| L1 migration helper for SigLIP | `tt/option_c/vision_slice.py::_migrate_tower_weights_to_l1` | yes, gated by `vision_weights_l1` |
| L1 migration for prefill/denoise | (would mirror SigLIP helper for VLM/expert blocks) | **not built** — would migrate but forward crashes |
| Pipeline flag chain | `Pi0_5PipelineC.{layer_paired_l1, device_siglip, vision_weights_l1, expert_layers_per_chip}` | yes |
| 7+7+7+6 SigLIP redistribution | `Pi0_5OptionCVisionSliceSplit` default `layers_per_chip` | yes |
| L1 footprint probe | `tests/test_option_c_l1_footprint_probe.py` | yes; env-knob driven |
| Smoke test | `tests/test_option_c_smoke.py` #11 (device-SigLIP split dry-run) | yes; 7+7+7+6 layout |
| `_output_memory_config` (output DRAM-bounce) | reverted (commit `d60d268b6ef`) | no |

## Open questions for the next session

1. **Is Option B's TP=8 with L1-resident weights actually unblocked,
   or does it hit the same CB clash at a smaller magnitude?** Per-chip
   weight load for Option B TP=8 VLM is ~125 MB (= 1.04 MB / bank,
   above 0.70). BUT TP=8 shrinks matmul shapes, which should shrink the
   CB region per-bank. Net result depends on whether the CB shrinks
   faster than the weight load grows. **Needs a probe.** See the
   Option B analytical assessment companion section below.

2. **Does the prefill TP=2 plan actually clear the CB clash in
   practice?** The per-bank math says yes (0.46 < 0.70) but matmul
   `in0_block_w` retuning may be needed and there could be other shape
   constraints — needs measurement on real silicon.

3. **What's the actual CB region size as a function of matmul shape?**
   We treated 0.73 MB / bank as a constant for the SigLIP MLP matmul.
   For other shapes (VLM MLP at TP=2, expert MLP, etc.) it'll differ.
   Worth a script that computes / measures expected CB region for
   different shapes so we can predict per-bank fit before running.

4. **Is the kernel-engineering path (shrink the CB region directly)
   tractable for someone on the tt_metal side?** A 5x reduction in CB
   size would unblock all Option C and Option B L1-resident paths at
   once. The L1 small allocator already partitions L1 — maybe the same
   mechanism could be extended to reserve "high L1" for weights and
   keep "low L1" for kernels.

## Lessons (for whoever picks this up)

1. **Test with REAL workload, not synthetic.** The 9+9+9 SigLIP
   layout's L1 fit looked OK on paper (~157 MB / chip per the
   deployment plan) but exceeded the 175 MB cap at the actual padded
   shapes (head_dim 72 → 96 inflation, per-bank bias overhead) by ~5
   MB. The probe caught it on the real checkpoint where the analytical
   sketch had not.

2. **CB region size is shape-dependent.** Don't treat the 0.73 MB /
   bank number as a universal constant — it's specific to the SigLIP
   GELU MLP matmul at Option C's per-chip shape.

3. **TP within stage doesn't always reduce per-chip weights.** If
   total chips × total layers is fixed, TP and layers/chip move
   inversely → per-chip load is constant. Only works where chips ≥
   layers (prefill).

4. **The migration helper pattern (post-construction walk +
   to_memory_config + deallocate) is solid.** It works structurally;
   only blocked by the CB clash, not by anything wrong with the
   migration logic. Reusable for any future placement experiment.

5. **`l1_small_size` is per-BANK, not per-chip.** 1 MB per bank =
   120 MB per chip reserved; 24 KB per bank = 2.88 MB per chip. The
   pi0_5 single-device convention is 24576; don't deviate without
   recalculating.
