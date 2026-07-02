# DiffusionDrive TTNN — Performance Report

**Platform:** Wormhole N300s · **Batch:** 1 · **Resolution:** production (camera 256×1024, LiDAR 256×256)
**Weights:** real checkpoint `diffusiondrive_navsim_88p1_PDMS` (`latent=False`) · **torch:** 2.11.0+cpu

All numbers are produced by [`scripts/profile_forward.py`](scripts/profile_forward.py) (median of 20
iterations after 3 warm-up passes). Latency is hardware- and build-dependent — reproduce on your own
setup rather than treating these as fixed. This report is the committed "performance report" deliverable.

```bash
source python_env/bin/activate && export PYTHONPATH="${TT_METAL_HOME:-$PWD}"
export DD_CHECKPOINT_PATH=/mnt/diffusion-drive/weights/diffusiondrive_navsim_88p1_PDMS.pth
export DD_ANCHOR_PATH=/mnt/diffusion-drive/resnet34/kmeans_navsim_traj_20.npy
python models/demos/diffusion_drive/scripts/profile_forward.py --iters 20
```

## 1. Forward-pass latency

| Path | Median latency | FPS (median) | Notes |
|---|---|---|---|
| eager `__call__` | 71.4 ms | 14.0 | op-by-op host dispatch |
| **traced `execute_compiled`** | **44.9 ms** | **22.3** | backbone **and perception** replayed as two `execute_trace`s; deployed path |

- **Trace speedup:** ~1.58× on the full forward (was ~1.43× with only the backbone traced). The perception
  forward is now a second trace (§3) — that is the −8.6% step from 49→45 ms. The still-eager DDIM head +
  agent head dilute the remaining gain.
- The traced path is what the NavSim in-process agent runs, so it is the number that matters for deployment.

## 2. Conv+ReLU fusion (Stage 2 "relu with conv")

38 `conv → ttnn.relu` pairs were fused into the conv's output writeback via
`Conv2dConfig(activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU))` — 32 ResNet-34 BasicBlock `conv1`,
2 stems, 3 FPN convs, 1 grid-sample `value_proj`. This removes 38 standalone elementwise ops per forward.

| Path | Before fusion | After fusion | Δ |
|---|---|---|---|
| eager `__call__` | 71.8 ms | 71.4 ms | ~wash |
| traced `execute_compiled` | 52.3 ms | 50.0 ms | **−4.5%** |

**Correctness is unaffected** (ReLU is monotonic pointwise — fusing vs. running it separately is the same
math): trajectory PCC = 1.000000 (stage3_6, random weights), 0.999777 (real checkpoint, production res),
traced-vs-eager = 1.000000. All 8 conv-path PCC gates still pass.

The gain is modest-but-real on the traced (deployed) path and a wash on eager: the model's cost is
dominated by the conv matmuls and the per-conv DRAM round-trip, not by the ReLU ops themselves. The
fusion is kept because it satisfies the named Stage 2 requirement at zero accuracy cost and zero downside.

## 3. Perception forward trace — the largest single win

Originally only the backbone `[stage→fusion]×4` loop was captured as a trace; the FPN, perception
TransformerDecoder, DDIM head, and agent head ran eagerly (op-by-op host dispatch) after the trace replay.
The **perception forward** (`TtnnPerceptionForward`: bev_downscale conv → keyval → on-device bilinear
upsample → bev_proj → 3-layer TransformerDecoder) is a *static device graph with no interleaved host scalar
glue*, so it is captured as a **second trace** replayed by `execute_compiled` after the backbone trace.

| Path | Backbone trace only | + Perception trace | Δ |
|---|---|---|---|
| traced `execute_compiled` | 49.1 ms | **44.9 ms** | **−8.6%** (20.4 → 22.3 FPS) |

**Correctness:** traced-vs-eager trajectory PCC = 1.000000; full PCC suite 29 passed. This is the first
change with an above-noise latency win — because the perception tail *was* dispatch-bound (~30-40 per-op
host dispatches), and collapsing them into one `execute_trace` removes that overhead (unlike the fusions,
which were dispatch-count reductions that the noise floor swallowed).

**Not traced — the DDIM head.** `TrajectoryHead`'s 2-step denoiser interleaves device ops with host
control-flow (`scheduler.step`, `gen_sineembed` trig, `argmax`/gather best-mode select, norm/denorm, tanh).
A monolithic trace cannot cross those host scalars; tracing it would require capturing many device
micro-sequences with host glue between replays — poor effort/value given the host-bound eval (§6). Left eager.

## 4. Where the remaining cost is — conv memory layout

The `profile_forward.py` core-grid probe reports the memory layout each auto-sharded `ttnn.conv2d`
(`shard_layout=None`) chooses for its **output**, at each ResNet-34 stage resolution:

| Stage conv | Output layout |
|---|---|
| layer1 3×3 64ch 64×256 | interleaved (not sharded) |
| layer2 3×3 128ch 32×128 | interleaved (not sharded) |
| layer3 3×3 256ch 16×64 | interleaved (not sharded) |
| layer4 3×3 512ch 8×32 | interleaved (not sharded) |

Every conv writes its output to **interleaved DRAM**, so each conv in a BasicBlock reads from and writes
to DRAM rather than handing an L1-resident, sharded activation to the next conv. This looked like the
highest-value remaining lever, so it was tested directly — see §5. Per-op *compute* core-utilisation
numbers require a profiler-enabled build (`TT_METAL_DEVICE_PROFILER`); this report uses host wall-clock +
the layout probe.

## 5. Optimization experiments — measured, not adopted

Two conv-config levers aimed at the §4 round-trip were tested and **not adopted** because the measured
gain was not significant or not robust for this model's (small) conv shapes. Documented here so the
decision is reproducible (Stage 3 "document advanced tuning / known issues").

**(a) Explicit L1 sharding** (`shard_layout=HEIGHT_SHARDED` / `BLOCK_SHARDED`), isolated 2-conv
BasicBlock chain vs. the interleaved baseline, per stage:

| Stage | HEIGHT_SHARDED | BLOCK_SHARDED | Output stayed in L1? |
|---|---|---|---|
| layer1 64ch 64×256 | 1.02× | 0.86× | no — still interleaved |
| layer2 128ch 32×128 | 0.91× | 0.84× | no — still interleaved |
| layer3 256ch 16×64 | 1.18× | 1.06× | no — still interleaved |
| layer4 512ch 8×32 | **crash** (`TT_FATAL` sliding_window) | 1.00× | — |

Two problems: (1) even with an explicit shard layout, `ttnn.conv2d` returns an **interleaved** output at
these shapes, so the "keep it L1-resident across convs" premise never actually holds; (2) the latency
effect is stage-specific and offsetting (layer3 gains, layer1/2 regress) and HEIGHT_SHARDED **crashes** on
layer4 — so there is no viable single global config. PCC held (≥0.997) where it ran.

**(b) Activation/weights double-buffering** (`enable_act_double_buffer` + `enable_weights_double_buffer`),
applied to the whole model, end-to-end traced forward:

| | median | avg | min |
|---|---|---|---|
| baseline | 50.0 ms | 50.7 ms | 48.7 ms |
| + double-buffer | 49.2 ms | **53.2 ms** | 47.1 ms |

Median improves ~1.5% but the **average regresses** (higher variance) — within noise, not a robust win.
PCC unchanged. Reverted.

**(c) bfloat8_b conv weights + LoFi math fidelity** (`Conv2dConfig(weights_dtype=bfloat8_b)` +
`compute_config=init_device_compute_kernel_config(..., MathFidelity.LoFi)`), whole conv path, traced:

| | median | avg | min |
|---|---|---|---|
| baseline (bf16 + HiFi2) | 44.9 ms | 45.8 ms | 44.3 ms |
| + bf8 + LoFi | 46.7 ms | 49.1 ms | 44.8 ms |

**No gain — if anything slightly worse.** Crucially, **PCC held** (checkpoint trajectory 0.999900, stage3_6
scores 0.998) — so bf8/LoFi are accuracy-safe; they just don't help *latency*. This was the hypothesis-test
for "the backbone is conv-matmul/bandwidth-bound, so low precision should help." It **isn't**: the small
convs are bound by **per-op device overhead** (fixed kernel/sharding setup), not by weight bandwidth or
matmul FLOPs — so halving the weight width and lowering fidelity touch nothing on the critical path.
Reverted (env-gated `DD_CONV_BF8`/`DD_LOFI` knobs removed — dead weight with no gain).

**Conclusion:** the backbone conv activations are small (≤512 ch, ≤64×256), so sharding/double-buffer/
low-precision all fail for the same reason — **the model is per-op-overhead-bound, not compute/bandwidth/
dispatch-bound.** That is why only *tracing* (which collapses per-op overhead wholesale, §3) helped, and
why op-fusion, sharding, double-buffering, and bf8/LoFi were all measured neutral. The auto
(`shard_layout=None`, bf16, HiFi2) config is the right default here.

## 6. Context — end-to-end throughput is host-bound

The headline NavSim PDM eval wall (~26 min for 12146 scenes with the thread-pool funnel) is gated by
host-side NavSim CPU (scene loading, metric scoring), **not** by the model forward. So forward-latency
improvements above have limited leverage on the eval wall — they matter for the per-scene model cost and
for any deployment that is not host-bound. See README §9.

**Comparison to the paper.** DiffusionDrive reports 45 FPS; that is on a data-center GPU, so it is not a
like-for-like comparison with Wormhole N300s. The traced path here is 20.0 FPS at batch=1, full
production resolution, real weights — the same hardware caveat applies to any absolute-FPS comparison.

## 7. Accuracy (unchanged by the perf work)

| Metric | Value |
|---|---|
| trajectory PCC vs PyTorch (real checkpoint, production res) | 0.999777 |
| traced-vs-eager trajectory PCC | 1.000000 |
| NavSim PDM — full navtest (12146 scenes) | 0.8789 (vs 0.8795 CPU reference, 0.8804 paper) |
