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
| **traced `execute_compiled`** | **50.0 ms** | **20.0** | backbone loop replayed as one `execute_trace`; deployed path |

- **Trace speedup:** ~1.43× on the full forward (backbone loop is captured; the still-eager
  FPN/perception/DDIM tail dilutes the loop's own ~1.76× — consistent with the README).
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

## 3. Where the remaining cost is — conv memory layout

The `profile_forward.py` core-grid probe reports the memory layout each auto-sharded `ttnn.conv2d`
(`shard_layout=None`) chooses for its **output**, at each ResNet-34 stage resolution:

| Stage conv | Output layout |
|---|---|
| layer1 3×3 64ch 64×256 | interleaved (not sharded) |
| layer2 3×3 128ch 32×128 | interleaved (not sharded) |
| layer3 3×3 256ch 16×64 | interleaved (not sharded) |
| layer4 3×3 512ch 8×32 | interleaved (not sharded) |

Every conv writes its output to **interleaved DRAM**, so each conv in a BasicBlock reads from and writes
to DRAM rather than handing an L1-resident, sharded activation to the next conv. This per-conv round-trip
is the dominant remaining overhead in the backbone and is *not* addressed by ReLU fusion. Closing it —
keeping activations height/block-sharded in L1 across a block, matching the residual-add's shard spec —
is the highest-value remaining optimization (tracked as the L1-sharding rewrite; it is the risky one and
is gated on this profile + maintainer input). Per-op *compute* core-utilisation numbers require a
profiler-enabled build (`TT_METAL_DEVICE_PROFILER`); this report uses host wall-clock + the layout probe.

## 4. Context — end-to-end throughput is host-bound

The headline NavSim PDM eval wall (~26 min for 12146 scenes with the thread-pool funnel) is gated by
host-side NavSim CPU (scene loading, metric scoring), **not** by the model forward. So forward-latency
improvements above have limited leverage on the eval wall — they matter for the per-scene model cost and
for any deployment that is not host-bound. See README §9.

**Comparison to the paper.** DiffusionDrive reports 45 FPS; that is on a data-center GPU, so it is not a
like-for-like comparison with Wormhole N300s. The traced path here is 20.0 FPS at batch=1, full
production resolution, real weights — the same hardware caveat applies to any absolute-FPS comparison.

## 5. Accuracy (unchanged by the perf work)

| Metric | Value |
|---|---|
| trajectory PCC vs PyTorch (real checkpoint, production res) | 0.999777 |
| traced-vs-eager trajectory PCC | 1.000000 |
| NavSim PDM — full navtest (12146 scenes) | 0.8789 (vs 0.8795 CPU reference, 0.8804 paper) |
