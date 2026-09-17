# SD3.5-Large — 4-chip Blackhole QuietBox Optimization Notes (nkira)

Branch: `nkira/sd35-bh-4chip-perf`, cut off Teja's `sd35-bh-qb-enable` @ `07ae2950`.
Config: `1x4cfg0sp0tp1` (`cfg=1, sp=1, tp=4`, Ring topology), 1024x1024, CFG on, bf16, 20 denoising steps.

**Result: 8.54s → 6.77s (-20.7%)**, exceeding the sprint's 15% target. Reproduced across 5+ consecutive PASSED runs (6.76 / 6.78 / 6.77 / 6.78 / 6.77s).

This doc covers the optimization work only. Setup/context (revert of Teja's bf8 quant, the 2x2→1x4 grid switch itself, step count 28→20) is prerequisite work requested by Dalar, not an optimization in itself, and is covered briefly in "Prerequisite changes" below for completeness.

---

## Summary of wins

| Step | Change | Before → After | Δ |
|---|---|---|---|
| 1 | VAE decoder fixes (dead clone, conv mesh-axis, SDPA config) | 8.54s → 8.38s | -1.9% |
| 2 | CCL topology: Linear → Ring | 8.38s → ~7.2s | -14 to -16% |
| 3 | Fused MM+RS+addcmul for FFN's `ff2` | ~7.2s → 6.77s | -6.5% |
| | **Total** | **8.54s → 6.77s** | **-20.7%** |

---

## Prerequisite changes (not optimizations, but required before any of the above)

- **Reverted Teja's bf8 DiT quantization** (weight/activation quant + matmul blockings tuned for it). Per Dalar — the model now always runs in plain bf16.
- **Fixed a VAE submesh-selection bug**: the pipeline's encoder/VAE placement logic assumed any submesh with 4 columns came from a `cfg>1` config with a second cfg-submesh to offload VAE onto. With `cfg=1` on a mesh that's *already* natively `1x4`, there's only one submesh, and the old code indexed a nonexistent second one. Guarded on `len(submesh_devices) > 1` instead.
- **Grid 2x2 → 1x4, steps 28 → 20**: `sp` and `tp` can't share one mesh axis for tensor-sharding purposes — a single tensor can't be sharded along two independent dims on one physical axis (confirmed by a real `TT_THROW` when attempted). So the 1x4 grid goes fully tensor-parallel (`tp=4`, `sp=1`) instead of splitting `sp=2/tp=2` across two axes like the old 2x2 did.

---

## Win 1: VAE decoder fixes (8.54s → 8.38s)

Found via a set of targeted read-spikes over the VAE decoder (`models/tt_dit/models/vae/vae_sd35.py`), each verified on real hardware before landing:

- **Dead `ttnn.clone(x)` in every `ResnetBlock.forward`** — a full-tensor copy that served no purpose (norm/silu/conv never mutate in place; confirmed by comparing against the equivalent, clone-free `VaeResnetBlock` used by other models' VAEs in the shared `vae.py`). 14 occurrences per decode call, some at 1024×1024 resolution. Trivial, zero-risk fix.
- **ResnetBlock convs sharding the wrong (larger) dimension**: `conv1`/`conv_shortcut` always used `out_mesh_axis` (which all-gathers `in_channels` worth of data before the conv), even when `in_channels > out_channels` — in which case sharding on `in_mesh_axis` instead (reduce-scatter of the smaller `out_channels`) moves less data. Fixed to pick whichever of in/out channels is smaller, mirroring the equivalent logic (`out_is_greater`) already used in the shared `vae.py`. Hits hardest on the 512→256ch and 256→128ch (1024×1024, largest stage) transitions.
- **VAE mid-block attention had zero SDPA tuning** — no `program_config`/`compute_kernel_config` at all, unlike the shared `VaeAttention` used elsewhere. Wired in the same defaults (grid-sized `SDPAProgramConfig`, HiFi2 compute kernel) that the more mature sibling implementation already uses.

**One thing tried and reverted**: overriding `GroupNorm`'s default 8×8 core grid to the full 11×10 Blackhole grid. This *sounds* like free parallelism, but `group_norm`'s valid grid is shape-dependent (virtual-row/col constraints tied to each call's `Ht`/`W`/`num_groups`) — confirmed via a real `TT_THROW` that 8×8 was already the *correct maximal* grid for at least one of the VAE's many differently-shaped norm calls. Reverted rather than force it; the original developer's hardcoded 8×8 wasn't blind underutilization everywhere, as first assumed.

Also worth noting: a broader profiling pass showed VAE decode is only ~5-8% of total pipeline time in steady state, so this class of fix alone was never going to reach the 15% target — flagged early, which is why the next two wins (both DiT-side) mattered more.

---

## Win 2: CCL topology Linear → Ring (8.38s → ~7.2s)

This came from checking Jonathan's suggestion to look at whether fused AGMM / fused MMRS / fused RMSNorm / experimental ring attention were actually in use. Tracing the code found the common root cause: **fused AGMM and fused MMRS both gate on `ccl_manager.topology == Topology.Ring`**, but every SD3.5 config in this file — including this one — was hardcoded to `Topology.Linear`.

It wasn't obvious `Ring` would even be *physically valid* here: every other `Topology.Ring` usage anywhere in this codebase is on an 8-chip T3000 or 32-chip Galaxy mesh, never a small 4-chip QuietBox. Tested it directly (with the matching `ring_params_req_exact_devices` device_params, i.e. `fabric_config=FABRIC_1D_RING`) — it works, and correctness (PASSED assertions) was never in question across every run. Switching the `1x4cfg0sp0tp1` config from `Topology.Linear` to `Topology.Ring` was, on its own, the single largest lever found in this whole sprint.

Also checked while here: **fused RMSNorm already appears to be in place** for the one place this model uses `RMSNorm` (attention QK-norm) — it unconditionally calls a fused kernel (`ttnn.experimental.dit_rms_norm_unary_fused`), so nothing to fix there. **Ring attention (regular or experimental)** never triggers under this `sp=1` config either way — the call site is gated on `sequence_parallel.factor > 1`, and this grid deliberately uses `sp=1, tp=4` — so switching to the "experimental" ring-attention op specifically wouldn't do anything here regardless of topology.

---

## Win 3: Fused MM+RS+addcmul for `ff2` (~7.2s → 6.77s)

The FFN's second linear (`ff2`, row-parallel: matmul + reduce-scatter) was doing that reduce-scatter as a separate op, followed by a separate elementwise gate-multiply and residual-add — three ops where one exists (`RowParallelLinear.forward_fused_addcmul`, computing `residual + scalar * ff2(x) * gate` in one kernel). Wiring this in required solving two real problems, not just flipping a flag:

**1. This device's 11×10 grid isn't covered by the op's built-in tuning.** `get_fused_mmrs_config`'s swept table and its v2.3 rule engine both explicitly assume a 12-wide Blackhole grid (`ttnn.CoreCoord(12, 8)` in the default fallback) — this box's grid is 11 wide, and 11 being prime means it can't split evenly the way 12 does either. The default fallback failed outright (`compute_with_storage_grid_size must be <= device grid size`). Fixed by registering an explicit `FusedMMRSConfig` constrained to an **11×8 matmul grid** (fits inside 11×10, leaving 2 rows for the reduce-scatter), with a simple, hand-picked blocking. Validated standalone against a torch reference before wiring in: **PCC ~1.0** at both small and full production scale.

**2. The fused kernel requires batch size 1.** `MinimalMatmulStridedReduceScatterAsync` explicitly asserts `padded_shape[0]==1 and padded_shape[1]==1` — but this model batches CFG conditional+unconditional together as `batch=2` (not split via a separate cfg-parallel mesh axis, since `cfg=1` here). Fixed by flattening `(1, 2, 4096, K) → (1, 1, 8192, K)` before the call and reshaping the result back to `(1, 2, 4096, N)` after. The gate tensor (`spatial_gate_ff`) complicated this further: it's a broadcast-over-tokens shape `(1, 2, 1, D)` with *different values per batch element* (cond vs. uncond gates differ), and the fused kernel has no broadcast support internally — so it has to be materialized to `(1, 2, 4096, D)` via `ttnn.repeat` before flattening, not just reshaped. Also validated standalone against a torch reference with the real batch=2 + broadcast-gate shapes before wiring in: **PCC ~1.0**.

Both fixes are scoped narrowly: the fused path is only taken when the runtime shape exactly matches what's registered (`M=8192, K=2432, N=2432` on this device's 11×10 grid); anything else — a different resolution, a different CFG factor — falls back to the original three-op path unchanged.

---

## Investigated, not adopted: fused AGMM

Also tried wiring up fused AGMM (`to_out`/`to_add_out`'s all-gather + matmul, currently done as two separate ops). Unlike MMRS, this hit what looks like a genuine **kernel-level limitation, not a tuning gap**:

- First attempt (auto-resolved blocking) failed a `K_tiles_per_device % K_block_size == 0` assertion.
- Enabling the heuristic blocking path hit the same class of failure with a different `K_block_size`.
- Tracing further: every grid/worker-count permutation tried (adjusting `num_workers_per_link`, constraining the grid to 10×10 to avoid 11 being prime) hit the **same generic host-level `Illegal NOC usage` safety assertion** during warmup — a program-factory core/NOC assignment conflict, unconditional on blocking parameters.

This points at `all_gather_minimal_matmul_async`'s program factory not correctly handling a **1-row mesh** (1×4) — every existing usage of this op anywhere in this codebase is on a genuine 2D mesh (4×8, 8×8, 12×9), never a degenerate 1-row shape. This isn't fixable from Python/model-level config; it needs actual kernel engineering on the op itself. Left unwired.

---

## How to reproduce

```bash
cd tt-metal && source python_env/bin/activate  # or venv, depending on checkout
export TT_METAL_HOME=$(pwd)
export ARCH_NAME=blackhole
export PYTHONPATH=$TT_METAL_HOME
unset TT_MESH_GRAPH_DESC_PATH   # let auto-discovery pick the mesh graph descriptor
unset TT_DIT_CACHE_DIR          # causes a hang during on-device weight distribution on this setup

NO_PROMPT=1 pytest models/tt_dit/tests/models/sd35/test_pipeline_sd35.py \
  -k "test_sd35_pipeline and 1x4cfg0sp0tp1 and yes_traced" -x -s
```

See `StableDiffusion35.md`'s "Blackhole (4-chip QuietBox, 1x4 mesh)" section for the full run instructions and updated performance table.
