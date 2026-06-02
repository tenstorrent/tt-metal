# pi0.5 Option B — implementation status (2026-06-02, updated 19:06 UTC)

**Branch**: `sdawle/dvartanians/pi0.5_openpi_upstream_blaze`
**Goal**: 4-stage pipeline (vision + VLM/2 + VLM/2 + expert/denoise) on 32-chip
Blackhole Galaxy with TP=8 within each stage. See
`PI0_5_GALAXY_DEPLOYMENT_PLAN.md` §3 + §6 for design.

**2026-06-02 19:06 UTC update**: tt-metal rebuilt (build dir timestamp
2026-06-02 18:56). `get_cluster_type()` now returns `ClusterType.BLACKHOLE_GALAXY`.
Both initial smoke tests pass; the build mismatch is resolved.

**2026-06-02 22:21 UTC — Denoise loop wired end-to-end. 18 smoke tests pass
in 396s on g11blx01:**

### Denoise loop landed

`Stage3Expert.denoise(noisy_actions, prefix_kv_cache, attention_mask)` runs
the 10-step Euler integrator on stage 3's submesh:
  - `Pi0_5SubmeshSuffixSlice` holds action_in_proj, time_mlp_in/out,
    action_out_proj (replicated, ~2 MB total at bf8).
  - Each step computes `adarms_cond` from a scalar timestep (sincos
    computed on HOST in torch, then MLP+silu+MLP+silu on device — bypass
    the on-device sincos's tiny matmul that picks an invalid program-config
    on MeshDevice).
  - Suffix embed → expert forward (replicated 18 layers, consumes
    migrated VLM KV via past_key_value) → project_output → Euler update.

E2E real-weights perf (first run, includes JIT setup):
```
VLM stages 1+2:           231.3 ms
KV migrate:                19.0 ms
denoise(10 steps):       3986.8 ms   (399 ms/step incl. step-1 cold JIT)
─────────────────────────
total e2e:               4237.0 ms
clean_actions shape:     [1, 64, 32]  ← [B, padded horizon, action_dim]
clean_actions mean_abs:  0.0772 (finite, no NaN/Inf)
```

Warm-pass denoise will be a follow-up measurement (step 1 cold, steps 2-10
should warm). The denoise loop runs correctly end-to-end now.

**2026-06-02 21:51 UTC — Real-weights PCC validated. 16 smoke tests pass
in 342s on g11blx01** (kernels fully cached, 3 new PCC tests included):

### Real-weights PCC validation (17 tests in 353s)

Pi05 upstream LIBERO checkpoint loaded via `Pi0_5WeightLoader`. Threshold 0.99.

**Submesh-vs-submesh PCC** (TP=8 vs the already-validated replicated path,
both on a 4×2 submesh with same input):

| Test | PCC |
|---|---:|
| VLM layer 0 | **0.996749** |
| VLM layers 0–9 (stage-1 workload) | **0.997468** |
| Expert layer 0 (adaRMS) | **0.997576** |

**Single-chip torch baseline PCC** (TP=8 TTNN vs pure-PyTorch CPU fp32
`GemmaBlock`, real layer-0 weights, same input):

| Test | PCC |
|---|---:|
| **TP=8 TTNN vs torch fp32 reference (VLM layer 0)** | **0.996184** |

All paths produce structurally correct outputs vs both the validated
TTNN replicated reference AND the canonical PyTorch fp32 CPU reference.
The numerical drift comes from bf8 quantization on weights + activations
and from different op orderings (fused-QKV in single-chip path vs
unfused-Q + KV-fused in TP path). PCC ≥ 0.99 means the linear correlation
is strong — the TP=8 math is verified correct.

**2026-06-02 21:03 UTC — TP=8 expert landed; full TP path warm E2E = 186.5 ms.
13 smoke tests pass in 284.8s on g11blx01** (kernels fully cached):

| Test                                                | Wall  | What it proves                                              |
|-----------------------------------------------------|-------|-------------------------------------------------------------|
| `test_default_layout_shape`                         | <1s   | StageLayout dataclass                                       |
| `test_open_32_chip_mesh_and_partition`              | ~13s  | 8×4 parent + 4× (4,2) submeshes open & close cleanly        |
| `test_vlm_slice_forward_one_layer_on_submesh`       | ~13s  | Pi0_5SubmeshVLMSlice forward on submesh 1 — 1 layer (replicated) |
| `test_expert_slice_forward_one_layer_on_submesh`    | ~12s  | Pi0_5SubmeshExpertSlice (adaRMS) forward on submesh 3       |
| `test_inter_submesh_host_bounce_transport`          | ~13s  | send_activation_via_host between submesh 1 and submesh 2    |
| `test_e2e_vlm_to_expert_shrunk_pipeline`            | ~17s  | Stage 1 → 2 → KV-migrate → stage 3 with depth-2 VLM / depth-1 expert |
| `test_vlm_slice_tp_shard_one_layer`                 | ~13s  | TP=8 sharded single VLM layer at real Gemma-2B width        |
| `test_vlm_slice_tp_shard_nine_layers`               | ~17s  | 9 TP=8 layers — the real Option B per-stage workload        |
| `test_tp_vlm_kv_migrate_to_replicated_expert`       | ~13s  | TP=8 VLM captures KV → host-bounce → expert cross-attention |
| `test_e2e_real_config_tp_vlm_replicated_expert`     | ~66s  | Real Gemma-2B (depth=18) + Gemma-300M (depth=18) E2E (TP VLM, repl expert) |
| `test_expert_slice_tp_shard_one_layer`              | ~27s  | TP=8 expert single layer at Gemma-300M dims                 |
| `test_expert_slice_tp_shard_eighteen_layers`        | ~26s  | TP=8 expert full 18-layer stage                             |
| `test_e2e_real_config_full_tp`                      | ~67s  | **Full TP=8 (VLM + expert) E2E with warm-pass perf**        |

### Real-config E2E perf — full TP=8 (2026-06-02 21:03, warm pass)

Prefix 64 / suffix 64, second forward in the same session (kernels cached):

```
[perf] stage 1 (TP=8 VLM, 9 layers):           31.2 ms
[perf] transport 1→2 (host bounce):             2.5 ms
[perf] stage 2 (TP=8 VLM, 9 + norm):           31.5 ms
[perf] KV migrate (18 × (K,V) host bounce):    37.6 ms
[perf] stage 3 (TP=8 expert, 18 layers):       83.8 ms
[perf] ──────────────────────────────────────────
[perf] total warm e2e:                        186.5 ms
```

First-run E2E for comparison (includes first-call program-config setup of
all kernels):

```
[perf] stage 1: 127.3 ms  transport: 7.3 ms  stage 2: 69.7 ms
[perf] KV migrate: 20.0 ms  stage 3 (TP expert): 170.5 ms
[perf] total first-run: 394.8 ms
```

### TP=8 expert vs replicated expert (warm, same workload)

| Stage 3 variant | Warm time | Notes |
|---|---:|---|
| Replicated expert (18 layers) | 127.9 ms | full weights on each chip; no all_reduce |
| **TP=8 expert (18 layers)**   | **83.8 ms** | 1 head/chip; +adaRMS Dense replicated; +2 all_reduce/layer |

TP=8 expert is **1.5× faster** than replicated despite the extra adaRMS and
all_reduce overhead, because the MLP gate/up/down sharding dominates the
saving.

### Memory footprint per chip (real config, depth=18 each)

Stage 1 / 2 (TP=8 VLM, 9 layers each, all kernels JIT'd and cached):
~125 MB / chip (well inside 180 MB cap).

Stage 3 (TP=8 expert, 18 layers, adaRMS Dense replicated):
~100 MB / chip (the replicated 6W modulation Dense is 3 MB/layer at bf8 ×
18 layers = 54 MB; the rest is sharded attn + MLP).

### Real-config E2E perf (first measurement, 2026-06-02 20:40)

Prefix len 64 (tile-aligned; real prefill will be 544–968), suffix len 64:

```
[perf] stage 1 forward (TP=8, 9 VLM layers):              126.1 ms
[perf] transport submesh1 → submesh2 (host bounce):       406.0 ms
[perf] stage 2 forward (TP=8, 9 VLM layers + norm):        70.2 ms
[perf] KV migration (18 layers × (K,V), host bounce):      20.9 ms
[perf] stage 3 forward_expert_step (replicated, 18 exp):  127.9 ms
[perf] ────────────────────────────────────────────────────────
[perf] total e2e:                                         751.2 ms
```

Init costs (one-time per session):
- Weight construction (CPU torch random): 13.5 s
- Stage 1 init (TP=8 weight upload + 9-layer block construction): 15.1 s
- Stage 2 init: 14.5 s
- Stage 3 init (replicated, 18 layers): 4.3 s

**Notes on this perf:**
- This is a **first-measurement baseline**, not optimized — the analytical
  perf model predicts ~7 ms E2E. Big gaps to close:
  - **Transport host-bounce dominates (406 ms / 54% of E2E).** A direct
    D2D `ttnn.copy` or fabric-socket transport would cut this to <5 ms.
    KV migration host-bounce is already small (21 ms for 36 tensors) so
    the main win is the single-activation transport.
  - Stage 1 first call (126 ms) vs stage 2 (70 ms) — 56 ms difference is
    likely first-call kernel program-config setup on stage 1's submesh.
    Stage 2 reuses warm kernels.
  - Stage 3 (128 ms) at depth=18 replicated; TP=8 expert will cut this.
  - Prefix length 64 is short; real prefill is 968 → matmul cost will
    scale up but transport stays constant.

### TP=8 design notes

The 4-stage pipeline is structurally complete and the **real per-stage
workload (9 TP=8-sharded VLM layers) fits and runs end-to-end on a 4×2
submesh of a Blackhole Galaxy**.

### TP=8 design notes

The TP path lives in `tp_block.py` and is gated behind
`Pi0_5SubmeshVLMSlice(..., tp_shard=True)`:

- **Q col-parallel** (sharded along output dim 1, 8 heads → 1 head/chip).
- **K, V replicated** (num_kv_heads=1 doesn't split into 8; KV is small
  enough that replication is cheap).
- **O row-parallel + all_reduce** (sharded along input dim 0, partial sum
  → `ttnn.all_reduce(cluster_axis=None)`).
- **MLP gate, up col-parallel** (output dim sharded).
- **MLP down row-parallel + all_reduce** (input dim sharded).
- **Norms replicated** (small, broadcast-friendly).
- **Unfused QKV** — the single-chip's fused QKV concat doesn't divide cleanly
  with num_kv_heads=1, so the TP path emits three separate linears.

### Fabric init recipe

Collective ops (`ttnn.all_reduce`) need the fabric initialized **before**
`open_mesh_device`. `open_galaxy_mesh(layout, enable_fabric=True)` handles
this — sets `FabricConfig.FABRIC_1D` on entry, `FabricConfig.DISABLED` on
exit. Without it, `ttnn.all_reduce` errors with
`TT_FATAL: Trying to get un-initialized fabric context`.

---

## What was built this session

### Module scaffolding — `models/experimental/pi0_5/tt/option_b/`

| File                 | Purpose                                      | State            |
|----------------------|----------------------------------------------|------------------|
| `__init__.py`        | Package marker, exports `StageLayout`        | ✓ Complete       |
| `README.md`          | File map + key tt-metal APIs we depend on    | ✓ Complete       |
| `stages.py`          | `StageSpec` / `StageLayout` dataclass + `build_default_layout()` | ✓ Complete |
| `mesh_setup.py`      | `open_galaxy_mesh()` ctx manager: opens 8×4 → slices 4× 4×2 submeshes | ✓ Complete |
| `transport.py`       | `send_activation_via_host()` — host-bounce inter-stage transfer | ✓ Working (uses `get_device_tensors[0]` to extract one replica) |
| `kv_migration.py`    | `KVMigration` class — one-shot VLM K/V → expert submesh | ✓ Host-bounce, not yet exercised E2E |
| `vlm_slice.py`       | `Pi0_5SubmeshVLMSlice` — submesh-aware layer range, replicated weights | ✓ **Forward verified on real HW** |
| `expert_slice.py`    | `Pi0_5SubmeshExpertSlice` — submesh-aware expert (adaRMS) layer range | ✓ **Forward verified on real HW** |
| `vision_slice.py`    | `Pi0_5SubmeshVisionSlice` — host SigLIP + projector + host embed lookup | ✓ Wired (host-side SigLIP, hardware untested but no on-device kernels) |
| `stage_0_vision.py`  | Thin wrapper around `Pi0_5SubmeshVisionSlice` | ✓ Wired |
| `stage_vlm.py`       | Thin wrapper around `Pi0_5SubmeshVLMSlice`    | ✓ Wired, exercised by smoke test |
| `stage_3_expert.py`  | Thin wrapper around `Pi0_5SubmeshExpertSlice` + suffix/denoise stubs | 🟡 Slice exercised; `denoise()` still NotImplementedError |
| `pipeline.py`        | `Pi0_5PipelineB.run_one_step` orchestrator (vision → 1 → 2 → KV-migrate → expert step) | ✓ Wired against new APIs; runnable once TP=8 sharding lands |

### Tests — `models/experimental/pi0_5/tests/`

| Test                          | What it checks                              | State           |
|-------------------------------|---------------------------------------------|-----------------|
| `test_option_b_smoke.py::test_default_layout_shape` | Layout dataclass shape (no HW) | ✓ Passing |
| `test_option_b_smoke.py::test_open_32_chip_mesh_and_partition` | 32-chip mesh open + 4× 4×2 submesh partition | ✓ **Passing on g11blx01 (20.24s call duration)** |

**Build-vs-hardware mismatch is resolved.** The earlier May 20 tt-metal build
predated the `ClusterType` enum entry for this Galaxy SKU; the new build
(`build/lib/_ttnn.so` timestamp 2026-06-02 18:56) returns
`ClusterType.BLACKHOLE_GALAXY` cleanly. **Last-session reboot was caused by an
open_mesh hang on the old build** — the safe-mesh-open recipe below avoids
that even if a future build regresses.

### Safe mesh-open recipe (to avoid the reboot-required hang)

Before touching `open_mesh_device()` from any script — especially after a
build or driver update — do this:

```bash
# 1. Confirm no stale processes hold the device.
ps -ef | grep -Ei "ttnn|tt_metal|pytest|python.*pi0" | grep -v grep
ls /dev/tenstorrent/             # expect 32 device nodes + by-id
lsof /dev/tenstorrent/0 2>/dev/null    # expect no output

# 2. Probe get_cluster_type() FIRST in a subprocess with a hard timeout.
timeout --kill-after=5 30 python -c "
import ttnn; print(ttnn._ttnn.cluster.get_cluster_type())
"
# If this hangs or errors, do NOT call open_mesh_device. Rebuild / debug first.

# 3. Run the actual test under setsid + timeout so a hang dies cleanly.
setsid timeout --kill-after=10 --foreground 240 \
  python -m pytest models/experimental/pi0_5/tests/test_option_b_smoke.py -v -s
```

The `setsid` ensures any subprocess pytest spawns gets SIGKILL'd as a group
when the outer timeout fires — so a wedged device-driver wait can't strand
file descriptors that would require a reboot to release.

---

## What's not yet built (call-out for next session)

1. **Submesh-aware backbone wrapper** (task #6). `Pi0_5PaliGemmaBackboneTTNN`
   takes a single `ttnn.Device` today (`ttnn_paligemma.py:51`). The wrapper needs
   to accept a `submesh` plus a `vlm_layer_range` tuple, and only allocate the
   layer-slice it owns. Largest single piece of work — touches block construction,
   weight sharding, and the `forward_vlm` / `forward_expert` iterators.

2. **TP=8 weight sharding inside each stage**. Stages 1, 2, 3 need:
   - Q/K/V/gate/up weights col-parallel sharded on submesh.
   - O/down weights row-parallel sharded.
   - `ttnn.all_reduce(cluster_axis=...)` inserted after row-parallel matmuls.
   - The current per-layer weight uploads in `ttnn_paligemma.py` and `ttnn_gemma.py`
     pass a single device; they need a `mesh_mapper` parameter and the matmul
     program configs need shard-aware sizing.

3. **SigLIP TP=8 sharding** (Stage 0). SigLIP layers are width=1152 (not div 8),
   intermediate=4304 (not div 8). Existing BS path already pads intermediate to
   4608. For TP=8, also need to pad attention width 1152 → 1280 (next multiple of
   128), OR pick TP=4 within stage 0 with 2-way replication.

4. **KV migration — D2D fast path**. Current implementation host-bounces (slow,
   ~2-5 ms for 9 MB). Need to investigate:
   - Whether `MeshDevice.copy()` or similar exposes a submesh→submesh primitive.
   - Whether we can construct a parent-mesh view that overlaps both submeshes
     and reshape across them.
   - Worst case: keep host bounce, profile, decide later if optimization needed.

5. **Inter-stage activation transport — D2D fast path**. Same status as #4.

6. **adaLN modulation precompute** (already in main `ttnn_pi0_5_model.py` via
   `_precompute_bs1_modulations` — needs to be replicated on stage 3's submesh,
   ~1 MB).

7. **End-to-end inference test** comparing Option B output vs single-device
   `Pi0_5ModelTTNN` baseline. PCC threshold: matches existing single-device
   PCC (≥0.99) within bf8 rounding noise.

---

## Tasks remaining

- **#4** [done] Verify 32-chip mesh opens on g11blx01 — both smoke tests pass.
- **#5** [done] Scaffold Option B module layout.
- **#6** [done] Wrap Pi0_5PaliGemmaBackboneTTNN for submesh placement — see
  `vlm_slice.py` (`Pi0_5SubmeshVLMSlice`) and `expert_slice.py`
  (`Pi0_5SubmeshExpertSlice`). Replicated weights for now.
- **#7** [done] Implement KV migration primitive (host-bounce fallback).
- **#8** [done] Host-bounce transport between submeshes (`transport.py` —
  smoke-tested with `get_device_tensors[0]` + `replicate_tensor_to_mesh_mapper`).
- **#9** [done] TP=8 weight sharding for VLM slice (see `tp_block.py`,
  `Pi0_5SubmeshVLMSlice(..., tp_shard=True)`). 9 layers on a 4×2 submesh.
- **#10** [done] TP=8 KV-cache plumbing — TP block emits replicated
  `[B, num_kv_heads, S, head_dim]` K/V via `use_cache=True`, host-bounce
  migrates to the expert submesh, replicated expert consumes via
  `past_key_value` and runs cross-attention. Verified end-to-end by
  `test_tp_vlm_kv_migrate_to_replicated_expert`.
- **#11** [done] TP=8 sharding for expert slice — see `tp_expert_block.py`
  and `Pi0_5SubmeshExpertSlice(..., tp_shard=True)`. adaRMS modulation Dense
  is replicated per-chip (~3 MB / layer at bf8); attention + MLP follow the
  same Q-col / KV-replicated / O-row / MLP-col-then-row pattern as the VLM
  TP block. Warm 18-layer forward 83.8 ms.
- **#12** [pending] Suffix MLP + 10-step Euler denoise wiring in
  `stage_3_expert.Stage3Expert.denoise`.
- **#13** [done] End-to-end shrunk-config smoke test with a depth-2 VLM /
  depth-1 expert PaliGemmaConfig.
- **#14** [done] End-to-end smoke test with TP=8 VLM (depth=18) + replicated
  expert at real config — `test_e2e_real_config_tp_vlm_replicated_expert`
  passes; baseline perf 751 ms E2E.
- **#15** [pending] On-device SigLIP slice (`Pi0_5SubmeshSigLIPSlice`) with
  TP=8 sharding — replaces the host-SigLIP fallback in `vision_slice.py`.

---

## Final perf comparison summary (analytical, from §6 of deployment plan)

| Metric                  | A (TP=32)         | **B (4×8, lead)** | C (heterogeneous 28-chip) | C' (1×2 submeshes)   |
|-------------------------|-------------------|-------------------|---------------------------|----------------------|
| End-to-end latency      | 6.82 ms           | **7.09 ms**       | 8.90 ms                   | 8.26 ms              |
| All-reduce overhead     | 6.57 ms (96%)     | **5.93 ms**       | 0 ms                      | 3.39 ms              |
| Throughput (pipelined)  | 147 inf/s         | **3107 inf/s**    | 1180 inf/s                | 2360 inf/s           |
| Chips used              | 32                | **32**            | 28 (4 spare)              | 56 (over by 24)      |
| Memory tightness        | OK                | **148.5 MB tight** | 156.9 MB very tight       | varies               |
| tt-blaze compatibility  | None (no pipeline) | **Direct** (uniform 4×2 = tt-blaze default) | Needs SubmeshPartition extension | Needs +1 Galaxy |

**Decision (per user, this session):**
- ✅ **Lead with Option B** — uniform 4×2 submeshes are the tt-blaze canonical
  shape; this option needs no new tt-blaze primitives beyond a SigLIP FusedOp
  and one-shot KV migration.
- ✅ **Iterate toward Option C** after B lands. Requires extending tt-blaze's
  `SubmeshPartition` for per-stage shapes, but pays back +1.8 ms latency
  reduction and zero-collective overhead.
- ❌ **Drop Option A** — 96% of e2e is all-reduce on this workload.
- ❌ **Drop Option C'** — exceeds 32 chips.

---

## Caveats on this status doc

- All perf numbers in the table above are analytical (from `perf_model.py`),
  not measured. Real numbers need the build + run cycle that hasn't happened yet.
- The "tt-blaze compatibility" row reflects today's tt-blaze (`SubmeshPartition`
  hard-coded uniform-shape; no vision-encoder precedent; no cross-stage KV op).
  All three of those are tractable extensions.
- Memory-tightness flags ignore kernel binaries and CB scratch; both can take
  another 5-15 MB / chip in practice.
