# Option C — Sharded matmul within stage (TP within stage) plan

**Status**: Research complete (2026-06-03). Implementation not yet started.
**Source**: 4-agent research workflow + synthesis (`wf_63b15681-e43`).

This document is the synthesized output of the research. It is the
load-bearing decision artifact for whether/how to pursue TP-within-stage
as a workaround for the MLP CB clash documented in
[OPEN_ISSUE_MLP_CB_CLASH.md](./OPEN_ISSUE_MLP_CB_CLASH.md).

---

## Headline finding

**TP=N within stage works for prefill but NOT for denoise** at the current
Option C chip counts. The asymmetry comes from how layers map to chips:

- **Prefill (6,3) = 18 chips, 18 VLM layers**: in the layer-paired plan
  each chip carries 1 layer. TP=2 via `(2,1)` col-pairs gives 9
  sub-meshes × 2 layers/sub-mesh, with per-layer weights split N-ways:
  per-chip drops from **110 MB → 55 MB** → 0.46 MB/bank → **below the
  0.70 MB CB threshold**. ✓
- **Denoise (6,1) = 6 chips, 18 expert layers**: each chip carries 3
  layers. Under TP=N, sub-meshes drop from 6 to 6/N but layers per
  sub-mesh climb from 3 to 3N, so per-chip load = `(3N · 32.8) / N =
  98.4 MB` — **invariant for any N**. Still ~0.72 MB/bank, still clashes.

**TP within stage is not a uniform fix.** It works where chips ≥ layers
(prefill) and breaks where chips < layers (denoise).

---

## Decisions

### Prefill: **TP=2 via `(2,1)` col-pairs**

| Sub-layout | Tiles (6,3)? | Per-chip wt | Per-bank | CB ok? |
|---|---|---|---|---|
| TP=2 `(2,1)` col-pairs (9 sub-meshes × 2 layers) | yes | **55 MB** | **0.46 MB** | ✓ |
| TP=2 `(1,2)` row-pairs | no — 3 not divisible by 2 | — | — | — |
| TP=3 `(1,3)` stripes (6 sub-meshes × 3 layers) | yes | 110 MB | 0.92 MB | ✗ |
| TP=6 `(2,3)` (3 sub-meshes × 6 layers) | yes | 110 MB | 0.92 MB | ✗ |

TP=2 col-pairs is the only choice — TP=3+ doesn't help because layers
per sub-mesh scales up to match the weight reduction.

### Denoise: **do nothing in this PR**

Per-chip weight is invariant under TP-within-(6,1). Reshaping (6,1) →
(3,2) gives the same 0.72 MB/bank with significant code churn (2-D
walker in `expert_slice.py:389`, layout redesign in `stages.py:55-58`).
Not worth it.

**Real denoise fix** is one of:
1. Shard the adaRMS modulation Dense `[1024, 6144]` along its 6144 axis
   (`expert_slice.py:314`, mirror `tp_expert_block.py:163`). Saves
   ~12 MB/layer × 3 = 36 MB/chip → 0.57 MB/bank → fits.
2. Drop adaRMS mod weight bf16 → bf8. Saves ~18 MB/chip → same effect.
3. The structural MLP DRAM-bounce per `vlm_slice.py:298` rms_norm
   pattern (the issue logged in `OPEN_ISSUE_MLP_CB_CLASH.md`).

All three are separate PRs. Out of scope for the prefill TP=2 work.

---

## Code surface (prefill TP=2 only)

| File | Change |
|---|---|
| `tt/option_c/stages.py:55-58` | Add `tp_size: int = 1` to prefill `StageSpec`; default `2` behind a flag. |
| `tt/option_c/mesh_setup.py` | Add `create_tp_submeshes(stage_submesh, tp_shape=MeshShape(2,1))` carver; wire `set_fabric_config(FABRIC_1D)` at Galaxy open (mirror `option_b/mesh_setup.py:44-50`). |
| `tt/option_c/vlm_slice.py` | New `Pi0_5OptionCVLMSliceTP` (or `tp_shard=True` branch) using TP block on each `(2,1)` child instead of replicating on the (6,3) stage. 9 sub-meshes × 2 layers each. |
| `tt/option_c/tp_block.py` (new) | Near-verbatim copy of `models/experimental/pi0_5/tt/option_b/tp_block.py`; thread `tp_size=2`; KV stays replicated (num_kv_heads=1 still doesn't divide). **`in0_block_w` retuning needed** — TP=8 used `8/4`; TP=2 per-chip widths are 4× larger. |
| `tt/option_c/stage_prefill.py` | Route forward through the new per-sub-mesh TP block; replicate activation into each `(2,1)` sub-mesh; all_reduce output → DRAM (`option_b/tp_block.py:215-224` pattern, TP-agnostic). |
| `tt/option_c/__init__.py` | Re-export new TP slice + flag. |

**Untouched**: `stage_denoise.py`, `expert_slice.py` (denoise is out of scope).

---

## API risks (from the ttnn-API research)

| Surface | Status | Notes |
|---|---|---|
| `ttnn.shard_tensor_to_mesh_mapper(submesh, dim)` | works | Any submesh shape; flattens to 1-D; valid for `(2,1)` |
| `ttnn.all_reduce(t, cluster_axis=..., topology=Linear, num_links=1)` | works | Pass `cluster_axis=None` or explicit; Ring auto-downgrades to Linear on a 2-chip axis |
| `MeshDevice.create_submesh` nested (grandparent → parent → child) | works | No `is_parent_mesh` guard; chain via `parent_mesh_` |
| `(1,2)` tiling on (6,3) | **does not tile** | Use `(2,1)` col-pairs; cols are 3 (not div by 2), rows are 6 (div by 2) |
| `set_fabric_config(FABRIC_1D)` | works | Must be called BEFORE parent open; covers all descendants |
| `num_links > 1` on 2-chip axis | unknown | Option B uses `num_links=1` unconditionally; defer tuning |
| `in0_block_w` for TP=2 shapes | needs measurement | TP=8 used `8/4` at `tp_block.py:313/389/412`; retune for TP=2 |

---

## Test sequence

1. **Smoke** — extend `tests/test_option_c_smoke.py` with a `--tp-size 2`
   path that opens the prefill (6,3) stage, carves 9 × `(2,1)` children,
   runs one VLM block forward, asserts no crash + output shape.
2. **L1 probe** — re-run `test_option_c_l1_footprint_probe.py` with the
   new flag. Expected: per-chip drops from 119 MB → ~64 MB (55 MB
   weights + ~9 MB transient activations); per-bank 0.92 → ~0.53 MB.
   **CB threshold cleared.**
3. **PCC** — new `tests/test_option_c_tp_pcc.py`: single VLM block
   forward at TP=1 vs TP=2 vs torch reference; assert PCC > 0.999
   against torch. Mirror Option B's PCC harness.
4. **Full prefill e2e** at `vlm_depth=18` only after (1)–(3) pass.

---

## Scope vs alternative

| Path | Scope | Risk | Pi0_5 perf benefit |
|---|---|---|---|
| **Prefill TP=2 within stage** (this plan) | medium (~6 files; 2 near-verbatim from Option B) | low — APIs all in production use | unblocks L1-resident prefill weights, CB-clash gone |
| **tt-blaze port** | large — new runtime, new collective semantics, full re-validation | medium — first pi0.5 port to tt-blaze | targets the deployment plan's 8.9 ms total |

Strongly prefer the prefill-TP=2 path as the next incremental step.

---

## Recommended first PR (minimum to prove CB clash gone)

**Scope: one VLM layer, TP=2, on a single `(2,1)` sub-mesh of prefill — smoke + footprint only. No PCC, no full depth.**

1. Add `create_submesh_tp2(stage_submesh, offset=(0,0)) -> MeshDevice`
   returning a single `(2,1)` child in `tt/option_c/mesh_setup.py`;
   ensure `set_fabric_config(FABRIC_1D)` is wired at parent open.
2. Copy `models/experimental/pi0_5/tt/option_b/tp_block.py` →
   `models/experimental/pi0_5/tt/option_c/tp_block.py`; default
   `tp_size=2`; leave `in0_block_w` values for now (document as TBD).
3. Add `--tp-size` flag to `tests/test_option_c_smoke.py` that opens
   Galaxy → carves (6,3) prefill → carves one `(2,1)` child → uploads
   one VLM layer's weights via the new `tp_block` → runs a forward pass
   with random inputs → asserts shape + no crash.
4. Re-run L1 footprint probe with the same flag; record per-chip MB +
   per-bank MB.
5. Assert per-bank weight load ≤ 0.55 MB (target 0.46 MB). If above,
   retune `in0_block_w` and re-measure before merge.
6. Defer full 9-sub-mesh tiling, PCC, denoise modulation sharding, and
   MLP DRAM-bounce to follow-ups.

---

## Pointers

- Critical files: `option_b/tp_block.py`, `option_c/vlm_slice.py`,
  `option_c/mesh_setup.py`, `option_c/stages.py`,
  `tests/test_option_c_smoke.py`.
- Open issue: [OPEN_ISSUE_MLP_CB_CLASH.md](./OPEN_ISSUE_MLP_CB_CLASH.md)
  — the reason TP-within-stage is being considered.
- Measurement context:
  [OPTION_C_L1_FOOTPRINT_PROBE.md](./OPTION_C_L1_FOOTPRINT_PROBE.md)
  — per-chip / per-bank numbers come from here.
- Deployment plan: [PI0_5_GALAXY_DEPLOYMENT_PLAN.md](./PI0_5_GALAXY_DEPLOYMENT_PLAN.md)
  §3 for the original analytical layout.
