# FIBO denoise matmul tuning (Phase 1) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the generic `(8,8,8)` matmul-block fallback that every FIBO denoise matmul currently hits with per-shape tuned `MinimalMatmulConfig` blockings, registered additively, to raise matmul FLOP utilization on the 2×2 Blackhole mesh without changing numerics.

**Architecture:** Enumerate the exact per-device `(M,K,N,grid)` of every matmul in one full-depth denoise forward → feed those shapes into the existing device-profiler block-size sweeper (`utils/sweep_mm_block_sizes.py`) → register the winning blocks via `register_matmul_configs()` from the FIBO transformer module at import time. Block sizes are a pure tiling choice (same accumulation, same result), so this is numerically neutral; correctness is confirmed with the existing PCC test and the win is measured by re-profiling.

**Tech Stack:** tt-metal / ttnn, `ttnn.experimental.minimal_matmul`, `MinimalMatmulConfig`, Tracy device profiler, `tt-perf-report`, pytest.

## Global Constraints

- Target device/topology: **2×2 Blackhole mesh, sp=2 (axis 0), tp=2 (axis 1)**, `num_links=1`, `topology=Linear`, `fabric_config=FABRIC_1D`. Values copied from `pipeline_bria_fibo.py` (`sp=(2,0)`, `tp=(2,1)`) and the profile test's `_PROFILE_DEVICE_PARAMS`.
- FIBO config (verbatim from cached `transformer/config.json`): `num_attention_heads=24`, `attention_head_dim=128` → `inner_dim=3072`; `num_layers=8`, `num_single_layers=38`; `in_channels=48`; `joint_attention_dim=4096`; `text_encoder_dim=2048`; `patch_size=1`.
- Correctness gate (per decision — **PCC only**): `test_fibo_transformer_mesh` must pass `pcc=0.99` after the registration change. Block-size registration is numerically neutral, so this test's role is to confirm the mechanism didn't break the build/run; production-shape config *validity* is confirmed by the Task 4 re-profile completing with a correctly shaped output.
- Env prefix for every device run in this repo: `HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole python_env/bin/python`.
- Phase 1 is **additive** — no change to shared block/attention/linear *behavior*. The only temporary edit to shared code (the Task 1 dump shim in `linear.py`) is reverted in Task 3.
- Scratch artifacts (dump JSONL, collector script) go in the session scratchpad, not the repo.

---

## File map

- `models/tt_dit/layers/linear.py` — **temporary** env-gated dump shim (Task 1), reverted in Task 3.
- `models/tt_dit/utils/sweep_mm_block_sizes.py` — **permanent** additions: a `bh_2x2` device config + FIBO shapes in `SHAPES` (Task 2).
- `models/tt_dit/models/transformers/transformer_bria_fibo.py` — **permanent**: a module-level `_register_fibo_matmul_configs()` call registering the winning blocks (Task 3).
- Scratchpad: `fibo_mm_dump.jsonl`, `collect_fibo_shapes.py` (Task 1), sweep CSV parsing (Task 2). Not committed.

---

### Task 1: Enumerate FIBO's denoise matmul shapes

**Files:**
- Modify (temporary): `models/tt_dit/layers/linear.py` — add env-gated dump helper + call it from the 4 `get_matmul_config` sites.
- Create (scratch): `<scratchpad>/collect_fibo_shapes.py`
- Produces: `<scratchpad>/fibo_mm_dump.jsonl` and a printed shape table.

**Interfaces:**
- Produces for Task 2: a deduped list of `(M, K, N, grid_x, grid_y, is_agmm, use_case)` tuples — the exact per-device kernel dims FIBO's denoise forward issues.

- [ ] **Step 1: Add the dump shim to `linear.py`.**

At the top of `models/tt_dit/layers/linear.py`, after the existing imports, add:

```python
import json as _json
import os as _os


def _dump_mm_shape(M, K, N, core_grid, *, is_agmm, use_case):
    """Env-gated one-line-per-matmul shape dump for offline block-size tuning.

    No-op unless FIBO_MM_DUMP is set. Writes JSONL to $FIBO_MM_DUMP so an offline
    collector can build the sweeper's SHAPE table. Temporary tuning instrument.
    """
    path = _os.environ.get("FIBO_MM_DUMP")
    if not path:
        return
    gx = getattr(core_grid, "x", None)
    gy = getattr(core_grid, "y", None)
    with open(path, "a") as f:
        f.write(_json.dumps({"M": int(M), "K": int(K), "N": int(N),
                             "gx": gx, "gy": gy, "is_agmm": is_agmm, "use_case": use_case}) + "\n")
```

Then add a `_dump_mm_shape(...)` call immediately after each `matmul_config = get_matmul_config(...)` line, passing the use-case for that site. There are 4 sites; use these exact `is_agmm`/`use_case` values:

- `Linear.forward` (~line 79), after `matmul_config = get_matmul_config(M, K, N, core_grid, default_block_size)`:
  ```python
  _dump_mm_shape(M, K, N, core_grid, is_agmm=False, use_case=("plain_gelu" if self.fused_activation_fn == (ttnn.UnaryOpType.GELU, False) else "plain"))
  ```
- `ColParallelLinear.forward` AGMM branch (~line 215), after its `matmul_config = get_matmul_config(...)`:
  ```python
  _dump_mm_shape(M, K, N, core_grid, is_agmm=True, use_case=("plain_gelu" if self.fused_activation_fn == (ttnn.UnaryOpType.GELU, False) else "plain"))
  ```
- `ColParallelLinear.forward` non-AGMM branch (~line 250), after its `matmul_config = get_matmul_config(...)`:
  ```python
  _dump_mm_shape(M, K, N, core_grid, is_agmm=False, use_case=("plain_gelu" if self.fused_activation_fn == (ttnn.UnaryOpType.GELU, False) else "plain"))
  ```
- `RowParallelLinear.forward` (~line 370), after its `matmul_config = get_matmul_config(...)`:
  ```python
  _dump_mm_shape(M, K, N, core_grid, is_agmm=False, use_case="plain")
  ```

- [ ] **Step 2: Run one full-depth denoise forward with the dump on.**

Run (plain pytest, no Tracy needed — the shim writes at every matmul):

```bash
rm -f <scratchpad>/fibo_mm_dump.jsonl
FIBO_MM_DUMP=<scratchpad>/fibo_mm_dump.jsonl \
  HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole \
  python_env/bin/python -m pytest \
  models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py::test_fibo_denoise_device_profile \
  -q --timeout=1200
```

Expected: test PASSES; `fibo_mm_dump.jsonl` exists with thousands of lines (warmup + measured forward, all 46 blocks).

- [ ] **Step 3: Collect + dedup into a shape table.**

Create `<scratchpad>/collect_fibo_shapes.py`:

```python
import json, sys
from collections import Counter
seen = Counter()
meta = {}
for line in open(sys.argv[1]):
    d = json.loads(line)
    key = (d["M"], d["K"], d["N"], d["gx"], d["gy"], d["is_agmm"], d["use_case"])
    seen[key] += 1
    meta[key] = d
print(f"{len(seen)} distinct (M,K,N,grid,agmm,use_case); grids seen: "
      f"{sorted({(k[3],k[4]) for k in seen})}")
print("\n# SHAPES entries (M, K, N, cgx, cgy, is_agmm, use_case) — count per forward-pair:")
for key, n in sorted(seen.items(), key=lambda kv: (-kv[0][0]*kv[0][1]*kv[0][2])):
    M,K,N,gx,gy,agmm,uc = key
    print(f"    ({M}, {K}, {N}, {gx}, {gy}, {agmm}, {uc!r}),  # x{n}")
```

Run:

```bash
python_env/bin/python <scratchpad>/collect_fibo_shapes.py <scratchpad>/fibo_mm_dump.jsonl
```

Expected: a single grid (e.g. one `(gx,gy)` pair) for all matmuls, `is_agmm=False` throughout, `use_case` `plain` for all except the `ff1` shape(s) which show `plain_gelu`. ~15–25 distinct shapes. **Record this printed block** — it is the input to Task 2.

- [ ] **Step 4: Commit nothing yet.** Task 1 leaves only the temporary shim (reverted in Task 3) and scratch artifacts. Sanity-check the shape families match expectations (big M≈2048 spatial; M=128 prompt twins; M=32 modulation; caption M=128 K=2048 N=1536). If a grid other than one uniform grid appears, note it — Task 2's SHAPES must use whatever grid the dump reports.

---

### Task 2: Add FIBO shapes to the sweeper and run the block-size sweep

**Files:**
- Modify: `models/tt_dit/utils/sweep_mm_block_sizes.py` — add `bh_2x2` to `DEVICE_CONFIGS`; append FIBO shapes to `SHAPES`.
- Produces: `sweep_results_mm.csv` (repo root, gitignored scratch) with per-combo timings.

**Interfaces:**
- Consumes from Task 1: the `(M,K,N,cgx,cgy,is_agmm,use_case)` tuples.
- Produces for Task 3: for each FIBO shape, the min-`device_kernel_duration_ns` combo `(M_block, K_block, N_block, subblock_h, subblock_w)`.

- [ ] **Step 1: Add a `bh_2x2` device config.**

In `models/tt_dit/utils/sweep_mm_block_sizes.py`, add to `DEVICE_CONFIGS` (matches FIBO's real topology; `cluster_axis` is the tp axis since AGMM gathers along tp — harmless for the all-`mm` FIBO shapes but correct if any AGMM shape appears):

```python
    "bh_2x2": {
        "mesh_shape": (2, 2),
        "fabric_config": "FABRIC_1D",
        "fabric_router_config_payload": None,
        "topology": "Linear",
        "num_links": 1,
        "num_workers_per_link": 13,  # full_grid.x // num_links on Blackhole
        "sp_axis": 0,
        "tp_axis": 1,
        "cluster_axis": 1,
    },
```

- [ ] **Step 2: Append FIBO shapes to `SHAPES`.**

Paste the recorded Task 1 tuples into `SHAPES` under a comment, using the grid the dump reported. Example form (substitute the real dumped values — do NOT invent them):

```python
    # --- FIBO denoise (2x2 BH, sp=2/tp=2) — from Task 1 dump 2026-07-10 ---
    (2048, 3072, 4608, <gx>, <gy>, False, "plain"),      # to_qkv spatial
    (2048, 3072, 6144, <gx>, <gy>, False, "plain_gelu"), # ff1 spatial (fused exact GELU)
    (2048, 6144, 3072, <gx>, <gy>, False, "plain"),      # ff2 spatial (RowParallel)
    (128,  3072, 4608, <gx>, <gy>, False, "plain"),      # to_qkv prompt twin
    (32,   3072, 9216, <gx>, <gy>, False, "plain"),      # norm1_linear modulation
    (128,  2048, 1536, <gx>, <gy>, False, "plain"),      # caption_projection
    # ... every distinct tuple from Task 1 ...
```

- [ ] **Step 3: Run the sweep for the FIBO shapes only.**

The `-k` filter selects the new `bh_2x2` params. Run one shape first to confirm plumbing:

```bash
HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole \
  python_env/bin/python -m pytest \
  models/tt_dit/utils/sweep_mm_block_sizes.py::test_mm_sweep \
  -k "bh_2x2 and 2048_3072_6144" -x -s --timeout=7200
```

Expected: worker prints candidate lists + `combos to measure: N`, then rows append to `sweep_results_mm.csv`. Then run all FIBO shapes:

```bash
HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole \
  python_env/bin/python -m pytest \
  models/tt_dit/utils/sweep_mm_block_sizes.py::test_mm_sweep \
  -k "bh_2x2" -s --timeout=7200
```

Expected: PASS/skip per shape; `sweep_results_mm.csv` has `status=ok` rows for each shape.

- [ ] **Step 4: Extract the winner per shape.**

```bash
python_env/bin/python - <<'PY'
import csv
from collections import defaultdict
best = {}
for d in csv.DictReader(open("sweep_results_mm.csv")):
    if d["device_config"] != "bh_2x2" or d["status"] != "ok":
        continue
    key = (int(d["M"]), int(d["K"]), int(d["N"]), d["core_grid"])
    dur = float(d["device_kernel_duration_ns"])
    if key not in best or dur < best[key][0]:
        best[key] = (dur, int(d["M_block"]), int(d["K_block"]), int(d["N_block"]),
                     int(d["subblock_h"]), int(d["subblock_w"]))
for key, v in sorted(best.items()):
    M,K,N,grid = key
    _, mb,kb,nb,sh,sw = v
    print(f"({M}, {K}, {N}): ({mb}, {kb}, {nb}, ({sh}, {sw})),")
PY
```

Expected: one `(M,K,N): (M_block,K_block,N_block,(sub_h,sub_w))` line per FIBO shape. **Record these** for Task 3.

- [ ] **Step 5: Commit the sweeper additions.**

```bash
git add models/tt_dit/utils/sweep_mm_block_sizes.py
git commit -m "perf(fibo-pipeline): add bh_2x2 config + FIBO denoise shapes to mm block sweeper"
```

---

### Task 3: Register the tuned blocks and gate on PCC

**Files:**
- Modify: `models/tt_dit/models/transformers/transformer_bria_fibo.py` — add `_register_fibo_matmul_configs()` + a module-level call.
- Modify (revert): `models/tt_dit/layers/linear.py` — remove the Task 1 dump shim.

**Interfaces:**
- Consumes from Task 2: the winning `(M,K,N) -> (M_block,K_block,N_block,(sub_h,sub_w))` map and the grid string (`"<gx>x<gy>"`).

- [ ] **Step 1: Add the registration function to `transformer_bria_fibo.py`.**

After the imports (the file already imports from `...utils`), add:

```python
from ...utils.matmul import register_matmul_configs

# FIBO denoise matmul blockings tuned via utils/sweep_mm_block_sizes.py on the
# 2x2 Blackhole mesh (sp=2/tp=2). Keyed by (M, K, N) under the runtime grid;
# additive (register_matmul_configs is keyed by shape) so it cannot affect
# other models. Regenerate with the bh_2x2 sweep if the topology/shapes change.
_FIBO_MM_CONFIGS_REGISTERED = False


def _register_fibo_matmul_configs() -> None:
    global _FIBO_MM_CONFIGS_REGISTERED
    if _FIBO_MM_CONFIGS_REGISTERED:
        return
    register_matmul_configs({
        "<gx>x<gy>": {
            # (M, K, N): (M_block, K_block, N_block, (sub_h, sub_w))  — from Task 2
            (2048, 3072, 6144): (<mb>, <kb>, <nb>, (<sh>, <sw>)),
            # ... every winner from Task 2 ...
        },
    })
    _FIBO_MM_CONFIGS_REGISTERED = True


_register_fibo_matmul_configs()
```

Substitute the real grid string and the Task 2 winners. Only include grid keys that `register_matmul_configs` accepts (`8x8`, `8x9`, `11x10`, `12x10`, `12x9`, `13x9`); if the dumped grid is not among these, add it to the `grid_map` and a matching `grid_XX_configs` dict in `utils/matmul.py` in the same commit (small, additive change) and note it.

- [ ] **Step 2: Revert the Task 1 dump shim from `linear.py`.**

Remove `_dump_mm_shape`, the `import json as _json` / `import os as _os` lines, and the 4 call sites. Confirm clean:

```bash
git diff --stat models/tt_dit/layers/linear.py
```

Expected: no changes to `linear.py` (fully reverted).

- [ ] **Step 3: Run the PCC gate.**

```bash
HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole \
  python_env/bin/python -m pytest \
  models/tt_dit/tests/models/bria_fibo/test_transformer.py::test_fibo_transformer_mesh \
  -q --timeout=1200
```

Expected: PASS (`assert_quality ... pcc=0.99`). If it fails, a registered tuple is malformed (wrong grid key or an invalid block for that shape) — recheck the grid string and that each `(M,K,N)` matches a real dumped shape.

- [ ] **Step 4: Commit the registration.**

```bash
git add models/tt_dit/models/transformers/transformer_bria_fibo.py models/tt_dit/utils/matmul.py
git commit -m "perf(fibo-pipeline): register tuned minimal_matmul blockings for FIBO denoise shapes"
```

---

### Task 4: Re-profile and measure the win

**Files:**
- Produces: refreshed `denoise_report/` + a before/after comparison.

- [ ] **Step 1: Re-run the denoise device profile.**

```bash
TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=6000 \
  HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole \
  python_env/bin/python -m tracy -r -p -v --dump-device-data-mid-run -m pytest \
  models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py::test_fibo_denoise_device_profile \
  --timeout=1800
mkdir -p denoise_report_after
tt-perf-report generated/profiler/reports/<ts>/ops_perf_results_<ts>.csv \
  --start-signpost "fibo denoise" --end-signpost "fibo denoise" \
  --csv denoise_report_after/ops.csv
```

Expected: run PASSES (`assert out is not None` — confirms the registered production-shape configs are valid), and `denoise_report_after/ops.csv` is written.

- [ ] **Step 2: Compare the matmul-util histogram before vs after.**

```bash
python_env/bin/python - <<'PY'
import csv
def hist(path):
    buckets=[(0,15),(15,30),(30,45),(45,60),(60,101)]; agg={b:[0.0,0] for b in buckets}; tot=0.0
    for d in csv.DictReader(open(path)):
        if 'MinimalMatmul' not in d['OP Code']: continue
        t=float((d['Device Time'] or '0').replace(',','')); fp=float((d['FLOPs %'] or '0').replace(',',''))
        tot+=t
        for b in buckets:
            if b[0]<=fp<b[1]: agg[b][0]+=t; agg[b][1]+=1; break
    return tot, agg
for label,p in [("BEFORE","denoise_report/ops.csv"),("AFTER","denoise_report_after/ops.csv")]:
    tot,agg=hist(p); print(f"\n{label}: total MM {tot/1000:.1f} ms")
    for b,(ms,n) in agg.items(): print(f"  {b[0]:>3}-{b[1]-1:<3}%  {ms/1000:6.1f} ms  {100*ms/tot:5.1f}%  ({n})")
PY
```

Expected: total matmul ms drops and mass shifts out of the `<15%` bucket toward higher-util buckets. Record the delta (total MM ms before/after, and the shift in the `<15%` share).

- [ ] **Step 3: Commit results + update memory.**

```bash
rm -rf denoise_report && mv denoise_report_after denoise_report
git add denoise_report
git commit -m "perf(fibo-pipeline): re-profile denoise after matmul-block tuning (before/after in commit msg)"
```

Update the `fibo-device-profiling` memory (and add a `fibo-denoise-opt` memory) with the Phase 1 result and the sweep command, so Phase 2 starts from the new baseline.

- [ ] **Step 4: Report Phase 1 outcome** (total MM-time delta, `<15%`-bucket reduction, PCC pass) and hand back for the Phase 2 target-selection decision from the new `denoise_report`.

---

## Self-Review

- **Spec coverage:** Phase 1 steps (enumerate → sweep → register → gate+measure) map to Tasks 1→4. Correctness gate = `test_fibo_transformer_mesh` (Task 3 Step 3). Metric = re-profile histogram (Task 4). Additive/registration-only constraint honored (Task 3 reverts the shim). ✓
- **Placeholder scan:** The `<gx>`, `<gy>`, `<mb>…`, `<ts>` are *runtime-produced values*, explicitly flagged as "substitute the real dumped/swept values" — they are outputs of Tasks 1–2, not unspecified design. All authored code (dump shim, bh_2x2 config, register function, comparison scripts) is complete. ✓
- **Type consistency:** `_dump_mm_shape(M,K,N,core_grid,*,is_agmm,use_case)` signature matches all 4 call sites; `register_matmul_configs({grid_str: {(M,K,N): (mb,kb,nb,(sh,sw))}})` matches the documented API in `utils/matmul.py`. ✓
