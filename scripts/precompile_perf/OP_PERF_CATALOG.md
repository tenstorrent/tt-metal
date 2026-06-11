# Per-op precompile perf — catalog & result template

The representative op set for measuring up-front precompile, the exact test targets in both the
**main** and **nightly** trees, the trace cases that stay excluded, the cheap-compute collect
status per op, and the tables to record results in. The procedure these numbers come from is in
`METHODOLOGY.md`; the manifest and runner are `perf_set.sh` + `run_op_perf.sh`.

## How to run

Inside tmux (the sweep is long and must survive disconnects):

```bash
tmux new -s perf

# everything, both trees:
scripts/precompile_perf/run_op_perf.sh both 2>&1 | tee /tmp/perf_run.log

# one tree, or a subset of ops:
scripts/precompile_perf/run_op_perf.sh main conv2d matmul
scripts/precompile_perf/run_op_perf.sh nightly

# prime the one-time cluster-descriptor capture first (so it isn't charged to op #1):
scripts/run_safe_pytest.sh --precompile tests/ttnn/unit_tests/operations/eltwise/test_add.py -k test_add_2D_tensors
```

Results land in `/tmp/perf_<stamp>/results.tsv` (plus a `*_precompile.log` and `*_cold.log` per
op/suite). Paste the TSV into the tables below.

## The op set

Tree bases: main = `tests/ttnn/unit_tests/operations/`, nightly =
`tests/ttnn/nightly/unit_tests/operations/`.

| # | Op | Main target | Nightly target | Notes |
|---|----|-------------|----------------|-------|
| 1 | conv2d | `conv/test_conv2d.py` | `conv/test_conv2d.py` | — |
| 2 | conv3d | `conv/test_conv3d.py` | `conv/test_conv3d.py` | `test_conv3d_sweep_shapes` (~1536 cases) pruned by default; `PERF_INCLUDE_CONV3D_SWEEP=1` to include |
| 3 | matmul | `matmul/test_matmul.py` | `matmul/test_matmul.py`, `test_matmul2.py` | non-trace files only; trace matmul files excluded by selection |
| 4 | reductions | `reduce/` (whole dir) | `reduction/` (whole dir) | sum / mean / max / var / std / prod / topk |
| 5 | layernorm | `fused/test_layer_norm.py` | `fused/test_layernorm.py` | — |
| 6 | rms_norm | `fused/test_rms_norm.py` | `fused/test_rmsnorm.py` | — |
| 7 | groupnorm | `fused/test_group_norm.py` | `fused/test_group_norm.py` | — |
| 8 | eltwise | `eltwise/` (whole dir) | `eltwise/` (whole dir) | add / mul / activations / … |
| 9 | tilize | `data_movement/test_tilize.py` | — *(no nightly counterpart)* | trace case deselected (below) |
| 10 | tilize_with_val_padding | `data_movement/test_tilize_with_val_padding.py` | — *(no nightly counterpart)* | — |

Coverage gap: **tilize and tilize_with_val_padding have no standalone nightly test** (nightly
only has tilize folded into matmul fusion tests), so ops 9–10 are main-tree only.

## Trace cases — kept excluded (precompile can't warm a traced command stream)

| Tree | Test | How it's excluded |
|------|------|-------------------|
| main | `data_movement/test_tilize.py::test_deepseek_v3_mla_tilize_trace_mode` | `-k "not deepseek_v3_mla_tilize_trace_mode"` (manifest) |
| main | `matmul/test_experimental.py::test_ttnn_linear` (graph tracer) | not in our matmul target (`test_matmul.py` only) |
| nightly | `matmul/test_rs_matmul_1d_gather_in0.py` (metal trace) | not in our matmul target |
| nightly | `sdpa/test_sdpa_chunked.py`, `sdpa/test_sdpa_prefill.py` | not in our set |
| nightly | `experimental/test_moe_compute_single_card.py` | not in our set |

For the whole-directory selections (reductions, eltwise) `run_op_perf.sh` does a pre-flight grep
for trace markers (`begin_trace_capture|execute_trace|trace_region_size|tracer.trace`) and warns
if any selected file carries one without a covering `-k` filter. None were flagged at authoring
time; re-check the warning on each run.

## Cheap-compute collect routing ("fused-off") status

Stand-ins already present in `tests/plugins/up_front_collect.py` make collect cheap for the heavy
ops. Add one for any op whose **collect** phase turns out to dominate (see METHODOLOGY → the lever
is shape-correct, value-irrelevant, so it never changes the programs collected).

| Op | Cheap-compute stand-in today | Action |
|----|------------------------------|--------|
| conv2d / conv3d | `_fast_conv2d` / `_fast_conv3d` | covered |
| matmul | `_fast_matmul` / `_fast_bmm` / tensor `__matmul__` | covered |
| layernorm / groupnorm | `_fast_norm` (patches `F.layer_norm`, `F.group_norm`) | covered |
| rms_norm | none (reference is a torch expression, usually cheap) | add a stand-in only if collect dominates |
| reductions / eltwise | none (torch reductions/eltwise are already cheap) | none expected |
| tilize / tilize_with_val_padding | `from_torch`/`torch2tt_tensor` already skip host tilize | revisit if collect dominates |

## Result tables — fill from `results.tsv`

Columns: **programs** = distinct programs compiled in the warm-up; **cold** = Arm A e2e (s);
**precompile** = Arm B e2e (s); **collect / compile / warm-run** = Arm B phase split (s);
**speedup** = cold / precompile. Report medians of 2–3 runs for headline rows.

### Main suite (`tests/ttnn/unit_tests/operations/`)

| Op | programs | cold (s) | precompile (s) | collect (s) | compile (s) | warm-run (s) | speedup | notes |
|----|---------:|---------:|---------------:|------------:|------------:|-------------:|--------:|-------|
| conv2d | | | | | | | | |
| conv3d | | | | | | | | |
| matmul | | | | | | | | |
| reductions | | | | | | | | |
| layernorm | | | | | | | | |
| rms_norm | | | | | | | | |
| groupnorm | | | | | | | | |
| eltwise | | | | | | | | |
| tilize | | | | | | | | |
| tilize_with_val_padding | | | | | | | | |

### Nightly suite (`tests/ttnn/nightly/unit_tests/operations/`)

| Op | programs | cold (s) | precompile (s) | collect (s) | compile (s) | warm-run (s) | speedup | notes |
|----|---------:|---------:|---------------:|------------:|------------:|-------------:|--------:|-------|
| conv2d | | | | | | | | |
| conv3d | | | | | | | | |
| matmul | | | | | | | | |
| reductions | | | | | | | | |
| layernorm | | | | | | | | |
| rms_norm | | | | | | | | |
| groupnorm | | | | | | | | |
| eltwise | | | | | | | | |
| tilize | — | — | — | — | — | — | — | no nightly target |
| tilize_with_val_padding | — | — | — | — | — | — | — | no nightly target |

### Reviewer roll-up

- **Parallelism (compile phase):** programs built and `compile` wall at `nproc` workers, per op.
- **Barrier cost (collect + compile):** the up-front wait before any test executes — the cost a
  streaming/overlap design would hide. Compare to where it lands inside the cold run.
- **Where precompile loses:** ops where `precompile (s)` ≥ `cold (s)` — expected for tiny program
  counts where up-front overhead isn't amortized; call these out explicitly.
