# LTX prompt-enhancer experiments

Device experiments behind the Gemma-4-E2B prompt enhancer (`models/tt_dit/pipelines/ltx/prompt_enhancer.py`).
Pytest drivers without the `test_` prefix: not collected by directory sweeps, run them by explicit path.
Every driver writes its videos, logs and JSON under `$LTX_ENHANCER_EXP_DIR` (default `~/ltx_enhancer_experiments`),
never into the tree. Measured results from the Blackhole Galaxy `bh-glx-120` are in `results/`.
The full hand-off is `models/tt_dit/pipelines/ltx/PROMPT_ENHANCER_HANDOFF.md`.

## Common environment (Blackhole Galaxy 4x8, one mpirun, no MCP)

```bash
cd $TT_METAL_HOME && source python_env/bin/activate
export PYTHONPATH=$TT_METAL_HOME
export HF_HUB_OFFLINE=1 HF_HOME=/mnt/models/huggingface TT_DIT_CACHE_DIR=$HOME/.cache/tt-dit
export TT_MESH_ID=0 TT_MESH_HOST_RANK=0 TT_METAL_SHM_TRACKING_DISABLED=1
export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/models/tt_dit/tests/models/ltx/scaleout_configs/single_galaxy/mesh_graph_descriptor.textproto
export NUM_FRAMES=153 FPS=25 SEED=10
export LTX_TRACED=0            # standing rule for experiments; the serving path is LTX_TRACED=1
export GEMMA4_HOST_SAMPLE=1    # enhancer samples on host
run() { mpirun -np 1 --bind-to none --tag-output --wdir $TT_METAL_HOME python3 -m pytest -s --timeout 2400 "$@"; }
```

The E2B weight cache (11 GB, converted for the 4x8 mesh) is resolved by the enhancer itself:
`LTX_ENHANCER_CACHE_DIR` → `/mnt/models/huggingface/tt_cache/gemma-4-E2B-it` if writable → `~/.cache/tt-gemma4-e2b`.
On a machine without a converted cache the first load takes ~50 s longer and writes it.

## Drivers

| Driver | Purpose | Command |
|---|---|---|
| `submesh_overlap.py` | Can a sibling submesh overlap the (4,8) DiT submesh safely? (No: per-handle allocators alias DRAM.) | `run <dir>/submesh_overlap.py`; add `SUBMESH_EXP_GUARD_FIRST=1 SUBMESH_EXP_TAG=_guardfirst` for the guard-reservation variant |
| `e2b_bringup.py` | Gemma-4-E2B-it alone on the full (4,8) handle: layout, load/prefill/decode timings, 64 greedy tokens | `run <dir>/e2b_bringup.py` |
| `e2b_hf_reference.py` | CPU HF greedy reference for the same templated prompt (no device) | `OMP_NUM_THREADS=32 python <dir>/e2b_hf_reference.py` |
| `e2b_parity_probe.py` | Teacher-forced device-vs-HF agreement over the HF tokens (needs `hf_reference.json`) | `run <dir>/e2b_parity_probe.py` |
| `ab_prompt_enhancer.py` | A/B inside the real LTX distilled pipeline: raw prompt vs enhanced, host or device backend | `AB_ENHANCER=device AB_PROMPT=beekeeper AB_MAX_NEW_TOKENS=300 run <dir>/ab_prompt_enhancer.py` |
| `tracker_report_only.py` | pytest plugin: trace allocation tracker reports instead of raising, flushes JSON per flag | `TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_TRACEBACKS=1 TRACKER_REPORT_PATH=... run -p models.tt_dit.tests.models.ltx.prompt_enhancer_experiments.tracker_report_only <any device test>` |
| `classify_tracker_report.py` | Attribute every flagged buffer to LTX, the enhancer, or other, from the plugin's JSON | `python <dir>/classify_tracker_report.py <TRACKER_REPORT_PATH>` |

`<dir>` = `models/tt_dit/tests/models/ltx/prompt_enhancer_experiments`. Driver-specific env knobs are documented in each file's docstring.

## Serving-configuration check (open at hand-off)

```bash
LTX_TRACED=1 TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_TRACEBACKS=1 \
LTX_PROMPT_ENHANCER=device PROMPT="beekeeper" PROMPT_STEADY_STATE="a cat on a windowsill watching the rain" \
NO_PROMPT=1 RUN_VBENCH=0 RUN_CLIP=0 TRACKER_REPORT_PATH=$HOME/ltx_enhancer_experiments/tracker_report.json \
run -p models.tt_dit.tests.models.ltx.prompt_enhancer_experiments.tracker_report_only \
    models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py::test_pipeline_distilled -k 4x8sp1tp0nl2_ring_is_fsdp0
python models/tt_dit/tests/models/ltx/prompt_enhancer_experiments/classify_tracker_report.py $HOME/ltx_enhancer_experiments/tracker_report.json
```

Pass criteria: all three generations complete, gen 2 (pure replay) has healthy latents (whiteness 0.25–0.71, zeros < 1%) and a valid mp4, and every flagged buffer is either an LTX per-step temporary or justified.
