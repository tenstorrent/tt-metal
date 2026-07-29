# Gemma2 — verified run commands (clean env)

Reproducible commands for accuracy (PCC) and decode speed (t/s/u).

## IMPORTANT: always start from a clean env

The streaming matmul path is **opt-in and still buggy**. It is gated on the
`TT_STREAM_MM` environment variable. If that variable is left set to `1` in your
shell (e.g. from earlier streaming experiments), it silently forces the broken
path on EVERY run and produces gibberish output / ~0.07 PCC — even though the
committed model is correct.

Always unset it (and its friends) before measuring:

```bash
unset TT_STREAM_MM TT_STREAM_MM_FF2 TT_STREAM_MM_FF13
```

## Common env setup (run once per shell)

```bash
cd /home/ttuser/Teja/tt-metal
source python_env/bin/activate
unset TT_STREAM_MM TT_STREAM_MM_FF2 TT_STREAM_MM_FF13        # clean env guard
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole
```

---

## gemma2-9B — 1xP150

```bash
export TT_VISIBLE_DEVICES=0 MESH_DEVICE=P150 HF_MODEL=google/gemma-2-9b-it
export TT_MESH_GRAPH_DESC_PATH=$PWD/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
```

Accuracy (PCC, teacher-forced):

```bash
pytest "models/demos/gemma2/tests/test_model.py::test_model_inference[blackhole-device_params0-1-performance-256-1-page_params0-paged_attention-full-False]" -s -q
```

Decode speed (t/s/u):

```bash
pytest "models/tt_transformers/demo/simple_text_demo.py::test_demo_text[blackhole-mesh_device0-device_params0-False-performance-batch-1]" -s -q
```

Verified results (2026-07-28): **PCC 0.976 (Passed)**, **~39.8 t/s/u avg** (steady ~39.5, 25 ms/token).

---

## gemma2-9B — 2xP150 / P300, TP=2

```bash
export TT_VISIBLE_DEVICES=0,1 MESH_DEVICE=P300 HF_MODEL=google/gemma-2-9b-it
export TT_CREATE_HEADS_MD=1
export TT_MESH_GRAPH_DESC_PATH=$PWD/tt_metal/fabric/mesh_graph_descriptors/p300_mesh_graph_descriptor.textproto
```

NOTE: `TT_CREATE_HEADS_MD=1` is required on P300. If device enumeration throws
`IndexError: unordered_map::at` at import (cluster/fabric links not trained),
reset the board first: `tt-smi -r 0 1`.

Accuracy (PCC, teacher-forced, note the `-2-` = num_devices=2):

```bash
pytest "models/demos/gemma2/tests/test_model.py::test_model_inference[blackhole-device_params0-2-performance-256-1-page_params0-paged_attention-full-False]" -s -q
```

Decode speed (t/s/u):

```bash
pytest "models/demos/gemma2/demo/text_demo.py::test_demo_text" -k "performance and batch-1" -s -q
```

Verified results (2026-07-28): **PCC 0.93-0.96 (all 9 iters Passed)**.
Decode: **42.5 t/s/u baseline -> 44.3 t/s/u** with the 2xP150 sweep-tuned FF core
grids (mlp_core_grid / mlp2_core_grid = 7x4 = 28c for num_devices==2), **+4.2%**.
The `quick` PCC variant hits a pre-existing harness `KeyError: 'gemma-2-9b-it'`;
use the `full` variant above.

---

## gemma2-27B — 2xP150 / P300, TP=2 (TODO: rerun clean)

```bash
export TT_VISIBLE_DEVICES=0,1 MESH_DEVICE=P300 HF_MODEL=google/gemma-2-27b-it
export TT_MESH_GRAPH_DESC_PATH=$PWD/tt_metal/fabric/mesh_graph_descriptors/p300_mesh_graph_descriptor.textproto
```

---

## Notes

- `mlp.py`: the untested approximate-SiLU LUT (`fast_and_approximate_mode`) was
  removed from the default decode gate multiply. Correctness confirmed without it.
- Do NOT commit with `TT_STREAM_MM` set in your shell; it only affects runtime,
  but leaving it set will corrupt any perf/accuracy numbers you collect.
