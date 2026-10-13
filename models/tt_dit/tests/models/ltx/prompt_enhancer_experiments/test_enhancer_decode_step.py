# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""One traced decode step of the device prompt enhancer (Gemma-4-E2B) on a 1x8 row, for device profiling.

The enhancer runs TP=8 along a mesh row, and every row of the LTX 4x8 mesh holds the same program, so one row
opened on its own carries the per-chip work of the served configuration while the profiler only has 8 chips to
drain. The ring fabric matches the pipeline's; the collectives run over a single row's ring instead of a row of
the full mesh, so check CCL numbers against the 4x8 bench's ms/token.

Load, eager warmup and decode-trace capture as in the bench, then per temperature:
  1. a reference request decoding ``skip + steps + 2`` tokens past any stop id;
  2. the same request with decode steps ``[skip, skip + steps)`` inside ``start``/``stop`` signposts and, under
     the device profiler, a profiler drain right before ``start`` and right after ``stop``, so the window alone
     has to fit the profiler buffer.
The two token streams must match: the window may not change what the model produces.

    STEP_TEMPS   comma list of temperatures (default "0.7")
    STEP_COUNT   decode steps inside the window (default 1)
    STEP_SKIP    decode steps before the window (default 2; step 0 restages the host inputs)
    LTX_ENHANCER_* / GEMMA4_*  the enhancer knobs, read as in the bench
    LTX_ENHANCER_EXP_DIR  output root (JSON lands under <root>/decode_step/)

Profile with the ops report (one row per op per device; bucket with profile_buckets.py):

    TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=<n> LTX_PIN_CORES=0 python -m tracy -p -r -v --no-web-server \\
        -m pytest -s models/tt_dit/tests/models/ltx/prompt_enhancer_experiments/test_enhancer_decode_step.py
"""

import json
import os

import pytest
from loguru import logger

import ttnn
from models.tt_dit.pipelines.ltx.prompt_enhancer import ENHANCER_SEED, DevicePromptEnhancer, default_ltx_enhancer
from models.tt_dit.tests.models.ltx.prompt_enhancer_experiments.enhancer_bench import BENCH_PROMPTS, _profile_decode
from models.tt_dit.utils.ltx import log_dram
from models.tt_dit.utils.test import ring_params_8k

OUT_DIR = os.path.join(
    os.environ.get("LTX_ENHANCER_EXP_DIR", os.path.expanduser("~/ltx_enhancer_experiments")), "decode_step"
)
# The bench's 4x8 ring device params, without require_exact_physical_num_devices (a row is not the whole system).
ROW_DEVICE_PARAMS = {**ring_params_8k, "trace_region_size": 500_000_000, "l1_small_size": 32768}


@pytest.mark.parametrize(
    "mesh_device, device_params, topology",
    [((1, 8), ROW_DEVICE_PARAMS, ttnn.Topology.Ring)],
    ids=["1x8_ring"],
    indirect=["mesh_device", "device_params"],
)
def test_enhancer_decode_step(mesh_device, device_params, topology):
    temps = [float(t) for t in os.environ.get("STEP_TEMPS", "0.7").split(",") if t.strip()]
    steps = int(os.environ.get("STEP_COUNT", "1"))
    skip = int(os.environ.get("STEP_SKIP", "2"))
    assert steps >= 1 and skip >= 1, "the window needs at least one step, after the restaging step 0"
    os.makedirs(OUT_DIR, exist_ok=True)

    enhancer = DevicePromptEnhancer(
        default_ltx_enhancer(), mesh_device=mesh_device, ccl_topology=topology, enable_trace=True
    )
    knobs = {k: v for k, v in os.environ.items() if k.startswith(("LTX_ENHANCER_", "GEMMA4_", "STEP_"))}
    results = {"mesh": list(mesh_device.shape), "knobs": knobs, "dram": []}
    enhancer.ensure_loaded()
    enhancer.warmup(capture=False)
    enhancer.capture_trace()
    results["dram"] = [d for d in [log_dram(mesh_device, "after capture")] if d]

    # Reference: the same short requests with nothing around the steps.
    prompt = BENCH_PROMPTS[0]
    budget, enhancer.max_new_tokens = enhancer.max_new_tokens, skip + steps + 2
    enhancer.ignore_stop = True
    reference = {}
    for temperature in temps:
        enhancer.temperature = temperature
        enhancer.enhance(prompt, mode="t2v", seed=ENHANCER_SEED)
        reference[temperature] = list(enhancer.last_token_ids)
    enhancer.ignore_stop = False
    enhancer.max_new_tokens = budget

    _profile_decode(enhancer, mesh_device, temps, steps, results, skip=skip, prompt=prompt)
    out_path = os.path.join(OUT_DIR, "decode_step.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    enhancer.unload()

    for window in results["profile_windows"]:
        temperature = window["temperature"]
        logger.info(
            f"decode step T={temperature}: steps {skip}..{skip + steps - 1}, window {window['window_wall_ms']} ms "
            f"wall ({window['window_wall_ms'] / steps:.3f} ms/step), tokens {window['token_ids']}"
        )
        assert window["token_ids"] == reference[temperature], (
            f"T={temperature}: the profiled request diverged from its reference: "
            f"{window['token_ids']} vs {reference[temperature]}"
        )
