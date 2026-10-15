# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tracy op capture of GPT-OSS-20B batch-1 prefill on a 1x4 mesh, reduced depth (default 2 decoder layers).

Same model build and entry point as prefill_ttft.py (demo build, paged KV sized for 128K, one user,
Generator.prefill_forward_text), but with num_layers=PREFILL_TRACY_LAYERS so the device profiler buffer holds a
whole forward. For each length L: one warm-up/compile prefill, drain the device profiler, then exactly ONE profiled
prefill between tracy signposts "L<L>", "start" ... "stop", then drain again. Dense expert path only.

    python -m tracy -p -r -v --no-web-server --op-support-count 12000 -m \
      pytest models/demos/gpt_oss/tests/perf/prefill_tracy.py -k 1x4

Env:
    PREFILL_LENGTHS       'all' (default) | 'fast' | comma list, e.g. '128,1k,8k'
    PREFILL_TRACY_LAYERS  decoder layers to build (default 2: one sliding + one full-attention layer)
    PREFILL_TRACY_OUT     JSON with the wall clock of each warm-up and profiled call (optional)

The weight cache marker records n_layers=24, so a reduced build would fail the completeness check and cold-load
the HF weights; this test passes state_dict={} instead (every tensor of layers 0..N-1 exists in the warm cache).
"""

import json
import os
import time
from pathlib import Path

import pytest
from loguru import logger
from tracy import signpost

import ttnn
from models.demos.gpt_oss.tests.perf.prefill_common import (
    PREFILL_LENGTHS,
    build_prefill_generator,
    git_info,
    parse_lengths,
    random_prompt_tokens,
    run_prefill,
)
from models.demos.gpt_oss.tests.test_factory import parametrize_mesh_with_fabric

MAX_SEQ_LEN = max(PREFILL_LENGTHS)


@pytest.mark.timeout(6 * 3600)
@parametrize_mesh_with_fabric([(1, 4)])
def test_prefill_tracy(mesh_device, device_params, monkeypatch):
    # The routed path is the default on Blackhole 1xN, so the dense baseline needs the flag set explicitly.
    assert os.getenv("GPT_OSS_ROUTED_PREFILL") == "0", "dense baseline capture: set GPT_OSS_ROUTED_PREFILL=0"
    num_layers = int(os.getenv("PREFILL_TRACY_LAYERS", "2"))
    lengths = parse_lengths(os.getenv("PREFILL_LENGTHS", "all"))

    import models.demos.gpt_oss.demo.text_demo as text_demo

    orig_create = text_demo.create_tt_model

    def create_reduced(*args, **kwargs):
        kwargs["num_layers"] = num_layers
        if kwargs.get("state_dict") is None:
            kwargs["state_dict"] = {}  # load from the warm 24-layer cache, skip the HF host load
        return orig_create(*args, **kwargs)

    monkeypatch.setattr(text_demo, "create_tt_model", create_reduced)

    record = {
        "harness": "models/demos/gpt_oss/tests/perf/prefill_tracy.py",
        "num_layers": num_layers,
        "git": git_info(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": {},
    }
    out = os.getenv("PREFILL_TRACY_OUT")

    t0 = time.perf_counter()
    ctx = build_prefill_generator(mesh_device, MAX_SEQ_LEN)
    assert len(ctx["model"][0].layers) == num_layers, "reduced build did not take effect"
    logger.info(f"[tracy] {num_layers}-layer model built in {time.perf_counter() - t0:.1f}s")

    for L in lengths:
        tokens = random_prompt_tokens(L)
        warm_s, _ = run_prefill(ctx, tokens)
        ttnn.synchronize_device(mesh_device)
        ttnn.ReadDeviceProfiler(mesh_device)  # drain the warm-up so the profiled forward has the whole buffer

        signpost(f"L{L}")
        signpost("start")
        prof_s, logits = run_prefill(ctx, tokens)  # logits to host: device work is complete on return
        signpost("stop")
        ttnn.synchronize_device(mesh_device)
        ttnn.ReadDeviceProfiler(mesh_device)

        record["results"][str(L)] = {"warmup_ms": warm_s * 1000.0, "profiled_ms": prof_s * 1000.0}
        logger.info(f"[tracy] L={L}: warm-up {warm_s * 1000.0:.1f} ms, profiled {prof_s * 1000.0:.1f} ms")
        if out:
            record["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_text(json.dumps(record, indent=2))
