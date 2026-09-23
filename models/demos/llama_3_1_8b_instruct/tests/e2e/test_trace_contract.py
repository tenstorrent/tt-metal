# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Command 3 — the per-stage trace contract and the fully-on-device check.

`PIPELINE_STAGES` is derived from the HF reference config: `architectures =
["LlamaForCausalLM"]`, not an encoder-decoder, no sub-configs, text output ->
["prefill", "decode"].

  * trace_capture_selftest — for EACH stage, capture ONE step inside
    begin/end_trace_capture, execute it, PCC it against the eager result, then
    RELEASE the trace before the next stage (stage traces never co-reside).
  * host_op_selftest — run the model math under host_op_observer with tokenization
    and the one-time weight build OUTSIDE the observed region. A truly on-device
    forward fires ZERO host aten ops.

Run on device:
  ./python_env/bin/python -m pytest models/demos/llama_3_1_8b_instruct/tests/e2e/test_trace_contract.py -s

`TT_TRACE_LAYERS` caps the decoder depth (the op mix is identical at any depth;
2 layers surface the same op set as 32 at a fraction of the cost).
"""
from __future__ import annotations

import os

import pytest

import ttnn
from models.demos.llama_3_1_8b_instruct.tt import pipeline as pl

LAYERS = int(os.environ.get("TT_TRACE_LAYERS", "2"))
CAPACITY = int(os.environ.get("TT_TRACE_CAPACITY", pl.TRACE_PREFILL_CAPACITY))


@pytest.fixture(scope="module")
def _built(request):
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    device = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 4),
        l1_small_size=24576,
        trace_region_size=pl.TRACE_REGION_SIZE,
    )
    pipe = pl.build_pipeline(device, layers=LAYERS)
    yield device, pipe
    ttnn.close_mesh_device(device)


def test_stage_contract_surface(_built):
    """Every stage exposes the full generic contract the perf engine binds."""
    _, pipe = _built
    assert pl.PIPELINE_STAGES == ["prefill", "decode"]
    for stage in pl.PIPELINE_STAGES:
        for suffix in ("trace_setup", "trace_step", "trace_inputs", "trace_items"):
            assert callable(getattr(pipe, f"{stage}_{suffix}")), f"missing {stage}_{suffix}"
        assert getattr(pipe, f"{stage}_trace_inputs")() is not None  # ZERO-ARG
        assert int(getattr(pipe, f"{stage}_trace_items")()) >= 1
    # AR decode contract
    assert callable(pipe.decode_prefill) and callable(pipe.decode_step)
    # every stack is discoverable as a plain list of same-typed elements
    assert isinstance(pipe.layers, list) and len({type(x) for x in pipe.layers}) == 1
    assert pipe.hf is not None, "the HF reference must stay reachable (ground truth for stack structure)"
    print(f"[contract] stages={pl.PIPELINE_STAGES} items={[getattr(pipe, f'{s}_trace_items')() for s in pl.PIPELINE_STAGES]}")


def test_trace_capture_selftest(_built):
    device, pipe = _built
    ok = pl.trace_capture_selftest(device, pipe=pipe, capacity=CAPACITY)
    assert ok, "trace capture selftest failed (see the per-stage PCC lines above)"


def test_host_op_selftest(_built):
    _, pipe = _built
    verdict = pl.host_op_selftest(pipe=pipe)
    print(f"[host-ops] {verdict['reason']}")
    assert verdict["on_device"], f"host compute in the forward: {verdict['host_ops']}"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-svv"]))
