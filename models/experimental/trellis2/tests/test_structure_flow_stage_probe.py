# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Diagnostic, not a gate: saves the structure DiT's tensors around the output head (h before the final norm, the
normed h, the output) for a few golden calls, to locate the ~0.4% uniform output shrink seen on every call.
Runs only with TRELLIS2_STAGE_PROBE=1; writes TRELLIS2_RESULTS_DIR/stage_probe.pt."""

from __future__ import annotations

import os

import pytest
import torch

import ttnn
from models.experimental.trellis2.reference import structure_flow_host as host
from models.experimental.trellis2.tests import goldens
from models.experimental.trellis2.tests.test_structure_flow_pcc import _model

CALLS = [int(c) for c in os.environ.get("TRELLIS2_PROBE_CALLS", "0,20").split(",")]

pytestmark = pytest.mark.skipif(
    goldens.missing_reason() is not None or os.environ.get("TRELLIS2_STAGE_PROBE") != "1",
    reason="set TRELLIS2_STAGE_PROBE=1, TRELLIS2_GOLDEN_DIR and TRELLIS2_WEIGHTS",
)


def _host(t: ttnn.Tensor) -> torch.Tensor:
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_structure_flow_stage_probe(mesh_device):
    model = _model(mesh_device)
    captured = {}
    final_norm, out_layer = model.final_norm.forward, model.out_layer.forward

    def norm_hook(h):
        captured["h_pre"] = _host(h)
        y = final_norm(h)
        captured["normed"] = _host(y)
        return y

    def out_hook(h, *args, **kwargs):
        y = out_layer(h, *args, **kwargs)
        captured["out_tokens"] = _host(y)
        return y

    model.final_norm.forward, model.out_layer.forward = norm_hook, out_hook
    # TRELLIS2_PROBE_BLOCKS=1: also the input layer output and every block output (first call only, ~750 MB)
    if os.environ.get("TRELLIS2_PROBE_BLOCKS") == "1":
        input_layer = model.input_layer.forward

        def input_hook(*args, **kwargs):
            y = input_layer(*args, **kwargs)
            if "blocks" not in captured:
                captured["h_in"] = _host(y)
                captured["blocks"] = []
            return y

        def block_hook(block_forward):
            def hook(*args, **kwargs):
                y = block_forward(*args, **kwargs)
                if len(captured["blocks"]) < host.NUM_BLOCKS:
                    captured["blocks"].append(_host(y))
                return y

            return hook

        model.input_layer.forward = input_hook
        for block in model.blocks:
            block.forward = block_hook(block.forward)
    # TRELLIS2_PROBE_SUBLAYERS=28,29: input and output of self_attn, cross_attn and mlp of those blocks (first call)
    for b in [int(v) for v in os.environ.get("TRELLIS2_PROBE_SUBLAYERS", "").split(",") if v]:
        for name in ("self_attn", "cross_attn", "mlp"):
            sub = getattr(model.blocks[b], name)

            def sub_hook(*args, _fwd=sub.forward, _key=f"b{b}.{name}", **kwargs):
                y = _fwd(*args, **kwargs)
                if _key not in captured:
                    captured[_key] = {"in": _host(args[0]), "out": _host(y)}
                return y

            sub.forward = sub_hook
    probe = {}
    for i in CALLS:
        x, t, cond, gold, branch = goldens.call_inputs(i)
        out = model(x, t, cond)
        probe[i] = {**captured, "out": out, "gold": gold, "branch": branch}
        captured.pop("blocks", None), captured.pop("h_in", None)
    os.makedirs(os.environ["TRELLIS2_RESULTS_DIR"], exist_ok=True)
    torch.save(probe, f"{os.environ['TRELLIS2_RESULTS_DIR']}/stage_probe.pt")
