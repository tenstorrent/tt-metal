# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0

"""Host-only ttnn checks (no device): the torch BFP emulation used by moe_pcc.py matches ttnn's round trip, and
cache.py loads what the converter's Writer wrote, under the §4 names, dtypes and layouts.

    PYTHONPATH=<worktree> /work/tt-metal/python_env/bin/python -m pytest --noconftest -c /dev/null \
        models/demos/inkling2_mini/convert/test_cache.py
"""

import json

import pytest
import torch

ttnn = pytest.importorskip("ttnn")

from models.demos.inkling2_mini.convert import cache
from models.demos.inkling2_mini.convert.convert import DTYPES, LAYOUTS, Writer
from models.demos.inkling2_mini.convert.moe_pcc import bfp_roundtrip


@pytest.mark.parametrize("dtype", ["BFP4", "BFP8"])
def test_bfp_emulation_matches_ttnn(dtype):
    g = torch.Generator().manual_seed(0)
    x = (torch.randn(1, 1, 256, 512, generator=g) * torch.rand(1, 1, 256, 1, generator=g) * 3).bfloat16()
    got = ttnn.to_torch(ttnn.from_torch(x, dtype=DTYPES[dtype], layout=ttnn.TILE_LAYOUT)).float()
    exact = (got == bfp_roundtrip(x, dtype)).float().mean().item()
    assert exact > 0.999, exact


def test_load_layer_and_global(tmp_path):
    w = Writer(tmp_path)
    w.write("attn.w_o", 3, torch.randn(3072, 2304), "BFP8", "TILE", ["src"])
    w.write("mlp_norm", 3, torch.randn(72, 32), "BF16", "RM", ["src"])
    w.write("router.w", 3, torch.randn(2304, 128), "FP32", "TILE", ["src"])
    w.write("final_norm", None, torch.randn(72, 32), "BF16", "RM", ["src"])
    (tmp_path / "manifest.json").write_text(json.dumps({"schema": 1, "tensors": w.entries}))

    layer = cache.load_layer(tmp_path, 3, device=None)
    assert set(layer) == {"attn.w_o", "mlp_norm", "router.w"}
    for e in w.entries[:3]:
        t = layer[e["name"]]
        assert t.dtype == DTYPES[e["dtype"]] and t.layout == LAYOUTS[e["layout"]]
        assert list(t.shape) == e["shape"]
    assert set(cache.load_global(tmp_path, device=None)) == {"final_norm"}
    with pytest.raises(KeyError):
        cache.load_layer(tmp_path, 4, device=None)


def test_expert_dtype_map():
    from models.demos.inkling2_mini.convert.convert import expert_tensor_dtype

    m = {"L08/experts.w_down": "BFP8", "L02/experts.w_gate_up": {"default": "BFP8", "BFP8": [1, 2]}}
    assert expert_tensor_dtype(m, 8, "experts.w_down", "BFP4") == "BFP8"
    assert expert_tensor_dtype(m, 8, "experts.w_gate_up", "BFP4") == "BFP4"
    assert expert_tensor_dtype(m, 2, "experts.w_gate_up", "BFP4") == "BFP8"
    with pytest.raises(SystemExit):  # mixed dtypes inside one §4 tensor
        expert_tensor_dtype({"L03/experts.w_down": {"default": "BFP4", "BFP8": [5]}}, 3, "experts.w_down", "BFP4")
