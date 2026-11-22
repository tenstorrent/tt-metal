# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0

"""Host-only ttnn checks (no device): the torch BFP emulation used by moe_pcc.py matches ttnn's round trip, and
cache.py loads what the converter's Writer wrote, under the §4 names, dtypes and layouts.

    TT_METAL_MOCK_CLUSTER_DESC_PATH=<p100 cluster desc yaml> pytest --noconftest -c /dev/null \
        models/demos/inkling2_mini/convert/test_cache.py

(the mock cluster keeps ttnn's host conversions off the card; see README.md)
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
    (tmp_path / "manifest.json").write_text(json.dumps({"schema": 1, "complete": True, "tensors": w.entries}))

    layer = cache.load_layer(tmp_path, 3, device=None)
    assert set(layer) == {"attn.w_o", "mlp_norm", "router.w"}
    for e in w.entries[:3]:
        t = layer[e["name"]]
        assert t.dtype == DTYPES[e["dtype"]] and t.layout == LAYOUTS[e["layout"]]
        assert list(t.shape) == e["shape"]
    assert set(cache.load_global(tmp_path, device=None)) == {"final_norm"}
    with pytest.raises(KeyError):
        cache.load_layer(tmp_path, 4, device=None)


def test_incomplete_manifest_refused(tmp_path):
    w = Writer(tmp_path)
    w.write("final_norm", None, torch.randn(72, 32), "BF16", "RM", ["src"])
    (tmp_path / "manifest.json").write_text(json.dumps({"schema": 1, "complete": False, "tensors": w.entries}))
    with pytest.raises(ValueError, match="not complete"):
        cache.load_global(tmp_path)


def test_dtype_map_keys():
    from models.demos.inkling2_mini.convert.convert import check_dtype_map

    check_dtype_map({"L00/experts.w_down": "BFP8", "L19/experts.w_gate_up": "BFP8"})
    for bad in ("L21/experts.w_down", "L99/experts.w_down", "L3/experts.w_down", "L05/attn.w_o", "x"):
        with pytest.raises(SystemExit):
            check_dtype_map({bad: "BFP8"})


def test_unit_key():
    """A unit's key changes with the settings that shape it, and only with those of its own layer."""
    from models.demos.inkling2_mini.convert.convert import unit_key

    base = unit_key("L08", "BFP4", {}, (), None)
    assert base == unit_key("L08", "BFP4", {"L09/experts.w_down": "BFP8"}, (), None)
    changed = [
        unit_key("L08", "BFP8", {}, (), None),
        unit_key("L08", "BFP4", {"L08/experts.w_down": "BFP8"}, (), None),
        unit_key("L08", "BFP4", {}, ("BFP4",), None),
        unit_key("L08", "BFP4", {}, (), "/some/bf16-store"),
    ]
    assert base not in changed and len(set(changed)) == len(changed)
    assert unit_key("global", None, {"L08/experts.w_down": "BFP8"}, ("BFP4",), None) == unit_key(
        "global", None, {}, (), None
    )


def test_expert_dtype_map():
    from models.demos.inkling2_mini.convert.convert import expert_tensor_dtype

    m = {"L08/experts.w_down": "BFP8", "L02/experts.w_gate_up": {"default": "BFP8", "BFP8": [1, 2]}}
    assert expert_tensor_dtype(m, 8, "experts.w_down", "BFP4") == "BFP8"
    assert expert_tensor_dtype(m, 8, "experts.w_gate_up", "BFP4") == "BFP4"
    assert expert_tensor_dtype(m, 2, "experts.w_gate_up", "BFP4") == "BFP8"
    with pytest.raises(SystemExit):  # mixed dtypes inside one §4 tensor
        expert_tensor_dtype({"L03/experts.w_down": {"default": "BFP4", "BFP8": [5]}}, 3, "experts.w_down", "BFP4")


def _e12_sticks(w, nb):
    """The stick page order as written by the fused MoE kernel's device-side builder (torch, [E, K, N])."""
    E, K, N = w.shape
    kt, nt = K // 32, N // 32
    s = w.reshape(E, kt, 32, nt, 32).permute(0, 3, 1, 2, 4).reshape(E * nt, kt, 32, 32)
    s = torch.cat([s, s.new_zeros((-s.shape[0]) % nb, kt, 32, 32)])
    return s.reshape(-1, nb, kt, 32, 32).permute(0, 2, 1, 3, 4).reshape(1, 1, -1, 32)


@pytest.mark.parametrize("dtype", ["BFP4", "BFP8"])
def test_stick_round_trip(tmp_path, dtype):
    """Stick file: same values in the kernel's page order, un-sticks to the exact source bytes, loader reports it."""
    from models.demos.inkling2_mini.convert import stick
    from models.demos.inkling2_mini.convert.convert import stick_layer

    g = torch.Generator().manual_seed(1)
    w = Writer(tmp_path)
    w.write("experts.w_gate_up", 0, torch.randn(1, 2, 2304, 2304, generator=g), dtype, "TILE", ["src"])
    w.write("experts.w_down", 0, torch.randn(1, 2, 1152, 2304, generator=g), dtype, "TILE", ["src"])
    tiles = {e["name"]: (tmp_path / e["file"]).read_bytes() for e in w.entries}
    vals = {e["name"]: ttnn.to_torch(ttnn.load_tensor(tmp_path / e["file"])) for e in w.entries}
    assert not stick_layer(w, 0, ("BFP4", "BFP8")[dtype == "BFP4"])  # other dtype: left alone
    assert stick_layer(w, 0, (dtype,))
    (tmp_path / "manifest.json").write_text(json.dumps({"schema": 1, "complete": True, "tensors": w.entries}))

    assert cache.layouts(tmp_path, 0) == {"experts.w_gate_up": "STICK", "experts.w_down": "STICK"}
    layer = cache.load_layer(tmp_path, 0)
    for e in w.entries:
        t = layer[e["name"]]
        assert t.dtype == DTYPES[dtype] and t.layout == ttnn.TILE_LAYOUT and list(t.shape) == e["shape"]
        assert torch.equal(ttnn.to_torch(t), _e12_sticks(vals[e["name"]][0], e["stick_banks"]))
        back = stick.unstick(tmp_path / e["file"], e)
        hdr = len(tiles[e["name"]]) - back.size
        assert back.tobytes() == tiles[e["name"]][hdr:]


def test_stick_pages_small():
    from models.demos.inkling2_mini.convert.stick import stick_pages

    w = torch.arange(3 * 2 * 3, dtype=torch.float32).reshape(3, 2, 3)  # tile ids in (e, k, n) order, kt=2, nt=3
    ref = w.permute(0, 2, 1).reshape(9, 2)  # sticks q = e*3 + n
    ref = torch.cat([ref, torch.full((5, 2), -1.0)]).reshape(2, 7, 2).permute(0, 2, 1).reshape(-1)
    assert stick_pages(3, 2, 3, 7).tolist() == ref.long().tolist()


@pytest.mark.parametrize("dtype", ["BFP4", "BFP8"])
def test_stick_header_matches_ttnn(tmp_path, dtype):
    """The stored file header equals a fresh ttnn dump of a [1, 1, rows, 32] TILE tensor."""
    from models.demos.inkling2_mini.convert import stick

    rows = 32 * 41
    ttnn.dump_tensor(
        tmp_path / "h.tensorbin",
        ttnn.from_torch(torch.zeros(1, 1, rows, 32), dtype=DTYPES[dtype], layout=ttnn.TILE_LAYOUT),
    )
    raw = (tmp_path / "h.tensorbin").read_bytes()
    assert stick.header(rows, dtype) == raw[: 8 + int.from_bytes(raw[:8], "little")]
