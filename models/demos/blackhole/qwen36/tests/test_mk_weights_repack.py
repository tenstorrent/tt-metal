"""Megakernel weight repack (integ/weights_mk.py) on the real model: extraction from the model's own decode weights +
host quantize + mesh assembly is LOSSLESS. With the tile-order test layout the repacked tensor's per-device pages must
equal the model weights' tiles exactly (BFP4 MLP from w_gate_up/w2, BFP8 qkv/wo from the KV-split pair weights).

  pytest models/demos/blackhole/qwen36/tests/test_mk_weights_repack.py -s -x   (prod env + full-mesh conftest)
"""
import torch

import ttnn
from models.demos.blackhole.qwen36.tests.test_spec_chain_engine import _env, _marks, _setup
from models.demos.blackhole.qwen36.tt.megakernel.integ import weights_mk as wm

LAYERS = (0, 3)  # one GDN-type layer and the first full-attention layer


@_marks
def test_mk_weights_repack_lossless(mesh_device, monkeypatch):
    _env(monkeypatch)
    model, _eng, _tok, _prompts = _setup(mesh_device)
    lays = {}
    for kind in ("mlp", "qkv", "wo", "gdn_in", "gdn_out"):
        src = {"mlp": model.layers[0].feed_forward.weights.w_gate_up, "qkv": model.layers[3].attention.tw["wqkvg_pair"],
               "wo": model.layers[3].attention.tw["wo_pair"], "gdn_in": model.layers[0].attention.tw["qkvz"],
               "gdn_out": model.layers[0].attention.tw["out"]}[kind]
        lays[kind] = wm.TileOrderLayout(kind, str(src.dtype).split(".")[-1].lower())
    lays["gamma"] = None  # fixed layout (gamma_pages)
    rep = wm.repack(model, lays, only_layers=LAYERS)
    g = wm.device_shards(rep.pop("gamma")["tensor"])
    want_g = wm.gamma_pages(model).reshape(-1, wm.H).bfloat16().float()
    assert all(torch.equal(x.reshape(-1, wm.H), want_g) for x in g), "gamma pages differ from the model norm weights"
    l0 = model.layers[0]
    assert torch.equal(want_g[0], (wm.norm_gamma(l0.attention_norm.norm).float() * wm.norm_fix()).bfloat16().float())
    assert not want_g[1:wm.TILE].any(), "gamma rows 1-31 must be zero"
    print(f"repack gamma: {want_g.shape[0] // wm.TILE} (layer, norm) rows x {len(g)} devices exact")
    for kind, r in rep.items():
        got = wm.device_shards(r["tensor"])
        for j, li in enumerate(r["layers"]):
            mats = wm.SOURCES[kind][1](model.layers[li])
            for d, m in enumerate(mats):
                want = lays[kind].pack(m, li).reshape(-1, wm.TILE)
                ppl = r["pages_per_layer"] * wm.TILE
                g = got[d][j * ppl:(j + 1) * ppl]
                assert torch.equal(g, want.float()), f"{kind} layer {li} device {d}: max |diff| {(g - want).abs().max()}"
        print(f"repack {kind}: layers {r['layers']} x {len(got)} devices lossless ({r['dtype']}, {r['pages_per_layer']} pages/layer)")
