"""Dump real GDN in-proj weights (device 0's host block, as integ weights_mk.gdn_in_mats sees it) for track-side real-weight
checks: {layer_idx: float tensor} -> DUMP_GDN_IN_OUT (default /tmp/gdn_in_real.pt). DUMP_GDN_IN_LAYERS="0,1,2,4".
  pytest models/demos/blackhole/qwen36/tests/test_mk_dump_gdn_in.py -s -x   (prod env + full-mesh conftest)"""
import os

import torch

from models.demos.blackhole.qwen36.tests.test_spec_chain_engine import _env, _marks, _setup
from models.demos.blackhole.qwen36.tt.megakernel.integ import weights_mk as wm

LAYERS = [int(x) for x in os.environ.get("DUMP_GDN_IN_LAYERS", "0,1,2,4").split(",")]
OUT = os.environ.get("DUMP_GDN_IN_OUT", "/tmp/gdn_in_real.pt")


@_marks
def test_mk_dump_gdn_in(mesh_device, monkeypatch):
    _env(monkeypatch)
    model, _eng, _tok, _p = _setup(mesh_device)
    out = {}
    for li in LAYERS:
        l = model.layers[li]
        assert not l.is_full_attention, f"layer {li} is not a GDN layer"
        out[li] = wm.device_shards(l.attention.tw["qkvz"])[0].float().reshape(-1, wm.device_shards(l.attention.tw["qkvz"])[0].shape[-1])
        print(f"gdn_in layer {li}: {tuple(out[li].shape)} dtype {l.attention.tw['qkvz'].dtype}")
    torch.save(out, OUT)
    print(f"saved {sorted(out)} -> {OUT}")
