# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device check that GPTQ-BFP4 weights survive the REAL TP load path bit-exactly (MLP + GDN, all device copies).

GPTQ output values sit on the bfloat4_b grid of the device layout, and that grid is a subset of the bfloat8_b grid
(same shared exponent, mantissa << 4). So loading the same grid weights with the BFP4 flags ON must read back
identically to loading them with the flags OFF (BFP8). A mismatch means the offline grouping (utils/gptq_bfp4.py)
disagrees with how gdn/tp.py / mlp.py lay the weight out on device for this TP degree. A negative control (plain
random weights) must differ, so the check is not vacuous. Random weights; only config.json of HF_MODEL is needed.

Run (Blackhole Galaxy, 1x8 submesh):
    MESH_DEVICE=P150x8 HF_MODEL=<Qwen3.8-27B dir> pytest models/demos/blackhole/qwen36/tests/test_gptq_bfp4_tp.py -v
"""
import os

import torch

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import model_path, parametrize_mesh_tp, random_gdn_state_dict
from models.demos.blackhole.qwen36.tt.gdn.tp import load_gdn_weights_tp
from models.demos.blackhole.qwen36.tt.mlp import load_mlp_weights
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs
from models.demos.blackhole.qwen36.utils import gptq_bfp4 as G

BFP4_FLAGS = ("QWEN36_BFP4_GDN_IN", "QWEN36_BFP4_GDN_OUT", "QWEN36_BFP4_MLP_DOWN")


def _readback(mesh, t, dim):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=dim)).to(torch.bfloat16)


def _load_all(mesh, args, gdn_sd, mlp_sd, monkeypatch, bfp4):
    monkeypatch.delenv("QWEN36_GPTQ_BFP4", raising=False)
    for f in BFP4_FLAGS:
        monkeypatch.setenv(f, "1" if bfp4 else "0")
    monkeypatch.setenv("QWEN36_MLP_GATEUP_BFP8", "0" if bfp4 else "1")
    tw = load_gdn_weights_tp(mesh, gdn_sd, args)
    mw = load_mlp_weights(mesh, mlp_sd, None, args=args)
    want = {"qkvz": (tw, -1), "ab": (tw, -1), "out": (tw, 0), "out_colpar": (tw, -1)}
    out = {k: _readback(mesh, src[k], d) for k, (src, d) in want.items() if k in src}
    out["w1"], out["w3"], out["w2"] = _readback(mesh, mw.w1, -1), _readback(mesh, mw.w3, -1), _readback(mesh, mw.w2, 0)
    if mw.w_gate_up is not None:
        out["w_gate_up"] = _readback(mesh, mw.w_gate_up, -1)
    dtypes = {k: v.dtype for k, v in tw.items() if k in want}
    dtypes.update(w1=mw.w1.dtype, w2=mw.w2.dtype, w3=mw.w3.dtype)
    return out, dtypes


@parametrize_mesh_tp()
def test_gptq_grid_weights_lossless_through_tp_load(mesh_device, monkeypatch, reset_seeds, ensure_gc):
    os.environ.setdefault("HF_MODEL", model_path())
    args = Qwen36ModelArgs(mesh_device, max_batch_size=1, max_seq_len=256)
    tp = mesh_device.get_num_devices()
    li = next(i for i, t in enumerate(args.attention_type_list) if t == "linear_attention")
    raw = random_gdn_state_dict(args, seed=li)
    g = torch.Generator().manual_seed(li)
    raw_mlp = {
        "gate_proj.weight": (torch.randn(args.hidden_dim, args.dim, generator=g) * 0.02).to(torch.bfloat16),
        "up_proj.weight": (torch.randn(args.hidden_dim, args.dim, generator=g) * 0.02).to(torch.bfloat16),
        "down_proj.weight": (torch.randn(args.dim, args.hidden_dim, generator=g) * 0.02).to(torch.bfloat16),
    }

    # Snap every target onto its device-layout BFP4 grid exactly as scripts/gptq_bfp4.py groups it (RTN = GPTQ
    # without error feedback; the grid, not the rounding choice, is what this test checks).
    P = "linear_attn."
    names = [P + n + ".weight" for n in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b")]
    stack = torch.cat([raw[n] for n in names])
    groups = G.gdn_in_groups(raw[names[0]].shape[0], raw[names[1]].shape[0], args.gdn_nv, tp)
    snapped = G.ttnn_bfp_roundtrip(stack, groups).to(torch.bfloat16)
    gdn = dict(raw)
    off = 0
    for n in names:
        gdn[n] = snapped[off : off + raw[n].shape[0]].contiguous()
        off += raw[n].shape[0]
    gdn[P + "out_proj.weight"] = G.ttnn_bfp_roundtrip(raw[P + "out_proj.weight"], G.consecutive_groups(args.dim)).to(
        torch.bfloat16
    )
    gu = torch.cat([raw_mlp["gate_proj.weight"], raw_mlp["up_proj.weight"]])
    gu = G.ttnn_bfp_roundtrip(gu, G.consecutive_groups(gu.shape[0])).to(torch.bfloat16)
    mlp = {
        "gate_proj.weight": gu[: args.hidden_dim].contiguous(),
        "up_proj.weight": gu[args.hidden_dim :].contiguous(),
        "down_proj.weight": G.ttnn_bfp_roundtrip(raw_mlp["down_proj.weight"], G.consecutive_groups(args.dim)).to(
            torch.bfloat16
        ),
    }

    q4, dt4 = _load_all(mesh_device, args, gdn, mlp, monkeypatch, bfp4=True)
    q8, dt8 = _load_all(mesh_device, args, gdn, mlp, monkeypatch, bfp4=False)
    assert all(d == ttnn.bfloat4_b for d in dt4.values()), dt4
    assert all(d == ttnn.bfloat8_b for d in dt8.values()), dt8
    for k in q8:
        assert torch.equal(q4[k], q8[k]), f"{k}: GPTQ grid weight not lossless in bfloat4_b on TP={tp}"

    # Negative control: off-grid weights must lose bits in bfloat4_b.
    n4, _ = _load_all(mesh_device, args, raw, raw_mlp, monkeypatch, bfp4=True)
    n8, _ = _load_all(mesh_device, args, raw, raw_mlp, monkeypatch, bfp4=False)
    assert not torch.equal(n4["qkvz"], n8["qkvz"]) and not torch.equal(n4["w2"], n8["w2"])
