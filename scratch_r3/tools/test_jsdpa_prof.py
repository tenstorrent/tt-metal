# Round 3 matmul (#58714): ttnn.transformer.joint_scaled_dot_product_attention at the per-device shapes of tt_dit's Flux1 on a
# P300 1x2 mesh (sequence parallelism 1, tensor parallelism 2: 12 of 24 heads, 4096 image and 512 text tokens, head dim 128,
# q chunk 128, k chunk 512, HiFi2, 16-bit DEST, exp_approx_mode False) and a QwenImage-like shape (12 heads, 4096 + 256), with a
# PCC check against torch (BITID-style raw outputs saved under JSDPA_OUT for a bit comparison across farms).
import os as _os
import sys as _sys

if not (_os.environ.get("HWLOCK_HELD") or _os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH")):
    _sys.exit("not under hwlock")
import os

import pytest
import torch
import ttnn

CASES = [
    # (name, heads, seq, joint_seq, head_dim, q_chunk, k_chunk)
    ("flux1_p300_sp1", 12, 4096, 512, 128, 128, 512),
    ("qwenimage_sp1", 12, 4096, 256, 128, 128, 512),
]
_INPUTS = {}
_REF = {}


@pytest.fixture(scope="module")
def device():
    dev = ttnn.CreateDevice(device_id=0, l1_small_size=32768)
    ttnn.SetDefaultDevice(dev)
    yield dev
    ttnn.close_device(dev)


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_jsdpa_prof(device, case):
    name, nh, s, js, d, qc, kc = case
    if name not in _INPUTS:
        _INPUTS.clear()
        g = torch.Generator().manual_seed(0)
        t = [torch.randn(1, nh, n, d, generator=g).to(torch.bfloat16) for n in (s, s, s, js, js, js)]
        _INPUTS[name] = (t, [ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device) for x in t])
    t, tt = _INPUTS[name]
    cfg = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=qc,
        k_chunk_size=kc,
        exp_approx_mode=False,
    )
    ckc = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
    )
    out, jout = ttnn.transformer.joint_scaled_dot_product_attention(
        *tt, joint_strategy="rear", program_config=cfg, compute_kernel_config=ckc
    )
    o = ttnn.to_torch(out)[:, :, :s, :]
    jo = ttnn.to_torch(jout)[:, :, :js, :]
    if os.environ.get("JSDPA_OUT"):
        torch.save((o.contiguous().view(torch.int16), jo.contiguous().view(torch.int16)), os.path.join(os.environ["JSDPA_OUT"], f"{name}.pt"))
    if not os.environ.get("JSDPA_OUT"):  # profiler runs: device time only; the bit runs check the PCC
        return
    if name + "_ref" not in _REF:  # the reference once per case
        q = torch.cat([t[0], t[3]], dim=2).float()
        k = torch.cat([t[1], t[4]], dim=2).float()
        v = torch.cat([t[2], t[5]], dim=2).float()
        _REF[name + "_ref"] = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    ref = _REF[name + "_ref"]
    got = torch.cat([o, jo], dim=2).float()
    pcc = torch.corrcoef(torch.stack([ref.flatten(), got.flatten()]))[0, 1].item()
    assert pcc > 0.99, f"{name}: PCC {pcc}"
