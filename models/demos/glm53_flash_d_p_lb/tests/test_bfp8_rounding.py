# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Precision analysis: is the device's conversion to bfp8 biased (vs the host quantizer, round to nearest)? On one
chip, real GLM activations (layer 4 component golden: the experts' input x, and h = act(x Wg, x Wu) of expert 0) and
Gaussian data, converted to bfloat8_b on the device by tilize (row-major bf16 -> TILE bfp8, as the flat op's x relays
do) and by typecast (bf16 / fp32 TILE -> bfp8, the packer path), each compared with the host's from_torch(bfloat8_b):
the fraction of elements equal to the host's, the scale <dev, v> / <v, v>, sum|dev| / sum|v|, and of the elements
that differ from the host's, how many are larger in magnitude."""

import pytest
import torch

import ttnn
from models.demos.common.bringup.testing.component import _step
from models.demos.common.bringup.testing.harness import component_golden, spec

S = spec()


def _report(tag, dev, host, v):
    dev, host, v = dev.double().reshape(-1), host.double().reshape(-1), v.double().reshape(-1)
    dif = dev != host
    nd = int(dif.sum())
    up = int((dev.abs() > host.abs())[dif].sum())
    print(
        f"[bfp8] {tag:34s} equal-to-host {1 - nd / dev.numel():.5f} (differ {nd}, of which larger |.| {up})  "
        f"<dev,v>/<v,v> {float((dev * v).sum() / (v * v).sum()):.6f}  <host,v>/<v,v> {float((host * v).sum() / (v * v).sum()):.6f}  "
        f"sum|dev|/sum|v| {float(dev.abs().sum() / v.abs().sum()):.6f}  sum|host|/sum|v| {float(host.abs().sum() / v.abs().sum()):.6f}",
        flush=True,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
def test_bfp8_rounding(device):
    from models.demos.glm53_flash_d_p.bringup import hooks
    from models.demos.glm53_flash_d_p.reference.weights import PackedExpert

    g, c = component_golden(S)
    ref = S.hooks().reference(S, layers=[4], dtype=torch.float32)
    st = _step(ref, 4, "experts")
    x = g.layer(c, 4)[st.inputs[0]].float().reshape(-1, 4096)
    loader, _ = hooks._loader_cfg(S)
    gw, uw, _ = PackedExpert(loader, 4, 0).weights(torch.float32)
    xs = x[:512]
    h = torch.nn.functional.silu((xs @ gw.T).clamp(max=10.0)) * (xs @ uw.T).clamp(-10.0, 10.0)
    torch.manual_seed(0)
    data = {"x (ffn_norm, real)": x, "h (expert 0, real)": h, "gaussian": torch.randn(1024, 4096)}
    hq = lambda t: ttnn.to_torch(ttnn.from_torch(t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)).float()  # noqa
    for name, v in data.items():
        for src in ("bf16", "fp32"):
            vs = v.bfloat16().float() if src == "bf16" else v.float()
            host = hq(vs)
            if src == "bf16":  # tilize: row-major bf16 -> TILE bfp8 on the device
                t = ttnn.from_torch(vs, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
                d = ttnn.to_torch(ttnn.tilize(t, dtype=ttnn.bfloat8_b)).float()
                _report(f"{name} tilize bf16->bfp8", d, host, vs)
            dt = ttnn.bfloat16 if src == "bf16" else ttnn.float32
            t = ttnn.from_torch(vs, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
            d = ttnn.to_torch(ttnn.typecast(t, ttnn.bfloat8_b)).float()
            _report(f"{name} typecast {src}->bfp8", d, host, vs)
