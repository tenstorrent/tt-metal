# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""`TtConv1d`'s once-per-geometry verify-and-resolve (`tt/hifigan/conv.py`), checked
against a float64 torch conv -- the resolved result must be *accurate*, not merely
"agree with the fallback".

Regression test for the resolver picking the worse result. It used to compare the
fast path (prepared weight + `accurate_compute_config`) with a raw-weight +
`safe_compute_config` reference by `max|out|` within 2%, and on a disagreement
switch to the reference on the assumption it was the truth. Measured at
`Conv1d(128->128, k=11)`, length 18560 (a real HiFT resblock geometry at the
464-frame test utterance), the accurate config has ~0.4% relative error and the
"safe" fallback ~2.7% -- so the check fired and then chose the less accurate one.
Prepared and raw weights were bit-identical there: it is the compute-config axis,
not the weight-prep axis.
"""

from __future__ import annotations

import pytest
import torch

needs_l1_small = pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)

# (in_ch, out_ch, kernel, stride, padding, dilation, length)
GEOMETRIES = [
    (128, 128, 11, 1, 5, 1, 18560),  # resblock, the live-warning geometry
    (128, 128, 11, 1, 15, 3, 18560),  # same, dilated
    (128, 128, 7, 1, 3, 1, 18560),  # the other geometry that warned
    (18, 64, 1, 1, 0, 1, 55681),  # source_downs[2] at real length
    (18, 128, 6, 3, 1, 1, 55681),  # source_downs[1]
    (18, 256, 30, 15, 7, 1, 55681),  # source_downs[0]
]


@needs_l1_small
@pytest.mark.parametrize("in_ch,out_ch,k,stride,pad,dil,length", GEOMETRIES)
def test_conv1d_resolved_path_is_accurate(device, in_ch, out_ch, k, stride, pad, dil, length):
    import ttnn
    from models.demos.audio.cosyvoice2.tt.hifigan.conv import TtConv1d

    g = torch.Generator().manual_seed(0)
    w = torch.randn(out_ch, in_ch, k, generator=g) / (in_ch * k) ** 0.5
    b = torch.randn(out_ch, generator=g) * 0.1
    x = torch.randn(1, length, in_ch, generator=g) * 0.5  # channels-last

    want = torch.nn.functional.conv1d(
        x.transpose(1, 2).double(), w.double(), b.double(), stride=stride, padding=pad, dilation=dil
    ).transpose(1, 2)

    conv = TtConv1d(device, w, b, stride=stride, padding=pad, dilation=dil, dtype=ttnn.float32)
    x_dev = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    out, out_len = conv(x_dev, length, 1)
    got = ttnn.to_torch(out).float().reshape(1, out_len, out_ch).double()

    assert got.shape == want.shape, (got.shape, want.shape)
    rel = float((got - want).norm() / want.norm())
    print(f"\n  Conv1d({in_ch}->{out_ch}, k={k}, s={stride}, d={dil}) L={length}: rel err vs fp64 = {rel:.4f}")
    # bf16-stored weights alone account for ~0.4%; the "safe" fallback measured ~2.7%.
    assert rel < 0.01, rel


@needs_l1_small
def test_conv1d_resolver_rejects_a_corrupted_prepared_weight(device):
    """The tie-break path itself. Healthy geometries never reach it (the fast path
    and the reference agree), so simulate the #55545 failure -- a prepared weight
    that is silently wrong -- by handing back the prepared weight of a *different*
    (0.03x scaled, the ~30x-too-small symptom `TtStft` showed) conv. The resolver
    must notice, arbitrate against the float64 host conv, and land on a raw-weight
    candidate that is actually accurate."""
    import ttnn
    from models.demos.audio.cosyvoice2.tt.hifigan.conv import TtConv1d

    in_ch, out_ch, k, pad, length = 128, 128, 11, 5, 4000
    g = torch.Generator().manual_seed(1)
    w = torch.randn(out_ch, in_ch, k, generator=g) / (in_ch * k) ** 0.5
    b = torch.randn(out_ch, generator=g) * 0.1
    x = torch.randn(1, length, in_ch, generator=g) * 0.5
    want = torch.nn.functional.conv1d(x.transpose(1, 2).double(), w.double(), b.double(), padding=pad).transpose(1, 2)

    conv = TtConv1d(device, w, b, padding=pad, dtype=ttnn.float32)
    corrupt = TtConv1d(device, w * 0.03, b, padding=pad, dtype=ttnn.float32)
    conv._prepared = lambda x_, length_, batch_: corrupt._prepared(x_, length_, batch_)

    x_dev = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    out, out_len = conv(x_dev, length, 1)
    got = ttnn.to_torch(out).float().reshape(1, out_len, out_ch).double()

    rel = float((got - want).norm() / want.norm())
    print(f"\n  corrupted prepared weight -> resolved rel err vs fp64 = {rel:.4f}")
    assert conv._verified_config[(length, 1)][0] is conv.weight, "resolver kept the corrupted prepared weight"
    assert rel < 0.01, rel


# (in_ch, out_ch, kernel, stride, padding, length): the vocoder's three upsample stages at the
# 464-frame test utterance (padding = (k - stride) // 2, as `TorchHiFTDecodeRef` builds them).
UPSAMPLE_GEOMETRIES = [
    (512, 256, 16, 8, 4, 464),
    (256, 128, 11, 5, 3, 3712),
    (128, 64, 7, 3, 2, 18560),
]


@needs_l1_small
@pytest.mark.parametrize("in_ch,out_ch,k,stride,pad,length", UPSAMPLE_GEOMETRIES)
def test_conv_transpose1d_resolved_path_is_accurate(device, in_ch, out_ch, k, stride, pad, length):
    import ttnn
    from models.demos.audio.cosyvoice2.tt.hifigan.upsample import TtConvTranspose1d

    g = torch.Generator().manual_seed(0)
    w = torch.randn(in_ch, out_ch, k, generator=g) / (in_ch * k) ** 0.5
    b = torch.randn(out_ch, generator=g) * 0.1
    x = torch.randn(1, length, in_ch, generator=g) * 0.5
    want = torch.nn.functional.conv_transpose1d(
        x.transpose(1, 2).double(), w.double(), b.double(), stride=stride, padding=pad
    ).transpose(1, 2)

    conv = TtConvTranspose1d(device, w, b, stride=stride, padding=pad, dtype=ttnn.float32)
    x_dev = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    out, out_len = conv(x_dev, length, 1)
    got = ttnn.to_torch(out).float().reshape(1, out_len, out_ch).double()

    assert got.shape == want.shape, (got.shape, want.shape)
    rel = float((got - want).norm() / want.norm())
    print(f"\n  ConvTranspose1d({in_ch}->{out_ch}, k={k}, s={stride}) L={length}: rel err vs fp64 = {rel:.4f}")
    assert rel < 0.01, rel


@needs_l1_small
def test_conv_transpose1d_resolver_rejects_a_corrupted_prepared_weight(device):
    import ttnn
    from models.demos.audio.cosyvoice2.tt.hifigan.upsample import TtConvTranspose1d

    in_ch, out_ch, k, stride, pad, length = 256, 128, 11, 5, 3, 800
    g = torch.Generator().manual_seed(1)
    w = torch.randn(in_ch, out_ch, k, generator=g) / (in_ch * k) ** 0.5
    b = torch.randn(out_ch, generator=g) * 0.1
    x = torch.randn(1, length, in_ch, generator=g) * 0.5
    want = torch.nn.functional.conv_transpose1d(
        x.transpose(1, 2).double(), w.double(), b.double(), stride=stride, padding=pad
    ).transpose(1, 2)

    conv = TtConvTranspose1d(device, w, b, stride=stride, padding=pad, dtype=ttnn.float32)
    corrupt = TtConvTranspose1d(device, w * 0.03, b, stride=stride, padding=pad, dtype=ttnn.float32)
    conv._prepared = lambda nhwc_, length_, batch_: corrupt._prepared(nhwc_, length_, batch_)

    x_dev = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    out, out_len = conv(x_dev, length, 1)
    got = ttnn.to_torch(out).float().reshape(1, out_len, out_ch).double()

    rel = float((got - want).norm() / want.norm())
    print(f"\n  corrupted prepared weight -> resolved rel err vs fp64 = {rel:.4f}")
    assert conv._verified_config[(length, 1)][0] is conv.weight, "resolver kept the corrupted prepared weight"
    assert rel < 0.01, rel
