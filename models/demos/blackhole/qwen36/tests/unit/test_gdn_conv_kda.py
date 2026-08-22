# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""KDA fused decode conv (gdn/conv_kda.py) vs the composite shift-register.

Runs several sequential decode steps so state EVOLUTION is checked, not one
step: both paths start from the same nonzero window (exercising
rebuild_window, the prefill->decode sync) and must track a torch golden and
each other step after step. Widths cover the TP=8 and TP=4 per-device q/k/v
splits plus the 27B campaign width. Single device; no checkpoint needed.
"""
import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.tests.test_factory import compute_pcc
from models.demos.blackhole.qwen36.tt.gdn import conv_kda

pytestmark = [
    run_for_blackhole(),
    pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True),
]

K = 4
B = 8


def _dev(device, t, layout=ttnn.TILE_LAYOUT):
    kw = {"mesh_mapper": ttnn.ReplicateTensorToMesh(device)} if isinstance(device, ttnn.MeshDevice) else {}
    return ttnn.from_torch(t.bfloat16(), dtype=ttnn.bfloat16, layout=layout, device=device, **kw)


def _host(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])


def _composite_step(st, taps, qkv):
    """The exact forward_decode composite branch: shift-register + MAC + SiLU + split."""
    for j in range(K - 1):
        ttnn.copy(st[j + 1], st[j])
    ttnn.copy(qkv, st[K - 1])
    ttnn.deallocate(qkv)
    conv = ttnn.multiply(st[0], taps[0])
    for j in range(1, K):
        conv = ttnn.mac(st[j], taps[j], conv)
    return ttnn.silu(conv)


def _torch_step(win, taps, x):
    """Golden: roll the window, K-tap conv, SiLU. All float32."""
    win[:, :-1, :] = win[:, 1:, :].clone()
    win[:, -1, :] = x
    return torch.nn.functional.silu((win * taps.reshape(1, K, -1)).sum(dim=1))


@pytest.mark.parametrize(
    ("kd", "vd"),
    [
        (256, 512),  # TP=8 per-device widths (key_dim 2048 / value_dim 4096)
        (512, 1024),  # TP=4 per-device widths
        (320, 640),  # 27B campaign decode width (C=1280)
    ],
    ids=["tp8", "tp4", "c1280"],
)
def test_kda_decode_conv_matches_composite(device, kd, vd):
    torch.manual_seed(1234)
    C = 2 * kd + vd

    taps_t = torch.randn(K, C) * 0.5
    taps = [_dev(device, taps_t[j].reshape(1, 1, C)) for j in range(K)]

    # Nonzero starting window (as left by prefill), oldest tap first.
    win0 = torch.randn(B, K, C) * 0.5
    st = [_dev(device, win0[:, m, :].reshape(1, B, C)) for m in range(K)]

    bufs = conv_kda.KDAConvBuffers(device, B, K, C)
    conv_kda.rebuild_window(bufs, st)  # prefill->decode sync under test
    assert torch.equal(_host(bufs.win), win0.bfloat16()), "rebuild_window must copy conv_states bit-exactly"

    # Golden tracks in bf16-quantized inputs (both device paths consume bf16).
    win_ref = win0.bfloat16().float()
    taps_ref = taps_t.bfloat16().float()

    for step in range(6):
        x = torch.randn(B, C) * 0.5
        ref = _torch_step(win_ref, taps_ref, x.bfloat16().float())

        conv_c = _composite_step(st, taps, _dev(device, x.reshape(1, B, C)))
        q_c = _host(ttnn.slice(conv_c, (0, 0, 0), (1, B, kd))).reshape(B, kd)
        k_c = _host(ttnn.slice(conv_c, (0, 0, kd), (1, B, 2 * kd))).reshape(B, kd)
        v_c = _host(ttnn.slice(conv_c, (0, 0, 2 * kd), (1, B, C))).reshape(B, vd)
        ttnn.deallocate(conv_c)

        q_f, k_f, v_f = conv_kda.decode_conv(bufs, _dev(device, x.reshape(1, B, C)), taps, kd, vd)
        q_k, k_k, v_k = (_host(t).reshape(B, -1) for t in (q_f, k_f, v_f))

        for name, ref_s, comp, fused in (
            ("q", ref[:, :kd], q_c, q_k),
            ("k", ref[:, kd : 2 * kd], k_c, k_k),
            ("v", ref[:, 2 * kd :], v_c, v_k),
        ):
            pcc_cr = compute_pcc(comp, ref_s)
            pcc_kr = compute_pcc(fused, ref_s)
            pcc_ck = compute_pcc(fused, comp)
            assert pcc_cr > 0.999, f"step {step} {name}: composite vs golden pcc={pcc_cr}"
            assert pcc_kr > 0.999, f"step {step} {name}: kda vs golden pcc={pcc_kr}"
            assert pcc_ck > 0.9999, f"step {step} {name}: kda vs composite pcc={pcc_ck}"

        # The window stores raw inputs, so both state representations must stay
        # bit-exact with the golden window (state evolution, not just outputs).
        assert torch.equal(_host(bufs.win), win_ref.bfloat16())
        for m in range(K):
            assert torch.equal(_host(st[m]).reshape(B, C), win_ref[:, m, :].bfloat16())


def test_kda_window_slot_edits_match_conv_states(device):
    """write_window_slot / gather_window_slots mirror the conv_states per-slot edits."""
    torch.manual_seed(99)
    C = 320
    win0 = torch.randn(B, K, C)
    st = [_dev(device, win0[:, m, :].reshape(1, B, C)) for m in range(K)]
    bufs = conv_kda.KDAConvBuffers(device, B, K, C)
    conv_kda.rebuild_window(bufs, st)

    slot, user = 2, torch.randn(K, C)
    convs = [_dev(device, user[m].reshape(1, 1, C)) for m in range(K)]
    conv_kda.write_window_slot(bufs, convs, slot)
    ref = win0.bfloat16().clone()
    ref[slot] = user.bfloat16()
    assert torch.equal(_host(bufs.win), ref)

    remap = [1, 0, 2, 2, 4, 5, 7, 6]
    conv_kda.gather_window_slots(bufs, remap)
    assert torch.equal(_host(bufs.win), ref[remap])

    bufs.reset()
    assert torch.equal(_host(bufs.win), torch.zeros(B, K, C, dtype=torch.bfloat16))
