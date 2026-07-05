# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


PCC = 0.999


def _golden(x_btc, taps, stride):
    # Depthwise FIR with taps shared across channels: weight (C, 1, K), groups=C.
    B, T_pad, C = x_btc.shape
    K = len(taps)
    w = torch.tensor(taps, dtype=torch.float32).reshape(1, 1, K).expand(C, 1, K).contiguous()
    x_bct = x_btc.permute(0, 2, 1).contiguous()  # (B, C, T_pad)
    y = torch.nn.functional.conv1d(x_bct, w, stride=stride, groups=C)  # (B, C, T_out)
    return y.permute(0, 2, 1).contiguous()  # (B, T_out, C)


def _make_taps(K):
    g = torch.Generator().manual_seed(K)
    return torch.randn(K, generator=g).tolist()


def _run(device, B, T_pad, C, K, stride, mesh_mapper=None, mesh_composer=None):
    torch.manual_seed(0)
    x = torch.randn(B, T_pad, C, dtype=torch.float32)
    taps = _make_taps(K)
    golden = _golden(x, taps, stride)

    x_tt = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, mesh_mapper=mesh_mapper)
    out_tt = ttnn.experimental.conv1d_depthwise(x_tt, taps=taps, stride=stride)
    out = ttnn.to_torch(out_tt, mesh_composer=mesh_composer)

    passing, pcc_msg = assert_with_pcc(golden, out, pcc=PCC)
    logger.info(f"B={B} T_pad={T_pad} C={C} K={K} stride={stride} -> {pcc_msg}")


# Smoke shapes: small, fast, single-device. Covers C padding (1/2/16/24/48), strides, K.
SMOKE = [
    (1, 256, 1, 12, 1),  # C<TILE padding path (HannUp uses C=1), small T
    (1, 256, 2, 12, 1),  # C<TILE padding path (HannUp uses C=2), small T
    (1, 256, 64, 12, 1),
    (1, 256, 64, 12, 2),
    (1, 512, 32, 12, 1),
    (1, 512, 16, 12, 2),
    (1, 300, 24, 12, 1),
    (1, 320, 48, 4, 1),
    (1, 600, 128, 29, 1),
]


@pytest.mark.parametrize("B,T_pad,C,K,stride", SMOKE)
def test_conv1d_depthwise_smoke(device, B, T_pad, C, K, stride):
    _run(device, B, T_pad, C, K, stride)


# All 35 production vocoder + BWE filter shapes (t_frames=120). Global T_pad.
STAGE_B = [
    (1536, 616, 11, 1),
    (768, 1241, 12, 1),
    (768, 611, 12, 2),
    (768, 1205, 4, 1),
    (384, 2441, 12, 1),
    (384, 1211, 12, 2),
    (384, 2405, 4, 1),
    (192, 4841, 12, 1),
    (192, 2411, 12, 2),
    (192, 4805, 4, 1),
    (96, 9641, 12, 1),
    (96, 4811, 12, 2),
    (96, 9605, 4, 1),
    (48, 19241, 12, 1),
    (48, 9611, 12, 2),
    (48, 19205, 4, 1),
    (24, 38441, 12, 1),
    (24, 19211, 12, 2),
]
STAGE_C = [
    (512, 1457, 12, 1),
    (256, 2921, 12, 1),
    (256, 1451, 12, 2),
    (256, 7216, 11, 1),
    (128, 14441, 12, 1),
    (128, 7211, 12, 2),
    (128, 14405, 4, 1),
    (64, 28841, 12, 1),
    (64, 14411, 12, 2),
    (64, 28805, 4, 1),
    (32, 57641, 12, 1),
    (32, 28811, 12, 2),
    (32, 57605, 4, 1),
    (16, 115241, 12, 1),
    (16, 57611, 12, 2),
]
HANNUP = [(1, 19228, 29, 1), (2, 19228, 29, 1)]

# (C, T_pad, K, stride) -> single-device, B=1. Tail (T_pad > 30k) marked slow.
PROD = [(c, t, k, s) for (c, t, k, s) in (STAGE_B + STAGE_C)] + [
    pytest.param(
        c,
        t,
        k,
        s,
        marks=pytest.mark.skip(
            reason="C=1/2 padding path covered by smoke; HannUp T=19228 tail redundant with prod-tail"
        ),
    )
    for (c, t, k, s) in HANNUP
]


@pytest.mark.parametrize("C,T_pad,K,stride", [(c, t, k, s) for (c, t, k, s) in (STAGE_B + STAGE_C) if t <= 30000])
def test_conv1d_depthwise_prod_head(device, C, T_pad, K, stride):
    _run(device, 1, T_pad, C, K, stride)


@pytest.mark.slow
@pytest.mark.parametrize("C,T_pad,K,stride", [(c, t, k, s) for (c, t, k, s) in (STAGE_B + STAGE_C) if t > 30000])
def test_conv1d_depthwise_prod_tail(device, C, T_pad, K, stride):
    _run(device, 1, T_pad, C, K, stride)


# Mesh: the op carries no mesh-specific logic — it runs the same per-chip program on every
# device of the mesh. Replicating the input and checking every device reproduces the golden
# proves mesh-broadcast correctness on 2x4 (8 chips) and 4x8 (32 chips). Cross-shard halo is a
# pipeline concern (validated e2e), not an op concern.
@pytest.mark.parametrize("mesh_device", [(2, 4), (4, 8)], indirect=True)
@pytest.mark.parametrize("C,T_pad,K,stride", [(768, 1241, 12, 1), (64, 2811, 12, 2), (32, 4811, 4, 1)])
def test_conv1d_depthwise_mesh(mesh_device, C, T_pad, K, stride):
    torch.manual_seed(0)
    x = torch.randn(1, T_pad, C, dtype=torch.float32)
    taps = _make_taps(K)
    golden = _golden(x, taps, stride)

    x_tt = ttnn.from_torch(
        x,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    out_tt = ttnn.experimental.conv1d_depthwise(x_tt, taps=taps, stride=stride)
    # dim=0 concat over replicated (1, T_out, C) -> (num_devices, T_out, C); every chip must match.
    out = ttnn.to_torch(out_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    n = mesh_device.get_num_devices()
    for i in range(n):
        assert_with_pcc(golden[0], out[i], pcc=PCC)
    logger.info(f"mesh {tuple(mesh_device.shape)} C={C} T_pad={T_pad} K={K} stride={stride}: {n} chips OK")


# Repro for #46395: why a dedicated depthwise op exists instead of stock ttnn.conv1d.
# The vocoder anti-alias resample filters are depthwise (one shared tap set per channel) and run
# at long sequence lengths after the upsample stages (T_pad up to ~115k at t_frames=120). A stock
# ttnn.conv1d(groups=C) with HEIGHT_SHARDED buffers the whole sequence in L1, so its static
# circular buffers exceed the 1.5 MB L1 and it throws (single device) / hung on a T-sharded mesh —
# hence ttnn.experimental.conv1d_depthwise, which streams the taps and handles the shape.
# Topology: single Blackhole device, l1_small_size=32768 (the vocoder's setting); shape is a real
# STAGE_C upsample tail.
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_stock_conv1d_l1_oom_vs_depthwise(device, device_params):
    C, T_pad, K, stride = 64, 28841, 12, 1
    torch.manual_seed(0)
    x = torch.randn(1, T_pad, C, dtype=torch.float32)
    taps = _make_taps(K)
    x_tt = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # Stock ttnn.conv1d depthwise (the pre-op path) overflows L1 at this length.
    wt = torch.tensor(taps, dtype=torch.float32).reshape(1, 1, K).expand(C, 1, K).contiguous()
    weight = ttnn.from_torch(wt, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.float32)
    cc = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    conv_config = ttnn.Conv1dConfig(weights_dtype=ttnn.float32, shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED)
    with pytest.raises(RuntimeError):  # TT_THROW: static circular buffers beyond max L1 size
        ttnn.conv1d(
            input_tensor=ttnn.reshape(x_tt, (1, T_pad, 1, C)),
            weight_tensor=weight,
            device=device,
            in_channels=C,
            out_channels=C,
            batch_size=1,
            input_length=T_pad,
            kernel_size=K,
            stride=stride,
            padding=0,
            dilation=1,
            groups=C,
            dtype=ttnn.float32,
            conv_config=conv_config,
            compute_config=cc,
        )

    # The dedicated op handles the same shape and matches the depthwise golden.
    out = ttnn.to_torch(ttnn.experimental.conv1d_depthwise(x_tt, taps=taps, stride=stride))
    assert_with_pcc(_golden(x, taps, stride), out, pcc=PCC)
