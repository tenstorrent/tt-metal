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
def test_stock_conv1d_l1_oom_vs_depthwise(device, device_params, expect_error):
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
    with expect_error(RuntimeError, "circular buffers"):  # TT_THROW: static CBs beyond max L1 size
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


# Repro for the SECOND, distinct stock-conv1d failure mode: the L1_SMALL OOM that the LTX distilled
# vocoder hits when the mesh is opened with l1_small_size=0. This is NOT the static-CB overflow above
# (that needs a long sequence + an existing L1_SMALL pool); it fires on ANY depthwise shape and is a
# pure device-config issue:
#   stock ttnn.conv1d(groups=C) runs an UntilizeWithHalo sliding-window gather, which allocates its
#   small sharding/config tensors from the device's dedicated L1_SMALL pool. l1_small_size defaults
#   to 0, and the LTX distilled test's device params (ring_trace_params) never set it — so the pool
#   is 0 bytes and even the ~240 B config tensor fails:
#     TT_FATAL bank_manager.cpp: Out of Memory ... L1_SMALL buffer ... bank size is 0 B
#   (move_config_tensor_to_device -> to_device(L1_SMALL) -> BankManager::allocate_buffer).
# The experimental conv1d_depthwise op streams its taps and never touches L1_SMALL, so it runs on the
# same l1_small_size=0 device. Giving the device an L1_SMALL budget (32768, the vocoder's setting)
# lets the config tensor allocate and the stock call succeeds. Small smoke shape — the failure is
# independent of sequence length, unlike the static-CB overflow above.
@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}, {"l1_small_size": 32768}], indirect=True)
def test_stock_conv1d_depthwise_needs_l1_small(device, device_params, expect_error):
    C, T_pad, K, stride = 64, 256, 12, 1  # tiny: no CB overflow, isolates the L1_SMALL config alloc
    torch.manual_seed(0)
    x = torch.randn(1, T_pad, C, dtype=torch.float32)
    taps = _make_taps(K)
    x_tt = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    wt = torch.tensor(taps, dtype=torch.float32).reshape(1, 1, K).expand(C, 1, K).contiguous()
    weight = ttnn.from_torch(wt, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.float32)
    cc = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    conv_config = ttnn.Conv1dConfig(weights_dtype=ttnn.float32, shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED)

    def _stock_conv1d():
        return ttnn.conv1d(
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
            return_output_dim=True,
            return_weights_and_bias=True,
        )

    if device_params["l1_small_size"] == 0:
        # The exact failure the LTX distilled vocoder hits on an l1_small_size=0 mesh.
        with expect_error(RuntimeError, "L1_SMALL"):
            _stock_conv1d()
    else:
        # 32 KB L1_SMALL pool: the halo config tensor allocates and the stock call matches golden.
        out, _, _ = _stock_conv1d()
        out = ttnn.to_layout(ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG), ttnn.ROW_MAJOR_LAYOUT)
        out = ttnn.to_torch(out).reshape(1, T_pad - K + 1, C)
        assert_with_pcc(_golden(x, taps, stride), out, pcc=PCC)


# Why the custom op is needed, demonstrated WITHOUT a crash: on the mesh, stock ttnn.conv1d(groups=C)
# cannot keep its depthwise weights in the device-ready (pre-tilized/sharded) format — every call it
# logs `conv2d: Device weights not properly prepared, pulling back to host and trying to reprocess`
# (conv2d.cpp) and re-prepares them on the host. That host round-trip is what makes the op unusable
# in the LTX vocoder: it is slow eager, and under tracing it cannot be captured at all (the host work
# isn't a device program), which is the runtime-args failure the traced distilled pipeline hits.
# The dedicated ttnn.experimental.conv1d_depthwise streams its taps and stays fully on device, so it
# emits no such warning. We capture fd-level stdout/stderr (the C++ Op logger) to prove the contrast.
_HOST_FALLBACK = "Device weights not properly prepared"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("C,T_pad,K,stride", [(512, 1457, 12, 1), (256, 2921, 12, 1)])  # real STAGE_C
def test_stock_conv1d_depthwise_host_fallback_vs_custom(mesh_device, device_params, capfd, C, T_pad, K, stride):
    torch.manual_seed(0)
    x = torch.randn(1, T_pad, C, dtype=torch.float32)
    taps = _make_taps(K)
    x_tt = ttnn.from_torch(
        x,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # (1) Custom op: stays on device -> no host-fallback warning.
    capfd.readouterr()  # drain prior output
    ttnn.experimental.conv1d_depthwise(x_tt, taps=taps, stride=stride)
    ttnn.synchronize_device(mesh_device)
    cap = capfd.readouterr()
    custom_hits = (cap.out + cap.err).count(_HOST_FALLBACK)
    logger.info(f"[custom op] C={C} T_pad={T_pad}: host-fallback warnings = {custom_hits}")

    # (2) Stock ttnn.conv1d depthwise, weights built the vocoder's way (from_torch on the mesh,
    # no mesh_mapper) -> conv2d can't use them as-is and re-prepares on host every call.
    wt = torch.tensor(taps, dtype=torch.float32).reshape(1, 1, K).expand(C, 1, K).contiguous()
    weight = ttnn.from_torch(wt, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.float32)
    cc = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    conv_config = ttnn.Conv1dConfig(weights_dtype=ttnn.float32, shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED)
    capfd.readouterr()  # drain
    ttnn.conv1d(
        input_tensor=ttnn.reshape(x_tt, (1, T_pad, 1, C)),
        weight_tensor=weight,
        device=mesh_device,
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
        return_output_dim=True,
        return_weights_and_bias=True,
    )
    ttnn.synchronize_device(mesh_device)
    cap = capfd.readouterr()
    stock_hits = (cap.out + cap.err).count(_HOST_FALLBACK)
    logger.info(f"[stock conv1d] C={C} T_pad={T_pad}: host-fallback warnings = {stock_hits}")

    # The point: stock conv1d round-trips weights to host; the custom op never does.
    assert stock_hits > 0, "expected stock ttnn.conv1d depthwise to fall back to host weight prep"
    assert custom_hits == 0, f"custom op unexpectedly fell back to host ({custom_hits}x)"


# Decisive question for "can the pipeline fix this without touching the op?": does REUSING the
# prepared weight that ttnn.conv1d returns (return_weights_and_bias=True) avoid the host fallback on
# the next call? depthwise_tap_filter already caches that returned weight. If call #2 (fed the
# returned weight) is clean, the 401 fallbacks in the vocoder are a caller-side caching bug ->
# pipeline-fixable. If call #2 still falls back, the prepared depthwise weight isn't recognized as
# valid on reuse -> op-side, the caller can't fix it.
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize(
    "C,T_pad,K,stride",
    [(256, 2921, 12, 1), (64, 28841, 12, 1)],  # short (no slice) + long STAGE_C tail (DRAM-sliced)
)
def test_stock_conv1d_depthwise_weight_reuse(mesh_device, device_params, capfd, C, T_pad, K, stride):
    torch.manual_seed(0)
    x = torch.randn(1, T_pad, C, dtype=torch.float32)
    taps = _make_taps(K)
    x_tt = ttnn.from_torch(
        x,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    wt = torch.tensor(taps, dtype=torch.float32).reshape(1, 1, K).expand(C, 1, K).contiguous()
    raw_weight = ttnn.from_torch(wt, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.float32)
    cc = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    conv_config = ttnn.Conv1dConfig(weights_dtype=ttnn.float32, shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED)

    def _call(w):
        out, _, (w_out, _b) = ttnn.conv1d(
            input_tensor=ttnn.reshape(x_tt, (1, T_pad, 1, C)),
            weight_tensor=w,
            device=mesh_device,
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
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        ttnn.synchronize_device(mesh_device)
        cap = capfd.readouterr()
        return w_out, (cap.out + cap.err).count(_HOST_FALLBACK)

    capfd.readouterr()  # drain
    w1, hits1 = _call(raw_weight)  # call #1: raw weight -> expect fallback
    w2, hits2 = _call(w1)  # call #2: feed back the returned (prepared) weight
    w3, hits3 = _call(w2)  # call #3: reuse again
    logger.info(f"[weight-reuse] C={C} T_pad={T_pad}: hits call1={hits1} call2={hits2} call3={hits3}")

    assert hits1 > 0, "call #1 (raw weight) should fall back to host"
    # The diagnostic: is reuse clean (caller-fixable) or still falling back (op-side)?
    if hits2 == 0 and hits3 == 0:
        logger.info(
            "[weight-reuse] VERDICT: reuse is clean -> host fallback is caller-side fixable (cache prepared weight)"
        )
    else:
        logger.info(f"[weight-reuse] VERDICT: reuse still falls back ({hits2},{hits3}) -> op-side, caller cannot fix")
