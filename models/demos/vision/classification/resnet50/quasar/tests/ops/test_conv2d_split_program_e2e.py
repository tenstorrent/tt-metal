# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Option B — end-to-end two-program split conv, with a PCC correctness gate.

The Quasar tilize 0x19 (per-tile MATH A2D datacopy never freeing the FPU dest-dvalid ring) is fixed by
UnpackToDestEn (TT_METAL_QSR_TILIZE_UNPACK_TO_DEST), but only in a tilize kernel WITHOUT an interleaved matmul
(the fused conv re-faults — dvalid-synced tilize + semaphore-synced matmul in one kernel). So the conv is split
into two Metal programs, orchestrated host-side in conv2d.cpp under TT_METAL_QSR_CONV_SPLIT_PROGRAM:
  - Program A: reader im2col gather + UnpackToDestEn tilize -> tilized activation tensor [M, K].
  - Program B: quasar matmul::linear(act_tilized, weights) -> conv output [M, N] (same GEMM the 1x1 path uses).

This runs the FULL split conv on the L1 path (pre-sharded L1 input, so no DRAM slicing / unported slice_write)
and checks the ACTUAL conv output against a torch golden — validating BOTH that the fix clears the 0x19 AND
that the unpack-to-dest tilize produced CORRECT data (fed through the matmul). Stem-like K=16-tile shape shrunk
to fit L1; act_block_h_override forces >=2 height blocks so the multi-block tilize (which faulted) is exercised.

Run (craq-sim / emulator, slow dispatch + forced JIT):
  TT_METAL_QSR_CONV_SPLIT_PROGRAM=1 TT_METAL_QSR_TILIZE_UNPACK_TO_DEST=1 \
  TT_METAL_SIMULATOR=~/sim/libttsim.so \
  TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 \
  pytest -s models/demos/vision/classification/resnet50/quasar/tests/ops/test_conv2d_split_program_e2e.py
"""

import os

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

PCC = 0.99


def _run(
    mesh_device,
    *,
    with_bias_relu,
    in_channels=32,  # default = folded-stem-like: K = 32*4*4 = 16 tiles
    out_channels=64,
    kernel_size=(4, 4),
    out_h=16,
    out_w=32,  # 16x32 = 512 sticks
    stride=(1, 1),
    padding=(0, 0),
    act_block_h_override=128,  # None -> let the factory pick (force_conv_no_spill caps it)
):
    device = mesh_device
    torch.manual_seed(0)

    batch_size = 1
    # stride 1: out = in + 2*pad - kernel + 1  ->  in = out + kernel - 1 - 2*pad
    input_height = out_h + kernel_size[0] - 1 - 2 * padding[0]
    input_width = out_w + kernel_size[1] - 1 - 2 * padding[1]
    # full im2col K in tiles (must stay <= kQuasarConvNoSpillMaxKTiles=32 for the no-spill/split path)
    import math as _math

    k_tiles = _math.ceil(in_channels * kernel_size[0] * kernel_size[1] / 32)
    print(
        f"  DIAG shape: in_ch={in_channels} out_ch={out_channels} k={kernel_size} out=({out_h},{out_w}) "
        f"K_tiles={k_tiles} N_tiles={_math.ceil(out_channels/32)}"
    )

    torch_input_nchw = torch.randn((batch_size, in_channels, input_height, input_width), dtype=torch.bfloat16).float()
    torch_weight = torch.randn((out_channels, in_channels, *kernel_size), dtype=torch.bfloat16).float()
    torch_bias = torch.randn((out_channels,), dtype=torch.bfloat16).float() if with_bias_relu else None
    torch_golden = torch.nn.functional.conv2d(
        torch_input_nchw, torch_weight, bias=torch_bias, stride=stride, padding=padding
    )
    if with_bias_relu:
        torch_golden = torch.relu(torch_golden)

    # --- pre-shard the activation into L1 (height-sharded) so conv2d takes the L1 path (not DRAM slicing) ---
    nhw = batch_size * input_height * input_width
    flat = torch.permute(torch_input_nchw, (0, 2, 3, 1)).reshape(1, 1, nhw, in_channels).contiguous()
    grid = device.compute_with_storage_grid_size()
    max_cores = grid.x * grid.y
    num_cores = max(c for c in range(1, max_cores + 1) if nhw % c == 0)
    shard_h = nhw // num_cores
    core_grid = ttnn.num_cores_to_corerangeset(num_cores, grid, True)
    in_mem = ttnn.create_sharded_memory_config(
        shape=(1, 1, shard_h, in_channels),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    tt_input = ttnn.from_torch(flat, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_input = tt_input.to(device, in_mem)
    tt_weight = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16)
    tt_bias = (
        ttnn.from_torch(torch_bias.reshape(1, 1, 1, out_channels), dtype=ttnn.bfloat16) if with_bias_relu else None
    )

    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        full_inner_dim=True,  # single K-block -> factory split_program_tilize_only eligibility + host split gate
        # act_block_h_override forces >=2 height blocks (exercises the multi-block tilize that faulted); None lets
        # the factory choose (force_conv_no_spill caps it to fit the DFB ring anyway).
        act_block_h_override=(act_block_h_override if act_block_h_override is not None else 0),
        reshard_if_not_optimal=True,
        activation=(ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU) if with_bias_relu else None),
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.LoFi, packer_l1_acc=True
    )

    # Two-program split: Program A tilize (UnpackToDestEn) + Program B matmul. Both flags set; no leak after.
    saved = {k: os.environ.get(k) for k in ("TT_METAL_QSR_CONV_SPLIT_PROGRAM", "TT_METAL_QSR_TILIZE_UNPACK_TO_DEST")}
    os.environ["TT_METAL_QSR_CONV_SPLIT_PROGRAM"] = "1"
    os.environ["TT_METAL_QSR_TILIZE_UNPACK_TO_DEST"] = "1"
    try:
        out, [oh, ow], _wb = ttnn.experimental.quasar.conv2d(
            input_tensor=tt_input,
            weight_tensor=tt_weight,
            bias_tensor=tt_bias,
            in_channels=in_channels,
            out_channels=out_channels,
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=(1, 1),
            groups=1,
            device=device,
            conv_config=conv_config,
            compute_config=compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=ttnn.bfloat16,
        )
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    print("  DIAG raw out.shape        =", tuple(out.shape))
    try:
        print("  DIAG raw out.padded_shape =", tuple(out.padded_shape()))
    except Exception as e:
        print("  DIAG padded_shape n/a:", repr(e))
    try:
        print("  DIAG raw out.layout       =", out.layout, "| mem =", out.memory_config())
    except Exception as e:
        print("  DIAG mem_config n/a:", repr(e))
    tt_out = ttnn.to_torch(ttnn.from_device(out))
    print("  DIAG to_torch raw shape   =", tuple(tt_out.shape))
    tt_out = tt_out.reshape(batch_size, oh, ow, tt_out.shape[-1])[:, :, :, :out_channels]
    tt_out = torch.permute(tt_out, (0, 3, 1, 2))
    print(f"split conv (Program A tilize + Program B matmul) completed. out shape={tuple(tt_out.shape)}")

    # -------- DIAGNOSTIC: which matmul did Program B actually compute? --------
    # The tilize (Program A) is confirmed to produce A[M,K] with K ordered [kh,kw,c] (readback test). The conv
    # golden = A_khkwc @ W_khkwc. Compare the DEVICE output against several host candidates to localize the fault:
    #   * torch_golden           : correct conv (A & W both [kh,kw,c])
    #   * W in [c,kh,kw] order    : weight K-order mismatch vs the activation
    #   * W transposed ([N,K])    : linear used the wrong weight orientation
    def _pcc(a, b):
        a = a.reshape(-1).float()
        b = b.reshape(-1).float()
        return float(torch.corrcoef(torch.stack([a, b]))[0, 1])

    with torch.no_grad():
        # host im2col A[M,K] with K = [kh,kw,c] (matches the confirmed tilize order)
        inp = torch_input_nchw  # [1, C, iH, iW]
        patches = torch.nn.functional.unfold(
            inp, kernel_size=kernel_size, stride=stride, padding=padding
        )  # [1, C*kh*kw, M]
        M = patches.shape[-1]
        # unfold gives K order [c][kh][kw]; permute to [kh][kw][c]
        A_ckhkw = patches[0].transpose(0, 1).reshape(M, in_channels, kernel_size[0], kernel_size[1])
        A_khkwc = A_ckhkw.permute(0, 2, 3, 1).reshape(M, -1)  # [M, kh*kw*c]
        # weights: torch_weight [O, C, kh, kw]
        W_khkwc = torch_weight.permute(2, 3, 1, 0).reshape(-1, out_channels)  # [kh*kw*c, O]
        W_ckhkw = torch_weight.reshape(out_channels, -1).transpose(0, 1)  # [c*kh*kw, O]
        dev = tt_out.permute(0, 2, 3, 1).reshape(M, out_channels).float()  # [M, O]
        print(
            "  DIAG PCC(device, torch_golden)          =",
            _pcc(dev, torch_golden.permute(0, 2, 3, 1).reshape(M, out_channels)),
        )
        print("  DIAG PCC(device, A_khkwc @ W_khkwc)      =", _pcc(dev, A_khkwc @ W_khkwc))
        print("  DIAG PCC(device, A_khkwc @ W_ckhkw)      =", _pcc(dev, A_khkwc @ W_ckhkw))

        # value-distribution: are the output VALUES present (permuted) or totally wrong?
        print(
            "  DIAG sorted-flat PCC(device, golden)     =",
            _pcc(torch.sort(dev.reshape(-1))[0], torch.sort((A_khkwc @ W_khkwc).reshape(-1))[0]),
        )

        # read back the PREPARED on-device weight and compare its K-order directly to the candidates
        try:
            wdev = _wb[0] if isinstance(_wb, (tuple, list)) else _wb
            w_host = ttnn.to_torch(ttnn.from_device(wdev)).float()
            w_host2 = w_host.reshape(w_host.shape[-2], w_host.shape[-1])  # [K_padded, N_padded]
            K = W_khkwc.shape[0]
            N = out_channels
            w_kn = w_host2[:K, :N]
            print("  DIAG prepared-weight shape               =", tuple(w_host.shape), "-> KxN used", (K, N))
            print("  DIAG PCC(prep_weight, W_khkwc)           =", _pcc(w_kn, W_khkwc))
            print("  DIAG PCC(prep_weight, W_ckhkw)           =", _pcc(w_kn, W_ckhkw))
            print(
                "  DIAG PCC(sorted prep_weight, sorted Wref)=",
                _pcc(torch.sort(w_kn.reshape(-1))[0], torch.sort(W_khkwc.reshape(-1))[0]),
            )
        except Exception as e:
            print("  DIAG weight readback failed:", repr(e))

        # ---- localize the output permutation (values are correct per sorted-flat PCC) ----
        gold_mn = A_khkwc @ W_khkwc  # [M, N]
        # transpose check
        if dev.shape[0] == gold_mn.shape[1] or dev.numel() == gold_mn.numel():
            print("  DIAG PCC(device_flat, golden.T_flat)     =", _pcc(dev.reshape(-1), gold_mn.t().reshape(-1)))
        # per-row (M) multiset preserved?  -> only N within each row permuted
        row_ok = torch.isclose(torch.sort(dev, dim=1)[0], torch.sort(gold_mn, dim=1)[0], atol=0.1).all(dim=1)
        # per-col (N) multiset preserved?  -> only M within each col permuted
        col_ok = torch.isclose(torch.sort(dev, dim=0)[0], torch.sort(gold_mn, dim=0)[0], atol=0.1).all(dim=0)
        print(f"  DIAG rows(M) with matching multiset      = {int(row_ok.sum())}/{dev.shape[0]}")
        print(f"  DIAG cols(N) with matching multiset      = {int(col_ok.sum())}/{dev.shape[1]}")
        # eyeball first row / first col
        print("  DIAG device[0,:6] =", [round(float(x), 2) for x in dev[0, :6]])
        print("  DIAG golden[0,:6] =", [round(float(x), 2) for x in gold_mn[0, :6]])
        print("  DIAG device[:6,0] =", [round(float(x), 2) for x in dev[:6, 0]])
        print("  DIAG golden[:6,0] =", [round(float(x), 2) for x in gold_mn[:6, 0]])

    assert_with_pcc(torch_golden, tt_out.float(), pcc=PCC)


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_quasar_conv2d_split_program_e2e_pure(mesh_device):
    # Primary gate: pure conv (no bias/relu) — isolates tilize+matmul correctness.
    _run(mesh_device, with_bias_relu=False)


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_quasar_conv2d_split_program_e2e_bias_relu(mesh_device):
    # Bonus: bias + RELU folded into Program B's matmul.
    _run(mesh_device, with_bias_relu=True)


# Larger-shape sweep — stress the pieces that only bite at scale:
#   * wider N (out_channels)  -> more per_core_N tiles + wider in1 mcast (the mcast_in1 NOC path)
#   * larger K (in_ch / kernel) -> wider tilize block + bigger [full_K, N] weights (must stay K<=32 tiles for
#     the no-spill/split path; kQuasarConvNoSpillMaxKTiles)
#   * larger M (out_h*out_w)  -> more sharded cores / more mcast receivers
# act_block_h_override=None lets the factory pick (force_conv_no_spill caps act_block_h to fit the DFB ring).
_LARGER_SHAPES = [
    dict(in_channels=32, out_channels=128, kernel_size=(4, 4), out_h=16, out_w=32, act_block_h_override=None),  # N=4t
    dict(in_channels=32, out_channels=256, kernel_size=(4, 4), out_h=16, out_w=32, act_block_h_override=None),  # N=8t
    dict(
        in_channels=32, out_channels=64, kernel_size=(3, 3), out_h=16, out_w=32, act_block_h_override=None
    ),  # 3x3 K=9t
    dict(in_channels=64, out_channels=64, kernel_size=(3, 3), out_h=16, out_w=32, act_block_h_override=None),  # K=18t
    dict(in_channels=32, out_channels=64, kernel_size=(4, 4), out_h=32, out_w=32, act_block_h_override=None),  # M=1024
]
_LARGER_IDS = [
    "N128_k4x4",
    "N256_k4x4",
    "k3x3_K9",
    "inch64_k3x3_K18",
    "M1024_k4x4",
]


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("shape", _LARGER_SHAPES, ids=_LARGER_IDS)
@pytest.mark.parametrize("with_bias_relu", [False, True], ids=["pure", "bias_relu"])
def test_quasar_conv2d_split_program_e2e_shapes(mesh_device, shape, with_bias_relu):
    _run(mesh_device, with_bias_relu=with_bias_relu, **shape)
