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

Program A runs the DATACOPY tilize (per-tile FPU dest-dvalid clear = the 0x19 fix), NOT UnpackToDestEn — the
test sets TT_METAL_QSR_CONV_SPLIT_PROGRAM itself and deliberately DROPS TT_METAL_QSR_TILIZE_UNPACK_TO_DEST
(the batched UNP_DEST tilize intermittently hangs on Quasar).

Run (emulator / WH, slow dispatch + forced JIT; the split env is set inside the test):
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
    skip_known_quasar_fail=True,  # gap tests set False to actually exercise the known-fail combos
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
    n_tiles = _math.ceil(out_channels / 32)
    print(
        f"  DIAG shape: in_ch={in_channels} out_ch={out_channels} k={kernel_size} out=({out_h},{out_w}) "
        f"K_tiles={k_tiles} N_tiles={n_tiles}"
    )
    # KNOWN Quasar limitation: the FUSED-bias matmul (Program B) hangs at program completion for wide N
    # (out_channels >= 256, N >= 8 tiles); pure wide-N and bias with N <= 4 pass. Needs a matmul-kernel fix
    # (the fused-bias wide-N epilogue). Skip that combo on Quasar so the sweep is green; WH/BH run it all.
    _is_quasar = "QUASAR" in str(device.arch()).upper()
    if skip_known_quasar_fail and with_bias_relu and _is_quasar and n_tiles > 4:
        import pytest as _pytest

        _pytest.skip(
            f"Quasar fused-bias wide-N (out_channels={out_channels}, N={n_tiles} tiles) matmul hang — TODO kernel fix"
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

    # Two-program split: Program A tilize + Program B matmul.
    # Program A uses the DATACOPY tilize (per-tile FPU dest-dvalid clear = the 0x19 fix in tilize.h), NOT
    # UnpackToDestEn. On WH datacopy is the only path anyway; on Quasar we explicitly DROP
    # TT_METAL_QSR_TILIZE_UNPACK_TO_DEST so tilize_block takes the datacopy branch. The batched UNP_DEST tilize
    # intermittently HANGS on Quasar (pack frozen inside tilize_block mid DEST-dvalid handshake, ~block 4 —
    # dprint_spe1 / utd10-12); the datacopy path is what validated on WH.
    saved = {k: os.environ.get(k) for k in ("TT_METAL_QSR_CONV_SPLIT_PROGRAM", "TT_METAL_QSR_TILIZE_UNPACK_TO_DEST")}
    os.environ["TT_METAL_QSR_CONV_SPLIT_PROGRAM"] = "1"
    os.environ.pop("TT_METAL_QSR_TILIZE_UNPACK_TO_DEST", None)
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

        # ---- localize a partial (PCC~0.5) failure: per-M-shard (core) and per-N-tile PCC ----
        # If one core's rows are correct and the other's are garbage -> per-core Program-B issue (mcast/weights).
        M = dev.shape[0]
        for frac, lbl in [(4, "quarter"), (2, "half")]:
            step = M // frac
            if step == 0:
                continue
            segs = " ".join(
                f"[{i*step}:{(i+1)*step}]={_pcc(dev[i*step:(i+1)*step], gold_mn[i*step:(i+1)*step]):.3f}"
                for i in range(frac)
            )
            print(f"  DIAG PCC by M-{lbl}: {segs}")
        Ncols = dev.shape[1]
        if Ncols >= 64:
            print(
                f"  DIAG PCC by N-tile: [0:32]={_pcc(dev[:, :32], gold_mn[:, :32]):.3f} "
                f"[32:64]={_pcc(dev[:, 32:64], gold_mn[:, 32:64]):.3f}"
            )

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
# act_block_h_override=128 (4 tiles) bounds the reader no-spill gather to <=4 M-tiles/block — REQUIRED on the
# Quasar emulator (read_activation_data mis-reads beyond ~4 M-tiles in one gather). Harmless on WH/BH.
_LARGER_SHAPES = [
    dict(in_channels=32, out_channels=128, kernel_size=(4, 4), out_h=16, out_w=32, act_block_h_override=128),  # N=4t
    dict(in_channels=32, out_channels=256, kernel_size=(4, 4), out_h=16, out_w=32, act_block_h_override=128),  # N=8t
    dict(in_channels=32, out_channels=64, kernel_size=(3, 3), out_h=16, out_w=32, act_block_h_override=128),  # 3x3 K=9t
    dict(in_channels=64, out_channels=64, kernel_size=(3, 3), out_h=16, out_w=32, act_block_h_override=128),  # K=18t
    dict(in_channels=32, out_channels=64, kernel_size=(4, 4), out_h=32, out_w=32, act_block_h_override=128),  # M=1024
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


# ============================================================================
# GAP TESTS — combos the split conv does NOT yet handle on the Quasar emulator.
# Each PASSES on WH/BH and FAILS (hang/assert/PCC) on Quasar today; they are the
# standalone repros to drive to green so the full ResNet model can run on Quasar.
# (These bypass the sweep's known-fail skip via skip_known_quasar_fail=False.)
# ============================================================================


# Deep-reduction K (tiles) sweep for the no-spill/split path. K = in_channels*3*3/32.
#   K36 (C=128, layer2 3x3): fixed by raising kQuasarConvNoSpillMaxKTiles to 64 -> should PASS.
#   K72 (C=256, layer3 3x3): K > 64 -> force_conv_no_spill does NOT fire (act_block_w != full_K) -> overrun/hang,
#     AND even if the limit were raised the single full-K K-block (72 tiles) may not fit L1 -> OOM. This test
#     pins WHICH it is (assert/overrun/hang vs allocation error), to decide raise-limit vs K-spill vs more-L1.
#   K144 (C=512, layer4 3x3) is deferred (certain OOM at one K-block).
_DEEP_K_CASES = [(128, "K36_layer2"), (256, "K72_layer3")]


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("in_channels,cid", _DEEP_K_CASES, ids=[c[1] for c in _DEEP_K_CASES])
def test_quasar_gap_deep_k_over_32(mesh_device, in_channels, cid):
    """GAP 1 — deep reduction K > 32 tiles (ResNet layer2-4 3x3 convs, in_channels >= 128).

    force_conv_no_spill only fires (sets act_block_w = full_K) when K <= kQuasarConvNoSpillMaxKTiles; above it
    the split runs with act_block_w = window-row while the reader gathers full_K -> ACT-CB overrun / tilize
    starvation hang (K36 hung on WH+Quasar at block 1 before the limit was raised 32->64).
    FIX DIRECTION: raise the limit + single-K-block matmul/tilize (L1 fit), OR K-spill without the accumulate.
    """
    _run(
        mesh_device,
        with_bias_relu=False,
        in_channels=in_channels,
        out_channels=64,
        kernel_size=(3, 3),
        out_h=8,
        out_w=16,  # small spatial to minimize L1 (act CB = act_block_h*full_K, weights = full_K*N)
        act_block_h_override=32,  # 1 tile -> <=4 M-tiles/gather (isolate the K gap from the reader gap)
        skip_known_quasar_fail=False,
    )


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_quasar_gap_wide_n_fused_bias(mesh_device):
    """GAP 2 — wide N (out_channels >= 256, N >= 8 tiles) + FUSED bias.

    Program B's fused-bias matmul HANGS at program completion for wide N (pure wide-N and bias N<=4 pass).
    Shrinking the M block (out_block_h=1) avoids the hang but TRANSPOSES the output on Quasar
    (a multi-M-block output-order bug). FIX DIRECTION: the fused-bias wide-N epilogue in
    bmm_large_block_zm_fused_bias_activation_metal2 (compute-kernel DPRINT is no-op'd there), and/or the
    out_block_h<per_core_M transpose so N can be blocked.
    """
    _run(
        mesh_device,
        with_bias_relu=True,
        out_channels=256,  # N = 8 tiles
        kernel_size=(4, 4),
        out_h=16,
        out_w=32,
        act_block_h_override=128,
        skip_known_quasar_fail=False,
    )


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_quasar_gap_large_m_reader_gather(mesh_device):
    """GAP 3 — large per-core M with the reader gather UNCAPPED (act_block_h_override=None).

    The no-spill full-window gather (read_activation_data / host reader-indices) mis-reads output rows
    beyond ~4 M-tiles in a single gather -> M-tiles >= 4 come out wrong (readback PCC by M-quarter
    1,0,1,0). We currently paper over it by capping act_block_h <= 4 tiles. Here act_block_h_override=None
    lets force_conv_no_spill pick a larger act_block_h (per_core_M is 16 tiles for M=1024 / 2 cores),
    re-exposing the bug. FIX DIRECTION: the reader-indices generation / read_activation_data for
    act_block_h > 4 tiles, so the cap can be dropped.
    """
    _run(
        mesh_device,
        with_bias_relu=False,
        out_channels=64,
        kernel_size=(4, 4),
        out_h=32,
        out_w=32,  # M = 1024 sticks -> large per-core M
        act_block_h_override=None,  # UNCAPPED -> gather > 4 M-tiles
        skip_known_quasar_fail=False,
    )
