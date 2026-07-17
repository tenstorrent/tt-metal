# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
TILIZE ISOLATION (Option 1) — read back the tilized activation and compare to a host golden, with NO matmul.

Why: the fused UnpackToDestEn conv (test_conv2d_unpack_to_dest.py) runs end-to-end but PCC ~= 0.001. The
per-block DPRINT probes (TZINIT / TZL1) proved the batched tilize-to-DEST INDEX MATH is exactly correct
(l1idx = y*FULL_CT*fpe, sst=1, srcz=2 dstz=64 soff0=8 — all match the LLK reference test). So the scramble is
NOT the index computation. It must be one of: (a) the UNP_DEST tilize MOP data production, (b) the packer
reading tile j out of DEST, or (c) the MATMUL consuming act_tilized. This test splits (a)+(b) from (c).

How: run ONLY the gather+tilize half (Program A). TT_METAL_QSR_CONV_SPLIT_PROGRAM=1 makes the factory select
conv_tilize_only_metal2.cpp; TT_METAL_QSR_TILIZE_UNPACK_TO_DEST=1 injects QSR_TILIZE_UNPACK_TO_DEST so it runs
the SAME batched UNP_DEST tilize as the fused kernel; TT_METAL_QSR_CONV_TILIZE_ONLY_NO_MATMUL=1 makes conv2d.cpp
STOP after Program A and RETURN the tilized activation [M, full_K] itself (skipping the Program B matmul it
would otherwise chain). OUT is borrowed_from the output tensor, so to_torch(out) untilizes the tilized
activation back to the row-major im2col matrix A[M, K].

Shape = the proven-routing 4x4 / in_ch=32 stem-like conv (same one test_conv2d_split_program.py routes to
tilize-only), so no new-shape routing surprises. K = 4*4*32 = 512 = 16 tiles; each tilize block is act_block_w =
in_ch*kw/32 = 4 tiles wide — the IDENTICAL batched-tilize block width the failing fused test hits.

Golden without guessing the K column order: for output position m = (oi, oj), the reader gathers the receptive
field input[:, oi:oi+4, oj:oj+4] — exactly 4*4*32 = 512 values, no K padding. The tilize only REARRANGES those
512 values within row m; it must not move values BETWEEN rows. So:
  * sorted-row PCC (sort each row's 512 values, compare) is INVARIANT to K column order and only drops if the
    tilize scrambles values across rows — i.e. a real intra-tile FACE scramble.
  * exact-order PCC (vs the two likely K flattenings, [kh,kw,c] and [c,kh,kw]) pins whether the column order
    also matches the weights' flattening.

Verdict:
  * sorted-row PCC ~1  AND some exact-order PCC ~1 -> tilize is CORRECT; the fused PCC~0 is the MATMUL / K-order.
  * sorted-row PCC ~1  AND both exact-order PCC low -> rows preserved but K column order != either candidate:
    a reader-vs-weights K-ORDERING mismatch, NOT a tilize scramble.
  * sorted-row PCC low -> the UNP_DEST tilize MOP / pack-from-DEST SCRAMBLES data across rows -> LLK (#49445);
    hand to /debug-kernel with the confirmed strides.

Run (craq-sim / emulator, slow dispatch + forced JIT). NOTE: rebuild first — conv2d.cpp changed.
  TT_METAL_QSR_CONV_SPLIT_PROGRAM=1 TT_METAL_QSR_TILIZE_UNPACK_TO_DEST=1 TT_METAL_QSR_CONV_TILIZE_ONLY_NO_MATMUL=1 \
  TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 \
  pytest -s models/demos/vision/classification/resnet50/quasar/tests/ops/test_conv2d_tilize_readback.py
"""

import os

import pytest
import torch

import ttnn

PCC = 0.99


def _pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    if torch.allclose(a, b):
        return 1.0
    va, vb = a - a.mean(), b - b.mean()
    denom = (va.norm() * vb.norm()).item()
    if denom == 0:
        return 0.0
    return (torch.dot(va, vb).item()) / denom


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_quasar_conv2d_tilize_readback(mesh_device):
    device = mesh_device
    torch.manual_seed(0)

    # Proven tilize-only-routing shape (== test_conv2d_split_program.py): 4x4 / s1 / p0, in=32, K=512=16 tiles.
    batch_size = 1
    in_channels = 32
    out_channels = 64  # unused by Program A (no matmul); kept for API shape inference
    kernel_size = (4, 4)
    stride = (1, 1)
    padding = (0, 0)
    out_h, out_w = 16, 32  # M = 512 = 16 tiles
    kh, kw = kernel_size
    input_height = out_h + kh - 1  # 19
    input_width = out_w + kw - 1  # 35

    torch_input_nchw = torch.randn((batch_size, in_channels, input_height, input_width), dtype=torch.bfloat16).float()
    torch_weight = torch.randn((out_channels, in_channels, *kernel_size), dtype=torch.bfloat16).float()

    # ---- host goldens (order-independent multiset, plus two exact-order candidates) ----
    m_total = out_h * out_w
    k_total = in_channels * kh * kw  # 512
    golden_sorted = torch.empty((m_total, k_total), dtype=torch.float32)
    golden_khkwc = torch.empty((m_total, k_total), dtype=torch.float32)  # K order [kh][kw][c]
    golden_ckhkw = torch.empty((m_total, k_total), dtype=torch.float32)  # K order [c][kh][kw]
    for oi in range(out_h):
        for oj in range(out_w):
            m = oi * out_w + oj
            win = torch_input_nchw[0, :, oi : oi + kh, oj : oj + kw]  # [c, kh, kw]
            golden_sorted[m] = torch.sort(win.reshape(-1))[0]
            golden_khkwc[m] = win.permute(1, 2, 0).reshape(-1)  # [kh, kw, c]
            golden_ckhkw[m] = win.reshape(-1)  # [c, kh, kw]

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

    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        full_inner_dim=True,  # single K-block (in0_num_blocks_w == 1) — the split-program factory gate
        act_block_h_override=128,  # 4-tile height blocks => >= 2 height blocks: exercises the multi-block tilize
        reshard_if_not_optimal=True,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    # Program A only (tilize-only), STOP before the Program B matmul so the op returns the tilized activation.
    # NO TT_METAL_QSR_TILIZE_UNPACK_TO_DEST -> uses the DATACOPY (MOVA2D) tilize path with the per-tile FPU
    # dest-dvalid clear (the 0x19 fix), NOT the racy UNP_DEST path. Restore envs after so they don't leak.
    to_set = {
        "TT_METAL_QSR_CONV_SPLIT_PROGRAM": "1",
        "TT_METAL_QSR_CONV_TILIZE_ONLY_NO_MATMUL": "1",
    }
    prev = {k: os.environ.get(k) for k in to_set}
    os.environ.update(to_set)
    try:
        out, [oh, ow], _wb = ttnn.experimental.quasar.conv2d(
            input_tensor=tt_input,
            weight_tensor=tt_weight,
            bias_tensor=None,
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
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    # OUT borrowed from the output tensor => this IS the tilized activation. to_torch untilizes to row-major.
    tt_out = ttnn.to_torch(ttnn.from_device(out)).float()
    tt_flat = tt_out.reshape(-1, tt_out.shape[-1])
    print(
        f"tilize-only readback: raw shape={tuple(tt_out.shape)} flat={tuple(tt_flat.shape)} (expected M={m_total} K={k_total})"
    )

    assert tt_flat.shape[0] >= m_total, f"readback rows {tt_flat.shape[0]} < M {m_total}"
    assert tt_flat.shape[1] >= k_total, f"readback width {tt_flat.shape[1]} < K {k_total} (got {tt_flat.shape[1]})"
    tt_A = tt_flat[:m_total, :k_total]

    sorted_pcc = _pcc(torch.sort(tt_A, dim=1)[0], golden_sorted)
    pcc_khkwc = _pcc(tt_A, golden_khkwc)
    pcc_ckhkw = _pcc(tt_A, golden_ckhkw)
    row_match = torch.isclose(torch.sort(tt_A, dim=1)[0], golden_sorted, atol=0.05).all(dim=1)
    print(
        f"TILIZE ISOLATION RESULTS:\n"
        f"  sorted-row PCC (rows preserved / no cross-row scramble) = {sorted_pcc:.4f}\n"
        f"  exact-order PCC [kh,kw,c]                               = {pcc_khkwc:.4f}\n"
        f"  exact-order PCC [c,kh,kw]                               = {pcc_ckhkw:.4f}\n"
        f"  rows with matching value-multiset                       = {int(row_match.sum())}/{m_total}\n"
        f"  => sorted~1 & some exact~1: tilize CORRECT (fused PCC~0 is matmul/K-order)\n"
        f"  => sorted~1 & exact low : reader-vs-weights K-ORDER mismatch (not a tilize scramble)\n"
        f"  => sorted low           : intra-tile FACE scramble (tilize MOP / pack-from-DEST bug)"
    )

    # Primary assert: rows must be preserved (order-independent). This is the decisive tilize-correctness signal.
    assert sorted_pcc >= PCC, (
        f"tilize moves values BETWEEN rows (sorted-row PCC {sorted_pcc:.4f}) => intra-tile face scramble in the "
        f"UNP_DEST tilize / pack-from-DEST path"
    )
