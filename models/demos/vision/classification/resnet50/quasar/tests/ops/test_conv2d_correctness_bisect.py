# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Conv2d CORRECTNESS bisection for Quasar (independent of Option C / the tilize race).

test_conv2d_split_tilize_hypothesis.py surfaced a Quasar conv that runs to completion but is numerically
WRONG (PCC ~0.42), IDENTICAL on the fused and split compute kernels -> the bug is in the shared conv path
(halo / reader / matmul / pack / output), not the tilize split. This test localizes it by isolating each
stage across a small variant matrix, all fed an L1-sharded input (so they reach the Quasar compute instead
of dying in the unported DRAM-slicing sharded_to_interleaved). Run it on BOTH WH and Quasar and compare the
pass/fail matrix:

  variant              path exercised                              if WH passes & Quasar fails => bug in
  -------------------  ------------------------------------------  ------------------------------------
  mm_1x1               matmul / mm_conv (NO halo, NO tilize)       base matmul / pack / output / harness
  conv_3x3_p0_fidF     halo GATHER + tilize + K-spill              halo gather OR tilize OR spill
  conv_3x3_p0_fidT     halo GATHER + tilize + single K-block       full_inner_dim (vs fidF)
  conv_3x3_p1_fidF     halo ZERO-PAD gather + tilize               halo zero-pad (Quasar noc zero-write)

Reading it:
  - ALL variants fail on WH too            => the harness/golden/pre-shard is wrong (fix the test first).
  - mm_1x1 passes on Quasar, 3x3 all fail  => bug is in the halo+tilize conv path (not base matmul).
  - 3x3_p0_fidF passes, 3x3_p0_fidT fails  => full_inner_dim is wrong on Quasar.
  - 3x3_p0 passes, 3x3_p1 fails            => halo zero-pad (padding) is wrong on Quasar.
  - mm_1x1 also fails on Quasar            => base matmul/pack/output is wrong (deepest).

Run (WH reference):
  pytest models/demos/vision/classification/resnet50/quasar/tests/ops/test_conv2d_correctness_bisect.py
Run (craq-sim / emulator):
  TT_METAL_SIMULATOR=~/sim/libttsim.so TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 \
  pytest models/demos/vision/classification/resnet50/quasar/tests/ops/test_conv2d_correctness_bisect.py
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

PCC = 0.97


def _run(
    mesh_device,
    *,
    kernel,
    padding,
    full_inner_dim,
    math_fidelity,
    with_bias=True,
    with_relu=True,
    packer_l1_acc=True,
    reshard=True,
):
    device = mesh_device
    torch.manual_seed(0)

    batch_size = 1
    in_channels = 64
    out_channels = 64
    kh, kw = kernel
    stride = (1, 1)
    out_h = out_w = 16
    # s1: out = in + 2*pad - (k-1)  =>  in = out - 2*pad + (k-1)
    input_height = out_h - 2 * padding + (kh - 1)
    input_width = out_w - 2 * padding + (kw - 1)

    torch_input_nchw = torch.randn((batch_size, in_channels, input_height, input_width), dtype=torch.bfloat16).float()
    torch_weight = torch.randn((out_channels, in_channels, kh, kw), dtype=torch.bfloat16).float()
    torch_bias = torch.randn((1, 1, 1, out_channels), dtype=torch.bfloat16).float()
    torch_golden = torch.nn.functional.conv2d(
        torch_input_nchw,
        torch_weight,
        bias=(torch_bias.reshape(-1) if with_bias else None),
        stride=stride,
        padding=padding,
    )
    if with_relu:
        torch_golden = torch.relu(torch_golden)

    # pre-shard the activation into L1 (height-sharded) so conv2d takes the L1 path (not DRAM slicing)
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
    tt_input = ttnn.from_torch(flat, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT).to(device, in_mem)
    tt_weight = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16)
    tt_bias = ttnn.from_torch(torch_bias, dtype=ttnn.bfloat16)

    conv_kwargs = dict(
        weights_dtype=ttnn.bfloat16,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        full_inner_dim=full_inner_dim,
        reshard_if_not_optimal=reshard,
    )
    if with_relu:
        conv_kwargs["activation"] = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
    conv_config = ttnn.Conv2dConfig(**conv_kwargs)
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=math_fidelity, packer_l1_acc=packer_l1_acc
    )

    out, [oh, ow], [tt_weight, tt_bias] = ttnn.experimental.quasar.conv2d(
        input_tensor=tt_input,
        weight_tensor=tt_weight,
        bias_tensor=(tt_bias if with_bias else None),
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        kernel_size=kernel,
        stride=stride,
        padding=(padding, padding),
        dilation=(1, 1),
        groups=1,
        device=device,
        conv_config=conv_config,
        compute_config=compute_config,
        return_output_dim=True,
        return_weights_and_bias=True,
        dtype=ttnn.bfloat16,
    )

    tt_out = ttnn.to_torch(ttnn.from_device(out))
    tt_out = tt_out.reshape(batch_size, oh, ow, tt_out.shape[-1])[:, :, :, :out_channels]
    tt_out = torch.permute(tt_out, (0, 3, 1, 2))  # [1, C, oh, ow]
    _report_error_pattern(torch_golden, tt_out.float(), oh, ow, out_channels)
    assert_with_pcc(torch_golden, tt_out.float(), pcc=PCC)


def _pcc(a, b):
    a = a.flatten().double() - a.flatten().double().mean()
    b = b.flatten().double() - b.flatten().double().mean()
    denom = a.norm() * b.norm()
    return 1.0 if denom == 0 else float((a @ b) / denom)


def _report_error_pattern(golden, tt, oh, ow, c):
    """Localize where the conv error lives: per-output-row PCC (boundary rows bad => halo), per-channel
    PCC (specific channels bad => weight/pack), and worst-element location. Printed always (cheap)."""
    from loguru import logger

    g = golden[0]  # [C, oh, ow]
    t = tt[0]
    logger.info(f"[BISECT] overall PCC = {_pcc(g, t):.6f}  shape C={c} oh={oh} ow={ow}")
    # DECISIVE: are the VALUES right but misplaced (layout/placement bug) or genuinely wrong (arithmetic)?
    # sorted-PCC ~1.0 => the multiset of outputs is correct, only positions are scrambled (tilize face order
    # / pack tile-index / reader gather order). sorted-PCC also ~0.85 => the math itself is wrong.
    gs = g.flatten().double().sort().values
    ts = t.flatten().double().sort().values
    logger.info(f"[BISECT] sorted-values PCC = {_pcc(gs, ts):.6f}  (~1.0 => right values, WRONG PLACEMENT)")
    # histogram overlap of the sorted values (independent of PCC): max abs diff between the two sorted vectors
    logger.info(
        f"[BISECT] sorted max|g-t| = {float((gs - ts).abs().max()):.4f}  "
        f"mean|g-t| = {float((gs - ts).abs().mean()):.4f}  (small => same value distribution)"
    )
    # per output row (spatial H): halo bugs hit shard-boundary / first-last rows
    row_pcc = [round(_pcc(g[:, r, :], t[:, r, :]), 4) for r in range(oh)]
    logger.info(f"[BISECT] per-oh-row PCC (len {oh}): {row_pcc}")
    # per channel: pack / weight-column bugs hit specific channels
    ch_pcc = [round(_pcc(g[ch], t[ch]), 4) for ch in range(min(c, 64))]
    logger.info(f"[BISECT] per-channel PCC (first {len(ch_pcc)}): {ch_pcc}")
    # worst elements
    err = (g - t).abs()
    flat = err.flatten()
    k = min(8, flat.numel())
    top = torch.topk(flat, k)
    locs = [tuple(int(x) for x in torch.unravel_index(i, err.shape)) for i in top.indices]  # (C, oh, ow)
    logger.info(f"[BISECT] top-{k} abs-err {[round(float(v),3) for v in top.values]} at (C,row,col) {locs}")
    logger.info(f"[BISECT] golden absmax={float(g.abs().max()):.3f} tt absmax={float(t.abs().max()):.3f}")


@pytest.mark.timeout(600)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
# PER-ELEMENT-STAGE discriminator. The window-shape sweep (k3x3/k3x1/k1x3/k2x2) all gave ~0.85 regardless
# of tap count => the corruption is NOT in the reduction (gather/weight-K/matmul-K would scale with taps).
# It's a per-element stage that runs the same for every conv: input reshard, output pack accumulation, bias,
# or RELU. Toggle each on a fixed 3x3 conv; whichever flip lifts PCC to ~1.0 is the culprit. If none do, the
# bug is in the core matmul_block / conv tilize (values). (Quasar output absmax > golden => extra/stale term.)
@pytest.mark.parametrize(
    "label, with_bias, with_relu, packer_l1_acc, reshard",
    [
        pytest.param("base", True, True, True, True, id="base"),
        pytest.param("no_l1acc", True, True, False, True, id="no_l1acc"),
        pytest.param("no_bias", False, True, True, True, id="no_bias"),
        pytest.param("no_relu", True, False, True, True, id="no_relu"),
        pytest.param("no_reshard", True, True, True, False, id="no_reshard"),
        pytest.param("bare", False, False, False, True, id="bare_matmul_only"),
    ],
)
def test_conv2d_correctness_bisect(mesh_device, label, with_bias, with_relu, packer_l1_acc, reshard):
    # WH passes all of these. Quasar 3x3 base = 0.8547, values wrong (sorted-PCC too), uniform, fidelity- AND
    # window-shape-independent. `bare` = matmul only (no bias/relu/l1acc) isolates the core matmul+tilize+pack.
    _run(
        mesh_device,
        kernel=(3, 3),
        padding=0,
        full_inner_dim=False,
        math_fidelity=ttnn.MathFidelity.LoFi,
        with_bias=with_bias,
        with_relu=with_relu,
        packer_l1_acc=packer_l1_acc,
        reshard=reshard,
    )
