# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.linear`` — the BEV encoder linears / FFN, the can_bus MLP, the
spatial-cross-attention projections, and the ResNet/FPN 1x1 convs (which conv2d lowers
to ``ttnn.linear``; ``conv2d.cpp:340-366``).

Shape source (kernel cache): compute ``bmm_large_block_zm_fused_bias_activation`` + readers
``reader_bmm_tile_layout_in0_sender_padding`` / ``..._in0_sender_receiver_padding_block_sharded``
/ ``..._in0_receiver`` + ``reader_bmm_tile_layout_in1_{sender,receiver}_writer_padding``.
Compute ct = ``[in0_block_w, in0_num_subblocks, in0_block_num_tiles, ..., in1_per_core_w,
num_blocks, ..., out_subblock_h, out_subblock_w, ...]`` (factory ``matmul_..._1d/2d``);
the in0 sender carries Kt and ``last_ktile_w`` (=> logical K), the sharded tensor
accessors carry shard shapes and dtypes. So per-core blocking, K, N, dtypes, shard
specs, fused bias / activation and compute config are cache-determined [H]; logical M of
interleaved inputs is known only to the tile [M].

Compute-config fingerprint (from the factory defaults):
  * HiFi2 + approx off (+ packer L1 acc): no program config — ttnn picked it  -> ``AUTO_CASES``
  * LoFi + approx off: explicit program config                              -> ``PC_CASES``
  * LoFi + approx on + L1 acc off, sharded, bias: a 1x1 conv (TtConv2D cfg)  -> ``CONV1X1_CASES``

Program configs target the model's 5x4 grid (``compute_with_storage_grid_size=(5, 4)``).
Ids carry the compute-variant hash prefix.
"""

import pytest
import torch

import ttnn
from models.experimental.ops.quasar.tests.vad_variant_ops import op_utils as U

Q = U.NUM_QUERY
LOFI = ttnn.MathFidelity.LoFi
RELU = "relu"
GELU = "gelu"


def _pc1d(in0_block_w, sbh, sbw, per_core_m, per_core_n, fuse_batch=True, mcast_in0=False, act=None):
    return ("1d", in0_block_w, sbh, sbw, per_core_m, per_core_n, fuse_batch, mcast_in0, act)


def _pc2d(in0_block_w, sbh, sbw, per_core_m, per_core_n, act=None):
    return ("2d", in0_block_w, sbh, sbw, per_core_m, per_core_n, True, False, act)


def _activation(act):
    if act is None:
        return None
    return ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU if act == RELU else ttnn.UnaryOpType.GELU)


def _program_config(spec, mesh):
    U.model_grid(mesh)  # skip when the device can't host the 5x4 program config
    kind, bw, sbh, sbw, pm, pn, fuse_batch, mcast_in0, act = spec
    common = dict(
        compute_with_storage_grid_size=(U.MODEL_GRID_X, U.MODEL_GRID_Y),
        in0_block_w=bw,
        out_subblock_h=sbh,
        out_subblock_w=sbw,
        out_block_h=pm,
        out_block_w=pn,
        per_core_M=pm,
        per_core_N=pn,
        fused_activation=_activation(act),
    )
    if kind == "1d":
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(**common, fuse_batch=fuse_batch, mcast_in0=mcast_in0)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(**common, transpose_mcast=True)


def _run_linear(
    mesh,
    a_shape,
    a_dt,
    a_mem,
    n,
    w_dt,
    bias,
    out_dt,
    out_mem,
    *,
    pc=None,
    fidelity=None,
    approx=False,
    fp32=False,
    l1_acc=False,
    act=None,
    w_mem="dram",
):
    k = a_shape[-1]
    a_torch = U.torch_rand(a_shape, dtype=torch.float32 if a_dt == "fp32" else torch.bfloat16)
    w_torch = U.torch_rand((k, n), std=k**-0.5)
    b_torch = U.torch_rand((1, n)) if bias else None

    a = U.to_tt_mem(a_torch, mesh, a_mem, dtype=U.DTYPES[a_dt])
    w = U.to_tt_mem(w_torch, mesh, w_mem, dtype=U.DTYPES[w_dt])
    b = U.to_tt(b_torch, mesh, dtype=ttnn.bfloat8_b) if bias else None

    kwargs = dict(bias=b, memory_config=U.resolve_memcfg(out_mem, mesh), dtype=U.DTYPES[out_dt])
    if pc is not None:
        kwargs["program_config"] = _program_config(pc, mesh)
        act = pc[-1]
    elif act is not None:
        kwargs["activation"] = act
    if fidelity is not None:
        kwargs["compute_kernel_config"] = ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=fidelity, math_approx_mode=approx, fp32_dest_acc_en=fp32, packer_l1_acc=l1_acc
        )
    out = ttnn.linear(a, w, **kwargs)

    # Reference on the (bfp8-quantized) values the device saw.
    a_seen = U.from_tt(a, mesh).reshape(-1)[: a_torch.numel()].reshape(a_shape)
    w_seen = U.from_tt(w, mesh).reshape(-1)[: w_torch.numel()].reshape(k, n)
    ref = a_seen @ w_seen
    if bias:
        ref = ref + U.from_tt(b, mesh).reshape(-1)[:n]
    if act == RELU:
        ref = torch.relu(ref)
    elif act == GELU:
        ref = torch.nn.functional.gelu(ref)

    low_precision = "bfp8" in (a_dt, out_dt) or fidelity == LOFI
    U.assert_pcc(ref, out, pcc=0.99 if low_precision else 0.999, mesh_device=mesh)


# fmt: off
# --- explicit program config, LoFi (BEV encoder, height-sharded 20 cores x 512 rows) ---
# (a_shape, a_dt, a_mem, n, w_dt, bias, out_dt, out_mem, pc, l1_acc)
PC_CASES = [
    pytest.param((1, 1, Q, 64), "bf16", ("hs", (512, 64)), 32, "bfp8", True, "bf16", ("hs", (512, 32)), _pc1d(2, 4, 1, 16, 1), True, id="L2-64to32-133343"),
    pytest.param((1, 1, Q, 512), "bf16", ("hs", (512, 512)), 64, "bfp8", True, "bf16", ("hs", (512, 64)), _pc1d(16, 4, 2, 16, 2, act=GELU), True, id="L3-512to64-gelu-837964"),
    pytest.param((1, 1, Q, 256), "bf16", ("hs", (512, 256)), 256, "bfp8", True, "bf16", ("hs", (512, 256)), _pc1d(4, 1, 8, 16, 8), True, id="L6-256to256-l1acc-230573"),
    pytest.param((1, 1, Q, 256), "bf16", ("hs", (512, 256)), 256, "bfp8", True, "bf16", ("hs", (512, 256)), _pc1d(8, 1, 8, 16, 8), False, id="L7-256to256-145533"),
    pytest.param((1, 1, Q, 256), "bf16", ("hs", (512, 256)), 512, "bfp8", True, "bf16", ("hs", (512, 512)), _pc1d(8, 1, 8, 16, 16, act=RELU), False, id="L8-ffn-fc1-relu-105187"),
    pytest.param((1, 1, Q, 512), "bf16", ("hs", (512, 512)), 256, "bfp8", True, "bf16", ("hs", (512, 256)), _pc1d(16, 1, 8, 16, 8), False, id="L9-ffn-fc2-164039"),
    # Temporal value projection over the 2-frame BEV queue (626 tiles fused), bfp8 DRAM out.
    pytest.param((2, 1, Q, 256), "bf16", ("hs", (1024, 256)), 256, "bfp8", True, "bfp8", "dram", _pc1d(8, 1, 1, 32, 8), False, id="L10-tsa-value-607524"),
]

# --- ttnn-chosen config (HiFi2, approx off, packer L1 acc) ---
# (a_shape, a_dt, a_mem, n, w_dt, w_mem, bias, out_dt, out_mem)
AUTO_CASES = [
    pytest.param((1, Q, 512), "bf16", "dram", 64, "bfp8", "dram", True, "bf16", "dram", id="L4-512to64-232251"),
    pytest.param((1, Q, 512), "bf16", "dram", 128, "bfp8", "dram", True, "bf16", "dram", id="L5-512to128-918311"),
    # SCA per-camera-group projections (rebatched query counts 3072/3680, batch 2 x 992/1376).
    pytest.param((1, 1, 3072, 256), "bf16", "l1", 64, "bfp8", "dram", False, "bf16", "l1", id="L11a-q3072-to64-442633"),
    pytest.param((1, 1, 3072, 256), "bf16", "l1", 128, "bfp8", "dram", False, "bf16", "l1", id="L11b-q3072-to128-751689"),
    pytest.param((1, 1, 3680, 256), "bf16", "l1", 64, "bfp8", "dram", False, "bf16", "l1", id="L11c-q3680-to64-616238"),
    pytest.param((1, 1, 3680, 256), "bf16", "l1", 128, "bfp8", "dram", False, "bf16", "l1", id="L11d-q3680-to128-778054"),
    pytest.param((2, 1, 992, 256), "bf16", "l1", 64, "bfp8", "dram", False, "bf16", "l1", id="L11e-2xq992-to64-737131"),
    pytest.param((2, 1, 992, 256), "bf16", "l1", 128, "bfp8", "dram", False, "bf16", "l1", id="L11f-2xq992-to128-101458"),
    pytest.param((2, 1, 1376, 256), "bf16", "l1", 64, "bfp8", "dram", False, "bf16", "l1", id="L11g-2xq1376-to64-150136"),
    pytest.param((2, 1, 1376, 256), "bf16", "l1", 128, "bfp8", "dram", False, "bf16", "l1", id="L11h-2xq1376-to128-110294"),
    # K = 10000 contraction over the BEV grid; in1 bf16 in L1. M <= 32 (assumed 32).
    pytest.param((1, 1, 32, Q), "bf16", "l1", 512, "bf16", "l1", False, "bf16", "dram", id="L12-k10000-587205"),
    pytest.param((1, 1, 32, 256), "bf16", "dram", 256, "bfp8", "dram", True, "bf16", "dram", id="L13-256to256-small-m-993269"),
    pytest.param((1, 1, 32, 256), "bf16", "dram", 32, "bfp8", "dram", True, "bf16", "dram", id="L14-256to32-small-m-993269"),
    pytest.param((1, 1, 32, 256), "bf16", "l1", 512, "bfp8", "dram", False, "bf16", "l1", id="L15-256to512-144483"),
    # can_bus MLP (fp32): 18 -> 128 -> 256.
    pytest.param((1, 1, 1, 18), "fp32", "dram", 128, "bfp8", "dram", True, "fp32", "dram", id="L16-canbus-18to128-131776"),
    pytest.param((1, 1, 1, 128), "fp32", "dram", 256, "bfp8", "dram", True, "fp32", "dram", id="L17-canbus-128to256-120677"),
]

# --- ResNet-50 / FPN 1x1 convs lowered to linear (bias, LoFi, approx on, L1 acc off) ---
# (a_shape, a_dt, a_mem, n, out_dt, out_mem, pc)
CONV1X1_CASES = [
    pytest.param((1, 1, 15360, 64), "bf16", ("hs", (768, 64)), 64, "bfp8", ("hs", (768, 64)), _pc1d(2, 4, 2, 24, 2, act=RELU), id="C1-layer1.0.conv1-226904"),
    pytest.param((1, 1, 15360, 256), "bf16", ("hs", (768, 256)), 64, "bfp8", ("hs", (768, 64)), _pc1d(8, 4, 2, 24, 2, act=RELU), id="C2-layer1.conv1-900817"),
    pytest.param((1, 1, 15360, 256), "bf16", ("hs", (768, 256)), 128, "bfp8", ("hs", (768, 128)), _pc1d(8, 2, 4, 24, 4, act=RELU), id="C3-layer2.0.conv1-126255"),
    pytest.param((1, 1, 15360, 64), "bfp8", ("hs", (768, 64)), 256, "bfp8", ("hs", (768, 256)), _pc1d(2, 1, 8, 24, 8), id="C4-layer1.conv3-847055"),
    pytest.param((1, 1, 15360, 64), "bf16", ("hs", (768, 64)), 256, "bf16", ("hs", (768, 256)), _pc1d(2, 1, 8, 24, 8), id="C5-layer1.0.downsample-399932"),
    pytest.param((1, 1, 3840, 128), "bfp8", ("hs", (192, 128)), 512, "bfp8", ("hs", (192, 512)), _pc1d(4, 1, 8, 6, 16), id="C6-layer2.conv3-775239"),
    pytest.param((1, 1, 3840, 512), "bfp8", ("hs", (192, 512)), 128, "bfp8", ("hs", (192, 128)), _pc1d(16, 2, 4, 6, 4, act=RELU), id="C7-layer2.conv1-183439"),
    pytest.param((1, 1, 3840, 512), "bfp8", ("hs", (192, 512)), 256, "bfp8", ("hs", (192, 256)), _pc1d(16, 1, 8, 6, 8, act=RELU), id="C8-layer3.0.conv1-713887"),
    pytest.param((1, 1, 1920, 256), "bfp8", ("hs", (96, 256)), 1024, "bfp8", ("hs", (96, 1024)), _pc1d(8, 1, 8, 3, 32), id="C9-layer3.conv3-hs-327941"),
    pytest.param((1, 1, 1920, 1024), "bfp8", ("hs", (96, 1024)), 256, "bfp8", ("hs", (96, 256)), _pc1d(32, 1, 8, 3, 8, act=RELU), id="C10-layer3.conv1-165489"),
    pytest.param((1, 1, 960, 256), "bfp8", ("bs", (192, 64)), 1024, "bfp8", ("bs", (192, 256)), _pc2d(2, 1, 8, 6, 8), id="C11-layer3.conv3-bs-253993"),
    pytest.param((1, 1, 1920, 1024), "bfp8", ("bs", (384, 256)), 512, "bf16", ("bs", (384, 128)), _pc2d(8, 2, 4, 12, 4, act=RELU), id="C12-layer4.0.conv1-230257"),
    pytest.param((1, 1, 480, 2048), "bf16", ("bs", (96, 512)), 512, "bf16", ("bs", (96, 128)), _pc2d(16, 1, 4, 3, 4, act=RELU), id="C13-layer4.conv1-128259"),
    pytest.param((1, 1, 480, 512), "bf16", ("bs", (96, 128)), 2048, "bf16", ("bs", (96, 512)), _pc2d(4, 1, 8, 3, 16), id="C14-layer4.conv3-182057"),
    # FPN lateral 1x1 over all 6 cameras' 12x20 maps (6 x 240 = 1440 rows).
    pytest.param((1, 1, 1440, 2048), "bf16", ("bs", (288, 512)), 256, "bf16", ("bs", (288, 64)), _pc2d(16, 3, 2, 9, 2), id="C15-fpn-lateral-731454"),
]
# fmt: on


@pytest.mark.blackhole_scale
@U.with_default_mesh()
@pytest.mark.parametrize("a_shape, a_dt, a_mem, n, w_dt, bias, out_dt, out_mem, pc, l1_acc", PC_CASES)
def test_linear_program_config(
    ttnn_mesh_device, reset_seeds, a_shape, a_dt, a_mem, n, w_dt, bias, out_dt, out_mem, pc, l1_acc
):
    _run_linear(
        ttnn_mesh_device, a_shape, a_dt, a_mem, n, w_dt, bias, out_dt, out_mem, pc=pc, fidelity=LOFI, l1_acc=l1_acc
    )


@U.with_default_mesh()
@pytest.mark.parametrize("a_shape, a_dt, a_mem, n, w_dt, w_mem, bias, out_dt, out_mem", AUTO_CASES)
def test_linear(ttnn_mesh_device, reset_seeds, a_shape, a_dt, a_mem, n, w_dt, w_mem, bias, out_dt, out_mem):
    # No program / compute config: ttnn's defaults gave the cached HiFi2 (+ fp32 acc for fp32 out).
    _run_linear(ttnn_mesh_device, a_shape, a_dt, a_mem, n, w_dt, bias, out_dt, out_mem, w_mem=w_mem)


@pytest.mark.blackhole_scale
@U.with_default_mesh()
@pytest.mark.parametrize("a_shape, a_dt, a_mem, n, out_dt, out_mem, pc", CONV1X1_CASES)
def test_linear_conv1x1(ttnn_mesh_device, reset_seeds, a_shape, a_dt, a_mem, n, out_dt, out_mem, pc):
    _run_linear(
        ttnn_mesh_device, a_shape, a_dt, a_mem, n, "bfp8", True, out_dt, out_mem, pc=pc, fidelity=LOFI, approx=True
    )
