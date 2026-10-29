# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""ttnn.experimental.quasar.matmul with MX (OCP microscaling) operands and outputs (issue #62742).

The unpacker expands every MX format to Float16_b in the Src registers, so the FPU multiplies bf16-class values
and the partial sums stay Float16_b / Float32; only an MX output is packed to MX. MXFP4 is the exception: matmul
unpacks it as the 2x-packed src format (MxFp4_2x_B), which needs both operands in MXFP4 and the unpacker / ALU
restored (mm_uninit) before the partials reload and the bias broadcast.

Run on the Quasar simulator:
    source <sim>/env.sh && pytest tests/ttnn/nightly/unit_tests/operations/experimental/quasar/test_matmul_mx.py
"""

import pytest
import torch

import ttnn
from tests.ttnn.nightly.unit_tests.operations.experimental.quasar.mx_test_utils import (
    check,
    mx_round_trip,
    pcc_floor,
    skip_unless_quasar,
    to_device,
)

qsr = ttnn.experimental.quasar
qsr_cpp = ttnn._ttnn.operations.experimental.quasar  # program-config types live on the C++ module
TILE = 32

MX_NO_FP4 = [d for d in ttnn.MX_DTYPES if d != ttnn.mxfp4]


@pytest.fixture(autouse=True)
def _quasar_only(device):
    skip_unless_quasar(device)


def _inputs(M, K, N, seed=0):
    torch.manual_seed(seed)
    return torch.randn(1, 1, M, K), torch.randn(1, 1, K, N)


def _golden(a, b, in0_dtype, in1_dtype, out_dtype):
    qa = mx_round_trip(a, in0_dtype)
    qb = mx_round_trip(b, in1_dtype)
    return mx_round_trip(qa @ qb, out_dtype)


def _run(device, a, b, in0_dtype, in1_dtype, out_dtype, **kwargs):
    out = qsr.matmul(to_device(device, a, in0_dtype), to_device(device, b, in1_dtype), dtype=out_dtype, **kwargs)
    assert out.dtype == out_dtype
    return ttnn.to_torch(out)


@pytest.mark.parametrize("dtype", MX_NO_FP4)
def test_mx_in_bf16_out(device, dtype):
    a, b = _inputs(2 * TILE, 2 * TILE, 2 * TILE)
    out = _run(device, a, b, dtype, dtype, ttnn.bfloat16)
    check(_golden(a, b, dtype, dtype, ttnn.bfloat16), out, pcc_floor(dtype))


@pytest.mark.parametrize("dtype", MX_NO_FP4)
def test_mx_in_mx_out(device, dtype):
    a, b = _inputs(2 * TILE, 2 * TILE, 2 * TILE)
    out = _run(device, a, b, dtype, dtype, dtype)
    check(_golden(a, b, dtype, dtype, dtype), out, pcc_floor(dtype))


@pytest.mark.parametrize("dtype", ttnn.MX_DTYPES)
def test_bf16_in_mx_out(device, dtype):
    # The MX output is produced by the packer; it must decode on the host like host-packed MX.
    a, b = _inputs(2 * TILE, 2 * TILE, 2 * TILE)
    out = _run(device, a, b, ttnn.bfloat16, ttnn.bfloat16, dtype)
    check(_golden(a, b, ttnn.bfloat16, ttnn.bfloat16, dtype), out, pcc_floor(dtype))


@pytest.mark.parametrize("dtype", [ttnn.mxfp8_e4m3, ttnn.mxint8])
def test_mixed_mx_and_bf16_operands(device, dtype):
    a, b = _inputs(2 * TILE, 2 * TILE, 2 * TILE)
    out = _run(device, a, b, dtype, ttnn.bfloat16, ttnn.bfloat16)
    check(_golden(a, b, dtype, ttnn.bfloat16, ttnn.bfloat16), out, pcc_floor(dtype))


@pytest.mark.parametrize("dtype", MX_NO_FP4)
def test_partials_reload(device, dtype):
    # K_chunk_tiles=1 with four K tiles: every K step after the first reloads the Float16_b partials into Dest
    # (copy_block) before the next matmul, the path that needs mm_uninit for MXFP4 and Float16_b partials for an
    # MX output.
    a, b = _inputs(2 * TILE, 4 * TILE, 2 * TILE)
    config = qsr_cpp.MatmulUnifiedProgramConfig(K_chunk_tiles=1)
    out = _run(device, a, b, dtype, dtype, dtype, program_config=config)
    check(_golden(a, b, dtype, dtype, dtype), out, pcc_floor(dtype))


@pytest.mark.parametrize("dtype", [ttnn.mxfp8_e4m3, ttnn.mxint8])
def test_fused_bias(device, dtype):
    a, b = _inputs(2 * TILE, 2 * TILE, 2 * TILE)
    bias = torch.randn(1, 1, 1, 2 * TILE)
    out = qsr.linear(
        to_device(device, a, dtype),
        to_device(device, b, dtype),
        bias=to_device(device, bias, ttnn.bfloat16),
        dtype=ttnn.bfloat16,
    )
    golden = mx_round_trip(a, dtype) @ mx_round_trip(b, dtype) + bias.to(torch.bfloat16).to(torch.float32)
    check(golden.to(torch.bfloat16), ttnn.to_torch(out), pcc_floor(dtype))


# MXFP4 takes the 2x-packed src-register path (FP4 x FP4 on the FPU, one fidelity phase).
@pytest.mark.parametrize("K_chunk_tiles", [None, 1], ids=["single_chunk", "reload_partials"])
def test_mxfp4_pair(device, K_chunk_tiles):
    a, b = _inputs(2 * TILE, 2 * TILE, 2 * TILE)
    kwargs = (
        {"program_config": qsr_cpp.MatmulUnifiedProgramConfig(K_chunk_tiles=K_chunk_tiles)} if K_chunk_tiles else {}
    )
    out = _run(device, a, b, ttnn.mxfp4, ttnn.mxfp4, ttnn.bfloat16, **kwargs)
    check(_golden(a, b, ttnn.mxfp4, ttnn.mxfp4, ttnn.bfloat16), out, pcc_floor(ttnn.mxfp4))


def test_mxfp4_needs_both_operands(device, expect_error):
    a, b = _inputs(2 * TILE, 2 * TILE, 2 * TILE)
    with expect_error(RuntimeError, "MXFP4 matmul needs both operands"):
        _run(device, a, b, ttnn.mxfp4, ttnn.bfloat16, ttnn.bfloat16)


def test_mx_rejects_transpose(device, expect_error):
    a, b = _inputs(2 * TILE, 2 * TILE, 2 * TILE)
    with expect_error(RuntimeError, "not supported with MX operands"):
        _run(device, a, b.transpose(-2, -1).contiguous(), ttnn.mxint8, ttnn.mxint8, ttnn.bfloat16, transpose_b=True)


def test_mx_rejects_ragged_k(device, expect_error):
    a, b = _inputs(2 * TILE, TILE + 16, 2 * TILE)
    with expect_error(RuntimeError, "K to be a tile multiple"):
        _run(device, a, b, ttnn.mxint8, ttnn.mxint8, ttnn.bfloat16)


# ----------------------------------------------------------------------------------------------------
# Legacy Quasar factories (explicit program configs): 2D multicast, 1D in0- and in1-multicast, reuse (batched).
# Their rings hold tiles at the tile's own size: the unpacker walks MX tiles at the hardware stride (1056 B for an
# 8-bit MX tile), not the 64 B DRAM-aligned page stride the WH / BH factories pad to.
# ----------------------------------------------------------------------------------------------------


def _legacy(family, in0_block_w):
    """(M, K, N, batch, program_config) for a two-K-block (in0_block_w=2) run of each legacy factory."""
    if family == "mcast_2d":
        return (
            4 * TILE,
            4 * TILE,
            4 * TILE,
            1,
            qsr_cpp.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(2, 2),
                in0_block_w=in0_block_w,
                out_subblock_h=1,
                out_subblock_w=2,
                out_block_h=2,
                out_block_w=2,
                per_core_M=2,
                per_core_N=2,
                transpose_mcast=False,
                fused_activation=None,
            ),
        )
    if family == "mcast_1d_in0":
        return (
            TILE,
            4 * TILE,
            8 * TILE,
            1,
            qsr_cpp.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(4, 1),
                in0_block_w=in0_block_w,
                out_subblock_h=1,
                out_subblock_w=2,
                out_block_h=1,
                out_block_w=2,
                per_core_M=1,
                per_core_N=2,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            ),
        )
    if family == "mcast_1d_in1":
        return (
            4 * TILE,
            4 * TILE,
            2 * TILE,
            1,
            qsr_cpp.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(4, 1),
                in0_block_w=in0_block_w,
                out_subblock_h=1,
                out_subblock_w=2,
                out_block_h=1,
                out_block_w=2,
                per_core_M=1,
                per_core_N=2,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=False,
            ),
        )
    assert family == "reuse"
    return (
        TILE,
        4 * TILE,
        2 * TILE,
        4,
        qsr_cpp.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(4, 1),
            in0_block_w=in0_block_w,
            out_subblock_h=1,
            out_subblock_w=2,
            per_core_M=1,
            per_core_N=2,
        ),
    )


LEGACY_FAMILIES = ["mcast_2d", "mcast_1d_in0", "mcast_1d_in1", "reuse"]


def _compute_config(device, fp32_dest_acc_en=False, packer_l1_acc=False):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=packer_l1_acc,
        dst_full_sync_en=False,
    )


def _run_legacy(device, family, in0_dtype, in1_dtype, out_dtype, in0_block_w=2, bias=False, **compute):
    grid = device.compute_with_storage_grid_size()
    if grid.x < 4 or grid.y < 2:
        pytest.skip(f"needs a 4x2 grid, device has {grid.x}x{grid.y}")
    M, K, N, batch, config = _legacy(family, in0_block_w)
    torch.manual_seed(0)
    a, b = torch.randn(1, batch, M, K), torch.randn(1, batch, K, N)
    kwargs = dict(program_config=config, dtype=out_dtype, compute_kernel_config=_compute_config(device, **compute))
    golden = mx_round_trip(a, in0_dtype) @ mx_round_trip(b, in1_dtype)
    if bias:
        bias_t = torch.randn(1, 1, 1, N)
        out = qsr.linear(
            to_device(device, a, in0_dtype),
            to_device(device, b, in1_dtype),
            bias=to_device(device, bias_t, ttnn.bfloat16),
            **kwargs,
        )
        golden = golden + bias_t.to(torch.bfloat16).to(torch.float32)
    else:
        out = qsr.matmul(to_device(device, a, in0_dtype), to_device(device, b, in1_dtype), **kwargs)
    assert out.dtype == out_dtype
    check(mx_round_trip(golden, out_dtype), ttnn.to_torch(out), pcc_floor(in0_dtype, in1_dtype, out_dtype))


@pytest.mark.parametrize("family", LEGACY_FAMILIES)
@pytest.mark.parametrize("dtype", [ttnn.mxint8, ttnn.mxfp8_e4m3])
def test_legacy_mx_in_mx_out(device, family, dtype):
    # Two K blocks: the partials spill to L1 and reload into Dest between them.
    _run_legacy(device, family, dtype, dtype, dtype)


@pytest.mark.parametrize("family", LEGACY_FAMILIES)
def test_legacy_mx_in_bf16_out(device, family):
    _run_legacy(device, family, ttnn.mxint8, ttnn.mxint8, ttnn.bfloat16)


@pytest.mark.parametrize("family", LEGACY_FAMILIES)
def test_legacy_bf16_in_mx_out(device, family):
    _run_legacy(device, family, ttnn.bfloat16, ttnn.bfloat16, ttnn.mxint8)


@pytest.mark.parametrize("family", LEGACY_FAMILIES)
def test_legacy_mxfp4_pair(device, family):
    _run_legacy(device, family, ttnn.mxfp4, ttnn.mxfp4, ttnn.bfloat16)


@pytest.mark.parametrize("family", [f for f in LEGACY_FAMILIES if f != "reuse"])  # reuse (bmm) has no bias
def test_legacy_fused_bias(device, family):
    _run_legacy(device, family, ttnn.mxint8, ttnn.mxint8, ttnn.bfloat16, bias=True)


@pytest.mark.parametrize("family", LEGACY_FAMILIES)
def test_legacy_packer_l1_acc(device, family):
    # Four K blocks accumulated by the packer in a Float16_b partials ring, then packed to MX.
    _run_legacy(device, family, ttnn.mxint8, ttnn.mxint8, ttnn.mxint8, in0_block_w=1, packer_l1_acc=True)


@pytest.mark.parametrize("family", LEGACY_FAMILIES)
def test_legacy_fp32_dest_acc(device, family):
    pytest.skip(
        "fp32_dest_acc_en fails host validation on every legacy Quasar factory for bf16 too: the compute KernelSpec "
        "gives no unpack_modes entry for the Float32 partials DFB (hardware_config.cpp dfbs_with_entry)"
    )
    _run_legacy(device, family, ttnn.mxint8, ttnn.mxint8, ttnn.mxint8, fp32_dest_acc_en=True)


@pytest.mark.parametrize("mode", ["packer_l1_acc", "fp32_dest_acc"])
def test_unified_accumulation_modes_mx_out(device, mode):
    a, b = _inputs(2 * TILE, 4 * TILE, 2 * TILE)
    compute = {"packer_l1_acc": True} if mode == "packer_l1_acc" else {"fp32_dest_acc_en": True}
    out = _run(
        device,
        a,
        b,
        ttnn.mxint8,
        ttnn.mxint8,
        ttnn.mxint8,
        program_config=qsr_cpp.MatmulUnifiedProgramConfig(K_chunk_tiles=1),
        compute_kernel_config=_compute_config(device, **compute),
    )
    check(_golden(a, b, ttnn.mxint8, ttnn.mxint8, ttnn.mxint8), out, pcc_floor(ttnn.mxint8))


def test_dram_sharded_factories_unreachable(device):
    pytest.skip(
        "MatmulMultiCoreReuseMultiCastDRAMSharded / BatchedDRAMSharded throw 'Unsupported architecture for DRAM "
        "sharded matmul' on Quasar for every dtype (matmul_utilities.cpp noc_max_page_size), so MX cannot reach them"
    )
