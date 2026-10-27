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
