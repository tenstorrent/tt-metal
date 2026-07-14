# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Precision matrix for eltwise_chain.

The functional suites are bf16-dominant. This sweeps a fidelity-sensitive chain (out = A * B, FPU
multiply) over input/output dtype x fp32_dest_acc_en x math_fidelity. Each config is asserted
against its OWN per-(dtype,fidelity) threshold — never one config vs another (relative-ordering
asserts flip on sub-1e-3 noise). Thresholds reflect inherent precision, not bugs: LoFi truncates FPU
source mantissa bits and bfloat8_b quantizes inputs, so their PCC floors are lower by design.
"""

import torch
import pytest
import ttnn
from loguru import logger
from tests.ttnn.utils_for_testing import comp_pcc
import tests.ttnn.unit_tests.kernel_lib.chain_test_lib as lib

KERNEL = "ttnn/cpp/ttnn/kernel_lib/tests/axes/precision_mul.cpp"

DTYPES = [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b]
FIDELITIES = [ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi4]
DTYPE_ID = {ttnn.bfloat16: "bf16", ttnn.float32: "fp32", ttnn.bfloat8_b: "bfp8"}
FID_ID = {ttnn.MathFidelity.LoFi: "LoFi", ttnn.MathFidelity.HiFi4: "HiFi4"}


def _threshold(dtype, fidelity):
    """Per-config PCC floor for A*B. LoFi (mantissa-truncated multiply) and bfloat8_b (quantized
    inputs) are inherently lower precision — not bugs."""
    if dtype == ttnn.bfloat8_b:
        return 0.98
    if fidelity == ttnn.MathFidelity.LoFi:
        return 0.99  # LoFi multiply on bf16/fp32 sources
    if dtype == ttnn.float32:
        return 0.9995
    return 0.9995  # bf16 HiFi4


@pytest.mark.parametrize("dtype", DTYPES, ids=[DTYPE_ID[d] for d in DTYPES])
@pytest.mark.parametrize("fp32_dest_acc_en", [False, True], ids=["acc_off", "acc_on"])
@pytest.mark.parametrize("fidelity", FIDELITIES, ids=[FID_ID[f] for f in FIDELITIES])
def test_precision_mul(device, dtype, fp32_dest_acc_en, fidelity):
    n = 8
    shape = [1, 1, 32, 32 * n]
    cg = lib.single_core_grid()

    # Moderate magnitudes so the FPU multiply doesn't over/underflow bfp8/bf16 range.
    ta, tt_a = lib.make_input(shape, dtype, device, seed=1501, scale=0.6, bias=0.2)
    tb, tt_b = lib.make_input(shape, dtype, device, seed=1502, scale=0.6, bias=0.2)
    tt_out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dtype, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)

    cbs = [lib.cb_descriptor(i, dtype, 2, cg) for i in (0, 1)] + [lib.cb_descriptor(16, dtype, 2, cg)]
    reader = lib.build_reader_kernel([tt_a, tt_b], n, cg)
    writer = lib.build_writer_1out_kernel(tt_out, n, cg)
    compute = lib.build_compute_kernel(KERNEL, [n], cg, fp32_dest_acc_en=fp32_dest_acc_en, math_fidelity=fidelity)

    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    output = ttnn.generic_op([tt_a, tt_b, tt_out], program)

    golden = ta.to(torch.float32) * tb.to(torch.float32)
    out = ttnn.to_torch(output).to(torch.float32)
    thr = _threshold(dtype, fidelity)
    pcc_ok, msg = comp_pcc(golden, out, thr)
    logger.info(
        f"precision A*B | {DTYPE_ID[dtype]:4s} acc={int(fp32_dest_acc_en)} {FID_ID[fidelity]:5s} | "
        f"thr={thr} | {msg}"
    )
    assert pcc_ok, f"{DTYPE_ID[dtype]} acc={fp32_dest_acc_en} {FID_ID[fidelity]}: {msg}"
