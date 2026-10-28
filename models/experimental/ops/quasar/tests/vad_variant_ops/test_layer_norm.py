# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.layer_norm`` — BEV encoder / decoder norms and the can_bus norm.

Shape source (kernel cache): ``layernorm`` compute + ``reader_unary_interleaved_ln`` +
``writer_unary_interleaved_start_id_blocked``. Ct args map via
``layernorm_op_multi_core.cpp:393-417,502-512`` (fork revision ``6b9b6a0e0e2~1``):
compute ``[Wt, block, has_gamma, has_beta, fp32_dest, ...]`` => W is cache-determined [H].
  * N1  bf16, W=256 (embed), input L1 or DRAM, output L1  — BEV encoder norms  [H W / M rows]
  * N2  bf16, W=512, output L1                                                 [H W / L rows]
  * N3  fp32, W=256, DRAM in/out — the can_bus MLP norm                         [H W / M rows]
gamma/beta are bfloat8_b TILE in DRAM [H]. Default compute config (HiFi4, fp32 dest acc,
approx off) [H]. Epsilon is a runtime arg (lost); 1e-5 (PyTorch default) is used.
"""

import pytest
import torch

import ttnn
from models.experimental.ops.quasar.tests.vad_variant_ops import op_utils as U

EPS = 1e-5

# fmt: off
CASES = [
    pytest.param((1, U.NUM_QUERY, U.EMBED), "bf16", "l1", "l1", id="N1-bev-l1-210765"),
    pytest.param((1, U.NUM_QUERY, U.EMBED), "bf16", "dram", "l1", id="N1-bev-dram-210765"),
    pytest.param((1, 300, 2 * U.EMBED), "bf16", "l1", "l1", id="N2-w512-631029"),
    pytest.param((1, 1, U.EMBED), "fp32", "dram", "dram", id="N3-canbus-fp32-170177"),
]
# fmt: on


@U.with_default_mesh()
@pytest.mark.parametrize("shape, dt, in_mem, out_mem", CASES)
def test_layer_norm(ttnn_mesh_device, reset_seeds, shape, dt, in_mem, out_mem):
    mesh = ttnn_mesh_device
    w = shape[-1]
    torch_dtype = torch.float32 if dt == "fp32" else torch.bfloat16
    x_torch = U.torch_rand(shape, dtype=torch_dtype)
    gamma = U.torch_rand((1, 1, 1, w))
    beta = U.torch_rand((1, 1, 1, w))

    x = U.to_tt_mem(x_torch, mesh, in_mem, dtype=U.DTYPES[dt])
    tt_gamma = U.to_tt(gamma, mesh, dtype=ttnn.bfloat8_b)
    tt_beta = U.to_tt(beta, mesh, dtype=ttnn.bfloat8_b)
    out = ttnn.layer_norm(x, epsilon=EPS, weight=tt_gamma, bias=tt_beta, memory_config=U.resolve_memcfg(out_mem, mesh))

    # Reference uses the bfp8-quantized gamma/beta the device saw.
    g = U.from_tt(tt_gamma, mesh).reshape(-1)[:w]
    b = U.from_tt(tt_beta, mesh).reshape(-1)[:w]
    ref = torch.nn.functional.layer_norm(x_torch.float(), (w,), g, b, EPS)
    U.assert_pcc(ref, out, pcc=0.999, mesh_device=mesh)
