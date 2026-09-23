# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.typecast``.

Model call sites:
  * model.py:897              x = ttnn.typecast(x, activation_dtype)          # activation dtype cast
  * attention_1d.py:489/497  ttnn.typecast(..., dtype=ttnn.bfloat16)         # q/k pre-rotary -> bf16
  * attention_1d.py:507/510  ttnn.typecast(..., dtype=keys.dtype)            # -> KV cache dtype
  * attention_1d.py:541      ttnn.typecast(..., dtype=cfg.activation_dtype or ttnn.bfloat8_b)
  * attention_1d.py:1160     ttnn.typecast(output, cfg.prefill_reduce_ccl_dtype)

typecast is a value-preserving dtype conversion (NOT a no-op even when src==tgt,
per the comment at attention_1d.py:486). Reference is ``x.to(dtype)``. For casts
to a lower-precision target (bf16, bfloat8_b) PCC is 0.99; for the bf16->fp32
widening it is 0.999.
"""

import pytest
import torch

import ttnn
from models.experimental.llama32_1b_quasar.tests.prototype_ops import op_utils as U

# (id, src_dtype, tgt_dtype, pcc)
_CAST_SITES = [
    ("bf16_to_fp32", ttnn.bfloat16, ttnn.float32, 0.999),  # widening -> exact
    ("fp32_to_bf16", ttnn.float32, ttnn.bfloat16, 0.99),  # attention q/k -> bf16
    ("bf16_to_bfp8", ttnn.bfloat16, ttnn.bfloat8_b, 0.99),  # activation / cache dtype
]


@U.with_default_mesh()
@pytest.mark.parametrize(
    "name, src_dtype, tgt_dtype, pcc",
    [pytest.param(*s, id=s[0]) for s in _CAST_SITES],
)
def test_typecast(ttnn_mesh_device, reset_seeds, name, src_dtype, tgt_dtype, pcc):
    mesh = ttnn_mesh_device

    x_torch = U.torch_rand((1, 1, U.MAX_BATCH, U.DIM), dtype=torch.float32)
    x = U.to_tt(x_torch, mesh, dtype=src_dtype)

    out = ttnn.experimental.quasar.typecast(x, dtype=tgt_dtype)
    assert out.dtype == tgt_dtype, f"expected dtype {tgt_dtype}, got {out.dtype}"

    ref = x_torch.float()
    U.assert_pcc(ref, out, pcc=pcc, mesh_device=mesh)
