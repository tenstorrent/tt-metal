# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.atan2`` — the OFT encoder's angle decode, ``atan2(sin, cos)`` on the two
angle-offset channels of the head output.

Shape source (kernel cache): ``ttnn.atan2`` is a composite. Its pieces all appear in the cache
at 25 tiles over 20 cores, i.e. [1, 159, 159] bf16 TILE, L1 interleaved in and out [H]:
reciprocal, SFPU mul, atan, gtz / ltz / eqz, ``ge(y, 0)``, 5x ``logical_and``, ``+/- pi``,
and ``where`` in tensor-scalar-scalar / tensor-scalar-tensor / tensor-tensor-tensor forms.
Each piece is also tested on its own (``test_unary.py``, ``test_comparison.py``,
``test_where.py``, ``test_binary.py``).

Second cache (in-tree OFT at ``628a4797bf7``): ``atan2`` is native, a single binary_ng op
``eltwise_binary_sfpu_no_bcast/453912`` (``atan2_binary_tile``, bf16, HiFi4, approx off, no fp32
dest acc). Its reader is ``reader_interleaved_no_bcast/179439`` with both operands L1 interleaved,
and the writer is L1. The operands are the ``angle_offsets[:, 0]`` / ``[:, 1]`` tile slices,
[1, 159, 159] [H op/mem, M shape]. None of the composite's pieces are in that cache.

Current main (and ``ttnn.experimental.quasar.atan2``) lowers ``ttnn.atan2`` to that same native
binary_ng ``ATAN2`` op, so this case is the cache-#2 form. Cache #1's composite pieces are
tested on their own in the files listed above. Quasar's ``atan2`` SFPU kernel landed in
``9a2c0e18337``; ``QUASAR_LLK_GAPS.md`` still lists it as missing, but that doc is stale.

``atan`` is a polynomial approximation, so this checks PCC plus a max-abs bound.
"""

import pytest
import torch

import ttnn
from models.experimental.ops.quasar.tests.oft_ops import op_utils as U


@U.with_default_mesh()
@pytest.mark.parametrize("shape", [pytest.param(U.HEATMAP, id="atan2-sin-cos-1x159x159-453912")])
def test_atan2(ttnn_mesh_device, reset_seeds, shape):
    mesh = ttnn_mesh_device
    theta = (torch.rand(shape) * 2 - 1) * torch.pi
    # The head regresses (sin, cos) only approximately; scale them so |v| != 1.
    scale = 0.5 + torch.rand(shape)
    sin, cos = (theta.sin() * scale).to(torch.bfloat16), (theta.cos() * scale).to(torch.bfloat16)
    # Exercise the special cases the composite handles: axis points and the origin.
    sin.view(-1)[:4] = torch.tensor([0.0, 0.0, 1.0, -1.0], dtype=torch.bfloat16)
    cos.view(-1)[:4] = torch.tensor([1.0, -1.0, 0.0, 0.0], dtype=torch.bfloat16)

    out = U.op("atan2")(U.to_tt_mem(sin, mesh, "l1"), U.to_tt_mem(cos, mesh, "l1"), memory_config=ttnn.L1_MEMORY_CONFIG)

    ref = torch.atan2(sin.float(), cos.float())
    got = U.from_tt(out, mesh).reshape(-1)[: ref.numel()].reshape(ref.shape)
    U.assert_pcc(ref, host=got, pcc=0.999)
    # Away from the +/-pi branch cut, the angle must be close in absolute terms.
    away = ref.abs() < 3.0
    err = (got - ref).abs()[away].max().item()
    assert err < 0.05, f"atan2 max |error| {err:.4f} rad"
