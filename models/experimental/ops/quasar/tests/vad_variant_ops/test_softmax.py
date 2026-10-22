# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.softmax`` — deformable-attention weight softmax (dim=-1) and the
moreh "h_large" softmax over 10000 BEV rows (dim=-2).

Shape source (kernel cache):
  * dim=-1: ``softmax`` (attention-optimized, rank 4, ``NUMERIC_STABLE``) +
    ``reader_unary_interleaved_sm`` + ``writer_unary_interleaved_start_id_blocked_sm``. The
    writer's ct[0] = padded - logical width => W mod 32 is cache-determined: 4 (DRAM out),
    8 (L1 out), 16 (DRAM and L1 out) [H]. These match levels x points of the deformable
    attentions: TSA 1x4, SCA 1x8, and a 16-wide one. Row count is inferred [M/L].
  * dim=-2: ``moreh_softmax_h_large`` + reader/writer (``softmax_program_factory_general_h_large``):
    compute ct ``[cols_per_core, Ht=313]`` => H = 10000 [H]; width <= 20 tiles; L1 [M].
Default softmax compute config (HiFi4, approx on, fp32 acc off) [H].
"""

import pytest
import torch

import ttnn
from models.experimental.ops.quasar.tests.vad_variant_ops import op_utils as U

Q, H = U.NUM_QUERY, U.NUM_HEADS

# fmt: off
CASES = [
    pytest.param((1, Q, H, U.TSA_POINTS), -1, "dram", "dram", id="N4-tsa-w4-122883"),
    pytest.param((1, Q, H, U.SCA_POINTS), -1, "l1", "l1", id="N5-sca-w8-141177"),
    pytest.param((1, 300, H, 16), -1, "dram", "dram", id="N6-w16-dram-143724"),
    pytest.param((1, 300, H, 16), -1, "l1", "l1", id="N6-w16-l1-554236"),
    pytest.param((1, 1, Q, 32), -2, "l1", "l1", id="N7-h10000-dim-2-557004"),
]
# fmt: on


@U.with_default_mesh()
@pytest.mark.parametrize("shape, dim, in_mem, out_mem", CASES)
def test_softmax(ttnn_mesh_device, reset_seeds, shape, dim, in_mem, out_mem):
    mesh = ttnn_mesh_device
    x_torch = U.torch_rand(shape)
    x = U.to_tt_mem(x_torch, mesh, in_mem)
    out = ttnn.softmax(x, dim=dim, memory_config=U.resolve_memcfg(out_mem, mesh), numeric_stable=True)
    U.assert_pcc(torch.softmax(x_torch.float(), dim=dim), out, pcc=0.999, mesh_device=mesh)
