# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Step 4 of the 3D-neighborhood generalization: the ``backend="op"`` NA3D executor.

``neighborhood_attention_3d(..., backend="op")`` runs the SDPA op's on-device ``neighborhood_3d``
mask instead of the grouped gather + dense masked attention. This holds it against the same host
reference the gather backend uses (``na3d_torch``), and against the gather backend directly, so the
two executors are interchangeable at the block level. Grids are kept small: the op leaves the
K-range full (step 2), so its compute is O(S^2) until step 3 narrows it.
"""

from __future__ import annotations

import pytest
import torch

import ttnn

from ...layers.na3d import build_device_plan, na3d_torch, neighborhood_attention_3d, plan_na3d
from ...models.vae.diffvae_ltx import NABlock, default_rope_dim_split, rope_tables
from ...utils.check import assert_quality

# Real DiffVAE kernels ((3,7,7) stages 1/2, (3,5,5) stages 3/4, (3,3,3)/(11,11,11) elsewhere) on
# small grids covering interior + both boundary regimes and the axis-shorter-than-kernel case.
CASES = [
    ((3, 7, 7), (3, 7, 7)),  # exact fit
    ((2, 5, 5), (3, 7, 7)),  # every axis shorter than the kernel
    ((6, 9, 9), (3, 7, 7)),  # interior plus both boundary regimes
    ((8, 12, 6), (3, 5, 5)),  # non-cubic stage-3/4 kernel
    ((4, 8, 8), (3, 3, 3)),  # small cubic kernel
]


@pytest.mark.parametrize("dims, kernel", CASES)
@pytest.mark.parametrize("heads, head_dim", [(4, 64)])
def test_na3d_op_backend_matches_host(*, device, dims, kernel, heads, head_dim):
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, *dims, heads, head_dim, dtype=torch.float32) for _ in range(3))

    expected = na3d_torch(q, k, v, kernel, scale=1.0).reshape(1, *dims, heads * head_dim)

    tt_q, tt_k, tt_v = (
        ttnn.from_torch(x, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT) for x in (q, k, v)
    )
    actual = neighborhood_attention_3d(tt_q, tt_k, tt_v, kernel_size=kernel, scale=1.0, backend="op")

    assert tuple(actual.shape) == tuple(expected.shape), f"{tuple(actual.shape)} != {tuple(expected.shape)}"
    assert_quality(expected, ttnn.to_torch(actual), pcc=0.999)


def test_na3d_op_backend_matches_gather(*, device):
    """The two executors agree on device, so a block can swap backends without a parity shift."""
    torch.manual_seed(0)
    dims, kernel, heads, head_dim = (6, 9, 9), (3, 7, 7), 4, 64
    q, k, v = (torch.randn(1, *dims, heads, head_dim, dtype=torch.float32) for _ in range(3))

    tt = lambda x: ttnn.from_torch(x, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    plan = plan_na3d(dims, kernel)
    device_plan = build_device_plan(plan, mesh_device=device, dtype=ttnn.bfloat16)
    gather = ttnn.to_torch(
        neighborhood_attention_3d(tt(q), tt(k), tt(v), kernel_size=kernel, scale=1.0, device_plan=device_plan)
    )
    op = ttnn.to_torch(neighborhood_attention_3d(tt(q), tt(k), tt(v), kernel_size=kernel, scale=1.0, backend="op"))

    assert_quality(gather, op, pcc=0.999)


def test_na_block_op_matches_gather(*, device):
    """A whole DiffVAE NA block (norm + qkv + q/k-norm + RoPE + NA3D + proj + SwiGLU) gives the
    same result with either NA3D backend, so a stage can select ``na3d_backend="op"`` without a
    parity shift. Small geometry with random weights, mirroring the row-chunking exactness test.
    """
    torch.manual_seed(0)
    dim, head_dim, kernel = 128, 64, (3, 3, 3)
    hidden = (int(dim * 4.0) + 15) // 16 * 16
    dims = (5, 8, 7)
    tokens = dims[0] * dims[1] * dims[2]

    weights = {
        "norm1.weight": (dim,),
        "norm2.weight": (dim,),
        "attn.qkv.weight": (3 * dim, dim),
        "attn.qkv.bias": (3 * dim,),
        "attn.proj.weight": (dim, dim),
        "attn.proj.bias": (dim,),
        "attn.q_norm.weight": (head_dim,),
        "attn.k_norm.weight": (head_dim,),
        "mlp.w_gate.weight": (hidden, dim),
        "mlp.w_up.weight": (hidden, dim),
        "mlp.w_down.weight": (dim, hidden),
    }
    state = {name: torch.randn(shape) * 0.1 for name, shape in weights.items()}
    hidden_states = torch.randn(tokens, dim)

    cos, sin = rope_tables(dims, default_rope_dim_split(head_dim), mesh_device=device)
    plan = build_device_plan(plan_na3d(dims, kernel), mesh_device=device)

    def run(backend: str) -> torch.Tensor:
        block = NABlock(dim, kernel, head_dim=head_dim, mesh_device=device, na3d_backend=backend)
        block.load_state_dict({key: value.clone() for key, value in state.items()})
        tt_hidden = ttnn.from_torch(hidden_states, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        return ttnn.to_torch(block(tt_hidden, dims=dims, cos=cos, sin=sin, device_plan=plan))

    assert_quality(run("gather"), run("op"), pcc=0.999)
