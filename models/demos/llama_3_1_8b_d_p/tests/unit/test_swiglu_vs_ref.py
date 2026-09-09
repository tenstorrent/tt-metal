# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The activation alone, at this model's exact variant and constants.

Structure follows `minimax_m3/tests/unit/test_swiglu_vs_ref.py`, but the variant is the whole point
of the test. The donor's activation is clamped **swigluoai** — `gate.clamp(max=limit)`, a 1.702
alpha inside the sigmoid, and a `+1` on the up branch. Llama's is plain SiLU SwiGLU:

    silu(gate) * up      where silu(x) = x * sigmoid(x)

no clamp, no alpha, no offset. Both donors on this engine use the OAI variant, so this is precisely
the place a borrowed MLP goes wrong while still producing plausible numbers, and it is checked on
its own before the surrounding matmuls can mask it.

There is no `ttnn.swiglu`; the activation is composed from `ttnn.silu` and `ttnn.multiply`, which is
the right answer rather than a workaround.
"""

import torch

import ttnn
from models.demos.llama_3_1_8b_d_p.reference.model import REF_DTYPE

from ..test_factory import ACT_DTYPE, assert_pcc, parametrize_target_mesh

SEQ = 512
INTERMEDIATE_LOCAL = 3584  # 14336 / tp 4 — the activation runs on the per-chip shard


@parametrize_target_mesh()
def test_swiglu_vs_ref(mesh_device, device_params, topology_name):
    """silu(gate) * up on device vs the torch reference, on the per-chip intermediate shard."""
    torch.manual_seed(0)
    gate = torch.randn(1, 1, SEQ, INTERMEDIATE_LOCAL, dtype=REF_DTYPE)
    up = torch.randn(1, 1, SEQ, INTERMEDIATE_LOCAL, dtype=REF_DTYPE)
    golden = torch.nn.functional.silu(gate) * up

    def to_dev(t):
        return ttnn.from_torch(
            t,
            device=mesh_device,
            dtype=ACT_DTYPE,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    tt_out = ttnn.multiply(ttnn.silu(to_dev(gate)), to_dev(up))
    ttnn.synchronize_device(mesh_device)
    got = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0])
    assert_pcc("swiglu", golden, got, topology_name)


@parametrize_target_mesh()
def test_swiglu_is_not_swigluoai(mesh_device, device_params):
    """The device activation must NOT match the donor's clamped swigluoai.

    A positive control for the test above: if the two were numerically indistinguishable at this
    scale, `test_swiglu_vs_ref` passing would say nothing about which variant actually ran.
    """
    torch.manual_seed(0)
    gate = torch.randn(1, 1, 32, 256, dtype=REF_DTYPE) * 4.0  # scaled so the clamp at 7.0 bites
    up = torch.randn(1, 1, 32, 256, dtype=REF_DTYPE)
    plain = torch.nn.functional.silu(gate) * up
    clamped = gate.clamp(max=7.0)
    oai = (clamped * torch.sigmoid(1.702 * clamped.float()).to(clamped.dtype)) * (up.clamp(-7.0, 7.0) + 1)
    a, b = plain.float().flatten(), oai.float().flatten()
    a, b = a - a.mean(), b - b.mean()
    pcc = float((a @ b) / (a.norm() * b.norm()))
    assert pcc < 0.99, f"plain SwiGLU and swigluoai are indistinguishable here (PCC {pcc}); test is vacuous"
