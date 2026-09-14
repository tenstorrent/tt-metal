# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""SwiGLU activation and the dense MLP, at the model's real dims (4096 -> 14336 -> 4096).

Two tests, because they fail differently: the activation is a fused one-op kernel and the MLP adds
the column/row-parallel split plus the closing TP all-reduce. If only the composite were tested, an
activation with the wrong variant and a compensating sharding bug could still land near the target.

The activation test also pins Llama's variant against gpt-oss/M3's clamped ``swigluoai``: a source
borrowed for the MLP *structure* carries that activation, and at ordinary activation magnitudes the
clamp at 7.0 almost never binds, so the wrong one scores ~0.999 on random data. Feeding it inputs
that DO cross the clamp is what makes the difference visible.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.llama_3_1_8b.tests.common import (
    assert_pcc,
    assert_tp_replicas_agree,
    cfg_full,
    galaxy_mesh,
    pcc,
    random_layer_weights,
    reference_mlp,
    spec_mesh_config,
    sp_shard_activation,
)
from models.demos.llama_3_1_8b.tt.mlp import MLP, swiglu


@galaxy_mesh()
def test_swiglu_vs_ref(mesh_device, device_params):
    """``silu(gate) * up`` as the fused op, vs torch, over a range that includes large magnitudes."""
    from models.demos.llama_3_1_8b.reference.model import swiglu as ref_swiglu

    mc = spec_mesh_config(mesh_device)
    torch.manual_seed(0)
    gate = torch.randn(1, 1, 512, 2048) * 4.0
    up = torch.randn(1, 1, 512, 2048) * 4.0

    tt_gate = sp_shard_activation(gate, mesh_device, mc)
    tt_up = sp_shard_activation(up, mesh_device, mc)
    got = assert_tp_replicas_agree(swiglu(tt_gate, tt_up), mesh_device, mc, name="swiglu", tol=0.2)

    ref = ref_swiglu(gate.to(torch.float16), up.to(torch.float16)).float()
    assert_pcc("swiglu", pcc(ref, got))

    # Negative control: the clamped swigluoai variant the borrowed MLP structure comes with.
    alpha, limit = 1.702, 7.0
    g32, u32 = gate.float(), up.float()
    g_c = g32.clamp(max=limit)
    u_c = u32.clamp(min=-limit, max=limit)
    oai = g_c * torch.sigmoid(alpha * g_c) * (u_c + 1.0)
    p_llama, p_oai = pcc(ref, got), pcc(oai, got)
    logger.info(f"llama silu-swiglu={p_llama:.6f} gpt-oss swigluoai={p_oai:.6f}")
    # Measured separation at these magnitudes: Llama ~0.99999, swigluoai ~0.955. Note what that
    # means — the WRONG activation still clears the spec's pcc_lower_bound of 0.85, so it would pass
    # every assert in the suite and show up only as a below-target number in the README. That is
    # precisely why the recipe says to port the MLP's structure and never its math.
    assert p_llama > 0.999, "device activation does not match plain silu-swiglu"
    assert p_oai < 0.99, "device activation is indistinguishable from the swigluoai variant"


@galaxy_mesh()
@pytest.mark.parametrize("seq", [1024, 5120], ids=["s1024", "chunk5120"])
def test_dense_mlp_vs_ref(mesh_device, device_params, seq, topology_name):
    """The whole FFN at real dims, column/row-parallel across TP=4 with the closing all-reduce."""
    cfg = cfg_full()
    mc = spec_mesh_config(mesh_device)
    ccl = None
    from models.demos.llama_3_1_8b.tests.common import make_ccl

    ccl = make_ccl(mesh_device)

    weights = random_layer_weights(cfg, seed=7)
    mlp = MLP(
        mesh_device,
        cfg,
        mc,
        ccl,
        state_dict={k[len("mlp.") :]: v for k, v in weights.items() if k.startswith("mlp.")},
    )

    torch.manual_seed(1)
    x = torch.randn(1, 1, seq, cfg.hidden_size) * 0.5
    tt_out = mlp(sp_shard_activation(x, mesh_device, mc))
    got = assert_tp_replicas_agree(tt_out, mesh_device, mc, name="dense_mlp", tol=0.05)

    ref = reference_mlp(cfg, weights)(x.to(torch.float16)).float()
    assert_pcc(f"dense_mlp[{topology_name}] s={seq}", pcc(ref, got))
