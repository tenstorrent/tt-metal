# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""RMSNorm on the target mesh vs the torch reference, at the model's real width.

Llama's norm is the PLAIN form (``x_normed * w``), not Gemma's ``(1 + w)``. The negative control at
the bottom is the point of this file: if the Gemma fold were accidentally carried over from the
borrowed M3 norm, the plain comparison would still score ~0.99 on small random gains and nobody
would notice until the whole model was 0.02 off. So the test also asserts the *folded* form does
NOT match.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.llama_3_1_8b.reference.model import RMSNorm as RefRMSNorm
from models.demos.llama_3_1_8b.tests.common import (
    assert_pcc,
    assert_tp_replicas_agree,
    cfg_full,
    galaxy_mesh,
    pcc,
    spec_mesh_config,
    sp_shard_activation,
)
from models.demos.llama_3_1_8b.tt.rms_norm import RMSNorm


def _reference(weight, x, eps, hidden):
    ref = RefRMSNorm(hidden, eps)
    ref.weight.data = weight.to(torch.float16).clone()
    return ref(x.to(torch.float16)).float()


@galaxy_mesh()
@pytest.mark.parametrize("seq", [256, 1024], ids=["s256", "s1024"])
def test_norm_vs_ref(mesh_device, device_params, seq, topology_name):
    """The decoder layernorm instance at hidden 4096, sequence sharded across the 8 SP rows."""
    cfg = cfg_full()
    mc = spec_mesh_config(mesh_device)
    torch.manual_seed(0)
    x = torch.randn(1, 1, seq, cfg.hidden_size) * 0.5
    weight = torch.randn(cfg.hidden_size) * 0.1

    norm = RMSNorm(mesh_device, cfg.hidden_size, cfg.rms_norm_eps, state_dict={"weight": weight})
    tt_out = norm(sp_shard_activation(x, mesh_device, mc))
    got = assert_tp_replicas_agree(tt_out, mesh_device, mc, name="rms_norm")

    ref = _reference(weight, x, cfg.rms_norm_eps, cfg.hidden_size)
    assert_pcc(f"rms_norm[{topology_name}] s={seq}", pcc(ref, got))


@galaxy_mesh()
def test_norm_is_plain_not_gemma(mesh_device, device_params):
    """Negative control: Llama's norm must NOT match the Gemma ``(1 + w)`` fold.

    A large-magnitude gain is used deliberately — at the ~0.1 gains a model actually has, the two
    forms agree to about 0.99 PCC and the mistake hides inside the accept band.
    """
    cfg = cfg_full()
    mc = spec_mesh_config(mesh_device)
    torch.manual_seed(0)
    x = torch.randn(1, 1, 256, cfg.hidden_size)
    weight = torch.randn(cfg.hidden_size) * 2.0

    norm = RMSNorm(mesh_device, cfg.hidden_size, cfg.rms_norm_eps, state_dict={"weight": weight})
    got = assert_tp_replicas_agree(
        norm(sp_shard_activation(x, mesh_device, mc)), mesh_device, mc, name="rms_norm"
    )

    plain = _reference(weight, x, cfg.rms_norm_eps, cfg.hidden_size)
    gemma = _reference(weight + 1.0, x, cfg.rms_norm_eps, cfg.hidden_size)
    p_plain, p_gemma = pcc(plain, got), pcc(gemma, got)
    logger.info(f"plain={p_plain:.6f} gemma-fold={p_gemma:.6f}")
    assert p_plain > 0.99
    assert p_gemma < p_plain - 0.05, "device norm is indistinguishable from the Gemma fold"
