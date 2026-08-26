# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DEVICE (1 chip): TT RMSNorm vs the torch reference (which is itself pinned to HF).

Mistral's norm is the plain form with eps 1e-5 — identical in mechanism to Llama-3.3-70B's.

NOTE: RMSNorm is **shared substrate**, not one of the two scoped modules (attention / MLP). Under the
sharded-residual contract the decoder layer owns the all-gather that turns the ``emb/tp`` residual
back into full emb before calling this; the norm itself is TP-agnostic and unchanged.

Run:  pytest models/demos/mistral_medium_d_p/tests/unit/test_rms_norm_vs_ref.py
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.mistral_medium_d_p.reference.torch_reference import rms_norm
from models.demos.mistral_medium_d_p.tt.rms_norm import RMSNorm

from ..test_factory import mesh_setup, parametrize_mesh_with_fabric, replicate
from .shapes import EPS, HIDDEN, HFConfigStub


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
@pytest.mark.parametrize("seq_len", [128, 2048], ids=["s128", "s2k"])
def test_rms_norm_vs_ref(mesh_device, device_params, seq_len, reset_seeds):
    torch.manual_seed(0)
    x = torch.randn(1, seq_len, HIDDEN) * 0.5
    gain = 1.0 + torch.randn(HIDDEN) * 0.02

    ref = rms_norm(x.float(), gain, EPS)

    mesh_config, _ = mesh_setup(mesh_device)
    norm = RMSNorm(mesh_device, HFConfigStub(), {"weight": gain}, mesh_config=mesh_config)
    x_tt = replicate(x.reshape(1, 1, seq_len, HIDDEN), mesh_device)
    out = ttnn.to_torch(ttnn.get_device_tensors(norm(x_tt))[0]).reshape(1, seq_len, HIDDEN)

    passing, pcc = comp_pcc(ref, out, 0.999)
    logger.info(f"rms_norm vs ref (s={seq_len}): {pcc}")
    assert passing, f"RMSNorm PCC fail: {pcc}"


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
def test_rms_norm_rejects_gemma_fold(mesh_device, device_params, reset_seeds, expect_error):
    """A Gemma-style (1+w) config must fail loud, not shift every norm by one."""
    with expect_error(NotImplementedError, "use_gemma_norm"):
        RMSNorm(mesh_device, HFConfigStub(use_gemma_norm=True), {"weight": torch.ones(HIDDEN)})
