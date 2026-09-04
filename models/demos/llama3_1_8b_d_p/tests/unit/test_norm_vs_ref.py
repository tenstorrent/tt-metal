# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC: tt RMSNorm vs the torch reference. Random weights, identical on both sides.

Llama uses a PLAIN RMSNorm (no Gemma ``(1 + w)`` fold), so only the plain form is exercised — the
fold is not a Llama code path and testing it here would test the donor, not this model.

This one class is used for all three norm instances (input_layernorm, post_attention_layernorm, and
the model's final norm), so this test covers the M2 "final-norm instance" row as well.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.llama3_1_8b_d_p.reference.model import LlamaRMSNorm
from models.demos.llama3_1_8b_d_p.tt.rms_norm import RMSNorm

from ..test_factory import dev0, llama_config, parametrize_mesh_with_fabric, replicate

PCC = 0.99


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1), (8, 4)])
@pytest.mark.parametrize("m", [32, 128, 512], ids=["m32", "m128", "m512"])
def test_rms_norm_vs_ref(mesh_device, device_params, m, reset_seeds):
    cfg = llama_config()
    width = cfg.hidden_size
    x = torch.randn(1, 1, m, width)
    weight = torch.randn(width)

    ref_norm = LlamaRMSNorm(width, cfg.rms_norm_eps)
    ref_norm.weight.data = weight.clone()
    ref = ref_norm(x)

    norm = RMSNorm(mesh_device=mesh_device, hf_config=cfg, state_dict={"weight": weight}, tensor_cache_path=None)
    out = dev0(norm(replicate(mesh_device, x))).reshape(1, 1, m, width)

    passing, pcc = comp_pcc(ref, out, PCC)
    logger.info(f"rms_norm m={m}: {pcc}")
    assert passing, f"PCC fail: {pcc}"
