# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""PCC tests for the KREA-2 (Krea2) single-stream MMDiT port.

Validates the tt_dit `Krea2Transformer` against fp32 reference goldens produced by
`reference/generate_goldens.py` (diffusers `main` Krea2Transformer2DModel). Goldens
carry the config, inputs, reference state_dict and reference output, so the test is
self-contained and does not require the diffusers `main` shadow at test time.

Run (single Blackhole/Wormhole chip):
    pytest models/tt_dit/tests/models/krea2/test_transformer_krea2.py
"""
import os

import pytest
import torch

import ttnn

from ....models.transformers.transformer_krea2 import Krea2Checkpoint
from ....utils.check import assert_quality
from ....utils.tensor import bf16_tensor

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "reference", "goldens")


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize(
    "golden_name",
    [
        "transformer_full_nomask_small",
        "transformer_full_mask_small",
        "transformer_full_nomask_real1",
        "transformer_full_mask_real1",
    ],
)
def test_krea2_transformer_pcc(*, mesh_device: ttnn.MeshDevice, golden_name: str) -> None:
    path = os.path.join(GOLDEN_DIR, f"{golden_name}.pt")
    if not os.path.exists(path):
        pytest.skip(f"golden {golden_name} not present (regenerate with reference/generate_goldens.py)")

    g = torch.load(path, weights_only=False)
    cfg, inp, sd, ref_out = g["config"], g["inputs"], g["state_dict"], g["output"]

    model = Krea2Checkpoint(config=cfg, state_dict=sd).build(mesh_device=mesh_device)

    hs = bf16_tensor(inp["hidden_states"], device=mesh_device)
    ehs = bf16_tensor(inp["encoder_hidden_states"], device=mesh_device)
    tt_out = model.forward(
        hs,
        ehs,
        inp["timestep"],
        inp["position_ids"],
        encoder_attention_mask=inp.get("encoder_attention_mask", None),
    )
    tt_out_torch = ttnn.to_torch(tt_out).reshape(ref_out.shape)

    assert_quality(ref_out, tt_out_torch, pcc=0.99)
