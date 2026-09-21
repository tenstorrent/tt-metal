# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Isolate FLUX.2 timestep/guidance conditioning from attention and denoising."""

import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from safetensors import safe_open
import torch
import ttnn

from models.tt_dit.layers.embeddings import CombinedTimestepGuidanceTextProjEmbeddings
from models.tt_dit.utils.tensor import from_torch


@pytest.fixture
def device_params():
    return {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 65536}


def quality(actual, reference):
    a, b = actual.double().flatten(), reference.double().flatten()
    return dict(l2_pct=100 * float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b)),
                pcc=float(torch.corrcoef(torch.stack([a, b]))[0, 1]))


@pytest.mark.parametrize("mesh_device", [(2, 4)], ids=["bh_lb"], indirect=True)
@pytest.mark.timeout(600)
def test_conditioning(mesh_device):
    from diffusers.models.transformers.transformer_flux2 import Flux2TimestepGuidanceEmbeddings

    torch.set_num_threads(8)
    directory = Path(os.environ["FLUX2_CHECKPOINT"]) / "transformer"
    index = json.loads((directory / "diffusion_pytorch_model.safetensors.index.json").read_text())
    prefix = "time_guidance_embed."
    state = {}
    for key, filename in index["weight_map"].items():
        if key.startswith(prefix):
            with safe_open(directory / filename, framework="pt", device="cpu") as handle:
                state[key[len(prefix):]] = handle.get_tensor(key)
    reference_model = Flux2TimestepGuidanceEmbeddings(embedding_dim=6144).bfloat16().eval()
    reference_model.load_state_dict(state, strict=True)
    model = CombinedTimestepGuidanceTextProjEmbeddings(embedding_dim=6144, pooled_projection_dim=0,
                                                      bias=False, with_guidance=True, mesh_device=mesh_device)
    model.load_torch_state_dict(state)
    original_factor = model.time_proj_factor
    factor = torch.exp(-math.log(10000) * torch.arange(128, dtype=torch.float32) / 128)
    precise_factor = ttnn.unsqueeze_to_4D(from_torch(factor, device=mesh_device, dtype=ttnn.float32))
    records = []
    original_forward = model.forward
    for timestep in (1000.0, 500.0, 100.0):
        with torch.no_grad():
            reference = reference_model(torch.tensor([timestep], dtype=torch.bfloat16),
                                        torch.tensor([4000.0], dtype=torch.bfloat16))
        for label, scale, fmt, projection in (
            ("stock", 1, ttnn.bfloat16, original_factor),
            ("guidance_x1000", 1000, ttnn.bfloat16, original_factor),
            ("guidance_x1000_fp32_phase", 1000, ttnn.float32, precise_factor),
        ):
            model.time_proj_factor = projection
            t = from_torch(torch.tensor([[timestep]]), device=mesh_device, dtype=ttnn.float32)
            g = from_torch(torch.tensor([[4.0 * scale]]), device=mesh_device, dtype=fmt)
            out = model.forward(timestep=t, guidance=g)
            actual = ttnn.to_torch(ttnn.get_device_tensors(out)[0])
            assert bool(torch.isfinite(actual).all())
            row = dict(timestep=timestep, mode=label, **quality(actual, reference))
            records.append(row)
            print("CONDITIONING_QUALIFICATION", json.dumps(row), flush=True)
        from conditioning import install
        expected_fixed = actual.clone()
        assert records[-1]["l2_pct"] < 1.5
        install(SimpleNamespace(time_guidance_embed=model))
        out = model.forward(timestep=t, guidance=from_torch(torch.tensor([[4.0]]), device=mesh_device))
        actual_fixed = ttnn.to_torch(ttnn.get_device_tensors(out)[0])
        assert torch.equal(actual_fixed, expected_fixed), "Repair wrapper differs from isolated FP32-phase test"
        model.forward = original_forward
    output = Path(os.environ["FLUX2_CONDITION_REPORT"])
    with output.open("x") as handle:
        json.dump(records, handle, indent=2)
