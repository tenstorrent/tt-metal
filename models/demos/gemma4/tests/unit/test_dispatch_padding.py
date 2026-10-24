# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Padding-aware token-dispatch prefill (GEMMA4_DISPATCH_PADDING): a chunk with fewer real rows than
CHUNK_TOKENS must give the same expert output on its real rows with the padded rows sentinel-marked and
skipped by dispatch as with every row routed, and the short chunk must get cheaper (a 128-token prompt
no longer pays for 4096 rows of expert work)."""

import os
import time

import pytest
import torch

import ttnn
from models.demos.gemma4.tests.test_factory import (
    lanes_fabric_config,
    parametrize_mesh_with_fabric,
    skip_if_config_only_checkpoint,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("HF_MODEL", "google/gemma-4-26B-A4B-it").find("26B") < 0, reason="MoE only"
)


@parametrize_mesh_with_fabric(
    mesh_shapes=[(8, 4)],
    device_params_extra={"fabric_config": lanes_fabric_config(), "l1_small_size": 65536},
)
@pytest.mark.parametrize("n_tokens", [128, 2065], ids=["n128", "n2065"])
def test_dispatch_padding_matches_full_chunk(mesh_device, n_tokens, reset_seeds, request, monkeypatch):
    skip_if_config_only_checkpoint()
    monkeypatch.setenv("GEMMA4_GALAXY_FRACTURE", "1")
    monkeypatch.setenv("GEMMA4_GALAXY_LANES", "1")
    monkeypatch.setenv("GEMMA4_MOE_DISPATCH_PREFILL", "1")
    from models.demos.gemma4.config import MeshConfig, ModeConfig
    from models.demos.gemma4.tests.test_factory import TestFactory
    from models.demos.gemma4.tt.ccl import CCLManager
    from models.demos.gemma4.tt.experts import Gemma4ExpertConfig, Gemma4Experts

    # layer-0 routed experts on the real checkpoint, the Galaxy one-instance orientation (as test_experts.py)
    hf_text_config = TestFactory.create_hf_text_config()
    hf_layer = TestFactory.create_hf_reference_layer(hf_text_config, layer_idx=0)
    state_dict = {
        "gate_up_proj": hf_layer.experts.gate_up_proj.data.clone(),
        "down_proj": hf_layer.experts.down_proj.data.clone(),
    }
    hf_config = TestFactory.create_hf_config()
    tp, ep = mesh_device.shape[0], mesh_device.shape[1]
    mesh_config = MeshConfig(
        mesh_device.shape,
        decode=ModeConfig(tp=tp, ep=ep),
        prefill=ModeConfig(tp=tp, ep=ep),
        tp_axis=0,
        weight_fracture=True,
    )
    tt_experts = Gemma4Experts(
        mesh_device=mesh_device,
        config=Gemma4ExpertConfig(hf_config),
        state_dict=state_dict,
        ccl_manager=CCLManager(mesh_device, num_links=1),
        mesh_config=mesh_config,
        program_config=None,
        weight_dtype=ttnn.bfloat8_b,
    )
    experts = tt_experts.dispatch_experts
    assert experts is not None, "GEMMA4_MOE_DISPATCH_PREFILL did not build the dispatch experts"
    H, E = experts.hidden, experts.num_experts
    torch.manual_seed(0)
    hidden = torch.randn(1, 1, n_tokens, H, dtype=torch.bfloat16) * 0.5
    routing = torch.zeros(1, 1, n_tokens, E, dtype=torch.bfloat16)
    top = torch.rand(n_tokens, E).topk(experts.top_k, dim=-1)
    routing[0, 0].scatter_(1, top.indices, torch.softmax(top.values, dim=-1).to(torch.bfloat16))

    def run(padding):
        experts.PADDING_ENABLED = padding
        h = ttnn.from_torch(
            hidden,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        r = ttnn.from_torch(
            routing,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        experts.forward(h, r)  # warm (program cache)
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        out = experts.forward(h, r)
        ttnn.synchronize_device(mesh_device)
        dt = time.perf_counter() - t0
        host = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0:1].float()
        return host[0, 0, :n_tokens], dt

    full, t_full = run(False)
    padded, t_pad = run(True)
    assert torch.isfinite(padded).all()
    pcc = torch.corrcoef(torch.stack([full.flatten(), padded.flatten()]))[0, 1].item()
    print(
        f"[n_tokens={n_tokens}] pcc(full, padded)={pcc:.6f} max|diff|={(full - padded).abs().max():.4f} "
        f"time full={t_full*1e3:.1f} ms padded={t_pad*1e3:.1f} ms"
    )
    assert pcc > 0.999, f"padded dispatch diverges from the full chunk: pcc={pcc}"
    if n_tokens <= 128:
        assert t_pad < 0.8 * t_full, f"short chunk did not get cheaper: {t_pad*1e3:.1f} vs {t_full*1e3:.1f} ms"
