# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC: :class:`~models.experimental.mistral_small_4_119b.tt.mistral4_moe_mesh.TtMistral4MoeRoutedExpertParallelSkeleton` vs HF ``Mistral4NaiveMoe`` (routed branch only).

Validates full **TTNN** forward (router, experts; replicated expert weights, no combine reduce) against
``mlp.experts(...)`` for a randomly initialized ``Mistral4MoE`` (no hub download). ``TT_CCL`` is
constructed in the test and passed into the skeleton like Mixtral-style MoE bring-up.
"""

import pytest
import torch
import torch.nn as nn
import ttnn
from loguru import logger

from models.common.auto_compose import to_torch_auto_compose
from models.common.utility_functions import comp_allclose, comp_pcc, run_for_wormhole_b0_or_blackhole
from models.tt_transformers.tt.ccl import TT_CCL
from models.experimental.mistral_small_4_119b.constants import HF_MODEL_ID
from models.experimental.mistral_small_4_119b.tests.mesh_param import mesh_device_request_param
from models.experimental.mistral_small_4_119b.tt.mistral4_moe_mesh import (
    TtMistral4MoeRoutedExpertParallelSkeleton,
    route_tokens_to_experts_torch,
)


pytest.importorskip("transformers")
pytest.importorskip(
    "transformers.models.mistral4.modeling_mistral4", reason="Mistral4 MoE requires recent transformers"
)


def _text_config_eager_attn():
    from transformers import AutoConfig

    try:
        cfg = AutoConfig.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
    except Exception as exc:
        pytest.skip(f"Could not load HF config: {exc}")
    text = cfg.text_config
    if text is None:
        pytest.skip("Config has no text_config")
    if hasattr(text, "attn_implementation"):
        text.attn_implementation = "eager"
    if hasattr(text, "_attn_implementation"):
        text._attn_implementation = "eager"
    return text


def _init_moe_weights(mlp: nn.Module) -> None:
    with torch.no_grad():
        for m in mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif hasattr(m, "weight") and isinstance(getattr(m, "weight"), nn.Parameter):
                nn.init.normal_(m.weight, std=0.02)


@torch.no_grad()
@run_for_wormhole_b0_or_blackhole()
@pytest.mark.parametrize("mesh_device", [mesh_device_request_param()], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 30000000, "num_command_queues": 1}],
    indirect=True,
)
@pytest.mark.parametrize("seq_len", (8,))
@pytest.mark.parametrize("shard_routed_experts", (None, False))
def test_mistral4_moe_routed_skeleton_pcc_vs_hf_naive_moe(seq_len, shard_routed_experts, reset_seeds, mesh_device):
    from transformers.models.mistral4.modeling_mistral4 import Mistral4MoE

    text = _text_config_eager_attn()
    mlp = Mistral4MoE(text).eval()
    _init_moe_weights(mlp)
    mlp = mlp.to(torch.bfloat16)

    torch.manual_seed(0)
    bsz = 1
    h = text.hidden_size
    x = torch.randn(bsz, seq_len, h, dtype=torch.bfloat16)
    router_logits = mlp.gate(x)
    topk_indices, topk_weights = mlp.route_tokens_to_experts(router_logits)
    hidden_flat = x.reshape(-1, h)
    ref_routed = mlp.experts(hidden_flat, topk_indices, topk_weights).view(bsz, seq_len, h)

    tt_ccl = TT_CCL(mesh_device)
    sk = TtMistral4MoeRoutedExpertParallelSkeleton(
        mesh_device,
        text,
        mlp.state_dict(),
        tt_ccl=tt_ccl,
        shard_routed_experts=shard_routed_experts,
    )
    x_tt = ttnn.from_torch(
        x.unsqueeze(1),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device=mesh_device),
    )
    y_tt = sk(x_tt)
    yt = to_torch_auto_compose(y_tt, device=mesh_device)
    # ``to_torch_auto_compose`` may concatenate replicated mesh replicas (see
    # ``models.common.auto_compose``); trim to the reference element count.
    expected_numel = ref_routed.numel()
    if yt.numel() != expected_numel:
        if yt.numel() % expected_numel != 0:
            pytest.fail(
                f"composed output numel {yt.numel()} not a multiple of reference {expected_numel}; "
                f"shape={tuple(yt.shape)}"
            )
        yt = yt.flatten()[:expected_numel]
    y = yt.view_as(ref_routed).contiguous()

    passing, msg = comp_pcc(ref_routed, y, pcc=0.99)
    logger.info(comp_allclose(ref_routed, y))
    logger.info(f"Mistral4 routed MoE skeleton PCC: {msg}")
    assert passing, msg


@torch.no_grad()
def test_route_tokens_to_experts_torch_matches_hf_moe():
    from transformers.models.mistral4.modeling_mistral4 import Mistral4MoE

    text = _text_config_eager_attn()
    mlp = Mistral4MoE(text).eval()
    _init_moe_weights(mlp)
    mlp = mlp.to(torch.bfloat16)
    torch.manual_seed(1)
    x = torch.randn(1, 4, text.hidden_size, dtype=torch.bfloat16)
    logits = mlp.gate(x)
    ref_idx, ref_w = mlp.route_tokens_to_experts(logits)
    flat = logits.reshape(-1, logits.shape[-1])
    tt_idx, tt_w = route_tokens_to_experts_torch(flat, text)
    assert torch.equal(ref_idx, tt_idx)
    assert torch.allclose(ref_w, tt_w, rtol=0, atol=0)
