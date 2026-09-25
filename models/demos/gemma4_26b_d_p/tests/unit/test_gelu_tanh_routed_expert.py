# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Single-chip PCC of ``unified_routed_expert_ffn`` with ``RoutedExpertActivation.GeluTanh``
at Gemma-4 26B-A4B expert dims (emb 2816, moe_intermediate 704), real layer-0 expert weights."""

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, is_blackhole
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from models.demos.gemma4_26b_d_p.reference.weights import CheckpointReader

EMB, INTER = 2816, 704


def _expert_weights(layer=0, expert=0):
    r = CheckpointReader()
    gu = r.get(f"layers.{layer}.experts.gate_up_proj")[expert].float()  # [2I, H], gate rows first
    down = r.get(f"layers.{layer}.experts.down_proj")[expert].float()  # [H, I]
    return {"gate_proj": gu[:INTER].contiguous(), "up_proj": gu[INTER:].contiguous(), "down_proj": down.contiguous()}


def _ref(x, w):
    return F.linear(F.gelu(F.linear(x, w["gate_proj"]), approximate="tanh") * F.linear(x, w["up_proj"]), w["down_proj"])


@pytest.mark.skipif(not is_blackhole(), reason="unified_routed_expert op is Blackhole-only")
@pytest.mark.parametrize("weights_dtype", [ttnn.bfloat8_b, ttnn.bfloat4_b], ids=["bfp8", "bfp4"])
@pytest.mark.parametrize("num_tokens", [128, 1024, 4096], ids=["t128", "t1k", "t4k"])
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [pytest.param(1, {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-chip")],
    indirect=["mesh_device", "device_params"],
)
def test_gelu_tanh_routed_expert(mesh_device, device_params, num_tokens, weights_dtype):
    torch.manual_seed(0)
    w = _expert_weights()
    # Pre-FFN-norm activations are O(1); real experts see rmsnorm'd inputs.
    x = torch.randn(num_tokens, EMB)
    ref = _ref(x, w)

    tt_x = ttnn.from_torch(x, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device), layout=ttnn.TILE_LAYOUT, device=mesh_device, dtype=ttnn.bfloat8_b)
    idx = lambda v: ttnn.from_torch(torch.tensor(v, dtype=torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, dtype=ttnn.uint32)
    expert = TtRoutedExpert(
        mesh_device=mesh_device,
        experts_per_chip=1,
        global_expert_idx_table=idx([0]),
        emb_dim=EMB,
        hidden_dim=INTER,
        max_tokens=num_tokens,
        torch_weights=[w],
        activations_dtype=ttnn.bfloat8_b,
        weights_dtype=weights_dtype,
        activation=ttnn.RoutedExpertActivation.GeluTanh,
    )
    out = expert(tt_x, idx([num_tokens]), idx([0]))
    out = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:num_tokens]
    thr = 0.99 if weights_dtype == ttnn.bfloat8_b else 0.97
    ok, pcc = comp_pcc(ref, out, thr)
    logger.info(f"GeluTanh expert tokens={num_tokens} w={weights_dtype}: {pcc}")
    assert not torch.isnan(out).any()
    assert ok, pcc
