# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-layer parity: TT ``DiffusionGemmaMoE`` (wrapping demos/gemma4 MoEBlock) vs
the actual ``transformers.models.gemma4.modeling_gemma4.Gemma4TextRouter`` +
``Gemma4TextExperts``.

Uses a tiny config (8 experts, 256 hidden, 64 intermediate) so the test fits
in memory and runs fast. Real-config validation (128 experts, 2816 hidden) is
exercised at the layer / encoder integration tests (M4).

    pytest models/tt_dit/tests/models/diffusion_gemma/test_moe.py -s
"""

import pytest
import torch
from loguru import logger

import ttnn

from ....models.transformers.diffusion_gemma.moe import DiffusionGemmaMoE
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....utils.check import assert_quality
from ....utils.tensor import bf16_tensor, local_device_to_torch
from ....utils.test import line_params, ring_params

# Looser thresholds for MoE — sparse_matmul + topk paths add rounding vs HF's per-expert loop.
PCC_THRESHOLD = 0.99
ALLCLOSE_ATOL = 5e-2
ALLCLOSE_RTOL = 5e-2


@pytest.mark.parametrize(
    ("mesh_device", "tp_axis", "num_links", "device_params", "topology"),
    [
        pytest.param((2, 4), 0, 1, line_params, ttnn.Topology.Linear, id="bh_qb2_tp2"),
        pytest.param((2, 4), 0, 1, ring_params, ttnn.Topology.Ring, id="wh_t3k_tp2"),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("seq_len", [64])
def test_diffusion_gemma_moe(
    mesh_device: ttnn.MeshDevice, tp_axis: int, num_links: int, topology: ttnn.Topology, seq_len: int
) -> None:
    """TT MoE vs HF Gemma4TextRouter + Gemma4TextExperts (tiny config)."""
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaTextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextExperts, Gemma4TextRouter

    torch.manual_seed(0)
    dtype = torch.float32

    hidden_size = 256
    intermediate_size = 64
    num_experts = 8
    top_k = 2
    eps = 1e-6
    B = 1

    tp_factor = tuple(mesh_device.shape)[tp_axis]

    hf_config = DiffusionGemmaTextConfig(
        hidden_size=hidden_size,
        intermediate_size=hidden_size,  # unused by router/experts; placeholder
        moe_intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k_experts=top_k,
        rms_norm_eps=eps,
        hidden_activation="gelu_pytorch_tanh",
        num_hidden_layers=6,
    )

    # HF router + experts. Initialize their parameters from random so we have
    # something to compare; the modules use Gemma4-style init internally
    # (ones for scales, normal for projections), but for the parity test we just
    # need them to be deterministic and shared with TT.
    hf_router = Gemma4TextRouter(hf_config).to(dtype).eval()
    hf_experts = Gemma4TextExperts(hf_config).to(dtype).eval()
    # MixtralExperts.__init__ uses torch.empty; populate with reproducible random.
    with torch.no_grad():
        hf_experts.gate_up_proj.data = torch.randn_like(hf_experts.gate_up_proj.data) * 0.02
        hf_experts.down_proj.data = torch.randn_like(hf_experts.down_proj.data) * 0.02

    router_input = torch.randn(B, seq_len, hidden_size, dtype=dtype)
    expert_input = torch.randn(B, seq_len, hidden_size, dtype=dtype)

    with torch.no_grad():
        # Match the HF DiffusionGemma layer's routing call: flatten, then run.
        r_flat = router_input.reshape(-1, hidden_size)
        e_flat = expert_input.reshape(-1, hidden_size)
        _probs, top_k_weights, top_k_index = hf_router(r_flat)
        torch_out_flat = hf_experts(e_flat, top_k_index, top_k_weights)
        torch_out = torch_out_flat.reshape(B, seq_len, hidden_size)

    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp_factor),
        sequence_parallel=ParallelFactor(mesh_axis=1 - tp_axis, factor=tuple(mesh_device.shape)[1 - tp_axis]),
        cfg_parallel=None,
    )

    # Build the demos/gemma4-style state_dict that DiffusionGemmaMoE consumes.
    # The router substate fields: proj.weight, scale, per_expert_scale (norm has no weight).
    # The experts substate fields: gate_up_proj, down_proj.
    state_dict = {
        "router.proj.weight": hf_router.proj.weight.data,
        "router.scale": hf_router.scale.data,
        "router.per_expert_scale": hf_router.per_expert_scale.data,
        "experts.gate_up_proj": hf_experts.gate_up_proj.data,
        "experts.down_proj": hf_experts.down_proj.data,
    }

    tt_model = DiffusionGemmaMoE(
        hidden_size=hidden_size,
        num_experts=num_experts,
        top_k_experts=top_k,
        moe_intermediate_size=intermediate_size,
        rms_norm_eps=eps,
        state_dict=state_dict,
        mesh_device=mesh_device,
        parallel_config=parallel_config,
        num_links=num_links,
        topology=topology,
        expert_dtype=ttnn.bfloat16,
        router_dtype=ttnn.bfloat16,
    )

    tt_router_in = bf16_tensor(router_input.unsqueeze(0), device=mesh_device)
    tt_expert_in = bf16_tensor(expert_input.unsqueeze(0), device=mesh_device)
    tt_out = tt_model(tt_router_in, tt_expert_in)
    tt_out_torch = local_device_to_torch(tt_out).squeeze(0)

    logger.info(f"torch_out: {torch_out.shape}, tt_out: {tt_out_torch.shape}")
    assert_quality(torch_out, tt_out_torch, pcc=PCC_THRESHOLD)

    abs_diff = (torch_out - tt_out_torch.to(dtype)).abs()
    logger.info(f"max abs diff: {abs_diff.max().item():.3e}")
    assert torch.allclose(torch_out, tt_out_torch.to(dtype), atol=ALLCLOSE_ATOL, rtol=ALLCLOSE_RTOL)
