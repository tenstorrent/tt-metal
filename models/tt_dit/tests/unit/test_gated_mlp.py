# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Layer-level parity: tt_dit ``GatedMLP`` vs the actual
``transformers.models.diffusion_gemma.modeling_diffusion_gemma.DiffusionGemmaText4MLP``
and ``transformers.models.gemma4.modeling_gemma4.Gemma4VisionMLP``.

Both HF classes are gate*up → down gated MLPs; the only TT-side specialization
is intermediate-dim padding when ``intermediate_size`` isn't tile-aligned for the
TP factor (handled by ``GatedMLP._prepare_torch_state``).

    pytest models/tt_dit/tests/unit/test_gated_mlp.py -s
"""

import pytest
import torch
from loguru import logger

import ttnn

from ...layers.feedforward import GatedMLP
from ...parallel.config import DiTParallelConfig, ParallelFactor
from ...parallel.manager import CCLManager
from ...utils.check import assert_quality
from ...utils.tensor import bf16_tensor, local_device_to_torch
from ...utils.test import line_params, ring_params

PCC_THRESHOLD = 0.9995
ALLCLOSE_ATOL = 2e-2
ALLCLOSE_RTOL = 2e-2


@pytest.mark.parametrize(
    ("mesh_device", "tp_axis", "num_links", "device_params", "topology"),
    [
        pytest.param((2, 4), 0, 1, line_params, ttnn.Topology.Linear, id="bh_qb2_tp2"),
        pytest.param((2, 4), 0, 1, ring_params, ttnn.Topology.Ring, id="wh_t3k_tp2"),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    ("kind", "hidden_size", "intermediate_size"),
    [
        # DiffusionGemma text dense MLP (intermediate=2112 is tile-aligned for tp=2).
        pytest.param("text", 2816, 2112, id="text-2816x2112"),
        # Vision MLP — intermediate=4304 is NOT tile-aligned; GatedMLP pads it.
        pytest.param("vision", 1152, 4304, id="vision-1152x4304"),
    ],
)
@pytest.mark.parametrize("seq_len", [64])
def test_gated_mlp(
    mesh_device: ttnn.MeshDevice,
    tp_axis: int,
    num_links: int,
    topology: ttnn.Topology,
    kind: str,
    hidden_size: int,
    intermediate_size: int,
    seq_len: int,
) -> None:
    """TT GatedMLP vs HF MLP class (text or vision branch)."""
    torch.manual_seed(0)
    dtype = torch.float32
    B = 1

    tp_factor = tuple(mesh_device.shape)[tp_axis]

    if kind == "text":
        from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
            DiffusionGemmaText4MLP,
            DiffusionGemmaTextConfig,
        )

        hf_config = DiffusionGemmaTextConfig(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_activation="gelu_pytorch_tanh",
        )
        torch_model = DiffusionGemmaText4MLP(hf_config, layer_idx=0).to(dtype).eval()
    else:
        from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig
        from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionMLP

        hf_config = Gemma4VisionConfig(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_activation="gelu_pytorch_tanh",
            num_attention_heads=16,
            num_key_value_heads=16,
            head_dim=72,
            patch_size=16,
            num_hidden_layers=1,
        )
        torch_model = Gemma4VisionMLP(hf_config).to(dtype).eval()

    x = torch.randn(B, seq_len, hidden_size, dtype=dtype)
    with torch.no_grad():
        torch_out = torch_model(x)

    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp_factor),
        sequence_parallel=ParallelFactor(mesh_axis=1 - tp_axis, factor=tuple(mesh_device.shape)[1 - tp_axis]),
        cfg_parallel=None,
    )

    tt_model = GatedMLP(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
    )
    tt_model.load_state_dict(torch_model.state_dict())

    tt_x = bf16_tensor(x.unsqueeze(0), device=mesh_device)
    tt_out = tt_model(tt_x)
    tt_out_torch = local_device_to_torch(tt_out).squeeze(0)

    logger.info(f"[{kind}] torch_out: {torch_out.shape}, tt_out: {tt_out_torch.shape}")
    assert_quality(torch_out, tt_out_torch, pcc=PCC_THRESHOLD)

    abs_diff = (torch_out - tt_out_torch.to(dtype)).abs()
    logger.info(f"max abs diff: {abs_diff.max().item():.3e}")
    assert torch.allclose(torch_out, tt_out_torch.to(dtype), atol=ALLCLOSE_ATOL, rtol=ALLCLOSE_RTOL)
