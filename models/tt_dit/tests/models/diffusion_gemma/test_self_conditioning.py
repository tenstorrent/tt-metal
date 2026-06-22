# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-layer parity: TT ``DiffusionGemmaSelfConditioning`` vs reference torch impl
(ported from transformers.models.diffusion_gemma.modeling_diffusion_gemma).

    pytest models/tt_dit/tests/models/diffusion_gemma/test_self_conditioning.py -s
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

import ttnn

from ....models.transformers.diffusion_gemma.self_conditioning import DiffusionGemmaSelfConditioning
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils.check import assert_quality
from ....utils.tensor import bf16_tensor, local_device_to_torch
from ....utils.test import line_params

# Tight thresholds for self-conditioning (small MLP, no attention).
PCC_THRESHOLD = 0.9995
ALLCLOSE_ATOL = 2e-2
ALLCLOSE_RTOL = 2e-2


class _GemmaRMSNorm(nn.Module):
    """Plain (no (1+w)) RMSNorm matching Gemma3nRMSNorm."""

    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        ms = x_fp32.pow(2).mean(-1, keepdim=True) + self.eps
        normed = x_fp32 * torch.pow(ms, -0.5)
        if self.with_scale:
            normed = normed * self.weight.float()
        return normed.type_as(x)


class _TorchSelfConditioning(nn.Module):
    """Reference impl mirroring transformers' ``DiffusionGemmaSelfConditioning``."""

    def __init__(self, hidden_size: int, intermediate_size: int, eps: float):
        super().__init__()
        self.pre_norm = _GemmaRMSNorm(hidden_size, eps=eps, with_scale=True)
        self.post_norm = _GemmaRMSNorm(hidden_size, eps=eps, with_scale=False)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, inputs_embeds: torch.Tensor, sc_signal: torch.Tensor) -> torch.Tensor:
        normed = self.pre_norm(sc_signal)
        sc = self.down_proj(F.gelu(self.gate_proj(normed), approximate="tanh") * self.up_proj(normed))
        return self.post_norm(inputs_embeds + sc)


@pytest.mark.parametrize(
    ("mesh_device", "tp_axis", "num_links", "device_params"),
    [
        # BH QB2 (2x4) — TP only on axis 1 (factor=4); axis 0 will eventually hold EP.
        # NOTE: intermediate_size=2112 is not tile-aligned for tp=4; we use tp=2 here
        # until output padding is added to the gated-MLP wrapper. See M3.
        pytest.param((2, 4), 0, 1, line_params, id="bh_qb2_tp2"),
        # BH galaxy (4x8) — same constraint applies; tp=2 along axis 0 keeps the
        # intermediate tile-aligned (1056 dim/dev = 33 tiles).
        pytest.param((4, 8), 0, 2, line_params, id="bh_galaxy_tp4"),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_self_conditioning(
    mesh_device: ttnn.MeshDevice,
    tp_axis: int,
    num_links: int,
) -> None:
    """Verify DiffusionGemmaSelfConditioning matches the torch reference under TP."""

    torch.manual_seed(0)
    torch_dtype = torch.float32

    # DiffusionGemma-26B-A4B-it text config values.
    hidden_size = 2816
    intermediate_size = 2112
    rms_eps = 1e-6
    canvas_length = 256
    B = 1

    tp_factor = tuple(mesh_device.shape)[tp_axis]
    # Skip TP factors that break tile alignment until padding is wired in.
    if (intermediate_size // tp_factor) % ttnn.TILE_SIZE != 0:
        pytest.skip(f"intermediate_size={intermediate_size} / tp={tp_factor} is not tile-aligned")

    # Build torch reference & random inputs.
    torch_model = _TorchSelfConditioning(hidden_size, intermediate_size, rms_eps).eval()
    inputs_embeds = torch.randn(B, canvas_length, hidden_size, dtype=torch_dtype)
    sc_signal = torch.randn(B, canvas_length, hidden_size, dtype=torch_dtype)

    with torch.no_grad():
        torch_out = torch_model(inputs_embeds, sc_signal)

    # CCL & parallel config.
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=ttnn.Topology.Linear)
    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp_factor),
        sequence_parallel=ParallelFactor(mesh_axis=1 - tp_axis, factor=tuple(mesh_device.shape)[1 - tp_axis]),
        cfg_parallel=None,
    )

    tt_model = DiffusionGemmaSelfConditioning(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        rms_norm_eps=rms_eps,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
    )
    tt_model.load_state_dict(torch_model.state_dict())

    # Inputs are replicated on the mesh (single-batch, no SP fracturing).
    tt_inputs_embeds = bf16_tensor(inputs_embeds.unsqueeze(0), device=mesh_device)
    tt_sc_signal = bf16_tensor(sc_signal.unsqueeze(0), device=mesh_device)

    tt_out = tt_model(tt_inputs_embeds, tt_sc_signal)
    tt_out_torch = local_device_to_torch(tt_out).squeeze(0)

    logger.info(f"torch_out: {torch_out.shape} {torch_out.dtype}, tt_out: {tt_out_torch.shape} {tt_out_torch.dtype}")
    assert_quality(torch_out, tt_out_torch, pcc=PCC_THRESHOLD)

    # Allclose check on top of the PCC bar.
    abs_diff = (torch_out - tt_out_torch.to(torch_dtype)).abs()
    rel_diff = abs_diff / (torch_out.abs() + 1e-6)
    logger.info(f"max abs diff: {abs_diff.max().item():.3e}, max rel diff: {rel_diff.max().item():.3e}")
    assert torch.allclose(
        torch_out, tt_out_torch.to(torch_dtype), atol=ALLCLOSE_ATOL, rtol=ALLCLOSE_RTOL
    ), f"allclose failed: max abs={abs_diff.max().item():.3e}, max rel={rel_diff.max().item():.3e}"
