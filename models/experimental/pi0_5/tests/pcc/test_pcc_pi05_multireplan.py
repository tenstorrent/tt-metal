# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Pi0.5 multi-replan regression test (TTNN vs PyTorch).

This guards the closed-loop failure mode that every single-call test misses:
calling ``sample_actions`` repeatedly with *different* observations (as a rollout
does on every replan). The TTNN expert attention keeps a persistent KV-concat
buffer for the prefix (VLM) KV; if that prefix is cached across calls instead of
refilled per observation, the policy attends to the FIRST frame's prefix forever
— silently wrong, fatal to closed-loop control, yet invisible to:

  - test_pcc_pi05_model        (one sample_actions call)
  - test_determinism_pi05      (repeated calls with the SAME obs -> stale == correct)
  - test_pcc_pi05_per_step     (one prefix build)

The guard: feed observation A, then a DIFFERENT observation B, and require TTNN to
respond to the change about as much as the PyTorch reference does. A plain final-PCC
threshold is too weak here (synthetic obs only dent it slightly), so we compare the
*responsiveness* — how much each backend's output changes from A to B. The stale
prefix-KV bug collapses TTNN's response (it reuses obs A's prefix).

Usage:
    pytest models/experimental/pi0_5/tests/pcc/test_pcc_pi05_multireplan.py -v -s
"""

from pathlib import Path

import pytest
import torch
import ttnn

from models.experimental.pi0_5.reference.torch_pi0_model import PI0Model as PI0ModelTorch
from models.experimental.pi0_5.tt.ttnn_pi0_model import PI0ModelTTNN
from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader
from models.experimental.pi0_5.common.configs import PI0ModelConfig, SigLIPConfig

CHECKPOINT_PATH = "lerobot/pi05_base"
SEED = 42
PCC_THRESHOLD = 0.93
# TTNN must respond to a new observation at least this fraction as much as torch does.
RESPONSE_RATIO_MIN = 0.6


def create_pi05_config() -> PI0ModelConfig:
    config = PI0ModelConfig(
        action_dim=32,
        action_horizon=50,
        state_dim=32,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        pi05=True,
    )
    config.siglip_config = SigLIPConfig(
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        image_size=224,
        patch_size=14,
    )
    return config


def compute_pcc(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    t1 = tensor1.flatten().float()
    t2 = tensor2.flatten().float()
    mean1, mean2 = torch.mean(t1), torch.mean(t2)
    std1, std2 = torch.std(t1), torch.std(t2)
    if std1 == 0 or std2 == 0:
        return 1.0 if torch.allclose(t1, t2) else 0.0
    covariance = torch.mean((t1 - mean1) * (t2 - mean2))
    return (covariance / (std1 * std2)).item()


def _resolve_checkpoint() -> Path:
    ckpt = Path(CHECKPOINT_PATH)
    if ckpt.is_absolute() and not ckpt.exists():
        pytest.skip(f"Checkpoint not found: {ckpt}")
    return ckpt


def _make_inputs(config: PI0ModelConfig, seed: int) -> dict:
    """A distinct (image, state, lang) observation, seeded for reproducibility."""
    g = torch.Generator().manual_seed(seed)
    image_size = config.siglip_config.image_size
    return {
        "images": [torch.randn(1, 3, image_size, image_size, generator=g) for _ in range(2)],
        "img_masks": [torch.ones(1, dtype=torch.bool) for _ in range(2)],
        "lang_tokens": torch.randint(0, 256000, (1, 32), generator=g),
        "lang_masks": torch.ones(1, 32, dtype=torch.bool),
        "state": torch.randn(1, config.state_dim, generator=g),
    }


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_pcc_pi05_multireplan(device):
    """TTNN must track torch on a SECOND sample_actions call with a new observation."""
    checkpoint = _resolve_checkpoint()
    config = create_pi05_config()
    weight_loader = PI0WeightLoader(str(checkpoint))

    model_torch = PI0ModelTorch(config, weight_loader)
    torch.manual_seed(SEED)
    model_ttnn = PI0ModelTTNN(config, weight_loader, device)

    # Fixed flow-matching noise: the ONLY thing that varies between the two calls is the
    # observation, so any cross-call divergence is a prefix-KV staleness bug.
    x0 = torch.randn(1, config.action_horizon, config.action_dim, dtype=torch.float32)

    inputs_a = _make_inputs(config, seed=1)
    inputs_b = _make_inputs(config, seed=2)  # a DIFFERENT observation

    def torch_sample(inp):
        saved = model_torch.denoising.sample_noise
        model_torch.denoising.sample_noise = lambda bs, device=None, dtype=torch.float32: x0.clone()
        try:
            with torch.no_grad():
                return model_torch.forward_inference(
                    images=inp["images"],
                    img_masks=inp["img_masks"],
                    lang_tokens=inp["lang_tokens"],
                    lang_masks=inp["lang_masks"],
                    state=inp["state"],
                )
        finally:
            model_torch.denoising.sample_noise = saved

    def ttnn_sample(inp):
        model_ttnn.x_t_ttnn = ttnn.from_torch(
            x0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        imgs = [
            ttnn.from_torch(
                im, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            for im in inp["images"]
        ]
        lt = ttnn.from_torch(inp["lang_tokens"], dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        lm = ttnn.from_torch(inp["lang_masks"].float(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        st = ttnn.from_torch(inp["state"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        with torch.no_grad():
            out = model_ttnn.sample_actions(
                images=imgs, img_masks=inp["img_masks"], lang_tokens=lt, lang_masks=lm, state=st
            )
        return ttnn.to_torch(out) if isinstance(out, ttnn.Tensor) else out

    # Torch reference for both observations.
    torch_a = torch_sample(inputs_a)
    torch_b = torch_sample(inputs_b)
    # The test is only meaningful if A and B genuinely produce different actions.
    assert compute_pcc(torch_a, torch_b) < 0.999, "observations A and B are not distinct enough"

    # TTNN: call A first (this populates the persistent prefix-KV buffer), THEN B.
    ttnn_a = ttnn_sample(inputs_a)
    ttnn_b = ttnn_sample(inputs_b)

    pcc_a = compute_pcc(torch_a, ttnn_a)
    pcc_b = compute_pcc(torch_b, ttnn_b)
    # Responsiveness: how much each backend's output changes from obs A -> obs B. The
    # stale-prefix-KV bug makes TTNN reuse obs A's prefix on the second call, so its
    # response to the observation change collapses relative to torch even while the
    # absolute PCC stays deceptively high. This ratio is the sensitive regression signal.
    resp_torch = 1.0 - compute_pcc(torch_a, torch_b)
    resp_ttnn = 1.0 - compute_pcc(ttnn_a, ttnn_b)
    ratio = resp_ttnn / resp_torch if resp_torch > 1e-6 else 1.0
    print(
        f"\n✅ multi-replan: PCC(A)={pcc_a:.4f} PCC(B)={pcc_b:.4f} | "
        f"resp_torch={resp_torch:.4f} resp_ttnn={resp_ttnn:.4f} ratio={ratio:.3f}"
    )

    assert pcc_a >= PCC_THRESHOLD, f"first-call PCC {pcc_a:.4f} < {PCC_THRESHOLD}"
    assert pcc_b >= PCC_THRESHOLD, f"second-call PCC {pcc_b:.4f} < {PCC_THRESHOLD}"
    # Primary regression guard: TTNN must respond to the new observation about as much as
    # torch does. A stale prefix-KV cache makes TTNN ignore the prefix (image/lang) change.
    assert ratio >= RESPONSE_RATIO_MIN, (
        f"TTNN under-responded to the new observation (only {ratio:.0%} of torch's response): "
        f"stale prefix-KV cache reused across sample_actions() calls."
    )
