# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TT-NN-backed X-VLA policy loader.

The benchmark harness routes `--backend ttnn` to `load_policy_ttnn()`. This
file is the only place where the optimization branch diverges from the
upstream lerobot reference; the harness itself stays frozen.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# Make ttnn imports route to the xvla tt-metal tree. Must happen before the
# first `import ttnn` anywhere in the process.
_HERE = Path(__file__).resolve().parent
_TTNN_ENV_FILE = _HERE / "ttnn_env.py"
_spec = importlib.util.spec_from_file_location("xvla_ttnn_env", str(_TTNN_ENV_FILE))
_mod = importlib.util.module_from_spec(_spec)
sys.modules["xvla_ttnn_env"] = _mod
_spec.loader.exec_module(_mod)
_mod.install()

import torch  # noqa: E402

# Bootstrap the lerobot import patches.
_BOOTSTRAP_FILE = _HERE.parent / "benchmark" / "lerobot_bootstrap.py"
_bspec = importlib.util.spec_from_file_location("xvla_lerobot_bootstrap", str(_BOOTSTRAP_FILE))
_bmod = importlib.util.module_from_spec(_bspec)
sys.modules["xvla_lerobot_bootstrap"] = _bmod
_bspec.loader.exec_module(_bmod)
_bmod.install()

_TTNN_DEVICE = None


def _open_device():
    """Open (and cache) the Blackhole device at id 3."""
    global _TTNN_DEVICE
    if _TTNN_DEVICE is None:
        import ttnn

        _TTNN_DEVICE = ttnn.open_device(device_id=3)
    return _TTNN_DEVICE


def _load_module_file(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, str(_HERE / filename))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def load_policy_ttnn(weights: Path):
    """Load X-VLA with the optimized backend.

    Iter1-iter16: see results.tsv. Best so far: iter10 = 118.4 fps, PCC 99.98%
        (24 SoftPromptedTransformer blocks on chip, MLP weights bfp8_b).
    Iter17 (current): also offload Florence-2's 12-layer BART encoder to
        the Blackhole device. This is ~75% of the remaining torch CPU
        time. Each layer is post-LN BART (fused QKV, attention with
        additive mask, output proj, FFN). Embedding lookups stay on
        torch (small integer-indexed tables).
    """
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy

    torch.set_grad_enabled(False)
    config = PreTrainedConfig.from_pretrained(str(weights))
    config.dtype = "bfloat16"
    config.num_denoising_steps = 1
    policy = XVLAPolicy.from_pretrained(str(weights), config=config)
    policy.eval()

    device = _open_device()
    stack_mod = _load_module_file("xvla_ttnn_block_stack", "ttnn_block_stack.py")
    transformer = policy.model.transformer
    transformer.blocks = stack_mod.TTNNTransformerBlockStack(
        transformer.blocks, device, num_heads=transformer.blocks[0].attn.num_heads
    )

    # Replace Florence-2 BART encoder with on-device version.
    bart_mod = _load_module_file("xvla_ttnn_bart_encoder", "ttnn_bart_encoder.py")
    lm = policy.model.vlm.language_model
    lm.model.encoder = bart_mod.TTNNBartEncoder(lm.model.encoder, device).to(torch.bfloat16)

    return policy
