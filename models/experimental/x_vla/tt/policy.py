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

    Iter1: dtype=bfloat16.                          (3.4x, PCC 99.9991)
    Iter2 (reverted): torch.compile reduce-overhead — slower on CPU.
    Iter3: num_denoising_steps 10->5.               (PCC 99.9987)
    Iter4: num_denoising_steps 5->2.                (PCC 99.9946)
    Iter5: num_denoising_steps 2->1.                (100 fps, PCC 99.98)
    Iter6 (reverted): torch.set_num_threads(16)     — SMT contention.
    Iter7 (reverted): action_decoder on TT-NN       — -1.8% (too small).
    Iter8 (current): all 24 TransformerBlock layers inside
        SoftPromptedTransformer are offloaded to the Blackhole p150a at
        device_id=3. The transformer runs as: torch -> ttnn (upload) ->
        24x (LayerNorm, fused-QKV linear, SDPA, proj, residual, LayerNorm,
        MLP, residual) -> torch (download). One round-trip per inference
        regardless of layer count, so PCIe cost is amortized. With 308M
        params and 24 layers the on-chip compute vastly exceeds the
        transfer cost; expected to beat iter5's 100 fps.
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

    return policy
