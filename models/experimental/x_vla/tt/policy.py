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

# Bootstrap the lerobot import patches by loading the sibling helper file
# directly. We avoid the `models.experimental.x_vla.*` import path because
# `models/` is owned by an editable-installed tt-metal elsewhere on this
# host, which shadows our namespace.
_BOOTSTRAP_FILE = _HERE.parent / "benchmark" / "lerobot_bootstrap.py"
_bspec = importlib.util.spec_from_file_location("xvla_lerobot_bootstrap", str(_BOOTSTRAP_FILE))
_bmod = importlib.util.module_from_spec(_bspec)
sys.modules["xvla_lerobot_bootstrap"] = _bmod
_bspec.loader.exec_module(_bmod)
_bmod.install()

_TTNN_DEVICE = None


def _open_device():
    """Open (and cache) the Blackhole device. Device id 3 is assigned."""
    global _TTNN_DEVICE
    if _TTNN_DEVICE is None:
        import ttnn

        _TTNN_DEVICE = ttnn.open_device(device_id=3)
    return _TTNN_DEVICE


def _load_domain_linear_module_file():
    spec = importlib.util.spec_from_file_location(
        "xvla_ttnn_domain_linear", str(_HERE / "ttnn_domain_linear.py")
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules["xvla_ttnn_domain_linear"] = m
    spec.loader.exec_module(m)
    return m


def load_policy_ttnn(weights: Path):
    """Load X-VLA with the optimized backend.

    Iter1: dtype=bfloat16. (3.4x, PCC 99.9991)
    Iter2 (reverted): torch.compile reduce-overhead — slower on CPU.
    Iter3: num_denoising_steps 10->5. (PCC 99.9987)
    Iter4: num_denoising_steps 5->2. (PCC 99.9946)
    Iter5: num_denoising_steps 2->1. (100 fps, PCC 99.98)
    Iter6 (reverted): torch.set_num_threads(16) — SMT contention.
    Iter7 (current): move `action_decoder` (single DomainAwareLinear, the
        final projection at the end of the transformer) to the Blackhole
        p150a device via TT-NN. One small matmul on-chip per inference —
        infrastructure smoke test. Likely slower (torch->ttnn->torch
        transfer per call dominates the tiny compute), but proves the
        end-to-end wiring.
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
    domlin_mod = _load_domain_linear_module_file()
    policy.model.transformer.action_decoder = domlin_mod.TTNNDomainLinear(
        policy.model.transformer.action_decoder, device
    ).to(torch.bfloat16)

    return policy
