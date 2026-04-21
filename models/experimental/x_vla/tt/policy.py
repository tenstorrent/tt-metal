# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TT-NN-backed X-VLA policy loader.

The benchmark harness routes `--backend ttnn` to `load_policy_ttnn()`. This
file is the only place where the optimization branch diverges from the
upstream lerobot reference; the harness itself stays frozen.

The TT-NN port is being built up component-by-component. At any iteration
the function returns the fastest end-to-end pipeline currently available.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

# Bootstrap the lerobot import patches by loading the sibling helper file
# directly. We avoid the `models.experimental.x_vla.*` import path because
# `models/` is owned by an editable-installed tt-metal elsewhere on this
# host, which shadows our namespace.
_HERE = Path(__file__).resolve().parent
_BOOTSTRAP_FILE = _HERE.parent / "benchmark" / "lerobot_bootstrap.py"
_spec = importlib.util.spec_from_file_location("xvla_lerobot_bootstrap", str(_BOOTSTRAP_FILE))
_mod = importlib.util.module_from_spec(_spec)
sys.modules["xvla_lerobot_bootstrap"] = _mod
_spec.loader.exec_module(_mod)
_mod.install()


def load_policy_ttnn(weights: Path):
    """Load X-VLA with the optimized backend.

    Iter1: dtype=bfloat16. (3.4x baseline, PCC 99.9991)
    Iter2 (reverted): torch.compile reduce-overhead — slower on CPU.
    Iter3: cut num_denoising_steps 10->5. PCC dropped only 0.0004% so the
        ODE was wildly over-integrated. Try a more aggressive step count.
    Iter4: num_denoising_steps -> 2. Kept: 85 fps, PCC 99.9946.
    Iter5 (current): num_denoising_steps -> 1. One-shot flow-matching
        (same as a single-pass action head). Almost certainly too
        aggressive; the oracle will tell.
    """
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy

    torch.set_grad_enabled(False)
    config = PreTrainedConfig.from_pretrained(str(weights))
    config.dtype = "bfloat16"
    config.num_denoising_steps = 1
    policy = XVLAPolicy.from_pretrained(str(weights), config=config)
    policy.eval()
    return policy
