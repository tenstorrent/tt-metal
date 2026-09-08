# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The drafter's inference path must stay entirely on device. No device needed to check.

Setup may use host freely -- weights are loaded there, the RoPE table is built there once,
and masks are pure functions of shape built once per context length. But once a forward
begins, nothing may round-trip: no ``ttnn.to_torch``/``from_torch``, no ``torch`` ops, no
``.item()`` (which forces a device read and serialises the pipeline).

Enforced by source inspection rather than by review, because a single ``to_torch`` added for
debugging is easy to leave behind and costs a device sync per call.
"""

from __future__ import annotations

import inspect
import re

import pytest

from models.demos.blackhole.qwen36.tt.dflash.attention import DFlashAttention
from models.demos.blackhole.qwen36.tt.dflash.drafter import DFlashDrafter
from models.demos.blackhole.qwen36.tt.dflash.encoder import DFlashContextEncoder
from models.demos.blackhole.qwen36.tt.dflash.layer import DFlashLayer
from models.demos.blackhole.qwen36.tt.dflash.mlp import DFlashMLP
from models.demos.blackhole.qwen36.tt.dflash.rope import DFlashRoPE

# Every method reachable from DFlashDrafter.forward.
INFERENCE_METHODS = [
    (DFlashDrafter, "forward"),
    (DFlashContextEncoder, "forward"),
    (DFlashLayer, "forward"),
    (DFlashLayer, "project_context"),
    (DFlashAttention, "forward"),
    (DFlashAttention, "project_context"),
    (DFlashMLP, "forward"),
    (DFlashRoPE, "apply"),
    (DFlashRoPE, "tables_for"),
]

FORBIDDEN = [
    (r"\bto_torch\b", "reads the tensor back to host"),
    (r"\bfrom_torch\b", "uploads from host mid-forward"),
    (r"\btorch\.\w+", "host torch op"),
    (r"\.item\(\)", "forces a device read and serialises the pipeline"),
    (r"\bnumpy\b", "host numpy op"),
]


@pytest.mark.parametrize(
    "cls, method",
    INFERENCE_METHODS,
    ids=[f"{c.__name__}.{m}" for c, m in INFERENCE_METHODS],
)
def test_inference_path_is_device_only(cls, method):
    src = inspect.getsource(getattr(cls, method))
    # Strip comments and docstrings -- they legitimately discuss torch and to_torch.
    body = re.sub(r'""".*?"""', "", src, flags=re.DOTALL)
    body = "\n".join(line.split("#")[0] for line in body.splitlines())

    for pattern, why in FORBIDDEN:
        found = re.search(pattern, body)
        assert not found, f"{cls.__name__}.{method} uses {found.group(0)!r} -- {why}"


def test_mask_cache_is_prewarmable():
    """``prewarm_masks`` exists so the timed path never builds a mask.

    Mask construction is host-side by nature (it is pure shape arithmetic), so the contract
    is that it happens ahead of time, not that it never happens.
    """
    assert callable(DFlashDrafter.prewarm_masks)
    src = inspect.getsource(DFlashDrafter.forward)
    assert "_mask_for" in src, "forward no longer goes through the mask cache"
