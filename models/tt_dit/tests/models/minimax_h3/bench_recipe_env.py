# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bench-only (not for merge): pick the SDPA recipe of a MiniMax-H3 run from the environment.

H3_SDPA_RECIPE = FAST | STANDARD | LOW_PRECISION (unset: the model default)
H3_SDPA_KV     = bf16 | bfp8 | bfp4 (LOW_PRECISION K/V storage; default bf16)
"""

import os

import ttnn

_KV = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}


def recipe_from_env():
    name = os.environ.get("H3_SDPA_RECIPE")
    if not name:
        return None, None
    precision = getattr(ttnn.SDPAPrecision, name)
    kv = _KV[os.environ.get("H3_SDPA_KV", "bf16")]
    return precision, (kv if precision == ttnn.SDPAPrecision.LOW_PRECISION else None)


def recipe_tag():
    name = os.environ.get("H3_SDPA_RECIPE", "default")
    if name == "LOW_PRECISION":
        name += "_" + os.environ.get("H3_SDPA_KV", "bf16")
    return name
