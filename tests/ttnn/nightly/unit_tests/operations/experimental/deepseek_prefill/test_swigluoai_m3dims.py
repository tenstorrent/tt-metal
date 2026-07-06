# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Confirm the fused swiglu_oai routed expert at MiniMax-M3 dims (emb=6144, hidden=3072) — the shapes
where the composite path produced the token-0 stale bf8 block at output channels 4128-4143 (N=6144).
Reuses Maciek's harness; just drives it at M3 dims. PCC vs torch swigluoai + no nan/inf is the pre-flight
that the fused op writes the full 6144-wide output with no stale tile-face, before the full-model run."""

import pytest
from models.common.utility_functions import is_blackhole

from .test_swigluoai_routed_expert import SINGLE_CHIP_MESH_PARAMS, run_swigluoai_routed_expert

EMB_M3, HID_M3 = 6144, 3072  # MiniMax-M3 hidden_size / moe intermediate_size


@pytest.mark.skipif(not is_blackhole(), reason="unified_routed_expert op is Blackhole-only")
@pytest.mark.parametrize("num_tokens", [128, 1024, 2048], ids=["t128", "t1k", "t2k"])
@pytest.mark.parametrize(
    "mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
def test_swigluoai_routed_expert_m3dims(mesh_device, device_params, num_tokens):
    """M3-shape (6144/3072) clamped swigluoai vs torch — confirms full-width output, no stale block."""
    run_swigluoai_routed_expert(
        mesh_device, num_tokens=num_tokens, emb_dim=EMB_M3, hidden_dim=HID_M3, swiglu_oai=True
    )
