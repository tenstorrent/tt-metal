# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Construction-only sanity test for TtQwen36DeltaAttention.

This is a stub for the V2-7 phase. Real device construction requires a
live BH GLX 8×4 mesh, so we skip unless ARCH_NAME indicates blackhole. The
file's purpose is to be in place for the eventual full device test.
"""
import os

import pytest


@pytest.mark.skipif(
    os.getenv("ARCH_NAME") != "blackhole",
    reason="requires BH GLX mesh; placeholder for V2-7 phase",
)
def test_delta_attention_init_smoke():
    """Construct TtQwen36DeltaAttention on a real BH GLX 8x4 mesh.

    Skipped in CI / on non-BH hardware. Filled in during V2-7.
    """
    pytest.skip("requires BH GLX mesh")
