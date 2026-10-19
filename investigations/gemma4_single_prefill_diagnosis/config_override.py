# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_call(item):
    from models.demos.gemma4_d_p.tt.attention import striped_prefill
    from models.demos.gemma4_d_p.tt.attention.ring_prefill import ring_sdpa_chunk_sizes

    variant = os.environ["GEMMA4_PERF_VARIANT"]
    assert variant in ("global_legacy", "swa_k128"), variant
    original = striped_prefill.ring_prefill_program_config

    def configure(mesh, ccl, head_dim, q_chunk, k_chunk, **kwargs):
        if variant == "global_legacy" and head_dim == 512:
            q_chunk, k_chunk, splits, segmented = ring_sdpa_chunk_sizes(
                int(os.getenv("GEMMA4_PERF_CHUNK_SIZE", "8192")) // 8,
                sliding=False,
                num_heads=8,
                num_cores=(ccl.compute_grid_size.x - 1) * ccl.compute_grid_size.y,
            )
            kwargs.update(max_k_splits=splits, segmented_accumulation=segmented)
        if variant == "swa_k128" and head_dim == 256:
            k_chunk = 128
        return original(mesh, ccl, head_dim, q_chunk, k_chunk, **kwargs)

    item.funcargs["monkeypatch"].setattr(striped_prefill, "ring_prefill_program_config", configure)
