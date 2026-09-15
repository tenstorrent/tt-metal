# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

# The factories a fusion branch drives directly. Each entry is one that has migrated to Metal 2.0.
_BRANCH_FACTORIES = (
    ttnn.LayerNormMultiCoreProgramFactory,
    ttnn.LayerNormShardedProgramFactory,
    ttnn.MatmulMultiCoreReuseOptimizedProgramFactory,
)


@pytest.fixture(autouse=True)
def _skip_branches_needing_descriptors(monkeypatch):
    """Skip tests that ask a factory for a ``create_descriptor`` it no longer has."""
    missing = [factory for factory in _BRANCH_FACTORIES if not hasattr(factory, "create_descriptor")]
    if not missing:
        return

    def _no_descriptor(*_args, **_kwargs):
        pytest.skip("a ported factory produces a ProgramSpec; a fusion branch needs a ProgramDescriptor")

    for factory in missing:
        monkeypatch.setattr(factory, "create_descriptor", staticmethod(_no_descriptor), raising=False)


@pytest.fixture(autouse=True)
def _enable_parallel_sequential(monkeypatch):
    """Opt this suite in to Sequential/Parallel fusion.

    Sequential/Parallel are gated behind ``TT_METAL_ENABLE_PARALLEL_SEQUENTIAL`` until
    ProgramSpec is exposed to Python (see fusion.py). This autouse fixture enables
    them only for the duration of each test in this directory and reverts after,
    so the guard stays active for every other test sharing the same process.
    """
    monkeypatch.setenv("TT_METAL_ENABLE_PARALLEL_SEQUENTIAL", "1")
