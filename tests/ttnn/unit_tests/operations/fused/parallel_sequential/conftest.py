# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

# The layernorm factories a fusion branch drives directly.
_LAYERNORM_FACTORIES = (
    ttnn.LayerNormMultiCoreProgramFactory,
    ttnn.LayerNormShardedProgramFactory,
)


@pytest.fixture(autouse=True)
def _skip_branches_needing_layernorm_descriptors(monkeypatch):
    """Skip tests that ask a layernorm factory for a ``create_descriptor`` it no longer has.

    A fusion branch drives a factory's ``create_descriptor`` and consumes the ``ProgramDescriptor``
    it returns; both layernorm factories now produce a ``ProgramSpec``, which no branch can consume
    yet. Standing in for the missing method, rather than marking whole tests, skips only the tests
    that actually reach the ``create_descriptor`` call, and will become a no-op once a factory
    exposes it again. Issue #54365.
    """
    missing = [factory for factory in _LAYERNORM_FACTORIES if not hasattr(factory, "create_descriptor")]
    if not missing:
        return

    def _no_descriptor(*_args, **_kwargs):
        pytest.skip("layernorm factories produce a ProgramSpec; a fusion branch needs a ProgramDescriptor")

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
