# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

# The program factories a fusion branch drives directly. Each entry is a factory whose op may
# already have moved to Metal 2.0; the fixture below checks rather than assumes.
_FUSION_FACTORIES = (
    ttnn.LayerNormMultiCoreProgramFactory,
    ttnn.LayerNormShardedProgramFactory,
    ttnn.SliceTileProgramFactory,
)


@pytest.fixture(autouse=True)
def _skip_branches_needing_legacy_descriptors(monkeypatch):
    """Skip tests that ask a factory for a ``create_descriptor`` it no longer has.

    A fusion branch drives a factory's ``create_descriptor`` and consumes the ``ProgramDescriptor``
    it returns. A factory whose op has moved to Metal 2.0 produces a ``ProgramSpec`` instead, which
    no branch can consume yet. Standing in for the missing method, rather than marking whole tests,
    skips only the tests that actually reach the ``create_descriptor`` call, and becomes a no-op
    once a factory exposes it again. Issue #54365.
    """
    missing = [factory for factory in _FUSION_FACTORIES if not hasattr(factory, "create_descriptor")]
    if not missing:
        return

    def _no_descriptor(*_args, **_kwargs):
        pytest.skip("this factory produces a ProgramSpec; a fusion branch needs a ProgramDescriptor")

    for factory in missing:
        monkeypatch.setattr(factory, "create_descriptor", staticmethod(_no_descriptor), raising=False)


@pytest.fixture(autouse=True)
def _skip_branches_needing_matmul_descriptors(monkeypatch):
    """Skip tests whose matmul branch selects a factory that has migrated to Metal 2.0.

    A matmul factory that no longer produces a ``ProgramDescriptor`` is not bound to Python at all,
    so the call to ``matmul_select_program_factory`` raises a ``TypeError`` before the branch ever
    asks for a descriptor.
    """
    select_program_factory = ttnn.matmul_select_program_factory

    def _select(*args, **kwargs):
        try:
            return select_program_factory(*args, **kwargs)
        except TypeError as exc:
            # Only the unregistered-alternative conversion failure; anything else is a real error.
            if "Unable to convert function return value" not in str(exc):
                raise
            pytest.skip("the selected matmul factory produces a ProgramSpec; a fusion branch needs a ProgramDescriptor")

    monkeypatch.setattr(ttnn, "matmul_select_program_factory", _select)


@pytest.fixture(autouse=True)
def _enable_parallel_sequential(monkeypatch):
    """Opt this suite in to Sequential/Parallel fusion.

    Sequential/Parallel are gated behind ``TT_METAL_ENABLE_PARALLEL_SEQUENTIAL`` until
    ProgramSpec is exposed to Python (see fusion.py). This autouse fixture enables
    them only for the duration of each test in this directory and reverts after,
    so the guard stays active for every other test sharing the same process.
    """
    monkeypatch.setenv("TT_METAL_ENABLE_PARALLEL_SEQUENTIAL", "1")
