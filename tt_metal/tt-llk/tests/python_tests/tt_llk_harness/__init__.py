# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The supported interface for driving this harness from another repository.

Suites outside tt-llk import from here and from nowhere else. Everything under
``helpers`` is implementation: it can be renamed, split, or restructured
without notice, and it regularly is. This module is the part that does not
change without a deprecation period.

Usage::

    # conftest.py, at the pytest rootdir of your suite
    import sys
    sys.path.insert(0, "<tt-llk>/tests/python_tests")

    import tt_llk_harness
    pytest_plugins = ["tt_llk_harness.plugin"]

    from tt_llk_harness import TestConfig
    TestConfig.add_include_dirs(my_headers)

Two kinds of thing live here, and they carry different promises.

**Mechanism** — the names listed in ``__all__``. This is the contract: how a
suite attaches to the harness, configures search paths, describes a variant,
and runs it. Removing or changing the meaning of one of these is a breaking
change and needs a deprecation period plus a migration note in
``docs/tests/getting_started.md``.

**Catalogues** — ``goldens`` and ``params``. These are open sets that grow with
the LLKs under test, so enumerating them here would be churn with no benefit.
Reach a specific entry through the alias (``goldens.DataCopyGolden``,
``params.DEST_INDEX``). Adding an entry is not a breaking change; removing or
renaming one is, and is reviewed the same way.

``tests/out_of_tree/`` is a consumer that uses only this module, and it runs in
tt-llk's PR CI. If something a real consumer needs is missing here, that suite
is where it should be noticed.
"""

from __future__ import annotations

import sys

# --- Catalogues: open sets, reached through a stable alias --------------------
from helpers import golden_generators as goldens
from helpers import test_variant_parameters as params

# Register the aliases as real submodules so both spellings work:
#
#     from tt_llk_harness import params;  params.DEST_INDEX(0)
#     from tt_llk_harness.params import DEST_INDEX;  DEST_INDEX(0)
#
# The second matters more than it looks: in-tree tests spell it that way, and
# lifting one into an out-of-tree suite is a normal thing to do. Forcing a
# rewrite at that moment is how people end up importing ``helpers`` instead.
sys.modules[f"{__name__}.goldens"] = goldens
sys.modules[f"{__name__}.params"] = params
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    ELEMENTS_PER_TILE,
    TILE_DIM,
    golden_registry,
    register_golden,
    round_to_dest_width,
)

# NOTE: ``get_golden_generator`` is deliberately absent from that list. See the
# ``__getattr__`` at the bottom of this module.
from helpers.llk_params import (
    BlocksCalculationAlgorithm,
    DestAccumulation,
    DestSync,
    L1Accumulation,
    Tilize,
    VectorMode,
    format_dict,
)

# --- Arch markers -------------------------------------------------------------
from helpers.llk_pytest_plugin import (
    blackhole_only,
    quasar_only,
    skip_for_blackhole,
    skip_for_coverage,
    skip_for_quasar,
    skip_for_wormhole,
    wormhole_only,
)
from helpers.param_config import (
    get_num_blocks_and_num_tiles_in_block,
    input_output_formats,
    parametrize,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli

# --- Mechanism ----------------------------------------------------------------
from helpers.test_config import TestConfig
from helpers.tilize_untilize import tilize_block, untilize_block
from helpers.utils import passed_test

#: Name to put in your suite's ``pytest_plugins``. Prefer the literal
#: ``"tt_llk_harness.plugin"``; this constant exists so tooling that builds the
#: list programmatically does not hardcode an implementation module path.
PYTEST_PLUGIN = "tt_llk_harness.plugin"

#: Version of *this interface*, not of tt-llk. Bumped when the contract
#: changes: minor for additions, major for a breaking change that has completed
#: its deprecation period. See :func:`require_version`.
HARNESS_API_VERSION = (1, 0)


def require_version(major: int, minor: int = 0) -> None:
    """Fail fast, and legibly, if the harness is incompatible with this suite.

    Without this a consumer pinned to an older submodule discovers the mismatch
    as an ``AttributeError`` or a missing keyword argument somewhere deep in a
    test run. Call it from your ``conftest.py`` right after importing::

        tt_llk_harness.require_version(1, 0)

    Compatible means *same major, minor at least as new as requested*. A newer
    major is a rejection, not an upgrade: a major bump only happens for a
    breaking change that has completed its deprecation period, so a suite
    written against 1.x has no reason to expect 2.x to work. Treating "newer is
    always fine" as compatible is the failure mode this function exists to
    prevent — it would let a consumer pass the advertised gate and then break
    later, in a worse place.
    """
    have_major, have_minor = HARNESS_API_VERSION
    if major != have_major:
        raise RuntimeError(
            f"This suite needs tt-llk harness API {major}.x, but the harness on "
            f"LLK_HOME provides {have_major}.{have_minor}. A different major "
            "version means a breaking change to the out-of-tree contract: see "
            "the migration note in docs/tests/getting_started.md §9, then move "
            "the suite to the new spelling."
        )
    if minor > have_minor:
        raise RuntimeError(
            f"This suite needs tt-llk harness API {major}.{minor}, but the "
            f"harness on LLK_HOME provides {have_major}.{have_minor}. Bump the "
            "tt-llk submodule, or stop using whatever was added in "
            f"{major}.{minor}."
        )


__all__ = [
    # attaching
    "PYTEST_PLUGIN",
    "HARNESS_API_VERSION",
    "require_version",
    # configuring and running
    "TestConfig",
    "StimuliConfig",
    "generate_stimuli",
    # describing a variant
    "DataFormat",
    "input_output_formats",
    "parametrize",
    "get_num_blocks_and_num_tiles_in_block",
    "BlocksCalculationAlgorithm",
    "DestAccumulation",
    "DestSync",
    "L1Accumulation",
    "Tilize",
    "VectorMode",
    "format_dict",
    # goldens
    "register_golden",
    "get_golden_generator",
    "golden_registry",
    "round_to_dest_width",
    "TILE_DIM",
    "ELEMENTS_PER_TILE",
    # checking results
    "passed_test",
    "tilize_block",
    "untilize_block",
    # arch
    "ChipArchitecture",
    "get_chip_architecture",
    "blackhole_only",
    "wormhole_only",
    "quasar_only",
    "skip_for_blackhole",
    "skip_for_wormhole",
    "skip_for_quasar",
    "skip_for_coverage",
    # catalogues
    "goldens",
    "params",
]


# --- Names the harness rebinds at runtime -------------------------------------
#
# ``TestConfig.setup_mode`` swaps ``helpers.golden_generators.get_golden_generator``
# during ``pytest_configure``: a stand-in generator under ``--compile-producer``,
# a caching proxy under ``--stimuli-only`` / ``--use-stimuli``. This package is
# imported *earlier* than that — a consumer's rootdir ``conftest.py`` declares
# ``pytest_plugins = ["tt_llk_harness.plugin"]``, which imports it — so binding
# the name eagerly would freeze the pre-swap function.
#
# The damage from that is quiet: ``tt_llk_harness.goldens.get_golden_generator``
# would see the swap while the flat ``__all__`` name would not, so two spellings
# this module presents as equivalent would disagree. An out-of-tree suite would
# compute real goldens under ``--compile-producer`` instead of the stand-in, and
# ``--stimuli-only`` would fail inside the proxy.
#
# Resolving through ``__getattr__`` defers the lookup to attribute-access time,
# which for a consumer's test module is collection time — after
# ``pytest_configure``, same as for in-tree tests.
_REBOUND_AT_RUNTIME = frozenset({"get_golden_generator"})


def __getattr__(name: str):
    if name in _REBOUND_AT_RUNTIME:
        return getattr(goldens, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
