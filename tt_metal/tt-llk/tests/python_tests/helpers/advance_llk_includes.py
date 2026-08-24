# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Per-test include paths for the demo-fork experimental-LLK ADVANCE TESTS.

The three advance tests (tt-metal#47554 / tt-blaze#1971) resolve their primitive headers out of a
shadow tree under `models/demos/deepseek_v3_b1/kernel_includes/`. These roots are a TEMPORARY
scaffold: on promotion into `tt_llk_blackhole/{llk_lib,common/inc/sfpu}/experimental/` the canonical
`-I` already covers them and the matching entry here gets deleted.

Because they are temporary, they are appended per test rather than pushed into
`TestConfig.INCLUDES` from `pytest_configure` -- `INCLUDES` is a session-wide ClassVar, so a scaffold
parked there would land in the compile command for *every* Blackhole test, not just these three.
This mirrors the `compressed_mm_include_paths` fixture in `test_matmul_custom_compressed.py`, which
scopes the byte-identical root 1 the same way.

Each advance test imports `advance_llk_include_paths` at module level; pytest picks it up as an
autouse fixture for that module only.
"""

import pytest

from .test_config import TestConfig

# Appended LAST so canonical llk_lib/sfpu headers win on any name collision; none of the shadow
# filenames collide with canonical Blackhole headers.
#   root 1 -- compressed_custom_mm
#   root 2 -- sdpa_bcast_col_srcb_reuse, sdpa_bcast_col_srca_srcb_reuse
# The sfpu sibling root that sdpa_reduce_row needed went with that test (owned by #53361).
ADVANCE_LLK_INCLUDES = [
    "-I../../../models/demos/deepseek_v3_b1/kernel_includes/tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib",
    "-I../../../models/demos/deepseek_v3_b1/kernel_includes/tt_llk/tt_llk_blackhole/llk_lib",
]


@pytest.fixture(autouse=True)
def advance_llk_include_paths():
    added = [inc for inc in ADVANCE_LLK_INCLUDES if inc not in TestConfig.INCLUDES]
    TestConfig.INCLUDES.extend(added)
    yield
    for inc in added:
        TestConfig.INCLUDES.remove(inc)
