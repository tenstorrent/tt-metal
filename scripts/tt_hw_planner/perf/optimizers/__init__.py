# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Optimizer-block catalog.

Each module under `optimizers/` defines exactly one block that implements
the `OptimizerBlock` protocol from `base.py`. Blocks are registered in
`catalog.py`; the report sidebar and the `perf blocks` CLI both read the
catalog so users see a single source of truth.
"""

from .base import (  # noqa: F401
    Finding,
    OptimizerBlock,
    Patch,
    PatchKind,
    Severity,
    SourceLocation,
    VerificationResult,
)
from .catalog import (  # noqa: F401
    catalog_for_sidebar,
    get_block,
    list_blocks,
)
