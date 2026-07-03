# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Layer-completion aggregation for pipelined prefill (deepseek_v3_d_p, num_ranks > 1).

This is exclusive to the pipelined-prefill runner — NOT a ttnn API and not general tt_metal infra.
The C++ (host-local SHM ring, reorder buffer, MPI router, and the test-only scheduler stand-in
consumer) is built into a standalone `_layer_completion` nanobind extension that links only
TT::Metalium; its .so is emitted into the build tree and symlinked next to this file (like _ttnn).
Import is intentionally at point-of-use in the runner so single-rank / no-extension builds don't fail.
"""

from models.demos.common.prefill.runners.pipelined_prefill._layer_completion import (
    LayerCompletionConsumer,
    LayerCompletionQueue,
    LayerCompletionRouter,
)

__all__ = ["LayerCompletionConsumer", "LayerCompletionQueue", "LayerCompletionRouter"]
