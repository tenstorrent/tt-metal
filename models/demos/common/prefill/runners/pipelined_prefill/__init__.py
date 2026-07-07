# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Layer-completion aggregation for pipelined prefill (deepseek_v3_d_p, num_ranks > 1).

This is exclusive to the pipelined-prefill runner — NOT a ttnn API. The C++ implementation lives in
`tt_metal/distributed/layer_completion/` and is built into a standalone `_layer_completion`
nanobind extension that links only TT::Metalium; the .so is installed next to this file during the
CMake install step (like _ttnn). Import is intentionally at point-of-use in the runner so single-rank
/ no-extension builds don't fail.
"""

from models.demos.common.prefill.runners.pipelined_prefill._layer_completion import (
    LayerCompletionConsumer,
    LayerCompletionQueue,
    LayerCompletionRouter,
)

__all__ = ["LayerCompletionConsumer", "LayerCompletionQueue", "LayerCompletionRouter"]
