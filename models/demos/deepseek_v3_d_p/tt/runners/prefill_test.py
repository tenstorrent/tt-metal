# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Internal test-only access to the C++ layer-completion consumer (NOT a public ttnn API).

The native-thread ``LayerCompletionConsumer`` is bound by the ttnn deepseek_prefill ``prefill_test``
component (a GIL-immune scheduler stand-in that drains the master router's scheduler counter channel).
``nb::class_`` lands on the experimental module — there is no ``bind_function`` namespace machinery for
classes — so it is exposed at ``ttnn._ttnn.operations.experimental.LayerCompletionConsumer``. Rather
than surface that internal helper in the public ``ttnn`` namespace, re-export it here under a clean
runner-local name; it is used only by the prefill runner's completion check.
"""

import ttnn

LayerCompletionConsumer = ttnn._ttnn.operations.experimental.LayerCompletionConsumer

__all__ = ["LayerCompletionConsumer"]
