# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Model-agnostic, test-only access to the C++ layer-completion consumer (NOT a public ttnn API).

The native-thread ``LayerCompletionConsumer`` (a GIL-immune scheduler stand-in that drains the master
router's scheduler counter channel) is built as a standalone ``_prefill_test`` nanobind extension —
see ``models/demos/test``. It is deliberately NOT part of the ttnn module/API; its ``.so`` is
emitted next to this file by the build. Re-export it here under a clean name for the prefill runner's
completion check.
"""

from models.demos.test._prefill_test import LayerCompletionConsumer

__all__ = ["LayerCompletionConsumer"]
