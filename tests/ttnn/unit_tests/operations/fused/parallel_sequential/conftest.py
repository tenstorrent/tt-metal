# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import os

# Opt in to Sequential/Parallel fusion for this test suite.
# See models/experimental/ops/descriptors/fusion/fusion.py for context.
os.environ.setdefault("TTNN_ENABLE_PARALLEL_SEQUENTIAL", "1")
