# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Thin performant runners around ACE-Step v1.5 TTNN models.

Mirrors the ``models/experimental/swin_v2/runner`` package: callers construct a
runner with a device handle, and the runner auto-captures the appropriate TTNN
trace + 2CQ replay buffers in ``__init__`` so subsequent ``run(...)`` calls land
on the steady-state perf path without any extra setup or env-var ceremony.
"""

from models.demos.ace_step_v1_5.runner.performant_runner import AceStepPerformantRunner

__all__ = ["AceStepPerformantRunner"]
