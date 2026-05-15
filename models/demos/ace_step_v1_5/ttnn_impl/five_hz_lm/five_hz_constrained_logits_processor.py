# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Re-export of ``five_hz_lm.five_hz_constrained_logits_processor`` (TTNN-aware branches).

See ``ttnn_impl/five_hz_lm/README.md`` for the layout vs ``torch_ref/five_hz_lm``.
"""

from models.demos.ace_step_v1_5.five_hz_lm.five_hz_constrained_logits_processor import *  # noqa: F403
