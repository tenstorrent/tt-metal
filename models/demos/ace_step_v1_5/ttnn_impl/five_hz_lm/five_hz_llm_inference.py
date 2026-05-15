# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Re-export of the production 5 Hz LM handler (``five_hz_lm.five_hz_llm_inference``).

The implementation lives under ``models.demos.ace_step_v1_5.five_hz_lm`` together with
``ttnn_impl/lm_*`` helpers. This package exists so PCC tests can import a **symmetric** pair:

- ``models.demos.ace_step_v1_5.torch_ref.five_hz_lm`` — PyTorch-only reference
- ``models.demos.ace_step_v1_5.ttnn_impl.five_hz_lm`` — same API with TTNN logits assist
"""

from models.demos.ace_step_v1_5.five_hz_lm.five_hz_llm_inference import *  # noqa: F403
