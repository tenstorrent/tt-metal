# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""5 Hz LM handler entrypoint for tt-metal ACE-Step demos (no ``acestep`` package imports).

The implementation is vendored under ``five_hz_lm`` so the demo can run when
ACE-Step is not installed as a Python package. Checkpoint weights are still loaded
from disk (HuggingFace layout).
"""

from .five_hz_llm_inference import LocalFiveHzLMHandler

__all__ = ["LocalFiveHzLMHandler"]
