# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""5 Hz LM package used with TTNN assist; re-exports the canonical ``five_hz_lm`` implementation."""

from .five_hz_llm_inference import LocalFiveHzLMHandler

__all__ = ["LocalFiveHzLMHandler"]
