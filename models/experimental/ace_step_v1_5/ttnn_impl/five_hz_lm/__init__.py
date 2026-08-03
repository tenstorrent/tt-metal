# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Vendored ACE-Step 5 Hz language model stack for tt-metal ACE-Step demos."""

from .local_five_hz_llm import LocalFiveHzLMHandler

__all__ = ["LocalFiveHzLMHandler"]
