# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tenstorrent device module for the Kimi-K2.6-DFlash speculative-decoding drafter's PREFILL path.

Front-loads the DFlash drafter's *context* KV cache during the verifier (Kimi/DeepSeek MLA) prefill.
Kimi-K2.6-DFlash-specific: the drafter dims and deepseek-yarn rope defaults live in
``dflash_drafter_config.py``. See issue #49586.
"""
