# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""llkaudit — Python detector layer for the LLK race-audit recall tool.

The C++ `extractor/llk_extract` emits a semantics-free fact base; the checkers
here classify it (via `registry`) into recall candidates. Every checker is an
augmentor — recall of known patterns, advisory to the /*-audit skills, never a
gate. See README.md.
"""
