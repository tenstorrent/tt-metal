# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Reuse the generic tt_transformers text-demo CLI options (--input_prompts,
# --max_seq_len, --enable_trace, ...) so the Gemma-2 demo behaves identically.
from models.tt_transformers.demo.conftest import pytest_addoption  # noqa: F401
