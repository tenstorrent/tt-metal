# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

# Import RoPE utilities directly from tt-transformers to avoid duplication
from models.tt_transformers.tt.rope import RotarySetup

__all__ = ["RotarySetup"]
