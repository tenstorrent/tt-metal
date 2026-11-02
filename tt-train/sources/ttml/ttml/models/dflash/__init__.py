# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""DFlash speculative-decoding drafter components."""

from .block_attention import build_dflash_square_mask, dflash_square_attention

__all__ = ["build_dflash_square_mask", "dflash_square_attention"]
