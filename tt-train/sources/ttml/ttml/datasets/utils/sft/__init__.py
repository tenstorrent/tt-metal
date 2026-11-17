# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""What the SFT dataset providers share: batching (``collate``) and each dataset's prompt templates (``templates``)."""

from .collate import IGNORED_INDEX, collate_causal_lm

__all__ = ["IGNORED_INDEX", "collate_causal_lm"]
