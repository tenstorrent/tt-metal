# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

MAX_TOKENS_ALL_USERS = 131_072


class ModelCapabilitiesMixin:
    """Defines interface for hardware- or model-specific configurations."""

    @classmethod
    def get_max_tokens_all_users(cls, **kwargs) -> int:
        """Returns the default total token budget across all users."""
        return MAX_TOKENS_ALL_USERS
