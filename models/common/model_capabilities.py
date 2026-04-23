# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

MAX_TOKENS_ALL_USERS = 131_072


class ModelCapabilitiesMixin:
    """Defines interface for hardware- or model-specific configurations.

    NOTE: The default values here and per-model overrides will eventually be
    unified with the corresponding vLLM scheduler configuration so that both
    paths derive from the same source of truth.
    """

    @classmethod
    def get_max_tokens_all_users(cls, **kwargs) -> int:
        """Returns the default total token budget across all users."""
        return MAX_TOKENS_ALL_USERS
