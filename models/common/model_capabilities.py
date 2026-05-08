# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Maximum number of tokens that can simultaneously occupy the KV cache
# across all concurrent users. This fallback applies to model/device
# configurations not covered by a model-specific override.
# Derived from the default branch of the per-model KV-cache rules in
# the TT vLLM worker (tenstorrent/vllm#315).
# See also: https://github.com/tenstorrent/vllm/issues/315
FALLBACK_MAX_TOKENS_ALL_USERS = 131_072


class ModelCapabilitiesMixin:
    """Defines interface for hardware- or model-specific configurations.

    NOTE: The default values here and per-model overrides will eventually be
    unified with the corresponding vLLM scheduler configuration so that both
    paths derive from the same source of truth.
    """

    @classmethod
    def get_max_tokens_all_users(cls, **kwargs) -> int:
        """Returns the fallback all-user KV-cache token capacity.

        Used when no model- or device-specific override applies.
        """
        return FALLBACK_MAX_TOKENS_ALL_USERS
