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

    Generator classes also carry a class-level ``model_capabilities`` dict.
    The vLLM TT plugin snapshots that class attribute during
    ``check_and_update_config``, before any generator instance exists, so a
    capability narrowed on ``self.model_capabilities`` inside ``__init__`` does
    not reach the scheduler configuration. A subclass dict replaces the
    inherited one rather than merging into it, so an absent key means "not
    supported" and no reader assumes otherwise.

    Keys:

    ``supports_prefix_caching`` (bool)
        The generator accepts a nonzero ``start_pos`` for a prompt whose prefix
        is already in the KV cache.
    ``supports_async_decode`` (bool)
        ``decode_forward(..., read_from_device=False)`` followed by
        ``read_decode_output(..., async_read=True)`` is implemented, so the
        engine may overlap scheduling with device execution.
    ``supports_sample_on_device`` (bool)
        The full on-device sampling pipeline is implemented.
    ``supports_chunked_prefill`` (bool)
        One prompt may be prefilled across several engine steps. Independent of
        ``supports_prefix_caching``: both make the scheduler hand the generator a
        nonzero ``start_pos``, and either one is enough to need the resume
        plumbing, but they gate on different validation.
    ``resumed_prefill_token_alignment`` (positive int)
        The ``q_chunk_size`` the model's chunked-SDPA program is built with.
        Required only of a model whose ``model_args`` exposes no
        ``get_attn_sdpa_program_config`` for the generator to read it from, and
        read only by the generator: the plugin never needs it, because the
        generator floors every resume offset itself.
    ``output_tokens_per_step`` (positive int)
        Tokens committed per engine step. The plugin treats a value greater
        than 1 as a block-output model. Defaults to 1 when absent.
    ``accepts_trace_mode`` (bool)
        The generator honours a ``trace_mode`` argument. Used by the common
        LLM runtime, not by the vLLM TT plugin.
    """

    @classmethod
    def get_max_tokens_all_users(cls, **kwargs) -> int:
        """Returns the fallback all-user KV-cache token capacity.

        Used when no model- or device-specific override applies.
        """
        return FALLBACK_MAX_TOKENS_ALL_USERS
