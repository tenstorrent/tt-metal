# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Overview
--------

The vLLM path has three layers:

1. vLLM scheduler/engine (caller)
2. ``tt/generator_vllm.py`` adapter (this contract)
3. ``tt/generator.py`` low-level methods (prefill/decode)

The adapter should be thin. It translates vLLM-facing inputs to the existing
generator's low-level interface and preserves cache ownership semantics:

- Standalone generation path: generator owns and resets its cache.
- vLLM serving path: vLLM owns attention KV cache allocation and passes that
  cache through prefill/decode calls.

Async decode split contract
---------------------------

If ``model_capabilities["supports_async_decode"]`` is true, decode must support
the split submit/read/process sequence:

1. ``decode_forward(..., read_from_device=False)`` submits decode and returns
   device-resident output handles.
2. ``read_decode_output(..., async_read=True)`` performs minimal deferred host
   reads and may return read events/futures.
3. ``process_decode_output_host(...)`` performs host formatting only.
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol, TypedDict

import torch


class ModelCapabilities(TypedDict, total=False):
    """Capability flags consumed by the TT vLLM integration path."""

    supports_prefix_caching: bool
    supports_async_decode: bool
    supports_sample_on_device: bool


class VllmGeneratorAdapter(Protocol):
    """
    Protocol for ``tt/generator_vllm.py`` adapter classes.

    Implementations are typically named ``TT<Arch>ForCausalLM`` (or
    ``...ForConditionalGeneration``) and are registered in the TT vLLM plugin.

    This protocol mirrors what the TT vLLM plugin calls today.
    """

    model_capabilities: ClassVar[ModelCapabilities]
    """
    Model capability flags interpreted by plugin/runtime.

    When present, the plugin currently reads:
    ``supports_sample_on_device``, ``supports_async_decode``, and
    ``supports_prefix_caching``.
    """

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config: Any,
        mesh_device: Any,
        max_batch_size: int,
        max_seq_len: int | None = None,
        tt_data_parallel: int = 1,
        optimizations: str | None = None,
        **kwargs: Any,
    ) -> "VllmGeneratorAdapter":
        """
        Build the adapter and underlying generator/model for vLLM serving.

        Called by the TT vLLM model loader.

        The adapter should load the selected full-model precision policy and
        preserve full-model behavior, rather than introducing an adapter-only
        decoding path.
        """

    @classmethod
    def get_max_tokens_all_users(
        cls,
        model_name: str = "",
        num_devices: int = 1,
        tt_data_parallel: int = 1,
        max_model_len: int | None = None,
        max_num_seqs: int | None = None,
        **kwargs: Any,
    ) -> int:
        """
        Return the serving token budget for vLLM's scheduler constraints.

        Called by the TT worker to size KV blocks.

        Implementations may validate supported batch/context limits and raise
        ``ValueError`` for unsupported configurations.
        """

    @classmethod
    def get_kv_cache_spec(cls, vllm_config: Any):
        """
        Optional hybrid-attention cache grouping contract.

        Required only for hybrid-attention adapters that expose per-layer KV
        cache groups.

        Adapters for hybrid attention models may expose this to let vLLM build
        per-layer cache groups and page-table routing.
        """

    def allocate_kv_cache(self, kv_cache_shape: Any, dtype: torch.dtype, num_layers: int) -> Any:
        """
        Allocate or bind vLLM-owned attention KV cache.

        In serving mode, vLLM owns cache lifecycle and passes the returned cache
        back into ``prefill_forward``/``decode_forward``.
        """

    def warmup_model_prefill(
        self,
        *,
        kv_cache: Any,
        can_sample_on_device: bool,
        enable_trace: bool,
        **kwargs: Any,
    ) -> None:
        """
        Warmup/compile prefill path for serving.

        This should exercise the same serving execution path used at runtime.
        """

    def warmup_model_decode(
        self,
        *,
        kv_cache: Any,
        max_batch_size: int,
        num_blocks: int,
        can_sample_on_device: bool,
        enable_trace: bool,
        **kwargs: Any,
    ) -> None:
        """
        Warmup/compile decode path for serving.

        This should exercise traced decode and the serving sampling path.
        """

    def prefill_forward(
        self,
        *,
        tokens: torch.Tensor,
        page_table: torch.Tensor,
        kv_cache: Any,
        enable_trace: bool,
        prompt_lens: Any,
        start_pos: Any,
        page_tables_per_layer: Any | None = None,  # Passed for hybrid KV-cache groups; absent for single-group models.
        sampling_params: Any | None = None,  # Passed only when device sampling is active.
        empty_slots: Any | None = None,  # Passed for multi-DP user-slot mapping; absent in single-DP flows.
        **kwargs: Any,
    ) -> Any:
        """
        Submit a serving prefill step.

        Additional model-specific kwargs (for example multimodal payloads) may
        be provided via ``**kwargs``.

        The adapter should delegate to generator/model low-level prefill
        behavior whenever possible.
        """

    def decode_forward(
        self,
        *,
        tokens: torch.Tensor,
        page_table: torch.Tensor,
        kv_cache: Any,
        start_pos: Any,
        enable_trace: bool,
        read_from_device: bool,
        page_tables_per_layer: Any | None = None,  # Passed for hybrid KV-cache groups; absent for single-group models.
        sampling_params: Any | None = None,  # Passed only when device sampling is active.
        prompt_tokens: Any
        | None = None,  # Passed on device-sampling decode when prompt/output token history is available.
        output_tokens: Any | None = None,  # Passed with prompt_tokens for device-sampling decode state updates.
        reset_batch: bool | None = None,  # Passed only on device-sampling decode; controls sampler state reset.
        slot_remap: Any | None = None,  # Passed only when scheduler remaps slots between decode steps.
        rope_deltas_all_users: Any
        | None = None,  # Passed for request-specific mRoPE models; may be None on steady requests.
        **kwargs: Any,
    ) -> Any:
        """
        Submit one serving decode step.

        When async split is supported, ``read_from_device=False`` should return
        device-resident output suitable for deferred read.

        If ``read_decode_output`` / ``process_decode_output_host`` are not
        implemented, ``decode_forward`` must already return host tensors.
        """

    def read_decode_output(self, tt_out: Any, async_read: bool = False) -> Any:
        """
        Read decode output to host buffers.

        For async split, ``async_read=True`` performs deferred/minimal host
        reads and may return completion handles/events.

        Required for async split decode (and for sync decode when ``decode_forward`` returns
        device tensors instead of host tensors).
        """

    def process_decode_output_host(self, tt_out: Any, is_tokens: bool = False) -> Any:
        """
        Format host-side decode output for vLLM.

        This stage should not submit new device work; it is the host formatting
        boundary after ``read_decode_output``.

        Required for async split decode (and for sync decode when ``decode_forward`` returns
        device tensors instead of host tensors).

        """


GENERATOR_VLLM_MODULE_RELPATH = "tt/generator_vllm.py"
"""Path of the vLLM adapter file relative to ``<model_dir>``."""


__all__ = [
    "GENERATOR_VLLM_MODULE_RELPATH",
    "ModelCapabilities",
    "VllmGeneratorAdapter",
]
