# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Serving capabilities the generator and the vLLM adapter declare to vllm-tt-plugin.

This module imports no serving framework or device runtime; tt/generator_vllm.py starts from this dictionary and adds
the QWEN_FAST / QWEN_SPEC / QWEN_PREFIX_CACHE capabilities.
"""


class QwenServingCapabilities:
    serving_capabilities = {
        # Owner accepts HF-reproduced resume PCC sensitivity; top-1 and retained-state negative controls pass.
        "supports_chunked_prefill": True,
        "supports_sample_on_device": True,
        "supports_async_decode": True,
        # Structural bounds; runtime acceptance remains separately recorded.
        "output_tokens_per_step": 1,
        "max_device_top_k": 32,
        # These exclusions are deliberate, rather than pending implementations.
        "supports_prefix_caching": False,
        "supports_device_penalties": False,
        "supports_device_grammar": False,
        "supports_compact_host_logits": False,
        "supports_selective_host_readback": False,
        "tt_adaptive_block_output": False,
        "tt_adaptive_block_max_prompt_tokens": 0,
        "tt_block_kv_extent_tokens": 0,
        "folds_dp_into_lanes": False,
        "supports_stage_inputs": False,
        "supports_stage_outputs": False,
        "supports_spec_decode": False,
        "supports_async_spec_decode": False,
        "spec_ids_are_sampled": False,
        "spec_requirements": (),
        "spec_hidden_handoff": (),
        "fabric_config": None,
    }

    # Explicitly disabled sliding-window sizing extension. Qwen's hybrid here
    # means full attention plus recurrent layers, not a sliding-window cache.
    _HYBRID_KV_CACHE_GROUPS_ENABLED = False
    # The pinned runner's None default pads to the configured capacity. A
    # dynamic bucket list would advertise additional physical trace shapes.
    tt_supported_decode_batch_sizes = None
