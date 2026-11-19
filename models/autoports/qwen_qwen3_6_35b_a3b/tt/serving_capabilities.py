# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Model-neutral serving declarations for the full-model generator.

This module imports no serving framework or device runtime. The generator can
inherit this mixin or expose its class dictionaries. These are not vLLM
registrations. Transport capabilities have reduced explicit-command evidence
and full40 capacity32 trace evidence. Accuracy and context acceptance are
separately tracked in doc/full_model/serving_contract.json.
"""


class QwenServingCapabilities:
    @staticmethod
    def recommended_trace_region_size(max_batch_size=1):
        """Mesh-owner reservation, including model and sampling traces.

        All-layer capacity-32 capture measured 467,361,792 model bytes.
        Reserve 512 MiB for fixed-capacity batched graphs; solo needs 200 MB.
        """
        if not 1 <= max_batch_size <= 32:
            raise ValueError("max_batch_size must be between 1 and 32")
        return 200_000_000 if max_batch_size == 1 else 512 * 1024 * 1024

    serving_capabilities = {
        # Owner accepts HF-reproduced resume PCC sensitivity; top-1 and
        # retained-state negative controls pass. See doc/full_model/owner_acceptance.json.
        "supports_chunked_prefill": True,
        "supports_sample_on_device": True,
        "supports_async_decode": True,
        # Structural bounds; runtime acceptance remains separately recorded.
        "output_tokens_per_step": 1,
        "max_device_top_k": 64,
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

    pending_serving_capabilities = {}

    verified_serving_capabilities = {
        "supports_chunked_prefill": {
            "evidence": "doc/full_model/owner_acceptance.json",
            "scope": "Retained-state continuation; measured PCC failures are an owner-accepted HF-reproduced limitation, not a numerical repair",
        },
        "supports_sample_on_device": {
            "evidence": "doc/full_model/serving_commands_probe/result.json; doc/full_model/generator_replay_probe/final.json; doc/full_model/full_batch32_trace_budget.json",
            "scope": "layers0/3 capacity3 explicit commands/stochastic isolation; full40 capacity32 traced greedy feedback",
        },
        "supports_async_decode": {
            "evidence": "doc/full_model/serving_commands_probe/result.json",
            "scope": "layers 0/3, capacity 3; resident feedback, immutable output leases, remap and reload commands",
        },
    }

    serving_limits = {
        "model_scope": "text-only Qwen/Qwen3.6-35B-A3B",
        "mesh_shape": (1, 4),
        "parallel_strategy": "selected TP4/EP4 optimized multichip decoder",
        "required_fabric": "FABRIC_1D; actual TTNN enum setup belongs to the mesh owner",
        "trace_reservation_owner": "mesh owner, before opening the mesh and constructing the generator",
        "trace_reservation_basis": "configured max_batch_size, never the currently active row count",
        "trace_region_size_batch1_bytes": 200_000_000,
        "trace_region_size_batched_bytes": 536_870_912,
        "full40_batch32_model_trace_measured_bytes": 467_361_792,
        "trace_reservation_accounting": "include the selected per-device reservation in prepared DRAM capacity; do not double-count live trace buffers",
        "trace_reservation_acceptance": "batch1 full40 gates pass at200MB; full40 capacity32 capture/replay passes at512MiB",
        "pinned_plugin_trace_region_override": "tt.trace_region_size; pinned worker defaults to 50000000 bytes and does not discover this helper",
        "context_ceiling_tokens": 262144,
        "context_status": "full-model 262144-token capacity acceptance passed; doc/full_model/context_acceptance/final.json",
        "aggregate_default_kv_token_capacity": 262144,
        "max_configured_request_slots": 32,
        "slot_status": "capacity32 full40 trace/state transport verified; context1024, state payload checks on chip0",
        "verified_transport_batch_capacity": 32,
        "verified_explicit_command_batch_capacity": 3,
        "maximum_batch_acceptance": "full40 capacity32/context1024 passes on1x4; inactive states checked on chip0, not claimed all-chip payload checks",
        "maximum_batch_evidence": "doc/full_model/full_batch32_trace_budget.json",
        "page_size_tokens": 32,
        "page_table_window_entries": 8,
        "local_kv_heads": 1,
        "kv_head_size": 256,
        "kv_dtype_policy": "cache_spec() reports selected layer backing dtype; preserve decode/pre-fill compute split",
        "cache_allocation_phase": "before preparation/capture; no live cache replacement",
        "residual_policy": "selected replicated BF16 inter-layer residual",
        "decode_entrypoint": "decode_forward_v1",
        "decode_command_contract_version": 1,
        "decode_dispatch": "vLLM adapter binds v1 explicitly; legacy decode_forward serves standalone readiness",
        "decode_bucket_policy": "standalone defaults to capacity trace; vLLM prepares and retains B2/B32 at capacity32, selecting by highest active physical slot",
        "decode_requires_trace": True,
        "prefill_graph_trace": False,
        "prefill_sampling_trace": True,
        "sampling_mode_selector": "sampling_params present => device tokens; absent => caller-sampled full logits",
        "stochastic_device_top_k": (1, 64),
        "unbounded_or_larger_top_k": "explicit host-logit compatibility; mixed admission may change seeded host/device routes, see doc/vllm_integration/sampling_debug/SAMPLING_DISPOSITION.md",
        "device_logprobs": False,
        "host_logits": "FP32 real-vocabulary [submitted_rows,1,V], independent writable storage",
        "output_snapshot_depth": 2,
        "async_queue_id": 0,
        "prefix_cache_exclusion": "recurrent_state: KV pages alone do not restore DeltaNet matrix/convolution history",
        "retained_state_resumed_prefill": "supported with owner-accepted HF-reproduced PCC limitation; automatic KV-only prefix caching remains excluded",
        "session_identity_tracking": False,
        "multimodal_inputs": False,
        "uses_request_specific_mrope": False,
        "host_control_owner": "caller applies penalties, logprobs, grammar, token filters and stopping",
        "standalone_stopping": "fixed-length default; explicit stop_on_eos and stop_token_ids are available",
        "standalone_unknown_kwargs": "ignored per Metal readiness ABC; recognized nonneutral semantic controls reject",
    }
