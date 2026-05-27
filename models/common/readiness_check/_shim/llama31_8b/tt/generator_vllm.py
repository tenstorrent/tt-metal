# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
vLLM adapter shim for Llama 3.1 8B Instruct (tt-transformers backend).

This file is the vLLM-facing counterpart to `tt/generator.py`. It exists
only to exercise the `run_vllm_server` readiness check against an
already-working tt-transformers model; it is not a real auto-ported
model.

The class is a one-line delegate to the upstream tt-transformers
`LlamaForCausalLM` — same construction, same prefill / decode / KV
allocation. We subclass it (rather than re-register the upstream class)
so the shim has its own architecture entry in
`vllm-tt-plugin/src/vllm_tt_plugin/platform.py::register_tt_models`
without colliding with the production `TTLlamaForCausalLM` mapping.
"""

from __future__ import annotations

from models.tt_transformers.tt.generator_vllm import LlamaForCausalLM


class Llama31_8BReadinessShimForCausalLM(LlamaForCausalLM):
    """Thin alias for the upstream Llama vLLM adapter.

    All `initialize_vllm_model`, `prefill_forward`, `decode_forward`,
    `allocate_kv_cache`, and `cache_path` behavior is inherited unchanged.
    """
