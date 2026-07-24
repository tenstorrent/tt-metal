# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Resident-generator factory for the self-contained Llama-3.1-8B-Instruct demo.

``build_pipeline(mesh_device)`` builds the Llama-3.1-8B-Instruct model ONCE
(weights loaded + uploaded, resident for the whole run) on the supplied device
and returns the ready-to-forward ``Generator``. It exposes exactly the decode
contract the demo and the auto-generated perf test drive:

    generator.prefill_forward_text(...)   # process the prompt, seed the KV cache
    generator.decode_forward(...)          # one steady-state decode step

The build reuses the copied, self-contained tt_transformers plumbing that lives
UNDER this demo package (``models.demos.llama3_1_8b_p150.*``) -- never
``models.tt_transformers`` directly -- so the identity stays pinned to
Llama-3.1-8B-Instruct and there is no wiring drift between the demo, the e2e
correctness gate and the perf profile.

The build mirrors the perf test's bounded 'batch-1 latency' config: single DP
group, paged attention, ``performance`` decoder precision, and the profiled
depth honoring ``TT_PERF_LAYERS`` (full 32-layer model when unset).
"""
from __future__ import annotations

import os

# Pinned identity for this single-model demo directory (matches the demo/test module tops).
os.environ.setdefault("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")

from models.demos.llama3_1_8b_p150.demo.simple_text_demo import prepare_generator_args
from models.demos.llama3_1_8b_p150.tt.generator import Generator
from models.demos.llama3_1_8b_p150.tt.model_config import DecodersPrecision

HF_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"


def build_pipeline(
    mesh_device,
    *,
    instruct: bool = True,
    max_seq_len: int = 1024,
    batch_size: int = 1,
    data_parallel: int = 1,
    paged_attention: bool = True,
    page_params: dict | None = None,
    num_layers: int | None = None,
    use_prefetcher: bool = False,
    use_hf_rope: bool = False,
    optimizations=None,
):
    """Build the RESIDENT Llama-3.1-8B-Instruct generator on ``mesh_device`` and
    return it (a ``Generator`` exposing ``prefill_forward_text`` / ``decode_forward``).

    The model is built ONCE: ``prepare_generator_args`` loads the checkpoint and
    uploads every weight to the device, so no parameter is streamed during a
    forward. ``num_layers`` (defaulting to ``TT_PERF_LAYERS`` when that env var is
    set, else the full model) caps the profiled depth without touching the model
    math. All other arguments mirror the demo's bounded batch-1 latency config.
    """
    import ttnn

    num_devices = mesh_device.get_num_devices() if isinstance(mesh_device, ttnn.MeshDevice) else 1

    if page_params is None:
        page_params = {"page_block_size": 32, "page_max_num_blocks_per_dp": 1024}

    if num_layers is None:
        _perf_layers = os.environ.get("TT_PERF_LAYERS")
        num_layers = int(_perf_layers) if _perf_layers else None

    if optimizations is None:
        optimizations = lambda model_args: DecodersPrecision.performance(model_args.n_layers, model_args.model_name)

    global_batch_size = batch_size * data_parallel

    (
        model_args,
        model,
        page_table,
        tt_kv_cache,
        tokenizer,
        processor,
        local_data_parallel,
        local_submesh_indices,
    ) = prepare_generator_args(
        num_devices=num_devices,
        data_parallel=data_parallel,
        mesh_device=mesh_device,
        instruct=instruct,
        global_batch_size=global_batch_size,
        optimizations=optimizations,
        max_seq_len=max_seq_len,
        page_params=page_params,
        paged_attention=paged_attention,
        num_layers=num_layers,
        use_prefetcher=use_prefetcher,
        use_hf_rope=use_hf_rope,
    )

    generator = Generator(model, model_args, mesh_device, processor=processor, tokenizer=tokenizer)

    # Stash the build artifacts the perf/e2e drivers need alongside the resident
    # generator, so a caller that only holds the pipeline object can drive a full
    # prefill+decode without re-deriving the page table / KV cache.
    generator.model_args = model_args
    generator.page_table = page_table
    generator.tt_kv_cache = tt_kv_cache
    generator.tokenizer = tokenizer
    generator.processor = processor
    generator.local_data_parallel = local_data_parallel
    generator.local_submesh_indices = local_submesh_indices

    return generator
