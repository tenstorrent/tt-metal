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

import torch

from models.demos.llama3_1_8b_p150.demo.simple_text_demo import prepare_generator_args
from models.demos.llama3_1_8b_p150.tt.generator import Generator, SamplingParams
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

    _attach_decode_contract(generator)

    return generator


def _normalize_prompt_ids(generator, input_ids) -> torch.Tensor:
    """Coerce ``input_ids`` into a ``[batch, seq]`` int64 prompt tensor.

    Accepts an already-encoded token tensor / list of ids, a batch of such
    lists, or a raw prompt string (encoded here with the instruct template).
    """
    if isinstance(input_ids, str):
        input_ids = generator.model_args[0].encode_prompt(input_ids, instruct=True)
    if isinstance(input_ids, torch.Tensor):
        toks = input_ids.to(torch.int64)
        return toks.unsqueeze(0) if toks.dim() == 1 else toks
    if len(input_ids) > 0 and isinstance(input_ids[0], (list, tuple, torch.Tensor)):
        rows = [torch.as_tensor(list(r), dtype=torch.int64) for r in input_ids]
        return torch.stack(rows, dim=0)
    return torch.as_tensor(list(input_ids), dtype=torch.int64).unsqueeze(0)


def _greedy_sampling_params(generator):
    """Build the on-device greedy (temperature=0 argmax) sampling params, or
    ``None`` when this model cannot sample on device (host argmax fallback)."""
    supports = getattr(generator.model[0], "_supports_on_device_sampling", False) and (
        getattr(generator.model[0], "sampling", None) is not None
    )
    if not supports:
        return None
    return SamplingParams(temperature=0, top_k=32, top_p=0.08)


def _attach_decode_contract(generator) -> None:
    """Bind the standard DECODE CONTRACT (decode_prefill / decode_step) onto the
    resident ``Generator``.

    The Generator already owns its decode trace capture (``trace_ids_decode``)
    and on-device sampling, and ``decode_forward(enable_trace=True)`` does the
    host<->device I/O plus ``execute_trace`` internally -- the persistent-buffer,
    vLLM-style decode. So this pipeline is ``self_traced``: ``decode_step`` drives
    exactly one traced token the same way ``simple_text_demo`` does, and the
    harness times the native step rather than wrapping it in its own capture.
    """

    def decode_prefill(input_ids):
        prompt = _normalize_prompt_ids(generator, input_ids)
        batch_size, _ = prompt.shape
        sampling_params = _greedy_sampling_params(generator)
        decoding_pos = [int(prompt.shape[1])] * batch_size

        prefill_out = generator.prefill_forward_text(
            prompt,
            page_table=generator.page_table,
            kv_cache=generator.tt_kv_cache,
            prompt_lens=decoding_pos,
            sampling_params=sampling_params,
            warmup_prefill=True,
            enable_trace=True,
        )
        if sampling_params is not None and isinstance(prefill_out, tuple):
            prefilled_token, _ = prefill_out
        else:
            prefilled_token = torch.argmax(prefill_out, dim=-1)

        return {
            "out_tok": prefilled_token,
            "current_pos": torch.tensor(decoding_pos),
            "prompt_tokens": prompt,
            "sampling_params": sampling_params,
            "iteration": 0,
            "generated": [],
        }

    def decode_step(state):
        out = generator.decode_forward(
            state["out_tok"],
            state["current_pos"],
            enable_trace=True,
            page_table=generator.page_table,
            kv_cache=generator.tt_kv_cache,
            reset_batch=(state["iteration"] == 0),
            sampling_params=state["sampling_params"],
            prompt_tokens=state["prompt_tokens"],
            output_tokens=state["out_tok"],
        )
        if state["sampling_params"] is not None:
            logits, _ = out
            state["out_tok"] = logits.unsqueeze(1)
        else:
            state["out_tok"] = torch.argmax(out, dim=-1)
        state["current_pos"] = state["current_pos"] + 1
        state["iteration"] += 1
        return state

    def trace_path():
        return "trace+1cq"

    generator.decode_prefill = decode_prefill
    generator.decode_step = decode_step
    generator.trace_path = trace_path
    generator.self_traced = True
