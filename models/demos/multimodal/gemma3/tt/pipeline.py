# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Resident gemma-3 TEXT pipeline exposing the standard DECODE CONTRACT.

WHY THIS EXISTS
    The perf tooling measures a steady-state decode token via trace+1CQ. To do that it needs a
    pipeline that exposes ``decode_step(state) -> state`` (see
    ``models/experimental/perf_automation/agent/perf_adapter.py``). Without it,
    ``PipelineDecodeAdapter.setup`` raises ``NotTraceCapable``, trace_replay emits
    ``TRACE_NOT_TRACE_CAPABLE=1``, and the model can only ever be measured EAGER -- which folds in
    per-op dispatch overhead that trace removes, so the headline number is not the one anyone ships.

    gemma3's demo drives decode inline in ``text_demo.py``'s loop, so there was nothing for the
    harness to call. This module re-exposes that same loop body as the contract, without changing
    how the model runs.

SELF-TRACED
    ``Generator.decode_forward(enable_trace=True)`` already owns its decode trace capture and does
    the host<->device I/O plus ``execute_trace`` internally. So this pipeline is ``self_traced``:
    ``decode_step`` drives exactly one traced token the way the demo does, and the harness TIMES the
    native step rather than wrapping it in a second capture (a nested capture fatals and hangs the
    device).
"""
from __future__ import annotations

import torch

# Match the demo's device configuration so what is measured is the path that actually ships.
_PAGE_PARAMS = {"page_block_size": 32, "page_max_num_blocks_per_dp": 1024}


def _greedy_sampling_params(generator):
    """On-device greedy sampling when this build supports it, else None (host argmax).

    Keeping sampling ON DEVICE is what the demo does when it can, and it keeps the timed step free
    of a host round-trip. When unsupported the caller falls back to argmax on the returned logits.
    """
    from models.common.sampling import SamplingParams

    model0 = generator.model[0] if isinstance(generator.model, (list, tuple)) else generator.model
    if not getattr(model0, "_supports_on_device_sampling", False):
        return None
    return SamplingParams(temperature=0.0, top_k=1, top_p=1.0)


def _normalize_prompt_ids(input_ids):
    """Accept a tensor, a list of ids, or None; return a [B, S] LongTensor."""
    if input_ids is None:
        return torch.tensor([[2, 818, 6037, 529, 6081, 603]])  # "The capital of France is"-ish
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    return input_ids.long()


def _attach_decode_contract(generator, page_table, tt_kv_cache) -> None:
    """Bind decode_prefill / decode_step onto the resident Generator.

    gemma3 differs from the llama pipeline in one respect: ``page_table`` and ``tt_kv_cache`` are
    returned alongside the model rather than living on the generator, so they are captured in the
    closure here and passed on every call.
    """

    def decode_prefill(input_ids):
        prompt = _normalize_prompt_ids(input_ids)
        batch_size, _ = prompt.shape
        sampling_params = _greedy_sampling_params(generator)
        decoding_pos = [int(prompt.shape[1])] * batch_size

        prefill_out = generator.prefill_forward_text(
            prompt,
            page_table=page_table,
            kv_cache=tt_kv_cache,
            prompt_lens=decoding_pos,
            sampling_params=sampling_params,
            warmup_prefill=True,
        )
        if sampling_params is not None and isinstance(prefill_out, tuple):
            prefilled_token, _ = prefill_out
        else:
            logits = prefill_out[0] if isinstance(prefill_out, tuple) else prefill_out
            prefilled_token = torch.argmax(logits, dim=-1)

        # decode_forward expects batch-major token ids shaped [B, 1] (text_demo.py).
        out_tok = prefilled_token.reshape(batch_size, 1)
        return {
            "out_tok": out_tok,
            "current_pos": torch.tensor(decoding_pos),
            "sampling_params": sampling_params,
            "batch_size": batch_size,
            "iteration": 0,
        }

    def decode_step(state):
        out = generator.decode_forward(
            state["out_tok"],
            state["current_pos"],
            enable_trace=True,
            page_table=page_table,
            kv_cache=tt_kv_cache,
            sampling_params=state["sampling_params"],
        )
        logits = out[0] if isinstance(out, (tuple, list)) else out
        b = state["batch_size"]
        if state["sampling_params"] is not None:
            # Device decode returns [B] or [B, 1] token ids; normalize to [B, 1] (text_demo.py).
            tok = logits.reshape(b, 1) if logits.dim() == 1 else logits.reshape(b, -1)[:, -1:]
        else:
            tok = torch.argmax(logits, dim=-1).reshape(b, 1)
        state["out_tok"] = tok
        state["current_pos"] = state["current_pos"] + 1
        state["iteration"] += 1
        return state

    def trace_path():
        return "trace+1cq"

    generator.decode_prefill = decode_prefill
    generator.decode_step = decode_step
    generator.trace_path = trace_path
    generator.self_traced = True

    # ---- PER-STAGE contract -------------------------------------------------------------------
    # decode alone is not the whole pipeline. PREFILL sets TTFT and dominates long-context cost, and
    # a perf run that only ever times decode leaves it unranked and unoptimized. Exposing both
    # stages means measure_adapter reports TRACE_STAGE_MS[prefill] and TRACE_STAGE_MS[decode]
    # separately, so optimize can target either.
    #
    # Prefill is the variable-shape stage, which is exactly what the setup/step split is for: pin
    # the prompt in _trace_setup, then _trace_step runs one FIXED-shape prefill over it.
    _st = {}

    def prefill_trace_setup(inputs=None):
        ids = inputs.get("input_ids") if isinstance(inputs, dict) else inputs
        prompt = _normalize_prompt_ids(ids)
        _st["prompt"] = prompt
        _st["lens"] = [int(prompt.shape[1])] * int(prompt.shape[0])
        _st["sampling"] = _greedy_sampling_params(generator)
        # Warm up OUTSIDE the timed step so compile/capture is not charged to prefill.
        generator.prefill_forward_text(
            prompt,
            page_table=page_table,
            kv_cache=tt_kv_cache,
            prompt_lens=_st["lens"],
            sampling_params=_st["sampling"],
            warmup_prefill=True,
            enable_trace=True,
        )

    def prefill_trace_step():
        return generator.prefill_forward_text(
            _st["prompt"],
            page_table=page_table,
            kv_cache=tt_kv_cache,
            prompt_lens=_st["lens"],
            sampling_params=_st["sampling"],
            warmup_prefill=False,
            enable_trace=True,
        )

    def decode_trace_setup(inputs=None):
        ids = inputs.get("input_ids") if isinstance(inputs, dict) else inputs
        _st["decode_state"] = decode_prefill(ids)

    def decode_trace_step():
        _st["decode_state"] = decode_step(_st["decode_state"])
        return _st["decode_state"]

    generator.prefill_trace_setup = prefill_trace_setup
    generator.prefill_trace_step = prefill_trace_step
    generator.decode_trace_setup = decode_trace_setup
    generator.decode_trace_step = decode_trace_step
    # Both stages are natively traced by the Generator; declaring this stops the harness wrapping
    # them in a SECOND begin_trace_capture (a nested capture fatals and hangs the device).
    generator.prefill_self_traced = True
    generator.decode_self_traced = True
    generator.PIPELINE_STAGES = ["prefill", "decode"]


def build_pipeline(mesh_device, max_seq_len: int = 1024, batch_size: int = 1, instruct: bool = True):
    """Build the resident gemma-3 text pipeline EXACTLY as the demo does, with the decode contract.

    Returns the Generator; the harness calls decode_prefill once then times decode_step.
    """
    from models.demos.multimodal.gemma3.demo.text_demo import prepare_generator_args
    from models.demos.multimodal.gemma3.tt.gemma_multimodal_generator import GemmaMultimodalGenerator as Generator
    from models.tt_transformers.tt.model_config import DecodersPrecision

    model_args, model, page_table, tt_kv_cache, tokenizer = prepare_generator_args(
        num_devices=mesh_device.get_num_devices(),
        data_parallel=1,
        mesh_device=mesh_device,
        instruct=instruct,
        global_batch_size=batch_size,
        # A CALLABLE of model_args, exactly as the demo's parametrize supplies it -- the precision
        # preset needs the layer count and model name, which only exist once ModelArgs is built.
        optimizations=lambda model_args: DecodersPrecision.performance(model_args.n_layers, model_args.model_name),
        max_seq_len=max_seq_len,
        page_params=_PAGE_PARAMS,
        paged_attention=True,
    )
    generator = Generator(model, model_args, mesh_device, tokenizer=tokenizer)
    # Keep them reachable for callers that expect the llama-style attributes.
    generator.page_table = page_table
    generator.tt_kv_cache = tt_kv_cache
    _attach_decode_contract(generator, page_table, tt_kv_cache)
    return generator
