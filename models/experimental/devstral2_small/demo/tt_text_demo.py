# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# Tenstorrent demo: Devstral Small 2 (Mistral3) **text** LM on TT, with **autoregressive generation** using **TT** ``embed_tokens`` and **TT** ``TtMinistral3RotaryEmbedding`` inside ``TtMinistral3Model`` (via ``forward_prefill``), **TT** ``LMHead``, and on-device ``Sampling1D``.

from __future__ import annotations

import argparse
import json
import os
import time
import types
from pathlib import Path
from typing import Any

# Keep normal demo output focused; set TT_LOGGER_LEVEL=warn/Debug externally when debugging TT-Metal.
os.environ.setdefault("TT_LOGGER_LEVEL", "Error")

import pytest
import torch
from loguru import logger
from tracy import signpost as _profiler_signpost
from transformers import AutoProcessor
from transformers.masking_utils import create_causal_mask
from transformers.models.ministral3.configuration_ministral3 import Ministral3Config
from transformers.models.mistral3.modeling_mistral3 import Mistral3Model

import ttnn
from models.common.sampling import SamplingParams, format_sampling_params
from models.common.utility_functions import comp_pcc
from models.experimental.devstral2_small.devstral_utils import (
    DEFAULT_MODEL_ID,
    DevstralSampling1DAdapter,
    apply_devstral_hf_trust_patches,
    resolve_rope_parameters,
    apply_fp8_dequantize_compat,
    close_devstral_demo_mesh,
    demo_lm_head_max_columns_per_device,
    devstral_supports_on_device_sampling,
    eos_token_ids,
    host_input_ids_to_tt_replicated,
    open_devstral_demo_mesh,
    text_model_root,
    tt_alloc_decode_input_buffers,
    tt_capture_decode_trace,
    tt_execute_decode_trace,
    tt_forward_prefill_from_device_ids,
    tt_lm_head_logits_block,
    tt_prefill_hidden_states_from_ids,
    tt_read_decode_traced_token,
    tt_release_decode_trace,
    tt_sampling_output_token_id,
    tt_update_decode_input_buffers,
    tt_warmup_decode_trace_path,
    tt_warmup_prefill_lm_head_sampling,
)
from models.experimental.devstral2_small.tt.pipeline.tt_ministral3_model import TtMinistral3Model
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import Mode, PagedAttentionConfig
from models.tt_transformers.tt.lm_head import LMHead
from models.tt_transformers.tt.model_config import ModelArgs

apply_fp8_dequantize_compat()

# Default HF-style generation kwargs (--hf-generate and argparse defaults).
DEFAULT_GENERATE_KWARGS = {
    "max_new_tokens": 200,
    "do_sample": True,
    "temperature": 0.15,
}


def _tokenize_demo_prompt(tokenizer, prompt: str, system_prompt: str | None):
    """Build input_ids from this demo's --prompt / --system-prompt (no tools)."""
    if system_prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
    else:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    return tokenizer.apply_chat_template(
        conversation=messages,
        return_tensors="pt",
        return_dict=True,
    )


def _load_messages_json(path: Path, scenario: str | None) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if "scenarios" in raw:
        key = scenario or raw.get("default_scenario")
        if not key or key not in raw["scenarios"]:
            raise ValueError(f"{path}: scenario {key!r} not in {list(raw.get('scenarios', {}).keys())}")
        block = raw["scenarios"][key]
    else:
        block = raw
    if "messages" not in block:
        raise ValueError(f"{path}: missing messages")
    return block["messages"]


def _truncate_input_ids_tail(
    input_ids: torch.Tensor,
    input_seq_len: int | None,
) -> tuple[torch.Tensor, int, int]:
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"Expected input_ids [1, L], got {tuple(input_ids.shape)}")
    full_len = int(input_ids.shape[1])
    if input_seq_len is None or input_seq_len >= full_len:
        return input_ids, full_len, full_len
    if input_seq_len <= 0:
        raise ValueError(f"--input-seq-len must be positive, got {input_seq_len}")
    return input_ids[:, -input_seq_len:], full_len, int(input_seq_len)


def _pad_page_table_blocks(table: torch.Tensor) -> torch.Tensor:
    """Pad a ``[1, n_blocks]`` page-table slice up to a multiple of 8 blocks with -1."""
    n = int(table.shape[1])
    aligned = ((n + 7) // 8) * 8
    if aligned == n:
        return table
    pad = torch.full((1, aligned - n), -1, dtype=table.dtype)
    return torch.cat([table, pad], dim=1)


def _tt_prefill_paged_chunked(
    merged_embeds_bsh: torch.Tensor,
    pad_row_1d: torch.Tensor,
    mesh_device,
    tt_language_model,
    real_seq_len: int,
    chunk_size: int,
    block_size: int,
    page_table_host: torch.Tensor,
    page_table_tt: ttnn.Tensor,
) -> tuple[ttnn.Tensor, int]:
    """Chunked + paged prefill from merged embeddings (same path as ``tt_image_demo``)."""
    hidden = int(merged_embeds_bsh.shape[-1])
    s = int(real_seq_len)
    s_pad = ((s + chunk_size - 1) // chunk_size) * chunk_size
    if s_pad > s:
        pr = pad_row_1d.to(device=merged_embeds_bsh.device, dtype=merged_embeds_bsh.dtype).view(1, 1, -1)
        merged = torch.cat([merged_embeds_bsh, pr.expand(1, s_pad - s, hidden)], dim=1)
    else:
        merged = merged_embeds_bsh

    last_chunk_start = ((s - 1) // chunk_size) * chunk_size
    sample_idx = (s - 1) - last_chunk_start

    replicate = ttnn.ReplicateTensorToMesh(mesh_device)
    tt_out_last: ttnn.Tensor | None = None
    for chunk_start in range(0, s_pad, chunk_size):
        chunk = merged[:, chunk_start : chunk_start + chunk_size, :].unsqueeze(1).contiguous()
        h_tt = ttnn.from_torch(
            chunk,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
        )
        positions = torch.arange(chunk_start, chunk_start + chunk_size, dtype=torch.int32).view(1, -1)
        pos_tt = ttnn.from_torch(
            positions, device=mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=replicate
        )
        cpt = _pad_page_table_blocks(
            page_table_host[:, chunk_start // block_size : (chunk_start + chunk_size) // block_size]
        )
        cpt_tt = ttnn.from_torch(
            cpt, device=mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=replicate
        )
        out = tt_language_model.forward_prefill_from_embeddings(
            h_tt,
            None,
            pos_tt,
            rope_start_pos=chunk_start,
            page_table=page_table_tt,
            chunk_page_table=cpt_tt,
            chunk_start_idx=chunk_start,
        )
        ttnn.deallocate(h_tt)
        ttnn.deallocate(pos_tt)
        ttnn.deallocate(cpt_tt)
        if chunk_start == last_chunk_start:
            tt_out_last = out
        else:
            ttnn.deallocate(out)
    if tt_out_last is None:
        raise RuntimeError("Chunked paged prefill produced no output tensor.")
    return tt_out_last, sample_idx


def pytest_addoption(parser):
    """Register demo CLI flags for ``pytest -p ...tt_text_demo ... --mesh-width N``."""
    parser.addoption("--mesh-width", action="store", default=None, type=int, help="Device mesh width (1 x N)")
    parser.addoption("--max-new-tokens", action="store", default=None, type=int, help="New tokens after prompt")


def _argv_from_pytest(request) -> list[str]:
    argv = ["tt_text_demo.py"]
    mesh_width = request.config.getoption("--mesh-width")
    if mesh_width is not None:
        argv.extend(["--mesh-width", str(mesh_width)])
    max_new_tokens = request.config.getoption("--max-new-tokens")
    if max_new_tokens is not None:
        argv.extend(["--max-new-tokens", str(max_new_tokens)])
    return argv


@pytest.mark.timeout(900)
def test_demo(request):
    """Pytest entrypoint; same path as ``python .../tt_text_demo.py [args]``."""
    mesh_width = request.config.getoption("--mesh-width")
    if mesh_width is not None and mesh_width > ttnn.get_num_devices():
        pytest.skip(f"--mesh-width {mesh_width} requested but only {ttnn.get_num_devices()} device(s) are visible.")
    main(_argv_from_pytest(request))


def main(argv: list[str] | None = None):
    ref_max = int(DEFAULT_GENERATE_KWARGS["max_new_tokens"])
    ref_temp = float(DEFAULT_GENERATE_KWARGS["temperature"])
    ref_do_sample = bool(DEFAULT_GENERATE_KWARGS["do_sample"])

    parser = argparse.ArgumentParser(
        description="Devstral-2 on TT: Ministral3 prefill + TT LMHead + on-device Sampling1D."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="HF repo id (also set HF_MODEL).")
    parser.add_argument(
        "--prompt",
        default=(
            "Can you implement in Python a method to compute the fibonnaci sequence at the `n`th element "
            "with `n` a parameter passed to the function? Start the sequence from 1."
        ),
        help="User message text (default prompt is defined in this file).",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional system message (default: none).",
    )
    parser.add_argument(
        "--text-layers",
        type=int,
        default=None,
        help="Decoder layers on TT/HF cache after load (default: all). Required for quality matching full inference.",
    )
    parser.add_argument("--mesh-width", type=int, default=1, help="Device mesh width (1 × N).")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="PCC: HF reference (HF prompt embeddings) vs full TT prefill (TT embed + TT rotary).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=ref_max,
        help=f"New tokens after prompt (default {ref_max}). 0 = skip generation.",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help=f"Greedy argmax (default: sample with temperature {ref_temp}).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=ref_temp,
        help="Sampling temperature when not --greedy.",
    )
    parser.add_argument(
        "--hf-generate",
        action="store_true",
        help="Use Hugging Face model.generate() instead of TT stack + TT LM head (baseline only; not TT).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Torch/CUDA RNG seed before generation (recommended for sampling).",
    )
    parser.add_argument(
        "--lm-head-max-device-cols",
        type=int,
        default=None,
        help="On-device LMHead: max vocab columns per matmul shard (default 4096; try 2048 if L1 error remains).",
    )
    parser.add_argument(
        "--messages-json",
        type=Path,
        default=None,
        help="Benchmark messages JSON (e.g. .cpmcache/messages_256k_text.json). Overrides --prompt.",
    )
    parser.add_argument(
        "--benchmark-scenario",
        default=None,
        help="Scenario key when --messages-json is a combined file (default: file default_scenario).",
    )
    parser.add_argument(
        "--input-seq-len",
        type=int,
        default=None,
        metavar="N",
        help="Use the last N prompt tokens from the benchmark JSON (tail slice). "
        "Omit to use the full measured prompt length.",
    )
    parser.add_argument(
        "--prefill-chunk-size",
        type=int,
        default=8192,
        metavar="N",
        help="Chunk size for chunked + paged prefill when prompt exceeds N tokens "
        "(must be a positive multiple of 512).",
    )
    args = parser.parse_args(None if argv is None else argv[1:])
    prefer_stochastic_sampling = (not args.greedy) and ref_do_sample

    if args.prefill_chunk_size <= 0 or args.prefill_chunk_size % 512 != 0:
        parser.error("--prefill-chunk-size must be a positive multiple of 512.")

    if args.hf_generate and args.text_layers is not None:
        logger.warning("--hf-generate uses full HF depth; ignoring --text-layers.")
        args.text_layers = None

    os.environ["HF_MODEL"] = args.model_id
    apply_devstral_hf_trust_patches()

    mesh_device = open_devstral_demo_mesh(args.mesh_width)
    try:
        dtype_tt = ttnn.bfloat16

        processor = AutoProcessor.from_pretrained(
            args.model_id,
            trust_remote_code=True,
            fix_mistral_regex=True,
            cache_dir=os.getenv("HF_TOKENIZER_CACHE") or os.getenv("HF_HUB_CACHE") or None,
            local_files_only=False,
            token=os.getenv("HF_TOKEN") or None,
        )
        tokenizer = getattr(processor, "tokenizer", processor)
        if args.messages_json is not None:
            bench_messages = _load_messages_json(args.messages_json, args.benchmark_scenario)
            raw_ids = tokenizer.apply_chat_template(
                conversation=bench_messages,
                return_tensors="pt",
                return_dict=True,
            )["input_ids"]
            input_ids, full_prompt_len, prompt_len = _truncate_input_ids_tail(raw_ids, args.input_seq_len)
            logger.info(
                f"Benchmark prompt from {args.messages_json}: full={full_prompt_len} tokens, "
                f"using tail={prompt_len} tokens."
            )
        else:
            input_ids = _tokenize_demo_prompt(tokenizer, args.prompt, args.system_prompt)["input_ids"]
            prompt_len = int(input_ids.shape[1])

        extra_tokens = max(0, args.max_new_tokens)
        max_seq = max(4096, prompt_len + extra_tokens + 2048)
        chunk_size = int(args.prefill_chunk_size)
        use_paged = prompt_len > chunk_size
        if use_paged:
            max_seq += chunk_size
        max_seq = ((max_seq + 511) // 512) * 512

        paged_attention_config = None
        if use_paged:
            paged_block_size = 64
            max_num_blocks = (max_seq + paged_block_size - 1) // paged_block_size
            max_num_blocks = ((max_num_blocks + 7) // 8) * 8
            paged_attention_config = PagedAttentionConfig(block_size=paged_block_size, max_num_blocks=max_num_blocks)
            logger.info(
                f"Chunked+paged prefill: prompt {prompt_len} tok, chunk_size {chunk_size}, "
                f"block_size {paged_block_size}, max_num_blocks {max_num_blocks}, max_seq {max_seq}."
            )

        model_args = ModelArgs(
            mesh_device,
            max_batch_size=1,
            max_seq_len=max_seq,
            dummy_weights=False,
            use_hf_rope=True,
            cache_hf=True,
        )
        # Multi-chip: prefill uses gathered residual (replicated norm); decode keeps width-sharded residual (distributed norm). Single device: replicated both.
        model_args.is_distributed_norm = types.MethodType(
            lambda self, mode: self.is_multichip and mode == Mode.DECODE,
            model_args,
        )

        try:
            meta_state_dict = model_args.load_state_dict()
        except Exception as exc:
            raise RuntimeError(
                f"Checkpoint load failed (memory, hub, FP8, etc.): {exc}\n"
                "Ensure HF access, enough RAM, and compatible transformers."
            ) from exc

        if args.text_layers is not None:
            if args.text_layers < 1 or args.text_layers > model_args.full_model_n_layers:
                raise ValueError(
                    f"--text-layers must be in [1, {model_args.full_model_n_layers}], got {args.text_layers}"
                )
            model_args.n_layers = args.text_layers
            if args.max_new_tokens > 0 and not args.hf_generate:
                logger.warning("Partial --text-layers: TT generation will not match full-model HF quality.")

        hf_full = model_args.cached_hf_model
        if hf_full is None:
            raise RuntimeError("Expected cached HF model after load_state_dict with cache_hf=True.")
        hf_inner = hf_full.model
        if not isinstance(hf_inner, Mistral3Model):
            raise TypeError(f"Expected Mistral3Model, got {type(hf_inner)}")

        text_cfg = model_args.hf_config.text_config
        if not isinstance(text_cfg, Ministral3Config):
            raise TypeError(f"Demo expects Ministral3Config as text_config, got {type(text_cfg)!r}")
        rope_params = resolve_rope_parameters(text_cfg)

        shared_tt_ccl = TT_CCL(mesh_device)

        tt_model = TtMinistral3Model(
            mesh_device=mesh_device,
            tt_ccl=shared_tt_ccl,
            model_args=model_args,
            meta_state_dict=meta_state_dict,
            weight_cache_path=model_args.weight_cache_path(dtype_tt),
            dtype=dtype_tt,
            transformation_mats={"decode": None, "prefill": None},
            configuration=model_args,
            llama_4_scaling_beta=rope_params.get("llama_4_scaling_beta"),
            original_max_position_embeddings=rope_params.get("original_max_position_embeddings"),
            ministral_text_config=text_cfg,
            paged_attention_config=paged_attention_config,
        )

        sd_prefix = model_args.get_state_dict_prefix("", None)
        out_key = f"{sd_prefix}output.weight"
        if out_key not in meta_state_dict:
            raise RuntimeError(f"Missing {out_key!r} in meta state dict (required for LM head).")
        lm_head_max_cols = demo_lm_head_max_columns_per_device(model_args, cli_cap=args.lm_head_max_device_cols)
        tt_lm_head = LMHead(
            args=model_args,
            mesh_device=mesh_device,
            tt_ccl=shared_tt_ccl,
            dtype=dtype_tt,
            state_dict=meta_state_dict,
            state_dict_prefix=sd_prefix,
            weight_cache_path=model_args.weight_cache_path(dtype_tt),
            max_columns_per_device=lm_head_max_cols,
        )

        if not devstral_supports_on_device_sampling(model_args, mesh_device):
            raise RuntimeError("On-device Sampling1D is required but unsupported for this vocab/mesh.")

        sampling = DevstralSampling1DAdapter(
            args=model_args,
            mesh_device=mesh_device,
            tt_ccl=shared_tt_ccl,
        )
        sampling_empty_slots = list(range(sampling.max_batch_size))
        seed_for_params = args.seed if args.seed is not None else None
        if not prefer_stochastic_sampling:
            sampling_in = SamplingParams(temperature=0.0, top_k=32, top_p=1.0, seed=seed_for_params)
        else:
            sampling_in = SamplingParams(
                temperature=float(args.temperature),
                top_k=32,
                top_p=1.0,
                seed=seed_for_params,
            )
        formatted_sampling = format_sampling_params(sampling_in, len(sampling_empty_slots))
        sampling.reset_sampling_params(formatted_sampling)

        input_ids = input_ids.to(hf_inner.get_input_embeddings().weight.device)
        seq_len_lm = int(input_ids.shape[1])
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        pad_token_id = 0 if pad_token_id is None else int(pad_token_id)

        if args.verify:
            if use_paged:
                logger.warning(
                    "--verify skipped: single-shot prefill PCC does not fit L1 at this prompt length "
                    f"({seq_len_lm} tokens). Use a shorter prompt or omit --verify."
                )
            else:
                tt_lm_torch = tt_prefill_hidden_states_from_ids(
                    input_ids, pad_token_id, mesh_device, tt_model, seq_len_lm, model_args
                )
                text_root = text_model_root(hf_inner)
                rotary = text_root.rotary_emb
                rotary.eval()
                merged = hf_inner.get_input_embeddings()(input_ids).to(torch.bfloat16)
                position_ids_lm = torch.arange(seq_len_lm, dtype=torch.long, device=merged.device).unsqueeze(0)
                position_embeddings_hf = rotary(merged, position_ids=position_ids_lm)
                causal_mask = create_causal_mask(
                    config=text_cfg,
                    inputs_embeds=merged,
                    attention_mask=None,
                    past_key_values=None,
                    position_ids=position_ids_lm,
                )
                hidden = merged
                for layer in text_root.layers[: model_args.n_layers]:
                    hidden = layer(
                        hidden_states=hidden,
                        attention_mask=causal_mask,
                        position_ids=position_ids_lm,
                        past_key_values=None,
                        use_cache=False,
                        position_embeddings=position_embeddings_hf,
                    )
                ref_out = text_root.norm(hidden)
                tt_cmp = tt_lm_torch
                if tt_cmp.shape != ref_out.shape:
                    tt_cmp = tt_cmp.reshape(ref_out.shape)
                pcc_ok, msg = comp_pcc(ref_out, tt_cmp, 0.90)
                if not pcc_ok:
                    logger.warning(f"PCC check did not reach threshold: {msg}")

        if args.max_new_tokens > 0:
            if args.hf_generate:
                gen_device = next(hf_full.parameters()).device
                if args.seed is not None:
                    torch.manual_seed(args.seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(args.seed)
                prompt_len_before_gen = int(input_ids.shape[1])
                with torch.inference_mode():
                    out = hf_full.generate(input_ids.to(gen_device), **DEFAULT_GENERATE_KWARGS)
                answer_text = tokenizer.decode(out[0, prompt_len_before_gen:].tolist(), skip_special_tokens=False)
                logger.info(answer_text)
            else:
                if args.seed is not None:
                    torch.manual_seed(args.seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(args.seed)

                eos_ids = eos_token_ids(hf_full.config, tokenizer)
                if not eos_ids:
                    logger.warning(
                        "No eos_token_id on config/text_config/tokenizer; TT loop will only stop at --max-new-tokens."
                    )
                gen_sl = seq_len_lm
                ids_tt_gen = host_input_ids_to_tt_replicated(mesh_device, input_ids)
                page_table_host: torch.Tensor | None = None
                page_table_tt: ttnn.Tensor | None = None
                if use_paged:
                    assert paged_attention_config is not None
                    page_table_host = torch.arange(paged_attention_config.max_num_blocks, dtype=torch.int32).reshape(
                        1, -1
                    )
                    page_table_tt = ttnn.from_torch(
                        page_table_host,
                        device=mesh_device,
                        dtype=ttnn.int32,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                    )
                pad_row = hf_inner.get_input_embeddings()(torch.tensor([[pad_token_id]], device=input_ids.device))[
                    0, 0
                ].detach()
                # Keep token history on host: growing ids_tt_gen during trace risks allocator overlap with trace outputs.
                generated_token_ids: list[int] = []

                stats = {
                    "prefill_s": 0.0,
                    "first_sample_s": 0.0,
                    "decode_s": 0.0,
                    "lmhead_s": 0.0,
                    "sample_post_s": 0.0,
                    "trace_capture_s": 0.0,
                    "steps": 0,
                    "ttft_s": None,
                    "first_traced_step_s": None,
                    "wall_s": 0.0,
                }

                def _sample_from_prefill_out(tt_out: ttnn.Tensor, last_token_index: int) -> int:
                    """Sample next token at ``last_token_index``; update ``stats`` lm-head/sample timings."""
                    tok_slot = last_token_index % 32
                    t0 = time.perf_counter()
                    logits_tt = tt_lm_head_logits_block(tt_out, last_token_index, model_args, tt_lm_head)
                    stats["lmhead_s"] += time.perf_counter() - t0
                    t0 = time.perf_counter()
                    sample_result = sampling.sample(logits_tt, enable_trace=False)
                    tt_next = sample_result[0] if isinstance(sample_result, tuple) else sample_result
                    out = tt_sampling_output_token_id(tt_next, tok_slot)
                    ttnn.deallocate(logits_tt)
                    stats["sample_post_s"] += time.perf_counter() - t0
                    ttnn.deallocate(tt_out)
                    return out

                decode_trace_ctx = None
                decode_buffers = None
                eos_set = set(eos_ids)
                try:
                    if use_paged:
                        assert page_table_tt is not None and page_table_host is not None
                        assert paged_attention_config is not None
                        warm_merged_bf = hf_inner.get_input_embeddings()(input_ids).to(torch.bfloat16)
                        warm_out, warm_sample_idx = _tt_prefill_paged_chunked(
                            warm_merged_bf,
                            pad_row,
                            mesh_device,
                            tt_model,
                            seq_len_lm,
                            chunk_size,
                            paged_attention_config.block_size,
                            page_table_host,
                            page_table_tt,
                        )
                    else:
                        warm_out = tt_forward_prefill_from_device_ids(
                            ids_tt_gen, gen_sl, pad_token_id, mesh_device, tt_model, model_args
                        )
                        warm_sample_idx = gen_sl - 1
                    tt_warmup_prefill_lm_head_sampling(
                        warm_out,
                        warm_sample_idx,
                        model_args,
                        tt_lm_head,
                        sampling=sampling,
                    )

                    if args.max_new_tokens > 1:
                        decode_buffers = tt_alloc_decode_input_buffers(mesh_device)
                        tt_update_decode_input_buffers(mesh_device, decode_buffers, 0, gen_sl)
                        tt_warmup_decode_trace_path(
                            mesh_device,
                            tt_model,
                            model_args,
                            decode_buffers,
                            tt_lm_head=tt_lm_head,
                            sampling=sampling,
                            page_table=page_table_tt,
                        )

                    run_t0 = time.perf_counter()
                    _profiler_signpost("prefill-start")
                    t0 = time.perf_counter()
                    if use_paged:
                        assert page_table_tt is not None and page_table_host is not None
                        assert paged_attention_config is not None
                        merged_bf = hf_inner.get_input_embeddings()(input_ids).to(torch.bfloat16)
                        tt_out, sample_idx = _tt_prefill_paged_chunked(
                            merged_bf,
                            pad_row,
                            mesh_device,
                            tt_model,
                            seq_len_lm,
                            chunk_size,
                            paged_attention_config.block_size,
                            page_table_host,
                            page_table_tt,
                        )
                    else:
                        tt_out = tt_forward_prefill_from_device_ids(
                            ids_tt_gen, gen_sl, pad_token_id, mesh_device, tt_model, model_args
                        )
                        sample_idx = gen_sl - 1
                    stats["prefill_s"] = time.perf_counter() - t0

                    t0 = time.perf_counter()
                    next_scalar = _sample_from_prefill_out(tt_out, sample_idx)
                    stats["first_sample_s"] = time.perf_counter() - t0
                    _profiler_signpost("prefill-end")
                    stats["ttft_s"] = time.perf_counter() - run_t0
                    stats["steps"] = 1

                    if next_scalar in eos_set or args.max_new_tokens <= 1:
                        if next_scalar not in eos_set:
                            generated_token_ids.append(next_scalar)
                            gen_sl += 1
                    else:
                        generated_token_ids.append(next_scalar)
                        gen_sl += 1

                        if decode_buffers is None:
                            decode_buffers = tt_alloc_decode_input_buffers(mesh_device)
                        tt_update_decode_input_buffers(mesh_device, decode_buffers, int(next_scalar), gen_sl - 1)
                        _profiler_signpost("trace-capture-start")
                        t_capture = time.perf_counter()
                        decode_trace_ctx = tt_capture_decode_trace(
                            mesh_device,
                            tt_model,
                            model_args,
                            decode_buffers,
                            tt_lm_head=tt_lm_head,
                            sampling=sampling,
                            page_table=page_table_tt,
                            prewarmed=True,
                        )
                        stats["trace_capture_s"] = time.perf_counter() - t_capture
                        _profiler_signpost("trace-capture-end")

                        _profiler_signpost("decode-loop-start")
                        for _ in range(1, args.max_new_tokens):
                            if next_scalar in eos_set:
                                break

                            decode_pos = gen_sl - 1  # absolute position of the just-appended token
                            step_t0 = time.perf_counter()
                            t0 = time.perf_counter()
                            tt_update_decode_input_buffers(
                                mesh_device, decode_trace_ctx.buffers, int(next_scalar), decode_pos
                            )
                            tt_execute_decode_trace(mesh_device, decode_trace_ctx)
                            stats["decode_s"] += time.perf_counter() - t0

                            t0 = time.perf_counter()
                            if decode_trace_ctx.output_tokens is not None:
                                next_scalar = tt_read_decode_traced_token(decode_trace_ctx, batch_slot=0)
                                stats["sample_post_s"] += time.perf_counter() - t0
                            else:
                                raise RuntimeError("Decode trace did not produce on-device sampled tokens.")

                            if stats["first_traced_step_s"] is None:
                                stats["first_traced_step_s"] = time.perf_counter() - step_t0

                            if next_scalar in eos_set:
                                break
                            generated_token_ids.append(next_scalar)
                            gen_sl += 1
                            stats["steps"] += 1
                        _profiler_signpost("decode-loop-end")

                    stats["wall_s"] = time.perf_counter() - run_t0

                    wall_s = stats["wall_s"]
                    steps = stats["steps"]
                    decode_steps = max(steps - 1, 0)
                    ttft_s = stats["ttft_s"] or 0.0

                    def _pct(part: float) -> float:
                        return 100.0 * part / wall_s if wall_s > 0 else 0.0

                    def _decode_avg_ms(part: float) -> float:
                        return 1000.0 * part / decode_steps if decode_steps > 0 else 0.0

                    decode_loop_total_s = stats["decode_s"] + stats["lmhead_s"] + stats["sample_post_s"]
                    steady_per_tok_s = decode_loop_total_s / decode_steps if decode_steps > 0 else 0.0
                    steady_tok_s = 1.0 / steady_per_tok_s if steady_per_tok_s > 0 else 0.0
                    thr = steps / wall_s if wall_s > 0 else 0.0

                    print()
                    print("──────────────────────────────────────────────────────────────")
                    print(
                        f"  TT · traced decode  ({steps} new token(s); {decode_steps} traced decode step(s); wall {wall_s:.2f} s)"
                    )
                    print("──────────────────────────────────────────────────────────────")
                    print(f"  {'Phase':<22} {'total':>14}     %")
                    print(
                        f"  {'prefill (1x)':<22} {stats['prefill_s'] * 1000:>10.2f} ms  {_pct(stats['prefill_s']):>5.1f}%"
                    )
                    print(
                        f"  {'first-token sample':<22} {stats['first_sample_s'] * 1000:>10.2f} ms  {_pct(stats['first_sample_s']):>5.1f}%"
                    )
                    print(
                        f"  {'trace capture (1x)':<22} {stats['trace_capture_s'] * 1000:>10.2f} ms  {_pct(stats['trace_capture_s']):>5.1f}%   (decode + sampling)"
                    )
                    print(
                        f"  {'traced decode submit':<22} {stats['decode_s'] * 1000:>10.2f} ms  {_pct(stats['decode_s']):>5.1f}%"
                        f"   (avg {_decode_avg_ms(stats['decode_s']):.2f} ms / decoded tok)"
                    )
                    print(
                        f"  {'lm head (decode)':<22} {stats['lmhead_s'] * 1000:>10.2f} ms  {_pct(stats['lmhead_s']):>5.1f}%"
                        f"   (avg {_decode_avg_ms(stats['lmhead_s']):.2f} ms / decoded tok)"
                    )
                    print(
                        f"  {'sample / post-decode':<22} {stats['sample_post_s'] * 1000:>10.2f} ms  {_pct(stats['sample_post_s']):>5.1f}%"
                        f"   (avg {_decode_avg_ms(stats['sample_post_s']):.2f} ms / decoded tok)"
                    )
                    print("──────────────────────────────────────────────────────────────")
                    print(f"  TTFT (prompt -> 1st new tok)        {ttft_s * 1000:>10.2f} ms")
                    if stats["first_traced_step_s"] is not None:
                        print(f"  First traced decode step latency    {stats['first_traced_step_s'] * 1000:>10.2f} ms")
                    if decode_steps > 0:
                        print(f"  Steady-state decode latency / tok   {steady_per_tok_s * 1000:>10.2f} ms")
                        print(f"  Steady-state decode throughput      {steady_tok_s:>10.3f} tok/s")
                    print(f"  End-to-end throughput               {thr:>10.3f} tok/s")
                    print("──────────────────────────────────────────────────────────────")

                    answer_ids = torch.tensor(generated_token_ids, dtype=torch.long)
                    answer_text = tokenizer.decode(answer_ids.tolist(), skip_special_tokens=False)
                    print(answer_text)
                finally:
                    if decode_trace_ctx is not None:
                        tt_release_decode_trace(mesh_device, decode_trace_ctx)
                    elif decode_buffers is not None:
                        ttnn.deallocate(decode_buffers.token_ids)
                        ttnn.deallocate(decode_buffers.pos_uint32)
                        ttnn.deallocate(decode_buffers.pos_int32)
                    if page_table_tt is not None:
                        ttnn.deallocate(page_table_tt)
                    ttnn.deallocate(ids_tt_gen)
    finally:
        close_devstral_demo_mesh(mesh_device)


if __name__ == "__main__":
    main()
