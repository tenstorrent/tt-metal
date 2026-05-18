# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Tenstorrent demo: Devstral Small 2 (Mistral3) **text** LM on TT, with **autoregressive generation**
using **TT** ``embed_tokens`` and **TT** ``TtMinistral3RotaryEmbedding`` inside ``TtMinistral3Model`` (via
``forward_prefill``), **TT** ``LMHead`` or ``--lm-head-cpu`` chunked torch logits, and sampling either via
**SamplingGenerator** on device (mirroring ``Transformer`` + ``format_sampling_params`` / ``seed_manager``) or
PyTorch softmax/multinomial on the host when ``--cpu-sampling`` is set or the vocab/mesh layout is unsupported.

``--verify`` still builds an HF reference with **HF** embeddings for layer-wise PCC; TT side is full TT prefill.

**Prompting** defaults to the **same** chat template as ``reference/inference.py`` (``inference_fixtures``:
system prompt, user fibonacci/tools task, and tool schemas). Use ``--simple-chat`` for a single user turn
from ``--prompt`` only.

**Generation** is **one TT prefill** of the full prompt (fills KV cache) followed by **traced TT decode**
for each new token. The decode trace covers ``forward_decode → LM head → SamplingGenerator`` when
on-device sampling is supported, ``forward_decode → LM head`` when sampling falls back to host, or
``forward_decode`` alone when ``--lm-head-cpu`` is set. A one-shot warmup compiles the kernels before
``ttnn.begin_trace_capture`` so the very first replayed iteration already runs at steady-state cost.
Prompt token ids stay on device (**``ttnn.concat``** to extend, **``ttnn.pad``** / **``ttnn.arange``**
inside prefill). Only ``tokenizer.decode`` pulls ids to host. Matching ``inference.py`` setup requires
**full** ``--text-layers`` (omit the flag) and ``--cpu-sampling`` if you need host-side sampling to match
reference RNG exactly.

Usage (from repo root)::

    export HF_MODEL=mistralai/Devstral-Small-2-24B-Instruct-2512
    python models/experimental/devstarl2_small/demo/demo_devstral2_tt_multimodal.py --seed 0

    # Bring-up: fewer layers + PCC check
    python models/experimental/devstarl2_small/demo/demo_devstral2_tt_multimodal.py \\
        --text-layers 1 --verify --max-new-tokens 20

    # Optional: pure HF ``generate()`` baseline (not TT)
    python models/experimental/devstarl2_small/demo/demo_devstral2_tt_multimodal.py --hf-generate --seed 0

Embeddings are padded so sequence length is **128**-divisible, **KV-shard tile**-aligned, and (when
``L > 1024``) a multiple of **1024** for the attention ``wo`` chunk reshape; hidden states are sliced
back to the real length for PCC. Logits use **TT** ``LMHead`` on the last-token 32-row block.

If the LM head hits Wormhole L1 circular-buffer limits, use ``--lm-head-cpu`` (chunked torch matmul),
lower shards via ``--lm-head-max-device-cols``, or env ``DEVSTRAL2_LM_HEAD_MAX_COLUMNS_PER_DEVICE``.
"""

from __future__ import annotations

import argparse
import os
import types

import torch
from loguru import logger
from transformers import MistralCommonBackend
from transformers.masking_utils import create_causal_mask
from transformers.models.ministral3.configuration_ministral3 import Ministral3Config
from transformers.models.mistral3.modeling_mistral3 import Mistral3Model

import ttnn
from models.common.sampling import SamplingGenerator, SamplingParams, format_sampling_params
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.devstarl2_small.devstral_utils import (
    DEFAULT_MODEL_ID,
    apply_devstral_hf_trust_patches,
    apply_fp8_dequantize_compat,
    close_devstral_demo_mesh,
    cpu_lm_head_logits_last_token,
    demo_lm_head_max_columns_per_device,
    devstral_supports_on_device_sampling,
    eos_token_ids,
    host_input_ids_to_tt_replicated,
    open_devstral_demo_mesh,
    text_model_root,
    tt_alloc_decode_input_buffers,
    tt_append_uint32_token,
    tt_capture_decode_trace,
    tt_execute_decode_trace,
    tt_forward_prefill_from_device_ids,
    tt_lm_head_logits_block,
    tt_lm_head_logits_last_token,
    tt_prefill_hidden_states_from_ids,
    tt_prefill_target_seqlen,
    tt_read_decode_traced_hidden,
    tt_read_decode_traced_logits,
    tt_read_decode_traced_token,
    tt_release_decode_trace,
    tt_replicated_ids_to_torch_long,
    tt_sampling_output_token_id,
    tt_update_decode_input_buffers,
)
from models.experimental.devstarl2_small.reference.inference_fixtures import (
    REFERENCE_GENERATE_KWARGS,
    REFERENCE_MESSAGES,
    REFERENCE_TOOLS,
)
from models.experimental.devstarl2_small.tt.tt_ministral3_model import TtMinistral3Model
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.lm_head import LMHead
from models.tt_transformers.tt.model_config import DecodersPrecision, ModelArgs

apply_fp8_dequantize_compat()


def main():
    ref_max = int(REFERENCE_GENERATE_KWARGS["max_new_tokens"])
    ref_temp = float(REFERENCE_GENERATE_KWARGS["temperature"])
    ref_do_sample = bool(REFERENCE_GENERATE_KWARGS["do_sample"])

    parser = argparse.ArgumentParser(
        description="Devstral-2 on TT: Ministral3 prefill + LMHead; on-device SamplingGenerator when supported, else CPU logits sampling."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="HF repo id (also set HF_MODEL).")
    parser.add_argument(
        "--prompt",
        default=(
            "Can you implement in Python a method to compute the fibonnaci sequence at the `n`th element "
            "with `n` a parameter passed to the function? Start the sequence from 1."
        ),
        help="Only used with --simple-chat (single user turn).",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional system message for --simple-chat only.",
    )
    parser.add_argument(
        "--simple-chat",
        action="store_true",
        help="Tokenize a minimal chat from --prompt/--system-prompt only. "
        "Default: same messages+tools as reference/inference.py (inference_fixtures).",
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
        help=f"New tokens after prompt (default {ref_max}, same as reference/inference.py). 0 = skip generation.",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help=f"Greedy argmax (default: sample with temperature {ref_temp}, like reference/inference.py).",
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
        "--lm-head-cpu",
        action="store_true",
        help="LM head as chunked torch matmul on CPU (avoids Wormhole L1 static CB limits on TT lm_head).",
    )
    parser.add_argument(
        "--lm-head-max-device-cols",
        type=int,
        default=None,
        help="On-device LMHead: max vocab columns per matmul shard (default 4096; try 2048 if L1 error remains).",
    )
    parser.add_argument(
        "--cpu-sampling",
        action="store_true",
        help="Force PyTorch softmax/argmax on host from TT logits (disables on-device SamplingGenerator).",
    )
    args = parser.parse_args()
    prefer_stochastic_sampling = (not args.greedy) and ref_do_sample

    if args.hf_generate and args.text_layers is not None:
        logger.warning("--hf-generate uses full HF depth; ignoring --text-layers.")
        args.text_layers = None

    os.environ["HF_MODEL"] = args.model_id
    apply_devstral_hf_trust_patches()

    mesh_device = open_devstral_demo_mesh(max(1, min(args.mesh_width, ttnn.get_num_devices())))
    try:
        dtype_tt = ttnn.bfloat16

        tokenizer = MistralCommonBackend.from_pretrained(
            args.model_id,
            trust_remote_code=True,
            local_files_only=os.getenv("CI") == "true",
        )
        if args.simple_chat:
            if args.system_prompt:
                messages = [
                    {"role": "system", "content": args.system_prompt},
                    {"role": "user", "content": [{"type": "text", "text": args.prompt}]},
                ]
                tokenized = tokenizer.apply_chat_template(
                    conversation=messages,
                    return_tensors="pt",
                    return_dict=True,
                )
            else:
                messages = [{"role": "user", "content": [{"type": "text", "text": args.prompt}]}]
                tokenized = tokenizer.apply_chat_template(
                    conversation=messages,
                    return_tensors="pt",
                    return_dict=True,
                )
        else:
            tokenized = tokenizer.apply_chat_template(
                conversation=REFERENCE_MESSAGES,
                tools=REFERENCE_TOOLS,
                return_tensors="pt",
                return_dict=True,
            )

        input_ids = tokenized["input_ids"]
        prompt_len = int(input_ids.shape[1])
        extra_tokens = max(0, args.max_new_tokens)
        # Upper bound padded prefill length: 128-step fixes for KV shard tile height + generation growth.
        # Round up to a multiple of 512 so SDPA decode k_chunk_size is always >= 512 (a multiple of 32).
        max_seq = max(4096, prompt_len + extra_tokens + 2048)
        max_seq = ((max_seq + 511) // 512) * 512

        # ``optimizations``: bfp4 MLP FF1/FF3 + bfp8 elsewhere. Cuts decode
        # bandwidth from ~48 GB/token (bf16) to ~14 GB/token. MLP and
        # attention modules read this via ``args.decoders_optimizations``.
        model_args = ModelArgs(
            mesh_device,
            max_batch_size=1,
            max_seq_len=max_seq,
            dummy_weights=False,
            use_hf_rope=True,
            cache_hf=True,
            optimizations=lambda ma: DecodersPrecision.performance(ma.n_layers, ma.model_name),
        )
        # Multi-chip: PREFILL is gathered to full-dim inside the decoder layer
        # (replicated norm), DECODE leaves residual width-fractured across chips
        # (distributed norm). Single chip: replicated norm for both modes.
        model_args.is_distributed_norm = types.MethodType(
            lambda self, mode: self.is_multichip and mode == Mode.DECODE,
            model_args,
        )

        logger.info("Loading checkpoint via ModelArgs.load_state_dict() …")
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
                logger.warning(
                    "Partial --text-layers: TT generation will not match full-model reference/inference.py quality."
                )

        hf_full = model_args.cached_hf_model
        if hf_full is None:
            raise RuntimeError("Expected cached HF model after load_state_dict with cache_hf=True.")
        hf_inner = hf_full.model
        if not isinstance(hf_inner, Mistral3Model):
            raise TypeError(f"Expected Mistral3Model, got {type(hf_inner)}")

        text_cfg = model_args.hf_config.text_config
        if not isinstance(text_cfg, Ministral3Config):
            raise TypeError(f"Demo expects Ministral3Config as text_config, got {type(text_cfg)!r}")
        rope_params = getattr(text_cfg, "rope_parameters", None) or {}
        if not isinstance(rope_params, dict):
            rope_params = dict(rope_params)

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
        )

        sd_prefix = model_args.get_state_dict_prefix("", None)
        out_key = f"{sd_prefix}output.weight"
        if out_key not in meta_state_dict:
            raise RuntimeError(f"Missing {out_key!r} in meta state dict (required for LM head).")
        lm_head_weight_cpu: torch.Tensor | None = None
        tt_lm_head: LMHead | None = None
        if args.lm_head_cpu:
            lm_head_weight_cpu = meta_state_dict[out_key].detach().to(torch.bfloat16).cpu().contiguous()
            logger.info(
                f"CPU LM head: chunked torch matmul; weight {tuple(lm_head_weight_cpu.shape)} {lm_head_weight_cpu.dtype}."
            )
        else:
            lm_head_max_cols = demo_lm_head_max_columns_per_device(model_args, cli_cap=args.lm_head_max_device_cols)
            logger.info(
                f"On-device LMHead max columns per shard: {lm_head_max_cols} "
                f"(ModelArgs value {model_args.max_columns_per_device_lm_head})."
            )
            if lm_head_max_cols < int(model_args.max_columns_per_device_lm_head):
                logger.info(
                    "Tune with --lm-head-max-device-cols or DEVSTRAL2_LM_HEAD_MAX_COLUMNS_PER_DEVICE, "
                    "or use --lm-head-cpu if L1 CB errors persist."
                )
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

        use_device_sampling = (
            not args.lm_head_cpu
            and not args.cpu_sampling
            and tt_lm_head is not None
            and devstral_supports_on_device_sampling(model_args, mesh_device)
        )
        if (
            tt_lm_head is not None
            and not args.cpu_sampling
            and not devstral_supports_on_device_sampling(model_args, mesh_device)
        ):
            logger.warning(
                "Vocab size / mesh splits exceed on-device sampling limit (64k per split); "
                "using CPU softmax on logits."
            )

        sampling: SamplingGenerator | None = None
        sampling_empty_slots: list[int] | None = None
        if use_device_sampling:
            sampling = SamplingGenerator(
                args=model_args,
                mesh_device=mesh_device,
                tt_ccl=shared_tt_ccl,
                enable_internal_trace=False,
            )
            sampling_empty_slots = list(range(sampling.tt_sampling.max_batch_size))
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
            sampling.seed_manager.reset_seed(formatted_sampling.seed, sampling_empty_slots)

        _sampling_splits = model_args.num_devices if list(mesh_device.shape) != [1, 1] else 2
        if sampling is not None and model_args.vocab_size // _sampling_splits <= 64 * 1024:
            logger.info(
                f"Using on-device SamplingGenerator (vocab split check OK; {len(sampling_empty_slots)} sampling slots)."
            )

        input_ids = input_ids.to(hf_inner.get_input_embeddings().weight.device)
        seq_len_lm = int(input_ids.shape[1])
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = 0
        else:
            pad_token_id = int(pad_token_id)

        target_lm = tt_prefill_target_seqlen(seq_len_lm, int(model_args.n_kv_heads), int(model_args.cluster_shape[1]))
        if target_lm != seq_len_lm:
            logger.info(
                f"Padded language sequence length {seq_len_lm} → {target_lm} for TT prefill "
                "(128-divisible seq + KV prefill shard tile alignment)."
            )

        logger.info(
            f"TT language model prefill ({model_args.n_layers} decoder layers; TT embed + TtMinistral3RotaryEmbedding); "
            f"prompt template: {'simple-chat' if args.simple_chat else 'inference_fixtures (inference.py)'}."
        )
        tt_lm_torch = tt_prefill_hidden_states_from_ids(
            input_ids, pad_token_id, mesh_device, tt_model, seq_len_lm, model_args
        )

        logger.info(f"TT prefill hidden states shape (batch, seq, dim): {tuple(tt_lm_torch.shape)}")

        if args.verify:
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
            logger.info(comp_allclose(ref_out, tt_cmp))
            logger.info(f"PCC (HF ref on HF embeddings vs full TT prefill): {msg}")
            if not pcc_ok:
                logger.warning(f"PCC check did not reach threshold: {msg}")

        if args.max_new_tokens <= 0:
            pass
        elif args.hf_generate:
            gen_device = next(hf_full.parameters()).device
            if args.seed is not None:
                torch.manual_seed(args.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(args.seed)
            prompt_vec = tokenized["input_ids"][0]
            input_ids_gen = tokenized["input_ids"].to(gen_device)
            logger.info(f"HF baseline generate {REFERENCE_GENERATE_KWARGS} on {gen_device} …")
            with torch.inference_mode():
                out = hf_full.generate(input_ids_gen, **REFERENCE_GENERATE_KWARGS)
            seq = out[0]
            answer_text = tokenizer.decode(seq[len(prompt_vec) :].tolist(), skip_special_tokens=False)
            logger.info(f"HF generate baseline ({seq.numel() - prompt_vec.numel()} new tokens):\n{answer_text}")
        else:
            do_sample = prefer_stochastic_sampling
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

            mode = "greedy" if not prefer_stochastic_sampling else f"sample (T={args.temperature})"
            lm_mode = "CPU lm_head (chunked torch)" if args.lm_head_cpu else "TT lm_head"
            samp_mode = (
                "on-device SamplingGenerator (mirrors Transformer + format_sampling_params / seed_manager)"
                if sampling is not None
                else "CPU softmax / argmax on TT logits (host)"
            )
            logger.info(
                f"TT generation: prompt prefill once, then traced decode (warmup + capture) for up to "
                f"{args.max_new_tokens} new tokens; mode {mode}; TT decoder + {lm_mode}; {samp_mode}."
            )

            def _sample_from_prefill_out(tt_out: ttnn.Tensor, last_token_index: int) -> int:
                """Single token sample from a prefill hidden-states block at ``last_token_index``."""
                if sampling is not None:
                    assert tt_lm_head is not None
                    sampling.seed_manager.get_new_values()
                    logits_tt = tt_lm_head_logits_block(tt_out, last_token_index, model_args, tt_lm_head)
                    sample_result = sampling.sample(logits_tt, enable_trace=False)
                    tt_next = sample_result[0] if isinstance(sample_result, tuple) else sample_result
                    out = tt_sampling_output_token_id(tt_next, last_token_index % 32)
                    ttnn.deallocate(logits_tt)
                    return out
                if args.lm_head_cpu:
                    assert lm_head_weight_cpu is not None
                    logits_row = cpu_lm_head_logits_last_token(
                        tt_out, last_token_index, mesh_device, lm_head_weight_cpu, int(model_args.vocab_size)
                    )
                else:
                    assert tt_lm_head is not None
                    logits_row = tt_lm_head_logits_last_token(
                        tt_out, last_token_index, mesh_device, model_args, tt_lm_head
                    )
                if do_sample:
                    probs = torch.softmax(logits_row.float().squeeze(0) / max(args.temperature, 1e-6), dim=-1)
                    return int(torch.multinomial(probs, num_samples=1).item())
                return int(logits_row.argmax(dim=-1).item())

            decode_trace_ctx = None
            try:
                # Step 0: one full TT prefill of the prompt (fills the KV cache).
                tt_out = tt_forward_prefill_from_device_ids(
                    ids_tt_gen, gen_sl, pad_token_id, mesh_device, tt_model, model_args
                )
                next_scalar = _sample_from_prefill_out(tt_out, gen_sl - 1)
                ttnn.deallocate(tt_out)

                if next_scalar in eos_ids or args.max_new_tokens <= 1:
                    if next_scalar not in eos_ids:
                        ids_tt_gen = tt_append_uint32_token(ids_tt_gen, next_scalar, mesh_device)
                        gen_sl += 1
                else:
                    ids_tt_gen = tt_append_uint32_token(ids_tt_gen, next_scalar, mesh_device)
                    gen_sl += 1

                    decode_buffers = tt_alloc_decode_input_buffers(mesh_device)
                    tt_update_decode_input_buffers(mesh_device, decode_buffers, int(next_scalar), gen_sl - 1)
                    logger.info("Capturing decode trace (warmup + capture)…")
                    decode_trace_ctx = tt_capture_decode_trace(
                        mesh_device,
                        tt_model,
                        model_args,
                        decode_buffers,
                        tt_lm_head=None if args.lm_head_cpu else tt_lm_head,
                        sampling=sampling,
                    )
                    logger.info("Decode trace ready; entering replay loop.")

                    for _step in range(1, args.max_new_tokens):
                        decode_pos = gen_sl - 1  # absolute position of the just-appended token
                        if sampling is not None:
                            # Advance the SamplingGenerator host-side RNG / push fresh device
                            # seeds before each replay (no-op in steady unseeded state).
                            sampling.seed_manager.get_new_values()
                        tt_update_decode_input_buffers(
                            mesh_device, decode_trace_ctx.buffers, int(next_scalar), decode_pos
                        )
                        tt_execute_decode_trace(mesh_device, decode_trace_ctx)

                        if decode_trace_ctx.output_tokens is not None:
                            next_scalar = tt_read_decode_traced_token(decode_trace_ctx, batch_slot=0)
                        elif decode_trace_ctx.output_logits is not None:
                            logits_row = tt_read_decode_traced_logits(
                                decode_trace_ctx, mesh_device, model_args, batch_slot=0
                            )
                            if do_sample:
                                probs = torch.softmax(
                                    logits_row.float().squeeze(0) / max(args.temperature, 1e-6), dim=-1
                                )
                                next_scalar = int(torch.multinomial(probs, num_samples=1).item())
                            else:
                                next_scalar = int(logits_row.argmax(dim=-1).item())
                        else:
                            h_clone = tt_read_decode_traced_hidden(decode_trace_ctx, mesh_device)
                            next_scalar = _sample_from_prefill_out(h_clone, 0)
                            ttnn.deallocate(h_clone)

                        if next_scalar in eos_ids:
                            break
                        ids_tt_gen = tt_append_uint32_token(ids_tt_gen, next_scalar, mesh_device)
                        gen_sl += 1

                ids_host = tt_replicated_ids_to_torch_long(mesh_device, ids_tt_gen, gen_sl)
                answer_ids = ids_host[prompt_len:]
                answer_text = tokenizer.decode(answer_ids.tolist(), skip_special_tokens=False)
                logger.info(f"TT stack + TT lm_head generated ({answer_ids.numel()} tokens):\n{answer_text}")
            finally:
                if decode_trace_ctx is not None:
                    tt_release_decode_trace(mesh_device, decode_trace_ctx)
                ttnn.deallocate(ids_tt_gen)
    finally:
        close_devstral_demo_mesh(mesh_device)


if __name__ == "__main__":
    main()
