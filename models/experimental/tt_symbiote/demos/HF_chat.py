# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Interactive chatbot for HF models with TTNN backend."""

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Fix import path: ensure project root comes before script directory in sys.path
# This prevents importing the local 'models/' subdirectory instead of the project 'models/' package
script_dir = str(Path(__file__).resolve().parent)
project_root = str(Path(__file__).resolve().parents[3])

# Remove script directory from the beginning of sys.path if present
if sys.path and sys.path[0] == script_dir:
    sys.path.pop(0)

# Ensure project root is in sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import (
    LogitsProcessorList,
    NoRepeatNGramLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

import ttnn
from models.experimental.tt_symbiote.core.run_config import DispatchManager, TracedRun
from models.experimental.tt_symbiote.modules.activation import TTNNSilu
from models.experimental.tt_symbiote.modules.linear import TTNNLinearIColShardedWRowSharded
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict
from models.experimental.tt_symbiote.modules.attention import (
    PagedAttentionConfig,
    TTNNPagedAttentionKVCache,
)
from models.experimental.tt_symbiote.modules.decoder_layer import TTNNBailingMoEDecoderLayerPadded
from models.experimental.tt_symbiote.modules.normalization import TTNNDistributedRMSNorm
from models.experimental.tt_symbiote.modules.embedding import TTNNBailingPaddedEmbedding, TTNNBailingRotaryEmbedding
from models.experimental.tt_symbiote.models.bailing_moe_v2 import TTNNBailingMoeV2Model

MESH_DEVICE_MAP = {
    "N150": (1, 1),
    "N300": (1, 2),
    "N150x4": (1, 4),
    "T3K": (1, 8),
    "TG": (8, 4),
    "P150": (1, 1),
    "P300": (1, 2),
    "P150x4": (1, 4),
    "P150x8": (1, 8),
    "BHGLX": (8, 4),
}


def get_mesh_shape():
    env = os.environ.get("MESH_DEVICE")
    if env and env in MESH_DEVICE_MAP:
        return MESH_DEVICE_MAP[env]
    num_devices = len(ttnn.get_device_ids())
    return (1, num_devices)


def setup_mesh_device():
    mesh_shape = get_mesh_shape()
    fabric_config = ttnn.FabricConfig.FABRIC_1D_RING
    ttnn.set_fabric_config(
        fabric_config,
        ttnn.FabricReliabilityMode.STRICT_INIT,
    )
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*mesh_shape),
        num_command_queues=1,
        trace_region_size=200_000_000,
    )
    print(f"Opened mesh device with {mesh_device.get_num_devices()} devices (shape={mesh_shape})")
    return mesh_device


def cleanup(mesh_device):
    TracedRun.release_all()
    for submesh in mesh_device.get_submeshes():
        ttnn.close_mesh_device(submesh)
    ttnn.close_mesh_device(mesh_device)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def create_paged_kv_cache(model_config, device, batch_size=1):
    config = PagedAttentionConfig(
        block_size=64,
        max_num_blocks=32,
        batch_size=batch_size,
    )
    return TTNNPagedAttentionKVCache(
        num_layers=model_config.num_hidden_layers,
        num_kv_heads=model_config.num_key_value_heads,
        head_dim=model_config.head_dim,
        config=config,
        device=None,
    ).to_device(device)


def preprocess_generation_inputs(inputs, model_config, paged_cache, max_new_tokens, device):
    """Strip unused fields, enforce prompt length vs model/KV budget, then move tensors to device."""
    out = {k: v for k, v in inputs.items() if k != "token_type_ids"}

    kv_max = paged_cache.config.max_seq_length
    model_max = getattr(model_config, "max_position_embeddings", kv_max)
    max_total = min(model_max, kv_max)
    reserve = max(1, max_new_tokens)
    max_prompt_len = max(1, max_total - reserve)

    input_ids = out["input_ids"]
    seq_len = input_ids.shape[-1]
    if seq_len > max_prompt_len:
        print(
            f"Warning: prompt truncated from {seq_len} to {max_prompt_len} tokens "
            f"(context {max_total}, reserving {reserve} for generation)."
        )
        for key, value in list(out.items()):
            if isinstance(value, torch.Tensor) and value.shape[-1] == seq_len:
                out[key] = value[..., -max_prompt_len:]

    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in out.items()}


@dataclass
class DecodeParams:
    """Settings for logits post-processing during autoregressive decoding (HF processor / warper semantics)."""

    temperature: float = 0.0
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0


def build_logits_postprocess_processors(params: DecodeParams) -> LogitsProcessorList:
    """Processors applied to raw next-step logits (repetition penalty, banned n-grams)."""
    procs = LogitsProcessorList()
    if params.repetition_penalty != 1.0:
        procs.append(RepetitionPenaltyLogitsProcessor(penalty=params.repetition_penalty))
    if params.no_repeat_ngram_size > 0:
        procs.append(NoRepeatNGramLogitsProcessor(params.no_repeat_ngram_size))
    return procs


def build_logits_postprocess_warpers(params: DecodeParams) -> LogitsProcessorList:
    """Sampling warpers (temperature, top-k, top-p). Only used when ``params.temperature > 0``."""
    if params.temperature <= 0:
        return LogitsProcessorList()
    warp = LogitsProcessorList()
    warp.append(TemperatureLogitsWarper(params.temperature))
    if params.top_k > 0:
        warp.append(TopKLogitsWarper(top_k=params.top_k))
    if params.top_p < 1.0:
        warp.append(TopPLogitsWarper(top_p=params.top_p))
    return warp


def logits_postprocess_generation_kwargs(
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
):
    """
    Build `model.generate` kwargs so Hugging Face applies the same logits post-processing internally.

    Used by warmup; interactive chat uses :func:`decode_with_logit_postprocess` so processing is explicit on ``logits``.
    """
    kwargs = {}
    if repetition_penalty is not None and repetition_penalty != 1.0:
        kwargs["repetition_penalty"] = repetition_penalty
    if no_repeat_ngram_size > 0:
        kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
    if temperature > 0:
        kwargs["do_sample"] = True
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p
        kwargs["top_k"] = top_k
    else:
        kwargs["do_sample"] = False
    return kwargs


def _token_is_eos(token_id: int, eos_token_id) -> bool:
    if eos_token_id is None:
        return False
    if isinstance(eos_token_id, (list, tuple)):
        return token_id in eos_token_id
    return token_id == eos_token_id


def decode_with_logit_postprocess(
    model,
    input_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    past_key_values,
    max_new_tokens: int,
    decode_params: DecodeParams,
):
    """
    Autoregressive decoding with explicit logits post-processing.

    Each step: ``outputs = model(**model_inputs)``, ``logits = outputs.logits[:, -1, :]``, then applies HF
    ``LogitsProcessor`` / warper lists before ``argmax`` or ``multinomial``. Cache bookkeeping uses
    ``GenerationMixin`` helpers only (not ``model.generate``).
    """
    logits_processor = build_logits_postprocess_processors(decode_params)
    logits_warper = build_logits_postprocess_warpers(decode_params)
    do_sample = decode_params.temperature > 0

    model_kwargs: dict = {
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "use_cache": True,
    }
    cur_len = input_ids.shape[-1]
    model_kwargs = model._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

    eos_token_id = model.config.eos_token_id
    is_prefill = True

    for _ in range(max_new_tokens):
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        if is_prefill:
            outputs = model(**model_inputs, return_dict=True)
            is_prefill = False
        else:
            outputs = model(**model_inputs, return_dict=True)

        next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
        next_token_scores = logits_processor(input_ids, next_token_logits)
        if do_sample:
            next_token_scores = logits_warper(input_ids, next_token_scores)
            probs = F.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        model_kwargs = model._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=getattr(model.config, "is_encoder_decoder", False),
        )
        del outputs

        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

        if _token_is_eos(int(next_tokens.item()), eos_token_id):
            break

    return input_ids


def load_model(mesh_device, model_name="inclusionAI/Ling-mini-2.0"):
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    nn_to_ttnn = {
        model.model.layers[0].__class__: TTNNBailingMoEDecoderLayerPadded,
        model.model.norm.__class__: TTNNDistributedRMSNorm,
        nn.Embedding: TTNNBailingPaddedEmbedding,
        model.model.rotary_emb.__class__: TTNNBailingRotaryEmbedding,
    }
    nn_to_ttnn2 = {
        nn.Linear: TTNNLinearIColShardedWRowSharded,
        nn.SiLU: TTNNSilu,
    }

    nn_to_ttnn3 = {
        model.model.__class__: TTNNBailingMoeV2Model,
    }

    modules1 = register_module_replacement_dict(model, nn_to_ttnn, model_config=None)
    modules2 = register_module_replacement_dict(model, nn_to_ttnn2, model_config=None)
    modules3 = register_module_replacement_dict(model, nn_to_ttnn3, model_config=None)
    type(model).device = property(lambda self: torch.device("cpu"))
    set_device(model, mesh_device)

    all_modules = {**modules1, **modules2, **modules3}
    print(f"Preprocessing {len(all_modules)} TTNN module weights...")
    for k, v in tqdm(all_modules.items()):
        v.preprocess_weights()
        v.move_weights_to_device()

    model.eval()
    torch.set_grad_enabled(False)
    paged_cache = create_paged_kv_cache(model.config, mesh_device, batch_size=1)
    return model, tokenizer, paged_cache


def warmup(model, tokenizer, mesh_device, paged_cache, logits_gen_kwargs=None):
    """Uses ``model.generate`` so the HF generation path is exercised; logits processing kwargs match CLI defaults."""
    logits_gen_kwargs = dict(logits_gen_kwargs or {})
    print("Warming up with zero inputs at seq_len = 256 ...")
    for seq_len in [256, 1024]:
        input_ids = torch.zeros((1, seq_len), dtype=torch.long, device=model.device)
        attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=model.device)
        model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=2,
            use_cache=True,
            past_key_values=paged_cache,
            **logits_gen_kwargs,
        )
        paged_cache.reset()
        print(f"  seq_len={seq_len} done")
    TracedRun.release_all()
    print("Warmup complete.")


def chat_loop(
    model,
    tokenizer,
    paged_cache,
    mesh_device,
    max_new_tokens=256,
    decode_params=None,
):
    decode_params = decode_params or DecodeParams()
    messages = []
    print("\n--- Ling-mini-2.0 Chatbot ---")
    print("Type 'quit' or 'exit' to stop, '/clear' to reset history.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if user_input.lower() == "/clear":
            messages = []
            paged_cache.reset()
            print("History cleared.\n")
            continue
        if user_input.lower() == "/clear_trace":
            TracedRun.release_all()
            print("Traces cleared.\n")
            continue

        messages.append({"role": "user", "content": user_input})

        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = preprocess_generation_inputs(
            inputs,
            model.config,
            paged_cache,
            max_new_tokens,
            model.device,
        )

        # Reset KV cache values in-place (preserves device buffer addresses so
        # decode traces remain valid) and release only prefill traces (different
        # prompt lengths require new prefill captures each turn).
        paged_cache.reset()

        prompt_len = inputs["input_ids"].shape[-1]
        outputs = decode_with_logit_postprocess(
            model,
            inputs["input_ids"],
            inputs["attention_mask"],
            paged_cache,
            max_new_tokens=max_new_tokens,
            decode_params=decode_params,
        )

        response = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
        print(f"\nAssistant: {response}\n")

        messages.append({"role": "assistant", "content": response})


def main():
    parser = argparse.ArgumentParser(description="HF Chatbot with TTNN acceleration")
    parser.add_argument("--model", default="inclusionAI/Ling-mini-2.0", help="HuggingFace model name")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max tokens to generate per turn")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Logits temperature; 0=greedy, >0 enables sampling (top-p/top-k apply)",
    )
    parser.add_argument("--top-p", type=float, default=0.95, dest="top_p")
    parser.add_argument("--top-k", type=int, default=50, dest="top_k")
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        dest="repetition_penalty",
        help=">1.0 discourages repeating tokens (1.0 disables)",
    )
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=0,
        dest="no_repeat_ngram_size",
        help="If >0, blocks repeating n-grams of this size",
    )
    args = parser.parse_args()
    decode_params = DecodeParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )
    logits_gen_kwargs = logits_postprocess_generation_kwargs(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )
    DispatchManager.DisableTiming()  # Disable timing during interactive chat
    mesh_device = setup_mesh_device()
    try:
        model, tokenizer, paged_cache = load_model(mesh_device, args.model)
        warmup(model, tokenizer, mesh_device, paged_cache, logits_gen_kwargs)
        chat_loop(model, tokenizer, paged_cache, mesh_device, args.max_new_tokens, decode_params)
    finally:
        cleanup(mesh_device)


if __name__ == "__main__":
    main()
