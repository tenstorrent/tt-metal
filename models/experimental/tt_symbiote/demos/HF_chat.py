# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Interactive chatbot for HF models with TTNN backend."""

from __future__ import annotations

import argparse
import os
import sys
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
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import ttnn
from models.experimental.tt_symbiote.core.run_config import DispatchManager, TracedRun
from models.experimental.tt_symbiote.modules.activation import TTNNSilu
from models.experimental.tt_symbiote.modules.linear import (
    TTNNLinearIColShardedWRowSharded,
    TTNNLinearIColShardedWAllReduced,
)
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict
from models.experimental.tt_symbiote.modules.decoder_layer import TTNNBailingMoEDecoderLayerPadded
from models.experimental.tt_symbiote.modules.normalization import TTNNDistributedRMSNorm
from models.experimental.tt_symbiote.modules.embedding import TTNNBailingPaddedEmbedding, TTNNBailingRotaryEmbedding
from models.experimental.tt_symbiote.models.bailing_moe_v2 import TTNNBailingMoeV2Model
from models.experimental.tt_symbiote.models.ling import (
    DecodeParams,
    create_paged_kv_cache,
    decode_with_logit_postprocess,
    generation_torch_device,
    preprocess_generation_inputs,
    replicated_mesh_tt_to_torch,
    token_ids_list_for_hf_decode,
)

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

    if mesh_device.get_num_devices() > 1 and isinstance(model.lm_head, TTNNLinearIColShardedWRowSharded):
        model.lm_head.__class__ = TTNNLinearIColShardedWAllReduced
        print("lm_head: TTNNLinearIColShardedWAllReduced (full vocab on each device after lm_head).")

    all_modules = {**modules1, **modules2}
    print(f"Preprocessing {len(all_modules)} TTNN module weights...")
    for k, v in tqdm(all_modules.items()):
        v.preprocess_weights()
        v.move_weights_to_device()

    model.eval()
    torch.set_grad_enabled(False)
    paged_cache = create_paged_kv_cache(model.config, mesh_device, batch_size=1)
    return model, tokenizer, paged_cache


def warmup(model, _tokenizer, mesh_device, paged_cache, decode_params=None):
    decode_params = decode_params or DecodeParams()
    print("Warming up with zero inputs at seq_len = 256 ...")
    for seq_len in [256, 1024]:
        prompt_tt = ttnn.zeros(
            (1, seq_len),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        mask_tt = ttnn.ones(
            (1, seq_len),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        input_ids = replicated_mesh_tt_to_torch(prompt_tt, mesh_device).long()
        attention_mask = replicated_mesh_tt_to_torch(mask_tt, mesh_device).long()
        ttnn.deallocate(mask_tt)
        out_tt = decode_with_logit_postprocess(
            model,
            input_ids,
            attention_mask,
            paged_cache,
            max_new_tokens=2,
            decode_params=decode_params,
            mesh_device=mesh_device,
            prompt_ids_tt=prompt_tt,
        )
        ttnn.deallocate(out_tt)
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

        torch_dev = generation_torch_device(model)
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
            torch_dev,
        )

        # Reset KV cache values in-place (preserves device buffer addresses so
        # decode traces remain valid) and release only prefill traces (different
        # prompt lengths require new prefill captures each turn).
        paged_cache.reset()

        prompt_len = inputs["input_ids"].shape[-1]
        outputs_tt = decode_with_logit_postprocess(
            model,
            inputs["input_ids"],
            inputs["attention_mask"],
            paged_cache,
            max_new_tokens=max_new_tokens,
            decode_params=decode_params,
            mesh_device=mesh_device,
        )
        try:
            outputs = replicated_mesh_tt_to_torch(outputs_tt, mesh_device).long().to(torch_dev)
            gen_ids = outputs[0, prompt_len:].tolist()
            response = tokenizer.decode(
                token_ids_list_for_hf_decode(gen_ids, tokenizer),
                skip_special_tokens=True,
            )
        finally:
            ttnn.deallocate(outputs_tt)
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
    DispatchManager.DisableTiming()  # Disable timing during interactive chat
    mesh_device = setup_mesh_device()
    try:
        model, tokenizer, paged_cache = load_model(mesh_device, args.model)
        warmup(model, tokenizer, mesh_device, paged_cache, decode_params)
        chat_loop(model, tokenizer, paged_cache, mesh_device, args.max_new_tokens, decode_params)
    finally:
        cleanup(mesh_device)


if __name__ == "__main__":
    main()
