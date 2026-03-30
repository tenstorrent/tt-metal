# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Interactive chatbot for HF models with TTNN backend."""

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
from models.experimental.tt_symbiote.modules.linear import TTNNLinearIColShardedWRowSharded
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict
from models.experimental.tt_symbiote.modules.attention import (
    PagedAttentionConfig,
    TTNNPagedAttentionKVCache,
)
from models.experimental.tt_symbiote.modules.decoder_layer import TTNNBailingMoEDecoderLayer
from models.experimental.tt_symbiote.modules.normalization import TTNNDistributedRMSNorm


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


def load_model(mesh_device, model_name="inclusionAI/Ling-mini-2.0"):
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    nn_to_ttnn = {
        model.model.layers[0].__class__: TTNNBailingMoEDecoderLayer,
        model.model.norm.__class__: TTNNDistributedRMSNorm,
    }
    nn_to_ttnn2 = {
        nn.Linear: TTNNLinearIColShardedWRowSharded,
        nn.SiLU: TTNNSilu,
    }

    modules1 = register_module_replacement_dict(model, nn_to_ttnn, model_config=None)
    modules2 = register_module_replacement_dict(model, nn_to_ttnn2, model_config=None)
    set_device(model, mesh_device)

    all_modules = {**modules1, **modules2}
    print(f"Preprocessing {len(all_modules)} TTNN module weights...")
    for k, v in tqdm(all_modules.items()):
        v.preprocess_weights()
        v.move_weights_to_device()

    model.eval()
    torch.set_grad_enabled(False)
    return model, tokenizer


def warmup(model, tokenizer, mesh_device):
    print("Warming up...")
    messages = [{"role": "user", "content": "Hello there. What is your name. Tell me about yourself."}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    paged_cache = create_paged_kv_cache(model.config, mesh_device, batch_size=1)
    model.generate(**inputs, max_new_tokens=2, use_cache=True, past_key_values=paged_cache)
    print("Warmup complete.")


def chat_loop(model, tokenizer, mesh_device, max_new_tokens=256):
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
            print("History cleared.\n")
            continue

        messages.append({"role": "user", "content": user_input})

        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]

        # Each turn needs a fresh KV cache and fresh traces.  Traces
        # reference the cache's device buffers, so they become invalid
        # when the cache is recreated.  The prompt length also changes
        # each turn, which requires new prefill trace captures.
        TracedRun.release_all()
        paged_cache = create_paged_kv_cache(model.config, mesh_device, batch_size=1)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            past_key_values=paged_cache,
        )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        print(f"\nAssistant: {response}\n")

        messages.append({"role": "assistant", "content": response})


def main():
    parser = argparse.ArgumentParser(description="HF Chatbot with TTNN acceleration")
    parser.add_argument("--model", default="inclusionAI/Ling-mini-2.0", help="HuggingFace model name")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max tokens to generate per turn")
    args = parser.parse_args()
    DispatchManager.DisableTiming()  # Disable timing during interactive chat
    mesh_device = setup_mesh_device()
    try:
        model, tokenizer = load_model(mesh_device, args.model)
        warmup(model, tokenizer, mesh_device)
        chat_loop(model, tokenizer, mesh_device, args.max_new_tokens)
    finally:
        cleanup(mesh_device)


if __name__ == "__main__":
    main()
