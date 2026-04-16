# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Interactive chatbot for HF models with TTNN backend."""

import argparse
import os
import sys
from pathlib import Path

script_dir = str(Path(__file__).resolve().parent)
project_root = str(Path(__file__).resolve().parents[3])

if sys.path and sys.path[0] == script_dir:
    sys.path.pop(0)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import ttnn
from models.experimental.tt_symbiote.core.run_config import DispatchManager, TracedRun
from models.experimental.tt_symbiote.models.ling import load_model, DEFAULT_MODEL_NAME

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


def warmup(model, tokenizer, mesh_device, paged_cache):
    import torch

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
        )
        paged_cache.reset()
        print(f"  seq_len={seq_len} done")
    TracedRun.release_all()
    print("Warmup complete.")


def chat_loop(model, tokenizer, paged_cache, mesh_device, max_new_tokens=256):
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
        ).to(model.device)
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]

        paged_cache.reset()

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
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="HuggingFace model name")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max tokens to generate per turn")
    args = parser.parse_args()
    DispatchManager.DisableTiming()
    mesh_device = setup_mesh_device()
    try:
        model, tokenizer, paged_cache = load_model(mesh_device, args.model)
        warmup(model, tokenizer, mesh_device, paged_cache)
        chat_loop(model, tokenizer, paged_cache, mesh_device, args.max_new_tokens)
    finally:
        cleanup(mesh_device)


if __name__ == "__main__":
    main()
