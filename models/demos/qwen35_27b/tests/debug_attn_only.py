"""Debug: Run ONLY the 16 full attention layers (skip GDN) to test attention path."""
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn

os.environ.setdefault("HF_MODEL", os.path.expanduser("~/models/Qwen3.5-27B-FP8"))

from transformers import AutoTokenizer

from models.demos.qwen35_27b.tt.model import Transformer
from models.demos.qwen35_27b.tt.model_config import Qwen35ModelArgs, load_qwen35_state_dict, load_weights_to_mesh


@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "N150x4": (1, 4), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_attn_only(mesh_device, reset_seeds, ensure_gc):
    if mesh_device.get_num_devices() < 4:
        pytest.skip("Need TP>=4")

    model_path = os.environ["HF_MODEL"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    state_dict = load_qwen35_state_dict(model_path)

    args = Qwen35ModelArgs(mesh_device, max_batch_size=32, max_seq_len=256)

    # Extract only the full_attention layer indices from the full model
    full_attn_indices = [i for i, lt in enumerate(args.layer_types) if lt == "full_attention"]
    logger.info(f"Full attention layer indices: {full_attn_indices}")
    n_attn_layers = len(full_attn_indices)  # 16

    # Override to use only these layers
    args.n_layers = n_attn_layers
    args.layer_types = ["full_attention"] * n_attn_layers

    # Remap state dict: rename layers to be contiguous 0..15
    remapped_sd = {}
    for key, val in state_dict.items():
        if not key.startswith("layers."):
            remapped_sd[key] = val
            continue
        parts = key.split(".", 2)
        orig_layer = int(parts[1])
        if orig_layer in full_attn_indices:
            new_layer = full_attn_indices.index(orig_layer)
            remapped_sd[f"layers.{new_layer}.{parts[2]}"] = val

    cache_dir = os.path.expanduser("~/models/Qwen3.5-27B-mesh-tp4-attn-only")
    all_weights, embed_table, final_norm_w, lm_head_w = load_weights_to_mesh(remapped_sd, mesh_device, cache_dir, args)

    weight_cache_path = Path(cache_dir) / "framework"
    os.makedirs(weight_cache_path, exist_ok=True)

    model = Transformer(
        args=args,
        dtype=ttnn.bfloat8_b,
        mesh_device=mesh_device,
        state_dict=remapped_sd,
        weight_cache_path=weight_cache_path,
    )
    del remapped_sd

    prompt = "The capital of France is"
    tokens = tokenizer.encode(prompt)
    logger.info(f"Prompt: '{prompt}' -> tokens: {tokens}")

    batch_size = 32
    # Prefill
    for i, tok in enumerate(tokens[:-1]):
        tok_batch = torch.full((batch_size,), tok, dtype=torch.long)
        cur_pos = torch.full((batch_size,), i, dtype=torch.long)
        tt_tokens, tt_pos, tt_rot, _ = model.prepare_inputs_decode(tok_batch, cur_pos)
        model.ttnn_decode_forward(tt_tokens, tt_pos, rot_mat_idxs=tt_rot)

    # Decode 5 tokens
    current_token = tokens[-1]
    for step in range(5):
        pos = len(tokens) - 1 + step
        tok_batch = torch.full((batch_size,), current_token, dtype=torch.long)
        cur_pos = torch.full((batch_size,), pos, dtype=torch.long)
        tt_tokens, tt_pos, tt_rot, _ = model.prepare_inputs_decode(tok_batch, cur_pos)
        tt_logits, _ = model.ttnn_decode_forward(tt_tokens, tt_pos, rot_mat_idxs=tt_rot)

        logits_torch = ttnn.to_torch(tt_logits, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
        logits_1d = logits_torch[0, 0, 0, : args.vocab_size]

        probs = torch.softmax(logits_1d.float(), dim=-1)
        topk = torch.topk(probs, k=10)
        logger.info(f"\nStep {step}: pos={pos}, input='{tokenizer.decode([current_token])}'")
        logger.info(f"  Logit range: [{logits_1d.min():.2f}, {logits_1d.max():.2f}]")
        top5 = ", ".join(f"'{tokenizer.decode([topk.indices[j].item()])}' ({topk.values[j]:.3f})" for j in range(5))
        logger.info(f"  Top-5: [{top5}]")

        current_token = logits_1d.argmax().item()
