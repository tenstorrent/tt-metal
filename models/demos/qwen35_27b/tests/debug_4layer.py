"""Debug: Run 4-layer subset and inspect logit quality."""
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
def test_debug_4layer(mesh_device, reset_seeds, ensure_gc):
    if mesh_device.get_num_devices() < 4:
        pytest.skip("Need TP>=4")

    model_path = os.environ["HF_MODEL"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    state_dict = load_qwen35_state_dict(model_path)

    args = Qwen35ModelArgs(mesh_device, max_batch_size=32)
    orig_n_layers = args.n_layers
    args.n_layers = 4
    args.layer_types = args.layer_types[:4]
    logger.info(f"Layer types (4 of {orig_n_layers}): {args.layer_types}")

    cache_dir = os.path.expanduser("~/models/Qwen3.5-27B-mesh-tp4-4layer")
    all_weights, embed_table, final_norm_w, lm_head_w = load_weights_to_mesh(state_dict, mesh_device, cache_dir, args)

    weight_cache_path = Path(cache_dir) / "framework"
    os.makedirs(weight_cache_path, exist_ok=True)

    model = Transformer(
        args=args,
        dtype=ttnn.bfloat8_b,
        mesh_device=mesh_device,
        state_dict=state_dict,
        weight_cache_path=weight_cache_path,
    )
    del state_dict

    prompt = "The capital of France is"
    tokens = tokenizer.encode(prompt)
    logger.info(f"Prompt: '{prompt}' -> tokens: {tokens}")

    # Prefill token by token
    for i, tok in enumerate(tokens):
        tok_batch = torch.full((32,), tok, dtype=torch.long)
        cur_pos = torch.full((32,), i, dtype=torch.long)
        tt_tokens, tt_pos, tt_rot, _ = model.prepare_inputs_decode(tok_batch, cur_pos)
        tt_logits, _ = model.ttnn_decode_forward(tt_tokens, tt_pos, rot_mat_idxs=tt_rot)

    # Decode 5 tokens
    current_token = tokens[-1]
    for step in range(5):
        pos = len(tokens) - 1 + step
        tok_batch = torch.full((32,), current_token, dtype=torch.long)
        cur_pos = torch.full((32,), pos, dtype=torch.long)
        tt_tokens, tt_pos, tt_rot, _ = model.prepare_inputs_decode(tok_batch, cur_pos)
        tt_logits, _ = model.ttnn_decode_forward(tt_tokens, tt_pos, rot_mat_idxs=tt_rot)

        logits_torch = ttnn.to_torch(tt_logits, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
        logits_1d = logits_torch[0, 0, 0, : args.vocab_size]

        probs = torch.softmax(logits_1d.float(), dim=-1)
        topk = torch.topk(probs, k=10)
        logger.info(f"\nStep {step}: pos={pos}, input='{tokenizer.decode([current_token])}'")
        logger.info(f"  Logit range: [{logits_1d.min():.2f}, {logits_1d.max():.2f}], mean={logits_1d.mean():.4f}")
        logger.info(f"  Top-10:")
        for j in range(10):
            tid = topk.indices[j].item()
            p = topk.values[j].item()
            text = tokenizer.decode([tid])
            logger.info(f"    {j+1}. token={tid:6d} prob={p:.4f} '{text}'")

        current_token = logits_1d.argmax().item()
