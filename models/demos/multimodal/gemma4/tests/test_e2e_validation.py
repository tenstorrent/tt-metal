# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""End-to-end validation for Gemma4 31B IT on Tenstorrent Blackhole.
Tests prefill accuracy against CPU reference using raw and chat-template prompts.
Pass criteria: top-1 token match on all test prompts.
"""

import gc
import os

import pytest
import torch
from loguru import logger
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

import ttnn
from models.demos.multimodal.gemma4.tt.gemma4_model import Gemma4Transformer
from models.demos.multimodal.gemma4.tt.model_config import ModelArgs


def compute_pcc(ref, test):
    """Compute Pearson Correlation Coefficient between two tensors."""
    ref = ref.float().flatten()
    test = test.float().flatten()
    mask = torch.isfinite(ref) & torch.isfinite(test)
    ref = ref[mask]
    test = test[mask]
    if len(ref) == 0:
        return 0.0
    ref_std = ref.std()
    test_std = test.std()
    if ref_std == 0 or test_std == 0:
        return 1.0 if torch.allclose(ref, test) else 0.0
    vr = ref - ref.mean()
    vt = test - test.mean()
    return (vr * vt).mean().item() / (ref_std * test_std).item()


RAW_PROMPTS = [
    "The capital of France is",
    "Water boils at",
    "The speed of light is approximately",
]

CHAT_TEST_CASES = [
    {"messages": [{"role": "user", "content": "What is the capital of France? Answer in one word."}]},
    {"messages": [{"role": "user", "content": "What is 2 + 2? Answer with just the number."}]},
    {"messages": [{"role": "user", "content": "What language is Python named after? One word."}]},
]


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4), "P150x4": (1, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_e2e_validation(mesh_device, reset_seeds, ensure_gc):
    model_name = "google/gemma-4-31B-it"
    tok = AutoTokenizer.from_pretrained(model_name)

    # --- CPU reference ---
    logger.info("Loading CPU reference model...")
    cpu_model = Gemma4ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    cpu_model.eval()

    cpu_raw_results = {}
    for p in RAW_PROMPTS:
        tokens = tok.encode(p, return_tensors="pt")
        logits = cpu_model(input_ids=tokens).logits[0, -1, :]
        cpu_raw_results[p] = (logits, torch.argmax(logits).item())

    cpu_chat_results = []
    for tc in CHAT_TEST_CASES:
        inputs = tok.apply_chat_template(
            tc["messages"], return_tensors="pt", add_generation_prompt=True, return_dict=True
        )
        logits = cpu_model(input_ids=inputs["input_ids"]).logits[0, -1, :]
        cpu_chat_results.append((logits, torch.argmax(logits).item(), inputs["input_ids"]))

    del cpu_model
    gc.collect()

    # --- TT model ---
    args = ModelArgs(mesh_device=mesh_device, instruct=True, max_batch_size=1, max_seq_len=2048)
    sd = args.load_state_dict()
    model = Gemma4Transformer(
        args=args,
        dtype=ttnn.bfloat16,
        mesh_device=mesh_device,
        state_dict=sd,
        weight_cache_path=args.weight_cache_path(ttnn.bfloat16),
    )

    def run_prefill(input_ids):
        seq_len = input_ids.shape[1]
        padded_len = max(256, ((seq_len + 127) // 128) * 128)
        padded = torch.nn.functional.pad(input_ids, (0, padded_len - seq_len), value=0)
        last_idx = seq_len - 1
        tt_in, rot_g, rot_l, _, _ = model.prepare_inputs_prefill(padded, start_pos=0, last_token_idx=last_idx)
        tt_out = model.ttnn_prefill_forward(
            tt_in, rot_mats_global=rot_g, rot_mats_local=rot_l, get_last_token=(last_idx // 32) * 32
        )
        return model.process_output_prefill(ttnn.from_device(tt_out), last_idx % 32)

    # --- Test raw prompts ---
    logger.info("=== Raw Prompt Accuracy ===")
    raw_matches = 0
    for p in RAW_PROMPTS:
        tokens = tok.encode(p, return_tensors="pt")
        tt_logits = run_prefill(tokens)
        cpu_logits, cpu_tok = cpu_raw_results[p]
        tt_tok = torch.argmax(tt_logits[: len(cpu_logits)]).item()
        min_v = min(len(cpu_logits), len(tt_logits))
        pcc = compute_pcc(cpu_logits[:min_v], tt_logits[:min_v])
        match = cpu_tok == tt_tok
        if match:
            raw_matches += 1
        status = "PASS" if match else "FAIL"
        logger.info(f"  '{p}' -> CPU:'{tok.decode([cpu_tok])}' TT:'{tok.decode([tt_tok])}' PCC={pcc:.4f} {status}")

    # --- Test chat prompts ---
    logger.info("=== Chat Template Accuracy ===")
    chat_matches = 0
    for i, tc in enumerate(CHAT_TEST_CASES):
        cpu_logits, cpu_tok, input_ids = cpu_chat_results[i]
        tt_logits = run_prefill(input_ids)
        tt_tok = torch.argmax(tt_logits[: len(cpu_logits)]).item()
        match = cpu_tok == tt_tok
        if match:
            chat_matches += 1
        status = "PASS" if match else "FAIL"
        logger.info(
            f"  Q: '{tc['messages'][0]['content'][:50]}' -> "
            f"CPU:'{tok.decode([cpu_tok])}' TT:'{tok.decode([tt_tok])}' {status}"
        )

    # --- Summary ---
    total = len(RAW_PROMPTS) + len(CHAT_TEST_CASES)
    total_matches = raw_matches + chat_matches
    logger.info(f"Raw prompts:  {raw_matches}/{len(RAW_PROMPTS)} top-1 match")
    logger.info(f"Chat prompts: {chat_matches}/{len(CHAT_TEST_CASES)} top-1 match")
    logger.info(f"Overall:      {total_matches}/{total} = {total_matches / total * 100:.0f}%")

    assert total_matches == total, f"Top-1 accuracy {total_matches}/{total} below 100%"
