# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""S1 gate: PaddleOCR-VL's ERNIE decoder on one Blackhole die.

The decoder is Llama-shaped GQA, so the bet is that ``models/tt_transformers``
serves it unmodified and the only bring-up work is the vision half. This scores
the stock ``Transformer``, built from the checkpoint's own weights and prefilling
a text prompt, against the HuggingFace reference.

Run::

    MESH_DEVICE=P150 TT_VISIBLE_DEVICES=3 \
    TT_MESH_GRAPH_DESC_PATH=$PWD/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
    HF_MODEL=PaddlePaddle/PaddleOCR-VL-1.6 \
    pytest models/demos/blackhole/paddleocr_vl/tests/test_text_decoder_pcc.py -s
"""

from __future__ import annotations

import torch

import ttnn
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.model import Transformer
from models.tt_transformers.tt.model_config import ModelArgs
from tests.ttnn.utils_for_testing import assert_with_pcc

PROMPT = "The capital of France is Paris. The capital of Japan is"
TOP_K = 5
PCC_TARGET = 0.98  # margin below the observed 0.989 (see tt/model.py docstring)


def test_text_decoder_logits_match_hf(device):
    args = ModelArgs(device, instruct=False, max_batch_size=1, max_seq_len=2048)
    tokenizer = args.tokenizer
    tokens = tokenizer.encode(PROMPT, add_special_tokens=True)
    state_dict = args.load_state_dict()

    # Drive `language_model` + `lm_head` directly rather than the
    # conditional-generation wrapper. That pair is exactly the module
    # tt_transformers reproduces, so a mismatch here cannot be blamed on
    # multimodal plumbing.
    from transformers import AutoModelForImageTextToText

    hf_full = AutoModelForImageTextToText.from_pretrained(args.CKPT_DIR, dtype=torch.bfloat16)
    hf_full.eval()
    with torch.no_grad():
        hidden = hf_full.model.language_model(input_ids=torch.tensor([tokens])).last_hidden_state
        ref_logits = hf_full.lm_head(hidden)[0, -1].float()
    del hf_full

    ref_top = torch.topk(ref_logits, TOP_K)

    tt_model = Transformer(
        args=args,
        mesh_device=device,
        dtype=ttnn.bfloat8_b,
        state_dict=state_dict,
        weight_cache_path=args.weight_cache_path(ttnn.bfloat8_b),
    )
    generator = Generator([tt_model], [args], device)
    tt_logits = generator.prefill_forward_text(
        torch.tensor([tokens], dtype=torch.int32),
        page_table=None,
        kv_cache=None,
        prompt_lens=[len(tokens)],
    )
    tt_last = tt_logits[0, -1].float() if tt_logits.dim() == 3 else tt_logits[-1].float()
    tt_top = torch.topk(tt_last, TOP_K)

    top1_match = bool(ref_top.indices[0] == tt_top.indices[0])
    overlap = len(set(ref_top.indices.tolist()) & set(tt_top.indices.tolist()))
    print(f"top-1 match={top1_match} top-{TOP_K} overlap={overlap}/{TOP_K}")

    assert_with_pcc(ref_logits, tt_last, PCC_TARGET)
    assert top1_match is True
    # Allow one miss out of 5 for quantization margin; 5/5 is the observed result,
    # not a guaranteed one (bf8 weights).
    assert overlap >= 4, f"top-{TOP_K} overlap {overlap}/{TOP_K} below floor"
