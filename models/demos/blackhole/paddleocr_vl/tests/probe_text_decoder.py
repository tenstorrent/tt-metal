# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""S1 probe: run PaddleOCR-VL's ERNIE decoder on one Blackhole die.

The decoder is Llama-shaped GQA, so the bet is that ``models/tt_transformers``
serves it unmodified and the only bring-up work is the vision half. This probe
settles that by building the stock ``Transformer`` from the checkpoint's own
weights and prefilling a text prompt, then scoring the logits against the
HuggingFace reference.

It is a script rather than a pytest case on purpose: at this stage the useful
output is *where it breaks*, and a traceback from a plain run is easier to read
than one filtered through fixtures. It graduates to ``test_text_decoder.py``
once it passes.

Run::

    MESH_DEVICE=P150 TT_VISIBLE_DEVICES=3 \
    TT_MESH_GRAPH_DESC_PATH=$PWD/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
    HF_MODEL=PaddlePaddle/PaddleOCR-VL-1.6 \
    python models/demos/blackhole/paddleocr_vl/tests/probe_text_decoder.py
"""

from __future__ import annotations

import torch
from loguru import logger

import ttnn
from models.tt_transformers.tt.model_config import ModelArgs

PROMPT = "The capital of France is Paris. The capital of Japan is"
TOP_K = 5


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    return 1.0 if denom == 0 else (a @ b).item() / denom


def main() -> int:
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        args = ModelArgs(mesh, instruct=False, max_batch_size=1, max_seq_len=2048)
        logger.info(f"model={args.model_name} device={args.device_name} multimodal={args.is_multimodal}")

        tokenizer = args.tokenizer
        tokens = tokenizer.encode(PROMPT, add_special_tokens=True)
        logger.info(f"prompt tokens: {len(tokens)} -> {tokens[:12]}...")

        state_dict = args.load_state_dict()
        text_keys = {k: v for k, v in state_dict.items() if not k.startswith(("visual", "projector"))}
        logger.info(f"state dict: {len(state_dict)} keys total, {len(text_keys)} text keys")

        # ---- HuggingFace reference (text only, no image) --------------------
        # Drive `language_model` + `lm_head` directly rather than the
        # conditional-generation wrapper. That pair is exactly the module
        # tt_transformers reproduces, so a mismatch here cannot be blamed on
        # multimodal plumbing. (It also sidesteps ModelArgs.reference_transformer,
        # whose construction path trips over rope_deltas on text-only input while
        # a plain from_pretrained does not.)
        logger.info("running HF reference on cpu")
        from transformers import AutoModelForImageTextToText

        hf_full = AutoModelForImageTextToText.from_pretrained(args.CKPT_DIR, dtype=torch.bfloat16)
        hf_full.eval()
        with torch.no_grad():
            hidden = hf_full.model.language_model(input_ids=torch.tensor([tokens])).last_hidden_state
            ref_logits = hf_full.lm_head(hidden)[0, -1].float()
        del hf_full

        ref_top = torch.topk(ref_logits, TOP_K)
        logger.info(f"HF top-{TOP_K}: {[tokenizer.decode([i]) for i in ref_top.indices.tolist()]}")

        # ---- TT model -------------------------------------------------------
        from models.tt_transformers.tt.model import Transformer

        logger.info("building TT Transformer (first run converts + caches weights)")
        tt_model = Transformer(
            args=args,
            mesh_device=mesh,
            dtype=ttnn.bfloat8_b,
            state_dict=state_dict,
            weight_cache_path=args.weight_cache_path(ttnn.bfloat8_b),
        )
        logger.info("TT model built")

        from models.tt_transformers.tt.generator import Generator

        generator = Generator([tt_model], [args], mesh)
        logger.info("running TT prefill")
        tt_logits = generator.prefill_forward_text(
            torch.tensor([tokens], dtype=torch.int32),
            page_table=None,
            kv_cache=None,
            prompt_lens=[len(tokens)],
        )
        tt_last = tt_logits[0, -1].float() if tt_logits.dim() == 3 else tt_logits[-1].float()

        tt_top = torch.topk(tt_last, TOP_K)
        logger.info(f"TT top-{TOP_K}: {[tokenizer.decode([i]) for i in tt_top.indices.tolist()]}")

        p = pcc(ref_logits, tt_last)
        top1 = int(ref_top.indices[0] == tt_top.indices[0])
        overlap = len(set(ref_top.indices.tolist()) & set(tt_top.indices.tolist()))
        print("\n================ S1 RESULT ================")
        print(f"logits PCC      : {p:.6f}")
        print(f"top-1 match     : {bool(top1)}")
        print(f"top-{TOP_K} overlap  : {overlap}/{TOP_K}")
        print(f"HF  argmax      : {ref_top.indices[0].item()} {tokenizer.decode([ref_top.indices[0]])!r}")
        print(f"TT  argmax      : {tt_top.indices[0].item()} {tokenizer.decode([tt_top.indices[0]])!r}")
        print("===========================================")
        return 0
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    raise SystemExit(main())
