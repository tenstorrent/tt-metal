# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""S3 probe: read a page end to end on Blackhole and score the text.

Vision tower plus text decoder, wired together through the host splice and
M-RoPE, decoding greedily. This is the first point the port produces OCR output
rather than tensors, and the gate that actually matters: character error rate
against the HuggingFace reference, with ground-truth CER reported alongside so a
port regression can be told apart from a model limitation.

Greedy throughout, so the comparison is deterministic. Everything runs one image
at a time; batching multimodal prefill is not supported by any TT model today.

Run::

    MESH_DEVICE=P150 TT_VISIBLE_DEVICES=3 \
    TT_MESH_GRAPH_DESC_PATH=$PWD/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
    HF_MODEL=PaddlePaddle/PaddleOCR-VL-1.6 \
    python models/demos/blackhole/paddleocr_vl/tests/probe_ocr_e2e.py --limit 3
"""

from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata

import torch
from loguru import logger
from PIL import Image

import ttnn
from models.demos.blackhole.paddleocr_vl.tt.common import multimodal_rope_from_hf, splice_image_embeddings
from models.demos.blackhole.paddleocr_vl.tt.model import Transformer
from models.demos.blackhole.paddleocr_vl.tt.vision.model import DropInVisionTransformer
from models.demos.blackhole.paddleocr_vl.tt.vision.vision_model_config import VisionModelArgs
from models.demos.blackhole.paddleocr_vl.tt.weight_mapping import map_vision_state_dict
from models.demos.qwen3_vl.tt.generator import Generator as VLGenerator
from models.tt_transformers.tt.common import get_padded_prefill_len
from models.tt_transformers.tt.model_config import ModelArgs

DEMO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "demo"))
GOLDEN = os.path.join(DEMO, "golden", "hf_goldens.json")
OCR_PROMPT = "OCR:"


def to_torch_logits(x, mesh, vocab_size: int, row: int = -1) -> torch.Tensor:
    """One position's logit vector, from either a ttnn or torch return value.

    ``ttnn_prefill_forward`` is asked for ``get_last_token=(idx // 32) * 32``, so
    prefill hands back the 32-row block that *contains* the last prompt token
    rather than that token alone. Taking the final row of the block reads a
    position past the end of the prompt, which produces a plausible-looking but
    wrong first token (it prefixed a stray "j" before this was pinned down).
    Callers pass ``row = last_token_idx % 32`` to select correctly; decode
    returns a single row and keeps the default.
    """
    if not isinstance(x, torch.Tensor):
        x = ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1))
    return x.float().reshape(-1, vocab_size)[row]


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s)).strip()


def cer(ref: str, hyp: str) -> float:
    ref, hyp = normalize(ref), normalize(hyp)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i]
        for j, hc in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / len(ref)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--dump-dir", default=None, help="write full TT/HF texts here for diffing")
    a = ap.parse_args()

    with open(GOLDEN) as f:
        manifest = json.load(f)
    samples = manifest["samples"]
    if a.only:
        samples = [s for s in samples if s["name"] == a.only]
    if a.limit:
        samples = samples[: a.limit]
    logger.info(f"scoring {len(samples)} sample(s) from {GOLDEN}")

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        # ---- args: one for the tower's shapes, one for the decoder's ---------
        vision_args = VisionModelArgs(mesh, instruct=True, max_batch_size=1, max_seq_len=8192)
        text_args = ModelArgs(mesh, instruct=True, max_batch_size=1, max_seq_len=8192)

        full_sd = vision_args.load_state_dict()
        device_sd, host_sd = map_vision_state_dict(full_sd, vision_head_dim=vision_args.head_dim, strict=True)

        logger.info("building vision tower")
        tower = DropInVisionTransformer(
            model_args=vision_args,
            device_state_dict=device_sd,
            host_state_dict=host_sd,
            dtype=ttnn.bfloat8_b,
        )

        logger.info("building text decoder")
        text_model = Transformer(
            args=text_args,
            dtype=ttnn.bfloat8_b,
            mesh_device=mesh,
            state_dict=device_sd,
            weight_cache_path=text_args.weight_cache_path(ttnn.bfloat8_b),
        )
        # qwen3_vl's Generator, the same one the vLLM path uses: it owns the
        # last_token_idx % 32 row selection and update_rope_deltas, so probe and
        # server exercise one code path rather than two.
        generator = VLGenerator(text_model, text_args, mesh, tokenizer=text_args.tokenizer)

        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor = AutoProcessor.from_pretrained(text_args.CKPT_DIR)
        # The reference model is kept for three host-side jobs only: the embedding
        # table, get_rope_index, and the text rotary module. Its decoder weights
        # are never used for inference here.
        ref = AutoModelForImageTextToText.from_pretrained(text_args.CKPT_DIR, dtype=torch.bfloat16)
        ref.eval()
        embed_tokens = ref.model.language_model.embed_tokens
        image_token_id = ref.config.image_token_id
        pad_token_id = getattr(ref.config, "pad_token_id", 0) or 0
        tokenizer = text_args.tokenizer

        rows = []
        for s in samples:
            img = Image.open(os.path.join(DEMO, s["image"])).convert("RGB")
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": OCR_PROMPT}]}]
            text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            inputs = processor(text=[text], images=[img], return_tensors="pt")

            input_ids = inputs["input_ids"][0]
            grid = inputs["image_grid_thw"]

            # ---- vision ------------------------------------------------------
            image_embeds = tower(inputs["pixel_values"].to(torch.bfloat16), grid)

            # ---- splice ------------------------------------------------------
            with torch.no_grad():
                text_embeds = embed_tokens(input_ids.unsqueeze(0))[0]
            merged = splice_image_embeddings(input_ids, text_embeds, image_embeds, image_token_id)

            # ---- prefill padding ---------------------------------------------
            # Prefill kernels require a sequence length that is a multiple of 128,
            # so pad the embedding stream and keep last_token_idx at the real end.
            prompt_len = merged.shape[0]
            padded_len = get_padded_prefill_len(prompt_len)

            # ---- rope --------------------------------------------------------
            # Tables must span the padded prefill and every position decode will
            # reach, or prepare_inputs_prefill/decode indexes off the end.
            cos, sin, rope_deltas = multimodal_rope_from_hf(
                input_ids,
                grid,
                ref,
                text_args,
                pad_token_id=pad_token_id,
                min_positions=max(padded_len, prompt_len + a.max_new_tokens + 1),
            )

            if padded_len > prompt_len:
                merged = torch.cat(
                    [merged, torch.zeros(padded_len - prompt_len, merged.shape[-1], dtype=merged.dtype)], dim=0
                )
            embeds_tt = ttnn.from_torch(
                merged.unsqueeze(0).to(torch.bfloat16),
                device=mesh,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            logits = generator.prefill_forward_single_user_text(
                ttnn.unsqueeze(embeds_tt, 0) if len(embeds_tt.shape) == 2 else embeds_tt,
                page_table=None,
                user_id=0,
                last_token_idx=prompt_len - 1,
                rot_mats=(cos, sin),
                kv_cache=None,
            )
            generator.update_rope_deltas([int(rope_deltas.reshape(-1)[0].item())])

            token = int(to_torch_logits(logits, mesh, text_args.vocab_size).argmax().item())
            generated = [token]

            # ---- decode ------------------------------------------------------
            eos = set(tokenizer.all_special_ids or [])
            for step in range(a.max_new_tokens - 1):
                if token in eos:
                    break
                out = generator.decode_forward(
                    torch.tensor([[token]], dtype=torch.int32),
                    start_pos=torch.tensor([prompt_len + step], dtype=torch.int32),
                    page_table=None,
                    kv_cache=None,
                    enable_trace=False,
                )
                lg = out[0] if isinstance(out, tuple) else out
                token = int(to_torch_logits(lg, mesh, text_args.vocab_size).argmax().item())
                generated.append(token)

            tt_text = tokenizer.decode([t for t in generated if t not in eos], skip_special_tokens=True)

            c_hf = cer(s["hf_output"], tt_text)
            c_gt = cer(s["ground_truth"], tt_text)
            c_hf_gt = cer(s["ground_truth"], s["hf_output"])
            rows.append((s["name"], s["bucket"], len(generated), c_hf, c_gt, c_hf_gt))
            logger.info(f"[{s['name']}] gen={len(generated)} CERvsHF={c_hf*100:.2f}% CERvsGT={c_gt*100:.2f}%")
            logger.info(f"[{s['name']}] TT : {tt_text[:160]!r}")
            logger.info(f"[{s['name']}] HF : {s['hf_output'][:160]!r}")
            if a.dump_dir:
                os.makedirs(a.dump_dir, exist_ok=True)
                for tag, body in (("tt", tt_text), ("hf", s["hf_output"]), ("gt", s["ground_truth"])):
                    with open(os.path.join(a.dump_dir, f"{s['name']}.{tag}.txt"), "w") as fh:
                        fh.write(body)

        print("\n================== S3 RESULT ==================")
        print(f"{'sample':20s} {'bkt':>5s} {'gen':>4s} {'CER vs HF':>10s} {'CER vs GT':>10s} {'HF vs GT':>9s}")
        for name, bkt, n, c_hf, c_gt, c_hf_gt in rows:
            print(f"{name:20s} {bkt:5d} {n:4d} {c_hf*100:9.2f}% {c_gt*100:9.2f}% {c_hf_gt*100:8.2f}%")
        if rows:
            m_hf = sum(r[3] for r in rows) / len(rows)
            print(f"\nmean CER vs HF reference : {m_hf*100:.2f}%   (gate: <= 1.00%)")
        print("===============================================")
        return 0
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    raise SystemExit(main())
