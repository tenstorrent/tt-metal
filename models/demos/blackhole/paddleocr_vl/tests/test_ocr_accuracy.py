# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""S3 gate: read a page end to end on Blackhole and score the text.

Vision tower plus text decoder, wired together through the host splice and
M-RoPE, decoding greedily. This is the gate that actually matters: character
error rate against the HuggingFace reference, over the whole recorded corpus.

Greedy throughout, so the comparison is deterministic. Everything runs one image
at a time; batching multimodal prefill is not supported by any TT model today.

Uses the same generator (``models.demos.qwen3_vl.tt.generator``) the vLLM serving
path uses, not the base ``tt_transformers`` one, so this test and the served
endpoint exercise one code path rather than two.

Run::

    MESH_DEVICE=P150 TT_VISIBLE_DEVICES=3 \
    TT_MESH_GRAPH_DESC_PATH=$PWD/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
    HF_MODEL=PaddlePaddle/PaddleOCR-VL-1.6 \
    pytest models/demos/blackhole/paddleocr_vl/tests/test_ocr_accuracy.py -s
"""

from __future__ import annotations

import os
import re
import unicodedata

import torch
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
OCR_PROMPT = "OCR:"
MAX_NEW_TOKENS = 256
MEAN_CER_GATE = 0.01  # "gate: <= 1.00%" -- see tests/probe_ocr_e2e.py history


def _to_torch_logits(x, mesh, vocab_size: int, row: int = -1) -> torch.Tensor:
    """One position's logit vector, from either a ttnn or torch return value.

    ``ttnn_prefill_forward`` is asked for ``get_last_token=(idx // 32) * 32``, so
    prefill hands back the 32-row block that *contains* the last prompt token
    rather than that token alone. Callers pass ``row = last_token_idx % 32`` to
    select correctly; decode returns a single row and keeps the default.
    """
    if not isinstance(x, torch.Tensor):
        x = ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1))
    return x.float().reshape(-1, vocab_size)[row]


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s)).strip()


def _cer(ref: str, hyp: str) -> float:
    ref, hyp = _normalize(ref), _normalize(hyp)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i]
        for j, hc in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / len(ref)


def _read_page(sample: dict, tower, generator, embed_tokens, image_token_id, ref, text_args, device) -> str:
    img = Image.open(os.path.join(DEMO, sample["image"])).convert("RGB")
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(text_args.CKPT_DIR)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": OCR_PROMPT}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[text], images=[img], return_tensors="pt")

    input_ids = inputs["input_ids"][0]
    grid = inputs["image_grid_thw"]

    image_embeds = tower(inputs["pixel_values"].to(torch.bfloat16), grid)

    with torch.no_grad():
        text_embeds = embed_tokens(input_ids.unsqueeze(0))[0]
    merged = splice_image_embeddings(input_ids, text_embeds, image_embeds, image_token_id)

    prompt_len = merged.shape[0]
    padded_len = get_padded_prefill_len(prompt_len)

    pad_token_id = getattr(ref.config, "pad_token_id", 0) or 0
    cos, sin, rope_deltas = multimodal_rope_from_hf(
        input_ids,
        grid,
        ref,
        text_args,
        pad_token_id=pad_token_id,
        min_positions=max(padded_len, prompt_len + MAX_NEW_TOKENS + 1),
    )

    if padded_len > prompt_len:
        merged = torch.cat([merged, torch.zeros(padded_len - prompt_len, merged.shape[-1], dtype=merged.dtype)], dim=0)
    embeds_tt = ttnn.from_torch(
        merged.unsqueeze(0).to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
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

    tokenizer = text_args.tokenizer
    token = int(_to_torch_logits(logits, device, text_args.vocab_size).argmax().item())
    generated = [token]

    eos = set(tokenizer.all_special_ids or [])
    for step in range(MAX_NEW_TOKENS - 1):
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
        token = int(_to_torch_logits(lg, device, text_args.vocab_size).argmax().item())
        generated.append(token)

    return tokenizer.decode([t for t in generated if t not in eos], skip_special_tokens=True)


def test_ocr_mean_cer_vs_hf_reference(device, hf_goldens):
    samples = hf_goldens["samples"]
    assert samples, "hf_goldens.json has no samples"

    vision_args = VisionModelArgs(device, instruct=True, max_batch_size=1, max_seq_len=8192)
    text_args = ModelArgs(device, instruct=True, max_batch_size=1, max_seq_len=8192)

    full_sd = vision_args.load_state_dict()
    device_sd, host_sd = map_vision_state_dict(full_sd, vision_head_dim=vision_args.head_dim, strict=True)

    tower = DropInVisionTransformer(
        model_args=vision_args,
        device_state_dict=device_sd,
        host_state_dict=host_sd,
        dtype=ttnn.bfloat8_b,
    )
    text_model = Transformer(
        args=text_args,
        dtype=ttnn.bfloat8_b,
        mesh_device=device,
        state_dict=device_sd,
        weight_cache_path=text_args.weight_cache_path(ttnn.bfloat8_b),
    )
    generator = VLGenerator(text_model, text_args, device, tokenizer=text_args.tokenizer)

    from transformers import AutoModelForImageTextToText

    # Kept for three host-side jobs only: the embedding table, get_rope_index, and
    # the text rotary module. Its decoder weights are never used for inference.
    ref = AutoModelForImageTextToText.from_pretrained(text_args.CKPT_DIR, dtype=torch.bfloat16)
    ref.eval()
    embed_tokens = ref.model.language_model.embed_tokens
    image_token_id = ref.config.image_token_id

    cers = []
    for s in samples:
        text = _read_page(s, tower, generator, embed_tokens, image_token_id, ref, text_args, device)
        c = _cer(s["hf_output"], text)
        cers.append(c)
        print(f"[{s['name']}] CERvsHF={c * 100:.2f}%  TT={text[:120]!r}")

    mean_cer = sum(cers) / len(cers)
    print(f"mean CER vs HF reference: {mean_cer * 100:.2f}%  (gate: <= {MEAN_CER_GATE * 100:.2f}%)")
    assert mean_cer <= MEAN_CER_GATE
