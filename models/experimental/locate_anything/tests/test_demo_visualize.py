# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""End-to-end LocateAnything-3B demo on a single Blackhole p150a (chip 2):
run the full on-device pipeline (MoonViT vision + Qwen2.5-3B LLM) on a real
image, parse the generated <ref>/<box> tokens into pixel boxes, draw them, and
save the visualization.

Env knobs:
  LA_IMAGE   path to input image       (default: <repo>/../image.png)
  LA_QUERY   detection query           (default: "car")
  LA_OUT     output visualization path (default: <repo>/../image_result.png)
  LA_IN_TOKEN_LIMIT  vision token cap  (default: 1024)
  LA_MAX_NEW max new decode tokens     (default: 128)
  HF_MODEL   extracted LA-Qwen2.5-3B dir (required)

Run (chip 2, no cross-chip fabric):
  TT_METAL_HOME=$(pwd) ARCH_NAME=blackhole TT_METAL_VISIBLE_DEVICES=2 MESH_DEVICE=N150 \
    HF_MODEL=~/.cache/locate_anything/LA-Qwen2.5-3B LA_QUERY="car" \
    ./python_env/bin/python -m pytest -svq \
    models/experimental/locate_anything/tests/test_demo_visualize.py
"""
import os
import re
import time

import pytest
import torch
from loguru import logger
from PIL import Image, ImageDraw, ImageFont

import ttnn
from models.demos.qwen25_vl.tt.common import merge_vision_tokens, preprocess_inputs_prefill
from models.experimental.locate_anything.reference import la_inputs
from models.experimental.locate_anything.tests.bench_locate_anything import create_tt_model, create_tt_page_table
from models.experimental.locate_anything.tt.vision import MoonViT
from models.tt_transformers.tt.common import Mode, sample_host
from models.tt_transformers.tt.generator import Generator as TTTGenerator
from models.tt_transformers.tt.model_config import DecodersPrecision

EOS_TOKEN_ID = 151645
PAGE_PARAMS = {"page_block_size": 32, "page_max_num_blocks": 1024}

_BOX_RE = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")
_POINT_RE = re.compile(r"<box><(\d+)><(\d+)></box>")
_REF_RE = re.compile(r"<ref>(.*?)</ref>")
# token-aware iterator: a <ref>label</ref> OR a 4-coord box OR a 2-coord point
_ITEM_RE = re.compile(
    r"<ref>(?P<ref>.*?)</ref>|<box><(?P<x1>\d+)><(?P<y1>\d+)><(?P<x2>\d+)><(?P<y2>\d+)></box>|<box><(?P<px>\d+)><(?P<py>\d+)></box>"
)

_PALETTE = [
    (255, 64, 64),
    (64, 200, 64),
    (64, 128, 255),
    (255, 180, 0),
    (200, 64, 255),
    (0, 200, 200),
    (255, 100, 180),
    (140, 220, 60),
]


def parse_detections(answer: str, W: int, H: int):
    """Walk the answer in order, attaching each box/point to the current <ref> label."""
    dets = []
    cur_label = None
    for m in _ITEM_RE.finditer(answer):
        if m.group("ref") is not None:
            cur_label = m.group("ref").strip()
        elif m.group("x1") is not None:
            x1, y1, x2, y2 = (int(m.group(k)) for k in ("x1", "y1", "x2", "y2"))
            dets.append(
                {
                    "label": cur_label,
                    "box": (x1 / 1000 * W, y1 / 1000 * H, x2 / 1000 * W, y2 / 1000 * H),
                }
            )
        elif m.group("px") is not None:
            px, py = int(m.group("px")), int(m.group("py"))
            dets.append({"label": cur_label, "point": (px / 1000 * W, py / 1000 * H)})
    return dets


def _font(size):
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def visualize(image: Image.Image, dets, out_path: str):
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    W, H = img.size
    lw = max(2, round(min(W, H) / 300))
    font = _font(max(14, round(min(W, H) / 45)))
    labels = sorted({d.get("label") or "obj" for d in dets})
    color_of = {lab: _PALETTE[i % len(_PALETTE)] for i, lab in enumerate(labels)}
    for d in dets:
        col = color_of.get(d.get("label") or "obj")
        lab = d.get("label") or ""
        if "box" in d:
            x1, y1, x2, y2 = d["box"]
            draw.rectangle([x1, y1, x2, y2], outline=col, width=lw)
            if lab:
                tb = draw.textbbox((0, 0), lab, font=font)
                tw, th = tb[2] - tb[0], tb[3] - tb[1]
                ty = max(0, y1 - th - 4)
                draw.rectangle([x1, ty, x1 + tw + 6, ty + th + 4], fill=col)
                draw.text((x1 + 3, ty + 2), lab, fill=(255, 255, 255), font=font)
        elif "point" in d:
            px, py = d["point"]
            r = lw * 3
            draw.ellipse([px - r, py - r, px + r, py + r], outline=col, width=lw)
            if lab:
                draw.text((px + r + 2, py - r), lab, fill=col, font=font)
    img.save(out_path)
    return out_path


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": False, "trace_region_size": 50000000, "num_command_queues": 1}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [{"N150": (1, 1)}.get(os.environ.get("MESH_DEVICE"), (1, 1))],
    indirect=True,
)
def test_demo_visualize(mesh_device):
    from transformers import Qwen2ForCausalLM

    repo_parent = os.path.abspath(os.path.join(os.environ.get("TT_METAL_HOME", "."), ".."))
    image_path = os.environ.get("LA_IMAGE", os.path.join(repo_parent, "image.png"))
    query = os.environ.get("LA_QUERY", "car")
    out_path = os.environ.get("LA_OUT", os.path.join(repo_parent, "image_result.png"))
    in_token_limit = int(os.environ.get("LA_IN_TOKEN_LIMIT", "1024"))
    max_new = int(os.environ.get("LA_MAX_NEW", "128"))
    max_seq_len = 4096
    hf_model_dir = os.environ.get("HF_MODEL")
    assert hf_model_dir, "Set HF_MODEL to the extracted LA-Qwen2.5-3B directory"
    assert os.path.isfile(image_path), f"image not found: {image_path}"

    logger.info(f"image={image_path} query={query!r} out={out_path}")
    image = Image.open(image_path).convert("RGB")
    W, H = image.size

    # --- Build model (LLM) + page table ---
    model_args, model, paged_attention_config, tt_kv_cache = create_tt_model(
        mesh_device,
        instruct=False,
        max_batch_size=1,
        optimizations=lambda ma: DecodersPrecision.accuracy(ma.n_layers, ma.model_name),
        max_seq_len=max_seq_len,
        page_params=PAGE_PARAMS,
        dtype=ttnn.bfloat8_b,
        use_paged_kv_cache=True,
    )
    tokenizer = model_args.tokenizer
    generator = TTTGenerator([model], [model_args], mesh_device)
    page_table = create_tt_page_table(paged_attention_config, model_args)

    # --- Build inputs from the real image ---
    bundle = la_inputs.build_inputs(tokenizer, image, query, in_token_limit=in_token_limit)
    input_ids = bundle["input_ids"]
    attention_mask = bundle["attention_mask"]
    grid_hw = bundle["grid_hw"]
    real_seq_len = input_ids.shape[1]
    last_token_idx = real_seq_len - 1
    logger.info(f"grid_hw={grid_hw} n_img_tokens={bundle['n_img_tokens']} seq_len={real_seq_len}")

    # --- Vision on device (MoonViT + projector) ---
    logger.info("MoonViT vision on device (chip 2)...")
    vis = MoonViT(mesh_device, la_inputs.find_model_path(), grid_hw, dtype=ttnn.bfloat16)
    vit_proj = vis.forward(bundle["pixel_values"].float()).to(torch.float32)  # [N,2048]

    # --- Host text-embed + merge ---
    hf_model = Qwen2ForCausalLM.from_pretrained(hf_model_dir, torch_dtype=torch.float32)
    embed_tokens = hf_model.get_input_embeddings()
    with torch.no_grad():
        text_embeds = embed_tokens(input_ids)
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        pad_embedding = embed_tokens(torch.tensor(pad_token_id))

    class _MergeConfig:
        image_token_id = la_inputs.IMAGE_TOKEN_INDEX

    input_embeds = merge_vision_tokens(input_ids, text_embeds, vit_proj.to(text_embeds.dtype), _MergeConfig())
    input_prefill_pt, decoding_pos, prefill_lens = preprocess_inputs_prefill(
        [input_embeds[0]], model_args, attention_mask, pad_embedding=pad_embedding
    )
    decode_start_pos = decoding_pos[0]
    embeds = input_prefill_pt[0].unsqueeze(0).to(torch.float32)

    # --- Prefill ---
    logger.info("Prefill...")
    model.switch_mode(Mode.PREFILL)
    t0 = time.time()
    tokens_embd, rot_mats_global, tt_page_table, _ = model.prepare_inputs_prefill_embeds(
        embeds, start_pos=0, page_table=page_table, last_token_idx=last_token_idx
    )
    tt_logits = model.ttnn_prefill_forward(
        tokens_embd,
        rot_mats_global=rot_mats_global,
        rot_mats_local=None,
        user_id=0,
        page_table=tt_page_table,
        get_last_token=(last_token_idx // 32) * 32,
        kv_cache=tt_kv_cache,
    )
    prefill_last_logits = model.process_output_prefill(tt_logits.cpu(), last_token_idx=last_token_idx % 32)
    ttnn.deallocate(tt_logits)
    ttnn.deallocate(tokens_embd)
    if tt_page_table is not None:
        ttnn.deallocate(tt_page_table)

    # --- Greedy AR decode ---
    logger.info("Greedy AR decode...")
    first_tok = int(torch.argmax(prefill_last_logits[: model.vocab_size]).item())
    generated_ids = [first_tok]
    out_tok = torch.tensor([[first_tok]], dtype=torch.int64)
    current_pos = torch.tensor([decode_start_pos], dtype=torch.int64)
    if first_tok != EOS_TOKEN_ID:
        for step in range(max_new - 1):
            logits, _ = generator.decode_forward(
                out_tok,
                current_pos,
                page_table=page_table,
                kv_cache=[tt_kv_cache],
                enable_trace=False,
                reset_batch=(step == 0),
            )
            _, next_tok_t = sample_host(logits, temperature=0, top_p=1.0, on_host=True)
            next_tok = int(next_tok_t.reshape(-1)[0].item())
            current_pos = current_pos + 1
            generated_ids.append(next_tok)
            if next_tok == EOS_TOKEN_ID:
                break
            out_tok = torch.tensor([[next_tok]], dtype=torch.int64)
    dt = time.time() - t0

    answer = tokenizer.decode(generated_ids, skip_special_tokens=False)
    logger.info(f"Generated {len(generated_ids)} tokens in {dt:.1f}s")
    print(f"RAW ANSWER: {answer}")

    dets = parse_detections(answer, W, H)
    n_box = sum(1 for d in dets if "box" in d)
    n_pt = sum(1 for d in dets if "point" in d)
    print(f"PARSED detections: {n_box} boxes, {n_pt} points")
    for d in dets:
        print(f"  {d}")

    saved = visualize(image, dets, out_path)
    print(f"SAVED visualization -> {saved}")
    print(f"num_detections={len(dets)}")
    assert os.path.isfile(saved)
