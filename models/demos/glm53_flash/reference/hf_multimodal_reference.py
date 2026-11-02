# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host (CPU) HF reference for one image or one video (+ question) prompt through GLM-5.3-Flash.

    # image
    python models/demos/glm53_flash/reference/hf_multimodal_reference.py \\
        --image models/demos/glm53_flash/demo/sample_prompts/shapes_hello.png \\
        --question "What shapes and what word are in this image? Answer in one sentence." \\
        --generate 16 --out bringup/references/glm53_flash/MM_shapes_hello
    # video: a synthetic 4-frame clip of the same image (frame k rotated by 10 k degrees, 2 fps -> 2 temporal
    # groups of 256 tokens), tower features + the merged embeddings only (--generate 0; the 45-layer CPU
    # generation takes hours per token batch and is resumable like the image reference)
    python models/demos/glm53_flash/reference/hf_multimodal_reference.py \\
        --image models/demos/glm53_flash/demo/sample_prompts/shapes_hello.png --clip-frames 4 \\
        --question "What happens in this clip? Answer in one sentence." \\
        --generate 0 --out bringup/references/glm53_flash/MMV_shapes_clip4
    # a directory of frame images instead of the synthetic clip: --video <dir> (sorted by name)

Pipeline (what ``Glm5NextForConditionalGeneration`` does, without instantiating the 320 B text model
at once): processor + chat template (``tt/multimodal.py::encode_prompt``, the encoding the device
uses) -> ``Glm5NextVisionModel`` with the real ``model.visual.*`` weights in fp32 on the CPU
(``get_image_features`` / ``get_video_features``: the video grid flattened to one ``(1, h, w)`` window per
temporal group, ``pooler_output`` split per item) -> ``embed_tokens[input_ids]`` (bf16) with the features cast to
bf16 scattered at EVERY ``image_token_id`` row (images and video frames alike, in prompt order:
``masked_scatter``) -> the 4 mHC streams -> all 45 text layers in bf16 streamed one module at a time per token
(``hf_layer_reference.generate_streaming``: prefill pass, then greedy decode passes; resumable).

Writes to ``--out``: ``prompt.json`` (ids, question, media, grids, item kinds, vision-token rows), ``features.pt``
(``fp32`` tower features and ``bf16_model`` = the tower run in bf16 as the shipped model does, per item),
``inputs_embeds.pt`` (the merged ``[1, S, 4096]`` bf16 embeddings and the vision mask: the reference of the
device stream construction), ``generate.json`` / ``generate_state.pt`` (``generate_streaming``) and ``text.txt``
(the decoded greedy tokens) when ``--generate > 0``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checkpoint import EMBED_KEY, default_snapshot_dir, load_tensors, shard_headers  # noqa: E402
from hf_layer_reference import generate_streaming, load_config  # noqa: E402


def build_hf_vision(snapshot: Path, weights: dict, dtype=torch.float32):
    """``Glm5NextVisionModel`` from the raw tower weights (non-persistent rope buffer recomputed); the
    same construction as ``tests/test_vision.py``."""
    from transformers import AutoConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionModel, Glm5NextVisionRotaryEmbedding

    vcfg = AutoConfig.from_pretrained(str(snapshot)).vision_config
    vcfg._attn_implementation = "sdpa"
    with torch.device("meta"):
        hf = Glm5NextVisionModel(vcfg)
    hf = hf.to_empty(device="cpu")
    missing, unexpected = hf.load_state_dict(weights, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    inv_freq, _ = Glm5NextVisionRotaryEmbedding.compute_axial_rope_parameters(vcfg)
    hf.rotary_pos_emb.inv_freq = inv_freq.clone()
    hf.rotary_pos_emb.original_inv_freq = inv_freq.clone()
    return hf.to(dtype).eval()


@torch.no_grad()
def hf_image_features(snapshot: Path, pixel_values: torch.Tensor, grid_thw: torch.Tensor, dtype=torch.float32):
    """HF ``get_image_features``: per-image ``[tokens, 4096]`` features (``dtype`` tower)."""
    from models.demos.glm53_flash.tt.vision import load_vision_weights

    weights = load_vision_weights(snapshot)
    hf = build_hf_vision(snapshot, weights, dtype)
    out = hf(pixel_values.to(dtype), grid_thw=grid_thw).pooler_output.float()
    split = (grid_thw.prod(-1) // hf.spatial_merge_size**2).tolist()
    return list(torch.split(out, split))


@torch.no_grad()
def hf_video_features(
    snapshot: Path, pixel_values_videos: torch.Tensor, video_grid_thw: torch.Tensor, dtype=torch.float32
):
    """HF ``get_video_features``: the grid flattened to ``[1, h, w] * t`` (one attention window per temporal
    group), per-video ``[t * h * w / 4, 4096]`` features."""
    from models.demos.glm53_flash.tt.mm_inputs import flatten_grid

    flat = flatten_grid(video_grid_thw)
    per_window = hf_image_features(snapshot, pixel_values_videos, flat, dtype)
    out, i = [], 0
    for t in video_grid_thw[:, 0].tolist():
        out.append(torch.cat(per_window[i : i + int(t)], dim=0))
        i += int(t)
    return out


def hf_item_features(snapshot: Path, items, dtype=torch.float32) -> list[torch.Tensor]:
    """Per ``MMItem`` (prompt order) the HF features: images through ``hf_image_features``, videos through
    ``hf_video_features`` (one tower construction per call)."""
    out = []
    for it in items:
        if it.kind == "image":
            out.extend(hf_image_features(snapshot, it.pixel_values, it.grid_thw, dtype))
        else:
            out.extend(hf_video_features(snapshot, it.pixel_values, it.grid_thw, dtype))
    return out


def merge_image_embeddings(
    embed: torch.Tensor, input_ids: torch.Tensor, features: list[torch.Tensor], image_token_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """``Glm5NextModel.forward``'s scatter over every vision row (images and video frames share
    ``image_token_id``; the features arrive in prompt order): ``(inputs_embeds [1, S, H] bf16, mask [1, S] bool)``."""
    emb = embed[input_ids]  # bf16 [1, S, H]
    mask = input_ids == image_token_id
    feats = torch.cat(features, dim=0).to(emb.dtype)
    assert int(mask.sum()) == feats.shape[0], (int(mask.sum()), feats.shape)
    return emb.masked_scatter(mask.unsqueeze(-1), feats), mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, default=None)
    ap.add_argument("--image", type=Path, default=None, help="the image (or the source of --clip-frames)")
    ap.add_argument("--video", type=Path, default=None, help="a directory of frame images (sorted by name)")
    ap.add_argument("--clip-frames", type=int, default=0, help="a synthetic clip of N rotated copies of --image")
    ap.add_argument("--clip-step-deg", type=float, default=10.0)
    ap.add_argument("--fps", type=float, default=2.0, help="the clip's frame rate (the processor samples at 2 fps)")
    ap.add_argument("--question", type=str, required=True)
    ap.add_argument("--think", action="store_true", help="keep the template's <think> (no </think> appended)")
    ap.add_argument("--generate", type=int, default=16, help="greedy tokens through the 45 CPU layers (0 = skip)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.image is None and args.video is None:
        ap.error("--image and/or --video is required")

    from transformers import AutoProcessor

    from models.demos.glm53_flash.tt.multimodal import encode_prompt, load_frames, synthetic_clip

    snapshot = args.snapshot or default_snapshot_dir()
    cfg, text = load_config(snapshot)
    image_token_id = int(cfg.image_token_id)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(str(snapshot))
    tok = processor.tokenizer

    content: list[dict] = []
    media: dict = {}
    frames = None
    if args.video is not None:
        frames = load_frames(args.video)
        media["video"] = str(args.video)
    elif args.clip_frames:
        frames = synthetic_clip(args.image, args.clip_frames, args.clip_step_deg)
        media[
            "video"
        ] = f"synthetic clip of {args.image}: {args.clip_frames} frames, {args.clip_step_deg} deg per frame"
    if frames is not None:
        content.append({"type": "video", "video": frames})
        media.update({"frames": int(frames.shape[0]), "frame_size": list(frames.shape[1:3]), "fps": args.fps})
    if args.image is not None and frames is None:
        content.append({"type": "image", "image": args.image})
        media["image"] = str(args.image)
    content.append({"type": "text", "text": args.question})

    t0 = time.time()
    enc = encode_prompt(processor, content, fps=args.fps, think=args.think)
    input_ids = enc.input_ids
    S = input_ids.shape[1]
    n_vis = int((input_ids == image_token_id).sum())
    kinds = [it.kind for it in enc.items]
    print(
        f"prompt: {S} tokens, {n_vis} vision tokens over {kinds}, grids {[it.grid_thw.tolist() for it in enc.items]}, "
        f"patches {[tuple(it.pixel_values.shape) for it in enc.items]}",
        flush=True,
    )
    (out / "prompt.json").write_text(
        json.dumps(
            {
                **media,
                "question": args.question,
                "think": bool(args.think),
                "input_ids": input_ids[0].tolist(),
                "items": kinds,
                "image_grid_thw": None if enc.image_grid_thw is None else enc.image_grid_thw.tolist(),
                "video_grid_thw": None if enc.video_grid_thw is None else enc.video_grid_thw.tolist(),
                "image_token_id": image_token_id,
                "n_image_tokens": n_vis,
                "image_rows": torch.nonzero(input_ids[0] == image_token_id).reshape(-1).tolist(),
                "text": tok.decode(input_ids[0]),
            },
            indent=1,
        )
    )

    feats = hf_item_features(snapshot, enc.items, torch.float32)
    t1 = time.time()
    feats16 = hf_item_features(snapshot, enc.items, torch.bfloat16)
    print(
        f"tower: fp32 {t1 - t0:.0f} s, bf16 {time.time() - t1:.0f} s; features {[tuple(f.shape) for f in feats]}",
        flush=True,
    )
    torch.save(
        {
            "fp32": feats,
            "bf16_model": feats16,
            "items": kinds,
            "image_grid_thw": enc.image_grid_thw,
            "pixel_values": enc.pixel_values,
            "video_grid_thw": enc.video_grid_thw,
            "pixel_values_videos": enc.pixel_values_videos,
            "frames": None if frames is None else torch.from_numpy(np.ascontiguousarray(frames)),
            "fps": args.fps,
        },
        out / "features.pt",
    )

    headers = shard_headers(snapshot)
    embed = load_tensors(snapshot, [EMBED_KEY], headers)[EMBED_KEY]
    inputs_embeds, mask = merge_image_embeddings(embed, input_ids, feats, image_token_id)
    torch.save(
        {
            "input_ids": input_ids,
            "inputs_embeds": inputs_embeds,
            "image_mask": mask,
            "question": args.question,
            **{k: v for k, v in media.items() if isinstance(v, str)},
        },
        out / "inputs_embeds.pt",
    )
    del embed
    if args.generate <= 0:
        print(f"inputs_embeds {tuple(inputs_embeds.shape)} {inputs_embeds.dtype}; no generation requested", flush=True)
        return
    x0 = inputs_embeds.unsqueeze(2).expand(-1, -1, text.hc_mult, -1).contiguous()
    print(
        f"inputs_embeds {tuple(inputs_embeds.shape)} {inputs_embeds.dtype}; generating {args.generate} tokens",
        flush=True,
    )
    res = generate_streaming(snapshot, text, input_ids, args.generate, out, resume=not args.no_resume, x0=x0)
    gen_text = tok.decode(res["generated"])
    (out / "text.txt").write_text(gen_text)
    print(f"generated ids {res['generated']}\ntext: {gen_text!r}", flush=True)


if __name__ == "__main__":
    main()
