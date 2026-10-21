# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host (CPU) HF reference for one image + question prompt through GLM-5.3-Flash.

    python models/demos/glm53_flash/reference/hf_multimodal_reference.py \\
        --image models/demos/glm53_flash/demo/sample_prompts/shapes_hello.png \\
        --question "What shapes and what word are in this image? Answer in one sentence." \\
        --generate 16 --out bringup/references/glm53_flash/MM_shapes_hello

Pipeline (what ``Glm5NextForConditionalGeneration`` does, without instantiating the 320 B text model
at once): processor + chat template (``tt/multimodal.py::encode_multimodal``, the encoding the device
uses) -> ``Glm5NextVisionModel`` with the real ``model.visual.*`` weights in fp32 on the CPU
(``get_image_features``: ``pooler_output`` split per image) -> ``embed_tokens[input_ids]`` (bf16) with
the features cast to bf16 scattered at the ``image_token_id`` rows (``masked_scatter``) -> the 4 mHC
streams -> all 45 text layers in bf16 streamed one module at a time per token
(``hf_layer_reference.generate_streaming``: prefill pass, then greedy decode passes; resumable).

Writes to ``--out``: ``prompt.json`` (ids, question, image, grid, image-token rows), ``features.pt``
(``fp32`` tower features and ``bf16_model`` = the tower run in bf16 as the shipped model does, per
image), ``inputs_embeds.pt`` (the merged ``[1, S, 4096]`` bf16 embeddings and the image mask: the
reference of the device stream construction), ``generate.json`` / ``generate_state.pt``
(``generate_streaming``) and ``text.txt`` (the decoded greedy tokens).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

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


def merge_image_embeddings(
    embed: torch.Tensor, input_ids: torch.Tensor, features: list[torch.Tensor], image_token_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """``Glm5NextModel.forward``'s scatter: ``(inputs_embeds [1, S, H] bf16, image_mask [1, S] bool)``."""
    emb = embed[input_ids]  # bf16 [1, S, H]
    mask = input_ids == image_token_id
    feats = torch.cat(features, dim=0).to(emb.dtype)
    assert int(mask.sum()) == feats.shape[0], (int(mask.sum()), feats.shape)
    return emb.masked_scatter(mask.unsqueeze(-1), feats), mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, default=None)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--question", type=str, required=True)
    ap.add_argument("--think", action="store_true", help="keep the template's <think> (no </think> appended)")
    ap.add_argument("--generate", type=int, default=16)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    from transformers import AutoProcessor

    from models.demos.glm53_flash.tt.multimodal import encode_multimodal

    snapshot = args.snapshot or default_snapshot_dir()
    cfg, text = load_config(snapshot)
    image_token_id = int(cfg.image_token_id)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(str(snapshot))
    tok = processor.tokenizer

    t0 = time.time()
    input_ids, pixel_values, grid = encode_multimodal(processor, args.image, args.question, think=args.think)
    S = input_ids.shape[1]
    n_img = int((input_ids == image_token_id).sum())
    print(
        f"prompt: {S} tokens, {n_img} image tokens, grid {grid.tolist()}, patches {tuple(pixel_values.shape)}",
        flush=True,
    )
    (out / "prompt.json").write_text(
        json.dumps(
            {
                "image": str(args.image),
                "question": args.question,
                "think": bool(args.think),
                "input_ids": input_ids[0].tolist(),
                "image_grid_thw": grid.tolist(),
                "image_token_id": image_token_id,
                "n_image_tokens": n_img,
                "image_rows": torch.nonzero(input_ids[0] == image_token_id).reshape(-1).tolist(),
                "text": tok.decode(input_ids[0]),
            },
            indent=1,
        )
    )

    feats = hf_image_features(snapshot, pixel_values, grid, torch.float32)
    t1 = time.time()
    feats16 = hf_image_features(snapshot, pixel_values, grid, torch.bfloat16)
    print(
        f"tower: fp32 {t1 - t0:.0f} s, bf16 {time.time() - t1:.0f} s; features {[tuple(f.shape) for f in feats]}",
        flush=True,
    )
    torch.save(
        {"fp32": feats, "bf16_model": feats16, "image_grid_thw": grid, "pixel_values": pixel_values},
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
            "image": str(args.image),
        },
        out / "inputs_embeds.pt",
    )
    del embed
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
