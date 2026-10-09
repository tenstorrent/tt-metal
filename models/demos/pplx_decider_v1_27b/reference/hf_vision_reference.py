# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""HF vision-tower golden for pplx-decider-v1-27b (``Qwen3_5VisionModel``, 27 ViT blocks + merger), CPU, bf16.

The tower is built from the snapshot ``visual.*`` tensors only (``strict=True``, 333 tensors, 0.92 GB
bf16); the 54 GB checkpoint is never loaded. It is constructed with the target dtype as the torch
default dtype, as ``from_pretrained(dtype=...)`` does, so the rotary ``inv_freq`` buffer stays fp32.

``capture`` mirrors ``Qwen3_5VisionModel.forward`` step by step (``modeling_qwen3_5.py`` 1084-1125,
same HF submodules and helpers) and records patch-embed output, pos-embed-added block-0 input, the
output of every block, and the merger output (the 5120-wide features spliced into the text), plus
the rotary position ids, cos/sin, bilinear pos-embed indices/weights and cu_seqlens HF used.
``--verify`` proves the capture equals the HF tower's own ``forward`` and
``Qwen3_5Model.get_image_features`` (on a ``Qwen3_5Model`` whose text model is a 1-layer stub).

Usage::

    python -m models.demos.pplx_decider_v1_27b.reference.hf_vision_reference --verify
    python -m models.demos.pplx_decider_v1_27b.reference.hf_vision_reference
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from functools import cached_property
from pathlib import Path

import torch
from loguru import logger
from safetensors.torch import save_file
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model, Qwen3_5VisionModel
from transformers.vision_utils import (
    get_vision_bilinear_indices_and_weights,
    get_vision_cu_seqlens,
    get_vision_position_ids,
)

from models.demos.pplx_decider_v1_27b.reference.hf_reference import SnapshotReader

VISION_PREFIX = "visual."
GOLDEN_DIR = Path("/local/ttuser/gtobar/artifacts/pplx_decider/goldens/vision")


class VisionSnapshotReader(SnapshotReader):
    """``SnapshotReader`` plus the full composite config and the vision config."""

    @cached_property
    def config(self):
        cfg = AutoConfig.from_pretrained(self.path, local_files_only=True)
        cfg._attn_implementation = "sdpa"
        return cfg

    @cached_property
    def vision_config(self):
        vc = self.config.vision_config
        vc._attn_implementation = "sdpa"
        return vc

    def vision_state_dict(self) -> dict[str, torch.Tensor]:
        return self.tensors_with_prefix(VISION_PREFIX)


@contextlib.contextmanager
def default_dtype(dtype: torch.dtype):
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old)


def build_vision_tower(reader: VisionSnapshotReader, dtype=torch.bfloat16) -> Qwen3_5VisionModel:
    with default_dtype(dtype):
        model = Qwen3_5VisionModel(reader.vision_config)
    model.load_state_dict(reader.vision_state_dict(), strict=True)
    assert model.rotary_pos_emb.inv_freq.dtype == torch.float32
    assert all(p.dtype == dtype for p in model.parameters())
    return model.eval()


@torch.no_grad()
def capture(model: Qwen3_5VisionModel, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> dict[str, torch.Tensor]:
    """``Qwen3_5VisionModel.forward`` unrolled, every intermediate kept (keys sorted for safetensors)."""
    pixel_values = pixel_values.type(model.dtype)  # Qwen3_5Model.get_image_features does this first
    idx, wts = get_vision_bilinear_indices_and_weights(
        grid_thw, num_grid_per_side=model.num_grid_per_side, spatial_merge_size=model.config.spatial_merge_size
    )
    pos_ids = get_vision_position_ids(grid_thw, model.spatial_merge_size)
    cu_seqlens = get_vision_cu_seqlens(grid_thw)
    out: dict[str, torch.Tensor] = {
        "bilinear_indices": idx,
        "bilinear_weights": wts,
        "rotary_pos_ids": pos_ids,
        "cu_seqlens": cu_seqlens,
    }
    h = model.patch_embed(pixel_values)
    out["patch_embed"] = h
    pos_embeds = (model.pos_embed(idx) * wts[:, :, None]).sum(0)
    h = h + pos_embeds.to(h.dtype)
    out["pos_embed"] = pos_embeds
    out["embed_with_pos"] = h
    rotary = model.rotary_pos_emb(pos_ids)
    seq_len = h.shape[0]
    h = h.reshape(seq_len, -1)
    rotary = rotary.reshape(seq_len, -1)
    emb = torch.cat((rotary, rotary), dim=-1)
    cos, sin = emb.cos(), emb.sin()
    out["rotary_cos"], out["rotary_sin"] = cos, sin
    for i, blk in enumerate(model.blocks):
        h = blk(h, cu_seqlens=cu_seqlens, position_embeddings=(cos, sin))
        out[f"block_{i:02d}"] = h
    out["merger"] = model.merger(h)
    return {k: v.contiguous() for k, v in out.items()}


def max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max())


@torch.no_grad()
def verify(reader: VisionSnapshotReader, rows: list[dict], dtype=torch.bfloat16) -> dict:
    """Capture == HF tower forward == ``Qwen3_5Model.get_image_features`` (bit-exact expected)."""
    from models.demos.pplx_decider_v1_27b.reference.hf_decision_golden import pcc

    tower = build_vision_tower(reader, dtype)
    # A real Qwen3_5Model with the real vision tower and a 1-layer, tiny text stub (not used here).
    cfg = AutoConfig.from_pretrained(reader.path, local_files_only=True)
    tc = cfg.text_config
    tc.num_hidden_layers, tc.layer_types, tc.vocab_size = 1, tc.layer_types[:1], 16
    tc.hidden_size, tc.intermediate_size, tc.num_attention_heads, tc.num_key_value_heads = 64, 64, 1, 1
    tc.linear_num_key_heads, tc.linear_num_value_heads = 1, 1
    full = Qwen3_5Model._from_config(cfg, dtype=dtype, attn_implementation="sdpa").eval()
    full.visual.load_state_dict(reader.vision_state_dict(), strict=True)
    assert full.visual.rotary_pos_emb.inv_freq.dtype == torch.float32
    res = {}
    for r in rows:
        pv, grid = r["enc"]["pixel_values"], r["enc"]["image_grid_thw"]
        cap = capture(tower, pv, grid)
        fwd = tower(pv.type(tower.dtype), grid)
        feats = torch.cat(full.get_image_features(pv, grid).pooler_output, 0)
        res[r["id"]] = {
            "capture_vs_forward_last_hidden_max_diff": max_diff(cap["block_26"], fwd.last_hidden_state),
            "capture_vs_forward_merger_max_diff": max_diff(cap["merger"], fwd.pooler_output),
            "capture_vs_get_image_features_max_diff": max_diff(cap["merger"], feats),
            "capture_vs_get_image_features_pcc": pcc(cap["merger"], feats),
            "bit_exact": bool(
                torch.equal(cap["merger"], feats) and torch.equal(cap["block_26"], fwd.last_hidden_state)
            ),
        }
        logger.info(f"verify {r['id']}: {res[r['id']]}")
    return res


def main() -> None:
    from models.demos.pplx_decider_v1_27b.reference.hf_decision_golden import peak_rss_gb
    from models.demos.pplx_decider_v1_27b.reference.image_decision_prompts import (
        build_image_prompt_set,
        image_tokenizer,
    )

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=GOLDEN_DIR)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--verify", action="store_true", help="only the capture == HF identity check")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.time()
    reader = VisionSnapshotReader()
    rows = build_image_prompt_set(image_tokenizer(), args.out / "images")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.verify:
        res = {"dtype": "bf16", "rows": verify(reader, rows), "peak_rss_gb": round(peak_rss_gb(), 2)}
        (args.out / "vision_identity_check.json").write_text(json.dumps(res, indent=2) + "\n")
        return
    (args.out / "inputs").mkdir(exist_ok=True)
    (args.out / "tower").mkdir(exist_ok=True)
    tower = build_vision_tower(reader, torch.bfloat16)
    summary = []
    for r in rows:
        enc = r["enc"]
        # App inputs exactly as the processor returns them (pixel_values is the processor's fp32 output).
        save_file({k: v.contiguous() for k, v in enc.items()}, str(args.out / "inputs" / f"{r['id']}.safetensors"))
        start = time.time()
        cap = capture(tower, enc["pixel_values"], enc["image_grid_thw"])
        secs = time.time() - start
        save_file(
            cap,
            str(args.out / "tower" / f"{r['id']}.bf16.safetensors"),
            metadata={
                "dtype": "bf16",
                "grid_thw": json.dumps(r["grid_thw"]),
                "note": "block_NN = output of block NN; embed_with_pos = block_00 input; merger = pooler_output; "
                "rotary_cos/sin, pos_embed and bilinear_weights are fp32 as HF computes them",
            },
        )
        summary.append(
            {k: r[k] for k in ("id", "image_size", "grid_thw", "patches", "image_tokens")} | {"tower_s": round(secs, 2)}
        )
        logger.info(f"{r['id']}: patches {r['patches']} -> merger {tuple(cap['merger'].shape)} in {secs:.1f}s")
    res = {
        "dtype": "bf16",
        "runtime_s": round(time.time() - t0, 1),
        "peak_rss_gb": round(peak_rss_gb(), 2),
        "rows": summary,
    }
    (args.out / "tower_summary.json").write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()
