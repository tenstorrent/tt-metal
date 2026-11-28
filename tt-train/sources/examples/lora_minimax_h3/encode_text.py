# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Encode the cached captions with the Qwen3-VL text tower on a Tenstorrent mesh.

    python encode_text.py --cache cache/pixelart-150 --weights <hf snapshot>/text_encoder --mesh 1,8

Reads `tokens/N.npy` from a `prepare.py` cache and writes `embeds/N.npy`: bf16 `(L, 5120)`, the residual
stream after decoder layer 50, matching the GPU cache's `prompt_embeds`. `--reference-dir` holds `N.npy`
float32 dumps of those GPU embeddings to compare against.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import ml_dtypes
import numpy as np
import ttnn
import ttml
from ttml.common.utils import build_causal_mask, no_grad

from prepare import MAX_TEXT_TOKENS
from text_encoder import TEXT_ENCODER_LAYER, TextTower, TextTowerConfig, load_text_tower

log = logging.getLogger("minimax_h3.encode_text")

TILE = 32
MODEL_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
DEFAULT_WEIGHTS = (
    Path.home() / ".cache/huggingface/hub/models--MiniMaxAI--MiniMax-H3/snapshots" / MODEL_REVISION / "text_encoder"
)


def open_mesh(dp: int, tp: int, seed: int = 0):
    if "TT_MESH_GRAPH_DESC_PATH" not in os.environ:
        log.warning("TT_MESH_GRAPH_DESC_PATH is not set; fabric picks a default descriptor")
    ttml.open_device_mesh(ttml.Mesh((dp, tp), ("dp", "tp")))
    ctx = ttml.autograd.AutoContext.get_instance()
    if not ctx.is_parallelism_context_initialized():
        ctx.initialize_parallelism_context(ttml.autograd.DistributedConfig(enable_ddp=dp > 1, enable_tp=tp > 1))
    ctx.set_seed(seed)
    return ctx


def ids_tensor(tokens: np.ndarray, padded_len: int):
    ids = np.zeros((1, 1, 1, padded_len), dtype=np.uint32)
    ids[0, 0, 0, : tokens.shape[0]] = tokens
    device = ttml.autograd.AutoContext.get_instance().get_device()
    mapper = ttml.core.distributed.replicate_tensor_to_mesh_mapper(device)
    return ttml.autograd.Tensor.from_numpy(ids, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32, mapper)


def pcc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel() - a.astype(np.float64).mean()
    b = b.astype(np.float64).ravel() - b.astype(np.float64).mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def row_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-token cosine. The first token's row is ~50x larger than the rest, so a flat PCC hides the others."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-30)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", required=True)
    p.add_argument(
        "--weights", default=str(DEFAULT_WEIGHTS), help="HF text_encoder/ folder with the safetensors shards"
    )
    p.add_argument("--mesh", default="1,8", help="dp,tp")
    p.add_argument(
        "--layers", type=int, default=TEXT_ENCODER_LAYER, help="decoder layers to run; the output is pre-norm"
    )
    p.add_argument("--max-tokens", type=int, default=MAX_TEXT_TOKENS)
    p.add_argument(
        "--pad-to",
        type=int,
        default=0,
        help="fixed padded length so kernels compile once (default: longest caption rounded up to 32)",
    )
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--reference-dir", default=None, help="N.npy float32 GPU prompt_embeds to compare against")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    weights = Path(args.weights)
    if not (weights / "model.safetensors.index.json").exists():
        raise SystemExit(
            f"no text_encoder weights at {weights}\n"
            f"  hf download MiniMaxAI/MiniMax-H3 --revision {MODEL_REVISION} --include 'text_encoder/*'"
        )
    cache = Path(args.cache)
    manifest = json.loads((cache / "manifest.json").read_text())
    keys = sorted(manifest, key=lambda k: int(k) if k.isdigit() else k)
    if args.limit:
        keys = keys[: args.limit]
    (cache / "embeds").mkdir(exist_ok=True)

    dp, tp = (int(v) for v in args.mesh.split(","))
    open_mesh(dp, tp)
    config = TextTowerConfig.from_hf(weights / "config.json", num_layers=args.layers, max_seq_len=args.max_tokens)
    log.info("text tower: %d layers, hidden %d, tp %d", config.num_layers, config.hidden_size, tp)
    t0 = time.perf_counter()
    model = TextTower(config)
    load_text_tower(model, weights)
    log.info("weights loaded in %.0f s", time.perf_counter() - t0)

    longest = max(manifest[k]["num_tokens"] for k in keys)
    padded = args.pad_to or (longest + TILE - 1) // TILE * TILE
    if padded < longest or padded % TILE or padded > config.max_seq_len:
        raise SystemExit(f"--pad-to {padded} must be a multiple of {TILE} in [{longest}, {config.max_seq_len}]")
    mask = build_causal_mask(padded, device=True)
    log.info("padded length %d for %d captions (longest %d)", padded, len(keys), longest)

    scores = []
    t0 = time.perf_counter()
    for i, key in enumerate(keys):
        out_path = cache / "embeds" / f"{key}.npy"
        if out_path.exists() and not args.overwrite and not args.reference_dir:
            continue
        tokens = np.load(cache / "tokens" / f"{key}.npy")
        length = int(tokens.shape[0])
        with no_grad():
            hidden = model(ids_tensor(tokens, padded), mask)
        full = ttml.Sharding.from_tensor(hidden).gather(hidden)
        embeds = np.asarray(full).reshape(padded, config.hidden_size)[:length].astype(ml_dtypes.bfloat16)
        np.save(out_path, embeds)
        if args.reference_dir:
            ref = np.load(Path(args.reference_dir) / f"{key}.npy").astype(np.float32)
            got = embeds.astype(np.float32)
            cos = row_cosine(got, ref)
            scores.append(float(cos.min()))
            log.info(
                "%s: L=%d row-cos min %.5f mean %.5f | flat pcc %.5f | max|d| %.2f",
                key,
                length,
                cos.min(),
                cos.mean(),
                pcc(got, ref),
                np.abs(got - ref).max(),
            )
        if (i + 1) % 10 == 0 or i + 1 == len(keys):
            log.info("[%d/%d] %.1f s/sample", i + 1, len(keys), (time.perf_counter() - t0) / (i + 1))
    if scores:
        log.info(
            "reference row-cos min: worst %.5f mean %.5f over %d samples",
            min(scores),
            float(np.mean(scores)),
            len(scores),
        )
    ttml.close_device_mesh()
    log.info("done: embeds in %s", cache / "embeds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
