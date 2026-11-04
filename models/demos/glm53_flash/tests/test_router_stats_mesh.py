# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Routing statistics of the routed experts over a text corpus (measurement tool, not a CI gate): prefills every
corpus file through the mesh model in ``GLM53_STATS_CHUNK``-token chunks with the router hooked, and saves per MoE
layer the ``[tokens, k]`` top-k expert ids to ``GLM53_STATS_OUT`` (torch file). ``tests/tools/router_stats.py``
turns that into per-EP-row load statistics under the contiguous placement and its alternatives.

    GLM53_MESH_SHAPE=8,4 GLM53_STATS_CORPUS=<dir of .md/.txt/.py files> GLM53_STATS_OUT=<file.pt> pytest ... -s

The router hook reads ``idx`` back on every chip of mesh column 0: the prefill spreads a chunk over the rows as
tile-aligned slabs (``prefill_chunked``), so the rows' slabs concatenated in row order are the chunk's tokens.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.mesh import mesh_device_params
from models.demos.glm53_flash.tt.mesh_model import GLM53MeshModel
from models.demos.glm53_flash.tt.moe import TtRouter

MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "8,4").split(","))
CHUNK = int(os.environ.get("GLM53_STATS_CHUNK", "2048"))
MAX_TOKENS = int(os.environ.get("GLM53_STATS_MAX_TOKENS", "40000"))  # corpus cap (prefill time)
B = int(os.environ.get("GLM53_STATS_B", "4"))  # user slots per row (the partitioned cache needs one per column)
TRACE_MB = int(os.environ.get("GLM53_MODEL_TRACE_MB", "512"))
_DT = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
EXPERT_DTYPE = _DT[os.environ.get("GLM53_MODEL_EXPERT_DTYPE", "bfp8")]
EXPERT_DOWN_DTYPE = _DT[os.environ.get("GLM53_MODEL_EXPERT_DOWN_DTYPE", "bfp8")]
_cache_env = os.environ.get("GLM53_MESH_CACHE", "/home/ttuser/ashai/glm53/_cache/mesh")
CACHE_ROOT = None if _cache_env.lower() == "none" else Path(_cache_env)
CORPUS = os.environ.get("GLM53_STATS_CORPUS", "")
OUT = os.environ.get("GLM53_STATS_OUT", "/tmp/glm53_router_stats.pt")


def _layer_range() -> list[int]:
    spec = os.environ.get("GLM53_MODEL_LAYERS", "0-44")
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(v) for v in spec.split(",")]


def _corpus_ids(tok, max_tokens: int) -> list[torch.Tensor]:
    """Every corpus file tokenized (no chat template: plain text statistics), truncated to the token cap."""
    files = sorted(p for p in Path(CORPUS).rglob("*") if p.suffix in (".md", ".txt", ".py", ".rst"))
    if not files:
        raise ValueError(f"no corpus files under {CORPUS!r} (GLM53_STATS_CORPUS)")
    out, total = [], 0
    for p in files:
        text = p.read_text(errors="replace")
        ids = tok(text, return_tensors="pt", add_special_tokens=True)["input_ids"][0]
        if ids.numel() < 64:
            continue
        ids = ids[: max_tokens - total]
        out.append(ids)
        total += int(ids.numel())
        if total >= max_tokens:
            break
    logger.info(f"corpus: {len(out)} files, {total} tokens (cap {max_tokens})")
    return out


@pytest.mark.timeout(5400)
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH, trace_region_size=TRACE_MB << 20)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_router_stats(mesh_device):
    from transformers import AutoTokenizer

    snapshot = default_snapshot_dir()
    tok = AutoTokenizer.from_pretrained(str(snapshot))
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    layers = _layer_range()
    rows, cols = MESH
    model = GLM53MeshModel(
        mesh_device,
        cfg,
        layers,
        max_batch=B,
        max_seq_len=CHUNK + 256,
        weight_cache_path=CACHE_ROOT,
        expert_dtype=EXPERT_DTYPE,
        expert_down_dtype=EXPERT_DOWN_DTYPE,
    )
    # the hook: every MoE layer's router records its prefill top-k ids per row slab (column 0 of every row)
    # the mesh model's layers are FunctionalDecoder wrappers: the MoE sublayer is ``.layer.mlp`` (TtMoE, ``.router``)
    routers = {}
    for i, lay in zip(layers, model.layers):
        mlp = getattr(getattr(lay, "layer", lay), "mlp", None)
        router = getattr(mlp, "router", None)
        if isinstance(router, TtRouter):
            routers[i] = router
    logger.info(f"router hook on {len(routers)} MoE layers: {sorted(routers)}")
    records: dict[int, list[torch.Tensor]] = {i: [] for i in routers}
    chunk_tokens = [0]
    k = int(cfg.num_experts_per_tok)

    def hooked(layer: int, orig):
        def forward(x, **kw):
            out = orig(x, **kw)
            idx, dense = out[1], out[2]
            if kw.get("prefill") and (idx is not None or dense is not None):
                T = chunk_tokens[0]
                per_row = []
                chips = model.mesh.device_tensors(idx if idx is not None else dense)
                for r in range(rows):
                    t = ttnn.to_torch(chips[r * cols])
                    t = t.float() if t.dtype in (torch.float32, torch.bfloat16) else t.to(torch.int64).float()
                    t = t.reshape(-1, t.shape[-1])[: T // rows]
                    if idx is None:  # the dense [T, E] routing matrix: its k non-zero columns per token are the ids
                        t = torch.topk(t.abs(), k, dim=-1).indices.float()
                    per_row.append(t)
                records[layer].append(torch.cat(per_row, dim=0).to(torch.int16))
            return out

        return forward

    originals = {}
    for i, r in routers.items():
        originals[i] = r.forward
        r.forward = hooked(i, r.forward)
    try:
        prompts = _corpus_ids(tok, MAX_TOKENS)
        done = 0
        t_all = time.perf_counter()
        for n, ids in enumerate(prompts):
            user = n % B
            for s0 in range(0, int(ids.numel()), CHUNK):
                piece = ids[s0 : s0 + CHUNK]
                if piece.numel() < 32 * rows:
                    break
                piece = piece[: (piece.numel() // (32 * rows)) * (32 * rows)]
                chunk_tokens[0] = int(piece.numel())
                t0 = time.perf_counter()
                out = model.prefill(piece, user=user, start=0)
                ttnn.synchronize_device(mesh_device)
                for t in out[:2]:
                    if t is not None:
                        ttnn.deallocate(t)
                done += int(piece.numel())
                logger.info(
                    f"file {n} chunk {s0}: {piece.numel()} tokens in {time.perf_counter() - t0:.1f} s ({done} so far)"
                )
        stats = {i: torch.cat(v, dim=0) for i, v in records.items() if v}
        torch.save(
            {"layers": stats, "E": cfg.n_routed_experts, "k": cfg.num_experts_per_tok, "rows": rows, "cols": cols}, OUT
        )
        n_tok = next(iter(stats.values())).shape[0] if stats else 0
        logger.info(
            f"ROUTER STATS: {len(stats)} MoE layers x {n_tok} tokens x k={cfg.num_experts_per_tok} -> {OUT} ({time.perf_counter() - t_all:.0f} s)"
        )
        assert n_tok > 0
    finally:
        for i, r in routers.items():
            r.forward = originals[i]
