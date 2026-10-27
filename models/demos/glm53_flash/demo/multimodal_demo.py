# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash multimodal demo: one image + question through the resident mesh model (vision tower on
device, image features in the prefill streams, traced greedy decode), TTFT and decode tok/s like
``text_demo.py``.

    # rows-6-7 (2,4) block of the Galaxy (bfp4 experts fit a 2x4; bfp8 needs the 8x4)
    GLM53_MESH_SHAPE=2,4 GLM53_MODEL_EXPERT_DTYPE=bfp4 GLM53_MODEL_EXPERT_DOWN_DTYPE=bfp4 \\
      TT_VISIBLE_DEVICES=25,24,29,28,21,20,17,16 TT_MESH_GRAPH_DESC_PATH=.../p150_x8_mesh_graph_descriptor.textproto MESH_DEVICE=P150x8 \\
      pytest models/demos/glm53_flash/demo/multimodal_demo.py -s
    # plumbing check on one chip (a layer subset; the answer is meaningless)
    GLM53_MESH_SHAPE=1,1 GLM53_DEMO_LAYERS=0-5 TT_VISIBLE_DEVICES=<chip> pytest models/demos/glm53_flash/demo/multimodal_demo.py -s

Flow: model build (layers resident, decode trace captured) -> vision tower built on the mesh (replicated)
-> ``encode_multimodal`` (processor + chat template; ``</think>`` appended so the answer is immediate,
``GLM53_MM_THINK=1`` keeps the reasoning trace) -> tower forward (image features) -> ``prefill_multimodal``
(features scattered into the prefill streams at the image placeholder rows) -> first token -> traced decode
replayed until an EOS or ``GLM53_MM_MAX_TOKENS`` -> the answer, TTFT (tower + prefill + first token), decode
tok/s (per user and aggregate over the ``rows x 1`` slots the step runs) and ``generated/glm53_flash_demo/mm_<tag>.json``.

Video: ``GLM53_MM_VIDEO=<directory of frame images>`` or ``GLM53_MM_CLIP_FRAMES=N`` (a synthetic clip of N
rotated copies of the image, ``tt/multimodal.py::synthetic_clip``) runs the clip through the same path
(``encode_multimodal_video`` -> ``video_features``: one tower window per 2-frame temporal group, the features
scattered at the frames' image-token runs); ``GLM53_MM_FPS`` (2.0).

Knobs: ``GLM53_MM_IMAGE`` (default ``sample_prompts/shapes_hello.png``: a red circle, a blue square, a green
triangle and the word HELLO), ``GLM53_MM_QUESTION``, ``GLM53_MM_MAX_TOKENS`` (default 64), ``GLM53_MESH_SHAPE``,
``GLM53_DEMO_LAYERS``, ``GLM53_MODEL_EXPERT_DTYPE`` / ``GLM53_MODEL_EXPERT_DOWN_DTYPE``, ``GLM53_MODEL_TRACE_MB``,
``GLM53_MESH_CACHE``, ``GLM53_DEMO_WARMUP``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.demo.text_demo import (
    CACHE_ROOT,
    EXPERT_DOWN_DTYPE,
    EXPERT_DTYPE,
    MESH,
    TRACE_MB,
    WARMUP,
    _layers,
    eos_ids,
)
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.mesh import mesh_device_params
from models.demos.glm53_flash.tt.mesh_model import GLM53MeshModel
from models.demos.glm53_flash.tt.multimodal import (
    build_vision_tower,
    encode_multimodal,
    encode_multimodal_video,
    image_features,
    load_frames,
    prefill_multimodal,
    synthetic_clip,
    video_features,
)
from models.demos.glm53_flash.tt.vision import GLM53VisionConfig

SAMPLE_DIR = Path(__file__).parent / "sample_prompts"
IMAGE = Path(os.environ.get("GLM53_MM_IMAGE", SAMPLE_DIR / "shapes_hello.png"))
QUESTION = os.environ.get("GLM53_MM_QUESTION", "What shapes and what word are in this image? Answer in one sentence.")
MAX_TOKENS = int(os.environ.get("GLM53_MM_MAX_TOKENS", "64"))
THINK = os.environ.get("GLM53_MM_THINK", "0") == "1"
VIDEO_DIR = os.environ.get("GLM53_MM_VIDEO")  # a directory of frame images
CLIP_FRAMES = int(os.environ.get("GLM53_MM_CLIP_FRAMES", "0"))  # a synthetic clip of the image instead
FPS = float(os.environ.get("GLM53_MM_FPS", "2.0"))
VIDEO = VIDEO_DIR is not None or CLIP_FRAMES > 0


@pytest.mark.timeout(14400)
@pytest.mark.parametrize(
    "device_params", [mesh_device_params(*MESH, trace_region_size=TRACE_MB * 1_000_000)], indirect=True
)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_multimodal_demo(mesh_device):
    from transformers import AutoProcessor

    rows, cols = MESH
    layers = _layers()
    snapshot = default_snapshot_dir()
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    proc = AutoProcessor.from_pretrained(str(snapshot))
    tok = proc.tokenizer
    stop = eos_ids(snapshot, tok)
    if VIDEO:
        frames = load_frames(VIDEO_DIR) if VIDEO_DIR else synthetic_clip(IMAGE, CLIP_FRAMES)
        question = os.environ.get("GLM53_MM_QUESTION", "What happens in this clip? Answer in one sentence.")
        ids, pv, grid = encode_multimodal_video(proc, frames, question, fps=FPS, think=THINK)
        media = f"video {VIDEO_DIR or f'synthetic clip of {IMAGE}'} ({frames.shape[0]} frames, {FPS} fps)"
    else:
        question = QUESTION
        ids, pv, grid = encode_multimodal(proc, IMAGE, question, think=THINK)
        media = f"image {IMAGE}"
    features = video_features if VIDEO else image_features
    S = int(ids.numel())
    n_img = int((ids == GLM53VisionConfig.image_token_id).sum())
    logger.info(f"{media} ({grid.tolist()} -> {n_img} vision tokens), question {question!r}: {S} prompt tokens")

    t0 = time.perf_counter()
    model = GLM53MeshModel(
        mesh_device,
        cfg,
        layers,
        max_batch=1,
        max_seq_len=max(1024, S + MAX_TOKENS + 64),
        weight_cache_path=CACHE_ROOT,
        expert_dtype=EXPERT_DTYPE,
        expert_down_dtype=EXPERT_DOWN_DTYPE,
    )
    U = model.total_users
    t_build = time.perf_counter() - t0
    logger.info(
        f"model built in {t_build:.0f} s ({len(layers)} layers resident, experts {EXPERT_DTYPE}/{EXPERT_DOWN_DTYPE})"
    )
    trace_id, (_, _, ids_tr) = model.capture(torch.zeros(U, dtype=torch.long), [0] * U)
    model.reset_kda_states()
    t0 = time.perf_counter()
    tower = build_vision_tower(mesh_device, snapshot)
    t_tower_build = time.perf_counter() - t0
    logger.info(f"vision tower built on the mesh in {t_tower_build:.1f} s")

    # warm-up (compiles the tower and prefill programs); the state of slot 0 is overwritten by the timed run
    feats = features(mesh_device, pv, grid, tower=tower)
    streams, logits_last, _ = prefill_multimodal(model, ids, feats, user=0)
    ttnn.deallocate(streams)
    ttnn.deallocate(logits_last)
    ttnn.synchronize_device(mesh_device)

    # timed: tower -> prefill -> first token
    t0 = time.perf_counter()
    feats = features(mesh_device, pv, grid, tower=tower)
    t_feat = time.perf_counter() - t0
    streams, logits_last, owner_row = prefill_multimodal(model, ids, feats, user=0)
    first = model.first_token(logits_last, owner_row)
    ttft_ms = (time.perf_counter() - t0) * 1000
    ttnn.deallocate(streams)
    ttnn.deallocate(logits_last)
    logger.info(
        f"TTFT {ttft_ms:.0f} ms (tower {t_feat * 1000:.0f} ms + prefill {S} tokens), first token {first} {tok.decode([first])!r}"
    )

    gen = [first]
    positions = [S] * U
    step_ms = []
    for _ in range(MAX_TOKENS - 1):
        model.load_ids(torch.tensor([gen[-1]] * U, dtype=torch.long))
        model.set_positions(positions)
        t0 = time.perf_counter()
        model.replay(trace_id)
        nxt = model.ids_to_host(ids_tr)
        step_ms.append((time.perf_counter() - t0) * 1000)
        tid = int(nxt[0])
        gen.append(tid)
        positions = [p + 1 for p in positions]
        if tid in stop:
            break
    rate = step_ms[WARMUP:] or step_ms
    mean_ms = sum(rate) / len(rate)
    tok_s_user = 1000.0 / mean_ms
    text = tok.decode([t for t in gen if t not in stop], skip_special_tokens=True)
    logger.info(f"=== ANSWER ({len(gen)} tokens) ===\nQUESTION: {question}\nANSWER: {text}")
    logger.info(
        f"PERF GLM-5.3-Flash multimodal {len(layers)} layers on {rows}x{cols}: ISL {S} ({n_img} vision tokens, {media}), "
        f"TTFT {ttft_ms:.0f} ms (tower {t_feat * 1000:.0f} ms); decode step {mean_ms:.2f} ms -> {tok_s_user:.1f} tok/s/user, "
        f"{tok_s_user * U:.0f} tok/s aggregate over {U} slots ({len(rate)} steps, {WARMUP} warm-up)"
    )
    out_dir = Path("generated/glm53_flash_demo")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"mm{'v' if VIDEO else ''}_{rows}x{cols}_{IMAGE.stem}_{time.strftime('%Y%m%d_%H%M%S')}"
    (out_dir / f"{tag}.json").write_text(
        json.dumps(
            {
                "model": "GLM-5.3-Flash",
                "layers": len(layers),
                "mesh": [rows, cols],
                "expert_dtype": str(EXPERT_DTYPE),
                "image": str(IMAGE),
                "video": (VIDEO_DIR or f"synthetic clip x{CLIP_FRAMES}") if VIDEO else None,
                "question": question,
                "think": THINK,
                "isl": S,
                "image_tokens": n_img,
                "build_s": t_build,
                "tower_build_s": t_tower_build,
                "tower_ms": t_feat * 1000,
                "ttft_ms": ttft_ms,
                "decode_step_ms": step_ms,
                "decode_tok_s_user": tok_s_user,
                "decode_tok_s_aggregate": tok_s_user * U,
                "generated_ids": gen,
                "text": text,
            },
            indent=1,
        )
    )
    logger.info(f"metrics written to {out_dir / (tag + '.json')}")
    ttnn.release_trace(mesh_device, trace_id)
    assert len(gen) > 1, "no tokens generated"
