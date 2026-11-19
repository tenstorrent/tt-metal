# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Video (frames -> features -> the prefill streams -> greedy tokens) on a mesh block against the HF reference,
``tests/test_multimodal.py::test_multimodal_mesh``'s shape for a clip:

    # the reference (host, CPU tower + the merged embeddings; --generate N adds the 45-layer greedy tokens, hours)
    python models/demos/glm53_flash/reference/hf_multimodal_reference.py \\
        --image models/demos/glm53_flash/demo/sample_prompts/shapes_hello.png --clip-frames 4 \\
        --question "What happens in this clip? Answer in one sentence." --generate 0 \\
        --out bringup/references/glm53_flash/MMV_shapes_clip4
    # the (2,4) block: every layer with bfp4 experts (the README's multimodal row), or a layer subset
    GLM53_MESH_SHAPE=2,4 GLM53_MODEL_EXPERT_DTYPE=bfp4 GLM53_MODEL_EXPERT_DOWN_DTYPE=bfp4 \\
        TT_VISIBLE_DEVICES=... MESH_DEVICE=P150x8 TT_MESH_GRAPH_DESC_PATH=... \\
        pytest models/demos/glm53_flash/tests/test_multimodal_video_mesh.py -s --timeout 10800
    GLM53_MODEL_LAYERS=0-3 ... (plumbing only: the generation comparison is reported, not asserted)

The clip is ``synthetic_clip`` (4 rotated copies of the demo image at 2 fps -> ``video_grid_thw [[2, 32, 32]]``,
2048 patches, 512 vision tokens in two timestamped runs, 549 tokens). Checks:
1. the processor encoding == the reference ids (``encode_multimodal_video``);
2. the tower on the clip (two 1024-patch windows in ONE tower call: ``vision_batches`` at the 4096 budget) vs the
   HF fp32 features: PCC above ``min(0.995, HF-bf16 floor - 0.03)`` per temporal group;
3. the prefill streams (``multimodal_streams`` on the mesh model) vs HF ``inputs_embeds``: text rows bit-exact, the
   vision rows at the tower's PCC;
4. ``prefill_multimodal`` + the traced decode: first token / 16 tokens vs ``generate.json`` when the reference holds it
   (strict with all 45 layers), else reported.
Expected on a full-layer block: the tower PCC at the image test's level (0.98-0.99 on this flat-shapes clip: the
HF-bf16 floor itself is ~0.99), text rows bit-exact, the first token == HF when ``generate.json`` exists. First
failure candidates: the two-window tower call (``cu_window`` of two 1024-patch windows + the pad window) vs the
single-image run -- compare ``image_features`` per group against the batched call; the timestamp text rows between
the runs (ordinary text: bit-exact or the scatter wrote past a run); the 549-token prompt's 320-row slabs.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

ttnn = pytest.importorskip("ttnn")

from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir
from models.demos.glm53_flash.reference.hf_layer_reference import parse_layers
from models.demos.glm53_flash.tests.test_multimodal import _compare_generation, _tower_bar
from models.demos.glm53_flash.tests.utils import pcc
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.mesh import mesh_device_params
from models.demos.glm53_flash.tt.mesh_model import GLM53MeshModel
from models.demos.glm53_flash.tt.mm_inputs import IMAGE_TOKEN_ID, flatten_grid, vision_batches
from models.demos.glm53_flash.tt.multimodal import (
    build_vision_tower,
    encode_multimodal_video,
    multimodal_streams,
    prefill_multimodal,
    synthetic_clip,
    video_features,
)
from models.demos.glm53_flash.tt.vllm_config import DEFAULT_VISION_BATCH_PATCHES

REF_DIR = Path("bringup/references/glm53_flash") / os.environ.get("GLM53_MMV_REF", "MMV_shapes_clip4")
N_TOKENS = int(os.environ.get("GLM53_MM_N", "16"))
MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "2,4").split(","))
TRACE_MB = int(os.environ.get("GLM53_MODEL_TRACE_MB", "512"))
_DT = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
EXPERT_DTYPE = _DT[os.environ.get("GLM53_MODEL_EXPERT_DTYPE", "bfp8")]
EXPERT_DOWN_DTYPE = _DT[os.environ.get("GLM53_MODEL_EXPERT_DOWN_DTYPE", "bfp8")]
_cache_env = os.environ.get("GLM53_MESH_CACHE", "/home/ttuser/ashai/glm53/_cache/mesh")
CACHE_ROOT = None if _cache_env.lower() == "none" else Path(_cache_env)
SAMPLE_DIR = Path(__file__).resolve().parents[1] / "demo" / "sample_prompts"


@pytest.fixture(scope="module")
def ref():
    if not (REF_DIR / "features.pt").exists():
        pytest.fail(
            f"HF reference {REF_DIR} missing: run reference/hf_multimodal_reference.py --clip-frames 4 (docstring)"
        )
    prompt = json.loads((REF_DIR / "prompt.json").read_text())
    feats = torch.load(REF_DIR / "features.pt")
    emb = torch.load(REF_DIR / "inputs_embeds.pt")
    gen_file = REF_DIR / "generate.json"
    return dict(
        prompt=prompt,
        feats=feats["fp32"],
        feats16=feats["bf16_model"],
        frames=feats["frames"],
        fps=float(feats.get("fps", 2.0)),
        video_grid_thw=feats["video_grid_thw"],
        pixel_values_videos=feats["pixel_values_videos"],
        input_ids=emb["input_ids"],
        inputs_embeds=emb["inputs_embeds"],
        mask=emb["image_mask"][0],
        gen=json.loads(gen_file.read_text()) if gen_file.exists() else None,
    )


def _encode(ref):
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(str(default_snapshot_dir()))
    p = ref["prompt"]
    frames = ref["frames"].numpy() if ref["frames"] is not None else synthetic_clip(SAMPLE_DIR / "shapes_hello.png", 4)
    ids, pvv, vg = encode_multimodal_video(proc, frames, p["question"], fps=ref["fps"], think=p["think"])
    assert ids[0].tolist() == p["input_ids"], "the processor encoding differs from the reference's"
    assert torch.equal(vg, ref["video_grid_thw"]) and torch.allclose(
        pvv, ref["pixel_values_videos"]
    ), "video patches differ"
    logger.info(f"clip prompt {ids.numel()} tokens ({int((ids == IMAGE_TOKEN_ID).sum())} vision), grid {vg.tolist()}")
    return proc, ids, pvv, vg


@pytest.mark.timeout(10800)
@pytest.mark.parametrize(
    "device_params", [mesh_device_params(*MESH, trace_region_size=TRACE_MB * 1_000_000)], indirect=True
)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_multimodal_video_mesh(mesh_device, ref):
    proc, ids, pvv, vg = _encode(ref)
    tok = proc.tokenizer
    S = int(ids.numel())
    layers = parse_layers(os.environ.get("GLM53_MODEL_LAYERS", "0-44"))
    full = layers == list(range(45))
    cfg = GLM53FlashConfig.from_hf_config_json(default_snapshot_dir() / "config.json")
    t0 = time.perf_counter()
    model = GLM53MeshModel(
        mesh_device,
        cfg,
        layers,
        max_batch=1,
        max_seq_len=1024,
        weight_cache_path=CACHE_ROOT,
        expert_dtype=EXPERT_DTYPE,
        expert_down_dtype=EXPERT_DOWN_DTYPE,
    )
    U = model.total_users
    logger.info(
        f"mesh model {MESH}: layers {layers[0]}..{layers[-1]} ({len(layers)}), {U} user slots, built in {time.perf_counter() - t0:.0f} s"
    )
    trace_id, (_, _, ids_tr) = model.capture(torch.zeros(U, dtype=torch.long), [0] * U)
    model.reset_kda_states()

    # 2. the tower: the clip's two temporal groups in one call (two windows of 1024 patches)
    t0 = time.perf_counter()
    tower = build_vision_tower(mesh_device)
    t_build = time.perf_counter() - t0
    assert vision_batches(flatten_grid(vg), DEFAULT_VISION_BATCH_PATCHES) == [(0, int(vg[0, 0]))]
    t0 = time.perf_counter()
    feats = video_features(mesh_device, pvv, vg, tower=tower)
    t_tower = time.perf_counter() - t0
    assert len(feats) == 1 and feats[0].shape == ref["feats"][0].shape
    per_group = int(vg[0, 1] * vg[0, 2]) // 4
    for g in range(int(vg[0, 0])):
        sl = slice(g * per_group, (g + 1) * per_group)
        p32 = pcc(feats[0][sl], ref["feats"][0][sl])
        floor = pcc(ref["feats16"][0][sl], ref["feats"][0][sl])
        logger.info(
            f"group {g}: tower PCC vs HF fp32 {p32:.6f} (HF bf16 floor {floor:.6f}, bar {_tower_bar(floor):.4f})"
        )
        assert p32 > _tower_bar(floor), f"group {g}: tower PCC {p32}"
    p_all = pcc(feats[0], ref["feats"][0])
    logger.info(
        f"tower on the mesh: build {t_build:.1f} s, run {t_tower * 1000:.0f} ms for {int(pvv.shape[0])} patches, PCC {p_all:.6f}"
    )

    # 3. the streams vs HF inputs_embeds: text rows bit-exact, vision rows at the tower's PCC
    streams = multimodal_streams(model, ids, feats)
    host = (
        model.mesh.to_host(streams, rows=2 if model.mesh.rows > 1 else None)
        .float()
        .reshape(-1, 4 * cfg.hidden_size)[:S]
    )
    ttnn.deallocate(streams)
    ref_emb = ref["inputs_embeds"][0].float().repeat(1, 4)
    mask = ref["mask"]
    assert torch.equal(
        host[~mask], ref_emb[~mask]
    ), "text rows (the timestamps between the frame runs included) must be bit-exact"
    p_vis = pcc(host[mask], ref_emb[mask])
    logger.info(f"streams: text rows bit-exact, vision rows PCC {p_vis:.6f}")
    assert p_vis > _tower_bar(pcc(ref["feats16"][0], ref["feats"][0]))

    # 4. the prefill + traced decode
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    streams, logits_last, owner_row = prefill_multimodal(model, ids, feats, user=0)
    first = model.first_token(logits_last, owner_row)
    ttft = time.perf_counter() - t0
    ttnn.deallocate(streams)
    ttnn.deallocate(logits_last)
    logger.info(
        f"prefill {S} tokens: TTFT {ttft * 1000:.0f} ms (without the tower), first token {first} {tok.decode([first])!r}"
    )
    gen = [first]
    positions = [S] * U
    for _ in range(N_TOKENS - 1):
        model.load_ids(torch.tensor([gen[-1]] * U, dtype=torch.long))
        model.set_positions(positions)
        model.replay(trace_id)
        gen.append(int(model.ids_to_host(ids_tr)[0]))
        positions = [p + 1 for p in positions]
    ttnn.release_trace(mesh_device, trace_id)
    logger.info(f"VIDEO_RESULT {len(layers)} layers: {tok.decode(gen)!r}")
    if ref["gen"] is not None:
        _compare_generation(tok, gen, [], ref["gen"], strict=full)
    else:
        logger.warning(f"{REF_DIR}: no generate.json (run the reference with --generate 16 for the token comparison)")
