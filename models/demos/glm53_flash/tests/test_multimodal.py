# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Image + question prompt through the GLM-5.3-Flash multimodal path (``tt/multimodal.py``) vs the HF
reference ``bringup/references/glm53_flash/MM_shapes_hello`` (``reference/hf_multimodal_reference.py``:
fp32 tower, bf16 text layers, greedy; the same ``encode_multimodal`` prompt on both sides).

    TT_VISIBLE_DEVICES=<chip> pytest models/demos/glm53_flash/tests/test_multimodal.py -s -k streams    # ~3 min
    TT_VISIBLE_DEVICES=<chip> pytest models/demos/glm53_flash/tests/test_multimodal.py -s -k single_chip --timeout 7000
    GLM53_MESH_SHAPE=2,4 GLM53_MODEL_EXPERT_DTYPE=bfp4 GLM53_MODEL_EXPERT_DOWN_DTYPE=bfp4 \\
        TT_VISIBLE_DEVICES=... pytest models/demos/glm53_flash/tests/test_multimodal.py -s -k mesh

1. ``test_multimodal_streams`` (one chip, no text layers): the processor encoding reproduces the
   reference ids; the tower on device vs the HF fp32 features (PCC, bar relative to the HF-bf16 floor as
   in ``test_vision.py``); the stream construction: built from the HF features the image rows equal the HF
   ``inputs_embeds`` bit-exactly and the text rows equal the device text embedding bit-exactly (PCC 1.0 on
   both), built from the TT features the image rows score the tower PCC; all 4 streams identical.
2. ``test_multimodal_single_chip``: ``GLM53Model`` (45 layers streamed through one chip, tensorbin cache)
   generates 16 greedy tokens from the multimodal streams; the first token must equal HF's, the
   position-wise matches / identical prefix are reported (after a divergence the two continue different
   sentences).
3. ``test_multimodal_mesh`` (``GLM53_MESH_SHAPE``): the resident ``GLM53MeshModel`` with the tower
   replicated on the mesh: ``prefill_multimodal`` + traced decode, same comparison.

Env: ``GLM53_MM_REF`` (reference subdirectory, default ``MM_shapes_hello``), ``GLM53_MM_N`` (tokens,
default 16), ``GLM53_MM_LAYERS`` / ``GLM53_MODEL_LAYERS`` (layer subset for mechanics: no token gate),
``GLM53_FULLMODEL_CACHE`` / ``GLM53_MESH_CACHE`` (tensorbin caches as in the text tests),
``GLM53_MODEL_EXPERT_DTYPE`` / ``GLM53_MODEL_EXPERT_DOWN_DTYPE`` (mesh), ``GLM53_MODEL_TRACE_MB``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir, shard_headers
from models.demos.glm53_flash.reference.hf_layer_reference import parse_layers
from models.demos.glm53_flash.tests.utils import pcc
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.embedding import TtEmbedding, padded_rows
from models.demos.glm53_flash.tt.functional_decoder import device_streams_to_host
from models.demos.glm53_flash.tt.mesh import GLMMesh, mesh_device_params
from models.demos.glm53_flash.tt.mesh_model import GLM53MeshModel
from models.demos.glm53_flash.tt.model import GLM53Model
from models.demos.glm53_flash.tt.multimodal import (
    build_vision_tower,
    encode_multimodal,
    image_features,
    image_rows,
    multimodal_streams,
    prefill_multimodal,
)
from models.demos.glm53_flash.tt.vision import GLM53VisionConfig

REF_DIR = Path("bringup/references/glm53_flash") / os.environ.get("GLM53_MM_REF", "MM_shapes_hello")
N_TOKENS = int(os.environ.get("GLM53_MM_N", "16"))
LAYERS = parse_layers(os.environ["GLM53_MM_LAYERS"]) if os.environ.get("GLM53_MM_LAYERS") else None
_cache_env = os.environ.get("GLM53_FULLMODEL_CACHE", "/home/ttuser/ashai/glm53/_cache/fullmodel")
CACHE_ROOT = None if _cache_env.lower() == "none" else Path(_cache_env)
_mesh_cache_env = os.environ.get("GLM53_MESH_CACHE", "/home/ttuser/ashai/glm53/_cache/mesh")
MESH_CACHE_ROOT = None if _mesh_cache_env.lower() == "none" else Path(_mesh_cache_env)
MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "1,1").split(","))
TRACE_MB = int(os.environ.get("GLM53_MODEL_TRACE_MB", "256"))
_DT = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
EXPERT_DTYPE = _DT[os.environ.get("GLM53_MODEL_EXPERT_DTYPE", "bfp8")]
EXPERT_DOWN_DTYPE = _DT[os.environ.get("GLM53_MODEL_EXPERT_DOWN_DTYPE", "bfp8")]
IMAGE_TOKEN_ID = GLM53VisionConfig.image_token_id
STREAM_PCC_BAR = 0.999
TOWER_PCC_BAR = 0.995  # test_vision.py's bar on well-conditioned images ...
CHAOTIC_MARGIN = 0.03  # ... relative to the HF-bf16 floor otherwise


@pytest.fixture(scope="module")
def ref():
    if not (REF_DIR / "inputs_embeds.pt").exists():
        pytest.fail(f"HF reference {REF_DIR} missing: run reference/hf_multimodal_reference.py (see the docstring)")
    prompt = json.loads((REF_DIR / "prompt.json").read_text())
    emb = torch.load(REF_DIR / "inputs_embeds.pt")
    feats = torch.load(REF_DIR / "features.pt")
    gen_file = REF_DIR / "generate.json"
    gen = json.loads(gen_file.read_text()) if gen_file.exists() else None
    return dict(
        prompt=prompt,
        input_ids=emb["input_ids"],
        inputs_embeds=emb["inputs_embeds"],
        mask=emb["image_mask"][0],
        feats=feats["fp32"],
        feats16=feats["bf16_model"],
        pixel_values=feats["pixel_values"],
        grid=feats["image_grid_thw"],
        gen=gen,
    )


def _encode(ref):
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(str(default_snapshot_dir()))
    p = ref["prompt"]
    ids, pv, grid = encode_multimodal(proc, Path(p["image"]), p["question"], think=p["think"])
    assert ids[0].tolist() == p["input_ids"], "the processor encoding differs from the reference's"
    assert torch.equal(grid, ref["grid"]) and torch.allclose(pv, ref["pixel_values"]), "pixel patches differ"
    logger.info(f"prompt {ids.numel()} tokens ({int((ids == IMAGE_TOKEN_ID).sum())} image), grid {grid.tolist()}")
    return proc, ids, pv, grid


def _tower_bar(floor: float) -> float:
    return min(TOWER_PCC_BAR, floor - CHAOTIC_MARGIN)


def _compare_generation(tok, tt_ids: list[int], records: list[dict], hf: dict, strict: bool):
    hf_ids = hf["generated"][: len(tt_ids)]
    matches = sum(int(a == b) for a, b in zip(tt_ids, hf_ids))
    prefix = 0
    while prefix < len(hf_ids) and tt_ids[prefix] == hf_ids[prefix]:
        prefix += 1
    lines = []
    for k, (t, h) in enumerate(zip(tt_ids, hf_ids)):
        top5 = records[k].get("top5_ids") if k < len(records) else None
        extra = f" (HF token in TT top-5: {h in top5})" if top5 else ""
        lines.append(
            f"{k:2d}: TT {t:6d} {tok.decode([t])!r:16s} HF {h:6d} {tok.decode([h])!r:16s} {'==' if t == h else '!='}{extra}"
        )
    logger.info("per step (each side continues its OWN sequence after a divergence):\n" + "\n".join(lines))
    logger.info(
        f"MULTIMODAL_RESULT first token {'==' if tt_ids[0] == hf_ids[0] else '!='} HF, matches {matches}/{len(hf_ids)} "
        f"position-wise, identical prefix {prefix}; TT {tok.decode(tt_ids)!r} vs HF {tok.decode(hf_ids)!r}"
    )
    if strict:
        assert tt_ids[0] == hf_ids[0], f"first generated token differs: TT {tt_ids[0]} vs HF {hf_ids[0]}"
    return matches, prefix


# ---------------------------------------------------------------------------------------------
@pytest.mark.timeout(1500)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_multimodal_streams(mesh_device, ref):
    proc, ids, pv, grid = _encode(ref)
    S = int(ids.numel())
    mask = ref["mask"]
    rows = image_rows(ids, IMAGE_TOKEN_ID)
    assert torch.equal(rows, torch.nonzero(mask).reshape(-1))

    # tower on device (built, run and released inside image_features)
    t0 = time.perf_counter()
    feats_tt = image_features(mesh_device, pv, grid)
    t_tower = time.perf_counter() - t0
    assert [tuple(f.shape) for f in feats_tt] == [tuple(f.shape) for f in ref["feats"]]
    p_tower = pcc(feats_tt[0], ref["feats"][0])
    floor = pcc(ref["feats16"][0], ref["feats"][0])
    logger.info(
        f"tower on device: {t_tower:.1f} s (build + run + release), features {tuple(feats_tt[0].shape)}, "
        f"PCC vs HF fp32 {p_tower:.6f} (HF-bf16 floor {floor:.6f}, bar {_tower_bar(floor):.4f})"
    )

    # stream construction on the embedding only (no text layers)
    snapshot = default_snapshot_dir()
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    emb = TtEmbedding.from_checkpoint(mesh_device, cfg, snapshot)
    stub = SimpleNamespace(embedding=emb, mesh=GLMMesh.from_device(mesh_device), device=mesh_device)
    text_dev = emb(emb.tokens_to_device(ids.reshape(-1), pad_to=padded_rows(S)))
    text_host = device_streams_to_host(text_dev, cfg.hc_mult, S)[0]  # [S, 4, 4096]
    ttnn.deallocate(text_dev)
    hf_emb = ref["inputs_embeds"].float()[0]  # [S, 4096]
    text_mask = ~mask
    results = {}
    for name, feats in (("hf-features", ref["feats"]), ("tt-features", feats_tt)):
        dev = multimodal_streams(stub, ids, feats)
        T_loc = dev.shape[-2]
        assert T_loc == padded_rows(S), (T_loc, S)
        st = device_streams_to_host(dev, cfg.hc_mult, S)[0]  # [S, 4, 4096]
        ttnn.deallocate(dev)
        s0 = st[:, 0]
        for k in range(1, cfg.hc_mult):
            assert torch.equal(st[:, k], s0), f"{name}: stream {k} differs from stream 0"
        text_exact_dev = torch.equal(s0[text_mask], text_host[:, 0][text_mask])
        text_exact_hf = torch.equal(s0[text_mask], hf_emb[text_mask])
        p_text = pcc(s0[text_mask], hf_emb[text_mask])
        p_img = pcc(s0[mask], hf_emb[mask])
        img_exact_hf = torch.equal(s0[mask], hf_emb[mask])
        logger.info(
            f"streams from {name}: text rows PCC {p_text:.6f} (bit-exact vs device text embedding {text_exact_dev}, "
            f"vs HF inputs_embeds {text_exact_hf}), image rows PCC {p_img:.6f} (bit-exact vs HF {img_exact_hf}), "
            f"max |diff| text {(s0[text_mask] - hf_emb[text_mask]).abs().max():.3e} image "
            f"{(s0[mask] - hf_emb[mask]).abs().max():.3e}"
        )
        results[name] = (p_text, p_img, text_exact_dev, text_exact_hf, img_exact_hf)
    p_text, p_img, ex_dev, ex_hf, img_ex = results["hf-features"]
    assert ex_dev and ex_hf, "text rows are not bit-exact against the text embedding path"
    assert img_ex, "image rows built from the HF features are not bit-exact against HF inputs_embeds"
    assert p_text >= STREAM_PCC_BAR and p_img >= STREAM_PCC_BAR, (p_text, p_img)
    p_text, p_img, ex_dev, ex_hf, _ = results["tt-features"]
    assert ex_dev and ex_hf and p_text >= STREAM_PCC_BAR, (ex_dev, ex_hf, p_text)
    assert abs(p_img - p_tower) < 2e-3, "image rows carry something else than the tower features"
    assert p_tower >= _tower_bar(floor), (p_tower, floor)


# ---------------------------------------------------------------------------------------------
@pytest.mark.timeout(7000)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_multimodal_single_chip(mesh_device, ref):
    if ref["gen"] is None:
        pytest.fail(f"{REF_DIR / 'generate.json'} missing (HF generation still running?)")
    hf = ref["gen"]
    if len(hf["generated"]) < N_TOKENS:
        pytest.fail(f"HF reference holds {len(hf['generated'])} tokens (still running?)")
    proc, ids, pv, grid = _encode(ref)
    assert hf["prompt_ids"] == ids[0].tolist(), "the HF generation was made from another prompt"
    tok = proc.tokenizer
    S = int(ids.numel())
    snapshot = default_snapshot_dir()
    t0 = time.perf_counter()
    feats = image_features(mesh_device, pv, grid)  # the tower is released before the text layers stream
    t_tower = time.perf_counter() - t0
    p_tower = pcc(feats[0], ref["feats"][0])
    logger.info(f"tower: {t_tower:.1f} s, PCC vs HF fp32 {p_tower:.6f}")
    model = GLM53Model.from_snapshot(
        mesh_device,
        snapshot,
        headers=shard_headers(snapshot),
        layers=LAYERS,
        max_batch=1,
        max_seq_len=S + N_TOKENS + 32,
        weight_cache_root=CACHE_ROOT,
    )
    full = LAYERS is None
    logger.info(f"model: layers {model.layers[0]}..{model.layers[-1]} ({len(model.layers)}), cache {CACHE_ROOT}")

    def on_token(rec):
        logger.info(
            f"TT step {rec['step']} (pos {rec['position']}): token {rec['token']} {tok.decode([rec['token']])!r} "
            f"(device greedy {rec['device_greedy']}), top-5 {rec['top5_ids']} {rec['top5_logits']}, pass {rec['pass_s']} s"
        )

    streams = multimodal_streams(model, ids, feats)
    t0 = time.perf_counter()
    tt_ids, records = model.generate(ids, N_TOKENS, streams=streams, on_token=on_token)
    logger.info(
        f"TT greedy {N_TOKENS} tokens in {time.perf_counter() - t0:.0f} s: {tt_ids}\nTT text: {tok.decode(tt_ids)!r}"
    )
    logger.info("per-layer timing (seconds, means over the passes):\n" + model.timing_table())
    for rec in records:
        assert rec["device_greedy"] is None or rec["device_greedy"] == rec["token"], f"device argmax differs: {rec}"
    _compare_generation(tok, tt_ids, records, hf, strict=full)


# ---------------------------------------------------------------------------------------------
def _mesh_layers() -> list[int]:
    return parse_layers(os.environ.get("GLM53_MODEL_LAYERS", "0-44"))


@pytest.mark.timeout(10800)
@pytest.mark.parametrize(
    "device_params", [mesh_device_params(*MESH, trace_region_size=TRACE_MB * 1_000_000)], indirect=True
)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_multimodal_mesh(mesh_device, ref):
    if ref["gen"] is None:
        pytest.fail(f"{REF_DIR / 'generate.json'} missing (HF generation still running?)")
    hf = ref["gen"]
    proc, ids, pv, grid = _encode(ref)
    assert hf["prompt_ids"] == ids[0].tolist(), "the HF generation was made from another prompt"
    tok = proc.tokenizer
    S = int(ids.numel())
    layers = _mesh_layers()
    full = layers == list(range(45))
    cfg = GLM53FlashConfig.from_hf_config_json(default_snapshot_dir() / "config.json")
    t0 = time.perf_counter()
    model = GLM53MeshModel(
        mesh_device,
        cfg,
        layers,
        max_batch=1,
        max_seq_len=512,
        weight_cache_path=MESH_CACHE_ROOT,
        expert_dtype=EXPERT_DTYPE,
        expert_down_dtype=EXPERT_DOWN_DTYPE,
    )
    U = model.total_users
    logger.info(
        f"mesh model {MESH}: layers {layers[0]}..{layers[-1]} ({len(layers)}), experts {EXPERT_DTYPE}/{EXPERT_DOWN_DTYPE}, "
        f"{U} user slots, built in {time.perf_counter() - t0:.0f} s"
    )
    trace_id, (_, _, ids_tr) = model.capture(torch.zeros(U, dtype=torch.long), [0] * U)
    model.reset_kda_states()

    t0 = time.perf_counter()
    tower = build_vision_tower(mesh_device)
    t_build = time.perf_counter() - t0
    t0 = time.perf_counter()
    feats = image_features(mesh_device, pv, grid, tower=tower)
    t_tower = time.perf_counter() - t0
    p_tower = pcc(feats[0], ref["feats"][0])
    logger.info(f"tower on the mesh: build {t_build:.1f} s, run {t_tower:.2f} s, PCC vs HF fp32 {p_tower:.6f}")

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
    step_ms = []
    for _ in range(N_TOKENS - 1):
        model.load_ids(torch.tensor([gen[-1]] * U, dtype=torch.long))
        model.set_positions(positions)
        t0 = time.perf_counter()
        model.replay(trace_id)
        nxt = model.ids_to_host(ids_tr)
        step_ms.append((time.perf_counter() - t0) * 1000)
        gen.append(int(nxt[0]))
        positions = [p + 1 for p in positions]
    mean_ms = sum(step_ms[2:] or step_ms) / len(step_ms[2:] or step_ms)
    logger.info(
        f"decode: {len(step_ms)} traced steps, mean {mean_ms:.1f} ms -> {1000 / mean_ms:.1f} tok/s/user, "
        f"{1000 / mean_ms * U:.0f} tok/s aggregate over {U} slots"
    )
    ttnn.release_trace(mesh_device, trace_id)
    _compare_generation(tok, gen, [], hf, strict=full)
