# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Image prompts through metal_2's whole-mesh prefill (``DPPrefill(image_rows=...)``, what ``demo/prefill_server.py``
serves).

``test_d_p_images`` (metal_2's runtime with images, 8 x 4, where its expert weight cache is) checks, exactly, that

* the embedding reads written vision rows (and the delimiters, text and pad rows) through the device id reorder,
* the MoE's image pass equals the layer built with ``gate.bias_vl`` as its bias (the image twin of the head's weights)
  row for row, and the text pass on the other rows,

then prefills a prompt of real text around two images (real vision rows, the ``vision_rows/*.pt`` the decode server
saved, when this host has them, else random rows of their scale) and checks the entries are finite. With
``D_P_IMAGE_STATES`` from ``test_kv_only_image_states`` it runs that file's prompt and compares every entry with it by
PCC (``D_P_MIN_PCC``, default 0.99).

``test_kv_only_image_states``: this model's traced KV-only prefill (eight ``1 x 4`` stages, ``image_rows``) of that
prompt, saved to ``D_P_IMAGE_STATES`` (layers 0-20's ``kv_tail`` / ``compressed_kv`` / ``idx_keys``, the ids and the
rows). The two never share a process (``test_d_p_bridge.py``). NOTE: at 1d26f6c5615 this model's prefill SDPA
(``deepseek_v4_prefill_sdpa``) fails its compile-time argument check (sparse_sdpa's split-K, 6d0edbc14a7, grew
``compute_ct_arg``), as ``test_d_p_bridge.py::test_d_p_vs_kv_only`` does.

Run (ttnn venv, repo root)::

    pytest -s models/experimental/deepseek_v41_flash/tests/prefill/test_d_p_images.py::test_d_p_images
    D_P_IMAGE_STATES=/tmp/d_p_image_states.pt DEEPSEEK_V4_CACHE_DIR=../cache \\
      pytest -s models/experimental/deepseek_v41_flash/tests/prefill/test_d_p_images.py::test_kv_only_image_states
"""

from __future__ import annotations

import copy
import glob
import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn

STATES = Path(os.environ.get("D_P_IMAGE_STATES", "/tmp/d_p_image_states.pt"))  # the KV-only oracle
PREFILL_LEN = int(os.environ.get("D_P_IMAGE_LEN", 8192))
MIN_PCC = float(os.environ.get("D_P_MIN_PCC", 0.99))
IMAGE_ROWS = 4096  # rows past the vocabulary on both sides: 3 delimiters, then the vision rows
FIRST_IMAGE_ROW = 3
_TP_SIZE = 4
_TEXT = (
    "The river locks were built in the last century to carry barges past the rapids. Each lock is a stone chamber "
    "with a gate at either end; a keeper opens the sluices, the water levels, and the gate swings. In winter the "
    "keepers grease the hinges, replace the seals and log every passage in a ledger kept in the lock house. "
)


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    den = a.norm() * b.norm()
    return float((a @ b) / den) if den > 0 else float(torch.equal(a, b))


def _vision_rows(hidden: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Two images' (types, IMAGE rows): the decode server's saved ones when present, else random of their scale."""
    from models.experimental.deepseek_v41_flash.tt.config import CACHE_DIR
    from models.experimental.deepseek_v41_flash.tt.vision import IMAGE, IMAGE_END, IMAGE_NEW_LINE, IMAGE_START

    where = os.environ.get("D_P_VISION_ROWS", os.path.join(CACHE_DIR, "vision_rows"))
    saved = sorted(glob.glob(os.path.join(where, "*.pt")), key=os.path.getsize)
    picks = [saved[len(saved) // 2], saved[-1]] if len(saved) >= 2 else []
    out = []
    for path in picks:
        d = torch.load(path)
        if d["rows"].shape[1] == hidden:
            out.append((d["types"].long(), d["rows"].to(torch.bfloat16)))
    if len(out) == 2:
        logger.info(f"real vision rows: {[tuple(r.shape) for _, r in out]}")
        return out
    gen = torch.Generator().manual_seed(7)
    for h, w in ((16, 16), (24, 30)):
        types = torch.tensor([IMAGE_START] + ([IMAGE] * w + [IMAGE_NEW_LINE]) * h + [IMAGE_END])
        out.append((types, (0.04 * torch.randn(h * w, hidden, generator=gen)).to(torch.bfloat16)))
    logger.info("random vision rows (no saved ones)")
    return out


def _image_prompt(tokenizer, vocab: int, hidden: int):
    """(ids [T] int64, rows [n, hidden] bf16, their ids [n]): text, image A, text, image B, text; T = PREFILL_LEN."""
    from models.experimental.deepseek_v41_flash.tt.vision import span_ids

    text = tokenizer.encode(_TEXT * 400, add_special_tokens=False)
    images = _vision_rows(hidden)
    spans, rows, row_ids, next_row = [], [], [], 0
    for types, image_rows in images:
        slots = torch.arange(next_row, next_row + image_rows.shape[0])
        next_row += image_rows.shape[0]
        spans.append(span_ids(types, slots, vocab))
        rows.append(image_rows)
        row_ids.append(vocab + FIRST_IMAGE_ROW + slots)
    n_text = PREFILL_LEN - sum(s.numel() for s in spans)
    cuts = [n_text * 3 // 8, n_text * 3 // 8 + n_text // 4]
    pieces = [text[: cuts[0]], spans[0].tolist(), text[cuts[0] : cuts[1]], spans[1].tolist(), text[cuts[1] : n_text]]
    ids = torch.tensor([t for piece in pieces for t in piece], dtype=torch.int64)
    assert ids.numel() == PREFILL_LEN, ids.numel()
    assert FIRST_IMAGE_ROW + next_row <= IMAGE_ROWS
    return ids, torch.cat(rows), torch.cat(row_ids)


@pytest.mark.timeout(6 * 3600)
@torch.no_grad()
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY}], indirect=True, ids=["fabric_2d"]
)
@pytest.mark.parametrize("mesh_device", [(8, _TP_SIZE)], indirect=True, ids=["galaxy_8x4"])
def test_kv_only_image_states(mesh_device):
    from transformers import AutoTokenizer

    from models.experimental.deepseek_v41_flash.tests.decode.test_full_model_decode_demo import _build_rope
    from models.experimental.deepseek_v41_flash.tt.config import CACHE_DIR, DEFAULT_MODEL_DIR, load_config
    from models.experimental.deepseek_v41_flash.tt.engram_lookup import EngramHostLookup, load_engram_table
    from models.experimental.deepseek_v41_flash.tt.pipeline_parallel_prefill.model import DeepSeekV41PrefillModel
    from models.experimental.deepseek_v41_flash.tt.prefill.d_p_bridge import _delimiter_rows
    from models.experimental.deepseek_v41_flash.tt.vision import write_embedding_rows
    from models.experimental.deepseek_v4_flash.tt.model import plan_layer_placement
    from models.experimental.deepseek_v4_flash.tt.system_config import load_system_config, set_active_system_config
    from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache
    from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader

    loader = DeepseekV4WeightLoader(os.environ.get("D_P_MODEL_DIR", DEFAULT_MODEL_DIR))
    config = load_config(loader.snapshot_dir)
    tok = AutoTokenizer.from_pretrained(loader.snapshot_dir)
    vocab, hidden = config.vocab_size, config.hidden_size
    ids, rows, row_ids = _image_prompt(tok, vocab, hidden)
    T = ids.numel()
    logger.info(f"prompt: {T} tokens, {int((ids >= vocab).sum())} of them in image spans, {rows.shape[0]} vision rows")

    set_active_system_config(load_system_config(mesh_device=mesh_device))
    num_layers = max(config.kv_source_layer_ids) + 1
    placement = plan_layer_placement(num_layers, tuple(mesh_device.shape)[0], 1)
    submeshes = [
        mesh_device.create_submesh(ttnn.MeshShape(1, _TP_SIZE), ttnn.MeshCoordinate(k, 0))
        for k in range(placement[-1] + 1)
    ]
    layer_devices = [submeshes[k] for k in placement]
    rope = _build_rope(config, loader.snapshot_dir, T)
    engram_ids = [i for i in config.engram_layer_ids if i < num_layers]
    tables = {i: load_engram_table(loader, i) for i in engram_ids}
    engram_lookup = EngramHostLookup(config, loader.snapshot_dir, tables, batch=1, max_seq=T, tokenizer=tok)
    model = DeepSeekV41PrefillModel(
        config,
        loader,
        layer_devices[0],
        rope,
        num_layers=num_layers,
        cache=WeightCache(CACHE_DIR) if CACHE_DIR else None,
        weight_dtype=ttnn.bfloat8_b,
        expert_dtype=ttnn.bfloat4_b,
        tp_size=_TP_SIZE,
        layer_devices=layer_devices,
        engram_lookup=engram_lookup,
        kv_only=True,
        image_rows=IMAGE_ROWS,
    )
    write_embedding_rows(
        model.embedding_weight, _delimiter_rows(str(loader.snapshot_dir)), [vocab, vocab + 1, vocab + 2], vocab
    )
    write_embedding_rows(model.embedding_weight, rows, row_ids.tolist(), vocab)
    model.synchronize("uploads")
    model.prepare_traced_prefill(T, 1024)
    t0 = time.perf_counter()
    _, states = model.prefill_traced(ids.unsqueeze(0))
    logger.info(f"KV-only prefill of {T} tokens in {time.perf_counter() - t0:.2f} s")

    layers = {}
    for i, state in enumerate(states):
        fields = [("kv_tail", state.kv_tail)] if i != num_layers - 1 else []  # (layer 20: metal_2 never writes it)
        if state.compressed_kv is not None:
            fields += [("compressed_kv", state.compressed_kv), ("idx_keys", state.idx_keys)]
        layers[i] = {
            name: model.to_host(t, model.layer_devices[i]).reshape(-1, t.shape[-1]).to(torch.bfloat16)
            for name, t in fields
        }
        for name, t in layers[i].items():
            assert torch.isfinite(t.float()).all(), (i, name)
    model.free_traced_states(states)
    model.release_traced_prefill()
    torch.save({"ids": ids, "rows": rows, "row_ids": row_ids, "seq_len": T, "layers": layers}, STATES)
    logger.info(f"saved layers {sorted(layers)} of {T} tokens to {STATES}")


class _BiasVl:
    """A checkpoint whose ``gate.bias`` reads ``gate.bias_vl``: the MoE layer a router of the image bias builds."""

    def __init__(self, ck):
        self._ck = ck

    def weight(self, name, *args, **kwargs):
        if name.endswith("ffn.gate.bias"):
            name += "_vl"
        return self._ck.weight(name, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._ck, name)


def _check_embedding(d_p, rows, row_ids) -> None:
    """The device id reorder + embedding of a chunk of text, delimiters, image rows and pad, against the table."""
    model, V = d_p.runtime.model, d_p.vocab
    c, valid = d_p.chunk, d_p.chunk - 96
    emb = model.ck.get("embed.weight")
    ids = torch.randint(0, V, (c,), generator=torch.Generator().manual_seed(3))
    ids[100:103] = torch.tensor([V, V + 1, V + 2])
    take = torch.arange(0, rows.shape[0], 7)[: c // 4]
    ids[200 : 200 + take.numel()] = row_ids[take]
    from models.demos.deepseek_v41_flash_d_p.tt.cores import max_grid

    inp = d_p.runtime.make_chunk_input(ids[:valid].tolist(), 0)
    dev = model.reorder_ids(inp, 0, valid)
    e = ttnn.bringup.embedding(
        dev,
        model.embed_w,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        max_grid=max_grid(),
    )
    sp, tp = tuple(model.mesh.shape)
    parts = [ttnn.to_torch(t).reshape(c // sp, -1) for t in ttnn.get_device_tensors(e)]
    got = torch.cat([torch.cat(parts[r * tp : (r + 1) * tp], dim=1) for r in range(sp)], dim=0)
    ttnn.deallocate(e)
    ttnn.deallocate(dev)
    ttnn.deallocate(inp)
    from models.experimental.deepseek_v41_flash.tt.prefill.d_p_bridge import _delimiter_rows

    want = emb[ids.clamp(max=V - 1)].clone()
    want[100:103] = _delimiter_rows(d_p.checkpoint)
    want[200 : 200 + take.numel()] = rows[take]
    want[valid:] = emb[V - 1]
    bad = (got != want).any(dim=1).nonzero().flatten()
    assert not bad.numel(), f"embedding rows {bad[:8].tolist()} differ (of {bad.numel()})"
    logger.info("embedding: text, delimiter, image and pad rows exact")


def _check_moe(d_p, layer: int) -> None:
    """The layer's image pass is the layer built with gate.bias_vl, row for row, and the text pass elsewhere."""
    from models.demos.deepseek_v41_flash_d_p.tt.layout import own_rows_to_device, own_rows_to_host

    model = d_p.runtime.model
    assert model.own, "the check reads own rows (OWN_ROWS=on)"
    moe = model.blocks[layer].moe_block
    twin = copy.copy(moe)
    twin.ck, twin.head_w, twin.head_w_image = _BiasVl(moe.ck), None, None
    c, hidden = d_p.chunk, int(model.cfg.dim)
    gen = torch.Generator().manual_seed(11)
    x_host = torch.randn(c, hidden, generator=gen)
    m = (torch.rand(c, generator=gen) < 0.3).float().reshape(-1, 1)

    def run(layer_moe, **kw):
        x = own_rows_to_device(model.mesh, x_host, ttnn.bfloat16)
        y = layer_moe(x, **kw)
        out = own_rows_to_host(model.mesh, y).float()
        ttnn.deallocate(x)
        return out

    y_text = run(moe)
    y_vl = run(twin)
    masks = (own_rows_to_device(model.mesh, 1 - m, ttnn.bfloat16), own_rows_to_device(model.mesh, m, ttnn.bfloat16))
    y_sel = run(moe, image=masks)
    y_text2 = run(moe)
    for t in masks:
        ttnn.deallocate(t)
    img, txt = m.flatten() > 0, m.flatten() == 0
    assert torch.equal(y_text2, y_text), "the MoE layer is not deterministic: compare with a tolerance"
    assert torch.equal(y_sel[txt], y_text[txt]), f"text rows: max |diff| {(y_sel[txt] - y_text[txt]).abs().max()}"
    assert torch.equal(y_sel[img], y_vl[img]), f"image rows: max |diff| {(y_sel[img] - y_vl[img]).abs().max()}"
    moved = (y_vl != y_text).any(dim=1).float().mean()
    logger.info(f"MoE layer {layer}: image pass == bias_vl layer, text pass elsewhere; bias_vl moves {moved:.0%} rows")
    assert moved > 0.05, "bias_vl changed almost no row's experts: is the twin's bias the image bias?"


@pytest.mark.timeout(6 * 3600)
def test_d_p_images():
    from transformers import AutoTokenizer

    from models.demos.common.bringup.core.spec import Spec
    from models.demos.common.bringup.testing.harness import close_mesh, open_mesh
    from models.experimental.deepseek_v41_flash.demo.prefill_server import _GALAXY_SPEC, _checkpoint
    from models.experimental.deepseek_v41_flash.tt.prefill.d_p_bridge import DPPrefill

    checkpoint = _checkpoint()
    saved = torch.load(STATES) if STATES.exists() else None
    if saved is None:
        logger.info(f"{STATES} not found (test_kv_only_image_states): the exact checks and the prompt's own run only")
    os.environ.setdefault("DS41_ENGRAM_TABLE", "resident")
    spec = Spec.load(_GALAXY_SPEC)
    mesh, fab = open_mesh(spec)
    try:
        d_p = DPPrefill(mesh, max_seq=PREFILL_LEN, slots=1, checkpoint=checkpoint, spec=spec, image_rows=IMAGE_ROWS)
        d_p.checkpoint = checkpoint
        if saved is not None:
            ids, rows, row_ids = saved["ids"], saved["rows"], saved["row_ids"]
        else:
            tok = AutoTokenizer.from_pretrained(checkpoint)
            ids, rows, row_ids = _image_prompt(tok, d_p.vocab, int(d_p.cfg.dim))
        assert d_p.vocab + FIRST_IMAGE_ROW <= int(row_ids.min()) and int(row_ids.max()) < d_p.vocab + IMAGE_ROWS
        d_p.write_image_rows(rows, row_ids.tolist())
        _check_embedding(d_p, rows, row_ids)
        _check_moe(d_p, layer=3)
        t0 = time.perf_counter()
        n = d_p.prefill(ids.tolist())
        logger.info(
            f"metal_2 prefill of {n} tokens ({int((ids >= d_p.vocab).sum())} in image spans) in "
            f"{time.perf_counter() - t0:.2f} s"
        )
        host = d_p.host_states(n)
    finally:
        close_mesh(mesh, fab)

    for i, layer in host["layers"].items():
        for name, t in layer.items():
            assert torch.isfinite(t.float()).all(), (i, name)
    if saved is None:
        return
    bad = []
    for i, layer in saved["layers"].items():
        for name, want in layer.items():
            have = host["layers"][i][name].reshape(-1, want.shape[-1]).float()
            assert have.shape == want.shape, (i, name, tuple(have.shape), tuple(want.shape))
            p = _pcc(have, want.float())
            logger.info(f"layer {i:>2} {name:<13} PCC {p:.6f}  max |diff| {(have - want.float()).abs().max():.4f}")
            if p < MIN_PCC:
                bad.append(f"layer {i} {name}: PCC {p:.6f} < {MIN_PCC}")
    assert not bad, "metal_2's image prefill differs from this model's:\n" + "\n".join(bad)
