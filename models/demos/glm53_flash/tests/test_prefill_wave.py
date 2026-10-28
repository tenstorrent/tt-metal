# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The row-local prefill wave (``GLM53MeshModel.prefill_wave_ids``, tt-blaze G53F-0250) vs the sequence-parallel prefill.

On the 8x4 mesh with a few layers (``GLM53_MODEL_LAYERS``, default 2-4: KDA + dense MLP, DSA + MoE, KDA + MoE), eight
different prompts of one length (``GLM53_WAVE_S``, default 512) go through the existing SP prefill one at a time (the
reference: each prompt's last-token logits and the state slot 0 then holds, read on the host) and then through ONE
wave call (prompt r on mesh row r). Per row r, against prompt r's reference: the last-token logits PCC >=
``GLM53_WAVE_PCC`` (0.999) and the same greedy token, every KDA layer's recurrent / conv carry and every DSA layer's
latent rows, complete pooled keys and ring rows PCC >= the same bar. A second wave in the reversed prompt order must
give every prompt the same results (PCC >= 0.9999): rows are independent.

    GLM53_MESH_SHAPE=8,4 GLM53_MODEL_LAYERS=2-4 GLM53_PREFILL_EXPERTS=regions \\
      pytest models/demos/glm53_flash/tests/test_prefill_wave.py -s --timeout 3000

Every reference pass runs before the waves (results kept on the host); every pass writes slot 0 of every row.
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
from models.demos.glm53_flash.tests.utils import pcc
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.mesh import mesh_device_params
from models.demos.glm53_flash.tt.mesh_model import GLM53MeshModel

MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "8,4").split(","))
S = int(os.environ.get("GLM53_WAVE_S", "512"))
MAX_SEQ = int(os.environ.get("GLM53_WAVE_MAX_SEQ", "2048"))
PCC_BAR = float(os.environ.get("GLM53_WAVE_PCC", "0.999"))
SWAP_BAR = float(os.environ.get("GLM53_WAVE_SWAP_PCC", "0.9999"))
# real-text prompts (comma-separated ids; 8 windows of S tokens); "random" = uniform random ids (near-tie logits)
IDS = os.environ.get("GLM53_WAVE_IDS", "/mnt/tt-data/sdawle/glm53f/needle/needle_260000_d10.ids")
_DT = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
EXPERT_DTYPE = _DT[os.environ.get("GLM53_MODEL_EXPERT_DTYPE", "bfp4")]
EXPERT_DOWN_DTYPE = _DT[os.environ.get("GLM53_MODEL_EXPERT_DOWN_DTYPE", "bfp4")]
_cache_env = os.environ.get("GLM53_MESH_CACHE", "none")
CACHE_ROOT = None if _cache_env.lower() == "none" else Path(_cache_env)


def _layer_range() -> list[int]:
    spec = os.environ.get("GLM53_MODEL_LAYERS", "2-4")
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(v) for v in spec.split(",")]


def _chip(t: ttnn.Tensor, chip: int) -> torch.Tensor:
    return ttnn.to_torch(ttnn.get_device_tensors(t)[chip])


def _logits_row(model: GLM53MeshModel, logits: ttnn.Tensor, row: int) -> torch.Tensor:
    return _chip(logits, row * model.mesh.cols).float().reshape(-1).clone()


def _slot_state(model: GLM53MeshModel, chip: int, user: int, S: int) -> dict[str, torch.Tensor]:
    """What slot ``user`` holds on chip ``chip`` after a prefill of ``S`` tokens (test_trace_prefill's readback on one
    chip): per KDA layer the recurrent / conv carries, per DSA layer the latent rows [0, S), the complete pooled keys
    and the ring rows of the open pool."""
    out = {}
    for dec in model.layers:
        li = dec.layer_idx
        if dec.is_kda:
            out[f"L{li}.recurrent"] = _chip(dec.kda_batched.recurrent, chip).float()[user].clone()
            conv = _chip(dec.kda_batched.convolution, chip).float()
            out[f"L{li}.conv"] = (conv[:, user] if dec.layer.attn.conv_tiled else conv[user]).clone()
        else:
            cache = dec.cache
            slot = cache.prefill_slot(user)
            nblk = -(-S // cache.block_size)
            ids = cache.page_table_host[slot, :nblk].tolist()
            lat = _chip(cache.cache, chip).float()
            out[f"L{li}.latent"] = lat[ids].reshape(nblk * cache.block_size, -1)[:S].clone()
            idx = cache.index
            pools = _chip(idx.pool_cache, chip).float()[slot * idx.pool_blocks : (slot + 1) * idx.pool_blocks]
            out[f"L{li}.pools"] = pools.reshape(-1, idx.D)[: S // idx.P].clone()
            tail = S % idx.P
            if tail:
                out[f"L{li}.ring_k"] = _chip(idx.ring_k, chip).float()[slot, 0, :tail].clone()
                out[f"L{li}.ring_g"] = _chip(idx.ring_g, chip).float()[slot, 0, :tail].clone()
    return out


def _worst(ref: dict, got: dict, tag: str) -> float:
    worst = 1.0
    for k, a in ref.items():
        b = got[k]
        p = pcc(a, b) if a.numel() > 1 and float(a.std()) > 0 else float(torch.equal(a, b))
        if p < PCC_BAR:
            logger.warning(f"  {tag} {k}: pcc {p:.6f} max|diff| {float((a - b).abs().max()):.3e}")
        worst = min(worst, p)
    return worst


@pytest.mark.timeout(3000)
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_prefill_wave_vs_sp(mesh_device):
    layers = _layer_range()
    cfg = GLM53FlashConfig.from_hf_config_json(default_snapshot_dir() / "config.json")
    t0 = time.perf_counter()
    model = GLM53MeshModel(
        mesh_device,
        cfg,
        layers,
        max_batch=1,
        max_seq_len=MAX_SEQ,
        weight_cache_path=CACHE_ROOT,
        expert_dtype=EXPERT_DTYPE,
        expert_down_dtype=EXPERT_DOWN_DTYPE,
    )
    rows, cols = model.mesh.rows, model.mesh.cols
    logger.info(f"model: mesh {MESH} layers {layers} S {S} built in {time.perf_counter() - t0:.0f} s")
    if IDS == "random":
        gen = torch.Generator().manual_seed(250)
        prompts = [torch.randint(1, cfg.vocab_size - 1, (S,), generator=gen) for _ in range(rows)]
    else:
        text = torch.tensor([int(v) for v in open(IDS).read().split(",") if v.strip()], dtype=torch.long)
        stride = max(S, (text.numel() - S) // rows)
        prompts = [text[r * stride : r * stride + S].clone() for r in range(rows)]
        assert all(p.numel() == S for p in prompts), "the ids file is too short for the wave's prompts"

    # -- references: each prompt alone through the SP prefill (slot 0 on every row), kept on the host
    ref = []
    for r, p in enumerate(prompts):
        t_r = time.perf_counter()
        streams, logits, owner = model.prefill(p, user=0)
        lg = _logits_row(model, logits, owner)
        ttnn.synchronize_device(mesh_device)
        ref.append(
            dict(logits=lg, token=int(lg.argmax()), state=_slot_state(model, 0, 0, S), s=time.perf_counter() - t_r)
        )
        for t in (streams, logits):
            ttnn.deallocate(t)
    logger.info(f"SP reference prefills: {[round(x['s'], 2) for x in ref]} s per prompt")

    def wave(order):
        t_w = time.perf_counter()
        streams, logits, _ = model.prefill_wave_ids([prompts[i] for i in order], user=0)
        ttnn.synchronize_device(mesh_device)
        wall = time.perf_counter() - t_w
        toks = model.wave_first_tokens(logits)
        res = {}
        for r, i in enumerate(order):
            res[i] = dict(logits=_logits_row(model, logits, r), token=toks[r], state=_slot_state(model, r * cols, 0, S))
        for t in (streams, logits):
            ttnn.deallocate(t)
        return res, wall

    fwd, wall_f = wave(list(range(rows)))
    fwd2, wall_f2 = wave(list(range(rows)))  # the second call: compiled, the wave's TTFT for 8 prompts
    rev, wall_r = wave(list(range(rows - 1, -1, -1)))
    report, bad = [], []
    for i in range(rows):
        lp = pcc(ref[i]["logits"], fwd[i]["logits"])
        sp = _worst(ref[i]["state"], fwd[i]["state"], f"prompt {i} row {i}")
        swap_l = pcc(fwd[i]["logits"], rev[i]["logits"])
        swap_s = min(
            pcc(a, rev[i]["state"][k]) for k, a in fwd[i]["state"].items() if a.numel() > 1 and float(a.std()) > 0
        )
        report.append(
            dict(
                prompt=i,
                logits_pcc=round(lp, 6),
                state_pcc_min=round(sp, 6),
                token_ref=ref[i]["token"],
                token_wave=fwd[i]["token"],
                swap_logits_pcc=round(swap_l, 6),
                swap_state_pcc_min=round(swap_s, 6),
            )
        )
        if lp < PCC_BAR or sp < PCC_BAR or fwd[i]["token"] != ref[i]["token"] or swap_l < SWAP_BAR or swap_s < SWAP_BAR:
            bad.append(i)
    for rr in report:
        logger.info(f"[prefill-wave] {rr}")
    logger.info(
        f"[prefill-wave] S {S} x {rows} prompts: SP one at a time {sum(x['s'] for x in ref):.2f} s total "
        f"(last {ref[-1]['s']:.2f} s), wave {wall_f:.2f} s first / {wall_f2:.2f} s second / {wall_r:.2f} s reversed"
    )
    assert not bad, f"prompts {bad} differ between the wave and their own SP prefill (see the report)"
