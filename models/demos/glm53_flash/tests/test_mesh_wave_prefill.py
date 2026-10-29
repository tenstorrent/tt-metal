# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Per-row prefill ownership (waves) on a multi-row mesh (``doc/mesh/row_ownership.md`` mechanism (a), phase 1:
one slot index per wave): ``GLM53MeshModel.prefill_wave`` against the sequence-parallel ``prefill`` on the KDA +
dense layers (0-2 by default) or, with GLM53_MODEL_LAYERS=0-3 and the partitioned heads2users cache (GLM53_PARTITION_USERS=1 GLM53_PREFILL_PARTITION=heads2users GLM53_WAVE_B=4), the DSA + MoE layer too, real weights, two short prompts:

(i)   the same prompt through SP and through a wave on row 0 (row 1 idle) and on row 1: the first token equal, the
      last-token logits within bf16 noise (the local and the sequence-parallel scans group the recurrence
      differently);
(ii)  two different prompts in one wave vs each alone in a wave: bit-identical per row (the wave must not mix);
(iii) an idle row's slot is untouched by a wave (its KDA carry before == after, bit for bit);
(iv)  per-row starts: row 0 continues its prompt at 64 while row 1 starts another at 0, then row 1 continues at 96
      with row 0 idle; against the unchunked wave and the SP chunked prefill; a non-contiguous start raises.

    GLM53_MESH_SHAPE=2,4 GLM53_MODEL_LAYERS=0-2 pytest models/demos/glm53_flash/tests/test_mesh_wave_prefill.py -s
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
from models.demos.glm53_flash.tt.vllm_config import WaveCall, WaveMember

os.environ.setdefault("GLM53_KDA_FUSED", "1")  # the block KDA decode: tiled conv history (the wave carry blend)
MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "2,4").split(","))
_cache_env = os.environ.get("GLM53_MESH_CACHE", "/home/ttuser/ashai/glm53/_cache/mesh")
CACHE_ROOT = None if _cache_env.lower() == "none" else Path(_cache_env)
PCC_BAR = 0.999  # SP vs local scan grouping, chunk boundaries: bf16 noise
B = int(
    os.environ.get("GLM53_WAVE_B", "2")
)  # user slots per row (a multiple of the columns with the partitioned cache)


def _layers() -> list[int]:
    raw = os.environ.get("GLM53_MODEL_LAYERS", "0-2")
    out: list[int] = []
    for part in raw.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


def _wave(model, members: dict[int, tuple[torch.Tensor, int]], user: int) -> WaveCall:
    """``members``: row -> (chunk ids, absolute start); the wave's ``t_pad`` is the longest chunk rounded to 32."""
    t_pad = max(-(-int(ids.numel()) // 32) * 32 for ids, _ in members.values())
    ms = tuple(
        WaveMember(logical=row, row=row, user=user, start=start, end=start + int(ids.numel()))
        for row, (ids, start) in sorted(members.items())
    )
    call = WaveCall(rows=model.mesh.rows, t_pad=t_pad, members=ms, user=user)
    for row, (ids, _) in members.items():
        call.ids[row] = ids.clone()
    return call


def _sp(model, ids: torch.Tensor, user: int, start: int = 0) -> tuple[int, torch.Tensor]:
    streams, logits, owner = model.prefill(ids, user=user, start=start)
    first = model.first_token(logits, owner)
    host = model.mesh.to_host(logits, rows=0).float().reshape(model.mesh.rows, -1)[owner]
    ttnn.deallocate(streams)
    ttnn.deallocate(logits)
    return first, host


def _kda_slot_state(model, row: int, user: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Every KDA layer's (recurrent [H,K,V], conv tiles [taps,W]) of slot ``user`` on row ``row``'s first chip."""
    out = []
    for dec in model.layers:
        if not dec.is_kda:
            continue
        chip = row * model.mesh.cols
        rec = ttnn.to_torch(model.mesh.device_tensors(dec.kda_batched.recurrent)[chip])[user].clone()
        conv = ttnn.to_torch(model.mesh.device_tensors(dec.kda_batched.convolution)[chip])[:, user].clone()
        out.append((rec, conv))
    return out


def _same_state(a, b) -> bool:
    return all(torch.equal(ra, rb) and torch.equal(ca, cb) for (ra, ca), (rb, cb) in zip(a, b))


def _timed(label: str, fn):
    t0 = time.perf_counter()
    out = fn()
    logger.info(f"  {label}: {time.perf_counter() - t0:.2f} s")
    return out


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_wave_prefill_vs_sequence_parallel(mesh_device, expect_error):
    from transformers import AutoTokenizer

    snapshot = default_snapshot_dir()
    tok = AutoTokenizer.from_pretrained(str(snapshot))
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    layers = _layers()
    ids_a = torch.tensor(
        tok.encode(
            "The lighthouse keeper collected stamps from every ship that passed, and on winter nights he sorted them by the colour of their seas. "
            * 8
        ),
        dtype=torch.long,
    )
    ids_b = torch.tensor(tok.encode("量子计算机使用量子比特，它可以同时处于多个状态；这使得某些问题的求解速度远远超过经典计算机。" * 10), dtype=torch.long)
    n_a, n_b = int(ids_a.numel()), int(ids_b.numel())
    assert n_a > 96 and n_b > 128 and n_a != n_b, (n_a, n_b)
    model = GLM53MeshModel(mesh_device, cfg, layers, max_batch=B, max_seq_len=2048, weight_cache_path=CACHE_ROOT)
    rows = model.mesh.rows
    assert rows > 1, "waves are the multi-row case"
    logger.info(f"wave test: {len(layers)} layers {layers}, prompts {n_a} / {n_b} tokens, mesh {MESH}")

    # (i) SP vs a wave on row 0, then on row 1; (iii) the idle row's slot untouched
    first_sp, logits_sp = _timed("SP prefill a (slot 0)", lambda: _sp(model, ids_a, user=0))
    row1_before = _kda_slot_state(model, 1, 0)
    res0 = _timed("wave a on row 0 (row 1 idle)", lambda: model.prefill_wave(_wave(model, {0: (ids_a, 0)}, user=0)))
    assert set(res0) == {0}
    p0 = pcc(logits_sp, res0[0][1])
    logger.info(f"(i) row 0: wave first token {res0[0][0]} vs SP {first_sp}; logits pcc {p0:.6f}")
    assert res0[0][0] == first_sp and p0 > PCC_BAR
    assert _same_state(row1_before, _kda_slot_state(model, 1, 0)), "(iii) the idle row's slot changed"
    res1 = _timed("wave a on row 1 (row 0 idle)", lambda: model.prefill_wave(_wave(model, {1: (ids_a, 0)}, user=0)))
    p1 = pcc(logits_sp, res1[1][1])
    p01 = pcc(res0[0][1], res1[1][1])
    logger.info(
        f"(i) row 1: first token {res1[1][0]} vs SP {first_sp}; pcc vs SP {p1:.6f}, vs row 0's wave {p01:.6f} (equal: {torch.equal(res0[0][1], res1[1][1])})"
    )
    assert res1[1][0] == first_sp and p1 > PCC_BAR and p01 > 0.9999

    # (ii) two prompts in one wave vs each alone (slot 1)
    res_ab = _timed(
        "wave a | b (slot 1)", lambda: model.prefill_wave(_wave(model, {0: (ids_a, 0), 1: (ids_b, 0)}, user=1))
    )
    res_a = model.prefill_wave(_wave(model, {0: (ids_a, 0)}, user=1))
    res_b = model.prefill_wave(_wave(model, {1: (ids_b, 0)}, user=1))
    first_sp_b, logits_sp_b = _sp(model, ids_b, user=1)
    pb = pcc(logits_sp_b, res_ab[1][1])
    logger.info(
        f"(ii) a|b vs a alone: equal {torch.equal(res_ab[0][1], res_a[0][1])}; b alone: equal "
        f"{torch.equal(res_ab[1][1], res_b[1][1])}; b first {res_ab[1][0]} vs SP {first_sp_b}, pcc {pb:.6f}"
    )
    assert res_ab[0][0] == res_a[0][0] == first_sp and torch.equal(res_ab[0][1], res_a[0][1])
    assert res_ab[1][0] == res_b[1][0] == first_sp_b and torch.equal(res_ab[1][1], res_b[1][1]) and pb > PCC_BAR

    # (iv) per-row starts: chunked continuation on each row, against the unchunked wave and the SP chunked prefill
    model.prefill_wave(_wave(model, {0: (ids_a[:64], 0), 1: (ids_b[:96], 0)}, user=0))
    res2 = _timed(
        "wave a[64:] on row 0 (start 64), row 1 idle",
        lambda: model.prefill_wave(_wave(model, {0: (ids_a[64:], 64)}, user=0)),
    )
    res3 = model.prefill_wave(_wave(model, {1: (ids_b[96:], 96)}, user=0))
    _sp(model, ids_a[:64], user=1)
    first_sp_c, logits_sp_c = _sp(model, ids_a[64:], user=1, start=64)
    p2, p2s = pcc(res0[0][1], res2[0][1]), pcc(logits_sp_c, res2[0][1])
    p3 = pcc(res_ab[1][1], res3[1][1])
    logger.info(
        f"(iv) row 0 continued at 64: first {res2[0][0]} (unchunked wave {res0[0][0]}, SP chunked {first_sp_c}), pcc vs "
        f"unchunked wave {p2:.6f}, vs SP chunked {p2s:.6f}; row 1 continued at 96: first {res3[1][0]} vs {res_ab[1][0]}, pcc {p3:.6f}"
    )
    assert res2[0][0] == res0[0][0] == first_sp_c and p2 > PCC_BAR and p2s > PCC_BAR
    assert res3[1][0] == res_ab[1][0] and p3 > PCC_BAR
    with expect_error(ValueError, "not contiguous"):
        model.prefill_wave(_wave(model, {0: (ids_a[64:], 32)}, user=0))
    logger.info("waves: (i)-(iv) PASS")
