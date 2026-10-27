# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The DSpark drafter's weight loader (``tt/v41/dspark_weights.py``) against the checkpoint, CPU only.

No device: run with ``python -m pytest --noconftest -p no:cacheprovider tests/v41/test_dspark_weights.py``
from the
model's root (the tree's conftest opens the cluster at collection) or as a script. The schema test needs the
checkpoint's index only; the round-trip tests read one expert and two projections of stage 0 from shard 44 and
the Markov tables from shard 46 (``TT_V41_CHECKPOINT`` or the HF cache, else skipped) and check, against a
slow
re-dequantization written here from the format's definition: the dequantized values bit for bit, the per-chip
shards (expert placement, TP splits, the main projection's reorder) as slices of the checkpoint tensor, and
the
bfp8 round trip through a host ttnn tensor (what the device holds) within the block format's error.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig as cfg
from models.demos.deepseek_v3_d_p.tt.v41 import dspark_weights as dw
from models.demos.deepseek_v3_d_p.tt.v41.weights import (
    CHECKPOINT_REVISION,
    INDEX_FILE,
    V41Checkpoint,
    resolve_checkpoint,
)

MESH = (8, 4)  # (sp, tp): the Galaxy mesh of the served model
EXPERTS_PER_CHIP = dw.DSPARK_ROUTED_EXPERTS // (MESH[0] * MESH[1])


def _checkpoint(need_shards: bool) -> V41Checkpoint:
    ckpt = resolve_checkpoint()
    if ckpt is None:
        pytest.skip("no checkpoint (TT_V41_CHECKPOINT or the HF cache)")
    if need_shards:
        files = {ckpt.weight_map[n] for n in dw.stage_tensor_names(cfg, 0)}
        if missing := sorted(f for f in files if not (ckpt.root / f).is_file()):
            pytest.skip(f"DSpark shards not downloaded: {missing}")
    return ckpt


def _slow_fp8_block(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """The format's definition, element by element over blocks: value = e4m3 * 2^e8m0 of its 32x32 block."""
    out = torch.empty(weight.shape, dtype=torch.float32)
    s = scale.float()
    for i in range(0, weight.shape[0], 32):
        for j in range(0, weight.shape[1], 32):
            out[i : i + 32, j : j + 32] = weight[i : i + 32, j : j + 32].float() * s[i // 32, j // 32]
    return out.to(torch.bfloat16)


def _slow_mxfp4_rows(packed: torch.Tensor, scale: torch.Tensor, rows: range) -> torch.Tensor:
    """MXFP4 rows ``rows``: nibble ``2j`` low, ``2j + 1`` high; E2M1 by its bit fields; one scale per 32
    elements."""
    table = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    out = torch.empty(len(rows), packed.shape[1] * 2, dtype=torch.float32)
    raw = packed.view(torch.uint8)
    for r_out, r in enumerate(rows):
        row = raw[r].tolist()
        s = scale[r].float().tolist()
        for j, byte in enumerate(row):
            for half, code in enumerate((byte & 0x0F, byte >> 4)):
                mag = table[code & 0x7] * (-1.0 if code & 0x8 else 1.0)
                col = 2 * j + half
                out[r_out, col] = mag * s[col // 32]
    return out.to(torch.bfloat16)


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().flatten(), b.float().flatten()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def _bfp8_roundtrip(t: torch.Tensor) -> torch.Tensor:
    """``t`` [rows, cols] bf16 -> the values a bfp8 tiled device tensor would hold (host ttnn, no device)."""
    ttnn = pytest.importorskip("ttnn")
    tt = ttnn.from_torch(t[None, None], dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    return ttnn.to_torch(tt)[0, 0].to(torch.bfloat16)


def test_stage_schema_against_index():
    """Every name the loader reads exists in the checkpoint index, with the stored dtype the schema expects,
    and
    the three stages differ only by the stage-specific heads (2,401 tensors in all)."""
    ckpt = _checkpoint(need_shards=False)
    assert ckpt.root.name == CHECKPOINT_REVISION or (ckpt.root / INDEX_FILE).is_file()
    total = 0
    for stage in range(cfg.NUM_DSPARK_LAYERS):
        names = dw.stage_tensor_names(cfg, stage)
        assert len(names) == len(set(names))
        missing = [n for n in names if n not in ckpt]
        assert not missing, missing[:5]
        in_index = [n for n in ckpt.weight_map if n.startswith(dw.stage_prefix(stage))]
        assert sorted(in_index) == sorted(names), (stage, len(in_index), len(names))
        total += len(names)
    assert total == 2401
    assert dw.dspark_layer_id(0) == 40 and cfg.compress_ratio(dw.dspark_layer_id(2)) == 0


def test_expert_roundtrip():
    """Stage 0, expert 0: the loader's dequantized w1 / w2 / w3 equal the slow definition on sampled rows; the
    expert's chip under EP-32 is (row 0, column 0, local 0) and expert 127's (7, 3, 3); bfp8 keeps the values
    within the block format (PCC > 0.999, every value within 1/64 of its tile row's largest)."""
    ckpt = _checkpoint(need_shards=True)
    (expert,) = dw.load_stage_experts(ckpt, 0, cfg, experts=range(0, 1))
    stem = f"{dw.stage_prefix(0)}ffn.experts.0"
    raw = ckpt.read([f"{stem}.{w}.{p}" for w in ("w1", "w2", "w3") for p in ("weight", "scale")])
    for key, w, down in dw._EXPERT_PROJ:
        t = expert[key]
        assert t.dtype == torch.bfloat16
        assert tuple(t.shape) == (
            (cfg.EMB_SIZE, cfg.MOE_INTERMEDIATE_SIZE) if down else (cfg.MOE_INTERMEDIATE_SIZE, cfg.EMB_SIZE)
        )
        rows = range(0, t.shape[0], max(1, t.shape[0] // 7))
        slow = _slow_mxfp4_rows(raw[f"{stem}.{w}.weight"], raw[f"{stem}.{w}.scale"], rows)
        assert torch.equal(t[list(rows)], slow), key
    assert dw.expert_chip(0, *MESH, EXPERTS_PER_CHIP) == (0, 0, 0)
    assert dw.expert_chip(127, *MESH, EXPERTS_PER_CHIP) == (7, 3, 3)
    assert dw.expert_chip(8 * 4 + 1, *MESH, EXPERTS_PER_CHIP) == (0, 1, 1)  # chip 8 = group 1, chip 0; local 1
    w1 = expert["gate_proj"].T.contiguous()  # the converter stacks w.T: [in, out] per expert
    rt = _bfp8_roundtrip(w1)
    assert _pcc(w1, rt) > 0.999
    step = w1.float().abs().reshape(-1, 32, 32).amax(dim=2, keepdim=True).expand(-1, 32, 32).reshape(w1.shape) / 64
    assert ((w1.float() - rt.float()).abs() <= step + 1e-6).all()


def test_projection_roundtrip():
    """Stage 0's ``wq_a`` [1280, 5120] and ``wq_b`` [32768, 1280]: the loader's dequantization equals the slow
    definition; the TP-4 row and column shards are slices of the transposed tensor; bfp8 within the format."""
    ckpt = _checkpoint(need_shards=True)
    dense = dw.load_stage_dense(ckpt, 0, cfg)
    attn = dense["attn"]
    raw = ckpt.read([f"{dw.stage_prefix(0)}attn.{n}.{p}" for n in ("wq_a", "wq_b") for p in ("weight", "scale")])
    for name in ("wq_a", "wq_b"):
        t = attn[name]
        slow = _slow_fp8_block(
            raw[f"{dw.stage_prefix(0)}attn.{name}.weight"], raw[f"{dw.stage_prefix(0)}attn.{name}.scale"]
        )
        assert torch.equal(t, slow), name
    rows = dw.split_tp_rows(attn["wq_a"], MESH[1])
    assert [tuple(r.shape) for r in rows] == [(cfg.EMB_SIZE // MESH[1], cfg.Q_LORA_RANK)] * MESH[1]
    assert torch.equal(torch.cat(rows, dim=0), attn["wq_a"].T)
    cols = dw.split_tp_cols(attn["wq_b"], MESH[1])
    heads_per_chip = cfg.NUM_ATTENTION_HEADS // MESH[1]
    assert [tuple(c.shape) for c in cols] == [(cfg.Q_LORA_RANK, heads_per_chip * cfg.HEAD_DIM)] * MESH[1]
    assert torch.equal(torch.cat(cols, dim=1), attn["wq_b"].T)
    rt = _bfp8_roundtrip(rows[1])
    assert _pcc(rows[1], rt) > 0.999
    assert tuple(dense["main_proj"].shape) == (cfg.EMB_SIZE, 3 * cfg.EMB_SIZE)
    shards = dw.main_proj_shards(dense["main_proj"], 3, MESH[1])
    full_t = dense["main_proj"].T  # [3 * hidden, hidden], rows (tap, j)
    h, q = cfg.EMB_SIZE, cfg.EMB_SIZE // MESH[1]
    for t_idx, shard in enumerate(shards):
        assert tuple(shard.shape) == (3 * q, h)
        for tap in range(3):  # shard t holds tap k's rows [t * q, (t + 1) * q) in tap order
            assert torch.equal(shard[tap * q : (tap + 1) * q], full_t[tap * h + t_idx * q : tap * h + (t_idx + 1) * q])


def test_markov_tables_and_seeding_dict():
    """The last stage's Markov tables are bf16 [vocab, 256] (vocab / tp = 32,320 rows a chip when
    column-parallel
    as the tied head); ``seeding_weights`` is ``TtV41DSpark``'s dict and equals the stage loaders' tensors."""
    ckpt = _checkpoint(need_shards=True)
    last = cfg.NUM_DSPARK_LAYERS - 1
    dense = dw.load_stage_dense(ckpt, last, cfg)
    for name in ("embed", "head"):
        t = dense["markov"][name]
        assert t.dtype == torch.bfloat16 and tuple(t.shape) == (cfg.VOCAB_SIZE, dw.DSPARK_MARKOV_RANK)
    assert cfg.VOCAB_SIZE % MESH[1] == 0 and cfg.VOCAB_SIZE // MESH[1] == 32320
    assert tuple(dense["confidence"].shape) == (1, cfg.EMB_SIZE + dw.DSPARK_MARKOV_RANK)
    assert tuple(dense["norm"].shape) == (cfg.EMB_SIZE,)
    seed = dw.seeding_weights(ckpt, cfg)
    assert len(seed["layers"]) == cfg.NUM_DSPARK_LAYERS
    assert torch.equal(seed["layers"][last]["wkv"], dense["attn"]["wkv"])
    assert torch.equal(seed["layers"][last]["kv_norm"], dense["attn"]["kv_norm"])
    stage0 = dw.load_stage_dense(ckpt, 0, cfg)
    assert torch.equal(seed["main_proj"], stage0["main_proj"]) and torch.equal(seed["main_norm"], stage0["main_norm"])
    assert torch.equal(seed["layers"][0]["wkv"], stage0["attn"]["wkv"])


def test_cache_dir_is_namespaced(tmp_path: Path):
    """The drafter's cache directory is its own, keyed by dtype and mesh, never the backbone's ``real-`` one."""
    a = dw.dspark_cache_dir(tmp_path, MESH, "BFLOAT8_B")
    b = dw.dspark_cache_dir(tmp_path, MESH, "BFLOAT4_B")
    c = dw.dspark_cache_dir(tmp_path, (2, 4), "BFLOAT8_B")
    assert a.parent == tmp_path and a.name.startswith("dspark-") and a.name.endswith("-mesh8x4")
    assert len({a, b, c}) == 3 and c.name.endswith("-mesh2x4")
    assert a == dw.dspark_cache_dir(tmp_path, MESH, "BFLOAT8_B")  # deterministic


if __name__ == "__main__":
    sys.exit(pytest.main(["--noconftest", "-p", "no:cacheprovider", "-v", os.path.abspath(__file__)]))
