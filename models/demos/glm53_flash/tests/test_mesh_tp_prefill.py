# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Phase 2: the decoder layers tensor-parallel over a single mesh row (1, 4): prefill (ttKDA TP /
DSA with head-sharded projections / hybrid MoE with 4 expert groups) and decode vs HF with real
weights, through the stage-1 FunctionalDecoder API unchanged. The outputs must be replicated across
the four chips.

    TT_VISIBLE_DEVICES=25,29,21,17 TT_MESH_GRAPH_DESC_PATH=~/g4cov/p150_x4_galaxy_2ch.textproto MESH_DEVICE=P150x4 \
    pytest models/demos/glm53_flash/tests/test_mesh_tp_prefill.py -s
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir, load_layer_raw, shard_headers
from models.demos.glm53_flash.reference.hf_layer_reference import build_layer, load_config
from models.demos.glm53_flash.tests.utils import pcc
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.functional_decoder import (
    FunctionalDecoder,
    device_streams_to_host,
    host_streams_to_device,
)
from models.demos.glm53_flash.tt.mesh import mesh_device_params

DUMP = Path("bringup/references/glm53_flash/R4097")
ROWS, COLS = 1, 4
# the diagnostics can run on another mesh shape (e.g. "1,1" for the single-chip control)
DIAG_MESH = tuple(int(v) for v in __import__("os").environ.get("GLM53_DIAG_MESH", "1,4").split(","))
S = 200  # non-aligned prompt
N_DECODE = 3


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [mesh_device_params(ROWS, COLS)], indirect=True)
@pytest.mark.parametrize("mesh_device", [(ROWS, COLS)], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 3, 4])
def test_tp4_prefill_decode_vs_hf(mesh_device, layer_idx):
    from transformers import DynamicCache

    snapshot = default_snapshot_dir()
    _, text = load_config(snapshot)
    headers = shard_headers(snapshot)
    raw = load_layer_raw(snapshot, layer_idx, headers)
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    hf, _ = build_layer(snapshot, text, layer_idx, headers)
    base = torch.load(DUMP / f"layer{layer_idx}_prefill.pt")["x_in"]
    x_prompt, x_next = base[:, 100 : 100 + S], base[:, 100 + S : 100 + S + N_DECODE]
    cache = DynamicCache(config=text)
    with torch.no_grad():
        y_hf, _ = hf(x_prompt, attention_mask=torch.ones(1, S, dtype=torch.bool), past_key_values=cache, use_cache=True)
        y_hf_dec = []
        for t in range(N_DECODE):
            yt, _ = hf(
                x_next[:, t : t + 1],
                attention_mask=torch.ones(1, 1, dtype=torch.bool),
                past_key_values=cache,
                use_cache=True,
            )
            y_hf_dec.append(yt)

    dec = FunctionalDecoder.from_state_dict(
        raw, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh_device, max_batch=1, max_seq_len=256
    )
    mesh = dec.mesh
    assert (mesh.rows, mesh.cols) == (ROWS, COLS)
    y = dec.prefill_forward(host_streams_to_device(x_prompt, mesh_device), user=0, seq_len=S)
    shards = [ttnn.to_torch(d) for d in mesh.device_tensors(y)]
    for c in range(1, COLS):
        assert torch.equal(shards[c][:, :, :S], shards[0][:, :, :S]), c
    out = device_streams_to_host(y, 4, S, mesh=mesh)
    p = pcc(y_hf, out)
    logger.info(f"layer {layer_idx} TP4 prefill S={S}: pcc vs HF {p:.6f} (replicated across the 4 chips)")
    assert p > 0.995
    dp = dec.make_decode_positions([S])
    for t in range(N_DECODE):
        dec.update_decode_positions(dp, [S + t])
        yd = dec.decode_forward(host_streams_to_device(x_next[:, t : t + 1], mesh_device, pad_to=1), positions=dp)
        p_d = pcc(y_hf_dec[t], device_streams_to_host(yd, 4, 1, mesh=mesh))
        logger.info(f"layer {layer_idx} TP4 decode at position {S + t}: pcc vs HF {p_d:.6f}")
        assert p_d > 0.995


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [mesh_device_params(ROWS, COLS)], indirect=True)
@pytest.mark.parametrize("mesh_device", [(ROWS, COLS)], indirect=True)
def test_tp4_sparse_prefill_layer3(mesh_device):
    """The sparse DSA prefill rows (>= 2048) with 16 local heads per chip: the R4097 random-prompt
    dump of layer 3 through the TP4 decoder, dense rows and sparse rows scored separately."""
    layer_idx = 3
    snapshot = default_snapshot_dir()
    headers = shard_headers(snapshot)
    raw = load_layer_raw(snapshot, layer_idx, headers)
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    pre = torch.load(DUMP / f"layer{layer_idx}_prefill.pt")
    x_in, x_out = pre["x_in"], pre["x_out"]
    T = x_in.shape[1]
    dec = FunctionalDecoder.from_state_dict(
        raw, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh_device, max_batch=1, max_seq_len=4160
    )
    mesh = dec.mesh
    y = dec.prefill_forward(host_streams_to_device(x_in, mesh_device), user=0, seq_len=T)
    out = device_streams_to_host(y, 4, T, mesh=mesh)
    split = dec.layer.attn.sparse_row_start
    p_all, p_dense, p_sparse = (
        pcc(x_out, out),
        pcc(x_out[:, :split], out[:, :split]),
        pcc(x_out[:, split:], out[:, split:]),
    )
    logger.info(
        f"layer 3 TP4 sparse prefill T={T}: pcc all {p_all:.6f}, rows < {split} {p_dense:.6f}, rows >= {split} {p_sparse:.6f}"
    )
    assert min(p_all, p_dense, p_sparse) > 0.995


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [mesh_device_params(*DIAG_MESH)], indirect=True)
@pytest.mark.parametrize("mesh_device", [DIAG_MESH], indirect=True)
def test_tp4_sparse_diagnostic_layer3(mesh_device):
    """Diagnostic for the TP4 sparse-row gap: per-row PCC distribution of the sparse rows, the
    indexer selection of every chip vs HF's index_ids (and across chips), and the sparse rows
    recomputed with the sparse path disabled (dense attention over the whole prefix)."""
    from models.demos.glm53_flash.tests.utils import attention_input, per_row_stats

    layer_idx = 3
    snapshot = default_snapshot_dir()
    headers = shard_headers(snapshot)
    raw = load_layer_raw(snapshot, layer_idx, headers)
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    pre = torch.load(DUMP / f"layer{layer_idx}_prefill.pt")
    x_in, x_out = pre["x_in"], pre["x_out"]
    T = x_in.shape[1]
    dec = FunctionalDecoder.from_state_dict(
        raw, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh_device, max_batch=1, max_seq_len=4160
    )
    mesh = dec.mesh
    y = dec.prefill_forward(host_streams_to_device(x_in, mesh_device), user=0, seq_len=T)
    out = device_streams_to_host(y, 4, T, mesh=mesh)
    split = dec.layer.attn.sparse_row_start
    logger.info(f"sparse rows per-row stats: {per_row_stats(x_out[:, split:], out[:, split:])}")
    logger.info(f"dense rows per-row stats: {per_row_stats(x_out[:, :split], out[:, :split])}")
    # selection of every chip at the first decode position vs HF
    d = torch.load(DUMP / f"layer{layer_idx}_decode0.pt")
    hf_set = set(int(i) for i in d["index_ids"][0, 0].tolist() if i >= 0)
    x_attn = attention_input(raw, cfg, d["x_in"])
    attn = dec.layer.attn
    idx = attn.indexer
    x_tt = ttnn.from_torch(
        x_attn.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device
    )
    position = int(d["position"])
    q_resid = attn._q_resid(x_tt)
    q_i, w_i = idx.queries_and_weights(x_tt, q_resid)
    start, r = position - position % 32, position % 32
    ids32 = idx.select(
        attn._place_row(q_i, r),
        attn._place_row(w_i, r),
        dec.cache.index,
        0,
        query_start=start,
        num_pools_valid=(position + 1) // idx.P,
    )
    row = ttnn.slice(ids32, [0, 0, r, 0], [1, 1, r + 1, ids32.shape[-1]])
    per_chip = [[int(v) for v in ttnn.to_torch(t).flatten().tolist()] for t in mesh.device_tensors(row)]
    sets = []
    for c, ids in enumerate(per_chip):
        first = next((i for i, v in enumerate(ids) if v == 0xFFFFFFFF), len(ids))
        s = set(ids[:first])
        sets.append(s)
        logger.info(
            f"chip {c}: {len(s)} ids, common with HF {len(s & hf_set)} ({len(s & hf_set) / max(1, len(hf_set)):.4f}), identical to chip 0: {s == sets[0]}"
        )
    # sparse rows through the dense attention (whole prefix) for comparison
    attn.sparse_row_start = 10**9
    y2 = dec.prefill_forward(host_streams_to_device(x_in, mesh_device), user=0, seq_len=T)
    out2 = device_streams_to_host(y2, 4, T, mesh=mesh)
    logger.info(
        f"rows >= {split} via the DENSE path at TP4: pcc {pcc(x_out[:, split:], out2[:, split:]):.6f} "
        f"(sparse path: {pcc(x_out[:, split:], out[:, split:]):.6f}); rows < {split}: {pcc(x_out[:, :split], out2[:, :split]):.6f}"
    )


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [mesh_device_params(*DIAG_MESH)], indirect=True)
@pytest.mark.parametrize("mesh_device", [DIAG_MESH], indirect=True)
def test_tp4_sparse_attention_level_layer3(mesh_device):
    """Attention-sublayer output at TP4 vs the HF attention (forward hook) on the R4097 prompt, dense
    and sparse rows separately, plus the TT attention output saved for a single-chip comparison."""
    import os

    from models.demos.glm53_flash.reference.hf_layer_reference import build_layer, load_config
    from models.demos.glm53_flash.tests.utils import per_row_stats

    layer_idx = 3
    snapshot = default_snapshot_dir()
    _, text = load_config(snapshot)
    headers = shard_headers(snapshot)
    raw = load_layer_raw(snapshot, layer_idx, headers)
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    pre = torch.load(DUMP / f"layer{layer_idx}_prefill.pt")
    x_in = pre["x_in"]
    T = x_in.shape[1]
    hf, _ = build_layer(snapshot, text, layer_idx, headers)
    captured = {}
    hf.self_attn.register_forward_hook(lambda m, i, o: captured.__setitem__("attn", o[0].detach().float()))
    with torch.no_grad():
        hf(x_in, attention_mask=torch.ones(1, T, dtype=torch.bool))
    attn_hf = captured["attn"][0]  # [T, 4096]
    dec = FunctionalDecoder.from_state_dict(
        raw, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh_device, max_batch=1, max_seq_len=4160
    )
    mesh = dec.mesh
    attn = dec.layer.attn
    orig = attn.prefill_forward

    def wrapped(*a, **k):
        y = orig(*a, **k)
        captured["tt"] = mesh.to_host(y)[0, 0, :T].float()
        return y

    attn.prefill_forward = wrapped
    dec.prefill_forward(host_streams_to_device(x_in, mesh_device), user=0, seq_len=T)
    attn_tt = captured["tt"]
    split = attn.sparse_row_start
    logger.info(
        f"TP4 attention vs HF attention: all {pcc(attn_hf, attn_tt):.6f}, rows < {split} {pcc(attn_hf[:split], attn_tt[:split]):.6f}, "
        f"rows >= {split} {pcc(attn_hf[split:], attn_tt[split:]):.6f}"
    )
    logger.info(
        f"sparse rows per-row: {per_row_stats(attn_hf[split:].reshape(1, -1, 1, 4096), attn_tt[split:].reshape(1, -1, 1, 4096))}"
    )
    out = os.environ.get("GLM53_SAVE_ATTN")
    if out:
        torch.save({"hf": attn_hf, "tt": attn_tt, "split": split}, out)
        logger.info(f"saved attention outputs to {out}")
