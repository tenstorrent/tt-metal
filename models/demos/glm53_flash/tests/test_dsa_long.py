# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Long-context DSA behaviour that the 4-step sparse test cannot see (stage review P1 items):

* ``test_sparse_selection_vs_hf``: layer 3 on a NON-repetitive 4097-token prompt (random ids,
  dump ``R4097``), 12 decode steps through positions 4097..4108: layer-output PCC per step, the
  pooled-key cache rows re-derived on the host from the HF indexer stems after the steps (every
  complete pool must match), and the TT-selected token-id set vs the HF indexer's selection
  (``index_ids`` recorded by the reference runner) per step.
* ``test_dense_to_sparse_transition``: prefill 2040 tokens on the dense path (user slot 1 of 2 with
  a permuted page table), then decode positions 2040..2063 (dump ``R2040``): the index must have
  been maintained by the dense path so the switch at 2048 and the HF-exact region up to 2050 and
  the tails 0/1/2/3 all match.

    TT_VISIBLE_DEVICES=0 pytest models/demos/glm53_flash/tests/test_dsa_long.py -s
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir, load_layer_raw, shard_headers
from models.demos.glm53_flash.tests.utils import pcc
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.functional_decoder import (
    FunctionalDecoder,
    device_streams_to_host,
    host_streams_to_device,
)

LAYER = 3
R4097 = Path("bringup/references/glm53_flash/R4097")
R2040 = Path("bringup/references/glm53_flash/R2040")
R16385 = Path("bringup/references/glm53_flash/R16385")


def row_pcc_split(ref: torch.Tensor, out: torch.Tensor, P: int) -> dict[str, float]:
    """PCC of a [1,S,4,4096] prefill output per row class: positions < 2048 vs >= 2048 and position % P."""
    S = ref.shape[1]
    pos = torch.arange(S)
    classes = {"rows<2048": pos < 2048, "rows>=2048": pos >= 2048}
    classes.update({f"pos%{P}=={j}": (pos % P) == j for j in range(P)})
    return {k: pcc(ref[:, m], out[:, m]) for k, m in classes.items() if m.any()}


def row_pcc_report(ref, out, P) -> str:
    return " ".join(f"{k} {v:.5f}" for k, v in row_pcc_split(ref, out, P).items()) + "  " + per_row_stats(ref, out)


def per_row_stats(ref: torch.Tensor, out: torch.Tensor) -> str:
    """Distribution of the per-token PCC (each token row = 4 streams x 4096 values): a single-token
    prefill is one sample of this distribution, which is why its PCC is noisier than an aggregate."""
    S = ref.shape[1]
    r = ref[0].reshape(S, -1).float()
    o = out[0].reshape(S, -1).float()
    rc = r - r.mean(1, keepdim=True)
    oc = o - o.mean(1, keepdim=True)
    per = (rc * oc).sum(1) / (rc.norm(dim=1) * oc.norm(dim=1) + 1e-12)
    q = torch.quantile(per, torch.tensor([0.0, 0.05, 0.5]))
    return f"per-row pcc min {q[0]:.5f} p5 {q[1]:.5f} median {q[2]:.5f} rows<0.997 {int((per < 0.997).sum())}/{S}"


def _hf_index_stems(raw, cfg, x_normed: torch.Tensor):
    """Host k (LayerNorm'd) and gate logits of the indexer for normalized attention inputs [S, 4096]."""
    sd = {k[len("self_attn.indexer.") :]: v for k, v in raw.items() if k.startswith("self_attn.indexer.")}
    k = torch.nn.functional.layer_norm(
        x_normed.float() @ sd["wk.weight"].float().t(),
        (cfg.index_head_dim,),
        sd["k_norm.weight"].float(),
        sd["k_norm.bias"].float(),
        1e-6,
    )
    g = x_normed.float() @ sd["index_kpool_compress_gate"].float().t()
    return k, g, sd["index_kpool_compress_ape"].float()


def _hf_pool(k, g, ape, P):
    """[n*P, D] -> [n, D] HF k-pool compression over complete pools."""
    n = k.shape[0] // P
    kp, gp = k[: n * P].reshape(n, P, -1), g[: n * P].reshape(n, P, -1)
    prob = torch.softmax(gp + ape[None], dim=1)
    return (prob * kp).sum(1)


def _attention_input(raw, cfg, streams: torch.Tensor) -> torch.Tensor:
    """HF attn_hc collapse + input_layernorm on mHC streams [1, S, 4, 4096] -> [S, 4096] (bf16 as the model)."""
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextHyperConnection, Glm5NextTextRMSNorm

    from models.demos.glm53_flash.reference.hf_layer_reference import load_config

    _, text = load_config(default_snapshot_dir())
    hc = Glm5NextTextHyperConnection(text)
    hc.fn.data, hc.base.data, hc.scale.data = (
        raw["hc_attn_fn"].float(),
        raw["hc_attn_base"].float(),
        raw["hc_attn_scale"].float(),
    )
    norm = Glm5NextTextRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
    norm.weight.data = raw["input_layernorm.weight"]
    with torch.no_grad():
        _, _, collapsed = hc(streams)
        return norm(collapsed)[0].to(torch.bfloat16)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_sparse_selection_vs_hf(mesh_device):
    snapshot = default_snapshot_dir()
    raw = load_layer_raw(snapshot, LAYER, shard_headers(snapshot))
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    P = cfg.index_kpool
    pre = torch.load(R4097 / f"layer{LAYER}_prefill.pt")
    x_in, x_out = pre["x_in"], pre["x_out"]
    S = x_in.shape[1]
    n_steps = 12
    dec = FunctionalDecoder.from_state_dict(
        raw, hf_config=cfg, layer_idx=LAYER, mesh_device=mesh_device, max_batch=1, max_seq_len=S + 64
    )

    y = dec.prefill_forward(host_streams_to_device(x_in, mesh_device), user=0, seq_len=S)
    y_h = device_streams_to_host(y, 4, S)
    p = pcc(x_out, y_h)
    logger.info(f"layer 3 random-prompt sparse prefill S={S}: pcc {p:.6f}  {row_pcc_report(x_out, y_h, P)}")
    assert p > 0.995
    for name, v in row_pcc_split(x_out, y_h, P).items():
        assert v > 0.995, f"row class {name}: pcc {v}"

    # HF selection for the prefill queries vs TT: compare the *selected id sets* of the last 32 queries
    # by re-running the indexer selection on device is implicit in the layer output; here we check the
    # cache contents and the per-step selection explicitly.
    attn_inputs = [_attention_input(raw, cfg, x_in)]
    for step in range(n_steps):
        d = torch.load(R4097 / f"layer{LAYER}_decode{step}.pt")
        dp = dec.make_decode_positions([d["position"]])
        yd = dec.decode_forward(host_streams_to_device(d["x_in"], mesh_device, pad_to=1), positions=dp)
        p_d = pcc(d["x_out"], device_streams_to_host(yd, 4, 1))
        attn_inputs.append(_attention_input(raw, cfg, d["x_in"]))
        # TT selection for this query vs HF's index_ids (sets of valid token ids)
        hf_ids = d["index_ids"][0, 0]
        hf_set = set(int(i) for i in hf_ids.tolist() if i >= 0)
        tt_raw = _tt_selected_ids(dec, raw, d["x_in"], d["position"], mesh_device)
        # contiguous-prefix contract of sparse_sdpa: valid ids first, then only sentinels; all ids in range
        first_sentinel = next((i for i, v in enumerate(tt_raw) if v == 0xFFFFFFFF), len(tt_raw))
        assert all(v == 0xFFFFFFFF for v in tt_raw[first_sentinel:]), "sentinel inside the valid prefix"
        tt_ids = tt_raw[:first_sentinel]
        assert all(0 <= v <= d["position"] for v in tt_ids), "selected id out of range"
        tt_set = set(tt_ids)
        inter = len(hf_set & tt_set)
        logger.info(
            f"  step {step} (pos {d['position']}): pcc {p_d:.6f}; selected ids HF {len(hf_set)} TT {len(tt_set)} "
            f"common {inter} ({inter / max(1, len(hf_set)):.4f})"
        )
        assert p_d > 0.995
        assert inter / len(hf_set) > 0.97, "TT top-k selection drifted from HF"

    # pooled-key cache rows after the steps vs host recomputation over all tokens seen so far
    x_all = torch.cat(attn_inputs, dim=0)  # [S + n_steps, 4096]
    k, g, ape = _hf_index_stems(raw, cfg, x_all)
    pools_ref = _hf_pool(k, g, ape, P)  # [n_complete, D]
    n_complete = pools_ref.shape[0]
    cache_rows = ttnn.to_torch(dec.cache.index.pools_batched())[0, 0, :n_complete].float()
    p_pool = pcc(pools_ref, cache_rows)
    worst = min(pcc(pools_ref[i], cache_rows[i]) for i in range(n_complete - 8, n_complete))
    logger.info(
        f"  pooled-key cache after {n_steps} steps: {n_complete} pools, pcc {p_pool:.6f}, worst of last 8 rows {worst:.5f}"
    )
    assert p_pool > 0.995 and worst > 0.98


def _tt_selected_ids(dec: FunctionalDecoder, raw, x_in_streams: torch.Tensor, position: int, device) -> list[int]:
    """The indexer's selected token ids for one decode query on the current device index cache."""
    x = _attention_input(raw, dec.cfg, x_in_streams).reshape(1, 1, 1, -1)
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    return dec.layer.attn.selected_ids(x_tt, dec.cache, 0, position)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_dense_to_sparse_transition(mesh_device):
    snapshot = default_snapshot_dir()
    raw = load_layer_raw(snapshot, LAYER, shard_headers(snapshot))
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    pre = torch.load(R2040 / f"layer{LAYER}_prefill.pt")
    x_in, x_out = pre["x_in"], pre["x_out"]
    S = x_in.shape[1]
    assert S < cfg.index_topk
    n_steps = 24
    user = 1
    dec = FunctionalDecoder.from_state_dict(
        raw, hf_config=cfg, layer_idx=LAYER, mesh_device=mesh_device, max_batch=2, max_seq_len=S + 64
    )
    # permuted page table: both users' blocks interleaved and reversed
    bps = dec.cache.blocks_per_seq
    perm = torch.randperm(2 * bps, generator=torch.Generator().manual_seed(5)).to(torch.int32).reshape(2, bps)
    dec.cache.set_page_table(perm)

    y = dec.prefill_forward(host_streams_to_device(x_in, mesh_device), user=user, seq_len=S)
    y_h = device_streams_to_host(y, 4, S)
    p = pcc(x_out, y_h)
    logger.info(
        f"layer 3 dense prefill S={S} (slot {user}, permuted page table): pcc {p:.6f}  {per_row_stats(x_out, y_h)}"
    )
    assert p > 0.995
    # slot 0: a short second user (37 tokens) with its own HF reference, checked at every decode step too
    from transformers import DynamicCache

    from models.demos.glm53_flash.reference.hf_layer_reference import build_layer, load_config

    _, text = load_config(snapshot)
    hf0, _ = build_layer(snapshot, text, LAYER, shard_headers(snapshot))
    cache0 = DynamicCache(config=text)
    with torch.no_grad():
        hf0(x_in[:, :37], attention_mask=torch.ones(1, 37, dtype=torch.bool), past_key_values=cache0, use_cache=True)
    dec.prefill_forward(host_streams_to_device(x_in[:, :37], mesh_device), user=0, seq_len=37)

    for step in range(n_steps):
        d = torch.load(R2040 / f"layer{LAYER}_decode{step}.pt")
        pos = d["position"]
        xin = torch.cat([x_in[:, 200 + step : 201 + step], d["x_in"]], dim=1)  # row 0 dummy, row 1 = reference
        dp = dec.make_decode_positions([37 + step, pos])
        yd = dec.decode_forward(host_streams_to_device(xin, mesh_device, pad_to=2), positions=dp)
        out = device_streams_to_host(yd, 4, 2)
        p_d = pcc(d["x_out"], out[:, user : user + 1])
        with torch.no_grad():
            y0, _ = hf0(
                xin[:, :1], attention_mask=torch.ones(1, 1, dtype=torch.bool), past_key_values=cache0, use_cache=True
            )
        p_0 = pcc(y0, out[:, 0:1])
        mode = "sparse" if pos > cfg.index_topk + 2 else "dense"
        logger.info(
            f"  decode pos {pos} ({mode}, tail {(pos + 1) % cfg.index_kpool}): pcc {p_d:.6f}; short user 0 (pos {37 + step}) {p_0:.6f}"
        )
        assert p_d > 0.995 and p_0 > 0.995


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_sparse_16k(mesh_device):
    """Largest single-chip context of the Phase-1 contract: 16385-token sparse prefill + 2 decode steps."""
    snapshot = default_snapshot_dir()
    raw = load_layer_raw(snapshot, LAYER, shard_headers(snapshot))
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    pre = torch.load(R16385 / f"layer{LAYER}_prefill.pt")
    x_in, x_out = pre["x_in"], pre["x_out"]
    S = x_in.shape[1]
    dec = FunctionalDecoder.from_state_dict(
        raw, hf_config=cfg, layer_idx=LAYER, mesh_device=mesh_device, max_batch=1, max_seq_len=S + 64
    )
    y = dec.prefill_forward(host_streams_to_device(x_in, mesh_device), user=0, seq_len=S)
    y_h = device_streams_to_host(y, 4, S)
    p = pcc(x_out, y_h)
    p_tail = pcc(x_out[:, -2048:], y_h[:, -2048:])
    logger.info(
        f"layer 3 sparse prefill S={S}: pcc {p:.6f} (last 2048 rows {p_tail:.6f})  {row_pcc_report(x_out, y_h, cfg.index_kpool)}"
    )
    assert p > 0.995 and p_tail > 0.995
    for name, v in row_pcc_split(x_out, y_h, cfg.index_kpool).items():
        assert v > 0.995, f"row class {name}: pcc {v}"
    for step in range(2):
        d = torch.load(R16385 / f"layer{LAYER}_decode{step}.pt")
        dp = dec.make_decode_positions([d["position"]])
        yd = dec.decode_forward(host_streams_to_device(d["x_in"], mesh_device, pad_to=1), positions=dp)
        p_d = pcc(d["x_out"], device_streams_to_host(yd, 4, 1))
        logger.info(f"  sparse decode pos {d['position']}: pcc {p_d:.6f}")
        assert p_d > 0.995


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("seq_len", [2042, 2043])
def test_prefill_tail_2_3_then_decode(mesh_device, seq_len):
    """Prompt lengths == 2 and 3 (mod 4): the ring tail rows written one per call, then 12 decode steps
    through the completion of that pool and across the 2048/2050 switch, with the pooled-key cache
    re-derived on the host (every complete pool must match) and per-step PCC vs HF (dumps R2042/R2043)."""
    snapshot = default_snapshot_dir()
    raw = load_layer_raw(snapshot, LAYER, shard_headers(snapshot))
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    P = cfg.index_kpool
    R = Path(f"bringup/references/glm53_flash/R{seq_len}")
    pre = torch.load(R / f"layer{LAYER}_prefill.pt")
    x_in, x_out = pre["x_in"], pre["x_out"]
    S = x_in.shape[1]
    assert S == seq_len and S % P in (2, 3)
    dec = FunctionalDecoder.from_state_dict(
        raw, hf_config=cfg, layer_idx=LAYER, mesh_device=mesh_device, max_batch=1, max_seq_len=S + 64
    )
    y = dec.prefill_forward(host_streams_to_device(x_in, mesh_device), user=0, seq_len=S)
    p = pcc(x_out, device_streams_to_host(y, 4, S))
    logger.info(f"layer 3 dense prefill S={S} (tail {S % P}): pcc {p:.6f}")
    assert p > 0.995
    attn_inputs = [_attention_input(raw, cfg, x_in)]
    for step in range(12):
        d = torch.load(R / f"layer{LAYER}_decode{step}.pt")
        dp = dec.make_decode_positions([d["position"]])
        yd = dec.decode_forward(host_streams_to_device(d["x_in"], mesh_device, pad_to=1), positions=dp)
        p_d = pcc(d["x_out"], device_streams_to_host(yd, 4, 1))
        attn_inputs.append(_attention_input(raw, cfg, d["x_in"]))
        logger.info(f"  decode pos {d['position']} (tail {(d['position'] + 1) % P}): pcc {p_d:.6f}")
        assert p_d > 0.995
    x_all = torch.cat(attn_inputs, dim=0)
    k, g, ape = _hf_index_stems(raw, cfg, x_all)
    pools_ref = _hf_pool(k, g, ape, P)
    n_complete = pools_ref.shape[0]
    cache_rows = ttnn.to_torch(dec.cache.index.pools_batched())[0, 0, :n_complete].float()
    worst = min(pcc(pools_ref[i], cache_rows[i]) for i in range(n_complete))
    logger.info(
        f"  pooled-key cache: {n_complete} complete pools vs host, pcc {pcc(pools_ref, cache_rows):.6f}, worst row {worst:.5f}"
    )
    assert worst > 0.98
