# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Every decoder layer (0..44) of GLM-5.3-Flash on the single-chip ``FunctionalDecoder`` vs HF,
real weights, each layer on its own true input distribution.

Realistic inputs come from the host reference dump ``bringup/references/glm53_flash/L45_S128``: a
128-token random-id prompt plus 2 decode steps through the whole HF stack, produced layer by layer
with::

    python models/demos/glm53_flash/reference/hf_layer_reference.py --layers 0-44 --seq-len 128 \\
        --random-ids 45 --num-decode 2 --stream --out bringup/references/glm53_flash/L45_S128

For layer n the TT input is the dump's ``x_in`` of layer n (no error accumulation across layers);
the reference is the live HF layer (``build_layer``, strict-fp32 tensors preserved) run on the same
``x_in`` with a fresh ``DynamicCache``. The dump's ``x_out`` (= the next layer's ``x_in``) is
reported too, so the stack is validated layer by layer. For MoE layers the router top-8 sets of
the HF gate and the TT router are compared row by row.

    TT_VISIBLE_DEVICES=2 GLM53_LAYERS=0-11 pytest models/demos/glm53_flash/tests/test_all_layers.py -s --timeout 1800
    pytest models/demos/glm53_flash/tests/test_all_layers.py -s -k "L07 or L43"

Env: ``GLM53_LAYERS=a-b`` (or ``a,b,c``) restricts the parametrization; ``GLM53_RESULTS=<file>``
appends one JSON line per layer; ``GLM53_WEIGHT_CACHE=<dir>`` uses ``<dir>/layer<n>`` as the KDA
weight cache (none by default); ``GLM53_DUMP_WAIT_S`` waits for a dump still being produced;
``GLM53_SUBBLOCK=1`` also logs the attention / FFN sub-block PCCs (hooks on both sides).
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
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir, load_layer_raw, shard_headers
from models.demos.glm53_flash.reference.hf_layer_reference import build_layer, load_config, parse_layers
from models.demos.glm53_flash.tests.utils import pcc
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.functional_decoder import (
    FunctionalDecoder,
    device_streams_to_host,
    host_streams_to_device,
)

DUMP = Path("bringup/references/glm53_flash/L45_S128")
NUM_DECODE = 2
PCC_BAR = 0.995
LAYERS = parse_layers(os.environ.get("GLM53_LAYERS", "0-44"))


def _wait_for_dump(path: Path):
    """The dump is produced layer by layer; optionally wait for the file (size stable for 2 s)."""
    budget = float(os.environ.get("GLM53_DUMP_WAIT_S", "0"))
    t0 = time.time()
    while True:
        if path.exists():
            s1 = path.stat().st_size
            time.sleep(2)
            if path.stat().st_size == s1 and s1 > 0:
                return
        if time.time() - t0 > budget:
            if not path.exists():
                raise FileNotFoundError(f"{path} (run reference/hf_layer_reference.py, see module docstring)")
            return
        time.sleep(10)


def _row_pcc(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-token PCC over the [4, 4096] stream vector: a, b [1, S, 4, 4096] -> [S]."""
    a = a.detach().float().reshape(a.shape[1], -1)
    b = b.detach().float().reshape(b.shape[1], -1)
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-30)


def _set_agreement(idx_hf: torch.Tensor, idx_tt: torch.Tensor) -> tuple[int, int]:
    """Rows whose top-8 expert sets are identical (idx_* [T, 8])."""
    T = idx_hf.shape[0]
    same = sum(set(idx_hf[t].tolist()) == set(idx_tt[t].tolist()) for t in range(T))
    return int(same), int(T)


class _Hooks:
    """Captures the HF gate top-k ids and (optionally) the attention / FFN sub-block outputs on
    both sides; the TT side is captured by wrapping the module forwards (test-time only)."""

    def __init__(self, hf, dec, sublayers: bool):
        self.hf_idx: list[torch.Tensor] = []
        self.hf_gate_in: list[torch.Tensor] = []
        self.tt_idx: list[torch.Tensor] = []
        self.gate = getattr(hf.mlp, "gate", None)
        self.hf_sub: dict[str, list[torch.Tensor]] = {"attn": [], "ffn": []}
        self.tt_sub: dict[str, list[torch.Tensor]] = {"attn": [], "ffn": []}
        self.sublayers = sublayers
        if hasattr(hf.mlp, "gate"):

            def gate_hook(m, i, o):  # returns None: a non-None hook result would replace the gate output
                self.hf_idx.append(o[2].detach().clone())
                self.hf_gate_in.append(i[0].detach())

            hf.mlp.gate.register_forward_hook(gate_hook)
            router = dec.layer.mlp.router
            orig_router = router.forward

            def router_fwd(x):
                w, idx, dense = orig_router(x)
                self.tt_idx.append(ttnn.to_torch(idx).reshape(idx.shape[-2], -1).long())
                return w, idx, dense

            router.forward = router_fwd
        if not sublayers:
            return
        hf.self_attn.register_forward_hook(
            lambda m, i, o: self.hf_sub["attn"].append((o[0] if isinstance(o, tuple) else o).detach().clone())
        )
        hf.mlp.register_forward_hook(lambda m, i, o: self.hf_sub["ffn"].append(o.detach().clone()))
        mlp = dec.layer.mlp
        orig_mlp = mlp.forward

        def mlp_fwd(x):
            y = orig_mlp(x)
            self.tt_sub["ffn"].append(ttnn.to_torch(y).float()[0, 0])
            return y

        mlp.forward = mlp_fwd
        attn = dec.layer.attn
        for name in ("prefill_forward", "decode_forward"):
            orig = getattr(attn, name)

            def attn_fwd(*a, _orig=orig, **k):
                out = _orig(*a, **k)
                y = out[0] if isinstance(out, tuple) else out
                self.tt_sub["attn"].append(ttnn.to_torch(y).float().reshape(-1, y.shape[-1]))
                return out

            setattr(attn, name, attn_fwd)

    def router_report(self, step: int, S: int) -> tuple[int, int, str]:
        """Rows whose HF / TT top-8 sets agree, plus for the disagreeing rows the HF margin between
        the 8th and 9th biased scores (a near tie flips under the attention-side bf16 error of the
        TT MoE input) next to the median margin of all rows."""
        idx_hf = self.hf_idx[step].reshape(-1, 8)[:S]
        idx_tt = self.tt_idx[step][:S]
        same, tot = _set_agreement(idx_hf, idx_tt)
        if same == tot:
            return same, tot, ""
        x = self.hf_gate_in[step].reshape(-1, 4096)[:S].float()
        with torch.no_grad():
            choice = torch.sigmoid(x @ self.gate.weight.float().t()) + self.gate.e_score_correction_bias.float()
        top = choice.sort(dim=-1, descending=True).values
        margin = top[:, 7] - top[:, 8]
        rows = [t for t in range(tot) if set(idx_hf[t].tolist()) != set(idx_tt[t].tolist())]
        detail = (
            f" (rows {rows}: 8th-9th margins "
            + " ".join(f"{float(margin[t]):.2e}" for t in rows)
            + f"; median margin {float(margin.median()):.2e}, disagreeing max {float(margin[rows].max()):.2e})"
        )
        return same, tot, detail

    def sublayer_pccs(self, step: int, S: int) -> dict[str, float]:
        out = {}
        for key in ("attn", "ffn"):
            if len(self.hf_sub[key]) <= step or len(self.tt_sub[key]) <= step:
                continue
            h = self.hf_sub[key][step].reshape(-1, 4096)[:S]
            t = self.tt_sub[key][step].reshape(-1, 4096)[:S]
            out[key] = pcc(h, t)
        return out


@pytest.fixture(scope="module")
def snapshot_ctx():
    snapshot = default_snapshot_dir()
    _, text = load_config(snapshot)
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    return snapshot, text, cfg, shard_headers(snapshot)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("layer_idx", LAYERS, ids=[f"L{i:02d}" for i in LAYERS])
def test_layer_vs_hf(mesh_device, snapshot_ctx, layer_idx):
    from transformers import DynamicCache

    snapshot, text, cfg, headers = snapshot_ctx
    kind = f"{'KDA' if cfg.is_kda(layer_idx) else 'DSA'}+{'MoE' if cfg.is_moe(layer_idx) else 'dense'}"
    _wait_for_dump(DUMP / f"layer{layer_idx}_decode{NUM_DECODE - 1}.pt")  # (pass --timeout above 300 s to wait)
    t_start = time.time()
    pre = torch.load(DUMP / f"layer{layer_idx}_prefill.pt")
    decs = [torch.load(DUMP / f"layer{layer_idx}_decode{s}.pt") for s in range(NUM_DECODE)]
    x_in = pre["x_in"]  # [1, S, 4, 4096] bf16: this layer's true HF input
    S = x_in.shape[1]

    # ---- weights once for both sides
    raw = load_layer_raw(snapshot, layer_idx, headers)
    hf, sd = build_layer(snapshot, text, layer_idx, headers, raw=raw)
    del sd
    cache_root = os.environ.get("GLM53_WEIGHT_CACHE")
    dec = FunctionalDecoder.from_state_dict(
        raw,
        hf_config=cfg,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        max_batch=1,
        max_seq_len=S + 64,
        weight_cache_path=Path(cache_root) / f"layer{layer_idx}" if cache_root else None,
    )
    del raw
    t_loaded = time.time()
    hooks = _Hooks(hf, dec, sublayers=os.environ.get("GLM53_SUBBLOCK", "0") == "1")
    result = {"layer": layer_idx, "kind": kind, "S": S}

    # ---- prefill
    cache = DynamicCache(config=text)
    with torch.no_grad():
        y_hf, _ = hf(x_in, attention_mask=torch.ones(1, S, dtype=torch.bool), past_key_values=cache, use_cache=True)
    y = dec.prefill_forward(host_streams_to_device(x_in, mesh_device), user=0, seq_len=S)
    y_h = device_streams_to_host(y, 4, S)
    p = pcc(y_hf, y_h)
    p_dump = pcc(pre["x_out"], y_h)
    p_hf_dump = pcc(pre["x_out"], y_hf)
    rows = _row_pcc(y_hf, y_h)
    streams = [pcc(y_hf[:, :, i], y_h[:, :, i]) for i in range(4)]
    msg = (
        f"layer {layer_idx} ({kind}) prefill S={S}: pcc {p:.6f} (vs dump x_out {p_dump:.6f}; hf live vs dump "
        f"{p_hf_dump:.6f}); worst row {int(rows.argmin())} {float(rows.min()):.5f}; streams "
        + " ".join(f"{s:.5f}" for s in streams)
    )
    result.update(prefill=p, prefill_vs_dump=p_dump, hf_vs_dump=p_hf_dump, prefill_worst_row=float(rows.min()))
    if hooks.hf_idx:
        same, tot, detail = hooks.router_report(0, S)
        msg += f"; router top-8 sets identical {same}/{tot}{detail}"
        result["router_prefill"] = f"{same}/{tot}"
    sub = hooks.sublayer_pccs(0, S)
    if sub:
        msg += "; sub-blocks " + " ".join(f"{k} {v:.6f}" for k, v in sub.items())
        result["sub_prefill"] = sub
    logger.info(msg)

    # ---- decode steps on the dump's decode inputs (positions S, S+1)
    for step, d in enumerate(decs):
        xt = d["x_in"]  # [1, 1, 4, 4096]
        assert d["position"] == S + step
        with torch.no_grad():
            y_hf_d, _ = hf(xt, attention_mask=torch.ones(1, 1, dtype=torch.bool), past_key_values=cache, use_cache=True)
        dp = dec.make_decode_positions([S + step])
        yd = dec.decode_forward(host_streams_to_device(xt, mesh_device, pad_to=1), positions=dp)
        yd_h = device_streams_to_host(yd, 4, 1)
        p_d = pcc(y_hf_d, yd_h)
        p_d_dump = pcc(d["x_out"], yd_h)
        msg = f"  decode step {step} (pos {S + step}): pcc {p_d:.6f} (vs dump x_out {p_d_dump:.6f})"
        result[f"decode{step}"] = p_d
        result[f"decode{step}_vs_dump"] = p_d_dump
        if hooks.hf_idx:
            same, tot, detail = hooks.router_report(step + 1, 1)
            msg += f"; router top-8 set identical {same}/{tot}{detail}"
            result[f"router_decode{step}"] = f"{same}/{tot}"
        sub = hooks.sublayer_pccs(step + 1, 1)
        if sub:
            msg += "; sub-blocks " + " ".join(f"{k} {v:.6f}" for k, v in sub.items())
            result[f"sub_decode{step}"] = sub
        logger.info(msg)

    result["load_s"] = round(t_loaded - t_start, 1)
    result["wall_s"] = round(time.time() - t_start, 1)
    logger.info(f"LAYER_RESULT {json.dumps(result)}")
    if os.environ.get("GLM53_RESULTS"):
        with open(os.environ["GLM53_RESULTS"], "a") as f:
            f.write(json.dumps(result) + "\n")

    assert p >= PCC_BAR, f"layer {layer_idx} prefill pcc {p:.6f} < {PCC_BAR}"
    for step in range(NUM_DECODE):
        assert result[f"decode{step}"] >= PCC_BAR, f"layer {layer_idx} decode {step} pcc {result[f'decode{step}']:.6f}"
    if hooks.hf_idx:
        # Exact routing on an identical MoE input is asserted by test_moe.py; here the TT MoE input
        # carries the attention-side bf16 error, so near-tie flips (logged with their margins) are
        # expected on a few rows. This bound only catches a broken / bf16-ranked router.
        same, tot = (int(v) for v in result["router_prefill"].split("/"))
        assert same >= 0.8 * tot, f"router top-8 sets differ on {tot - same}/{tot} rows"
