# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Expert dtype gate evidence (design section 0 "Expert dtype" / section 9): routed-expert weight
formats on the two MoE layer kinds with real weights and realistic activations, one chip.

Policies (``expert_dtype`` = gate/up W13, ``expert_down_dtype`` = down W2):
    bfp8      bfp8_b / bfp8_b   the default (baseline)
    bfp4_gu   bfp4_b / bfp8_b   the design's candidate (halves the dominant expert-streaming term)
    bfp4_all  bfp4_b / bfp4_b   the lower bound (down in bfp4 too)

Per (layer, policy): MoE-only PCC at T=256 on ``realistic_mlp_input`` (router top-8 sets vs HF
reported; the router is fp32 and does not depend on the expert dtype), the full layer on the random
prompt dump ``R4097`` (prefill S=4097: PCC on the first 1024 rows = the dense region of layer 3,
on all rows, and the per-row PCC distribution; then the 4 decode steps 4097..4100), logged as one
table at the end of the module. The baseline must hold the 0.995 bar; the bfp4 policies are
measurements (loose sanity bar) because the design's decision needs the eval set.

    TT_VISIBLE_DEVICES=0 pytest models/demos/glm53_flash/tests/test_expert_dtype.py -s
    TT_VISIBLE_DEVICES=0 GLM53_DTYPE_LAYERS=4 GLM53_DTYPE_POLICIES=bfp8,bfp4_gu pytest ... -s   # subset
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir, load_layer_raw, shard_headers
from models.demos.glm53_flash.reference.hf_layer_reference import load_config
from models.demos.glm53_flash.tests.test_moe import _hf_moe, realistic_mlp_input
from models.demos.glm53_flash.tests.utils import pcc
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.functional_decoder import (
    FunctionalDecoder,
    device_streams_to_host,
    host_streams_to_device,
)

DUMP = Path("bringup/references/glm53_flash/R4097")
POLICIES = {
    "bfp8": (ttnn.bfloat8_b, ttnn.bfloat8_b),
    "bfp4_gu": (ttnn.bfloat4_b, ttnn.bfloat8_b),
    "bfp4_all": (ttnn.bfloat4_b, ttnn.bfloat4_b),
}
LAYERS = [int(x) for x in os.environ.get("GLM53_DTYPE_LAYERS", "3,4").split(",")]
POLICY_NAMES = [p for p in os.environ.get("GLM53_DTYPE_POLICIES", ",".join(POLICIES)).split(",") if p]
MOE_T = 256
PREFILL_ROWS = 1024  # the "1024 rows" prefill PCC (dense region of the DSA layer)
N_DECODE = 4
BAR = 0.995  # the design's PCC bar; enforced on the baseline policy
SANITY = 0.98  # bfp4 policies: measurement only, but a broken upload must still fail


def per_row_pcc(ref: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Per-token PCC of HF ``[1, S, 4, 4096]`` streams (one row = 4 streams x 4096 values)."""
    S = ref.shape[1]
    r = ref[0].reshape(S, -1).float()
    o = out[0].reshape(S, -1).float()
    rc = r - r.mean(1, keepdim=True)
    oc = o - o.mean(1, keepdim=True)
    return (rc * oc).sum(1) / (rc.norm(dim=1) * oc.norm(dim=1) + 1e-12)


def row_stats(per: torch.Tensor) -> dict[str, float]:
    q = torch.quantile(per, torch.tensor([0.0, 0.05, 0.5]))
    return {"min": float(q[0]), "p5": float(q[1]), "median": float(q[2]), "rows<0.997": int((per < 0.997).sum())}


_LAYER_CACHE: dict[int, tuple] = {}


def layer_weights(layer_idx: int):
    """Raw layer tensors + the HF MoE module (strict-fp32 router bias) + the stacked expert tensors,
    one layer resident at a time (each is ~45 GB of host memory)."""
    if layer_idx not in _LAYER_CACHE:
        _LAYER_CACHE.clear()
        snapshot = default_snapshot_dir()
        _, text = load_config(snapshot)
        raw = load_layer_raw(snapshot, layer_idx, shard_headers(snapshot))
        hf, sd = _hf_moe(text, {k: v for k, v in raw.items() if k.startswith("mlp.")})
        _LAYER_CACHE[layer_idx] = (snapshot, text, raw, hf, sd)
    return _LAYER_CACHE[layer_idx]


@pytest.fixture(scope="module")
def sweep_table():
    rows: dict[tuple[int, str], dict] = {}
    yield rows
    if not rows:
        return
    hdr = (
        "| layer | policy (gate/up, down) | MoE T=256 PCC | MoE per-row p5 / min | router sets | "
        f"prefill PCC rows<{PREFILL_ROWS} | prefill PCC all 4097 | per-row p5 / min (all rows) | "
        "decode PCC 4097..4100 (min) |"
    )
    lines = [hdr, "|" + "---|" * 9]
    for (layer, policy), r in sorted(rows.items()):
        lines.append(
            f"| {layer} | {policy} ({r['dtypes']}) | {r['moe_pcc']:.5f} | {r['moe_p5']:.5f} / {r['moe_min']:.5f} | "
            f"{r['sets']} | {r['prefill_pcc_head']:.5f} | {r['prefill_pcc_all']:.5f} | "
            f"{r['row_p5']:.5f} / {r['row_min']:.5f} | "
            + " ".join(f"{p:.5f}" for p in r["decode"])
            + f" ({min(r['decode']):.5f}) |"
        )
    logger.info("expert dtype sweep (R4097 random prompt, realistic MoE inputs):\n" + "\n".join(lines))


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("policy", POLICY_NAMES)
@pytest.mark.parametrize("layer_idx", LAYERS)
def test_expert_dtype_policy(mesh_device, layer_idx, policy, sweep_table):
    gu_dtype, down_dtype = POLICIES[policy]
    snapshot, text, raw, hf, sd = layer_weights(layer_idx)
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    assert cfg.is_moe(layer_idx)
    kind = f"{'KDA' if cfg.is_kda(layer_idx) else 'DSA'}+MoE"
    # the decoder takes the pre-stacked experts (the HF module build already stacked them)
    raw_dec = {k: v for k, v in raw.items() if not k.startswith("mlp.experts.")}
    raw_dec["mlp.experts.gate_up_proj"] = sd["experts.gate_up_proj"]
    raw_dec["mlp.experts.down_proj"] = sd["experts.down_proj"]
    pre = torch.load(DUMP / f"layer{layer_idx}_prefill.pt")
    x_in, x_out = pre["x_in"], pre["x_out"]
    S = x_in.shape[1]
    dec = FunctionalDecoder.from_state_dict(
        raw_dec,
        hf_config=cfg,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        max_batch=1,
        max_seq_len=S + 64,
        expert_dtype=gu_dtype,
        expert_down_dtype=down_dtype,
    )
    moe = dec.layer.mlp
    dtypes = f"{moe.experts.w13.dtype}, {moe.experts.w2.dtype}"
    assert moe.experts.w13.dtype == gu_dtype and moe.experts.w2.dtype == down_dtype
    assert moe.shared.w13.dtype == ttnn.bfloat8_b, "the shared expert stays on weight_dtype"
    logger.info(
        f"layer {layer_idx} ({kind}) policy {policy}: experts W13 {moe.experts.w13.dtype}, W2 {moe.experts.w2.dtype}"
    )

    # ---- MoE alone at T=256 on realistic inputs (vs the HF MoE module)
    x = realistic_mlp_input(snapshot, text, raw, MOE_T, layer=layer_idx)
    with torch.no_grad():
        y_hf = hf(x)
        _, w_hf, idx_hf = hf.gate(x)
    x_tt = ttnn.from_torch(x.reshape(1, 1, MOE_T, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    _, idx_tt, dense_tt = moe.router(x_tt)
    idx_t = ttnn.to_torch(idx_tt).reshape(MOE_T, -1).long()
    same_sets = sum(set(idx_t[t].tolist()) == set(idx_hf[t].tolist()) for t in range(MOE_T))
    ttnn.deallocate(dense_tt)
    y_tt = ttnn.to_torch(moe(x_tt)).reshape(1, MOE_T, -1)
    moe_pcc = pcc(y_hf, y_tt)
    moe_rows = row_stats(per_row_pcc(y_hf.reshape(1, MOE_T, 1, -1), y_tt.reshape(1, MOE_T, 1, -1)))
    logger.info(
        f"  MoE T={MOE_T}: pcc {moe_pcc:.6f}, router identical top-8 sets {same_sets}/{MOE_T}, "
        f"per-row min {moe_rows['min']:.5f} p5 {moe_rows['p5']:.5f} median {moe_rows['median']:.5f}"
    )

    # ---- full layer: prefill the 4097-token random prompt (layer 3: dense rows < 2048, sparse beyond)
    y = dec.prefill_forward(host_streams_to_device(x_in, mesh_device), user=0, seq_len=S)
    y_h = device_streams_to_host(y, 4, S)
    ttnn.deallocate(y)
    p_head = pcc(x_out[:, :PREFILL_ROWS], y_h[:, :PREFILL_ROWS])
    p_all = pcc(x_out, y_h)
    per = per_row_pcc(x_out, y_h)
    rows_all = row_stats(per)
    rows_head = row_stats(per[:PREFILL_ROWS])
    logger.info(
        f"  prefill S={S}: pcc rows<{PREFILL_ROWS} {p_head:.6f} (per-row min {rows_head['min']:.5f} p5 {rows_head['p5']:.5f} "
        f"median {rows_head['median']:.5f}); all rows {p_all:.6f} (per-row min {rows_all['min']:.5f} "
        f"p5 {rows_all['p5']:.5f} median {rows_all['median']:.5f}, rows<0.997 {rows_all['rows<0.997']}/{S})"
    )
    # ---- the 4 decode steps of the dump (positions 4097..4100; layer 3 on the sparse path)
    decode = []
    for step in range(N_DECODE):
        d = torch.load(DUMP / f"layer{layer_idx}_decode{step}.pt")
        dp = dec.make_decode_positions([d["position"]])
        yd = dec.decode_forward(host_streams_to_device(d["x_in"], mesh_device, pad_to=1), positions=dp)
        decode.append(pcc(d["x_out"], device_streams_to_host(yd, 4, 1)))
        ttnn.deallocate(yd)
    logger.info("  decode steps 4097..4100: " + " ".join(f"{p:.6f}" for p in decode))

    sweep_table[(layer_idx, policy)] = dict(
        dtypes=dtypes,
        moe_pcc=moe_pcc,
        moe_p5=moe_rows["p5"],
        moe_min=moe_rows["min"],
        sets=f"{same_sets}/{MOE_T}",
        prefill_pcc_head=p_head,
        prefill_pcc_all=p_all,
        row_p5=rows_all["p5"],
        row_min=rows_all["min"],
        row_p5_head=rows_head["p5"],
        row_min_head=rows_head["min"],
        decode=decode,
    )
    bar = BAR if policy == "bfp8" else SANITY
    assert same_sets >= MOE_T - max(1, MOE_T // 64)
    assert moe_pcc > bar and p_head > bar and p_all > bar and min(decode) > bar, f"policy {policy} below {bar}"
