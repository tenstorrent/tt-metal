# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic (not a gate): where layer 47's error on isl56320's last chunk [51200, 56320) comes from.

The e2e test (test_prefill_e2e.py) failed only there: layer 47 pcc 0.96716, rel 0.27206, worst row rel 16.2. The golden's
layer 47 holds massive activations (token 54853 at 77x the chunk's median row norm). This runs the same isl56320 prefill
with the same model and settings (no model change), then on the last chunk:

  A  layer 47 on the device's own layer 46 output (the e2e path), every boundary of the block read back;
  B  layer 47 on the golden's layer 46 output (from_host), the same boundaries: layer 47's own error, no upstream drift;
  R  the host reference (reference/model.py, fp32 activations, bf16-held weights) on the golden's layer 46 output, with
     the golden's layer 47 K/V of [0, 51200) as its cache: the fp32 values of the same boundaries.

and prints per token (worst row, the massive positions) and per stage (attn_out, h = in + attn_out, mlp_out, out), plus
layer 47 rel L2 with the rows over 20x the median golden row norm left out. Tensors: $DEVICE_TEST_OUT/diag_layer47.pt.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.mimo_v26_flash_rl.reference._bringup_interface import Ctx, run_block
from models.demos.mimo_v26_flash_rl.reference.model import Layer as RefLayer
from models.demos.mimo_v26_flash_rl.reference.model import LayerState, rope_tables
from models.demos.mimo_v26_flash_rl.reference.weights import Checkpoint
from models.demos.mimo_v26_flash_rl.tests.e2e import goldens
from models.demos.mimo_v26_flash_rl.tests.e2e.test_prefill_e2e import (
    CAPACITY,
    CHUNK,
    DEVICE_PARAMS,
    MESH,
    NUM_LAYERS,
    SLOTS,
    layer_errors,
    pcc,
    rel_l2,
)
from models.demos.mimo_v26_flash_rl.tt.model import MimoModel
from models.demos.mimo_v26_flash_rl.tt.runners.kv_caches import allocate_kv_caches
from models.demos.mimo_v26_flash_rl.tt.runners.runtime import _Spec
from models.demos.mimo_v26_flash_rl.tt.settings import settings

LAYER = 47
ISL = 56320
S0 = ISL - CHUNK  # 51200: the last chunk
WATCH = (54853, 51831, 55762, 52977)  # the golden's largest rows on this chunk
MASSIVE = 20.0  # x the median golden row norm
STAGES = ("attn_norm", "attn_out", "h", "ffn_norm", "mlp_out", "out")  # residual-shaped boundaries (not the router)


def out_dir() -> Path:
    d = Path(os.environ.get("DEVICE_TEST_OUT") or "generated/mimo_v26_flash_rl_diag")
    d.mkdir(parents=True, exist_ok=True)
    return d


def device_block(model: MimoModel, i: int, h, start: int, state) -> dict:
    """model.layer(i, h, start, state) with every residual-shaped boundary read back to the host (natural order)."""
    b = model.blocks[i]
    got = {}

    def rec(name, t):
        if name in STAGES:
            got[name] = model.to_host(t).float()

    prev = ttnn.CONFIG.enable_fast_runtime_mode
    ttnn.CONFIG.enable_fast_runtime_mode = False
    inner = []
    try:
        out = run_block(
            b.steps,
            b.component,
            Ctx(layer=i, start=start, length=model.chunk, state=state),
            h,
            rec=lambda name, t: (inner.append((name, t)), rec(name, t)),
        )
    finally:
        ttnn.CONFIG.enable_fast_runtime_mode = prev
    for name, t in inner:
        if name not in ("in", "out"):
            t.deallocate(True)
    out.deallocate(True)
    return got


def host_block(hf: str, cfg: dict, x: torch.Tensor, g: goldens.Golden) -> dict:
    """The reference layer 47 on x (the golden's layer 46 rows of the last chunk), fp32, the golden's K/V as its cache."""
    t0 = time.time()
    ck = Checkpoint(hf, torch.bfloat16)  # reference.weight_dtype
    L = RefLayer(ck, LAYER, cfg)
    print(f"reference layer {LAYER} loaded in {time.time() - t0:.0f} s", flush=True)
    st = LayerState(L.kv_heads, ISL, L.head_dim, L.v_head_dim, torch.float32)
    kv = g.kv(LAYER, S0)
    st.key[:, :S0] = kv["key"].float()
    st.value[:, :S0] = kv["value"].float()
    cos, sin = rope_tables(L.inv_freq, S0, CHUNK)
    ctx = Ctx(layer=LAYER, start=S0, length=CHUNK, state=st, extra={"cos": cos, "sin": sin})
    got = {}
    t0 = time.time()
    run_block(
        L.graph(), L.component, ctx, x.float(), rec=lambda n, t: got.__setitem__(n, t.float()) if n in STAGES else None
    )
    print(f"reference layer {LAYER} ran in {time.time() - t0:.0f} s", flush=True)
    return got


def row_rel(got, want):
    return (got - want).norm(dim=1) / want.norm(dim=1).clamp_min(1e-30)


def compare(tag: str, got: torch.Tensor, want: torch.Tensor, keep: torch.Tensor) -> dict:
    rr = row_rel(got, want)
    w = int(rr.argmax())
    e = layer_errors(got, want)
    d = {
        "pcc": pcc(got, want),
        "rel": rel_l2(got, want),
        "rel_wo_massive": rel_l2(got[keep], want[keep]),
        "pcc_wo_massive": pcc(got[keep], want[keep]),
        "bias": e["bias"],
        "worst_row": {
            "index": w,
            "pos": S0 + w,
            "rel": float(rr[w]),
            "got_norm": float(got[w].norm()),
            "want_norm": float(want[w].norm()),
            "err_norm": float((got[w] - want[w]).norm()),
        },
        "err_share_massive": float(((got - want)[~keep].norm() ** 2) / ((got - want).norm() ** 2).clamp_min(1e-30)),
    }
    print(
        f"  {tag:<34} pcc {d['pcc']:.5f} rel {d['rel']:.4f} | w/o massive pcc {d['pcc_wo_massive']:.5f} "
        f"rel {d['rel_wo_massive']:.4f} | err share of massive rows {d['err_share_massive']:.3f} | worst row "
        f"{w} (pos {S0 + w}) rel {d['worst_row']['rel']:.3f} norm {d['worst_row']['got_norm']:.1f} vs "
        f"{d['worst_row']['want_norm']:.1f}",
        flush=True,
    )
    return d


@pytest.mark.timeout(5400)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True, ids=["ring"])
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True, ids=["8x4"])
def test_diag_layer47(mesh_device):
    assert tuple(mesh_device.shape) == MESH
    hf = settings.get("SERVE_HF_DIR")
    cfg = json.loads((Path(hf) / "config.json").read_text())
    g = goldens.load("isl56320")
    ids = g.ids(ISL)
    layers = list(range(NUM_LAYERS))
    kv = allocate_kv_caches(mesh_device, cfg, layers=layers, num_users=SLOTS, max_seq_len=CAPACITY, chunk_size=CHUNK)
    model = MimoModel(
        mesh_device, _Spec(hf, CHUNK, kv.max_seq_len), layers, lm_head=False, slots=kv.num_users, kv_caches=kv.layers
    )
    state = model.new_state(CAPACITY)
    print(f"model loaded in {model.load_seconds:.0f} s", flush=True)

    # the e2e prefill up to the last chunk's layer 46, then A
    t0 = time.time()
    for c in range(ISL // CHUNK):
        s0 = c * CHUNK
        h = model.embed(ids[s0 : s0 + CHUNK], start=s0)
        for i in layers if s0 != S0 else layers[:LAYER]:
            h2 = model.layer(i, h, s0, state)
            model.free(h)
            h = h2
        if s0 != S0:
            model.free(h)
    dev_in = model.to_host(h).float()
    A = device_block(model, LAYER, h, S0, state)
    model.free(h)
    # B: the same chunk's layer 47 again on the golden's layer 46 rows (its K/V write lands on the same positions)
    gold_in = g.layer_out(LAYER - 1, S0, ISL)
    hb = model.from_host(gold_in.to(torch.bfloat16))
    B = device_block(model, LAYER, hb, S0, state)
    model.free(hb)
    model.sync()
    print(f"device prefill + layer {LAYER} x2 in {time.time() - t0:.0f} s", flush=True)

    R = host_block(hf, cfg, gold_in, g)
    gold = g.layer_out(LAYER, S0, ISL).float()
    gold46 = gold_in.float()

    norms = gold.norm(dim=1)
    med = float(norms.median())
    keep = norms <= MASSIVE * med
    massive = torch.nonzero(~keep).flatten()
    print(
        f"golden layer {LAYER} rows on [{S0}, {ISL}): median norm {med:.1f}, {int((~keep).sum())} rows > "
        f"{MASSIVE:.0f}x: "
        + ", ".join(
            f"{S0 + int(r)} ({float(norms[r]):.0f}, id {int(ids[S0 + r])})"
            for r in massive[norms[massive].argsort(descending=True)]
        ),
        flush=True,
    )

    res = {
        "chunk_start": S0,
        "median_norm": med,
        "massive_rows": [S0 + int(r) for r in massive],
        "out": {},
        "stages": {},
        "watch": {},
    }
    print("layer 47 output vs the golden:", flush=True)
    res["out"]["A_device_e2e"] = compare("A device (own layer 46) vs golden", A["out"], gold, keep)
    res["out"]["B_device_gold_in"] = compare("B device (golden layer 46) vs golden", B["out"], gold, keep)
    res["out"]["R_reference_fp32"] = compare("R reference fp32 vs golden", R["out"], gold, keep)
    res["out"]["B_vs_R"] = compare("B device vs R reference", B["out"], R["out"], keep)
    print("layer 47 input (layer 46 output):", flush=True)
    res["in"] = compare("device layer 46 vs golden", dev_in, gold46, keep)

    print("per stage, B (device, golden input) vs R (reference, same input):", flush=True)
    for s in STAGES:
        res["stages"][f"B_vs_R.{s}"] = compare(f"B vs R  {s}", B[s], R[s], keep)
    print("per stage, A (device e2e) vs R:", flush=True)
    for s in STAGES:
        res["stages"][f"A_vs_R.{s}"] = compare(f"A vs R  {s}", A[s], R[s], keep)

    rr = {k: row_rel(v["out"], gold) for k, v in (("A", A), ("B", B), ("R", R))}
    worst = sorted({int(rr["A"].argmax()), int(rr["B"].argmax())} | {p - S0 for p in WATCH})
    print(
        "rows (pos, id): golden norm | A, B, R out norm | rel vs golden A, B, R | B vs R per stage rel "
        "(attn_out, h, mlp_out, out)",
        flush=True,
    )
    for r in worst:
        row = {
            "pos": S0 + r,
            "token_id": int(ids[S0 + r]),
            "golden_norm": float(gold[r].norm()),
            "golden_in_norm": float(gold46[r].norm()),
            "norm": {k: float(v["out"][r].norm()) for k, v in (("A", A), ("B", B), ("R", R))},
            "rel": {k: float(v[r]) for k, v in rr.items()},
            "B_vs_R": {
                s: float(row_rel(B[s][r : r + 1], R[s][r : r + 1])) for s in ("attn_out", "h", "mlp_out", "out")
            },
            "norms_R": {s: float(R[s][r].norm()) for s in ("attn_out", "h", "mlp_out")},
        }
        res["watch"][str(S0 + r)] = row
        print(
            f"  {S0 + r} (id {row['token_id']}): {row['golden_norm']:.1f} (in {row['golden_in_norm']:.1f}) | "
            + " ".join(f"{row['norm'][k]:.1f}" for k in "ABR")
            + " | "
            + " ".join(f"{row['rel'][k]:.4f}" for k in "ABR")
            + " | "
            + " ".join(f"{row['B_vs_R'][s]:.4f}" for s in ("attn_out", "h", "mlp_out", "out"))
            + f" | R norms attn_out {row['norms_R']['attn_out']:.1f} h {row['norms_R']['h']:.1f} "
            f"mlp_out {row['norms_R']['mlp_out']:.1f}",
            flush=True,
        )

    d = out_dir()
    (d / "diag_layer47.json").write_text(json.dumps(res, indent=1))
    torch.save(
        {
            "A": {k: v.bfloat16() for k, v in A.items()},
            "B": {k: v.bfloat16() for k, v in B.items()},
            "R": {k: v.bfloat16() for k, v in R.items()},
            "device_in": dev_in.bfloat16(),
            "ids": ids[S0:ISL],
        },
        d / "diag_layer47.pt",
    )
    print(f"report: {d / 'diag_layer47.json'}", flush=True)
