# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Precision stress of the routed-expert ops on the same inputs (one chip, MiMo expert shape by default):

  unified   unified_routed_expert_moe (TtRoutedExpert)
  fused     moe_fused_swiglu (TtRoutedExpert with every expert in the fused band)
  flat      flat_routed_expert (flatpy: the Python builder, with its MIMO_FL_* accumulation knobs)
Every op gets the same row-major bf16 dispatch buffer (MIMO_PREC_X_TILED=1: unified / fused get host-tiled bfp8).

Ragged counts (0, 1, 31, 33, ..., the full capacity); x distributions (scales, all positive, heavy tails, outlier
channels, one-hot / near one-hot rows, sparse, per-row scale spread, constant rows) per weight set (normal, large,
all positive). Against two references: quantized (bfp8 x, bfp4 weights, fp32 math: the op's own arithmetic error)
and fp32 (x and weights unquantized: end to end). Result lines go to the log and MIMO_PREC_OUT (a TSV).
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping, extract_mesh_config, get_ep_mesh_mapper
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from models.demos.mimo_v2_d_p.tt.flat_expert import FlatExpert, FlatRoutedExpert

H = int(os.environ.get("MIMO_PREC_H", "4096"))
I = int(os.environ.get("MIMO_PREC_I", "2048"))
M = 512  # capacity per expert
COUNTS = [512, 1, 31, 33, 0, 200, 64, 300]
E = len(COUNTS)
ROWS_CHECK = 96  # rows per expert in the metrics (host reference cost)
OPS = [o for o in os.environ.get("MIMO_PREC_OPS", "unified,fused,flat").split(",") if o]  # flatpy: the Python builder
X_TILED = int(os.environ.get("MIMO_PREC_X_TILED", "0"))  # 1: unified / fused get host-tiled bfp8 x
TAG = os.environ.get("MIMO_PREC_TAG", "")  # a label (e.g. the builder's MIMO_FL_GU_ACC / _DN_ACC combination)


def _x_case(name, n, h, g):
    r = lambda *s: torch.randn(*s, generator=g)
    if name == "normal":
        return r(n, h)
    if name == "normal_x0.01":
        return r(n, h) * 0.01
    if name == "normal_x100":
        return r(n, h) * 100
    if name == "abs_normal":  # all positive
        return r(n, h).abs()
    if name == "uniform01":  # all positive
        return torch.rand(n, h, generator=g)
    if name == "student_t2":  # heavy tails
        return torch.distributions.StudentT(2.0).sample((n, h)).clamp(-1e3, 1e3)
    if name == "outlier_channels":  # a few channels 100x (LLM activation outliers)
        x = r(n, h)
        x[:, torch.randperm(h, generator=g)[:8]] *= 100
        return x
    if name == "one_hot":  # one 1 per row
        x = torch.zeros(n, h)
        x[torch.arange(n), torch.randint(0, h, (n,), generator=g)] = 1.0
        return x
    if name == "one_hot_x10":
        x = torch.zeros(n, h)
        x[torch.arange(n), torch.randint(0, h, (n,), generator=g)] = 10.0
        return x
    if name == "near_one_hot":  # one large element + tiny noise (bfp8 shares an exponent per 16 elements)
        x = r(n, h) * 1e-3
        x[torch.arange(n), torch.randint(0, h, (n,), generator=g)] = 10.0
        return x
    if name == "sparse90":  # 90% zeros
        return r(n, h) * (torch.rand(n, h, generator=g) > 0.9)
    if name == "row_scales":  # rows 1e-3 .. 1e3
        return r(n, h) * (10.0 ** (torch.rand(n, 1, generator=g) * 6 - 3))
    if name == "constant":
        return torch.ones(n, h)
    raise ValueError(name)


X_CASES = [
    "normal",
    "normal_x0.01",
    "normal_x100",
    "abs_normal",
    "uniform01",
    "student_t2",
    "outlier_channels",
    "one_hot",
    "one_hot_x10",
    "near_one_hot",
    "sparse90",
    "row_scales",
    "constant",
]
X_CASES = [c for c in X_CASES if c in os.environ.get("MIMO_PREC_CASES", ",".join(X_CASES)).split(",")]
W_SETS = {"w_normal": (0.02, False), "w_large": (0.1, False), "w_pos": (0.02, True)}


def _metrics(ref, got):
    ref, got = ref.double(), got.double()
    bad = int((~torch.isfinite(got)).sum())
    if bad:
        return dict(pcc=float("nan"), rel=float("nan"), norm=float("nan"), row=float("nan"), nonfinite=bad)
    rn = ref.norm()
    rel = float((got - ref).norm() / rn) if rn > 0 else float((got - ref).norm())
    rows = (got - ref).norm(dim=1) / ref.norm(dim=1).clamp_min(1e-30)
    a, b = ref.flatten() - ref.mean(), got.flatten() - got.mean()
    pcc = float((a @ b) / (a.norm() * b.norm())) if a.norm() > 0 and b.norm() > 0 else float("nan")
    return dict(
        pcc=pcc, rel=rel, norm=float(got.norm() / rn) if rn > 0 else float("nan"), row=float(rows.max()), nonfinite=0
    )


def _report(line):
    logger.info(line)
    if os.environ.get("MIMO_PREC_OUT"):
        with open(os.environ["MIMO_PREC_OUT"], "a") as f:
            f.write(line + "\n")


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("wset", list(W_SETS), ids=list(W_SETS))
def test_expert_precision(mesh_device, device_params, wset):
    std, pos = W_SETS[wset]
    g = torch.Generator().manual_seed(7)
    mk = lambda *s: (torch.randn(*s, generator=g).abs() if pos else torch.randn(*s, generator=g)) * std
    # nn.Linear layout per expert: gate / up [I, H], down [H, I]
    w = [{"gate_proj": mk(I, H), "up_proj": mk(I, H), "down_proj": mk(H, I)} for _ in range(E)]
    mc = extract_mesh_config(mesh_device)
    gidx_host = ExpertMapping.create_global_expert_idx_table(
        experts_per_chip=E, dispatch_group_size=mc.dispatch_group_size, num_dispatch_groups=mc.num_dispatch_groups
    )
    ids = [int(v) for v in gidx_host[0, 0]]
    NG = max(ids) + 1
    gidx = ttnn.squeeze(
        ttnn.squeeze(
            ttnn.from_torch(
                gidx_host,
                mesh_mapper=get_ep_mesh_mapper(mesh_device),
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
                dtype=ttnn.uint32,
            ),
            0,
        ),
        0,
    )
    ops = {}
    for name in OPS:
        if name in ("unified", "fused"):
            ops[name] = TtRoutedExpert(
                mesh_device=mesh_device,
                experts_per_chip=E,
                global_expert_idx_table=gidx,
                emb_dim=H,
                hidden_dim=I,
                max_tokens=M,
                torch_weights=w,
                activations_dtype=ttnn.bfloat8_b,
                weights_dtype=ttnn.bfloat4_b,
                activation=ttnn.RoutedExpertActivation.Silu,
                hybrid_token_threshold=None if name == "unified" else M,
            )
        else:
            ops[name] = (FlatExpert if name == "flatpy" else FlatRoutedExpert)(
                mesh_device,
                [
                    [
                        (d["gate_proj"].T.contiguous(), d["up_proj"].T.contiguous(), d["down_proj"].T.contiguous())
                        for d in w
                    ]
                ],
                m=M,
                H=H,
                I=I,
                gids=[ids],
                n_global=NG,
                pin=1,
            )
    q = lambda t, dt: ttnn.to_torch(ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT)).float()
    qw = [tuple(q(d[k].T.contiguous(), ttnn.bfloat4_b) for k in ("gate_proj", "up_proj", "down_proj")) for d in w]
    fw = [tuple(d[k].T.contiguous() for k in ("gate_proj", "up_proj", "down_proj")) for d in w]
    offs = [sum(-(-c // 32) * 32 for c in COUNTS[:e]) for e in range(E)]
    rows_total = offs[-1] + -(-COUNTS[-1] // 32) * 32
    c_h = torch.zeros(1, NG, dtype=torch.int32)
    r_h = torch.zeros(1, NG, dtype=torch.int32)
    for e, gid in enumerate(ids):
        c_h[0, gid], r_h[0, gid] = COUNTS[e], offs[e]
    dev = lambda t, dt, lay: ttnn.from_torch(
        t, dtype=dt, layout=lay, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    counts, regions = dev(c_h, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT), dev(r_h, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    silu = torch.nn.functional.silu
    for case in X_CASES:
        gx = torch.Generator().manual_seed(11)
        x = torch.zeros(rows_total, H)
        for e in range(E):
            x[offs[e] : offs[e] + COUNTS[e]] = _x_case(case, COUNTS[e], H, gx)
        xq = q(x, ttnn.bfloat8_b)
        sel = {
            e: list(range(COUNTS[e]))[:: max(1, COUNTS[e] // ROWS_CHECK)][:ROWS_CHECK] for e in range(E) if COUNTS[e]
        }
        ref_q = torch.cat(
            [
                (silu(xq[offs[e] + torch.tensor(s)] @ qw[e][0]) * (xq[offs[e] + torch.tensor(s)] @ qw[e][1])) @ qw[e][2]
                for e, s in sel.items()
            ]
        )
        ref_f = torch.cat(
            [
                (silu(x[offs[e] + torch.tensor(s)] @ fw[e][0]) * (x[offs[e] + torch.tensor(s)] @ fw[e][1])) @ fw[e][2]
                for e, s in sel.items()
            ]
        )
        for name, op in ops.items():
            if name in ("unified", "fused") and X_TILED:  # (opt-in: the model's pre-tiled bfp8 path)
                y = op(dev(x, ttnn.bfloat8_b, ttnn.TILE_LAYOUT), counts, regions)
            else:  # every op the same input: the row-major bf16 dispatch buffer (each tilizes / packs x itself)
                y = op(dev(x, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT), counts, regions)
            yh = ttnn.to_torch(ttnn.get_device_tensors(y)[0]).float().reshape(-1, H)
            got = torch.cat([yh[offs[e] + torch.tensor(s)] for e, s in sel.items()])
            mq, mf = _metrics(ref_q, got), _metrics(ref_f, got)
            _report(
                f"PREC\t{wset}\t{case}\t{name}{TAG}\tq_pcc {mq['pcc']:.5f}\tq_rel {mq['rel']:.4f}\tq_norm {mq['norm']:.4f}"
                f"\tq_row {mq['row']:.3g}\tf_pcc {mf['pcc']:.5f}\tf_rel {mf['rel']:.4f}\tnonfinite {mq['nonfinite']}"
                f"\t|ref| {float(ref_q.abs().max()):.3g}"
            )
            y.deallocate(True)
