# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Scratch: routed-expert precision at the production model shapes, one chip, the same seeded bf16 weights and inputs
for every op (the file runs unchanged in two worktrees: origin/main for the production op, this branch for the flat
expert). Dumps each op's output rows; zz_expert_precision_report.py grades them all against one reference.

  prod        TtRoutedExpert as the model's TtMoe builds it on main (bfp8 activations, bfp4 weights, LoFi, bf16 DEST,
              packer L1 accumulation; the model's hybrid threshold: GLM 5.3 / K2.7 experts with >= 320 rows on
              moe_fused_swiglu, the rest unified_routed_expert_moe; K3 unified only)
  flat        flat_routed_expert as the token-split K3 block runs it (fp32 full-sync down; bfp8 h)
  flat_hbf16  the same with bf16 h (the h transfer's share)
Models (MODELS=glm53,k27,k3): GLM 5.3 6144 x 2048 SiLU, Kimi K2.7 7168 x 2048 SiLU, Kimi K3 3584 x 3072 SiTU."""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping, extract_mesh_config, get_ep_mesh_mapper
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert

MODELS = {  # name: (H, I, activation, hybrid threshold)
    "glm53": (6144, 2048, "silu", 320),
    "k27": (7168, 2048, "silu", 320),
    "k3": (3584, 3072, "situ", None),
}
M = 512  # capacity per expert
COUNTS = [512, 1, 31, 33, 0, 200, 64, 384]
E = len(COUNTS)
ROWS_CHECK = 96
X_CASES = ["normal", "normal_x100", "outlier_channels", "row_scales", "near_one_hot", "student_t2"]
W_STD = 0.02


def x_case(name, n, h, g):
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    if name == "normal":
        return r(n, h)
    if name == "normal_x100":
        return r(n, h) * 100
    if name == "outlier_channels":
        x = r(n, h)
        x[:, torch.randperm(h, generator=g)[:8]] *= 100
        return x
    if name == "row_scales":
        return r(n, h) * (10.0 ** (torch.rand(n, 1, generator=g) * 6 - 3))
    if name == "near_one_hot":
        x = r(n, h) * 1e-3
        x[torch.arange(n), torch.randint(0, h, (n,), generator=g)] = 10.0
        return x
    if name == "student_t2":
        return torch.distributions.StudentT(2.0).sample((n, h)).clamp(-1e3, 1e3)
    raise ValueError(name)


def weights(model):
    """bf16-valued weights (nn.Linear layout: gate / up [I, H], down [H, I]): every upload path rounds them alike."""
    H, I, _, _ = MODELS[model]
    g = torch.Generator().manual_seed(7)
    mk = lambda *s: (torch.randn(*s, generator=g) * W_STD).bfloat16().float()  # noqa: E731
    return [{"gate_proj": mk(I, H), "up_proj": mk(I, H), "down_proj": mk(H, I)} for _ in range(E)]


def layout():
    offs = [sum(-(-c // 32) * 32 for c in COUNTS[:e]) for e in range(E)]
    rows = offs[-1] + -(-COUNTS[-1] // 32) * 32
    sel = {e: list(range(COUNTS[e]))[:: max(1, COUNTS[e] // ROWS_CHECK)][:ROWS_CHECK] for e in range(E) if COUNTS[e]}
    return offs, rows, sel


def inputs(model, case):
    H = MODELS[model][0]
    offs, rows, _ = layout()
    gx = torch.Generator().manual_seed(11)
    x = torch.zeros(rows, H)
    for e in range(E):
        x[offs[e] : offs[e] + COUNTS[e]] = x_case(case, COUNTS[e], H, gx)
    return x.bfloat16().float()


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("model", [m for m in os.environ.get("MODELS", "glm53,k27,k3").split(",") if m])
def test_expert_precision_models(mesh_device, device_params, model):
    H, I, act, thr = MODELS[model]
    ops_wanted = os.environ.get("PREC_OPS", "prod").split(",")
    w = weights(model)
    mc = extract_mesh_config(mesh_device)
    gidx_host = ExpertMapping.create_global_expert_idx_table(
        experts_per_chip=E, dispatch_group_size=mc.dispatch_group_size, num_dispatch_groups=mc.num_dispatch_groups
    )
    ids = [int(v) for v in gidx_host[0, 0]]
    NG = max(ids) + 1
    gidx = ttnn.squeeze(ttnn.squeeze(ttnn.from_torch(
        gidx_host, mesh_mapper=get_ep_mesh_mapper(mesh_device), layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device,
        dtype=ttnn.uint32), 0), 0)
    ops = {}
    for name in ops_wanted:
        if name == "prod":
            ops[name] = TtRoutedExpert(
                mesh_device=mesh_device, experts_per_chip=E, global_expert_idx_table=gidx, emb_dim=H, hidden_dim=I,
                max_tokens=M, torch_weights=w, activations_dtype=ttnn.bfloat8_b, weights_dtype=ttnn.bfloat4_b,
                activation={"silu": ttnn.RoutedExpertActivation.Silu, "situ": ttnn.RoutedExpertActivation.SituGlu}[act],
                hybrid_token_threshold=thr,
            )
        else:  # the flat expert (this branch only)
            from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import FlatRoutedExpert

            ops[name] = FlatRoutedExpert(
                mesh_device,
                [[(d["gate_proj"].T.contiguous(), d["up_proj"].T.contiguous(), d["down_proj"].T.contiguous()) for d in w]],
                m=M, H=H, I=I, gids=[ids], n_global=NG, wdtype="bf4", act=act, pin=1, h_bf16=name == "flat_hbf16",
            )
    offs, rows, sel = layout()
    c_h = torch.zeros(1, NG, dtype=torch.int32)
    r_h = torch.zeros(1, NG, dtype=torch.int32)
    for e, gid in enumerate(ids):
        c_h[0, gid], r_h[0, gid] = COUNTS[e], offs[e]
    dev = lambda t, dt, lay: ttnn.from_torch(t, dtype=dt, layout=lay, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # noqa: E731
    counts, regions = dev(c_h, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT), dev(r_h, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    out = {}
    for case in X_CASES:
        x = inputs(model, case)
        for name, op in ops.items():
            xt = dev(x, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)  # (every op: the row-major bf16 dispatch buffer)
            y = op(xt, counts, regions, down_fp32=True) if name.startswith("flat") else op(xt, counts, regions)
            yh = ttnn.to_torch(ttnn.get_device_tensors(y)[0]).float().reshape(-1, H)
            out[(name, case)] = torch.cat([yh[offs[e] + torch.tensor(s)] for e, s in sel.items()])
            logger.info(f"PRECM {model} {name} {case}: rows {tuple(out[(name, case)].shape)} |y| max {out[(name, case)].abs().max().item():.3g}")
            y.deallocate(True)
            xt.deallocate(True)
    path = os.environ.get("PREC_DUMP")
    if path:
        torch.save(out, f"{path}_{model}.pt")
        logger.info(f"PRECM dump {path}_{model}.pt")
