# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Numerics of the persistent-stack port against our HF layer reference: ONE layer's decode step at position 0 (fresh
state) on one chip (the full layer: vtp = 1, no fabric) from the HF golden's input residual, PCC of the output
residual against the golden (reference/hf_layer_reference.py --layers 0-4 --seq-len 1 --num-decode 0 --stream).

    GLM53_MK_GOLDEN=<dir> GLM53_MK_LAYER=0 pytest .../test_mk_layer_check.py -s   # dense KDA layer
    GLM53_MK_GOLDEN=<dir> GLM53_MK_LAYER=4 ...                                    # sparse KDA layer (288 experts)

test_mk_layer_verify_groups (GLM53_SPEC_MULTI items 1 + 2, doc/perf/mtp_multi_user_design.md; written device-free):
one dense KDA layer, U users x (K + 1) verify rows in ONE verify launch (kda_rec REC_CHAIN_GROUPS + kda_pre
KDA_VERIFY_GROUPS for U > 1; U = 1 = the single-user REC_CHAIN / KDA_VERIFY, the knob-off path) vs the plain layer
stepping each user's K + 1 tokens one launch at a time from the same state: per row the output and the candidate
state / conv history equal the plain ones (PCC, max |diff| logged).
    GLM53_MESH_SHAPE=1,1 GLM53_MK_VG_USERS=4 GLM53_MK_VG_K=3 pytest .../test_mk_layer_check.py::test_mk_layer_verify_groups -s
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt.config import DENSE_KDA, SPARSE_KDA, load_hf_text_config
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import OptimizedDecoderConfig
from models.autoports.zai_org_glm_5_3_flash.tt.stack_model import layer_kind, set_knobs
from models.autoports.zai_org_glm_5_3_flash.tt.tp8_layer import TPDenseKDALayer
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir
from models.demos.glm53_flash.tests.test_mk_layer_probe import RawHFSource
from models.demos.glm53_flash.tt.mesh import mesh_device_params

LAYER = int(os.environ.get("GLM53_MK_LAYER", "0"))
GOLDEN = os.environ.get("GLM53_MK_GOLDEN", "/home/ttuser/ashai/glm53/_logs/mesh/hf_golden_t1")
MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "1,1").split(","))
VTP = int(os.environ.get("GLM53_MK_VTP", "0")) or MESH[0] * MESH[1]  # the TP the shards are sized for (1 chip: 1)
STEPS = int(os.environ.get("GLM53_MK_STEPS", "1"))  # golden steps: the first token, then decode0 .. (state carry)
TRACE_MB = int(os.environ.get("GLM53_MODEL_TRACE_MB", "256"))
WORKER_L1 = int(os.environ.get("GLM53_MK_WORKER_L1", "1264512"))


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "device_params",
    [mesh_device_params(*MESH, trace_region_size=TRACE_MB << 20, worker_l1_size=WORKER_L1)],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_mk_layer_check(mesh_device):
    snapshot = default_snapshot_dir()
    cfg = load_hf_text_config(str(snapshot))
    ocfg = OptimizedDecoderConfig()
    kind = layer_kind(cfg, LAYER)
    golds = [torch.load(os.path.join(GOLDEN, f"layer{LAYER}_prefill.pt"))] + [
        torch.load(os.path.join(GOLDEN, f"layer{LAYER}_decode{k}.pt")) for k in range(STEPS - 1)
    ]
    gold = golds[0]
    x_in, x_out = gold["x_in"].float(), gold["x_out"].float()  # [1, T, 4, H]
    assert x_in.shape[1] == 1, f"golden T = {x_in.shape[1]}: this check decodes one token at position 0"
    H = cfg.hidden_size
    if os.environ.get("GLM53_MK_KNOBS", "1") == "1":
        set_knobs(kind == SPARSE_KDA, VTP, 1)
    source = RawHFSource(snapshot, [LAYER])
    t0 = time.perf_counter()
    if kind == DENSE_KDA:
        layer = TPDenseKDALayer(mesh_device, cfg, LAYER, source, ocfg, b=1, probe=False, vtp=VTP)
    elif kind == SPARSE_KDA:
        from models.autoports.zai_org_glm_5_3_flash.tt.tp8_sparse import TPSparseKDALayer, build_moe

        moe = build_moe(mesh_device, cfg, source, LAYER, ocfg)
        layer = TPSparseKDALayer(mesh_device, cfg, LAYER, source, ocfg, b=1, moe=moe, probe=False, vtp=VTP)
    else:
        pytest.skip(f"layer {LAYER} is {kind}: checked through the backend")
    logger.info(f"{kind} layer {LAYER} built in {time.perf_counter() - t0:.0f} s (vtp {VTP})")
    state = layer.attn.allocate_state(1)
    active = layer.masks(torch.ones(1))
    worst = 1.0
    for step, gold in enumerate(golds):  # the same state object through the steps (the KDA conv history / state)
        x_in, x_out = gold["x_in"].float(), gold["x_out"].float()
        # the residual row of the token: the 4 streams side by side [1, 1, 1, 4 H] fp32
        x = layer.replicated(x_in[0, 0].reshape(1, 1, 1, 4 * H).contiguous(), ttnn.float32)
        s1, out, details, _ = layer(x, state.C, active, state.S, t=1)
        ttnn.synchronize_device(mesh_device)
        o = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float().reshape(4, H)
        g = x_out[0, 0].reshape(4, H)
        p = pcc(o, g)
        per_stream = [round(pcc(o[s_], g[s_]), 5) for s_ in range(4)]
        delta = pcc(o - x_in[0, 0].reshape(4, H), g - x_in[0, 0].reshape(4, H))
        logger.info(
            f"MK_LAYER_CHECK: {kind} layer {LAYER} mesh {MESH} vtp {VTP} step {step}: pcc {p:.6f} per stream "
            f"{per_stream}; delta (out - in) pcc {delta:.6f}; golden x_out std {x_out.std():.4f}"
        )
        worst = min(worst, p)
    logger.info(
        f"MK_LAYER_CHECK_SUMMARY: {kind} layer {LAYER} mesh {MESH} vtp {VTP} steps {len(golds)}: worst pcc {worst:.6f}"
    )
    assert worst > 0.98, f"layer {LAYER} output pcc {worst}"


VG_USERS = int(os.environ.get("GLM53_MK_VG_USERS", "4"))  # U users (1: the single-user verify, the knob-off path)
VG_K = int(os.environ.get("GLM53_MK_VG_K", "3"))  # drafts per user: K + 1 verify rows each


def _snap(t) -> torch.Tensor:
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).clone()


def _restore(mesh, t, host: torch.Tensor, dtype) -> None:
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(host, dtype=dtype, layout=ttnn.TILE_LAYOUT,
                                                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)), t)  # fmt: skip


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "device_params",
    [mesh_device_params(*MESH, trace_region_size=TRACE_MB << 20, worker_l1_size=WORKER_L1)],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_mk_layer_verify_groups(mesh_device):
    from models.autoports.zai_org_glm_5_3_flash.tt.spec_mtp import SHARED_ATTN_KNOBS, dram_rows, env

    snapshot = default_snapshot_dir()
    cfg = load_hf_text_config(str(snapshot))
    ocfg = OptimizedDecoderConfig()
    if layer_kind(cfg, LAYER) != DENSE_KDA:
        pytest.skip(f"layer {LAYER}: the verify-groups check runs a dense KDA layer")
    U, K = VG_USERS, VG_K
    g = K + 1
    W = U * g  # the verify rows: user u's group = rows u g .. u g + K (generator.spec_multi_plan)
    assert W <= 16 and (U == 1 or g % 2 == 0), (U, K)
    H = cfg.hidden_size
    if os.environ.get("GLM53_MK_KNOBS", "1") == "1":
        set_knobs(False, VTP, W)
    source = RawHFSource(snapshot, [LAYER])
    layer = TPDenseKDALayer(mesh_device, cfg, LAYER, source, ocfg, b=W, probe=False, vtp=VTP)
    sel = dram_rows(mesh_device, torch.tensor([[u * g for u in range(U)] + [0] * (32 - U)], dtype=torch.int32))
    attn = layer.w.attn
    keep = {a: getattr(attn, a) for a in SHARED_ATTN_KNOBS if hasattr(attn, a)}
    attn.kda_verify_sel = sel
    if U > 1:  # (the user count: kda_pre's KDA_VERIFY_GROUPS = G reads it as an int, kda_rec as a flag)
        attn.kda_verify_groups = U
    try:
        with env(GLM53_REC_UGROUPS=str(U)):
            vlayer = TPDenseKDALayer(mesh_device, cfg, LAYER, source, ocfg, b=W, probe=False, vtp=VTP, weights=layer.w)
    finally:
        del attn.kda_verify_sel
        if U > 1:
            del attn.kda_verify_groups
        for a, v in keep.items():
            setattr(attn, a, v)
    state = layer.attn.allocate_state(32)
    gen = torch.Generator().manual_seed(17)
    scale = 1.0
    gp = os.path.join(GOLDEN, f"layer{LAYER}_prefill.pt")
    if os.path.exists(gp):
        scale = float(torch.load(gp)["x_in"].float().std())
    rows = lambda: torch.randn(W, 4 * H, generator=gen) * scale  # noqa: E731
    run = lambda lyr, x, act: lyr(layer.replicated(x.reshape(1, 1, W, 4 * H).contiguous(), ttnn.float32),  # noqa: E731
                                  state.C, act, state.S, t=W)[1]  # fmt: skip
    all_on = layer.masks(torch.ones(W))
    for _ in range(2):  # warm-up: every row's state / conv history non-trivial (the users sit on rows u g)
        run(layer, rows(), all_on)
    ttnn.synchronize_device(mesh_device)
    S0, C0 = _snap(state.S), _snap(state.C)
    tok = rows()  # verify row u g + j = user u's token j
    # ---- the reference: the plain layer, one token per launch, only the users' rows active
    bases = torch.zeros(W)
    bases[[u * g for u in range(U)]] = 1
    act_users = layer.masks(bases)
    ref_out, ref_S, ref_C = [], [], []
    for j in range(g):
        x = torch.zeros(W, 4 * H)
        for u in range(U):
            x[u * g] = tok[u * g + j]
        out = run(layer, x, act_users)
        ttnn.synchronize_device(mesh_device)
        o, S_, C_ = _snap(out).float().reshape(W, 4 * H), _snap(state.S), _snap(state.C)
        ref_out.append([o[u * g] for u in range(U)])
        ref_S.append([S_[u * g] for u in range(U)])
        ref_C.append([C_[:, :, u * g] for u in range(U)])
    # ---- the verify: ONE launch of every user's K + 1 rows from the restored state
    _restore(mesh_device, state.S, S0, ttnn.float32)
    _restore(mesh_device, state.C, C0, ttnn.bfloat16)
    out = run(vlayer, tok, layer.masks(torch.ones(W)))
    ttnn.synchronize_device(mesh_device)
    vo, vS, vC = _snap(out).float().reshape(W, 4 * H), _snap(state.S), _snap(state.C)
    worst = 1.0
    for u in range(U):
        for j in range(g):
            r = u * g + j
            po = pcc(vo[r], ref_out[j][u])
            ps = pcc(vS[r], ref_S[j][u])
            pc = pcc(vC[:, :, r], ref_C[j][u])
            dmax = float((vo[r] - ref_out[j][u]).abs().max())
            logger.info(f"MK_VERIFY_GROUPS: U {U} K {K} user {u} row {j}: out pcc {po:.6f} max|d| {dmax:.3e}, "
                        f"state pcc {ps:.6f}, conv pcc {pc:.6f}")  # fmt: skip
            worst = min(worst, po, ps, pc)
    logger.info(f"MK_VERIFY_GROUPS_SUMMARY: layer {LAYER} mesh {MESH} U {U} K {K} rows {W}: worst pcc {worst:.6f}")
    assert worst > 0.9999, f"verify groups vs plain: worst pcc {worst}"
