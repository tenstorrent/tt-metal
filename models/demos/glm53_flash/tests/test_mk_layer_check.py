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
GLM53_MK_VG_LAYERS=4,5 (one KDA kind, dense or sparse): the layers as ONE stack program per launch (the backend's
verify stacks). GLM53_REC_SPLIT_VERIFY=2: also the verify (and the plain launch) with the kda_rec value-column split,
asserted BYTE-identical to spread 1 on every chip (outputs, state, conv history), after a split-1 determinism control.
    GLM53_MESH_SHAPE=8,4 GLM53_MK_VTP=32 GLM53_MK_VG_USERS=8 GLM53_MK_VG_K=1 GLM53_MK_VG_LAYERS=4,5 \
      GLM53_REC_SPLIT_VERIFY=2 pytest .../test_mk_layer_check.py::test_mk_layer_verify_groups -s
The split's odd-item cases: an odd item count per core (one user, K 2: 3 chained rows; the plain t = 3 launch on 3
user groups) and user groups that do not divide the plain rows (GLM53_MK_VG_PLAIN_UGROUPS=3 on W = 8: 3 / 3 / 2 rows):
    GLM53_MESH_SHAPE=8,4 GLM53_MK_VTP=32 GLM53_MK_VG_USERS=1 GLM53_MK_VG_K=2 GLM53_MK_VG_LAYERS=4,5 \
      GLM53_REC_SPLIT_VERIFY=2 pytest .../test_mk_layer_check.py::test_mk_layer_verify_groups -s
    GLM53_MESH_SHAPE=8,4 GLM53_MK_VTP=32 GLM53_MK_VG_USERS=4 GLM53_MK_VG_K=1 GLM53_MK_VG_LAYERS=4,5 \
      GLM53_MK_VG_PLAIN_UGROUPS=3 GLM53_REC_SPLIT_VERIFY=2 \
      pytest .../test_mk_layer_check.py::test_mk_layer_verify_groups -s
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
# GLM53_MK_VG_LAYERS = "4,5" (KDA layers of one kind): ONE stack program per launch (TPStack / TPSparseStack, as the
# backend's verify runs; a stack repeats every phase per layer inside one program); default: GLM53_MK_LAYER alone as
# its own layer program (the original check)
VG_LAYERS = [int(v) for v in os.environ.get("GLM53_MK_VG_LAYERS", str(LAYER)).split(",")]
# with GLM53_REC_SPLIT_VERIFY = 2 (tt/spec_mtp.py rec_split): also the verify with the kda_rec value-column split; it
# must be BYTE-identical to the split-1 verify (outputs, every layer's state and conv history, every chip), after a
# determinism control (the split-1 verify twice); GLM53_MK_VG_PLAIN_SPLIT=1 (default on with the split): the plain
# t = W launch with the split too (its GLM53_REC_UGROUPS groups x 2 cores per head), byte-identical to the plain one
VG_PLAIN_SPLIT = os.environ.get("GLM53_MK_VG_PLAIN_SPLIT", "1") == "1"
# GLM53_MK_VG_PLAIN_UGROUPS = u: the plain t = W programs (and plain_split) on u kda_rec user groups instead of
# set_knobs' pick for the width -- a u that does not divide W gives a short (or an empty) last group
VG_PLAIN_UG = os.environ.get("GLM53_MK_VG_PLAIN_UGROUPS")


def _snap(t) -> torch.Tensor:
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).clone()


def _snap_all(t) -> torch.Tensor:
    """every chip's copy (the byte-equality checks)"""
    return torch.stack([ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)])


def _restore(mesh, t, host: torch.Tensor, dtype) -> None:
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(host, dtype=dtype, layout=ttnn.TILE_LAYOUT,
                                                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)), t)  # fmt: skip


def _same(a: tuple, b: tuple) -> tuple:
    """(identical, [(what, max |diff|)] of the differing pieces) of two (out, [S per layer], [C per layer]) snapshots"""
    pieces = [("out", a[0], b[0])] + [(f"S{i}", x, y) for i, (x, y) in enumerate(zip(a[1], b[1]))]
    pieces += [(f"C{i}", x, y) for i, (x, y) in enumerate(zip(a[2], b[2]))]
    bits = lambda x: x.contiguous().view(torch.int16 if x.element_size() == 2 else torch.int32)  # noqa: E731 (NaN-safe)
    diff = lambda x, y: float((x.float() - y.float()).nan_to_num(0.0, 1e30, -1e30).abs().max())  # noqa: E731
    bad = [(n, diff(x, y)) for n, x, y in pieces if x.shape != y.shape or not torch.equal(bits(x), bits(y))]
    return not bad, bad


@pytest.mark.timeout(7200)  # (a sparse_kda stack: the layers' raw weights + EP packing, 4 stack programs)
@pytest.mark.parametrize(
    "device_params",
    [mesh_device_params(*MESH, trace_region_size=TRACE_MB << 20, worker_l1_size=WORKER_L1)],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_mk_layer_verify_groups(mesh_device):
    from models.autoports.zai_org_glm_5_3_flash.tt.spec_mtp import SHARED_ATTN_KNOBS, dram_rows, env, rec_split

    snapshot = default_snapshot_dir()
    cfg = load_hf_text_config(str(snapshot))
    ocfg = OptimizedDecoderConfig()
    kinds = {layer_kind(cfg, L) for L in VG_LAYERS}
    if len(kinds) != 1 or not kinds <= {DENSE_KDA, SPARSE_KDA}:
        pytest.skip(f"layers {VG_LAYERS} ({sorted(kinds)}): the verify-groups check runs KDA layers of one kind")
    sparse, stacked, split = kinds == {SPARSE_KDA}, len(VG_LAYERS) > 1, rec_split()
    U, K = VG_USERS, VG_K
    g = K + 1
    W = U * g  # the verify rows: user u's group = rows u g .. u g + K (generator.spec_multi_plan)
    assert W <= 16 and (U == 1 or g % 2 == 0), (U, K)
    H = cfg.hidden_size
    if os.environ.get("GLM53_MK_KNOBS", "1") == "1":
        set_knobs(sparse, VTP, W)
    if VG_PLAIN_UG:  # (after set_knobs: the layers' construction and first builds read it; the verify sets its own)
        os.environ["GLM53_REC_UGROUPS"] = VG_PLAIN_UG
    source = RawHFSource(snapshot, VG_LAYERS)
    t0 = time.perf_counter()
    if sparse:
        from models.autoports.zai_org_glm_5_3_flash.tt.ep_experts import build_moe_ep
        from models.autoports.zai_org_glm_5_3_flash.tt.tp8_sparse import TPSparseKDALayer, build_moe

        # the backend's expert weights on a mesh: EP-sharded (StackBackend ep_weights), replicated on one chip
        moe_of = build_moe_ep if MESH[0] * MESH[1] > 1 else build_moe
        moes = {L: moe_of(mesh_device, cfg, source, L, ocfg) for L in VG_LAYERS}

    def make(L, **kw):
        if sparse:
            return TPSparseKDALayer(mesh_device, cfg, L, source, ocfg, b=W, moe=moes[L], probe=False, vtp=VTP, **kw)
        return TPDenseKDALayer(mesh_device, cfg, L, source, ocfg, b=W, probe=False, vtp=VTP, **kw)

    tps = []
    for L in VG_LAYERS:  # one fabric phase for every program (the backend shares a width's phase likewise)
        tps.append(make(L, fabric=tps[0].fabric if tps else None))
    layer = tps[0]
    sel = dram_rows(mesh_device, torch.tensor([[u * g for u in range(U)] + [0] * (32 - U)], dtype=torch.int32))

    def variant(spread: int, verify: bool) -> list:
        """the resident layers' programs at this kda_rec spread (verify: REC_CHAIN(_GROUPS) + KDA_VERIFY, U user groups;
        else the plain t = W programs) — knobs set on the shared attention objects and restored, as spec_mtp does"""
        out = []
        for L, tp in zip(VG_LAYERS, tps):
            attn = tp.w.attn
            keep = {a: getattr(attn, a) for a in SHARED_ATTN_KNOBS if hasattr(attn, a)}
            if verify:
                attn.kda_verify_sel = sel
                if U > 1:  # (the user count: kda_pre's KDA_VERIFY_GROUPS = G reads it as an int, kda_rec as a flag)
                    attn.kda_verify_groups = U
            try:
                attn.kda_rec_spread = spread
                with env(GLM53_REC_UGROUPS=str(U) if verify else os.environ.get("GLM53_REC_UGROUPS", "1")):
                    out.append(make(L, weights=tp.w, fabric=layer.fabric))
            finally:
                if verify:
                    del attn.kda_verify_sel
                    if U > 1:
                        del attn.kda_verify_groups
                for a, v in keep.items():
                    setattr(attn, a, v)
            assert out[-1].mk.rec.spread == spread and out[-1].mk.kda.rec_spread == spread, (L, spread)
        return out

    progs = {"plain": tps, "verify": variant(1, True)}
    if split > 1:
        progs["verify_split"] = variant(split, True)
        if VG_PLAIN_SPLIT:
            progs["plain_split"] = variant(split, False)
    ugs = {k: ls[0].mk.rec.ug for k, ls in progs.items()}  # (the kda_rec user groups of each program)
    logger.info(f"{sorted(kinds)} layers {VG_LAYERS} built in {time.perf_counter() - t0:.0f} s: {sorted(progs)}, "
                f"kda_rec user groups {ugs}, rows {W}")  # fmt: skip
    if VG_PLAIN_UG:
        assert all(ugs[k] == int(VG_PLAIN_UG) for k in ("plain", "plain_split") if k in ugs), ugs
    states = [tp.attn.allocate_state(32) for tp in tps]
    hists, ss = [s.C for s in states], [s.S for s in states]
    all_on = layer.masks(torch.ones(W))
    if stacked:
        from models.autoports.zai_org_glm_5_3_flash.tt.mk_stack import AllocPool
        from models.autoports.zai_org_glm_5_3_flash.tt.tp_stack import TPSparseStack, TPStack

        pool = AllocPool()  # one L1 set of intermediates for every stack (they never run concurrently)
        cls = TPSparseStack if sparse else TPStack
        progs = {k: cls(mesh_device, [v.mk for v in ls], ls[0], t=W, pool=pool) for k, ls in progs.items()}
        for st in progs.values():  # every program built (pool / persistent L1 allocated) before the first launch;
            st.build(hists, all_on, ss)  # (a stack bakes its mask tensor in: every launch runs all W rows)
        logger.info(f"stacks: {[(k, st.info.get('sem_x')) for k, st in progs.items()]}")
    else:
        progs = {k: ls[0] for k, ls in progs.items()}

    def run(name, x, act):
        p = progs[name]
        if stacked:
            res = p.residual()
            _restore(mesh_device, res, x.reshape(tuple(res.shape)).contiguous(), ttnn.float32)
            return p(hists, act, ss)
        x = layer.replicated(x.reshape(1, 1, W, 4 * H).contiguous(), ttnn.float32)
        return p(x, states[0].C, act, states[0].S, t=W)[1]

    gen = torch.Generator().manual_seed(17)
    scale = 1.0
    gp = os.path.join(GOLDEN, f"layer{VG_LAYERS[0]}_prefill.pt")
    if os.path.exists(gp):
        scale = float(torch.load(gp)["x_in"].float().std())
    rows = lambda: torch.randn(W, 4 * H, generator=gen) * scale  # noqa: E731
    for _ in range(2):  # warm-up: every row's state / conv history non-trivial (the users sit on rows u g)
        run("plain", rows(), all_on)
    ttnn.synchronize_device(mesh_device)
    S0, C0 = [_snap(s.S) for s in states], [_snap(s.C) for s in states]
    tok = rows()  # verify row u g + j = user u's token j

    def restore():
        for st, s0, c0 in zip(states, S0, C0):
            _restore(mesh_device, st.S, s0, ttnn.float32)
            _restore(mesh_device, st.C, c0, ttnn.bfloat16)

    def launch(name, x, act) -> tuple:
        """(out, [S], [C]) of every chip after one launch from the restored state"""
        restore()
        out = run(name, x, act)
        ttnn.synchronize_device(mesh_device)
        return _snap_all(out), [_snap_all(s.S) for s in states], [_snap_all(s.C) for s in states]

    # ---- the reference: the plain programs, one token per launch (single layer: only the users' rows active; a
    # stack runs every row, the other rows' inputs zero -- rows are independent)
    bases = torch.zeros(W)
    bases[[u * g for u in range(U)]] = 1
    act_ref = all_on if stacked else layer.masks(bases)
    restore()
    ref_out, ref_S, ref_C = [], [], []
    for j in range(g):
        x = torch.zeros(W, 4 * H)
        for u in range(U):
            x[u * g] = tok[u * g + j]
        out = run("plain", x, act_ref)
        ttnn.synchronize_device(mesh_device)
        o = _snap(out).float().reshape(W, 4 * H)
        S_, C_ = [_snap(s.S) for s in states], [_snap(s.C) for s in states]
        ref_out.append([o[u * g] for u in range(U)])
        ref_S.append([[S_[i][u * g] for u in range(U)] for i in range(len(states))])
        ref_C.append([[C_[i][:, :, u * g] for u in range(U)] for i in range(len(states))])
    # ---- every launch below from the same restored state; a launch never follows a later program's first build
    # (single-layer programs allocate on their first call): plain, verify, verify again, the split variants
    snaps = {}
    if "plain_split" in progs:
        snaps["plain"] = launch("plain", tok, all_on)
    snaps["verify"] = launch("verify", tok, all_on)
    # ---- the verify: ONE launch of every user's K + 1 rows vs the plain per-token launches (PCC, chip 0)
    vo = snaps["verify"][0][0].float().reshape(W, 4 * H)
    worst = 1.0
    for u in range(U):
        for j in range(g):
            r = u * g + j
            po = pcc(vo[r], ref_out[j][u])
            ps = min(pcc(snaps["verify"][1][i][0][r], ref_S[j][i][u]) for i in range(len(states)))
            pc = min(pcc(snaps["verify"][2][i][0][:, :, r], ref_C[j][i][u]) for i in range(len(states)))
            dmax = float((vo[r] - ref_out[j][u]).abs().max())
            logger.info(f"MK_VERIFY_GROUPS: U {U} K {K} user {u} row {j}: out pcc {po:.6f} max|d| {dmax:.3e}, "
                        f"state pcc {ps:.6f}, conv pcc {pc:.6f}")  # fmt: skip
            worst = min(worst, po, ps, pc)
    logger.info(
        f"MK_VERIFY_GROUPS_SUMMARY: layers {VG_LAYERS} ({'stack' if stacked else 'layer'}) mesh {MESH} U {U} K {K} "
        f"rows {W}: worst pcc {worst:.6f}"
    )
    verdicts = []
    if split > 1:
        ok, bad = _same(snaps["verify"], launch("verify", tok, all_on))
        logger.info(f"MK_VERIFY_SPLIT: control (split-1 verify twice) identical {ok} {bad}")
        verdicts.append(("control", ok, bad))
        for name, ref in (("verify_split", "verify"), ("plain_split", "plain")):
            if name in progs:
                ok, bad = _same(snaps[ref], launch(name, tok, all_on))
                logger.info(f"MK_VERIFY_SPLIT: {name} vs {ref} (spread {split} vs 1, layers {VG_LAYERS}, "
                            f"{'stack' if stacked else 'layer'}, every chip) identical {ok} {bad}")  # fmt: skip
                verdicts.append((name, ok, bad))
        logger.info(f"MK_VERIFY_SPLIT_SUMMARY: {[(n, ok) for n, ok, _ in verdicts]}")
    assert worst > 0.9999, f"verify groups vs plain: worst pcc {worst}"
    assert all(ok for _, ok, _ in verdicts), f"GLM53_REC_SPLIT_VERIFY={split}: not byte-identical {verdicts}"
