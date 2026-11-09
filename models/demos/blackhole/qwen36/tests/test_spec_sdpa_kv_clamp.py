# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""paged SDPA decode ``clamp_kv_tail`` (QWEN36_SPEC_SDPA_KV_CLAMP; on by default with QWEN36_SPEC_ADAPTIVE_K=1).

The dynamic k-chunk is 128 rows (two 64-row blocks) from position 64 on and the last chunk ends on a chunk boundary, so for
a group whose last position p_max has p_max % 128 < 64 (p_max >= 128) the last chunk's second block lies past the block
holding p_max. Every row of that block is masked to -inf for every q row, yet the legacy reader resolves its page-table
entry. Inside a page-table row the entry is the null block or a stale valid block (finite K/V, masked to an exact zero);
one entry past a row's end it is the row-major page's DRAM padding, which the host never writes (stale command-queue
bytes): a wild block id whose K/V can be Inf / NaN, and Inf - inf = NaN poisons the softmax of every q row of the group.

  test_kv_tail_clamp_poison      deterministic version of that failure: rows 16 entries wide (64 B, no page padding),
                                 entries past each user's real blocks point at a POISON block whose K/V are NaN (bf16
                                 cache: exact). Geometries with p_max % 128 < 64 resolve the poison entry without the
                                 clamp. clamp=on must match torch; clamp=off is the control (expected corrupted, i.e.
                                 the test exercises the entry). Spec packed T = 4 (r5) and T = 5 / 6 / 8 (adaptive-K wide
                                 family, one group per user), and the non-spec decode the drafter runs.
  test_kv_tail_clamp_identical   the serving layout (rows spec_table_width = 4097 entries, zero-padded; null block 0
                                 zeroed; bf8 cache): clamp on vs off bit-identical on every chip (masked columns
                                 contribute exactly zero, whatever finite K/V sit under them), both close to torch. r5's
                                 T = 4 buckets 1 / 2 / 4 / 8, the wide family T = 5 / 6 / 8, the T = 8 split, and the
                                 non-spec decode, at short bases on both sides of every 128-row boundary and at 32K.

Run (inside the tt-metal container, on a Blackhole Galaxy / P150x8 TP mesh):
    source /tmp/penv.sh
    pytest models/demos/blackhole/qwen36/tests/test_spec_sdpa_kv_clamp.py -v -s
"""
import math

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import parametrize_mesh_tp
from models.demos.blackhole.qwen36.tt.spec_engine import spec_table_width

BLOCK, NH, HD = 64, 3, 256
PROD_WIDTH = spec_table_width(262144, 7, BLOCK)  # 4097
POISON_WIDTH = 16  # 16 int32 entries = 64 B: one DRAM-aligned page, no padding the host leaves unwritten

# (tag, users, T, Tg, cap, bases); Tg = 0: the non-spec paged decode (the drafter's call: one q row-tile per user)
POISON_CASES = [
    ("r5-T4-u2", 2, 4, 4, 64, [127, 297]),  # p_max 130 / 300: entries 3 / 5 past 3 / 5 real blocks
    ("r5-T4-u1", 1, 4, 4, 64, [517]),  # p_max 520: entry 9
    ("wide-T6-u1", 1, 6, 6, 64, [125]),  # p_max 130
    ("wide-T5-u4", 4, 5, 5, 64, [126, 296, 516, 1]),  # p_max 130 / 300 / 520 / 5
    ("wide-T8-u2", 2, 8, 8, 64, [515, 127]),  # p_max 522 / 134
    ("split-T8-u2", 2, 8, 4, 55, [515, 127]),
    ("decode-u2", 2, 1, 0, 64, [130, 300]),
]
SHORT8 = [127, 28, 1000, 0, 125, 30, 200, 3]
LONG4 = [32760, 32289, 30000, 31999]
IDENT_CASES = [
    ("r5-T4-u8", 8, 4, 4, 64, SHORT8),
    ("r5-T4-u4", 4, 4, 4, 64, SHORT8[:4]),
    ("r5-T4-u2", 2, 4, 4, 64, [127, 28]),
    ("r5-T4-u1", 1, 4, 4, 64, [127]),
    ("r5-T4-u2-b", 2, 4, 4, 64, [253, 382]),  # p_max 256 / 385
    ("r5-T4-u4-32k", 4, 4, 4, 64, LONG4),
    ("wide-T6-u4", 4, 6, 6, 64, SHORT8[:4]),
    ("wide-T6-u2", 2, 6, 6, 64, [127, 28]),
    ("wide-T5-u1", 1, 5, 5, 64, [127]),
    ("wide-T8-u1", 1, 8, 8, 64, [125]),
    ("wide-T6-u4-32k", 4, 6, 6, 64, LONG4),
    ("split-T8-u2", 2, 8, 4, 55, [127, 28]),
    ("decode-u8", 8, 1, 0, 64, [130, 28, 1000, 0, 300, 64, 520, 3]),
]


def _pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    d = a.norm() * b.norm()
    return float((a @ b) / d) if d > 0 else float(torch.equal(a, b))


def _rep(mesh, t, dtype, layout):
    return ttnn.from_torch(
        t, dtype=dtype, layout=layout, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh), memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _all(mesh, t):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))


def _setup(mesh, users, T, Tg, cap, bases, layout, kv_dtype, seed):
    """Inputs of one call. layout "poison": POISON_WIDTH-wide rows, entries past the real blocks = the poison block (last
    pool block, K/V NaN). layout "prod": PROD_WIDTH-wide rows zero-padded, null block 0 zeroed."""
    nb_real = math.ceil((max(bases) + T) / BLOCK)
    g = torch.Generator().manual_seed(seed)
    if layout == "poison":
        assert nb_real < POISON_WIDTH
        nblocks = users * nb_real + 1
        poison = nblocks - 1
        real = torch.randperm(users * nb_real, generator=g).reshape(users, nb_real).to(torch.int32)
        pt_u = torch.full((users, POISON_WIDTH), poison, dtype=torch.int32)
        pt_u[:, :nb_real] = real
    else:
        nblocks = 1 + users * nb_real
        real = (1 + torch.randperm(nblocks - 1, generator=g)).reshape(users, nb_real).to(torch.int32)
        pt_u = torch.zeros(users, PROD_WIDTH, dtype=torch.int32)
        pt_u[:, :nb_real] = real
    k = torch.randn(nblocks, 1, BLOCK, HD, generator=g).to(torch.bfloat16)
    v = torch.randn(nblocks, 1, BLOCK, HD, generator=g).to(torch.bfloat16)
    if layout == "poison":
        k[-1] = float("nan")
        v[-1] = float("nan")
    else:
        k[0].zero_()
        v[0].zero_()
    spec = Tg > 0
    if spec:
        G = users * T // Tg
        pos = torch.tensor([bases[u] + j for u in range(users) for j in range(T)], dtype=torch.int32)
        q = torch.randn(1, users * T, 32, HD, generator=g).to(torch.bfloat16)
        q_dev = q.reshape(1, G, Tg * 32, HD)
        pt = pt_u.repeat_interleave(T // Tg, dim=0)
    else:  # the drafter's non-spec call: one row per user, NH valid q heads
        assert T == 1
        pos = torch.tensor(bases, dtype=torch.int32)
        q = torch.randn(1, users, 32, HD, generator=g).to(torch.bfloat16)
        q_dev = q[:, :, :NH, :]
        pt = pt_u
        G = users
    grid = mesh.compute_with_storage_grid_size()
    cfg = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        exp_approx_mode=False,
        q_chunk_size=0,
        k_chunk_size=0,
        max_cores_per_head_batch=min(cap, max(1, (grid.x * grid.y) // G)),
    )
    dev = dict(
        k=_rep(mesh, k, kv_dtype, ttnn.TILE_LAYOUT),
        v=_rep(mesh, v, kv_dtype, ttnn.TILE_LAYOUT),
        pos=_rep(mesh, pos, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
        pt=_rep(mesh, pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
        q=_rep(mesh, q_dev, ttnn.bfloat16, ttnn.TILE_LAYOUT),
    )
    return dict(dev=dev, cfg=cfg, q=q, real=real, nb_real=nb_real, spec=spec, Tg=Tg, nblocks=nblocks)


def _call(s, clamp):
    d = s["dev"]
    kw = dict(page_table_tensor=d["pt"], cur_pos_tensor=d["pos"], scale=HD**-0.5, program_config=s["cfg"])
    kw["memory_config"] = ttnn.L1_MEMORY_CONFIG
    if s["spec"]:
        kw.update(spec_multi_pos_tiles=s["Tg"], spec_pack_heads=NH)
    if clamp:
        kw["clamp_kv_tail"] = True
    return ttnn.transformer.paged_scaled_dot_product_attention_decode(d["q"], d["k"], d["v"], **kw)


def _outputs(mesh, s, o, users, T):
    """[chips, users, T, NH, HD] float."""
    a = _all(mesh, o).float()
    nd = mesh.get_num_devices()
    if s["spec"]:
        return a.reshape(nd, users, T, 32, HD)[:, :, :, :NH, :]
    return a.reshape(nd, users, -1, HD)[:, :, :NH, :].reshape(nd, users, 1, NH, HD)


def _reference(mesh, s, users, T, bases):
    kd = _all(mesh, s["dev"]["k"])[: s["nblocks"]].float()
    vd = _all(mesh, s["dev"]["v"])[: s["nblocks"]].float()
    nb = s["nb_real"]
    if s["spec"]:
        qv = s["q"].reshape(users, T, 32, HD)[:, :, :NH, :].float()
    else:
        qv = s["q"][0, :, :NH, :].float().reshape(users, 1, NH, HD)
    ref = torch.zeros(users, T, NH, HD)
    for u in range(users):
        k_u = kd[s["real"][u].long()].reshape(nb * BLOCK, HD)
        v_u = vd[s["real"][u].long()].reshape(nb * BLOCK, HD)
        for j in range(T):
            p = bases[u] + j
            ref[u, j] = torch.softmax((qv[u, j] @ k_u[: p + 1].T) * HD**-0.5, dim=-1) @ v_u[: p + 1]
    return ref


def _free(s):
    for t in s["dev"].values():
        ttnn.deallocate(t)


def _min_pcc(out, ref):
    """Worst PCC over (chip, user, candidate) after nan_to_num, and the NaN count."""
    worst = 1.0
    for d in range(out.shape[0]):
        for u in range(out.shape[1]):
            for j in range(out.shape[2]):
                worst = min(worst, _pcc(torch.nan_to_num(out[d, u, j]), ref[u, j]))
    return worst, int(torch.isnan(out).sum())


@torch.no_grad()
@parametrize_mesh_tp()
@pytest.mark.parametrize("clamp", [True, False], ids=["clamp", "control-noclamp"])
@pytest.mark.parametrize("tag, users, T, Tg, cap, bases", POISON_CASES, ids=[c[0] for c in POISON_CASES])
def test_kv_tail_clamp_poison(mesh_device, tag, users, T, Tg, cap, bases, clamp, reset_seeds, ensure_gc):
    s = _setup(mesh_device, users, T, Tg, cap, bases, "poison", ttnn.bfloat16, seed=101 + users)
    o = _call(s, clamp)
    out = _outputs(mesh_device, s, o, users, T)
    ttnn.deallocate(o)
    ref = _reference(mesh_device, s, users, T, bases)
    worst, nans = _min_pcc(out, ref)
    oob = [b + T - 1 for b in bases if b + T - 1 >= 128 and (b + T - 1) % 128 < 64]
    logger.info(
        f"[kv-clamp] poison {tag} clamp={clamp}: worst PCC vs torch {worst:.6f}, NaN {nans}, max|x| "
        f"{float(torch.nan_to_num(out).abs().max()):.3e}; p_max reaching the poison entry {oob}"
    )
    _free(s)
    assert oob, f"[{tag}] no group reaches past its real blocks: the case does not exercise the clamp"
    if clamp:
        assert nans == 0 and worst >= 0.99, f"[{tag}] clamp_kv_tail: worst PCC {worst}, NaN {nans}"
    else:
        assert nans > 0 or worst < 0.9, f"[{tag}] control without the clamp did not read the poison block ({worst}, {nans})"


@torch.no_grad()
@parametrize_mesh_tp()
@pytest.mark.parametrize("tag, users, T, Tg, cap, bases", IDENT_CASES, ids=[c[0] for c in IDENT_CASES])
def test_kv_tail_clamp_identical(mesh_device, tag, users, T, Tg, cap, bases, reset_seeds, ensure_gc):
    s = _setup(mesh_device, users, T, Tg, cap, bases, "prod", ttnn.bfloat8_b, seed=202 + users)
    outs = {}
    for clamp in (False, True, False, True):
        o = _call(s, clamp)
        outs.setdefault(clamp, []).append(_outputs(mesh_device, s, o, users, T))
        ttnn.deallocate(o)
    ref = _reference(mesh_device, s, users, T, bases)
    _free(s)
    off, on = outs[False][0], outs[True][0]
    det = all(torch.equal(outs[c][0], outs[c][1]) for c in (False, True))
    same = bool(torch.equal(off, on))
    xchip = all(torch.equal(on[0], on[d]) for d in range(1, on.shape[0]))
    worst_on, nan_on = _min_pcc(on, ref)
    worst_off, nan_off = _min_pcc(off, ref)
    logger.info(
        f"[kv-clamp] identical {tag}: clamp on == off bit-identical {same} (max|d| {float((off - on).abs().max()):.3e}), "
        f"deterministic {det}, chips identical {xchip}, worst PCC vs torch off {worst_off:.6f} on {worst_on:.6f}, "
        f"NaN off {nan_off} on {nan_on}"
    )
    assert det, f"[{tag}] repeated calls differ"
    assert same, f"[{tag}] clamp_kv_tail changed the output on the serving layout"
    assert nan_on == 0 and worst_on >= 0.99, f"[{tag}] worst PCC vs torch {worst_on}, NaN {nan_on}"
