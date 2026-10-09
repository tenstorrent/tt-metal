# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit checks for the Qwen context-parallel long prefill (tt/qwen_cp_prefill.py, design B) on the WH Galaxy 8x4.

Layout B: mesh column c owns ring slices c and 7 - c (S/4 tokens, local order [slice c ; slice 7 - c]); mesh row r
owns q heads 8r..8r+7 and kv head r.

test_cp_q_prep_local: qwen_q_prep(src_local=True) reading a column-local fused QKV buffer [1, 1, S/4, 1280]
    against _qwen_qkv_prep_direct on the natural buffer: Q on the owned slices, K (all_rows over the local rows with
    the local rot mats) and V (column slice) against the matching rows of today's K / V. torch.equal on all 32
    devices.

Run on the Wormhole Galaxy (8x4 mesh):
    pytest models/demos/llama3_70b_galaxy/tests/test_qwen_cp_prefill.py -s
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.rmsnorm import RMSNorm
from models.demos.llama3_70b_galaxy.tt.custom_ops.qwen_q_prep import column_slices, qwen_q_prep
from models.demos.llama3_70b_galaxy.tt.llama_attention import TtLlamaAttention
from models.demos.llama3_70b_galaxy.tt.llama_common import get_rot_transformation_mat

N_LOCAL_HEADS = 8
HEAD_DIM = 128


def _norm(mesh_device, name, gamma_torch):
    return RMSNorm(
        device=mesh_device,
        dim=HEAD_DIM,
        state_dict={f"layers.0.attention.{name}.weight": gamma_torch},
        state_dict_prefix=None,
        weight_dtype=ttnn.bfloat16,
        weight_key=f"layers.0.attention.{name}",
        eps=1e-06,
    )


class _Attention:
    """The attributes the production methods below read, nothing else."""

    _qwen_local_q_setup = TtLlamaAttention._qwen_local_q_setup
    _qwen_qkv_prep_direct = TtLlamaAttention._qwen_qkv_prep_direct

    def __init__(self, mesh_device, q_gamma, k_gamma, trans_mat):
        self.mesh_device = mesh_device
        self.tt_ccl = type("TtCclStub", (), {})()
        self.n_local_heads = N_LOCAL_HEADS
        self.n_local_kv_heads = 1
        self.head_dim = HEAD_DIM
        self.qk_norm = True
        self.q_norm = _norm(mesh_device, "q_norm", q_gamma)
        self.k_norm = _norm(mesh_device, "k_norm", k_gamma)
        self.transformation_mats = {"prefill": trans_mat}


def _replicated(mesh_device, t, dtype):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _local_rows(t, c, cols, dim=2):
    """Column c's local order [slice c ; slice 2 * cols - 1 - c] of a natural-order tensor along dim."""
    sl = t.shape[dim] // (2 * cols)
    s0, s1 = column_slices(c, cols)
    return torch.cat([t.narrow(dim, s0 * sl, sl), t.narrow(dim, s1 * sl, sl)], dim=dim)


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
@pytest.mark.parametrize("seq_len", [16384, 40960])
def test_cp_q_prep_local(mesh_device, seq_len):
    torch.manual_seed(0)
    rows, cols = list(mesh_device.shape)
    assert (rows, cols) == (8, 4)
    H, D = N_LOCAL_HEADS, HEAD_DIM
    width = (H + 2) * D
    cos = _replicated(mesh_device, torch.rand(1, 1, seq_len, D) * 2 - 1, ttnn.bfloat16)
    sin = _replicated(mesh_device, torch.rand(1, 1, seq_len, D) * 2 - 1, ttnn.bfloat16)
    trans_mat = _replicated(mesh_device, get_rot_transformation_mat(D), ttnn.bfloat16)
    attn = _Attention(mesh_device, torch.randn(D).to(torch.bfloat16), torch.randn(D).to(torch.bfloat16), trans_mat)
    rot = [cos, sin]
    sl = seq_len // (2 * cols)

    # One natural fused QKV, quantized once on the host so the local copy holds exactly the same bf8 values.
    xh = ttnn.to_torch(
        ttnn.from_torch(torch.randn(1, 1, seq_len, width) * 4.0, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    ).to(torch.bfloat16)
    xqkv = _replicated(mesh_device, xh, ttnn.bfloat8_b)
    xqkv_local = ttnn.from_torch(
        torch.cat([_local_rows(xh, c, cols) for c in range(cols)], dim=0),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 0), mesh_shape=(rows, cols)),
    )
    q_ref, k_ref, v_ref = attn._qwen_qkv_prep_direct(xqkv, rot, seq_len)
    rot_local = attn._qwen_local_q_setup(rot, seq_len)[3]

    def run_local():
        q_new = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, H, seq_len, D]), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG
        )
        k_loc = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, seq_len // cols, D]),
            ttnn.bfloat8_b,
            ttnn.TILE_LAYOUT,
            mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        qwen_q_prep(
            q_new,
            cos,
            sin,
            trans_mat,
            attn.q_norm.weight,
            eps=attn.q_norm.eps,
            src=xqkv_local,
            src_col=0,
            src_local=True,
        )
        qwen_q_prep(
            k_loc,
            rot_local[0],
            rot_local[1],
            trans_mat,
            attn.k_norm.weight,
            eps=attn.k_norm.eps,
            all_rows=True,
            src=xqkv_local,
            src_col=H * D,
        )
        v_loc = ttnn.slice(
            xqkv_local,
            [0, 0, 0, (H + 1) * D],
            [1, 1, seq_len // cols, (H + 2) * D],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return q_new, k_loc, v_loc

    for tag in ("first call", "fresh buffers"):
        q_new, k_loc, v_loc = run_local()
        ttnn.synchronize_device(mesh_device)
        bad = []
        devs = zip(*(ttnn.get_device_tensors(t) for t in (q_ref, q_new, k_ref, k_loc, v_ref, v_loc)))
        for i, (qr, qn, kr, kn, vr, vn) in enumerate(devs):
            c = i % cols
            qr, qn = ttnn.to_torch(qr), ttnn.to_torch(qn)
            for s in column_slices(c, cols):
                rs = slice(s * sl, (s + 1) * sl)
                if not torch.equal(qr[:, :, rs], qn[:, :, rs]):
                    bad.append(f"{tag} device {i} Q slice {s} differs")
            if not torch.equal(_local_rows(ttnn.to_torch(kr), c, cols), ttnn.to_torch(kn)):
                bad.append(f"{tag} device {i} local K differs from today's K rows")
            if not torch.equal(_local_rows(ttnn.to_torch(vr), c, cols), ttnn.to_torch(vn)):
                bad.append(f"{tag} device {i} local V differs from today's V rows")
        assert not bad, "\n".join(bad[:16])
        logger.info(f"S={seq_len} {tag}: local-src Q (owned slices), local K, local V == today's on all 32 devices")
        for t in (q_new, k_loc, v_loc):
            ttnn.deallocate(t)


# ---------------------------------------------------------------------------------------------------------------
# Slice 2: layout helpers, weight gather, K/V reorder, entry / exit
# ---------------------------------------------------------------------------------------------------------------
from models.demos.llama3_70b_galaxy.tt import qwen_cp_prefill as cp  # noqa: E402

FABRIC_PARAMS = [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": True}]


@pytest.mark.parametrize("seq_len", [16384, 40960])
def test_cp_layout_host(seq_len):
    lay = cp.CPLayout(seq_len)
    x = torch.arange(seq_len).reshape(1, 1, seq_len, 1)
    chunks = lay.to_cp(x)
    assert torch.equal(lay.from_cp(chunks), x)
    # every device chunk is the expected local rows of its column, and owner() inverts it
    for (r, c), t in chunks.items():
        nat = [lay.local_to_natural(c, r * lay.s32 + o) for o in range(lay.s32)]
        assert t.flatten().tolist() == nat
        for o in (0, lay.s32 - 1):
            assert lay.owner(nat[o]) == (r, c, o)
    assert sorted(lay.kv_perm()) == list(range(seq_len // cp.KV_BLOCK))
    assert sorted(lay.exit_perm()) == list(range(32))


def _prefill_ccl(mesh_device, seq_len):
    from models.demos.llama3_70b_galaxy.tt.llama_ccl import TT_CCL
    from models.demos.llama3_70b_galaxy.tt.model_config import LlamaOptimizations
    from models.demos.llama3_70b_galaxy.tt.prefetcher_common import TtLlamaPrefetcherSetup
    from models.demos.llama3_70b_galaxy.tt.qwen_model_config import TtQwenModelArgs

    margs = TtQwenModelArgs(
        mesh_device, max_batch_size=32, max_seq_len=seq_len, optimizations=LlamaOptimizations.performance
    )
    pf = TtLlamaPrefetcherSetup(
        mesh_device, n_tensors=0, n_layers=1, mode="prefill", save_tensor_addresses=True, is_qwen=True
    )
    mesh_device.set_sub_device_stall_group([pf.worker_sub_device_id])
    ccl = TT_CCL(mesh_device, margs, pf.worker_sub_device_id, mode="prefill", is_qwen=True)
    return margs, pf, ccl


def _stored_layer(mesh_device, margs, layer=10):
    """The model's stored prefill weights of one layer, loaded from the tensor cache (never written), plus the
    per-row full matrices W_r rebuilt from the HF checkpoint with the model's transform (job139 recipe)."""
    import json
    import os

    from safetensors import safe_open

    from models.demos.llama3_70b_galaxy.tt.load_checkpoints import convert_hf_to_meta

    hf = os.environ["HF_MODEL"]
    wmap = json.load(open(os.path.join(hf, "model.safetensors.index.json")))["weight_map"]
    names = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj"]
    names += ["mlp.up_proj", "mlp.down_proj"]
    sd = {}
    for n in names:
        key = f"model.layers.{layer}.{n}.weight"
        with safe_open(os.path.join(hf, wmap[key]), framework="pt") as f:
            sd[key] = f.get_tensor(key)
    sd = convert_hf_to_meta(sd, margs.head_dim)
    pa, pm = f"layers.{layer}.attention", f"layers.{layer}.feed_forward"
    qkv = [
        torch.cat(
            [torch.chunk(sd[f"{pa}.{w}.weight"], ROWS, dim=0)[i].transpose(-2, -1) for w in ("wq", "wk", "wv")], -1
        )
        for i in range(ROWS)
    ]
    host = {
        "wqkv": (torch.cat(qkv, dim=-1)[None, None], (3, 2), f"{pa}.wqkv_sharded_2d_dram"),
        "wo": (sd[f"{pa}.wo.weight"].transpose(-1, -2)[None, None], (2, 3), f"{pa}.wo_width_sharded_2d_dram"),
        "w1": (sd[f"{pm}.w1.weight"].transpose(-2, -1)[None, None], (-1, -2), f"{pm}.w1_interleavedprefetcher"),
        "w3": (sd[f"{pm}.w3.weight"].transpose(-2, -1)[None, None], (-1, -2), f"{pm}.w3_interleavedprefetcher"),
        "w2": (sd[f"{pm}.w2.weight"].transpose(-2, -1)[None, None], (-2, -1), f"{pm}.w2_interleavedprefetcher"),
    }
    cache_dir = margs.weight_cache_path(ttnn.bfloat8_b)
    stored = {}
    for k, (t, dims, cname) in host.items():
        f = cache_dir / f"{cname}_dtype_BFLOAT8_B_layout_TILE.tensorbin"
        assert f.exists(), f"missing stored weight cache {f}"
        stored[k] = ttnn.as_tensor(
            t,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=dims, mesh_shape=(ROWS, COLS)),
            cache_file_name=cache_dir / cname,
        )

    def w_row(k, r):
        t = host[k][0][0, 0]
        n = {"wqkv": 1280, "wo": 1024, "w1": 3200, "w3": 3200, "w2": 3200}[k]
        return t[:, n * r : n * (r + 1)] if k in ("wqkv", "w1", "w3") else t[n * r : n * (r + 1), :]

    return stored, w_row


ROWS, COLS = 8, 4


def _q_host(t, dtype):
    return ttnn.to_torch(ttnn.from_torch(t.to(torch.bfloat16), dtype=dtype, layout=ttnn.TILE_LAYOUT))


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("device_params", FABRIC_PARAMS, indirect=True)
@pytest.mark.parametrize("seq_len", [40960])
def test_cp_gather_reorder_entry_exit(mesh_device, seq_len):
    margs, pf, ccl = _prefill_ccl(mesh_device, seq_len)
    lay = cp.CPLayout(seq_len)
    bad = []

    # (1) per-layer weight gather over the columns: concat of the row's stored shards and bf8(W_r) from HF
    stored, w_row = _stored_layer(mesh_device, margs)
    attn = type("A", (), {"wqkv_interleaved": stored["wqkv"], "wo_interleaved": stored["wo"]})()
    mlp = type("M", (), {k + "_interleaved": stored[k] for k in ("w1", "w3", "w2")})()
    w = cp.gather_layer_weights(ccl, attn, mlp)
    ttnn.synchronize_device(mesh_device)
    for k, g in w.items():
        sh = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(stored[k])]
        go = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(g)]
        ok_sh = ok_hf = 0
        for d in range(ROWS * COLS):
            r = d // COLS
            ok_sh += int(torch.equal(go[d], torch.cat([sh[r * COLS + c] for c in range(COLS)], dim=cp.GATHER_DIMS[k])))
            if d % COLS == 0:
                ref = _q_host(w_row(k, r), ttnn.bfloat8_b).to(go[d].dtype)
                ok_hf += int(torch.equal(go[d].reshape(ref.shape), ref))
        logger.info(f"weight gather {k} {tuple(g.shape)}: eq_shards {ok_sh}/32 eq_hf_Wr {ok_hf}/8")
        if ok_sh != 32 or ok_hf != 8:
            bad.append(f"weight gather {k}: eq_shards {ok_sh}/32 eq_hf_Wr {ok_hf}/8")
    cp.free_weights(w)
    for t in stored.values():
        ttnn.deallocate(t)

    # (2) K/V gather + reorder: scratch == natural K on all devices (column-local K built from one natural K per row)
    ctx = cp.get_context(ccl, mesh_device, seq_len)
    torch.manual_seed(60)
    k_nat = _q_host(torch.randn(ROWS, 1, seq_len, 128), ttnn.bfloat8_b)  # kv head r on row r
    loc = torch.cat([_local_rows(k_nat, c, COLS) for c in range(COLS)], dim=1)  # [ROWS, COLS, S/4, 128]
    kl = ttnn.from_torch(
        loc.reshape(ROWS * COLS, 1, lay.s4, 128),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    knat = cp.kv_to_natural(ctx, ccl, kl, ctx.scratch()[0])
    ttnn.synchronize_device(mesh_device)
    ok = sum(
        int(torch.equal(ttnn.to_torch(t).reshape(seq_len, 128), k_nat[d // COLS, 0]))
        for d, t in enumerate(ttnn.get_device_tensors(knat))
    )
    logger.info(f"K reorder: scratch == natural K on {ok}/32 devices, shape {tuple(knat.shape)}")
    if ok != 32:
        bad.append(f"K reorder {ok}/32")
    ttnn.deallocate(kl)

    # (3) entry / exit round trip: natural [S, 1280] quarter per column -> residual chunks -> natural
    torch.manual_seed(61)
    xh = torch.randn(1, 1, seq_len, cp.HIDDEN).to(torch.bfloat16)
    x = ttnn.from_torch(
        xh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 3), mesh_shape=(ROWS, COLS)),
    )
    h = cp.entry(ctx, ccl, x)
    ttnn.synchronize_device(mesh_device)
    ok = 0
    for d, t in enumerate(ttnn.get_device_tensors(h)):
        r, c = divmod(d, COLS)
        ok += int(torch.equal(ttnn.to_torch(t), xh[:, :, lay.block_start(r, c) : lay.block_start(r, c) + lay.s32]))
    logger.info(f"entry: residual chunk == natural rows on {ok}/32 devices, shape {tuple(h.shape)}")
    if ok != 32:
        bad.append(f"entry {ok}/32")
    y = cp.exit_natural(ctx, ccl, h)
    ttnn.synchronize_device(mesh_device)
    ok = 0
    for d, t in enumerate(ttnn.get_device_tensors(y)):
        c = d % COLS
        ok += int(torch.equal(ttnn.to_torch(t), xh[..., c * 1280 : (c + 1) * 1280]))
    logger.info(f"exit: natural quarter == input on {ok}/32 devices, shape {tuple(y.shape)}")
    if ok != 32:
        bad.append(f"exit {ok}/32")
    for t in (x, h, y):
        ttnn.deallocate(t)
    ccl.close()
    assert not bad, "\n".join(bad)
