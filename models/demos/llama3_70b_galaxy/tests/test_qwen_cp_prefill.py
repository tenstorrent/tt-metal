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
from models.demos.llama3_70b_galaxy.tt.qwen_opt_limits import prefill_norm_eps

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
@pytest.mark.parametrize(
    "device_params", [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "worker_l1_size": 1345000}], indirect=True
)  # production dispatch / L1
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

        # Split output (q_out1): one [1, H, sl, D] buffer per owned slice == that slice of the natural-row Q.
        q_split = [
            ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, H, sl, D]), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG
            )
            for _ in range(2)
        ]
        qwen_q_prep(
            q_split[0],
            cos,
            sin,
            trans_mat,
            attn.q_norm.weight,
            eps=attn.q_norm.eps,
            src=xqkv_local,
            src_col=0,
            src_local=True,
            q_out1=q_split[1],
        )
        ttnn.synchronize_device(mesh_device)
        devs = zip(*(ttnn.get_device_tensors(t) for t in (q_new, q_split[0], q_split[1])))
        for i, (qn, q0, q1) in enumerate(devs):
            qn = ttnn.to_torch(qn)
            for j, (s, qj) in enumerate(zip(column_slices(i % cols, cols), (q0, q1))):
                if not torch.equal(qn[:, :, s * sl : (s + 1) * sl], ttnn.to_torch(qj)):
                    bad.append(f"{tag} device {i} split Q local slice {j} (slice {s}) differs")
        assert not bad, "\n".join(bad[:16])
        logger.info(f"S={seq_len} {tag}: split-output Q == the natural-row Q slices on all 32 devices")
        for t in (q_new, k_loc, v_loc, *q_split):
            ttnn.deallocate(t)


# ---------------------------------------------------------------------------------------------------------------
# Slice 2: layout helpers, weight gather, K/V reorder, entry / exit
# ---------------------------------------------------------------------------------------------------------------
from models.demos.llama3_70b_galaxy.tt import qwen_cp_prefill as cp  # noqa: E402

FABRIC_PARAMS = [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": True}]


@pytest.mark.host_only
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
    import os

    from models.demos.llama3_70b_galaxy.tt.load_checkpoints import convert_hf_to_meta
    from models.tt_transformers.tt.load_checkpoints import load_hf_state_dict_filtered

    hf = os.getenv("HF_MODEL")
    if not hf:
        pytest.skip("HF_MODEL is not set (Qwen3-32B repo id or checkpoint directory)")
    names = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj"]
    names += ["mlp.up_proj", "mlp.down_proj"]
    keys = [f"model.layers.{layer}.{n}.weight" for n in names]
    sd = load_hf_state_dict_filtered(hf, keys)  # local dir or HF repo id
    sd = convert_hf_to_meta({k: sd[k] for k in keys}, margs.head_dim)
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


# ---------------------------------------------------------------------------------------------------------------
# Slices 3 / 4: one real layer (layer 0 of a 1-layer model), today's path vs layout B on the same inputs
# ---------------------------------------------------------------------------------------------------------------
def _pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def _one_layer_model(mesh_device, seq_len):
    from models.demos.llama3_70b_galaxy.demo.text_qwen_demo import create_tt_qwen_model
    from models.demos.llama3_70b_galaxy.tt.model_config import LlamaOptimizations

    model_args, model, page_table, kv = create_tt_qwen_model(
        mesh_device,
        instruct=True,
        max_batch_size=1,
        optimizations=LlamaOptimizations.performance,
        max_seq_len=seq_len,
        num_layers=1,
        dummy_weights=False,
        page_params={"page_block_size": 64, "page_max_num_blocks": 4096},
        dtype=ttnn.bfloat8_b,
        use_paged_kv_cache=True,
    )
    nb = seq_len // 64
    pt_host = torch.full((COLS, nb), -1, dtype=torch.int32)  # user 0: column 0 fills, the others skip (-1)
    pt_host[0] = page_table[0, :nb]
    pt = ttnn.from_torch(
        pt_host,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 0), mesh_shape=(ROWS, COLS)),
    )
    return model, pt, page_table[0, :nb].long()


def _natural_quarters(mesh_device, t, seq_len):
    """Row 0's 4 devices of a natural [1, 1, S, 1280]-per-column tensor -> [S, 5120] fp32."""
    devs = ttnn.get_device_tensors(t)[:COLS]
    return torch.cat([ttnn.to_torch(d).float().reshape(seq_len, -1) for d in devs], dim=-1)


def _read_cache(cache, blocks, seq_len):
    """Column 0's cache (user 0) per row -> [ROWS, S, 128] fp32."""
    devs = ttnn.get_device_tensors(cache)
    return torch.stack([ttnn.to_torch(devs[r * COLS]).float()[blocks].reshape(seq_len, -1) for r in range(ROWS)])


def _x_natural(mesh_device, xh, dtype):
    return ttnn.from_torch(
        xh,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 3), mesh_shape=(ROWS, COLS)),
    )


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("device_params", FABRIC_PARAMS, indirect=True)
@pytest.mark.parametrize("seq_len", [40960])
def test_cp_attention_layer(mesh_device, seq_len, reset_seeds):
    """Slice 3: cp.attention_prefill vs today's forward_prefill (seq_parallel) on the same bf8 norm output (B through
    entry / exit, natural order). Against an fp32 host reference of the same quantized inputs and weights: B's output
    (128 query positions) and K / V caches reach PCC >= 0.99, and B's output vs today's reaches PCC >= 0.99."""
    model, pt, blocks = _one_layer_model(mesh_device, seq_len)
    layer, ccl = model.layers[0], model.tt_ccl
    attn = layer.attention
    rot = model.get_or_create_prefill_rot_mats()
    torch.manual_seed(5)
    xh = _q_host(torch.randn(1, 1, seq_len, cp.HIDDEN), ttnn.bfloat8_b).to(torch.bfloat16)

    # today
    out_t = attn.forward_prefill(_x_natural(mesh_device, xh, ttnn.bfloat8_b), rot, 0, page_table=pt, seq_parallel=True)
    nat_t = ccl.ring_all_gather_seq(out_t, ttnn.DRAM_MEMORY_CONFIG, "FF2")
    ttnn.synchronize_device(mesh_device)
    y_t = _natural_quarters(mesh_device, nat_t, seq_len)
    k_t = _read_cache(attn.layer_past[0], blocks, seq_len)
    v_t = _read_cache(attn.layer_past[1], blocks, seq_len)
    for t in (out_t, nat_t):
        ttnn.deallocate(t)

    # B
    ctx = cp.get_context(ccl, mesh_device, seq_len)
    w = cp.gather_layer_weights(ccl, attn, layer.feed_forward, keys=("wqkv", "wo"))
    x_nat = _x_natural(mesh_device, xh, ttnn.bfloat8_b)
    xl = cp.entry(ctx, ccl, x_nat)
    ttnn.deallocate(x_nat)
    out_b = cp.attention_prefill(ctx, ccl, attn, xl, rot, w, 0, pt, None)
    nat_b = cp.exit_natural(ctx, ccl, out_b)
    ttnn.synchronize_device(mesh_device)
    y_b = _natural_quarters(mesh_device, nat_b, seq_len)
    k_b = _read_cache(attn.layer_past[0], blocks, seq_len)
    v_b = _read_cache(attn.layer_past[1], blocks, seq_len)
    wqkv = [
        ttnn.to_torch(ttnn.get_device_tensors(w["wqkv"])[r * COLS]).float().reshape(cp.HIDDEN, -1) for r in range(ROWS)
    ]
    wo = torch.cat(
        [ttnn.to_torch(ttnn.get_device_tensors(w["wo"])[r * COLS]).float().reshape(-1, cp.HIDDEN) for r in range(ROWS)]
    )  # [8192, 5120], heads in row order
    cp.free_weights(w)

    # fp32 host reference of K (k_norm + RoPE) and V from the same bf8 input and weights
    x32 = xh.float().reshape(seq_len, cp.HIDDEN)
    g = ttnn.to_torch(ttnn.get_device_tensors(attn.k_norm.weight)[0]).float().reshape(-1)
    cos = ttnn.to_torch(ttnn.get_device_tensors(rot[0])[0]).float().reshape(-1, 128)[:seq_len]
    sin = ttnn.to_torch(ttnn.get_device_tensors(rot[1])[0]).float().reshape(-1, 128)[:seq_len]
    tm = ttnn.to_torch(ttnn.get_device_tensors(attn.transformation_mats["prefill"])[0]).float().reshape(32, 32)
    tfull = torch.block_diag(*[tm] * 4)
    k_ref, v_ref = [], []
    for r in range(ROWS):
        kp = x32 @ wqkv[r][:, 1024:1152]
        kn = kp * torch.rsqrt(kp.pow(2).mean(-1, keepdim=True) + prefill_norm_eps(attn.k_norm, seq_len)) * g
        k_ref.append(kn * cos + (kn @ tfull) * sin)
        v_ref.append(x32 @ wqkv[r][:, 1152:1280])
    k_ref, v_ref = torch.stack(k_ref), torch.stack(v_ref)

    # fp32 host reference of the attention block output on 128 query positions (16 per ring slice)
    sl = seq_len // 8
    gen = torch.Generator().manual_seed(9)
    pos = torch.cat([s * sl + torch.randint(0, sl, (16,), generator=gen) for s in range(8)])
    gq = ttnn.to_torch(ttnn.get_device_tensors(attn.q_norm.weight)[0]).float().reshape(-1)
    heads = []
    for r in range(ROWS):
        qp = (x32[pos] @ wqkv[r][:, :1024]).reshape(-1, 8, 128)
        qn = qp * torch.rsqrt(qp.pow(2).mean(-1, keepdim=True) + prefill_norm_eps(attn.q_norm, seq_len)) * gq
        q = qn * cos[pos][:, None] + (qn @ tfull) * sin[pos][:, None]  # [P, 8, 128]
        sc = torch.einsum("phd,kd->phk", q, k_ref[r]) * attn.scale
        sc = sc.masked_fill(torch.arange(seq_len)[None, None, :] > pos[:, None, None], float("-inf"))
        heads.append(torch.einsum("phk,kd->phd", torch.softmax(sc, -1), v_ref[r]).reshape(len(pos), -1))
    y_ref = torch.cat(heads, dim=1) @ wo
    blk = [
        float((y_b[i : i + 1280] - y_t[i : i + 1280]).norm() / y_t[i : i + 1280].norm())
        for i in range(0, seq_len, 1280)
    ]
    logger.info("CP_S3 per-1280-block rel L2 B vs today: " + " ".join("%.4f" % v for v in blk))

    out_pcc = _pcc(y_b, y_t)
    res = dict(
        out_pcc_b_vs_today=out_pcc,
        k_pcc_today=_pcc(k_t, k_ref),
        k_pcc_b=_pcc(k_b, k_ref),
        v_pcc_today=_pcc(v_t, v_ref),
        v_pcc_b=_pcc(v_b, v_ref),
        out_rel_l2=float((y_b - y_t).norm() / y_t.norm()),
        ref_pcc_today=_pcc(y_t[pos], y_ref),
        ref_pcc_b=_pcc(y_b[pos], y_ref),
        ref_rel_l2_today=float((y_t[pos] - y_ref).norm() / y_ref.norm()),
        ref_rel_l2_b=float((y_b[pos] - y_ref).norm() / y_ref.norm()),
    )
    logger.info("CP_S3 " + " ".join(f"{k}={v:.7f}" for k, v in res.items()))
    ccl.close()
    # On these random inputs today's own output is only pcc ~0.9973 (rel L2 7.5%) against the fp32 reference, and B
    # lands within ~0.001 of it, so B is held to an absolute 0.99 against the fp32 reference (output on 128
    # positions, K and V caches on all positions) and against today; a layout bug collapses these far below 0.99.
    # Model-level accuracy at 40960 is checked end to end by test_qwen_topk40k.py.
    assert res["ref_pcc_b"] >= 0.99, res
    assert res["k_pcc_b"] >= 0.99 and res["v_pcc_b"] >= 0.99, res
    assert out_pcc >= 0.99, res


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("device_params", FABRIC_PARAMS, indirect=True)
@pytest.mark.parametrize("seq_len", [40960])
def test_cp_mlp_norm_layer(mesh_device, seq_len, reset_seeds):
    """Slice 4: (a) cp.local_norm (2 width halves) vs an fp32 host RMSNorm with the layer's gamma: pcc >= 0.99995.
    (b) cp.mlp_prefill vs today's forward_prefill (seq_parallel) on the same bf8 input: output PCC >= 0.99; both
    also against an fp32 host MLP of the same quantized input and weights on 1024 positions (logged)."""
    model, pt, blocks = _one_layer_model(mesh_device, seq_len)
    layer, ccl = model.layers[0], model.tt_ccl
    mlp = layer.feed_forward
    lay = cp.CPLayout(seq_len)
    res = {}

    # (a) local norm on a residual-sized chunk
    torch.manual_seed(7)
    xr = (torch.randn(1, 1, lay.s32, cp.HIDDEN) * 3).to(torch.bfloat16)
    xd = ttnn.from_torch(
        xr,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    for name, dn in (("attn_norm", layer.attention_norm), ("ff_norm", layer.ff_norm)):
        yd = cp.local_norm(dn, xd, layer.args.compute_kernel_config_hifi2)
        ttnn.synchronize_device(mesh_device)
        y = ttnn.to_torch(ttnn.get_device_tensors(yd)[5]).float().reshape(lay.s32, -1)
        g = ttnn.to_torch(ttnn.get_device_tensors(dn.norm.weight)[0]).float().reshape(-1)
        x32 = xr.float().reshape(lay.s32, -1)
        ref = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + prefill_norm_eps(dn.norm, seq_len)) * g
        res[f"{name}_pcc"] = _pcc(y, ref)
        res[f"{name}_rel_l2"] = float((y - ref).norm() / ref.norm())
        ttnn.deallocate(yd)
    ttnn.deallocate(xd)

    # (b) MLP
    torch.manual_seed(8)
    xh = _q_host(torch.randn(1, 1, seq_len, cp.HIDDEN), ttnn.bfloat8_b).to(torch.bfloat16)
    out_t = mlp.forward_prefill(_x_natural(mesh_device, xh, ttnn.bfloat8_b), seq_parallel=True)
    nat_t = ccl.ring_all_gather_seq(out_t, ttnn.DRAM_MEMORY_CONFIG, "FF2")
    ttnn.synchronize_device(mesh_device)
    y_t = _natural_quarters(mesh_device, nat_t, seq_len)
    for t in (out_t, nat_t):
        ttnn.deallocate(t)
    ctx = cp.get_context(ccl, mesh_device, seq_len)
    w = cp.gather_layer_weights(ccl, layer.attention, mlp, keys=("w1", "w3", "w2"))
    x_nat = _x_natural(mesh_device, xh, ttnn.bfloat8_b)
    xl = cp.entry(ctx, ccl, x_nat)
    ttnn.deallocate(x_nat)
    out_b = cp.mlp_prefill(ctx, ccl, mlp, xl, w)
    nat_b = cp.exit_natural(ctx, ccl, out_b)
    ttnn.synchronize_device(mesh_device)
    y_b = _natural_quarters(mesh_device, nat_b, seq_len)
    wh = {
        k: torch.cat(
            [ttnn.to_torch(ttnn.get_device_tensors(w[k])[r * COLS]).float().squeeze() for r in range(ROWS)],
            dim=1 if k != "w2" else 0,
        )
        for k in ("w1", "w3", "w2")
    }
    cp.free_weights(w)
    idx = torch.cat([torch.arange(512), torch.arange(seq_len - 512, seq_len)])
    x32 = xh.float().reshape(seq_len, -1)[idx]
    ref = (torch.nn.functional.silu(x32 @ wh["w1"]) * (x32 @ wh["w3"])) @ wh["w2"]
    res.update(
        mlp_out_pcc_b_vs_today=_pcc(y_b, y_t),
        mlp_rel_l2_b_vs_today=float((y_b - y_t).norm() / y_t.norm()),
        mlp_ref_pcc_today=_pcc(y_t[idx], ref),
        mlp_ref_pcc_b=_pcc(y_b[idx], ref),
        mlp_ref_rel_l2_today=float((y_t[idx] - ref).norm() / ref.norm()),
        mlp_ref_rel_l2_b=float((y_b[idx] - ref).norm() / ref.norm()),
    )
    logger.info("CP_S4 " + " ".join(f"{k}={v:.7f}" for k, v in res.items()))
    ccl.close()
    assert res["attn_norm_pcc"] >= 0.99995 and res["ff_norm_pcc"] >= 0.99995, res
    assert res["mlp_out_pcc_b_vs_today"] >= 0.99, res


# ---------------------------------------------------------------------------------------------------------------
# Slice 6: layer-major driver and last-token exit
# ---------------------------------------------------------------------------------------------------------------
@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("device_params", FABRIC_PARAMS, indirect=True)
@pytest.mark.parametrize("seq_len", [40960])
def test_cp_layer_major(mesh_device, seq_len, reset_seeds):
    """cp.layer_major_prefill (2 users on different columns, 1 real layer) against cp.model_prefill per user:
    last-token windows and KV caches torch.equal; last-token logits (process_output_prefill_logits on the window vs
    on the full natural output) torch.equal. Logs the wall time of both tails."""
    import time

    model, _, _ = _one_layer_model(mesh_device, seq_len)
    attn = model.layers[0].attention
    rot = model.get_or_create_prefill_rot_mats()
    nb = seq_len // 64
    users = [
        dict(slot=0, last=seq_len - 1, blocks=torch.arange(nb)),
        dict(slot=9, last=seq_len - 400, blocks=torch.arange(nb, 2 * nb)),
    ]
    for u in users:
        pt_host = torch.full((COLS, nb), -1, dtype=torch.int32)
        pt_host[u["slot"] // 8] = u["blocks"].int()
        u["pt"] = ttnn.from_torch(
            pt_host,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 0), mesh_shape=(ROWS, COLS)),
        )
        torch.manual_seed(100 + u["slot"])
        u["xh"] = torch.randn(1, 1, seq_len, cp.HIDDEN).to(torch.bfloat16)

    def caches(u):
        devs = [ttnn.get_device_tensors(t) for t in attn.layer_past]
        col = u["slot"] // 8
        return [torch.stack([ttnn.to_torch(d[r * COLS + col])[u["blocks"]] for r in range(ROWS)]) for d in devs]

    def window_of(nat, last):
        w0 = (last // cp.EXIT_WINDOW) * cp.EXIT_WINDOW
        return [ttnn.to_torch(d)[:, :, w0 : w0 + cp.EXIT_WINDOW] for d in ttnn.get_device_tensors(nat)]

    def timed(fn):
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(mesh_device)
        return out, (time.perf_counter() - t0) * 1e3

    # user-major (today's CP forward) per user
    ref = {}
    for u in users:
        nat = cp.model_prefill(model, _x_natural(mesh_device, u["xh"], ttnn.bfloat16), rot, u["slot"], u["pt"], None)
        ttnn.synchronize_device(mesh_device)
        ref[u["slot"]] = dict(win=window_of(nat, u["last"]), cache=caches(u))
        model.process_output_prefill_logits(nat, last_token_idx=u["last"])  # compile
        lg, ms = timed(lambda: model.process_output_prefill_logits(nat, last_token_idx=u["last"]))
        ref[u["slot"]]["logits"] = ttnn.to_torch(ttnn.get_device_tensors(lg[0])[0])
        ref[u["slot"]]["tail_ms"] = ms
        ttnn.deallocate(nat)
    # wipe both users' blocks so the layer-major run has to write them again
    for u in users:
        z = ttnn.from_torch(
            torch.zeros(1, 1, seq_len, 128),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        for t in attn.layer_past:
            ttnn.experimental.paged_fill_cache(t, z, u["pt"], batch_idx=0)
        ttnn.deallocate(z)

    lm = [
        dict(
            h=cp.model_entry(model, _x_natural(mesh_device, u["xh"], ttnn.bfloat16)),
            user_id=u["slot"],
            page_table=u["pt"],
            last_idx=u["last"],
        )
        for u in users
    ]
    windows = cp.layer_major_prefill(model, lm, rot, None)
    ttnn.synchronize_device(mesh_device)
    bad, res = [], {}
    for u, win in zip(users, windows):
        r = ref[u["slot"]]
        got = [ttnn.to_torch(d) for d in ttnn.get_device_tensors(win)]
        nw = sum(int(torch.equal(a, b)) for a, b in zip(got, r["win"]))
        nc = sum(int(torch.equal(a, b)) for a, b in zip(caches(u), r["cache"]))
        model.process_output_prefill_logits(win, last_token_idx=u["last"] % cp.EXIT_WINDOW)  # compile
        lg, ms = timed(lambda: model.process_output_prefill_logits(win, last_token_idx=u["last"] % cp.EXIT_WINDOW))
        eq_logits = torch.equal(ttnn.to_torch(ttnn.get_device_tensors(lg[0])[0]), r["logits"])
        res[
            u["slot"]
        ] = f"window {nw}/32 caches {nc}/2 logits_equal {eq_logits} tail_ms natural {r['tail_ms']:.2f} window {ms:.2f}"
        if nw != 32 or nc != 2 or not eq_logits:
            bad.append(f"user {u['slot']}: {res[u['slot']]}")
    # exit cost alone (b1): natural exit vs last-token exit on the same residual
    ctx = cp.get_context(model.tt_ccl, mesh_device, seq_len)
    h = cp.model_entry(model, _x_natural(mesh_device, users[0]["xh"], ttnn.bfloat16))
    for fn in (
        lambda: cp.exit_natural(ctx, model.tt_ccl, h),
        lambda: cp.exit_last_token(ctx, model.tt_ccl, h, seq_len - 1),
    ):
        ttnn.deallocate(fn())
    o1, t_nat = timed(lambda: cp.exit_natural(ctx, model.tt_ccl, h))
    o2, t_last = timed(lambda: cp.exit_last_token(ctx, model.tt_ccl, h, seq_len - 1))
    logger.info(f"CP_S6 exit_ms natural {t_nat:.2f} last_token {t_last:.2f} (S={seq_len})")
    for u in users:
        logger.info(f"CP_S6 user {u['slot']}: {res[u['slot']]}")
    model.tt_ccl.close()
    assert not bad, "\n".join(bad)


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("device_params", FABRIC_PARAMS, indirect=True)
@pytest.mark.parametrize("seq_len", [16384])
def test_cp_slot_coverage(mesh_device, seq_len, reset_seeds):
    """One layer-major CP prefill of the same prompt in slots 0, 8, 16 and 31 (one user per mesh column), with the
    device page tables llama_model builds (the user's row on its owner column slot // 8, -1 on the others) and
    disjoint blocks per user. Each slot must give the slot-0 result: last-token logits (all 32 devices) and the
    owner column's K / V cache contents in the user's blocks (8 rows) torch.equal. The prefill must not write the
    KV cache outside the owner column: on every column, the blocks of the users of the other columns (row 0 device)
    are unchanged (the cache is column-local and block ids repeat across columns in serving)."""
    model, _, _ = _one_layer_model(mesh_device, seq_len)
    attn = model.layers[0].attention
    rot = model.get_or_create_prefill_rot_mats()
    nb = seq_len // 64
    slots = [0, 8, 16, 31]
    torch.manual_seed(7)
    xh = torch.randn(1, 1, seq_len, cp.HIDDEN).to(torch.bfloat16)
    users = []
    for k, slot in enumerate(slots):
        blocks = torch.arange(k * nb, (k + 1) * nb)
        pt_host = torch.full((COLS, nb), -1, dtype=torch.int32)
        pt_host[slot // ROWS] = blocks.int()
        pt = ttnn.from_torch(
            pt_host,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 0), mesh_shape=(ROWS, COLS)),
        )
        users.append(dict(slot=slot, col=slot // ROWS, blocks=blocks, pt=pt))

    def owner_cache(u):
        out = []
        for t in attn.layer_past:
            d = ttnn.get_device_tensors(t)
            out.append(torch.stack([ttnn.to_torch(d[r * COLS + u["col"]])[u["blocks"]] for r in range(ROWS)]))
        return out

    def foreign_blocks():
        """Row-0 device of each column: the blocks of the users of the other columns, K and V."""
        res = {}
        for c in range(COLS):
            idx = torch.cat([u["blocks"] for u in users if u["col"] != c])
            res[c] = [ttnn.to_torch(ttnn.get_device_tensors(t)[c])[idx] for t in attn.layer_past]
        return res

    before = foreign_blocks()
    lm = [
        dict(
            h=cp.model_entry(model, _x_natural(mesh_device, xh, ttnn.bfloat16)),
            user_id=u["slot"],
            page_table=u["pt"],
            last_idx=seq_len - 1,
        )
        for u in users
    ]
    windows = cp.layer_major_prefill(model, lm, rot, None)
    ttnn.synchronize_device(mesh_device)
    res = {}
    for u, win in zip(users, windows):
        lg = model.process_output_prefill_logits(win, last_token_idx=(seq_len - 1) % cp.EXIT_WINDOW)
        res[u["slot"]] = dict(logits=[ttnn.to_torch(d) for d in ttnn.get_device_tensors(lg[0])], cache=owner_cache(u))
    after = foreign_blocks()
    ref = res[slots[0]]
    bad = []
    for u in users:
        r = res[u["slot"]]
        nl = sum(int(torch.equal(a, b)) for a, b in zip(r["logits"], ref["logits"]))
        nc = sum(int(torch.equal(a, b)) for a, b in zip(r["cache"], ref["cache"]))
        nz = sum(int(torch.count_nonzero(t) > 0) for t in r["cache"])
        line = f"slot {u['slot']} (column {u['col']}): logits equal {nl}/32, K/V cache equal {nc}/2, nonzero {nz}/2"
        logger.info(f"CP_SLOTS {line}")
        if nl != 32 or nc != 2 or nz != 2:
            bad.append(line)
    for c in range(COLS):
        same = sum(int(torch.equal(a, b)) for a, b in zip(before[c], after[c]))
        logger.info(f"CP_SLOTS column {c}: other columns' blocks unchanged {same}/2")
        if same != 2:
            bad.append(f"column {c}: other columns' user blocks written ({same}/2 unchanged)")
    model.tt_ccl.close()
    assert not bad, "\n".join(bad)


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("device_params", FABRIC_PARAMS, indirect=True)
@pytest.mark.parametrize("seq_len", [16384, 32768, 40960])
def test_cp_chunked_sdpa_pair(mesh_device, seq_len):
    """cp.chunked_sdpa_pair (two chunked SDPA calls, per-device chunk start) against the ring_distributed SDPA call it
    replaces, on the scratch-shaped K / V [S/64, 1, 64, 128] through the identity page table (as attention_prefill
    calls it). Replicated random bf8 Q [1, 8, S, 128] / K / V. Per column (row 0) and slice: PCC vs ring >= 0.9997;
    all 8 rows of every column equal; a second call equal (program cache hit). Not bit-identical (streaming compute)."""
    from models.demos.llama3_70b_galaxy.tt.qwen_model_config import TtQwenModelArgs

    margs = TtQwenModelArgs(mesh_device, max_batch_size=32, max_seq_len=seq_len)
    mc = margs.get_model_config()
    base = margs.compute_kernel_config_ring_sdpa_long
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=base.math_fidelity,
        math_approx_mode=base.math_approx_mode,
        fp32_dest_acc_en=False,
        packer_l1_acc=base.packer_l1_acc,
    )
    ctx = cp.CPContext(mesh_device, seq_len)
    lay = ctx.lay
    torch.manual_seed(seq_len)
    q = _replicated(mesh_device, torch.randn(1, N_LOCAL_HEADS, seq_len, HEAD_DIM), ttnn.bfloat8_b)
    kv = [
        _replicated(mesh_device, torch.randn(seq_len // cp.KV_BLOCK, 1, cp.KV_BLOCK, HEAD_DIM), ttnn.bfloat8_b)
        for _ in range(2)
    ]
    nat = [ttnn.reshape(t, [1, 1, seq_len, HEAD_DIM]) for t in kv]
    ring_pc = mc["SDPA_RING_PROGCFG"](seq_len, chunk_start_idx=0)
    scale = HEAD_DIM**-0.5
    ring = ttnn.transformer.ring_distributed_scaled_dot_product_attention(
        q, nat[0], nat[1], ring_size=COLS, scale=scale, compute_kernel_config=ckc, program_config=ring_pc
    )
    pc = cp.chunked_sdpa_config(ring_pc, lay.sl)
    outs = [cp.chunked_sdpa_pair(ctx, q, kv[0], kv[1], ctx.identity_page_table(), scale, ckc, pc) for _ in range(2)]
    rd = [ttnn.to_torch(d).float() for d in ttnn.get_device_tensors(ring)]
    nd, nd2 = ([ttnn.to_torch(d).float() for d in ttnn.get_device_tensors(o)] for o in outs)
    bad = []
    for c in range(COLS):
        for j, s in enumerate(lay.slices(c)):
            part = slice(j * lay.sl, (j + 1) * lay.sl)
            p = _pcc(nd[c][:, :, part], rd[c][:, :, part])
            logger.info(f"CP_CHUNKED S={seq_len} column {c} slice {s}: pcc vs ring {p:.6f}")
            if p < 0.9997:
                bad.append(f"column {c} slice {s}: pcc {p:.6f}")
    rows_eq = all(torch.equal(nd[r * COLS + c], nd[c]) for r in range(ROWS) for c in range(COLS))
    rep_eq = all(torch.equal(a, b) for a, b in zip(nd, nd2))
    logger.info(f"CP_CHUNKED S={seq_len} q{pc.q_chunk_size} k{pc.k_chunk_size}: rows equal {rows_eq}, repeat {rep_eq}")

    # The model's form: Q as one [1, H, S/8, D] buffer per owned slice (qwen_q_prep q_out1) == the natural-Q form,
    # bit for bit, on all 32 devices.
    tabs = ctx._chunked_sdpa[(N_LOCAL_HEADS, HEAD_DIM)]
    q_split = [ttnn.slice(q, starts=st, ends=en, slice_dim=2, num_devices=2 * COLS) for st, en, _ in tabs]
    out_split = cp.chunked_sdpa_pair(ctx, q_split, kv[0], kv[1], ctx.identity_page_table(), scale, ckc, pc)
    split_eq = sum(int(torch.equal(ttnn.to_torch(d), r)) for d, r in zip(ttnn.get_device_tensors(out_split), nd))
    logger.info(f"CP_CHUNKED S={seq_len} split-Q form == natural-Q form on {split_eq}/32 devices")
    assert rows_eq and rep_eq and not bad, bad
    assert split_eq == ROWS * COLS, split_eq
