# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-op device perf of Qwen3.5-397B-A17B attention layers on a quad-Blackhole mesh (TP=4).

Four tests, one per (layer type, phase):

    test_gdn_prefill    Gated DeltaNet (linear attention) prefill, ISL sweep      ids gdn_pf_isl<N>
    test_gdn_decode     Gated DeltaNet decode, batch sweep                        ids gdn_dec_B<N>
    test_attn_prefill   gated full attention (GQA) prefill, ISL sweep             ids attn_pf_isl<N>
    test_attn_decode    gated full attention decode, batch x context              ids attn_dec_B<N>-ctx<8k|128k>

RANDOM weights - PERF ONLY, no PCC. The 397B checkpoint is ~800 GB and is not needed:
Qwen36ModelArgs reads only config.json, so GDN_CFG_DIR points at a directory named Qwen3.5-397B
holding the shim config from ./configs plus tokenizer files (copied in automatically from any cached
Qwen3.x-27B snapshot; a dir with only config.json builds a silently EMPTY tokenizer).

Prefill runs the model's real 2048-token outer chunks, carry-chained, plus the masked tail bucket.
  * GDN: per-chunk cost is constant (fixed-size recurrent state), so cost is linear in ISL.
  * Attention: chunk N attends over chunks 0..N via `forward_prefill_paged`, so per-chunk cost
    rises with chunk index and the real chunk sequence must be run.
Decode:
  * GDN `forward_decode(x)` takes no position and reads no cache: cost is independent of ISL.
    B>32 silently routes through the prefill in/out-proj branches (tp.py `S > TILE_SIZE`), which
    expect a K-sharded activation; the test feeds whichever layout the branch expects. The
    recurrent step pins [B, Nv_tp, Dk, Dv] fp32 intermediates to L1, so B=32 OOMs at default
    dtypes (QWEN35_GDN_DECODE_BF16=1 + QWEN35_GDN_STATE_BF16=1 reach 32) and B>=64 does not run.
  * Attention walks a paged KV cache, so decode is parametrized on context too.

Profile ONE -k selector per invocation (the signpost filter keeps every region in the CSV):
    python -m tracy -r -p -v -n gdn_pf_isl4096 -m pytest <this file> -k "gdn_pf_isl4096-device"
Ids end in -device_params0-1x4; the -device suffix keeps B1 from also matching B16/B128.
"""
import math
import os
import shutil
from glob import glob
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import parametrize_mesh_tp, replicate_to_device, shard_to_device

try:
    from tracy import signpost
except Exception:  # profiler-less build

    def signpost(_name):
        pass


CHUNK = 2048  # outer prefill chunk, fixed by tt/model.py
BLOCK = 64  # paged KV block size, the value every repo test uses
DECODE_STEPS = 16  # measured steps inside the signposts
TILE = 32  # tpc.TILE_SIZE - the batch threshold that flips GDN's projection branches
# kv_update_shard_cfg builds a CoreGrid(x=cols, y=B//cols) that must fit the 8x10 device, so a
# GDN args object at max_batch_size=128 would fail to construct. GDN never uses that config, so
# build args at 32 and set the layer's own batch via gdn.B. Attention DOES use it, so attention
# constructs args at the real batch (legal up to 80 when 8 | B).
GDN_ARGS_MAX_BATCH = 32

# Config-driven: point at a different model's config dir to profile that model instead.
CFG_DIR = os.environ.get("GDN_CFG_DIR") or os.environ.get("GDN397B_CFG_DIR", "/tmp/Qwen3.5-397B")
TOKENIZER_SRC_GLOB = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3.*-27B/snapshots/*/")

START, END = "GDN_PREFILL_START", "GDN_PREFILL_END"  # parse_gdn_perf.py's markers, shared by all four


# ------------------------------------------------------------------------------------------ shared
def _sync(mesh_device):
    return ttnn.synchronize_device(mesh_device)


def _ensure_cfg_dir():
    p = Path(CFG_DIR)
    assert (p / "config.json").is_file(), f"missing {p}/config.json"
    if not (p / "tokenizer.json").is_file():
        snaps = glob(TOKENIZER_SRC_GLOB)
        assert snaps, f"no cached Qwen3 snapshot under {TOKENIZER_SRC_GLOB} to copy tokenizer files from"
        for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
            src = Path(snaps[0]) / f
            if src.is_file():
                shutil.copy(src, p / f)
    os.environ["HF_MODEL"] = str(p)  # plain assignment: model_config uses setdefault
    return str(p)


def _args(mesh_device, max_batch_size, max_seq_len):
    from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs

    _ensure_cfg_dir()
    args = Qwen36ModelArgs(mesh_device, max_batch_size=max_batch_size, max_seq_len=max_seq_len)
    assert mesh_device.get_num_devices() > 1, "TP path needs >1 device; full 397B width does not fit one chip"
    return args


def _randn(g, *shape, scale=0.02):
    return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16)


def _blocks(n_tokens):
    return math.ceil(n_tokens / BLOCK)


def _rm_int32(mesh_device, t):
    return ttnn.from_torch(t.contiguous(), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)


# --------------------------------------------------------------------------------------------- GDN
def _random_gdn_state_dict(args, seed=0):
    """Full-width weights keyed as tests/test_factory.py::load_gdn_layer emits; load_gdn_weights_tp shards."""
    g = torch.Generator().manual_seed(seed)
    dim, kd, vd = args.dim, args.gdn_key_dim, args.gdn_value_dim
    nv, dv, K = args.gdn_nv, args.gdn_dv, args.gdn_conv_kernel_size
    return {
        "linear_attn.in_proj_qkv.weight": _randn(g, 2 * kd + vd, dim),
        "linear_attn.in_proj_z.weight": _randn(g, vd, dim),
        "linear_attn.in_proj_a.weight": _randn(g, nv, dim),
        "linear_attn.in_proj_b.weight": _randn(g, nv, dim),
        "linear_attn.out_proj.weight": _randn(g, dim, vd),
        "linear_attn.conv1d.weight": _randn(g, 2 * kd + vd, 1, K, scale=0.5),
        # HF init A ~ U(1,16), A_log = log(A): g = -exp(A_log)*softplus(a+dt_bias) stays finite.
        "linear_attn.A_log": torch.log(torch.empty(nv).uniform_(1.0, 16.0, generator=g)).to(torch.float32),
        "linear_attn.dt_bias": torch.empty(nv).uniform_(-4.0, -2.0, generator=g).to(torch.float32),
        "linear_attn.norm.weight": torch.ones(dv, dtype=torch.float32),
    }


def _build_gdn(mesh_device, max_batch_size, max_seq_len):
    from models.demos.blackhole.qwen36.tt.gdn.tp import TPGatedDeltaNet, load_gdn_weights_tp
    from models.tt_transformers.tt.ccl import TT_CCL

    args = _args(mesh_device, max_batch_size, max_seq_len)
    logger.info(
        f"GDN [{Path(CFG_DIR).name}]: devices={mesh_device.get_num_devices()} dim={args.dim} Nk={args.gdn_nk} "
        f"Nv={args.gdn_nv} Dk={args.gdn_dk} Dv={args.gdn_dv} K={args.gdn_conv_kernel_size} | per-device: "
        f"Nk_tp={args.gdn_nk_tp} Nv_tp={args.gdn_nv_tp} qkv_dim_tp={args.gdn_qkv_dim_tp} "
        f"qkvzab_dim_tp={args.gdn_qkvzab_dim_tp} value_dim_tp={args.gdn_value_dim_tp}"
    )
    tw = load_gdn_weights_tp(mesh_device, _random_gdn_state_dict(args), args)
    gdn = TPGatedDeltaNet(mesh_device, args, tw, TT_CCL(mesh_device))
    gdn._stable_state = True  # matches allocate_kv_caches: in-place state carry, as in production
    logger.info(
        f"flags: _fuse_agmm={gdn._fuse_agmm} _fuse_out_mmrs_prefill={gdn._fuse_out_mmrs_prefill} "
        f"_gdn_conv1d={gdn._gdn_conv1d} _gdn_flat_qkv={gdn._gdn_flat_qkv} _dram_sharded={gdn._dram_sharded}"
    )
    return args, gdn


@torch.no_grad()
@pytest.mark.timeout(7200)
@parametrize_mesh_tp()
@pytest.mark.parametrize(
    "isl",
    [
        pytest.param(1024, id="gdn_pf_isl1024"),  # the 1024 tail bucket alone (MAC-FIR conv path)
        pytest.param(4096, id="gdn_pf_isl4096"),
        pytest.param(8192, id="gdn_pf_isl8192"),
        pytest.param(16384, id="gdn_pf_isl16384"),
        pytest.param(32768, id="gdn_pf_isl32768"),
        pytest.param(65536, id="gdn_pf_isl65536"),
        pytest.param(128000, id="gdn_pf_isl128000"),  # 62 x 2048 + 1024 tail = 63 calls
    ],
)
def test_gdn_prefill(mesh_device, isl, reset_seeds, ensure_gc):
    args, gdn = _build_gdn(mesh_device, max_batch_size=1, max_seq_len=4096)
    gdn.reset_state()
    n_full, tail = divmod(isl, CHUNK)
    logger.info(f"GDN prefill ISL={isl}: {n_full} x {CHUNK}" + (f" + {tail} tail" if tail else ""))

    # Prefill activations are K-sharded: the fused in-proj all-gathers them itself.
    x = shard_to_device(mesh_device, torch.randn(1, 1, CHUNK, args.dim, dtype=torch.bfloat16), dim=-1)
    xt = shard_to_device(mesh_device, torch.randn(1, 1, tail, args.dim, dtype=torch.bfloat16), dim=-1) if tail else None
    cs = args.gdn_chunk_size  # inner sub-chunk arg; the fused op ignores it and runs at 32

    for _ in range(2):  # warm-up: compile + program cache, outside the signposts
        ttnn.deallocate(gdn.forward_prefill(x, chunk_size=cs, valid_len=None))
    if xt is not None:
        ttnn.deallocate(gdn.forward_prefill(xt, chunk_size=cs, valid_len=tail))
    _sync(mesh_device)
    gdn.reset_state()

    signpost(START)
    for _ in range(n_full):
        ttnn.deallocate(gdn.forward_prefill(x, chunk_size=cs, valid_len=None))
    if xt is not None:  # valid_len set -> MAC-FIR conv instead of ttnn.conv1d
        ttnn.deallocate(gdn.forward_prefill(xt, chunk_size=cs, valid_len=tail))
    _sync(mesh_device)
    signpost(END)
    logger.info(f"PASSED: GDN prefill ISL={isl}")


@torch.no_grad()
@pytest.mark.timeout(7200)
@parametrize_mesh_tp()
@pytest.mark.parametrize("B", [pytest.param(b, id=f"gdn_dec_B{b}") for b in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)])
def test_gdn_decode(mesh_device, B, reset_seeds, ensure_gc):
    args, gdn = _build_gdn(mesh_device, max_batch_size=GDN_ARGS_MAX_BATCH, max_seq_len=4096)
    gdn.B = B
    gdn.reset_state()  # rec_state [B, Nv_tp, Dk, Dv] fp32 + K conv states [1, B, qkv_dim_tp]

    branch = "prefill-arm (all_gather_matmul + 2D out-proj)" if B > TILE else "decode-arm (1D matmul)"
    logger.info(
        f"GDN decode B={B} steps={DECODE_STEPS} | rec_state={B * args.gdn_nv_tp * args.gdn_dk * args.gdn_dv * 4 / 2**20:.1f} "
        f"MiB per device per layer | branch={branch}"
    )
    x = torch.randn(1, 1, B, args.dim, dtype=torch.bfloat16)
    # The AG-matmul arm gathers K-sharded activations itself; the 1D decode matmul wants full width.
    x_tt = shard_to_device(mesh_device, x, dim=-1) if B > TILE else replicate_to_device(mesh_device, x)

    for _ in range(3):
        ttnn.deallocate(gdn.forward_decode(x_tt))
    _sync(mesh_device)
    gdn.reset_state()

    signpost(START)
    for _ in range(DECODE_STEPS):
        ttnn.deallocate(gdn.forward_decode(x_tt))
    _sync(mesh_device)
    signpost(END)
    logger.info(f"PASSED: GDN decode B={B} x {DECODE_STEPS} steps")


# --------------------------------------------------------------------------------------- attention
def _random_attn_state_dict(args, seed=0):
    """Keys as tests/test_factory.py::load_attn_layer emits. q_proj is 2x wide (per head [q_h | gate_h]).
    q_norm/k_norm are stored zero-centred - load_attention_weights_tp adds the +1 itself."""
    g = torch.Generator().manual_seed(seed)
    dim, NH, NKV, HD = args.dim, args.n_heads, args.n_kv_heads, args.head_dim
    return {
        "q_proj.weight": _randn(g, NH * HD * 2, dim),
        "k_proj.weight": _randn(g, NKV * HD, dim),
        "v_proj.weight": _randn(g, NKV * HD, dim),
        "o_proj.weight": _randn(g, dim, NH * HD),
        "q_norm.weight": (torch.randn(HD, generator=g) * 0.02).to(torch.float32),
        "k_norm.weight": (torch.randn(HD, generator=g) * 0.02).to(torch.float32),
    }


def _build_attn(mesh_device, max_batch_size, max_seq_len):
    from models.demos.blackhole.qwen36.tt.attention.tp import TPAttention, load_attention_weights_tp
    from models.tt_transformers.tt.ccl import TT_CCL

    args = _args(mesh_device, max_batch_size, max_seq_len)
    logger.info(
        f"397B attention: devices={mesh_device.get_num_devices()} dim={args.dim} NH={args.n_heads} "
        f"NKV={args.n_kv_heads} HD={args.head_dim} rope_dim={args.rope_head_dim} theta={args.rope_theta} | "
        f"per-device: NH_tp={args.n_local_heads} NKV_tp={args.n_local_kv_heads}"
    )
    tw = load_attention_weights_tp(mesh_device, _random_attn_state_dict(args), args)
    attn = TPAttention(mesh_device, args, tw, TT_CCL(mesh_device))
    logger.info(
        f"flags: _fuse_agmm={attn._fuse_agmm} _fused_qkv={attn._fused_qkv} _qg_deint={attn._qg_deint} "
        f"_dram_sharded={attn._dram_sharded} _wo_sharded={attn._wo_sharded} _sdpa_bf8={attn._sdpa_bf8}"
    )
    return args, attn


def _alloc_paged_kv(mesh_device, args, num_blocks):
    """[num_blocks, n_local_kv_heads, BLOCK, head_dim] bf16 DRAM, replicated per device. At TP=4
    n_local_kv_heads = 1, so K+V cost ISL x 1024 B per user per layer per device."""
    shape = (num_blocks, args.n_local_kv_heads, BLOCK, args.head_dim)

    def _mk():
        return ttnn.as_tensor(
            torch.zeros(shape, dtype=torch.bfloat16),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    logger.info(f"paged KV: {shape} -> {2 * math.prod(shape) * 2 / 2**20:.1f} MiB for K+V per device")
    return _mk(), _mk()


@torch.no_grad()
@pytest.mark.timeout(10800)
@parametrize_mesh_tp()
@pytest.mark.parametrize(
    "isl", [pytest.param(n, id=f"attn_pf_isl{n}") for n in (4096, 8192, 16384, 32768, 65536, 128000)]
)
def test_attn_prefill(mesh_device, isl, reset_seeds, ensure_gc):
    from models.demos.blackhole.qwen36.tt.attention.rope_tp import rot_mats_prefill

    n_full, tail = divmod(isl, CHUNK)
    args, attn = _build_attn(mesh_device, max_batch_size=1, max_seq_len=max(isl, 4096))

    num_blocks = _blocks(isl) + 8
    attn.set_paged_kv_cache(*_alloc_paged_kv(mesh_device, args, num_blocks))
    pt = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)
    full_pt = _rm_int32(mesh_device, pt)
    logger.info(f"attention prefill ISL={isl}: {n_full} x {CHUNK}" + (f" + {tail} tail" if tail else ""))

    x = shard_to_device(mesh_device, torch.randn(1, 1, CHUNK, args.dim, dtype=torch.bfloat16), dim=-1)
    xt = shard_to_device(mesh_device, torch.randn(1, 1, tail, args.dim, dtype=torch.bfloat16), dim=-1) if tail else None
    cos, sin = rot_mats_prefill(mesh_device, args.rope_head_dim, CHUNK, args.rope_theta)
    cos_t, sin_t = rot_mats_prefill(mesh_device, args.rope_head_dim, tail, args.rope_theta) if tail else (None, None)

    def _chunk(cs, length, xx, c, s):
        # Only this chunk's blocks are written; SDPA reads the whole page table up to cs+length,
        # so chunk N attends over chunks 0..N.
        chunk_pt = _rm_int32(mesh_device, pt[:, cs // BLOCK : _blocks(cs + length)])
        out = attn.forward_prefill_paged(xx, c, s, full_pt, chunk_page_table=chunk_pt, chunk_start_idx=cs, user_id=0)
        ttnn.deallocate(chunk_pt)
        ttnn.deallocate(out)

    _chunk(0, CHUNK, x, cos, sin)  # warm-up; program cache is keyed on shape, not chunk_start_idx
    if xt is not None:
        _chunk(0, tail, xt, cos_t, sin_t)
    _sync(mesh_device)

    signpost(START)
    for c in range(n_full):
        _chunk(c * CHUNK, CHUNK, x, cos, sin)
    if xt is not None:
        _chunk(n_full * CHUNK, tail, xt, cos_t, sin_t)
    _sync(mesh_device)
    signpost(END)
    logger.info(f"PASSED: attention prefill ISL={isl}")


@torch.no_grad()
@pytest.mark.timeout(10800)
@parametrize_mesh_tp()
@pytest.mark.parametrize("ctx", [pytest.param(8192, id="ctx8k"), pytest.param(128000, id="ctx128k")])
@pytest.mark.parametrize("B", [pytest.param(b, id=f"attn_dec_B{b}") for b in (1, 16, 32)])
def test_attn_decode(mesh_device, B, ctx, reset_seeds, ensure_gc):
    from models.demos.blackhole.qwen36.tt.attention.rope_tp import rot_mats_decode

    args, attn = _build_attn(mesh_device, max_batch_size=B, max_seq_len=max(2 ** math.ceil(math.log2(ctx)), 4096))
    blocks_per_user = _blocks(ctx) + 1
    num_blocks = B * blocks_per_user
    attn.set_paged_kv_cache(*_alloc_paged_kv(mesh_device, args, num_blocks))
    page_table = _rm_int32(mesh_device, torch.arange(num_blocks, dtype=torch.int32).reshape(B, blocks_per_user))
    logger.info(
        f"attention decode B={B} ctx={ctx} steps={DECODE_STEPS} | KV {ctx * 1024 / 2**20:.1f} MiB per user per layer"
    )

    # Decode activation is replicated (post distributed-norm all-gather), unlike prefill.
    x_tt = replicate_to_device(mesh_device, torch.randn(1, 1, B, args.dim, dtype=torch.bfloat16))
    cur = torch.full((B,), ctx - 1, dtype=torch.int32)
    cur_tt = ttnn.from_torch(
        cur, dtype=ttnn.int32, device=mesh_device, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    )
    cos, sin = rot_mats_decode(mesh_device, args.rope_head_dim, args.max_seq_len, args.rope_theta, cur)

    for _ in range(3):
        ttnn.deallocate(attn.forward_decode(x_tt, cur_tt, cos, sin, page_table=page_table))
    _sync(mesh_device)

    signpost(START)
    for _ in range(DECODE_STEPS):
        ttnn.deallocate(attn.forward_decode(x_tt, cur_tt, cos, sin, page_table=page_table))
    _sync(mesh_device)
    signpost(END)
    logger.info(f"PASSED: attention decode B={B} ctx={ctx} x {DECODE_STEPS} steps")
