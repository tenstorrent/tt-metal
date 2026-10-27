# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device tests: Muse-Glimmer MLP and decoder layer vs the float32 torch reference (real layer 0/3 weights)."""

import os

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.muse_glimmer_30b.reference.model import MLP, DecoderLayer, RotaryEmbedding, build_attention_mask
from models.demos.muse_glimmer_30b.tt.ccl import MuseCCL
from models.demos.muse_glimmer_30b.tt.common import RopeTables
from models.demos.muse_glimmer_30b.tt.mesh_utils import mesh_test_params, skip_and_submesh
from models.demos.muse_glimmer_30b.tt.model_config import MuseGlimmerModelArgs, checkpoint_dir

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(checkpoint_dir(), "config.json")), reason="checkpoint not available"
)

PCC = 0.99


_mesh_params = mesh_test_params()


def _make_args(mesh_device):
    return MuseGlimmerModelArgs.from_checkpoint(
        mesh_device=mesh_device, num_layers=4, max_batch_size=4, max_seq_len=4608, max_prefill_len=4096
    )


def _to_tt(x, mesh_device):
    return ttnn.from_torch(
        x,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _to_torch(t, mesh_device):
    out = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    return out[0:1].float()  # replicated: take rank 0 -> [1, 1, S, H]


def _int_tt(t, mesh_device, dtype):
    return ttnn.from_torch(
        t,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _ref_layer(args, layer_idx, sd):
    layer = DecoderLayer(args.reference_config(), layer_idx).float()
    layer.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True)
    return layer.eval()


def _page_table(args, batch):
    nb = args.max_num_blocks_per_user
    return torch.arange(batch * nb, dtype=torch.int32).reshape(batch, nb)


def _alloc_kv(args, mesh_device):
    from models.demos.muse_glimmer_30b.tt.attention import allocate_kv_cache

    return allocate_kv_cache(args, mesh_device)


def _ref_forward(ref, args, layer_idx, x):
    """x [1, L, H] float32 -> [1, L, H] full-sequence reference."""
    cfg = args.reference_config()
    L = x.shape[1]
    window = cfg.sliding_window if cfg.is_sliding(layer_idx) else None
    mask = build_attention_mask(L, L, 0, window, torch.float32)
    pe = None
    if not cfg.is_nope(layer_idx):
        pe = RotaryEmbedding(cfg)(x, torch.arange(L)[None])
    with torch.no_grad():
        out, _ = ref(x, pe, mask, None)
    return out


@_mesh_params
@pytest.mark.parametrize("seq_len", [32, 4096])
def test_mlp(mesh_device, tp, seq_len):
    sub = skip_and_submesh(mesh_device, tp)
    from models.demos.muse_glimmer_30b.tt.mlp import MuseGlimmerMLP

    args = _make_args(sub)
    sd = args.load_layer_state_dict(0)
    ref = MLP(args.reference_config()).float()
    ref.load_state_dict({k[len("mlp.") :]: v.float() for k, v in sd.items() if k.startswith("mlp.")})
    ref.eval()

    torch.manual_seed(0)
    x = torch.randn(1, 1, seq_len, args.hidden_size).bfloat16()
    with torch.no_grad():
        expected = ref(x.float())

    mlp = MuseGlimmerMLP(args, sub, MuseCCL(sub), 0, sd, use_weight_cache=False)
    out = _to_torch(mlp(_to_tt(x, sub)), sub)
    ok, msg = comp_pcc(expected, out, PCC)
    print(f"mlp S={seq_len}: {msg}")
    assert ok, msg


@_mesh_params
@pytest.mark.parametrize("seq_len", [128, 2304])
@pytest.mark.parametrize("layer_idx", [0, 3])
def test_decoder_layer_prefill(mesh_device, tp, layer_idx, seq_len):
    sub = skip_and_submesh(mesh_device, tp)
    from models.demos.muse_glimmer_30b.tt.decoder import MuseGlimmerDecoderLayer

    args = _make_args(sub)
    sd = args.load_layer_state_dict(layer_idx)
    ref = _ref_layer(args, layer_idx, sd)

    torch.manual_seed(0)
    x = (torch.randn(1, 1, seq_len, args.hidden_size) * 2.0).bfloat16()
    expected = _ref_forward(ref, args, layer_idx, x[0].float())

    ccl = MuseCCL(sub)
    layer = MuseGlimmerDecoderLayer(args, sub, ccl, layer_idx, sd, use_weight_cache=False)
    kv_cache = _alloc_kv(args, sub)
    page_table = _int_tt(_page_table(args, 1), sub, ttnn.int32)
    rope = None
    if args.uses_rope(layer_idx):
        tables = RopeTables(sub, args.head_dim, args.rope_theta, args.max_seq_len)
        rope = tables.prefill(0, seq_len)

    out = layer.forward_prefill(_to_tt(x, sub), rope, kv_cache, page_table, 0)
    out = _to_torch(out, sub)[0]
    ok, msg = comp_pcc(expected, out, PCC)
    print(f"decoder prefill layer={layer_idx} S={seq_len}: {msg}")
    assert ok, msg


@_mesh_params
@pytest.mark.parametrize("layer_idx", [0, 3])
def test_decoder_layer_decode(mesh_device, tp, layer_idx):
    sub = skip_and_submesh(mesh_device, tp)
    """B=2 users with histories [100, 2200]: prefill each, then one decode step vs full-sequence reference."""
    from models.demos.muse_glimmer_30b.tt.decoder import MuseGlimmerDecoderLayer

    args = _make_args(sub)
    sd = args.load_layer_state_dict(layer_idx)
    ref = _ref_layer(args, layer_idx, sd)
    histories = [100, 2200]
    batch, b_pad = len(histories), 32
    hidden = args.hidden_size

    torch.manual_seed(0)
    seqs = [(torch.randn(1, h + 1, hidden) * 2.0).bfloat16() for h in histories]  # last row = decode token
    expected = [_ref_forward(ref, args, layer_idx, s.float())[0, -1] for s in seqs]

    ccl = MuseCCL(sub)
    layer = MuseGlimmerDecoderLayer(args, sub, ccl, layer_idx, sd, use_weight_cache=False)
    kv_cache = _alloc_kv(args, sub)
    pt_host = _page_table(args, args.max_batch_size)
    page_table = _int_tt(pt_host, sub, ttnn.int32)
    tables = RopeTables(sub, args.head_dim, args.rope_theta, args.max_seq_len) if args.uses_rope(layer_idx) else None

    # prefill each user (S padded up to a multiple of 128; padded rows are zeros and causally harmless)
    for u, h in enumerate(histories):
        s_pad = ((h + 127) // 128) * 128
        xp = torch.zeros(1, 1, s_pad, hidden).bfloat16()
        xp[0, 0, :h] = seqs[u][0, :h]
        rope = tables.prefill(0, s_pad) if tables is not None else None
        out = layer.forward_prefill(_to_tt(xp, sub), rope, kv_cache, page_table, u)
        ttnn.deallocate(out)

    # one decode step; padded users use position -1 (no cache update)
    x = torch.zeros(1, 1, b_pad, hidden).bfloat16()
    pos = torch.full((b_pad,), -1, dtype=torch.int32)
    rope_idx = torch.zeros(1, b_pad, dtype=torch.int32)
    for u, h in enumerate(histories):
        x[0, 0, u] = seqs[u][0, h]
        pos[u] = h
        rope_idx[0, u] = h
    pos_tt = _int_tt(pos, sub, ttnn.int32)
    rope = None
    if tables is not None:
        rope = tables.decode(_int_tt(rope_idx, sub, ttnn.uint32))
    pt_decode = torch.zeros(b_pad, pt_host.shape[1], dtype=torch.int32)
    pt_decode[: pt_host.shape[0]] = pt_host
    page_table_dec = _int_tt(pt_decode, sub, ttnn.int32)

    out = layer.forward_decode(_to_tt(x, sub), pos_tt, rope, kv_cache, page_table_dec)
    out = _to_torch(out, sub)[0, 0]  # [B_pad, H]
    for u in range(batch):
        ok, msg = comp_pcc(expected[u], out[u], PCC)
        print(f"decoder decode layer={layer_idx} user={u} hist={histories[u]}: {msg}")
        assert ok, msg
