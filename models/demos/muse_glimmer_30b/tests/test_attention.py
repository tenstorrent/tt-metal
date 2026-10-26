# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device test: MuseGlimmerAttention (prefill + decode) vs the torch reference, real weights, layers 0 (sliding+RoPE) and 3 (full+NoPE)."""

import pytest
import torch
from loguru import logger

import models.demos.gemma4.tt.attention.operations as gemma4_ops
import models.demos.muse_glimmer_30b.tt.attention as attention_mod
import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.muse_glimmer_30b.reference.model import Attention, RotaryEmbedding, build_attention_mask
from models.demos.muse_glimmer_30b.tt.attention import MuseGlimmerAttention, allocate_kv_cache
from models.demos.muse_glimmer_30b.tt.ccl import MuseCCL
from models.demos.muse_glimmer_30b.tt.common import RopeTables
from models.demos.muse_glimmer_30b.tt.mesh_utils import mesh_test_params, skip_and_submesh
from models.demos.muse_glimmer_30b.tt.model_config import MuseGlimmerModelArgs

PCC_THRESHOLD = 0.99


mesh_fabric = mesh_test_params()


def _setup(mesh_device, layer_idx):
    args = MuseGlimmerModelArgs.from_checkpoint(
        mesh_device=mesh_device, num_layers=4, max_batch_size=4, max_seq_len=4608, max_prefill_len=4096
    )
    sd = args.load_layer_state_dict(layer_idx)
    ccl = MuseCCL(mesh_device)
    tt_attn = MuseGlimmerAttention(args, mesh_device, ccl, layer_idx, sd, use_weight_cache=False)

    cfg = args.reference_config()
    ref = Attention(config=cfg, layer_idx=layer_idx)
    ref.load_state_dict({k[len("self_attn.") :]: v.float() for k, v in sd.items() if k.startswith("self_attn.")})
    ref = ref.float().eval()
    return args, cfg, ccl, tt_attn, ref


def _ref_forward(ref, cfg, args, layer_idx, x):
    """x [1, S, H] float32 -> [1, S, H] reference attention output."""
    S = x.shape[1]
    pos = None
    if args.uses_rope(layer_idx):
        pos = RotaryEmbedding(cfg)(x, torch.arange(S)[None])
    sw = args.sliding_window if args.is_sliding(layer_idx) else None
    mask = build_attention_mask(S, S, 0, sw, torch.float32)
    with torch.no_grad():
        out, _ = ref(x, pos, mask)
    return out


def _to_dev(t, mesh_device, dtype, layout):
    return ttnn.from_torch(
        t, device=mesh_device, dtype=dtype, layout=layout, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    )


def _read(t, mesh_device):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0]


def _page_table(args, num_users):
    nb = args.max_num_blocks_per_user
    return torch.arange(num_users * nb, dtype=torch.int32).reshape(num_users, nb)


def _run_prefill(mesh_device, tp, layer_idx, seq_len, user_id=0):
    sub = skip_and_submesh(mesh_device, tp)
    args, cfg, ccl, tt_attn, ref = _setup(sub, layer_idx)
    torch.manual_seed(0)
    x = (torch.randn(1, seq_len, args.hidden_size) * 1.0).to(torch.bfloat16)
    ref_out = _ref_forward(ref, cfg, args, layer_idx, x.float())

    rope = None
    if args.uses_rope(layer_idx):
        rope = RopeTables(sub, args.head_dim, args.rope_theta, args.max_seq_len).prefill(0, seq_len)
    kv_cache = allocate_kv_cache(args, sub)
    pt = _to_dev(_page_table(args, args.max_batch_size), sub, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    tt_x = _to_dev(x.unsqueeze(0), sub, ttnn.bfloat16, ttnn.TILE_LAYOUT)

    tt_out = tt_attn.forward_prefill(tt_x, rope, kv_cache, pt, user_id=user_id)
    out = _read(tt_out, sub).reshape(1, seq_len, args.hidden_size).float()

    passing, pcc = comp_pcc(ref_out, out, PCC_THRESHOLD)
    logger.info(f"prefill layer={layer_idx} S={seq_len} user={user_id} PCC: {pcc}")
    assert passing, f"PCC {pcc} < {PCC_THRESHOLD}"


@pytest.mark.timeout(1800)
@mesh_fabric
@pytest.mark.parametrize("layer_idx", [0, 3], ids=["sliding_rope", "full_nope"])
@pytest.mark.parametrize("seq_len", [128, 4096])
def test_attention_prefill(mesh_device, tp, layer_idx, seq_len):
    _run_prefill(mesh_device, tp, layer_idx, seq_len)


@pytest.mark.timeout(1800)
@mesh_fabric
@pytest.mark.parametrize("layer_idx", [0, 3], ids=["sliding_rope", "full_nope"])
@pytest.mark.parametrize("seq_len", [4096, 3328])
def test_attention_prefill_chunked(mesh_device, tp, layer_idx, seq_len, monkeypatch):
    """Long-prompt paths (gemma4 chunked SDPA) forced at short lengths; 3328 has a short tail chunk,
    user 1 checks the page-table row slice."""
    monkeypatch.setattr(attention_mod, "PREFILL_SDPA_MAX_SEQ", 1024)
    monkeypatch.setattr(gemma4_ops, "PREFILL_CHUNK_SIZE", 1024)
    monkeypatch.setattr(gemma4_ops, "PREFILL_SLIDING_CHUNK_SIZE", 1024)
    _run_prefill(mesh_device, tp, layer_idx, seq_len, user_id=1)


@pytest.mark.timeout(1800)
@mesh_fabric
@pytest.mark.parametrize("layer_idx", [0, 3], ids=["sliding_rope", "full_nope"])
def test_attention_decode(mesh_device, tp, layer_idx):
    sub = skip_and_submesh(mesh_device, tp)
    args, cfg, ccl, tt_attn, ref = _setup(sub, layer_idx)
    torch.manual_seed(0)
    lengths = [37, 128, 2100, 3000]
    B = len(lengths)
    H = args.hidden_size
    tables = RopeTables(sub, args.head_dim, args.rope_theta, args.max_seq_len)
    kv_cache = allocate_kv_cache(args, sub)
    pt = _to_dev(_page_table(args, args.max_batch_size), sub, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)

    hist = [(torch.randn(1, L, H) * 1.0).to(torch.bfloat16) for L in lengths]
    new = (torch.randn(1, B, H) * 1.0).to(torch.bfloat16)  # one new token per user

    # Prefill each user's history (zero-padded to a multiple of 128).
    for u, L in enumerate(lengths):
        S = ((L + 127) // 128) * 128
        xp = torch.zeros(1, S, H, dtype=torch.bfloat16)
        xp[:, :L] = hist[u]
        rope = tables.prefill(0, S) if args.uses_rope(layer_idx) else None
        out = tt_attn.forward_prefill(
            _to_dev(xp.unsqueeze(0), sub, ttnn.bfloat16, ttnn.TILE_LAYOUT), rope, kv_cache, pt, user_id=u
        )
        ttnn.synchronize_device(sub)
        out.deallocate(True)

    # One decode step: x_dec [1,1,B,H] at current_pos = lengths.
    cur_pos = _to_dev(torch.tensor(lengths, dtype=torch.int32), sub, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    rope = None
    if args.uses_rope(layer_idx):
        idx = torch.zeros(1, 32, dtype=torch.int32)
        idx[0, :B] = torch.tensor(lengths, dtype=torch.int32)
        rope_idx = _to_dev(idx, sub, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        rope = tables.decode(rope_idx)
    tt_x = _to_dev(new.unsqueeze(0), sub, ttnn.bfloat16, ttnn.TILE_LAYOUT)
    tt_out = tt_attn.forward_decode(tt_x, cur_pos, rope, kv_cache, pt)
    out = _read(tt_out, sub).reshape(-1, H)[:B].float()

    failures = []
    for u, L in enumerate(lengths):
        full = torch.cat([hist[u], new[:, u : u + 1]], dim=1).float()
        ref_out = _ref_forward(ref, cfg, args, layer_idx, full)[:, -1]  # [1, H]
        passing, pcc = comp_pcc(ref_out, out[u : u + 1], PCC_THRESHOLD)
        logger.info(f"decode layer={layer_idx} user={u} len={L} PCC: {pcc}")
        if not passing:
            failures.append((u, L, pcc))
    assert not failures, f"PCC below {PCC_THRESHOLD}: {failures}"
