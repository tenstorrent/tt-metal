# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCRATCH: A/B for the per-layer "glue" reductions on the Qwen3.6-27B TP=4 prefill.

Three independent, env-gated changes, each compared against the current default on the SAME input:

1. QWEN36_GDN_PROJ_CHUNKS=1 — the GDN in-projection AGMM writes qkv | z | ab as three tensors
   instead of one [1,S,4128] tensor plus three ttnn.slice ops. Needs the tile-padded qkvzab weight
   (load_gdn_weights_tp pads 4120 -> 4128 with zero columns when the env is set at load time).
   Expected: BIT-IDENTICAL qkv/z/a/b (same matmul, same blocking, the extra 8 columns are zeros).

2. QWEN36_GDN_PROJ_CHUNKS=1 on the full-attention in-projection — qkv3 | gate chunks instead of two
   slices. No weight change (both widths are multiples of head_dim). Expected: BIT-IDENTICAL.

3. QWEN36_GDN_GB_BF16=1 — g/beta stay bf16 into chunk_gdn_prep instead of being widened to fp32 on
   the host, which halves the bytes moved by the [BH,T] -> [BH,NC,C,1] column-tile reshape. Expected:
   bit-identical layer output (the fp32 values were exact bf16 widenings; prep only feeds g/beta into
   HiFi4 matmuls / column broadcasts with an fp32 DEST).

Run with MESH_DEVICE=P150x4 and the HF_MODEL env of the checkpoint. This is a scratch A/B, not CI.
"""
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen36.tests.test_factory import (
    load_attn_layer,
    load_gdn_layer,
    model_path,
    parametrize_mesh_tp,
    shard_to_device,
)
from models.demos.blackhole.qwen36.tt.attention.tp import TPAttention, load_attention_weights_tp
from models.demos.blackhole.qwen36.tt.gdn.tp import TPGatedDeltaNet, load_gdn_weights_tp
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs
from models.tt_transformers.tt.ccl import TT_CCL

T = 2048


def _report(label, ref, out):
    """Bit-exactness first (that is the contract here), PCC only as a fallback diagnostic."""
    same = ref.shape == out.shape and torch.equal(ref, out)
    if same:
        logger.info(f"GDN_GLUE {label:24s} BIT-IDENTICAL  shape={tuple(ref.shape)}")
        return True
    _, p = comp_pcc(ref.float(), out.float(), 0.99)
    logger.error(
        f"GDN_GLUE {label:24s} DIFFERS  PCC={p} max|d|={float((ref.float() - out.float()).abs().max()):.6f} "
        f"shapes {tuple(ref.shape)} vs {tuple(out.shape)}"
    )
    return False


@torch.no_grad()
@parametrize_mesh_tp()
def test_gdn_proj_chunks(mesh_device, reset_seeds, ensure_gc):
    """GDN in-proj: chunked AGMM outputs vs the sliced default (bit-for-bit on qkv, z, a, b)."""
    os.environ.setdefault("HF_MODEL", model_path())
    os.environ.pop("QWEN36_GDN_PROJ_CHUNKS", None)
    args = Qwen36ModelArgs(mesh_device, max_batch_size=1, max_seq_len=4096)
    li = next(i for i, t in enumerate(args.attention_type_list) if t == "linear_attention")
    sd = load_gdn_layer(args.CKPT_DIR, li)
    tt_ccl = TT_CCL(mesh_device)
    comp = ttnn.ConcatMeshToTensor(mesh_device, dim=-1)

    # Default: unpadded weight, sliced outputs.
    tw_ref = load_gdn_weights_tp(mesh_device, sd, args)
    gdn_ref = TPGatedDeltaNet(mesh_device, args, tw_ref, tt_ccl)
    # Chunked: the env must be set BEFORE load (it pads the fused weight to a tile multiple) and
    # before the module is constructed (TPGatedDeltaNet caches the mode).
    os.environ["QWEN36_GDN_PROJ_CHUNKS"] = "1"
    tw_chunk = load_gdn_weights_tp(mesh_device, sd, args)
    gdn_chunk = TPGatedDeltaNet(mesh_device, args, tw_chunk, tt_ccl)
    os.environ.pop("QWEN36_GDN_PROJ_CHUNKS", None)

    assert gdn_ref._proj_chunks == 0 and gdn_chunk._proj_chunks == 1, "env gating did not take effect"
    _w = tw_ref["qkvz"].shape[-1]
    assert tw_chunk["qkvz"].shape[-1] == -(-_w // 32) * 32, "fused qkvzab weight was not tile-padded"

    x = torch.randn(1, 1, T, args.dim, dtype=torch.bfloat16)
    x_tt = shard_to_device(mesh_device, x, dim=-1)

    ref = gdn_ref._project_qkvzab(x_tt, T, out_mc=ttnn.L1_MEMORY_CONFIG)
    out = gdn_chunk._project_qkvzab(x_tt, T, out_mc=ttnn.L1_MEMORY_CONFIG)
    ok = True
    for name, r, o in zip(("qkv", "z", "a", "b"), ref, out):
        ok &= _report(f"gdn_proj {name}", ttnn.to_torch(r, mesh_composer=comp), ttnn.to_torch(o, mesh_composer=comp))
    assert ok, "chunked GDN in-projection is not bit-identical to the sliced default"


@torch.no_grad()
@parametrize_mesh_tp()
def test_attn_proj_chunks(mesh_device, reset_seeds, ensure_gc):
    """Attention in-proj: qkv3 | gate chunks vs the two slices (bit-for-bit). No weight change."""
    os.environ.setdefault("HF_MODEL", model_path())
    os.environ.pop("QWEN36_GDN_PROJ_CHUNKS", None)
    args = Qwen36ModelArgs(mesh_device, max_batch_size=1, max_seq_len=4096)
    ai = next(i for i, t in enumerate(args.attention_type_list) if t == "full_attention")
    sd = load_attn_layer(args.CKPT_DIR, ai)
    tt_ccl = TT_CCL(mesh_device)
    comp = ttnn.ConcatMeshToTensor(mesh_device, dim=-1)
    tw = load_attention_weights_tp(mesh_device, sd, args)
    attn = TPAttention(mesh_device, args, tw, tt_ccl)
    if not attn._fused_qkv or not attn._fuse_agmm:
        pytest.skip("attention in-proj is not on the fused AGMM path")

    x = torch.randn(1, 1, T, args.dim, dtype=torch.bfloat16)
    x_tt = shard_to_device(mesh_device, x, dim=-1)

    qkv3_r, gate_r, _ = attn._qkv(x_tt)  # sliced (env unset)
    os.environ["QWEN36_GDN_PROJ_CHUNKS"] = "1"
    qkv3_c, gate_c, _ = attn._qkv(x_tt)  # chunked (read fresh per call)
    os.environ.pop("QWEN36_GDN_PROJ_CHUNKS", None)

    ok = _report("attn_proj qkv3", ttnn.to_torch(qkv3_r, mesh_composer=comp), ttnn.to_torch(qkv3_c, mesh_composer=comp))
    ok &= _report(
        "attn_proj gate", ttnn.to_torch(gate_r, mesh_composer=comp), ttnn.to_torch(gate_c, mesh_composer=comp)
    )
    assert ok, "chunked attention in-projection is not bit-identical to the sliced default"


@torch.no_grad()
@parametrize_mesh_tp()
def test_gdn_gb_bf16(mesh_device, reset_seeds, ensure_gc):
    """chunk_gated_delta_rule with bf16 g/beta vs the fp32 widening, on one real GDN layer."""
    os.environ.setdefault("HF_MODEL", model_path())
    os.environ.pop("QWEN36_GDN_GB_BF16", None)
    args = Qwen36ModelArgs(mesh_device, max_batch_size=1, max_seq_len=4096)
    li = next(i for i, t in enumerate(args.attention_type_list) if t == "linear_attention")
    sd = load_gdn_layer(args.CKPT_DIR, li)
    tt_ccl = TT_CCL(mesh_device)
    tw = load_gdn_weights_tp(mesh_device, sd, args)
    gdn = TPGatedDeltaNet(mesh_device, args, tw, tt_ccl)
    comp = ttnn.ConcatMeshToTensor(mesh_device, dim=-1)

    x = torch.randn(1, 1, T, args.dim, dtype=torch.bfloat16)
    x_tt = shard_to_device(mesh_device, x, dim=-1)

    def run():
        gdn.reset_state()
        return ttnn.to_torch(gdn.forward_prefill(x_tt, chunk_size=128), mesh_composer=comp)[0, 0]

    ref = run()
    os.environ["QWEN36_GDN_GB_BF16"] = "1"
    out = run()
    os.environ.pop("QWEN36_GDN_GB_BF16", None)
    assert _report("gdn layer gb_bf16", ref, out), "bf16 g/beta changed the GDN layer output"
