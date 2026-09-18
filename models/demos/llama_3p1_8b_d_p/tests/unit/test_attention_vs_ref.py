# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC tests for the Llama-3.1-8B GQA attention (tt-blaze#4144).

Device cases:

  1. ``single-card-tp1`` — one chip, TP=1, all 32 Q-heads / 8 KV-heads, one-shot prefill with no
     KV cache. Isolates the GQA math, the Meta-frame RoPE and the causal SDPA from every collective.

  2. ``tp8-1x8`` — eight chips as a 1x8 mesh, real TP=8: **the production per-chip width** — 4
     Q-heads and exactly one KV head per chip, fused QKV 768/chip, o_proj output 512/chip — plus a
     genuine ``reduce_scatter`` over the TP axis. SP is 1 here (one mesh row), so the cache-read
     path is not involved.

  3. ``sp4-4x2`` — 4x2 mesh, TP=2 and SP=4: the sequence-parallel cache-read path, where the
     ring-joint SDPA reads the block-cyclic KV cache the module just wrote. TP=2 (4 KV heads per
     chip) rather than the production TP=8 because TP=8 *with* SP>1 needs 4x8 = 32 chips; an 8-chip
     loudbox can have production TP or nonzero SP but not both, and the SP path is what is untested
     elsewhere. Case 2 covers production TP.

The device-free tests carry most of the correctness argument for the part that is easy to get
silently wrong: the HF -> Meta head-frame permute. Attention *output* is invariant to it (Q and K
are permuted identically and q·k is unchanged), so cases 1-3 passing says nothing about whether the
stored K is in the frame blaze decode reads. ``test_meta_frame_matches_hf_rotation`` grades the
permute against an HF-frame restatement directly, and ``test_stored_k_is_in_the_meta_frame`` pins
that what the module hands the cache writer is ``to_meta_frame`` of HF's K.

Run (single card — case 1):
    pytest models/demos/llama_3p1_8b_d_p/tests/unit/test_attention_vs_ref.py
Eight-chip loudbox (all three):
    pytest models/demos/llama_3p1_8b_d_p/tests/unit/test_attention_vs_ref.py -k "tp8 or sp4"
"""

from __future__ import annotations

import subprocess
import sys

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.llama_3p1_8b_d_p.reference.llama_3p1_8b_config import Llama31_8BConfig
from models.demos.llama_3p1_8b_d_p.reference.model import Llama31Attention
from models.demos.llama_3p1_8b_d_p.reference.model import apply_rope as hf_apply_rope
from models.demos.llama_3p1_8b_d_p.reference.model import build_hf_cos_sin, to_meta_frame
from models.demos.llama_3p1_8b_d_p.tests.mesh_profiles import drop_sp_replicas, galaxy_torus_xy_device_params
from models.demos.llama_3p1_8b_d_p.tt.attention import TtLlamaAttention, hf_to_meta_head_frame
from models.demos.llama_3p1_8b_d_p.tt.ccl import CCLManager
from models.demos.llama_3p1_8b_d_p.tt.config import MeshConfig
from models.demos.llama_3p1_8b_d_p.tt.kv_cache import allocate_kv_cache
from models.demos.llama_3p1_8b_d_p.tt.rope import build_indexed_rope, build_llama3_cos_sin, build_transformation_mat

EMB_DIM = Llama31_8BConfig.EMB_SIZE  # 4096
N_HEADS = Llama31_8BConfig.NUM_ATTENTION_HEADS  # 32
N_KV_HEADS = Llama31_8BConfig.NUM_KEY_VALUE_HEADS  # 8
HEAD_DIM = Llama31_8BConfig.HEAD_DIM  # 128

# #4144's acceptance floor, the same as the MLP's. Attention is harder on numerics than the MLP —
# the softmax amplifies QK error — so this is bf16 projections and HiFi4 throughout.
PCC_REQUIRED = 0.99


def _reference_weights(reference: Llama31Attention) -> dict:
    return {
        "q_proj": reference.q_proj.weight.detach(),
        "k_proj": reference.k_proj.weight.detach(),
        "v_proj": reference.v_proj.weight.detach(),
        "o_proj": reference.o_proj.weight.detach(),
    }


# =====================================================================================
# Frame / layout (device-free) — the part cases 1-3 cannot see
# =====================================================================================
def test_meta_frame_matches_hf_rotation():
    """Meta-interleaved rotation of un-permuted weights == HF rotation of HF weights.

    This is the whole justification for :func:`hf_to_meta_head_frame`. HF ships q/k rows in
    half-split order precisely so its ``rotate_half`` reproduces Meta's adjacent-pair rotation; the
    device op rotates adjacent pairs, so the rows must be interleaved back. If that reasoning is
    wrong, the device and HF disagree — and they disagree in a way that is *exactly zero at
    position 0*, so a short-sequence test would not see it. Positions are therefore pushed well
    past ``original_max_position_embeddings`` (8192), where llama3 frequency scaling is also live.
    """
    torch.manual_seed(0)
    seq = 8
    positions = torch.arange(20000, 20000 + seq)  # past 8192: scaled band, and far from RoPE's identity
    w_hf = torch.randn(N_KV_HEADS * HEAD_DIM, EMB_DIM)
    x = torch.randn(1, seq, EMB_DIM)

    # HF path: HF-layout weights, half-split rotation.
    k_hf = (x @ w_hf.T).view(1, seq, N_KV_HEADS, HEAD_DIM).transpose(1, 2)
    cos, sin = build_hf_cos_sin(positions)
    k_hf_rot = hf_apply_rope(k_hf, cos, sin)

    # Device path: un-permuted weights, Meta-interleaved rotation with adjacently-duplicated tables.
    w_meta = hf_to_meta_head_frame(w_hf, N_KV_HEADS, HEAD_DIM)
    k_meta = (x @ w_meta.T).view(1, seq, N_KV_HEADS, HEAD_DIM).transpose(1, 2)
    cos_m, sin_m = build_llama3_cos_sin(int(positions[-1]) + 1)
    cos_m = cos_m[0, 0][positions]
    sin_m = sin_m[0, 0][positions]
    # Meta rotation: pairs are (x0,x1), (x2,x3), ... -> (-x1, x0) interleaved back.
    rotated = torch.stack((-k_meta[..., 1::2], k_meta[..., 0::2]), dim=-1).flatten(-2)
    k_meta_rot = k_meta * cos_m + rotated * sin_m

    torch.testing.assert_close(k_meta_rot, to_meta_frame(k_hf_rot), rtol=1e-4, atol=1e-4)


def test_hf_to_meta_head_frame_is_a_per_head_row_permutation():
    """The permute only reorders rows, and only within each head.

    A permute that leaked across heads (e.g. reshaping with the head axis in the wrong place) would
    still satisfy the round-trip identity for a single head, so check both properties directly.
    """
    w = torch.arange(N_KV_HEADS * HEAD_DIM * 4, dtype=torch.float32).reshape(N_KV_HEADS * HEAD_DIM, 4)
    got = hf_to_meta_head_frame(w, N_KV_HEADS, HEAD_DIM)

    assert got.shape == w.shape
    for head in range(N_KV_HEADS):
        lo, hi = head * HEAD_DIM, (head + 1) * HEAD_DIM
        block_in, block_out = w[lo:hi], got[lo:hi]
        # Same multiset of rows within the head: nothing entered or left this head's block.
        assert torch.equal(block_in.sum(0), block_out.sum(0)), f"head {head} rows crossed a head boundary"
        # And the specific interleave: out row 2i is in row i, out row 2i+1 is in row i + head_dim/2.
        half = HEAD_DIM // 2
        assert torch.equal(block_out[0::2], block_in[:half])
        assert torch.equal(block_out[1::2], block_in[half:])


def test_stored_k_is_in_the_meta_frame():
    """What attention hands the cache writer is ``to_meta_frame`` of HF's post-RoPE K.

    The property the KV migration depends on: blaze decode reads K Meta-interleaved. Nothing in the
    attention *output* can detect a violation, so it is pinned here on the host against the
    reference, which computes in the HF frame by construction.
    """
    torch.manual_seed(0)
    seq = 64
    reference = Llama31Attention().eval()
    x = torch.randn(1, seq, EMB_DIM)
    cos, sin = build_hf_cos_sin(torch.arange(seq))
    with torch.no_grad():
        _, (k_hf, _) = reference(x, cos, sin, return_kv=True)

    w_meta = hf_to_meta_head_frame(reference.k_proj.weight.detach(), N_KV_HEADS, HEAD_DIM)
    k_meta = (x @ w_meta.T).view(1, seq, N_KV_HEADS, HEAD_DIM).transpose(1, 2)
    cos_m, sin_m = build_llama3_cos_sin(seq)
    rotated = torch.stack((-k_meta[..., 1::2], k_meta[..., 0::2]), dim=-1).flatten(-2)
    k_meta_rot = k_meta * cos_m[0, 0] + rotated * sin_m[0, 0]

    torch.testing.assert_close(k_meta_rot, to_meta_frame(k_hf), rtol=1e-4, atol=1e-4)


def test_fused_qkv_gives_each_device_its_own_qkv_block():
    """Each TP shard of the fused weight is that device's ``[wq_i | wk_i | wv_i]``.

    ``ShardTensor2dMesh`` splits the last dim into ``tp`` equal chunks in device order, so the fuse
    has to interleave per device. A plain ``cat([q, k, v])`` would hand the first devices Q only —
    correct shapes, nonsense attention — which is why this is checked rather than assumed.
    """
    tp = 8
    q = torch.randn(N_HEADS * HEAD_DIM, EMB_DIM)
    k = torch.randn(N_KV_HEADS * HEAD_DIM, EMB_DIM)
    v = torch.randn(N_KV_HEADS * HEAD_DIM, EMB_DIM)

    fused = TtLlamaAttention._fuse_qkv(q, k, v, tp)
    q_local, kv_local = N_HEADS * HEAD_DIM // tp, N_KV_HEADS * HEAD_DIM // tp
    assert fused.shape == (EMB_DIM, tp * (q_local + 2 * kv_local))

    for i in range(tp):
        shard = torch.chunk(fused, tp, dim=-1)[i]
        assert torch.equal(shard[:, :q_local], torch.chunk(q, tp, dim=0)[i].T)
        assert torch.equal(shard[:, q_local : q_local + kv_local], torch.chunk(k, tp, dim=0)[i].T)
        assert torch.equal(shard[:, q_local + kv_local :], torch.chunk(v, tp, dim=0)[i].T)


def test_attention_module_is_import_light():
    """Importing ``tt.attention`` must not drag in reference modelling or checkpoint loading."""
    forbidden = ("safetensors", "transformers", "models.demos.llama_3p1_8b_d_p.reference.model")
    probe = (
        "import sys;"
        "import models.demos.llama_3p1_8b_d_p.tt.attention;"
        f"print(','.join(m for m in {forbidden!r} if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "", f"tt.attention import pulled in {out}"


def test_reference_attention_is_sinkless_and_biasfree():
    """The reference must be Llama's GQA, not a neighbour's.

    Pins what "drop the sinks and the sliding window" means in practice: gpt-oss's attention adds a
    learned per-head logit to the softmax denominator and masks to a 128-token window on half its
    layers. Either would still produce plausible PCC on a short sequence.
    """
    attn = Llama31Attention()
    for proj in (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj):
        assert proj.bias is None, "Llama-3.1-8B's attention projections carry no biases"
    assert not hasattr(attn, "sinks"), "Llama-3.1-8B has no attention sinks"
    assert Llama31_8BConfig.SLIDING_WINDOW is None
    assert attn.n_rep == N_HEADS // N_KV_HEADS == 4

    # A 128-token sliding window would bind at seq=256; full causal must not.
    torch.manual_seed(0)
    seq = 256
    x = torch.randn(1, seq, EMB_DIM)
    cos, sin = build_hf_cos_sin(torch.arange(seq))
    with torch.no_grad():
        out_full, _ = attn(x, cos, sin)
        # Zeroing only the first 8 tokens must change the LAST token's output. Under a 128-window it
        # could not: token 255 would not attend to token 0 at all.
        x_masked = x.clone()
        x_masked[:, :8] = 0.0
        out_masked, _ = attn(x_masked, cos, sin)
    assert not torch.allclose(out_full[:, -1], out_masked[:, -1], atol=1e-5), "last token ignores the sequence start"


# =====================================================================================
# Device PCC
# =====================================================================================
@pytest.mark.parametrize("seq_len", [256], ids=["s256"])
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param((1, 1), {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-card-tp1"),
        pytest.param((1, 8), {"fabric_config": ttnn.FabricConfig.FABRIC_1D}, id="tp8-1x8"),
        # Production TP=8 on the geometry that ships; the four SP rows replicate this SP=1 case.
        # The only arm here a Galaxy can open, since it refuses every partial mesh.
        pytest.param((4, 8), galaxy_torus_xy_device_params(), id="galaxy-tp8-4x8"),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_attention_vs_ref(mesh_device, device_params, seq_len, reset_seeds):
    """One-shot prefill attention vs the torch reference, PCC >= 0.99. No KV cache, SP=1."""
    torch.manual_seed(0)
    rows, cols = mesh_device.shape
    tp = cols
    mesh_config = MeshConfig((rows, cols), tp=tp, tp_axis=1)

    reference = Llama31Attention().eval()
    tt_attn = TtLlamaAttention(
        mesh_device=mesh_device,
        mesh_config=mesh_config,
        torch_weights=_reference_weights(reference),
    )

    # Production per-chip width must be what the module actually allocated, not just what the
    # parametrization asked for.
    assert tt_attn.n_local_heads == N_HEADS // tp
    assert tt_attn.n_local_kv_heads == N_KV_HEADS // tp
    logger.info(
        f"mesh={rows}x{cols} tp={tp} -> {tt_attn.n_local_heads}Q + {tt_attn.n_local_kv_heads}KV/chip, "
        f"out {tt_attn.emb_dim_per_chip}/chip"
    )

    torch_input = torch.randn(1, seq_len, EMB_DIM, dtype=torch.float32)
    cos_hf, sin_hf = build_hf_cos_sin(torch.arange(seq_len))
    with torch.no_grad():
        torch_output, _ = reference(torch_input, cos_hf, sin_hf)

    tt_input = ttnn.from_torch(
        torch_input.unsqueeze(0),  # [1, 1, seq, emb]
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
    )
    cos, sin = build_llama3_cos_sin(seq_len)
    rope_mats = [
        ttnn.from_torch(
            t,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            dtype=ttnn.bfloat16,
        )
        for t in (cos, sin)
    ]
    transformation_mat = build_transformation_mat(mesh_device)
    ccl_manager = CCLManager(mesh_device, num_links=1) if tp > 1 else None

    tt_output = tt_attn(tt_input, rope_mats, transformation_mat, ccl_manager=ccl_manager)
    ttnn.synchronize_device(mesh_device)

    if tp > 1:
        assert tt_output.shape[-1] == EMB_DIM // tp, (
            f"reduce_scatter should leave {EMB_DIM // tp}/chip on the hidden dim (the layout the "
            f"residual stream is in, matching tt/mlp.py), got {tt_output.shape[-1]}"
        )
        tt_output_torch = drop_sp_replicas(
            ttnn.to_torch(
                tt_output,
                mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_device.shape, dims=(0, -1)),
            ),
            rows,
        )
    else:
        assert tt_output.shape[-1] == EMB_DIM
        tt_output_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_output)[0])

    tt_output_torch = tt_output_torch.reshape(torch_output.shape).to(torch.float32)
    assert not torch.isnan(tt_output_torch).any(), "NaN in attention output"
    assert not torch.isinf(tt_output_torch).any(), "Inf in attention output"

    passing, pcc = comp_pcc(torch_output, tt_output_torch, PCC_REQUIRED)
    logger.info(f"attention PCC: {pcc}")
    assert passing, f"attention PCC {pcc} below {PCC_REQUIRED}"


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param((4, 2), {"fabric_config": ttnn.FabricConfig.FABRIC_1D}, id="sp4-4x2"),
        # The production SP=4 x TP=8, which only a Galaxy can open. Worth both arms: this one is
        # the geometry that ships and the only one where each chip owns exactly one KV head, while
        # the 4x2 above keeps the same code path reachable on an eight-chip box.
        pytest.param((4, 8), galaxy_torus_xy_device_params(), id="galaxy-sp4-tp8-4x8"),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("chunk_size, max_seq_len", [(256, 1024)], ids=["c256-s1024"])
def test_attention_sp_cache_read_vs_ref(mesh_device, device_params, chunk_size, max_seq_len, reset_seeds):
    """Sequence-parallel chunk 0: write the KV cache, then read it back with the ring-joint SDPA.

    The cache has room for four chunks while this request is one chunk long, so Q is shorter than
    K/V and the ring reader accepts chunk 0 — which is the whole point of the cache-backed design
    (one ring program for the entire prefill, no separate bootstrap).

    The 4x2 arm runs TP=2, so 4 KV heads land on each chip; the 4x8 Galaxy arm is production's
    one-KV-head-per-chip. See the module docstring.
    """
    torch.manual_seed(0)
    rows, cols = mesh_device.shape
    tp, sp_axis = cols, 0
    sp = rows
    mesh_config = MeshConfig((rows, cols), tp=tp, tp_axis=1)
    chunk_local = chunk_size // sp

    reference = Llama31Attention().eval()
    tt_attn = TtLlamaAttention(
        mesh_device=mesh_device, mesh_config=mesh_config, torch_weights=_reference_weights(reference)
    )

    kv_cache = allocate_kv_cache(
        mesh_device,
        num_layers=1,
        max_seq_len=max_seq_len,
        sp_axis=sp_axis,
        num_users=1,
        head_dim=HEAD_DIM,
        chunk_size=chunk_size,
        num_kv_heads_per_chip=N_KV_HEADS // tp,
    )

    torch_input = torch.randn(1, chunk_size, EMB_DIM, dtype=torch.float32)
    cos_hf, sin_hf = build_hf_cos_sin(torch.arange(chunk_size))
    with torch.no_grad():
        torch_output, _ = reference(torch_input, cos_hf, sin_hf)

    # SP-shard the sequence contiguously across the mesh rows: for chunk 0 the block-cyclic walk
    # degenerates to exactly this split (pinned by test_kv_cache_vs_ref's block-cyclic tests), and
    # the indexed RoPE tables below are reordered to match.
    shard_dims = [None, None]
    shard_dims[sp_axis] = 2
    tt_input = ttnn.from_torch(
        torch_input.unsqueeze(0),
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=tuple(shard_dims)),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
    )
    rope_mats = build_indexed_rope(
        mesh_device, head_dim=HEAD_DIM, max_seq_len=max_seq_len, chunk_size=chunk_size, sp_axis=sp_axis
    )
    transformation_mat = build_transformation_mat(mesh_device)
    ccl_manager = CCLManager(mesh_device, num_links=1)

    tt_output = tt_attn(
        tt_input,
        rope_mats,
        transformation_mat,
        kv_cache=kv_cache,
        ccl_manager=ccl_manager,
        cache_layer_idx=0,
        user_id=0,
        cached_len=0,
        indexed_rope=True,
    )
    ttnn.synchronize_device(mesh_device)

    assert tt_output.shape[-2] == chunk_local, f"expected {chunk_local} rows/chip, got {tt_output.shape[-2]}"
    assert tt_output.shape[-1] == EMB_DIM // tp

    # Rows concat on the sequence dim (SP), columns on hidden (TP) -> the full chunk output.
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_device.shape, dims=(2, -1)),
    )
    tt_output_torch = tt_output_torch.reshape(torch_output.shape).to(torch.float32)
    assert not torch.isnan(tt_output_torch).any(), "NaN in SP attention output"
    assert not torch.isinf(tt_output_torch).any(), "Inf in SP attention output"

    passing, pcc = comp_pcc(torch_output, tt_output_torch, PCC_REQUIRED)
    logger.info(f"SP(4) ring cache-read attention PCC: {pcc}")
    assert passing, f"SP attention PCC {pcc} below {PCC_REQUIRED}"
