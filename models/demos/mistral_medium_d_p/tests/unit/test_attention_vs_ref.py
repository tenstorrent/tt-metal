# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DEVICE (32 chips, BH Galaxy): the WHOLE prefill attention module, end to end, vs torch.

Where ``test_ring_joint_sp_vs_ref.py`` isolates the ring op with Q/K/V placed by hand, this drives
``tt/attention/prefill.py::attention_forward`` in full, at the production shape (8,4) = SP=8 x TP=4:

    QKV proj -> head split (GQA 96/8) -> indexed YaRN RoPE -> KV-cache write
      -> ring-joint SDPA over the block-cyclic SP cache -> concat heads -> o_proj -> TP reduce-scatter

So a failure here that the ring test does not also show is in the projections, the head split, the
RoPE seam, the cache write, or the TP close — never the ring itself.

**Conventions.** The reference runs HF-style (``rotate_half`` + ``cat([half, half])`` cos/sin); the
module runs Meta interleaved with ``convert_hf_qkv_to_meta_format``-swizzled q/k. That pair is proven
equivalent on the host by ``test_checkpoint_ingest.py::test_meta_qkv_swizzle_is_the_inverse_of_hf_rope``,
so a mismatch here is a device/sharding issue, not a convention one.

**Block contract** (shared with the MLP — see ``tests/test_factory.py``), now SP-sharded:

    in :  [1, 1, s_local, 12288]   full emb, replicated across TP, SP-sharded over the ring
    out:  [1, 1, s_local,  3072]   emb/tp, reduce-scattered across TP  (the sharded residual)

**Placement.** Rows are laid out in the KV writer's block-cyclic chip order via
``rotated_chip_positions``, the same as the ring test, and undone on the way out. For a
slab-aligned chunk that order is the identity, but it is derived rather than assumed so the chunked
case below stays honest.

Coverage:
  * both SDPA branches — the cache-backed ring (``cache_global > chunk_global``) and the one-shot
    all-gather bootstrap (``cache_global == chunk_global``), which is the branch a request that
    exactly fills its cache takes;
  * two-chunk sequential prefill, i.e. ``cached_len > 0``, which is the only path that exercises
    the on-device cache rotation and a Q slab sitting at a nonzero global offset. Checked
    device-to-device against a single-shot run of the same length, because the bf8 cache's error
    grows with prefix length and would otherwise be charged to chunking;
  * the fused-QKV per-device interleave, and the config/bias guards.

Run:  pytest models/demos/mistral_medium_d_p/tests/unit/test_attention_vs_ref.py -k 8x4
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.tt.mla.utils import rotated_chip_positions
from models.demos.mistral_medium_d_p.config import MeshConfig
from models.demos.mistral_medium_d_p.reference.torch_reference import gqa_attention
from models.demos.mistral_medium_d_p.tt.attention import Attention, AttentionConfig, ProgramConfig, allocate_kv_cache
from models.demos.mistral_medium_d_p.tt.rope import build_indexed_rope, build_transformation_mat
from models.demos.mistral_medium_d_p.tt.rope_tables import build_hf_cos_sin
from models.tt_transformers.tt.load_checkpoints import convert_hf_qkv_to_meta_format

from ..test_factory import mesh_setup, parametrize_mesh_with_fabric
from .shapes import EPS, HEAD_DIM, HIDDEN, N_KV, N_Q, YARN, per_chip

# One SP chunk per chip. 128 rows/chip x SP=8 = a 1024-token global chunk, matching the ring test's
# per-chip load exactly so the two tests are directly comparable.
CHUNK_LOCAL = 128


def _random_attn_weights(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "q": torch.randn(N_Q * HEAD_DIM, HIDDEN, generator=g) * 0.02,
        "k": torch.randn(N_KV * HEAD_DIM, HIDDEN, generator=g) * 0.02,
        "v": torch.randn(N_KV * HEAD_DIM, HIDDEN, generator=g) * 0.02,
        "o": torch.randn(HIDDEN, N_Q * HEAD_DIM, generator=g) * 0.02,
    }


def _build_attention(mesh_device, mesh_config, ccl, w, max_seq_len, sequence_parallel=True):
    state = convert_hf_qkv_to_meta_format(
        {f"{n}_proj.weight": w[n] for n in ("q", "k", "v", "o")},
        HEAD_DIM,
    )
    return Attention(
        mesh_device=mesh_device,
        config=AttentionConfig(
            hidden_size=HIDDEN,
            num_heads=N_Q,
            num_kv_heads=N_KV,
            head_dim=HEAD_DIM,
            max_seq_len=max_seq_len,
            rms_norm_eps=EPS,
            sequence_parallel=sequence_parallel,
        ),
        state_dict=state,
        ccl_manager=ccl,
        mesh_config=mesh_config,
        program_config=ProgramConfig(),
        layer_idx=0,
        transformation_mats={"prefill": build_transformation_mat(mesh_device)},
        weight_dtype=ttnn.bfloat16,
    )


def _chunk_order(cached_len, sp, chunk_local):
    """Chunk-local row order the KV writer expects, and its inverse.

    ``rotated_chip_positions[c][r]`` is the GLOBAL position carried by chip c's r-th row; subtracting
    ``cached_len`` makes it an index into this chunk. Identity for a slab-aligned chunk — derived, not
    assumed.
    """
    positions = rotated_chip_positions(cached_len, sp, chunk_local)
    idx = torch.tensor([positions[c][r] - cached_len for c in range(sp) for r in range(chunk_local)], dtype=torch.long)
    inv = torch.empty_like(idx)
    inv[idx] = torch.arange(idx.numel())
    return idx, inv


def _place_sp(t, mesh_device, mesh_config, idx):
    """[1, 1, chunk_global, H] -> per-chip [1, 1, chunk_local, H]: SP-shard the seq, replicate on TP."""
    dims = [None, None]
    dims[mesh_config.sp_axis] = 2
    return ttnn.from_torch(
        t[:, :, idx, :],
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=tuple(dims)),
    )


def _gather_sp_tp(out_tt, mesh_device, mesh_config, inv):
    """Per-chip [1, 1, chunk_local, hidden/tp] -> [1, 1, chunk_global, hidden], natural row order."""
    dims = [None, None]
    dims[mesh_config.sp_axis] = 2  # sequence back together over the ring
    dims[mesh_config.tp_axis] = 3  # emb/tp shards back to full emb
    got = ttnn.to_torch(
        out_tt,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=tuple(dims), mesh_shape=tuple(mesh_device.shape)),
    ).float()
    return got[:, :, inv, :]


def _reference(x_global, w, seq_global):
    """Full-sequence dense causal GQA with YaRN RoPE, in HF convention. ``x_global`` is [1, S, HIDDEN]."""
    cos_hf, sin_hf = build_hf_cos_sin(seq_global, HEAD_DIM, **YARN)
    return gqa_attention(x_global.float(), w, cos_hf, sin_hf, n_q=N_Q, n_kv=N_KV, head_dim=HEAD_DIM)


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)])
@pytest.mark.parametrize(
    "cache_mult",
    [2, 1],
    ids=["ringcache", "oneshot"],
)
def test_attention_prefill_sp_vs_ref(mesh_device, device_params, cache_mult, reset_seeds):
    """One full ``attention_forward`` at SP=8 x TP=4, both SDPA branches, vs the torch reference.

    ``cache_mult`` selects the branch inside ``prefill.py``:
      * 2 -> ``cache_global > chunk_global``, so ``use_cache_backed_ring`` is True: the production
        ring-joint SDPA reading the block-cyclic SP cache.
      * 1 -> a one-shot request whose cache is exactly the request length. Q and K/V slabs are then
        equal-sized, which the ring reader rejects, so the module falls back to the explicit
        all-gather / SDPA / reduce-scatter bootstrap. Different code, same expected answer.
    """
    torch.manual_seed(0)
    mesh_config, ccl = mesh_setup(mesh_device)
    sp, tp = mesh_config.sp, mesh_config.tp
    chunk_global = sp * CHUNK_LOCAL
    cache_global = cache_mult * chunk_global

    w = _random_attn_weights()
    x = torch.randn(1, chunk_global, HIDDEN) * 0.1
    ref = _reference(x, w, chunk_global)

    attn = _build_attention(mesh_device, mesh_config, ccl, w, cache_global)
    kv_cache = allocate_kv_cache(
        mesh_device,
        num_layers=1,
        max_seq_len=cache_global,
        sp_axis=mesh_config.sp_axis,
        num_users=1,
        head_dim=HEAD_DIM,
        n_kv_local=per_chip(tp)["n_kv"],
    )
    rope_mats = build_indexed_rope(
        mesh_device,
        head_dim=HEAD_DIM,
        max_seq_len=cache_global,
        chunk_size=chunk_global,
        sp_axis=mesh_config.sp_axis,
        **YARN,
    )

    idx, inv = _chunk_order(0, sp, CHUNK_LOCAL)
    out_tt = attn(
        _place_sp(x.reshape(1, 1, chunk_global, HIDDEN), mesh_device, mesh_config, idx),
        rope_mats=rope_mats,
        kv_cache=kv_cache,
        cached_len=0,
        indexed_rope=True,
    )

    assert tuple(out_tt.shape)[-2:] == (
        CHUNK_LOCAL,
        HIDDEN // tp,
    ), f"expected per-chip [.., {CHUNK_LOCAL}, {HIDDEN // tp}], got {tuple(out_tt.shape)}"
    got = _gather_sp_tp(out_tt, mesh_device, mesh_config, inv).reshape(1, chunk_global, HIDDEN)

    passing, pcc = comp_pcc(ref, got, 0.99)
    branch = "cache-backed ring" if cache_mult > 1 else "one-shot all-gather bootstrap"
    logger.info(
        f"attention prefill SP={sp} TP={tp} ({branch}, seq={chunk_global} global / {CHUNK_LOCAL} local, "
        f"cache={cache_global}): {pcc}"
    )
    assert passing, f"attention prefill PCC fail ({branch}): {pcc}"


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)])
def test_attention_prefill_chunked_matches_single(mesh_device, device_params, reset_seeds):
    """Two chunks must reproduce what ONE chunk of the same total length produces.

    This is the only test that drives ``cached_len > 0``, so it is what exercises the on-device cache
    rotation, the indexed RoPE's per-chip start-row derivation, and a Q slab at a nonzero global
    offset.

    **Why device-to-device rather than against the float reference.** The bf8 KV cache's error grows
    with the length of the attended prefix — measured here on a SINGLE unchunked call, where no
    rotation or chunk seam exists at all:

        global tokens    1024      2048      4096
        overall PCC      0.99854   0.99763   0.99119
        tail-block PCC   0.99185   0.98379   0.93517

    K is read from a bf8 cache into the scores, and YaRN's ``attention_factor`` (1.4159 on both cos
    and sin) makes those scores run ~2x hot, so softmax amplifies the quantisation noise; the longer
    the prefix, the more terms compete. Random test activations are the worst case for this, since
    unstructured scores give a near-uniform softmax that reshuffles under small perturbations.

    Comparing chunk 1 against a float reference would therefore conflate two unrelated errors —
    chunking, and prefix-length numerics — and chunk 1 holds precisely the longest-prefix rows, so it
    absorbs the worst of the second while being blamed for the first. An earlier version of this test
    did exactly that and failed at 0.9874 for reasons that had nothing to do with chunking.

    Comparing the chunked run against a single-shot run of the same length puts identical cache
    numerics on both sides, so the only thing that can differ is the chunking itself.
    """
    torch.manual_seed(0)
    mesh_config, ccl = mesh_setup(mesh_device)
    sp, tp = mesh_config.sp, mesh_config.tp
    n_chunks = 2
    total_global = n_chunks * sp * CHUNK_LOCAL  # 2048
    cache_global = total_global

    w = _random_attn_weights(seed=3)
    x = torch.randn(1, total_global, HIDDEN) * 0.1
    ref = _reference(x, w, total_global)

    def _run(chunk_local, chunks):
        """Drive the whole sequence in `chunks` calls of `chunk_local` rows/chip; return [1, S, H]."""
        attn = _build_attention(mesh_device, mesh_config, ccl, w, cache_global)
        kv_cache = allocate_kv_cache(
            mesh_device,
            num_layers=1,
            max_seq_len=cache_global,
            sp_axis=mesh_config.sp_axis,
            num_users=1,
            head_dim=HEAD_DIM,
            n_kv_local=per_chip(tp)["n_kv"],
        )
        chunk_global = sp * chunk_local
        rope_mats = build_indexed_rope(
            mesh_device,
            head_dim=HEAD_DIM,
            max_seq_len=cache_global,
            chunk_size=chunk_global,
            sp_axis=mesh_config.sp_axis,
            **YARN,
        )
        pieces = []
        for c in range(chunks):
            cached_len = c * chunk_global
            idx, inv = _chunk_order(cached_len, sp, chunk_local)
            xc = x[:, cached_len : cached_len + chunk_global, :].reshape(1, 1, chunk_global, HIDDEN)
            out_tt = attn(
                _place_sp(xc, mesh_device, mesh_config, idx),
                rope_mats=rope_mats,
                kv_cache=kv_cache,
                cached_len=cached_len,
                indexed_rope=True,
            )
            pieces.append(_gather_sp_tp(out_tt, mesh_device, mesh_config, inv).reshape(1, chunk_global, HIDDEN))
        return torch.cat(pieces, dim=1)

    got_chunked = _run(CHUNK_LOCAL, n_chunks)  # 2 x 1024
    got_single = _run(CHUNK_LOCAL * n_chunks, 1)  # 1 x 2048

    # The claim under test: chunking changes nothing.
    passing, pcc = comp_pcc(got_single, got_chunked, 0.999)
    logger.info(f"chunked({n_chunks}x{sp * CHUNK_LOCAL}) vs single({total_global}) on device: {pcc}")
    assert passing, (
        f"chunked prefill does not reproduce the single-shot run: {pcc}. Cache numerics are identical "
        f"on both sides, so this is a chunking bug — rotation, RoPE start row, or Q global offset."
    )

    # Sanity floor against the torch reference, over the FULL sequence. Loose by design: see the
    # prefix-length table above for why a tight bound here would be measuring bf8, not correctness.
    for name, got in (("chunked", got_chunked), ("single", got_single)):
        passing, pcc = comp_pcc(ref, got, 0.995)
        logger.info(f"{name} vs torch reference (full {total_global} seq): {pcc}")
        assert passing, f"{name} vs reference PCC fail over {total_global} tokens: {pcc}"


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)])
def test_fused_qkv_shard_is_per_device_qkv(mesh_device, device_params, reset_seeds):
    """Device *i* must hold ``[Q_i(24 heads) | K_i(2) | V_i(2)]`` = 3584 columns, contiguously.

    A naive ``cat([q, k, v], -1)`` sharded afterwards would give device 0 a slice of Q only. Checked
    on the built weight so a failure names the cause instead of showing up as a low PCC. Only the
    TP cols carry the split, so column 0 of the mesh is representative.
    """
    mesh_config, ccl = mesh_setup(mesh_device)
    tp = mesh_config.tp
    pc = per_chip(tp)
    w = _random_attn_weights(seed=11)
    attn = _build_attention(mesh_device, mesh_config, ccl, w, 2048)

    state = convert_hf_qkv_to_meta_format(
        {"q_proj.weight": w["q"], "k_proj.weight": w["k"], "v_proj.weight": w["v"]}, HEAD_DIM
    )
    q_t, k_t, v_t = (state[f"{n}_proj.weight"].t() for n in ("q", "k", "v"))  # [H, n*hd]

    rows, cols = tuple(mesh_device.shape)
    per_dev = ttnn.get_device_tensors(attn.weights.wqkv)
    nq_l, nkv_l = pc["n_q"] * HEAD_DIM, pc["n_kv"] * HEAD_DIM
    for col in range(cols):
        got = ttnn.to_torch(per_dev[col]).reshape(HIDDEN, pc["qkv"])  # row 0 of the mesh
        want = torch.cat(
            [
                q_t[:, col * nq_l : (col + 1) * nq_l],
                k_t[:, col * nkv_l : (col + 1) * nkv_l],
                v_t[:, col * nkv_l : (col + 1) * nkv_l],
            ],
            dim=-1,
        )
        passing, pcc = comp_pcc(want, got, 0.999)
        assert passing, (
            f"TP column {col}'s fused QKV shard is wrong (pcc {pcc}): must be "
            f"[Q_{col}({pc['n_q']} heads) | K_{col}({pc['n_kv']}) | V_{col}({pc['n_kv']})]"
        )
    logger.info(f"fused QKV interleave correct on all {tp} TP columns ({pc['qkv']} cols each)")


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)])
def test_attention_config_guards(mesh_device, device_params, reset_seeds, expect_error):
    """Partial rotary and a non-divisible head count must not be silently accepted."""
    with expect_error(NotImplementedError, "FULL rotary"):
        AttentionConfig(
            hidden_size=HIDDEN,
            num_heads=N_Q,
            num_kv_heads=N_KV,
            head_dim=HEAD_DIM,
            max_seq_len=128,
            rotary_dim=HEAD_DIM // 2,
        )
    with expect_error(ValueError, "divisible"):
        AttentionConfig(hidden_size=HIDDEN, num_heads=N_Q, num_kv_heads=7, head_dim=HEAD_DIM, max_seq_len=128)


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)])
def test_attention_weights_reject_bias(mesh_device, device_params, reset_seeds, expect_error):
    """gpt-oss has attention_bias=True; Mistral does not. A stray bias must fail loud."""
    from models.demos.mistral_medium_d_p.tt.attention.weights import load_attention_weights

    w = _random_attn_weights()
    state = {
        "q_proj.weight": w["q"],
        "q_proj.bias": torch.zeros(N_Q * HEAD_DIM),
        "k_proj.weight": w["k"],
        "v_proj.weight": w["v"],
        "o_proj.weight": w["o"],
    }
    with expect_error(AssertionError, "bias-free"):
        load_attention_weights(
            mesh_device=mesh_device,
            config=AttentionConfig(
                hidden_size=HIDDEN, num_heads=N_Q, num_kv_heads=N_KV, head_dim=HEAD_DIM, max_seq_len=128
            ),
            state_dict=state,
            mesh_config=MeshConfig(tuple(mesh_device.shape), tp=mesh_device.shape[1]),
        )
