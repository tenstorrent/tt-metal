# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One complete decoder layer with residuals vs a torch reference — the composition, after every
piece above passes alone. Pattern: ``minimax_m3/tests/unit/test_decoder_layer_vs_ref.py``.

    input_layernorm -> Attention -> residual add -> post_attention_layernorm -> MLP -> residual add

Everything inside has its own row in the suite; what is new here is only the wiring: which norm feeds
which block, and that both residuals are added to the PRE-norm stream rather than the post-norm one.
Those are the mistakes a per-block test structurally cannot catch, and they produce a plausible
output with a slowly-degrading PCC over depth.

Run at real dims (hidden 12288, intermediate 28672, 96/8 heads, head_dim 128) on the target mesh, in
both configurations the model uses: one-shot (SP bootstrap) and sequence-parallel chunked.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.mistral_3_5_d_p.reference import model as reference
from models.demos.mistral_3_5_d_p.reference.mistral_config import MistralMedium35Config as C
from models.demos.mistral_3_5_d_p.reference.mistral_config import reduced_text_config
from models.demos.mistral_3_5_d_p.spec import SPEC
from models.demos.mistral_3_5_d_p.tt.attention import allocate_kv_cache
from models.demos.mistral_3_5_d_p.tt.layer import DecoderLayer
from models.demos.mistral_3_5_d_p.tt.rope import build_indexed_rope, build_transformation_mat

from ..test_factory import build_mesh_and_ccl, meta_swizzle, parametrize_mesh, sp_tp_shard_mapper

YARN = dict(
    rope_theta=C.ROPE_THETA,
    yarn_factor=C.YARN_FACTOR,
    yarn_orig_max_pos=C.YARN_ORIG_MAX_POS,
    yarn_beta_fast=C.YARN_BETA_FAST,
    yarn_beta_slow=C.YARN_BETA_SLOW,
    truncate=C.YARN_TRUNCATE,
)


def random_layer_state(hf_config, *, seed=0):
    """A full layer's HF sub-state with random weights, in HF (unswizzled) convention."""
    torch.manual_seed(seed)
    return reference.hf_layer_state_dict(reference.random_layer_weights(hf_config, seed=seed))


def build_layer(mesh_device, hf_config, hf_state, *, max_seq_len, ccl, mesh_config, layer_idx=0):
    """The production DecoderLayer, fed Meta-swizzled q/k exactly as ``tt/model_config.py`` does."""
    return DecoderLayer(
        mesh_device=mesh_device,
        hf_config=hf_config,
        state_dict=meta_swizzle(hf_state, hf_config.head_dim),
        layer_idx=layer_idx,
        ccl_manager=ccl,
        tensor_cache_path=None,
        mesh_config=mesh_config,
        transformation_mats={"prefill": build_transformation_mat(mesh_device)},
        max_seq_len=max_seq_len,
        sequence_parallel=True,
    )


@parametrize_mesh()
@pytest.mark.parametrize("n_chunks", [1, 2], ids=["one_shot", "chunked2"])
@pytest.mark.parametrize("chunk", [512], ids=["c512"])
def test_decoder_layer_vs_ref(mesh_device, device_params, chunk, n_chunks, reset_seeds):
    """One decoder layer vs the torch reference, at real dims, one-shot and chunked.

    ``n_chunks=1`` sizes the cache to exactly the chunk, which is the one-shot shape the ring reader
    rejects, so the layer takes the all-gather bootstrap. ``n_chunks=2`` gives the cache room and the
    layer takes the cache-backed ring path from chunk 0 — the same module, two different attention
    cores, and the residual/norm wiring must be right for both.
    """
    rows, cols = tuple(mesh_device.shape)
    sp = rows
    assert (rows, cols) == SPEC.mesh_shape
    total = chunk * n_chunks
    chunk_local = chunk // sp

    hf_config = reduced_text_config()  # real dims; only the layer COUNT is irrelevant here
    hf_state = random_layer_state(hf_config, seed=5)
    x = torch.randn(1, total, hf_config.hidden_size) * 0.1

    # Golden: the whole sequence through the reference layer at once.
    ref, _ = reference.decoder_layer_reference(x, hf_state, hf_config)

    mesh_config, ccl = build_mesh_and_ccl(mesh_device)
    layer = build_layer(mesh_device, hf_config, hf_state, max_seq_len=total, ccl=ccl, mesh_config=mesh_config)
    kv_cache = allocate_kv_cache(
        mesh_device,
        num_layers=1,
        max_seq_len=total,
        sp_axis=SPEC.sp_axis,
        num_users=1,
        head_dim=hf_config.head_dim,
    )
    rope_mats = build_indexed_rope(
        mesh_device, head_dim=hf_config.head_dim, max_seq_len=total, chunk_size=chunk, sp_axis=SPEC.sp_axis, **YARN
    )
    seq_mapper = sp_tp_shard_mapper(mesh_device, seq_dim=2)

    worst = 1.0
    for c in range(n_chunks):
        lo = c * chunk
        x_tt = ttnn.from_torch(
            x[:, lo : lo + chunk, :].reshape(1, 1, chunk, hf_config.hidden_size),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=seq_mapper,
        )
        out = layer(x_tt, position_embeddings=rope_mats, kv_cache=kv_cache, user_id=0, cached_len=lo, indexed_rope=True)
        ttnn.synchronize_device(mesh_device)
        shards = ttnn.get_device_tensors(out)
        for r in range(rows):
            got = ttnn.to_torch(shards[r * cols]).float().reshape(1, chunk_local, hf_config.hidden_size)
            start = lo + r * chunk_local
            ok, pcc = comp_pcc(ref[:, start : start + chunk_local, :], got, SPEC.pcc)
            worst = min(worst, float(pcc))
            assert ok, f"decoder layer output disagrees at chunk {c}, SP row {r}: {pcc}"
    logger.info(f"decoder layer ({n_chunks} x {chunk}, hidden={hf_config.hidden_size}): worst pcc={worst}")


@parametrize_mesh()
def test_decoder_layer_residuals_are_pre_norm(mesh_device, device_params, reset_seeds):
    """The residual added after each sublayer must be the PRE-norm stream, not the normed one.

    Adding the post-norm tensor instead still yields a well-shaped output whose PCC degrades only
    gradually with depth, so a single-layer output test at a loose bar can miss it. Comparing against
    both candidate references makes the difference explicit.
    """
    chunk = 256
    rows, cols = tuple(mesh_device.shape)
    sp = rows
    chunk_local = chunk // sp

    hf_config = reduced_text_config()
    hf_state = random_layer_state(hf_config, seed=6)
    x = torch.randn(1, chunk, hf_config.hidden_size) * 0.1

    correct, _ = reference.decoder_layer_reference(x, hf_state, hf_config)

    # The wrong wiring: each residual add uses the NORMED tensor instead of the incoming stream.
    attn_sd = {k[len("self_attn.") :]: v for k, v in hf_state.items() if k.startswith("self_attn.")}
    mlp_sd = {k[len("mlp.") :]: v for k, v in hf_state.items() if k.startswith("mlp.")}
    normed = reference.rms_norm_reference(x, hf_state["input_layernorm.weight"], hf_config.rms_norm_eps)
    wrong = normed + reference.attention_reference(normed, attn_sd, hf_config).output
    normed2 = reference.rms_norm_reference(wrong, hf_state["post_attention_layernorm.weight"], hf_config.rms_norm_eps)
    wrong = normed2 + reference.mlp_reference(normed2, mlp_sd, hf_config)

    mesh_config, ccl = build_mesh_and_ccl(mesh_device)
    layer = build_layer(mesh_device, hf_config, hf_state, max_seq_len=chunk * 2, ccl=ccl, mesh_config=mesh_config)
    kv_cache = allocate_kv_cache(
        mesh_device,
        num_layers=1,
        max_seq_len=chunk * 2,
        sp_axis=SPEC.sp_axis,
        num_users=1,
        head_dim=hf_config.head_dim,
    )
    rope_mats = build_indexed_rope(
        mesh_device, head_dim=hf_config.head_dim, max_seq_len=chunk * 2, chunk_size=chunk, sp_axis=SPEC.sp_axis, **YARN
    )
    x_tt = ttnn.from_torch(
        x.reshape(1, 1, chunk, hf_config.hidden_size),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=sp_tp_shard_mapper(mesh_device, seq_dim=2),
    )
    out = layer(x_tt, position_embeddings=rope_mats, kv_cache=kv_cache, user_id=0, cached_len=0, indexed_rope=True)
    ttnn.synchronize_device(mesh_device)

    shards = ttnn.get_device_tensors(out)
    got = torch.cat(
        [ttnn.to_torch(shards[r * cols]).float().reshape(1, chunk_local, hf_config.hidden_size) for r in range(rows)],
        dim=1,
    )
    ok_correct, pcc_correct = comp_pcc(correct, got, SPEC.pcc)
    _, pcc_wrong = comp_pcc(wrong, got, SPEC.pcc)
    logger.info(f"decoder layer residual wiring: pre-norm pcc={pcc_correct} vs post-norm pcc={pcc_wrong}")
    assert ok_correct, f"device layer does not match the pre-norm residual reference: {pcc_correct}"
    assert pcc_wrong < pcc_correct, "the two residual wirings are indistinguishable here; the test proves nothing"
