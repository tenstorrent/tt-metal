# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The whole attention block vs a torch reference, and the chunked path through the same module.

QKV proj -> head split -> RoPE -> KV-cache write -> ring SDPA -> o_proj -> TP all-reduce, against
``reference/model.py::Attention`` with identical random weights. The reference is handed the *same*
cos/sin frequencies (both derive from the config), so what is measured is attention and sharding —
not the RoPE constants, which ``test_rope_vs_ref.py`` owns.

The chunked test pushes a 2-chunk sequence through the **same** ``Attention`` instance twice and
asserts the second chunk's output matches a one-shot run over the whole sequence. That is what
proves the cache-read path is wired into the block rather than merely callable: chunk 1 goes through
``ring_sdpa_cache_read`` with ``cached_len=chunk``, and its answer has to equal the one-shot result
for the same query rows.

Cached K is also checked directly against the reference's post-RoPE K — after the Meta/HF column
permutation — because the KV cache, not the attention output, is what P1/P2 are graded on.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.llama_3_1_8b.reference import model as ref
from models.demos.llama_3_1_8b.tests.common import (
    assert_pcc,
    assert_tp_replicas_agree,
    cfg_full,
    galaxy_mesh,
    make_ccl,
    pcc,
    random_layer_weights,
    reference_attention,
    spec_mesh_config,
    sp_shard_activation,
)
from models.demos.llama_3_1_8b.tt.attention import Attention, AttentionConfig, allocate_kv_cache, read_slot_kv
from models.demos.llama_3_1_8b.tt.rope import RopeSetup
from models.demos.llama_3_1_8b.tt.runners.prefill_kv_validation import naturalize
from models.demos.llama_3_1_8b.utils.rope_layout import hf_to_meta_perm

CHUNK = 5120
NUM_LAYERS = 2
LAYER = 1  # not 0: a wrong cache slot index is invisible at layer 0


def _build(mesh_device, cfg, mc, weights, max_seq_len):
    ccl = make_ccl(mesh_device)
    rope = RopeSetup(mesh_device, cfg, mc)
    attn = Attention(
        mesh_device,
        AttentionConfig.from_model_config(cfg, max_seq_len=max_seq_len, sequence_parallel=True),
        mc,
        ccl,
        rope,
        state_dict={k[len("self_attn.") :]: v for k, v in weights.items() if k.startswith("self_attn.")},
        layer_idx=LAYER,
    )
    cache = allocate_kv_cache(
        mesh_device,
        num_layers=NUM_LAYERS,
        max_seq_len=max_seq_len,
        num_kv_heads=cfg.num_key_value_heads,
        tp=mc.tp,
        sp_axis=mc.sp_axis,
        head_dim=cfg.head_dim,
    )
    return attn, cache, rope


def _cached_k_v(mesh_device, cache, mc, n_tokens, max_seq_len):
    k_blk, v_blk = read_slot_kv(mesh_device, cache, 0, NUM_LAYERS)
    return (
        naturalize(k_blk[LAYER], n_tokens, mc.sp, CHUNK, max_seq_len).unsqueeze(0),
        naturalize(v_blk[LAYER], n_tokens, mc.sp, CHUNK, max_seq_len).unsqueeze(0),
    )


@galaxy_mesh()
def test_attention_vs_ref(mesh_device, device_params, topology_name):
    """One-shot attention over a full 5120-token chunk, plus the K/V it cached."""
    cfg = cfg_full()
    mc = spec_mesh_config(mesh_device)
    weights = random_layer_weights(cfg, seed=21)
    attn, cache, rope = _build(mesh_device, cfg, mc, weights, CHUNK)

    torch.manual_seed(2)
    x = torch.randn(1, 1, CHUNK, cfg.hidden_size) * 0.5
    out = attn(
        sp_shard_activation(x, mesh_device, mc),
        rope.build_indexed_rope(CHUNK, CHUNK),
        kv_cache=cache,
        cached_len=0,
        indexed_rope=True,
    )
    got = assert_tp_replicas_agree(out, mesh_device, mc, name="attention", tol=0.05)

    cos, sin = ref.rope_cos_sin(cfg, CHUNK, dtype=torch.float32)
    ref_attn = reference_attention(cfg, weights)
    ref_out, ref_k, ref_v = ref_attn(x[0].to(torch.float16), cos, sin)
    assert_pcc(f"attention[{topology_name}]", pcc(ref_out.unsqueeze(0).float(), got))

    perm = hf_to_meta_perm(cfg.head_dim)
    dev_k, dev_v = _cached_k_v(mesh_device, cache, mc, CHUNK, CHUNK)
    assert_pcc(f"attention cached K[{topology_name}]", pcc(ref_k.float()[..., perm], dev_k))
    assert_pcc(f"attention cached V[{topology_name}]", pcc(ref_v.float(), dev_v))


@galaxy_mesh()
def test_attention_chunked_vs_ref(mesh_device, device_params, topology_name):
    """A 2-chunk sequence through the SAME module, two ways; chunk 1's output must match one-shot.

    Both runs use their own cache, so the comparison is between two complete prefills rather than
    between a run and its own leftovers.
    """
    cfg = cfg_full()
    mc = spec_mesh_config(mesh_device)
    total = 2 * CHUNK
    weights = random_layer_weights(cfg, seed=22)
    attn, cache, rope = _build(mesh_device, cfg, mc, weights, total)
    indexed = rope.build_indexed_rope(total, CHUNK)

    torch.manual_seed(4)
    x = torch.randn(1, 1, total, cfg.hidden_size) * 0.5

    # --- chunked: chunk 0 then chunk 1 reading the accumulated prefix
    outs = []
    for start in (0, CHUNK):
        piece = x[:, :, start : start + CHUNK]
        outs.append(
            assert_tp_replicas_agree(
                attn(
                    sp_shard_activation(piece, mesh_device, mc),
                    indexed,
                    kv_cache=cache,
                    cached_len=start,
                    indexed_rope=True,
                ),
                mesh_device,
                mc,
                name=f"attention chunk@{start}",
                tol=0.05,
            )
        )
    chunked_k, chunked_v = _cached_k_v(mesh_device, cache, mc, total, total)

    # --- reference over the whole sequence, sliced to chunk 1's rows
    cos, sin = ref.rope_cos_sin(cfg, total, dtype=torch.float32)
    ref_attn = reference_attention(cfg, weights)
    ref_out, ref_k, ref_v = ref_attn(x[0].to(torch.float16), cos, sin)
    p_chunk1 = pcc(ref_out[:, CHUNK:].unsqueeze(0).float(), outs[1])
    logger.info(f"chunk-1 attention output vs whole-sequence reference: PCC {p_chunk1:.6f}")
    assert_pcc(f"attention_chunked[{topology_name}] chunk1", p_chunk1)

    perm = hf_to_meta_perm(cfg.head_dim)
    assert_pcc(f"attention_chunked K[{topology_name}]", pcc(ref_k.float()[..., perm], chunked_k))
    assert_pcc(f"attention_chunked V[{topology_name}]", pcc(ref_v.float(), chunked_v))

    # Control: chunk 1 must NOT look like a chunk that attended only itself.
    ref_local, _, _ = ref_attn(x[0, 0, CHUNK:].unsqueeze(0).to(torch.float16), *ref.rope_cos_sin(cfg, CHUNK, start_pos=CHUNK, dtype=torch.float32))
    p_local = pcc(ref_local.unsqueeze(0).float(), outs[1])
    logger.info(f"chunk-1 output vs prefix-blind attention: PCC {p_local:.6f}")
    assert p_local < p_chunk1 - 0.01, "chunk 1 did not attend the cached prefix"
