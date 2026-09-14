# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Whole model at target SP x TP vs a composed torch reference — M3's closing test.

**These runs are DEPTH- AND VOCAB-REDUCED, and are diagnostics, not results** (recipe §4). A host
cannot hold two copies of a 32-layer 8B model as random weights, so the stack is 4 layers and the
vocab is 8192; ``hidden_size``, ``intermediate_size``, the head counts, the chunk size and the mesh
are all the real ones. Every number this file prints is labelled ``REDUCED``. The graded whole-model
number is ``tests/test_prefill_acceptance.py`` at full depth and width with real weights.

What the reduction still buys, and single-layer tests cannot: per-layer weight slicing (layer 3 must
get layer 3's tensors), per-layer KV slot addressing across a stack, and the residual stream
surviving four layers of collectives.

Three checks:
  1. per-layer K/V against the reference's — the same comparison acceptance makes, on random weights;
  2. the e2e logits through the real embedding and LM head;
  3. chunked == one-shot, which is P2's property stated at model scale.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.llama_3_1_8b.reference.model import LlamaReference, REF_DTYPE
from models.demos.llama_3_1_8b.tests.common import (
    assert_pcc,
    cfg_full,
    galaxy_mesh,
    make_ccl,
    pcc,
    spec_mesh_config,
)
from models.demos.llama_3_1_8b.tt.attention.kv_cache import read_slot_kv
from models.demos.llama_3_1_8b.tt.model_config import random_state_dict
from models.demos.llama_3_1_8b.tt.runners.prefill_kv_validation import naturalize
from models.demos.llama_3_1_8b.tt.tt_prefill_runtime import PrefillRuntimeConfig, TtPrefillRuntime
from models.demos.llama_3_1_8b.utils.rope_layout import hf_to_meta_perm

REDUCED_LAYERS = 4
REDUCED_VOCAB = 8192
CHUNK = 5120


def _reduced_cfg():
    cfg = cfg_full()
    cfg.num_hidden_layers = REDUCED_LAYERS
    cfg.vocab_size = REDUCED_VOCAB
    return cfg


def _runtime(mesh_device, cfg, state_dict, *, max_seq_len, chunk_size, build_lm_head=False):
    from models.demos.llama_3_1_8b.conftest import CCL_TOPOLOGY

    return TtPrefillRuntime(
        mesh_device,
        cfg,
        state_dict,
        PrefillRuntimeConfig(
            num_layers=cfg.num_hidden_layers,
            max_seq_len=max_seq_len,
            chunk_size=chunk_size,
            mesh_shape=tuple(mesh_device.shape),
            topology=CCL_TOPOLOGY,
            build_lm_head=build_lm_head,
        ),
    )


def _reference(cfg, state_dict):
    model = LlamaReference(cfg)
    remapped = {(k[len("model.") :] if k.startswith("model.") else k): v.to(REF_DTYPE) for k, v in state_dict.items()}
    model.load_state_dict(remapped, strict=True)
    model.eval()
    return model


def _device_kv(runtime, kv_cache, n_tokens, layer):
    k_blk, v_blk = read_slot_kv(runtime.mesh_device, kv_cache, 0, runtime.config.num_layers)
    c = runtime.config
    return (
        naturalize(k_blk[layer], n_tokens, c.sp, c.chunk_size, c.max_seq_len).unsqueeze(0),
        naturalize(v_blk[layer], n_tokens, c.sp, c.chunk_size, c.max_seq_len).unsqueeze(0),
    )


@galaxy_mesh()
def test_model_sp_vs_ref(mesh_device, device_params, topology_name):
    """REDUCED (4 of 32 layers, vocab 8192): per-layer KV and e2e logits vs the torch reference."""
    cfg = _reduced_cfg()
    torch.manual_seed(0)
    sd = random_state_dict(cfg, num_layers=REDUCED_LAYERS, dtype=torch.float16, seed=41)
    runtime = _runtime(mesh_device, cfg, sd, max_seq_len=CHUNK, chunk_size=CHUNK, build_lm_head=True)
    kv_cache = runtime.allocate_kv_cache()

    ids = torch.randint(0, REDUCED_VOCAB, (CHUNK,)).tolist()
    logits_tt = runtime.prefill_chunk(
        runtime.make_chunk_input(ids), kv_cache, actual_start=0, actual_end=CHUNK, skip_lm_head=False
    )

    ref_logits, ref_kv = _reference(cfg, sd)(torch.tensor(ids).unsqueeze(0))

    perm = hf_to_meta_perm(cfg.head_dim)
    worst = 1.0
    for L in range(REDUCED_LAYERS):
        dev_k, dev_v = _device_kv(runtime, kv_cache, CHUNK, L)
        k_ref, v_ref = ref_kv[L]
        pk = pcc(k_ref.float()[..., perm], dev_k)
        pv = pcc(v_ref.float(), dev_v)
        logger.info(f"REDUCED model layer {L}: K={pk:.6f} V={pv:.6f}")
        assert_pcc(f"REDUCED model[{topology_name}] L{L} K", pk)
        assert_pcc(f"REDUCED model[{topology_name}] L{L} V", pv)
        worst = min(worst, pk, pv)
    logger.info(f"REDUCED whole-model min per-layer KV PCC over {REDUCED_LAYERS} layers: {worst:.6f}")

    mc = spec_mesh_config(mesh_device)
    dims = [None, None]
    dims[mc.tp_axis] = -1
    dims[mc.sp_axis] = 2
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=tuple(dims), mesh_shape=mesh_device.shape)
    got_logits = ttnn.to_torch(logits_tt, mesh_composer=composer).float()[..., : cfg.vocab_size]
    assert_pcc(f"REDUCED model e2e logits[{topology_name}]", pcc(ref_logits.unsqueeze(0).float(), got_logits))


@galaxy_mesh()
def test_model_chunked_matches_one_shot(mesh_device, device_params, topology_name):
    """REDUCED (4 of 32 layers): multi-chunk prefill must produce the same KV as one-shot.

    P2's goal, stated on random weights and a reduced stack so it can be checked against a host
    reference. The full-depth, real-weights statement is the acceptance test's two modes.
    """
    cfg = _reduced_cfg()
    total = 2 * CHUNK
    torch.manual_seed(0)
    sd = random_state_dict(cfg, num_layers=REDUCED_LAYERS, dtype=torch.float16, seed=42)
    ids = torch.randint(0, REDUCED_VOCAB, (total,)).tolist()

    results = {}
    for mode, chunk in (("one_shot", total), ("chunked", CHUNK)):
        runtime = _runtime(mesh_device, cfg, sd, max_seq_len=total, chunk_size=chunk)
        kv_cache = runtime.allocate_kv_cache()
        runtime.prefill_sequence(ids, kv_cache)
        results[mode] = [_device_kv(runtime, kv_cache, total, L) for L in range(REDUCED_LAYERS)]

    ref_logits, ref_kv = _reference(cfg, sd)(torch.tensor(ids).unsqueeze(0))
    perm = hf_to_meta_perm(cfg.head_dim)
    for L in range(REDUCED_LAYERS):
        k1, v1 = results["one_shot"][L]
        kc, vc = results["chunked"][L]
        agree_k, agree_v = pcc(k1, kc), pcc(v1, vc)
        logger.info(f"REDUCED L{L}: chunked-vs-one-shot K={agree_k:.6f} V={agree_v:.6f}")
        assert_pcc(f"REDUCED chunked==one_shot[{topology_name}] L{L} K", agree_k, target=0.999)
        assert_pcc(f"REDUCED chunked==one_shot[{topology_name}] L{L} V", agree_v, target=0.999)

        k_ref, v_ref = ref_kv[L]
        assert_pcc(f"REDUCED chunked vs ref[{topology_name}] L{L} K", pcc(k_ref.float()[..., perm], kc))
        assert_pcc(f"REDUCED chunked vs ref[{topology_name}] L{L} V", pcc(v_ref.float(), vc))
