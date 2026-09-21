# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The whole model at target SP x TP vs a composed torch reference.

**Reduced run** (recipe section 4): depth 8 of 64 and vocab 4096 of 248320, so the reference fits
in host memory alongside the device run. Width, head geometry, the hybrid schedule and both
carried-state families are all at their real values. Every number this file prints is a
reduced-run number and is labelled as such in ``README.md``; the graded numbers come from P1/P2
at full depth with real weights.

What this catches that the single-layer tests cannot: per-layer weight slicing (``layers.N.*``),
layer-type dispatch across the stack, a KV cache sized for the 16 attention layers rather than
for 64, and the GDN state being kept per layer rather than shared.
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.qwen_3_8_27b_d_p.reference.modeling import Qwen35TextModel, init_random_weights
from models.demos.qwen_3_8_27b_d_p.tt.model import Qwen35Model

from ..test_factory import mesh_setup, parametrize_mesh, unit_test_config
from .helpers import CACHE_DTYPE, WEIGHT_DTYPE, assert_tp_replicated, check_pcc, from_sp_sharded

S_LOCAL = 128
REDUCED = dict(num_hidden_layers=8, vocab_size=4096)


def _build(mesh, mesh_config, ccl, cfg, seed: int = 81):
    ref = Qwen35TextModel(cfg).eval()
    init_random_weights(ref, seed=seed)
    model = Qwen35Model(
        mesh,
        cfg,
        ref.state_dict(),
        mesh_config=mesh_config,
        ccl_manager=ccl,
        weight_dtype=WEIGHT_DTYPE,
        cache_dtype=CACHE_DTYPE,
    )
    return ref, model


@parametrize_mesh()
def test_model_sp_vs_ref(mesh, submesh_shape, device_params):
    """One-shot: token ids in, final-norm output out, residual SP-sharded through all 8 layers."""
    cfg = unit_test_config(**REDUCED)
    mesh_config, ccl = mesh_setup(mesh)
    total = S_LOCAL * mesh_config.sp
    ref, model = _build(mesh, mesh_config, ccl, cfg)

    token_ids = torch.randint(0, cfg.vocab_size, (1, total), generator=torch.Generator().manual_seed(82))
    with torch.no_grad():
        expected, _states = ref(input_ids=token_ids, skip_lm_head=True)

    caches = model.allocate_caches(max_seq_len=total)
    out = model.prefill_chunk(token_ids, start_pos=0, caches=caches, skip_lm_head=True)
    assert_tp_replicated(out, mesh_config, "model output")
    got = from_sp_sharded(out, mesh_config)
    check_pcc("model_sp[REDUCED 8L/v4096]", expected.reshape(1, 1, total, cfg.hidden_size), got)


@parametrize_mesh()
def test_model_layer_type_dispatch(mesh, submesh_shape, device_params):
    """The stack must build 6 Gated DeltaNet layers and 2 attention layers at depth 8, and the KV
    cache must hold 2 slots per user — not 8. A cache sized by ``layer_idx`` still runs."""
    cfg = unit_test_config(**REDUCED)
    mesh_config, ccl = mesh_setup(mesh)
    _ref, model = _build(mesh, mesh_config, ccl, cfg)
    from models.demos.qwen_3_8_27b_d_p.tt.attention.prefill import Attention
    from models.demos.qwen_3_8_27b_d_p.tt.gdn.prefill import GatedDeltaNet

    kinds = [type(layer.mixer) for layer in model.layers]
    assert kinds.count(GatedDeltaNet) == 6 and kinds.count(Attention) == 2
    assert [i for i, k in enumerate(kinds) if k is Attention] == list(cfg.full_attention_layers)

    caches = model.allocate_caches(max_seq_len=S_LOCAL * mesh_config.sp)
    assert caches.kv.num_kv_layers == 2, "the KV cache is sized by layer index, not by attention layers"
    assert sorted(caches.gdn) == list(cfg.linear_attention_layers)
    assert caches.kv.k.shape[0] == 2  # num_users * num_kv_layers


@parametrize_mesh()
def test_model_chunked_equals_one_shot(mesh, submesh_shape, device_params):
    """Two chunks vs one — the host-side statement of P2's goal, on the device.

    Both carried-state families have to be right simultaneously: the attention layers' packed K/V
    prefix and the GDN layers' conv + recurrent state. A failure here with the per-block chunked
    tests green points at the model-level threading, not at either mixer.
    """
    cfg = unit_test_config(**REDUCED)
    mesh_config, ccl = mesh_setup(mesh)
    chunk = S_LOCAL * mesh_config.sp
    total = 2 * chunk
    _ref, model = _build(mesh, mesh_config, ccl, cfg)
    token_ids = torch.randint(0, cfg.vocab_size, (1, total), generator=torch.Generator().manual_seed(83))

    one_shot_caches = model.allocate_caches(max_seq_len=total)
    one_shot = from_sp_sharded(
        model.prefill_chunk(token_ids, start_pos=0, caches=one_shot_caches, skip_lm_head=True), mesh_config
    )

    chunked_caches = model.allocate_caches(max_seq_len=total)
    pieces = []
    for c in range(2):
        out = model.prefill_chunk(
            token_ids[:, c * chunk : (c + 1) * chunk],
            start_pos=c * chunk,
            caches=chunked_caches,
            skip_lm_head=True,
        )
        pieces.append(from_sp_sharded(out, mesh_config))
    got = torch.cat(pieces, dim=2)
    check_pcc("model_chunked_vs_one_shot[REDUCED 8L/v4096]", one_shot, got)


@parametrize_mesh()
def test_model_per_layer_kv_vs_ref(mesh, submesh_shape, device_params):
    """Per-layer K/V after a one-shot model run against the reference's captures.

    The same comparison P1 makes against the golden trace, but on random weights and reduced
    depth — so the harness is proven before real weights are in play.
    """
    cfg = unit_test_config(**REDUCED)
    mesh_config, ccl = mesh_setup(mesh)
    total = S_LOCAL * mesh_config.sp
    ref, model = _build(mesh, mesh_config, ccl, cfg)
    token_ids = torch.randint(0, cfg.vocab_size, (1, total), generator=torch.Generator().manual_seed(84))
    with torch.no_grad():
        _out, states = ref(input_ids=token_ids, skip_lm_head=True)

    caches = model.allocate_caches(max_seq_len=total)
    model.prefill_chunk(token_ids, start_pos=0, caches=caches, skip_lm_head=True)

    kv = caches.kv
    seq_local = total // mesh_config.sp
    kv_local = cfg.num_key_value_heads // mesh_config.tp
    for layer_idx in cfg.full_attention_layers:
        slot = kv.slot(0, cfg.kv_slot(layer_idx))
        for name, cache, expected in (
            ("k", kv.k, states[layer_idx].key),
            ("v", kv.v, states[layer_idx].value),
        ):
            dev = ttnn.get_device_tensors(ttnn.to_memory_config(cache, ttnn.DRAM_MEMORY_CONFIG))
            cols = []
            for c in range(mesh_config.tp):
                rows = [
                    ttnn.to_torch(dev[r * mesh_config.tp + c])[slot : slot + 1, :, :seq_local, :]
                    for r in range(mesh_config.sp)
                ]
                cols.append(torch.cat(rows, dim=2))
            got = torch.cat(cols, dim=1).reshape(1, cfg.num_key_value_heads, total, cfg.head_dim)
            check_pcc(f"model_kv_{name}[layer{layer_idx}]", expected, got)
