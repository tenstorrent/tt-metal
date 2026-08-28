# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-component parity for the graduated stubs AS THE PIPELINE HOLDS THEM.

`tests/pcc/` checks each stub in isolation, built fresh from the full HF checkpoint.  This file
checks the very objects `tt/pipeline.py` chained together -- `pipeline.layers[0].mlp.router` is the
`top_k_router` instance the MoE block actually calls -- against the golden inputs Source B captured
under `_captured/`.  It is the regression net for the re-routing the pipeline does (the MoE block
now calls the graduated router and shared expert, the delta net calls the graduated gated norm, the
decoder layer calls the graduated RMSNorm) and for the mesh-aware sharding.

The goldens are RECOMPUTED from the HF submodule on the captured inputs with the cache disabled,
because the pipeline's ports are cache-free by construction; the stored `output.pt` was captured
mid-generation with a live `DynamicCache` and is not the right target for a cache-free port.

    ./python_env/bin/python -m pytest models/demos/qwen3_coder_next/tests/e2e/test_captured_parity.py -s
"""
from __future__ import annotations

import os

import torch

import ttnn

from models.demos.qwen3_coder_next.tt import mesh as tt_mesh
from models.demos.qwen3_coder_next.tt.pipeline import _pcc
from models.demos.qwen3_coder_next._stubs.gated_delta_net import to_device

CAPTURED = os.path.join(os.path.dirname(__file__), "..", "..", "_captured")
COMPONENT_PCC = 0.99


def _load(component):
    args = torch.load(os.path.join(CAPTURED, component, "args.pt"), weights_only=False)
    kwargs = torch.load(os.path.join(CAPTURED, component, "kwargs.pt"), weights_only=False)
    return args, kwargs


def _first(x):
    return x[0] if isinstance(x, tuple) else x


def _host(x):
    return (x if isinstance(x, torch.Tensor) else tt_mesh.to_host(x)).float().reshape(-1)


def test_graduated_components_match_captured_golden(pipeline):
    """Every graduated stub instance the pipeline holds still reproduces its HF golden."""
    # `layer_types` runs 3 x linear_attention then 1 x full_attention, so a 4-layer build is the
    # shallowest one that holds BOTH token mixers -- and both are checked below.
    assert pipeline.depth >= 4, (
        f"built at depth {pipeline.depth}: a >=4-layer build is required for the stack to contain "
        f"the full_attention layer this test checks the `attention` stub against"
    )

    hf = pipeline.reference.model
    layer0, layer3 = hf.layers[0], hf.layers[3]
    device = pipeline.device

    tt_layer0, tt_layer3 = pipeline.layers[0], pipeline.layers[3]
    tt_moe = tt_layer0.mlp

    results = {}

    def check(name, golden, got):
        value = _pcc(golden.float().reshape(-1), _host(got))
        results[name] = value
        print(f"  {name:18s} PCC={value:.6f}")

    with torch.no_grad():
        # ---- rotary_embedding -----------------------------------------------------------
        args, _ = _load("rotary_embedding")
        x, position_ids = args
        ref_cos, ref_sin = hf.rotary_emb(x, position_ids)
        cos, sin = pipeline.rope(x, position_ids)
        check("rotary_embedding.cos", ref_cos, cos)
        check("rotary_embedding.sin", ref_sin, sin)

        # ---- r_m_s_norm -----------------------------------------------------------------
        # `_captured/r_m_s_norm` resolved to the GATED norm's path during bring-up, so its stored
        # golden is a gated-norm output. The pipeline routes this stub at `input_layernorm`, which
        # is what is checked here.
        hidden = _load("decoder_layer")[0][0]
        seq = hidden.shape[-2]
        ref_norm = layer0.input_layernorm(hidden)
        got_norm = tt_layer0.input_layernorm(to_device(hidden.float().reshape(1, 1, seq, -1), device))
        check("r_m_s_norm", ref_norm, got_norm)

        # ---- r_m_s_norm_gated -----------------------------------------------------------
        args, _ = _load("r_m_s_norm_gated")
        ref_gated = layer0.linear_attn.norm(args[0], args[1])
        got_gated = tt_layer0.mixer.norm(
            to_device(args[0].float().reshape(1, 1, *args[0].shape), device),
            to_device(args[1].float().reshape(1, 1, *args[1].shape), device),
        )
        check("r_m_s_norm_gated", ref_gated, got_gated)

        # ---- m_l_p (the shared expert) --------------------------------------------------
        args, _ = _load("m_l_p")
        ref_mlp = layer0.mlp.shared_expert(args[0])
        got_mlp = tt_moe.shared_expert(to_device(args[0].float().reshape(1, 1, *args[0].shape), device))
        check("m_l_p", ref_mlp, got_mlp)

        # ---- top_k_router ---------------------------------------------------------------
        args, _ = _load("top_k_router")
        ref_logits, ref_scores, ref_index = layer0.mlp.gate(args[0])
        logits, scores, index = tt_moe.router(to_device(args[0].float().reshape(1, 1, *args[0].shape), device))
        check("top_k_router.logits", ref_logits, logits)
        check("top_k_router.scores", ref_scores, scores)
        assert tt_mesh.to_host(index).flatten().tolist() == ref_index.flatten().tolist(), (
            "top_k_router selected different experts than the golden"
        )

        # ---- experts --------------------------------------------------------------------
        args, _ = _load("experts")
        hidden_e, index_e, weights_e = args
        ref_experts = layer0.mlp.experts(hidden_e, index_e, weights_e)
        x_e = to_device(hidden_e.float().reshape(1, 1, *hidden_e.shape), device)
        routing = tt_moe.experts.dense_routing(index_e, weights_e, hidden_e.shape[0])
        got_experts = tt_moe.experts.partial(x_e, tt_moe.experts.local_routing(routing), hidden_e.shape[0])
        if tt_moe.experts.num_devices > 1:
            # `partial()` returns this chip's partial sum over the model dim, exactly as
            # `sparse_moe_block` receives it; the block's own all_reduce is what completes it.
            got_experts = ttnn.all_reduce(got_experts)
        check("experts", ref_experts, got_experts)

        # ---- sparse_moe_block -----------------------------------------------------------
        args, _ = _load("sparse_moe_block")
        ref_moe = _first(layer0.mlp(args[0]))
        got_moe = tt_moe(to_device(args[0].float().reshape(1, 1, args[0].shape[-2], -1), device))
        check("sparse_moe_block", ref_moe, got_moe)

        # ---- gated_delta_net ------------------------------------------------------------
        _, kwargs = _load("gated_delta_net")
        h = kwargs["hidden_states"]
        ref_dn = layer0.linear_attn(hidden_states=h, cache_params=None, attention_mask=None)
        got_dn = tt_layer0.mixer(to_device(h.float().reshape(1, 1, h.shape[-2], -1), device))
        check("gated_delta_net", _first(ref_dn), got_dn)

        # ---- attention ------------------------------------------------------------------
        _, kwargs = _load("attention")
        h = kwargs["hidden_states"]
        cos, sin = kwargs["position_embeddings"]
        ref_attn = _first(
            layer3.self_attn(
                hidden_states=h, position_embeddings=(cos, sin), attention_mask=None, past_key_values=None
            )
        )
        got_attn = tt_layer3.mixer(
            to_device(h.float().reshape(1, 1, h.shape[-2], -1), device), position_embeddings=(cos, sin)
        )
        check("attention", ref_attn, got_attn)

        # ---- decoder_layer --------------------------------------------------------------
        args, kwargs = _load("decoder_layer")
        h = args[0]
        ref_layer = _first(
            layer0(
                h,
                position_embeddings=kwargs["position_embeddings"],
                attention_mask=None,
                position_ids=kwargs["position_ids"],
                past_key_values=None,
            )
        )
        got_layer = tt_layer0(
            to_device(h.float().reshape(1, 1, h.shape[-2], -1), device),
            position_embeddings=kwargs["position_embeddings"],
            attention_mask=None,
        )
        check("decoder_layer", ref_layer, got_layer)

    failed = {k: v for k, v in results.items() if not (v >= COMPONENT_PCC)}
    print(f"component parity: {len(results) - len(failed)}/{len(results)} at PCC >= {COMPONENT_PCC}")
    assert not failed, f"graduated components below PCC {COMPONENT_PCC}: {failed}"
