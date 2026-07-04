# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decisive torch-oracle probe for the commit MoE divergence (#47557).

The batched vs sequential commit KV disagree (0.43). ``probe_commit_l0attn.py``
localized the divergence to the layer-0 MoE EXPERT output (0.17), with everything
upstream (attn / shared_mlp / router) bit-exact. This probe settles WHICH commit
MoE is correct by running the **HF torch reference MoE** (hand-rolled to match
``transformers.models.diffusion_gemma`` ``DiffusionGemmaTextRouter`` +
``DiffusionGemmaTextExperts``, loaded from the real layer-0 checkpoint weights) on
the *identical bit-exact* layer-0 MoE input the device computed, and comparing:

  PCC(torch, batched-experts)   vs   PCC(torch, sequential-decode-experts)

The higher one is the correct kernel. Batched uses ``moe.experts`` /
``sparse_experts_forward`` (the CI-verified prefill MoE); sequential uses
``_commit_experts_decode_forward`` (decode ``sparse_matmul`` nnz=8, never verified).

Runs a 2-layer model (layer 0 is the decisive, bit-exact-input layer).

*** DEVICE-OWNERSHIP: run only when QB2 is free (one process per mesh). ***
"""
from __future__ import annotations

import os
import sys

import torch
from loguru import logger
from safetensors import safe_open


def _pcc(a, b):
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    if a.numel() != b.numel():
        return -1.0
    if torch.allclose(a, b):
        return 1.0
    a = a - a.mean()
    b = b - b.mean()
    d = a.norm() * b.norm()
    return float((a @ b) / d) if d > 0 else 0.0


# ---- HF-faithful torch reference MoE (matches modeling_diffusion_gemma.py) ----


def _rms_norm_scaleless(x, eps):
    # DiffusionGemmaRMSNorm(with_scale=False): x_fp32 * pow(mean(x^2)+eps, -0.5)
    x = x.float()
    ms = x.pow(2).mean(-1, keepdim=True) + eps
    return x * torch.pow(ms, -0.5)


def torch_router(router_input, w, *, hidden_size, num_experts, top_k, eps):
    """HF DiffusionGemmaTextRouter.forward — returns (top_k_weights, top_k_index)."""
    scalar_root_size = hidden_size**-0.5
    hs = _rms_norm_scaleless(router_input, eps)  # [T, H] fp32
    hs = hs * w["scale"].float() * scalar_root_size
    expert_scores = hs @ w["proj.weight"].float().t()  # [T, E]
    router_probs = torch.softmax(expert_scores, dim=-1, dtype=torch.float32)
    top_k_weights, top_k_index = torch.topk(router_probs, k=top_k, dim=-1)
    top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
    top_k_weights = top_k_weights * w["per_expert_scale"].float()[top_k_index]
    return top_k_weights, top_k_index


def torch_experts(expert_input, top_k_index, top_k_weights, w, *, num_experts):
    """HF DiffusionGemmaTextExperts.forward (eager reference)."""
    x = expert_input.float()  # [T, H]
    gate_up_proj = w["gate_up_proj"].float()  # [E, 2*inter, H]
    down_proj = w["down_proj"].float()  # [E, H, inter]
    final = torch.zeros_like(x)  # [T, H]
    expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=num_experts)  # [T, K, E]
    expert_mask = expert_mask.permute(2, 1, 0)  # [E, K, T]
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx in expert_hit:
        e = int(expert_idx[0])
        top_k_pos, token_idx = torch.where(expert_mask[e])
        current = x[token_idx]  # [n, H]
        gate, up = torch.nn.functional.linear(current, gate_up_proj[e]).chunk(2, dim=-1)
        h = torch.nn.functional.gelu(gate, approximate="tanh") * up
        h = torch.nn.functional.linear(h, down_proj[e])  # [n, H]
        h = h * top_k_weights[token_idx, top_k_pos, None]
        final.index_add_(0, token_idx, h.to(final.dtype))
    return final


def load_layer0_moe_weights(ckpt):
    idx_path = os.path.join(ckpt, "model.safetensors.index.json")
    import json

    wm = json.load(open(idx_path))["weight_map"]
    keys = {
        "proj.weight": "model.decoder.layers.0.router.proj.weight",
        "scale": "model.decoder.layers.0.router.scale",
        "per_expert_scale": "model.decoder.layers.0.router.per_expert_scale",
        "gate_up_proj": "model.decoder.layers.0.experts.gate_up_proj",
        "down_proj": "model.decoder.layers.0.experts.down_proj",
    }
    out = {}
    # group by shard file
    by_file = {}
    for short, full in keys.items():
        by_file.setdefault(wm[full], []).append((short, full))
    for fname, items in by_file.items():
        with safe_open(os.path.join(ckpt, fname), framework="pt") as f:
            for short, full in items:
                out[short] = f.get_tensor(full)
    return out


def main():
    import ttnn

    ckpt = os.environ["DG_CKPT"]
    HIDDEN, E, TOPK, EPS = 2816, 128, 8, 1e-6

    logger.info("loading layer-0 MoE weights from checkpoint (host torch reference)…")
    w = load_layer0_moe_weights(ckpt)
    logger.info(
        f"weights: proj{tuple(w['proj.weight'].shape)} scale{tuple(w['scale'].shape)} "
        f"per_expert_scale{tuple(w['per_expert_scale'].shape)} "
        f"gate_up{tuple(w['gate_up_proj'].shape)} down{tuple(w['down_proj'].shape)}"
    )

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D, ttnn.FabricReliabilityMode.STRICT_INIT, None)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4))
    try:
        from models.experimental.diffusion_gemma.checkpoint import (
            build_tt_model_from_checkpoint_dir,
            text_generation_prefixes_for_layers,
        )
        from models.experimental.diffusion_gemma.config import DiffusionConfig
        from models.experimental.diffusion_gemma.tt import commit_batched, commit_decode
        from models.experimental.diffusion_gemma.tt import denoise_forward as _dfwd
        from models.experimental.diffusion_gemma.tt.generate import (
            commit_canvas_tokens,
            denoise_and_commit_block,
            tokenize_prompt,
        )
        from models.experimental.diffusion_gemma.tt.serving import BlockDiffusionServingSession

        mi = build_tt_model_from_checkpoint_dir(
            mesh, ckpt, state_prefixes=text_generation_prefixes_for_layers(2), num_layers=2, max_seq_len=512
        )
        tt_model = mi.tt_model
        cfg = DiffusionConfig(canvas_length=256)
        prompt = tokenize_prompt(mi.tokenizer, "The capital of France is")
        session = BlockDiffusionServingSession(
            tt_model, mi.state_dict, config=cfg, tokenizer=mi.tokenizer, gumbel_mode="argmax", seed=0
        )
        session.prefill(prompt)
        start_pos = session.next_pos

        l0 = tt_model.layers[0]

        # capture committed tokens (no commit yet)
        captured = {}

        def _noop(_m, canvas_tokens, **_k):
            captured["committed"] = canvas_tokens.clone()

        denoise_and_commit_block(
            tt_model,
            session._logits_fn,
            session._init_canvas_fn(0, start_pos),
            cfg,
            start_pos=start_pos,
            gumbel_noise_fn=session._gumbel_noise_fn(0),
            noise_tokens_fn=session._noise_tokens_fn(0),
            commit_fn=_noop,
        )
        committed = captured["committed"]

        cap = {
            "bat_out": None,
            "bat_ri": None,
            "bat_ei": None,
            "bat_route": None,
            "seq_out": [],
            "seq_ri": [],
            "seq_ei": [],
        }

        def _h(x):
            return ttnn.to_torch(ttnn.get_device_tensors(x)[0]).float().clone()

        # capture batched MoE input+output for layer 0. commit_batched imports
        # _denoise_moe_forward into its OWN namespace, so patch it there.
        orig_bmoe = commit_batched._denoise_moe_forward
        orig_brouter = _dfwd._denoise_router_forward

        def bmoe_wrap(moe, router_input, expert_input, *a, **kw):
            out = orig_bmoe(moe, router_input, expert_input, *a, **kw)
            if moe is l0.moe:
                cap["bat_ri"] = _h(router_input)
                cap["bat_ei"] = _h(expert_input)
                cap["bat_out"] = _h(out)
            return out

        def brouter_wrap(router, hs, *a, **kw):
            out = orig_brouter(router, hs, *a, **kw)
            if router is l0.moe.router:
                cap["bat_route"] = _h(out)
            return out

        # capture sequential MoE input+output for layer 0 (per token, concat)
        orig_smoe = commit_decode._commit_moe_forward

        def smoe_wrap(moe, router_input, expert_input, *a, **kw):
            out = orig_smoe(moe, router_input, expert_input, *a, **kw)
            if moe is l0.moe:
                cap["seq_ri"].append(_h(router_input))
                cap["seq_ei"].append(_h(expert_input))
                cap["seq_out"].append(_h(out))
            return out

        # clone caches so both commits start from identical prefix
        def clone_caches(kvc):
            by, cl = {}, []
            for kv in kvc:
                k = id(kv[0])
                if k not in by:
                    by[k] = [ttnn.clone(kv[0]), ttnn.clone(kv[1])]
                cl.append(by[k])
            return cl

        orig_caches = tt_model.tt_kv_cache
        clones = clone_caches(orig_caches)

        # sequential into originals
        commit_decode._commit_moe_forward = smoe_wrap
        try:
            commit_canvas_tokens(tt_model, committed, start_pos=start_pos, page_table=None)
        finally:
            commit_decode._commit_moe_forward = orig_smoe

        # batched into clones
        tt_model.tt_kv_cache = clones
        commit_batched._denoise_moe_forward = bmoe_wrap
        _dfwd._denoise_router_forward = brouter_wrap
        try:
            commit_batched.commit_canvas_tokens_batched(tt_model, committed, start_pos=start_pos, page_table=None)
        finally:
            commit_batched._denoise_moe_forward = orig_bmoe
            _dfwd._denoise_router_forward = orig_brouter
            tt_model.tt_kv_cache = orig_caches

        # ---- assemble ----
        bat_out = cap["bat_out"].reshape(-1, HIDDEN)  # [256, H]
        bat_ri = cap["bat_ri"].reshape(-1, HIDDEN)
        bat_ei = cap["bat_ei"].reshape(-1, HIDDEN)
        seq_out = torch.cat(cap["seq_out"], dim=-2).reshape(-1, HIDDEN)
        seq_ri = torch.cat(cap["seq_ri"], dim=-2).reshape(-1, HIDDEN)
        seq_ei = torch.cat(cap["seq_ei"], dim=-2).reshape(-1, HIDDEN)
        n = min(bat_out.shape[0], seq_out.shape[0])
        bat_out, bat_ri, bat_ei = bat_out[:n], bat_ri[:n], bat_ei[:n]
        seq_out, seq_ri, seq_ei = seq_out[:n], seq_ri[:n], seq_ei[:n]

        print("=" * 72)
        print(f"INPUT bit-exactness (batched vs sequential):")
        print(f"  router_input PCC={_pcc(bat_ri, seq_ri):.6f} max={(bat_ri-seq_ri).abs().max():.3e}")
        print(f"  expert_input PCC={_pcc(bat_ei, seq_ei):.6f} max={(bat_ei-seq_ei).abs().max():.3e}")
        print(f"  batched vs sequential MoE OUTPUT PCC={_pcc(bat_out, seq_out):.6f}  (the 0.17 divergence)")

        # ---- torch reference on the batched input (its own routing) ----
        tw, ti = torch_router(bat_ri, w, hidden_size=HIDDEN, num_experts=E, top_k=TOPK, eps=EPS)
        torch_out = torch_experts(bat_ei, ti, tw, w, num_experts=E)

        # torch with routing on the SEQUENTIAL input too (should be near-identical input)
        tw2, ti2 = torch_router(seq_ri, w, hidden_size=HIDDEN, num_experts=E, top_k=TOPK, eps=EPS)
        torch_out_seqin = torch_experts(seq_ei, ti2, tw2, w, num_experts=E)

        print("-" * 72)
        print("TORCH ORACLE (HF DiffusionGemma MoE, fp32, real layer-0 weights):")
        print(f"  PCC(torch,  batched )  = {_pcc(torch_out, bat_out):.6f}   <-- higher = correct kernel")
        print(f"  PCC(torch, sequential) = {_pcc(torch_out_seqin, seq_out):.6f}")
        print(f"  (torch batched-input vs torch seq-input) = {_pcc(torch_out, torch_out_seqin):.6f}")
        # per-expert routing agreement between torch and device
        if cap["bat_route"] is not None:
            dev_route = cap["bat_route"].reshape(-1, E)[:n]
            dev_nnz = (dev_route.abs() > 1e-9).float()
            torch_nnz = torch.zeros_like(dev_nnz).scatter_(1, ti, 1.0)
            agree = (dev_nnz == torch_nnz).float().mean().item()
            print(f"  torch-vs-device selected-expert agreement = {agree:.4f}")
        print("=" * 72)
        verdict = (
            "BATCHED matches torch (sequential is the buggy kernel)"
            if _pcc(torch_out, bat_out) > _pcc(torch_out_seqin, seq_out)
            else "SEQUENTIAL matches torch (batched is wrong)"
        )
        print(f"VERDICT: {verdict}")
    finally:
        ttnn.close_mesh_device(mesh)
        if mesh.get_num_devices() > 1:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    sys.exit(main())
