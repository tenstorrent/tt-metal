#!/usr/bin/env python3
"""
ZAYA1-8B reference forward on CPU -> dump golden tensors for tt-metal PCC validation.

Run (after weights downloaded + Zyphra transformers fork installed):
    HF_HOME=/home/yito/work/hf_cache \
    /home/yito/work/zaya-ref-venv/bin/python dump_golden.py

Outputs:
    reference/golden/config_report.txt     <- resolves the open questions (head counts, layer list, shapes)
    reference/golden/inputs.pt             <- fixed input_ids / position_ids used
    reference/golden/<module>.pt           <- per-module input/output tensors (layer 0 ATT, layer 1 MoE, final, logits)

This is intentionally CPU + fp32 so it is a clean numerical reference; tt-metal bf16 is compared via PCC.
"""
import os
import sys
import json
from pathlib import Path

import torch

MODEL_ID = "Zyphra/ZAYA1-8B"
HERE = Path(__file__).resolve().parent
OUT = HERE / "golden"
OUT.mkdir(exist_ok=True)

# Small, deterministic prompt. Short seq keeps the dump light and fast on CPU.
PROMPT = "The capital of France is"
SEQ_CAP = 16  # truncate just in case

torch.manual_seed(0)


def main():
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

    print(f"[load] config {MODEL_ID}")
    cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)

    # ---- Resolve open questions from the actual checkpoint config ----
    report = []
    def rep(line):
        print(line)
        report.append(line)

    rep("=== ZAYA1-8B config resolution ===")
    for k in [
        "model_type", "hidden_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "head_dim", "kv_channels", "partial_rotary_factor",
        "rope_theta", "moe_router_topk", "ffn_hidden_size", "vocab_size",
        "zaya_use_mod", "zaya_use_eda", "scale_residual_merge", "residual_in_fp32",
        "tie_word_embeddings",
    ]:
        rep(f"{k} = {getattr(cfg, k, '<absent>')}")
    for k in ["zaya_layers", "cca_num_q_heads", "num_query_groups_list",
              "ffn_hidden_size_list", "zaya_mlp_expansion"]:
        v = getattr(cfg, k, None)
        if isinstance(v, list):
            rep(f"{k} (len {len(v)}) = {v[:8]}{' ...' if len(v) > 8 else ''}")
        else:
            rep(f"{k} = {v}")

    # ---- Patch checkpoint/modeling config skew ----
    # Published config has zaya_mlp_expansion as a scalar (256) but the fork's
    # modeling indexes it per-layer (config.zaya_mlp_expansion[layer_n]).
    # Expand to a per-layer list: 256 at MoE layers, 0 at attention layers.
    if not isinstance(cfg.zaya_mlp_expansion, (list, tuple)):
        scalar = int(cfg.zaya_mlp_expansion)
        cfg.zaya_mlp_expansion = [
            (scalar if isinstance(l, int) else 0) for l in cfg.zaya_layers
        ]
        rep(f"[patch] expanded zaya_mlp_expansion scalar {scalar} -> list len {len(cfg.zaya_mlp_expansion)}")

    # Published config has num_attention_heads=8, but checkpoint weight shapes
    # (linear_q=[1024,2048]=8*128, o_proj in=1024, key=2*128) only close with
    # head_dim=128 => num_attention_heads=16. Correct the skew.
    if cfg.hidden_size // cfg.num_attention_heads != cfg.head_dim:
        fixed = cfg.hidden_size // cfg.head_dim
        rep(f"[patch] num_attention_heads {cfg.num_attention_heads} -> {fixed} "
            f"(to match head_dim={cfg.head_dim})")
        cfg.num_attention_heads = fixed

    print(f"[load] tokenizer")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    print(f"[load] model (fp32, cpu) — this is large, be patient")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, config=cfg, trust_remote_code=True, dtype=torch.float32,
        attn_implementation="eager", low_cpu_mem_usage=True,
    )
    model.eval()

    # ---- Runtime attention geometry (ground truth for the port) ----
    sa = model.model.layers[0].self_attn
    rep("\n=== runtime attention geometry (layers[0].self_attn) ===")
    rep(f"num_heads={sa.num_heads} head_dim={sa.head_dim} "
        f"num_key_value_heads={sa.num_key_value_heads} "
        f"num_key_value_groups={sa.num_key_value_groups}")
    rep(f"cca: num_q_heads={sa.qkv.num_q_heads} num_kv_heads={sa.qkv.num_kv_heads} "
        f"cca_head_dim={sa.qkv.head_dim} latent_q={sa.qkv.latent_q_dim} latent_k={sa.qkv.latent_k_dim}")

    # Report concrete module structure for the first ATT + first MoE layer
    rep("\n=== layer types (first 6) ===")
    for i, layer in enumerate(model.model.layers[:6]):
        rep(f"layers[{i}] = {type(layer).__name__}")
    rep(f"lm_head.weight is embed_tokens.weight: "
        f"{model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()}")

    enc = tok(PROMPT, return_tensors="pt")
    input_ids = enc.input_ids[:, :SEQ_CAP]
    rep(f"\ninput_ids shape {tuple(input_ids.shape)}: {input_ids.tolist()}")
    torch.save({"input_ids": input_ids, "prompt": PROMPT}, OUT / "inputs.pt")

    # ---- Hooks: dump IO of a curated set of modules ----
    saved = {}

    def make_hook(name):
        def hook(mod, inp, out):
            def to_cpu(x):
                if torch.is_tensor(x):
                    return x.detach().to(torch.float32).cpu()
                if isinstance(x, (tuple, list)):
                    return [to_cpu(t) for t in x]
                return None
            saved[name] = {"in": [to_cpu(x) for x in inp], "out": to_cpu(out)}
        return hook

    handles = []
    def watch(name, mod):
        handles.append(mod.register_forward_hook(make_hook(name)))

    m = model.model
    watch("embed_tokens", m.embed_tokens)
    watch("rotary_emb", m.rotary_emb)
    # First attention layer (expected layers[0]) and its internals
    att = m.layers[0]
    watch("L0_att_layer", att)
    watch("L0_input_norm", att.input_norm)
    watch("L0_self_attn", att.self_attn)
    watch("L0_cca_qkv", att.self_attn.qkv)
    watch("L0_o_proj", att.self_attn.o_proj)
    # CCA internals (submodules) for fine-grained Phase 3 debugging
    watch("L0_linear_q", att.self_attn.qkv.linear_q)
    watch("L0_linear_k", att.self_attn.qkv.linear_k)
    watch("L0_conv_qk", att.self_attn.qkv.conv_qk)
    watch("L0_val1", att.self_attn.qkv.val_proj1)
    watch("L0_val2", att.self_attn.qkv.val_proj2)
    # First MoE layer (expected layers[1]) and its internals
    moe = m.layers[1]
    watch("L1_moe_layer", moe)
    watch("L1_input_norm", moe.input_norm)
    watch("L1_zaya_block", moe.zaya_block)
    watch("L1_router", moe.zaya_block.router)
    watch("L1_experts", moe.zaya_block.experts)
    watch("final_norm", m.final_norm)
    watch("lm_head", model.lm_head)

    print("[run] forward (no cache)")
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False, output_hidden_states=True)

    logits = out.logits
    saved["logits"] = {"in": [], "out": logits.detach().float().cpu()}
    rep(f"\nlogits shape {tuple(logits.shape)}")
    top = torch.topk(logits[0, -1], 5)
    rep(f"last-token top5 ids: {top.indices.tolist()}")
    rep(f"last-token top5 decoded: {[tok.decode([i]) for i in top.indices.tolist()]}")

    for h in handles:
        h.remove()

    # Per-layer residual-stream hidden states (input hidden to each layer + final)
    if out.hidden_states is not None:
        hs = [h.detach().float().cpu() for h in out.hidden_states]
        torch.save(hs, OUT / "hidden_states.pt")
        rep(f"[done] saved hidden_states: {len(hs)} entries, each {tuple(hs[0].shape)}")

    for name, io in saved.items():
        torch.save(io, OUT / f"{name}.pt")
    rep(f"\n[done] dumped {len(saved)} module IO files to {OUT}")

    (OUT / "config_report.txt").write_text("\n".join(report))
    print(f"[done] wrote {OUT/'config_report.txt'}")


if __name__ == "__main__":
    main()
