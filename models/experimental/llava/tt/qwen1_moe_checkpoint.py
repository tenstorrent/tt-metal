# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Checkpoint converter for MoE-LLaVA-Qwen-1.8B-4e.

The HF checkpoint uses the original Qwen1 naming and DeepSpeed's MoE wrapper:
    - transformer.h.{l}.attn.c_attn.weight  (fused QKV, [6144, 2048], nn.Linear layout)
    - transformer.h.{l}.attn.c_attn.bias    (fused QKV bias, [6144])
    - transformer.h.{l}.attn.c_proj.weight (output proj, [2048, 2048])
    - transformer.h.{l}.ln_1.weight, ln_2.weight
    - transformer.h.{l}.mlp.w1.weight, w2.weight, c_proj.weight
      (dense layers, layers NOT in moe_layers_idx)
    - transformer.h.{l}.mlp.wg.weight
    - transformer.h.{l}.mlp.experts.{e}.w1.weight, w2.weight, c_proj.weight
      (MoE layers, where l in moe_layers_idx)
    - transformer.wte.weight (token embedding)
    - transformer.ln_f.weight
    - output.weight

The tt-metal `ModelArgs` / `Transformer` expect Meta-format keys:
    - layers.{l}.attention.wq.weight, wk.weight, wv.weight, wo.weight, wo.bias
    - layers.{l}.attention_norm.weight, layers.{l}.ffn_norm.weight
    - layers.{l}.feed_forward.w1.weight, w2.weight, w3.weight
      (dense, LLaMA-style SwiGLU: w1=gate, w3=up, w2=down)
    - layers.{l}.block_sparse_moe.gate.weight (MoE router)
    - layers.{l}.block_sparse_moe.experts.{e}.w1.weight, w2.weight, w3.weight
    - tok_embeddings.weight, norm.weight, output.weight

Key remapping rules:
    Attention:
        c_attn.weight [3H, H] -> split into wq.weight [H, H], wk.weight [H, H], wv.weight [H, H]
        c_attn.bias [3H]      -> split into wq.bias [H], wk.bias [H], wv.bias [H]
        c_proj.weight [H, H]  -> wo.weight [H, H]
    MLP (dense layer):
        Qwen SwiGLU mapping: w1 -> feed_forward.w1 (gate), w2 -> feed_forward.w3 (up),
                             c_proj -> feed_forward.w2 (down)
        Note Qwen names are reversed from LLaMA: Qwen's w1 acts on gate, w2 acts on up.
    MoE (MoE layer):
        mlp.wg.weight -> block_sparse_moe.gate.weight
        mlp.experts.{e}.w1.weight -> block_sparse_moe.experts.{e}.w1.weight (gate)
        mlp.experts.{e}.w2.weight -> block_sparse_moe.experts.{e}.w3.weight (up)
        mlp.experts.{e}.c_proj.weight -> block_sparse_moe.experts.{e}.w2.weight (down)
    Norm:
        ln_1 -> attention_norm, ln_2 -> ffn_norm, ln_f -> norm
    Embedding:
        wte.weight -> tok_embeddings.weight

Reference: MoE-LLaVA/moellava/model/language_model/qwen/{modeling_qwen,llava_qwen_moe}.py
"""

from typing import Dict, Iterable

import torch


def split_qwen_fused_qkv(c_attn_weight: torch.Tensor, num_heads: int, head_dim: int) -> Dict[str, torch.Tensor]:
    """Split Qwen1's fused `c_attn` weight `[3H, H]` into separate Q, K, V tensors.

    nn.Linear stores weight as [out_features, in_features], so the serialized
    fused QKV is `[3H, H]` with the Q|K|V blocks stacked along dim 0 (the
    LanguageBind checkpoint: c_attn.weight [6144, 2048]). Per Qwen1's MHA,
    num_heads == num_kv_heads, so each Q, K, V split is the full hidden dim.

    Returns Meta-format tensors (no transpose — the linear layer transposes
    internally). Caller responsible for emitting `wq`, `wk`, `wv` keys.
    """
    H = c_attn_weight.shape[1]
    assert c_attn_weight.shape[0] == 3 * H, f"Expected c_attn shape [3H, H], got {c_attn_weight.shape}"
    q, k, v = torch.split(c_attn_weight, [H, H, H], dim=0)
    return {"wq": q, "wk": k, "wv": v}


def split_qwen_fused_qkv_bias(c_attn_bias: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Split Qwen1's fused `c_attn` bias `[3H]` into separate Q, K, V bias vectors.

    Qwen1 attention carries a QKV bias (`nn.Linear(..., bias=True)`,
    `modeling_qwen.py`); the serialized checkpoint has `attn.c_attn.bias [3H]`.
    tt-metal `Attention` consumes per-projection biases as `wq.bias/wk.bias/wv.bias`
    (models/tt_transformers/tt/attention.py:186-260), so emit them in Meta format.
    """
    H = c_attn_bias.shape[0]
    assert H % 3 == 0, f"Expected c_attn bias length 3H, got {H}"
    q, k, v = torch.split(c_attn_bias, [H // 3, H // 3, H // 3], dim=0)
    return {"wq": q, "wk": k, "wv": v}


def qwen_mlp_to_meta(mlp_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Map a single dense QwenMLP's w1/w2/c_proj weights to Meta/LaMA-format w1/w2/w3.

    Qwen1's SwiGLU forward (`MoE-LLaVA/.../qwen/modeling_qwen.py:570`):
        a1 = self.w1(hidden_states)            # up branch (NOT activated)
        a2 = self.w2(hidden_states)            # gate branch (silu-applied)
        intermediate_parallel = a1 * F.silu(a2)
        output = self.c_proj(intermediate_parallel)
    LLaMA/Meta SwiGLU expected by tt-metal `MLP.forward` (`mlp.py`):
        w2(silu(w1(x)) * w3(x))
    Because multiplication is commutative, Silicon-flow identity requires:
        silu(Meta.w1(x)) = silu(Qwen.w2(x)) -> Meta.w1 = Qwen.w2 (gate)
        Meta.w3(x)         = Qwen.w1(x)      -> Meta.w3 = Qwen.w1 (up)
        Meta.w2(...)      = Qwen.c_proj(...)  -> Meta.w2 = Qwen.c_proj (down)

    Args:
        mlp_state: {"w1.weight": (up), "w2.weight": (gate), "c_proj.weight": (down)}

    Returns:
        {"w1.weight": gate, "w3.weight": up, "w2.weight": down}
    """
    return {
        "w1.weight": mlp_state["w2.weight"],  # gate (LLaMA w1 takes silu)
        "w3.weight": mlp_state["w1.weight"],  # up (LLaMA w3 is unactivated)
        "w2.weight": mlp_state["c_proj.weight"],  # down
    }


def qwen_moe_expert_to_meta(expert_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Map a single Qwen1 MoE expert's weights to Meta/Mixtral-format w1/w2/w3.

    Same SwiGLU math as the dense MLP (Qwen training uses the same `QWenMLP`
    for both the dense layer and as the MoE expert class):
        Qwen.w2 (gate, silu-applied) -> Meta.w1
        Qwen.w1 (up, no activation)  -> Meta.w3
        Qwen.c_proj (down)           -> Meta.w2

    Args:
        expert_state: {"w1.weight": (up), "w2.weight": (gate), "c_proj.weight": (down)}

    Returns:
        {"w1.weight": gate, "w2.weight": down, "w3.weight": up}
    """
    return {
        "w1.weight": expert_state["w2.weight"],  # gate (LLaMA w1 takes silu)
        "w3.weight": expert_state["w1.weight"],  # up (LLaMA w3 is unactivated)
        "w2.weight": expert_state["c_proj.weight"],  # down
    }


def convert_qwen_moe_to_meta(
    state_dict: Dict[str, torch.Tensor],
    n_layers: int,
    moe_layer_ids: Iterable[int],
    num_heads: int = 16,
    head_dim: int = 128,
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Convert MoE-LLaVA-Qwen-1.8B-4e HF state dict to tt-metal Meta format.

    Args:
        state_dict: HF `model.state_dict()` from the MoE-LLaVA-Qwen-1.8B-4e checkpoint,
                    in PyTorch's `transformer.h.{l}...` naming. NOTE: this function MUTATES
                    the dict via `pop()` to consume keys; pass a copy if you need to
                    retain the original.
        n_layers:   Number of transformer layers (24 for Qwen-1.8B-4e)
        moe_layer_ids: Iterable of layer indices that have MoE (e.g., [0, 2, ..., 22])
        num_heads:  Attention head count (16 for Qwen-1.8B-4e, MHA — n_kv_heads == n_heads)
        head_dim:   Head dimension (128 for Qwen-1.8B-4e)

    Returns:
        (meta_state_dict, leftover_hf_state_dict) — the Meta-format keys consumed by
        tt-metal `Transformer` (`layers.{l}.attention.{wq,wk,wv,wo}.weight`,
        `layers.{l}.{attention_norm,ffn_norm}.weight`,
        `layers.{l}.feed_forward.w{1,2,3}.weight` for dense layers,
        `layers.{l}.block_sparse_moe.gate.weight` and
        `layers.{l}.block_sparse_moe.experts.{e}.w{1,2,3}.weight` for MoE layers,
        plus `tok_embeddings.weight`, `norm.weight`, `output.weight`),
        and the remaining unused HF keys (for auditing).
    """
    out: Dict[str, torch.Tensor] = {}

    for l in range(n_layers):
        attn_prefix = f"transformer.h.{l}.attn"
        # Fused QKV split: c_attn.weight [3H, H] -> wq, wk, wv each [H, H]
        c_attn_w = state_dict.pop(f"{attn_prefix}.c_attn.weight")
        qkv = split_qwen_fused_qkv(c_attn_w, num_heads=num_heads, head_dim=head_dim)
        out[f"layers.{l}.attention.wq.weight"] = qkv["wq"]
        out[f"layers.{l}.attention.wk.weight"] = qkv["wk"]
        out[f"layers.{l}.attention.wv.weight"] = qkv["wv"]
        # Fused QKV bias: c_attn.bias [3H] -> wq.bias, wk.bias, wv.bias (Qwen1
        # attention uses nn.Linear(bias=True); tt-metal Attention consumes the
        # split biases when they are present in the Meta state dict).
        if f"{attn_prefix}.c_attn.bias" in state_dict:
            qkv_b = split_qwen_fused_qkv_bias(state_dict.pop(f"{attn_prefix}.c_attn.bias"))
            out[f"layers.{l}.attention.wq.bias"] = qkv_b["wq"]
            out[f"layers.{l}.attention.wk.bias"] = qkv_b["wk"]
            out[f"layers.{l}.attention.wv.bias"] = qkv_b["wv"]
        # Output projection: c_proj -> wo (no bias in Qwen1)
        out[f"layers.{l}.attention.wo.weight"] = state_dict.pop(f"{attn_prefix}.c_proj.weight")

        # RMSNorm weights: ln_1 -> attention_norm, ln_2 -> ffn_norm (Qwen1 norms have no bias)
        out[f"layers.{l}.attention_norm.weight"] = state_dict.pop(f"transformer.h.{l}.ln_1.weight")
        out[f"layers.{l}.ffn_norm.weight"] = state_dict.pop(f"transformer.h.{l}.ln_2.weight")

        # MoE layer key layout differs between sources:
        #   - serialized LanguageBind checkpoints use the DeepSpeed MoE wrapper:
        #     mlp.deepspeed_moe.gate.wg / mlp.deepspeed_moe.experts.deepspeed_experts.{e}
        #   - the MoE-LLaVA modeling code (and our synthetic CPU tests) use the
        #     flat form: mlp.wg / mlp.experts.{e}
        layer_mlp_prefix = f"transformer.h.{l}.mlp"
        if f"{layer_mlp_prefix}.deepspeed_moe.gate.wg.weight" in state_dict:
            gate_key = f"{layer_mlp_prefix}.deepspeed_moe.gate.wg.weight"
            expert_template = f"{layer_mlp_prefix}.deepspeed_moe.experts.deepspeed_experts.{{e}}.{{name}}"
            expert_key_prefix = f"{layer_mlp_prefix}.deepspeed_moe.experts.deepspeed_experts."
        elif f"{layer_mlp_prefix}.wg.weight" in state_dict:
            gate_key = f"{layer_mlp_prefix}.wg.weight"
            expert_template = f"{layer_mlp_prefix}.experts.{{e}}.{{name}}"
            expert_key_prefix = f"{layer_mlp_prefix}.experts."
        else:
            gate_key = None
            expert_key_prefix = None

        if gate_key is not None:
            # MoE layer
            out[f"layers.{l}.block_sparse_moe.gate.weight"] = state_dict.pop(gate_key)
            # Count experts by scanning state_dict for this layer
            expert_keys = [k for k in state_dict if k.startswith(expert_key_prefix)]
            expert_ids = sorted({int(k[len(expert_key_prefix) :].split(".")[0]) for k in expert_keys})
            for e in expert_ids:
                expert_state = {
                    name: state_dict.pop(expert_template.format(e=e, name=name))
                    for name in ("w1.weight", "w2.weight", "c_proj.weight")
                }
                meta_expert = qwen_moe_expert_to_meta(expert_state)
                for meta_key, val in meta_expert.items():
                    out[f"layers.{l}.block_sparse_moe.experts.{e}.{meta_key}"] = val
        else:
            # Dense layer: mlp.w1, w2, c_proj
            mlp_state = {
                "w1.weight": state_dict.pop(f"{layer_mlp_prefix}.w1.weight"),
                "w2.weight": state_dict.pop(f"{layer_mlp_prefix}.w2.weight"),
                "c_proj.weight": state_dict.pop(f"{layer_mlp_prefix}.c_proj.weight"),
            }
            meta_mlp = qwen_mlp_to_meta(mlp_state)
            for meta_key, val in meta_mlp.items():
                out[f"layers.{l}.feed_forward.{meta_key}"] = val

    # Embedding (token), final norm, LM head
    if "transformer.wte.weight" in state_dict:
        out["tok_embeddings.weight"] = state_dict.pop("transformer.wte.weight")
    if "transformer.ln_f.weight" in state_dict:
        out["norm.weight"] = state_dict.pop("transformer.ln_f.weight")
    if "lm_head.weight" in state_dict:
        out["output.weight"] = state_dict.pop("lm_head.weight")

    # Place any leftover unused keys in a leftover dict so the caller can audit them
    return out, state_dict


def demo_roundtrip_smoke_test():
    """Tiny pure-Python syntactic smoke test of conversion shape math.

    Builds a synthetic mini-Qwen checkpoint (n_layers=2, hidden=64, head_dim=8,
    num_heads=8, 1 MoE layer with 2 experts, 1 dense layer), runs the converter,
    and asserts every Meta-format key exists with the expected shape.

    Run as `python -m models.demos.llava.tt.qwen1_moe_checkpoint` or directly:
        python models/demos/llava/tt/qwen1_moe_checkpoint.py
    """
    H = 64
    n_layers = 2
    num_heads = 8
    head_dim = H // num_heads
    I = 128  # intermediate size (Qwen uses 11008//2 = 5504 in production)
    num_experts = 2
    moe_layer_ids = [0]

    sd = {}
    # token embedding, final norm, lm_head
    sd["transformer.wte.weight"] = torch.zeros(151936, H)
    sd["transformer.ln_f.weight"] = torch.ones(H)
    sd["lm_head.weight"] = torch.zeros(151936, H)

    for l in range(n_layers):
        # attention (fused QKV + output proj) — nn.Linear layout [out, in]
        sd[f"transformer.h.{l}.attn.c_attn.weight"] = torch.zeros(3 * H, H)
        sd[f"transformer.h.{l}.attn.c_attn.bias"] = torch.zeros(3 * H)
        sd[f"transformer.h.{l}.attn.c_proj.weight"] = torch.zeros(H, H)
        sd[f"transformer.h.{l}.ln_1.weight"] = torch.ones(H)
        sd[f"transformer.h.{l}.ln_2.weight"] = torch.ones(H)
        if l in moe_layer_ids:
            sd[f"transformer.h.{l}.mlp.wg.weight"] = torch.zeros(H, num_experts)
            for e in range(num_experts):
                sd[f"transformer.h.{l}.mlp.experts.{e}.w1.weight"] = torch.zeros(I // 2, H)
                sd[f"transformer.h.{l}.mlp.experts.{e}.w2.weight"] = torch.zeros(I // 2, H)
                sd[f"transformer.h.{l}.mlp.experts.{e}.c_proj.weight"] = torch.zeros(H, I // 2)
        else:
            sd[f"transformer.h.{l}.mlp.w1.weight"] = torch.zeros(I // 2, H)
            sd[f"transformer.h.{l}.mlp.w2.weight"] = torch.zeros(I // 2, H)
            sd[f"transformer.h.{l}.mlp.c_proj.weight"] = torch.zeros(H, I // 2)

    out, leftover = convert_qwen_moe_to_meta(sd, n_layers=n_layers, moe_layer_ids=moe_layer_ids)

    # Assert every required Meta-format key exists with the expected shape
    expected = {
        "tok_embeddings.weight": (151936, H),
        "norm.weight": (H,),
        "output.weight": (151936, H),
    }
    for l in range(n_layers):
        expected[f"layers.{l}.attention.wq.weight"] = (H, H)
        expected[f"layers.{l}.attention.wk.weight"] = (H, H)
        expected[f"layers.{l}.attention.wv.weight"] = (H, H)
        expected[f"layers.{l}.attention.wq.bias"] = (H,)
        expected[f"layers.{l}.attention.wk.bias"] = (H,)
        expected[f"layers.{l}.attention.wv.bias"] = (H,)
        expected[f"layers.{l}.attention.wo.weight"] = (H, H)
        expected[f"layers.{l}.attention_norm.weight"] = (H,)
        expected[f"layers.{l}.ffn_norm.weight"] = (H,)
        if l in moe_layer_ids:
            expected[f"layers.{l}.block_sparse_moe.gate.weight"] = (H, num_experts)
            for e in range(num_experts):
                expected[f"layers.{l}.block_sparse_moe.experts.{e}.w1.weight"] = (I // 2, H)
                expected[f"layers.{l}.block_sparse_moe.experts.{e}.w3.weight"] = (I // 2, H)
                expected[f"layers.{l}.block_sparse_moe.experts.{e}.w2.weight"] = (H, I // 2)
        else:
            expected[f"layers.{l}.feed_forward.w1.weight"] = (I // 2, H)
            expected[f"layers.{l}.feed_forward.w3.weight"] = (I // 2, H)
            expected[f"layers.{l}.feed_forward.w2.weight"] = (H, I // 2)

    for k, exp_shape in expected.items():
        assert k in out, f"Missing Meta-format key: {k}"
        assert tuple(out[k].shape) == exp_shape, f"Wrong shape for {k}: {tuple(out[k].shape)} != {exp_shape}"

    # No Qwen-specific keys should be left over — every pop()'d key was consumed
    expect_residual = set()
    assert set(leftover.keys()) == expect_residual, f"Unexpected leftover keys: {set(leftover.keys())}"

    print("OK: Qwen1-MoE checkpoint converter smoke test passed")


if __name__ == "__main__":
    # First: structural shape-conversion smoke test
    demo_roundtrip_smoke_test()

    # Second: numerically verify the SwiGLU remapping is correct (this is the
    # subtle math: Qwen.w2 is the *gate* branch (silu-applied), not Qwen.w1).
    import torch.nn.functional as F

    H = 64
    I = 128
    B, S = 1, 8
    # Random Qwen1-MLP weights
    torch.manual_seed(42)
    qwen_w1 = torch.randn(I // 2, H)
    qwen_w2 = torch.randn(I // 2, H)
    qwen_c_proj = torch.randn(H, I // 2)

    x = torch.randn(B, S, H)

    # Qwen1 forward (per MoE-LLaVA qwen/modeling_qwen.py:570)
    a1 = x @ qwen_w1.T  # [B, S, I//2] (up branch, NOT activated)
    a2 = x @ qwen_w2.T  # [B, S, I//2] (gate branch, silu-applied)
    qwen_out = (a1 * F.silu(a2)) @ qwen_c_proj.T  # [B, S, H]

    # Now do the Meta remap and run LLaMA-style forward
    meta_state = qwen_mlp_to_meta(
        {
            "w1.weight": qwen_w1,
            "w2.weight": qwen_w2,
            "c_proj.weight": qwen_c_proj,
        }
    )
    # LLaMA tt-metal MLP convention (from mlp.py docstring):
    #   w1 -> gate_proj (with SiLU fused)
    #   w2 -> down_proj
    #   w3 -> up_proj
    # LLaMA forward: w2(silu(w1(x)) * w3(x))
    meta_w1 = meta_state["w1.weight"]  # gate (from Qwen.w2)
    meta_w3 = meta_state["w3.weight"]  # up (from Qwen.w1)
    meta_w2 = meta_state["w2.weight"]  # down (from Qwen.c_proj)
    gate = F.silu(x @ meta_w1.T)  # silu applied to Meta.w1 (= Qwen.w2)
    up = x @ meta_w3.T  # no activation on Meta.w3 (= Qwen.w1)
    meta_out = (gate * up) @ meta_w2.T

    max_diff = (qwen_out - meta_out).abs().max().item()
    assert max_diff < 1e-5, f"SwiGLU remap mismatch! max_diff={max_diff}"
    print(f"OK SwiGLU remap numerical equivalence: max_diff = {max_diff:.2e}")
