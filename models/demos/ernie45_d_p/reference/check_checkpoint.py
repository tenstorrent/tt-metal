# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Gate P1.1: every weight the reference needs exists with the expected shape."""

import os
import sys

from safetensors import safe_open

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
from models.demos.ernie45_d_p.bringup import metrics  # noqa: E402
from models.demos.ernie45_d_p.reference.ernie_ref import ErnieConfig, WeightLoader, resolve_model_path  # noqa: E402

TASK = os.environ.get("ERNIE_BRINGUP_TASK", "P1.1")


def expected(cfg: ErnieConfig) -> dict[str, tuple]:
    H, D = cfg.hidden_size, cfg.head_dim
    exp = {
        "model.embed_tokens.weight": (cfg.vocab_size, H),
        "model.norm.weight": (H,),
    }
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}."
        exp |= {
            p + "input_layernorm.weight": (H,),
            p + "post_attention_layernorm.weight": (H,),
            p + "self_attn.q_proj.weight": (cfg.num_attention_heads * D, H),
            p + "self_attn.k_proj.weight": (cfg.num_key_value_heads * D, H),
            p + "self_attn.v_proj.weight": (cfg.num_key_value_heads * D, H),
            p + "self_attn.o_proj.weight": (H, cfg.num_attention_heads * D),
        }
        if not cfg.is_moe_layer(i):
            I = cfg.intermediate_size
            exp |= {
                p + "mlp.gate_proj.weight": (I, H),
                p + "mlp.up_proj.weight": (I, H),
                p + "mlp.down_proj.weight": (H, I),
            }
            continue
        Is = cfg.moe_intermediate_size * cfg.moe_num_shared_experts
        Ie = cfg.moe_intermediate_size
        exp |= {
            p + "mlp.gate.weight": (cfg.moe_num_experts, H),
            p + "mlp.shared_experts.gate_proj.weight": (Is, H),
            p + "mlp.shared_experts.up_proj.weight": (Is, H),
            p + "mlp.shared_experts.down_proj.weight": (H, Is),
        }
        for e in range(cfg.moe_num_experts):
            q = f"{p}mlp.experts.{e}."
            exp |= {q + "gate_proj.weight": (Ie, H), q + "up_proj.weight": (Ie, H), q + "down_proj.weight": (H, Ie)}
    return exp


def main():
    path = resolve_model_path()
    cfg = ErnieConfig.from_json(os.path.join(path, "config.json"))
    loader = WeightLoader(path)
    exp = expected(cfg)
    missing, bad = [], []
    for name, shape in exp.items():
        if not loader.has(name):
            missing.append(name)
            continue
        with safe_open(os.path.join(path, loader.weight_map[name]), framework="pt") as f:
            got = tuple(f.get_slice(name).get_shape())
        if got != shape:
            bad.append((name, got, shape))
    ebias = [n for n in loader.weight_map if "e_score_correction_bias" in n]
    extra = sorted(set(loader.weight_map) - set(exp) - set(ebias))
    print(f"checked {len(exp)} weights: missing={len(missing)} bad_shape={len(bad)} e_bias={len(ebias)}")
    print("unused checkpoint tensors (first 10):", extra[:10], f"... total {len(extra)}")
    for m in missing[:10]:
        print("MISSING", m)
    for b in bad[:10]:
        print("BAD", b)
    metrics.record(TASK, "num_weights", len(exp))
    metrics.record(TASK, "missing_weights", len(missing) + len(bad) + (cfg.num_hidden_layers - 1 - len(ebias)))
    metrics.record(TASK, "unused_tensors", len(extra))
    return 0 if not (missing or bad) else 1


if __name__ == "__main__":
    sys.exit(main())
