# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0


def expand_fused_moe_experts(state_dict):
    """Split transformers 5.x fused Mixtral MoE expert params back to per-expert keys.

    transformers 5.x replaced the per-expert ``...block_sparse_moe.experts.{i}.w{1,2,3}.weight``
    tensors with 3D batched params under ``...mlp.experts.`` :
      - ``gate_up_proj`` : ``[num_experts, 2*intermediate, hidden]`` (rows ``:I`` = w1/gate, ``I:`` = w3/up)
      - ``down_proj``    : ``[num_experts, hidden, intermediate]`` (= w2)
    and renamed the router ``block_sparse_moe.gate`` -> ``mlp.gate``. The tt Mixtral model loads the
    per-expert / ``block_sparse_moe`` keys, so split them back here. Version- and model-tolerant:
    a no-op unless the fused ``mlp.experts.gate_up_proj`` keys are present (i.e. Mixtral on >=5.x).
    """
    fused_keys = [k for k in state_dict if k.endswith("mlp.experts.gate_up_proj")]
    if not fused_keys:
        return state_dict
    out = dict(state_dict)
    for gup_key in fused_keys:
        prefix = gup_key[: -len("mlp.experts.gate_up_proj")]  # e.g. "model.layers.0."
        gate_up = out.pop(gup_key)  # [E, 2I, H]
        down = out.pop(prefix + "mlp.experts.down_proj")  # [E, H, I]
        num_experts = gate_up.shape[0]
        inter = gate_up.shape[1] // 2
        for i in range(num_experts):
            base = f"{prefix}block_sparse_moe.experts.{i}."
            out[base + "w1.weight"] = gate_up[i, :inter, :].contiguous()  # gate -> w1, [I, H]
            out[base + "w3.weight"] = gate_up[i, inter:, :].contiguous()  # up   -> w3, [I, H]
            out[base + "w2.weight"] = down[i].contiguous()  # down -> w2, [H, I]
        # router gate: 5.x `...mlp.gate.weight` -> tt expects `...block_sparse_moe.gate.weight`
        gate_key = prefix + "mlp.gate.weight"
        if gate_key in out:
            out[prefix + "block_sparse_moe.gate.weight"] = out.pop(gate_key)
    return out
