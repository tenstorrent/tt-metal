# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""CPU reference for the DeepSeek-V4.1 MoE op test (``test_moe_v41``); torch only, no ttnn.

The MoE input is the §6 oracle's ``ffn_in`` of V4.1 layer 2 for one 5120-token chunk (random prompt),
synthetic (seeded) or checkpoint weights. The observable parts of ``MoE.forward`` (gate logits, routing,
routed sum, shared expert, output) are recomputed with the vendored modules and disk-cached next to the
oracle result; the CPU run takes tens of minutes, so warm the cache outside the device lock with
``python -m models.demos.deepseek_v3_d_p.tests.v41.moe_reference {synthetic,real}``.
"""

import hashlib
import sys

import pytest
import torch

from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import model as v41
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import oracle as O
from models.demos.deepseek_v3_d_p.tt.v41 import weights as W

LAYER = 2
SEQ = 5120  # one prefill chunk: 2560 tokens/chip on 2x4, 1280 on 4x2
REFERENCE_VERSION = 1  # bump when reference_parts changes what it records


def spec(source: str) -> O.OracleSpec:
    if source == "real" and not (O.HF_SNAPSHOT / W.INDEX_FILE).is_file():
        pytest.skip("V4.1 checkpoint shards not downloaded")
    return O.real_spec((LAYER,), SEQ, checkpoint=O.HF_SNAPSHOT if source == "real" else None)


@torch.no_grad()
def reference_parts(moe: v41.MoE, x: torch.Tensor) -> dict:
    """``MoE.forward`` (text tokens) split into its observable parts: same ops, same order."""
    logits = v41.linear(x.float(), moe.gate.weight.float())
    weights, indices = moe.gate(x)
    routed = torch.zeros_like(x, dtype=torch.float32)
    for e in range(moe.n_routed_experts):
        token, slot = torch.where(indices == e)
        if len(token):
            routed[token] += moe.experts[e](x[token], weights[token, slot, None])
    shared = moe.shared_experts(x)
    final = (routed + shared).type_as(x)
    return {
        "logits": logits,
        "weights": weights,
        "indices": indices,
        "routed": routed,
        "shared": shared,
        "final": final,
    }


def reference(source: str) -> tuple[dict, v41.Transformer | None]:
    """MoE input ``x`` and reference parts for ``source`` (cached), plus the reference model if it had to be
    built (synthetic device weights are read from it)."""
    s = spec(source)
    tokens = O.random_tokens(s)
    key = hashlib.sha256(f"{O.cache_path(s, tokens).name}:{REFERENCE_VERSION}".encode()).hexdigest()[:20]
    path = O.CACHE_DIR / f"moe-parts-{key}.pt"
    if path.is_file():
        return torch.load(path), None
    model = O.build_reference(s)
    result = O.oracle(s, tokens, model)
    x = result["blocks"][LAYER]["ffn_in"]
    with v41.set_dtype(torch.bfloat16):
        parts = reference_parts(model.layers[0].ffn, x)
    assert torch.equal(parts["final"], result["blocks"][LAYER]["ffn_out"]), "MoE split differs from MoE.forward"
    parts["x"] = x
    tmp = path.with_suffix(".tmp")
    torch.save(parts, tmp)
    tmp.replace(path)
    return parts, model


def device_weights(source: str, model: v41.Transformer | None) -> dict:
    """``TtV41Moe`` weights: the checkpoint layer via the F0 loader, or the synthetic reference's, dequantized
    (exact) with the F0 functions."""
    if source == "real":
        return W.load_layer(W.resolve_checkpoint(), LAYER)
    if model is None:
        model = O.build_reference(spec(source))
    moe = model.layers[0].ffn

    def expert(e: v41.Expert, dequant) -> dict:
        deq = lambda lin: dequant(lin.weight.data, lin.scale.data)
        return {"gate_proj": deq(e.w1), "up_proj": deq(e.w3), "down_proj": deq(e.w2)}

    return {
        "gate_weights": {
            "weight": moe.gate.weight.data.to(torch.bfloat16),
            "e_score_correction_bias": moe.gate.bias.data.float(),
        },
        "routed_expert_weights": [expert(e, W.dequant_mxfp4) for e in moe.experts],
        "shared_expert_weights": expert(moe.shared_experts, W.dequant_fp8_block),
    }


if __name__ == "__main__":
    torch.set_num_threads(16)
    for src in sys.argv[1:]:
        parts, _ = reference(src)
        print(src, {k: tuple(v.shape) for k, v in parts.items()}, flush=True)
