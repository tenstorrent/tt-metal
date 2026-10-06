# SPDX-License-Identifier: Apache-2.0
"""Execute provider routing/residual Python bodies without its CUDA/vLLM imports.

Checks the transcription boundary, not TT device accuracy. Attention and expert
math are checked by the hardware runners against ReferenceDecoder.
"""

import ast
import json
from pathlib import Path

import torch

from .reference import ReferenceDecoder, config, load_weights, norm

ROOT = Path(__file__).resolve().parents[1]


def extract(name, class_name=None):
    source = (ROOT / "doc/functional_decoder/reference/aleph_alpha_inference__kolibri1.py.txt").read_text()
    tree = ast.parse(source)
    nodes = tree.body
    if class_name:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name).body
    node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    tree = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(tree)
    namespace = {"torch": torch}
    exec(compile(tree, "pinned_provider_body", "exec"), namespace)
    return namespace[name]


class FusedNorm:
    def __init__(self, weight):
        self.weight = weight

    def __call__(self, x, residual=None):
        if residual is None:
            return norm(x, self.weight)
        residual = x + residual
        return norm(residual, self.weight), residual


class ProviderLayer:
    def __init__(self, reference):
        self.ref = reference
        for name in ["input_layernorm", "post_attn_norm", "post_attention_layernorm", "post_ffn_norm"]:
            setattr(self, name, FusedNorm(reference.w[name + ".weight"]))
        self.mlp = reference.moe

    def self_attn(self, *, positions, hidden_states):
        return self.ref.attention(hidden_states, int(positions[0]))


def main():
    torch.set_num_threads(8)
    torch.manual_seed(542)
    router = extract("sigmoid_logit_add_routing")
    forward = extract("forward", "Kolibri1DecoderLayer")
    rows = []
    for layer in [0, 4]:
        weights = load_weights(layer)
        reference = ReferenceDecoder(weights, layer)
        x = (torch.randn(1, 33, 2560) * 0.48495227).bfloat16()
        logits = torch.nn.functional.linear(x.flatten(0, 1).float(), weights["mlp.gate.weight"].float())
        bias = weights["moe.router.expert_bias"].float()
        scores, ids = router(x, logits, 6, False, bias)
        own_ids = (logits + bias).topk(6, dim=-1).indices
        own_scores = logits.gather(1, own_ids).sigmoid()
        assert torch.equal(ids.long(), own_ids) and torch.equal(scores, own_scores)
        expected = reference(x)
        reference.reset()
        y, residual = forward(ProviderLayer(reference), torch.arange(33), x, None)
        assert torch.equal(y + residual, expected)
        # Fused input residual contract used when stacking provider layers.
        residual_input = torch.randn_like(x) * 0.01
        reference.reset()
        expected = reference(x + residual_input)
        reference.reset()
        y, residual = forward(ProviderLayer(reference), torch.arange(33), x, residual_input)
        assert torch.equal(y + residual, expected)
        rows.append(dict(layer=layer, router_exact=True, unfused_residual_exact=True, fused_residual_exact=True))
    original = json.loads((ROOT / "tests/config.json").read_text())
    assert config().max_position_embeddings == original["max_position_embeddings"] == 262144
    result = dict(provider_revision="049a6a7bd2405b27d6d280d256bd3d585191c7ae", rows=rows)
    (ROOT / "doc/functional_decoder/provider_reference_check.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
