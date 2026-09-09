# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""M1 — the reference against the REAL HF checkpoint loaded on CPU. Ground truth for the whole model.

Host only, real weights. Structure follows `minimax_m3/tests/golden_hf_first_token.py`.

Everything else in this bring-up compares the reference either to upstream HF *classes* with random
weights, or to a second oracle written from the same architecture description. Both can agree while
being wrong about the actual checkpoint — a mis-mapped weight key, a transposed projection, a
permutation applied to the wrong tensor. This is the one test that runs the real 8B checkpoint end
to end and asserts the reference reproduces `LlamaForCausalLM`'s own logits.

Full depth (32 layers), full width, real weights, on CPU. Short ISL: the point is the WEIGHTS and the
key mapping, and a long prompt buys nothing that 128 tokens does not while costing minutes per run.

Skips (rather than fails) when the checkpoint is absent, so the host suite stays runnable off-box.
"""

from __future__ import annotations

import pytest
import torch

from models.demos.llama_3_1_8b_d_p.reference.config import LlamaConfigConstants
from models.demos.llama_3_1_8b_d_p.reference.model import REF_DTYPE, RefModel
from models.demos.llama_3_1_8b_d_p.tt.model_config import ModelArgs, resolve_weights_path

PROMPT = "The capital of France is"
ISL = 128
PCC_BAR = 0.99

_WEIGHTS = resolve_weights_path(required=False)
requires_checkpoint = pytest.mark.skipif(
    _WEIGHTS is None, reason="no Llama-3.1-8B-Instruct checkpoint; set HF_MODEL or stage /mnt/models"
)


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.detach().float().flatten(), b.detach().float().flatten()
    a, b = a - a.mean(), b - b.mean()
    denom = a.norm() * b.norm()
    if denom == 0:
        return 1.0 if torch.allclose(a, b) else 0.0
    return min(1.0, float((a @ b) / denom))


@pytest.fixture(scope="module")
def config():
    return LlamaConfigConstants.from_json()


@pytest.fixture(scope="module")
def state_dict_real():
    return ModelArgs.load_state_dict(_WEIGHTS)


@pytest.fixture(scope="module")
def input_ids():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(_WEIGHTS)
    ids = tok(PROMPT, return_tensors="pt").input_ids
    if ids.shape[1] < ISL:
        # Pad on the RIGHT with the prompt's own tokens so every position is a real token: a padded
        # tail would make the comparison depend on how each side masks padding rather than on the
        # weights, which is what this test is about.
        reps = -(-ISL // ids.shape[1])
        ids = ids.repeat(1, reps)
    return ids[:, :ISL]


@requires_checkpoint
def test_checkpoint_keys_map_onto_the_reference(config, state_dict_real):
    """Every reference parameter has a checkpoint tensor of the right shape, by HF name.

    Runs before the forward because a key mismatch is far cheaper to diagnose here than as a PCC
    number, and because it is what makes P1's loader mapping the identity.
    """
    ref_sd = RefModel(config).state_dict()
    missing, mismatched = [], []
    for name, tensor in ref_sd.items():
        ckpt_name = f"model.{name}" if not name.startswith("lm_head") else name
        if ckpt_name not in state_dict_real:
            missing.append(ckpt_name)
        elif tuple(state_dict_real[ckpt_name].shape) != tuple(tensor.shape):
            mismatched.append((ckpt_name, tuple(state_dict_real[ckpt_name].shape), tuple(tensor.shape)))
    assert not missing, f"reference parameters absent from the checkpoint: {missing[:8]}"
    assert not mismatched, f"shape mismatches: {mismatched[:8]}"


@requires_checkpoint
def test_reference_matches_hf_checkpoint(config, state_dict_real, input_ids):
    """Full-depth, real-weights logits: the vendored reference vs `LlamaForCausalLM` itself.

    Both sides load the SAME checkpoint tensors and run in fp16 on CPU. Disagreement here means the
    reference is wrong about the model, which would make every PCC number downstream meaningless.
    """
    from transformers import LlamaForCausalLM

    ours = RefModel(config).to(REF_DTYPE).eval()
    ours.load_state_dict(
        {name: state_dict_real[f"model.{name}" if not name.startswith("lm_head") else name].to(REF_DTYPE)
         for name in ours.state_dict()}
    )

    hf = LlamaForCausalLM(config.to_hf_config()).to(REF_DTYPE).eval()
    hf.load_state_dict({k: v.to(REF_DTYPE) for k, v in state_dict_real.items()}, strict=True)

    with torch.no_grad():
        ours_logits = ours(input_ids)
        hf_logits = hf(input_ids).logits

    pcc = _pcc(ours_logits, hf_logits)
    assert pcc >= PCC_BAR, f"reference vs HF checkpoint logits PCC {pcc:.6f} < {PCC_BAR}"

    # Top-1 agreement on the final position, as a second, harder-to-fake check: a high PCC on
    # 128k-wide logits can still hide a different argmax.
    assert int(ours_logits[0, -1].argmax()) == int(hf_logits[0, -1].argmax()), "next-token argmax disagrees with HF"


@requires_checkpoint
def test_reference_kv_matches_hf_cache(config, state_dict_real, input_ids):
    """Per-layer K/V from the reference vs HF's own KV cache — the artifact P1/P2 are graded on.

    The logits check above can pass while the cached K/V are laid out differently (pre- vs post-RoPE,
    or heads in a different order), and it is the K/V that every device test compares against.
    """
    from transformers import LlamaForCausalLM

    ours = RefModel(config).to(REF_DTYPE).eval()
    ours.load_state_dict(
        {name: state_dict_real[f"model.{name}" if not name.startswith("lm_head") else name].to(REF_DTYPE)
         for name in ours.state_dict()}
    )
    hf = LlamaForCausalLM(config.to_hf_config()).to(REF_DTYPE).eval()
    hf.load_state_dict({k: v.to(REF_DTYPE) for k, v in state_dict_real.items()}, strict=True)

    with torch.no_grad():
        _, our_kv = ours(input_ids, return_kv=True)
        hf_out = hf(input_ids, use_cache=True)

    # transformers 5.x: `past_key_values` is a `Cache` holding per-layer objects with .keys/.values.
    # It is no longer the 4.x tuple-of-tuples, and it is not subscriptable — indexing it raises
    # TypeError rather than returning the layer. Fall back to the 4.x shape so this test survives
    # either major.
    hf_cache = hf_out.past_key_values
    layers = getattr(hf_cache, "layers", None)
    if layers is not None:
        hf_pairs = [(layer.keys, layer.values) for layer in layers]
    else:
        hf_pairs = list(hf_cache)

    assert len(our_kv) == config.num_hidden_layers
    assert len(hf_pairs) == config.num_hidden_layers, f"HF cache has {len(hf_pairs)} layers"
    worst = 1.0
    for i, (k_ours, v_ours) in enumerate(our_kv):
        k_hf, v_hf = hf_pairs[i]
        pcc_k, pcc_v = _pcc(k_ours, k_hf), _pcc(v_ours, v_hf)
        worst = min(worst, pcc_k, pcc_v)
        assert pcc_k >= PCC_BAR, f"layer {i} K PCC {pcc_k:.6f}"
        assert pcc_v >= PCC_BAR, f"layer {i} V PCC {pcc_v:.6f}"
    print(f"reference vs HF per-layer KV, {config.num_hidden_layers} layers: worst PCC {worst:.6f}")
