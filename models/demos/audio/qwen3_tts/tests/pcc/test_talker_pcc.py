# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""PCC test for the TTNN Qwen3-TTS talker against the CPU reference.

Block boundary: embeddings [1, T, hidden] -> hidden states [1, T, hidden], where hidden is
2048 at 1.7B and 1024 at 0.6B.

Input is a real prompt, built from real token ids through the model's own embedding and
projection path. That choice matters more than it looks. Random embeddings land far outside
the activation distribution the weights were trained on, and a 28-layer residual stack
amplifies the difference: the same graph scores 0.936 on random input and 0.995 on this
one, with top-1 codec agreement going from 71% to 92%. Test with what the model will see.

Measured here, 26-token prompt, bf16 on Blackhole P150:

    per layer, each fed the reference's own fp32 input   0.9998 to 0.99999, mean 0.99996
    end to end                                           0.9949
    codec top-1 agreement                                24/26, both misses near-ties

The per-layer number is what says the implementation is right, because it removes
accumulated drift: a wiring error shows up as one bad layer, rounding shows up as nothing.
The end-to-end number carries 28 layers of bf16 rounding on top of that. For reference, the
CPU model in bf16 reaches only 0.956 against its own fp32 output on random input, so the
device is not the limiting factor.
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.audio.qwen3_tts import weights
from models.demos.audio.qwen3_tts.reference.qwen import transformers_compat  # noqa: F401  (registers rope "default")
from models.demos.audio.qwen3_tts.reference.qwen3_talker_ref import TalkerReference
from models.demos.audio.qwen3_tts.reference.qwen.talker import (
    Qwen3TTSTalkerConfig,
    Qwen3TTSTalkerRotaryEmbedding,
    apply_multimodal_rotary_pos_emb,
)
from models.demos.audio.qwen3_tts.tests.reference_helpers import codec_head, talker_prompt
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_talker import (
    MASK_FILL,
    TtTalker,
    causal_mask,
    preprocess_talker_parameters,
    rotary_tables,
)

# Per-layer with drift removed. Measured 0.9998 at worst.
LAYER_PCC = 0.999

# End to end through 28 layers of bf16. Measured 0.9949 and deterministic.
STACK_PCC = 0.99

# A diagnostic, not the verdict: a sub-ulp perturbation moves it over 22/26 to 24/26.
MIN_TOKEN_AGREEMENT = 0.80

# The verdict: distance between the two sampling distributions, 0.076 to 0.098 when right.
SAMPLER_TEMPERATURE = 0.9
MAX_MEAN_DISTRIBUTION_DISTANCE = 0.13
MAX_DISTRIBUTION_DISTANCE = 0.60


def mixed_position_ids(length):
    """Three different position sequences, one per MRoPE axis.

    Equal axes make the interleaved selection a no-op, which would let a wrong selection
    pass unnoticed. A text-only prompt does have equal axes, so the selection is pinned
    here rather than through the device tests.
    """
    return torch.stack([torch.arange(length), torch.arange(length) * 2 + 1, torch.arange(length) * 3 + 7]).reshape(
        3, 1, length
    )


@pytest.fixture(scope="module")
def talker_config():
    return weights.talker_config()


@pytest.fixture(scope="module")
def prompt():
    return talker_prompt()


@pytest.fixture(scope="module")
def reference_outputs(prompt):
    embeddings, positions = prompt
    reference = TalkerReference(dtype=torch.float32)
    return reference(embeddings, position_ids=positions, return_intermediates=True)


def _to_device(device, tensor):
    return ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


# ── host-side tables ────────────────────────────────────────────────────────


def test_rotary_tables_match_the_reference(talker_config):
    """The rotation is built on host, so it is pinned against upstream's own function."""
    cfg = dict(talker_config)
    cfg.setdefault("pad_token_id", None)
    config = Qwen3TTSTalkerConfig(**cfg)
    length = 24
    positions = mixed_position_ids(length)

    torch.manual_seed(1)
    query = torch.randn(1, cfg["num_attention_heads"], length, cfg["head_dim"])
    key = torch.randn(1, cfg["num_key_value_heads"], length, cfg["head_dim"])

    cos_ref, sin_ref = Qwen3TTSTalkerRotaryEmbedding(config)(query, positions)
    query_ref, key_ref = apply_multimodal_rotary_pos_emb(
        query,
        key,
        cos_ref,
        sin_ref,
        cfg["rope_scaling"]["mrope_section"],
        mrope_interleaved=cfg["rope_scaling"]["interleaved"],
    )

    cos, sin = rotary_tables(cfg, positions)

    def rotate_half(x):
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    assert torch.equal(query * cos + rotate_half(query) * sin, query_ref)
    assert torch.equal(key * cos + rotate_half(key) * sin, key_ref)


def test_rotary_selection_reads_all_three_axes(talker_config):
    """Guards the test above: with equal axes the selection cannot be got wrong."""
    length = 24
    equal_axes = torch.arange(length).reshape(1, 1, length).expand(3, 1, length).contiguous()

    mixed_cos, _ = rotary_tables(talker_config, mixed_position_ids(length))
    equal_cos, _ = rotary_tables(talker_config, equal_axes)
    assert not torch.equal(mixed_cos, equal_cos), "the three axes are not reaching the tables"


def test_causal_mask_blocks_the_future():
    mask = causal_mask(8)[0, 0]
    future = torch.ones_like(mask, dtype=torch.bool).triu(1)

    assert mask.shape == (8, 8)
    assert (mask[future] == MASK_FILL).all(), "positions must not attend forwards"
    assert (mask[~future] == 0).all(), "positions must attend to themselves and backwards"


# ── device ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_layers_match_the_reference_in_isolation(device, prompt, reference_outputs):
    """The correctness gate: each layer fed the reference's own fp32 input for that layer.

    Removing accumulated drift is what makes this measure the implementation rather than
    28 layers of rounding.
    """
    embeddings, positions = prompt
    _, gold = reference_outputs
    length = embeddings.shape[1]
    model = TtTalker(device, preprocess_talker_parameters(device))

    cos, sin, mask = model.host_inputs(length, positions)
    cos_tt, sin_tt, mask_tt = (_to_device(device, t) for t in (cos, sin, mask))

    failures = []
    for index, layer in enumerate(model.layers):
        source = embeddings if index == 0 else gold[f"layers.{index - 1}"]
        want = gold[f"layers.{index}"]
        got = ttnn.to_torch(layer(_to_device(device, source), cos_tt, sin_tt, mask_tt, length)).float()
        passed, message = comp_pcc(want, got.reshape(want.shape), pcc=LAYER_PCC)
        print(f"  [layers.{index:<2d}] {message}")
        if not passed:
            failures.append(f"layers.{index}: {message}")

    assert not failures, "layers below PCC gate: " + "; ".join(failures)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_full_stack_matches_the_reference(device, prompt, reference_outputs):
    """All 28 layers in one pass, on a real prompt."""
    embeddings, positions = prompt
    gold, _ = reference_outputs
    length = embeddings.shape[1]
    model = TtTalker(device, preprocess_talker_parameters(device))

    cos, sin, mask = model.host_inputs(length, positions)
    hidden = model(*(_to_device(device, t) for t in (embeddings, cos, sin, mask)))

    got = ttnn.to_torch(hidden).float().reshape(gold.shape)
    passed, message = comp_pcc(gold, got, pcc=STACK_PCC)
    print(f"full stack {tuple(got.shape)}  {message}")
    assert torch.isfinite(got).all()
    assert passed, f"talker below PCC {STACK_PCC}: {message}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_codec_tokens_agree_with_the_reference(device, prompt, reference_outputs):
    """What PCC is a proxy for: would the sampler draw from the same distribution?

    Judged on the distribution, not the argmax. On this prompt the reference's own top-1
    probability is under 0.1 at several positions, so a quarter-ulp perturbation scatters
    agreement over 22 to 24 of 26 and lands the device on its 8th choice. Agreement is still
    printed and floored, since a matching distribution with every pick different would be
    worth seeing.
    """
    embeddings, positions = prompt
    gold, _ = reference_outputs
    length = embeddings.shape[1]
    model = TtTalker(device, preprocess_talker_parameters(device))

    cos, sin, mask = model.host_inputs(length, positions)
    hidden = model(*(_to_device(device, t) for t in (embeddings, cos, sin, mask)))
    got = ttnn.to_torch(hidden).float().reshape(gold.shape)

    head = codec_head()
    reference_logits, device_logits = (gold @ head.T)[0], (got @ head.T)[0]
    reference_choice, device_choice = reference_logits.argmax(-1), device_logits.argmax(-1)

    agreement = (reference_choice == device_choice).float().mean().item()
    print(f"codec top-1 agreement {int(agreement * length)}/{length}")

    reference_probs = torch.softmax(reference_logits / SAMPLER_TEMPERATURE, dim=-1)
    device_probs = torch.softmax(device_logits / SAMPLER_TEMPERATURE, dim=-1)
    distance = 0.5 * (reference_probs - device_probs).abs().sum(-1)
    print(
        f"sampling distribution distance: mean {distance.mean():.4f}, worst {distance.max():.4f} "
        f"at position {int(distance.argmax())}"
    )

    for index in torch.nonzero(reference_choice != device_choice).flatten().tolist():
        rank = int((reference_logits[index] > reference_logits[index][device_choice[index]]).sum())
        print(
            f"  pos {index:2d}: reference rank {rank}, reference p(own pick) "
            f"{float(reference_probs[index].max()):.3f}, distance {float(distance[index]):.4f}"
        )

    assert distance.mean() < MAX_MEAN_DISTRIBUTION_DISTANCE, f"mean distribution distance {distance.mean():.4f}"
    assert distance.max() < MAX_DISTRIBUTION_DISTANCE, f"worst distribution distance {distance.max():.4f}"
    assert agreement >= MIN_TOKEN_AGREEMENT, f"codec top-1 agreement {agreement:.2f} below {MIN_TOKEN_AGREEMENT}"
