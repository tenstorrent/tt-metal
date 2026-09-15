# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The reference model graded against HuggingFace (tt-blaze#4147).

``reference/model.py`` is the oracle every other prefill issue is measured against, so it needs its
own oracle, and that has to be HuggingFace rather than anything in this repo.

The main test loads a **real** ``LlamaForCausalLM`` with the true Llama-3.1-8B dimensions and a
reduced layer count and vocabulary, copies its state dict into the reference, and compares logits.
Going through ``load_hf_state_dict`` rather than assigning tensors directly means the key mapping
is exercised by the same test — a mapping typo shows up as a logits mismatch instead of as a
silently untrained parameter.

Two layers is enough to catch anything architectural: residual wiring, pre- vs post-norm, the RMS
norm float32 upcast, GQA head grouping, the rope frame, and the causal mask all produce
order-of-magnitude logit differences on a single layer. What two layers adds over one is that the
second layer's input is the first's output, so a wrong residual would compound rather than cancel.

The full 32-layer, real-checkpoint logits comparison is a separate opt-in test: it needs the 16 GB
checkpoint and is the literal acceptance criterion, but it is far too heavy to run by default.
Enable it with ``LLAMA31_REF_CHECKPOINT=1``.
"""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

from models.demos.llama_3p1_8b_d_p.reference.llama_3p1_8b_config import Llama31_8BConfig
from models.demos.llama_3p1_8b_d_p.reference.model import (
    DEFAULT_CHECKPOINT,
    Llama31Model,
    from_meta_frame,
    to_meta_frame,
)

# Small enough to build quickly, real everywhere it matters: hidden size, intermediate size, head
# count, KV head count and head_dim are all the production values.
TEST_LAYERS = 2
TEST_VOCAB = 512


def _hf_config(num_layers: int = TEST_LAYERS, vocab_size: int = TEST_VOCAB):
    from transformers import LlamaConfig

    return LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=Llama31_8BConfig.EMB_SIZE,
        intermediate_size=Llama31_8BConfig.INTERMEDIATE_SIZE,
        num_hidden_layers=num_layers,
        num_attention_heads=Llama31_8BConfig.NUM_ATTENTION_HEADS,
        num_key_value_heads=Llama31_8BConfig.NUM_KEY_VALUE_HEADS,
        head_dim=Llama31_8BConfig.HEAD_DIM,
        max_position_embeddings=Llama31_8BConfig.MAX_POSITION_EMBEDDINGS,
        rms_norm_eps=Llama31_8BConfig.RMS_NORM_EPS,
        rope_theta=Llama31_8BConfig.ROPE_THETA,
        rope_scaling={
            "rope_type": Llama31_8BConfig.ROPE_TYPE,
            "factor": Llama31_8BConfig.ROPE_SCALING_FACTOR,
            "low_freq_factor": Llama31_8BConfig.ROPE_LOW_FREQ_FACTOR,
            "high_freq_factor": Llama31_8BConfig.ROPE_HIGH_FREQ_FACTOR,
            "original_max_position_embeddings": Llama31_8BConfig.ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS,
        },
        attention_bias=Llama31_8BConfig.ATTENTION_BIAS,
        mlp_bias=False,
        tie_word_embeddings=False,
        attn_implementation="eager",
    )


@pytest.fixture(scope="module")
def hf_and_reference():
    """An HF model and the reference loaded from its state dict, sharing exactly one set of weights."""
    from transformers import LlamaForCausalLM

    torch.manual_seed(0)
    config = _hf_config()
    hf_model = LlamaForCausalLM(config).to(torch.float32).eval()

    reference = Llama31Model(num_layers=TEST_LAYERS, vocab_size=TEST_VOCAB).to(torch.float32)
    reference.load_hf_state_dict(hf_model.state_dict(), strict=True)
    reference.eval()
    return hf_model, reference


@pytest.mark.parametrize("seq_len", [1, 8, 128], ids=["s1", "s8", "s128"])
def test_reference_logits_match_hf(hf_and_reference, seq_len):
    """The reference reproduces HF logits.

    Tolerance is tight on purpose. Both sides are float32 doing the same operations in the same
    order, so the only expected difference is matmul reduction order; anything structural (a wrong
    rope frame, a transposed GQA grouping, a missing residual) moves logits by far more than this.
    """
    hf_model, reference = hf_and_reference
    torch.manual_seed(1)
    input_ids = torch.randint(0, TEST_VOCAB, (1, seq_len))

    with torch.no_grad():
        hf_logits = hf_model(input_ids).logits
        ref_logits, _ = reference(input_ids)

    max_abs = (hf_logits - ref_logits).abs().max().item()
    logger.info(f"seq_len={seq_len} max |HF - reference| logit diff: {max_abs:.3e}")
    torch.testing.assert_close(ref_logits, hf_logits, rtol=2e-4, atol=2e-4)


def test_reference_matches_hf_past_the_original_context(hf_and_reference):
    """Agreement holds past 8192, where llama3 rope scaling dominates.

    RoPE is the identity at position 0 and the llama3 frequency scaling only diverges from plain
    theta at long wavelengths, so a short-prompt comparison is nearly blind to it. This drives the
    same short sequence at a large ``start_pos`` instead of materialising 8192 tokens.
    """
    hf_model, reference = hf_and_reference
    seq_len, start_pos = 8, 10_000
    torch.manual_seed(2)
    input_ids = torch.randint(0, TEST_VOCAB, (1, seq_len))
    position_ids = torch.arange(start_pos, start_pos + seq_len)[None]

    with torch.no_grad():
        hf_logits = hf_model(input_ids, position_ids=position_ids).logits
        ref_logits, _ = reference(input_ids, start_pos=start_pos)

    max_abs = (hf_logits - ref_logits).abs().max().item()
    logger.info(f"start_pos={start_pos} max |HF - reference| logit diff: {max_abs:.3e}")
    torch.testing.assert_close(ref_logits, hf_logits, rtol=2e-4, atol=2e-4)


def test_chunked_prefill_equals_single_shot(hf_and_reference):
    """Prefilling in two chunks equals prefilling in one pass.

    This is the property the chunk loop in the prefill model depends on, and it is exactly what a
    chunk-local causal mask would break: masking on the within-chunk index rather than the absolute
    position hides the entire prefix from chunk 1, which still produces plausible-looking logits.
    Comparing against the single-shot run is what makes that visible.
    """
    _, reference = hf_and_reference
    torch.manual_seed(3)
    seq_len = 64
    input_ids = torch.randint(0, TEST_VOCAB, (1, seq_len))
    split = 32

    with torch.no_grad():
        full_logits, _ = reference(input_ids, start_pos=0)

        _, kvs = reference(input_ids[:, :split], start_pos=0, return_kv=True)
        chunk_logits, _ = reference(input_ids[:, split:], start_pos=split, past_kvs=kvs)

    torch.testing.assert_close(chunk_logits, full_logits[:, split:], rtol=2e-4, atol=2e-4)


def test_hf_key_map_covers_every_parameter():
    """Every reference parameter is reachable from an HF key, and nothing is invented.

    ``load_hf_state_dict`` is strict, but only about keys it looks for. A parameter the map forgets
    would stay at its randomly initialised value and the logits test would catch it only if that
    parameter mattered on a 2-layer model.
    """
    reference = Llama31Model(num_layers=TEST_LAYERS, vocab_size=TEST_VOCAB)
    mapping = reference.hf_key_map()
    own_names = set(dict(reference.named_parameters()))
    mapped = set(mapping.values())

    assert mapped == own_names, (
        f"key map and parameters disagree; unmapped parameters: {sorted(own_names - mapped)[:5]}, "
        f"mapped-but-nonexistent: {sorted(mapped - own_names)[:5]}"
    )
    assert len(mapping) == len(mapped), "two HF keys map onto the same parameter"


def test_meta_frame_conversion_roundtrips():
    """``to_meta_frame`` and ``from_meta_frame`` are inverses, and actually reorder.

    The reorder assertion is the point: both frames hold the same values, so a conversion that
    accidentally did nothing would still round-trip perfectly.
    """
    torch.manual_seed(0)
    x = torch.randn(2, 8, 16, Llama31_8BConfig.HEAD_DIM)

    torch.testing.assert_close(from_meta_frame(to_meta_frame(x)), x, rtol=0, atol=0)
    torch.testing.assert_close(to_meta_frame(from_meta_frame(x)), x, rtol=0, atol=0)
    assert not torch.equal(to_meta_frame(x), x), "conversion is a no-op — the frames would be indistinguishable"

    # Spell out the permutation on a tiny case so the intent is checkable by eye.
    small = torch.tensor([[0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0]])
    expected = torch.tensor([[0.0, 10.0, 1.0, 11.0, 2.0, 12.0, 3.0, 13.0]])
    torch.testing.assert_close(to_meta_frame(small), expected, rtol=0, atol=0)


def test_reference_k_in_meta_frame_matches_the_device_rope_frame():
    """Converting the reference's K lands in the frame ``tt/rope.py`` produces.

    The golden's rope frame must match what blaze decode writes, not what prefill happens to do.
    The reference computes in the HF frame (that is what makes it comparable to HF logits), so a
    golden K has to go through ``to_meta_frame`` first. This pins that the conversion produces the
    same rotation the device applies, rather than merely being self-consistent.
    """
    from models.demos.llama_3p1_8b_d_p.reference.model import apply_rope, build_hf_cos_sin
    from models.demos.llama_3p1_8b_d_p.tt.rope import build_llama3_cos_sin

    torch.manual_seed(0)
    head_dim = Llama31_8BConfig.HEAD_DIM
    positions = torch.arange(500, 500 + 16)
    k_hf_layout = torch.randn(1, 2, 16, head_dim)

    # Reference path: HF tables + rotate_half, then convert the result to the Meta frame.
    cos_hf, sin_hf = build_hf_cos_sin(positions, head_dim=head_dim)
    ref_meta = to_meta_frame(apply_rope(k_hf_layout, cos_hf, sin_hf))

    # Device path: Meta interleaved tables + the interleaved rotation, on the converted input.
    cos_meta, sin_meta = build_llama3_cos_sin(int(positions[-1]) + 1, head_dim=head_dim)
    cos_rows = cos_meta[0, 0, positions]
    sin_rows = sin_meta[0, 0, positions]

    k_meta_layout = to_meta_frame(k_hf_layout)
    x1, x2 = k_meta_layout[..., ::2], k_meta_layout[..., 1::2]
    rotated = torch.stack((-x2, x1), dim=-1).flatten(-2)
    device_meta = k_meta_layout * cos_rows + rotated * sin_rows

    torch.testing.assert_close(ref_meta, device_meta, rtol=1e-5, atol=1e-5)


def test_rmsnorm_upcasts_to_float32():
    """The norm computes in float32 even for a bf16 input.

    HF does this upcast, and skipping it shifts logits enough to look like a device bug. Checked by
    feeding a bf16 input whose squared mean is not representable well in bf16.
    """
    from models.demos.llama_3p1_8b_d_p.reference.model import Llama31RMSNorm

    norm = Llama31RMSNorm(dim=64)
    torch.manual_seed(0)
    x = (torch.randn(1, 4, 64) * 1e3).to(torch.bfloat16)

    out = norm(x)
    assert out.dtype == torch.bfloat16, "output dtype must follow the input"

    manual = x.float()
    manual = manual * torch.rsqrt(manual.pow(2).mean(-1, keepdim=True) + norm.eps)
    torch.testing.assert_close(out.float(), manual.to(torch.bfloat16).float(), rtol=0, atol=0)


@pytest.mark.skipif(
    os.environ.get("LLAMA31_REF_CHECKPOINT") != "1",
    reason="needs the 16GB Llama-3.1-8B checkpoint; set LLAMA31_REF_CHECKPOINT=1",
)
def test_real_checkpoint_logits_match_hf():
    """The acceptance criterion: the reference reproduces HF logits for Llama-3.1-8B on a prompt.

    Opt-in because it loads the full checkpoint twice (once into HF, once into the reference) and
    runs 32 layers on CPU.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from models.demos.llama_3p1_8b_d_p.reference.model import load_reference_model

    tokenizer = AutoTokenizer.from_pretrained(str(DEFAULT_CHECKPOINT))
    input_ids = tokenizer("The capital of France is", return_tensors="pt").input_ids

    reference = load_reference_model(dtype=torch.float32)
    with torch.no_grad():
        ref_logits, _ = reference(input_ids)
    del reference

    hf_model = AutoModelForCausalLM.from_pretrained(str(DEFAULT_CHECKPOINT), dtype=torch.float32).eval()
    with torch.no_grad():
        hf_logits = hf_model(input_ids).logits

    max_abs = (hf_logits - ref_logits).abs().max().item()
    logger.info(f"real checkpoint max |HF - reference| logit diff: {max_abs:.3e}")
    assert torch.equal(hf_logits.argmax(-1), ref_logits.argmax(-1)), "argmax token predictions differ from HF"
    torch.testing.assert_close(ref_logits, hf_logits, rtol=1e-3, atol=1e-3)
