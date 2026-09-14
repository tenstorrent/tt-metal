# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""PCC test for the TTNN Qwen3-TTS talker against the CPU reference.

Block boundary: embeddings [1, T, 2048] -> hidden states [1, T, 2048].

**Read this before judging the end-to-end number.** A 28-layer residual stack amplifies
small perturbations, and this model amplifies them hard. Measured on a 24-position prompt:

    per layer, each fed the reference's own fp32 input   0.9998 to 0.99999, mean 0.99996
    end to end, TTNN bf16 vs CPU fp32                    0.936
    end to end, CPU bf16 vs CPU fp32                     0.956

The middle row is the one that says whether this port is correct, and it says yes. The
third row is the control: the reference drifts nearly as far from itself in bf16 as the
device does, so the end-to-end loss is the number format rather than the implementation.
Raising device tensors to fp32 moves it to 0.9396, and fp32 weights change nothing at all,
because the compute is bf16-class whatever the tensors say.

So `test_layers_match_the_reference_in_isolation` is the correctness gate, and the
end-to-end test gates loosely against a break rather than against drift.
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
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_talker import (
    MASK_FILL,
    TtTalker,
    causal_mask,
    preprocess_talker_parameters,
    rotary_tables,
)

LENGTH = 24

# Per-layer, drift removed. Measured 0.9998 at worst, so this catches a wiring error while
# leaving room for a different card or a compiler change.
LAYER_PCC = 0.999

# End to end through 28 layers. Measured 0.936 and deterministic; the reference in bf16
# manages only 0.956, so there is no implementation headroom to recover here. Gated to
# catch something breaking, not to certify precision.
STACK_PCC = 0.92


def mixed_position_ids(length):
    """Three different position sequences, one per MRoPE axis.

    Equal axes make the interleaved selection a no-op, which would let a wrong selection
    pass unnoticed.
    """
    return torch.stack([torch.arange(length), torch.arange(length) * 2 + 1, torch.arange(length) * 3 + 7]).reshape(
        3, 1, length
    )


@pytest.fixture(scope="module")
def talker_config():
    return weights.talker_config()


@pytest.fixture(scope="module")
def prompt(talker_config):
    torch.manual_seed(0)
    embeddings = torch.randn(1, LENGTH, talker_config["hidden_size"]) * 0.02
    return embeddings, mixed_position_ids(LENGTH)


@pytest.fixture(scope="module")
def reference_outputs(prompt):
    embeddings, positions = prompt
    reference = TalkerReference(dtype=torch.float32)
    hidden, intermediates = reference(embeddings, position_ids=positions, return_intermediates=True)
    return hidden, intermediates


# ── host-side tables ────────────────────────────────────────────────────────


def test_rotary_tables_match_the_reference(talker_config, prompt):
    """The rotation is built on host, so it is pinned against upstream's own function."""
    _, positions = prompt
    cfg = dict(talker_config)
    cfg.setdefault("pad_token_id", None)
    config = Qwen3TTSTalkerConfig(**cfg)

    torch.manual_seed(1)
    query = torch.randn(1, cfg["num_attention_heads"], LENGTH, cfg["head_dim"])
    key = torch.randn(1, cfg["num_key_value_heads"], LENGTH, cfg["head_dim"])

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


def test_rotary_selection_reads_all_three_axes(talker_config, prompt):
    """Guards the test above: with equal axes the selection cannot be got wrong."""
    _, positions = prompt
    equal_axes = torch.arange(LENGTH).reshape(1, 1, LENGTH).expand(3, 1, LENGTH).contiguous()

    mixed_cos, _ = rotary_tables(talker_config, positions)
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
def test_layers_match_the_reference_in_isolation(device, talker_config, prompt, reference_outputs):
    """The correctness gate: each layer fed the reference's own fp32 input for that layer.

    Removing accumulated drift is what makes this measure the implementation. A wiring
    error appears as one bad layer; rounding appears as nothing.
    """
    embeddings, positions = prompt
    _, gold = reference_outputs
    model = TtTalker(device, preprocess_talker_parameters(device))

    cos, sin, mask = model.host_inputs(LENGTH, positions)
    to_device = lambda tensor: ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    cos_tt, sin_tt, mask_tt = to_device(cos), to_device(sin), to_device(mask)

    failures = []
    for index, layer in enumerate(model.layers):
        source = embeddings if index == 0 else gold[f"layers.{index - 1}"]
        want = gold[f"layers.{index}"]
        got = ttnn.to_torch(layer(to_device(source), cos_tt, sin_tt, mask_tt, LENGTH)).float().reshape(want.shape)
        passed, message = comp_pcc(want, got, pcc=LAYER_PCC)
        print(f"  [layers.{index:<2d}] {message}")
        if not passed:
            failures.append(f"layers.{index}: {message}")

    assert not failures, "layers below PCC gate: " + "; ".join(failures)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_full_stack_runs_end_to_end(device, talker_config, prompt, reference_outputs):
    """All 28 layers in one pass. See the module docstring on why the gate is loose."""
    embeddings, positions = prompt
    gold, _ = reference_outputs
    model = TtTalker(device, preprocess_talker_parameters(device))

    cos, sin, mask = model.host_inputs(LENGTH, positions)
    to_device = lambda tensor: ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    hidden = model(to_device(embeddings), to_device(cos), to_device(sin), to_device(mask))

    got = ttnn.to_torch(hidden).float().reshape(gold.shape)
    passed, message = comp_pcc(gold, got, pcc=STACK_PCC)
    print(f"full stack {tuple(got.shape)}  {message}")
    assert passed, f"talker below PCC {STACK_PCC}: {message}"
    assert torch.isfinite(got).all()
