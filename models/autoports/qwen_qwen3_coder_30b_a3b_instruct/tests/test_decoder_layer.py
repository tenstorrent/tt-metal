# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The composed decoder layer against the HuggingFace reference layer.

This is the stage-01 deliverable: norm -> attention -> residual -> norm -> MoE
-> residual, end to end, at PCC >= 0.995.

The submodules already pass on their own (attention 0.9994+, MoE 0.9981+), so
a shortfall here is a *composition* error -- a residual added in the wrong
place, the router fed the un-normed tensor, the two norms swapped -- rather
than accumulated precision. ``test_residual_path_is_present`` exists to
separate those two explanations: it checks the layer's output actually depends
on the residual stream, which is the composition mistake most likely to still
score respectably on PCC.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc

from ..tt.functional_decoder import (
    DecoderLayerConfig,
    build_expert_sparsity,
    build_rope_cache,
    decoder_layer_prefill,
    upload_layer_weights,
)
from ..tt.weight_mapping import convert_layer_weights
from .reference import build_reference_layer, layer_state_dict, rotary_embeddings

LAYER_IDX = 0
PCC_REQUIRED = 0.995  # the stage-01 functional-decoder bar


@pytest.fixture(scope="module")
def reference():
    return build_reference_layer(LAYER_IDX)


@pytest.fixture(scope="module")
def torch_weights(reference):
    _, hf_config = reference
    return convert_layer_weights(layer_state_dict(LAYER_IDX), hf_config)


def _hidden(hf_config, seq_len, seed=0):
    torch.manual_seed(seed)
    return torch.randn(1, seq_len, hf_config.hidden_size, dtype=torch.float32) * 0.02


def _causal_mask(seq_len):
    return torch.full((seq_len, seq_len), float("-inf")).triu(1).reshape(1, 1, seq_len, seq_len)


def _reference_layer(layer, hf_config, hidden):
    seq_len = hidden.shape[1]
    cos, sin = rotary_embeddings(hf_config, seq_len)
    with torch.no_grad():
        out = layer(
            hidden,
            position_embeddings=(cos, sin),
            attention_mask=_causal_mask(seq_len),
        )
    return out[0] if isinstance(out, tuple) else out


def _run_layer(mesh_device, hf_config, torch_weights, hidden):
    config = DecoderLayerConfig.from_hf(hf_config)
    weights = upload_layer_weights(torch_weights, mesh_device, config)
    cos_cache, sin_cache = build_rope_cache(hf_config, hidden.shape[1], mesh_device)
    sparsity = build_expert_sparsity(mesh_device, config.moe.num_experts)

    tt_in = ttnn.from_torch(
        hidden.unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_out = decoder_layer_prefill(tt_in, weights, config, cos_cache, sin_cache, sparsity)
    return ttnn.to_torch(tt_out).squeeze(0)


@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=True)
@pytest.mark.parametrize(
    "seq_len",
    [32, 128, 512, 33, 100, 257],
    ids=["s32", "s128", "s512", "s33", "s100", "s257"],
)
def test_decoder_layer_vs_reference(mesh_device, reference, torch_weights, seq_len):
    """Tile-aligned and deliberately non-aligned sequence lengths.

    33, 100 and 257 exercise the zero-padding in ``moe_prefill``: one row past a
    tile, a mid-tile length, and one past a large power of two. Real prompts are
    almost never a multiple of 32, and padding bugs typically corrupt only the
    tail tokens -- which a sequence-wide PCC can absorb, so these run as
    separate cases rather than being folded into the aligned ones.
    """
    layer, hf_config = reference
    hidden = _hidden(hf_config, seq_len)

    ref_out = _reference_layer(layer, hf_config, hidden)
    tt_out = _run_layer(mesh_device, hf_config, torch_weights, hidden)

    passing, pcc_message = comp_pcc(ref_out, tt_out, PCC_REQUIRED)
    logger.info(comp_allclose(ref_out, tt_out))
    logger.info(f"decoder layer seq={seq_len}: {pcc_message}")
    assert passing, f"decoder layer (seq={seq_len}) below {PCC_REQUIRED}: {pcc_message}"


@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=True)
@pytest.mark.parametrize("seq_len", [33, 100], ids=["s33", "s100"])
def test_non_aligned_tail_tokens(mesh_device, reference, torch_weights, seq_len):
    """Tokens near the pad boundary must be no worse than the rest of the sequence.

    Stated *relatively*, on purpose. An absolute per-token bar cannot tell
    "zero-padding corrupted the tail" apart from "this token was always noisy":
    a couple of tokens sit near 0.9946 regardless of length because their router
    top-8 contains a near-tie, and they stay low at seq_len=32 where no padding
    exists at all. Comparing the tail against the sequence's own distribution
    isolates the padding question, which is the only thing this test is for.
    """
    layer, hf_config = reference
    hidden = _hidden(hf_config, seq_len)

    ref_out = _reference_layer(layer, hf_config, hidden)
    tt_out = _run_layer(mesh_device, hf_config, torch_weights, hidden)

    def token_pcc(pos):
        pair = torch.stack([ref_out[:, pos, :].flatten().float(), tt_out[:, pos, :].flatten().float()])
        return float(torch.corrcoef(pair)[0, 1])

    per_token = [token_pcc(p) for p in range(seq_len)]
    tail = per_token[-3:]
    body_worst = min(per_token[:-3])

    logger.info(
        f"seq={seq_len}: tail={[round(v, 5) for v in tail]} "
        f"body_worst={body_worst:.5f} median={sorted(per_token)[len(per_token) // 2]:.5f}"
    )

    assert min(tail) >= 0.99, f"tail tokens of seq_len {seq_len} are outright wrong: {tail}"
    # Padding, if broken, would make the tail distinctly worse than the body.
    assert min(tail) >= body_worst - 1e-3, (
        f"tail tokens ({min(tail):.5f}) are worse than the worst body token "
        f"({body_worst:.5f}) at seq_len {seq_len} -- suspect the moe_prefill zero-padding"
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=True)
def test_residual_path_is_present(mesh_device, reference, torch_weights):
    """The output must track the input, and must not merely echo it.

    A dropped residual still produces sane-looking activations; a layer that
    returns its input unchanged does too. Both are composition bugs that PCC
    against the reference would report as a single vague number.
    """
    layer, hf_config = reference
    hidden = _hidden(hf_config, 32)
    out = _run_layer(mesh_device, hf_config, torch_weights, hidden).float()
    flat_in = hidden.squeeze(0).float()

    assert torch.isfinite(out).all(), "layer produced non-finite values"
    assert not torch.allclose(out, flat_in, atol=1e-3), "output equals input -- layer body is a no-op"

    # With the residual intact the output stays correlated with the input;
    # without it the sublayer outputs alone would decorrelate.
    corr = torch.corrcoef(torch.stack([out.flatten(), flat_in.flatten()]))[0, 1]
    logger.info(f"corr(output, input) = {corr:.4f}")
    assert corr > 0.5, f"output barely tracks input (corr={corr:.4f}) -- residual likely dropped"
