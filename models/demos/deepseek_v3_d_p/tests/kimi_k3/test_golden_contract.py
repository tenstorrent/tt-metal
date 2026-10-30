# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""What the Kimi-K3 golden traces mean. Host only — no TTNN, no device.

Every device PCC number for K3 is scored against `decoder_output_layer_i`, so what that stream
*is* has to be settled before any of them are believable. Under AttnRes it is not the plain
residual every other model in this package writes: `AttnResStream.seal()` sets `running_sum` to
`None`, and the next `accumulate` therefore **replaces** rather than adds. Layer 0 seals, so its
output carries no embedding term at all. Read the stream as a plain residual and every K3 layer
would look ~0.45 PCC wrong for a reason no device change could fix.

Four things are pinned here, in the order they compose, and each is separated from its most
plausible wrong reading by a control that must fail:

1. `decoder_input_layer_0` is `embed_tokens[token_ids]` — bit-identical, so the trace enters at
   the raw embedding and the checkpoint on this box is the one that produced the trace.
2. `decoder_output_layer_0 == kda_output + moe_output`, with **no** embedding term. This is the
   seal, and it needs no weights at all: one subtraction separates the AttnRes schedule from a
   plain residual stream.
3. `kda_input_layer_0 == input_layernorm(embed)`. Layer 0's pre-attention read is skipped
   (nothing is sealed yet), so the KDA module sees the norm of the live stream directly.
4. `moe_input_layer_0 == post_attention_layernorm(attn_res(kda_out, [embed], q_post))` — the
   AttnRes read itself, against `reference/kimi_k3/attn_res/attn_res.py`. The reference's existing
   tests score it against the vendored HuggingFace function and against itself; this is the only
   check of it against activations a real model actually produced. The control — the same norm
   with the read omitted — reaches 0.9865, which is exactly why an eyeballed "high PCC" on the
   full read would prove nothing and the bar here is 0.99999.

The 100k trace is a 5-layer `Kimi-K3-SLIM-5L-PARTIAL` checkpoint that is not on the box, so (3)
and (4) double as evidence that its weights are the full checkpoint's: the queries and norms come
from the full model and reproduce the SLIM trace to bf16.
"""

from pathlib import Path

import pytest
import torch
from loguru import logger

from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.reference.kimi_k3.attn_res.attn_res import attn_res, fold_query
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tests.kimi_k3.golden import (
    TRACE_1M,
    TRACE_100K,
    checkpoint_prefix,
    embedding_rows,
    load_checkpoint_tensors,
    resolve_checkpoint,
    resolve_trace,
)

# Every comparison here is host fp32 against a bf16 trace, so the only error is the trace's own
# storage rounding. A real disagreement in schedule or query lands two decimal places lower, not
# in the fifth: the "no read at all" control for (4) reaches 0.9865.
GOLDEN_PCC = 0.99999

# Enough rows to make the correlation meaningful while every read stays a slice. The streams are
# per-token and causal, so more rows would restate the same result more slowly.
ROWS = 512


def _skip_without(trace, checkpoint):
    if trace is None:
        pytest.skip("no Kimi-K3 golden trace on this host; set KIMI_K3_GOLDEN_TRACE")
    if checkpoint is None:
        pytest.skip("no Kimi-K3 checkpoint on this host; set KIMI_K3_HF_MODEL")


def _rms_norm(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """The decoder's RMSNorm, in fp32. `KimiRMSNorm` with `KimiK3Config.RMS_NORM_EPS`."""
    scale = torch.rsqrt(hidden.pow(2).mean(-1, keepdim=True) + KimiK3Config.RMS_NORM_EPS)
    return hidden * scale * weight


@pytest.fixture(scope="module")
def checkpoint() -> Path | None:
    return resolve_checkpoint()


@pytest.mark.parametrize("default", [TRACE_100K, TRACE_1M], ids=["trace_100k", "trace_1m"])
def test_decoder_input_is_the_token_embedding(default, checkpoint):
    """(1) The stack enters at `embed_tokens[token_ids]`, bit-identically.

    Bit-identity rather than PCC on purpose. This is the one comparison where both sides are the
    same tensor read two ways, so anything short of equality means the checkpoint on this box is
    not the one behind the trace — and then nothing further in this file is evidence of anything.
    """
    trace = resolve_trace(default)
    _skip_without(trace, checkpoint)

    token_ids = trace.token_ids(ROWS)
    expected = embedding_rows(checkpoint, token_ids)
    golden = trace.decoder_input(0, ROWS)

    logger.info(f"{trace.path.name}: model_id={trace.metadata.get('model_id')}")
    assert torch.equal(expected.to(torch.bfloat16), golden.to(torch.bfloat16)), (
        f"decoder_input_layer_0 is not embed_tokens[token_ids] for {trace.path.name}; "
        f"max abs diff {(expected.float() - golden.float()).abs().max().item():.6g}"
    )


def test_layer0_output_is_a_post_seal_running_sum():
    """(2) The seal resets the stream — no weights needed, one subtraction decides it.

    `attn_res_layer` at layer 0: the pre-attention read is skipped, then `seal()` moves the
    embedding into `block_residual` and clears `running_sum`, then the two module outputs
    accumulate into it. So the layer's output is `attn + mlp`. A plain residual stream would carry
    `embed + attn + mlp`, and the embedding dominates — it is the control below.
    """
    trace = resolve_trace(TRACE_100K)
    if trace is None:
        pytest.skip("no Kimi-K3 100k golden trace on this host; set KIMI_K3_GOLDEN_TRACE")

    embed = trace.decoder_input(0, ROWS)
    attn_out = trace.rows("kda", "kda_output_layer_0", 0, ROWS)
    mlp_out = trace.rows("moe_io", "moe_output_layer_0", 0, ROWS)
    golden = trace.decoder_output(0, 0, ROWS)

    passed, message = comp_pcc(golden, attn_out + mlp_out, GOLDEN_PCC)
    logger.info(f"post-seal running sum: {message}")
    assert passed, f"decoder_output_layer_0 != kda_output + moe_output: {message}"

    plain_passed, plain_message = comp_pcc(golden, embed + attn_out + mlp_out, 0.99)
    assert not plain_passed, f"a plain residual stream also fits, so this proves nothing: {plain_message}"


def test_layer0_attention_input_skips_the_read(checkpoint):
    """(3) Nothing is sealed when layer 0 opens, so its attention sees `input_layernorm(embed)`."""
    trace = resolve_trace(TRACE_100K)
    _skip_without(trace, checkpoint)

    prefix = checkpoint_prefix(checkpoint)
    weight = load_checkpoint_tensors(checkpoint, [f"{prefix}layers.0.input_layernorm.weight"])[
        f"{prefix}layers.0.input_layernorm.weight"
    ].float()

    embed = trace.decoder_input(0, ROWS)
    golden = trace.rows("kda", "kda_input_layer_0", 0, ROWS)

    passed, message = comp_pcc(golden, _rms_norm(embed, weight), GOLDEN_PCC)
    logger.info(f"layer 0 pre-attention: {message}")
    assert passed, f"kda_input_layer_0 != input_layernorm(embed): {message}"


def test_attn_res_read_matches_the_reference(checkpoint):
    """(4) The AttnRes read, scored against real activations for the first time.

    Layer 0's pre-MLP site reads two candidates — the sealed embedding and the live `attn` sum —
    so this exercises the folded query, the epsilon, and the "keys are RMS-normalized, values are
    not" rule in one comparison. The trace captures the site *after*
    `post_attention_layernorm`, which is why the norm is applied here and why the raw read scores
    only ~0.78 against it.

    The control matters: with 96% of the softmax mass on the live stream, omitting the read
    entirely still reaches 0.9865. Only a bar in the fifth decimal distinguishes a correct read
    from no read at all.
    """
    trace = resolve_trace(TRACE_100K)
    _skip_without(trace, checkpoint)

    prefix = checkpoint_prefix(checkpoint)
    names = [
        f"{prefix}layers.0.{suffix}"
        for suffix in ("mlp_res_norm.weight", "mlp_res_proj.weight", "post_attention_layernorm.weight")
    ]
    weights = {k: v.float() for k, v in load_checkpoint_tensors(checkpoint, names).items()}
    query = fold_query(weights[names[0]], weights[names[1]])
    ffn_norm_weight = weights[names[2]]

    embed = trace.decoder_input(0, ROWS)
    attn_out = trace.rows("kda", "kda_output_layer_0", 0, ROWS)
    golden = trace.rows("moe_io", "moe_input_layer_0", 0, ROWS)

    read = attn_res(attn_out, embed.unsqueeze(1), query, eps=KimiK3Config.RMS_NORM_EPS)
    passed, message = comp_pcc(golden, _rms_norm(read, ffn_norm_weight), GOLDEN_PCC)
    logger.info(f"AttnRes read at layer 0 q_post: {message}")
    assert passed, f"moe_input_layer_0 != post_attention_layernorm(attn_res(...)): {message}"

    no_read_passed, no_read_message = comp_pcc(golden, _rms_norm(attn_out, ffn_norm_weight), GOLDEN_PCC)
    assert not no_read_passed, f"omitting the read also passes, so this proves nothing: {no_read_message}"
