# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The K3.1 golden is K3's golden format and K3's contract, and it is the trace it was converted from. Host only.

The golden is Miguel's chunked vLLM trace rewritten into the per-stream layout `golden.GoldenTrace` reads, by a
one-shot converter kept outside the repo (the golden's `converted_from` metadata names its source and version).
Pinned here, through both readers that consume a golden — K3's `GoldenTrace` (the *_matches_golden tests) and
the runner producer's loaders (the PCC gate): the converted K3.1 golden is the source trace row for row where it
copies a stream (all 24 DSA layers' KV and indexer keys, which the runner gate scores; the FFN output of layers
23 and 92, which no gate scores, pinned as the input of the rebuild), and enters at the K3.1 checkpoint's own
token embedding.

And `decoder_output_layer_i` means what it means in K3's golden (`test_golden_contract.py`): the AttnRes running
sum after layer i, restarted at each seal, which is what the stack's layer tap returns. The K3.1 tracer records
the layer's FFN output under that key; the converter rebuilds the running sum from it and `attn_output`. The
rebuild is checked against a stream the tracer wrote independently: the sealed block snapshots in
`attn_res/residual_layer_{12k}`, whose slot k is the running sum at the end of block k-1.
"""

import os
from pathlib import Path

import pytest
import torch

from models.common.utility_functions import comp_pcc
from models.demos.common.prefill.runners.runner_utils import load_trace_golden_span, load_trace_token_ids
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tests.kimi_k3.golden import TRACE_K31, GoldenTrace, embedding_rows, resolve_trace
from models.demos.deepseek_v3_d_p.tests.kimi_k3.test_golden_contract import GOLDEN_PCC, ROWS
from models.demos.deepseek_v3_d_p.tt.runners.adapters.kimi_k3_1 import KimiK31Adapter
from models.demos.deepseek_v3_d_p.tt.runners.prefill_kv_validation import (
    _load_golden_index_k,
    _load_golden_kv_post,
    _load_sharded_rows,
    index_golden_present,
    kvpe_golden_present,
)


@pytest.fixture(scope="module")
def k31_trace() -> GoldenTrace:
    """Fails, not skips, when the golden is missing: the runner gate reads the same path, so a skip here
    would turn a deleted golden into a green run."""
    trace = resolve_trace(TRACE_K31)
    assert trace is not None, f"no K3.1 golden at {os.getenv('KIMI_K3_GOLDEN_TRACE') or TRACE_K31}"
    return trace


def test_k31_golden_enters_at_the_k31_token_embedding(k31_trace):
    """Bit-identical, as K3's `test_decoder_input_is_the_token_embedding`: the checkpoint is the trace's."""
    checkpoint = Path(os.getenv(KimiK31Adapter.env_var) or KimiK31Adapter.hf_model_default)
    if not (checkpoint / "model.safetensors.index.json").is_file():
        pytest.skip(f"no K3.1 checkpoint at {checkpoint}; set {KimiK31Adapter.env_var}")
    rows = 512
    expected = embedding_rows(checkpoint, k31_trace.token_ids(rows)).to(torch.bfloat16)
    assert torch.equal(expected, k31_trace.decoder_input(0, rows).to(torch.bfloat16))


def test_k31_golden_serves_the_runner_gate(k31_trace):
    """What the runner gate scores is in the golden, as the source trace's prompt rows; so is the FFN output the
    running sum is rebuilt from.

    The producer compares the prompt span (`load_trace_golden_span`) of every DSA layer's kvpe and indexer key,
    read through the producer's own loaders. The producer only warns and skips the index cache when a trace carries
    no indexer key, so a missing one would leave the gate green: its presence is pinned here. `ffn_output` of
    layers 23 and 92 (the tracer's `decoder_output`) is pinned to the source; no gate scores it. The
    `decoder_output` the depth tests score is the rebuilt running sum, pinned by
    `test_k31_block_ends_match_the_sealed_snapshots`.
    """
    trace_dir = k31_trace.path
    prompt = k31_trace.metadata["n_prompt_tokens"]
    source = Path(k31_trace.metadata["converted_from"]["path"])
    assert load_trace_golden_span(trace_dir) == (0, prompt) and len(load_trace_token_ids(trace_dir)) == prompt

    for layer in KimiK3Config.mla_layer_ids():
        assert kvpe_golden_present(trace_dir, layer), layer
        golden = _load_golden_kv_post(trace_dir, layer, prompt)
        key = f"kv_post_transform_layer_{layer}"
        assert torch.equal(golden, _load_sharded_rows(source / "kv_cache" / f"layer_{layer}", key, 0, prompt)), layer
        assert index_golden_present(trace_dir, layer), layer
        golden = _load_golden_index_k(trace_dir, layer, prompt)
        key = f"indexer_k_layer_{layer}"
        assert torch.equal(golden, _load_sharded_rows(source / "indexer" / key, key, 0, prompt)), layer

    for layer in (23, KimiK3Config.NUM_LAYERS - 1):
        golden = k31_trace.rows("ffn_output", f"ffn_output_layer_{layer}", 0, prompt)
        key = f"decoder_output_layer_{layer}"  # the tracer's name for the FFN output
        assert torch.equal(golden, _load_sharded_rows(source / "decoder_io" / key, key, 0, prompt)), layer


def test_k31_layer0_output_is_a_post_seal_running_sum(k31_trace):
    """K3's contract (2): `decoder_output_layer_0 == attn + ffn`, with no embedding term (layer 0 seals). The
    control, a plain residual carrying the embedding, must fail."""
    golden = k31_trace.decoder_output(0, 0, ROWS)
    attn = k31_trace.rows("attn_output", "attn_output_layer_0", 0, ROWS)
    ffn = k31_trace.rows("ffn_output", "ffn_output_layer_0", 0, ROWS)
    passed, message = comp_pcc(golden, attn + ffn, GOLDEN_PCC)
    assert passed, f"decoder_output_layer_0 != attn_output + ffn_output: {message}"
    plain, plain_message = comp_pcc(golden, k31_trace.decoder_input(0, ROWS) + attn + ffn, 0.99)
    assert not plain, f"decoder_output_layer_0 also matches a plain residual with the embedding: {plain_message}"


def test_k31_block_ends_match_the_sealed_snapshots(k31_trace):
    """The rebuilt running sum at the end of each block is the snapshot the tracer sealed for it, written
    independently: slot k of `residual_layer_{12k}` (slot 0 is the embedding). Every block that has a successor.
    Bit-identical, a stronger form of K3's GOLDEN_PCC: both sides are vLLM's bf16 stream, so equality is
    reachable, and an fp32 rebuild misses it (block 6 reaches only 0.999988)."""
    block = KimiK3Config.ATTN_RES_BLOCK_SIZE
    for k in range(1, -(-KimiK3Config.NUM_LAYERS // block)):
        sealed = k31_trace.rows("attn_res", f"residual_layer_{k * block}", 0, ROWS)[:, k]
        rebuilt = k31_trace.decoder_output(k * block - 1, 0, ROWS)
        assert torch.equal(sealed, rebuilt), (
            f"decoder_output_layer_{k * block - 1} != sealed snapshot {k} of residual_layer_{k * block}: "
            f"{comp_pcc(sealed, rebuilt, GOLDEN_PCC)[1]}"
        )
        # Control: the tracer's own per-layer stream, the FFN output, is not the running sum.
        ffn_pcc = comp_pcc(sealed, k31_trace.rows("ffn_output", f"ffn_output_layer_{k * block - 1}", 0, ROWS))[1]
        assert ffn_pcc < 0.99, f"block {k}: the FFN output alone matches the sealed snapshot ({ffn_pcc})"
