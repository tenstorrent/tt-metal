# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks behind tests/test_model_full.py: the HF recorder that keeps only the decode rows
(reference/hf_harness.py record_layer ``first_row``), and the decode goldens of the cut depths.

A golden of the model cut to N layers holds, for the layers it shares with a deeper cut, the same
computation: a layer's output depends only on the layers below it. The goldens of 6, 12, 24, 48
and 78 layers share the 1022-token prompt, so two things can be compared between any two of them:

* HF's cache rows of the prompt, from its prefill pass over the prompt alone. Identical inputs:
  the rows must be bit-equal, and are. This pins the deep goldens, whose weights were read through
  a bounded page cache, to the shallow ones.
* The recorded first decode step (the last prompt token), from the pass over the prompt AND each
  depth's own continuation. The inputs of that row are identical too, and yet HF's BF16 result is
  not: with other tokens behind it in the sequence the row's attention output comes out one BF16
  step different on a few values (126 of 6144 at layer 5 between the 48- and the 78-layer golden,
  with bit-equal attention input, queries and picks), and that grows with depth: see what the
  test prints. It is the floor under every comparison with these goldens: a one-step
  perturbation at layer 5 is 1e-3 of PCC and more some forty layers later, in HF itself.
"""

from contextlib import ExitStack

import pytest
import torch

from models.experimental.hy4_preview.reference import hf_harness
from models.experimental.hy4_preview.reference.decoder import STAGES

DEPTHS = (6, 12, 24, 48, 78)
PROMPT_LEN = 1022


def test_recorder_keeps_the_last_rows():
    """record_layer(first_row) records what slicing a full record gives, on a small random HF model."""
    host = pytest.importorskip("models.experimental.hy4_preview.tests.test_decoder_reference_host")
    cfg = host._config()
    model = host._hf_model(cfg)
    ids = torch.randint(3, 64, (40,), generator=torch.Generator().manual_seed(5))
    first_row = 37

    def run(first):
        with torch.no_grad(), ExitStack() as stack:
            records = [stack.enter_context(hf_harness.record_layer(layer, first)) for layer in model.layers]
            model(input_ids=ids[None], use_cache=False)
        return records

    whole, tail = run(0), run(first_row)
    for layer, (a, b) in enumerate(zip(whole, tail)):
        assert "dsa_dropped" in a and "dsa_dropped" not in b
        for key in STAGES + ("q_resid", "topk_indices", "k_pass", "kv_a"):
            assert torch.equal(a[key][first_row:], b[key]), f"layer {layer} {key}"
        if "router_indices" in a:
            rows = ids.numel()
            assert torch.equal(a["router_indices"].reshape(rows, -1)[first_row:], b["router_indices"])
            assert torch.equal(a["router_weights"].reshape(rows, -1)[first_row:], b["router_weights"])


def _goldens():
    found = {depth: hf_harness.find_decode_golden(depth, PROMPT_LEN) for depth in DEPTHS}
    return {depth: golden for depth, golden in found.items() if golden is not None}


def _pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().double(), b.flatten().double()]))[0, 1].item()


# HF against HF on the first decode step of a shared layer (see the module docstring): a floor
# with room under what the goldens on disk give (the test prints the lows per pair and stage).
SHARED_LAYER_PCC = 0.99
SHARED_STAGES = ("attn_in", "attn_out", "mlp_in", "streams_out")


def test_goldens_of_the_cut_depths_agree_on_the_shared_layers():
    goldens = _goldens()
    if len(goldens) < 2:
        pytest.skip(f"fewer than two decode goldens of the {PROMPT_LEN}-token prompt under {hf_harness.GOLDEN_ROOT}")
    depths = sorted(goldens)
    deepest = goldens[depths[-1]]
    for depth in depths[:-1]:
        golden = goldens[depth]
        assert torch.equal(golden["input_ids"][:PROMPT_LEN], deepest["input_ids"][:PROMPT_LEN])
        assert golden["meta"]["positions"][0] == deepest["meta"]["positions"][0] == PROMPT_LEN - 1
        low, first = {key: (2.0, None) for key in SHARED_STAGES}, None
        for layer in range(depth):
            # the prefill pass saw the prompt and nothing else: the same rows, bit for bit
            for kind, rows in golden["cache"][layer].items():
                assert torch.equal(rows[:PROMPT_LEN], deepest["cache"][layer][kind][:PROMPT_LEN]), (depth, layer, kind)
            a, b = golden["layers"][layer], deepest["layers"][layer]
            for key in SHARED_STAGES:
                score = _pcc(a[key][0].float(), b[key][0].float())
                assert score >= SHARED_LAYER_PCC, f"L{depth} vs L{depths[-1]}: layer {layer} {key} pcc={score:.6f}"
                if score < low[key][0]:
                    low[key] = (score, layer)
                if first is None and not torch.equal(a[key][0], b[key][0]):
                    first = (layer, key)
        print(
            f"HF L{depth} vs HF L{depths[-1]}, first decode step: first tensor that is not bit-equal {first}; lowest "
            "pcc over the shared layers: "
            + ", ".join(f"{key} {score:.6f} (layer {layer})" for key, (score, layer) in low.items())
        )


@pytest.mark.parametrize("depth", DEPTHS)
def test_golden_shape(depth):
    """What tests/test_model_full.py reads of a golden is there, for every layer."""
    golden = hf_harness.find_decode_golden(depth, PROMPT_LEN)
    if golden is None:
        pytest.skip(f"no {depth}-layer decode golden under {hf_harness.GOLDEN_ROOT}")
    meta = golden["meta"]
    steps, n = meta["steps"], meta["prompt_len"]
    assert meta["num_layers"] == depth == len(golden["layers"]) == len(golden["cache"])
    assert golden["input_ids"].numel() == n + steps and golden["logits"].shape == (steps, 120832)
    assert torch.equal(golden["cached_logits_argmax"], golden["input_ids"][n:])
    for layer, rec in enumerate(golden["layers"]):
        assert rec["streams_out"].shape == (steps, 4, 6144), (layer, tuple(rec["streams_out"].shape))
        assert rec["topk_indices"].shape[0] == steps
        assert (layer == 0) == ("router_indices" not in rec)
        assert golden["cache"][layer]["latent"].shape == (n + steps - 1, 576)
        assert ("index_key" in golden["cache"][layer]) == (meta["indexer_types"][layer] == "full")
