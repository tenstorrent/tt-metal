# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Full-layer device tests: real GLM-5.3-Flash decoder layers, real checkpoint weights, real prompts.

Unlike ``test_functional_decoder.py`` (full width, synthetic weights, FFN shrunk to 16 experts) nothing is scaled
down here: each layer is built by ``FunctionalDecoder.from_state_dict`` from the checkpoint's fp8-dequantized weights
(``reference/hf_layer_reference.py::build_layer``) with the checkpoint's own config -- 288 routed experts top-8 of
I = 2048 stored as bfloat4_b, the shared expert, 64 x 128 KDA heads, 64-head MLA, dense I = 12288.

Inputs are the real residual streams the layer sees in the model: two chat prompts are embedded and run through the
HF layers 0 .. 4 on the host (fp32 compute on the checkpoint's bf16 values); layer L's device input is HF layer L-1's
output. Each device layer runs step by step in decode (both users batched) and, for MLA, as a prefill chunk followed
by decode. Tested layers: 0 (kda_dense), 3 (mla_moe, DSA in its dense-exact regime) and 4 (kda_moe).

Every run is graded against two HF references:

* **implementation** -- the HF layer with its routed experts rounded to their bfloat4_b values (what the device
  holds): isolates the device code, so the thresholds are tight.
* **deployment** -- the exact HF layer: what the 4-bit expert storage costs on real weights. On layer 3 that alone
  is ~0.996 median PCC (the HF MoE with bf4 experts scores the same against the exact one), so these thresholds are
  the storage budget, not a bug allowance.

Needs the checkpoint (~95 GB of shards; skipped when absent). KDA prefill is not tested here (blocked on Wormhole,
``test_kda.py::PREP_BLOCKER``).
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import pytest
import torch

import ttnn
from models.experimental.glm53_flash.tests.test_kda_block import assert_bf4_experts, bf4_round_experts, pcc
from models.experimental.glm53_flash.tt.config import Glm5NextTextConfig
from models.experimental.glm53_flash.tt.functional_decoder import HC_WIDTH, FunctionalDecoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
from checkpoint import default_snapshot_dir, load_embedding, load_layer_raw, shard_headers  # noqa: E402
from hf_layer_reference import build_layer, load_config  # noqa: E402

pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")

try:
    SNAPSHOT = default_snapshot_dir()
except FileNotFoundError:
    SNAPSHOT = None

pytestmark = [
    pytest.mark.skipif(SNAPSHOT is None, reason="GLM-5.3-Flash checkpoint not available"),
    pytest.mark.use_module_device({"l1_small_size": 24576}),
]

HC, HIDDEN = 4, 4096
KDA_DENSE, MLA_MOE, KDA_MOE = 0, 3, 4
TESTED = (KDA_DENSE, MLA_MOE, KDA_MOE)
LAST = max(TESTED)
SEQ = 24
MAX_SEQ = 128
PROMPTS = (
    "Explain why the sky is blue in two sentences, then give one everyday example of the same effect.",
    "Write a short Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
)
ROUTER_TIE_MARGIN = 2e-3

# (output PCC min, output PCC median, off-tie residual-update PCC min, its median), per reference.
THRESHOLDS = {
    "implementation": (0.999, 0.9999, 0.98, 0.995),
    "deployment": (0.99, 0.995, 0.97, 0.985),
}


# ---------------------------------------------------------------------------------------------
# Host: real prompts through the real HF layers 0 .. LAST
# ---------------------------------------------------------------------------------------------


def _hf_forward(layer, x: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    b, s = ids.shape
    with torch.no_grad():
        return layer(
            x,
            attention_mask=torch.ones(b, s, dtype=torch.bool),
            position_ids=torch.arange(s)[None].expand(b, -1),
            position_embeddings=None,
            input_ids=ids,
        )[0]


def _router_ties(layer, x: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """[B, S] tokens whose HF router is within ROUTER_TIE_MARGIN of a top-k boundary flip (bf16 inputs on device can
    pick the other expert there, which legitimately changes that token's whole FFN output)."""
    b, s = ids.shape
    if not hasattr(layer.mlp, "experts"):
        return torch.zeros(b, s, dtype=torch.bool)
    cap = {}
    hook = layer.mlp.register_forward_hook(lambda m, i, o: cap.setdefault("x", i[0].detach()))
    _hf_forward(layer, x, ids)
    hook.remove()
    h = cap["x"].reshape(b * s, -1).float()
    gate = layer.mlp.gate
    scores = (h @ gate.weight.float().t()).sigmoid() + gate.e_score_correction_bias.float()
    k = layer.mlp.config.num_experts_per_tok
    top = scores.topk(k + 1, dim=-1).values
    return (top[:, k - 1] - top[:, k] < ROUTER_TIE_MARGIN).reshape(b, s)


@pytest.fixture(scope="module")
def chain():
    """{layer: (input streams [B, S, hc, D], {reference: HF output}, router-tie mask, HF-named state dict)} for the
    TESTED layers. The chain itself propagates the exact HF outputs (the real model's streams)."""
    from models.experimental.glm53_flash.tt.tokenizer_chat import load_tokenizer

    headers = shard_headers(SNAPSHOT)
    _, text = load_config(SNAPSHOT)
    tok = load_tokenizer(SNAPSHOT)
    seqs = [tok.encode_chat([{"role": "user", "content": p}]) for p in PROMPTS]
    assert min(len(s) for s in seqs) >= SEQ
    ids = torch.tensor([s[:SEQ] for s in seqs])
    x = load_embedding(SNAPSHOT, headers)[ids].float()[:, :, None, :].expand(-1, -1, HC, -1).contiguous()

    out = {}
    for layer_idx in range(LAST + 1):
        raw = load_layer_raw(SNAPSHOT, layer_idx, headers)
        layer, sd = build_layer(SNAPSHOT, text, layer_idx, headers, raw=raw)
        layer = layer.float()  # checkpoint (bf16 / fp32) values, fp32 math: the oracle
        y = _hf_forward(layer, x, ids)
        if layer_idx in TESTED:
            refs = {"deployment": y}
            if hasattr(layer.mlp, "experts"):
                bf4_round_experts(layer)  # the fp32 copy only; sd keeps the checkpoint values the device rounds itself
                refs["implementation"] = _hf_forward(layer, x, ids)
            else:
                refs["implementation"] = y
            out[layer_idx] = (x, refs, _router_ties(layer, x, ids), sd)
        del layer, raw
        x = y
    gc.collect()
    return ids, out


@pytest.fixture(scope="module")
def tt_config():
    return Glm5NextTextConfig.from_json(str(SNAPSHOT))


@pytest.fixture(scope="module", params=TESTED, ids=["kda_dense_L0", "mla_moe_L3", "kda_moe_L4"])
def full_layer(request, _device_module_impl, chain, tt_config):
    """One real layer on device at a time (a 288-expert layer is ~4 GB of bfloat4_b)."""
    idx = request.param
    x, refs, ties, sd = chain[1][idx]
    dec = FunctionalDecoder.from_state_dict(sd, hf_config=tt_config, layer_idx=idx, mesh_device=_device_module_impl)
    yield idx, dec, x, refs, ties
    del dec
    gc.collect()


# ---------------------------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------------------------


def _decode(dec, cache, streams: torch.Tensor, start: int = 0, page_table=None) -> torch.Tensor:
    """Decode ``streams`` [B, n, hc, D] at positions start .. start + n - 1 for all B users -> [B, n, hc, D]."""
    b, n = streams.shape[:2]
    outs = []
    for t in range(n):
        rows = ttnn.from_torch(
            streams[:, t].reshape(1, 1, b, HC_WIDTH).float(),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=dec.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        pos = ttnn.from_torch(
            torch.full((b,), start + t, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=dec.device,
        )
        y = dec.decode_forward(rows, cache=cache, page_table=page_table, cur_pos=pos)
        outs.append(ttnn.to_torch(y).float().reshape(b, 1, HC, HIDDEN))
    return torch.cat(outs, dim=1)


def _grade(got, refs: dict, inp, ties, what):
    """Against each reference: per-token output PCC (every token) and residual-update PCC (tokens off a router
    near-tie, where bf16 rounding may legitimately pick another expert)."""
    b, s = got.shape[:2]
    keep = ~ties
    failures = []
    for name, ref in refs.items():
        ref = ref[:b, :s]
        out = torch.tensor([[pcc(got[i, t], ref[i, t]) for t in range(s)] for i in range(b)])
        upd = torch.tensor([[pcc(got[i, t] - inp[i, t], ref[i, t] - inp[i, t]) for t in range(s)] for i in range(b)])
        out_min, out_med, upd_min, upd_med = THRESHOLDS[name]
        print(
            f"\n{what} vs {name}: output PCC min {out.min():.5f} median {out.median():.6f}; update PCC (off-tie) "
            f"min {upd[keep].min():.4f} median {upd[keep].median():.5f}; router near-ties {int(ties.sum())}/{b * s}"
        )
        for value, bound, label in (
            (out.min(), out_min, "worst-token output"),
            (out.median(), out_med, "median output"),
            (upd[keep].min(), upd_min, "worst off-tie update"),
            (upd[keep].median(), upd_med, "median off-tie update"),
        ):
            if not value > bound:
                failures.append(f"{name}: {label} PCC {value:.5f} <= {bound}")
    assert not failures, f"{what}: " + "; ".join(failures)


# ---------------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------------


def test_full_layer_weights(full_layer, tt_config):
    """The layer is the checkpoint's, at full size, with the deployed expert storage."""
    idx, dec, *_ = full_layer
    assert dec.spec.layer_idx == idx and dec.spec.kind == {0: "dense_kda", 3: "sparse_dsa", 4: "sparse_kda"}[idx]
    if dec.spec.mlp == "moe":
        moe = dec.layer.mlp
        assert_bf4_experts(moe)
        assert tuple(moe.gate_up.shape) == (1, tt_config.n_routed_experts, HIDDEN, 2 * tt_config.moe_intermediate_size)
        assert tt_config.n_routed_experts == 288 and tt_config.num_experts_per_tok == 8


def test_full_layer_decode_matches_hf(full_layer):
    """Both prompts decoded together, one token per step from position 0, against the HF layer's full forward."""
    idx, dec, x, refs, ties = full_layer
    cache = dec.allocate_cache(max_batch=x.shape[0], max_seq_len=MAX_SEQ)
    pt = None if dec.is_kda else dec.make_page_table(cache)
    got = _decode(dec, cache, x, page_table=pt)
    _grade(got, refs, x, ties, f"layer {idx} decode")


def test_full_mla_layer_prefill_then_decode_matches_hf(full_layer):
    """MLA: prefill the prompt's first SEQ - 4 tokens as one chunk, then decode the last 4 from the latent cache."""
    idx, dec, x, refs, ties = full_layer
    if dec.is_kda:
        pytest.skip("KDA prefill is blocked on Wormhole (test_kda.py::PREP_BLOCKER)")
    n = SEQ - 4
    cache = dec.allocate_cache(max_batch=1, max_seq_len=MAX_SEQ)
    pt = dec.make_page_table(cache)
    pre = dec.prefill_forward(x[:1, :n], cache=cache, user_slot=0).reshape(1, n, HC, HIDDEN)
    tail = _decode(dec, cache, x[:1, n:], start=n, page_table=pt)
    got = torch.cat([pre, tail], dim=1)
    _grade(got, refs, x[:1], ties[:1], f"layer {idx} prefill {n} + decode 4")
