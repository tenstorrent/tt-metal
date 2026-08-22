# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Exact token agreement -- the bring-up's first gate.

This comes before everything else.

**Top-k overlap is not agreement.** Two gates, both deterministic, neither
involving the sampler:

1. **Teacher-forced argmax match.** Feed the reference's own hidden states and
   the device's through the same head, and compare `argmax(logits)` at every
   position. No sampling, no drift, no compounding -- a position either agrees or
   it does not. Gate: **> 95 %**.
2. **Free-running greedy decode.** `top_k = 1` on both sides from the same prefix,
   compared as full sequences. Reported as exact-match prefix length and
   full-sequence match. This one *does* compound: one disagreement diverges
   everything after it, which is exactly what makes it worth measuring separately.

RAS sampling is stochastic and is never the accuracy gate -- it is reported for
audio quality only.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from models.demos.cosyvoice.tt.common import GOLDEN_DIR, as_torch, load_golden
from models.demos.cosyvoice.tt.weights import default_weights_path

LLM_WEIGHTS = default_weights_path().replace("hift_", "llm_")

needs_l1_small = pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
needs_weights = pytest.mark.skipif(not os.path.exists(LLM_WEIGHTS), reason="export llm weights first")
needs_golden = pytest.mark.skipif(
    not os.path.exists(os.path.join(GOLDEN_DIR, "llm.ar_forward_chunk.npz")), reason="generate goldens first"
)


def _weights():
    with np.load(LLM_WEIGHTS) as z:
        meta = json.loads(bytes(z["__meta__"]).decode())
        w = {k: torch.from_numpy(np.ascontiguousarray(z[k])).float() for k in z.files if k != "__meta__"}
    return w, meta


def _head(w, hidden):
    return F.linear(hidden, w["llm_decoder.weight"], w["llm_decoder.bias"])


def agreement(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Match rate and the length of the agreeing prefix."""
    same = a == b
    n = int(same.numel())
    prefix = int(same.cumprod(0).sum()) if n else 0
    return {"rate": float(same.float().mean()) if n else 1.0, "matched": int(same.sum()), "n": n, "prefix": prefix}


# --------------------------------------------------------------------------
# host tier -- the reference's own greedy stream, so the gate has a target
# --------------------------------------------------------------------------
def torch_greedy(w, meta, prefix_xs: torch.Tensor, n_steps: int) -> list[int]:
    """Greedy decode in plain torch from the flat weight export.

    This is the comparison target for gate 2. It has to exist separately from the
    captured goldens because the reference *sampled* -- its recorded token stream
    is one draw from RAS, not the greedy stream, so it cannot be compared against
    a greedy device run.
    """
    from models.demos.cosyvoice.tt.llm import reference as R

    ar = meta["ar_decoder"]
    speech_emb = w["speech_embedding.weight"]
    with torch.no_grad():
        ys, caches = R.ar_forward_chunk(w, prefix_xs, ar)
        out = []
        for _ in range(n_steps):
            token = int(_head(w, ys[:, -1]).argmax())
            out.append(token)
            xs = speech_emb[token].reshape(1, 1, -1)
            ys, caches = R.ar_forward_chunk(w, xs, ar, caches)
    return out


@needs_weights
@needs_golden
def test_teacher_forced_argmax_on_the_reference_hidden_states():
    """Sanity floor for gate 1: the head applied to the reference's own hidden
    states must agree with itself, and must produce tokens in range.

    Establishes that the 209 prefill positions are a usable comparison surface
    before the device is involved at all.
    """
    w, meta = _weights()
    ys = as_torch(load_golden("llm.ar_forward_chunk")["call0.out_ys"])
    with torch.no_grad():
        pred = _head(w, ys[0]).argmax(dim=-1)
    assert pred.shape[0] == ys.shape[1]
    assert int(pred.min()) >= 0 and int(pred.max()) <= meta["speech_token_size"]
    # a degenerate head would emit one token everywhere; this one must not
    assert len(set(pred.tolist())) > 5, "the reference predictions are suspiciously uniform"


@needs_weights
@needs_golden
def test_torch_greedy_is_deterministic():
    """Gate 2's target must be reproducible or the gate means nothing."""
    w, meta = _weights()
    prefix = as_torch(load_golden("llm.ar_forward_chunk")["call0.in_xs"])
    a = torch_greedy(w, meta, prefix, 6)
    b = torch_greedy(w, meta, prefix, 6)
    assert a == b, (a, b)
    assert all(0 <= t <= meta["speech_token_size"] for t in a)


# --------------------------------------------------------------------------
# device tier -- the gates themselves
# --------------------------------------------------------------------------
@needs_weights
@needs_golden
@needs_l1_small
def test_gate1_teacher_forced_argmax_match(device):
    """**§6 gate 1.** Argmax agreement at every one of the 209 prefill positions.

    Teacher-forced: both sides see the reference's inputs, so nothing compounds
    and each position is an independent verdict. That is what makes >95 % a
    meaningful bar -- under free-running decode a single early disagreement would
    drag the rate to near zero regardless of model quality.
    """
    import ttnn
    from models.demos.cosyvoice.tt.llm.decoder import TtARDecoder, right_aligned_bias
    from models.demos.cosyvoice.tt.weights import WeightBag

    w, _ = _weights()
    g = load_golden("llm.ar_forward_chunk")
    xs = as_torch(g["call0.in_xs"])
    want_ys = as_torch(g["call0.out_ys"])
    length = xs.shape[1]

    bag = WeightBag.load(LLM_WEIGHTS)
    dec = TtARDecoder(device, bag.sub("llm"), bag.meta["ar_decoder"])
    max_len = ((length + 127) // 128) * 128

    def dev(v):
        return ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    ys, caches = dec.forward_chunk_fixed(
        dev(xs),
        dec.empty_cache(max_len, length),
        max_len,
        valid=length,
        mask=dev(right_aligned_bias(max_len, length, length, causal=True)),
    )
    got_ys = ttnn.to_torch(ys).float()
    ttnn.deallocate(ys)
    TtARDecoder.free_caches(caches)

    with torch.no_grad():
        got = _head(w, got_ys[0]).argmax(dim=-1)
        want = _head(w, want_ys[0]).argmax(dim=-1)

    a = agreement(got, want)
    print(f"\n  gate 1, teacher-forced argmax over {a['n']} positions")
    print(f"    exact match {a['matched']}/{a['n']} = {100*a['rate']:.2f}%   (gate > 95%)")
    print(f"    longest agreeing prefix: {a['prefix']}")
    if a["matched"] < a["n"]:
        bad = (got != want).nonzero().flatten()[:8].tolist()
        print(f"    first disagreements at positions {bad}")
    assert a["rate"] > 0.95, a


@needs_weights
@needs_golden
@needs_l1_small
@pytest.mark.parametrize("n_steps", [24])
def test_gate1b_teacher_forced_argmax_through_the_kv_cache(device, n_steps):
    """**§6 gate 1, extended to the decode path.** Gate 1 above only exercises
    *prefill*; every token after the first goes through the cached path, which is
    where the fixed-width buffer, the right-alignment and the growing positional
    window all live. None of that is covered by a prefill-only sweep.

    Still teacher-forced -- the reference's own emitted tokens are fed one at a
    time, so each step is an independent verdict and nothing compounds. And unlike
    greedy decode the token stream is non-degenerate, because it is the stream the
    reference actually produced.
    """
    import ttnn
    from models.demos.cosyvoice.tt.llm import reference as R
    from models.demos.cosyvoice.tt.llm.decoder import TtARDecoder, right_aligned_bias
    from models.demos.cosyvoice.tt.weights import WeightBag

    w, meta = _weights()
    prefix = as_torch(load_golden("llm.ar_forward_chunk")["call0.in_xs"])
    length = prefix.shape[1]
    # tokens[:n_prompt] are the prompt; what follows is what the LLM emitted
    all_tokens = torch.from_numpy(load_golden("flow.input_embedding")["call0.in_tokens"]).long()
    n_prompt = as_torch(load_golden("flow.length_regulator")["call0.in_x1"]).shape[1]
    fed = all_tokens[0, n_prompt : n_prompt + n_steps].tolist()
    speech_emb = w["speech_embedding.weight"]

    # --- torch side, same teacher-forced stream
    with torch.no_grad():
        ys_t, caches_t = R.ar_forward_chunk(w, prefix, meta["ar_decoder"])
        want = []
        for tok in fed:
            want.append(int(_head(w, ys_t[:, -1]).argmax()))
            ys_t, caches_t = R.ar_forward_chunk(w, speech_emb[tok].reshape(1, 1, -1), meta["ar_decoder"], caches_t)

    # --- device side
    bag = WeightBag.load(LLM_WEIGHTS)
    dec = TtARDecoder(device, bag.sub("llm"), bag.meta["ar_decoder"])
    max_len = ((length + n_steps + 1 + 127) // 128) * 128

    def dev(v):
        return ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    ys, caches = dec.forward_chunk_fixed(
        dev(prefix),
        dec.empty_cache(max_len, length),
        max_len,
        valid=length,
        mask=dev(right_aligned_bias(max_len, length, length, causal=True)),
    )
    got = []
    for i, tok in enumerate(fed):
        hidden = ttnn.to_torch(ys).float()[:, -1]
        ttnn.deallocate(ys)
        with torch.no_grad():
            got.append(int(_head(w, hidden).argmax()))
        ys, caches = dec.forward_chunk_fixed(
            dev(speech_emb[tok].reshape(1, 1, -1)),
            caches,
            max_len,
            valid=length + 1 + i,
            mask=dev(right_aligned_bias(max_len, length + 1 + i, 1)),
        )
    ttnn.deallocate(ys)
    TtARDecoder.free_caches(caches)

    a = agreement(torch.tensor(got), torch.tensor(want))
    print(f"\n  gate 1b, teacher-forced argmax through the KV cache, {n_steps} decode steps")
    print(f"    fed (reference's own tokens) {fed[:8]} ...")
    print(f"    device {got[:8]} ...")
    print(f"    torch  {want[:8]} ...")
    print(f"    exact match {a['matched']}/{a['n']} = {100*a['rate']:.2f}%   (gate > 95%)")
    print(f"    distinct predictions: {len(set(got))} (a degenerate path would be 1)")
    assert len(set(want)) > 1, "the teacher-forced stream degenerated; this gate would be vacuous"
    assert a["rate"] > 0.95, a


@needs_weights
@needs_golden
@needs_l1_small
@pytest.mark.parametrize("n_steps", [16])
def test_gate2_free_running_greedy(device, n_steps):
    """**§6 gate 2.** `top_k=1` on both sides from the same prefix.

    Unlike gate 1 this compounds: the device's token at step k becomes its input
    at step k+1, so one disagreement diverges the rest. Reported as the exact
    match prefix length rather than a pass/fail rate, because that is the number
    that actually describes the divergence.

    The comparison target is `torch_greedy`, not the captured token stream -- the
    reference *sampled*, so its recorded tokens are one RAS draw and are not the
    greedy sequence.
    """
    import ttnn
    from models.demos.cosyvoice.tt.llm.decoder import TtARDecoder, right_aligned_bias
    from models.demos.cosyvoice.tt.weights import WeightBag

    w, meta = _weights()
    prefix = as_torch(load_golden("llm.ar_forward_chunk")["call0.in_xs"])
    length = prefix.shape[1]
    want = torch_greedy(w, meta, prefix, n_steps)

    bag = WeightBag.load(LLM_WEIGHTS)
    dec = TtARDecoder(device, bag.sub("llm"), bag.meta["ar_decoder"])
    speech_emb = w["speech_embedding.weight"]
    max_len = ((length + n_steps + 1 + 127) // 128) * 128

    def dev(v):
        return ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    ys, caches = dec.forward_chunk_fixed(
        dev(prefix),
        dec.empty_cache(max_len, length),
        max_len,
        valid=length,
        mask=dev(right_aligned_bias(max_len, length, length, causal=True)),
    )
    got = []
    for i in range(n_steps):
        hidden = ttnn.to_torch(ys).float()[:, -1]
        ttnn.deallocate(ys)
        with torch.no_grad():
            token = int(_head(w, hidden).argmax())
        got.append(token)
        ys, caches = dec.forward_chunk_fixed(
            dev(speech_emb[token].reshape(1, 1, -1)),
            caches,
            max_len,
            valid=length + 1 + i,
            mask=dev(right_aligned_bias(max_len, length + 1 + i, 1)),
        )
    ttnn.deallocate(ys)
    TtARDecoder.free_caches(caches)

    a = agreement(torch.tensor(got), torch.tensor(want))
    print(f"\n  gate 2, free-running greedy, {n_steps} steps")
    print(f"    device {got}")
    print(f"    torch  {want}")
    print(f"    exact match {a['matched']}/{a['n']} = {100*a['rate']:.2f}%")
    print(f"    exact-match prefix length: {a['prefix']}/{n_steps}")
    print(f"    full-sequence match: {got == want}")
    distinct = len(set(got))
    print(f"    distinct tokens emitted: {distinct}")
    if distinct == 1:
        print(
            "    NOTE: greedy degenerates to a constant token here, so a 100% match is\n"
            "    a weak result -- it compares two constant sequences. That degeneracy is\n"
            "    expected (it is the failure mode RAS exists to break, see tt/llm/sampling.py)\n"
            "    and it is why gate 1b exists: teacher forcing with the reference's own\n"
            "    non-degenerate token stream is what actually exercises the cached path."
        )
    # A compounding gate cannot sensibly demand 100%: one bfloat16 tie-break at a
    # near-degenerate logit pair diverges everything after it. The prefix length
    # is the number that describes the divergence, and it is what gets reported.
    assert a["prefix"] >= 1, f"diverged on the very first token: {a}"
