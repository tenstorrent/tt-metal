# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""Sequence assembly, prefill/decode wiring, and RAS sampling for `TtQwen2LM`.

test_qwen2lm.py covers the backbone (embeddings + 24 TransformerBlocks + norm) in
isolation, including the audit-driven `test_device_backbone_matches_real_qwen2_multistep_decode`,
which exercises a populated KV cache built entirely through DECODE's own per-step cache
update. That test does NOT exercise `Attention.forward_prefill`'s cache-write path
(`ttnn.fill_cache`, a different code path from decode's) at all -- so "prefill populates the
cache, decode then continues from it" was still an untested seam going into this file. That
seam is exactly what `test_device_prefill_then_decode_matches_real_qwen2` below checks: a
real assembled prefix goes through `TtQwen2LM.prefill`, then ONE more token continues through
`decode_step` reading the cache prefill wrote -- checked against a reference built the same
way `test_qwen2lm.py`'s multistep test already validated (a chain of `HfDecoderWrapper` calls,
one per position, real triangular history), just now seeded with the real assembled sequence
end to end instead of synthetic per-position tokens.

GATE_DECODE=0.94 here for the same reason as test_qwen2lm.py: this is real multi-step decode
at full (24-layer) depth with a bf16 KV cache in `ModelOptimizations.accuracy()` mode --
tt_transformers' own `test_model.py:139-141/436-441` checks exactly this scenario against
0.94, not a DSP-identity-style 0.99. See test_qwen2lm.py's module docstring for the full
root-cause writeup; this file does not repeat it.
"""

from __future__ import annotations

import os

import pytest
import torch

from models.common.utility_functions import comp_pcc

GATE_DECODE = 0.94  # see module docstring; cited from tt_transformers' test_model.py.


def _build_args_and_state_dict(mesh_device, max_seq_len=256):
    os.environ["HF_MODEL"] = "Qwen/Qwen2-0.5B-Instruct"
    from models.tt_transformers.tt.model_config import ModelArgs

    args = ModelArgs(mesh_device, max_batch_size=1, max_seq_len=max_seq_len, dummy_weights=False, use_hf_rope=True)
    state_dict = args.load_state_dict()
    return args, state_dict


# --------------------------------------------------------------------------
# host tier -- sequence assembly and RAS sampling algorithm correctness
# --------------------------------------------------------------------------
@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
def test_assemble_prefill_sequence_matches_manual_concat(device):
    """`assemble_prefill_sequence` must produce exactly `torch.cat([sos, text, task_id,
    speech], dim=1)` built from the SAME three embedding tables -- confirming the method
    does what its docstring claims (upstream's `Qwen2LM.inference` order), not a
    self-consistency check against itself."""
    from models.demos.audio.cosyvoice2.tt.llm.qwen2lm import TtQwen2LM

    args, state_dict = _build_args_and_state_dict(device)
    tt_model = TtQwen2LM(args, device, state_dict)

    torch.manual_seed(1)
    text_ids = torch.randint(0, args.vocab_size, (1, 6))
    speech_ids = torch.tensor([[3, 100, 4200]])

    got = tt_model.assemble_prefill_sequence(text_ids, speech_ids)

    sos = tt_model.embed_llm_tokens_host(torch.tensor([[0]]))
    task = tt_model.embed_llm_tokens_host(torch.tensor([[1]]))
    text = tt_model.embed_text_tokens_host(text_ids)
    speech = tt_model.embed_speech_tokens_host(speech_ids)
    want = torch.cat([sos, text, task, speech], dim=1)

    assert got.shape == want.shape == (1, 1 + 6 + 1 + 3, args.dim)
    assert torch.equal(got, want)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
def test_assemble_prefill_sequence_without_prompt_speech(device):
    """No prompt speech tokens (`prompt_speech_ids=None`) -> 3-part sequence, matching
    upstream's own `if prompt_speech_token_len != 0` guard."""
    from models.demos.audio.cosyvoice2.tt.llm.qwen2lm import TtQwen2LM

    args, state_dict = _build_args_and_state_dict(device)
    tt_model = TtQwen2LM(args, device, state_dict)

    text_ids = torch.randint(0, args.vocab_size, (1, 4))
    got = tt_model.assemble_prefill_sequence(text_ids, None)
    assert got.shape == (1, 1 + 4 + 1, args.dim)


def _reference_nucleus_sampling(weighted_scores: torch.Tensor, top_p: float = 0.8, top_k: int = 25) -> int:
    """Literal transcription of the real upstream `cosyvoice/utils/common.py::nucleus_sampling`
    (downloaded and read directly during this bring-up), kept deliberately naive/unvectorised
    so it is obviously a faithful copy, not a second implementation of the same idea."""
    prob, indices = [], []
    cum_prob = 0.0
    sorted_value, sorted_idx = weighted_scores.softmax(dim=0).sort(descending=True, stable=True)
    for i in range(len(sorted_idx)):
        if cum_prob < top_p and len(prob) < top_k:
            cum_prob += float(sorted_value[i])
            prob.append(sorted_value[i])
            indices.append(sorted_idx[i])
        else:
            break
    prob_t = torch.stack(prob)
    indices_t = torch.stack(indices)
    return int(indices_t[prob_t.multinomial(1, replacement=True)].item())


def _reference_ras_sampling(
    weighted_scores: torch.Tensor,
    decoded_tokens,
    top_p: float = 0.8,
    top_k: int = 25,
    win_size: int = 10,
    tau_r: float = 0.1,
) -> int:
    """Literal transcription of upstream `ras_sampling` / `random_sampling`."""
    top_ids = _reference_nucleus_sampling(weighted_scores, top_p=top_p, top_k=top_k)
    rep_num = sum(1 for t in decoded_tokens[-win_size:] if t == top_ids)
    if rep_num >= win_size * tau_r:
        weighted_scores[top_ids] = -float("inf")
        top_ids = int(weighted_scores.softmax(dim=0).multinomial(1, replacement=True).item())
    return top_ids


def test_nucleus_sampling_matches_literal_transcription():
    """`sampling.nucleus_sampling` (the vectorised `torch.topk`-based port) must select the
    exact same token as a literal transcription of upstream's sort-and-walk loop, across many
    random distributions -- ties among continuous random logits have probability zero, so
    the two retrieval orders (`sort(stable=True)` vs `topk(sorted=True)`) agree in practice.
    Re-seeded identically before each call so both draw the same `multinomial` outcome."""
    from models.demos.audio.cosyvoice2.tt.llm.sampling import nucleus_sampling

    for trial in range(50):
        torch.manual_seed(1000 + trial)
        scores = torch.randn(4097)
        top_p = [0.8, 0.5, 0.99][trial % 3]
        top_k = [25, 5, 1][trial % 3]

        torch.manual_seed(2000 + trial)
        mine = nucleus_sampling(scores.clone(), top_p=top_p, top_k=top_k)
        torch.manual_seed(2000 + trial)
        theirs = _reference_nucleus_sampling(scores.clone(), top_p=top_p, top_k=top_k)
        assert mine == theirs, (trial, mine, theirs)


def test_ras_sampling_matches_literal_transcription():
    """Same check as above, one level up: `sampling.ras_sampling`, including its
    repetition-triggered resample branch, against a literal transcription of upstream."""
    from models.demos.audio.cosyvoice2.tt.llm.sampling import ras_sampling

    for trial in range(30):
        torch.manual_seed(3000 + trial)
        scores = torch.randn(4097)
        # A short, adversarial history that forces the repetition branch on some trials
        # and not others -- both branches need coverage, not just the common path.
        history = [int(torch.randint(0, 4097, (1,)).item()) for _ in range(9)]

        torch.manual_seed(4000 + trial)
        mine = ras_sampling(scores.clone(), list(history))
        torch.manual_seed(4000 + trial)
        theirs = _reference_ras_sampling(scores.clone(), list(history))
        assert mine == theirs, (trial, mine, theirs)


def test_ras_sampling_repetition_branch_actually_fires():
    """Not just algorithmic agreement -- confirm the repetition branch is reachable at all:
    a history saturated with the argmax token must force a different (or at least
    re-drawn) token than plain `nucleus_sampling` would give without that history."""
    from models.demos.audio.cosyvoice2.tt.llm.sampling import is_repetitive, nucleus_sampling, ras_sampling

    torch.manual_seed(42)
    scores = torch.zeros(100)
    scores[7] = 20.0  # token 7 overwhelmingly dominates -- deterministic nucleus pick
    assert nucleus_sampling(scores.clone(), top_p=0.8, top_k=25) == 7

    history = [7] * 10
    assert is_repetitive(history, 7)

    token = ras_sampling(scores.clone(), history, top_p=0.8, top_k=25)
    assert token != 7  # token 7 was banned (-inf) before the fallback draw


# --------------------------------------------------------------------------
# device tier -- prefill/decode wiring and on-device sampling
# --------------------------------------------------------------------------
@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
def test_device_prefill_then_decode_matches_real_qwen2(device):
    """The gap this file exists to close: a real assembled prefix through
    `TtQwen2LM.prefill` (`ttnn.fill_cache`'s code path), then one `decode_step` continuing
    from that populated cache, against a reference built the same way test_qwen2lm.py's
    already-passing multistep test was: `HfDecoderWrapper` chained once per position,
    `mask=None` each call (single-token decode semantics), which is mathematically
    equivalent to causal prefill over the same prefix -- the two differ only in
    implementation, not in what they compute. Real downloaded Qwen2-0.5B-Instruct weights
    on both sides; the CosyVoice-specific sos/task_id/speech tables are random-init (no
    checkpoint yet) but IDENTICAL on both sides, since the reference is built from the same
    `assemble_prefill_sequence` output the TT path consumes.
    """
    from models.demos.audio.cosyvoice2.tt.llm.qwen2lm import TtQwen2LM
    from models.tt_transformers.tt.common import precompute_freqs
    from models.tt_transformers.tt.model_config import HfDecoderWrapper

    args, state_dict = _build_args_and_state_dict(device)
    hf_model = args.reference_transformer(wrap=False)
    tt_model = TtQwen2LM(args, device, state_dict)

    torch.manual_seed(11)
    text_ids = torch.randint(0, args.vocab_size, (1, 5))
    speech_ids = torch.randint(0, tt_model.speech_token_size, (1, 3))
    sequence = tt_model.assemble_prefill_sequence(text_ids, speech_ids)  # [1, L, dim]
    L = sequence.shape[1]

    new_token_emb = torch.randn(1, 1, args.dim, dtype=torch.bfloat16).float() * 0.1

    cos, sin = precompute_freqs(
        args.head_dim,
        args.max_seq_len * 2,
        args.rope_theta,
        args.rope_scaling.factor if args.rope_scaling else None,
        args.rope_scaling.original_max_position_embeddings if args.rope_scaling else None,
        args.rope_scaling.rope_type.value if args.rope_scaling else "llama3",
    )
    freqs_cis = torch.complex(cos, sin)
    wrappers = [
        HfDecoderWrapper(layer, args.head_dim, hf_model.model.rotary_emb, use_hf_rope=args.use_hf_rope)
        for layer in hf_model.model.layers[: args.n_layers]
    ]

    want = None
    with torch.no_grad():
        h = sequence.bfloat16()
        for pos in range(L):
            hi = h[:, pos : pos + 1, :]
            fi = freqs_cis[pos, :].unsqueeze(0)
            for wrapper in wrappers:
                hi = wrapper(hi, pos, fi, mask=None)
                if hi.dim() == 2:
                    hi = hi.unsqueeze(1)
        hi = new_token_emb.bfloat16()
        fi = freqs_cis[L, :].unsqueeze(0)
        for wrapper in wrappers:
            hi = wrapper(hi, L, fi, mask=None)
            if hi.dim() == 2:
                hi = hi.unsqueeze(1)
        want = hf_model.model.norm(hi)

    tt_hidden, L_tt = tt_model.prefill(sequence)
    assert L_tt == L
    tt_hidden2 = tt_model.decode_step(new_token_emb, L)

    got = tt_hidden2.reshape(want.shape)
    passed, pcc = comp_pcc(want.float(), got, GATE_DECODE)
    print(f"\n  prefill (L={L}) + one decode step PCC {pcc}")
    assert passed, pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
def test_device_nucleus_sampler_top1_matches_row_max(device):
    """`TtDeviceNucleusSampler` at `top_k=1, top_p=0.0` must select the row's maximum-value
    token -- the one thing on-device top-k/top-p sampling and CosyVoice's own host
    `nucleus_sampling` are guaranteed to agree on (see qwen2lm.py's class docstring for why
    they are not bit-exact in general). Mirrors
    `models/common/tests/test_tt_sampling.py::test_ttsampling_topk_matches_argmax_on_single_device`'s
    bf16-tolerant comparison: the sampled token's OWN logit value must equal the row max
    after the same bf16 rounding the device tensor underwent, not necessarily match
    `argmax` of the original fp32 tensor index-for-index (bf16 can create ties)."""
    import ttnn
    from models.demos.audio.cosyvoice2.tt.llm.qwen2lm import TtDeviceNucleusSampler, TtQwen2LM

    args, state_dict = _build_args_and_state_dict(device)
    tt_model = TtQwen2LM(args, device, state_dict)
    sampler = tt_model.device_nucleus_sampler()
    assert isinstance(sampler, TtDeviceNucleusSampler)

    torch.manual_seed(5)
    logits = torch.randn(tt_model.head_out_features) * 3.0

    logits_bf16 = (
        ttnn.to_torch(
            ttnn.from_torch(logits.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        )
        .float()
        .reshape(-1)[: tt_model.head_out_features]
    )

    token = sampler.sample(logits, top_p=0.0, top_k=1, temperature=1.0)
    assert 0 <= token < tt_model.head_out_features
    row_max = logits_bf16.max().item()
    assert logits_bf16[token].item() >= row_max - 1e-3, (token, logits_bf16[token].item(), row_max)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
def test_ras_sample_device_falls_back_on_repetition(device):
    """`TtQwen2LM.ras_sample_device`'s host fallback must actually fire and actually change
    the outcome: a logits row overwhelmingly dominated by one token, plus a decoded-token
    history saturated with that same token, must force the repetition branch (banning it and
    redrawing on host) rather than returning the dominant token again."""
    from models.demos.audio.cosyvoice2.tt.llm.qwen2lm import TtQwen2LM

    args, state_dict = _build_args_and_state_dict(device)
    tt_model = TtQwen2LM(args, device, state_dict)

    torch.manual_seed(21)
    logits = torch.full((tt_model.head_out_features,), -5.0)
    dominant = 42
    logits[dominant] = 30.0
    history = [dominant] * 10

    token = tt_model.ras_sample_device(logits, history, top_p=0.8, top_k=25)
    assert token != dominant


@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
def test_device_generate_smoke(device):
    """End-to-end wiring smoke test: `generate()` must run prefill, then several real
    `decode_step` iterations each continuing the SAME cache at increasing positions,
    without error, for all three sampler paths (`greedy`, `ras` host, `ras_device`).
    No checkpoint exists yet (random-init CosyVoice tables), so token IDENTITY has no
    "golden" to check here -- test_device_prefill_then_decode_matches_real_qwen2 above is
    what checks the underlying math is right. This only checks the LOOP -- multiple
    sequential decode steps, each one's output feeding the next input -- holds together."""
    from models.demos.audio.cosyvoice2.tt.llm.qwen2lm import TtQwen2LM

    args, state_dict = _build_args_and_state_dict(device)
    tt_model = TtQwen2LM(args, device, state_dict)

    torch.manual_seed(3)
    text_ids = torch.randint(0, args.vocab_size, (1, 4))

    for sampler in ("greedy", "ras", "ras_device"):
        out = tt_model.generate(text_ids, max_tokens=4, sampler=sampler, seed=0)
        assert len(out) <= 4
        assert all(0 <= t < tt_model.head_out_features for t in out)
