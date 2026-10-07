# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""scripts/reference_env.py's two transformers shims, on upstream's own classes. Host only, reference venv only.

The reference venv runs python_env's transformers 5.12.1 under upstream code written for 4.51.3. Two shims make it
behave as 4.51.3 does (see reference_env.py). This file checks both on upstream's real `Qwen2Encoder` with the real
CosyVoice-BlankEN weights, each against a negative control:

1. **fp32:** the backbone loads in fp32. The unshimmed `from_pretrained` loads bf16.
2. **The decode mask:** upstream's `forward_one_step`, driven exactly as `inference_wrapper` drives it (a
   length-1 mask at each decode step), matches one no-cache forward over the same sequence within
   DECODE_REL_TOL of the hidden states' scale. The unshimmed method misses by orders of magnitude.

Also on record (2026-09-27): upstream under 5.12.1 with these shims, and under 4.51.3 with none, gave bit-identical
reference output (tokens and audio) on all seven corpus cases.

The reference venv has no pytest, so run this file there as a plain script (it exits non-zero on failure):

    COSYVOICE2_REPO=<upstream checkout> HF_HOME=<HF cache> \\
        $COSYVOICE2_REF_ENV/bin/python tests/reference/test_reference_env.py

Under python_env's pytest it skips: upstream's `cosyvoice` package, and its torchaudio / onnxruntime imports, aren't
available there.
"""
from __future__ import annotations

import os
import sys

import torch

try:
    import pytest
except ImportError:  # the reference venv
    pytest = None

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts")
)

PROMPT_LEN, N_STEPS = 24, 16
# max |d hidden| / max |hidden|, shimmed decode vs a no-cache forward, fp32. Post-norm hidden states reach ~100-160,
# so fp32 accumulation order alone gives absolute differences near 1e-3. Measured 2026-09-27 (seeds 0/1/2): relative
# 8.1e-6 / 2.2e-6 / 2.7e-6 shimmed; unshimmed 1.8 (seed 0).
DECODE_REL_TOL = 1e-4


def _upstream():
    """(reference_env, cosyvoice.llm.llm, BlankEN dir), shims installed; a skip outside the reference venv."""
    try:
        import torchaudio  # noqa: F401  (upstream's import chain needs it; python_env has none)

        import reference_env

        reference_env.setup_upstream()  # also refuses any transformers other than the pinned one
        import cosyvoice.llm.llm as llm
    except (ImportError, SystemExit) as e:
        if pytest is not None:
            pytest.skip(f"reference venv only ({e})")
        raise
    return reference_env, llm, os.path.join(reference_env.model_dir(), "CosyVoice-BlankEN")


def test_fp32_shim_loads_the_backbone_in_fp32():
    _, llm, blank_en = _upstream()
    encoder = llm.Qwen2Encoder(blank_en)  # upstream's own constructor: from_pretrained(pretrain_path), no dtype
    assert {p.dtype for p in encoder.parameters()} == {torch.float32}
    unshimmed = llm.Qwen2ForCausalLM.from_pretrained.__wrapped__(blank_en)
    assert next(unshimmed.parameters()).dtype == torch.bfloat16, "negative control: 5.12.1 unshimmed loads bf16"


def test_shimmed_decode_matches_a_no_cache_forward():
    _, llm, blank_en = _upstream()
    encoder = llm.Qwen2Encoder(blank_en)
    torch.manual_seed(0)
    ids = torch.randint(0, encoder.model.config.vocab_size, (1, PROMPT_LEN + N_STEPS))
    with torch.inference_mode():
        emb = encoder.model.model.embed_tokens(ids)
        out = encoder.model(inputs_embeds=emb, output_hidden_states=True, return_dict=True, use_cache=False)
        want = out.hidden_states[-1][:, PROMPT_LEN - 1 : PROMPT_LEN - 1 + N_STEPS]

        def decode(step):
            """inference_wrapper's calls: the prefix with its tril mask, then one token at a time with a 1x1 mask."""
            tril = torch.tril(torch.ones((1, PROMPT_LEN, PROMPT_LEN))).to(torch.bool)
            y, cache = step(encoder, emb[:, :PROMPT_LEN], masks=tril, cache=None)
            got = [y[:, -1]]
            for t in range(PROMPT_LEN, PROMPT_LEN + N_STEPS - 1):
                one = torch.tril(torch.ones((1, 1, 1))).to(torch.bool)
                y, cache = step(encoder, emb[:, t : t + 1], masks=one, cache=cache)
                got.append(y[:, -1])
            return torch.stack(got, dim=1)

        scale = want.abs().max().item()
        shimmed = (decode(llm.Qwen2Encoder.forward_one_step) - want).abs().max().item() / scale
        unshimmed = (decode(llm.Qwen2Encoder.forward_one_step.__wrapped__) - want).abs().max().item() / scale
    print(
        f"\n  decode vs no-cache forward over {N_STEPS} steps, max |d hidden| / max |hidden|: "
        f"shimmed {shimmed:.3g}, unshimmed {unshimmed:.3g}"
    )
    assert shimmed <= DECODE_REL_TOL, shimmed
    assert unshimmed > 100 * DECODE_REL_TOL, f"negative control: the unshimmed decode should be wrong ({unshimmed})"


if __name__ == "__main__":
    for test in (test_fp32_shim_loads_the_backbone_in_fp32, test_shimmed_decode_matches_a_no_cache_forward):
        test()
        print(f"PASSED {test.__name__}")
