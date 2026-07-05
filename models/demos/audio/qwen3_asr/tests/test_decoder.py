# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""PCC + regression tests for the ttnn Qwen3-1.7B text decoder of Qwen3-ASR (P150).

  test_decoder_prefill_pcc          last-token prefill logits vs golden (+ argmax match)
  test_decoder_generate             prefill golden embeds -> greedy decode -> non-empty text
  test_prefill_length_state_regression
                                    interleaving prefills of DIFFERENT real lengths must NOT
                                    change a fixed prompt's greedy output. This is the exact
                                    failure the fixed-512-multiple / fixed-14s workaround
                                    guards against (see tt/qwen3_asr_decoder.prefill_logits
                                    and server/LONGFORM_DESIGN.md). If a future change makes
                                    decoder state length-keyed again, this fails.

Golden tensors and the extracted text-decoder checkpoint are supplied via env vars; see
tests/conftest.py. Run on a P150 (single Blackhole) box that has them staged.
"""
import numpy as np
import pytest
import torch

import ttnn
from models.demos.audio.qwen3_asr.tt.qwen3_asr_decoder import Qwen3ASRDecoder
from models.tt_transformers.tt.model_config import ModelArgs
from tests.ttnn.utils_for_testing import assert_with_pcc

DECODER_DEVICE_PARAMS = {"l1_small_size": 32768, "trace_region_size": 200000000}

PCC_PREFILL = 0.97


def _build_decoder(device, max_seq_len):
    args = ModelArgs(device, max_batch_size=1, max_seq_len=max_seq_len)
    state_dict = args.load_state_dict()
    model = Qwen3ASRDecoder(
        args, ttnn.bfloat16, device, state_dict,
        args.weight_cache_path(ttnn.bfloat16), use_paged_kv_cache=False,
    )
    return model


@pytest.mark.parametrize("device_params", [DECODER_DEVICE_PARAMS], indirect=True)
def test_decoder_prefill_pcc(device, golden, text_decoder_ckpt):
    """Prefill over golden merged embeds: last-token logits PCC + argmax match vs golden."""
    model = _build_decoder(device, max_seq_len=1024)
    ie = golden("inputs_embeds.npy").unsqueeze(0)  # (1, S, 2048)

    logits, S = model.prefill_logits(ie)
    gold = golden("lm_head.npy")[0, S - 1]  # (vocab,)

    assert int(logits.argmax()) == int(gold.argmax()), "greedy next-token disagrees with golden"
    assert_with_pcc(gold, logits, PCC_PREFILL)


@pytest.mark.parametrize("device_params", [DECODER_DEVICE_PARAMS], indirect=True)
def test_decoder_generate(device, golden, text_decoder_ckpt):
    """Full prefill + greedy decode loop yields a non-trivial transcription (not just tag+EOS)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(text_decoder_ckpt)
    model = _build_decoder(device, max_seq_len=1024)
    ie = golden("inputs_embeds.npy").unsqueeze(0)

    ids = model.generate(ie, max_new_tokens=64)
    txt = tok.decode(ids, skip_special_tokens=True).strip()

    # A corrupt decoder emits only the language tag + EOS -> empty/near-empty transcript.
    assert len(ids) > 3, f"decode produced too few tokens ({len(ids)}): likely empty transcript"
    assert len(txt) > 0, "decoded transcript is empty"


@pytest.mark.parametrize("device_params", [DECODER_DEVICE_PARAMS], indirect=True)
def test_prefill_length_state_regression(device, golden, text_decoder_ckpt):
    """Greedy decode of a fixed prompt must be identical before/after prefilling other lengths.

    Root cause of the server's empty transcripts: mixing prefills of different real sequence
    lengths in one long-lived process corrupted the decoder (it locked to the first length).
    ``prefill_logits`` works around it by padding every prefill to the same 512-multiple path.
    This test drives the alternation directly and asserts determinism.
    """
    model = _build_decoder(device, max_seq_len=2048)
    X = golden("inputs_embeds.npy").unsqueeze(0)  # (1, S, 2048)
    Xs = X[:, :100, :].contiguous()  # shorter real length
    Xl = torch.cat([X, X[:, 20:, :]], dim=1).contiguous()  # longer real length

    base = model.generate(X, max_new_tokens=16)
    assert model.generate(X, max_new_tokens=16) == base, "non-deterministic with no interleave"

    model.generate(Xs, max_new_tokens=16)
    assert model.generate(X, max_new_tokens=16) == base, "output drifted after a shorter prefill"

    model.generate(Xl, max_new_tokens=16)
    assert model.generate(X, max_new_tokens=16) == base, "output drifted after a longer prefill"

    # a few rounds of alternation to catch cumulative drift
    for i in range(3):
        model.generate(Xs, max_new_tokens=16)
        model.generate(Xl, max_new_tokens=16)
        assert model.generate(X, max_new_tokens=16) == base, f"cumulative drift at round {i + 1}"
