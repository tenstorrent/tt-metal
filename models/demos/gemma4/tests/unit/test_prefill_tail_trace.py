# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""GEMMA4_PREFILL_TAIL_IN_TRACE (item 4 of the gemma4_12b_qb2 port): a prefill capture that ends with the last-token
tile, the lm_head and the softcap (TP-sharded, the host concatenating one row per chip) returns the same last-token
logits, bit for bit, as the capture that returns the hidden states for the eager tail -- for host logits and for the
device sampler's tokens -- and both agree with the eager prefill.

    pytest -k "1x4" models/demos/gemma4/tests/unit/test_prefill_tail_trace.py
"""

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.common.sampling import SamplingParams
from models.demos.gemma4.tt.generator import Gemma4Generator
from models.demos.gemma4.tt.model import PREFILL_TAIL_TRACE_ENV, Gemma4Model, prefill_tail_in_trace_default

from ...tests.test_factory import TestFactory, _get_model_path, compare_tensors, parametrize_mesh_with_fabric
from .test_prefill_trace import _allocate_fresh_kv_cache, _create_page_table, _page_params

PREFILL_LENS = (128, 1024)
MAX_NEW = 32


def _release_prefill_traces(generator):
    mesh = generator.model_args[0].mesh_device
    for tid in list(generator.trace_id_prefill.values()):
        if tid is not None:
            ttnn.release_trace(mesh, tid)
    generator.trace_id_prefill = defaultdict(lambda: None)
    generator.trace_inputs_prefill = defaultdict(lambda: None)
    generator.trace_output_prefill = defaultdict(lambda: None)


def _prefill(generator, kv, tokens, page_table, prompt_lens, *, enable_trace, sampling=None):
    out = generator.prefill_forward_text(
        tokens,
        page_table=page_table,
        kv_cache=kv,
        prompt_lens=prompt_lens,
        enable_trace=enable_trace,
        warmup_prefill=False,
        sampling_params=sampling,
    )
    if sampling is not None:
        tokens_out = out[0] if isinstance(out, tuple) else out
        return torch.as_tensor(tokens_out).reshape(-1).to(torch.int64)
    return (out.squeeze(1) if out.dim() == 3 else out).float()


@pytest.mark.gemma4_prefill_trace  # loads the checkpoint: skipped under a config-only HF_MODEL (tests/conftest.py)
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 4)])
def test_prefill_tail_in_trace_bit_exact(mesh_device, device_params, reset_seeds):
    """Traced prefill, tail outside vs inside the capture: identical host logits and sampler tokens; eager parity.
    Then the retained tail-in-trace capture of each bucket replays prompts whose last token sits in another tile
    (the generator rewrites the persistent tile bounds before every replay) and still matches the eager prefill."""
    hf_config = TestFactory.create_hf_config()
    if int(getattr(hf_config, "hidden_size_per_layer_input", 0) or 0) > 0:
        pytest.skip("PLI models disable prefill trace")
    max_seq_len = max(max(PREFILL_LENS) + MAX_NEW, 4096)
    paged_cfg = _page_params(1, max(PREFILL_LENS), MAX_NEW)
    page_table = _create_page_table(1, paged_cfg)
    generator, kv_caches, _tok = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=_get_model_path(),
        max_batch_size=1,
        max_seq_len=max_seq_len,
        paged_attention_config=paged_cfg,
        bounded_sliding_kv_cache=False,
    )
    model_args = generator.model_args[0]
    models = generator.model
    assert all(hasattr(m, "_prefill_tail_in_trace") for m in models)
    gen = torch.Generator().manual_seed(1234)
    greedy = SamplingParams(temperature=0.0, top_k=1, top_p=1.0)

    def fresh_kv():
        return [
            _allocate_fresh_kv_cache(
                models[0], max_batch_size=1, max_seq_len=max_seq_len, paged_attention_config=paged_cfg
            )
        ]

    for prefill_len in PREFILL_LENS:
        assert model_args.can_enable_trace(prefill_len, batch_size=1)
        prompt_len = prefill_len - 3  # the last token sits in the bucket's last tile, off its first row
        tokens = torch.zeros(1, prefill_len, dtype=torch.int32)
        tokens[0, :prompt_len] = torch.randint(10, hf_config.vocab_size - 10, (prompt_len,), generator=gen)
        prompt_lens = torch.tensor([prompt_len], dtype=torch.long)
        eager = _prefill(generator, fresh_kv(), tokens, page_table, prompt_lens, enable_trace=False)
        results = {}
        for tail_in_trace in (False, True):
            _release_prefill_traces(generator)
            for m in models:
                m._prefill_tail_in_trace = tail_in_trace
            logits = _prefill(generator, fresh_kv(), tokens, page_table, prompt_lens, enable_trace=True)
            tokens_dev = None
            if models[0].sampling is not None:
                tokens_dev = _prefill(
                    generator, fresh_kv(), tokens, page_table, prompt_lens, enable_trace=True, sampling=greedy
                )
            results[tail_in_trace] = (logits, tokens_dev)
        off_logits, off_tokens = results[False]
        on_logits, on_tokens = results[True]
        passing, pcc = compare_tensors(off_logits, eager, pcc_threshold=0.99)
        logger.info(
            f"prefill_len={prefill_len}: traced (eager tail) vs eager PCC {pcc}; argmax eager {int(eager.argmax(-1)[0])}"
        )
        assert passing, f"traced vs eager: {pcc}"
        assert off_logits.shape == on_logits.shape == (1, hf_config.vocab_size)
        assert torch.equal(off_logits, on_logits), (
            f"prefill_len={prefill_len}: the traced tail's logits differ from the eager tail's "
            f"(max abs {float((off_logits - on_logits).abs().max())})"
        )
        if off_tokens is not None:
            assert torch.equal(off_tokens, on_tokens), f"sampler tokens {off_tokens.tolist()} vs {on_tokens.tolist()}"
            assert int(on_tokens[0]) == int(on_logits.argmax(-1)[0]), "the device sampler's greedy token != argmax"
        # The tail-in-trace capture is still live (no release): replay it for prompts whose last token lands in
        # other tiles of the bucket (first tile, a middle tile, and the capture's tile again at another row); every
        # replay must slice the tile the generator wrote for that request, not the capture-time one.
        assert generator.trace_id_prefill.get(prefill_len) is not None or any(
            v is not None for v in generator.trace_id_prefill.values()
        ), "the tail-in-trace capture should be retained for the replays below"
        for prompt_len in (1, prefill_len // 2 + 5, prefill_len - 30):
            tokens = torch.zeros(1, prefill_len, dtype=torch.int32)
            tokens[0, :prompt_len] = torch.randint(10, hf_config.vocab_size - 10, (prompt_len,), generator=gen)
            prompt_lens = torch.tensor([prompt_len], dtype=torch.long)
            eager_pl = _prefill(generator, fresh_kv(), tokens, page_table, prompt_lens, enable_trace=False)
            traced_pl = _prefill(generator, fresh_kv(), tokens, page_table, prompt_lens, enable_trace=True)
            passing, pcc = compare_tensors(traced_pl, eager_pl, pcc_threshold=0.99)
            logger.info(
                f"prefill_len={prefill_len} prompt_len={prompt_len} (tile {((prompt_len - 1) // 32) * 32}): replayed tail vs eager PCC {pcc}"
            )
            assert passing, f"prompt_len={prompt_len}: replayed tail vs eager {pcc}"
            assert int(traced_pl.argmax(-1)[0]) == int(
                eager_pl.argmax(-1)[0]
            ), f"prompt_len={prompt_len}: the replayed tail's argmax differs from the eager prefill's"
            if models[0].sampling is not None:
                tokens_pl = _prefill(
                    generator, fresh_kv(), tokens, page_table, prompt_lens, enable_trace=True, sampling=greedy
                )
                assert int(tokens_pl[0]) == int(
                    eager_pl.argmax(-1)[0]
                ), f"prompt_len={prompt_len}: the replayed sampler token != the eager argmax"
    _release_prefill_traces(generator)


def test_prefill_tail_in_trace_default_policy(monkeypatch):
    """Host: the env var overrides, else the measured-shape policy; never on lanes / fractured meshes."""
    mesh = SimpleNamespace(shape=(1, 4))
    monkeypatch.setenv(PREFILL_TAIL_TRACE_ENV, "1")
    assert prefill_tail_in_trace_default(mesh) is True
    assert prefill_tail_in_trace_default(mesh, SimpleNamespace(lane_sharded=True, weight_fracture=False)) is False
    assert prefill_tail_in_trace_default(mesh, SimpleNamespace(lane_sharded=False, weight_fracture=True)) is False
    monkeypatch.setenv(PREFILL_TAIL_TRACE_ENV, "0")
    assert prefill_tail_in_trace_default(mesh) is False
    monkeypatch.delenv(PREFILL_TAIL_TRACE_ENV)
    from models.demos.gemma4.tt import model as model_module

    monkeypatch.setattr(model_module, "is_blackhole", lambda: True)
    assert prefill_tail_in_trace_default(mesh) is True  # the shipped (1,4) default
    monkeypatch.setattr(model_module, "PREFILL_TAIL_TRACE_DEFAULT_SHAPES", ())
    assert prefill_tail_in_trace_default(mesh) is False
    monkeypatch.setattr(model_module, "PREFILL_TAIL_TRACE_DEFAULT_SHAPES", ((1, 4), (1, 8)))
    assert prefill_tail_in_trace_default(mesh) is True
    assert prefill_tail_in_trace_default(SimpleNamespace(shape=(1, 8))) is True
    assert prefill_tail_in_trace_default(SimpleNamespace(shape=(8, 4))) is False
    monkeypatch.setattr(model_module, "is_blackhole", lambda: False)  # Wormhole (T3K is (1,8) too)
    assert prefill_tail_in_trace_default(mesh) is False
    assert prefill_tail_in_trace_default(SimpleNamespace(shape=(1, 8))) is False


def test_prefill_tail_applies_to_the_single_user_single_chunk_form_only():
    """Host: the batched form, the traced chunk walk and bounded sliding keep the eager tail."""
    m = Gemma4Model.__new__(Gemma4Model)
    m._prefill_tail_in_trace = True
    m.bounded_sliding_kv_cache = False
    assert m._prefill_tail_in_trace_applies(1, None) is True
    assert m._prefill_tail_in_trace_applies(2, None) is False
    assert m._prefill_tail_in_trace_applies(1, object()) is False
    m.bounded_sliding_kv_cache = True
    assert m._prefill_tail_in_trace_applies(1, None) is False
    m.bounded_sliding_kv_cache = False
    m._prefill_tail_in_trace = False
    assert m._prefill_tail_in_trace_applies(1, None) is False
