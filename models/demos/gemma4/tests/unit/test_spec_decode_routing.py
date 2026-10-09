# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from models.demos.gemma4.tt.spec_decode import SpeculativeDecoder


def _decoder(*, pli, device_pli=True, trace=True):
    decoder = object.__new__(SpeculativeDecoder)
    decoder.target = SimpleNamespace(max_seq_len=4096, init_pli_device_weights=lambda: None)
    decoder.target_has_pli = pli
    decoder.draft_len = 3
    decoder._pli_dev_host = device_pli
    decoder._use_trace = trace
    decoder.tt_kv_cache = None
    decoder._last_route = None
    return decoder


def _reached_host_loop(*args, **kwargs):
    raise RuntimeError("host loop reached")


@pytest.mark.parametrize(
    "pli,device_pli,trace,temperature,fused",
    [
        (True, True, True, 0.0, True),
        (False, True, True, 0.0, True),
        (False, False, True, 0.0, True),
        (True, False, True, 0.0, False),
        (True, True, False, 0.0, False),
        (False, True, True, 0.7, False),
    ],
)
def test_generate_dispatch(pli, device_pli, trace, temperature, fused, expect_error):
    decoder = _decoder(pli=pli, device_pli=device_pli, trace=trace)
    decoder.generate_batched = lambda *args, **kwargs: pytest.fail("single user reached the batched body")
    calls = []
    decoder.generate_fused = lambda *args, **kwargs: calls.append((args, kwargs)) or ([7], [0])
    decoder.seed = _reached_host_loop
    if fused:
        assert decoder.generate(1, 0, 1, temperature=temperature) == ([7], [0])
        assert calls == [((1, 0, 1), {"_nested": True})]
    else:
        with expect_error(RuntimeError, "host loop reached"):
            decoder.generate(1, 0, 1, temperature=temperature)
        assert calls == []


@pytest.mark.parametrize("trace", [True, False])
def test_batched_dispatch_follows_trace(trace, expect_error):
    decoder = _decoder(pli=False, trace=trace)
    decoder._fused_reseed = False
    seen = []

    def traced(*args):
        seen.append("fused")
        return [[7]], [[0]]

    def eager_seed(*args):
        seen.append(decoder._use_trace)
        raise RuntimeError("host loop reached")

    decoder._generate_fused_traced_batched = traced
    decoder._seed_batched = eager_seed
    if trace:
        assert decoder.generate_batched([1], [0], 1, 64) == ([[7]], [[0]])
        assert seen == ["fused"]
    else:
        with expect_error(RuntimeError, "host loop reached"):
            decoder.generate_batched([1], [0], 1, 64)
        assert seen == [False]


def test_batched_traced_host_pli_rejected_before_device_work(expect_error):
    decoder = _decoder(pli=True, device_pli=False)
    decoder._fused_reseed = False
    decoder._seed_batched = lambda *args: pytest.fail("device work started")
    with expect_error(ValueError, "GEMMA4_PLI=device"):
        decoder.generate_batched([1], [0], 1, 64)


def test_batched_seed_uses_selected_device_pli():
    decoder = _decoder(pli=True)
    calls = []

    class Tensor:
        def deallocate(self, force):
            pass

    decoder._tokens_tensor = lambda tokens: Tensor()
    decoder._pos_tensors = lambda positions: (Tensor(), Tensor())
    # Upstream width-matches the page table to the masks' S_k, which reads the KV
    # cache block size and passes a width kwarg; mock both.
    decoder.tt_kv_cache = [[SimpleNamespace(padded_shape=(1, 1, 32, 1))]]
    decoder._page_table_users = lambda batch, width=None: Tensor()
    decoder.target.ttnn_verify_forward = lambda **kwargs: (calls.append(kwargs) or Tensor(), Tensor())
    decoder._seed_batched([1, 2], [0, 0])
    assert calls[0]["pli_on_device"] is True
    assert calls[0]["token_ids_host"] is None


def test_batched_packed_verify_uses_selected_device_pli():
    decoder = _decoder(pli=True)
    decoder.target.hf_config = SimpleNamespace(sliding_window=1024)
    calls = []

    class Tensor:
        def deallocate(self, force):
            pass

    decoder._packed_H = lambda: 1
    decoder._packed_mask_host = lambda *args: (None, None)
    decoder._from_b = lambda *args, **kwargs: Tensor()
    # Upstream width-matches the page table to the masks' S_k, which reads the KV
    # cache block size and passes a width kwarg; mock both.
    decoder.tt_kv_cache = [[SimpleNamespace(padded_shape=(1, 1, 32, 1))]]
    decoder._page_table_users = lambda batch, width=None: Tensor()
    decoder._logits_to_host = lambda logits: torch.zeros(4, 8)
    decoder.target.ttnn_packed_verify_forward = lambda **kwargs: (calls.append(kwargs) or Tensor(), Tensor())
    decoder._verify_packed_batched([[1, 2], [3, 4]], [0, 0])
    assert calls[0]["pli_on_device"] is True
    assert calls[0]["token_ids_host"] is None


@pytest.mark.parametrize(
    "trace,packed_env,expected",
    [
        (True, "1", "fused-packed-traced"),
        (True, "0", "fused-batch-dim-traced"),
        (False, "1", "fused-batch-dim-eager"),
    ],
)
def test_fused_route_label_names_the_body_that_runs(monkeypatch, trace, packed_env, expected):
    monkeypatch.setenv("GEMMA4_SPEC_FUSED_PACKED", packed_env)
    decoder = _decoder(pli=False, trace=trace)
    decoder._fused_reseed = False
    decoder._metrics_active = False
    decoder._last_metrics = None
    decoder.generate_fused(1, 0, 0)
    assert decoder._last_route == expected
