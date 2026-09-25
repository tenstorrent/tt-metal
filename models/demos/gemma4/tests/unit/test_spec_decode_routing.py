# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from models.demos.gemma4.tt.spec_decode import SpeculativeDecoder


def _decoder(monkeypatch, *, pli, mechanism="device", route="auto", trace=True):
    monkeypatch.setenv("GEMMA4_PLI", mechanism)
    monkeypatch.delenv("GEMMA4_DECODE_PLI_DEV", raising=False)
    decoder = object.__new__(SpeculativeDecoder)
    decoder.target = SimpleNamespace(max_seq_len=4096)
    decoder.target_has_pli = pli
    decoder._pli_dev_host = mechanism == "device"
    decoder._route = route
    decoder._use_trace = trace
    decoder.tt_kv_cache = None
    decoder._last_route = None
    return decoder


@pytest.mark.parametrize(
    "pli,mechanism,expected",
    [
        (True, "device", "fused-packed"),
        (True, "host", "host-loop"),
        (False, "device", "fused-batch-dim"),
    ],
)
def test_auto_dispatch(monkeypatch, pli, mechanism, expected):
    decoder = _decoder(monkeypatch, pli=pli, mechanism=mechanism)
    assert decoder._effective_route() == expected


@pytest.mark.parametrize("route", ["fused-packed", "fused-batch-dim"])
def test_explicit_fusion_without_trace_fails_before_seed(monkeypatch, route, expect_error):
    decoder = _decoder(monkeypatch, pli=False, route=route, trace=False)
    decoder.seed = lambda *args, **kwargs: pytest.fail("device work started")
    with expect_error(ValueError, "requires GEMMA4_SPEC_TRACE"):
        decoder.generate(1, 0, 1)


def test_pli_batch_dim_rejected_before_seed(monkeypatch, expect_error):
    decoder = _decoder(monkeypatch, pli=True, route="fused-batch-dim")
    decoder.seed = lambda *args, **kwargs: pytest.fail("device work started")
    with expect_error(ValueError, "cannot use fused-batch-dim"):
        decoder.generate(1, 0, 1)


def test_mixed_pli_needs_diagnostic_override(monkeypatch, expect_error):
    decoder = _decoder(monkeypatch, pli=True)
    monkeypatch.setenv("GEMMA4_DECODE_PLI_DEV", "0")
    with expect_error(ValueError, "different PLI"):
        decoder._effective_route()
    monkeypatch.setenv("GEMMA4_PLI_ALLOW_MIXED", "1")
    assert decoder._effective_route() == "fused-packed"


def test_batched_seed_uses_selected_device_pli(monkeypatch):
    decoder = _decoder(monkeypatch, pli=True)
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


def test_batched_packed_verify_uses_selected_device_pli(monkeypatch):
    decoder = _decoder(monkeypatch, pli=True)
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
    decoder = _decoder(monkeypatch, pli=False, route="fused-batch-dim", trace=trace)
    decoder._fused_reseed = False
    decoder._metrics_active = False
    decoder._last_metrics = None
    decoder.generate_fused(1, 0, 0)
    assert decoder._last_route == expected
