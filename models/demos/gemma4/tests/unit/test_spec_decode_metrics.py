# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from models.demos.gemma4.tt import spec_decode


def test_phase_accounting_matches_tokens_and_time(monkeypatch):
    clock = iter((0.0, 2.0, 3.0, 5.0, 6.0, 9.0, 10.0))
    monkeypatch.setattr(spec_decode, "time", SimpleNamespace(perf_counter=lambda: next(clock)))
    decoder = object.__new__(spec_decode.SpeculativeDecoder)
    decoder._last_route = "host-loop-eager"
    decoder._metrics_active = False
    decoder._last_metrics = None

    decoder._metrics_begin("host-loop-eager")
    decoder._metrics_setup_done()
    first = spec_decode.time.perf_counter()
    decoder._metrics_iteration(first, 4)
    second = spec_decode.time.perf_counter()
    decoder._metrics_iteration(second, 2)
    decoder._metrics_finish()

    metrics = decoder._last_metrics
    assert metrics["setup_s"] == 2.0
    assert (metrics["first_s"], metrics["first_tokens"], metrics["first_iters"]) == (2.0, 4, 1)
    assert (metrics["rest_s"], metrics["rest_tokens"], metrics["rest_iters"]) == (3.0, 2, 1)
    assert metrics["wall_s"] == 10.0


def test_reuse_resets_phase_counters(monkeypatch):
    clock = iter((0.0, 1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0))
    monkeypatch.setattr(spec_decode, "time", SimpleNamespace(perf_counter=lambda: next(clock)))
    decoder = object.__new__(spec_decode.SpeculativeDecoder)
    decoder._last_route = "fused-packed-traced"
    decoder._metrics_active = False
    decoder._last_metrics = None

    decoder._metrics_begin(decoder._last_route)
    decoder._metrics_setup_done()
    start = spec_decode.time.perf_counter()
    decoder._metrics_iteration(start, 1)
    decoder._metrics_finish()
    assert decoder._last_metrics["rest_iters"] == 0
    assert decoder._last_metrics["rest_tokens"] == 0

    decoder._last_route = "host-loop-eager"
    decoder._metrics_begin(decoder._last_route)
    decoder._metrics_setup_done()
    decoder._metrics_finish()
    assert decoder._last_metrics["route"] == "host-loop-eager"
    assert decoder._last_metrics["first_iters"] == 0
    assert decoder._last_metrics["rest_tokens"] == 0


def test_early_eos_counts_only_emitted_first_iteration(monkeypatch):
    monkeypatch.delenv("GEMMA4_PLI_ALLOW_MIXED", raising=False)
    decoder = object.__new__(spec_decode.SpeculativeDecoder)
    decoder.target_has_pli = False
    decoder._pli_dev_host = True
    decoder._route = "host-loop"
    decoder._use_trace = False
    decoder._pv_a_prev = -1
    decoder._seed_mode = "reseed"
    decoder.stop_tokens = {42}
    decoder._metrics_active = False
    decoder._last_metrics = None

    class Hidden:
        def deallocate(self, force):
            pass

    decoder.seed = lambda *args, **kwargs: Hidden()
    decoder._draft = lambda *args, **kwargs: ([7], None)
    decoder._verify = lambda *args, **kwargs: (None, Hidden())
    decoder._accept_greedy = lambda *args, **kwargs: (0, [42])

    tokens, accepts = decoder.generate(1, 0, 8)
    assert tokens == [42]
    assert accepts == [0]
    assert decoder._last_metrics["first_tokens"] == 1
    assert decoder._last_metrics["first_iters"] == 1
    assert decoder._last_metrics["rest_tokens"] == 0
    assert decoder._last_metrics["rest_iters"] == 0


def test_zero_token_batched_call_has_no_steady_phase():
    decoder = object.__new__(spec_decode.SpeculativeDecoder)
    decoder.target_has_pli = False
    decoder._pli_dev_host = True
    decoder._route = "host-loop"
    decoder._use_trace = False
    decoder._metrics_active = False
    decoder._last_metrics = None
    outputs, accepts = decoder.generate_batched([1, 2], [0, 0], 0, 128)
    assert outputs == [[], []]
    assert accepts == [[], []]
    assert decoder._last_metrics["rest_iters"] == 0
    assert decoder._last_metrics["wall_s"] >= 0


def test_failed_request_clears_previous_timing(expect_error):
    decoder = object.__new__(spec_decode.SpeculativeDecoder)
    decoder._metrics_active = True
    decoder._last_metrics = {"stale": 1}
    decoder._last_route = "stale"

    def invalid_route(greedy):
        raise ValueError("invalid route")

    decoder._effective_route = invalid_route
    with expect_error(ValueError, "invalid route"):
        decoder.generate(1, 0, 1)
    assert decoder._last_metrics is None
    assert decoder._last_route is None
    assert not decoder._metrics_active
