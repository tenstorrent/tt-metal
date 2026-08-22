# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from models.autoports.poolside_laguna_xs_2_1.tests import serving_envelope_qualification as envelope


def _result(tokens, *, prompt_len=8192, ttft_ms=1000.0, tpot_ms=50.0, e2e_ms=4200.0):
    return envelope.prefix_q.CompletionResult(
        request_id="request",
        prompt_tokens=prompt_len,
        output_tokens=len(tokens),
        cached_tokens=None,
        prompt_token_ids_sha256="prompt",
        token_ids=list(tokens),
        token_ids_sha256=envelope.prefix_q.token_ids_sha256(tokens),
        finish_reason="length",
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        e2e_ms=e2e_ms,
    )


def test_completion_gate_is_exact_and_cache_off(expect_error):
    result = _result([3, 4])
    envelope._assert_completion(
        result,
        prompt_len=8192,
        output_len=2,
        oracle_ids=[3, 4],
    )

    result.cached_tokens = 64
    with expect_error(envelope.EnvelopeQualificationError, "cache-off"):
        envelope._assert_completion(result, prompt_len=8192, output_len=2)


def test_log_contracts_distinguish_the_two_fail_closed_envelopes(tmp_path, expect_error):
    log = tmp_path / "server.log"
    log.write_text(
        "profile: p150x2 | context: 65536 | seqs: 2 | hybrid KV: 0 | "
        "prefix cache: 0 | experimental: TT_LAGUNA_MULTI_SEQ_POOL=1 (qualified=0)\n"
    )
    path, offset, evidence = envelope._prepare_log(str(log), "multi-seq")

    assert path == log
    assert offset == log.stat().st_size
    assert evidence["validated"] is True
    with expect_error(envelope.EnvelopeQualificationError, "does not match"):
        envelope._prepare_log(str(log), "context-262k")


def test_log_tail_rejects_only_new_hard_faults(tmp_path, expect_error):
    log = tmp_path / "server.log"
    log.write_text("traceback (most recent call last) during an old rejected boot\n")
    offset = log.stat().st_size
    with log.open("a") as stream:
        stream.write("healthy measured request\n")
    assert envelope._scan_log_tail(log, offset)["hard_fault_markers"] == []

    offset = log.stat().st_size
    with log.open("a") as stream:
        stream.write("FATAL ERROR after request\n")
    with expect_error(envelope.EnvelopeQualificationError, "hard fault"):
        envelope._scan_log_tail(log, offset)


def test_context_gate_normalizes_ttft_by_causal_attention_work(monkeypatch):
    boundary = _result(
        [7, 8],
        prompt_len=131136,
        ttft_ms=100000.0,
        e2e_ms=100100.0,
    )
    cap = _result(
        [9, 10],
        prompt_len=262142,
        ttft_ms=190000.0,
        e2e_ms=190100.0,
    )
    responses = iter((boundary, boundary, cap))
    client = SimpleNamespace(completion=lambda _payload: next(responses))
    monkeypatch.setattr(
        envelope.prefix_q,
        "deterministic_token_ids",
        lambda length, **_kwargs: [100] * length,
    )
    args = SimpleNamespace(
        boundary_prompt_len=131136,
        boundary_output_len=2,
        boundary_repetitions=2,
        cap_prompt_len=262142,
        cap_output_len=2,
        model=envelope.DEFAULT_MODEL,
        run_id="unit",
        seed=1234,
        vocab_size=envelope.DEFAULT_VOCAB_SIZE,
        maximum_normalized_ttft_ratio=1.15,
    )

    evidence = envelope.run_context_262k(args, client)

    assert evidence["checks"]["no_power_of_two_ttft_cliff"] is True
    assert evidence["attention_work_ratio"] > 3.9
    assert evidence["normalized_ttft_ratio"] < 1.15


def test_context_gate_rejects_work_beyond_attention_scaling_bound(monkeypatch, expect_error):
    boundary = _result(
        [7, 8],
        prompt_len=131136,
        ttft_ms=100000.0,
        e2e_ms=100100.0,
    )
    cap = _result(
        [9, 10],
        prompt_len=262142,
        ttft_ms=500000.0,
        e2e_ms=500100.0,
    )
    responses = iter((boundary, boundary, cap))
    client = SimpleNamespace(completion=lambda _payload: next(responses))
    monkeypatch.setattr(
        envelope.prefix_q,
        "deterministic_token_ids",
        lambda length, **_kwargs: [100] * length,
    )
    args = SimpleNamespace(
        boundary_prompt_len=131136,
        boundary_output_len=2,
        boundary_repetitions=2,
        cap_prompt_len=262142,
        cap_output_len=2,
        model=envelope.DEFAULT_MODEL,
        run_id="unit",
        seed=1234,
        vocab_size=envelope.DEFAULT_VOCAB_SIZE,
        maximum_normalized_ttft_ratio=1.15,
    )

    with expect_error(envelope.EnvelopeQualificationError, "attention_work_ratio"):
        envelope.run_context_262k(args, client)
