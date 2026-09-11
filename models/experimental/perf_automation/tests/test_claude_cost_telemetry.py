import json


def _result_event():
    return {
        "type": "result",
        "result": "DECISION: continue\nREASON: valid",
        "duration_ms": 1234,
        "total_cost_usd": 0.456,
        "usage": {
            "input_tokens": 100,
            "cache_creation_input_tokens": 200,
            "cache_read_input_tokens": 300,
            "output_tokens": 40,
        },
    }


def test_cli_result_records_each_billable_token_type():
    from agent.probes import _cli_result_and_usage

    text, usage = _cli_result_and_usage(json.dumps(_result_event()))

    assert text.startswith("DECISION: continue")
    assert usage == {
        "tokens_in": 600,
        "tokens_input_uncached": 100,
        "tokens_cache_creation": 200,
        "tokens_cached": 300,
        "tokens_out": 40,
        "cost_usd": 0.456,
        "latency_s": 1.23,
    }


def test_round_stream_appends_cost_row(tmp_path):
    from cc_optimize.run import _record_round_agent_usage

    agent_log = tmp_path / "round.agent.log"
    agent_log.write_text('{"type":"assistant","message":"working"}\n' + json.dumps(_result_event()) + "\n")
    calls = tmp_path / "run-123" / "agent_calls.jsonl"

    _record_round_agent_usage(str(agent_log), 0, calls, iteration=1, task="main")

    row = json.loads(calls.read_text())
    assert row["run_id"] == "run-123"
    assert row["phase"] == "optimize"
    assert row["tokens_input_uncached"] == 100
    assert row["tokens_cache_creation"] == 200
    assert row["tokens_cached"] == 300
    assert row["tokens_out"] == 40
    assert row["cost_usd"] == 0.456
