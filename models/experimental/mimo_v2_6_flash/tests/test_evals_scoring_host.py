# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host checks of the multimodal eval harness's bookkeeping (tests/evals): a request that failed for good is scored
wrong and never through a task's answer extraction; compare_runs.py counts a step's last run only and recomputes the
"failed = wrong" score from the samples; a journal is resumed per endpoint and a dry run keeps its own.

The task and client checks need lmms-eval (the eval venv); the rest runs anywhere."""

import importlib
import json
import os

import pytest

EVALS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evals")
PLUGIN = os.path.join(EVALS, "lmms_plugin")


@pytest.fixture()
def plugin(monkeypatch):
    monkeypatch.syspath_prepend(PLUGIN)
    monkeypatch.syspath_prepend(EVALS)


def failure(status, body):
    from mimo_lmms.failures import failed_text

    return failed_text(status, body)


FAILURES = [
    (500, "500 Internal Server Error: a device op failed"),
    (400, "400 Error code: 400 - {'error': {'message': 'Failed to fetch media from URL'}}"),
    ("conn", "APITimeoutError: Request timed out."),
    ("no_choices", '200 without choices: {"error": {"message": "Provider returned error", "code": 429}}'),
]


def test_failed_marker(plugin):
    from mimo_lmms.failures import FAILED_PREFIX, request_failed

    for status, body in FAILURES:
        text = failure(status, body)
        assert text.startswith(FAILED_PREFIX) and request_failed(text) and request_failed("  " + text)
    for answer in ("Answer: D", "", None, "The request failed.", "MIMO_REQUEST_FAILED"):
        assert not request_failed(answer)


@pytest.mark.parametrize("status, body", FAILURES)
def test_failed_requests_are_scored_wrong_whatever_the_error_text_holds(plugin, status, body):
    """The extractors find a letter in an error text (AA's last fallback reads the D of FAILED, Seldon's the first
    standalone letter), so a failure must not reach them: scored 0 for every answer key."""
    pytest.importorskip("lmms_eval")
    aa = importlib.import_module("mimo_lmms.tasks.mmmu_pro_aa.utils")
    vgi = importlib.import_module("mimo_lmms.tasks.vgi_bench.utils")
    text = failure(status, body)
    assert aa.aa_extract(text) is not None or vgi.parse_answer(text, 4) is not None  # what made this necessary
    for letter in aa.LETTERS:
        assert aa.aa_process_results({"answer": letter}, [text]) == {"aa_accuracy": 0.0}
    for index in range(4):
        doc = {"answers": ["w", "x", "y", "z"], "correct_answer": index}
        assert vgi.vgi_process_results(doc, [text]) == {"vgi_accuracy": 0.0, "vgi_accuracy_last_answer_line": 0.0}
    # Real answers are scored as before.
    assert aa.aa_process_results({"answer": "D"}, ["Reasoning...\nAnswer: D"]) == {"aa_accuracy": 1.0}
    doc = {"answers": ["w", "x", "y", "z"], "correct_answer": 2}
    assert vgi.vgi_process_results(doc, ["Answer: c"])["vgi_accuracy"] == 1.0


def test_videomme_scores_a_failed_request_as_an_empty_answer(plugin):
    pytest.importorskip("lmms_eval")
    utils = importlib.import_module("mimo_lmms.tasks.videomme_short.utils")
    seen = []
    upstream, utils.upstream_process_results = utils.upstream_process_results, lambda doc, results: seen.append(results)
    try:
        utils.videomme_process_results({}, [failure(500, "A. B. C. D.")])
        utils.videomme_process_results({}, ["B"])
    finally:
        utils.upstream_process_results = upstream
    assert seen == [[""], ["B"]]


def test_mmmu_options_literal_is_bounded(plugin, expect_error):
    pytest.importorskip("lmms_eval")
    aa = importlib.import_module("mimo_lmms.tasks.mmmu_pro_aa.utils")
    assert aa.parse_options("['3 m/s', \"it's 4\", '5']") == ["3 m/s", "it's 4", "5"]
    for bad in ("[" * 200_000, "{'a': 1}", "[]", str(list(range(11))), 7):
        with expect_error(ValueError, "MMMU-Pro options"):
            aa.parse_options(bad)


def write_lines(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def test_compare_runs_counts_the_last_run_and_failed_as_wrong(tmp_path, monkeypatch):
    """Five docs; the last run answered 0, 1, 2 (3 and 4 failed). The journal also holds an earlier run's answers
    (other settings: other keys) to docs 3 and 4. The harness scored the failure text of doc 3 as correct."""
    monkeypatch.syspath_prepend(EVALS)
    compare_runs = importlib.import_module("compare_runs")
    d = str(tmp_path / "tt" / "mmmu_pro_aa" / "full")
    usage = {"prompt_tokens": 100, "completion_tokens": 10}
    journal = [{"key": f"new{i}", "doc_id": i, "usage": usage} for i in range(3)]
    journal += [
        {"key": f"old{i}", "doc_id": i, "usage": {"prompt_tokens": 999, "completion_tokens": 999}} for i in (3, 4)
    ]
    write_lines(os.path.join(d, "journal.jsonl"), journal)
    correct = [1.0, 0.0, 1.0, 1.0, 0.0]  # doc 3: a failure the extraction read as the right letter
    write_lines(
        os.path.join(d, "lmms", "run", "x_samples_mmmu_pro_aa.jsonl"),
        [{"doc_id": i, "aa_accuracy": c} for i, c in enumerate(correct)],
    )
    summary = {"metrics": {"aa_accuracy,none": 0.6}, "n_samples": {"effective": 5}, "wall_s": 1}
    with open(os.path.join(d, "summary.json"), "w") as f:
        json.dump(summary, f)

    # A journal from before journal.keys.json: every record counts (the old behaviour), and says so.
    row = compare_runs.row("tt", "mmmu_pro_aa", d, {})
    assert row["journal"] == "all records" and row["answered"] == 5 and row["failed"] == 0

    with open(os.path.join(d, "journal.keys.json"), "w") as f:
        json.dump({"endpoint": "vllm|http://h/v1", "dry_run": False, "keys": [f"new{i}" for i in range(5)]}, f)
    row = compare_runs.row("tt", "mmmu_pro_aa", d, {})
    assert row["journal"] == "last run" and row["answered"] == 3 and row["failed"] == 2
    assert row["input_tokens"] == 300 and row["output_tokens"] == 30  # not the earlier run's 999s
    assert row["score_harness"] == pytest.approx(60.0)  # counts doc 3's failure as correct
    assert row["score"] == pytest.approx(40.0)  # failed = wrong: 2 of 5
    assert row["score_answered"] == pytest.approx(100 * 2 / 3)


def client(plugin_module, journal, **kwargs):
    return plugin_module.OpenRouterChat(journal=journal, raw_log=False, **kwargs)


def test_journal_is_per_endpoint_and_a_dry_run_keeps_its_own(plugin, tmp_path):
    pytest.importorskip("lmms_eval")
    pytest.importorskip("openai")
    module = importlib.import_module("mimo_lmms.models.mimo_openrouter")
    journal = str(tmp_path / "journal.jsonl")

    dry = client(module, journal, dry_run=True)
    assert dry.responses.endswith("journal.dryrun-journal.jsonl") and dry.endpoint == "dry-run"
    dry._append_journal({"key": "k1", "text": "A", "endpoint": dry.endpoint})
    assert not os.path.exists(journal)  # a dry run's "A" never enters the journal

    tt = client(module, journal, base_url="http://127.0.0.1:8000/v1", target="vllm")
    assert tt._done == {} and tt.responses == journal
    tt._append_journal({"key": "k1", "text": "B", "endpoint": tt.endpoint})
    tt._append_journal({"key": "k0", "text": "legacy"})  # a record from before the endpoint field
    again = client(module, journal, base_url="http://127.0.0.1:8000/v1/", target="vllm")
    assert {k: r["text"] for k, r in again._done.items()} == {"k1": "B", "k0": "legacy"}
    other = client(module, journal, base_url="https://openrouter.ai/api/v1", target="openrouter")
    assert set(other._done) == {"k0"}  # not another endpoint's answer
    assert set(client(module, journal, base_url="https://x/v1", journal_any_endpoint=True)._done) == {"k0", "k1"}
    assert set(client(module, journal, dry_run=True)._done) == {"k1"} and dry._done == {}

    # The last run's request keys, for compare_runs.py.
    tt._run_keys.update({"k1", "k2"})
    tt._write_stats(1.0)
    with open(str(tmp_path / "journal.keys.json")) as f:
        assert json.load(f) == {"endpoint": tt.endpoint, "dry_run": False, "keys": ["k1", "k2"]}
