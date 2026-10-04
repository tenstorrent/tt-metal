"""Observe the frozen structured row; retain stock scoring and pytest selection."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
STOCK_SKIPS = {
    "test_a_solo_split_prefill_recalls_its_needle",
    "test_a_long_split_prefill_recalls_its_needle",
    "test_prefills_sharing_a_step_each_recall_their_own_needle",
    "test_chat_logprobs_all_vocab",
}


def fixture(path):
    encoded = path.read_bytes()
    digest = hashlib.sha256(encoded).hexdigest()
    assert digest == path.with_suffix(".sha256").read_text().strip()
    rows = json.loads(encoded)
    assert len(rows) == 32
    return rows, digest


def sampling(args, output, source_root):
    import pytest

    source_root = source_root.resolve(strict=True)

    class Provenance:
        def __init__(self):
            self.collected = []
            self.sources = {}
            self.outcomes = {}

        def pytest_collection_finish(self, session):
            assert len(session.items) == 47, "Original C1 sampling selection"
            self.collected = [item.nodeid for item in session.items]
            assert len(set(self.collected)) == 47
            for item in session.items:
                path = Path(item.module.__file__).resolve(strict=True)
                relative = path.relative_to(source_root)
                assert relative.suffix == ".py"
                self.sources[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()

        def pytest_runtest_logreport(self, report):
            assert report.nodeid in self.collected
            self.outcomes.setdefault(report.nodeid, {})[report.when] = report.outcome

        def pytest_sessionfinish(self, session, exitstatus):
            (output / "sampling-provenance.json").write_text(
                json.dumps(
                    {
                        "collected": self.collected,
                        "sources_sha256": self.sources,
                        "outcomes": self.outcomes,
                        "exit_status": int(exitstatus),
                    },
                    indent=2,
                )
                + "\n"
            )

    return pytest.main(args, plugins=[Provenance()])


def completed_sampling(output, code):
    receipt = json.loads((output / "sampling-provenance.json").read_text())
    names = receipt["collected"]
    assert len(names) == len(set(names)) == 47 and receipt["sources_sha256"]
    assert receipt["exit_status"] == code and code in (0, 1)
    assert set(receipt["outcomes"]) == set(names)
    skipped, failed = [], []
    for name, outcomes in receipt["outcomes"].items():
        assert outcomes["setup"] == outcomes["teardown"] == "passed"
        assert outcomes["call"] in ("passed", "failed", "skipped")
        if outcomes["call"] == "skipped":
            assert name.rsplit("::", 1)[1] in STOCK_SKIPS, "Unexpected sampling skip"
            skipped.append(name)
        if outcomes["call"] == "failed":
            failed.append(name)
    assert (code == 1) == bool(failed)
    return {"collected": len(names), "passed": 47 - len(skipped) - len(failed), "skipped": skipped, "failed": failed}


def completed(output, fixtures, source, source_sha256, code):
    """An assertion failure is complete only with all original request results."""
    assert code in (0, 1)
    assert json.loads((output / "phase-0.json").read_text())["exit_code"] == 0
    plain = json.loads((output / "vllm_result.json").read_text())
    assert plain["model_id"] == MODEL and plain["num_prompts"] == plain["completed"] == 32
    assert plain["failed"] == 0
    _, plain_digest = fixture(fixtures / "structured-plain.json")
    requests, structured_digest = fixture(fixtures / "structured-json-unique.json")
    assert all(row["structure_type"] == "json" and row["expected_output_len"] == 1024 for row in requests)
    result = json.loads((output / "vllm_result_structured.json").read_text())
    assert result["model_id"] == result["tokenizer_id"] == MODEL
    assert result["num_prompts"] == result["completed"] == 32
    assert (
        len(result["outputs"]) == len(result["errors"]) == len(result["input_lens"]) == len(result["output_lens"]) == 32
    )
    assert all(error == "" for error in result["errors"]), "Incomplete structured transport"
    assert result["input_lens"] == [row["prompt_len"] for row in requests]
    assert [row["expected"] for row in result["outputs"]] == [row["completion"] for row in requests]
    # Execute only the exact pinned stock evaluator; do not introduce schema
    # validation, substitute a score or change the strict original threshold.
    text = source.read_text()
    assert hashlib.sha256(text.encode()).hexdigest() == source_sha256
    tree = ast.parse(text)
    evaluator = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "evaluate")
    scope = {"json": json}
    exec(compile(ast.Module(body=[evaluator], type_ignores=[]), str(source), "exec"), scope)
    evaluated = copy.deepcopy(result["outputs"])
    score = scope["evaluate"](evaluated, SimpleNamespace(structure_type="json"))
    assert score == result["correct_rate(%)"]
    assert [row["correctness"] for row in evaluated] == [row["correctness"] for row in result["outputs"]]
    phase = json.loads((output / "phase-1.json").read_text())["exit_code"]
    if score != 100.0:
        assert phase == code == 1
        assert not (output / "phase-2.json").exists(), "Preserve stock assertion order"
        sampling_result = None
    else:
        assert phase == 0
        assert json.loads((output / "phase-2.json").read_text())["exit_code"] == code
        sampling_result = completed_sampling(output, code)
    return {
        "requests": 32,
        "completed": 32,
        "stock_correct_rate_percent": score,
        "json_valid": sum(row["correctness"] for row in evaluated),
        "empty_outputs": sum(not row["generated"] for row in evaluated),
        "plain_fixture_sha256": plain_digest,
        "structured_fixture_sha256": structured_digest,
        "stock_source_sha256": source_sha256,
        "sampling": sampling_result,
    }


if __name__ == "__main__":
    assert len(sys.argv) > 3 and sys.argv[1] == "sampling"
    raise SystemExit(sampling(sys.argv[3:], Path(sys.argv[2]), Path("/pair/plugin/tests/tt")))
