# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import model_trace_tests as trace_tests
import verify_time_budget
from prepare_test_matrix import build_test_matrix, collect_skus_from_tests, load_sku_config, load_tests


@pytest.fixture
def registry():
    return load_tests(str(trace_tests.E2E_TESTS))


@pytest.fixture
def sku_config():
    return load_sku_config(str(trace_tests.SKU_CONFIG))


def test_monthly_audit_covers_configured_entries_without_mutating_source(registry, sku_config):
    original = copy.deepcopy(registry)
    config = trace_tests.load_trace_config()
    llms = [entry for entry in registry if trace_tests.is_llm(entry, config)]
    source = build_test_matrix(llms, collect_skus_from_tests(llms), sku_config)
    audit = trace_tests.audit_matrix(registry, sku_config)
    assert len(audit) == len(source)
    assert {entry["tier"] for entry in audit} == {1, 2, 3}
    for before, after in zip(source, audit):
        assert after["cmd"] == trace_tests.tracked_command(before["cmd"])
        assert after["timeout"] == before["timeout"] * 2
        for key in ("sku", "tier", "runs_on", "model", "model_family", "owner_id", "multihost", "weights-cache-mode"):
            assert after.get(key) == before.get(key)
    assert registry == original


@pytest.mark.parametrize(
    "family,model",
    [
        ("Qwen", "qwenimage"),
        ("ResNet", "resnet50"),
        ("Flux", "flux.1-dev"),
        (None, "[tt-train] tinyllama"),
        ("DeepSeek", "deepseek-v3-tg"),
        ("Mistral", "mixtral-8x7b"),
        ("Falcon", "falcon7b"),
        ("Phi", "phi-3-mini"),
        ("Mamba", "mamba2-2.7b"),
    ],
)
def test_audit_excludes_models_outside_configured_scope(family, model):
    assert not trace_tests.is_llm({"model_family": family, "model": model}, trace_tests.load_trace_config())


def test_time_budget_skips_trace_config_and_counts_generated_sweeps():
    registries = dict(verify_time_budget.load_tests(str(trace_tests.TRACE_CONFIG.parent)))
    assert trace_tests.TRACE_CONFIG.name not in registries
    assert any("model_trace_tests.py run" in test["cmd"] for test in registries[trace_tests.SWEEP_TESTS.name])


def test_audit_filters_keep_exact_models_and_skus(registry, sku_config):
    matrix = trace_tests.audit_matrix(registry, sku_config, skus="bh_quietbox_2", model="QWEN3.6-27B,gemma-4-26b-a4b")
    assert {entry["model"] for entry in matrix} == {"qwen3.6-27b", "gemma-4-26b-a4b"}
    assert {entry["sku"] for entry in matrix} == {"bh_quietbox_2"}
    with pytest.raises(ValueError, match="No e2e tests match"):  # allow-pytest.raises: empty audits must fail visibly
        trace_tests.audit_matrix(registry, sku_config, model="resnet50")


@pytest.mark.parametrize("case", ["missing", "duplicate", "wrong-sku", "non-llm"])
def test_invalid_focused_selection_is_rejected(case):
    entry = {"name": "test", "model_family": "Llama", "model": "llama", "skus": {"wh_n150": {}}}
    tests = [entry]
    sku = "wh_n150"
    if case == "missing":
        tests = []
    elif case == "duplicate":
        tests.append(entry)
    elif case == "wrong-sku":
        sku = "bh_p150"
    else:
        entry["model_family"] = "Flux"
    with pytest.raises(ValueError):  # allow-pytest.raises: reject stale or invalid registry references
        trace_tests.select_test(tests, "test", sku)


def test_flags_reach_fresh_python_process_and_override_parent():
    # A fresh interpreter observes all flags before it can import TTNN. This also
    # guards against inherited skip-program-cache settings making the audit weaker.
    code = f"import json, os; print(json.dumps({{k: os.environ[k] for k in {list(trace_tests.TRACKER_ENV)!r}}}))"
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"
    env = dict(os.environ, **{key: "999" for key in trace_tests.TRACKER_ENV})
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", trace_tests.tracked_command(command)],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == trace_tests.TRACKER_ENV


@pytest.mark.parametrize("command", ["exit 7", "false\necho hidden-failure", "false | cat\necho hidden-failure"])
def test_command_failure_is_not_hidden(command):
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", trace_tests.tracked_command(command)], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "hidden-failure" not in result.stdout


def test_focused_sweeps_preserve_source_tier_and_sku(registry, sku_config):
    config = trace_tests.load_trace_config()
    trace_tests.sync_sweeps(registry, config, check=True)
    sweeps = load_tests(str(trace_tests.REPO_ROOT / "tests/pipeline_reorg/models_sweep_tests.yaml"))
    traces = [entry for entry in sweeps if "model_trace_tests.py run" in entry["cmd"]]
    assert traces
    covered = set()
    for sweep in traces:
        argv = shlex.split(sweep["cmd"])
        name = argv[argv.index("--test-name") + 1]
        assert argv[argv.index("--sku") + 1] == "{sku}"
        for sku, settings in sweep["skus"].items():
            source = trace_tests.select_test(registry, name, sku)
            assert settings["tier"] == source["skus"][sku]["tier"]
            assert settings["timeout"] == source["skus"][sku]["timeout"] * 2
            for key in ("model", "model_family", "owner_id", "team"):
                assert sweep[key] == source[key]
            assert not build_test_matrix([source], [sku], sku_config)[0][
                "multihost"
            ], "The focused sweep workflow supports single-host SKUs; use the monthly audit for multihost tests"
            covered.add((sweep["model"], sku))
    assert covered == {(model, sku) for model, skus in config["sweeps"].items() for sku in skus}


def test_new_family_and_model_need_only_configuration(registry, sku_config):
    # Simulate a new model with separate e2e entries for two architectures.
    first = copy.deepcopy(next(test for test in registry if test["model"] == "llama3.1-8b"))
    first.update(name="New model's Wormhole tests", model="new-llm", model_family="NewFamily", owner_id="new-owner")
    first["skus"] = {"wh_n150": {"timeout": 12, "tier": 2}}
    second = copy.deepcopy(first)
    second.update(name="New model's Blackhole tests", team="new-team")
    second["skus"] = {"bh_p150": {"timeout": 30, "tier": 3}}
    tests = [first, second]
    original = copy.deepcopy(tests)
    config = {"families": ["NewFamily"], "exclude_models": [], "sweeps": {"new-llm": ["wh_n150", "bh_p150"]}}
    focused = trace_tests.focused_tests(tests, config)
    assert len(focused) == 2
    for source, generated in zip(tests, focused):
        for key in ("model", "model_family", "owner_id", "team"):
            assert generated[key] == source[key]
        assert set(generated["skus"]) == set(source["skus"])
        for sku, settings in generated["skus"].items():
            assert settings == {"tier": source["skus"][sku]["tier"], "timeout": source["skus"][sku]["timeout"] * 2}
            args = shlex.split(generated["cmd"])
            assert trace_tests.select_test(tests, args[args.index("--test-name") + 1], sku, config) == source
    monthly = trace_tests.audit_matrix(tests, sku_config, config=config)
    assert {entry["sku"] for entry in monthly} == {"wh_n150", "bh_p150"}
    assert {entry["tier"] for entry in monthly} == {2, 3}
    assert tests == original


@pytest.mark.parametrize("skus", [[], "wh_n150", ["wh_n150", "wh_n150"], ["unknown"]])
def test_invalid_sweep_config_fails_before_dispatch(registry, skus):
    config = trace_tests.load_trace_config()
    config["sweeps"] = {"llama3.1-8b": skus}
    with pytest.raises(ValueError):  # allow-pytest.raises: invalid hardware selections must not silently disappear
        trace_tests.focused_tests(registry, config)


def test_excluded_model_cannot_enter_focused_sweeps(registry):
    config = trace_tests.load_trace_config()
    config["exclude_models"].append("llama3.1-8b")
    with pytest.raises(ValueError, match="excluded"):  # allow-pytest.raises: enforce the same scope in both schedules
        trace_tests.focused_tests(registry, config)


def test_stale_sweep_metadata_requires_regeneration(tmp_path, monkeypatch, registry):
    target = tmp_path / "sweeps.yaml"
    text = trace_tests.SWEEP_TESTS.read_text()
    target.write_text(text)
    monkeypatch.setattr(trace_tests, "SWEEP_TESTS", target)
    config = trace_tests.load_trace_config()
    source = next(test for test in registry if test["model"] == "llama3.1-8b")
    source["skus"]["wh_n150"]["tier"] = 3
    with pytest.raises(ValueError, match="sync-sweeps"):  # allow-pytest.raises: catch tier drift before dispatch
        trace_tests.sync_sweeps(registry, config, check=True)
    assert target.read_text() == text
    trace_tests.sync_sweeps(registry, config)
    trace_tests.sync_sweeps(registry, config, check=True)
    assert target.read_text().split(trace_tests.GENERATED_START)[0] == text.split(trace_tests.GENERATED_START)[0]


def test_runner_dry_run_keeps_existing_model_command():
    result = subprocess.run(
        [
            sys.executable,
            str(trace_tests.REPO_ROOT / ".github/scripts/utils/model_trace_tests.py"),
            "run",
            "--test-name",
            "Llama 3.1-8B e2e tests",
            "--sku",
            "wh_n150",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "TT_METAL_TRACE_ALLOC_TRACKING=1" in result.stdout
    assert "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0" in result.stdout
    assert "performance-ci-token-matching" in result.stdout
    assert "--use_prefetcher True --repeat_batches 1" in result.stdout


@pytest.mark.skipif(shutil.which("jq") is None, reason="workflow filter requires jq")
def test_sweep_workflow_filter_preserves_quoted_commands(tmp_path, registry, sku_config):
    workflow = yaml.safe_load((trace_tests.REPO_ROOT / ".github/workflows/models-sweep-tests-impl.yaml").read_text())
    step = next(step for step in workflow["jobs"]["load-test-matrix"]["steps"] if step.get("id") == "apply-filters")
    assert step["env"]["MATRIX"] == "${{ steps.build-matrix.outputs.matrix }}"
    tests = load_tests(str(trace_tests.REPO_ROOT / "tests/pipeline_reorg/models_sweep_tests.yaml"))
    matrix = build_test_matrix(tests, collect_skus_from_tests(tests), sku_config)
    output = tmp_path / "github-output"
    env = dict(os.environ, MATRIX=json.dumps(matrix), TIER="1", MODEL="GEMMA", GITHUB_OUTPUT=str(output))
    subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", step["run"]], env=env, check=True, capture_output=True, text=True
    )
    filtered = json.loads(output.read_text().splitlines()[1])
    expected = [entry for entry in matrix if entry["tier"] == 1 and "gemma" in entry["model"]]
    assert filtered == expected
    assert any("'Gemma-4-26B-A4B e2e tests'" in entry["cmd"] for entry in filtered)
