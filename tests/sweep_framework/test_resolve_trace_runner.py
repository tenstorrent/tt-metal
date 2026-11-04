# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from model_tracer.resolve_trace_runner import (
    arch_for,
    check_models,
    hf_model_coverage,
    is_hub_id,
    resolve_sku,
    weights_mount,
)

SKUS = {
    "wh_n150": {"runs_on": ["N150", "in-service"]},
    "wh_n300": {"runs_on": ["N300", "in-service"]},
    "bh_p150": {"runs_on": ["arch-blackhole", "P150"], "weights-cache-mode": "cloud-mlperf"},
    "bh_loudbox": {"runs_on": ["arch-blackhole", "BH-LoudBox"], "weights-cache-mode": "cloud-mlperf"},
    "bh_p150_perf": {"runs_on": ["arch-blackhole", "P150"], "weights-cache-mode": "local-disk"},
    "cpu_medium": {"runs_on": ["tt-ubuntu-2204-medium-stable"]},
    "bh_sc4": {"runs_on": ["exabox-multihost-ci-sc4-8kw"]},
    "bh_legacy_alloc": {"runs_on": ["arch-blackhole"], "allocation": {"hosts": 2}},
}


def test_alias_resolves_to_sku():
    assert resolve_sku("N150", SKUS)[0] == "wh_n150"
    assert resolve_sku("BH-LoudBox", SKUS)[0] == "bh_loudbox"


def test_sku_name_accepted_directly():
    assert resolve_sku("bh_p150_perf", SKUS)[0] == "bh_p150_perf"


def test_unknown_label_rejected(expect_error):
    with expect_error(ValueError, "unknown runner-label"):
        resolve_sku("N999", SKUS)


@pytest.mark.parametrize("sku", ["bh_sc4", "bh_legacy_alloc"])
def test_multihost_sku_rejected(sku, expect_error):
    with expect_error(ValueError, "multihost pool"):
        resolve_sku(sku, SKUS)


def test_arch_from_prefix_and_rejects_cpu(expect_error):
    assert arch_for("wh_n150", []) == "wormhole_b0"
    assert arch_for("bh_p150", []) == "blackhole"
    with expect_error(ValueError, "cannot tell the arch"):
        arch_for("cpu_medium", SKUS["cpu_medium"]["runs_on"])


@pytest.mark.parametrize(
    "mode, offline",
    [("", "1"), ("cloud-mlperf", "1"), ("yyz4-mnt-models", "1"), ("local-disk", "0"), ("lfc", "0")],
)
def test_weights_mount_lands_at_mlperf_huggingface(mode, offline):
    mount, off = weights_mount(mode)
    assert off == offline
    assert mount.split(":")[1] in ("/mnt/MLPerf", "/mnt/MLPerf/huggingface")


def test_unknown_mode_rejected(expect_error):
    with expect_error(ValueError, "unknown weights-cache-mode"):
        weights_mount("s3-direct")


def test_coverage_reads_hf_model_from_cmd_and_env(tmp_path):
    (tmp_path / "a.yaml").write_text(
        "- name: llama\n"
        "  cmd: export HF_MODEL=meta-llama/Llama-3.2-1B-Instruct MESH_DEVICE=N150\n"
        "  skus:\n"
        "    wh_n150: {timeout: 5}\n"
        "- name: qwen\n"
        "  env:\n"
        "    HF_MODEL: Qwen/Qwen3-0.6B\n"
        "  skus:\n"
        "    bh_p150: {}\n"
        "- name: no skus\n"
        "  cmd: HF_MODEL=ignored/model\n"
    )
    (tmp_path / "broken.yaml").write_text("- : [\n")
    cov = hf_model_coverage(tmp_path)
    assert cov == {"meta-llama/Llama-3.2-1B-Instruct": {"wh_n150"}, "Qwen/Qwen3-0.6B": {"bh_p150"}}


def test_coverage_reads_multiline_cmd_cleanly(tmp_path):
    (tmp_path / "a.yaml").write_text(
        "- name: a\n" "  cmd: |\n" "    export HF_MODEL=org/a\n" "    export X=1\n" "  skus:\n" "    wh_n150: {}\n"
    )
    assert hf_model_coverage(tmp_path) == {"org/a": {"wh_n150"}}


def test_coverage_credits_per_sku_hf_model_to_that_sku_only(tmp_path):
    (tmp_path / "a.yaml").write_text(
        "- name: perf\n"
        "  cmd: export HF_MODEL={hf_model}\n"
        "  skus:\n"
        "    wh_n150: {hf_model: org/small}\n"
        "    bh_p150: {hf_model: org/big}\n"
    )
    assert hf_model_coverage(tmp_path) == {"org/small": {"wh_n150"}, "org/big": {"bh_p150"}}


def test_coverage_drops_placeholders_and_shell_expansions(tmp_path):
    (tmp_path / "a.yaml").write_text(
        "- name: junk\n"
        "  cmd: |\n"
        "    export HF_MODEL={hf_model}\n"
        "    export HF_MODEL=$(python -c 'print(1)')\n"
        "    export HF_MODEL=hub-id\n"
        "    HF_MODEL=local\n"
        "    HF_MODEL=\n"
        "  env:\n"
        "    HF_MODEL: ''\n"
        "  skus:\n"
        "    wh_n150: {}\n"
    )
    assert hf_model_coverage(tmp_path) == {}


@pytest.mark.parametrize("model", ["meta-llama/Llama-3.2-1B-Instruct", "Qwen/Qwen3-0.6B", "org/a.b_c-d"])
def test_is_hub_id_accepts_org_name(model, tmp_path):
    assert is_hub_id(model, tmp_path)


@pytest.mark.parametrize(
    "model",
    ["/mnt/MLPerf/x/y", "models/demos/gemma4/configs/gemma-4-E2B-it", "org/name/extra", "single", "-org/name", ""],
)
def test_is_hub_id_rejects_paths_and_junk(model, tmp_path):
    assert not is_hub_id(model, tmp_path)


def test_is_hub_id_rejects_existing_repo_dir(tmp_path):
    (tmp_path / "configs" / "model").mkdir(parents=True)
    assert not is_hub_id("configs/model", tmp_path)
    assert is_hub_id("configs/other", tmp_path)


COVERAGE = {"meta-llama/Llama-3.2-1B-Instruct": {"wh_n150"}, "Qwen/Qwen3-0.6B": {"bh_p150"}}


def test_covered_model_passes():
    assert check_models(["meta-llama/Llama-3.2-1B-Instruct"], "wh_n150", SKUS, COVERAGE) == []


def test_same_mount_other_sku_warns():
    warnings = check_models(["Qwen/Qwen3-0.6B"], "bh_loudbox", SKUS, COVERAGE)
    assert len(warnings) == 1
    assert "bh_p150" in warnings[0]


def test_uncovered_model_on_offline_cache_warns_and_names_alternatives():
    warnings = check_models(["meta-llama/Llama-3.2-1B-Instruct"], "bh_loudbox", SKUS, COVERAGE)
    assert len(warnings) == 1
    assert "wh_n150" in warnings[0] and "bh_p150_perf" in warnings[0]


def test_online_cache_and_local_paths_skip_the_check(tmp_path):
    assert check_models(["anything/at-all"], "bh_p150_perf", SKUS, COVERAGE) == []
    assert check_models(["/mnt/MLPerf/tt_dnn-models/x", ""], "bh_loudbox", SKUS, COVERAGE) == []
    assert check_models(["models/demos/gemma4/configs/gemma-4-E2B-it"], "bh_loudbox", SKUS, COVERAGE) == []
    (tmp_path / "configs" / "model").mkdir(parents=True)
    assert check_models(["configs/model"], "bh_loudbox", SKUS, COVERAGE, repo_root=tmp_path) == []
