# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Run with pytest --noconftest; synthetic CPU contracts, never native receipts."""

import copy
from dataclasses import FrozenInstanceError

import pytest
import torch

from models.tt_dit.tests.encoders.gemma import gqa_requalification as gqa


@pytest.fixture(scope="module", params=gqa.ALL_CASES)
def prepared(request):
    recipe = gqa.Recipe(request.param)
    return recipe, gqa.prepare_fixture(recipe)


def synthetic_result(recipe, fixture):
    a, b = [sample["reference"].clone() for sample in fixture["cases"]]
    return {
        "recipe": recipe.identity(),
        "eager_outputs": {route: a.clone() for route in ("expanded", "native")},
        "replay_outputs": [
            {"fixture_index": i, "expanded": x.clone(), "native": x.clone()} for i, x in ((0, a), (1, b), (0, a))
        ],
    }


@pytest.mark.parametrize("mode", gqa.MODES)
def test_fp64_fp32_and_synthetic_positive(prepared, mode):
    recipe, fixture = prepared
    recipe = gqa.Recipe(recipe.case, mode)
    gqa.validate_fixture(fixture, recipe)
    report = gqa.verify_outputs(synthetic_result(recipe, fixture), fixture, recipe)
    assert report["status"] == "cpu_numerical_contract_pass" and report["native_qualified"] is False
    assert len(report["metrics"]) == 6
    assert all(len(m["per_head"]) == recipe.heads for m in report["metrics"])


@pytest.mark.parametrize("name", gqa.ORIGINAL_CASES)
def test_original_input_bytes_preserved(name):
    recipe = gqa.Recipe(name)
    for i, seed in enumerate((11, 29)):
        sample = gqa.make_sample(recipe, i)
        gen = torch.Generator().manual_seed(seed)
        for key, heads in (("q", recipe.heads), ("k", recipe.heads // 2), ("v", recipe.heads // 2)):
            assert torch.equal(sample[key], torch.randn((1, heads, 1024, 256), generator=gen).bfloat16())
        assert sample["real_tokens"] == ((257, 613)[i] if name.endswith("leftpad") else 1024)


def test_recipe_capture_does_not_follow_environment(monkeypatch):
    monkeypatch.setenv("GQA_EXP_MODE", "accurate")
    recipe = gqa.Recipe("tp4-causal")
    identity = recipe.identity()
    monkeypatch.setenv("GQA_EXP_MODE", "approximate")
    assert recipe.identity() == identity
    assert gqa.Recipe(recipe.case, "approximate").identity() != identity
    with pytest.raises(FrozenInstanceError):  # allow-pytest.raises: standalone CPU test without device conftest
        recipe.exp_mode = "approximate"


@pytest.mark.parametrize(
    "fault",
    [
        "scale_missing",
        "scale_twice",
        "block_tiled_heads",
        "mask_missing",
        "gain",
        "nan",
        "constant",
        "stale_B",
        "changed_A",
        "missing",
        "duplicate",
        "wrong_mode",
        "hifi4",
        "wrong_geometry",
    ],
)
def test_numerical_adverse_controls(fault):
    recipe = gqa.Recipe("signed-scaled-multi-inf")
    fixture = gqa.prepare_fixture(recipe)
    result = synthetic_result(recipe, fixture)
    sample = fixture["cases"][0]
    if fault in ("scale_missing", "scale_twice", "block_tiled_heads", "mask_missing"):
        q, k, v, mask = [sample[key] for key in ("q", "k", "v", "mask")]
        if fault == "block_tiled_heads":
            # Wrong [KV0,KV1,KV0,KV1] versus contiguous [KV0,KV0,KV1,KV1].
            actual = torch.nn.functional.scaled_dot_product_attention(
                q.float(),
                k.float().repeat(1, 2, 1, 1),
                v.float().repeat(1, 2, 1, 1),
                attn_mask=mask.float(),
                scale=recipe.scale,
            )
        else:
            scale = 1 if fault == "scale_missing" else recipe.scale**2 if fault == "scale_twice" else recipe.scale
            actual = gqa.fp32_oracle(q, k, v, None if fault == "mask_missing" else mask, scale, False)
        # Direct oracle gate must discriminate; replay mismatch is not enough.
        with pytest.raises(ValueError, match="oracle gate"):  # allow-pytest.raises: CPU-only
            gqa.gate(gqa.score(sample, actual))
        return
    if fault == "gain":
        for row in result["replay_outputs"]:
            for route in ("expanded", "native"):
                row[route] *= 1.03
        m = gqa.score(sample, result["replay_outputs"][0]["native"])
        assert m["pcc"] > 0.999 and m["relative_rmse"] > 0.02
    elif fault == "nan":
        result["replay_outputs"][0]["native"][0, 0, 0, 0] = torch.nan
    elif fault == "constant":
        result["replay_outputs"][0]["native"].fill_(1)
    elif fault == "stale_B":
        result["replay_outputs"][1]["native"] = result["replay_outputs"][0]["native"]
    elif fault == "changed_A":
        result["replay_outputs"][2]["native"][0, 0, 0, 0] += 0.001
    elif fault == "missing":
        result["replay_outputs"].pop()
    elif fault == "duplicate":
        result["replay_outputs"].insert(0, result["replay_outputs"][0])
    elif fault == "wrong_mode":
        result["recipe"] = gqa.Recipe(recipe.case, "approximate").identity()
    elif fault == "hifi4":
        result["recipe"]["math_fidelity"] = "HiFi4"
    elif fault == "wrong_geometry":
        result["recipe"]["kv_shape"][1] = 1
    with pytest.raises(ValueError):  # allow-pytest.raises: CPU-only
        gqa.verify_outputs(result, fixture, recipe)


def test_nonfinite_leftpad_rows_still_fail():
    recipe = gqa.Recipe("tp8-leftpad")
    fixture = gqa.prepare_fixture(recipe)
    result = synthetic_result(recipe, fixture)
    result["replay_outputs"][0]["native"][0, 0, 127, 0] = torch.inf
    with pytest.raises(ValueError, match="all-output finiteness"):  # allow-pytest.raises: CPU-only
        gqa.verify_outputs(result, fixture, recipe)


@pytest.mark.parametrize("fault", ["fixture_bytes", "oracle", "geometry"])
def test_fixture_tampering(fault):
    recipe = gqa.Recipe("signed-unit-one-finite")
    fixture = gqa.prepare_fixture(recipe)
    if fault == "fixture_bytes":
        fixture["cases"][0]["q"][0, 0, 0, 0] += 1
    elif fault == "oracle":
        fixture["cases"][0]["reference"].mul_(1.03)
    else:
        fixture["geometry"]["scale"] = 1 / 16
    with pytest.raises(ValueError):  # allow-pytest.raises: CPU-only
        gqa.validate_fixture(fixture, recipe)


def test_one_visible_key_and_chunk_boundaries():
    recipe = gqa.Recipe("one-visible-key")
    fixture = gqa.prepare_fixture(recipe)
    for i, sample in enumerate(fixture["cases"]):
        expected = sample["v"][:, :, 127 + i : 128 + i].repeat_interleave(2, dim=1).expand_as(sample["reference_fp64"])
        assert torch.equal(expected, sample["reference_fp64"])
        assert {"127", "128", "129", "257", "613"} <= sample["oracle_comparison"]["rows"].keys()


def binding_fixture(tmp_path):
    """Fabricated metadata for structural CPU tests, never write a native PASS."""

    def link(name):
        path = tmp_path / name
        path.write_text("CPU synthetic binding control: " + name)
        return {"name": name, "sha256": gqa.digest(path), "bytes": path.stat().st_size}

    recipe = gqa.Recipe("tp8-causal")
    source = {"commit": "a" * 40, "dirty_patch_sha256": None, "source_files": {p: "a" * 64 for p in gqa.SOURCE_PATHS}}
    build = {
        k: "CPU synthetic"
        for k in ("cpp_id", "data_id", "compiler", "python_environment", "tracy_revision", "runtime_firmware_driver")
    }
    build.update(status="attested", binary_hashes={"synthetic.so": "b" * 64}, attestation="attestation.json")
    provenance = {
        "run_id": "cpu-contract",
        "source": source,
        "build": build,
        "execution_configuration": {
            "arithmetic": {
                "math_fidelity": "HiFi2",
                "approximate_math": False,
                "fp32_destination": True,
                "packer_requested": True,
                "packer_effective": True,
                "dtypes": {k: "BF16" for k in ("q", "k", "v", "mask")},
            },
            "chunk_dispatch": {"selected_paths": [gqa.Recipe(name).identity() for name in gqa.ALL_CASES]},
        },
    }
    samples = [{"request_id": "cpu-contract", "index": i, "seed": seed} for i, seed in enumerate((11, 29, 11))]
    entries = [
        {
            "recipe": gqa.Recipe(name).identity(),
            "output": link(name + ".output"),
            "fixture": link(name + ".fixture"),
            "samples": samples,
            "routes": ["expanded", "native"],
        }
        for name in gqa.ALL_CASES
    ]
    execution = {
        "kind": "native_execution",
        "status": "completed",
        "provenance": copy.deepcopy(provenance),
        "raw_log": link("raw.log"),
        "results": entries,
    }
    links = [link("attestation.json"), execution["raw_log"]] + [e[k] for e in entries for k in ("output", "fixture")]
    artifacts = {row["name"]: row for row in links}
    result = {
        "status": "collected",
        "case": recipe.case,
        "provenance": copy.deepcopy(provenance),
        "commit": source["commit"],
        "tracked_diff_sha256": None,
        "source_sha256": source["source_files"].copy(),
        "native_binary_sha256": build["binary_hashes"].copy(),
        "recipe": recipe.identity(),
        "fixture_sha256": entries[0]["fixture"]["sha256"],
        "samples": samples,
        "request_id": "cpu-contract",
    }
    return (
        result,
        provenance,
        execution,
        recipe,
        tmp_path / entries[0]["output"]["name"],
        tmp_path / entries[0]["fixture"]["name"],
        tmp_path,
        artifacts,
    )


def test_synthetic_binding_positive(tmp_path):
    assert gqa.validate_native_binding(*binding_fixture(tmp_path)) is None
    assert not list(tmp_path.glob("qualification-*.json"))


@pytest.mark.parametrize(
    "fault",
    [
        "provenance",
        "commit",
        "source",
        "source_missing",
        "native",
        "unattested",
        "abi_missing",
        "capture_only",
        "synthetic",
        "fixture",
        "mode",
        "precision",
        "missing_case",
        "duplicate_case",
        "missing_sample",
        "duplicate_sample",
        "raw_overwritten",
        "result_overwritten",
        "log_unbound",
    ],
)
def test_binding_adverse_controls(tmp_path, fault):
    args = binding_fixture(tmp_path)
    result, provenance, execution, recipe, output, fixture, root, artifacts = args
    if fault == "provenance":
        result["provenance"]["run_id"] = "other"
    elif fault == "commit":
        result["commit"] = "f" * 40
    elif fault == "source":
        result["source_sha256"][gqa.SOURCE_PATHS[0]] = "f" * 64
    elif fault == "source_missing":
        result["source_sha256"].pop(gqa.SOURCE_PATHS[0])
    elif fault == "native":
        result["native_binary_sha256"]["synthetic.so"] = "f" * 64
    elif fault == "unattested":
        provenance["build"]["status"] = "unattested"
    elif fault == "abi_missing":
        provenance["build"]["runtime_firmware_driver"] = {}
    elif fault == "capture_only":
        execution["status"] = "capture_only"
    elif fault == "synthetic":
        execution["kind"] = "cpu_contract"
    elif fault == "fixture":
        result["fixture_sha256"] = "f" * 64
    elif fault == "mode":
        result["recipe"]["exp_mode"] = "approximate"
    elif fault == "precision":
        result["recipe"]["fp32_dest_acc_en"] = False
    elif fault == "missing_case":
        execution["results"].pop()
    elif fault == "duplicate_case":
        execution["results"][-1] = execution["results"][0]
    elif fault == "missing_sample":
        result["samples"] = result["samples"][:-1]
    elif fault == "duplicate_sample":
        result["samples"] = result["samples"] + result["samples"][:1]
    elif fault == "raw_overwritten":
        (tmp_path / "raw.log").write_text("overwritten")
    elif fault == "result_overwritten":
        output.write_text("overwritten")
    elif fault == "log_unbound":
        artifacts.pop("raw.log")
    with pytest.raises(ValueError):  # allow-pytest.raises: CPU-only
        gqa.validate_native_binding(*args)


def test_stale_success_removed_even_without_native_dependencies(tmp_path):
    report = tmp_path / "qualification-accurate.json"
    report.write_text('{"quality_pass":true}')
    with pytest.raises(ValueError, match="required"):  # allow-pytest.raises: CPU-only
        gqa.verify_suite(tmp_path, tmp_path, "accurate", None, None)
    assert not report.exists()


@pytest.mark.parametrize("text", ['{"mode":1,"mode":2}', '{"value":NaN}', '{"value":Infinity}'])
def test_malformed_json(text):
    with pytest.raises(ValueError):  # allow-pytest.raises: CPU-only
        gqa.strict_json(text)
