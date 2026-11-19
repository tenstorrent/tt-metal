# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Collect raw full-model trials and render traced teacher-forcing Pareto plots."""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import pickle
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "doc/datatype_sweep"


def score_evidence(references):
    """Keep sampling units separate; repeated prefill/decode positions are correlated."""
    rows = []
    for reference in references:
        for mode in ("prefill", "teacher"):
            for index, score in enumerate(reference.get(mode, [])):
                n = score.get("total")
                p = score.get("top1")
                se = math.sqrt(p * (1 - p) / n) if n and p is not None else None
                rows.append(
                    dict(
                        reference=reference["reference"],
                        reference_sha256=reference.get("sha256"),
                        mode=mode,
                        row=index,
                        positions=n,
                        top1_matches=score.get("matches_top1"),
                        top1=p,
                        top1_binomial_standard_error=se,
                        top1_binomial_margin_95=1.96 * se if se is not None else None,
                        interval_note="Normal binomial diagnostic; ignores within-text correlation; not an independent-repeat confidence interval",
                    )
                )
    return rows


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def trial_identity(raw, path):
    """Use captured runtime facts; never relabel a measurement from a mutable config."""
    runtime = raw.get("runtime_policy_after", raw.get("runtime_policy", {}))
    config = runtime.get("selected_policy", {})
    errors = []
    if not config or config.get("config_id") != raw["config_id"] or runtime.get("config_id") != raw["config_id"]:
        errors.append("missing_or_mismatched_runtime_policy")
    layers = runtime.get("layers", [])
    if sorted(row.get("layer_index", -1) for row in layers) != list(range(40)):
        errors.append("runtime_summary_does_not_cover_all40_layers")
    dtypes = {"bf16": "DataType.BFLOAT16", "bfp8": "DataType.BFLOAT8_B", "bfp4": "DataType.BFLOAT4_B"}
    for layer in layers:
        weights = dict(config.get("weight_groups", {}))
        computes = dict(config.get("compute_fidelities", {}))
        for exception in config.get("layer_exceptions", []):
            if layer["layer_index"] in exception["layers"]:
                weights.update(exception.get("weight_groups", {}))
                computes.update(exception.get("compute_fidelities", {}))
        required = {"expert_gu", "expert_down", "expert_ep", "shared_packed", "shared_down"}
        if layer.get("kind") == "full_attention":
            required |= {"attention_input", "attention_output"}
        for group in required:
            actual = layer.get("compute", {}).get(group, {})
            expected = dict(computes.get(group, {}))
            if "math_fidelity" in expected:
                expected["math_fidelity"] = "MathFidelity." + expected["math_fidelity"]
            if (
                layer.get("weights", {}).get(group, {}).get("dtype") != dtypes.get(weights.get(group))
                or actual != expected
            ):
                errors.append(f"runtime_layer_{layer['layer_index']}_{group}_mismatch")
        if layer.get("kv_dtype") != dtypes.get(config.get("kv_cache_dtype")) or layer.get("ccl_dtype") != config.get(
            "ccl_dtype"
        ):
            errors.append(f"runtime_layer_{layer['layer_index']}_cache_or_ccl_mismatch")
        scalar = layer.get("shared_scalar", {})
        if scalar.get("dtype") != config.get("shared_scalar_dtype") or scalar.get("fidelity") != config.get(
            "shared_scalar_fidelity"
        ):
            errors.append(f"runtime_layer_{layer['layer_index']}_shared_scalar_mismatch")
    if config:
        expected_head = dict(config["compute_fidelities"]["lm_head"])
        expected_head["math_fidelity"] = "MathFidelity." + expected_head["math_fidelity"]
        if (
            runtime.get("lm_head", {}).get("dtype") != dtypes[config["weight_groups"]["lm_head"]]
            or runtime.get("lm_head", {}).get("compute") != expected_head
        ):
            errors.append("runtime_head_mismatch")
        fixed = {
            k: v for k, v in config.items() if k not in ("weight_groups", "compute_fidelities", "layer_exceptions")
        }
        if runtime.get("fixed_boundaries") != fixed:
            errors.append("runtime_fixed_boundaries_mismatch")
        boundaries = runtime.get("observed_boundaries", {})
        sites = {
            "embedding.output": "activation_dtype",
            "embedding.residual": "residual_dtype",
            "terminal.input": "residual_dtype",
            "terminal.normalized_activation": "activation_dtype",
            "terminal.dram_logits": "logits_dtype",
            "public_logits.device_input": "logits_dtype",
            "bounded_prefill.decoder_input": "activation_dtype",
            "bounded_prefill.residual_read": "residual_dtype",
            "bounded_prefill.residual_write_source": "residual_dtype",
            "bounded_prefill.residual_write_buffer": "residual_dtype",
            "sampler.logits_buffer": "sampling_dtype",
            "sampler.model_logits_compatibility": "logits_dtype",
            "sampler.zero_logits_row": "sampling_dtype",
            "sampler.forward_input": "sampling_dtype",
        }
        sites.update(
            {f"decode.layer{layer}.{side}": "residual_dtype" for layer in range(40) for side in ("input", "output")}
        )
        for site, field in sites.items():
            if boundaries.get(site) != dict(field=field, selected=config[field], actual_dtype=dtypes[config[field]]):
                errors.append(f"runtime_boundary_{site}_missing_or_mismatched")
        for field in ("activation_dtype", "residual_dtype", "logits_dtype", "sampling_dtype"):
            if runtime.get("boundary_policy_support", {}).get(field) != dict(
                selected=config[field], supported=["bf16"], mode="fixed_model_contract"
            ):
                errors.append(f"runtime_boundary_{field}_support_mismatch")
        if runtime.get("embedding_dtype") != dtypes.get(config.get("embedding_dtype")):
            errors.append("runtime_embedding_mismatch")
        terminal = runtime.get("terminal_norm", {})
        if (
            terminal.get("dtype") != "DataType.FLOAT32"
            or terminal.get("compute", {}).get("math_fidelity") != "MathFidelity.HiFi2"
        ):
            errors.append("runtime_terminal_norm_mismatch")
        caches = runtime.get("cache", {}).get("layers", [])
        expected_cache_dtype = {"bf16": "bfloat16", "bfp8": "bfloat8_b"}.get(config.get("kv_cache_dtype"))
        if len(caches) != 10 or any(
            cache.get("dtype") != expected_cache_dtype or cache.get("layout") != config.get("kv_layout")
            for cache in caches
        ):
            errors.append("runtime_cache_metadata_mismatch")
    provenance_path = next(
        (p for p in (path.with_suffix(".provenance.json"), Path(str(path) + ".provenance.json")) if p.exists()), None
    )
    provenance = json.loads(provenance_path.read_text()) if provenance_path else {}
    source_hashes = {k: v for k, v in provenance.get("source_sha256", {}).items() if "/tt/" in k and k.endswith(".py")}
    harness_hashes = {
        k: v for k, v in provenance.get("source_sha256", {}).items() if "/tests/" in k and k.endswith(".py")
    }
    if not source_hashes or provenance.get("commit") != raw.get("commit"):
        errors.append("missing_or_mismatched_trial_source_provenance")
    return config, dict(
        valid=not errors,
        errors=errors,
        policy_sha256=digest(config),
        source_sha256=source_hashes,
        source_digest=digest(source_hashes),
        measurement_harness_sha256=harness_hashes,
        measurement_harness_digest=digest(harness_hashes),
        provenance_artifact=str(provenance_path) if provenance_path else None,
    )


def p1_binding(path, data, cid, row, runner):
    """Bind a runner result to its CPU score cache and the measured policy/source."""
    import torch

    model_dir = ROOT.parents[1]
    details = {"valid": False, "errors": []}
    try:
        assert row["measurement_identity"]["valid"], "trial identity is invalid"
        expected_sources = row["measurement_identity"]["source_sha256"]
        sha256 = lambda item: hashlib.sha256(Path(item).read_bytes()).hexdigest()

        def validate_python(identity):
            assert identity.get("payload"), "Missing P1 Python source identity"
            assert identity["digest"] == digest(identity["payload"]), "Invalid P1 Python identity digest"
            assert identity["source_digest"] == digest(
                {key: identity["payload"][key] for key in ("source_files", "missing_optional")}
            ), "Invalid P1 source digest"
            assert (
                identity["source_digest"] == row["python_runtime_identity"]["source_digest"]
            ), "P1 Python/readiness/harness sources differ from measured runtime"

        archive = path.parent / path.stem
        sidecar_proof = None
        if not (archive / "manifest.json").is_file():
            sidecar_proof = json.loads(path.with_suffix(".runtime_identity.json").read_text())
            assert sidecar_proof["result_sha256"] == sha256(path), "P1 native identity sidecar is stale"
            archive = Path(sidecar_proof["archive"])
            assert sha256(Path(str(archive) + ".json")) == sha256(path), "Canonical P1 differs from archived result"
        manifest = json.loads((archive / "manifest.json").read_text())
        process_path = archive / "process.json"
        assert sha256(process_path) == manifest["process_evidence_sha256"], "P1 process evidence hash differs"
        process = json.loads(process_path.read_text())
        assert process["status"] == "complete" and process["identity_unchanged"], "P1 process did not finish stably"
        native_identity = manifest["runtime_identity"]
        assert native_identity.get("digest") == digest(native_identity.get("payload")), "Invalid P1 native digest"
        assert (
            native_identity == process["runtime_identity_before"] == process["runtime_identity_after"]
        ), "Actual P1 process native identity changed or differs from archive"
        assert native_identity == row["native_runtime_identity"], "P1 native source/build differs from measured runtime"
        sources = process["source_identity_before"]
        validate_python(sources)
        assert sources == process["source_identity_after"] == manifest["source_identity"], "P1 Python sources changed"
        validate_python(process["run_input_identity"])
        run_inputs = {item["path"]: item["sha256"] for item in process["run_input_identity"]["payload"]["input_files"]}
        if sidecar_proof is not None:
            assert Path(sidecar_proof["process_evidence"]).resolve() == process_path.resolve()
            assert sidecar_proof["process_evidence_sha256"] == sha256(process_path)
            assert sidecar_proof["runtime_identity"] == native_identity
            assert sidecar_proof["score_bindings"] == manifest["score_bindings"]
        details["native_runtime_digest"] = native_identity["digest"]
        details["source_digest"] = sources["source_digest"]
        details["process_evidence"] = str(process_path)
        details["process_evidence_sha256"] = sha256(process_path)
        source_root = archive / "runtime"
        source_paths = sorted((source_root / "tt").rglob("*.py"))
        actual_sources = {
            str((model_dir / source.relative_to(source_root)).relative_to(Path.cwd())): sha256(source)
            for source in source_paths
        }
        assert expected_sources == actual_sources, "Measured runtime source differs from archived P1 source"
        assert manifest["source_sha256"] == actual_sources, "Archived source manifest differs from bytes"
        process_sources = {item["path"]: item["sha256"] for item in sources["payload"]["source_files"]}
        assert all(
            process_sources[str(Path(name).resolve())] == value for name, value in actual_sources.items()
        ), "Archived model bytes differ from actual P1 process source snapshot"

        def score_code_sha(score_id, kwargs):
            h = hashlib.sha256()
            for source in source_paths:
                h.update(str(source.relative_to(source_root)).encode() + b"\0" + source.read_bytes() + b"\0")
            for key, relative in sorted(manifest["input_files"][score_id].items()):
                assert isinstance(kwargs.get(key), str), "archived input is not a file-backed build kwarg"
                h.update(key.encode() + b"\0" + (archive / relative).read_bytes() + b"\0")
            return h.hexdigest()[:16]

        blobs, contexts, binding_hashes = {}, {}, {}
        for score_id in {cid, data["reference"]}:
            score_path = archive / "scores" / f"{score_id}.pt"
            assert score_path.is_file(), f"Missing archived score cache for {score_id}"
            score_sha = sha256(score_path)
            proof = manifest["score_bindings"][score_id]
            process_proof = process["score_bindings"][score_id]
            assert all(proof[key] == value for key, value in process_proof.items()), "Process score record differs"
            assert (
                score_sha == manifest["score_sha256"][score_id] == proof["score_sha256"]
            ), "Archived score hash differs"
            binding_path = archive / proof["archived_binding"]
            assert sha256(binding_path) == proof["binding_sha256"], "Immutable score binding hash differs"
            binding = json.loads(binding_path.read_text())
            context = binding["context"]
            assert binding["score_sha256"] == score_sha and binding["context_digest"] == digest(context)
            assert context["schema_version"] == 1 and context["candidate_id"] == score_id
            assert context["runtime_identity"] == native_identity, "Score was created with another native runtime"
            assert context["source_identity"] == sources, "Score was created with different Python sources"
            assert context["settings"] == process["settings"], "Cached P1 scoring settings differ"
            for key, value in data["thresholds"].items():
                if key in context["settings"]:
                    assert value == context["settings"][key], f"P1 reported threshold differs: {key}"
            assert context["text_sha"] == data["text"]["sha"], "Bound P1 text hash differs"
            if context["settings"].get("text_file"):
                text_path = str(Path(context["settings"]["text_file"]).resolve())
                assert context["text_file_sha256"] == run_inputs[text_path], "Bound reasoning text bytes differ"
            for item in context["settings_input_files"].values():
                assert run_inputs[item["path"]] == item["sha256"], "Scoring settings input changed"
            assert set(context["input_files"]) == set(manifest["input_files"][score_id]), "Bound config inputs differ"
            for key, item in context["input_files"].items():
                archived_input = archive / manifest["input_files"][score_id][key]
                assert (
                    sha256(archived_input) == item["sha256"] == run_inputs[item["path"]]
                ), f"Score config bytes differ for {score_id}/{key}"
            blob = torch.load(score_path, map_location="cpu", weights_only=True)
            kwargs = blob["build_kwargs"]
            assert context["build_kwargs"] == kwargs, "Immutable binding build kwargs differ from score"
            assert blob["text_sha"] == data["text"]["sha"], f"Text hash mismatch for {score_id}"
            assert blob["code_sha"] == score_code_sha(score_id, kwargs), f"Code/config hash mismatch for {score_id}"
            blobs[score_id], contexts[score_id], binding_hashes[score_id] = blob, context, sha256(binding_path)
            details[score_id] = dict(
                score_artifact=str(score_path),
                code_sha=blob["code_sha"],
                build_kwargs=kwargs,
                score_sha256=score_sha,
                binding_artifact=str(binding_path),
                binding_sha256=sha256(binding_path),
            )
        reference_id = data["reference"]
        assert contexts[reference_id]["scored_against"] is None, "Reference binding has a candidate role"
        if cid != reference_id:
            assert contexts[cid]["scored_against"] == dict(
                id=reference_id,
                score_sha256=manifest["score_sha256"][reference_id],
                binding_sha256=binding_hashes[reference_id],
                build_kwargs=blobs[reference_id]["build_kwargs"],
                input_files=contexts[reference_id]["input_files"],
            ), "Candidate was scored against different reference scores/config bytes"
        blob, reference = blobs[cid], blobs[data["reference"]]
        config_value = blob["build_kwargs"].get("precision_config")
        if isinstance(config_value, dict):
            p1_config = config_value
        else:
            config_path = archive / manifest["input_files"][cid]["precision_config"]
            assert config_path, "P1 precision config must be explicit and available"
            p1_config = json.loads(config_path.read_text())
        assert p1_config == row["dtype_policy"], "P1 config differs from measured runtime policy"
        # Context allocation is intentionally larger for long P1 texts; all other construction switches agree.
        normalized = lambda kwargs: {k: v for k, v in kwargs.items() if k not in ("precision_config", "max_seq_len")}
        assert normalized(blob["build_kwargs"]) == normalized(row["build_kwargs"]), "P1 construction switches differ"
        assert reference["scored_against"] is None, "reference score cache has the wrong role"
        if cid != data["reference"]:
            assert len(blob["texts"]) == len(reference["texts"]), "Candidate score text coverage differs"
            assert all(
                candidate_row["nll"].shape == reference_row["nll"].shape
                and torch.equal(candidate_row["target"], reference_row["target"])
                for candidate_row, reference_row in zip(blob["texts"], reference["texts"])
            ), "Candidate score positions/targets differ from reference"
            assert blob["scored_against"] == {
                "id": data["reference"],
                "kwargs": reference["build_kwargs"],
            }, "candidate reference identity differs"
            stats = runner._compact(runner.compare(reference["texts"], blob["texts"]))
            recorded = data["candidates"][cid]
            assert all(
                recorded.get(k) == value for k, value in stats.items()
            ), "cached scores do not reproduce archived paired statistics"
            thresholds = data["thresholds"]
            within_noise, _ = runner.within_noise(
                runner.compare(reference["texts"], blob["texts"]),
                nll_floor=thresholds["nll_floor"],
                kl_max_x1000=thresholds["kl_max_x1000"],
            )
            assert within_noise == recorded.get("within_noise"), "cached scores disagree with archived gate"
        assert (
            sum(int(r["nll"].numel()) for r in reference["texts"]) == data["text"]["positions"]
        ), "reference score count differs"
        details["valid"] = True
    except (AssertionError, KeyError, ValueError, OSError, RuntimeError, pickle.UnpicklingError) as exc:
        details["errors"].append(str(exc))
    return details


def selection_evidence(results, selected_id, baseline_id="baseline_bfp8"):
    """Aggregate independent trials, enforce serial baseline noise, prefer safer ties."""
    baselines = [
        r
        for r in results
        if r["config_id"] == baseline_id
        and r["traced"]
        and r["full_model_accuracy_pass"]
        and r["measurement_identity"]["valid"]
        and r["reference_identity_valid"]
        and r["native_runtime_identity_valid"]
        and r["python_runtime_identity_valid"]
        and r["finalized_successfully"]
    ]
    assert baselines or not selected_id, "Selection needs a valid full-model baseline"
    if not baselines:
        for row in results:
            row["measurement_matches_baseline"] = False
            row["eligible_for_selection"] = False
            row["eligibility_reasons"].append("valid_baseline_missing")
        return {"status": "baseline_missing", "baseline_id": baseline_id}
    signature = lambda r: (
        r["hardware"],
        tuple(r["mesh"]),
        r["measurement_regime"],
        digest(r["environment"]),
        digest(r.get("reference_sha256", {})),
        r.get("native_runtime_identity", {}).get("digest"),
        r.get("python_runtime_identity", {}).get("source_digest"),
        r["measurement_identity"]["source_digest"],
        r["measurement_identity"].get("measurement_harness_digest"),
        digest({k: v for k, v in r["build_kwargs"].items() if k != "precision_config"}),
    )
    control = baselines[-1]
    baselines = [r for r in baselines if signature(r) == signature(control)]
    assert control["measurement_regime"] == "serial" or not selected_id, "Selection requires serial measurements"
    independent = len({str(Path(r["raw_artifact"]).resolve()) for r in baselines})
    # Conservatively include within-trial spread as well as between-process mean spread.
    speeds = [speed for r in baselines for speed in r["repeat_speeds"]]
    noise = max(speeds) - min(speeds)
    groups = {}
    for row in results:
        comparable = signature(row) == signature(control) and row["measurement_identity"]["valid"]
        row["measurement_matches_baseline"] = comparable
        if not comparable:
            row["eligible_for_selection"] = False
            row["eligibility_reasons"].append("measurement_or_source_differs_from_baseline")
        groups.setdefault(row["config_id"], []).append(row)
    policies = {}
    for cid, rows in groups.items():
        assert (
            len({r["measurement_identity"]["policy_sha256"] for r in rows if r["measurement_identity"]["valid"]}) <= 1
        ), f"Config ID {cid} labels different measured policies"
        trials = [r for r in rows if r["measurement_matches_baseline"]]
        if trials and all(r["eligible_for_selection"] for r in trials):
            policies[cid] = dict(
                speed=statistics.mean(r["teacher_forcing_decode_t_s_u"] for r in trials), trial_count=len(trials)
            )
    baseline_config = control["dtype_policy"]

    def tie_risk(cid):
        config = groups[cid][0]["dtype_policy"]
        bfp4, lower_fidelity, changed = 0, 0, 0
        fidelity_rank = {"LoFi": 0, "HiFi2": 2, "HiFi4": 4}

        def resolved(policy, layer):
            weights, computes = dict(policy["weight_groups"]), dict(policy["compute_fidelities"])
            for exception in policy["layer_exceptions"]:
                if layer in exception["layers"]:
                    weights.update(exception.get("weight_groups", {}))
                    computes.update(exception.get("compute_fidelities", {}))
            return weights, computes

        for layer in range(40):
            weights, computes = resolved(config, layer)
            baseline_weights, baseline_computes = resolved(baseline_config, layer)
            active = {"expert_gu", "expert_down", "expert_ep", "shared_packed", "shared_down"}
            if control["layer_kinds"][str(layer)] == "full_attention":
                active |= {"attention_input", "attention_output"}
            bfp4 += sum(weights[g] == "bfp4" for g in active)
            lower_fidelity += sum(
                fidelity_rank[computes[g]["math_fidelity"]] < fidelity_rank[baseline_computes[g]["math_fidelity"]]
                for g in active
            )
            changed += sum(weights[g] != baseline_weights[g] for g in active)
            changed += sum(computes[g] != baseline_computes[g] for g in active)
        # The terminal head is one model-wide operation, not forty layer groups.
        bfp4 += config["weight_groups"]["lm_head"] == "bfp4"
        lower_fidelity += (
            fidelity_rank[config["compute_fidelities"]["lm_head"]["math_fidelity"]]
            < fidelity_rank[baseline_config["compute_fidelities"]["lm_head"]["math_fidelity"]]
        )
        changed += config["weight_groups"]["lm_head"] != baseline_config["weight_groups"]["lm_head"]
        changed += config["compute_fidelities"]["lm_head"] != baseline_config["compute_fidelities"]["lm_head"]
        transfers = sum(config[k] != baseline_config[k] for k in ("kv_cache_dtype", "ccl_dtype"))
        return (cid != baseline_id, bfp4, transfers, lower_fidelity, changed, len(config["layer_exceptions"]), cid)

    fastest = max((p["speed"] for p in policies.values()), default=None)
    tied = [cid for cid, p in policies.items() if fastest - p["speed"] <= noise]
    recommended = min(tied, key=tie_risk) if tied else None
    if selected_id:
        assert independent >= 2, "Selection needs at least two independent serial baseline artifacts"
        assert selected_id in policies, "Selected config has missing/failed/incomparable trials or P1 evidence"
        assert (
            selected_id == recommended
        ), f"Select {recommended}: fastest passing policy with noise ties resolved by safer/simpler policy"
    return dict(
        status="qualified" if independent >= 2 else "needs_independent_baseline_repeat",
        baseline_id=baseline_id,
        independent_baseline_trials=independent,
        baseline_artifacts=[r["raw_artifact"] for r in baselines],
        noise_t_s_u=noise,
        noise_definition="range of all repeated serial baseline measurements; includes within-run and between-run spread",
        policy_performance=policies,
        fastest_eligible_t_s_u=fastest,
        tied_policy_ids=tied,
        recommended=recommended,
        tie_rule="baseline first; then fewer BFP4 layer/groups, transfer changes, reduced-fidelity layer/groups, changed layer/groups, and exceptions; config ID breaks otherwise equal ties",
    )


def p1_evidence(paths, rows):
    """Interpret real runner records with the installed gate's schema/waiver logic."""
    records = {row["raw_artifact"]: [] for row in rows}
    if not paths:
        return records
    package = Path(os.environ["TT_MODEL_BRINGUP_ROOT"])
    spec = importlib.util.spec_from_file_location(
        "datatype_eval_readiness_gate", package / "scripts/check_eval_readiness.py"
    )
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    runner_spec = importlib.util.spec_from_file_location(
        "datatype_precision_reasoning", package / "runtime/readiness_check/run_precision_reasoning.py"
    )
    runner = importlib.util.module_from_spec(runner_spec)
    runner_spec.loader.exec_module(runner)
    model_dir = ROOT.parents[1]
    limitations = gate.load_limitations(model_dir)
    for path in sorted(set(paths), key=lambda item: json.loads(item.read_text()).get("created_utc", "")):
        raw = json.loads(path.read_text())
        checks = gate.validate_result(raw, "precision_reasoning", path)
        by_id = {row["id"]: row for row in checks}
        data = raw.get("data", {})
        text = data.get("text", {})
        positions = text.get("positions", 0)
        positions_ok = by_id.get("positions_scored", {}).get("status") == "pass" and positions >= max(
            1000, data.get("thresholds", {}).get("min_positions", 1000)
        )
        for row in rows:
            cid, config = row["config_id"], row["dtype_policy"]
            selected_here = data.get("selected") == cid
            stats = data.get("candidates", {}).get(cid, {})
            if not selected_here and "within_noise" not in stats:
                continue  # --only writes explicit not-scored rows for other cached candidates.
            selected_check = by_id.get("selected_within_noise", {})
            within_noise = (
                selected_check.get("status") == "pass" if selected_here else stats.get("within_noise") is True
            )
            waiver = None
            if selected_here and selected_check.get("status") == "fail":
                has_bfp4 = "bfp4" in config.get("weight_groups", {}).values() or any(
                    "bfp4" in str(exception) for exception in config.get("layer_exceptions", [])
                )
                for item in limitations:
                    reason = str(item.get("reason", "")).lower()
                    evidence = item.get("evidence", [])
                    resolved = [
                        next((base / value for base in (model_dir, Path.cwd()) if (base / value).exists()), None)
                        for value in evidence
                        if isinstance(value, str)
                    ]
                    capacity = any(value and ("context" in value.name or "memory" in value.name) for value in resolved)
                    if (
                        has_bfp4
                        and "required to fit" in reason
                        and resolved
                        and all(resolved)
                        and capacity
                        and gate.waiver_reason(model_dir, "precision_reasoning", selected_check, [item])
                    ):
                        waiver = dict(artifact=str(model_dir / "doc/eval_readiness/limitations.json"), entry=item)
                        break
            critical_ok = all(
                check["status"] == "pass" or (check["id"] == "selected_within_noise" and waiver is not None)
                for check in checks
                if check["severity"] == "critical"
            )
            binding = p1_binding(path, data, cid, row, runner)
            gate_pass = (
                binding["valid"]
                and selected_here
                and positions_ok
                and critical_ok
                and (within_noise or waiver is not None)
            )
            status = (
                "pass"
                if positions_ok and within_noise
                else "required_to_fit_limitation"
                if positions_ok and waiver
                else "fail"
            )
            if not binding["valid"]:
                status = "identity_unverified"
            records[row["raw_artifact"]].append(
                dict(
                    artifact=str(path),
                    created_utc=raw.get("created_utc"),
                    command=raw.get("command"),
                    status=status,
                    positions=positions,
                    text_sha=text.get("sha"),
                    reference=data.get("reference"),
                    selected_in_artifact=data.get("selected"),
                    candidate_stats=stats,
                    selected_checks=checks if selected_here else None,
                    runner_gate_pass=gate_pass,
                    limitation=waiver,
                    identity_binding=binding,
                )
            )
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected")
    parser.add_argument(
        "--baseline-id", default="baseline_bfp8", help="Measured BFP8 control policy ID for this runtime version"
    )
    parser.add_argument(
        "--p1-artifact",
        action="append",
        type=Path,
        default=[],
        help="Repeat for archived real P1 runner results, including different <=5-candidate subsets; canonical result is included automatically",
    )
    parser.add_argument("artifacts", nargs="+", type=Path)
    args = parser.parse_args()
    results = []
    assert len({path.resolve() for path in args.artifacts}) == len(
        args.artifacts
    ), "Duplicate trial artifact paths are not independent runs"
    for path in args.artifacts:
        raw = json.loads(path.read_text())
        kwargs = raw["build_kwargs"]
        config_path = Path(kwargs["precision_config"])
        config, identity = trial_identity(raw, path)
        refs = raw.get("references", [])
        scores = [s for r in refs for mode in ("prefill", "teacher") for s in r.get(mode, [])]
        perf = raw.get("teacher_performance", [])
        speed = [p["tokens_per_second_per_user"] for p in perf]
        row = dict(
            config_id=raw["config_id"],
            precision_config=str(config_path),
            dtype_policy=config,
            compute_fidelity_policy=config.get("compute_fidelities", {}),
            layer_exceptions=config.get("layer_exceptions", []),
            layer_kinds={
                str(layer["layer_index"]): layer["kind"] for layer in raw.get("runtime_policy", {}).get("layers", [])
            },
            measurement_identity=identity,
            build_kwargs=kwargs,
            environment=raw.get("environment", {}),
            top1=min((s["top1"] for s in scores), default=None),
            top5=min((s["top5"] for s in scores), default=None),
            top100=min((s["top100"] for s in scores), default=None),
            accuracy_aggregation="minimum across every prefill and teacher-forced reference",
            references=refs,
            reference_sha256=raw.get("reference_sha256", {}),
            reference_entry_counts=raw.get("reference_entry_counts", {}),
            python_runtime_identity=raw.get("python_runtime_identity", {}),
            python_runtime_identity_unchanged=bool(raw.get("python_runtime_identity"))
            and raw.get("python_runtime_identity") == raw.get("python_runtime_identity_after"),
            native_runtime_identity=raw.get("native_runtime_identity", {}),
            native_runtime_identity_unchanged=raw.get("native_runtime_identity_after")
            == raw.get("native_runtime_identity")
            and bool(raw.get("native_runtime_identity")),
            accuracy_sample_evidence=score_evidence(refs),
            ttft_ms=statistics.mean([p["ttft_ms"] for p in perf]) if perf else None,
            ttft_scope="warmed161-token prefill to resident logits; sampling/readback excluded",
            callback_ttft_ms=next(
                (s["ttft_ms"] for r in refs[:1] for s in r.get("teacher", []) if "ttft_ms" in s), None
            ),
            callback_ttft_scope="readiness161-token AIME prefill through first sampled token readback",
            teacher_forcing_decode_t_s_u=statistics.mean(speed) if speed else None,
            repeat_speeds=speed,
            repeat_spread=max(speed) - min(speed) if speed else None,
            traced=bool(perf)
            and all(
                p["direct_model_trace_replays"] == p["steps"] == 99
                and p["loop_counter_delta"]["captures"] == 0
                and p["loop_counter_delta"]["sampling_replays"] == 0
                and p["loop_counter_delta"]["readbacks"] == 0
                and math.isfinite(p["tokens_per_second_per_user"])
                and p["tokens_per_second_per_user"] > 0
                for p in perf
            ),
            measurement_regime=raw["measurement_regime"],
            performance_workload="B1,161-token AIME24 chat prompt,99 HF forced-token uploads to traced resident logits; no sampling/readback",
            command=raw["command"],
            hardware=raw["hardware"],
            mesh=raw["mesh"],
            commit=raw["commit"],
            runtime_policy_evidence=str(path)
            + ("#/runtime_policy_after" if "runtime_policy_after" in raw else "#/runtime_policy"),
            raw_artifact=str(path),
            status=raw["status"],
            device_closed=raw.get("device_closed", False),
            finalized_successfully=raw["status"] == "accuracy_pass" and raw.get("device_closed") is True,
            full_model_accuracy_pass=raw.get("passed", False),
            selected=raw["config_id"] == args.selected,
        )
        results.append(row)
    p1_paths = list(args.p1_artifact)
    canonical_p1 = ROOT.parents[1] / "doc/eval_readiness/precision_reasoning.json"
    if canonical_p1.exists():
        p1_paths.append(canonical_p1)
    evidence = p1_evidence(p1_paths, results)
    baseline_rows = [row for row in results if row["config_id"] == args.baseline_id and row["reference_sha256"]]
    coverage = lambda row: {
        (str(score["reference"]), score["reference_sha256"], score["mode"], score["row"], score["positions"])
        for score in row["accuracy_sample_evidence"]
    }
    baseline_coverage = max((coverage(row) for row in baseline_rows), key=len, default=set())
    for row in results:
        proofs = evidence[row["raw_artifact"]]
        latest = proofs[-1] if proofs else {}
        row["p1_status"] = latest.get("status", "not_run")
        row["p1_evidence"] = proofs
        row["p1_positions"] = latest.get("positions")
        row["p1_runner_gate_pass"] = latest.get("runner_gate_pass", False)
        row["p1_pass"] = row["p1_status"] in ("pass", "required_to_fit_limitation")
        row["reference_coverage_matches_baseline"] = not baseline_coverage or baseline_coverage.issubset(coverage(row))
        row["reference_identity_valid"] = (
            bool(row["references"])
            and all(
                reference.get("sha256")
                and reference["sha256"] == row["reference_sha256"].get(reference["reference"])
                and row["reference_entry_counts"].get(reference["reference"], 0) > 0
                and all(
                    len(reference.get(mode, [])) == row["reference_entry_counts"][reference["reference"]]
                    for mode in ("prefill", "teacher")
                )
                for reference in row["references"]
            )
            and set(row["reference_sha256"]) == {r["reference"] for r in row["references"]}
        )
        native = row["native_runtime_identity"]
        row["native_runtime_identity_valid"] = (
            bool(native.get("payload"))
            and native.get("digest") == digest(native.get("payload"))
            and row["native_runtime_identity_unchanged"]
        )
        python_identity = row["python_runtime_identity"]
        row["python_runtime_identity_valid"] = (
            bool(python_identity.get("payload"))
            and python_identity.get("digest") == digest(python_identity["payload"])
            and python_identity.get("source_digest")
            == digest({key: python_identity["payload"].get(key, []) for key in ("source_files", "missing_optional")})
            and row["python_runtime_identity_unchanged"]
        )
        row["full_model_accuracy_pass"] = bool(
            row["full_model_accuracy_pass"]
            and row["reference_coverage_matches_baseline"]
            and all(
                row[metric] is not None and row[metric] >= threshold
                for metric, threshold in (("top1", 0.9), ("top5", 0.98), ("top100", 1.0))
            )
        )
        row["eligible_for_selection"] = bool(
            row["full_model_accuracy_pass"]
            and row["traced"]
            and row["p1_pass"]
            and row["measurement_identity"]["valid"]
            and row["reference_identity_valid"]
            and row["native_runtime_identity_valid"]
            and row["python_runtime_identity_valid"]
            and row["finalized_successfully"]
        )
        row["eligibility_reasons"] = [
            name
            for name, passed in (
                ("full_model_accuracy_or_reference_coverage_failed", row["full_model_accuracy_pass"]),
                ("traced_teacher_forcing_performance_missing", row["traced"]),
                ("p1_missing_or_failed", row["p1_pass"]),
                ("runtime_or_source_identity_invalid", row["measurement_identity"]["valid"]),
                ("reference_identity_missing_or_invalid", row["reference_identity_valid"]),
                ("native_runtime_identity_missing_or_invalid", row["native_runtime_identity_valid"]),
                ("python_runtime_identity_missing_or_invalid", row["python_runtime_identity_valid"]),
                ("trial_not_finalized_successfully", row["finalized_successfully"]),
            )
            if not passed
        ]
    selection = selection_evidence(results, args.selected, args.baseline_id)
    for row in results:
        row["selected"] = row["selected"] and row["eligible_for_selection"]
    selected_rows = [row for row in results if row["selected"] and row["eligible_for_selection"]]
    if args.selected:
        assert selected_rows, "--selected must identify an eligible evaluated policy"
        selected = selected_rows[0]
        assert selected["eligible_for_selection"], f"Selected policy is ineligible: {selected['eligibility_reasons']}"
        assert selected[
            "p1_runner_gate_pass"
        ], "Selected policy needs a real P1 artifact whose selected ID matches and runner critical gate passes"
        assert (
            canonical_p1.exists() and json.loads(canonical_p1.read_text())["data"]["selected"] == args.selected
        ), "Final canonical P1 result must name the shipped selected policy"
        assert any(
            Path(proof["artifact"]).resolve() == canonical_p1.resolve() and proof["runner_gate_pass"]
            for proof in selected["p1_evidence"]
        ), "The canonical final P1 runner critical gate must pass; an older archived pass cannot replace it"
    # Full source manifests remain in each immutable raw trial; avoid duplicating
    # hundreds of identical file rows in both the consolidated JSON and CSV.
    for row in results:
        identity = row["python_runtime_identity"]
        row["python_runtime_identity"] = {
            **{key: identity[key] for key in ("schema_version", "digest", "source_digest") if key in identity},
            "evidence": row["raw_artifact"] + "#/python_runtime_identity",
        }
    payload = dict(
        p1_artifacts=[str(path) for path in p1_paths],
        thresholds={"top1": 0.9, "top5": 0.98, "top100": 1.0},
        selected=args.selected,
        selection_evidence=selection,
        results=results,
    )
    (ROOT / "sweep_results.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    columns = [
        "config_id",
        "precision_config",
        "top1",
        "top5",
        "top100",
        "ttft_ms",
        "ttft_scope",
        "callback_ttft_ms",
        "callback_ttft_scope",
        "teacher_forcing_decode_t_s_u",
        "repeat_spread",
        "traced",
        "measurement_regime",
        "hardware",
        "mesh",
        "status",
        "device_closed",
        "finalized_successfully",
        "full_model_accuracy_pass",
        "reference_coverage_matches_baseline",
        "reference_identity_valid",
        "reference_sha256",
        "native_runtime_identity",
        "native_runtime_identity_valid",
        "python_runtime_identity",
        "python_runtime_identity_valid",
        "accuracy_sample_evidence",
        "p1_status",
        "p1_positions",
        "p1_pass",
        "p1_runner_gate_pass",
        "p1_evidence",
        "eligible_for_selection",
        "eligibility_reasons",
        "selected",
        "dtype_policy",
        "compute_fidelity_policy",
        "command",
        "raw_artifact",
        "runtime_policy_evidence",
        "measurement_identity",
        "measurement_matches_baseline",
    ]
    with (ROOT / "sweep_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in results:
            writer.writerow(
                {key: json.dumps(row[key]) if isinstance(row[key], (dict, list)) else row[key] for key in columns}
            )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plotted_trials = [r for r in results if r["traced"] and r["top1"] is not None]
    points = []
    for config_id in sorted({r["config_id"] for r in plotted_trials}):
        trials = [r for r in plotted_trials if r["config_id"] == config_id]
        point = dict(trials[0])
        point["teacher_forcing_decode_t_s_u"] = sum(r["teacher_forcing_decode_t_s_u"] for r in trials) / len(trials)
        point["top1"] = min(r["top1"] for r in trials)
        point["top5"] = min(r["top5"] for r in trials)
        point["eligible_for_selection"] = all(r["eligible_for_selection"] for r in trials)
        point["selected"] = all(r["selected"] for r in trials)
        point["plot_repeats"] = [speed for r in trials for speed in r["repeat_speeds"]]
        points.append(point)
    for metric, threshold in [("top1", 0.90), ("top5", 0.98)]:
        fig, ax = plt.subplots(figsize=(12, 7), layout="constrained")
        fig.patch.set_facecolor("#f6f8fc")
        ax.set_facecolor("#f6f8fc")
        for row in points:
            x, y = row[metric] * 100, row["teacher_forcing_decode_t_s_u"]
            color = "#cf263d" if row["selected"] else "#226b9a" if row["eligible_for_selection"] else "#8796a5"
            repeats = row["plot_repeats"]
            ax.errorbar(x, y, yerr=[[y - min(repeats)], [max(repeats) - y]], color=color, capsize=4, zorder=3)
            ax.scatter(
                x, y, c=color, s=115 if row["selected"] else 65, marker="*" if row["selected"] else "o", zorder=4
            )
            ax.annotate(
                row["config_id"], (x, y), xytext=(5, 7), textcoords="offset points", fontsize=8, color="#273749"
            )
        frontier = [
            p
            for p in points
            if not any(
                q[metric] >= p[metric]
                and q["teacher_forcing_decode_t_s_u"] >= p["teacher_forcing_decode_t_s_u"]
                and (q[metric] > p[metric] or q["teacher_forcing_decode_t_s_u"] > p["teacher_forcing_decode_t_s_u"])
                for q in points
            )
        ]
        frontier.sort(key=lambda r: r[metric])
        ax.plot(
            [r[metric] * 100 for r in frontier],
            [r["teacher_forcing_decode_t_s_u"] for r in frontier],
            color="#226b9a",
            linewidth=1.5,
            marker="o",
            markersize=9,
            markerfacecolor="none",
            label="Evaluated Pareto frontier",
            zorder=2,
        )
        ax.axvline(
            threshold * 100, color="#69588e", linestyle=":", linewidth=2, label=f"Minimum {metric}: {threshold:.0%}"
        )
        ax.set_xlabel(f"Minimum {metric} agreement across prefill/teacher references (%)")
        ax.set_ylabel("Traced teacher forcing to resident logits (tokens/s/user)")
        ax.set_title(
            f"Qwen3.6-35B-A3B · {metric} / decode frontier\nAll40 layers · 1×4 Blackhole · mean per policy; bars show repeat range",
            loc="left",
        )
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="best")
        fig.savefig(ROOT / f"{metric}_perf_pareto.png", dpi=180)
        plt.close(fig)
    print("Collected", len(results), "trials;", len(points), "trace-verified plot points")


if __name__ == "__main__":
    main()
