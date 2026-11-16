# SPDX-License-Identifier: Apache-2.0
"""Capture full-vocabulary logits on the Requests-5414 shared branch history.

This is an explicitly invoked diagnostic, not a sampler replay or benchmark.
It writes private logits to a caller-selected local .pt file and a compact JSON
summary. Importing this module does not import torch, Transformers, TTNN, or a
device runtime.
"""

import argparse
import ast
import copy
import hashlib
import heapq
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
assert (ROOT / "bringup/run_contract.json").is_file(), "repository root resolution failed"
D = ROOT / "models/autoports/google_gemma_4_31b_it/doc/agentic_evals"
INVESTIGATION = D / "setup/swe85-loop-branch-investigation/actual-branch-logits"
DEFAULT_CASE = INVESTIGATION / "fixture/anomaly.psf__requests-5414.call11.onset.suffix2048.private.json"
DEFAULT_BRANCHES = INVESTIGATION / "branches.json"
DEFAULT_BRANCH_PROVENANCE = INVESTIGATION / "branch-provenance.json"
HF_REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"
TARGET_IDS = (138, 147, 1390, 107, 556, 172109)
SAMPLING = {"temperature": 1.0, "top_k": 20, "top_p": 0.95}
EFFECTIVE_BF16_TOP_P = 0.94921875
# generator.generate(host_sampling=True) currently permits only greedy settings.
# The callback forces every continuation token; these settings are compatibility
# plumbing and do not select the recorded branch histories.
TT_HOST_COMPAT_SAMPLING = {"temperature": 0.0, "top_k": 1}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def validate_case_and_branches(case, branches, *, later_loop=False):
    prompt = case.get("prompt_token_ids")
    if not isinstance(prompt, list) or len(prompt) != 2048 or any(type(x) is not int or x < 0 for x in prompt):
        raise ValueError("expected the exact nonempty 2048-ID saved prompt")
    if case.get("metadata", {}).get("hf_revision") != HF_REVISION:
        raise ValueError("case is not pinned to the required HF revision")
    if case.get("metadata", {}).get("sampling") != SAMPLING:
        raise ValueError("case sampling settings differ from the investigation")
    expected = {
        "tt_seed44_loop_history": [1390, 107, 147, 1390, 107, 147],
        "hf_seed44_ellipsis_history": [1390, 107, 138, 556, 58982, 236800],
    }
    if later_loop:
        expected = {"tt_seed44_loop_history": [1390, 107, 147] * 10}
    if branches != expected:
        raise ValueError(f"branch histories differ from the reviewed hypotheses: {branches!r}")
    for name, tokens in branches.items():
        if len(tokens) != (30 if later_loop else 6) or any(type(token) is not int or token < 0 for token in tokens):
            raise ValueError(f"invalid branch token sequence: {name}")
    if not later_loop and branches["tt_seed44_loop_history"][:2] != branches["hf_seed44_ellipsis_history"][:2]:
        raise ValueError("branches must share the first two generated tokens")
    if not later_loop and (
        branches["tt_seed44_loop_history"][2] != 147 or branches["hf_seed44_ellipsis_history"][2] != 138
    ):
        raise ValueError("branch split must remain at generated token index2")
    return {
        "prompt_tokens": len(prompt),
        "branch_count": len(branches),
        "rows_per_branch": len(prompt[:1]) + len(next(iter(branches.values()))),
        "branch_names": list(branches),
    }


def _softmax_subset(values, token_ids):
    peak = max(values[i] for i in token_ids)
    weights = [math.exp(values[i] - peak) for i in token_ids]
    total = math.fsum(weights)
    return {token: weight / total for token, weight in zip(token_ids, weights)}


def _nucleus_distribution(values, top_k=20, p=0.95):
    ids = _topk_ids(values, top_k)
    topk = _softmax_subset(values, ids)
    kept, cumulative = [], 0.0
    for token in ids:
        kept.append(token)
        cumulative += topk[token]
        if cumulative >= p:
            break
    mass = math.fsum(topk[token] for token in kept)
    return {token: topk[token] / mass for token in kept}, mass


def analyze_logits(logits, target_ids=TARGET_IDS, top_k=20):
    """Pure-Python summary for a single full-vocabulary logits vector.

    The reconstructed top-k/top-p distribution is a host estimate from the
    saved BF16 logits, not the device sampler's internal probabilities or RNG.
    """
    values = [float(x) for x in logits]
    if not values or any(not math.isfinite(x) for x in values):
        raise ValueError("logits must be a nonempty finite full-vocabulary vector")
    maximum = max(values)
    weights = [math.exp(x - maximum) for x in values]
    denominator = math.fsum(weights)
    raw = [w / denominator for w in weights]
    top32_ids = heapq.nsmallest(min(32, len(values)), range(len(values)), key=lambda i: (-values[i], i))
    rank_by_id = {
        token: 1
        + sum(value > values[token] or (value == values[token] and i < token) for i, value in enumerate(values))
        for token in target_ids
        if token < len(values)
    }
    topk_ids = top32_ids[: min(top_k, len(top32_ids))]
    topk_prob_by_id = _softmax_subset(values, topk_ids)
    nominal_topp, nominal_mass = _nucleus_distribution(values, top_k, 0.95)
    bf16_topp, bf16_mass = _nucleus_distribution(values, top_k, EFFECTIVE_BF16_TOP_P)
    return {
        "vocab_size": len(values),
        "top32": [
            {"token_id": token, "rank": rank + 1, "logit": values[token], "raw_softmax_probability": raw[token]}
            for rank, token in enumerate(top32_ids)
        ],
        "host_rank_tie_break": "descending logit, then ascending global token ID",
        "topk_nucleus_support_semantics": "PROXY only: strict top-k20 with deterministic token-ID tie-break; HF may retain all kth-logit ties and device support is not exposed. Raw logits and ranks are primary evidence.",
        "top_k20_distribution": [{"token_id": token, "probability": topk_prob_by_id[token]} for token in topk_ids],
        "top_p095_nominal_distribution": [
            {"token_id": token, "probability": nominal_topp[token]} for token in nominal_topp
        ],
        "top_p095_bf16_distribution": [{"token_id": token, "probability": bf16_topp[token]} for token in bf16_topp],
        "top_p_cutoffs": {"nominal_hf": 0.95, "effective_bfloat16": EFFECTIVE_BF16_TOP_P},
        "top_p_kept_counts": {"nominal_hf": len(nominal_topp), "effective_bfloat16": len(bf16_topp)},
        "top_p_pre_renormalization_masses": {"nominal_hf": nominal_mass, "effective_bfloat16": bf16_mass},
        "distribution_reconstruction_note": "Host-side reconstruction from saved logits; does not model device probability quantization.",
        "targets": [
            {
                "token_id": token,
                "rank": rank_by_id[token] if token < len(values) else None,
                "logit": values[token] if token < len(values) else None,
                "raw_softmax_probability": raw[token] if token < len(values) else 0.0,
                "top_k20_probability": topk_prob_by_id.get(token, 0.0),
                "top_p095_nominal_probability": nominal_topp.get(token, 0.0),
                "top_p095_bf16_probability": bf16_topp.get(token, 0.0),
            }
            for token in target_ids
        ],
    }


def validate_source_pins(investigation=None):
    investigation = INVESTIGATION if investigation is None else investigation
    pins_path = investigation / "source-pins.json"
    pins = read_json(pins_path)
    if pins.get("status") != "source_pins_prepared_candidate_only":
        raise ValueError("source pins status mismatch")
    for rel, digest in pins.get("files", {}).items():
        path = ROOT / rel
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"source hash mismatch: {rel}")
    for rel, digest in pins.get("candidate_files", {}).items():
        path = investigation / rel
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"candidate input hash mismatch: {rel}")
    return pins


def validate_inputs(case_path=DEFAULT_CASE, branches_path=DEFAULT_BRANCHES, *, later_loop=False):
    case = read_json(case_path)
    branches = read_json(branches_path)
    facts = validate_case_and_branches(case, branches, later_loop=later_loop)
    provenance_path = INVESTIGATION / "branch-provenance.json"
    provenance = read_json(provenance_path)
    validate_branch_provenance(branches, provenance, later_loop=later_loop)
    pins = validate_source_pins()
    if sha256(case_path) != pins["candidate_files"].get(
        "fixture/anomaly.psf__requests-5414.call11.onset.suffix2048.private.json"
    ):
        raise ValueError("fixture does not match its candidate pin")
    facts["branch_provenance_sha256"] = sha256(provenance_path)
    return case, branches, facts, pins


def validate_branch_provenance(branches, provenance, *, later_loop=False):
    """Bind each forced continuation to the exact recorded seed-44 prefix."""
    if later_loop:
        origin = provenance["origin"]
        if origin["mode"] != "tt-full" or origin["seed"] != 44:
            raise ValueError("later-loop provenance must be full-TT seed44")
        source_path = ROOT / origin["path"]
        if sha256(source_path) != origin["sha256"]:
            raise ValueError("later-loop source hash mismatch")
        source = read_json(source_path)
        trial = next(row for row in source["free_running"] if row["sampling"]["seed"] == 44)
        observed = trial["token_ids"][:30]
        if (
            observed != branches.get("tt_seed44_loop_history")
            or observed != provenance["branch_token_ids"]
            or len(observed) != 30
        ):
            raise ValueError("later-loop branch differs from observed 30-token prefix")
        return
    expected = {
        "tt_seed44_loop_history": ("full", 44, "free_running", "token_ids"),
        "hf_seed44_ellipsis_history": ("hf", 44, "results", "generated_token_ids"),
    }
    for branch_name, (source_name, seed, collection_name, token_key) in expected.items():
        record = provenance[source_name]
        source_path = ROOT / record["path"]
        if sha256(source_path) != record["sha256"]:
            raise ValueError(f"observed branch source hash mismatch: {source_name}")
        source = read_json(source_path)
        if source_name == "full":
            match = next((row for row in source[collection_name] if row.get("sampling", {}).get("seed") == seed), None)
        else:
            match = next((row for row in source[collection_name] if row.get("seed") == seed), None)
        if match is None:
            raise ValueError(f"observed seed-{seed} continuation is missing: {source_name}")
        observed = match[token_key][:6]
        if observed != branches[branch_name] or observed != record["first_six_token_ids"]:
            raise ValueError(f"branch does not match the exact seed-{seed} observed continuation: {branch_name}")


def _native_generator_host_sampling_guard():
    """Compile the exact guard branch from generator.py without importing TTNN."""
    source_path = ROOT / "models/autoports/google_gemma_4_31b_it/tt/generator.py"
    module = ast.parse(source_path.read_text(), filename=str(source_path))
    generate = next(
        node
        for node in ast.walk(module)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "generate"
    )
    guards = [
        node
        for node in ast.walk(generate)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "host_params['temperature'] > 0 and host_params['top_k'] != 1"
    ]
    if len(guards) != 1:
        raise RuntimeError("generator host-sampling guard changed; re-audit this diagnostic")
    function = ast.FunctionDef(
        name="native_guard",
        args=ast.arguments(
            posonlyargs=[], args=[ast.arg(arg="host_params")], kwonlyargs=[], kw_defaults=[], defaults=[]
        ),
        body=[copy.deepcopy(guards[0])],
        decorator_list=[],
    )
    isolated = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {}
    exec(compile(isolated, str(source_path), "exec"), namespace)
    return namespace["native_guard"]


def validate_tt_host_sampling_contract(params):
    """Apply generator.py's actual guard branch without importing TTNN."""
    host_params = {"temperature": 0, "top_k": 1}
    host_params.update(params)
    _native_generator_host_sampling_guard()(host_params)


def _rows_for_tokens(tokens, branch, logits_by_branch):
    rows = []
    for row_index, input_token in enumerate([tokens[-1], *branch]):
        logits = logits_by_branch[row_index].detach().to(device="cpu").contiguous()
        vector = logits.tolist()
        rows.append(
            {
                "row_index": row_index,
                "absolute_input_position": len(tokens) - 1 + row_index,
                "input_token_id": input_token,
                "history_after_input": [*tokens[-1:], *branch[:row_index]],
                "predicted_continuation_index": row_index,
                "expected_next_token_id": branch[row_index] if row_index < len(branch) else None,
                "summary": analyze_logits(vector),
            }
        )
        del vector
    return rows


def collect_tt_branch_logits(generator, prompt_ids, branch, torch_module):
    """Run one forced branch and collect every decode logit row."""
    captured = []

    def collect(step, predicted):
        row = (
            generator._host_logits(generator.logits)
            .reshape(1, -1)[0]
            .detach()
            .to(device="cpu", dtype=torch_module.float32)
            .contiguous()
        )
        captured.append(row)
        return branch[step] if step < len(branch) else predicted

    validate_tt_host_sampling_contract(TT_HOST_COMPAT_SAMPLING)
    generator.generate(
        prompt_ids,
        len(branch) + 1,
        next_input=collect,
        enable_trace=True,
        host_sampling=True,
        buffered=False,
        sampling_params=dict(TT_HOST_COMPAT_SAMPLING),
    )
    if len(captured) != len(branch) + 1:
        raise RuntimeError(f"incomplete TT logits history: {len(captured)} rows")
    return captured


def _report_base(mode, case_path, case, branches, pins, build_kwargs_path=None, runtime=None):
    config_path = None
    if build_kwargs_path:
        build_kwargs = read_json(build_kwargs_path)
        config_path = ROOT / build_kwargs["precision_config"]
    return {
        "status": "completed_logits_capture",
        "mode": mode,
        "case_path": str(case_path.resolve()),
        "case_sha256": sha256(case_path),
        "source_prompt_tokens": len(case["prompt_token_ids"]),
        "prompt_token_ids_sha256": hashlib.sha256(
            json.dumps(case["prompt_token_ids"], separators=(",", ":")).encode()
        ).hexdigest(),
        "branches": branches,
        "branches_sha256": hashlib.sha256(
            json.dumps(branches, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "source_pins_sha256": sha256(INVESTIGATION / "source-pins.json"),
        "source_hashes": pins["files"],
        "hf_revision": HF_REVISION,
        "sampling_settings_for_distribution_reconstruction": SAMPLING,
        "effective_bfloat16_top_p": EFFECTIVE_BF16_TOP_P,
        "tt_host_compatibility_sampling_settings": TT_HOST_COMPAT_SAMPLING,
        "target_token_ids": list(TARGET_IDS),
        "precision": runtime or {"reference": "pinned Hugging Face CPU bfloat16"},
        "build_kwargs_path": str(build_kwargs_path.resolve()) if build_kwargs_path else None,
        "build_kwargs_sha256": sha256(build_kwargs_path) if build_kwargs_path else None,
        "precision_config_path": str(config_path.resolve()) if config_path else None,
        "precision_config_sha256": sha256(config_path) if config_path else None,
        "runtime_limitation": (
            "TT logits are read back through generator.generate(next_input=..., host_sampling=True, buffered=False). "
            "This mode requires greedy compatibility parameters (temperature=0, top_k=1). Prefill computes the "
            "model logits and its greedy token is discarded in favor of the raw-logit argmax; the callback then "
            "forces the branch token. Subsequent host_sampling model replays skip sampling and the callback forces "
            "every next history token. Recorded histories and raw logits are therefore not stochastic samples. "
            "The T=1, top_k=20, top_p=0.95 distributions are reconstructed host proxies from those same logits, "
            "not sampler settings used during capture. "
            "This captures forced-history logits and host-side distribution summaries, not the canonical device "
            "sampler candidate mapping, probability quantization, PRNG state, or sampled draw."
        ),
        "interpretation_limit": "Branch-conditional diagnostic only; no model-answer, correctness, performance, or causal-bug claim. Top-k/nucleus probabilities are strict-20 host proxies, not exact HF/device support; raw logits and ranks are primary.",
    }


def run_hf(case_path, case, branches, output_prefix):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    import torch

    from models.autoports.google_gemma_4_31b_it.tests.generate_full_reference import load_reference
    from models.autoports.google_gemma_4_31b_it.tt.weights import HF_REVISION as PINNED_REVISION

    torch.set_num_threads(16)
    model = load_reference()
    if (
        PINNED_REVISION != HF_REVISION
        or next(model.parameters()).device.type != "cpu"
        or next(model.parameters()).dtype != torch.bfloat16
    ):
        raise RuntimeError("HF reference identity/device/dtype mismatch")
    model.eval()
    tokens = case["prompt_token_ids"]
    prompt = torch.tensor([tokens], dtype=torch.long)
    rows_by_branch, logits_by_branch = {}, {}
    with torch.inference_mode():
        for name, branch in branches.items():
            cache = None
            # Match precision_repro_hf.py exactly: prefill P[:-1] in chunks of512.
            for start in range(0, len(tokens) - 1, 512):
                end = min(start + 512, len(tokens) - 1)
                out = model(
                    input_ids=prompt[:, start:end],
                    attention_mask=torch.ones((1, end), dtype=torch.long),
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
                cache = out.past_key_values
                del out
            captured = []
            inputs = [tokens[-1], *branch]
            for i, token in enumerate(inputs):
                absolute_position = len(tokens) - 1 + i
                one = torch.tensor([[token]], dtype=torch.long)
                out = model(
                    input_ids=one,
                    attention_mask=torch.ones((1, absolute_position + 1), dtype=torch.long),
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
                cache = out.past_key_values
                captured.append(out.logits[0, -1, :].detach().to(device="cpu", dtype=torch.float32).contiguous())
                del out, one
            logits_by_branch[name] = torch.stack(captured)
            rows_by_branch[name] = _rows_for_tokens(tokens, branch, captured)
            del cache
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    pt_path = output_prefix.with_suffix(".logits.pt")
    json_path = output_prefix.with_suffix(".report.json")
    torch.save(
        {
            "format": "actual-branch-logits-v1",
            "mode": "hf",
            "case_sha256": sha256(case_path),
            "revision": HF_REVISION,
            "rows": logits_by_branch,
        },
        pt_path,
    )
    os.chmod(pt_path, 0o600)
    report = _report_base("hf", case_path, case, branches, validate_source_pins())
    report.update(
        {
            "logits_pt": str(pt_path.resolve()),
            "logits_pt_sha256": sha256(pt_path),
            "rows": rows_by_branch,
            "artifact_policy": "private local file; not uploaded or sent to telemetry",
        }
    )
    _write_private(json_path, report)
    del model
    return json_path


def run_tt(mode, case_path, case, branches, build_kwargs_path, output_prefix):
    if mode not in ("tt-selected", "tt-full"):
        raise ValueError("invalid TT mode")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    import torch
    from readiness_check.mesh_device import close_readiness_mesh_device, open_readiness_mesh_device

    from models.autoports.google_gemma_4_31b_it.tt.generator import build_generator

    kwargs = read_json(build_kwargs_path)
    if kwargs.get("batch_size") != 1:
        raise ValueError("TT branch diagnostic requires batch_size=1")
    config_path = ROOT / kwargs["precision_config"]
    expected_config = "stage12_head_bfp8_lofi" if mode == "tt-selected" else "stage12_bf16_hifi4_diagnostic"
    if read_json(config_path).get("config_id") != expected_config:
        raise ValueError(f"{mode} precision config must be {expected_config}")
    mesh = open_readiness_mesh_device("P300x2", "FABRIC_1D_RING")
    generator = None
    logits_by_branch, rows_by_branch = {}, {}
    primary_error = None
    try:
        generator = build_generator(model_dir=Path(__file__).parents[1], mesh_device=mesh, **kwargs)
        runtime = generator.model.precision_config
        for name, branch in branches.items():
            captured = collect_tt_branch_logits(generator, case["prompt_token_ids"], branch, torch)
            logits_by_branch[name] = torch.stack(captured)
            rows_by_branch[name] = _rows_for_tokens(case["prompt_token_ids"], branch, captured)
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        pt_path = output_prefix.with_suffix(".logits.pt")
        json_path = output_prefix.with_suffix(".report.json")
        torch.save(
            {
                "format": "actual-branch-logits-v1",
                "mode": mode,
                "case_sha256": sha256(case_path),
                "revision": HF_REVISION,
                "rows": logits_by_branch,
            },
            pt_path,
        )
        os.chmod(pt_path, 0o600)
        report = _report_base(mode, case_path, case, branches, validate_source_pins(), build_kwargs_path, runtime)
        report.update(
            {
                "logits_pt": str(pt_path.resolve()),
                "logits_pt_sha256": sha256(pt_path),
                "rows": rows_by_branch,
                "generator_counters": generator.counters,
                "artifact_policy": "private local file; not uploaded or sent to telemetry",
            }
        )
        _write_private(json_path, report)
        return json_path
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        _cleanup_preserving_primary(generator, mesh, close_readiness_mesh_device, primary_error)


def _close_tt_resources(generator, mesh, close_mesh):
    """Release generator state and mesh, retaining all cleanup failures."""
    errors = []
    if generator is not None:
        try:
            release = getattr(generator, "release_persistent_capture", None)
            if callable(release):
                release()
            else:
                teardown = getattr(generator, "teardown", None)
                if callable(teardown):
                    teardown()
        except BaseException as exc:
            errors.append(exc)
    try:
        close_mesh(mesh, "FABRIC_1D_RING")
    except BaseException as exc:
        errors.append(exc)
    return errors


def _cleanup_preserving_primary(generator, mesh, close_mesh, primary_error):
    cleanup_errors = _close_tt_resources(generator, mesh, close_mesh)
    if cleanup_errors:
        for index, cleanup_error in enumerate(cleanup_errors):
            try:
                if primary_error is None:
                    if index:
                        print(
                            f"additional cleanup failure: {type(cleanup_error).__name__}: {cleanup_error}",
                            file=sys.stderr,
                        )
                else:
                    primary_error.add_note(f"cleanup also failed: {type(cleanup_error).__name__}: {cleanup_error}")
            except AttributeError:
                print(f"cleanup also failed: {type(cleanup_error).__name__}: {cleanup_error}", file=sys.stderr)
        if primary_error is None:
            raise cleanup_errors[0]


def compare_saved(left_report_path, right_report_path, output_path, torch_module=None):
    if torch_module is None:
        import torch as torch_module

    left, right = read_json(left_report_path), read_json(right_report_path)
    if left.get("case_sha256") != right.get("case_sha256") or left.get("branches") != right.get("branches"):
        raise ValueError("reports do not share exact source case and branch definitions")
    left_pt, right_pt = Path(left["logits_pt"]), Path(right["logits_pt"])
    if sha256(left_pt) != left.get("logits_pt_sha256") or sha256(right_pt) != right.get("logits_pt_sha256"):
        raise ValueError("saved logits artifact hash mismatch")
    left_blob = torch_module.load(left_pt, map_location="cpu", weights_only=True)
    right_blob = torch_module.load(right_pt, map_location="cpu", weights_only=True)
    for report, blob in ((left, left_blob), (right, right_blob)):
        if (
            blob.get("format") != "actual-branch-logits-v1"
            or blob.get("case_sha256") != report.get("case_sha256")
            or blob.get("revision") != HF_REVISION
            or blob.get("mode") != report.get("mode")
        ):
            raise ValueError("saved logits metadata does not match its pinned report")
    a, b = left_blob["rows"], right_blob["rows"]
    comparisons = {}
    for branch_name in left["branches"]:
        if branch_name not in a or branch_name not in b or len(a[branch_name]) != len(b[branch_name]):
            raise ValueError(f"saved logits rows differ: {branch_name}")
        per_row = []
        for index, (x, y) in enumerate(zip(a[branch_name], b[branch_name])):
            xvals = x.tolist() if hasattr(x, "tolist") else [float(value) for value in x]
            yvals = y.tolist() if hasattr(y, "tolist") else [float(value) for value in y]
            metrics = compare_logit_vectors(xvals, yvals)
            dx, dy = _topk_distribution(xvals), _topk_distribution(yvals)
            support = set(dx) | set(dy)
            topk_tv = 0.5 * sum(abs(dx.get(token, 0.0) - dy.get(token, 0.0)) for token in support)
            top_p_nominal_x, top_p_nominal_y = _top_p_distribution(xvals, p=0.95), _top_p_distribution(yvals, p=0.95)
            support_nominal = set(top_p_nominal_x) | set(top_p_nominal_y)
            topp_nominal_tv = 0.5 * sum(
                abs(top_p_nominal_x.get(token, 0.0) - top_p_nominal_y.get(token, 0.0)) for token in support_nominal
            )
            top_p_x, top_p_y = _top_p_distribution(xvals, p=EFFECTIVE_BF16_TOP_P), _top_p_distribution(
                yvals, p=EFFECTIVE_BF16_TOP_P
            )
            support_p = set(top_p_x) | set(top_p_y)
            topp_tv = 0.5 * sum(abs(top_p_x.get(token, 0.0) - top_p_y.get(token, 0.0)) for token in support_p)
            per_row.append(
                {
                    "row_index": index,
                    **metrics,
                    "top_k20_distribution_tv": topk_tv,
                    "top_p095_nominal_distribution_tv": topp_nominal_tv,
                    "top_p095_bf16_distribution_tv": topp_tv,
                }
            )
        comparisons[branch_name] = per_row
    report = {
        "status": "saved_logits_comparison",
        "left_report": str(left_report_path.resolve()),
        "right_report": str(right_report_path.resolve()),
        "left_report_sha256": sha256(left_report_path),
        "right_report_sha256": sha256(right_report_path),
        "left_logits_sha256": left["logits_pt_sha256"],
        "right_logits_sha256": right["logits_pt_sha256"],
        "case_sha256": left["case_sha256"],
        "comparisons": comparisons,
        "interpretation_limit": "Logit/distribution comparison only; no performance or causal conclusion.",
    }
    _write_private(output_path, report)
    return output_path


def compare_logit_vectors(xvals, yvals, top_k=20):
    if len(xvals) != len(yvals) or not xvals:
        raise ValueError("logit vectors must have the same nonzero vocabulary size")
    xvals, yvals = [float(v) for v in xvals], [float(v) for v in yvals]
    if any(not math.isfinite(v) for v in (*xvals, *yvals)):
        raise ValueError("logit vectors must be finite")
    diff = [x - y for x, y in zip(xvals, yvals)]
    mx, my = math.fsum(xvals) / len(xvals), math.fsum(yvals) / len(yvals)
    xc, yc = [x - mx for x in xvals], [y - my for y in yvals]
    denom = math.sqrt(math.fsum(x * x for x in xc) * math.fsum(y * y for y in yc))
    pcc = math.fsum(x * y for x, y in zip(xc, yc)) / denom if denom else None
    px, py = _softmax_subset(xvals, range(len(xvals))), _softmax_subset(yvals, range(len(yvals)))
    tx, ty = set(_topk_ids(xvals, top_k)), set(_topk_ids(yvals, top_k))
    return {
        "pcc": pcc,
        "mae": math.fsum(abs(x) for x in diff) / len(diff),
        "rmse": math.sqrt(math.fsum(x * x for x in diff) / len(diff)),
        "max_abs_error": max(abs(x) for x in diff),
        "raw_full_softmax_tv": 0.5 * math.fsum(abs(px[i] - py[i]) for i in range(len(xvals))),
        "left_raw_softmax_mass_on_own_top20": math.fsum(px[i] for i in tx),
        "right_raw_softmax_mass_on_own_top20": math.fsum(py[i] for i in ty),
        "top20_intersection_count": len(tx & ty),
        "top20_union_count": len(tx | ty),
        "top20_support_jaccard": len(tx & ty) / max(1, len(tx | ty)),
    }


def _topk_distribution(logits, top_k=20):
    values = logits.tolist() if hasattr(logits, "tolist") else [float(value) for value in logits]
    ids = _topk_ids(values, top_k)
    return _softmax_subset(values, ids)


def _topk_ids(logits, top_k=20):
    values = logits.tolist() if hasattr(logits, "tolist") else logits
    return heapq.nsmallest(min(top_k, len(values)), range(len(values)), key=lambda i: (-values[i], i))


def _top_p_distribution(logits, p=EFFECTIVE_BF16_TOP_P):
    values = logits.tolist() if hasattr(logits, "tolist") else [float(value) for value in logits]
    return _nucleus_distribution(values, 20, p)[0]


def _write_private(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", opener=lambda name, flags: os.open(name, flags, 0o600)) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def mock_tests():
    case = {"prompt_token_ids": [1] * 2048, "metadata": {"hf_revision": HF_REVISION, "sampling": dict(SAMPLING)}}
    branches = {
        "tt_seed44_loop_history": [1390, 107, 147, 1390, 107, 147],
        "hf_seed44_ellipsis_history": [1390, 107, 138, 556, 58982, 236800],
    }
    assert validate_case_and_branches(case, branches)["rows_per_branch"] == 7
    later_loop_tested = False
    if INVESTIGATION.name == "later-loop-conditional-candidate":
        later_case_path = INVESTIGATION / "fixture/anomaly.psf__requests-5414.call11.onset.suffix2048.private.json"
        later_case = read_json(later_case_path)
        later_branches = read_json(INVESTIGATION / "branches.json")
        assert validate_case_and_branches(later_case, later_branches, later_loop=True)["rows_per_branch"] == 31
        later_provenance = read_json(INVESTIGATION / "branch-provenance.json")
        validate_branch_provenance(later_branches, later_provenance, later_loop=True)

        altered_history = {name: list(tokens) for name, tokens in later_branches.items()}
        altered_history["tt_seed44_loop_history"][29] += 1
        try:
            validate_case_and_branches(later_case, altered_history, later_loop=True)
        except ValueError:
            pass
        else:
            raise AssertionError("altered 30-token later-loop history accepted")

        altered_hash = copy.deepcopy(later_provenance)
        altered_hash["origin"]["sha256"] = "0" * 64
        try:
            validate_branch_provenance(later_branches, altered_hash, later_loop=True)
        except ValueError as exc:
            assert "source hash mismatch" in str(exc)
        else:
            raise AssertionError("altered later-loop source hash accepted")

        altered_seed = copy.deepcopy(later_provenance)
        altered_seed["origin"]["seed"] = 43
        try:
            validate_branch_provenance(later_branches, altered_seed, later_loop=True)
        except ValueError as exc:
            assert "seed44" in str(exc)
        else:
            raise AssertionError("later-loop provenance from seed43 accepted")
        later_loop_tested = True
    row = analyze_logits([0.0, 3.0, 2.0, 1.0], target_ids=(0, 1, 99), top_k=3)
    assert row["top32"][0]["token_id"] == 1
    assert row["targets"][1]["rank"] == 1 and row["targets"][2]["rank"] is None
    assert pytest_close(sum(entry["probability"] for entry in row["top_p095_bf16_distribution"]), 1.0)
    assert pytest_close(sum(entry["probability"] for entry in row["top_p095_nominal_distribution"]), 1.0)
    host_topk = _topk_distribution([4.0, 3.0, 2.0, 1.0])
    host_topp = _top_p_distribution([4.0, 3.0, 2.0, 1.0], p=0.8)
    assert len(host_topk) == 4 and len(host_topp) == 2 and host_topk != host_topp

    class FakeTensor:
        def __init__(self, values):
            self.values = values

        def reshape(self, *_shape):
            return self

        def __getitem__(self, _index):
            return self

        def detach(self):
            return self

        def to(self, **_kwargs):
            return self

        def contiguous(self):
            return self

        def tolist(self):
            return list(self.values)

    class FakeTorch:
        float32 = object()

    class FakeGenerator:
        def __init__(self):
            self.logits = None
            self.forced = []
            self.kwargs = None

        def _host_logits(self, logits):
            return logits

        def generate(self, prompt, steps, **kwargs):
            self.kwargs = (prompt, steps, kwargs)
            validate_tt_host_sampling_contract(kwargs["sampling_params"])
            callback = kwargs["next_input"]
            for step in range(steps):
                self.logits = FakeTensor([float(step), 1.0, -1.0])
                self.forced.append(callback(step, 99))

    fake_generator = FakeGenerator()
    collected = collect_tt_branch_logits(fake_generator, [7, 8], [31, 32], FakeTorch)
    assert len(collected) == 3 and fake_generator.kwargs[:2] == ([7, 8], 3)
    assert fake_generator.forced == [31, 32, 99]
    assert [row.tolist()[0] for row in collected] == [0.0, 1.0, 2.0]
    assert fake_generator.kwargs[2]["host_sampling"] is True and fake_generator.kwargs[2]["buffered"] is False
    assert fake_generator.kwargs[2]["sampling_params"] == {"temperature": 0.0, "top_k": 1}
    native_guard = _native_generator_host_sampling_guard()
    native_guard({"temperature": 0.0, "top_k": 1})
    try:
        native_guard({"temperature": 1.0, "top_k": 20})
    except ValueError as exc:
        assert "Host compatibility mode is greedy" in str(exc)
    else:
        raise AssertionError("stochastic host compatibility settings passed the generator guard")
    mock_rows = _rows_for_tokens([7, 8], [31, 32], collected)
    assert [row["history_after_input"] for row in mock_rows] == [[8], [8, 31], [8, 31, 32]]

    class Releaser:
        def release_persistent_capture(self):
            raise RuntimeError("release")

    cleanup_calls = []

    def bad_close(mesh, mode):
        cleanup_calls.append((mesh, mode))
        raise ValueError("mesh close")

    import contextlib
    import io

    primary = OSError("generation")
    with contextlib.redirect_stderr(io.StringIO()):
        _cleanup_preserving_primary(Releaser(), "mesh", bad_close, primary)
    assert str(primary) == "generation" and cleanup_calls == [("mesh", "FABRIC_1D_RING")]
    try:
        _cleanup_preserving_primary(None, "mesh", bad_close, None)
    except ValueError as exc:
        assert str(exc) == "mesh close"
    else:
        raise AssertionError("cleanup failure without primary error was swallowed")

    # Exercise compare_saved itself with a host-only loader and isolated private artifacts.
    import tempfile

    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        branches = {"b": [1]}
        case_hash = "a" * 64
        blobs = {}
        report_paths = []
        for index, (mode, logits) in enumerate((("hf", [[4.0, 3.0, 2.0, 1.0]]), ("tt-full", [[3.0, 4.0, 2.0, 1.0]]))):
            pt = temp / f"{index}.pt"
            pt.write_bytes(b"host-test-logits")
            report = {
                "case_sha256": case_hash,
                "branches": branches,
                "mode": mode,
                "logits_pt": str(pt),
                "logits_pt_sha256": sha256(pt),
            }
            rp = temp / f"{index}.json"
            rp.write_text(json.dumps(report))
            report_paths.append(rp)
            blobs[str(pt)] = {
                "format": "actual-branch-logits-v1",
                "case_sha256": case_hash,
                "revision": HF_REVISION,
                "mode": mode,
                "rows": {"b": logits},
            }

        class FakeLoader:
            @staticmethod
            def load(path, **_kwargs):
                return blobs[str(path)]

        out = temp / "comparison.json"
        compare_saved(report_paths[0], report_paths[1], out, torch_module=FakeLoader)
        compared = read_json(out)["comparisons"]["b"][0]
        assert compared["top_k20_distribution_tv"] > 0
        assert "top_p095_nominal_distribution_tv" in compared
        assert "top_p095_bf16_distribution_tv" in compared
    try:
        validate_case_and_branches(case, {**branches, "tt_seed44_loop_history": [1]})
    except ValueError:
        pass
    else:
        raise AssertionError("invalid branch fixture accepted")
    return {
        "fixture_and_branch_validation": True,
        "original_six_token_branches_retained": True,
        "later_loop_exact_30_token_branch": later_loop_tested,
        "later_loop_rejects_altered_history_source_hash_and_seed": later_loop_tested,
        "host_distribution_summary": True,
        "distinct_topk_and_topp": True,
        "tt_callback_history_and_capture_count": True,
        "cleanup_failure_preserves_primary": True,
        "compare_saved_host_mock": True,
        "no_torch_transformers_or_ttnn_imported": True,
    }


def pytest_close(a, b, tol=1e-12):
    return abs(a - b) <= tol


def main():
    global INVESTIGATION
    parser = argparse.ArgumentParser(description=__doc__)
    gate = parser.add_mutually_exclusive_group(required=True)
    gate.add_argument("--validate-only", action="store_true")
    gate.add_argument("--execute", action="store_true")
    parser.add_argument("--mode", choices=("hf", "tt-selected", "tt-full", "compare"))
    parser.add_argument("--case", type=Path, default=DEFAULT_CASE)
    parser.add_argument("--branches", type=Path, default=DEFAULT_BRANCHES)
    parser.add_argument("--build-kwargs-file", type=Path)
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--left-report", type=Path)
    parser.add_argument("--right-report", type=Path)
    parser.add_argument("--comparison-output", type=Path)
    parser.add_argument("--mock-tests", action="store_true")
    parser.add_argument("--later-loop", action="store_true", help="Use the separately pinned 30-token full-TT history")
    args = parser.parse_args()
    try:
        if args.later_loop:
            INVESTIGATION = INVESTIGATION / "later-loop-conditional-candidate"
            if args.branches == DEFAULT_BRANCHES:
                args.branches = INVESTIGATION / "branches.json"
        case, branches, facts, pins = validate_inputs(args.case, args.branches, later_loop=args.later_loop)
        if args.validate_only:
            if args.mock_tests:
                tests = mock_tests()
            else:
                tests = None
            print(
                json.dumps(
                    {
                        "status": "validated_inert_no_model_or_device",
                        "facts": facts,
                        "source_pin_count": len(pins["files"]),
                        "mock_tests": tests,
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.mock_tests or not args.mode:
            raise ValueError("--execute requires --mode; --mock-tests is validate-only")
        if args.mode == "compare":
            if not all((args.left_report, args.right_report, args.comparison_output)):
                raise ValueError("compare mode requires both reports and --comparison-output")
            comparison_output = args.comparison_output.resolve()
            if not comparison_output.is_relative_to((INVESTIGATION / "outputs").resolve()):
                raise ValueError(
                    "comparison artifact must remain under investigation outputs/; telemetry paths are forbidden"
                )
            if comparison_output.exists():
                raise FileExistsError(f"refusing to overwrite artifact: {comparison_output}")
            output_root = (INVESTIGATION / "outputs").resolve()
            for report_path in (args.left_report.resolve(), args.right_report.resolve()):
                if not report_path.is_relative_to(output_root):
                    raise ValueError("comparison inputs must be private outputs under this investigation")
            comparison_output.parent.mkdir(parents=True, exist_ok=True)
            compare_saved(args.left_report, args.right_report, comparison_output)
            return 0
        if not args.output_prefix:
            raise ValueError("capture mode requires --output-prefix")
        output_prefix = args.output_prefix.resolve()
        if not output_prefix.is_relative_to((INVESTIGATION / "outputs").resolve()):
            raise ValueError("logits artifacts must remain under investigation outputs/; telemetry paths are forbidden")
        for suffix in (".logits.pt", ".report.json"):
            if output_prefix.with_suffix(suffix).exists():
                raise FileExistsError(f"refusing to overwrite artifact: {output_prefix.with_suffix(suffix)}")
        if args.mode == "hf":
            run_hf(args.case, case, branches, output_prefix)
            return 0
        if not args.build_kwargs_file:
            raise ValueError("TT mode requires --build-kwargs-file")
        run_tt(args.mode, args.case, case, branches, args.build_kwargs_file, output_prefix)
        return 0
    except Exception as exc:
        print(f"actual_branch_logits: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
