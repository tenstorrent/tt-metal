import collections
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path.cwd()
A = ROOT / "models/autoports/google_gemma_4_31b_it/doc/agentic_evals"
S = A / "setup"
BASE = S / "swe85-precision-bisect/proposal-with-cap1024"
T = S / "swe85-third-cap-extensions"
CAP2 = S / "swe85-second-cap-extensions"
PAIRED = S / "swe85-precision-bisect"
OUT = PAIRED / "proposal-with-cap2048-pending-t-closure"
OUT.mkdir(exist_ok=True)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
write = lambda p, v: p.write_text(json.dumps(v, indent=2) + "\n")

base_cases = read(BASE / "candidate-cases.json")
oldcost = read(BASE / "runtime-estimate.json")
tprov = read(T / "provenance.json")
secondprov = read(CAP2 / "provenance.json")
assert len(base_cases) == 108
assert collections.Counter(x["free_run_tokens"] for x in base_cases) == {256: 72, 512: 7, 1024: 29}
base_by = {(x["source_onset"], x["seed"]): x for x in base_cases}
assert len(base_by) == 108

# Pin the 15 original served-anomalous/full-normal onset+seed target pairs.
RE = S / "swe85-remainder-execution85"
served = read(RE / "served-semantic-coordinator-all36.json")
full_rows = []
for letter in "abcd":
    full_rows.extend(read(RE / f"full-semantic-coordinator-{letter}.json")["rows"])
full_by = {(r["job"], r["seed"]): r for r in full_rows}
assert len(full_by) == 108
sensitive = []
for r in served["rows"]:
    fr = full_by[(r["job"], r["seed"])]
    full_verdict = fr.get("full_verdict", fr.get("coordinator_verdict", fr.get("verdict")))
    if r["coordinator_verdict"] in ("other_anomaly", "reproduced_native_motif") and full_verdict == "normal":
        sensitive.append(
            {
                "source_onset": r["job"],
                "seed": r["seed"],
                "served_verdict": r["coordinator_verdict"],
                "full_verdict": full_verdict,
            }
        )
assert len(sensitive) == 15

# Deduplicate eight T rows to seven onset/seed pairs; prefer full only on exact tie.
t_by = collections.defaultdict(list)
for r in tprov["cases"]:
    t_by[(r["source_job"], r["seed"])].append(r)
assert len(tprov["cases"]) == 8 and len(t_by) == 7
canonical = {}
for key, rows in t_by.items():
    assert key in base_by and base_by[key]["free_run_tokens"] == 1024
    if len(rows) > 1:
        assert len(rows) == 2 and {r["mode"] for r in rows} == {"served", "full"}
        a, b = [read(ROOT / r["fixture"]) for r in rows]
        assert (a["prompt_token_ids"], a["continuations"], a["metadata"]["sampling"], a["metadata"]["seed"]) == (
            b["prompt_token_ids"],
            b["continuations"],
            b["metadata"]["sampling"],
            b["metadata"]["seed"],
        )
        canonical[key] = next(r for r in rows if r["mode"] == "full")
    else:
        canonical[key] = rows[0]
assert len(canonical) == 7

# Copy all 108 identities; update only seven matched cap/fixture references.
cases = copy.deepcopy(base_cases)
changes = []
for job in cases:
    key = (job["source_onset"], job["seed"])
    if key not in canonical:
        continue
    rec = canonical[key]
    old_fixture = read(ROOT / job["case"])
    new_fixture = read(ROOT / rec["fixture"])
    assert (
        old_fixture["prompt_token_ids"],
        old_fixture["continuations"],
        old_fixture["metadata"]["sampling"],
        old_fixture["metadata"]["seed"],
    ) == (
        new_fixture["prompt_token_ids"],
        new_fixture["continuations"],
        new_fixture["metadata"]["sampling"],
        new_fixture["metadata"]["seed"],
    )
    assert new_fixture["metadata"]["cap_extension"]["previous_cap"] == 1024
    assert new_fixture["metadata"]["cap_extension"]["new_cap"] == 2048
    job["name"] = job["name"].replace(".cap1024", ".cap2048")
    job["case"] = rec["fixture"]
    job["free_run_tokens"] = 2048
    job["cap_source"] = "third_cap_" + rec["mode"]
    changes.append(
        {
            "source_onset": key[0],
            "seed": key[1],
            "source_mode": rec["mode"],
            "from_cap": 1024,
            "to_cap": 2048,
            "fixture": rec["fixture"],
            "fixture_sha256": rec["fixture_sha256"],
            "matched_1024_result": rec["source_result"],
            "matched_1024_result_sha256": rec["source_result_sha256"],
        }
    )
assert len(changes) == 7
assert {(x["source_onset"], x["seed"]) for x in changes} == set(canonical)
assert {(x["source_onset"], x["seed"]) for x in cases} == set(base_by)
assert collections.Counter(x["free_run_tokens"] for x in cases) == {256: 72, 512: 7, 1024: 22, 2048: 7}

# Actual completed cap1024 source results.
second_by = collections.defaultdict(list)
for r in secondprov["cases"]:
    second_by[(r["source_job"], r["seed"])].append(r)
oldper = {(r["source_onset"], r["seed"]): r for r in oldcost["per_trial"]}
assert len(oldper) == 108
per = []
hashrefs = {}
for job in cases:
    key = (job["source_onset"], job["seed"])
    cap = job["free_run_tokens"]
    fixture_path = ROOT / job["case"]
    fixture = read(fixture_path)
    hashrefs[str(fixture_path.relative_to(ROOT))] = sha(fixture_path)
    if cap in (256, 512):
        old = oldper[key]
        result_path = ROOT / old["source"]
        result = read(result_path)
        run = next(x for x in result["free_running"] if x["sampling"]["seed"] == job["seed"])
        dt = run["decode_timing"]
        generated = len(run["token_ids"])
        diag = run["diagnostic_wall_s_including_prefill"]
        mode = old["mode"]
        prefill = diag - dt["decode_wall_s"]
        scale = 1.0
        projected_decode = dt["decode_wall_s"]
        projected = diag
        assert sha(result_path) == old["source_sha256"] and abs(diag - old["diagnostic_wall_s"]) < 1e-7
    else:
        choices = second_by[key]
        if key in canonical:
            choices = [r for r in choices if r["mode"] == canonical[key]["mode"]]
        else:
            target_mode = job["cap_source"].replace("second_cap_", "")
            same_mode = [r for r in choices if r["mode"] == target_mode]
            if same_mode:
                choices = same_mode
            elif any(r["mode"] == "full" for r in choices):
                choices = [next(r for r in choices if r["mode"] == "full")]
        rec = canonical[key] if key in canonical else choices[0]
        mode = rec["mode"]
        result_path = (
            ROOT / rec["source_result"] if key in canonical else CAP2 / "runs" / rec["mode"] / (rec["job"] + ".json")
        )
        result = read(result_path)
        run = next(x for x in result["free_running"] if x["sampling"]["seed"] == job["seed"])
        dt = run["decode_timing"]
        generated = len(run["token_ids"])
        diag = run["diagnostic_wall_s_including_prefill"]
        prefill = diag - dt["decode_wall_s"]
        if key in canonical:
            assert sha(result_path) == rec["source_result_sha256"]
        source_fixture = read(ROOT / rec["fixture"])
        assert (
            fixture["prompt_token_ids"],
            fixture["continuations"],
            fixture["metadata"]["sampling"],
            fixture["metadata"]["seed"],
        ) == (
            source_fixture["prompt_token_ids"],
            source_fixture["continuations"],
            source_fixture["metadata"]["sampling"],
            source_fixture["metadata"]["seed"],
        )
        if cap == 1024:
            scale = 1.0
            projected_decode = dt["decode_wall_s"]
            projected = diag
        else:
            assert generated == 1024 and dt["decode_tokens"] == 1023
            scale = 2047 / 1023
            projected_decode = dt["decode_wall_s"] * scale
            projected = prefill + projected_decode
    assert generated > 0 and dt["decode_tokens"] == generated - 1
    hashrefs[str(result_path.relative_to(ROOT))] = sha(result_path)
    per.append(
        {
            "source_onset": job["source_onset"],
            "seed": job["seed"],
            "cap": cap,
            "source_mode": mode,
            "source_result": str(result_path.relative_to(ROOT)),
            "source_result_sha256": sha(result_path),
            "source_fixture": job["case"],
            "source_fixture_sha256": sha(fixture_path),
            "source_generated_tokens": generated,
            "source_decode_tokens": dt["decode_tokens"],
            "source_diagnostic_wall_s": diag,
            "prefill_host_overhead_s": prefill,
            "source_decode_wall_s": dt["decode_wall_s"],
            "decode_scale": scale,
            "projected_decode_wall_s": projected_decode,
            "projected_case_seconds": projected,
        }
    )

# Measured current T load and first-preparation proxy.
tcost = read(T / "cost-estimate.json")
load_rec = next(
    x for x in tcost["modes"]["served"]["load_sources"] if x["wall_s"] == tcost["modes"]["served"]["load_seconds_used"]
)
load_path = ROOT / load_rec["path"]
load = load_rec["wall_s"]
assert sha(load_path) == load_rec["sha256"]
prep_rec = tcost["first_preparation_source"]
prep_path = ROOT / prep_rec["path"]
prep_result = read(prep_path)
prep = prep_result["continuations"]["chosen"]["diagnostic_wall_s_including_prefill"]
assert sha(prep_path) == prep_rec["sha256"] and abs(prep - tcost["first_preparation_allowance_seconds"]) < 1e-9
hashrefs[load_rec["path"]] = load_rec["sha256"]
hashrefs[prep_rec["path"]] = prep_rec["sha256"]
variable = sum(x["projected_case_seconds"] for x in per)
fixed = load + prep
point = fixed + variable
planning_range = [fixed + 0.7 * variable, fixed + 1.7 * variable]

coord_paths = [RE / "served-semantic-coordinator-all36.json"] + [
    RE / f"full-semantic-coordinator-{x}.json" for x in "abcd"
]
source_hashes = dict(hashrefs)
for path in [
    BASE / "candidate-cases.json",
    BASE / "candidate-input-provenance.json",
    BASE / "runtime-estimate.json",
    PAIRED / "acceptance-decision.json",
    PAIRED / "acceptance-audit-c.json",
    T / "provenance.json",
    T / "cost-estimate.json",
    T / "served-candidate-cases.json",
    T / "full-candidate-cases.json",
    T / "review-d.json",
    S / "swe85-second-cap-extensions-completion-summary.json",
] + coord_paths:
    source_hashes[str(path.relative_to(ROOT))] = sha(path)
for rec in canonical.values():
    for path, expected_hash in [
        (rec["fixture"], rec["fixture_sha256"]),
        (rec["source_case"], rec["source_case_sha256"]),
        (rec["source_result"], rec["source_result_sha256"]),
        (rec["coordinator"], rec["coordinator_sha256"]),
    ]:
        assert sha(ROOT / path) == expected_hash
        source_hashes[path] = expected_hash

precision = read(BASE / "head_bfp8_hifi2_precision.json")
write(OUT / "head_bfp8_hifi2_precision.json", precision)
write(
    OUT / "head_bfp8_hifi2_build_kwargs.json",
    {"batch_size": 1, "precision_config": str((OUT / "head_bfp8_hifi2_precision.json").relative_to(ROOT))},
)
write(OUT / "candidate-cases.json", cases)
provenance = {
    "status": "candidate_copy_pending_T_closure_not_admitted_not_run",
    "source_panel": str((BASE / "candidate-cases.json").relative_to(ROOT)),
    "source_panel_sha256": sha(BASE / "candidate-cases.json"),
    "source_panel_input_provenance": str((BASE / "candidate-input-provenance.json").relative_to(ROOT)),
    "source_panel_input_provenance_sha256": sha(BASE / "candidate-input-provenance.json"),
    "selection_rule": "Preserve all 108 onset+seed identities from proposal-with-cap1024. Apply seven unique T cap2048 onset+seed pairs; where served/full duplicates are exact, prefer full. Keep prompt, continuation, sampling, seed, and HiFi2 policy unchanged.",
    "T_closure_status": "T is not closed; later coordinator outputs may add cap extensions, so this cap schedule remains provisional.",
    "counts": {
        "trials": 108,
        "precision_sensitive_served_anomalous_full_normal_pairs": 15,
        "cap_256": 72,
        "cap_512": 7,
        "cap_1024": 22,
        "cap_2048": 7,
        "T_rows": 8,
        "T_unique_pairs": 7,
    },
    "precision_sensitive_pairs": sensitive,
    "cap2048_changes": changes,
    "source_sha256": dict(sorted(source_hashes.items())),
}
write(OUT / "candidate-input-provenance.json", provenance)
forecast = {
    "status": "planning_only_pending_T_closure_not_performance_not_run",
    "formula": "Use exact matched diagnostic wall including prefill for retained 256/512 trials and actual matched 1024-token diagnostic wall for 22 cap1024 trials. For seven cap2048 trials, preserve matched 1024 prefill/host overhead and project decode as decode_wall_s * (2047/1023). Add one observed served-mode load and one measured chosen64 branch wall as first-preparation proxy. Load/preparation fixed; apply 0.7x–1.7x uncertainty multipliers to case-generation work only.",
    "panel": {
        "case_seed_trials": 108,
        "precision_sensitive_pairs": 15,
        "cap_256": 72,
        "cap_512": 7,
        "cap_1024": 22,
        "cap_2048": 7,
    },
    "load": {
        "source": load_rec["path"],
        "sha256": load_rec["sha256"],
        "wall_seconds": load,
        "basis": "Observed served BFP8 load; closest measured served load proxy for the proposed head-BFP8 HiFi2 build.",
    },
    "first_preparation": {
        "source": prep_rec["path"],
        "sha256": prep_rec["sha256"],
        "wall_seconds": prep,
        "field": "continuations.chosen.diagnostic_wall_s_including_prefill",
        "tokens": 64,
        "basis": "Measured chosen64 branch wall as first-preparation proxy, not direct HiFi2 cap2048 warmup measurement.",
    },
    "case_generation_seconds": variable,
    "fixed_load_plus_preparation_seconds": fixed,
    "estimated_total_seconds": point,
    "estimated_total_hours": point / 3600,
    "planning_range_seconds": planning_range,
    "planning_range_hours": [x / 3600 for x in planning_range],
    "uncertainty_variable_component_multipliers": [0.7, 1.7],
    "per_trial": per,
    "source_hashes": dict(sorted(source_hashes.items())),
}
write(OUT / "runtime-estimate.json", forecast)
cmd = {
    "status": "proposal_only_pending_T_closure_not_admitted_not_run",
    "working_directory": str(ROOT),
    "command": [
        "timeout",
        "21600",
        "env",
        "PYTHONPATH="
        + str(ROOT)
        + ":/home/mvasiljevic/v5-agentic/codex-home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.27/runtime",
        "READINESS_TRACE_REGION_SIZE=134217728",
        str(ROOT / "python_env/bin/python"),
        "-m",
        "models.autoports.google_gemma_4_31b_it.tools.precision_repro_batch",
        "--cases",
        str(OUT / "candidate-cases.json"),
        "--build-kwargs-file",
        str(OUT / "head_bfp8_hifi2_build_kwargs.json"),
        "--output-dir",
        str(OUT / "runs/head_bfp8_hifi2"),
    ],
    "cpu_validate_only_command": [
        "timeout",
        "21600",
        "env",
        "PYTHONPATH="
        + str(ROOT)
        + ":/home/mvasiljevic/v5-agentic/codex-home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.27/runtime",
        "READINESS_TRACE_REGION_SIZE=134217728",
        str(ROOT / "python_env/bin/python"),
        "-m",
        "models.autoports.google_gemma_4_31b_it.tools.precision_repro_batch",
        "--cases",
        str(OUT / "candidate-cases.json"),
        "--build-kwargs-file",
        str(OUT / "head_bfp8_hifi2_build_kwargs.json"),
        "--output-dir",
        str(OUT / "runs/head_bfp8_hifi2"),
        "--validate-only",
    ],
    "why_no_compare_dir": "All jobs have branches=[] and one free-running trial. Compare outputs by source_onset+seed in separate semantic adjudication.",
    "forecast": "runtime-estimate.json",
    "provenance": "candidate-input-provenance.json",
}
write(OUT / "command.json", cmd)
readme = (
    "# Pending cap-2048 HiFi2 panel\n\n"
    "Status: proposal only, pending T closure, not admitted and not run. This separate candidate retains the fixed 108 onset/seed identities and existing head-BFP8 HiFi2 precision policy. Current common caps are 72×256, 7×512, 22×1024, and 7×2048. The eight T extension rows contain one exact served/full duplicate; the full fixture is preferred for that identity.\n\n"
    "T has not closed, so further progressing outputs may require additional caps and change this provisional schedule. The 15 original served-anomalous/full-normal onset+seed pairs are pinned in candidate-input-provenance.json. Final precision selection requires review of all 108 trials, including baseline controls, and selection of the fastest measured passing policy. This proposal claims no passing candidate.\n\n"
    "The cost forecast uses actual matched cap-1024 timing for the 29 previously 1024-cap cases: 22 retain cap 1024 and seven project cap-2048 decode work by 2047/1023 with measured prefill/host overhead. Cap-256/512 cases retain matched diagnostic walls. Load and first-preparation values are measured proxies; the prep source is a chosen64 branch wall, not a direct HiFi2 cap-2048 warmup measurement. The 0.7×–1.7× range scales case generation only.\n\n"
    "Six-hour execution and CPU-only validation commands are recorded in command.json. Only --validate-only is run during preparation.\n"
)
(OUT / "README.md").write_text(readme)
print(
    json.dumps(
        {
            "output": str(OUT),
            "counts": dict(collections.Counter(x["free_run_tokens"] for x in cases)),
            "sensitive_pairs": len(sensitive),
            "point_seconds": point,
            "range_seconds": planning_range,
            "variable_seconds": variable,
            "source_hash_count": len(source_hashes),
        },
        indent=2,
    )
)
