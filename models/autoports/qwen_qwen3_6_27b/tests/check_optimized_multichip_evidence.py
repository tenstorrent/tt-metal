# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Read-only, CPU-only stage-5 evidence gate. No TTNN/torch imports.

Paths in the manifest are repo-relative (absolute paths also work). See
doc/optimized_multichip_decoder/evidence_gate_plan.md for schema and limits.
Missing/pending evidence fails; this gate does not replace independent review.
"""

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import runpy
import statistics
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

MODEL = "models/autoports/qwen_qwen3_6_27b"
MAIN = MODEL + "/tt/multichip_decoder.py"
SHAPES = {(layer, batch, seq) for layer in (0, 3) for batch, seq in ((1, 512), (32, 33), (32, 513))}
KINDS = {0: "linear_attention", 3: "full_attention"}
NATIVE_LIBRARIES = {"_ttnn.so", "_ttnncpp.so", "libtt_metal.so", "libtt_stl.so"}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_fused_decode_norm_resources(report):
    """Verify optional norm owners without weakening inherited state/CCL gates."""
    owners = report["owners"]
    require(len(owners) == 64, "Norm resource report needs all64 owners")
    old_sems = [a for o in owners for k in ("ccl_semaphores", "conv_semaphores") for a in o.get(k, [])]
    sems, addresses, active = [], [], 0
    rank_addresses = [[] for _ in range(4)]
    for o in owners:
        family = report["selected_policies"][o["kind"]].get("distributed_decode_norm", "off")
        require(family in ("off", "dit", "minimal"), "Invalid selected norm family")
        n = o.get("fused_decode_norm")
        if family == "off":
            require(n is None, "Unselected norm owner was installed")
            continue
        active += 1
        require(n is not None and n["family"] == family and n["layer_idx"] == o["layer"], "Wrong norm owner/family")
        require(n["phase_keys"] == ["input_layernorm.weight", "post_attention_layernorm.weight"], "Wrong norm phases")
        require(
            n["capacity_rows"] == 32
            and n["batch_signature_device_allocations"] == 1
            and n["metadata_views_share_backing"] is True,
            "Norm storage grew with batches",
        )
        require(report["batch"] in n["warmed_batches"], "Norm batch was not warmed")
        require(len(n["semaphores"]) == (10 if family == "dit" else 8), "Incomplete norm semaphores")
        require(len(n["persistent_addresses"]) == (4 if family == "dit" else 8), "Incomplete norm buffers")
        require(
            n["persistent_addresses"] == [t["address"] for t in n["persistent_tensors"]], "Norm address records differ"
        )
        for tensor in n["persistent_tensors"]:
            require(len(tensor["rank_addresses"]) == 4, "Missing TP4 norm rank addresses")
            for rank, address in enumerate(tensor["rank_addresses"]):
                rank_addresses[rank].append(address)
        if family == "minimal":
            require(n["worker_row"] == o["layer"] % 10, "Minimal norm owner placement differs")
        sems.extend(n["semaphores"])
        addresses.extend(n["persistent_addresses"])
    require(len(old_sems + sems) == len(set(old_sems + sems)), "Norm semaphore aliases another CCL owner")
    old_addresses = [a for o in owners for a in o.get("persistent_ccl_addresses", [])]
    old_addresses += [a for o in owners for a in o["state_addresses"].values()]
    require(len(old_addresses + addresses) == len(set(old_addresses + addresses)), "Norm tensor owners alias")
    for rank, rank_values in enumerate(rank_addresses):
        require(len(rank_values) == len(set(rank_values)), f"Norm tensor owner alias on rank{rank}")
    if active:
        require(report["fused_decode_norm_ownership_unique"] is True, "Missing norm uniqueness gate")
        require(
            report["resource_counts"]["fused_decode_norm_owners"] == active
            and report["resource_counts"]["fused_decode_norm_semaphores"] == len(sems),
            "Norm owner counts differ",
        )


def native_build_snapshot(repo):
    """Source/build inventory independent of staged/committed status; no imports."""
    repo = Path(repo).resolve()
    names = (
        subprocess.check_output(
            [
                "git",
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
                "--",
                "tt_metal",
                "ttnn/cpp",
                "ttnn/core",
            ],
            cwd=repo,
        )
        .decode()
        .split("\0")
    )
    suffixes = {".cpp", ".cc", ".c", ".hpp", ".h", ".S", ".s", ".ld", ".cmake"}
    sources = {
        name: file_sha256(repo / name)
        for name in sorted(set(names))
        if name and (Path(name).suffix in suffixes or Path(name).name == "CMakeLists.txt") and (repo / name).is_file()
    }
    require(sources, "native source inventory is empty")
    candidates = [repo / "ttnn/ttnn/_ttnn.so", *sorted((repo / "build/lib").glob("*.so*"))]
    binaries = {str(p.resolve()): file_sha256(p.resolve()) for p in candidates if p.is_file()}
    require(NATIVE_LIBRARIES <= {Path(p).name for p in binaries}, "native binary inventory is incomplete")
    metadata = {
        str(p.resolve()): file_sha256(p)
        for p in (
            repo / "CMakeLists.txt",
            repo / "CMakePresets.json",
            repo / "build/CMakeCache.txt",
            repo / "build/build.ninja",
            repo / "build/compile_commands.json",
        )
        if p.is_file()
    }
    return {
        "schema_version": 1,
        "native_source_sha256": sources,
        "binary_sha256": binaries,
        "build_metadata_sha256": metadata,
    }


def loaded_native_snapshot(repo, pid=None):
    """Hash actual repository native libraries mapped by a process, CPU-only."""
    repo = Path(repo).resolve()
    pid = os.getpid() if pid is None else int(pid)
    paths = set()
    allowed = {
        p.resolve() for p in (repo / "ttnn/ttnn/_ttnn.so", *sorted((repo / "build/lib").glob("*.so*"))) if p.is_file()
    }
    for line in Path(f"/proc/{pid}/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        raw = fields[5]
        if raw.endswith(" (deleted)") and (
            Path(raw.removesuffix(" (deleted)")).name in NATIVE_LIBRARIES
            or Path(raw.removesuffix(" (deleted)")).resolve() in allowed
        ):
            raise ValueError("mapped native library was replaced/deleted during the run: " + raw)
        p = Path(raw)
        if p.name in NATIVE_LIBRARIES or p.resolve() in allowed:
            paths.add(p.resolve())
    binaries = {str(p): file_sha256(p) for p in sorted(paths)}
    require(NATIVE_LIBRARIES <= {Path(p).name for p in binaries}, "process has not mapped all required TT libraries")
    return {"schema_version": 1, "pid": pid, "binary_sha256": binaries}


def validate_loaded_native(loaded, build):
    require(loaded["schema_version"] == build["schema_version"] == 1, "native provenance schema differs")
    binaries = loaded["binary_sha256"]
    require(NATIVE_LIBRARIES <= {Path(p).name for p in binaries}, "actual mapped TT libraries missing")
    for path, sha in binaries.items():
        require(build["binary_sha256"].get(path) == sha, f"mapped native binary differs from prepared build: {path}")


def paired_median_upper_bound(differences):
    """Exact >=95% one-sided median bound across independent paired jobs."""
    require(
        len(differences) >= 5 and all(type(x) in (int, float) and math.isfinite(x) for x in differences),
        "paired timing needs >=5 finite independent process differences",
    )
    n = len(differences)
    k = next(k for k in range(1, n + 1) if sum(math.comb(n, j) for j in range(k, n + 1)) / 2**n <= 0.05)
    return {
        "pairs": n,
        "median_difference_ms": statistics.median(differences),
        "upper_bound_ms": sorted(differences)[k - 1],
        "order_statistic": k,
        "coverage": 1 - sum(math.comb(n, j) for j in range(k, n + 1)) / 2**n,
    }


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def scores(value):
    if isinstance(value, dict):
        require(value, "empty PCC mapping")
        return [x for child in value.values() for x in scores(child)]
    if isinstance(value, list):
        require(value, "empty PCC list")
        return [x for child in value for x in scores(child)]
    require(
        type(value) in (int, float) and math.isfinite(value) and 0.995 <= value <= 1.000001,
        f"PCC below unchanged 0.995 bar or non-finite: {value!r}",
    )
    return [value]


def validate_hf_contract_report(d):
    """CPU-only schema/gate check shared by final evidence and job receipts."""
    require(d["pass"] is True and d["status"] == "pass" and d["mesh"] == [1, 4], "TP4 HF contract did not pass")
    require(d["contract_schema_version"] == 2 and d["pcc_threshold"] == 0.995, "HF contract schema/threshold differs")
    require(
        set(d["reference_types"]) == {"hf", "tp4_full_prefill"} and all(d["reference_types"].values()),
        "independent HF and TP4 self-consistency references must be labeled separately",
    )
    require(bool(d["hf_revision"]) and d["runtime_host_conversion_guard"] is True, "HF revision/runtime guard absent")
    checks = d["checks"]
    lengths = [v for v in checks if v["case"] == "length"]
    resumes = [v for v in checks if v["case"] == "resume"]
    mixed = [v for v in checks if v["case"] == "mixed_rows"]
    require(
        len(checks) == 20 and len(lengths) == 12 and len(resumes) == 7 and len(mixed) == 1,
        "HF contract case counts differ",
    )
    require(
        {v["length"] for v in lengths} == {1, 17, 31, 32, 33, 127, 128, 129, 511, 512, 513, 1025},
        "HF logical length coverage differs",
    )
    require({v["start"] for v in resumes} == {1, 32, 33, 128, 511, 512, 544}, "HF resume boundaries differ")
    for v in lengths:
        require(v["reference_type"] == "hf" and v["all_finite"] is True, "HF length reference/finite gate missing")
        scores(v["pcc"])
        scores(v["min_contribution_pcc"])
        scores(v["contribution_pcc"])
        require(len(v["contribution_pcc"]) == v["length"], "HF length per-row coverage differs")
        require(v["min_contribution_pcc"] == min(v["contribution_pcc"]), "HF contribution minimum differs")
    for v in resumes:
        require(
            v["reference_types"] == ["tp4_full_prefill", "hf"] and v["all_finite"] is True,
            "HF resume reference/finite gate missing",
        )
        rows = min(8, 545 - v["start"])
        require(v["rows"] == rows, "HF resume suffix length differs")
        for key in ("positive", "hf_output_row_pcc", "hf_contribution_pcc"):
            scores(v[key])
            require(len(v[key]) == rows, f"HF resume {key} per-row coverage differs")
        scores(v["hf_output_pcc"])
        negative = v["dropped_history"]
        require(
            len(negative) == rows
            and all(type(p) in (int, float) and math.isfinite(p) and -1.000001 <= p <= 1.000001 for p in negative)
            and min(negative) < 0.995,
            "TP4 dropped-history negative control missing/ineffective",
        )
    v = mixed[0]
    require(v["reference_type"] == "hf" and v["all_finite"] is True, "HF list reference/finite gate missing")
    require(v["starts"] == [32, 33] and v["lengths"] == [33, 17], "HF independent-list boundaries differ")
    scores(v["pcc"])
    require(len(v["pcc"]) == 2, "HF list output aggregate coverage differs")
    for key in ("output_row_pcc", "contribution_pcc"):
        scores(v[key])
        require([len(rows) for rows in v[key]] == [33, 17], f"HF list {key} per-row coverage differs")


def validate_boundary_report(d, *, tp4=True):
    instrumentation = d["boundary_instrumentation"]
    require(
        instrumentation["schema_version"] == 1 and d["no_boundary_conversion"] is True,
        "stack boundary instrumentation missing/failed",
    )
    runs = instrumentation["runs"]
    require(
        sum(r["phase"] == "prefill" for r in runs) >= 1 and sum(r["phase"] == "decode" for r in runs) >= 3,
        "boundary instrumentation omits prefill/eager/warm/captured decode",
    )
    pairs = list(zip(d["layers"], d["layers"][1:]))
    for run in runs:
        require(
            run["positive_control_rejected"] is True and run["boundary_calls"] == [], "boundary adapter/guard failure"
        )
        require(
            "ttnn.to_memory_config" in run["patched_apis"] and "ttnn.reshape" in run["patched_apis"],
            "boundary conversion interception missing",
        )
        require(
            [(b["from_layer"], b["to_layer"]) for b in run["boundaries"]] == pairs, "boundary pair coverage differs"
        )
        require(
            run["inside_layer_calls"] and all(type(v) is int and v > 0 for v in run["inside_layer_calls"].values()),
            "boundary guard did not observe legitimate layer operations",
        )
        if tp4:
            require(
                any(
                    "all_gather" in name or "reduce_scatter" in name or "all_reduce" in name
                    for name in run["inside_layer_calls"]
                ),
                "guard did not observe inside-layer TP4 collectives",
            )
        for boundary in run["boundaries"]:
            require(
                boundary["same_object"] is True and boundary["producer"] == boundary["consumer"],
                "inter-layer residual identity/layout/address changed",
            )
            contract = boundary["consumer"]
            require(
                contract["shape"][-1] == (1280 if tp4 else 5120)
                and len(contract["rank_buffer_addresses"]) == (4 if tp4 else 1),
                "boundary is not the expected mesh residual",
            )


class Gate:
    def __init__(self, manifest, repo):
        self.m, self.repo, self.errors, self.checked = manifest, repo, [], []
        self.validation_loaded = {}
        self.native_build = None

    def check(self, label, fn):
        try:
            fn()
            self.checked.append(label)
        except (
            KeyError,
            TypeError,
            ValueError,
            OSError,
            IndexError,
            AttributeError,
            subprocess.SubprocessError,
        ) as exc:
            self.errors.append(f"{label}: {type(exc).__name__}: {exc}")

    def path(self, name):
        require(isinstance(name, str) and name.strip(), "missing artifact path")
        p = Path(name)
        p = p if p.is_absolute() else self.repo / p
        require(p.is_file() and p.stat().st_size > 0, f"missing or empty artifact: {p}")
        return p

    def text(self, name):
        p = self.path(name)
        return gzip.decompress(p.read_bytes()).decode() if p.suffix == ".gz" else p.read_text()

    def read(self, name):
        d = json.loads(self.text(name))
        require(isinstance(d, dict), f"expected JSON object: {name}")
        require(
            d.get("pass") is not False and d.get("status") not in ("fail", "failed", "pending", "running"),
            f"failed/pending artifact: {name}",
        )
        return d

    def source(self, d):
        expected = self.m["sources"][MAIN]
        value = d.get("source_sha256", d.get("source_hashes"))
        if isinstance(value, dict):
            value = next((value[k] for k in (MAIN, "tt/multichip_decoder.py", "multichip_decoder") if k in value), None)
        require(value == expected, "artifact is not from final multichip source")

    def bundle(self, spec, *, current=True, default=True):
        d = self.read(spec["result"])
        if current:
            self.source(d)
        command = self.read(spec["command"])
        if current:
            snapshot = command["source_sha256"]
            runtime_paths = [
                name for name in self.m["sources"] if name.startswith(MODEL + "/tt/") and name.endswith(".py")
            ]
            for path in runtime_paths:
                short = path.removeprefix(MODEL + "/")
                require(
                    snapshot.get(short, snapshot.get(MODEL + "/" + short)) == self.m["sources"][MODEL + "/" + short],
                    f"run command snapshot missing or stale: {short}",
                )
            require(
                self.native_build is not None and command["native_build"] == self.native_build,
                "current command native sources/build differ from final inventory",
            )
            loaded = d.get("native_loaded", self.validation_loaded.get(str(self.path(spec["result"]).resolve())))
            require(loaded is not None, "actual mapped native-library provenance missing")
            validate_loaded_native(loaded, self.native_build)
        argv = command["command"]
        require(isinstance(argv, list) and argv, "command must retain exact argv list")
        require("--baseline" not in argv, "single-device baseline cannot satisfy TP4 evidence")
        if default:
            require(
                "--decoder-source" not in argv and d.get("decoder_source_override") is None,
                "stage4 test control cannot stand in for final-default evidence",
            )
            require("--legacy-decode-layout" not in argv, "final-default evidence uses legacy decode residual layout")
            require("--policy-file" not in argv, "final-default run has a policy-file override")
            if "--policy-json" in argv:
                require(json.loads(argv[argv.index("--policy-json") + 1]) == {}, "final-default run overrides policy")
        log = self.text(spec["log"])
        require(
            not any(x in log for x in ("Traceback (most recent call last)", "TT_FATAL", "markers were dropped")),
            "run log contains failure or dropped profiler markers",
        )
        return d, command

    def decoder(self, spec, *, current=True, performance=False, profile=False):
        d, command = self.bundle(spec, current=current, default=current)
        require(d["pass"] is True and d["finite"] is True, "numerical pass/finite missing")
        require(d["provenance"]["mesh"] == [1, 4], "measurement is not TP4")
        if current:
            require(d["candidate_policy"] == {}, "candidate override cannot stand in for final-default evidence")
            require(d["selected_policy"] == self.m["final_policy"][str(d["layer"])], "effective final policy differs")
            require(
                d["provenance"]["decode_input_shape_per_rank"] == [1, d["batch"], 1280],
                "final decode is not native compact TP4",
            )
        for key in ("prefill", "decode", "contribution", "per_user_pcc", "state"):
            scores(d[key])
        require(set(d["contribution"]) >= {"prefill", "decode"}, "missing phase contribution PCC")
        for phase in ("prefill", "decode"):
            require(len(d["per_user_pcc"][phase]) == d["batch"], "per-user coverage does not match logical batch")
        trace = d["trace"]
        require(
            trace["bit_equal_eager"] is True and trace["changed_input_control"] is True,
            "trace lacks exact eager/changed-input evidence",
        )
        require(trace["runtime_conversion_sentinels"] == "pass", "runtime conversion sentinel failed/missing")
        require(trace["captures"] >= 1 and trace["replays"] >= 2 and trace["cases"], "trace replay coverage missing")
        for case in trace["cases"]:
            require(
                case["output_bit_equal_eager"] is True and case["state_bit_equal_eager"] is True,
                "trace case changed output/state",
            )
            scores(case["baseline_output_pcc"])
            scores(case["baseline_state_pcc"])
        if performance:
            require(
                positive(d["timing"]["prefill_ms"]) and positive(d["timing"]["decode_trace_ms"]), "latencies missing"
            )
            require(d["timing"]["regime"] == d["provenance"]["regime"] == "serial", "performance regime is not serial")
            require(d["provenance"]["timed_trace_replays"] > 0, "warmed replay timing absent")
            require(str(d["provenance"]["watcher"]) in ("0", "None", ""), "performance run includes watcher")
            require(command["environment"].get("TT_METAL_WATCHER") in (None, ""), "performance command enables watcher")
            require(str(d["provenance"]["profiler"]) == ("1" if profile else "0"), "profiler regime mismatch")
            if not profile:
                require(d["timing"]["tracking_enabled"] is False, "final timing includes allocation tracking")
        return d, command

    def sources(self):
        require(self.m["schema_version"] == 1, "unsupported manifest schema")
        require(self.m["stage"] == "optimized-multichip-decoder", "wrong stage")
        require(self.m["pcc_threshold"] == 0.995, "accepted PCC threshold changed")
        for name in (
            MAIN,
            MODEL + "/tt/optimized_decoder.py",
            MODEL + "/tt/flat_prefill.py",
            MODEL + "/tt/recurrent_variants.py",
            MODEL + "/tests/run_multichip_decoder.py",
            MODEL + "/tests/run_multichip_contract.py",
            MODEL + "/tests/run_multichip_stack.py",
            MODEL + "/tests/run_optimized_multichip_validation.py",
            MODEL + "/tests/check_optimized_multichip_evidence.py",
            MODEL + "/tests/check_multichip_stack_resources.py",
            MODEL + "/doc/optimized_multichip_decoder/run_job.sh",
        ):
            require(name in self.m["sources"], f"final source pin missing: {name}")
        for name, sha in self.m["sources"].items():
            require(
                hashlib.sha256(self.path(name).read_bytes()).hexdigest() == sha, f"current source hash differs: {name}"
            )
        require(set(self.m["final_policy"]) == {"0", "3"}, "both effective layer policies required")

    def native(self):
        artifact = self.read(self.m["native_build"])
        inventory = artifact.get("native_build", artifact)
        require(
            inventory == native_build_snapshot(self.repo), "final native source/build inventory differs from checkout"
        )
        require(self.m["native_build_evidence"], "native build/install evidence missing")
        built_binaries = set()
        for name in self.m["native_build_evidence"]:
            build = self.read(name)
            require(
                build["build"].strip() and build["install"] and build["files"],
                "native build receipt lacks commands/files",
            )
            log = self.text(build["build_log"])
            require(
                not any(marker in log for marker in ("FAILED:", "ninja: error", "CMake Error")),
                "native build log failed",
            )
            for path, sha in build["files"].items():
                actual = self.path(path)
                require(file_sha256(actual) == sha, f"native build receipt file changed: {path}")
                if actual.name in NATIVE_LIBRARIES:
                    require(
                        inventory["binary_sha256"].get(str(actual.resolve())) == sha,
                        "build receipt binary is not inventoried",
                    )
                    built_binaries.add(actual.name)
        require(
            {"_ttnn.so", "_ttnncpp.so"} <= built_binaries, "native build receipts omit TTNN extension/install hashes"
        )
        self.native_build = inventory

    def validation_suite(self):
        spec = self.m["validation_suite"]
        plan, summary = self.read(spec["manifest"]), self.read(spec["summary"])
        planner = runpy.run_path(str(self.repo / MODEL / "tests/run_optimized_multichip_validation.py"))
        config = plan["configuration"]
        require(
            config["phases"] == list(planner["PHASES"]) and config["stress_runs"] == 2,
            "final validation must include all seven phases and both independent stress jobs",
        )
        require(
            len(config["context_lengths"]) == 1
            and config["context_lengths"][0] in (32768, 65536)
            and config["reserve_gib"] >= 20,
            "context coverage/reserve differs from final risk matrix",
        )
        canonical = planner["build_jobs"](SimpleNamespace(**config))
        require(
            len(canonical) == 68 and plan["jobs"] == canonical,
            "final validation job definitions differ from the 68-job plan",
        )
        require(plan["source_sha256"] == planner["source_snapshot"](), "validation source inventory is stale")
        require(
            plan["selected_default_sha256"] == self.m["sources"][MAIN] and plan["native_build"] == self.native_build,
            "validation default/native build differs",
        )
        definition = {
            key: plan[key]
            for key in ("jobs", "source_sha256", "selected_default_sha256", "native_build", "configuration")
        }
        signature = hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()
        require(
            plan["signature"] == summary["signature"] == signature and summary["pass"] is True,
            "validation signature/summary mismatch",
        )
        require(summary["jobs"] == 68 and summary["phases"] == config["phases"], "validation summary coverage differs")
        require(
            self.path(summary["manifest"]).resolve() == self.path(spec["manifest"]).resolve(),
            "summary refers to another plan",
        )
        expected_receipts = [str(planner["DOC"] / (job["name"] + ".validation.json")) for job in canonical]
        require(
            [str(self.path(p).resolve()) for p in summary["receipts"]]
            == [str(Path(p).resolve()) for p in expected_receipts],
            "validation receipt list differs from exact job inventory",
        )
        for job, path in zip(canonical, summary["receipts"]):
            receipt = self.read(path)
            require(
                receipt["pass"] is True
                and receipt["exit_code"] == 0
                and receipt["signature"] == signature
                and receipt["job"] == job,
                "validation receipt is failed/stale or belongs to another job",
            )
            require(
                receipt["native_build_verified_before_after"] is True,
                "native build was not stable across validation job",
            )
            validate_loaded_native(receipt["native_loaded"], self.native_build)
            result = planner["validate"](job)
            require(
                receipt["artifact_metadata"] == planner["artifact_metadata"](job),
                "validation artifact content/hash changed",
            )
            command_path = planner["DOC"] / (job["name"] + ".command.json")
            command = self.read(str(command_path))
            require(
                command["command"] == job["command"] and command["native_build"] == self.native_build,
                "validation actual command/build differs from planned job",
            )
            require(
                all(
                    command["environment"].get(key) == value
                    for key, value in job["environment"].items()
                    if key.startswith("TT_METAL_")
                ),
                "validation actual instrumentation environment differs",
            )
            if job["module"] == "run_multichip_contract":
                require(
                    receipt["reference_provenance"]
                    == {
                        key: result[key]
                        for key in ("contract_schema_version", "pcc_threshold", "hf_revision", "reference_types")
                    },
                    "HF validation receipt reference provenance differs",
                )
            if not job["baseline"]:
                self.validation_loaded[str(self.path(job["output"]).resolve())] = receipt["native_loaded"]
                self.source(result)
                if "selected_policy" in result:
                    layer = result.get("layer", result.get("metadata", {}).get("layer"))
                    require(
                        result["selected_policy"] == self.m["final_policy"][str(layer)], "validation job policy differs"
                    )

    def measurements(self):
        observed = set()
        for row in self.m["measurements"]:
            before, _ = self.decoder(row["before"], current=False, performance=True)
            after, _ = self.decoder(row["after"], performance=True)
            require(
                before["source_sha256"] == self.m["baseline_source_sha256"],
                "before row is not the declared accepted baseline",
            )
            shape = tuple(after[k] for k in ("layer", "batch", "length"))
            require(shape == tuple(before[k] for k in ("layer", "batch", "length")), "before/after shapes differ")
            require(shape not in observed, f"duplicate measurement row {shape}")
            observed.add(shape)
            require(
                before["reference_metadata"] == after["reference_metadata"], "before/after reference fixtures differ"
            )
        require(
            observed == SHAPES,
            f"need six final/default and baseline rows; missing={sorted(SHAPES-observed)}, extra={sorted(observed-SHAPES)}",
        )

    def b1_prefill_regression(self):
        spec = self.m["b1_prefill_regression"]
        require(
            spec["method"] == "exact_paired_median_upper_bound"
            and spec["confidence"] == 0.95
            and spec["margin_ms"] == 0,
            "B1 regression must use the declared 95% zero-margin paired method",
        )
        require(
            spec["control_source_sha256"]
            == self.m["baseline_source_sha256"]
            == "082c82faa6b9500f654dcc3f0e26f1c4fab656e987d3fd079bc1c79ec19a3c44",
            "B1 control is not the accepted original TP4 source",
        )
        observed, used_runs = set(), set()
        self.b1_statistics = []
        for workload in spec["workloads"]:
            layer, pairs = workload["layer"], workload["pairs"]
            require(
                layer in KINDS and layer not in observed and len(pairs) >= 5,
                "need >=5 independent B1 pairs for each layer kind",
            )
            observed.add(layer)
            differences = []
            previous_time = None
            for index, pair in enumerate(pairs):
                control, cc = self.decoder(pair["control"], current=False, performance=True)
                final, fc = self.decoder(pair["final"], performance=True)

                def comparable_command(argv):
                    result, index = [], 0
                    while index < len(argv):
                        if argv[index] in ("--decoder-source", "--output"):
                            index += 2
                        else:
                            result.append(argv[index])
                            index += 1
                    return result

                require(
                    comparable_command(cc["command"]) == comparable_command(fc["command"])
                    and "--diagnose" not in cc["command"]
                    and "--profile-drain" not in cc["command"],
                    "B1 paired commands differ beyond source/output or enable diagnostics",
                )
                require(
                    control["source_sha256"] == spec["control_source_sha256"] and "--decoder-source" in cc["command"],
                    "B1 control does not execute the pinned immutable source",
                )
                control_path = self.path(control["decoder_source_override"])
                require(file_sha256(control_path) == spec["control_source_sha256"], "immutable B1 control file changed")
                require(
                    control["candidate_policy"] == {} and "--legacy-decode-layout" not in cc["command"],
                    "B1 control policy/layout override",
                )
                require(control["reference_metadata"] == final["reference_metadata"], "B1 pair inputs/oracle differ")
                require(
                    control["provenance"]["runner_source_sha256"]
                    == final["provenance"]["runner_source_sha256"]
                    == file_sha256(self.repo / MODEL / "tests/run_multichip_decoder.py"),
                    "B1 pair is not measured with the same final harness",
                )
                require(
                    cc["native_build"] == fc["native_build"] == self.native_build, "B1 pair native sources/build differ"
                )
                validate_loaded_native(control["native_loaded"], self.native_build)
                for arm, command in ((control, cc), (final, fc)):
                    require((arm["layer"], arm["batch"], arm["length"]) == (layer, 1, 512), "B1 pair workload differs")
                    samples = arm["timing"]["prefill_samples_ms"]
                    require(len(samples) == 3 and all(positive(x) for x in samples), "B1 warmed call samples missing")
                    require(
                        arm["timing"]["prefill_estimator"] == "arithmetic_mean_of_three_warmed_calls"
                        and math.isclose(sum(samples) / 3, arm["timing"]["prefill_ms"], rel_tol=1e-12),
                        "B1 run mean disagrees with samples",
                    )
                    require(
                        command["environment"].get("TT_METAL_WATCHER") in (None, ""), "B1 timing watcher is enabled"
                    )
                for key in ("control", "final"):
                    run = str(self.path(pair[key]["result"]).resolve())
                    require(run not in used_runs, "B1 pair reuses a process result (pseudoreplication)")
                    used_runs.add(run)
                require(
                    pair["order"] == (["control", "final"] if index % 2 == 0 else ["final", "control"]),
                    "B1 independent pairs must alternate arm order",
                )
                times = {"control": cc["utc"], "final": fc["utc"]}
                first, last = (times[key] for key in pair["order"])
                require(
                    first < last and (previous_time is None or previous_time < first),
                    "B1 pairs are not sequential/alternating",
                )
                previous_time = last
                differences.append(final["timing"]["prefill_ms"] - control["timing"]["prefill_ms"])
            # For continuous paired differences, P(X_(k) < population median)
            # is the Binomial(n, .5) upper tail. Ties make this conservative.
            statistic = paired_median_upper_bound(differences)
            self.b1_statistics.append({"layer": layer, **statistic})
            require(
                statistic["upper_bound_ms"] <= 0,
                f"B1 layer{layer} zero-margin prefill nonregression not established: upper={statistic['upper_bound_ms']}ms",
            )
        require(observed == {0, 3}, "B1 prefill regression omitted a layer kind")

    def profiles(self):
        observed = set()
        for spec in self.m["profiles"]:
            d, cmd = self.decoder(spec, performance=True, profile=True)
            shape = tuple(d[k] for k in ("layer", "batch", "length"))
            require(shape not in observed, f"duplicate profile {shape}")
            observed.add(shape)
            require(
                str(cmd["environment"].get("TT_METAL_PROFILER_MID_RUN_DUMP", "0")) == "0",
                "unsafe profiler mid-run dump",
            )
            require(
                str(cmd["environment"].get("TT_METAL_WATCHER", "0")) in ("0", "None", ""), "watcher/profiler combined"
            )
            metrics, accounting = self.read(spec["metrics"]), self.read(spec["accounting"])
            require(
                self.path(metrics["correctness_result"]).resolve() == self.path(spec["result"]).resolve(),
                "metrics refer to another run",
            )
            require(metrics["provenance"] == d["provenance"], "metrics provenance differs from measured run")
            raw_path = self.path(spec["raw_ops"])
            raw_bytes = raw_path.read_bytes()
            if raw_path.suffix == ".gz":
                raw_bytes = gzip.decompress(raw_bytes)
            require(hashlib.sha256(raw_bytes).hexdigest() == metrics["raw_ops_sha256"], "raw profile hash mismatch")
            raw = raw_bytes.decode()
            raw_rows = list(csv.DictReader(io.StringIO(raw)))
            require(
                {r["DEVICE ID"] for r in raw_rows if r["DEVICE ID"]} == {"0", "1", "2", "3"},
                "profile missing TP4 ranks",
            )
            advice = set()
            for phase in ("prefill", "decode"):
                rows = list(csv.DictReader(io.StringIO(self.text(spec[phase + "_csv"]))))
                require(
                    rows and any(r.get("Device Time [μs]") for r in rows), f"{phase} CSV has no measured operations"
                )
                advice.update(r["Advice"].strip() for r in rows if r.get("Advice", "").strip())
                self.path(spec[phase + "_table"])
                require(
                    metrics[phase]["device_ops"] > 0 and positive(metrics[phase]["device_kernel_sum_us"]),
                    "empty profile window",
                )
            dispositions = spec["advice"]
            require({x["text"].strip() for x in dispositions} >= advice, "profiler advice lacks dispositions")
            for item in dispositions:
                require(
                    item["decision"] in ("kept", "rejected", "not_applicable", "not_tried_stop_rule"),
                    "unknown advice disposition",
                )
                require(item["reason"].strip() and item["evidence"], "advice requires reason and evidence")
                for name in item["evidence"]:
                    self.path(name)
            inventory = accounting["matrix_inventory_verified_against_run"]["matrices"]
            require(inventory, "missing selected-geometry inventory verification")
            for name, matrix in inventory.items():
                actual = d["matrix_allocations"][name]
                require(
                    actual["padded_shape"] == matrix["physical_shape"]
                    and actual["dtype"].split(".")[-1].lower() == matrix["dtype"],
                    f"accounting shape/dtype differs from uploaded matrix: {name}",
                )
            require(
                accounting["traffic_lower_bound"]["recurrent_read_write_bytes"]
                == (786432 * d["batch"] if d["layer"] == 0 else 0),
                "recurrent-state traffic omitted or miscounted",
            )
            require(
                math.isclose(accounting["same_run_host_ms"], d["timing"]["decode_trace_ms"], rel_tol=1e-9),
                "accounting host time from another run",
            )
            require(
                positive(accounting["roofline_ms"]) and positive(accounting["device_kernel_plus_gap_ms"]),
                "roofline/device reconciliation absent",
            )
            require(
                math.isclose(
                    accounting["roofline_fraction_of_host"],
                    accounting["roofline_ms"] / accounting["same_run_host_ms"],
                    rel_tol=1e-9,
                ),
                "roofline fraction inconsistent",
            )
            if d["batch"] == 1:
                require(
                    accounting["roofline_fraction_of_host"] >= 0.10,
                    "batch1 decoder below10% theoretical functional floor",
                )
        require(observed == SHAPES, f"profile coverage differs from six measured rows: {sorted(observed)}")

    def decoder_suite(self, group):
        specs = self.m["checks"][group]
        require(specs, f"{group} evidence pending")
        kinds, batches, pairs, functional_kinds = set(), set(), set(), set()
        for spec in specs:
            d, cmd = self.decoder(spec)
            kinds.add(d["layer"])
            batches.add(d["batch"])
            pairs.add((d["layer"], d["batch"]))
            if group == "stress":
                s = d["stress"]
                require(
                    s["runs"] >= 2 and s["steps"] > 0 and s["bit_equal"] is True and s["changed_inputs"] is True,
                    "stress must show repeated changed-input deterministic trajectories",
                )
                scores(s["output_pcc"])
                scores(s["state_pcc"])
                if d["layer"] == 3:
                    require(s["increasing_positions"] is True, "full-attention stress did not advance positions")
                if d["batch"] == 1 and d["length"] == 128 and s["steps"] >= 128:
                    functional_kinds.add(d["layer"])
                if d["batch"] == 32:
                    require(s["steps"] >= 512, "largest-batch stress is shorter than512 steps")
            elif group == "watcher":
                require(
                    str(d["provenance"]["watcher"]) == "10" and str(d["provenance"]["profiler"]) == "0",
                    "watcher regime missing or combined with profiler",
                )
                log = self.text(spec["watcher_log"])
                require(
                    not any(x in log for x in ("Watcher detected", "ASSERT", "TT_FATAL")), "watcher reported an error"
                )
            elif group == "trace_alloc":
                t = d["trace"]
                require(
                    t["allocation_tracking"] is True
                    and str(t["skip_program_cache"]) == "0"
                    and t["positive_control_rejected"] is True,
                    "allocation tracker/program-cache/positive-control evidence missing",
                )
                if t["acknowledged_buffers"]:
                    self.path(spec["acknowledgment_review"])
        require(kinds == {0, 3}, f"{group} must cover both layer kinds")
        if group in ("stress", "trace_alloc"):
            require(32 in batches, f"{group} must include largest served batch32")
        if group == "stress":
            require(functional_kinds == {0, 3}, "need inherited B1/128-input/128-step functional rows for both kinds")
            require({(0, 32), (3, 32)} <= pairs, "largest-batch stress must cover both layer kinds")

    def context(self):
        c = self.m["context"]
        contract = self.read(c["contract"])
        require(contract["supported_context"] >= 262144, "advertised context reduced")
        entry = contract["optimized_multichip_decoder"]
        require(
            entry["status"] == "validated" and entry.get("capability_reduction") is None,
            "optimized context acceptance pending/reduced",
        )
        require(
            entry["supported_context"] >= 262144 and entry["target_mesh"] == [1, 4],
            "stage5 TP4 context contract missing",
        )
        require(entry["decoder_source_sha256"] == self.m["sources"][MAIN], "context contract source pin differs")
        self.path(c["capacity"])
        observed = set()
        for spec in c["smokes"]:
            d, _ = self.bundle(spec)
            layer, length = d["metadata"]["layer"], d["metadata"]["length"]
            require(d["pass"] is True and d["full_decoder_all_tokens"] is True, "context did not execute full decoder")
            require(d["selected_policy"] == self.m["final_policy"][str(layer)], "context policy differs")
            require(32768 <= length <= 262144, "stage5 smoke shorter than32K")
            require(d["traced_decode_bit_equal"] is True, "context smoke lacks traced continuation")
            require(positive(d["largest_sampled_allocation_bytes_per_device"]), "context allocation evidence missing")
            scores(d["pcc"])
            scores(d["contribution_pcc"])
            observed.add(layer)
        require(observed == {0, 3}, "context smoke needs both kinds")
        if c["kv_dtype_or_layout_changed"]:
            require(c.get("max_context_validation"), "changed KV contract requires max-context validation evidence")
            max_kinds = set()
            for spec in c["max_context_validation"]:
                d, _ = self.bundle(spec)
                require(d["pass"] is True and d["metadata"]["length"] == 262144, "max-context validation did not pass")
                scores(d["pcc"])
                scores(d["contribution_pcc"])
                max_kinds.add(d["metadata"]["layer"])
            require(max_kinds == {0, 3}, "changed KV contract max-context coverage missing a layer kind")

    def hf_contract(self):
        specs = self.m["checks"]["hf_contract"]
        require(len(specs) == 2, "independent HF contract evidence needs both layer kinds")
        observed = set()
        harness = MODEL + "/tests/run_multichip_contract.py"
        require(harness in self.m["sources"], "HF contract harness source pin missing")
        for spec in specs:
            d, command = self.bundle(spec)
            validate_hf_contract_report(d)
            require(d["layer"] in KINDS and d["layer"] not in observed, "HF contract layer coverage differs")
            observed.add(d["layer"])
            require(d["selected_policy"] == self.m["final_policy"][str(d["layer"])], "HF contract policy differs")
            short = harness.removeprefix(MODEL + "/")
            require(
                d["source_sha256"].get(short) == self.m["sources"][harness], "HF contract result harness pin differs"
            )
            snapshot = command["source_sha256"]
            require(
                snapshot.get(short, snapshot.get(harness)) == self.m["sources"][harness],
                "HF contract command harness pin differs",
            )
        require(observed == {0, 3}, "independent HF contract omitted a layer kind")

    def interlayer(self):
        specs = self.m["checks"]["interlayer"]
        require(specs, "interlayer evidence pending")
        for spec in specs:
            d, _ = self.bundle(spec)
            require({0, 3} <= set(d["layers"]), "stack lacks both layer kinds")
            require(d["trace_bit_equal"] is True and d["boundary_shape"][-1] == 1280, "stack trace/layout failed")
            require(len(d["selected_policies"]) == len(d["layers"]), "stack effective policies missing")
            for layer, policy in zip(d["layers"], d["selected_policies"]):
                require(policy == self.m["final_policy"]["3" if layer % 4 == 3 else "0"], "stack uses another policy")
            scores(d["pcc"])
            validate_boundary_report(d)
            if d["no_boundary_conversion"] is not True:
                self.path(spec["measured_boundary_exception"])

    def resources(self):
        specs = self.m["checks"]["resources"]
        require(specs, "resource evidence pending")
        batches = set()
        for spec in specs:
            d, _ = self.bundle(spec)
            require(d["status"] == "pass" and d["mesh"] == [1, 4], "resource test failed")
            batches.add(d["batch"])
            require(
                d["resource_counts"]["layers"] == d["resource_counts"]["ccl_managers"] == 64
                and d["resource_counts"]["conv_exchange_buffers"] == 48,
                "resource test does not cover all64 owners",
            )
            validate_fused_decode_norm_resources(d)
            owners = d["owners"]
            require(
                len(owners) == 64 and {o["layer"] for o in owners} == set(range(64)),
                "resource owner records incomplete",
            )
            sems = [
                address for o in owners for key in ("ccl_semaphores", "conv_semaphores") for address in o.get(key, [])
            ]
            states = [address for o in owners for address in o["state_addresses"].values()]
            require(
                len(sems) == len(set(sems)) == 912 and len(states) == len(set(states)) == 128,
                "resource communication/state owners alias or are incomplete",
            )
            require(
                d["persistent_ccl_ownership_unique"] is True and d["prefill_batch_index_ownership_unique"] is True,
                "persistent scratch ownership audit missing",
            )
            for layer, kind in KINDS.items():
                require(
                    d["selected_policies"][kind] == self.m["final_policy"][str(layer)],
                    "resource test uses another policy",
                )
            require(
                d["late_program_cache_growth"] == 0 and d["all_main_persistent_allocations_before_capture"] is True,
                "late resource allocation or program-cache growth",
            )
            require(d["main_trace_allocated_bytes_per_device"] <= d["trace_region_bytes"], "trace region overflow")
            require(
                len(d["replays"]) >= 3 and all(r["pass"] is True for r in d["replays"]),
                "resource replay missing/failed",
            )
            require(
                {r["variant"] for r in d["replays"]} >= {"original", "changed"}, "resource changed-input replay missing"
            )
            for replay in d["replays"]:
                require(
                    len(replay["observed"]) == 64
                    and len(replay["checks"]) == 64 * 3 * 4
                    and all(c["exact"] is True for c in replay["checks"]),
                    "resource exact state/output coverage incomplete",
                )
            require(d["tracker_positive_control"]["pass"] is True, "resource tracker positive control missing")
            require(
                d["environment"]["TT_METAL_TRACE_ALLOC_TRACKING"] == "1"
                and d["environment"]["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"] == "0",
                "resource tracker bypass",
            )
            if d["allocation_acknowledgments"]:
                self.path(spec["acknowledgment_review"])
        require(batches == {1, 32}, "resource owner coverage needs both B1 and B32")

    def runtime(self):
        spec = self.m["checks"]["runtime_audit"]
        d = self.read(spec["result"])
        self.source(d)
        require(d["violations"] == [] and d["reachable_functions"], "runtime audit missing/violations")
        for name in ("optimized_decoder", "multichip_decoder", "flat_prefill", "recurrent_variants"):
            require(
                d["source_hashes"][name] == self.m["sources"][MODEL + "/tt/" + name + ".py"],
                f"runtime audit does not cover current {name}",
            )
        for name, sha in d["source_hashes"].items():
            require(
                self.m["sources"].get(MODEL + "/tt/" + name + ".py") == sha,
                f"reachable runtime helper missing/stale in final source pins: {name}",
            )
        self.path(spec["scope_review"])

    def stop_rule(self):
        s = self.m["stop_rule"]
        require(s["scope"].strip() and s["evidence"], "stop decision needs workload scope and evidence")
        for name in s["evidence"]:
            self.path(name)
        if s["reason"] == "explicit_user_wrapup":
            instruction = self.read(s["instruction"])
            require(
                instruction.get("optimization_search_stopped") is True
                and instruction.get("quality_gates_unchanged") is True,
                "User wrap-up must stop search while preserving quality gates",
            )
            require(
                instruction.get("instruction")
                == (
                    "Stage time/token cap reached. Keep the best version that passes the required gates, "
                    "run only the remaining required gates once, write untried ideas as hand-off notes "
                    "in the stage work log, and complete the stage (pass-with-carried is acceptable "
                    "for non-blocking items)."
                ),
                "Missing exact superseding user instruction",
            )
        elif s["reason"] == "device_time_cap":
            require(
                positive(s["device_hours"]) and s["device_hours"] >= 8, "user cap is8 device hours, not skill wall cap"
            )
        elif s["reason"] == "roofline_and_plateau":
            require(s["roofline_accounting"], "no target-workload roofline evidence")
            allowed = {self.path(p["accounting"]).resolve() for p in self.m["profiles"]}
            for name in s["roofline_accounting"]:
                require(self.path(name).resolve() in allowed, "stop roofline is not a final-default profile")
                require(self.read(name)["roofline_fraction_of_host"] >= 0.70, "70% roofline objective not achieved")
            require(len(s["last_two_kept"]) == 2, "need the last two kept candidates")
            for candidate in s["last_two_kept"]:
                before, after = candidate["before_ms"], candidate["after_ms"]
                require(
                    positive(before) and positive(after) and 0 <= (before - after) / before < 0.03,
                    "last kept gain is not below3%",
                )
                require(candidate["evidence"], "kept-candidate timing provenance missing")
                for name in candidate["evidence"]:
                    self.path(name)
        else:
            raise ValueError(
                "stop reason must be roofline_and_plateau or device_time_cap; weaker skill defaults do not apply"
            )

    def review_commit(self):
        r = self.m["review"]
        require(r["unresolved_blocking"] == [] and r["reports"], "independent review missing/blocking")
        require(
            r["verdict"] == "clean-pass" or (r["verdict"] == "pass-with-carried" and r["rounds"] >= 3),
            "review verdict/round count not accepted",
        )
        for path in r["reports"]:
            self.path(path)
        if r["verdict"] == "pass-with-carried":
            require(r["carried"], "carried verdict lacks explicit findings")
        commit = self.m["local_commit"]["sha"]
        require(
            isinstance(commit, str) and len(commit) == 40 and all(c in "0123456789abcdef" for c in commit),
            "local commit SHA pending/invalid",
        )
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=self.repo, check=True, capture_output=True
        )
        for path, sha in self.m["sources"].items():
            content = subprocess.run(
                ["git", "show", f"{commit}:{path}"], cwd=self.repo, check=True, capture_output=True
            ).stdout
            require(hashlib.sha256(content).hexdigest() == sha, f"local commit does not contain final source: {path}")
        owned = self.m["local_commit"]["stage_owned_paths"]
        require(owned and MAIN in owned, "stage-owned commit path inventory missing")
        for path in owned:
            content = subprocess.run(
                ["git", "show", f"{commit}:{path}"], cwd=self.repo, check=True, capture_output=True
            ).stdout
            require(content == self.path(path).read_bytes(), f"stage-owned path differs from local commit: {path}")

    def documents(self):
        for key in (
            "readme",
            "work_log",
            "topology_audit",
            "interlayer_contract",
            "candidate_decisions",
            "untried_handoff",
        ):
            self.path(self.m["documents"][key])

    def run(self):
        for name in (
            "sources",
            "native",
            "validation_suite",
            "measurements",
            "b1_prefill_regression",
            "profiles",
            "context",
            "hf_contract",
            "interlayer",
            "resources",
            "runtime",
            "stop_rule",
            "documents",
            "review_commit",
        ):
            self.check(name, getattr(self, name))
        for group in ("stress", "watcher", "trace_alloc"):
            self.check(group, lambda group=group: self.decoder_suite(group))
        extra = self.m.get("checks", {})
        extra = extra.get("correctness", []) if isinstance(extra, dict) else []
        require(isinstance(extra, list), "checks.correctness must be a list")
        for index, spec in enumerate(extra):
            self.check(f"additional_correctness[{index}]", lambda spec=spec: self.decoder(spec))
        return {
            "stage": "optimized-multichip-decoder",
            "pass": not self.errors,
            "checked": self.checked,
            "errors": self.errors,
            "b1_prefill_statistics": getattr(self, "b1_statistics", []),
            "scope": "Artifact consistency gate; independent numerical/kernel/performance review still required.",
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="JSON file, or '-' for stdin")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[4])
    args = parser.parse_args()
    try:
        m = json.loads(sys.stdin.read() if args.manifest == "-" else Path(args.manifest).read_text())
        require(isinstance(m, dict), "manifest must be a JSON object")
        report = Gate(m, args.repo_root.resolve()).run()
    except (OSError, ValueError, TypeError) as exc:
        report = {"pass": False, "errors": [f"manifest: {exc}"]}
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
