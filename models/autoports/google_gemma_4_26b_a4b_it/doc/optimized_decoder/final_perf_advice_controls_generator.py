# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Summarize the current advice controls without rerunning hardware."""

import datetime
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def ref(name):
    return dict(artifact=name, sha256=sha(ROOT / name))


def screen(directory):
    path = ROOT / directory / "commands.json"
    if not path.exists():
        return []
    rows = []
    for command in read(path):
        file = Path(command["output"]).resolve()
        result = read(file)
        assert sha(file) == command["report_sha256"]
        row = dict(
            **ref(str(file.relative_to(ROOT))),
            layer=command["layer"],
            candidate=command["candidate"],
            status=command["status"],
            returncode=command["returncode"],
            runtime_sha256=result.get("runtime_sha256"),
            input_fixture_sha256=result.get("input_fixture_sha256"),
            prefill_pcc=result.get("pcc"),
            minimum_decode_pcc=result.get("decode", {}).get("min_pcc"),
            source_hashes=command["hashes"],
        )
        if command["returncode"] == 0:
            assert result["passed"] and result["decode"]["passed"] and result["decode"]["repeated_equal"]
            assert result["program_cache_miss_guard"]
            row.update(
                prefill_host_samples_us=result["warmed_prefill_host_us"],
                prefill_host_median_us=statistics.median(result["warmed_prefill_host_us"]),
                decode_host_samples_us=result["traced_decode_host_us"],
                decode_host_median_us=statistics.median(result["traced_decode_host_us"]),
                metadata=result.get("minimal_advice_runtime", result.get("prefill_placement_runtime")),
            )
        rows.append(row)
    return rows


def acceptance():
    directory = ROOT / "minimal_hifi2_acceptance_v6"
    rows = []
    for phase in ("stress", "long"):
        for layer in (0, 5):
            path = directory / f"{phase}_layer{layer}.json"
            if not path.exists():
                continue
            result = read(path)
            row = dict(
                **ref(str(path.relative_to(ROOT))),
                phase=phase,
                layer=layer,
                runtime_sha256=result["runtime_sha256"],
                passed=result["passed"],
                prefill_pcc=result["pcc"],
                probe_sha256=result["minimal_advice_probe_sha256"],
            )
            if phase == "long":
                samples = result["sampled_row_diagnostics"]
                assert len(samples) == 291
                assert all(item["passed"] == (item["pcc"] >= 0.995) for item in samples)
                failed = [item for item in samples if not item["passed"]]
                assert result["prefill_sampled_rows_passed"] == (not failed)
                row.update(
                    input_fixture_sha256=result["input_fixture"]["sha256"],
                    sampled_rows=len(samples),
                    minimum_sampled_pcc=min(item["pcc"] for item in samples),
                    failed_rows=failed,
                    aggregate_passed=result["prefill_aggregate_passed"],
                    end_query_checks=result["prefill_tail_checks"],
                    decode_checks=result["decode"],
                    shim_sha256=result["minimal_advice_contract_shim_sha256"],
                )
                if layer == 0:
                    baseline = read(ROOT / "validated_v5_long_262144_layer0.json")
                    assert baseline["input_fixture"]["sha256"] == row["input_fixture_sha256"]
                    row["historical_hifi4_control"] = dict(
                        **ref("validated_v5_long_262144_layer0.json"),
                        runtime_sha256=baseline["runtime_sha256"],
                        passed=baseline["passed"],
                        matching_rows=[
                            item
                            for item in baseline["sampled_row_diagnostics"]
                            if item["position"] in {item["position"] for item in failed}
                        ],
                        source_delta=ref("source_delta_v6.json"),
                    )
            else:
                row.update(
                    input_fixture_sha256=result["input_fixture_sha256"],
                    minimum_decode_pcc=result["decode"]["min_pcc"],
                    decode_passed=result["decode"]["passed"],
                    repeated_equal=result["decode"]["repeated_equal"],
                    program_cache_miss_guard=result["program_cache_miss_guard"],
                )
            rows.append(row)
    return rows


def main():
    qkv = screen("minimal_advice_v6")
    placement = screen("prefill_placement_v6")
    grid = screen("prefill_shared_grid_v6")
    gates = acceptance()
    pairs = read(ROOT / "minimal_pairs_v6_summary.json")
    qkv_l1 = read(ROOT / "qkv_l1_results.json")
    baselines = {row["layer"]: row for row in qkv if row["candidate"] == "baseline"}
    for row in qkv + placement + grid:
        if "prefill_host_median_us" in row:
            row["delta_vs_separate_process_screen_baseline_us"] = (
                row["prefill_host_median_us"] - baselines[row["layer"]]["prefill_host_median_us"]
            )
    remaining = []
    if len(qkv) != 6 or len(placement) != 3 or len(grid) != 2 or len(gates) != 4:
        remaining.append("Await all scheduled control reports")
    remaining.extend(qkv_l1["pending"])
    router_file = ROOT / "prefill_router_results.json"
    router = read(router_file) if router_file.exists() else None
    if router is None or router.get("remaining"):
        remaining.append("Phase-matched prefill-router HiFi2 and input-L1 advice disposition")
    result = dict(
        generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status="completed controls; selected integration/native profile validation separate"
        if not remaining
        else "controls pending",
        scope="Advice controls on recorded real activations: original fidelity/shared/output controls on b585 and subsequent full-QKV geometry/placement/producer controls on daa82. Exact runtime hashes remain attached to each report. Separate-process warmed host screens and same-process alternating pairs are distinct. Selected integration/native validation is separate; this generator runs no hardware.",
        qkv_screen=qkv,
        placement_screen=placement,
        shared_grid_screen=grid,
        paired=ref("minimal_pairs_v6_summary.json"),
        paired_results=pairs["results"],
        full_qkv_l1_family=ref("qkv_l1_results.json"),
        full_qkv_l1_results=qkv_l1["results"],
        prefill_router_advice=ref("prefill_router_results.json") if router is not None else None,
        hifi2_acceptance=gates,
        acceptance_journal=ref("minimal_hifi2_acceptance_v6/commands.json"),
        metadata_driver_failure=ref("minimal_hifi2_acceptance_v6/metadata_failure.json"),
        full_long_resume=ref("minimal_hifi2_acceptance_v6/full_long_resume_command.json"),
        planned_only_alternate_driver="minimal_advice_acceptance_v6/plan.json was never executed; actual parent journal is minimal_hifi2_acceptance_v6/commands.json",
        decisions=dict(
            sliding_qkv="Retain minimalK8 HiFi4. HiFi2 paired7/8 faster but actual maximum-context position32 PCC0.9949930133358956 fails the0.995 gate. This is actual activation evidence, not a synthetic veto.",
            full_qkv="Select minimal M2/K16/N8 HiFi2 with direct input-norm gamma output to L1, subject to selected integration validation. HiFi2 passes headline,512-step andall291maximum-context rows. M2+copiedL1 beats DRAM/M4 in8/8 pairs (-809.1405us); matchedM2 L1 beats DRAM in8/8 (-664.9255us), establishing the coupled input/output placement benefit (unspecified output memory inherits input placement). Direct producer then beats copiedL1 in22/32 pairs (-128.295us), with unchanged headline PCC. All are median paired whole-prefill host deltas, not isolated device claims.",
            qkv_grid="Retain11x8: sliding M4 grid110 paired4/8 faster (+4.3035us), full M4 loses. The adapted full M2 L1 family also tests110 directly:4/8 faster with+2.8355us median paired delta, no resolved gain. Its L1 input/output tensor specs and matmul program match the direct-producer path.",
            placement="Retain DRAM inputs for shared MLP and full output projection: allthree L1 screens pass but cost+205.697us sliding shared,+692.552us full minimal-output,+370.497us full shared versus screening baselines. These are screens, not paired gain estimates. FullQKV independently selects producerL1 after the adapted paired controls.",
            shared_grid="Retain88active schedule: legal110-active transpose controls pass but screen at221857.644us sliding/188432.310us full, slower by766.511/726.980us than screening baselines. No source/API blocker or absolute optimum is claimed.",
        ),
        shared_grid_contract=dict(
            source="ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp:494,510,562,836",
            original="Mtiles32/perM4 ->8; gateN132/perN12 anddownN88/perN8 ->11,88active on11x10configured",
            candidate="transpose_mcast=True; perM3; gateperN14/sub3x2,downperN9/sub3x1;11x10active",
            padded_tile_work=dict(
                gate_original=4224,
                gate_candidate=4620,
                gate_extra_pct=9.375,
                down_original=2816,
                down_candidate=2970,
                down_extra_pct=5.46875,
            ),
        ),
        remaining=remaining,
    )
    (ROOT / "final_perf_advice_controls.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Current prefill advice controls",
        "",
        result["scope"],
        "",
        "| Layer | Candidate | Host prefill median µs | Delta from screen baseline µs | Prefill PCC | Minimum decode PCC |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in qkv + placement + grid:
        if "prefill_host_median_us" in row:
            lines.append(
                f"| {row['layer']} | {row['candidate']} | {row['prefill_host_median_us']:.3f} | {row['delta_vs_separate_process_screen_baseline_us']:+.3f} | {row['prefill_pcc']:.12f} | {row['minimum_decode_pcc']:.12f} |"
            )
    lines += [
        "",
        "Eight same-process alternating pairs confirm the HiFi2 gains before the long-context gate. The JSON links exact commands, fixtures, configs, all timing samples and source hashes. See [paired results](minimal_pairs_v6_summary.md).",
        "",
    ]
    lines += [f"- {value}" for value in result["decisions"].values()]
    lines += [
        "",
        "The [full QKV L1 family](qkv_l1_results.md) preserves the original M4 circular-buffer overlap without rejecting placement. Two legal geometry adaptations were measured; matched M2 comparisons isolate placement, grid and producer changes. The65-token case checks three tiled M rows with M2. All successful reports retain exact fixture/source hashes and program-cache guards. Candidate controls are complete; selected integration validation and native advice parsing are separate gates.",
        "",
        "The sliding long run executes291 sampled rows and fails only position32. Aggregate PCC0.9992968877546407, end-query and decode checks pass; aggregate success does not override the individual-row failure. The identical fixture’s v5 HiFi4 control gives0.9951303778312409 at position32. The source-delta manifest connects the unchanged math path to v6. Full HiFi2 passes all291 rows with minimum0.996029643684647.",
        "",
        "The parent acceptance driver crashed while formatting the long-run decode list as a dictionary after the child had finished. Its frozen source and partial journal are preserved; metadata_failure.json records that incident. The strict long harness and raw report retain the real failure, and the full-layer continuation has a separate command/result journal. No failure is converted to a passing return code.",
        "",
        "Shared grid110 uses transpose multicast with perM3, gateperN14/sub3×2 anddownperN9/sub3×1. It increases padded output-tile work9.375%/5.469% but activates110 workers; the original schedule is88 active on an already configured110 grid. This schedule is tested, not dismissed using grid metadata.",
        "",
        "Remaining: "
        + (
            "; ".join(remaining)
            if remaining
            else "current selected integration validation and native profiles are owned by the main stage campaign"
        )
        + ".",
    ]
    (ROOT / "final_perf_advice_controls.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(rows=len(qkv) + len(placement) + len(grid), gates=len(gates), remaining=remaining)))


if __name__ == "__main__":
    main()
