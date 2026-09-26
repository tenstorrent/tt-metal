# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Rebuild the current advice audit from completed native profiles; CPU only."""

import argparse
import collections
import csv
import datetime
import gzip
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT.parent.parent / "tt/optimized_decoder.py"
BOUNDS = {
    0: [
        ("input norm", 3, 7),
        ("qkv projection", 8, 8),
        ("qkv head layout", 9, 12),
        ("head norms", 13, 32),
        ("rope lookup and rotation", 33, 56),
        ("cache update", 57, 59),
        ("native sdpa", 60, 61),
        ("output projection and head concat", 62, 65),
        ("post attention and common norm", 66, 76),
        ("router", 77, 97),
        ("routed experts", 98, 109),
        ("shared experts", 110, 120),
        ("tail norm and residual", 121, 125),
    ],
    5: [
        ("input norm", 3, 7),
        ("qkv projection", 8, 10),
        ("qkv head layout", 11, 13),
        ("head norms", 14, 19),
        ("rope lookup and rotation", 20, 55),
        ("cache update", 56, 61),
        ("native sdpa", 62, 63),
        ("output projection and head concat", 64, 67),
        ("post attention and common norm", 68, 78),
        ("router", 79, 99),
        ("routed experts", 100, 112),
        ("shared experts", 113, 123),
        ("tail norm and residual", 124, 128),
    ],
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def evidence(name):
    path = ROOT / name
    return dict(artifact=name, sha256=sha(path))


def joined_rows(directory, phase):
    raw_file = "decode_one_replay.csv" if phase == "decode" else "ops.csv"
    with (directory / raw_file).open() as stream:
        raw = {
            int(float(row["GLOBAL CALL COUNT"])): row for row in csv.DictReader(stream) if row.get("GLOBAL CALL COUNT")
        }
    with (directory / f"{phase}_perf_report.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["id"] = int(row["ID"])
        row["device_us"] = float(row["Device Time"])
        row["raw"] = raw[int(float(row["Global Call Count"]))]
        native_ns = float(row["raw"]["DEVICE KERNEL DURATION [ns]"])
        assert abs(row["device_us"] - native_ns / 1000) < 0.002
    return rows


def native(row):
    raw = row["raw"]
    attributes = raw["ATTRIBUTES"]
    configs = dict(re.findall(r"'(program_config|config|compute_kernel_config)': '([^']*)'", attributes))
    return dict(
        id=row["id"],
        op=row["OP Code"],
        device_us=row["device_us"],
        cores=row["Cores"],
        math_fidelity=row["Math Fidelity"],
        advice=row["Advice"],
        attributes=attributes,
        parsed_configs=configs,
        tensors={
            key: value
            for key, value in raw.items()
            if value and (key.startswith("INPUT_") or key.startswith("OUTPUT_"))
        },
        global_call_count=raw["GLOBAL CALL COUNT"],
        kernel_sources={key: value for key, value in raw.items() if "KERNEL SOURCE" in key and value},
    )


def grouped_prefill(rows, span):
    groups = collections.defaultdict(lambda: dict(native_ops=0, device_us=0.0, advice=set()))
    for row in rows:
        group = groups[(row["OP Code"], row["Math Fidelity"])]
        group["native_ops"] += 1
        group["device_us"] += row["device_us"]
        group["advice"].add(row["Advice"])
    return [
        dict(
            op=key[0],
            math_fidelity=key[1],
            native_ops=value["native_ops"],
            device_us=value["device_us"],
            whole_prefill_span_pct=100 * value["device_us"] / span,
            advice=sorted(value["advice"] - {""}),
        )
        for key, value in sorted(groups.items(), key=lambda pair: pair[1]["device_us"], reverse=True)
    ]


def profile(layer, version):
    directory = ROOT / "tracy" / f"actual_optimized_{version}_layer{layer}"
    runner = read(ROOT / f"profile_actual_optimized_{version}_layer{layer}.json")
    whole = read(directory / "whole_layer.json")
    commands = read(directory / "summary_commands.json")
    assert all(command["returncode"] == 0 and command["advice_enabled"] for command in commands["report_commands"])
    assert commands["reconciliation_returncode"] == 0
    decode, prefill = joined_rows(directory, "decode"), joined_rows(directory, "prefill")
    # The v6 fixes and prefill-only tuning retain this verified decode sequence.
    previous = joined_rows(ROOT / "tracy" / f"actual_optimized_v5_layer{layer}", "decode")
    assert [(row["id"], row["OP Code"]) for row in decode] == [(row["id"], row["OP Code"]) for row in previous]
    kernel_sum = sum(row["device_us"] for row in decode)
    groups = {}
    for name, lo, hi in BOUNDS[layer]:
        selected = [row for row in decode if lo <= row["id"] <= hi]
        assert len(selected) == hi - lo + 1
        duration = sum(row["device_us"] for row in selected)
        groups[name] = dict(
            id_range=[lo, hi],
            native_ops=len(selected),
            device_us=duration,
            sampled_kernel_pct=100 * duration / kernel_sum,
        )
    assert sum(group["native_ops"] for group in groups.values()) == len(decode)
    assert len(prefill) == whole["whole_layer_windows"]["prefill"]["native_ops"]
    sparse = [native(row) for row in decode if "SparseMatmul" in row["OP Code"]]
    for row in sparse:
        tensors = row["tensors"]
        assert "'use_indices': 'true'" in row["attributes"]
        assert tensors["INPUT_3_DATATYPE"] == "UINT16" and tensors["INPUT_3_LAYOUT"] == "ROW_MAJOR"
        assert tensors["INPUT_3_X_PAD[LOGICAL]"] == "8[8]" and tensors["OUTPUT_0_Z_PAD[LOGICAL]"] == "8[8]"
        assert tensors["INPUT_1_Z_PAD[LOGICAL]"] == "128[128]"
    unique_prefill = {}
    for row in prefill:
        if "Matmul" not in row["OP Code"]:
            continue
        key = (row["OP Code"], row["raw"]["ATTRIBUTES"], row["raw"]["INPUT_0_MEMORY"])
        if key not in unique_prefill:
            unique_prefill[key] = dict(first_native_row=native(row), invocations=0, device_us=0.0)
        unique_prefill[key]["invocations"] += 1
        unique_prefill[key]["device_us"] += row["device_us"]
    qkv_producers = []
    for index, row in enumerate(prefill):
        if "MinimalMatmul" in row["OP Code"] and "2816 x 9216" in row["OP Code"]:
            qkv_producers.append(
                dict(
                    projection=native(row),
                    preceding_operations=[native(item) for item in prefill[max(0, index - 3) : index]],
                )
            )
    return dict(
        runtime_sha256=runner["runtime_sha256"],
        input_fixture_sha256=runner["input_fixture_sha256"],
        source=str(directory.relative_to(ROOT)),
        source_sha256={
            name: sha(directory / name)
            for name in (
                "whole_layer.json",
                "decode_one_replay.csv",
                "decode_perf_report.csv",
                "prefill_perf_report.csv",
                "summary_commands.json",
                "timing_reconciliation.json",
            )
        },
        policy=runner["precision_policy"],
        whole_layer_windows=whole["whole_layer_windows"],
        decode_groups=groups,
        decode_native_ops=len(decode),
        decode_group_sum_us=kernel_sum,
        decode_key_operations=[native(row) for row in decode if "Matmul" in row["OP Code"] or "Sdpa" in row["OP Code"]],
        prefill_native_groups=grouped_prefill(prefill, whole["prefill_device_us"]),
        prefill_projection_native_audit=list(unique_prefill.values()),
        full_prefill_qkv_producer_audit=qkv_producers,
        prefill_useful_flops_pct=whole["prefill_flops_pct"],
        decode_estimated_dram_pct=whole["decode_dram_pct"],
        decode_estimated_dram_bytes=whole["decode_dram_bytes"],
        peak_basis=whole["peak_basis"],
        sparse_weight_reads=whole["sparse_weight_reads"],
        native_sparse_metadata=sparse,
    )


def build(version):
    historical = read(ROOT / "final_perf_advice_v4_historical.json")
    journal_name = f"actual_optimized_{version}_profile_commands.json"
    journal = read(ROOT / journal_name)
    assert len(journal) == 2 and all(entry["returncode"] == 0 for entry in journal)
    profiles = {str(layer): profile(layer, version) for layer in (0, 5)}
    current = sha(RUNTIME)
    assert all(entry["runtime_sha256"] == current for entry in journal)
    assert all(item["runtime_sha256"] == current for item in profiles.values())
    validation_file = f"validated_{version}_validation_summary.json"
    validation = read(ROOT / validation_file)
    assert validation["runtime_sha256"] == current
    assert "passed" in validation["status"] and not validation.get("pending") and not validation.get("errors")
    reader = read(ROOT / "reader_layer_results.json")
    controls = read(ROOT / "final_perf_advice_controls.json")
    assert not controls["remaining"]
    assert len(controls["qkv_screen"]) == 6 and len(controls["placement_screen"]) == 3
    assert len(controls["shared_grid_screen"]) == 2 and len(controls["hifi2_acceptance"]) == 4
    router_prefill = read(ROOT / "prefill_router_results.json")
    assert not router_prefill["remaining"] and len(router_prefill["results"]) == 4
    assert router_prefill["runtime_sha256"] == current
    qkv_l1 = read(ROOT / "qkv_l1_results.json")
    assert not qkv_l1["pending"] and len(qkv_l1["results"]) == 7
    assert sum(item["returncode"] != 0 for item in qkv_l1["results"]) == 1
    assert next(item for item in qkv_l1["results"] if item["name"] == "m2_producer")["pairs"]["pairs_requested"] == 32
    for layer, fidelity in (("0", "HiFi4"), ("5", "HiFi2")):
        assert profiles[layer]["policy"]["prefill_qkv_projection"]["fidelity"].endswith(fidelity)
    full_qkv = profiles["5"]["full_prefill_qkv_producer_audit"]
    assert len(full_qkv) == 4
    for item in full_qkv:
        row = item["projection"]
        assert row["tensors"]["INPUT_0_MEMORY"].endswith("L1_INTERLEAVED")
        assert row["tensors"]["OUTPUT_0_MEMORY"].endswith("L1_INTERLEAVED")
        assert row["tensors"]["INPUT_0_DATATYPE"] == "FLOAT32"
        assert row["tensors"]["INPUT_1_DATATYPE"] == "BFLOAT8_B"
        assert row["tensors"]["OUTPUT_0_DATATYPE"] == "FLOAT32"
        config = row["parsed_configs"]["config"]
        assert all(
            fragment in config
            for fragment in (
                "M_block_size=2",
                "K_block_size=16",
                "N_block_size=8",
                "compute_with_storage_grid_size=11-8",
            )
        )
        producer = item["preceding_operations"][-1]
        assert producer["tensors"]["OUTPUT_0_MEMORY"].endswith("L1_INTERLEAVED")
        assert "Binary" in producer["op"] and "MUL" in producer["attributes"]
    advice = historical["advice_disposition"]
    for item in advice:
        item[
            "evidence_scope"
        ] = "Historical candidate controls: preserve original source/fixture hashes in the archived v4 audit; selected native rows below are current."
        item["reason"] = (
            item["reason"]
            .replace("final v4", "historical v4")
            .replace("final native input norm itself is6.233 us", "historical v4 native input norm itself was6.233 us")
        )

    def update(role, reason, files, status="completed"):
        item = next((item for item in advice if item["role"] == role), None)
        if item is None:
            item = dict(role=role)
            advice.append(item)
        item.update(
            status=status,
            reason=reason,
            evidence=files,
            evidence_scope="Current controls and explicitly hashed historical policy evidence; see each referenced artifact.",
        )

    update(
        "decode QKV",
        "All legal reader1/2/3 isolated controls and all precision-locked whole-layer integrations completed. K11 reader1 exceeds static L1; legal K1 was executed separately. Reader2 wins the K11 microbenchmark, but current whole-layer DRAM alternatives add17.002/6.908us for sliding/full versus selected interleaved QKV. Includes common padding and all movement. Retain current defaults.",
        ["dram_reader_results.json", "reader_layer_results.json"],
    )
    update(
        "decode output projection",
        "Reader1/2/3 controls use the actual BF16 SDPA/concat producer input, BFP8 weight and FP32 output, not an FP32-input surrogate. Isolated fastest is reader3 sliding/reader1 full; whole-layer alternatives all lose. Full reader1 adds9.544us; sliding reader2/3 add17.712/19.559us. Historical fidelity/subblock trials remain separate.",
        [
            "reader_layer_results.json",
            "dram_reader_results.json",
            "actual_direct_output_commands.json",
            "actual_final_precision_commands.json",
        ],
    )
    update(
        "decode router",
        "The native weights are BF16, so the report's BFP8 label is inaccurate. Independent actual-input fidelity controls nevertheless exist: sliding K44/HiFi2 passes the headline but fails the 512-step stress at position 1459, PCC 0.9943880922; K22/HiFi4 passes all 512. Full K44/LoFi passes headline, 512-step and recorded reuse-window controls. Larger subblocks on two/one cores all pass but are slower: historical sliding 989.866/998.113us versus987.932us, full1076.886/1080.291us versus1076.212us. Retain four cores and the selected fidelity per kind.",
        [
            "router_direct.md",
            "direct_router_commands.json",
            "router_selected_stress_layer0.json",
            "router_repair_hifi4_stress_layer0.json",
            "router_selected_stress_layer5.json",
            "router_subblock_commands.json",
        ],
    )
    update(
        "prefill QKV",
        "Select minimalK8 HiFi4 sliding and M2/K16/N8 HiFi2 full, grid11x8. Full QKV input is produced directly in L1 by the input-norm gamma multiply. HiFi2 wins8/8 pairs (-545.573us) and passes512steps/all291maximum-context rows; sliding HiFi2 instead fails actual long-context position32 atPCC0.9949930133358956 versus same-fixture HiFi4 control0.9951303778312409. Original M4 L1 fails a precise CB overlap; legal M2 and N4 adaptations both pass, with M2 stronger. MatchedM2 placement isolates a coupled L1-input/L1-output benefit in8/8 pairs (-664.9255us). Direct producer versus copiedL1 then wins22/32 (-128.295us) with unchanged PCC. M2L1 grid110 gives4/8 faster and+2.8355us median paired delta, no resolved gain. These are whole-prefill host comparisons including movement. Current native rows prove M2/K16/N8, input/outputL1, FP32/BFP8/FP32 and the preceding gamma-multiply L1 producer; output_mem_config=None inherits input placement. Selected integration correctness is separately hash-bound below.",
        [
            "minimal_selection.json",
            "final_perf_advice_controls.json",
            "minimal_pairs_v6_summary.json",
            "qkv_l1_results.json",
            "prospective_qkv_advice_v7.json",
            validation_file,
        ],
    )
    update(
        "prefill output projection",
        "Sliding retains 2D K16 LoFi. Full selects minimal K8 LoFi after7/8 faster alternating pairs (~204.839us median host-prefill saving in that historical comparison). Actual current minimal-inputL1 passes but screens692.552us slower, so keepDRAMinput/output. This control is separate from the older multicast L1 matrix. Explicit minimal blocks are in native config even when the report says no program_config.",
        ["minimal_selection.json", "prefill_output_results.json", "final_perf_advice_controls.json"],
    )
    update(
        "prefill shared MLP",
        "Native programs configure11x10 with88active blocks. The legal110-active transpose schedule (M3,N14/9,sub3x2/3x1) passes both realheadline controls but screens766.511/726.980us slower sliding/full. It adds9.375%/5.469% padded gate/down tile work. SharedinputL1 also passes but screens205.697/370.497us slower. Retain existing DRAMinput88active schedule; no API blocker or absolute optimum is claimed.",
        ["final_perf_advice_controls.json"],
    )
    update(
        "prefill router",
        "Four phase-matched actual4096/128 controls use32 alternating pairs each, exact FP32/BF16/FP32 precision and locked11x10/K8/perM4/perN1/sub4x1 geometry. HiFi2 changes only fidelity; producerL1 changes only the existing scaled-input multiply output, retaining its fused scalar-root operation and explicitDRAMprojection output. All pass HF/cache guards. Sliding HiFi2 has20/32 faster and-67.093us paired median, producerL1 has16/32 and+3.8725us; full HiFi2 has10/32 and+111.637us, producerL1 has20/32 and-32.762us. Paired IQRs cross zero in everycase; apparent savings are at most0.031% of wholeprefill and unresolved against observed variation. Retain HiFi4/DRAM bothkinds. Compute-object strings are opaque; hashed source and selected object identities prove requested fidelity, while current native rows prove the retained config. No historical decode result substitutes for these prefill controls.",
        ["prefill_router_results.json", "prefill_router_v8/commands.json", validation_file],
    )
    update(
        "decode RoPE producer layout",
        "Caller-provided row-major tables avoid full-table untilize. Selected-row slice/repeat/tilize remains. Historical TILE compatibility and the current validation source-delta chain are retained without relabeling old runs.",
        ["actual_decode_rope_row_major_layer0.json", "actual_decode_rope_row_major_layer5.json", validation_file],
    )
    update(
        "full decode input norm",
        "Sharded normalization remains selected at all hidden-width sites. The actual headline/stress controls establish its precision/latency choice; current native rows and current validation establish the integrated path.",
        ["full_all_norm_headline.json", "full_all_norm_stress.json", validation_file],
    )
    historical_audits = {
        key: {field: value for field, value in item.items() if field != "operations"}
        for key, item in historical["operator_audits"].items()
    }
    return dict(
        generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope=f"Current {version} native advice audit; CPU artifact parsing only",
        status="complete: current native profiles and applicable advice controls audited; independent stage review remains separate",
        current_runtime_sha256=current,
        timing_basis="Decode groups sum one sampled replay's native Device Time in microseconds; percentages use that sampled kernel sum. Whole-layer128-replay spans include gaps. Prefill percentages use complete prefill device span. Warm host comparisons remain separate.",
        profiles=profiles,
        native_profile_command_journal=evidence(journal_name),
        advice_disposition=advice,
        historical_v4=dict(
            json=evidence("final_perf_advice_v4_historical.json"),
            markdown=evidence("final_perf_advice_v4_historical.md"),
            runtime_sha256=historical["current_runtime_sha256"],
            operator_audits=historical_audits,
            note="Six operator controls plus their selected reference rows are historical; none is relabeled as a current native run.",
        ),
        correctness_scope=dict(
            **evidence(validation_file),
            status=validation["status"],
            runtime_sha256=current,
            basis=validation.get("validation_basis", "See validation manifest"),
        ),
        reader_controls=dict(
            **evidence("reader_layer_results.json"),
            runtime_sha256=reader["runtime_sha256"],
            isolated_runtime_sha256=reader["isolated_runtime_sha256"],
            decision=reader["reader_decision"],
        ),
        current_prefill_advice_controls=evidence("final_perf_advice_controls.json"),
        current_prefill_router_controls=evidence("prefill_router_results.json"),
        current_full_qkv_l1=dict(
            **evidence("qkv_l1_results.json"),
            scope="Seven controls on frozen daa82: original capacity failure plus six successful legal/boundary controls. Historical trial hashes are preserved; current integration is proved by the selected native producer audit and validation manifest.",
            current_native_producers=full_qkv,
        ),
        indexed_profiler_accounting=historical["indexed_profiler_accounting"],
        parser_limitations=dict(
            minimal_config="tt-perf-report1.3.0 perf_report.py1374 only parses program_config/in0_block_w/out_subblock; minimal uses config/K_block_size/subblock. Native explicit configs are retained above. Raw CSV unchanged.",
            active_cores="Configured grid and active workblocks differ; shared prefill already configures110. The110-active redistribution is a separate measured schedule, not a no-op grid setting.",
            source_sha256=sha(Path("python_env/lib/python3.10/site-packages/tt_perf_report/perf_report.py")),
        ),
        remaining_evidence=[],
    )


def markdown(a):
    p = a["profiles"]
    lines = [
        "# Current native profiler advice audit",
        "",
        f"Runtime SHA256: `{a['current_runtime_sha256']}`.",
        "",
        a["timing_basis"],
        "",
        "| Decode group | Sliding µs | Sliding kernel % | Full µs | Full kernel % |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, _, _ in BOUNDS[0]:
        left, right = p["0"]["decode_groups"][name], p["5"]["decode_groups"][name]
        lines.append(
            f"| {name} | {left['device_us']:.3f} | {left['sampled_kernel_pct']:.2f} | {right['device_us']:.3f} | {right['sampled_kernel_pct']:.2f} |"
        )
    lines += ["", "| Whole-layer native window | Sliding µs | Full µs |", "| --- | ---: | ---: |"]
    for phase in ("prefill", "decode"):
        lines.append(
            f"| {phase} | {p['0']['whole_layer_windows'][phase]['device_us']:.3f} | {p['5']['whole_layer_windows'][phase]['device_us']:.3f} |"
        )
    lines += [
        "",
        f"Useful prefill work reaches {p['0']['prefill_useful_flops_pct']:.3f}% / {p['5']['prefill_useful_flops_pct']:.3f}% of the stated single-P300 theoretical peak (sliding/full). Estimated decode DRAM rates reach {p['0']['decode_estimated_dram_pct']:.3f}% / {p['5']['decode_estimated_dram_pct']:.3f}%. These are model estimates, not hardware utilization counters; extra union-expert work is excluded from the useful FLOPs numerator.",
        "",
        "| Prefill sparse group | Sliding native µs (% of span) | Full native µs (% of span) |",
        "| --- | ---: | ---: |",
    ]
    for role, fragment in (("Gate/up", "2816 x 1408"), ("Down", "704 x 2816")):
        groups = [
            next(
                group
                for group in p[layer]["prefill_native_groups"]
                if "SparseMatmul" in group["op"] and fragment in group["op"]
            )
            for layer in ("0", "5")
        ]
        lines.append(
            f"| {role} | {groups[0]['device_us']:.3f} ({groups[0]['whole_prefill_span_pct']:.2f}%) | {groups[1]['device_us']:.3f} ({groups[1]['whole_prefill_span_pct']:.2f}%) |"
        )
    lines += [
        "",
        "The JSON joins every key operation to its raw native attributes, input/output dtypes, memory layout, configured blocks and kernel sources. It verifies all indexed sparse rows use UINT16 eight-element indices, E8 output and E128 resident weights. Decoder traffic accounts for8 executed experts; prefill active unions remain unknown rather than assuming8 per32-token chunk.",
        "",
        "The remaining small blocks have explicit bounds and controls. Compact expert mixing has eight logical routes padded to one K tile, so K1 is the only divisor of its actual tiled K; expanded and compact weighted-reduce alternatives were both executed and lost. Expert gate/up's44 output tiles distribute one per44 workers, giving subblock1×1; the22-core/subblock1×2 and separate gate/up families were measured, including the native operator controls below. Router subblocks1×2/1×4 reduce the active worker count from four to two/one and were measured slower.",
        "",
        "## Prefill native projection audit",
        "",
        "| Layer | Operation | Calls | Native µs | Fidelity / dtype | Explicit program |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for layer, item in p.items():
        for group in item["prefill_projection_native_audit"]:
            row = group["first_native_row"]
            if "Sparse" in row["op"]:
                continue
            config = row["parsed_configs"].get("config", row["parsed_configs"].get("program_config", ""))
            lines.append(
                f"| {layer} | {row['op']} | {group['invocations']} | {group['device_us']:.3f} | {row['math_fidelity']} | `{config}` |"
            )
    lines += [
        "",
        "`tt-perf-report` does not parse minimal matmul’s `config=MinimalMatmulConfig` fields, so its missing-program warning is false. This does not dismiss independent grid, fidelity or placement recommendations; their controls are recorded below.",
        "",
        "The [full QKV L1 controls](qkv_l1_results.md) retain the original M4 capacity failure, both legal M2/N4 adaptations, matched M2 placement and grid comparisons, a65-token boundary, and32 producer/copy pairs. Current native rows additionally verify each full-QKV input comes directly from an L1 gamma-multiply output; the selected path adds no DRAM-to-L1 copy.",
        "",
        "## Advice disposition",
    ]
    for item in a["advice_disposition"]:
        lines += [
            "",
            f"**{item['role']} — {item['status']}.** {item['reason']} Evidence: "
            + ", ".join(f"`{name}`" for name in item["evidence"])
            + ".",
        ]
    lines += [
        "",
        "## Historical expert operator controls",
        "",
        f"These six controls and their selected comparison rows ran on v4 `{a['historical_v4']['runtime_sha256']}`. They establish the measured topology/config decisions; they are not current-profile durations. Selected rows used the headline workload; alternatives used separately hashed 4096/1 inputs. End-to-end geometry journals in the archive establish the whole-layer selection.",
        "",
        "| Historical policy | Layer | Expert region µs | Gate/up µs | Down µs |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, control in a["historical_v4"]["operator_audits"].items():
        assert control["status"] == "complete"
        costs = control["by_operation_us"]
        lines.append(
            f"| {name.rsplit('_layer', 1)[0]} | {name[-1]} | {control['device_us']:.3f} | {costs.get('sparse_gate', 0):.3f} | {costs.get('sparse_down', 0):.3f} |"
        )
    lines += [
        "",
        "## Evidence scope",
        "",
        "The [v4 archive](final_perf_advice_v4_historical.md) retains the six expert operator controls and historical dtype/topology candidates with their original hashes. Current selected group costs above come only from the current native rows. The [reader audit](reader_layer_results.md) links28 isolated legal cases, two precise K11 reader1 capacity failures and22 complete whole-layer controls; it retains the formatting provenance incident. All22 integrations pass; no meaningful whole-layer reader winner was found. Full shared-down reader1’s0.047us apparent gain is within sample overlap, while reader3 is0.046us slower.",
        "",
        f"Correctness basis: `{a['correctness_scope']['artifact']}`; {a['correctness_scope']['basis']}",
        "",
        "Remaining advice evidence: "
        + (
            "; ".join(a["remaining_evidence"])
            or "none among the audited recommendations; stage review remains separate"
        )
        + ".",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="v8")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        profiles = {layer: profile(layer, args.version) for layer in (0, 5)}
        print(
            json.dumps(
                {
                    layer: dict(native_ops=item["decode_native_ops"], sampled_kernel_us=item["decode_group_sum_us"])
                    for layer, item in profiles.items()
                },
                indent=2,
            )
        )
        return
    result = build(args.version)
    payload = (json.dumps(result, indent=2) + "\n").encode()
    (ROOT / "final_perf_advice.json").write_bytes(payload)
    (ROOT / "final_perf_advice.json.gz").write_bytes(gzip.compress(payload, mtime=0))
    (ROOT / "final_perf_advice.md").write_text(markdown(result))


if __name__ == "__main__":
    main()
