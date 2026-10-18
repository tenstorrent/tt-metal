"""Verify and index current-source stage-5 artifacts; no device access."""

import hashlib
import json
from pathlib import Path
import torch
from models.autoports.qwen_qwen3_8_flash_next.tests.metrics import pcc_metrics

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/optimized_multichip_decoder"


def read(name):
    return json.loads((DOC / name).read_text())


def main():
    torch.set_num_threads(8)
    source = hashlib.sha256((ROOT / "tt/multichip_decoder.py").read_bytes()).hexdigest()
    suites = []
    for suite, batches in [("stress", [1]), ("batches", [2, 32]), ("geometry_boundary", [1])]:
        for layer in [0, 1, 3]:
            for batch in batches:
                tag = (
                    "s5_geometry4_boundary"
                    if suite == "geometry_boundary"
                    else "s5_final4_" + suite
                )
                path = f"{tag}/mesh_l{layer}_b{batch}_manifest.json"
                d = read(path)
                assert d["status"] == "passed" and d["source_sha256"] == source, path
                assert d["env"]["TT_METAL_TRACE_ALLOC_TRACKING"] == "1"
                assert d["env"]["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"] == "0"
                assert d["captures"] == d["releases"] and min(m["pcc"] for m in d["metrics"]) >= 0.995
                if batch == 1:
                    for length, memory in d["prefill_output_placements"].items():
                        assert ("DRAM" if int(length) > 512 else "L1") in memory, (path, length, memory)
                suites.append(
                    dict(
                        artifact=path,
                        layer=layer,
                        batch=batch,
                        comparisons=len(d["metrics"]),
                        minimum_pcc=min(m["pcc"] for m in d["metrics"]),
                        replays=d["replays"],
                        captures=d["captures"],
                        releases=d["releases"],
                    )
                )
    contexts = []
    baseline = json.loads((ROOT / "doc/multichip_decoder/final_evidence.json").read_text())
    stress_before_after = []
    for layer in (0, 1, 3):
        before = next(r for r in baseline["correctness"] if r["layer"] == layer and r["comparisons"] == 429)
        after = next(r for r in suites if r["layer"] == layer and r["comparisons"] == 429)
        original = json.loads((ROOT / "doc/multichip_decoder" / before["artifact"]).read_text())
        assert min(r["pcc"] for r in original["metrics"]) == before["minimum_pcc"]
        stress_before_after.append(
            dict(
                layer=layer,
                before_artifact="../multichip_decoder/" + before["artifact"],
                after_artifact=after["artifact"],
                before_minimum_pcc=before["minimum_pcc"],
                after_minimum_pcc=after["minimum_pcc"],
                threshold=0.995,
            )
        )
    for layer in [0, 1, 3]:
        path = f"s5_final4_context/mesh_l{layer}.json"
        d = read(path)
        assert d["status"] == "passed" and d["source_sha256"] == source
        assert d["context"] == d["validated_tokens"] == 262144
        contexts.append(
            dict(
                artifact=path,
                layer=layer,
                minimum_pcc=d["minimum_pcc"],
                comparisons=len(d["records"]),
                replays=d["trace_replays"],
            )
        )
    stack = read("s5_final4_stack/stack_contract.json")
    assert stack["status"] == "passed" and stack["source_sha256"] == source
    stack513 = read("s5_final4_stack513/stack_contract.json")
    assert stack513["status"] == "passed" and stack513["source_sha256"] == source
    perf = []
    for layer in [0, 1, 3]:
        for length in [128, 33, 2053]:
            before = f"s5_before_l{layer}" if length == 128 else f"s5_before_mesh_s{length}_l{layer}"
            after = f"s5_final4_mesh_l{layer}" if length == 128 else f"s5_final4_mesh_s{length}_l{layer}"
            a, b = read(before + "/timing.json"), read(after + "/timing.json")
            assert (
                b["source_sha256"] == source
                and a["source_sha256"] == "f292934e2268db39868676611f608dcba52ad2bc3dc309ebd3e7649b89c8d143"
            )
            assert a["tracking"] == b["tracking"] == "0"
            for field in [
                "recorded",
                "recorded_tag",
                "prefill_length",
                "decode_position",
                "advance_position",
                "chunk_size",
            ]:
                assert a["policy"][field] == b["policy"][field], (field, before, after)
            aa, bb = torch.load(a["outputs"], weights_only=False), torch.load(b["outputs"], weights_only=False)
            metrics = {k: pcc_metrics(aa[k], bb[k]) for k in aa}
            assert min(v["pcc"] for v in metrics.values()) >= 0.995
            perf.append(
                dict(
                    layer=layer,
                    length=length,
                    before=before + "/timing.json",
                    after=after + "/timing.json",
                    before_prefill_ms=a["median_prefill_host_ms"],
                    after_prefill_ms=b["median_prefill_host_ms"],
                    before_decode_ms=a["median_traced_decode_ms"],
                    after_decode_ms=b["median_traced_decode_ms"],
                    paired_metrics=metrics,
                )
            )
    ownership = []
    for n in [
        "s5_final4_attention_qsa_b1_s511.json",
        "s5_final4_attention_qsa_b1_s513.json",
        "s5_final4_attention_qsa_b1_s1025.json",
        "s5_final4_ownership_qsa_b2_s33.json",
    ]:
        d = read(n)
        assert d["status"] == "passed" and d["source_sha256"] == source
        ownership.append(dict(artifact=n, metrics=d["metrics"]))
    accounting = read("s5_final4_performance_accounting.json")
    assert all(r["source_sha256"] == source for r in accounting["runs"])
    capacity = read("capacity_probe.json")
    assert capacity["status"] == "capacity_pass"
    runtime = read("runtime_audit.json")
    assert runtime["status"] == "pass" and not runtime["violations"]
    policy = read("profile_policy_audit.json")
    assert policy["status"] == "passed"
    assert all(r["source_sha256"] == source for r in policy["runs"])
    native = read("triage_norm1/final_native_provenance.json")
    for name, digest in native.items():
        if name.endswith(".cpp"):
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest

    prefill_traces = []
    for layer in (0, 1, 3):
        for length in (128, 2053):
            checked_path = f"s5_final4_trace_watcher_s{length}_l{layer}/timing.json"
            measured_path = f"s5_final4_trace_s{length}_l{layer}/timing.json"
            checked, measured = read(checked_path), read(measured_path)
            assert checked["source_sha256"] == measured["source_sha256"] == source
            assert checked["tracking"] == "1" and checked["skip_program_cache_tracking"] == "0"
            assert checked["watcher"] == "10" and measured["tracking"] == "0"
            assert checked["prefill_trace_captures"] == checked["prefill_trace_releases"] == 1
            assert checked["prefill_trace_replays"] == 13 and len(checked["prefill_trace_checks"]) == 6
            assert all(r["bitwise_equal"] and r["appended_decode_equal"] for r in checked["prefill_trace_checks"])
            control = read(f"s5_prefill_control_s{length}_l{layer}/timing.json")
            aa, bb = torch.load(control["outputs"], weights_only=False), torch.load(
                measured["outputs"], weights_only=False
            )
            metrics = {k: pcc_metrics(aa[k], bb[k]) for k in aa}
            assert min(v["pcc"] for v in metrics.values()) >= 0.995
            prefill_traces.append(
                dict(
                    layer=layer,
                    length=length,
                    qualification=checked_path,
                    timing=measured_path,
                    prefill_ms=measured["median_prefill_host_ms"],
                    capture_ms=measured["prefill_capture_ms"],
                    bitwise_control_match=all(torch.equal(aa[k], bb[k]) for k in aa),
                    paired_metrics=metrics,
                )
            )
    repeats = []
    for layer, length in ((1, 33), (3, 128)):
        for repeat in (1, 2):
            before = f"s5_before_repeat{repeat}_s{length}_l{layer}/timing.json"
            after = f"s5_final4_repeat{repeat}_s{length}_l{layer}/timing.json"
            a, b = read(before), read(after)
            assert b["source_sha256"] == source
            aa, bb = torch.load(a["outputs"], weights_only=False), torch.load(b["outputs"], weights_only=False)
            metrics = {k: pcc_metrics(aa[k], bb[k]) for k in aa}
            assert min(v["pcc"] for v in metrics.values()) >= 0.995
            repeats.append(
                dict(
                    layer=layer,
                    length=length,
                    repeat=repeat,
                    before=before,
                    after=after,
                    before_prefill_ms=a["median_prefill_host_ms"],
                    after_prefill_ms=b["median_prefill_host_ms"],
                    metrics=metrics,
                )
            )

    result = dict(
        status="all_artifact_gates_passed",
        source_sha256=source,
        suites=suites,
        stress_before_after=stress_before_after,
        contexts=contexts,
        stack="s5_final4_stack/stack_contract.json",
        stack513="s5_final4_stack513/stack_contract.json",
        ownership=ownership,
        performance=perf,
        accounting="s5_final4_performance_accounting.json",
        capacity="capacity_probe.json",
        runtime="runtime_audit.json",
        profile_policy="profile_policy_audit.json",
        native_provenance="triage_norm1/final_native_provenance.json",
        prefill_traces=prefill_traces,
        paired_repeats=repeats,
    )
    review = DOC / "stage_review_final.md"
    if review.exists() and "Verdict: clean-pass" in review.read_text() and source in review.read_text():
        result.update(
            status="optimized_multichip_decoder_validated_clean_pass",
            stage_review="stage_review_final.md",
            stage_review_sha256=hashlib.sha256(review.read_bytes()).hexdigest(),
        )
    (DOC / "final_evidence.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Matched final-default performance",
        "",
        "B1, true1×4 mesh; median of five warmed prefill calls / five10-replay traced decode windows.",
        "128 uses recorded_inputs;33/2053 use recorded_inputs_long. Both sides use identical inputs and advancing positions.",
        "",
        "| Layer | Tokens | Before prefill ms | After prefill ms | Before decode ms | After decode ms | Decode reduction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in perf:
        lines.append(
            f"| {r['layer']} | {r['length']} | {r['before_prefill_ms']:.4f} | {r['after_prefill_ms']:.4f} | {r['before_decode_ms']:.4f} | {r['after_decode_ms']:.4f} | {100*(1-r['after_decode_ms']/r['before_decode_ms']):.1f}% |"
        )
    lines += [
        "",
        "Final-default caller-owned prefill replay (recorded_inputs_long, separate capture cost):",
        "",
        "| Layer | Tokens | Warmed traced prefill ms | Capture ms |",
        "|---|---:|---:|---:|",
    ]
    for r in prefill_traces:
        lines.append(f"| {r['layer']} | {r['length']} | {r['prefill_ms']:.4f} | {r['capture_ms']:.4f} |")
    lines += [
        "",
        "Paired repeats retained to resolve the elevated final2 short-prefill host timings:",
        "",
        "| Layer | Tokens | Repeat | Before prefill ms | Final prefill ms |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in repeats:
        lines.append(
            f"| {r['layer']} | {r['length']} | {r['repeat']} | {r['before_prefill_ms']:.4f} | {r['after_prefill_ms']:.4f} |"
        )
    (DOC / "final_performance.md").write_text("\n".join(lines) + "\n")
    print("STAGE5_ARTIFACT_GATES_PASS", source, len(suites), len(perf))


if __name__ == "__main__":
    main()
