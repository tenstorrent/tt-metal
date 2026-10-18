"""Build the final stage index from qualified manifests and paired output files."""

import hashlib
import json
from pathlib import Path
import torch
from models.autoports.qwen_qwen3_8_flash_next.tests.metrics import pcc_metrics

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/multichip_decoder"


def read(name):
    return json.loads((DOC / name).read_text())


def main():
    torch.set_num_threads(8)
    source = hashlib.sha256((ROOT / "tt/multichip_decoder.py").read_bytes()).hexdigest()
    baseline_source = hashlib.sha256((ROOT / "tt/optimized_decoder.py").read_bytes()).hexdigest()
    reuse = read("source_reuse.json")
    assert reuse["status"] == "passed" and reuse["selected_source_sha256"] == source
    prior_source = reuse["prior_source_sha256"]
    suites = []
    for suite, batches in [("stress", [1]), ("recorded", [1]), ("batches", [2, 32]), ("watcher", [1])]:
        for layer in [0, 1, 3]:
            for batch in batches:
                if layer == 3 and batch == 1:
                    if suite != "watcher":
                        continue  # The final watcher suite also contains the expanded recorded stress.
                    name = "selected_v6_watcher/mesh_l3_b1_manifest.json"
                    expected = source
                else:
                    name = f"selected_v4_{suite}/mesh_l{layer}_b{batch}_manifest.json"
                    expected = prior_source
                d = read(name)
                assert d["status"] == "passed" and d["source_sha256"] == expected, name
                assert d["env"]["TT_METAL_TRACE_ALLOC_TRACKING"] == "1"
                assert d["env"]["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"] == "0"
                suites.append(
                    dict(
                        artifact=name,
                        qualified_source_sha256=d["source_sha256"],
                        retained_equivalence="source_reuse.json" if expected == prior_source else None,
                        layer=layer,
                        batch=batch,
                        comparisons=len(d["metrics"]),
                        minimum_pcc=min(x["pcc"] for x in d["metrics"]),
                        captures=d["captures"],
                        releases=d["releases"],
                        replays=d["replays"],
                    )
                )
    contexts = []
    for layer in [0, 1, 3]:
        name = f"selected_v{6 if layer == 3 else 4}_context/mesh_l{layer}.json"
        d = read(name)
        assert d["status"] == "passed" and d["source_sha256"] == (source if layer == 3 else prior_source), name
        assert d["context"] == 262144 and d["validated_tokens"] == 262144
        contexts.append(
            dict(
                artifact=name,
                layer=layer,
                minimum_pcc=d["minimum_pcc"],
                comparisons=len(d["records"]),
                replays=d["trace_replays"],
            )
        )
    ownership = []
    for name in [
        "selected_v4_attention_gdn_b1_s513.json",
        "selected_v4_attention_gdn_b1_s1025.json",
        "selected_v6_attention_matched_qsa_b1_s511.json",
        "selected_v6_attention_matched_qsa_b1_s513.json",
        "selected_v6_attention_matched_qsa_b1_s1025.json",
        "selected_v4_ownership_gdn_b2_s33.json",
        "selected_v4_ownership_qsa_b2_s33.json",
        "selected_v4_rank3_gdn_b1_s513.json",
    ]:
        d = read(name)
        assert d["source_sha256"] == (source if name.startswith("selected_v6") else prior_source)
        assert min(m["pcc"] for m in d["metrics"].values()) >= 0.995
        ownership.append(
            dict(
                artifact=name,
                metrics=d["metrics"],
                local_shape=d["local_shape"],
                local_dtype=d["local_dtype"],
                reference_policy=(
                    "matched BF8-prefill/converted-BF4-decode single attention; original comparisons retained"
                    if "matched" in name
                    else "unchanged optimized single attention"
                ),
                three_way=d.get("comparisons"),
            )
        )
    stack = read("selected_v6_stack/stack_contract.json")
    assert stack["status"] == "passed" and stack["source_sha256"] == source
    performance = []
    for layer in [0, 1, 3]:
        for length, suffix in [(128, ""), (33, "_s33"), (2053, "_s2053")]:
            single_name = f"selected_v3_single{suffix}_l{layer}/timing.json"
            mesh_name = f"selected_v{6 if layer == 3 else 4}_mesh{suffix}_l{layer}/timing.json"
            if layer == 3 and length != 128:
                single_name = f"selected_v5_single{suffix}_l{layer}/timing.json"
            single, mesh = read(single_name), read(mesh_name)
            assert (
                mesh["source_sha256"] == (source if layer == 3 else prior_source)
                and mesh["tracking"] == single["tracking"] == "0"
            )
            assert single["source_sha256"] == baseline_source
            assert single["policy"]["recorded"] == mesh["policy"]["recorded"]
            assert single["policy"].get("recorded_tag", "recorded_inputs") == mesh["policy"].get(
                "recorded_tag", "recorded_inputs"
            )
            for field in [
                "batch",
                "prefill_length",
                "chunk_size",
                "position",
                "advance_position",
                "final_position",
                "decode_replays",
            ]:
                assert single[field] == mesh[field], (mesh_name, field)
            reference = torch.load(single["outputs"], weights_only=True)
            output = torch.load(mesh["outputs"], weights_only=True)
            pcc = {k: pcc_metrics(reference[k], v)["pcc"] for k, v in output.items()}
            assert min(pcc.values()) >= 0.995, (mesh_name, pcc)
            row = dict(layer=layer, prefill_length=length, single=single_name, mesh=mesh_name, pcc=pcc)
            for phase, field in [("prefill", "median_prefill_host_ms"), ("decode", "median_traced_decode_ms")]:
                ratio = single[field] / mesh[field]
                row[phase] = dict(single_ms=single[field], mesh_ms=mesh[field], speedup=ratio, efficiency=ratio / 4)
            performance.append(row)
    audit = read("runtime_audit.json")
    assert audit["status"] == "pass" and audit["sha256"]["tt/multichip_decoder.py"] == source
    capacity = read("capacity_probe.json")
    assert capacity["status"] == "capacity_pass"
    assert capacity["plan"] == read("memory_capacity_plan.json")
    profiles = read("profile_policy_audit.json")
    assert profiles["status"] == "passed" and len(profiles["runs"]) == 12
    assert all(r["source_sha256"] == source for r in profiles["runs"] if r["tag"].startswith("selected_v6"))
    assert all(r["source_sha256"] == prior_source for r in profiles["runs"] if r["tag"].startswith("selected_v4"))
    accounting = read("selected_v6_performance_accounting.json")
    assert all(r["source_sha256"] == (source if r["layer"] == 3 else prior_source) for r in accounting["runs"])
    result = dict(
        status="validation_pass_pending_independent_review",
        source_sha256=source,
        retained_evidence_equivalence="source_reuse.json",
        baseline_sha256=baseline_source,
        mesh=[1, 4],
        fabric="FABRIC_1D",
        supported_context=262144,
        capability_reduction=None,
        correctness=suites,
        full_context=contexts,
        ownership=ownership,
        stack=dict(
            artifact="selected_v6_stack/stack_contract.json", minimum_pcc=min(x["pcc"] for x in stack["metrics"])
        ),
        performance=performance,
        efficiency_definition="single-chip latency / mesh latency / 4",
        memory=dict(
            plan="memory_capacity_plan.json",
            probe="capacity_probe.json",
            allocated_bytes_per_device=8 * capacity["final"]["total_bytes_allocated_per_bank"],
            remaining_bytes_per_device=8 * capacity["final"]["total_bytes_free_per_bank"],
        ),
        runtime_audit="runtime_audit.json",
        accounting="selected_v6_performance_accounting.json",
        profile_policy_audit="profile_policy_audit.json",
        watcher_scope="Worker watcher=10; Ethernet instrumentation disabled after ACTIVE_ETH29040>26624-byte build limit. This is not full Ethernet-watcher coverage.",
        precision_limitations="QSA B1 BF8 prefill and converted BF4/LoFi decode are qualified on actual HF-derived activations and adjacent decoder-stack inputs. This phase policy fails correlated-stream synthetic stress in its historical advice implementation; actual-input expanded stress, stack and full context pass. This is not a claim of arbitrary-input numerical equivalence. Attention-only decode differs from unchanged attention (about0.976–0.987 PCC in direct component cases), reproduced exactly by the independent single-chip phase-matched control; whole-decoder PCC remains≥0.995. See qsa_bfp4_autofix.md and qsa_component_precision_autofix.md.",
        warnings="Unknown B850M-C motherboard uses bus_id as tray_id; explicit mesh mapping is validated. AICLK1343MHz is within firmware tolerance of1350MHz. Neither warning is a model runtime fallback.",
    )
    review = DOC / "stage_review_final.md"
    if review.exists() and "Verdict: clean-pass" in review.read_text() and source in review.read_text():
        result["status"] = "multichip_decoder_validated_clean_pass"
        result["stage_review"] = dict(artifact=review.name, sha256=hashlib.sha256(review.read_bytes()).hexdigest())
    (DOC / "final_evidence.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Final matched performance",
        "",
        "Warmed, uninstrumented timings in milliseconds. Decode uses 50 trace replays with advancing positions. Efficiency is speedup / 4. S128 uses recorded inputs; QSA S33/S2053 use expanded HF-derived inputs, while GDN S33/S2053 use matched seeded inputs. GDN measurements retain the unchanged code path proved by source_reuse.json. All paired outputs pass PCC ≥ 0.995. Source and exact arguments are in the linked timing JSONs.",
        "",
        "| Layer | Prefill tokens | Single prefill | Mesh prefill | Speedup / efficiency | Single decode | Mesh decode | Speedup / efficiency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in performance:
        p, d = row["prefill"], row["decode"]
        lines.append(
            f"| {row['layer']} | {row['prefill_length']} | [{p['single_ms']:.4f}]({row['single']}) | [{p['mesh_ms']:.4f}]({row['mesh']}) | {p['speedup']:.3f}× / {p['efficiency']:.1%} | {d['single_ms']:.4f} | {d['mesh_ms']:.4f} | {d['speedup']:.3f}× / {d['efficiency']:.1%} |"
        )
    (DOC / "final_performance.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            dict(status=result["status"], source=source, correctness_rows=len(suites), performance=performance),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
