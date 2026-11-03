# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Offline reconciliation of final optimized full-model evidence (no TT import)."""
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/optimized_full_model"


def main():
    before, after = [json.loads((DOC / f"performance_{tag}.json").read_text()) for tag in ("before", "after")]
    rows = []
    for tag, data in (("before", before), ("after", after)):
        for item in data["prefill_lengths"]:
            length = item["logical_length"]
            remain = length
            buckets = []
            while remain:
                size = max(n for n in (1, 32, 64, 128, 256, 512, 1024, 2048, 4096) if n <= remain)
                buckets.append(size)
                remain -= size
            rows.append(
                dict(
                    phase=tag,
                    logical_length=length,
                    buckets=str(buckets),
                    chunks=len(buckets),
                    first_observation_ms=item["ttft_ms"][0],
                    warmed_ttft_ms=statistics.mean(item["ttft_ms"][1:]),
                )
            )
    with (DOC / "prefill_performance.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    floor = 3 * 0.178316 + 45 * 0.317480
    sampler = json.loads((DOC / "greedy_strategies.json").read_text())["greedy_strategies"]
    split = statistics.mean(r["latency_ms"] for r in sampler if not r["force_argmax"])
    head = next(
        r["ms"]
        for r in json.loads((DOC / "head_geometry.json").read_text())
        if r["name"] == "dram_width32768_cores8_block1_workers4_pipelineTrue"
    )
    queued = after["queued_token_out"]["decode_ms"]
    summary = dict(
        workload=dict(
            profile="single_user_decode",
            prompt_len=128,
            gen_len=128,
            batch=1,
            layers=48,
            context_allocation=524288,
            mesh=[1, 4],
        ),
        ttft_ms=after["token_out"]["ttft_ms"],
        decode_ms_per_token_e2e=queued,
        decode_ms_per_token_device=None,
        device_time_scope="Full48 profiling deliberately excluded; actual reduced-profile timing is recorded separately below.",
        layer_stack_lower_bound_ms=floor,
        layer_stack_lower_bound_tps=1000 / floor,
        layer_stack_source="doc/optimized_multichip_decoder/performance.csv, dense3*0.178316 + MoE45*0.317480; optimistic short-capacity floor",
        terminal_head_ms=head,
        canonical_split_sampler_ms=split,
        overhead_above_floor_and_terminal_ms=queued - floor - head - split,
        overhead_fraction_above_floor_and_terminal=queued / (floor + head + split) - 1,
        named_limitations=[
            "Inherited long-capacity attention grid/fidelity differs from optimistic short-capacity layer floor",
            "Indexed active experts contain many small operations; stored active-weight traffic is an optimistic bandwidth floor",
            "Caller-read decode is separately measured; queued token-out has no per-token host read/sync",
            "Prefill tail tokens retain original sequential paged-update arithmetic; tracing removes host dispatch but not mathematical work",
        ],
    )
    profile = json.loads((DOC / "profile.json").read_text())
    provenance = json.loads((DOC / "tracy/final/mova_moe/provenance.json").read_text())
    raw = Path(provenance["source_csv"])
    with raw.open() as f:
        ops = list(csv.DictReader(f))
    windows = {}
    runtime = []
    for phase in ("decode", "prefill"):
        begin = next(i for i, r in enumerate(ops) if r["OP CODE"] == f"PERF_{phase.upper()}")
        end = next(i for i, r in enumerate(ops) if i > begin and r["OP CODE"] == f"PERF_{phase.upper()}_END")
        window = ops[begin + 1 : end]
        ranks = []
        for rank in range(4):
            rr = [r for r in window if r["DEVICE ID"] == str(rank) and r["DEVICE FW DURATION [ns]"]]
            rate = statistics.median(
                (float(r["DEVICE FW END CYCLE"]) - float(r["DEVICE FW START CYCLE"]))
                / float(r["DEVICE FW DURATION [ns]"])
                for r in rr
                if float(r["DEVICE FW DURATION [ns]"])
            )
            ranks.append(
                dict(
                    rank=rank,
                    elapsed_ms=(
                        max(float(r["DEVICE FW END CYCLE"]) for r in rr)
                        - min(float(r["DEVICE FW START CYCLE"]) for r in rr)
                    )
                    / rate
                    / 1e6,
                    kernel_sum_ms=sum(float(r["DEVICE KERNEL DURATION [ns]"] or 0) for r in rr) / 1e6,
                )
            )
        windows[phase] = dict(
            host_ms=profile[f"profile_{phase}_host_ms"],
            ranks=ranks,
            max_rank_elapsed_ms=max(r["elapsed_ms"] for r in ranks),
            report=provenance["windows"][phase],
        )
        assert windows[phase]["max_rank_elapsed_ms"] < windows[phase]["host_ms"] * 1.1
        for row in window:
            if row["DEVICE ID"] == "0" and row["CORE COUNT"]:
                runtime.append(
                    dict(
                        phase=phase,
                        **{
                            k: row[k]
                            for k in [
                                "OP CODE",
                                "GLOBAL CALL COUNT",
                                "ATTRIBUTES",
                                "MATH FIDELITY",
                                "CORE COUNT",
                                "DEVICE KERNEL DURATION [ns]",
                                "INPUT_0_DATATYPE",
                                "INPUT_1_DATATYPE",
                                "OUTPUT_0_DATATYPE",
                                "INPUT_0_MEMORY",
                                "INPUT_1_MEMORY",
                                "OUTPUT_0_MEMORY",
                            ]
                        },
                    )
                )
    # Derive one-read active projection traffic from the measured runtime rows.
    # Full capacity selects k_chunk1024 even for this short decode prefix.
    begin = next(i for i, r in enumerate(ops) if r["OP CODE"] == "PERF_DECODE")
    end = next(i for i, r in enumerate(ops) if r["OP CODE"] == "PERF_DECODE_END")
    matmuls = [r for r in ops[begin:end] if r["DEVICE ID"] == "0" and "Matmul" in r["OP CODE"]]
    roles = [
        "dense_attention",
        "dense_output",
        "dense_gate_up",
        "dense_down",
        "moe_attention",
        "value_experts",
        "moe_output",
        "shared_gate_router",
        "shared_down",
        "experts_gate_up",
        "experts_down",
        "head0",
        "head1",
    ]
    assert len(matmuls) == len(roles), "Reconcile changed projection topology"
    traffic_rows = []
    for role, row in zip(roles, matmuls):
        shape = [int(row[f"INPUT_1_{axis}_PAD[LOGICAL]"].split("[")[0]) for axis in ("W", "Z", "Y", "X")]
        dtype = row["INPUT_1_DATATYPE"]
        tile_bytes = {"BFLOAT4_B": 576, "BFLOAT8_B": 1088, "BFLOAT16": 2048}[dtype]
        active = 4 if role == "value_experts" else (8 if role.startswith("experts_") else None)
        copies = math.prod(shape[:2]) if active is None else active
        size = copies * math.ceil(shape[2] / 32) * math.ceil(shape[3] / 32) * tile_bytes
        traffic_rows.append(
            dict(
                role=role,
                weight_shape=shape,
                dtype=dtype,
                fidelity=row["MATH FIDELITY"],
                active_experts=active,
                mandatory_weight_bytes_per_chip=size,
            )
        )
    dense_weights = sum(r["mandatory_weight_bytes_per_chip"] for r in traffic_rows[:4])
    moe_weights = sum(r["mandatory_weight_bytes_per_chip"] for r in traffic_rows[4:11])
    kv = 1024 * 2 * 2 * 128 * 1088 // 1024 + 34816
    terminal = sum(r["mandatory_weight_bytes_per_chip"] for r in traffic_rows[11:]) + 2560 * 2 + 2 * 128 * 2
    full_bytes = 3 * dense_weights + 45 * moe_weights + 48 * kv + terminal
    reduced_bytes = dense_weights + moe_weights + 2 * kv + terminal
    summary["projection_traffic_rows"] = traffic_rows
    summary["kv_read_and_tile_update_bytes_per_layer_per_chip"] = kv
    summary["roofline_ms_per_token_estimate"] = full_bytes / 512e6
    summary["roofline_traffic_bytes_per_chip"] = full_bytes
    summary[
        "roofline_basis"
    ] = "Optimistic one-read active projection weights at measured stored tile dtypes +1024-token rounded KV window and tile read/write update /512GB/s/chip (2048GB/s mesh); no achieved-bandwidth claim. Excludes duplicated per-core reads, small gamma/bias and intermediate traffic. Embedding reads one row, not full table."
    summary["reduced_same_run_accounting"] = dict(
        layers=[0, 3],
        scope="Real full terminal, same full context allocation; not full48 device timing",
        roofline_ms=reduced_bytes / 512e6,
        traffic_bytes_per_chip=reduced_bytes,
        device_ms=windows["decode"]["max_rank_elapsed_ms"],
        host_ms=windows["decode"]["host_ms"],
        windows=windows,
        source_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
    )
    (DOC / "perf_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (DOC / "runtime_profile_rows.json").write_text(json.dumps(runtime, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
