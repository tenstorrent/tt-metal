# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Serial TP4 probe for the optional tiled native-math prefill convolution candidate.

No device access or TTNN import until main. Native prefill PCC >= .995; exact
history, explicit decode-arithmetic PCC >= .995 and changed-input traced
first/last checks. This is synthetic kernel evidence, not full-layer acceptance.
"""

import argparse
import hashlib
import json
import os
import statistics
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=["3x2", "3x33", "32x33", "3x257"])
    parser.add_argument("--cores", type=int)
    parser.add_argument("--block-ct", type=int, choices=(1, 2, 4, 8), default=8)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--replays", type=int, default=10)
    parser.add_argument("--poison-dead-padding", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.poison_dead_padding, "Tiled candidate owns zeroed window padding; use watcher instead"
    cases = [tuple(map(int, case.split("x"))) for case in args.cases]
    assert all(len(c) == 2 and 1 <= c[0] <= 32 and c[1] >= 1 for c in cases)
    assert args.steps >= 2 and args.samples > 0 and args.replays > 0
    assert args.cores is None or args.cores >= 1
    assert args.output.resolve().is_relative_to((ROOT / "doc/optimized_multichip_decoder").resolve())
    assert not args.output.exists(), "Preserve previous evidence; choose a new output"
    watcher = os.getenv("TT_METAL_WATCHER", "0") != "0"
    profiler = os.getenv("TT_METAL_DEVICE_PROFILER", "0") != "0"
    assert not (watcher and profiler), "Separate watcher and profiling runs"
    timed = not (watcher or args.poison_dead_padding)
    import torch
    import ttnn
    from models.autoports.qwen_qwen3_6_27b.tests.multichip_assertions import pcc
    from models.autoports.qwen_qwen3_6_27b.tt.tiled_prefill_conv_candidate import tiled_prefill_conv, NATIVE

    torch.set_num_threads(4)
    report = {
        "status": "running",
        "mesh": [1, 4],
        "arguments": vars(args) | {"output": str(args.output)},
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                ROOT / "tt/tiled_prefill_conv_candidate.py",
                ROOT / "tt/recurrent_next_candidate.py",
                ROOT / "tt/optimized_decoder.py",
            )
        },
        "watcher": watcher,
        "profiler": profiler,
        "timed": timed,
        "cases": [],
    }
    report["native_compute_sha256"] = hashlib.sha256(
        (NATIVE / "kernels/compute/qkv_causal_conv1d_silu.cpp").read_bytes()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    mesh = None
    save()
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=32 * 1024 * 1024)
        ckc = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        def upload(values, layout):
            return ttnn.from_torch(
                torch.cat(values, -1).contiguous(),
                device=mesh,
                dtype=ttnn.bfloat16,
                layout=layout,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
            )

        def scalar(value):
            return ttnn.from_torch(
                torch.tensor([value], dtype=torch.int64),
                device=mesh,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        def read(value):
            return [ttnn.to_torch(t).contiguous() for t in ttnn.get_device_tensors(value)]

        exchange_core = ttnn._ttnn.operations.experimental.kda.qkv_causal_conv1d_silu_exchange_core(
            mesh.compute_with_storage_grid_size()
        )
        exchange_grid = ttnn.CoreRangeSet({ttnn.CoreRange(exchange_core, exchange_core)})
        semaphores = [ttnn.create_global_semaphore(mesh, exchange_grid, 0) for _ in range(3)]
        exchange = upload([torch.zeros(1, 12, 2560) for _ in range(4)], ttnn.ROW_MAJOR_LAYOUT)
        start = scalar(0)
        for batch, seq in cases:
            entry = {"batch": batch, "sequence": seq, "checks": [], "timings_ms": {}}
            report["cases"].append(entry)
            save()
            generators = [torch.Generator().manual_seed(7111 + rank + 13 * batch + seq) for rank in range(4)]
            hist = [
                (torch.randn(batch, 3, 2560, generator=g) * 0.25 + rank * 0.03125).bfloat16()
                for rank, g in enumerate(generators)
            ]
            initial = upload(hist, ttnn.ROW_MAJOR_LAYOUT)
            histories = [upload(hist, ttnn.ROW_MAJOR_LAYOUT) for _ in range(2)]
            taps_host = [
                [(torch.randn(1, 1, 2560, generator=g) * 0.125).bfloat16() for g in generators] for _ in range(4)
            ]
            taps = [upload(t, ttnn.TILE_LAYOUT) for t in taps_host]
            taps_rm = [upload(t, ttnn.ROW_MAJOR_LAYOUT) for t in taps_host]
            bank_host = [
                [
                    (
                        torch.randn(batch, seq, 2560, generator=g) * 0.25
                        + torch.arange(batch).reshape(batch, 1, 1) * 0.0078125
                        + step * 0.03125
                    ).bfloat16()
                    for g in generators
                ]
                for step in range(args.steps)
            ]
            bank = [upload(t, ttnn.TILE_LAYOUT) for t in bank_host]
            token = ttnn.empty_like(bank[0])
            end = scalar(seq)
            padded = ((seq + 31) // 32) * 32

            def batched_prefill_conv(x, h, ignored_rm_taps, *, cores=None, poison_dead_padding=False):
                assert not poison_dead_padding
                return tiled_prefill_conv(x, h, taps, batch=batch, seq=x.shape[1], block_ct=args.block_ct, cores=cores)

            def invoke(index):
                if index:
                    return batched_prefill_conv(
                        token, histories[1], taps_rm, cores=args.cores, poison_dead_padding=args.poison_dead_padding
                    )
                x = token if seq == padded else ttnn.pad(token, ((0, 0), (0, padded - seq), (0, 0)), value=0.0)
                rows = [
                    ttnn.experimental.kda.qkv_causal_conv1d_silu_with_carry(
                        x[user : user + 1],
                        histories[0][user : user + 1],
                        *taps,
                        512,
                        512,
                        1536,
                        actual_start=start,
                        actual_end=end,
                        exchange_buffer=exchange,
                        exchange_semaphores=semaphores,
                        program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=256),
                        compute_kernel_config=ckc,
                    )
                    for user in range(batch)
                ]
                out = tuple(ttnn.concat([r[i] for r in rows], 0) if batch > 1 else rows[0][i] for i in range(4))
                ttnn.copy(out[3], histories[0])
                return out[:3]

            def explicit_reference():
                raw = ttnn.to_layout(token, ttnn.ROW_MAJOR_LAYOUT)
                joined = ttnn.concat([histories[1], raw], dim=1)
                products = [
                    ttnn.multiply(joined[:, i : i + seq], tap, dtype=ttnn.float32) for i, tap in enumerate(taps_rm)
                ]
                acc = ttnn.add(products[0], products[1], dtype=ttnn.float32)
                acc = ttnn.add(acc, products[2], dtype=ttnn.float32)
                return ttnn.silu(ttnn.add(acc, products[3], dtype=ttnn.bfloat16))

            def reset():
                for h in histories:
                    ttnn.copy(initial, h)

            def check(native, candidate, expected_history, label):
                scores, mismatches, native_bits = [], [], []
                for a, b in zip(native, candidate):
                    for x, y in zip(read(a), read(b)):
                        x, y = x[:, :seq], y[:, :seq]
                        assert torch.isfinite(x).all() and torch.isfinite(y).all()
                        scores.append(pcc(x, y))
                        native_bits.append(
                            int((x.contiguous().view(torch.int16) != y.contiguous().view(torch.int16)).sum())
                        )
                for h in histories:
                    for actual, expected in zip(read(h), expected_history):
                        mismatches.append(int((actual.view(torch.int16) != expected.view(torch.int16)).sum()))
                entry["checks"].append(
                    {
                        "label": label,
                        "native_pcc": scores,
                        "native_bit_mismatches": native_bits,
                        "history_bit_mismatches": mismatches,
                    }
                )
                save()
                assert min(scores) >= 0.995 and max(mismatches) == 0, label

            # Both branches start with distinct nonzero per-user/per-rank carry;
            # each subsequent input resumes the exact final 3 raw tokens.
            reset()
            expected_history = hist
            eager_host = []
            for step in range(args.steps):
                ttnn.copy(bank[step], token)
                explicit = read(explicit_reference())
                outputs = [invoke(i) for i in range(2)]
                candidate_parts = [read(t) for t in outputs[1]]
                explicit_scores = []
                for rank in range(4):
                    actual = torch.cat([part[rank][:, :seq] for part in candidate_parts], -1)
                    explicit_scores.append(pcc(actual, explicit[rank]))
                entry.setdefault("explicit_decode_arithmetic_pcc", []).append(explicit_scores)
                save()
                assert min(explicit_scores) >= 0.995
                expected_history = [
                    torch.cat([h, x], 1)[:, -3:].contiguous() for h, x in zip(expected_history, bank_host[step])
                ]
                check(*outputs, expected_history, f"eager_resume_{step}")
                eager_host.append([[read(t) for t in branch] for branch in outputs])
            reset()
            compact = upload(
                [
                    torch.nn.functional.pad(t.reshape(1, batch * seq, 2560), (0, 1600), value=float("nan"))
                    for t in bank_host[0]
                ],
                ttnn.TILE_LAYOUT,
            )
            compact_outputs = tiled_prefill_conv(
                compact, histories[1], taps, batch=batch, seq=seq, block_ct=args.block_ct, cores=args.cores
            )
            compact_errors = []
            for component in range(3):
                for rank, actual in enumerate(read(compact_outputs[component])):
                    expected = eager_host[0][1][component][rank]
                    compact_errors.append(
                        int(
                            (
                                actual[:, :seq].contiguous().view(torch.int16)
                                != expected[:, :seq].contiguous().view(torch.int16)
                            ).sum()
                        )
                    )
            entry["compact_packed_input_bit_mismatches"] = compact_errors
            save()
            assert max(compact_errors) == 0
            compact_expected = [torch.cat([h, x], 1)[:, -3:].contiguous() for h, x in zip(hist, bank_host[0])]
            assert all(torch.equal(a, b) for a, b in zip(read(histories[1]), compact_expected))
            # Candidate all-at-once vs sequence split includes short (<3) chunks.
            # Compare retained rows and carry exactly; state must have no padded rows.
            if seq > 1:
                reset()
                ttnn.copy(bank[0], token)
                whole = batched_prefill_conv(
                    token, histories[1], taps_rm, cores=args.cores, poison_dead_padding=args.poison_dead_padding
                )
                whole_host = [read(t) for t in whole]
                ttnn.copy(initial, histories[1])
                cut = min(2, seq - 1)
                pieces = [
                    batched_prefill_conv(
                        token[:, a:b],
                        histories[1],
                        taps_rm,
                        cores=args.cores,
                        poison_dead_padding=args.poison_dead_padding,
                    )
                    for a, b in ((0, cut), (cut, seq))
                ]
                errors = []
                for component in range(3):
                    left, right = read(pieces[0][component]), read(pieces[1][component])
                    for rank in range(4):
                        actual = torch.cat([left[rank][:, :cut], right[rank][:, : seq - cut]], 1)
                        errors.append(
                            int(
                                (
                                    actual.view(torch.int16)
                                    != whole_host[component][rank][:, :seq].contiguous().view(torch.int16)
                                ).sum()
                            )
                        )
                entry["split_resume_bit_mismatches"] = errors
                save()
                assert max(errors) == 0
                split_history = [torch.cat([h, x], 1)[:, -3:].contiguous() for h, x in zip(hist, bank_host[0])]
                assert all(torch.equal(a, b) for a, b in zip(read(histories[1]), split_history))
            # Capture each branch separately: first release its trace and result
            # before allocating another capture, avoiding allocation-lifetime ambiguity.
            trace_results = []
            for index in range(2):
                reset()
                ttnn.copy(bank[0], token)
                invoke(index)
                ttnn.synchronize_device(mesh)
                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                output = invoke(index)
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                try:
                    reset()
                    expected_history = hist
                    steps = []
                    for step in range(args.steps):
                        ttnn.copy(bank[step], token)
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                        expected_history = [
                            torch.cat([h, x], 1)[:, -3:].contiguous() for h, x in zip(expected_history, bank_host[step])
                        ]
                        values = [read(t) for t in output]
                        exact = [
                            int(
                                (
                                    values[c][r][:, :seq].contiguous().view(torch.int16)
                                    != eager_host[step][index][c][r][:, :seq].contiguous().view(torch.int16)
                                ).sum()
                            )
                            for c in range(3)
                            for r in range(4)
                        ]
                        entry.setdefault("trace_vs_eager_bit_mismatches", []).append(
                            {"branch": index, "step": step, "errors": exact}
                        )
                        save()
                        assert max(exact) == 0
                        assert all(torch.equal(a, b) for a, b in zip(read(histories[index]), expected_history))
                        steps.append(values)
                    trace_results.append(steps)
                    if timed:
                        samples = []
                        for _ in range(args.samples):
                            ttnn.synchronize_device(mesh)
                            before = time.perf_counter()
                            for _ in range(args.replays):
                                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                            ttnn.synchronize_device(mesh)
                            samples.append((time.perf_counter() - before) * 1000 / args.replays)
                        entry["timings_ms"]["native" if index == 0 else "candidate"] = {
                            "samples": samples,
                            "median": statistics.median(samples),
                        }
                finally:
                    ttnn.release_trace(mesh, trace)
                    del output
            traced_pcc = []
            for step in range(args.steps):
                for component in range(3):
                    for rank in range(4):
                        a, b = (trace_results[i][step][component][rank][:, :seq] for i in range(2))
                        assert torch.isfinite(a).all() and torch.isfinite(b).all()
                        traced_pcc.append(pcc(a, b))
            entry["changed_input_trace_pcc"] = traced_pcc
            assert min(traced_pcc) >= 0.995
            if profiler:
                ttnn.ReadDeviceProfiler(mesh)
            save()
        report["status"] = "pass"
    except BaseException:
        report["status"] = "fail"
        report["exception"] = traceback.format_exc()
        raise
    finally:
        save()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
