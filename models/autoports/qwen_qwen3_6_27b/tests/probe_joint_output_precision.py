# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Isolated direct joint-combine and real-weight projection precision probe.

CPU checks import neither torch nor TTNN. Parent serializes hardware runs.
This does not replace whole-layer/state/HF gates or select model defaults.
"""

import argparse
import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import time
import traceback

from models.autoports.qwen_qwen3_6_27b.tt import packed_correction_candidate as joint

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("control", "bf16_output", "fp32_geometry", "bf16_geometry", "bf16_acc4", "bf16_acc8")


def cpu_check():
    source = (ROOT / "tt/optimized_decoder.py").read_text()
    tree = ast.parse(source)
    reader = next(
        ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_NARROW_GDN_COMBINE_READER" for t in n.targets)
    )
    for old, new in ((516, 130), (320, 80), (512, 128), (192, 48)):
        reader = re.sub(r"\b" + str(old) + r"\b", str(new), reader)
    reader = reader.replace("{.page_id=row*rn+rc}", "{.page_id=row*rn+130+rc}")
    assert joint.combine_reader(reader, "float32") == reader
    bf16 = joint.combine_reader(reader, "bfloat16")
    assert bf16 == reader.replace("4096", "2048") and "4096" not in bf16
    assert joint.combine_storage("float32") == (
        (0, 8192, 4096, "float32"),
        (1, 8192, 4096, "float32"),
        (2, 4096, 2048, "bfloat16"),
        (3, 4096, 4096, "float32"),
    )
    assert joint.combine_storage("bfloat16") == tuple(
        (i, size, page, "bfloat16")
        for i, size, page in ((0, 4096, 2048), (1, 4096, 2048), (2, 4096, 2048), (3, 2048, 2048))
    )
    plans = mapping = 0
    for batch in range(1, 33):
        for seq in (1, 17, 31, 32, 33, 513):
            rows = batch * seq
            for n in (6912, 7168):
                for grid in ((8, 8), (8, 10), (7, 8), (11, 10)):
                    for acc in (True, False):
                        plan = joint.prefill_plan(rows, n, {}, {"grid": grid}, fp32_dest_acc_en=acc)
                        sh, sw = plan["out_subblock_h"], plan["out_subblock_w"]
                        assert sh * sw <= (4 if acc else 8)
                        assert plan["out_block_h"] % sh == plan["per_core_N"] % sw == 0
                        plans += 1
                for col in range(130):
                    rc = col if col < 80 else col - 48
                    if col < 80 or col >= 128:
                        assert 130 <= 130 + rc < 212 <= n // 32
                    for page in (2048, 4096):
                        assert (130 + rc + 1) * page <= (n // 32) * page
                    mapping += 1
    negative = 0
    for callback in (
        lambda: joint.precision_contract("bfloat8_b"),
        lambda: joint.precision_contract("float32", False),
        lambda: joint.precision_contract("bfloat16", 1),
        lambda: joint.prefill_plan(1056, 7168, {}, {"grid": (7, 8), "subblock_w": 8}),
        lambda: joint.combine_reader(reader + "4096", "bfloat16"),
        lambda: joint.PackedCorrection(None, combine="slice", prefill_output_dtype="bfloat16"),
    ):
        try:
            callback()
        except (ValueError, AssertionError):
            negative += 1
        else:
            raise AssertionError("Expected invalid precision/layout rejection")
    wide = joint.prefill_plan(1056, 7168, {}, {"grid": (7, 8), "subblock_w": 8}, fp32_dest_acc_en=False)
    assert wide["out_subblock_h"] * wide["out_subblock_w"] == 8
    return dict(
        pass_cpu=True,
        geometry_plans=plans,
        tile_mapping_checks=mapping,
        expected_rejections=negative,
        default_reader_byte_exact=True,
        default_cb_exact=True,
        limitations="CPU source/geometry only; no device numerical or latency claim",
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cpu-check", action="store_true")
    ap.add_argument("--mode", choices=("combine", "projection", "both"), default="combine")
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 3, 17, 32])
    ap.add_argument("--all-batches", action="store_true")
    ap.add_argument("--rows", nargs="+", type=int, default=[31, 33, 512, 1056])
    ap.add_argument(
        "--prefill-batch", type=int, default=1, help="Public prefill B; requested row counts must divide by B"
    )
    ap.add_argument("--phase", nargs="+", choices=("prefill", "decode"), default=["prefill", "decode"])
    ap.add_argument("--arms", nargs="+", choices=ARMS, default=["control", "bf16_output"])
    ap.add_argument("--grid", nargs=2, type=int, default=[7, 8])
    ap.add_argument("--k-block", type=int, default=8)
    ap.add_argument("--policy-file", type=Path)
    ap.add_argument("--activation-file", type=Path)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--queued", type=int, default=50)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--replays", type=int, default=20)
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if args.cpu_check:
        print(json.dumps(cpu_check(), indent=2))
        return
    if args.all_batches:
        args.batches = list(range(1, 33))
    assert args.output and not args.output.exists()
    assert args.output.resolve().is_relative_to((ROOT / "doc/optimized_multichip_decoder").resolve())
    assert all(1 <= b <= 32 for b in args.batches) and all(1 <= r <= 4096 for r in args.rows)
    assert 1 <= args.prefill_batch <= 32
    assert "prefill" not in args.phase or all(r % args.prefill_batch == 0 for r in args.rows)
    assert args.steps >= 2 and min(args.samples, args.replays, args.queued, *args.grid) > 0
    assert args.k_block > 0 and 160 % args.k_block == 0
    watcher, profiler = "TT_METAL_WATCHER" in os.environ, os.getenv("TT_METAL_DEVICE_PROFILER", "0") != "0"
    tracking = os.getenv("TT_METAL_TRACE_ALLOC_TRACKING") == "1"
    assert not (watcher and profiler)
    timed = not (watcher or profiler or tracking or args.no_timing)
    import torch
    import ttnn
    from models.autoports.qwen_qwen3_6_27b.tt import multichip_decoder as local
    from models.autoports.qwen_qwen3_6_27b.tests.multichip_assertions import pcc

    torch.set_num_threads(4)
    files = [
        Path(__file__),
        ROOT / "tt/packed_correction_candidate.py",
        ROOT / "tt/multichip_decoder.py",
        ROOT / "tt/optimized_decoder.py",
    ]
    report = dict(
        status="running",
        mesh=[1, 4],
        cases=[],
        cpu=cpu_check(),
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        watcher=watcher,
        profiler=profiler,
        allocation_tracking=tracking,
        timed=timed,
        skip_program_cache=os.getenv("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "unset"),
        reference="Exact FP32-add/RNE-BF16 combine oracle; projection control is accepted TTNN arithmetic, not HF",
        timing_scope="Complete combine or joint projection+combine; input copies excluded; allocation captured normally",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    mesh = None
    save()
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=32 * 1024 * 1024)

        def upload(values, dtype):
            return ttnn.from_torch(
                torch.cat(values, -1).contiguous(),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
                preserve_nan_values=True,
            )

        def read(value):
            return [ttnn.to_torch(t).contiguous() for t in ttnn.get_device_tensors(value)]

        def check(actual, expected, *, exact, entry, label):
            checks = []
            for rank, (a, b) in enumerate(zip(actual, expected)):
                assert a.shape == b.shape and a.dtype == b.dtype == torch.bfloat16
                axis = 1 if entry.get("phase") == "decode" else 0
                for user in range(a.shape[axis]):
                    au, bu = a.select(axis, user), b.select(axis, user)
                    for name, start, stop in (
                        ("qkv", 0, 2560),
                        ("z", 2560, 4096),
                        ("beta", 4096, 4108),
                        ("alpha", 4128, 4140),
                    ):
                        aa, bb = au[..., start:stop].contiguous(), bu[..., start:stop].contiguous()
                        equal = torch.equal(aa.view(torch.int16), bb.view(torch.int16))
                        score = 1.0 if equal else float(pcc(aa.float(), bb.float()))
                        checks.append(
                            dict(
                                rank=rank,
                                user=user,
                                group=name,
                                pcc=score,
                                bit_errors=int((aa.view(torch.int16) != bb.view(torch.int16)).sum()),
                                finite=bool(torch.isfinite(aa).all() and torch.isfinite(bb).all()),
                            )
                        )
            entry.setdefault("checks", []).append(dict(label=label, exact=exact, groups=checks))
            save()
            assert all(c["finite"] and c["pcc"] >= 0.995 and (not exact or c["bit_errors"] == 0) for c in checks), label
            if exact:
                assert all(
                    torch.equal(a.view(torch.int16), b.view(torch.int16)) for a, b in zip(actual, expected)
                ), label

        positive_done = False

        def exercise(entry, bank, invoke, expected, *, exact):
            nonlocal positive_done
            live, rebound = ttnn.clone(bank[0]), ttnn.clone(bank[1])
            eager = []
            for step in range(args.steps):
                ttnn.copy(bank[step], live)
                out = invoke(live)
                values = read(out)
                check(values, expected[step], exact=exact, entry=entry, label=f"eager{step}")
                eager.append(values)
                ttnn.deallocate(out)
            out = invoke(rebound)
            check(read(out), eager[1], exact=True, entry=entry, label="rebind")
            ttnn.deallocate(out)
            trace = None
            try:
                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                result = invoke(live)
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                if tracking and not positive_done:
                    unsafe = ttnn.empty((1, 32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
                    detected = False
                    try:
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    except RuntimeError as exc:
                        if not any(word in str(exc).lower() for word in ("unsafe", "allocation")):
                            raise
                        detected = True
                    finally:
                        ttnn.deallocate(unsafe)
                    entry["allocation_positive_control"] = detected
                    save()
                    assert detected
                    positive_done = True
                for step in range(args.steps):
                    ttnn.copy(bank[step], live)
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    check(read(result), eager[step], exact=True, entry=entry, label=f"trace{step}")
                for index in range(args.queued):
                    ttnn.copy(bank[index % args.steps], live)
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                check(
                    read(result), eager[(args.queued - 1) % args.steps], exact=True, entry=entry, label="queued_final"
                )
                entry["queued"] = args.queued
                if timed:
                    samples = []
                    for _ in range(2):
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    for _ in range(args.samples):
                        start = time.perf_counter_ns()
                        for _ in range(args.replays):
                            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                        ttnn.synchronize_device(mesh)
                        samples.append((time.perf_counter_ns() - start) / 1000 / args.replays)
                    entry.update(trace_us=samples, median_us=statistics.median(samples))
                    check(
                        read(result),
                        eager[(args.queued - 1) % args.steps],
                        exact=True,
                        entry=entry,
                        label="timed_final",
                    )
            finally:
                if trace is not None:
                    ttnn.release_trace(mesh, trace)
            entry["pass"] = True
            save()

        if args.mode in ("combine", "both"):
            helper = object.__new__(joint.PackedCorrection)
            helper.reader = local._LOCAL_NARROW_READER.replace("{.page_id=row*rn+rc}", "{.page_id=row*rn+130+rc}")
            helper.bf16_reader = joint.combine_reader(helper.reader, "bfloat16")
            helper.writer, helper.compute = local._NARROW_GDN_COMBINE_WRITER, local._LOCAL_NARROW_COMPUTE
            for rows in sorted(set(args.batches + args.rows)):
                for width in (6912, 7168):
                    for dtype_name in ("float32", "bfloat16"):
                        entry = dict(kind="combine", rows=rows, physical_width=width, source_dtype=dtype_name)
                        report["cases"].append(entry)
                        bank, expected = [], []
                        for step in range(args.steps):
                            values = []
                            for rank in range(4):
                                generator = torch.Generator().manual_seed(7713 + rows * 13 + rank * 3 + step * 997)
                                value = torch.randn((1, rows, width), generator=generator)
                                # Cancellation and tie-adjacent finite values, distinct per rank/step.
                                value[..., 4160:6720] = -value[..., :2560] + (step + 1) / 256
                                value[..., :8] = torch.tensor(
                                    [0.0, -0.0, 1.0, 1.00390625, -1.0, -1.00390625, 128.0, -128.0]
                                )
                                value[..., 6784:] = float("nan")  # physical N tail must never be consumed
                                values.append(value.to(getattr(torch, dtype_name)))
                            packed = upload(values, getattr(ttnn, dtype_name))
                            got = read(packed)
                            assert all(torch.isnan(v[..., 6784:]).all() for v in got), "Poison did not survive upload"
                            oracle = []
                            for value in got:
                                result = value[..., :4160].float().clone()
                                result[..., :2560] += value[..., 4160:6720].float()
                                result[..., 2560:4096] += 0.0
                                result[..., 4096:4160] += value[..., 6720:6784].float()
                                oracle.append(result.bfloat16())
                            bank.append(packed)
                            expected.append(oracle)
                        entry["verified_nan_tail_elements_per_step"] = 4 * rows * (width - 6784)
                        exercise(entry, bank, helper._combine, expected, exact=True)

        if args.mode in ("projection", "both"):
            from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
            from models.autoports.qwen_qwen3_6_27b.tests.run_decoder import load_layer

            cfg = Qwen3_5TextConfig(**json.loads((ROOT / "doc/hf_config.json").read_text())["text_config"])
            policy = json.loads(args.policy_file.read_text()) if args.policy_file else {}
            options = dict(policy.get("packed_correction_options") or {})
            options.update(
                prefill_output_dtype="float32",
                decode_output_dtype="float32",
                prefill_fp32_dest_acc_en=True,
                decode_fp32_dest_acc_en=True,
                combine="fused",
                prefill=True,
            )
            policy["packed_correction_options"] = options
            owner = local.MultichipDecoder.from_state_dict(
                load_layer(0), hf_config=cfg, layer_idx=0, mesh_device=mesh, policy=policy
            )
            base = owner._packed_correction_candidate
            source = args.activation_file or ROOT / "doc/optimized_decoder/text_activations/text_l0_b32_s254.pt"
            recorded = torch.load(source, weights_only=True)["x"].reshape(-1, 5120)
            assert recorded.shape[0] >= max(args.rows + args.batches) + args.steps
            report["activation_provenance"] = dict(
                file=str(source),
                size_bytes=source.stat().st_size,
                mtime_ns=source.stat().st_mtime_ns,
                boundary="actual TP4 input norm; token-independent slices of recorded text",
            )
            report["selected_policy"] = owner.policy
            for phase in args.phase:
                for rows in (args.batches if phase == "decode" else args.rows):
                    bank = []
                    input_shape = (
                        (1, rows, 5120) if phase == "decode" else (args.prefill_batch, rows // args.prefill_batch, 5120)
                    )
                    for step in range(args.steps):
                        x = recorded[step : step + rows].reshape(input_shape).contiguous()
                        bank.append(
                            owner._norm(
                                owner.upload_residual(x),
                                "input_layernorm.weight",
                                decode=phase == "decode",
                                logical_batch=rows if phase == "decode" else args.prefill_batch,
                            )
                        )
                    call = lambda candidate, tx: (
                        candidate.project(owner, tx) if phase == "decode" else candidate.project_prefill(owner, tx)
                    )
                    expected = []
                    for tx in bank:
                        out = call(base, tx)
                        expected.append(read(out))
                        ttnn.deallocate(out)
                    for arm in args.arms:
                        # DRAM-sharded compact factory picks subblock width itself;
                        # it cannot supply an independent reduced-accumulation 4-tile arm.
                        if rows <= 32 and arm in ("fp32_geometry", "bf16_geometry", "bf16_acc4"):
                            continue
                        candidate = copy.copy(base)
                        candidate.precision = {k: dict(v) for k, v in base.precision.items()}
                        candidate.prefill_config = dict(base.prefill_config)
                        if arm != "control":
                            candidate.precision[phase] = joint.precision_contract(
                                "float32" if arm == "fp32_geometry" else "bfloat16",
                                arm not in ("bf16_acc4", "bf16_acc8"),
                            )
                        if arm in ("fp32_geometry", "bf16_geometry", "bf16_acc4", "bf16_acc8") and rows > 32:
                            candidate.prefill_config.update(
                                grid=args.grid,
                                in0_block_w=args.k_block,
                                out_block_h=1,
                                subblock_h=1,
                                subblock_w=8 if arm == "bf16_acc8" else 4,
                            )
                        entry = dict(
                            kind="projection_combine",
                            phase=phase,
                            rows=rows,
                            input_shape=list(input_shape),
                            arm=arm,
                            precision=candidate.precision,
                            prefill_config=candidate.prefill_config,
                            compact_subblock="native automatic" if rows <= 32 else None,
                        )
                        report["cases"].append(entry)
                        exercise(entry, bank, lambda tx: call(candidate, tx), expected, exact=arm == "control")
        report.update(status="pass", pass_all=True)
    except BaseException:
        report.update(status="fail", exception=traceback.format_exc())
        raise
    finally:
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
            report["mesh_closed"] = True
        save()


if __name__ == "__main__":
    main()
