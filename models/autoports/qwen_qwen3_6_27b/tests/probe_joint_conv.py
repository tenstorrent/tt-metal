# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TP4 joint combine -> QKV slice -> untilize -> local convolution comparison.

CPU-only by --cpu-check. Hardware work belongs to the parent device queue.
Native baseline allocations remain in its actual boundary; candidate inputs,
outputs, histories and input banks are prepared before capture.
"""

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
import traceback

from models.autoports.qwen_qwen3_6_27b.tt import joint_conv_candidate as candidate

ROOT = Path(__file__).resolve().parents[1]


def cpu_check():
    addresses = owners = 0
    for batch in range(1, 33):
        for workers in (1, 8, 48, 80, 110, 130):
            columns = [
                start + i * stride for start, count, stride in candidate.plan(batch, workers) for i in range(count)
            ]
            assert sorted(columns) == list(range(130))
            owners += len(columns)
        for col in range(130):
            rc = candidate.residual_column(col)
            assert rc == (130 + col if col < 80 else 130 + col - 48 if col >= 128 else None)
            if rc is not None:
                assert 130 <= rc < 212
            for user in range(batch):
                for ch in range(32):
                    idx = candidate.tile_index(user, ch)
                    assert idx == (user // 16 * 2 + ch // 16) * 256 + user % 16 * 16 + ch % 16
                    if col < 80:
                        offset = col * 64
                        assert offset % 64 == 0 and offset + 64 <= 5120
                        assert (user * 3 + 2) * 5120 + offset + ch * 2 < batch * 3 * 5120
                    addresses += 1
        for residue in range(64):
            shift = (-residue) % 64
            assert shift + 6400 <= 6464 and shift + 8192 <= 8256
    assert sorted(candidate.tile_index(u, c) for u in range(32) for c in range(32)) == list(range(1024))
    own_tree = ast.parse(Path(__file__).read_text())
    verify = next(n for n in ast.walk(own_tree) if isinstance(n, ast.FunctionDef) and n.name == "verify")
    user_loop = next(
        n
        for n in ast.walk(verify)
        if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "user"
    )
    assert any(
        isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Attribute)
        and n.value.func.attr == "append"
        and isinstance(n.value.func.value, ast.Name)
        and n.value.func.value.id == "checks"
        for n in user_loop.body
    ), "Every user must append a comparison even when nonfinite"
    assert any(
        isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "finite_reference" for t in n.targets)
        for n in user_loop.body
    )
    native = ast.parse((ROOT / "tt/optimized_decoder.py").read_text())
    strings = {
        n.targets[0].id: ast.literal_eval(n.value)
        for n in native.body
        if isinstance(n, ast.Assign)
        and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Name)
        and isinstance(n.value, ast.Constant)
        and isinstance(n.value.value, str)
    }
    conv = strings["_conv107__CONV_RM_PIPE2_COMPUTE"]
    for line in (
        "mul_binary_tile_init(); mul_binary_tile(work,2,work);",
        "add_binary_tile_init(); add_binary_tile(0,1,0);",
        "MATH(SFPU_UNARY_CALL(DST_SYNC_MODE,true,calculate_silu,(false,8),idst,VectorMode::RC));",
    ):
        assert "".join(line.split()) in "".join(conv.split())
        assert "".join(line.split()) in "".join(candidate.COMPUTE.split())
    assert candidate.COMPUTE.count("reconfig_data_format_srca(0);") == 1
    assert candidate.COMPUTE.count("reconfig_data_format_srca(4);") == 1
    assert candidate.COMPUTE.index("reconfig_data_format_srca(0);") < candidate.COMPUTE.index("copy_init(0);")
    assert candidate.COMPUTE.index("reconfig_data_format_srca(4);") < candidate.COMPUTE.index(
        "for(uint32_t tap=0;tap<4;tap++)"
    )
    assert "round_bf16(0);" in candidate.COMPUTE
    assert candidate.COMPUTE.index("round_bf16(0);") < candidate.COMPUTE.index("for(uint32_t tap=0;tap<4;tap++)")
    # Writer retains old history1/2 and raw combined token, not SiLU output.
    assert "id=row==0?9:row==3?2:row+4" in candidate.WRITER
    for dtype in ("float32", "bfloat16"):
        for rest in (False, True):
            reader, compute, writer = candidate.sources(dtype, rest)
            assert "SOURCE_" not in reader and "REST_ONLY" not in writer
            assert compute == candidate.COMPUTE
    negative = 0
    for fn in (
        lambda: candidate.plan(0),
        lambda: candidate.plan(33),
        lambda: candidate.plan(1, 0),
        lambda: candidate.plan(1, True),
        lambda: candidate.residual_column(130),
        lambda: candidate.sources("bfloat4_b"),
        lambda: candidate.sources("float32", 1),
    ):
        try:
            fn()
        except ValueError:
            negative += 1
        else:
            raise AssertionError("Invalid geometry accepted")
    return dict(
        pass_cpu=True,
        element_addresses=addresses,
        column_ownership=owners,
        expected_rejections=negative,
        fp32_cb_bytes=43392,
        bf16_cb_bytes=39296,
        hardware_exactness="unverified",
        arithmetic="native combine RNE-BF16 before native FP32 tap products/sum, BF16 SiLU",
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cpu-check", action="store_true")
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 3, 17, 32])
    ap.add_argument("--all-batches", action="store_true")
    ap.add_argument("--dtypes", nargs="+", choices=("float32", "bfloat16"), default=["float32"])
    ap.add_argument("--placements", nargs="+", choices=("l1", "dram"), default=["l1"])
    ap.add_argument("--cores", nargs="+", type=int, default=[80])
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--queued", type=int, default=50)
    ap.add_argument("--samples", type=int, default=7)
    ap.add_argument("--replays", type=int, default=100)
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if args.cpu_check:
        print(json.dumps(cpu_check(), indent=2))
        return
    if args.all_batches:
        args.batches = list(range(1, 33))
    assert args.output and not args.output.exists() and args.steps >= 2
    assert args.output.resolve().is_relative_to((ROOT / "doc/optimized_multichip_decoder").resolve())
    assert min(args.queued, args.samples, args.replays) > 0
    watcher = "TT_METAL_WATCHER" in os.environ
    profiler = os.getenv("TT_METAL_DEVICE_PROFILER", "0") != "0"
    tracking = os.getenv("TT_METAL_TRACE_ALLOC_TRACKING") == "1"
    assert not (watcher and profiler)
    if tracking:
        assert os.getenv("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE") == "0"
    timed = not (watcher or profiler or tracking or args.no_timing)
    import torch
    import ttnn
    from models.autoports.qwen_qwen3_6_27b.tt import multichip_decoder as local
    from models.autoports.qwen_qwen3_6_27b.tt.packed_correction_candidate import PackedCorrection, combine_reader
    from models.autoports.qwen_qwen3_6_27b.tt.recurrent_next_candidate import local_conv
    from models.autoports.qwen_qwen3_6_27b.tests.multichip_assertions import pcc

    torch.set_num_threads(4)
    report = dict(
        status="running",
        mesh=[1, 4],
        cpu=cpu_check(),
        cases=[],
        timed=timed,
        watcher=watcher,
        profiler=profiler,
        allocation_tracking=tracking,
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        boundary="Actual joint FP32/BF16 -> corrected packed/rest and SiLU RM QKV, in-place raw history shift",
        reference="Current PackedCorrection._combine + material QKV slice + local_conv including untilize, not HF",
        allocations="All candidate outputs/banks/history allocated before capture; native intermediates allocated by original ops during capture",
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                ROOT / "tt/joint_conv_candidate.py",
                ROOT / "tt/packed_correction_candidate.py",
                ROOT / "tt/recurrent_next_candidate.py",
                ROOT / "tt/optimized_decoder.py",
                ROOT / "tt/multichip_decoder.py",
            ]
        },
    )

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    mesh = None
    save()
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=32 * 1024 * 1024)
        grid = mesh.compute_with_storage_grid_size()
        combine = PackedCorrection.__new__(PackedCorrection)
        combine.reader = local._LOCAL_NARROW_READER.replace("{.page_id=row*rn+rc}", "{.page_id=row*rn+130+rc}")
        combine.bf16_reader = combine_reader(combine.reader, "bfloat16")
        combine.compute, combine.writer = local._LOCAL_NARROW_COMPUTE, local._NARROW_GDN_COMBINE_WRITER

        def upload(values, dtype, layout):
            return ttnn.from_torch(
                torch.cat(values, -1).contiguous(),
                dtype=dtype,
                layout=layout,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
                preserve_nan_values=True,
            )

        def empty(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory=ttnn.DRAM_MEMORY_CONFIG):
            return ttnn.empty(shape, dtype=dtype, layout=layout, device=mesh, memory_config=memory)

        def read(t):
            return [ttnn.to_torch(s).contiguous() for s in ttnn.get_device_tensors(t)]

        def raw_copy(src, dst):
            assert src.dtype == dst.dtype and tuple(src.padded_shape) == tuple(dst.padded_shape)
            assert src.memory_config() == dst.memory_config() == ttnn.DRAM_MEMORY_CONFIG
            page = 4096 if src.dtype == ttnn.float32 else 2048
            core = ttnn.CoreCoord(0, 0)
            ranges = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
            rt = ttnn.RuntimeArgs()
            rt[0][0] = [src.buffer_address(), dst.buffer_address()]
            ct = list(ttnn.TensorAccessorArgs(src).get_compile_time_args()) + list(
                ttnn.TensorAccessorArgs(dst).get_compile_time_args()
            )
            source = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
 constexpr auto sa=TensorAccessorArgs<0>();constexpr auto da=TensorAccessorArgs<sa.next_compile_time_args_offset()>();
 const auto s=TensorAccessor(sa,get_arg_val<uint32_t>(0),PAGE);const auto d=TensorAccessor(da,get_arg_val<uint32_t>(1),PAGE);
 Noc noc;CircularBuffer cb(0);
 for(uint32_t i=0;i<224;i+=8) {
 cb.reserve_back(8);for(uint32_t j=0;j<8;j++)noc.async_read(s,cb,PAGE,{.page_id=i+j},{.offset_bytes=j*PAGE});
 noc.async_read_barrier();cb.push_back(8);cb.wait_front(8);
 for(uint32_t j=0;j<8;j++)noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(cb),d,PAGE,{.offset_bytes=j*PAGE},{.page_id=i+j});
 noc.async_write_barrier();cb.pop_front(8);
 }
}""".replace("PAGE", str(page))
            kernel = ttnn.KernelDescriptor(
                kernel_source=source,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=ranges,
                compile_time_args=ct,
                runtime_args=rt,
                config=ttnn.ReaderConfigDescriptor(),
            )
            cb = ttnn.CBDescriptor(
                total_size=page * 8,
                core_ranges=ranges,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=src.dtype, page_size=page)],
            )
            ttnn.generic_op([src, dst], ttnn.ProgramDescriptor(kernels=[kernel], cbs=[cb], semaphores=[]))

        positive_done = False
        for batch in args.batches:
            for dtype_name in args.dtypes:
                for placement in args.placements:
                    entry = dict(batch=batch, dtype=dtype_name, placement=placement, checks=[], trace_us={}, poison=[])
                    report["cases"].append(entry)
                    save()
                    dtype = getattr(ttnn, dtype_name)
                    memory = ttnn.DRAM_MEMORY_CONFIG
                    if placement == "l1":
                        workers = ttnn.num_cores_to_corerangeset(8, grid, row_wise=True)
                        memory = ttnn.create_sharded_memory_config(
                            (32, 896), workers, ttnn.ShardStrategy.WIDTH, use_height_and_width_as_shard_shape=True
                        )
                    banks, host_banks = [], []
                    for step in range(args.steps):
                        values = []
                        for rank in range(4):
                            gen = torch.Generator().manual_seed(541 + batch * 31 + rank * 997 + step * 3571)
                            value = torch.full((1, 32, 7168), float("nan"), dtype=torch.float32)
                            value[:, :batch, :6784] = torch.randn((1, batch, 6784), generator=gen) * (0.3 + rank * 0.2)
                            # Nontrivial cancellation and signed-zero input cases.
                            value[:, :batch, :8] = torch.tensor(
                                [0.0, -0.0, 1.0, -1.0, 1.00390625, 1.01171875, 64.0, -64.0]
                            )
                            value[:, :batch, 4160:4168] = torch.tensor(
                                [-0.0, 0.0, -0.999, 1.001, 0.0, 0.0, -63.875, 63.875]
                            )
                            values.append(value.to(torch.bfloat16 if dtype_name == "bfloat16" else torch.float32))
                        full = upload(values, dtype, ttnn.TILE_LAYOUT)
                        logical = empty((1, batch, 7168), dtype)
                        raw_copy(full, logical)
                        physical = [
                            s.cpu().to_torch_with_padded_shape().contiguous() for s in ttnn.get_device_tensors(logical)
                        ]
                        word = torch.int32 if dtype_name == "float32" else torch.int16
                        bad = [int((a.view(word) != b.view(word)).sum()) for a, b in zip(physical, values)]
                        entry["poison"].append(
                            dict(step=step, bit_errors=bad, nans=[int(torch.isnan(x).sum()) for x in physical])
                        )
                        assert bad == [0] * 4
                        banks.append(ttnn.to_memory_config(logical, memory))
                        host_banks.append(values)
                        ttnn.deallocate(full)
                    live = empty((1, batch, 7168), dtype, memory=memory)
                    rebound = empty((1, batch, 7168), dtype, memory=memory)
                    initial = []
                    tap_values = [[] for _ in range(4)]
                    for rank in range(4):
                        gen = torch.Generator().manual_seed(197 + rank * 671 + batch)
                        initial.append(torch.randn((batch, 3, 2560), generator=gen).bfloat16())
                        for tap in range(4):
                            tap_values[tap].append(
                                (torch.randn((1, 1, 2560), generator=gen) * (0.15 + tap * 0.13)).bfloat16()
                            )
                    history_bank = upload(initial, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
                    taps = [upload(v, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT) for v in tap_values]
                    rebound_taps = [upload(v, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT) for v in tap_values]
                    paths = [dict(name="native", core=None, rest_only=False)] + [
                        dict(name=f'fused_c{c}_{"rest" if r else "packed"}', core=c, rest_only=r)
                        for c in args.cores
                        for r in (False, True)
                    ]
                    for path in paths:
                        path["history"] = empty((batch, 3, 2560), layout=ttnn.ROW_MAJOR_LAYOUT)
                        path["output"] = empty((batch, 1, 2560), layout=ttnn.ROW_MAJOR_LAYOUT)
                        path["rest"] = (
                            None if path["core"] is None else empty((1, batch, 1600 if path["rest_only"] else 4160))
                        )

                    def reset_history(path):
                        ttnn.copy(history_bank, path["history"])

                    def set_input(step, dst=live):
                        ttnn.copy(banks[step], dst)

                    def verify_input(step, tensor, label):
                        physical_dram = ttnn.to_memory_config(tensor, ttnn.DRAM_MEMORY_CONFIG)
                        physical = [
                            s.cpu().to_torch_with_padded_shape().contiguous()
                            for s in ttnn.get_device_tensors(physical_dram)
                        ]
                        word = torch.int32 if dtype_name == "float32" else torch.int16
                        bad = [int((a.view(word) != b.view(word)).sum()) for a, b in zip(physical, host_banks[step])]
                        entry["poison"].append(
                            dict(
                                stage=label,
                                step=step,
                                bit_errors=bad,
                                nans=[int(torch.isnan(x).sum()) for x in physical],
                            )
                        )
                        save()
                        assert bad == [0] * 4

                    def invoke(path, inp=live, coefs=taps):
                        if path["core"] is None:
                            packed = combine._combine(inp)
                            local_conv(packed[:, :, :2560], path["history"], coefs, output=path["output"])
                            path["rest"] = packed
                        else:
                            candidate.joint_conv(
                                inp,
                                path["history"],
                                coefs,
                                output=path["output"],
                                rest=path["rest"],
                                rest_only=path["rest_only"],
                                cores=path["core"],
                            )

                    def snapshot(path):
                        return tuple(read(path[k]) for k in ("output", "history", "rest"))

                    def verify(path, expected, label):
                        got = snapshot(path)
                        checks = []
                        for key, actual, reference in zip(("output", "history", "rest"), got, expected):
                            for rank, (a, b) in enumerate(zip(actual, reference)):
                                if key == "rest" and path["rest_only"]:
                                    b = b[..., 2560:]
                                assert a.shape == b.shape
                                axis = 1 if key == "rest" else 0
                                for user in range(batch):
                                    aa = a.select(axis, user).contiguous()
                                    bb = b.select(axis, user).contiguous()
                                    errors = int((aa.view(torch.int16) != bb.view(torch.int16)).sum())
                                    finite_actual = bool(torch.isfinite(aa).all())
                                    finite_reference = bool(torch.isfinite(bb).all())
                                    score = None
                                    if finite_actual and finite_reference:
                                        if errors == 0:
                                            score = 1.0
                                        elif bool(torch.var(aa.double(), correction=0) > 0) and bool(
                                            torch.var(bb.double(), correction=0) > 0
                                        ):
                                            # The metric must never prevent exact-failure tensor capture.
                                            try:
                                                value = float(pcc(aa.float(), bb.float()))
                                                score = value if math.isfinite(value) else 0.0
                                            except AssertionError:
                                                score = 0.0
                                        else:
                                            score = 0.0
                                    checks.append(
                                        dict(
                                            tensor=key,
                                            rank=rank,
                                            user=user,
                                            bit_errors=errors,
                                            pcc=score,
                                            finite=finite_actual and finite_reference,
                                            actual_nonfinite=int((~torch.isfinite(aa)).sum()),
                                            reference_nonfinite=int((~torch.isfinite(bb)).sum()),
                                        )
                                    )
                        assert len(checks) == 3 * 4 * batch
                        entry["checks"].append(dict(path=path["name"], label=label, values=checks))
                        save()
                        passing = all(
                            c["bit_errors"] == 0 and c["pcc"] is not None and c["pcc"] >= 0.995 and c["finite"]
                            for c in checks
                        )
                        if not passing and "first_failure" not in report:
                            dump = args.output.with_suffix(".first_failure.pt")
                            assert not dump.exists(), "Preserve prior failure dump"
                            snapshots = {p["name"]: snapshot(p) for p in paths}
                            masks = {
                                name: [[torch.isfinite(t) for t in ranks] for ranks in tensors]
                                for name, tensors in snapshots.items()
                            }
                            torch.save(
                                dict(
                                    path=path["name"],
                                    label=label,
                                    batch=batch,
                                    tensors=snapshots,
                                    finite_masks=masks,
                                    expected=expected,
                                    joint=read(live),
                                    taps=[read(t) for t in taps],
                                ),
                                dump,
                            )
                            summaries = {}
                            for name, tensors in snapshots.items():
                                summaries[name] = {}
                                for key, ranks, refs in zip(("output", "history", "rest"), tensors, expected):
                                    items = []
                                    for rank, (a, b) in enumerate(zip(ranks, refs)):
                                        if key == "rest" and a.shape[-1] == 1600:
                                            b = b[..., 2560:]
                                        bad = a.contiguous().view(torch.int16) != b.contiguous().view(torch.int16)
                                        coords = torch.nonzero(bad, as_tuple=False)[:64]
                                        items.append(
                                            dict(
                                                rank=rank,
                                                bit_errors=int(bad.sum()),
                                                nonfinite=int((~torch.isfinite(a)).sum()),
                                                reference_nonfinite=int((~torch.isfinite(b)).sum()),
                                                mismatch_coordinates=coords.tolist(),
                                                actual_bits=[
                                                    int(a.contiguous().view(torch.int16)[tuple(c.tolist())])
                                                    for c in coords
                                                ],
                                                expected_bits=[
                                                    int(b.contiguous().view(torch.int16)[tuple(c.tolist())])
                                                    for c in coords
                                                ],
                                            )
                                        )
                                    summaries[name][key] = items
                            report["first_failure"] = dict(
                                path=path["name"], label=label, dump=str(dump), summaries=summaries
                            )
                            save()
                        assert passing, (path["name"], label)

                    eager = []
                    for step in range(args.steps):
                        set_input(step)
                        for path in paths:
                            reset_history(path)
                            invoke(path)
                        expected = snapshot(paths[0])
                        eager.append(expected)
                        for path in paths:
                            verify(path, expected, f"eager{step}")
                        verify_input(step, live, "post_eager_immutable_actual_placement")
                        for path in paths[1:]:
                            physical = [
                                s.cpu().to_torch_with_padded_shape() for s in ttnn.get_device_tensors(path["rest"])
                            ]
                            assert all(torch.count_nonzero(v[:, batch:]) == 0 for v in physical)

                    set_input(1, rebound)
                    for path in paths:
                        previous = {key: path[key] for key in ("history", "output", "rest")}
                        path["history"] = empty((batch, 3, 2560), layout=ttnn.ROW_MAJOR_LAYOUT)
                        path["output"] = empty((batch, 1, 2560), layout=ttnn.ROW_MAJOR_LAYOUT)
                        path["rest"] = (
                            None if path["core"] is None else empty((1, batch, 1600 if path["rest_only"] else 4160))
                        )
                        reset_history(path)
                        invoke(path, rebound, rebound_taps)
                        verify(path, eager[1], "rebound_joint_history_taps_outputs")
                        path.update(previous)
                    # Actual accumulating native history oracle, computed before traces.
                    reset_history(paths[0])
                    for step in range(args.queued):
                        set_input(step % args.steps)
                        invoke(paths[0])
                    queued_expected = snapshot(paths[0])
                    set_input(args.steps - 1)
                    for _ in range(4):
                        invoke(paths[0])
                    steady_expected = snapshot(paths[0])
                    traces = []
                    stable = [
                        (
                            p["history"].buffer_address(),
                            p["output"].buffer_address(),
                            None if p["core"] is None else p["rest"].buffer_address(),
                        )
                        for p in paths
                    ]
                    try:
                        for path in paths:
                            reset_history(path)
                            invoke(path)
                        for path in paths:
                            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                            invoke(path)
                            ttnn.end_trace_capture(mesh, trace, cq_id=0)
                            traces.append(trace)
                        if tracking and not positive_done:
                            unsafe = empty((1, 32, 32))
                            detected = False
                            try:
                                ttnn.execute_trace(mesh, traces[-1], cq_id=0, blocking=True)
                            except RuntimeError as exc:
                                if not any(w in str(exc).lower() for w in ("allocation", "unsafe")):
                                    raise
                                detected = True
                            finally:
                                ttnn.deallocate(unsafe)
                            entry["allocation_positive_control"] = detected
                            save()
                            assert detected
                            positive_done = True
                        for step in range(args.steps):
                            set_input(step)
                            for path, trace in reversed(list(zip(paths, traces))):
                                reset_history(path)
                                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                                verify(path, eager[step], f"trace{step}")
                        for path in paths:
                            reset_history(path)
                        for step in range(args.queued):
                            set_input(step % args.steps)
                            for trace in traces:
                                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                        ttnn.synchronize_device(mesh)
                        for path in paths:
                            verify(path, queued_expected, "queued_history")
                        if timed:
                            set_input(args.steps - 1)
                            for trace in traces:
                                for _ in range(4):
                                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                            ttnn.synchronize_device(mesh)
                            samples = [[] for _ in paths]
                            for sample in range(args.samples):
                                for offset in range(len(paths)):
                                    i = (sample + offset) % len(paths)
                                    start = time.perf_counter_ns()
                                    for _ in range(args.replays):
                                        ttnn.execute_trace(mesh, traces[i], cq_id=0, blocking=False)
                                    ttnn.synchronize_device(mesh)
                                    samples[i].append((time.perf_counter_ns() - start) / 1000 / args.replays)
                            for i, path in enumerate(paths):
                                entry["trace_us"][path["name"]] = dict(
                                    samples=samples[i], median=statistics.median(samples[i])
                                )
                                verify(path, steady_expected, "timed_steady")
                        assert stable == [
                            (
                                p["history"].buffer_address(),
                                p["output"].buffer_address(),
                                None if p["core"] is None else p["rest"].buffer_address(),
                            )
                            for p in paths
                        ]
                    finally:
                        for trace in traces:
                            ttnn.release_trace(mesh, trace)
                    entry.update(pass_all=True, queued=args.queued)
                    save()
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
