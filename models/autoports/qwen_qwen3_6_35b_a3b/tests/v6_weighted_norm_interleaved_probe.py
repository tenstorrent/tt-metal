# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Raw interleaved input reader for the exact weighted norm/compact-SiLU adapter."""

import argparse
import ast
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "doc/fused_decoder/endurance"
BASE = E / "v6_transport_conv_norm_decay_lookup_integration.py.snapshot"
BASE_SHA = "44f87c04349836a4ad8b09c7a6c573fe3afe1e558af285e0482cce2c0fe5b2b3"
FIXTURE_HELPER = ROOT / "tests/v6_gated_norm_tail_timing_probe.py"
FIXTURE_SHA = "48ba6764ac1b04586a21e364d97a08843b555fbe163abb4d2eb11e956d9b5f24"
READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
 constexpr uint32_t sum_offset=get_compile_time_arg_val(0);
 const auto core_input=TensorAccessor(TensorAccessorArgs<1>(),get_arg_val<uint32_t>(0));
 const auto sum_input=TensorAccessor(TensorAccessorArgs<sum_offset>(),get_arg_val<uint32_t>(1));
 DataflowBuffer core(0),sums(1);Noc noc;
 core.reserve_back(4);sums.reserve_back(1);
 for(uint32_t tile=0;tile<4;++tile) {
  noc.async_read(core_input,core,4096,{.page_id=tile},{.offset_bytes=tile*4096});
 }
 noc.async_read(sum_input,sums,4096,{.page_id=0},{.offset_bytes=0});
 noc.async_read_barrier();
 core.push_back(4);sums.push_back(1);
}
"""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_proof():
    assert sha(BASE) == BASE_SHA
    tree = ast.parse(BASE.read_text())
    original = ast.literal_eval(
        next(
            n.value
            for n in tree.body
            if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "WEIGHTED_NORM_COMPUTE_SOURCE"
        )
    )
    publication = " core.reserve_back(4); core.push_back(4);\n sums.reserve_back(1); sums.push_back(1);\n"
    assert original.count(publication) == 1
    compute = original.replace(publication, "")
    marker = " gamma.reserve_back(4); gamma.push_back(4);"
    assert compute.count(marker) == 1 and compute.replace(marker, publication + marker) == original
    cls = next(
        n
        for n in ast.parse(Path(__file__).read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "DirectWeightedNormTail"
    )
    runtime = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__call__")
    calls = [ast.unparse(n.func) for n in ast.walk(runtime) if isinstance(n, ast.Call)]
    assert calls.count("t.generic_op") == 1
    assert not any(
        s.startswith("torch.") or s.endswith((".from_torch", ".to_torch", ".to_memory_config")) for s in calls
    )
    return compute, dict(
        base_sha256=BASE_SHA,
        original_compute_sha256=hashlib.sha256(original.encode()).hexdigest(),
        candidate_compute_sha256=hashlib.sha256(compute.encode()).hexdigest(),
        reader_sha256=hashlib.sha256(READER.encode()).hexdigest(),
        exact_compute_inverse=True,
        removed_publication=publication,
        arithmetic_pack_unpack_and_compact_consumer_unchanged=True,
        raw_reader_pages=dict(core=4, sums=1),
        local_input_cb_bytes=20480,
        replaced_input_shard_bytes=20480,
        persistent_added_bytes=0,
    )


class DirectWeightedNormTail:
    """Same tail math/output buffers; raw interleaved reader replaces input shards."""

    def __init__(self, baseline, source):
        self.base = baseline
        self.t, self.mesh, self.source = baseline.t, baseline.mesh, source

    def __call__(self, core, sums):
        t, base = self.t, self.base
        assert tuple(core.shape) == (1, 32, 128) and tuple(sums.shape) == (1, 32, 1)
        assert core.dtype == sums.dtype == t.float32 and core.layout == sums.layout == t.TILE_LAYOUT
        assert core.memory_config() in (t.L1_MEMORY_CONFIG, t.DRAM_MEMORY_CONFIG)
        assert sums.memory_config() in (t.L1_MEMORY_CONFIG, t.DRAM_MEMORY_CONFIG)
        output = t.empty(
            (1, 32, 128), dtype=t.float32, layout=t.TILE_LAYOUT, device=self.mesh, memory_config=base.memory
        )
        bindings = [(2, base.gamma), (16, output)]
        debug_outputs = {}
        if base.debug:
            for index, name in ((17, "normalized_fp32"), (18, "weighted_fp32_carrier"), (19, "inverse_repeated")):
                value = t.empty(
                    (1, 32, 128), dtype=t.float32, layout=t.TILE_LAYOUT, device=self.mesh, memory_config=base.memory
                )
                bindings.append((index, value))
                debug_outputs[name] = value
        ac = list(t.TensorAccessorArgs(core).get_compile_time_args())
        bc = list(t.TensorAccessorArgs(sums).get_compile_time_args())
        runtime = t.RuntimeArgs()
        runtime[0][0] = [core.buffer_address(), sums.buffer_address()]
        read = t.KernelDescriptor(
            kernel_source=READER,
            source_type=t.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=base.cores,
            compile_time_args=[1 + len(ac)] + ac + bc,
            runtime_args=runtime,
            config=t.ReaderConfigDescriptor(),
        )
        compute = t.KernelDescriptor(
            kernel_source=self.source,
            source_type=t.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=base.cores,
            compile_time_args=[int(base.debug)],
            runtime_args=[],
            config=base.config,
        )
        cbs = [t.cb_descriptor_from_sharded_tensor(i, value) for i, value in bindings]
        for index, pages in ((0, 4), (1, 1)):
            cbs.append(
                t.CBDescriptor(
                    total_size=pages * 4096,
                    core_ranges=base.cores,
                    format_descriptors=[t.CBFormatDescriptor(index, t.float32, 4096)],
                )
            )
        program = t.ProgramDescriptor(kernels=[read, compute], semaphores=[], cbs=cbs)
        t.generic_op([core, sums] + [value for _, value in bindings], program)
        return output, debug_outputs


class CompleteAdapter:
    def __init__(self, t, mesh, module, weighted, compact):
        self.t, self.reshape, self.weighted, self.compact = t, module.reshape, weighted, compact
        self.compute = t.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=t.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def prepare_core(self, core):
        t = self.t
        core_bf16 = t.typecast(core, t.bfloat16)
        core_rows = self.reshape(core_bf16, (1, 32, 128))
        wide = t.typecast(core_rows, t.float32, memory_config=t.L1_MEMORY_CONFIG)
        squares = t.multiply(
            wide, wide, dtype=t.float32, fast_and_approximate_mode=False, memory_config=t.L1_MEMORY_CONFIG
        )
        sums = t.sum(
            squares, dim=-1, keepdim=True, compute_kernel_config=self.compute, memory_config=t.L1_MEMORY_CONFIG
        )
        return wide, sums

    def __call__(self, core, z):
        wide, sums = self.prepare_core(core)
        weighted, _ = self.weighted(wide, sums)
        merged, _ = self.compact(weighted, z)
        return merged


def bits(a, b):
    import torch

    assert a.shape == b.shape and a.dtype == b.dtype
    dtype = torch.int32 if a.dtype == torch.float32 else torch.int16
    count = int((a.contiguous().view(dtype) != b.contiguous().view(dtype)).sum())
    return dict(bit_differences=count, exact=count == 0)


def timed(t, mesh, operation, expected, args):
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.runtime_audit import device_only

    for _ in range(3):
        with device_only():
            out = operation()
    assert bits(expected, t.to_torch(out))["exact"]
    eager = []
    for _ in range(args.samples):
        t.synchronize_device(mesh)
        start = time.perf_counter()
        with device_only():
            out = operation()
        t.synchronize_device(mesh)
        eager.append((time.perf_counter() - start) * 1000)
    assert bits(expected, t.to_torch(out))["exact"]
    out = None
    tid = t.begin_trace_capture(mesh, cq_id=0)
    with device_only():
        out = operation()
    t.end_trace_capture(mesh, tid, cq_id=0)
    try:
        t.execute_trace(mesh, tid, cq_id=0, blocking=True)
        assert bits(expected, t.to_torch(out))["exact"]
        samples = []
        for _ in range(args.samples):
            t.synchronize_device(mesh)
            start = time.perf_counter()
            for _ in range(args.replays):
                t.execute_trace(mesh, tid, cq_id=0, blocking=False)
            t.synchronize_device(mesh)
            samples.append((time.perf_counter() - start) * 1000 / args.replays)
            assert bits(expected, t.to_torch(out))["exact"]
        return dict(
            warmed_eager_ms=statistics.median(eager),
            eager_samples_ms=eager,
            traced_warmed_ms=statistics.median(samples),
            trace_samples_ms=samples,
            repeated_executions=args.samples * args.replays,
            all_bit_exact=True,
        )
    finally:
        t.release_trace(mesh, tid)


def device(args, gamma, fixtures, compute, report, save):
    import torch

    import ttnn as t
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.runtime_audit import device_only

    for name in (
        "TT_VISIBLE_DEVICES",
        "TT_METAL_WATCHER",
        "TT_METAL_DEVICE_PROFILER",
        "TT_METAL_TRACE_ALLOC_TRACKING",
        "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE",
    ):
        assert name not in os.environ, name
    loader = importlib.machinery.SourceFileLoader("_weighted_norm_interleaved_base44", str(BASE))
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader(loader.name, loader))
    loader.exec_module(module)
    mesh = None
    try:
        mesh = t.open_mesh_device(
            t.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=67108864, l1_small_size=32768
        )
        weighted = module.WeightedNormTail(t, mesh, gamma, debug=False)
        weighted_debug = module.WeightedNormTail(t, mesh, gamma, debug=True)
        candidate_debug = DirectWeightedNormTail(weighted_debug, compute)
        compact = module.LocalSiluMultiply(t, mesh, debug=False)
        baseline = CompleteAdapter(t, mesh, module, weighted, compact)
        candidate = CompleteAdapter(t, mesh, module, DirectWeightedNormTail(weighted, compute), compact)

        def upload(value, dtype=t.float32, memory=t.L1_MEMORY_CONFIG):
            return t.from_torch(
                value.contiguous(), dtype=dtype, layout=t.TILE_LAYOUT, device=mesh, memory_config=memory
            )

        for fixture in fixtures:
            core, z = upload(fixture["core_input"]), upload(fixture["z_input"], t.bfloat16)
            row = dict(name=fixture["name"], checks={})
            report["cases"].append(row)
            with device_only():
                wide, sums = baseline.prepare_core(core)
                expected, old_debug = weighted_debug(wide, sums)
                out, new_debug = candidate_debug(wide, sums)
                base_full = baseline(core, z)
                candidate_full = candidate(core, z)
            row["actual_input_layouts"] = dict(
                core=str(core.memory_config()),
                wide=str(wide.memory_config()),
                sums=str(sums.memory_config()),
                z=str(z.memory_config()),
            )
            for key in old_debug:
                row["checks"][key] = bits(t.to_torch(old_debug[key]), t.to_torch(new_debug[key]))
            row["checks"]["weighted_fp32"] = bits(t.to_torch(expected), t.to_torch(out))
            row["checks"]["complete_bf16"] = bits(t.to_torch(base_full), t.to_torch(candidate_full))
            save()
            assert all(check["exact"] for check in row["checks"].values()), row
            del core, z, wide, sums, expected, old_debug, out, new_debug, base_full, candidate_full
        # Extra direct-tail finite transport controls: zeros, subnormals, cancellation and BF16 midpoints.
        codes = torch.tensor(
            [0, -2147483648, 1, -2147483647, 0x007FFFFF, 0x00800000, 0x3F808000, 0x3F818000, -1082097664, -1082032128],
            dtype=torch.int32,
        ).view(torch.float32)
        control_core = codes.repeat(410)[:4096].reshape(1, 32, 128).contiguous()
        control_sums = torch.linspace(0, 4, 32).reshape(1, 32, 1)
        for memory_name, memory in (("L1", t.L1_MEMORY_CONFIG), ("DRAM", t.DRAM_MEMORY_CONFIG)):
            core, sums = upload(control_core, memory=memory), upload(control_sums, memory=memory)
            with device_only():
                a, ad = weighted_debug(core, sums)
                b, bd = candidate_debug(core, sums)
            row = dict(
                name="finite_transport_controls_" + memory_name,
                checks={k: bits(t.to_torch(ad[k]), t.to_torch(bd[k])) for k in ad},
            )
            row["checks"]["weighted_fp32"] = bits(t.to_torch(a), t.to_torch(b))
            report["transport_controls"].append(row)
            save()
            assert all(check["exact"] for check in row["checks"].values()), row
            del core, sums, a, b, ad, bd
        if args.timing:
            fixture = next(x for x in fixtures if x["name"] == "position3271")
            core, z = upload(fixture["core_input"]), upload(fixture["z_input"], t.bfloat16)
            expected = t.to_torch(baseline(core, z)).clone()
            for label, adapter in (("candidate", candidate), ("baseline", baseline)):
                report["timing"][label] = timed(t, mesh, lambda: adapter(core, z), expected, args)
                save()
            del core, z
        report["status"] = "all_exact_complete_adapter"
    finally:
        if mesh is not None:
            t.close_mesh_device(mesh)
            report["device_closed"] = True
        save()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--run-device", action="store_true")
    p.add_argument("--timing", action="store_true")
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--replays", type=int, default=20)
    p.add_argument("--random-cases", type=int, default=8)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(os.environ.get("HF_HOME", "/home/ttuser/qwen36-35b-a3b/hf"))
        / "hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/345d467b117d1fd5eda9b879a5988de9ae5f47a7",
    )
    args = p.parse_args()
    assert not args.output.exists() and (not args.timing or args.run_device)
    assert args.samples > 0 and args.replays > 0
    assert sha(FIXTURE_HELPER) == FIXTURE_SHA
    import torch

    from models.autoports.qwen_qwen3_6_35b_a3b.tests import v6_gated_norm_tail_timing_probe as fixture_helper

    torch.set_num_threads(4)
    args.position, args.warmup, args.trace_region_size = 3271, 3, 67108864
    gamma, fixtures, _, fixture_report = fixture_helper.prepare(args)
    for sign in (1.0, -1.0):
        fixtures.append(
            dict(
                name="zero_" + str(sign),
                core_input=torch.zeros(1, 1, 32, 128) * sign,
                z_input=torch.zeros(1, 1, 4096, dtype=torch.bfloat16) * sign,
            )
        )
    compute, proof = source_proof()
    report = dict(
        status="source_setup_pass",
        source_sha256=sha(__file__),
        source_proof=proof,
        fixture_helper_sha256=FIXTURE_SHA,
        fixture_report=fixture_report,
        fixture_names=[x["name"] for x in fixtures],
        cases=[],
        transport_controls=[],
        timing={},
        device_closed=False,
        source_only=not args.run_device,
        acceptance_eligible=False,
        timing_scope="L1 FP32 recurrent core plus L1 BF16 Z to WIDTH-sharded merged BF16; native square/sum, full tail/compact lookup and every placement included",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    for suffix, content in (
        ("source.py.snapshot", Path(__file__).read_text()),
        ("reader.cpp.snapshot", READER),
        ("compute.cpp.snapshot", compute),
    ):
        path = Path(str(args.output) + "." + suffix)
        assert not path.exists()
        path.write_text(content)
    payload = Path(str(args.output) + ".fixtures.pt")
    torch.save(dict(gamma=gamma, fixtures=fixtures), payload)
    report["fixture_payload"] = dict(path=str(payload), sha256=sha(payload))
    save()
    if args.run_device:
        try:
            device(args, gamma, fixtures, compute, report, save)
        except Exception:
            report.update(status="failed", error=traceback.format_exc())
            save()
            raise
    else:
        assert "ttnn" not in sys.modules
    assert sha(__file__) == report["source_sha256"]
    save()
    print(report["status"], report["source_sha256"], flush=True)


if __name__ == "__main__":
    main()
