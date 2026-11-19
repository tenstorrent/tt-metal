# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device-only scalar selector diagnostic; no production selection is implied."""

import argparse
import hashlib
import json
import os
import statistics
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record
from models.autoports.qwen_qwen3_6_35b_a3b.tests.v5_router_tie_probe import DOC, assess, canonical, router_function


class InlineSelector:
    """Scratch and outputs are setup-owned; scalar work executes on BRISC."""

    def __init__(self, ttnn, mesh, selector_cpp):
        self.ttnn = ttnn
        self.mesh = mesh
        core = ttnn.CoreCoord(0, 0)
        self.cores = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
        self.shard = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(self.cores, (32, 32), ttnn.ShardOrientation.ROW_MAJOR),
        )
        ids_shard = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(self.cores, (1, 16), ttnn.ShardOrientation.ROW_MAJOR),
        )
        self.values = ttnn.from_torch(
            torch.zeros(1, 8), device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=self.shard
        )
        self.ids = ttnn.from_torch(
            torch.zeros(1, 8, dtype=torch.int32),
            device=mesh,
            dtype=ttnn.uint16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ids_shard,
        )
        self.compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.kernel_source = (
            '#include "api/dataflow/dataflow_api.h"\n'
            + selector_cpp
            + r"""
void kernel_main() {
    auto* src = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_arg_val<uint32_t>(0));
    auto* values = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_arg_val<uint32_t>(1));
    auto* indices = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_arg_val<uint32_t>(2));
    cb_reserve_back(0, 1);
    auto* scratch = reinterpret_cast<uint32_t*>(get_write_ptr(0));
    for (uint32_t i = 0; i < 256; ++i) scratch[i] = (uint32_t(src[i]) << 16) | i;
    v6_select_top8(scratch);
    for (uint32_t i = 0; i < 1024; ++i) values[i] = 0;
    for (uint32_t i = 0; i < 16; ++i) indices[i] = 0;
    for (uint32_t i = 0; i < 8; ++i) {
        values[i] = scratch[i] >> 16;
        indices[i] = scratch[i] & 65535;
    }
    cb_push_back(0, 1);
    cb_wait_front(0, 1);
    cb_pop_front(0, 1);
}
"""
        )
        runtime = ttnn.RuntimeArgs()
        runtime[0][0] = [0, self.values.buffer_address(), self.ids.buffer_address()]
        self.kernel = ttnn.KernelDescriptor(
            kernel_source=self.kernel_source,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=self.cores,
            runtime_args=runtime,
            config=ttnn.WriterConfigDescriptor(),
        )
        self.program = ttnn.ProgramDescriptor(
            kernels=[self.kernel],
            cbs=[
                ttnn.CBDescriptor(
                    total_size=4096,
                    core_ranges=self.cores,
                    format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.uint32, 4096)],
                )
            ],
        )

    def __call__(self, logits):
        ttnn = self.ttnn
        face = ttnn.reshape(logits, (1, 16, 16))
        face = ttnn.to_memory_config(face, self.shard)
        # Runtime arguments are updated, but the kernel/program definition is reused.
        self.program.kernels[0].runtime_args[0][0] = [
            face.buffer_address(),
            self.values.buffer_address(),
            self.ids.buffer_address(),
        ]
        ttnn.generic_op([face, self.values, self.ids], self.program)
        values = ttnn.to_memory_config(self.values, ttnn.DRAM_MEMORY_CONFIG)
        values = ttnn.typecast(values, ttnn.float32)
        scores = ttnn.softmax(values, dim=-1, numeric_stable=True, compute_kernel_config=self.compute)
        scores = ttnn.typecast(scores, ttnn.bfloat16)
        routing = ttnn.reshape(scores, (1, 8, 1, 1))
        ids = ttnn.reshape(ttnn.to_memory_config(self.ids, ttnn.DRAM_MEMORY_CONFIG), (1, 1, 1, 8))
        return routing, ids


def fixtures(random_cases):
    result = []
    proof_path = DOC / "v5_torch_topk_tie_permutation_proof.json"
    proof = json.loads(proof_path.read_text())
    for n, case in enumerate(proof["cases"]):
        x = torch.tensor(case["input_bf16_codes"], dtype=torch.int16).view(torch.bfloat16).reshape(1, 256)
        result.append((f"permutation_{n}", x))
    for filename in (
        "v5_combined_layer34_stream_probe.json.pos181.pt",
        "v5_accurate_residual_layer34_worst_probe.json.pos7944.pt",
    ):
        capture = torch.load(DOC / filename, weights_only=False, map_location="cpu")
        for name, x in (("hf", capture["hf_values"]["router"][0]), ("tt", capture["tt_values"]["router"])):
            result.append((f"pos{capture['position']}_{name}", x.reshape(1, 256).bfloat16()))
    generator = torch.Generator().manual_seed(74612)
    for n in range(random_cases):
        x = torch.randn(1, 256, generator=generator).bfloat16()
        if n % 2:
            x = (x.float() * 4).round().div(4).bfloat16()
        result.append((f"random_{n}", x))
    return result


def run(args):
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_cpu_topk_selector_proof import PORTABLE_CPP as SELECTOR_CPP

    assert not args.output.exists(), args.output
    for key in ("TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER", "TT_METAL_TRACE_ALLOC_TRACKING"):
        assert os.environ.get(key) in (None, "", "0"), key
    torch.set_num_threads(4)
    cases = fixtures(args.random_cases)
    result = {
        "status": "started",
        "classification": "component diagnostic; not model acceptance",
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "selector_cpp_sha256": hashlib.sha256(SELECTOR_CPP.encode()).hexdigest(),
        "cases": [],
        "timings": [],
        "device_closed": False,
        "scope": "Complete BF16-logit router to BF16 TILE routing and UINT16 RM sparse IDs, including sharding, output copies, casts and FP32 softmax; projection excluded equally.",
        "limits": "Raw BF16 ordering is checked against CPU softmax/topk per input; universal equivalence is not claimed. Device scalar selection executes on BRISC; no host tensor operation in the runtime call.",
    }

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    record(args.output)
    import ttnn
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.runtime_audit import device_only

    mesh = None
    try:
        mesh = ttnn.open_mesh_device(
            ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=32 * 1024 * 1024, l1_small_size=32768
        )
        selector = InlineSelector(ttnn, mesh, SELECTOR_CPP)
        result["kernel_source_sha256"] = hashlib.sha256(selector.kernel_source.encode()).hexdigest()
        obj = SimpleNamespace(compute=selector.compute, router_shard=selector.shard)

        def upload(x, dtype, layout=ttnn.TILE_LAYOUT, memory=ttnn.DRAM_MEMORY_CONFIG):
            return ttnn.from_torch(x.contiguous(), device=mesh, dtype=dtype, layout=layout, memory_config=memory)

        obj.router_bias = upload(torch.zeros(1, 16, 16), ttnn.bfloat16, memory=selector.shard)
        obj.router_indices = upload(
            torch.arange(256).reshape(1, 16, 16).transpose(1, 2), ttnn.uint16, memory=selector.shard
        )
        obj.router_out = upload(torch.zeros(1, 32, 32), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, selector.shard)
        obj.router_out_ids = upload(
            torch.zeros(1, 32, 32, dtype=torch.int32), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT, selector.shard
        )
        native, _ = router_function(DOC.parent / "selected_fused_decoder_v5.py.snapshot", ttnn)
        for name, x in cases:
            dx = upload(x, ttnn.bfloat16)
            expected = canonical(x)
            with device_only():
                values = selector(dx)
            host = [ttnn.to_torch(v) for v in values]
            assessment = assess(host, expected, expected[1].tolist())
            assessment["exact_cpu_id_order"] = host[1].flatten().long().tolist() == expected[1].tolist()
            assessment["input_immutable"] = torch.equal(ttnn.to_torch(dx), x)
            result["cases"].append(dict(name=name, **assessment))
            assert (
                assessment["same_input_cpu_membership"]
                and assessment["exact_cpu_id_order"]
                and assessment["input_immutable"]
            ), name
            if name == "pos7944_tt":
                for variant, call in [("native", lambda: native(obj, dx)), ("inline", lambda: selector(dx))]:
                    for _ in range(3):
                        with device_only():
                            out = call()
                    reference = [ttnn.to_torch(v) for v in out]
                    out = None
                    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                    with device_only():
                        out = call()
                    ttnn.end_trace_capture(mesh, trace, cq_id=0)
                    try:
                        samples = []
                        for _ in range(10):
                            ttnn.synchronize_device(mesh)
                            start = time.perf_counter()
                            for _ in range(10):
                                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                            ttnn.synchronize_device(mesh)
                            samples.append((time.perf_counter() - start) * 1000 / 10)
                        repeated = all(torch.equal(a, ttnn.to_torch(b)) for a, b in zip(reference, out))
                        assert repeated
                        result["timings"].append(
                            dict(
                                variant=variant,
                                median_ms=statistics.median(samples),
                                samples_ms=samples,
                                bit_equal_after_100_replays=repeated,
                            )
                        )
                    finally:
                        ttnn.release_trace(mesh, trace)
            save()
        result["status"] = "component_pass"
    except Exception:
        result.update(status="error", error=traceback.format_exc())
        raise
    finally:
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
            result["device_closed"] = True
        save()
    print(result["status"], len(result["cases"]), result["timings"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DOC / "v6_inline_selector_device.json")
    parser.add_argument("--random-cases", type=int, default=64)
    run(parser.parse_args())
