# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""64-core corrected-FMA adapter and ordered GEMM diagnostic. Device use is opt-in."""

import argparse
import ast
import ctypes
import hashlib
import json
import math
import time
import traceback
from pathlib import Path

import torch

from models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_fma_sfpu_corrected_probe import Graph

DOC = Path(__file__).resolve().parents[1] / "doc/fused_decoder/endurance"
MATRIX_INPUT = DOC / "v6_chunk64_sfpu_fma_compensated_device_v2.json.cpu_matrices.pt"


def adapter_source():
    graph = Graph().build()
    assert graph.dst(len(graph.nodes) - 1) + 31 < 224
    return r"""
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/experimental/reg_api.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_ternary_sfpu.h"
#endif
void kernel_main() {
 CircularBuffer a(0); CircularBuffer b(1); CircularBuffer c(2); CircularBuffer output(16);
 compute_kernel_hw_startup(0,16); copy_init(0);
 constexpr uint32_t count=get_compile_time_arg_val(0);
 a.reserve_back(count); a.push_back(count);
 b.reserve_back(count); b.push_back(count);
 c.reserve_back(count); c.push_back(count);
 for(uint32_t t=0;t<count;++t) {
  tile_regs_acquire_math_clear(); a.wait_front(1); b.wait_front(1); c.wait_front(1);
  copy_tile(0,0,0); copy_tile(1,0,1); copy_tile(2,0,2);
#ifdef TRISC_MATH
  sfpu::_init_sfpu_config_reg();
  addr_mod_t{.srca={.incr=0},.srcb={.incr=0},.dest={.incr=0}}.set(ADDR_MOD_3);
  math::reset_counters(p_setrwc::SET_ABD_F);
  _llk_math_eltwise_sfpu_start_(0);
  for(uint32_t lane=0;lane<32;++lane) {
   GRAPH
   { sfpi::vFloat result=sfpi::dst_reg[RESULT_SLOT]; sfpi::dst_reg[224]=result; }
   sfpi::dst_reg++;
  }
  _llk_math_eltwise_sfpu_done_();
#endif
  tile_regs_commit(); tile_regs_wait(); output.reserve_back(1);
  pack_tile(7,16); output.push_back(1);
  a.pop_front(1); b.pop_front(1); c.pop_front(1); tile_regs_release_math_clear();
 }
}
""".replace(
        "GRAPH", graph.cpp()
    ).replace(
        "RESULT_SLOT", str(graph.dst(graph.result))
    )


class CorrectedFma:
    """Return a*b+acc with the measured bounded SFPU correction, on device only.

    Full/row/column/scalar spatial inputs are expanded with native WHERE using
    identical branches. All inputs have identical leading dimensions. Output is
    height sharded; a following call can reuse that accumulator without a copy.
    This diagnostic does not claim general subnormal correctness or fast GEMM.
    """

    def __init__(self, ttnn, mesh, max_cores=64):
        self.t = ttnn
        self.mesh = mesh
        self.max_cores = max_cores
        self.source = adapter_source()
        self.plans = {}
        self.conditions = {}
        self.last_inputs = None
        self.last_plan = None

    def plan(self, shape):
        if shape in self.plans:
            return self.plans[shape]
        t = self.t
        assert len(shape) == 4 and shape[-2] % 32 == 0 and shape[-1] % 32 == 0, shape
        height = math.prod(shape[:-1])
        tile_rows = height // 32
        count = min(self.max_cores, tile_rows)
        while tile_rows % count:
            count -= 1
        assert count <= 64
        coords = [t.CoreCoord(i % 8, i // 8) for i in range(count)]
        cores = t.CoreRangeSet([t.CoreRange(x, x) for x in coords])
        shard_shape = (height // count, shape[-1])
        memory = t.MemoryConfig(
            t.TensorMemoryLayout.HEIGHT_SHARDED,
            t.BufferType.L1,
            t.ShardSpec(cores, shard_shape, t.ShardOrientation.ROW_MAJOR),
        )
        config = t.ComputeConfigDescriptor(
            math_fidelity=t.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, dst_full_sync_en=True
        )
        modes = [t.UnpackToDestMode.Default] * 64
        for index in (0, 1, 2):
            modes[index] = t.UnpackToDestMode.UnpackToDestFp32
        config.unpack_to_dest_mode = modes
        tiles = math.prod(shard_shape) // 1024
        result = dict(cores=cores, memory=memory, config=config, tiles=tiles, core_count=count, shard_shape=shard_shape)
        self.plans[shape] = result
        return result

    def broadcast(self, x, shape):
        t = self.t
        own_shape = tuple(x.shape)
        assert x.dtype == t.float32, x.dtype
        assert own_shape[:-2] == shape[:-2], (own_shape, shape)
        assert all(a == b or a == 1 for a, b in zip(own_shape[-2:], shape[-2:])), (own_shape, shape)
        if own_shape == shape:
            return x
        if shape not in self.conditions:
            self.conditions[shape] = t.zeros(
                shape, dtype=t.float32, layout=t.TILE_LAYOUT, device=self.mesh, memory_config=t.DRAM_MEMORY_CONFIG
            )
        # The branches are identical; the full-shaped predicate forces expansion
        # without changing signed zeros or full FP32 mantissas by arithmetic.
        x = t.to_memory_config(x, t.DRAM_MEMORY_CONFIG)
        return t.where(self.conditions[shape], x, x, memory_config=t.DRAM_MEMORY_CONFIG)

    def __call__(self, acc, a, b):
        t = self.t
        shape = tuple(acc.shape)
        plan = self.plan(shape)
        aa = t.to_memory_config(self.broadcast(a, shape), plan["memory"])
        bb = t.to_memory_config(self.broadcast(b, shape), plan["memory"])
        cc = t.to_memory_config(acc, plan["memory"])
        out = t.empty(shape, dtype=t.float32, layout=t.TILE_LAYOUT, device=self.mesh, memory_config=plan["memory"])
        kernel = t.KernelDescriptor(
            kernel_source=self.source,
            source_type=t.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=plan["cores"],
            compile_time_args=[plan["tiles"]],
            runtime_args=[],
            config=plan["config"],
        )
        program = t.ProgramDescriptor(
            kernels=[kernel],
            semaphores=[],
            cbs=[
                t.cb_descriptor_from_sharded_tensor(index, tensor)
                for index, tensor in ((0, aa), (1, bb), (2, cc), (16, out))
            ],
        )
        t.generic_op([aa, bb, cc, out], program)
        self.last_inputs = (aa, bb, cc)
        self.last_plan = dict(
            core_count=plan["core_count"], shard_shape=plan["shard_shape"], tiles_per_core=plan["tiles"]
        )
        return out


class CorrectedMatmul:
    """Diagnostic increasing-K FMA recurrence; one corrected kernel call per K."""

    def __init__(self, fma):
        self.fma = fma

    def __call__(self, a, b):
        t = self.fma.t
        ashape, bshape = tuple(a.shape), tuple(b.shape)
        assert ashape[:-2] == bshape[:-2] and ashape[-1] == bshape[-2] and ashape[-1] > 0
        shape = (*ashape[:-2], ashape[-2], bshape[-1])
        result = t.zeros(
            shape, dtype=t.float32, layout=t.TILE_LAYOUT, device=self.fma.mesh, memory_config=t.DRAM_MEMORY_CONFIG
        )
        for k in range(ashape[-1]):
            result = self.fma(result, a[..., :, k : k + 1], b[..., k : k + 1, :])
        return t.to_memory_config(result, t.DRAM_MEMORY_CONFIG)


def compare(expected, actual):
    expected, actual = expected.contiguous(), actual.contiguous()
    difference = expected - actual
    return dict(
        fp32_bits_different=int((expected.view(torch.int32) != actual.view(torch.int32)).sum()),
        numeric_different=int((expected != actual).sum()),
        bf16_different=int((expected.bfloat16() != actual.bfloat16()).sum()),
        max_abs=float(difference.abs().max()),
        shape=list(actual.shape),
    )


def fma_oracle(acc, a, b):
    libm = ctypes.CDLL("libm.so.6")
    libm.fmaf.argtypes = [ctypes.c_float] * 3
    libm.fmaf.restype = ctypes.c_float
    a, b, acc = torch.broadcast_tensors(a, b, acc)
    return torch.tensor(
        [libm.fmaf(x, y, z) for x, y, z in zip(a.flatten().tolist(), b.flatten().tolist(), acc.flatten().tolist())],
        dtype=torch.float32,
    ).reshape(acc.shape)


def adapter_fixtures(heads):
    generator = torch.Generator().manual_seed(613645)
    shape = (1, heads, 64, 128)
    result = []
    for name, ashape, bshape in [
        ("full", shape, shape),
        ("col_row", (1, heads, 64, 1), (1, heads, 1, 128)),
        ("row_col", (1, heads, 1, 128), (1, heads, 64, 1)),
        ("scalar_full", (1, heads, 1, 1), shape),
    ]:
        a = torch.randn(ashape, generator=generator) * 0.03
        b = torch.randn(bshape, generator=generator) * 0.01
        acc = torch.randn(shape, generator=generator) * 0.001
        # Exercise signed zero preservation through both broadcast directions.
        a.flatten()[::17] = -0.0
        b.flatten()[::19] = -0.0
        acc.flatten()[::23] = -0.0
        result.append(dict(name=name, a=a, b=b, acc=acc, expected=fma_oracle(acc, a, b)))
    return result


def matrix_fixtures(heads, selected, k_limit):
    saved = torch.load(MATRIX_INPUT, weights_only=False)
    case = saved["cases"][128]
    matrices = case["matrices"]
    key = case["operands"]["key"]
    fixtures = []
    for chunk in range(2):
        get = lambda name: matrices[f"chunk{chunk}/{name}"]
        state = saved["initial"] if chunk == 0 else matrices["chunk0/state"]
        pairs = [
            (f"chunk{chunk}_gram", get("k_beta"), key[:, :, chunk * 64 : (chunk + 1) * 64].transpose(-1, -2)),
            (f"chunk{chunk}_transformed_v", get("A_final"), get("v_beta")),
            (f"chunk{chunk}_transformed_k", get("A_final"), get("k_beta") * get("cumulative_g").exp()),
            (f"chunk{chunk}_state_contribution", get("transformed_k"), state),
            (f"chunk{chunk}_state_update", get("weighted_k").transpose(-1, -2), get("v_new")),
        ]
        for name, a, b in pairs:
            if selected and name not in selected:
                continue
            a, b = a[:, :heads].contiguous(), b[:, :heads].contiguous()
            if k_limit:
                a, b = a[..., :k_limit].contiguous(), b[..., :k_limit, :].contiguous()
            fixtures.append(dict(name=name, a=a, b=b, expected=a @ b))
    assert fixtures, selected
    return fixtures


def runtime_audit():
    tree = ast.parse(Path(__file__).read_text())
    forbidden = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in ("CorrectedFma", "CorrectedMatmul"):
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    call = ast.unparse(child.func)
                    if call.startswith("torch.") or call.endswith((".from_torch", ".to_torch")):
                        forbidden.append(call)
    assert not forbidden
    return dict(
        torch_or_host_conversion_calls=forbidden,
        broadcast="native FP32 WHERE with identical branches",
        placement="height-sharded L1, up to 8x8 cores; direct FP32 unpack for all three input CBs",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-device", action="store_true")
    parser.add_argument("--mode", choices=("adapter", "matrices"), default="adapter")
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--matrices", default="")
    parser.add_argument("--k-limit", type=int, default=0)
    args = parser.parse_args()
    assert not args.output.exists(), args.output
    assert 1 <= args.heads <= 32
    torch.set_num_threads(2)
    fixtures = (
        adapter_fixtures(args.heads)
        if args.mode == "adapter"
        else matrix_fixtures(args.heads, set(filter(None, args.matrices.split(","))), args.k_limit)
    )
    source = adapter_source()
    result = dict(
        status="source_emitted_unexecuted",
        classification="corrected FMA/GEMM component diagnostic; not decoder acceptance",
        mode=args.mode,
        heads=args.heads,
        k_limit=args.k_limit,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        arithmetic_source_sha256=hashlib.sha256(
            Path(__file__).with_name("v6_fma_sfpu_corrected_probe.py").read_bytes()
        ).hexdigest(),
        compute_sha256=hashlib.sha256(source.encode()).hexdigest(),
        arithmetic_cpp_sha256=hashlib.sha256(Graph().build().cpp().encode()).hexdigest(),
        runtime_audit=runtime_audit(),
        cases=[],
        device_closed=False,
        limitations="Bounded normal FMA only; FTZ/subnormal differences remain possible in actual matrices. Per-K dispatch, broadcasting and sharding are diagnostic, not optimized. No latency acceptance claim.",
    )
    for case in fixtures:
        result["cases"].append(
            dict(
                name=case["name"],
                input_shapes={k: list(case[k].shape) for k in ("a", "b")},
                expected_shape=list(case["expected"].shape),
                input_sha256={
                    k: hashlib.sha256(v.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                    for k, v in case.items()
                    if isinstance(v, torch.Tensor)
                },
            )
        )
    args.output.with_suffix(".compute.cpp.snapshot").write_text(source)
    fixtures_path = args.output.with_suffix(".fixtures.pt")
    torch.save(fixtures, fixtures_path)
    result["fixtures_sha256"] = hashlib.sha256(fixtures_path.read_bytes()).hexdigest()

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    if not args.run_device:
        return
    import ttnn
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.runtime_audit import device_only

    mesh = None
    try:
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], l1_small_size=32768)
        fma = CorrectedFma(ttnn, mesh)
        mm = CorrectedMatmul(fma)

        def upload(x):
            return ttnn.from_torch(
                x.contiguous(),
                device=mesh,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        measured = {}
        for case, row in zip(fixtures, result["cases"]):
            da, db = upload(case["a"]), upload(case["b"])
            dc = upload(case["acc"]) if args.mode == "adapter" else None
            ttnn.synchronize_device(mesh)
            start = time.perf_counter()
            with device_only():
                out = fma(dc, da, db) if args.mode == "adapter" else mm(da, db)
            ttnn.synchronize_device(mesh)
            row["diagnostic_cold_elapsed_ms"] = (time.perf_counter() - start) * 1000
            actual = ttnn.to_torch(out)
            measured[case["name"]] = actual
            row["comparison"] = compare(case["expected"], actual)
            row["plan"] = fma.last_plan
            if args.mode == "adapter":
                aa, bb, cc = torch.broadcast_tensors(case["a"], case["b"], case["acc"])
                row["expanded_inputs"] = [
                    compare(want, ttnn.to_torch(got)) for want, got in zip((aa, bb, cc), fma.last_inputs)
                ]
            save()
            print(case["name"], row["comparison"], flush=True)
        torch.save(measured, args.output.with_suffix(".device.pt"))
        exact = all(row["comparison"]["fp32_bits_different"] == 0 for row in result["cases"])
        inputs = all(
            all(v["fp32_bits_different"] == 0 for v in row.get("expanded_inputs", [])) for row in result["cases"]
        )
        result["status"] = "component_exact_pass" if exact and inputs else "component_measured_differences"
    except Exception:
        result.update(status="error", error=traceback.format_exc())
        raise
    finally:
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
            result["device_closed"] = True
        save()


if __name__ == "__main__":
    main()
