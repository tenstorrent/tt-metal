# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real-weight packed/split dense decode projection search, including conversions.

Run serially under timeout with TT_METAL_TRACE_ALLOC_TRACKING=1. Results are
component candidates, not decoder acceptance. Inputs are seeded normalized BF16
representative activations; they are not recorded full-model activations. API
errors are recorded as blockers to investigate, never optimization rejections.
"""

import argparse
import gc
import itertools
import json
import math
import os
import statistics
import time
import traceback
from pathlib import Path

import torch

from models.autoports.qwen_qwen3_8_flash_next.tests.probe_components import load_weights, pcc

ROLES = {
    "qsa": (3, ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight")),
    "gdn": (0, ("linear_attn.in_proj_z.weight", "linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight")),
    "shared": (
        0,
        ("mlp.shared_expert.gate_proj.weight", "mlp.shared_expert.up_proj.weight", "mlp.shared_expert_gate.weight"),
    ),
    "ple": (1, ("ple.key_proj.weight", "ple.value_proj.weight")),
}


def integers(value):
    return tuple(int(x) for x in value.split(","))


def host_operands(snapshot, role, batch):
    layer, names = ROLES[role]
    # Load only these matrices: loading all mlp weights would load every expert.
    weights = [load_weights(snapshot, layer, name)[name].bfloat16() for name in names]
    if len({w.shape[1] for w in weights}) != 1:
        raise ValueError("Packed projections must share their input width")
    torch.manual_seed(819)
    x = torch.randn(1, 1, batch, weights[0].shape[1])
    x = (x / x.square().mean(-1, keepdim=True).sqrt()).bfloat16()
    refs = [(x.float() @ w.float().T).bfloat16() for w in weights]
    return names, x, weights, refs


def candidates(args, k):
    for dtype, fidelity, packed in itertools.product(args.dtypes.split(","), args.fidelities.split(","), (False, True)):
        for cores in args.cores:
            if k % (32 * cores):
                continue
            common = dict(dtype=dtype, fidelity=fidelity, packed=packed, cores=cores)
            if args.storage in ("all", "interleaved"):
                yield dict(common, storage="interleaved", readers=1, block=None)
            if args.storage in ("all", "dram"):
                for readers, block in itertools.product(args.readers, args.blocks):
                    # Include the entire legal divisor family of the input shard.
                    if (k // (32 * cores)) % block == 0:
                        yield dict(common, storage="dram", readers=readers, block=block)


def make_group(ttnn, mesh, x, host_weights, config):
    """Return a callable from interleaved BF16 input to separate BF16 outputs."""
    cores, packed = config["cores"], config["packed"]
    device_grid = mesh.compute_with_storage_grid_size()
    grid = ttnn.num_cores_to_corerangeset(cores, device_grid, row_wise=True)
    k = x.shape[-1]
    height = math.ceil(x.shape[-2] / 32) * 32
    widths = [w.shape[0] for w in host_weights]
    padded_widths = [math.ceil(n / 32) * 32 for n in widths]
    # Internal tile padding keeps small a/b/router projections separate when
    # slicing the packed result. Public output widths remain the logical widths.
    matrices = [
        torch.nn.functional.pad(w.T.contiguous(), (0, n - w.shape[0])) for w, n in zip(host_weights, padded_widths)
    ]
    if packed:
        matrices = [torch.cat(matrices, dim=-1)]
    compute = ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=getattr(ttnn.MathFidelity, config["fidelity"]),
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    def l1_config(width):
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, (height, width), ttnn.ShardOrientation.ROW_MAJOR),
        )

    specs, weights = [], []
    for matrix in matrices:
        n = matrix.shape[-1]
        if config["storage"] == "dram":
            banks = mesh.dram_grid_size().x
            # The one-reader control and multi-reader arms use identical bank
            # padding. The pipeline can handle uneven reader subdivisions.
            bank_width = math.ceil(n / (32 * banks)) * 32
            dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
            memory = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(dram_grid, (k, bank_width), ttnn.ShardOrientation.ROW_MAJOR),
            )
            output_width = math.ceil(n / (32 * cores)) * 32
            program = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=config["block"],
                per_core_M=height // 32,
                per_core_N=output_width // 32,
                fused_activation=None,
                num_workers_per_dram_bank=config["readers"],
                use_multicore_pipeline=config["readers"] >= 2,
            )
            specs.append(dict(program_config=program, memory_config=l1_config(output_width)))
        else:
            memory = ttnn.DRAM_MEMORY_CONFIG
            # Explicit 1D core grid candidate rather than the default heuristic.
            legal_x = [n for n in range(1, device_grid.x + 1) if cores % n == 0 and cores // n <= device_grid.y]
            if not legal_x:
                raise ValueError(f"Interleaved core count {cores} cannot form this device's rectangular grid")
            core_grid = ttnn.CoreGrid(x=max(legal_x), y=cores // max(legal_x))
            specs.append(dict(core_grid=core_grid, memory_config=ttnn.DRAM_MEMORY_CONFIG))
        weights.append(
            ttnn.from_torch(
                matrix.reshape(1, 1, k, n),
                dtype=getattr(ttnn, config["dtype"]),
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=memory,
            )
        )

    def run():
        # Both conversions are inside the captured group. Every candidate has
        # the same external input/output layout contract as the fused decoder.
        activation = ttnn.to_memory_config(x, l1_config(k // cores)) if config["storage"] == "dram" else x
        outputs = []
        for weight, spec in zip(weights, specs):
            y = ttnn.linear(activation, weight, dtype=ttnn.bfloat16, compute_kernel_config=compute, **spec)
            if config["storage"] == "dram":
                y = ttnn.to_memory_config(y, ttnn.DRAM_MEMORY_CONFIG)
            outputs.append(y)
        if packed:
            offset, result = 0, []
            for width, padded in zip(widths, padded_widths):
                result.append(outputs[0][..., offset : offset + width])
                offset += padded
            return result
        return [value[..., :width] for value, width in zip(outputs, widths)]

    return run


def measure(ttnn, mesh, run, expected, repeats):
    from ttnn.tools import trace_allocation_tracker

    trace = None
    capturing = False
    try:
        # Warm every forward/readback program before capture. No host copies or
        # new program-cache allocations occur inside the measured trace region.
        warm = run()
        eager = [ttnn.to_torch(value) for value in warm]
        del warm
        warm = run()
        ttnn.synchronize_device(mesh)
        del warm
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        capturing = True
        outputs = run()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        capturing = False
        # These exact final outputs are the only intentionally overwritten
        # retained buffers; all weights/input storage remains live and immutable.
        for value in outputs:
            trace_allocation_tracker.acknowledge_corruptible(value)
        times = []
        for _ in range(repeats + 2):
            start = time.perf_counter()
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            times.append((time.perf_counter() - start) * 1000)
        actual = [ttnn.to_torch(value) for value in outputs]
        exact = all(torch.equal(a, e) for a, e in zip(actual, eager))
        scores = [pcc(e, a) if e.numel() > 1 else None for e, a in zip(expected, actual)]
        scores = [score if score is not None and math.isfinite(score) else None for score in scores]
        expected_all = torch.cat([e.flatten() for e in expected])
        actual_all = torch.cat([a.flatten() for a in actual])
        whole_pcc = pcc(expected_all, actual_all)
        finite = bool(torch.isfinite(actual_all).all())
        return dict(
            status="measured",
            traced_ms=statistics.median(times[2:]),
            samples_ms=times[2:],
            whole_pcc=whole_pcc if math.isfinite(whole_pcc) else None,
            projection_pcc=scores,
            max_abs=float((expected_all.float() - actual_all.float()).abs().max()) if finite else None,
            replay_exact=exact,
            finite=finite,
            passes_component_bar=finite and exact and whole_pcc >= 0.995,
        )
    finally:
        if trace is not None:
            if capturing:
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
            ttnn.release_trace(mesh, trace)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role", choices=ROLES, required=True)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--batch", type=int, default=1, choices=(1, 32))
    parser.add_argument("--dtypes", default="bfloat8_b,bfloat16")
    parser.add_argument("--fidelities", default="HiFi2,LoFi")
    parser.add_argument("--readers", type=integers, default=(1, 2, 3))
    parser.add_argument("--blocks", type=integers, default=(1, 2, 4, 5, 8, 10, 16))
    parser.add_argument("--cores", type=integers, default=(8, 16))
    parser.add_argument("--storage", choices=("all", "dram", "interleaved"), default="all")
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("--repeats must be at least 2")
    if os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") != "1":
        parser.error("Set TT_METAL_TRACE_ALLOC_TRACKING=1 for this trace experiment")
    torch.set_num_threads(8)
    names, host_x, weights, reference = host_operands(args.snapshot, args.role, args.batch)
    import ttnn

    report = dict(
        role=args.role,
        snapshot=str(args.snapshot.resolve()),
        names=names,
        shapes=[list(w.shape) for w in weights],
        batch=args.batch,
        activation_provenance="seed819 normalized random BF16; representative distribution, not recorded model activations",
        reference="FP32 torch matmul of BF16 checkpoint weights/inputs rounded to BF16",
        timing="serial warmed blocking trace replay; includes input/output memory conversion and packed output slicing",
        trace_alloc_tracking=True,
        rows=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        report["arch"] = str(mesh.arch())
        report["dram_banks"] = mesh.dram_grid_size().x
        x = ttnn.from_torch(host_x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        for config in candidates(args, host_x.shape[-1]):
            row = dict(config)
            run = None
            try:
                run = make_group(ttnn, mesh, x, weights, config)
                row.update(measure(ttnn, mesh, run, reference, args.repeats))
            except Exception as error:
                row.update(status="blocker_requires_investigation", error=str(error), traceback=traceback.format_exc())
            finally:
                # measure releases the trace before the next geometry/weights
                # are constructed; no allocation is reused across live traces.
                run = None
                gc.collect()
                report["rows"].append(row)
                save()
            print(
                json.dumps({k: v for k, v in row.items() if k not in ("traceback", "samples_ms", "error")}), flush=True
            )
    finally:
        ttnn.close_mesh_device(mesh)
        save()


if __name__ == "__main__":
    main()
