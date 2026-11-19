# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exact native empty-expert schedule probe; no serving or timing claim.

Run serialized after applying/rebuilding the archived native opt-in. CPU mode
validates metadata cases only and imports neither TTNN nor a device utility.
"""

import argparse
import gc
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
CAPACITY, EXPERTS, GLOBAL_EXPERTS, MAX_COUNT = 4096, 64, 256, 320
CASES = (
    ("empty", {}, 0),
    ("late63", {63: 1}, 0),
    ("boundary", {0: 1, 31: 31, 32: 32, 63: 33}, 0),
    ("mixed", {2: 289, 31: 97, 32: 33, 63: 1}, 0),
    ("dense", dict.fromkeys(range(64), 1), 0),
    ("remapped", {1: 32, 31: 1, 32: 31, 63: 65}, 17),
    ("remapped_unchanged", {1: 32, 31: 1, 32: 31, 63: 65}, 17),
    ("empty_again", {}, 0),
    ("late63_again", {63: 1}, 0),
)


def metadata(active, shift, minimum=0, maximum=MAX_COUNT):
    counts = torch.zeros((4, GLOBAL_EXPERTS), dtype=torch.int32)
    offsets = torch.zeros_like(counts)
    indices = torch.empty((4, EXPERTS), dtype=torch.int32)
    written = torch.zeros((4, CAPACITY), dtype=torch.bool)
    for rank in range(4):
        # Rank3 is deliberately all-empty in sparse cases. Dense tests all64 on
        # every rank; ranks1/2 exercise different ownership/order distributions.
        local = (
            active
            if rank == 0 or len(active) == 64
            else ({} if rank == 3 else {(e + rank) % 64: n for e, n in active.items()})
        )
        cursor = 0
        for expert in range(EXPERTS):
            gid = rank * EXPERTS + (expert + shift) % EXPERTS
            indices[rank, expert] = gid
            count = local.get(expert, 0)
            assert 0 <= count <= MAX_COUNT
            aligned = (count + 31) // 32 * 32
            counts[rank, gid] = count
            offsets[rank, gid] = cursor
            if minimum <= count <= maximum:
                written[rank, cursor : cursor + aligned] = True
            cursor += aligned + 32  # One untouched guard tile after every region.
        assert cursor <= CAPACITY
    return counts, indices, offsets, written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cpu", "acceptance"), default="acceptance")
    parser.add_argument("--embedding", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--weight-dtype", choices=("bfp8", "bfp4", "bf16"), default="bfp8")
    parser.add_argument("--min-active-tokens", type=int, default=0)
    parser.add_argument("--max-active-tokens", type=int, default=MAX_COUNT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert 0 <= args.min_active_tokens <= args.max_active_tokens <= MAX_COUNT
    assert args.embedding >= 256 and args.embedding % 256 == 0
    assert args.hidden >= 256 and args.hidden % 256 == 0
    assert not args.output.exists(), "Preserve previous evidence: choose a fresh output path"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(
        status="RUNNING",
        command=sys.argv,
        hardware_verified=False,
        mesh=[1, 4],
        core_grid=[8, 8],
        embedding=args.embedding,
        hidden=args.hidden,
        weight_dtype=args.weight_dtype,
        local_experts=EXPERTS,
        capacity=CAPACITY,
        max_count=MAX_COUNT,
        active_token_band=[args.min_active_tokens, args.max_active_tokens],
        tracking=True,
        skip_program_cache=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        captures=0,
        releases=0,
        cases=[],
        cleanup_errors=[],
        limitations=[
            "Native component protocol with synthetic weights; no full-model accuracy or serving performance claim.",
            "Bias and stochastic rounding are intentionally unsupported by the opt-in.",
            "Sentinel protection is checked outside written tile-rows; intra-tile padding is owned by the kernel.",
        ],
    )

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    mesh, traces = None, {}
    failure = None
    save()
    try:
        torch.set_num_threads(4)
        fixtures = [
            (name, metadata(active, shift, args.min_active_tokens, args.max_active_tokens))
            for name, active, shift in CASES
        ]
        for name, (count, idx, offset, written) in fixtures:
            assert count.shape == offset.shape == (4, 256) and idx.shape == (4, 64)
            assert torch.all(offset % 32 == 0)
            assert torch.all(offset + ((count + 31) // 32 * 32) <= CAPACITY)
            assert all(len(set(row.tolist())) == 64 for row in idx)
            assert written.shape == (4, CAPACITY)
        result["metadata_cases"] = [name for name, _ in fixtures]
        if args.mode == "cpu":
            result.update(status="PASS", tracking="not_exercised", skip_program_cache="not_exercised")
            return

        os.environ.setdefault("TT_METAL_TRACE_ALLOC_TRACKING", "1")
        os.environ.setdefault("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0")
        assert os.environ["TT_METAL_TRACE_ALLOC_TRACKING"] == "1"
        assert os.environ["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"] == "0"
        import ttnn as t

        native_root = ROOT.parents[2] / "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_fused_swiglu"
        result["native_source_sha256"] = {
            str(path.relative_to(native_root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(native_root.rglob("*"))
            if path.suffix in (".cpp", ".hpp")
        }
        t.set_fabric_config(t.FabricConfig.FABRIC_1D)
        mesh = t.open_mesh_device(t.MeshShape(1, 4), trace_region_size=64 * 1024 * 1024)

        def host(value, dtype, layout, sharded=False):
            return t.from_torch(
                value.contiguous(),
                dtype=dtype,
                layout=layout,
                mesh_mapper=t.ShardTensorToMesh(mesh, dim=0) if sharded else t.ReplicateTensorToMesh(mesh),
            )

        def upload(value, dtype, layout, sharded=False):
            return t.to_device(host(value, dtype, layout, sharded), mesh, memory_config=t.DRAM_MEMORY_CONFIG)

        random = torch.Generator().manual_seed(1729)
        x = upload(torch.randn(CAPACITY, args.embedding, generator=random) * 0.125, t.bfloat16, t.ROW_MAJOR_LAYOUT)
        weight_dtype = {"bfp8": t.bfloat8_b, "bfp4": t.bfloat4_b, "bf16": t.bfloat16}[args.weight_dtype]
        weights = {name: [] for name in ("gate", "up", "down")}
        for expert in range(EXPERTS):
            for name in weights:
                shape = (args.hidden, args.embedding) if name == "down" else (args.embedding, args.hidden)
                # Different local experts have different matrices. Renumbering
                # sparse experts as compact ordinals must fail exact comparison.
                weights[name].append(upload(torch.randn(shape, generator=random) * 0.02, weight_dtype, t.TILE_LAYOUT))
        count, idx, offset, _ = fixtures[0][1]
        inputs = {
            name: upload(value, t.uint32, t.ROW_MAJOR_LAYOUT, True)
            for name, value in (("counts", count), ("indices", idx), ("offsets", offset))
        }
        sentinel_value = torch.full((CAPACITY, args.embedding), 8.0, dtype=torch.bfloat16)
        sentinel = host(sentinel_value, t.bfloat16, t.TILE_LAYOUT)
        outputs = {
            name: t.to_device(sentinel, mesh, memory_config=t.DRAM_MEMORY_CONFIG) for name in ("reference", "candidate")
        }
        fresh_outputs = {name: t.to_device(sentinel, mesh, memory_config=t.DRAM_MEMORY_CONFIG) for name in outputs}
        assert len({int(value.buffer_address()) for value in [*outputs.values(), *fresh_outputs.values()]}) == 4
        compute = t.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=t.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        def call(name, output=None, omit_flag=False):
            target = outputs[name] if output is None else output
            returned = t.experimental.deepseek_prefill.moe_fused_swiglu(
                x,
                weights["gate"],
                weights["up"],
                weights["down"],
                inputs["counts"],
                inputs["indices"],
                input_m_tiles=MAX_COUNT // 32,
                core_grid=t.CoreCoord(8, 8),
                output=target,
                expert_region_offsets=inputs["offsets"],
                read_x_at_offset=True,
                compute_kernel_config=compute,
                stochastic_rounding=False,
                min_active_tokens=args.min_active_tokens,
                max_active_tokens=args.max_active_tokens,
                **({} if omit_flag else {"compact_empty_experts": name == "candidate"}),
            )
            assert returned.buffer_address() == target.buffer_address()

        def host_bits(value):
            tensors = [t.to_torch(part).contiguous() for part in t.get_device_tensors(value)]
            assert len(tensors) == 4 and all(tensor.dtype == torch.bfloat16 for tensor in tensors)
            return [tensor.reshape(CAPACITY, args.embedding).view(torch.int16).clone() for tensor in tensors]

        def refresh(values):
            for key, value in zip(("counts", "indices", "offsets"), values[:3]):
                t.copy_host_to_device_tensor(host(value, t.uint32, t.ROW_MAJOR_LAYOUT, True), inputs[key])

        def reset(targets):
            for target in targets.values():
                t.copy_host_to_device_tensor(sentinel, target)

        def verify(name, values, targets):
            bits = {key: host_bits(value) for key, value in targets.items()}
            sentinel_bits = sentinel_value.view(torch.int16)
            for rank in range(4):
                assert torch.equal(bits["reference"][rank], bits["candidate"][rank]), (name, rank, "output bits")
                untouched = ~values[3][rank]
                for key in targets:
                    assert torch.equal(bits[key][rank][untouched], sentinel_bits[untouched]), (
                        name,
                        rank,
                        key,
                        "sentinel",
                    )

        # Metadata refresh, output reset, readback and both native signatures are
        # warm before capture. Default omission and explicit False share a key;
        # True must create a distinct program for identical tensor metadata.
        refresh(fixtures[2][1])
        reset(outputs)
        call("reference", omit_flag=True)
        entries_default = mesh.num_program_cache_entries()
        call("reference")
        assert mesh.num_program_cache_entries() == entries_default
        call("candidate")
        entries_compact = mesh.num_program_cache_entries()
        assert entries_compact > entries_default
        verify("warm", fixtures[2][1], outputs)
        reset(fresh_outputs)
        for name in outputs:
            call(name, fresh_outputs[name])
        verify("fresh_output_bindings", fixtures[2][1], fresh_outputs)
        assert mesh.num_program_cache_entries() == entries_compact
        result["program_cache_separation"] = dict(default=entries_default, compact=entries_compact)
        refresh(fixtures[0][1])
        reset(outputs)
        t.synchronize_device(mesh)
        gc.collect()
        for name in outputs:
            trace = t.begin_trace_capture(mesh, cq_id=0)
            traces[name] = trace
            try:
                call(name)
            finally:
                t.end_trace_capture(mesh, trace, cq_id=0)
            result["captures"] += 1
        result["lifetime"] = (
            "All weights, activation, routing inputs and caller-owned output buffers predate capture. "
            "No captured tensor output allocation is retained or acknowledged; no broad exemption. "
            "Native program cache is warm and tracking uses skip0. Each replay reads current metadata. "
            "Host snapshots consume both outputs after nonblocking replay before any subsequent replay."
        )
        addresses = {key: int(value.buffer_address()) for key, value in {**inputs, **outputs}.items()}
        result["addresses"] = addresses
        entries = mesh.num_program_cache_entries()
        for mode in ("eager", "trace"):
            for case, values in fixtures:
                unchanged = case.endswith("unchanged")
                if not unchanged:
                    refresh(values)
                reset(outputs)
                if mode == "eager":
                    for name in outputs:
                        call(name)
                else:
                    gc.collect()
                    for name, trace in traces.items():
                        unsafe = t._ttnn.operations.trace.get_unsafe_tracked_ids(mesh, trace)
                        assert not unsafe, (name, list(unsafe))
                        t.execute_trace(mesh, trace, cq_id=0, blocking=False)
                verify(f"{mode}/{case}", values, outputs)
                assert mesh.num_program_cache_entries() == entries
                assert addresses == {key: int(value.buffer_address()) for key, value in {**inputs, **outputs}.items()}
                result["cases"].append(dict(mode=mode, case=case, unchanged_metadata=unchanged, exact_ranks=4))
                save()
                print(f"EMPTY_EXPERT_SCHEDULE_PASS {mode} {case}", flush=True)
        result.update(status="PASS", hardware_verified=True, program_cache_entries=entries)
    except BaseException as error:
        failure = error
        result.update(status="FAIL", error=repr(error), traceback=traceback.format_exc())
    finally:
        if mesh is not None:
            for trace in traces.values():
                try:
                    t.release_trace(mesh, trace)
                    result["releases"] += 1
                except BaseException as error:
                    result["cleanup_errors"].append(repr(error))
            try:
                t.close_mesh_device(mesh)
            except BaseException as error:
                result["cleanup_errors"].append(repr(error))
        if result["cleanup_errors"]:
            result["status"] = "FAIL"
        save()
        print(
            json.dumps(
                dict(status=result["status"], hardware_verified=result["hardware_verified"], cases=len(result["cases"]))
            )
        )
    if failure is not None:
        raise failure
    if result["status"] != "PASS":
        raise RuntimeError(result["cleanup_errors"])


if __name__ == "__main__":
    main()
