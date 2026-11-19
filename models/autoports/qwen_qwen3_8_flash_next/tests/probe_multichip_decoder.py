# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serial real-checkpoint baseline/mesh comparison with persisted exact operands."""

import argparse
import gc
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from transformers import AutoConfig
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_components import load_weights, pcc
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_linear_contract import runtime_audit
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_ple_decoder import load_reference as load_ple_reference
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_ple_decoder import token_rows
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import (
    Collectives,
    MeshPackedEmbedding,
    MultichipDecoder,
)
from models.autoports.qwen_qwen3_8_flash_next.tt.optimized_decoder import OptimizedDecoder
from models.autoports.qwen_qwen3_8_flash_next.tt.packed_embedding import PackedEmbedding


def validate_options(a):
    """Reject invalid probe requests before loading weights or opening hardware."""
    if a.layer not in (0, 1, 3):
        raise ValueError("This representative-layer probe supports layers 0, 1, and 3")
    if a.capacity < 128 or a.capacity > 262144 or a.capacity % 128:
        raise ValueError("Capacity must be a multiple of 128 in 128..262144")
    if not 1 <= a.length <= a.capacity - 3:
        raise ValueError("Length must leave capacity for the three subsequent decode tokens")
    if not 0 <= a.batch <= 32:
        raise ValueError("Batch must be 0 (disabled) or 1..32")
    if getattr(a, "batch_changed_inputs", False) and (a.batch < 2 or a.clean_timing):
        raise ValueError("Changed-input batch validation requires batch2..32 and tracked validation, not clean timing")
    if a.tails and a.length < 32:
        raise ValueError("Tail sweep requires at least 32 reference input rows")
    if a.components and (a.trace or a.batch or a.tails or a.stack or a.profile):
        raise ValueError("Component retention requires a standalone eager diagnostic run")
    if (a.trace or a.batch) and a.repeats < 3:
        raise ValueError("Trace timing requires at least three replays, including two warmups")
    if (a.trace or a.batch) and not a.clean_timing:
        if not trace_allocation_tracker.TRACE_ALLOC_TRACKING:
            raise ValueError("Set TT_METAL_TRACE_ALLOC_TRACKING=1 for trace validation")
        if os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") != "0":
            raise ValueError("Trace validation requires TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0")
    if a.profile and os.environ.get("TT_METAL_WATCHER"):
        raise ValueError("Watcher and profiler evidence must use separate runs")
    if a.reserve_bytes < 0 or (a.reserve_bytes and a.mode != "mesh"):
        raise ValueError("A positive full-stack reservation belongs to the target mesh profile")
    if a.expert_dtype not in ("bfloat4_b", "bfloat8_b", "bfloat16"):
        raise ValueError("Unsupported expert dtype")
    if getattr(a, "reduction", "default") != "default" and a.mode != "mesh":
        raise ValueError("Reduction candidates require mesh mode")
    if getattr(a, "moe_timing", False) and (a.mode != "mesh" or not a.trace):
        raise ValueError("MoE timing requires mesh mode and a warmed decode trace")


def reference_metadata(a):
    metadata = dict(schema=1, snapshot=str(a.snapshot.resolve()), layer=a.layer, length=a.length, capacity=a.capacity)
    if getattr(a, "batch_fixture", None):
        metadata["prompt_fixture"] = str(a.batch_fixture.resolve())
    return metadata


def validate_reference(a, saved):
    """Require matching operands and complete requested evidence before mesh open."""
    if saved.get("metadata") != reference_metadata(a):
        raise ValueError("Reference metadata is absent or mismatched; regenerate the baseline for these options")
    shapes = [(1, 1, a.length, 10240)] + [(1, 1, 1, 10240)] * 3
    for key in ("inputs", "outputs"):
        values = saved.get(key, [])
        if len(values) != 4 or any(tuple(v.shape) != shape for v, shape in zip(values, shapes)):
            raise ValueError(f"Reference {key} must contain the requested prefill and three decode tensors")
        if any(v.dtype != torch.bfloat16 or not torch.isfinite(v).all() for v in values):
            raise ValueError(f"Reference {key} must contain finite BF16 tensors")
    names = (
        {"position", "page_table", "k_cache", "v_cache", "raw_history", "compressed_keys"}
        if a.layer == 3
        else {"conv", "recurrent"} | ({"ple_conv", "token_history"} if a.layer == 1 else set())
    )
    states = saved.get("states", [])
    if len(states) != 4 or any(set(state) != names for state in states):
        raise ValueError("Reference must include every mutable state buffer after each step")
    if a.components and len(saved.get("components", [])) != 4:
        raise ValueError("Reference is missing component diagnostics")
    if a.tails and set(saved.get("tails", {})) != {str(n) for n in range(1, 33)}:
        raise ValueError("Reference is missing the complete logical tail sweep")
    if a.batch:
        batch = saved.get("batch", {})
        if any(len(batch.get(key, [])) != a.batch for key in ("inputs", "outputs")):
            raise ValueError("Reference batch size does not match the requested batch")
    if a.stack and tuple(saved.get("stack", torch.empty(0)).shape) != (1, 1, 1, 10240):
        raise ValueError("Reference is missing the stacked decoder output")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "mesh"), required=True)
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path(
            "/home/ttuser/qwen38-flash/hf/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/812d96e10a0dd82b69d749e68693698cf6c36daa"
        ),
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int, default=1)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert-dtype", default="bfloat4_b")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--repeats", type=int, default=16)
    parser.add_argument("--capacity", type=int, default=128)
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument(
        "--batch-changed-inputs", action="store_true", help="Alternate prepared batch inputs in one tracked trace"
    )
    parser.add_argument("--stack", action="store_true")
    parser.add_argument("--tails", action="store_true")
    parser.add_argument(
        "--profile", action="store_true", help="Signpost warmed eager operator windows for tt-perf-report"
    )
    parser.add_argument(
        "--reserve-bytes",
        type=int,
        default=0,
        help="Reserve a planned full-stack resident/state DRAM footprint before input allocation",
    )
    parser.add_argument("--clean-timing", action="store_true")
    parser.add_argument("--batch-fixture", type=Path)
    parser.add_argument("--moe-timing", action="store_true")
    parser.add_argument("--collective", choices=("default", "native-ar"), default="default")
    parser.add_argument("--reduction", choices=("default", "matmul", "matmul-b1"), default="default")
    parser.add_argument(
        "--components", action="store_true", help="Retain eager intermediate tensors for divergence diagnosis"
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("/home/ttuser/qwen38-flash/packed_ple_lloyd_lsq_group10_scale8_wide")
    )
    a = parser.parse_args()
    try:
        validate_options(a)
    except ValueError as error:
        parser.error(str(error))
    torch.set_num_threads(8)
    torch.manual_seed(2071 + a.layer)
    multi = a.mode == "mesh"
    if a.collective != "default":
        if not multi:
            parser.error("Collective candidates require mesh mode")

        def native_sum(self, value):
            return ttnn.experimental.all_reduce_async(
                ttnn.typecast(value, ttnn.float32),
                cluster_axis=1,
                mesh_device=mesh,
                barrier_semaphores=barriers,
                rs_global_semaphores=rs_sems,
                ag_global_semaphores=ag_sems,
                math_op=ttnn.ReduceType.Sum,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=ttnn.Topology.Ring,
                num_links=2,
            )

        Collectives.sum = native_sum
    if multi:
        saved = torch.load(a.reference, weights_only=False)
        validate_reference(a, saved)
    else:
        saved = dict(
            metadata=reference_metadata(a),
            inputs=[(torch.randn(1, 1, a.length, 10240) * 0.02).bfloat16()]
            + [(torch.randn(1, 1, 1, 10240) * 0.02).bfloat16() for _ in range(3)],
            outputs=[],
            states=[],
            components=[],
        )
    if not multi and a.batch_fixture:
        fixture = torch.load(a.batch_fixture, map_location="cpu", weights_only=False)
        sequence = fixture["sequence"]
        if sequence.shape[-2] < a.length + 3:
            parser.error("Prompt fixture lacks requested rows")
        saved["inputs"] = [sequence[:, :, : a.length, :]] + [
            sequence[:, :, a.length + i : a.length + i + 1, :] for i in range(3)
        ]
        saved["token_ids"] = [fixture["sequence_token_ids"][: a.length]] + [
            [fixture["sequence_token_ids"][a.length + i]] for i in range(3)
        ]
    cfg = AutoConfig.from_pretrained(a.snapshot, local_files_only=True).text_config
    if a.layer == 1:
        reference, weights = load_ple_reference(a.snapshot, cfg)
        del reference
    else:
        weights = load_weights(a.snapshot, a.layer, "")
    if multi:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4 if multi else 1), trace_region_size=0, l1_small_size=32768)
    if a.collective == "native-ar":
        # One persistent set shared by all serial CQ0 operations, including
        # stacked layers. Retain through trace release; kernels reset credits.
        grid = mesh.compute_with_storage_grid_size()
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})

        def make_semaphores(count):
            return [ttnn.create_global_semaphore(mesh, cores, 0, ttnn.BufferType.L1_SMALL) for _ in range(count)]

        barriers, rs_sems, ag_sems = make_semaphores(2), make_semaphores(3), make_semaphores(2)
    report = dict(mode=a.mode, layer=a.layer, length=a.length, passed=False, regime="serial", rows=[])
    report["collective"] = a.collective
    report["reduction"] = a.reduction
    if a.reduction != "default":
        from models.autoports.qwen_qwen3_8_flash_next.tests.multichip_reduction_candidate import make_matmul_reducer
        from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import MeshMoe

        matmul_reduce = make_matmul_reducer(mesh)
        native_reduce = MeshMoe._reduce

        def candidate_reduce(self, combined, weights, ids_rm, n):
            selected = native_reduce if a.reduction == "matmul-b1" and n > 1 else matmul_reduce
            return selected(self, combined, weights, ids_rm, n)

        MeshMoe._reduce = candidate_reduce
    audit = nullcontext if a.clean_timing else runtime_audit
    trace = None
    profiler_drains = []
    profiler_context = nullcontext()
    if a.profile:
        from models.autoports.qwen_qwen3_8_flash_next.tests.profile_fused_ops import bounded_profiler

        profiler_context = bounded_profiler(mesh, profiler_drains)
        report["profiler_scope"] = "Segmented per-operation timings; use separate trace runs for latency"
        report["profiler_drain_operation_ids"] = profiler_drains
    profiler_context.__enter__()
    try:
        embedding = None
        if a.layer == 1:
            embedding = (
                MeshPackedEmbedding.from_cache(a.cache_dir, mesh, Collectives(mesh))
                if multi
                else PackedEmbedding.from_checkpoint(a.snapshot, cache_dir=a.cache_dir, mesh_device=mesh)
            )
        factory = MultichipDecoder if multi else OptimizedDecoder
        model = factory.from_state_dict(
            weights,
            hf_config=cfg,
            layer_idx=a.layer,
            mesh_device=mesh,
            packed_embedding=embedding,
            **({"expert_dtype": getattr(ttnn, a.expert_dtype)} if multi else {}),
        )
        if a.profile:
            ttnn.ReadDeviceProfiler(mesh)
        intermediates = []
        native_reductions = []
        native_experts = []
        native_combines = []
        if a.components:
            for label, component in (("attention_input", model.attn_mix), ("moe_input", model.mlp_mix)):
                original = component.read

                def capture(value, original=original, label=label):
                    mixed, injection = original(value)
                    intermediates.append((label, mixed))
                    return mixed, injection

                component.read = capture

            class CaptureMoe:
                def __init__(self, component):
                    self.component = component

                def __call__(self, value):
                    result = self.component(value)
                    intermediates.append(("moe_output", result))
                    return result

            model.mlp = CaptureMoe(model.mlp)
            if multi:
                original_reduce = ttnn.experimental.deepseek_prefill.post_combine_reduce

                def capture_reduce(combined, weights, indices, table, **kwargs):
                    result = original_reduce(combined, weights, indices, table, **kwargs)
                    native_reductions.append((combined, weights, indices, result))
                    return result

                ttnn.experimental.deepseek_prefill.post_combine_reduce = capture_reduce
                if a.length <= 32:
                    original_experts = ttnn.experimental.deepseek_prefill.moe_fused_swiglu

                    def capture_experts(activations, gate, up, down, counts, ids, **kwargs):
                        result = original_experts(activations, gate, up, down, counts, ids, **kwargs)
                        native_experts.append((activations, counts, ids, kwargs["expert_region_offsets"], result))
                        return result

                    ttnn.experimental.deepseek_prefill.moe_fused_swiglu = capture_experts
                    original_combine = ttnn.experimental.deepseek_prefill.combine

                    def capture_combine(expert_output, metadata, counts, offsets, **kwargs):
                        result = original_combine(expert_output, metadata, counts, offsets, **kwargs)
                        native_combines.append((expert_output, metadata, counts, offsets, result))
                        return result

                    ttnn.experimental.deepseek_prefill.combine = capture_combine
        state = model.create_state(
            capacity=a.capacity, page_table=torch.arange(a.capacity // 32 - 1, -1, -1, dtype=torch.int32).reshape(1, -1)
        )

        def memory():
            view = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
            return dict(
                allocated_bytes=view.total_bytes_allocated_per_bank * view.num_banks,
                free_bytes=view.total_bytes_free_per_bank * view.num_banks,
                banks=view.num_banks,
            )

        report["memory_model_and_state"] = memory()
        reservation = []
        if a.reserve_bytes:
            if not multi:
                raise ValueError("Full-stack reservation belongs to the target mesh profile")
            current = memory()["allocated_bytes"]
            remaining = max(0, a.reserve_bytes - current)
            while remaining > 0:
                size = min(256 * 1024**2, ((remaining + 2047) // 2048) * 2048)
                reservation.append(
                    ttnn.empty(
                        (1, 1, 32, size // 64),
                        device=mesh,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                )
                remaining -= size
            report["reservation"] = dict(
                target_bytes=a.reserve_bytes,
                buffers=len(reservation),
                actual=memory(),
                claim="DRAM reservation, not an instantiated full model",
            )
        mapper = ttnn.ShardTensorToMesh(mesh, dim=3) if multi else ttnn.ReplicateTensorToMesh(mesh)
        composer = ttnn.ConcatMeshToTensor(mesh, dim=3) if multi else None

        def host_tensor(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, map_override=None):
            return ttnn.from_torch(
                host, dtype=dtype, layout=layout, mesh_mapper=mapper if map_override is None else map_override
            )

        inputs = [
            ttnn.from_torch(host, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
            for host in saved["inputs"]
        ]
        token_hosts = [
            token_rows(torch.arange(x.shape[-2]).reshape(1, -1) + 123 + i * 37) for i, x in enumerate(saved["inputs"])
        ]
        if "token_ids" in saved:
            token_hosts = [
                token_rows(torch.tensor(row, dtype=torch.int32).reshape(1, -1)) for row in saved["token_ids"]
            ]
        tokens = (
            [ttnn.from_torch(t, device=mesh, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT) for t in token_hosts]
            if a.layer == 1
            else [None] * len(inputs)
        )

        def call(i, token=None):
            with audit():
                if i == 0:
                    return model.prefill_forward(inputs[i], state=state, token_ids=tokens[i])
                return model.decode_forward(inputs[i] if token is None else token, state=state, token_ids=tokens[i])

        def read(t):
            return ttnn.to_torch(t, **({"mesh_composer": composer} if multi else {})).clone()

        owner = state.attention if state.attention is not None else state
        names = (
            ("position", "page_table", "k_cache", "v_cache", "raw_history", "compressed_keys")
            if state.attention is not None
            else ("conv", "recurrent") + (("ple_conv", "token_history") if a.layer == 1 else ())
        )
        buffers = {n: getattr(owner, n) for n in names}
        state_composer = ttnn.ConcatMeshToTensor(mesh, dim=0) if multi else None
        state_mapper = ttnn.ShardTensorToMesh(mesh, dim=0) if multi else ttnn.ReplicateTensorToMesh(mesh)

        def snapshot():
            return {
                n: ttnn.to_torch(t, **({"mesh_composer": state_composer} if multi else {})).clone()
                for n, t in buffers.items()
            }

        def prepare(raw):
            return {
                n: ttnn.from_torch(v, dtype=buffers[n].dtype, layout=buffers[n].layout, mesh_mapper=state_mapper)
                for n, v in raw.items()
            }

        def restore(prepared):
            for n, v in prepared.items():
                ttnn.copy_host_to_device_tensor(v, buffers[n])

        def mesh_state_reference(name, value):
            if name in ("k_cache", "v_cache"):
                return torch.cat([value[:, r // 2 : r // 2 + 1] for r in range(4)], dim=0)
            if name == "recurrent":
                return torch.cat(value.chunk(4, dim=1), dim=0)
            if name == "conv":
                q, k, v = value.split((2048, 2048, 6144), dim=-1)
                return torch.cat(
                    [
                        torch.cat(
                            (
                                q[..., r * 512 : (r + 1) * 512],
                                k[..., r * 512 : (r + 1) * 512],
                                v[..., r * 1536 : (r + 1) * 1536],
                            ),
                            dim=-1,
                        )
                        for r in range(4)
                    ],
                    dim=0,
                )
            if name == "ple_conv":
                return torch.cat(value.chunk(4, dim=-1), dim=0)
            return torch.cat([value] * 4, dim=0)

        initial = prepare(snapshot())
        prefix = None
        for i in range(len(inputs)):
            start = time.perf_counter()
            y = call(i)
            ttnn.synchronize_device(mesh)
            elapsed = (time.perf_counter() - start) * 1000
            out = read(y)
            metric = pcc(saved["outputs"][i], out) if multi else 1.0
            report["rows"].append(
                dict(
                    step=i,
                    pcc=metric,
                    finite=bool(torch.isfinite(out).all()),
                    max_absolute=float(out.abs().max()),
                    cold_eager_ms=elapsed,
                )
            )
            print(report["rows"][-1], flush=True)
            if a.components:
                host_components = []
                for label, value in intermediates:
                    host = ttnn.to_torch(value, **({"mesh_composer": state_composer} if multi else {}))
                    if multi:
                        ranks = host.chunk(4, dim=0)
                        expected_label, expected_value = saved["components"][i][len(host_components)]
                        assert label == expected_label
                        metric = dict(
                            label=label,
                            rank_pcc=[pcc(expected_value, rank) for rank in ranks],
                            rank_max_absolute=[float(rank.abs().max()) for rank in ranks],
                        )
                        report["rows"][-1].setdefault("components", []).append(metric)
                    host_components.append((label, host.clone()))
                if not multi:
                    saved["components"].append(host_components)
                intermediates.clear()
                for combined, weights, indices, reduced in native_reductions:
                    hosts = [
                        ttnn.to_torch(value, mesh_composer=state_composer).chunk(4, dim=0)
                        for value in (combined, weights, indices, reduced)
                    ]
                    metrics = []
                    for rank in range(4):
                        values, scores, ids, observed = [parts[rank] for parts in hosts]
                        ids = ids.long()
                        owned = (ids >= rank * 128) & (ids < (rank + 1) * 128)
                        selected = torch.where(owned[..., None], values.float(), 0.0)
                        expected_reduce = (selected * scores.float()[..., None]).sum(dim=-2)
                        metrics.append(
                            dict(
                                rank=rank,
                                pcc=pcc(expected_reduce, observed),
                                expected_max=float(expected_reduce.abs().max()),
                                actual_max=float(observed.abs().max()),
                                selected_finite=bool(torch.isfinite(selected).all()),
                            )
                        )
                    report["rows"][-1].setdefault("native_reduction", []).append(metrics)
                native_reductions.clear()
                for tensors in native_experts:
                    hosts = [ttnn.to_torch(v, mesh_composer=state_composer).chunk(4, dim=0) for v in tensors]
                    metrics = []
                    for rank in range(4):
                        activations, counts, ids, offsets, observed = [parts[rank] for parts in hosts]
                        for expert in ids.flatten().long().tolist():
                            count, offset = int(counts.flatten()[expert]), int(offsets.flatten()[expert])
                            if count:
                                metrics.append(
                                    dict(
                                        rank=rank,
                                        expert=expert,
                                        count=count,
                                        offset=offset,
                                        input_max=float(activations[offset : offset + count].abs().max()),
                                        output_max=float(observed[offset : offset + count].abs().max()),
                                    )
                                )
                    report["rows"][-1].setdefault("native_experts", []).append(metrics)
                native_experts.clear()
                for tensors in native_combines:
                    hosts = [ttnn.to_torch(v, mesh_composer=state_composer).chunk(4, dim=0) for v in tensors]
                    metrics = []
                    for rank in range(4):
                        down, metadata, counts, offsets, combined = [parts[rank] for parts in hosts]
                        down = down.reshape(-1, 2560)
                        metadata = metadata.reshape(-1, metadata.shape[-1]).long()
                        combined = combined.reshape(-1, 10, 2560)
                        for expert in range(rank * 128, (rank + 1) * 128):
                            count, offset = int(counts.flatten()[expert]), int(offsets.flatten()[expert])
                            for ordinal in range(count):
                                destination, token_index, slot = metadata[offset + ordinal].tolist()
                                valid = 0 <= token_index < combined.shape[0] and 0 <= slot < 10
                                actual = combined[token_index, slot] if valid else torch.empty(0)
                                expected = down[offset + ordinal]
                                metrics.append(
                                    dict(
                                        rank=rank,
                                        expert=expert,
                                        ordinal=ordinal,
                                        metadata=[destination, token_index, slot],
                                        exact=torch.equal(actual, expected),
                                        expected_max=float(expected.abs().max()),
                                        actual_max=float(actual.abs().max()) if valid else None,
                                    )
                                )
                    report["rows"][-1].setdefault("native_combine", []).append(metrics)
                native_combines.clear()
            if not multi:
                saved["outputs"].append(out)
                saved["states"].append(snapshot())
            elif "states" in saved:
                observed = snapshot()
                state_metrics = {}
                for name, value in saved["states"][i].items():
                    expected_state = mesh_state_reference(name, value)
                    actual_state = observed[name]
                    exact = torch.equal(expected_state, actual_state)
                    state_metrics[name] = dict(
                        exact=exact,
                        pcc=1.0 if exact else pcc(expected_state, actual_state),
                        finite=bool(torch.isfinite(actual_state).all()),
                    )
                report["rows"][-1]["state_parity"] = state_metrics
            if i == 0:
                prefix = prepare(snapshot())
            del y
            if a.profile:
                ttnn.ReadDeviceProfiler(mesh)
        if a.trace:
            if not a.clean_timing:
                assert trace_allocation_tracker.TRACE_ALLOC_TRACKING
                assert os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") == "0"
            expected = []
            prepared_inputs = [host_tensor(x) for x in saved["inputs"][1:]]
            token = inputs[1]
            for i in range(1, len(inputs)):
                restore(prefix)
                ttnn.copy_host_to_device_tensor(prepared_inputs[i - 1], token)
                y = call(i, token)
                expected.append((read(y), snapshot()))
                del y
            prefill_times = []
            for _ in range(5):
                restore(initial)
                ttnn.synchronize_device(mesh)
                start = time.perf_counter()
                y = call(0)
                ttnn.synchronize_device(mesh)
                prefill_times.append((time.perf_counter() - start) * 1000)
                del y
            restore(prefix)
            ttnn.copy_host_to_device_tensor(prepared_inputs[0], token)
            y = call(1, token)
            del y
            gc.collect()
            restore(prefix)
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            y = call(1, token)
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            if trace_allocation_tracker.TRACE_ALLOC_TRACKING:
                # The only retained capture output is overwritten on every replay,
                # read before the next execution, and released with this one trace.
                trace_allocation_tracker.acknowledge_corruptible(y)
            samples = []
            for j in range(a.repeats):
                i = j % len(expected)
                restore(prefix)
                ttnn.copy_host_to_device_tensor(prepared_inputs[i], token)
                # PLE hash token words are also a persistent trace input.
                if a.layer == 1:
                    ttnn.copy_host_to_device_tensor(
                        ttnn.from_torch(token_hosts[i + 1], dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT), tokens[1]
                    )
                ttnn.synchronize_device(mesh)
                start = time.perf_counter()
                with audit():
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                ms = (time.perf_counter() - start) * 1000
                actual = read(y)
                observed = snapshot()
                samples.append(
                    dict(
                        ms=ms,
                        input=i,
                        exact_output=torch.equal(actual, expected[i][0]),
                        exact_state=all(torch.equal(observed[n], expected[i][1][n]) for n in names),
                    )
                )
            report.update(
                trace_id=str(trace),
                captures=1,
                replays=a.repeats,
                trace_samples=samples,
                prefill_ms=float(torch.tensor(prefill_times).median()),
                decode_ms=float(torch.tensor([r["ms"] for r in samples[2:]]).median()),
                trace_tracking=bool(trace_allocation_tracker.TRACE_ALLOC_TRACKING),
                runtime_audit=not a.clean_timing,
            )
            ttnn.release_trace(mesh, trace)
            trace = None
            del y
        if a.moe_timing:
            from models.autoports.qwen_qwen3_8_flash_next.tests.multichip_moe_timing import measure

            restore(prefix)
            report["moe"] = measure(model, lambda: call(1), mesh)
            report["moe"]["share_percent"] = 100 * report["moe"]["traced_ms"] / report["decode_ms"]
        if a.profile:
            from tracy import signpost

            for _ in range(3):
                restore(initial)
                warmed = call(0)
                ttnn.synchronize_device(mesh)
                del warmed
                ttnn.ReadDeviceProfiler(mesh)
            restore(initial)
            ttnn.synchronize_device(mesh)
            signpost("PERF_PREFILL")
            profiled = call(0)
            ttnn.synchronize_device(mesh)
            signpost("PERF_PREFILL_END")
            del profiled
            ttnn.ReadDeviceProfiler(mesh)
            restore(prefix)
            ttnn.synchronize_device(mesh)
            signpost("PERF_DECODE")
            profiled = call(1)
            ttnn.synchronize_device(mesh)
            signpost("PERF_DECODE_END")
            ttnn.ReadDeviceProfiler(mesh)
            report["profile"] = dict(
                regime="warmed eager operator attribution; separate trace latency evidence",
                finite=bool(torch.isfinite(read(profiled)).all()),
            )
            del profiled
        if a.tails:
            if saved["inputs"][0].shape[-2] < 32:
                raise ValueError("Tail sweep requires at least32 reference input rows")
            if not multi:
                saved["tails"] = {}
            report["tails"] = []
            for n in range(1, 33):
                restore(initial)
                tail_input = ttnn.from_torch(
                    saved["inputs"][0][:, :, :n, :].contiguous(),
                    device=mesh,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=mapper,
                )
                tail_tokens = (
                    ttnn.from_torch(
                        token_hosts[0][:, :n, :].contiguous(),
                        device=mesh,
                        dtype=ttnn.uint32,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                    )
                    if a.layer == 1
                    else None
                )
                with audit():
                    tail_output = model.prefill_forward(tail_input, state=state, token_ids=tail_tokens)
                actual = read(tail_output)
                if not multi:
                    saved["tails"][str(n)] = actual
                report["tails"].append(
                    dict(
                        length=n,
                        pcc=pcc(saved["tails"][str(n)], actual),
                        finite=bool(torch.isfinite(actual).all()),
                        shape=list(actual.shape),
                    )
                )
                del tail_output, tail_input, tail_tokens
        if a.batch:
            if not 1 <= a.batch <= 32:
                raise ValueError("Batch must be1..32")
            if not multi:
                if a.batch_fixture:
                    fixture = torch.load(a.batch_fixture, map_location="cpu", weights_only=False)
                    saved["batch"] = {
                        "inputs": fixture["inputs"][: a.batch],
                        "token_ids": fixture["token_ids"][: a.batch],
                        "provenance": fixture["provenance"],
                    }
                else:
                    saved["batch"] = {
                        "inputs": [(torch.randn(1, 1, 1, 10240) * 0.02).bfloat16() for _ in range(a.batch)]
                    }
            batch_inputs = tuple(
                ttnn.from_torch(v, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
                for v in saved["batch"]["inputs"]
            )
            batch_states = tuple(model.create_state(capacity=a.capacity) for _ in batch_inputs)
            batch_tokens = tuple(tokens[1] for _ in batch_inputs) if a.layer == 1 else None
            if a.layer == 1 and "token_ids" in saved["batch"]:
                batch_tokens = tuple(
                    ttnn.from_torch(
                        torch.full((1, 1, 16), token, dtype=torch.int32),
                        device=mesh,
                        dtype=ttnn.uint32,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                    )
                    for token in saved["batch"]["token_ids"]
                )
            batch_input_variants = []
            batch_token_variants = []
            batch_changed_hidden = False
            batch_changed_tokens = False
            if a.batch_changed_inputs:
                # Prepare both host payloads before capture. Device input objects
                # remain fixed; replay only copies into those existing buffers.
                raw_variants = [saved["batch"]["inputs"], list(reversed(saved["batch"]["inputs"]))]
                batch_changed_hidden = any(not torch.equal(x, y) for x, y in zip(*raw_variants))
                if not batch_changed_hidden:
                    raise ValueError("Changed-input batch validation requires diverse hidden input rows")
                batch_input_variants = [tuple(host_tensor(value) for value in variant) for variant in raw_variants]
                if a.layer == 1:
                    raw_tokens = (
                        [torch.full((1, 1, 16), token, dtype=torch.int32) for token in saved["batch"]["token_ids"]]
                        if "token_ids" in saved["batch"]
                        else [token_hosts[1].clone() for _ in batch_inputs]
                    )
                    changed_tokens = [value.clone() for value in reversed(raw_tokens)]
                    if all(torch.equal(x, y) for x, y in zip(raw_tokens, changed_tokens)):
                        # Synthetic contract references use one repeated token.
                        # Change its ID while preserving the public word encoding.
                        for value in changed_tokens:
                            value[..., 0] = (value[..., 0] + 37) % cfg.vocab_size
                    batch_changed_tokens = any(not torch.equal(x, y) for x, y in zip(raw_tokens, changed_tokens))
                    batch_token_variants = [
                        tuple(
                            ttnn.from_torch(value, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT) for value in variant
                        )
                        for variant in (raw_tokens, changed_tokens)
                    ]
                    # Legacy synthetic batches alias tokens[1]. Distinct buffers
                    # are necessary when independently refreshing each request.
                    batch_tokens = tuple(
                        ttnn.from_torch(value, device=mesh, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
                        for value in raw_tokens
                    )

            def batch_refresh(selector):
                if not a.batch_changed_inputs:
                    return
                for source, target in zip(batch_input_variants[selector], batch_inputs):
                    ttnn.copy_host_to_device_tensor(source, target)
                if a.layer == 1:
                    for source, target in zip(batch_token_variants[selector], batch_tokens):
                        ttnn.copy_host_to_device_tensor(source, target)

            batch_buffers = {}
            for index, target in enumerate(batch_states):
                target_owner = target.attention if target.attention is not None else target
                for name in names:
                    batch_buffers[index, name] = getattr(target_owner, name)

            def batch_snapshot():
                return {
                    key: ttnn.to_torch(v, **({"mesh_composer": state_composer} if multi else {})).clone()
                    for key, v in batch_buffers.items()
                }

            batch_initial = {
                key: ttnn.from_torch(
                    value, dtype=batch_buffers[key].dtype, layout=batch_buffers[key].layout, mesh_mapper=state_mapper
                )
                for key, value in batch_snapshot().items()
            }

            def batch_restore():
                for key, value in batch_initial.items():
                    ttnn.copy_host_to_device_tensor(value, batch_buffers[key])

            def batch_call():
                with audit():
                    # Exercise the baseline's list contract as well as grouped MLP execution.
                    return model.decode_forward(list(batch_inputs), state=batch_states, token_ids=batch_tokens)

            batch_outputs = batch_call()
            own_expected = [read(v) for v in batch_outputs]
            own_state = batch_snapshot()
            if not multi:
                saved["batch"]["outputs"] = own_expected
            batch_pcc = [pcc(expected, actual) for expected, actual in zip(saved["batch"]["outputs"], own_expected)]
            del batch_outputs
            batch_expected_variants = [(own_expected, own_state)]
            if a.batch_changed_inputs:
                batch_restore()
                batch_refresh(1)
                batch_outputs = batch_call()
                batch_expected_variants.append(([read(value) for value in batch_outputs], batch_snapshot()))
                del batch_outputs
                batch_refresh(0)
            for _ in range(2):
                batch_restore()
                batch_outputs = batch_call()
                ttnn.synchronize_device(mesh)
                del batch_outputs
            batch_restore()
            gc.collect()
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            batch_outputs = batch_call()
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            if trace_allocation_tracker.TRACE_ALLOC_TRACKING:
                for value in batch_outputs:
                    trace_allocation_tracker.acknowledge_corruptible(value)
            batch_samples = []
            for replay in range(a.repeats):
                selector = replay % len(batch_expected_variants)
                batch_restore()
                batch_refresh(selector)
                ttnn.synchronize_device(mesh)
                start = time.perf_counter()
                with audit():
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                elapsed = (time.perf_counter() - start) * 1000
                actual = [read(v) for v in batch_outputs]
                observed = batch_snapshot()
                expected_output, expected_state = batch_expected_variants[selector]
                batch_samples.append(
                    dict(
                        ms=elapsed,
                        input_selector=selector,
                        exact_output=all(torch.equal(x, y) for x, y in zip(actual, expected_output)),
                        exact_state=all(torch.equal(value, expected_state[key]) for key, value in observed.items()),
                    )
                )
            ttnn.release_trace(mesh, trace)
            trace = None
            report["batch"] = dict(
                size=a.batch,
                pcc=batch_pcc,
                whole_output_pcc=pcc(torch.cat(saved["batch"]["outputs"], dim=2), torch.cat(own_expected, dim=2)),
                samples=batch_samples,
                decode_ms=float(torch.tensor([s["ms"] for s in batch_samples[2:]]).median()),
                trace_tracking=bool(trace_allocation_tracker.TRACE_ALLOC_TRACKING),
            )
            if a.batch_changed_inputs:
                report["batch"]["changed_inputs"] = dict(
                    variants=len(batch_expected_variants),
                    hidden_inputs_changed=batch_changed_hidden,
                    token_inputs_changed=batch_changed_tokens if a.layer == 1 else None,
                    hidden_permutation=list(reversed(range(a.batch))),
                    preparation="Both host input/token sets and exact eager output/state references precede capture",
                    replay="Copy prepared host payloads into persistent device inputs; restore identical initial states",
                    timing_scope="Tracked correctness stress with alternating inputs; use clean-timing plans for latency",
                )
            if a.moe_timing:
                if not multi:
                    raise ValueError("MoE attribution is defined for the mesh path")
                from models.autoports.qwen_qwen3_8_flash_next.tests.multichip_moe_timing import measure

                batch_restore()
                batch_refresh(0)
                report["batch"]["moe"] = measure(model, batch_call, mesh)
                report["batch"]["moe"]["share_percent"] = (
                    100 * report["batch"]["moe"]["traced_ms"] / report["batch"]["decode_ms"]
                )
            if "provenance" in saved["batch"]:
                report["batch"]["input_provenance"] = saved["batch"]["provenance"]
            del batch_outputs, batch_states, batch_buffers, batch_initial
        if a.stack:
            other_layer = 3 if a.layer == 0 else 0
            first_state = model.create_state(capacity=a.capacity)
            with audit():
                first_output = model.decode_forward(inputs[1], state=first_state, token_ids=tokens[1])
            shared_workspace = model.workspace if multi else None
            if not multi:
                # The single-chip PLE reference nearly fills DRAM. Preserve the
                # boundary tensor while releasing first-layer weights before
                # constructing the second reference layer.
                ttnn.synchronize_device(mesh)
                del model
                gc.collect()
            other = factory.from_state_dict(
                load_weights(a.snapshot, other_layer, ""),
                hf_config=cfg,
                layer_idx=other_layer,
                mesh_device=mesh,
                **({"expert_dtype": getattr(ttnn, a.expert_dtype), "workspace": shared_workspace} if multi else {}),
            )
            second_state = other.create_state(capacity=a.capacity)
            with audit():
                second_output = other.decode_forward(first_output, state=second_state)
            stacked = read(second_output)
            if not multi:
                saved["stack"] = stacked
            report["stack"] = dict(
                layers=[a.layer, other_layer],
                pcc=pcc(saved["stack"], stacked),
                finite=bool(torch.isfinite(stacked).all()),
                boundary_conversion=False,
            )
            del other, first_output, second_output
        report["passed"] = all(r["pcc"] >= 0.995 and r["finite"] for r in report["rows"]) and all(
            r["exact_output"] and r["exact_state"] for r in report.get("trace_samples", [])
        )
        report["passed"] &= all(
            metric["finite"]
            and (metric["exact"] if name in ("position", "page_table", "token_history") else metric["pcc"] >= 0.995)
            for row in report["rows"]
            for name, metric in row.get("state_parity", {}).items()
        )
        if a.batch:
            # The stage contract applies .995 to the whole output. Per-row
            # correlations remain diagnostics and are retained verbatim above.
            report["passed"] &= report["batch"]["whole_output_pcc"] >= 0.995 and all(
                s["exact_output"] and s["exact_state"] for s in report["batch"]["samples"]
            )
        if a.stack:
            report["passed"] &= report["stack"]["pcc"] >= 0.995 and report["stack"]["finite"]
        if a.tails:
            report["passed"] &= all(r["pcc"] >= 0.995 and r["finite"] for r in report["tails"])
        if not multi and report["passed"]:
            torch.save(saved, a.reference)
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        profiler_context.__exit__(None, None, None)
        a.output.write_text(json.dumps(report, indent=2) + "\n")
        ttnn.close_mesh_device(mesh)
    assert report["passed"], report


if __name__ == "__main__":
    main()
