# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated Gemma CP8/TP4 SDPA candidate target.

Use ``python -m models.demos.gemma4_d_p.tests.sweep_sdpa_perf`` to inspect the
search, or add ``--run-device`` only after reserving an idle Galaxy.
"""

import json
import os
import platform
import subprocess
import uuid
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.gemma4_d_p.config import MeshConfig
from models.demos.gemma4_d_p.tests.sdpa_perf_utils import (
    CAPACITY,
    CHUNK,
    CP,
    LOCAL_Q,
    PCC_THRESHOLD,
    PREFIXES,
    RMSE_THRESHOLD,
    TP,
    Candidate,
    accuracy_metrics,
    cache_to_chronological,
    cache_to_rank_major,
    pack_queries,
    reference_attention,
    sdpa_duration_ns,
    tilings,
    timing_summary,
)
from models.demos.gemma4_d_p.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gemma4_d_p.tt.attention.ring_prefill import migration_ring_memory_config, ring_prefill_gather_seq
from models.demos.gemma4_d_p.tt.ccl import CCLManager
from models.demos.gemma4_d_p.tt.prefill_metadata import PrefillMetadata
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program, require_realtime_profiler


def _cases():
    selected = os.environ.get("GEMMA4_SDPA_CANDIDATE")
    layers = [os.environ["GEMMA4_SDPA_LAYER"]] if "GEMMA4_SDPA_LAYER" in os.environ else ("global", "swa")
    prefixes = [int(os.environ["GEMMA4_SDPA_ISL"])] if "GEMMA4_SDPA_ISL" in os.environ else PREFIXES
    for layer in layers:
        for prefix in prefixes:
            for candidate in [Candidate(**json.loads(selected))] if selected else tilings(layer):
                yield pytest.param(layer, prefix, candidate, id=f"{layer}-isl{prefix}-{candidate.id}")


def _upload(mesh, host, dtype, memory_config):
    return ttnn.from_torch(
        host,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=memory_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(CP, TP), dims=(2, 1)),
    )


def _read(mesh, tensor):
    return ttnn.to_torch(
        tensor,
        mesh_composer=ttnn.create_mesh_composer(mesh, ttnn.MeshComposerConfig(2, 1)),
    )


def _cache_placement(tensor):
    # Production caches advertise replicated placement; writers populate distinct CP/TP contents.
    shape = ttnn.MeshShape(CP, TP)
    coords = [ttnn.MeshCoordinate([row, col]) for row in range(CP) for col in range(TP)]
    tensor.update_tensor_topology(
        ttnn.TensorTopology(shape, [ttnn.PlacementReplicate(), ttnn.PlacementReplicate()], coords)
    )
    return tensor


def _prepare_cache(mesh, layer, seed):
    """Populate all prefix slabs directly; packed-cache slicing is outside measurement."""
    generator = torch.Generator().manual_seed(seed)
    dim, heads = (512, 1) if layer == "global" else (256, 4)
    if layer == "global":
        # [Krot128 | Vordered512], with overlapping K=[0:512], V=[128:640].
        # Four global KV heads are split across TP, leaving one distinct head per device.
        packed = torch.randn((1, TP, CAPACITY, 640), generator=generator, dtype=torch.bfloat16)
        packed = cache_to_rank_major(packed)
        tt_packed = _upload(mesh, packed, ttnn.bfloat8_b, migration_ring_memory_config(mesh, 640))
        _cache_placement(tt_packed)
        del packed
        shape = tuple(tt_packed.shape)
        k = ttnn.slice(tt_packed, (0, 0, 0, 0), shape[:-1] + (512,), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        v = ttnn.slice(tt_packed, (0, 0, 0, 128), shape[:-1] + (640,), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tt_packed.deallocate(True)
    else:

        def make_cache():
            host = torch.randn((1, heads * TP, CAPACITY, dim), generator=generator, dtype=torch.bfloat16)
            return _cache_placement(
                _upload(mesh, cache_to_rank_major(host), ttnn.bfloat8_b, migration_ring_memory_config(mesh, dim))
            )

        k, v = make_cache(), make_cache()
    assert tuple(k.shape) == (1, heads, CAPACITY // CP, dim)
    assert tuple(v.shape) == tuple(k.shape)
    # Reference the actual BF8 operands, so this checks SDPA arithmetic rather than cache quantization.
    host_k = cache_to_chronological(_read(mesh, k))
    host_v = cache_to_chronological(_read(mesh, v))
    return k, v, host_k, host_v


def _check_output(mesh, output, host_q, host_k, host_v, positions, window, row):
    got = _read(mesh, output)
    kv_heads = host_k.shape[1] // TP
    per_device = []
    for rank in range(CP):
        count = len(positions[rank])
        if not count:
            continue
        seq = slice(rank * LOCAL_Q, rank * LOCAL_Q + count)
        for col in range(TP):
            heads, kv = slice(col * 8, (col + 1) * 8), slice(col * kv_heads, (col + 1) * kv_heads)
            expected = reference_attention(host_q[:, heads, seq], host_k[:, kv], host_v[:, kv], positions[rank], window)
            pcc, rmse = accuracy_metrics(expected, got[:, heads, seq])
            per_device.append({"cp": rank, "tp": col, "pcc": pcc, "rmse": rmse, "valid_rows": count})
    row["pcc"] = min(row.get("pcc", 1.0), min(item["pcc"] for item in per_device))
    row["rmse"] = max(row.get("rmse", 0.0), max(item["rmse"] for item in per_device))
    row.setdefault("accuracy", []).append(per_device)
    if row["pcc"] < PCC_THRESHOLD:
        worst = min(per_device, key=lambda item: item["pcc"])
        print(
            f"PCC FAILED: layer={row['layer']} ISL={row['isl']} "
            f"CP={worst['cp']} TP={worst['tp']} PCC={worst['pcc']:.10f} "
            f"required>={PCC_THRESHOLD} RMSE={worst['rmse']:.10f}",
            flush=True,
        )
    assert row["pcc"] >= PCC_THRESHOLD, f"PCC {row['pcc']} < {PCC_THRESHOLD}"
    assert row["rmse"] < RMSE_THRESHOLD, f"RMSE {row['rmse']} >= {RMSE_THRESHOLD}"


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Local performance search")
@parametrize_mesh_with_fabric(mesh_shapes=[(CP, TP)])
@pytest.mark.parametrize("layer, prefix, candidate", list(_cases()))
def test_sdpa_perf(mesh_device, layer, prefix, candidate):
    row = {
        "layer": layer,
        "isl": prefix,
        "config": asdict(candidate),
        "status": "error",
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "hardware": f"{platform.node()}:Blackhole:8x4:CP8:TP4:Linear:links2",
        "seed": int(os.environ.get("GEMMA4_SDPA_SEED", "1234")),
    }
    result_path = Path(os.environ.get("GEMMA4_SDPA_RESULT", f"generated/gemma4_sdpa/{uuid.uuid4().hex}.json"))
    try:
        row["stage"] = "setup"
        assert ttnn.get_arch_name() == "blackhole", "Requires Blackhole Galaxy"
        assert tuple(mesh_device.shape) == (CP, TP)
        require_realtime_profiler("Gemma SDPA parameter search")
        assert not any(
            name in os.environ for name in ("TT_METAL_WATCHER", "TT_METAL_LLK_ASSERTS", "TT_METAL_LLK_SANITIZER")
        ), "Unset Watcher and LLK instrumentation before timing"
        assert prefix in PREFIXES
        repeats = int(os.environ.get("GEMMA4_SDPA_REPEATS", "20"))
        assert repeats >= 20
        torch.set_num_threads(8)
        mesh_device.enable_program_cache()
        mesh_config = MeshConfig(mesh_device)
        manager = CCLManager(mesh_config, num_links=2, topology=ttnn.Topology.Linear)
        metadata = PrefillMetadata(mesh_config)
        dim, kv_heads = (512, 1) if layer == "global" else (256, 4)
        window = None if layer == "global" else 1024
        grid = manager.compute_grid_size
        program = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(grid.x - 1, grid.y),
            q_chunk_size=candidate.q,
            k_chunk_size=candidate.k,
            exp_approx_mode=candidate.exp_approx,
        )
        compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, candidate.fidelity),
            math_approx_mode=candidate.math_approx,
            fp32_dest_acc_en=candidate.fp32_dest,
            packer_l1_acc=candidate.packer_l1,
        )
        k, v, host_k, host_v = _prepare_cache(mesh_device, layer, row["seed"])
        gather_seq = ring_prefill_gather_seq(CAPACITY, CP, window, candidate.k)
        buffers = [
            manager.get_ring_gather_buffer(key, kv_heads, gather_seq, dim, ttnn.bfloat8_b, tensor.memory_config())
            for key, tensor in (("ring_k", k), ("ring_v", v))
        ]

        def prepare(prefix_len, length):
            generator = torch.Generator().manual_seed(row["seed"] + 1)
            queries = torch.randn((1, 8 * TP, length, dim), generator=generator, dtype=torch.bfloat16)
            host_q, positions = pack_queries(queries, prefix_len)
            q = _upload(mesh_device, host_q, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG)
            metadata.update(slot_idx=0, kv_actual_global=prefix_len)
            return q, host_q, positions

        def reset():
            for semaphore in manager.ring_attention_ccl_semaphore_handles:
                ttnn.reset_global_semaphore_value(semaphore, 0)
            ttnn.synchronize_device(mesh_device)

        def invoke(q, start, length, use_metadata=True):
            indexing = (
                dict(
                    slot_id=metadata.slot_idx,
                    kv_actual_isl_tensor=metadata.kv_actual_global,
                    kv_cache_num_layers=1,
                    kv_cache_layer_idx=0,
                )
                if use_metadata
                else dict(kv_cache_batch_idx=0, kv_actual_isl=start)
            )
            out, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
                q,
                k,
                v,
                None,
                None,
                None,
                persistent_output_buffer_k=buffers[0],
                persistent_output_buffer_v=buffers[1],
                joint_strategy="rear",
                logical_n=CAPACITY if use_metadata else start + length,
                scale=1.0,
                program_config=program,
                compute_kernel_config=compute,
                dim=2,
                multi_device_global_semaphore=manager.ring_attention_ccl_semaphore_handles,
                num_links=manager.num_links,
                cluster_axis=mesh_config.cp_axis,
                mesh_device=mesh_device,
                topology=ttnn.Topology.Linear,
                ccl_core_grid_offset=ttnn.CoreCoord(*manager.ring_attention_ccl_core_grid_offset),
                use_column_major_ccl=True,
                is_causal=True,
                is_balanced=False,
                sliding_window_size=window,
                **indexing,
            )
            return out

        q, host_q, positions = prepare(prefix, CHUNK)
        reset()
        row["stage"] = "op"
        out = invoke(q, prefix, CHUNK)
        row["stage"] = "accuracy"
        _check_output(mesh_device, out, host_q, host_k, host_v, positions, window, row)
        out.deallocate(True)

        # The scalar path expresses partial lengths; current tensor metadata always means a full chunk.
        # SWA rejects these geometries in op validation, so do not claim service-rotation coverage.
        if layer == "global":
            for start, length in ((max(32, prefix - 32), CHUNK), (prefix, CHUNK - 32)):
                edge_q, edge_host, edge_positions = prepare(start, length)
                reset()
                row["stage"] = "op"
                edge_out = invoke(edge_q, start, length, use_metadata=length == CHUNK)
                row["stage"] = "accuracy"
                _check_output(mesh_device, edge_out, edge_host, host_k, host_v, edge_positions, window, row)
                edge_out.deallocate(True)
                edge_q.deallocate(True)
            row["edge_status"] = "passed_metadata_rotated_and_scalar_partial"
        else:
            row["edge_status"] = "unsupported: checkout requires complete aligned SWA ring groups"

        metadata.update(slot_idx=0, kv_actual_global=prefix)
        row["stage"] = "timing"
        for _ in range(2):
            reset()
            out = invoke(q, prefix, CHUNK)
            ttnn.synchronize_device(mesh_device)
            out.deallocate(True)
        durations = []
        chips = set(mesh_device.get_device_ids())
        assert len(chips) == CP * TP
        for _ in range(repeats):
            reset()
            out, records = profile_realtime_program(
                mesh_device, lambda: invoke(q, prefix, CHUNK), collect_all=True, record_timeout_seconds=10.0
            )
            durations.append(sdpa_duration_ns(records, chips))
            out.deallocate(True)
        row["median_us"], row["p90_us"] = timing_summary(durations)
        row["durations_ns"] = durations
        row["status"] = "passed"
    except (Exception, pytest.fail.Exception) as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        message = str(exc).lower()
        if (
            row["stage"] == "op"
            and "l1" in message
            and any(word in message for word in ("overflow", "out of", "exceed", "does not fit", "not enough"))
        ):
            row["status"] = "l1_overflow"
        elif row["stage"] == "op" and any(
            word in message for word in ("unsupported", "not supported", "requires", "must be")
        ):
            row["status"] = "unsupported"
        elif isinstance(exc, AssertionError) and ("PCC" in str(exc) or "RMSE" in str(exc)):
            row["status"] = "accuracy_failed"
        raise
    finally:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(row, indent=2) + "\n")
        print(json.dumps({key: value for key, value in row.items() if key not in ("accuracy", "durations_ns")}))
