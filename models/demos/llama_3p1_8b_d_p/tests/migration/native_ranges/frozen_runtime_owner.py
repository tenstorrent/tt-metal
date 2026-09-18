#!/usr/bin/env python3
"""CLOSED preparation: one TTNN owner, five full32 H2D calls, no native manager."""
import argparse
import gc
import hashlib
import json
import os
import struct
import subprocess
import sys
import time
import traceback
from pathlib import Path

from edge_checks import drive_requests, seed_value
from edge_coverage import RUNTIME_CALLS, prepare_h2d_input, require_distinct_prompts
from edge_guard import preflight
from edge_snapshots import capture, load_decoder
from gate_contract import AckRecorder
from owned_cleanup import cleanup_each, cleanup_owner
from page_io import geometry
from support import CapturedCompletionSink, mapped_libraries, prepare_jit_cache, require, sha256, write_json


def seed_cache(ttnn, torch, mesh, cache, writer, *, cleanup_errors):
    calls = 0
    for slot in (0, 1):
        for layer in range(32):
            for begin in (0, 1024):
                tensors = []
                primary = None
                try:
                    for config in (0, 8):
                        host = torch.full((1, 1, 256, 128), seed_value(config, slot, layer), dtype=torch.bfloat16)
                        tensors.append(
                            ttnn.from_torch(
                                host,
                                device=mesh,
                                dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                            )
                        )
                    writer(cache, *tensors, slot_idx=slot, layer_idx=layer, actual_start=begin, actual_end=begin + 1024)
                    calls += 1
                except BaseException as exc:
                    primary = exc
                    raise
                finally:
                    before = len(cleanup_errors)
                    cleanup_each(
                        (
                            (f"seed input{index}", lambda tensor=tensor: tensor.deallocate(True))
                            for index, tensor in reversed(list(enumerate(tensors)))
                        ),
                        cleanup_errors,
                        [],
                    )
                    if len(cleanup_errors) > before and primary is None:
                        raise RuntimeError("seed input cleanup failed")
    ttnn.synchronize_device(mesh)
    return calls


def main(plan_path, plan_hash):
    plan, spec, resources = preflight(plan_path, plan_hash)
    output = Path(plan["output"])
    output.mkdir(exist_ok=False)
    report = dict(
        scope=plan["scope"],
        run_nonce=plan["run_nonce"],
        gate_passed=False,
        requests=[],
        acks=[],
        snapshots=[],
        errors=[],
        cleanup_errors=[],
        resources=resources,
        owner_cleanup_complete=False,
        model_executed=False,
        persistent_h2d_tested=False,
        packed_capture_performed=False,
        native_manager_tested=False,
        native_source_pin_or_retirement_tested=False,
        native_transfer_tested=False,
        kv_golden_comparison_performed=False,
        new_model_numerical_acceptance=False,
        decoder_tested=False,
    )
    recorder = AckRecorder()
    mesh = runtime = cache = service = router = producer = channel = None
    saved = {"value": None}
    sink = None
    try:
        report["jit_cache"] = prepare_jit_cache(output, os.environ)
        write_json(output / "jit-cache-before.json", report["jit_cache"])
        with (output / "native-environment.log").open("xb") as log:
            native = subprocess.run(
                [sys.executable, "-B", plan["native_probe"]], stdout=log, stderr=subprocess.STDOUT, timeout=120
            )
        write_json(
            output / "native-environment.json",
            dict(actual_exit=native.returncode, source_sha256=sha256(plan["native_probe"])),
        )
        require(native.returncode == 0, "accepted native environment probe failed")
        fixture = json.loads(Path(plan["fixtures"]).read_bytes())
        fixtures = fixture["tokens"]
        require_distinct_prompts(fixtures)
        sys.path.insert(0, spec["prepared_source"])
        os.environ["LLAMA31_8B_HF_MODEL"] = os.environ["PREFILL_HF_MODEL"] = plan["checkpoint"]
        import numpy as np
        import torch
        from ttnn._experimental.layer_completion import LayerCompletionQueue, LayerCompletionRouter

        import ttnn
        from models.demos.common.prefill.adapter import PrefillRunParams
        from models.demos.common.prefill.runners.migration import export_device_map_to_file, serialize_device_map
        from models.demos.common.prefill.runners.runner_utils import build_h2d_service, open_mesh_device
        from models.demos.llama_3p1_8b_d_p.tt.input import pack_token_ids
        from models.demos.llama_3p1_8b_d_p.tt.kv_cache import write_kv_chunk
        from models.demos.llama_3p1_8b_d_p.tt.runners.adapters.llama_3p1_8b import Llama31PrefillAdapter
        from models.demos.llama_3p1_8b_d_p.tt.runners.kv_chunk_table import build_kv_chunk_address_table
        from models.demos.llama_3p1_8b_d_p.tt.tt_prefill_runtime import build_runtime

        for fn in (build_runtime, pack_token_ids, write_kv_chunk):
            require(
                str(Path(fn.__code__.co_filename).resolve()).startswith(spec["prepared_source"] + "/"),
                "wrong prepared model source",
            )
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        report["loaded_libraries"] = mapped_libraries(os.getpid(), plan["native_libraries"])
        decode = load_decoder(plan["decoder_source"], plan["source_pins"][plan["decoder_source"]], torch, np)
        adapter = Llama31PrefillAdapter()
        params = PrefillRunParams(
            mesh_shape=(4, 8),
            num_layers=32,
            first_layer_idx=0,
            is_first_rank=True,
            is_last_rank=True,
            max_seq_len=2048,
            chunk_size=1024,
            num_users=2,
            capacity_factor=1,
            num_links=2,
            gate_mode_name="DEVICE_FP32",
            kv_only_last_layer=True,
            weight_cache_path=None,
            use_trace=False,
        )
        mesh = open_mesh_device((4, 8), adapter.model_config)
        require(mesh.get_num_devices() == 32, "full32-chip mesh required")
        cache = adapter.allocate_kv_cache(mesh_device=mesh, hf_config=None, params=params)
        runtime = build_runtime(mesh, params=params, checkpoint_path=Path(plan["checkpoint"]))
        require(
            runtime.model.num_layers == len(runtime.model.layers) == 32
            and [x.layer_idx for x in runtime.model.layers] == list(range(32)),
            "full32 model layers required",
        )
        require(
            runtime.model.max_seq_len == cache.max_seq_len == 2048 and cache.k.dtype == cache.v.dtype == ttnn.bfloat8_b,
            "wrong cache geometry/dtype",
        )
        report["model_executed"] = True
        runtime.compile(cache)
        report["warmup_full32_calls"] = 2
        report["seed_writer_calls"] = seed_cache(
            ttnn, torch, mesh, cache, write_kv_chunk, cleanup_errors=report["cleanup_errors"]
        )
        table = build_kv_chunk_address_table(mesh_device=mesh, kv_cache=cache, chunk_size=1024)
        require(geometry(table) == 2048, "wrong native table capacity")
        ttnn.experimental.disaggregation.export_to_protobuf_file(table, str(output / "table.pb"))
        export_device_map_to_file(mesh, (4, 8), str(output / "device-map.txt"))
        serialize_device_map(mesh, str(output / "device-map.json"))
        mapping = json.loads((output / "device-map.json").read_bytes())
        require(len(mapping) == len(set(mapping.values())) == 32, "incomplete physical map")
        bases = [int(cache.k.buffer_address()), int(cache.v.buffer_address())]
        require(bases[0] != bases[1], "K/V allocations overlap")
        identity = dict(
            run_nonce=plan["run_nonce"],
            table_sha256=sha256(output / "table.pb"),
            device_map_sha256=sha256(output / "device-map.json"),
            cache_bases=bases,
            fixture_sha256=sha256(plan["fixtures"]),
            plan_sha256=sha256(plan_path),
        )
        saved["value"], baseline = capture(table, output / "baseline", dict(identity, ordinal=-1), decode)
        report["baseline"] = baseline
        mesh.clear_loaded_sub_device_manager()
        service = build_h2d_service(
            mesh,
            mesh_shape=(4, 8),
            chunk_size=1024,
            mapper_config=ttnn.MeshMapperConfig(placements=[ttnn.PlacementShard(0), ttnn.PlacementReplicate()]),
            worker_cores=ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0)),
            metadata_size_bytes=12,
        )
        ring = "/llama_runtime_edge_ring_" + plan["run_nonce"]
        ack = "/llama_runtime_edge_acks_" + plan["run_nonce"]
        router = LayerCompletionRouter(
            rank=0,
            world_size=1,
            master_rank=0,
            ring_shm_name=ring,
            scheduler_channel_shm_name=ack,
            teardown_timeout_ms=30000,
        )
        producer = LayerCompletionQueue.connect(ring, connect_timeout_ms=30000)
        channel = ttnn.InterProcessCounterChannel.connect(ack, connect_timeout_ms=30000)
        require(channel.try_consume_all() == 0, "unexpected initial readiness")
        original_sync = runtime._synchronize

        def sync(device):
            original_sync(device)
            recorder.synchronized(time.monotonic_ns())

        runtime._synchronize = sync

        def take_snapshot(request):
            call = RUNTIME_CALLS[request]
            parent = baseline if not report["snapshots"] else report["snapshots"][-1]
            call_identity = dict(
                identity,
                ordinal=request,
                prompt=call.prompt,
                slot=call.slot,
                begin=call.begin,
                end=call.end,
                previous_snapshot_sha256=parent["receipt_sha256"],
                token_ids_sha256=hashlib.sha256(
                    json.dumps(fixtures[call.prompt][call.begin : call.end], separators=(",", ":")).encode()
                ).hexdigest(),
            )
            new, row = capture(
                table, output / f"call-{request}", call_identity, decode, previous=saved["value"], call=call
            )
            saved["value"].close()
            saved["value"] = new
            report["snapshots"].append(row)
            return row

        def push(layer, request):
            deadline = time.monotonic() + 30
            while not producer.try_push(seq=request * 32 + layer, source_rank=0, layer_idx=layer, request_id=request):
                require(time.monotonic() < deadline, "completion route timeout")
                time.sleep(0.001)

        sink = CapturedCompletionSink(recorder, take_snapshot, push)
        runtime.set_layer_completion_sink(sink)

        def receive(call, ids):
            host = prepare_h2d_input(ids, call, pack_token_ids).to(torch.uint32).reshape(4, 1, 256).contiguous()
            service.forward_to_tensor_bytes(host.numpy(), metadata=struct.pack("<III", call.slot, call.begin, call.end))
            tokens, metadata = ttnn.experimental.deepseek_prefill.inbound_socket_service_sync(
                service, metadata_size_bytes=12
            )
            rows = [
                ttnn.to_torch(x).view(torch.int32).flatten()[:3].tolist() for x in ttnn.get_device_tensors(metadata)
            ]
            return dict(tokens=tokens, metadata=metadata, metadata_rows=rows, host=host, began_ns=time.monotonic_ns())

        def check_input(packet, call):
            shards = ttnn.get_device_tensors(packet["tokens"])
            require(len(shards) == 32, "missing input shards")
            for chip, shard in enumerate(shards):
                require(
                    torch.equal(
                        ttnn.to_torch(shard).reshape(-1).to(torch.int64),
                        packet["host"][chip // 8].reshape(-1).to(torch.int64),
                    ),
                    "borrowed H2D input changed",
                )

        def consume():
            count = 0
            deadline = time.monotonic() + 30
            while count < 32:
                count += channel.try_consume_all()
                require(count <= 32, "extra routed acknowledgments")
                require(time.monotonic() < deadline, "missing routed acknowledgments")
                if count < 32:
                    time.sleep(0.001)
            return count

        def record(request, call, routed):
            row = dict(
                request_id=request,
                prompt=call.prompt,
                slot=call.slot,
                begin=call.begin,
                end=call.end,
                routed_acks=routed,
                borrowed_input_preserved=True,
                all32_metadata_equal=True,
                runtime_only_reuse=request in (3, 4),
                native_retirement_checked=False,
            )
            report["requests"].append(row)
            write_json(output / f"call-{request}-done.json", row)
            print(
                f"RUNTIME_EDGE_DONE request={request} slot={call.slot} start={call.begin} end={call.end} acks={routed}",
                flush=True,
            )

        drive_requests(runtime, cache, recorder, fixtures, receive, check_input, consume, record)
        require(channel.try_consume_all() == 0, "extra final acknowledgments")
        require(
            len(report["requests"]) == len(report["snapshots"]) == 5 and len(recorder.rows) == 160,
            "incomplete runtime edge inventory",
        )
        report.update(
            status="runtime_edges_complete",
            persistent_h2d_tested=True,
            packed_capture_performed=True,
            configs=16,
            table_entries=65536,
            page_bytes=4352,
            mesh_devices=32,
        )
    except BaseException as exc:
        report["errors"].append(repr(exc))
        traceback.print_exc()
    finally:
        report["acks"] = recorder.rows
        report["published"] = [] if sink is None else sink.published
        owned = dict(
            service=service,
            channel=channel,
            producer=producer,
            router=router,
            saved=saved["value"],
            model=None if runtime is None else runtime.model,
            mesh=mesh,
            **{"cache.k": None if cache is None else cache.k, "cache.v": None if cache is None else cache.v},
        )
        # Clear the caller's references so service/channel drops occur in the ordered helper.
        service = channel = producer = router = cache = runtime = None
        saved["value"] = None
        report["cleanup_attempts"] = cleanup_owner(
            owned, api=ttnn if mesh is not None else None, collect=gc.collect, errors=report["cleanup_errors"]
        )
        report["owner_cleanup_complete"] = not report["cleanup_errors"]
        # Cleanup ambiguity remains a failed run for the retained root supervisor and recovery protocol.
        report["recovery_required"] = bool(report["cleanup_errors"])
        report["gate_passed"] = (
            not report["errors"] and not report["cleanup_errors"] and report.get("status") == "runtime_edges_complete"
        )
        write_json(output / "report.json", report)
    return 0 if report["gate_passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    raise SystemExit(main(args.plan, args.plan_sha256))
