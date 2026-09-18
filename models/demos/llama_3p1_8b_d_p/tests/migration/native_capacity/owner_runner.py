#!/usr/bin/env python3
"""CLOSED real-prefill capacity owner: two crossed high-end transfers, no golden or performance claim.

Launch only after a separately reviewed device/lease/process/seat preflight. Each process owns
its TTNN allocation, local manager and bridge. The passive role creates no model or decoder.
"""

import argparse
import gc
import json
import os
import signal
import socket
import struct
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

from bridge_network import prepare_bridge_config
from capacity_execution import Call, make_cases, resources, sentinel_keys
from capacity_observation import check_capacity_ready, observe_manager_memory
from capacity_pages import (
    SelectedWriter,
    check_samples,
    compare_selected,
    live_selected_group_hashes,
    require_changed_groups,
    sample_pages,
    saved_selected_group_hashes,
    selected_config_hashes,
)
from capacity_warmup import warmup_barrier, warmup_geometry
from device_identity import device_namespace_receipt
from native_observation import observe_manager, process_identity
from owned_cleanup import cleanup_each, cleanup_owner
from page_io import geometry
from peer_stop import PeerStopGuard, guarded
from range_contract import AckRecorder
from range_driver import run_phases
from runner_support import (
    Bridge,
    CapturedCompletionSink,
    mapped_libraries,
    prepare_jit_cache,
    require,
    require_clean_manager_exit,
    sha256,
    validate_plan,
    wait_receipt,
    write_json,
)
from runtime_call import execute_call
from transfer_bootstrap import EtcdBootstrap
from transfer_lifetime import (
    await_pair_cleanup,
    check_unallocated_bootstrap_release,
    finish_native_pair,
    publish_native_stopped,
)


def manager_environment(plan, role, run, output, device_ids):
    local = plan[role]
    peer = "passive" if role == "source" else "source"
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("KV_MANAGER_", "TT_KVM_"))
        and key
        not in {
            "KVM_ID",
            "PEERS",
            "ROLE",
            "LEADER_NAME",
            "PREFILL_TABLE",
            "DECODE_TABLE",
            "DEVICE_MAP",
            "HEALTH_PORT",
            "CONTROL_PORT",
            "ETCD_ENDPOINT",
            "MC_TCP_BIND_ADDRESS",
        }
    }
    env.update(
        {
            "KVM_ID": f"llama-{plan['run_nonce']}-{role}",
            "PEERS": f"llama-{plan['run_nonce']}-{peer}",
            "ROLE": "leader",
            "KV_MANAGER_TABLE_HOST": socket.gethostname(),
            "KV_MANAGER_PREFILL_KV_CHUNK_TABLE_PATH": str(run / "source" / "table.pb"),
            "KV_MANAGER_DECODE_KV_CHUNK_TABLE_PATH": str(run / "passive" / "table.pb"),
            "KV_MANAGER_DEVICE_MAP_PATH": str(output / "device-map.txt"),
            "KV_MANAGER_DEVICE_IDS": ",".join(map(str, device_ids)),
            "KV_MANAGER_SEAT_LOCK_DIR": "/dev/shm/tt-kv",
            "KV_MANAGER_DEVICE_IO": "dmk",
            "KV_MANAGER_DMK_ELF_PATH": plan["dmk_elf"],
            "KV_MANAGER_DISCOVERY_BACKEND": "etcd",
            "KV_MANAGER_ETCD_ENDPOINT": plan["etcd_endpoint"],
            "KV_MANAGER_ADVERTISE_HOST": local["host"],
            "KV_MANAGER_ADVERTISE_PORT": str(local["manager_control_port"]),
            "KV_MANAGER_CONTROL_MSG_ENDPOINT": f"tcp://0.0.0.0:{local['manager_control_port']}",
            "KV_MANAGER_TRANSPORT_KIND": "zmq",
            "KV_MANAGER_TRANSPORT_ENDPOINT": f"tcp://127.0.0.1:{local['manager_port']}",
            "KV_MANAGER_HTTP_PORT": str(local["health_port"]),
            "KV_MANAGER_TRANSFER_ENGINE_PROTOCOL": "tcp",
            "KV_MANAGER_TRANSFER_ENGINE_METADATA": plan["etcd_endpoint"].replace("http://", "etcd://", 1),
            "KV_MANAGER_STAGING_BUFFER_BYTES": "268435456",
            "KV_MANAGER_SLAB_COUNT": "32",
            "KV_MANAGER_TABLE_LOAD_MAX_RETRIES": "12",
            "KV_MANAGER_MIGRATION_TIMEOUT_MS": "120000",
            "LD_LIBRARY_PATH": plan["ld_library_path"],
            "TT_LOG_LEVEL": "info",
        }
    )
    return env


def retain_for_recovery(output, report, resources):
    # A timed-out/failed DMK teardown is not proof that its device kernels stopped. Deliberately
    # keep strong references and the owner alive; only the supervising root may resolve this case.
    report["allocation_release_blocked"] = True
    report["resources_retained"] = sorted(resources)
    manager = resources.get("manager")
    report["recovery_handoff"] = dict(
        owner=process_identity(os.getpid()),
        manager_pid=None if manager is None else manager.pid,
        manager_exit=None if manager is None else manager.poll(),
        endpoints={r: resources["plan"][r] for r in ("source", "passive")},
        node_lock=resources["local"]["node_lock"],
        lease_end_utc=resources["local"]["lease_end_utc"],
        manager_log=str(output / "manager.log"),
        action="Root must coordinate both endpoints and prove native I/O stopped or reset before terminating either retained owner; lease expiry is not safe retention",
    )
    if manager is not None and manager.poll() is None:
        try:
            report["recovery_handoff"]["manager_identity"] = process_identity(manager.pid)
        except OSError as error:
            report["recovery_handoff"]["identity_error"] = repr(error)
    try:
        write_json(output / "recovery-required.json", report)
    except BaseException:
        traceback.print_exc()
    try:
        print("NATIVE_MIGRATION_RECOVERY_REQUIRED allocations retained; no reset or forced release", flush=True)
    except OSError:
        pass
    receipt = await_pair_cleanup(
        output / "root-cleanup.json", report["owner"], manager, resources["plan"], report["role"]
    )
    report["allocation_release_blocked"] = False
    report["root_authorized_cleanup"] = receipt
    write_json(output / "root-cleanup-accepted.json", receipt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--role", choices=("source", "passive"), required=True)
    args = parser.parse_args()
    require(sha256(args.plan) == args.plan_sha256, "Plan bytes differ")
    plan = json.loads(args.plan.read_bytes())
    validate_plan(plan)
    role, nonce = args.role, plan["run_nonce"]
    local = plan[role]
    peer_role = "passive" if role == "source" else "source"
    require(socket.gethostname() == local["host"], "Wrong assigned host")
    require(os.environ.get("SLURM_JOB_ID") == str(local["job_id"]), "Wrong assigned job")
    require(os.environ.get("PREFILL_FABRIC_MODE") == "1d_ring", "Accepted ring fabric required")
    require(os.environ.get("TT_METAL_SLOW_DISPATCH_MODE") not in ("1", "true"), "Fast dispatch required")
    run = Path(plan["run_dir"])
    require(run.is_dir(), "Controller must create a fresh shared run directory")
    output = run / role
    output.mkdir(exist_ok=False)
    lock_fd = int(os.environ["MIGRATION_NODE_LOCK_FD"])
    lock_stat, path_stat = os.fstat(lock_fd), Path(local["node_lock"]).stat()
    require((lock_stat.st_dev, lock_stat.st_ino) == (path_stat.st_dev, path_stat.st_ino), "Wrong inherited node lock")
    lock_info = Path(f"/proc/self/fdinfo/{lock_fd}").read_text()
    require(
        "FLOCK  ADVISORY  WRITE" in lock_info or "FLOCK ADVISORY WRITE" in " ".join(lock_info.split()),
        "Owner must inherit the supervisor's held node lock",
    )
    stopped = {"requested": False}
    peer_guard = PeerStopGuard(run, role, nonce, output)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopped.update(requested=True))

    def check_stop():
        request = run / (role + "-stop-requested.json")
        if request.exists():
            value = json.loads(request.read_bytes())
            require(value.get("run_nonce") == nonce and value.get("role") == role, "Foreign stop request")
            stopped["requested"] = True
        require(not stopped["requested"], "Cooperative supervisor stop requested")
        peer_guard.check()

    report = dict(
        run_nonce=nonce,
        role=role,
        ok=False,
        errors=[],
        cleanup_errors=[],
        native_transfer_tested=False,
        decoder_tested=False,
        full_model_accepted=False,
        new_model_numerical_acceptance=False,
        kv_golden_comparison_performed=False,
        owner=process_identity(os.getpid()),
        owner_cleanup_complete=False,
    )
    report["jit_cache"] = prepare_jit_cache(output, os.environ)
    write_json(output / "jit-cache-before.json", report["jit_cache"])
    mesh = runtime = cache = service = producer = router = manager = bridge = table = None
    tokens = metadata = shards = shard = None
    manager_log = None
    manager_safe = True
    pair_safe = False
    bootstrap = EtcdBootstrap(plan, role, output)
    saved = {"value": None}
    recorder = sink = None

    def publish(name, **fields):
        write_json(output / (name + ".json"), dict(run_nonce=nonce, role=role, ok=True, **fields))

    def wait_peer(name, timeout=300):
        def check():
            check_stop()
            require(not (run / peer_role / "failure.json").exists(), "Peer endpoint failed")
            if manager is not None:
                require(manager.poll() is None, "Local manager exited prematurely")

        return wait_receipt(run / peer_role / (name + ".json"), nonce, timeout, check)

    try:
        check_stop()
        bridge_config, network = prepare_bridge_config(plan, role)
        write_json(output / "bridge-network.json", network)
        report["etcd_preflight"] = bootstrap.start(check_stop)
        with (output / "native-environment.log").open("wb") as probe_log:
            probe = subprocess.run(
                [sys.executable, "-B", plan["native_probe"]], stdout=probe_log, stderr=subprocess.STDOUT, timeout=120
            )
        write_json(
            output / "native-environment.json",
            dict(actual_exit=probe.returncode, probe_sha256=sha256(plan["native_probe"])),
        )
        require(probe.returncode == 0, "Native environment probe failed")
        check_stop()
        spec = json.loads(Path(plan["accepted_gate_spec"]).read_bytes())
        for path, digest in json.loads(Path(plan["accepted_source_pins"]).read_bytes()).items():
            require(sha256(path) == digest, "Accepted source changed: " + path)
        for name, digest in spec["checkpoint_metadata"].items():
            require(sha256(Path(plan["checkpoint"]) / name) == digest, "Checkpoint identity differs")
        doc, fixtures = make_cases(plan["capacity"], plan["book_manifest"], plan["pins"][plan["book_manifest"]])
        calls = [call for phase in doc["phases"] for call in phase["compute_calls"]]
        sys.path.insert(0, spec["prepared_source"])
        os.environ["LLAMA31_8B_HF_MODEL"] = plan["checkpoint"]
        os.environ["PREFILL_HF_MODEL"] = plan["checkpoint"]
        import numpy as np  # noqa: F401 - retain the tested native bootstrap import order.
        import torch

        import ttnn
        from models.demos.common.prefill.adapter import PrefillRunParams
        from models.demos.common.prefill.runners.migration import export_device_map_to_file, serialize_device_map
        from models.demos.common.prefill.runners.runner_utils import build_h2d_service, open_mesh_device
        from models.demos.llama_3p1_8b_d_p.tt.input import pack_token_ids
        from models.demos.llama_3p1_8b_d_p.tt.kv_cache import write_kv_chunk
        from models.demos.llama_3p1_8b_d_p.tt.runners.adapters.llama_3p1_8b import Llama31PrefillAdapter
        from models.demos.llama_3p1_8b_d_p.tt.runners.kv_chunk_table import build_kv_chunk_address_table

        for fn in (pack_token_ids, write_kv_chunk, build_kv_chunk_address_table):
            require(
                str(Path(fn.__code__.co_filename).resolve()).startswith(spec["prepared_source"] + "/"),
                "Wrong prepared model/cache/table helper source",
            )
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        adapter = Llama31PrefillAdapter()
        params = PrefillRunParams(
            mesh_shape=(4, 8),
            num_layers=32,
            first_layer_idx=0,
            is_first_rank=True,
            is_last_rank=True,
            max_seq_len=plan["capacity"],
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
        require(mesh.get_num_devices() == 32, "Full Galaxy required")
        cache = adapter.allocate_kv_cache(mesh_device=mesh, hf_config=None, params=params)
        require(cache.k.dtype == cache.v.dtype == ttnn.bfloat8_b, "Packed BFP8 cache required")
        if role == "source":
            from ttnn._experimental.layer_completion import LayerCompletionQueue, LayerCompletionRouter

            from models.demos.llama_3p1_8b_d_p.tt.tt_prefill_runtime import build_runtime

            require(
                str(Path(build_runtime.__code__.co_filename).resolve()).startswith(spec["prepared_source"] + "/"),
                "Wrong prepared runtime source",
            )
            runtime = build_runtime(mesh, params=params, checkpoint_path=Path(plan["checkpoint"]))
            require(runtime.model.max_seq_len == cache.max_seq_len == plan["capacity"], "Model/cache capacity differs")
            require(
                runtime.model.num_layers == len(runtime.model.layers) == 32
                and [x.layer_idx for x in runtime.model.layers] == list(range(32)),
                "Full32 model required",
            )
            runtime.compile(cache)
            mesh.clear_loaded_sub_device_manager()
            service = build_h2d_service(
                mesh,
                mesh_shape=(4, 8),
                chunk_size=1024,
                mapper_config=ttnn.MeshMapperConfig(placements=[ttnn.PlacementShard(0), ttnn.PlacementReplicate()]),
                worker_cores=ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0)),
                metadata_size_bytes=12,
            )
            ring = "/llama_native_ring_" + nonce
            ack_name = "/llama_native_acks_" + nonce
            router = LayerCompletionRouter(
                rank=0,
                world_size=1,
                master_rank=0,
                ring_shm_name=ring,
                scheduler_channel_shm_name=ack_name,
                teardown_timeout_ms=30000,
            )
            producer = LayerCompletionQueue.connect(ring, connect_timeout_ms=30000)
        report["seed_writer_calls"] = 0
        report["initial_cache"] = "Production zero allocation; actual source model writes valid prompt KV"
        table = build_kv_chunk_address_table(mesh_device=mesh, kv_cache=cache, chunk_size=1024)
        require(geometry(table) == plan["capacity"], "Table/cache/plan capacity differs")
        ttnn.experimental.disaggregation.export_to_protobuf_file(table, str(output / "table.pb"))
        export_device_map_to_file(mesh, (4, 8), str(output / "device-map.txt"))
        serialize_device_map(mesh, str(output / "device-map.json"))
        mapping = json.loads((output / "device-map.json").read_bytes())
        require(len(mapping) == len(set(mapping.values())) == 32, "Incomplete physical ASIC identity map")
        namespaces = device_namespace_receipt(ttnn, mesh, mapping)
        write_json(output / "device-namespaces.json", namespaces)
        devices = namespaces["umd_device_ids"]
        write_json(output / "owner-loaded-libraries.json", mapped_libraries(os.getpid(), plan["manager_libraries"]))
        identity = dict(
            run_nonce=nonce,
            role=role,
            plan_sha256=args.plan_sha256,
            table_sha256=sha256(output / "table.pb"),
            map_sha256=sha256(output / "device-map.json"),
            cache_bases=[int(cache.k.buffer_address()), int(cache.v.buffer_address())],
            capacity=plan["capacity"],
            book_manifest_sha256=sha256(plan["book_manifest"]),
        )
        selected_receipts = []
        source_writers = {}
        pre_request_groups = {}
        phase_sentinels = {}
        final_sentinel_keys = sorted(
            {k for phase in doc["phases"] for k in sentinel_keys(plan["capacity"], phase, doc["phases"])}
        )
        final_sentinels = sample_pages(table, final_sentinel_keys) if role == "passive" else {}
        write_json(output / "initial-sentinels.json", final_sentinels)
        report["capacity_resources"] = resources(plan["capacity"])
        report["capacity"] = plan["capacity"]
        report["book_manifest_sha256"] = sha256(plan["book_manifest"])
        report["compile_warmup_full32_calls"] = 2 if role == "source" else 0
        report["sentinel_readback_plan"] = dict(
            per_phase_keys=[len(sentinel_keys(plan["capacity"], phase)) for phase in doc["phases"]],
            final_keys=len(final_sentinel_keys),
            method="packed page SHA256",
            excludes_selected_destination_bytes=True,
        )
        report["selected_captures"] = []
        publish(
            "allocation",
            owner=report["owner"],
            table_sha256=sha256(output / "table.pb"),
            map_sha256=sha256(output / "device-map.txt"),
            unique_asic_ids=sorted(mapping.values()),
            bases=[int(cache.k.buffer_address()), int(cache.v.buffer_address())],
        )
        other = wait_peer("allocation", timeout=plan["source_setup_timeout_seconds"])
        require(other["table_sha256"] == sha256(run / peer_role / "table.pb"), "Peer table changed")
        require(other["map_sha256"] == sha256(run / peer_role / "device-map.txt"), "Peer map changed")
        check_stop()

        def run_capacity_warmup():
            require(bridge is None and manager is None, "Native clients must not age during model warmup")

            def completed(row):
                print("CAPACITY_WARMUP_CHUNK " + json.dumps(row, sort_keys=True), flush=True)

            return warmup_geometry(runtime, cache, doc, fixtures, check_stop, completed)

        report["capacity_warmup"] = warmup_barrier(
            role, run_capacity_warmup, publish, wait_peer, plan["warmup_timeout_seconds"]
        )
        report["warmup_barrier_before_native_clients"] = True
        check_stop()
        env = manager_environment(plan, role, run, output, devices)
        write_json(
            output / "manager-environment.json",
            {
                k: v
                for k, v in env.items()
                if k.startswith("KV_MANAGER_") or k in {"KVM_ID", "PEERS", "ROLE", "LD_LIBRARY_PATH"}
            },
        )
        manager_log = (output / "manager.log").open("xb")
        manager = subprocess.Popen([plan["manager_binary"]], stdout=manager_log, stderr=subprocess.STDOUT, env=env)
        manager_safe = False
        write_json(
            output / "manager-started.json",
            dict(
                run_nonce=nonce, role=role, owner=report["owner"], endpoint=local, manager=process_identity(manager.pid)
            ),
        )
        deadline = time.monotonic() + 180
        while True:
            check_stop()
            require(manager.poll() is None, "Manager exited during startup")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{local['health_port']}/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except (urllib.error.URLError, TimeoutError):
                pass
            require(time.monotonic() < deadline, "Manager health readiness timed out")
            time.sleep(0.05)
        report["table_readiness"] = check_capacity_ready(
            plan["capacity"], role, local["host"], 200, (output / "manager.log").read_text(errors="replace")
        )
        report["manager_memory_after_tables"] = observe_manager_memory(
            manager,
            json.loads((output / "manager-started.json").read_bytes())["manager"],
            plan["manager_rss_limit_bytes"][role],
        )

        def observe(label):
            value = observe_manager(
                manager,
                os.getpid(),
                {int(x) for x in mapping.values()},
                plan["manager_libraries"],
                {
                    k: v
                    for k, v in env.items()
                    if k.startswith("KV_MANAGER_") or k in {"KVM_ID", "PEERS", "ROLE", "LD_LIBRARY_PATH"}
                },
            )
            write_json(output / (label + ".json"), value)

        observe("ownership-before-transfer")
        publish("manager-ready", pid=manager.pid)
        wait_peer("manager-ready")
        cfg = dict(bridge_config)
        if role == "source":
            cfg["ack_shm"] = ack_name
        bridge = Bridge(plan[role + "_client"], cfg, output, env, check=check_stop)
        if role == "source":
            write_json(
                output / "source-client-loaded-libraries.json",
                mapped_libraries(bridge.process.pid, plan["manager_libraries"]),
            )
            recorder = AckRecorder(calls)
            original_sync = runtime._synchronize

            def synchronized(device):
                original_sync(device)
                recorder.synchronized(time.monotonic_ns())

            runtime._synchronize = synchronized
            active = {}

            def take_snapshot(request):
                phase, call = active["phase"], active["call"]
                uuid = phase["source_command"]["uuid"]
                if uuid not in source_writers:
                    source_writers[uuid] = SelectedWriter(output / ("selected-" + str(uuid) + ".bin"), phase)
                writer = source_writers[uuid]
                count = writer.capture(table, call["begin"], call["end"])
                row = dict(request=request, selected_pages=count, begin=call["begin"], end=call["end"])
                if call["end"] == phase["valid_prompt_tokens"]:
                    row["selected"] = writer.finish()
                    row["changed_from_pre_request"] = require_changed_groups(
                        pre_request_groups[uuid], saved_selected_group_hashes(row["selected"])
                    )
                    row["selected"]["capture_complete_ns"] = time.monotonic_ns()
                    selected_receipts.append((phase, row["selected"]))
                report["selected_captures"].append(row)
                return row

            def push(layer, request):
                deadline = time.monotonic() + 30
                check_stop()
                while not producer.try_push(
                    seq=request * 32 + layer, source_rank=0, layer_idx=layer, request_id=request
                ):
                    check_stop()
                    require(time.monotonic() < deadline, "Real acknowledgment router backpressure timed out")
                    time.sleep(0.001)

            sink = CapturedCompletionSink(recorder, take_snapshot, push, check=check_stop)
            runtime.set_layer_completion_sink(sink)

        def run_call(phase, call, ordinal):
            check_stop()
            active.update(phase=phase, call=call)
            typed = Call(phase["name"], phase["fixture"], call["slot"], call["begin"], call["end"])
            host = (
                pack_token_ids(
                    fixtures[typed.prompt][typed.begin : typed.end],
                    actual_start=typed.begin,
                    actual_end=typed.end,
                    pad_id=0,
                    max_seq_len=plan["capacity"],
                )
                .to(torch.uint32)
                .reshape(4, 1, 256)
                .contiguous()
            )
            guarded(
                check_stop,
                service.forward_to_tensor_bytes,
                host.numpy(),
                metadata=struct.pack("<III", typed.slot, typed.begin, typed.end),
            )
            borrowed, meta = guarded(
                check_stop,
                ttnn.experimental.deepseek_prefill.inbound_socket_service_sync,
                service,
                metadata_size_bytes=12,
            )
            rows = [ttnn.to_torch(x).view(torch.int32).flatten()[:3].tolist() for x in ttnn.get_device_tensors(meta)]
            packet = dict(tokens=borrowed, metadata=meta, metadata_rows=rows, host=host, began_ns=time.monotonic_ns())

            def check_input(value, typed):
                parts = ttnn.get_device_tensors(value["tokens"])
                require(len(parts) == 32, "Borrowed H2D tensor lost shards")
                for chip, part in enumerate(parts):
                    require(
                        torch.equal(
                            ttnn.to_torch(part).reshape(-1).to(torch.int64),
                            value["host"][chip // 8].reshape(-1).to(torch.int64),
                        ),
                        "Borrowed input changed",
                    )

            execute_call(runtime, cache, recorder, bridge, typed, ordinal, packet, check_input, check_stop)
            report.setdefault("requests", []).append(
                dict(
                    ordinal=ordinal,
                    uuid=phase["source_command"]["uuid"],
                    **call,
                    routed_acks=32,
                    borrowed_input_preserved=True,
                    all32_metadata_equal=True,
                )
            )
            # Borrowed tensors belong to the persistent H2D service and are not deallocated here.
            return report["selected_captures"][-1]

        def before_source(phase):
            uuid = phase["source_command"]["uuid"]
            pre_request_groups[uuid] = live_selected_group_hashes(table, phase)
            write_json(output / (phase["name"] + "-selected-before-request.json"), pre_request_groups[uuid])

        def before_destination(phase):
            values = sample_pages(table, sentinel_keys(plan["capacity"], phase))
            phase_sentinels[phase["name"]] = values
            write_json(output / (phase["name"] + "-sentinels-before.json"), values)

        def verify_destination(phase, source_row):
            receipt = source_row["selected"]
            result = compare_selected(receipt, phase, table, destination=True)
            result["untouched_samples"] = check_samples(table, phase_sentinels[phase["name"]])
            result["source_sha256"] = receipt["sha256"]
            selected_receipts.append((phase, receipt))
            return result

        def verify_source(phase, source_row):
            return compare_selected(source_row["selected"], phase, table, destination=False)

        phases, terminal = run_phases(
            role,
            doc,
            nonce,
            bridge,
            publish,
            wait_peer,
            run_call,
            verify_destination,
            check_stop,
            before_destination,
            verify_source,
            before_source,
            plan["transfer_phase_timeout_seconds"],
        )
        report["phases"] = phases
        expected_calls = 2 * (plan["capacity"] // 1024)
        if role == "source":
            require(
                len(report["requests"]) == expected_calls
                and len(recorder.rows) == len(sink.published) == expected_calls * 32,
                "Incomplete real H2D/full32 call/ack inventory",
            )
            report["acks"], report["published"] = recorder.rows, sink.published
            hashes = [selected_config_hashes(row) for _, row in selected_receipts]
            require(
                len(hashes) == 2 and all(hashes[0][str(c)] != hashes[1][str(c)] for c in range(16)),
                "Selected source slots are not distinct in every config",
            )
            report["selected_source_config_hashes"] = hashes
        report["manager_memory_after_transfer"] = observe_manager_memory(
            manager,
            json.loads((output / "manager-started.json").read_bytes())["manager"],
            plan["manager_rss_limit_bytes"][role],
        )
        publish("transfers-complete", terminal=terminal)
        wait_peer("transfers-complete")
        observe("ownership-after-transfer")
        drained = bridge.drain()
        report["bridge_exit_code"] = drained["bridge_exit_code"]
        publish("client-drained", terminal=drained)
        wait_peer("client-drained")
        manager.send_signal(signal.SIGTERM)
        rc = manager.wait(timeout=150)
        manager_log.flush()
        require_clean_manager_exit(rc, (output / "manager.log").read_text(errors="replace"))
        manager_safe = True
        publish("manager-stopped", returncode=rc, log_sha256=sha256(output / "manager.log"))
        finish_native_pair(plan, role, report["owner"], manager, output)
        pair_safe = True
        report["after_manager_shutdown"] = [
            compare_selected(row, phase, table, destination=(role == "passive")) for phase, row in selected_receipts
        ]
        if role == "passive":
            report["final_untouched_samples"] = check_samples(table, final_sentinels)
        report.update(
            ok=True,
            native_transfer_tested=True,
            source_client_tested=True,
            persistent_h2d_tested=(role == "source"),
            model_executed=(role == "source"),
            manager_exit=rc,
            terminal=drained,
        )
        publish("bytes-verified")
        wait_receipt(run / peer_role / "bytes-verified.json", nonce, timeout=300, check=check_stop)
    except BaseException as error:
        report["errors"].append(repr(error))
        traceback.print_exc()
        write_json(output / "failure.json", dict(run_nonce=nonce, role=role, ok=False, error=repr(error)))
    finally:
        # Stop command producers first; a live peer manager may still have submitted device work.
        # Neither PID disappearance nor a local clean exit alone permits cache release.
        try:
            if bridge is not None and bridge.process.poll() is None:
                bridge.process.terminate()
                bridge.process.wait(timeout=10)
            if manager is not None and not manager_safe:
                if manager.poll() is None:
                    manager.send_signal(signal.SIGTERM)
                rc = manager.wait(timeout=150)
                manager_log.flush()
                require_clean_manager_exit(rc, (output / "manager.log").read_text(errors="replace"))
                manager_safe = True
            if cache is not None and not pair_safe:
                finish_native_pair(plan, role, report["owner"], manager, output)
                pair_safe = True
            elif cache is None:
                publish_native_stopped(plan, role, report["owner"], manager, output)
                # No local allocation receipt was published, so the peer cannot pass its
                # allocation rendezvous and start a manager. Assert the invariant before
                # reaping source-owned etcd on this pre-allocation failure path.
                check_unallocated_bootstrap_release(plan, role, output)
        except BaseException as error:
            report["cleanup_errors"].append(repr(error))
            retain_for_recovery(output, report, locals())
            manager_safe = pair_safe = True
        # The unchanged pair barrier above must complete before any owned cache cleanup action.
        require(pair_safe or cache is None, "Both-manager stop proof missing before cache release")
        attempts = []
        for writer in locals().get("source_writers", {}).values():
            cleanup_each((("selected file close", writer.close),), report["cleanup_errors"], attempts)
        cleanup_each((("etcd.close", bootstrap.close),), report["cleanup_errors"], attempts)
        tokens = metadata = shards = shard = None
        owned = dict(
            service=service,
            producer=producer,
            router=router,
            saved=saved["value"],
            model=None if runtime is None else runtime.model,
            mesh=mesh,
            **{"cache.k": None if cache is None else cache.k, "cache.v": None if cache is None else cache.v},
        )
        service = producer = router = runtime = cache = None
        saved["value"] = None
        attempts += cleanup_owner(
            owned, api=ttnn if mesh is not None else None, collect=gc.collect, errors=report["cleanup_errors"]
        )
        report["cleanup_attempts"] = attempts
        report["owner_cleanup_complete"] = not report["cleanup_errors"]
        if recorder is not None:
            report["acks"] = recorder.rows
            report["published"] = [] if sink is None else sink.published
        if manager_log is not None:
            cleanup_each(
                (("manager log close", manager_log.close),), report["cleanup_errors"], report["cleanup_attempts"]
            )
        cache_files = sorted(Path(report["jit_cache"]["path"]).rglob("*.elf"))
        write_json(
            output / "jit-cache-after.json",
            dict(
                files={str(p): sha256(p) for p in cache_files},
                bytes=sum(p.stat().st_size for p in cache_files),
                count=len(cache_files),
            ),
        )
        if report["ok"] and role == "source" and not cache_files:
            report["cleanup_errors"].append("Source produced no ELF evidence in its fresh JIT cache")
        report["ok"] = report["ok"] and not report["errors"] and not report["cleanup_errors"]
        write_json(output / "result.json", report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
