#!/usr/bin/env python3
"""One real first-chunk cancel and one real32-token restart under a retained owner."""

import argparse
import ast
import gc
import hashlib
import json
import os
import shutil
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
from cancel_restart_scenario import run_cancel_epoch, run_restart_epoch
from device_identity import device_namespace_receipt
from edge_checks import seed_value
from epoch_contract import check_fresh_identities, check_role_events, epoch_plan, require_epoch_b_allowed
from manager_environment import manager_environment
from native_observation import observe_manager, process_identity
from owned_cleanup import cleanup_each, cleanup_owner
from page_io import PAGE, geometry, range_keys, read_page
from peer_stop import PeerStopGuard
from runner_support import (
    Bridge,
    mapped_libraries,
    prepare_jit_cache,
    require,
    require_clean_manager_exit,
    sha256,
    validate_plan,
    wait_receipt,
    write_json,
)
from runtime_cancel import produce_real_chunk, prompt_pair
from transfer_bootstrap import EtcdBootstrap
from transfer_contract import check_manager_ready
from transfer_lifetime import await_pair_cleanup, finish_native_pair, peer_stopped


def selected_pages(table, output, label, slot, end=32):
    """Read every selected config/layer page; save actual packed bytes for the final verifier."""
    hashes = {}
    path = output / (label + ".bin")
    with path.open("xb") as stream:
        for config, _, layer, position in range_keys(slot, 0, end):
            raw = read_page(table, (config, slot, layer, position))
            require(any(raw), "Selected seeded page is all zero")
            stream.write(raw)
            key = f"{config}:{layer}" if end == 32 else f"{config}:{layer}:{position}"
            hashes[key] = hashlib.sha256(raw).hexdigest()
        stream.flush()
        os.fsync(stream.fileno())
    count = 512 * (end // 32)
    require(len(hashes) == count and path.stat().st_size == count * PAGE, "Selected page inventory differs")
    write_json(
        output / (label + ".json"),
        dict(
            path=str(path),
            sha256=sha256(path),
            pages=count,
            bytes=count * PAGE,
            slot=slot,
            begin=0,
            end=end,
            groups=hashes,
        ),
    )
    return hashes


def seeded_cache(plan, ttnn, torch, mesh, cache, writer, errors):
    path = Path(plan["seed_helper"])
    require(sha256(path) == plan["pins"][str(path)], "Seed helper changed")
    node = next(
        n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "seed_cache"
    )
    namespace = dict(seed_value=seed_value, cleanup_each=cleanup_each)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    count = namespace["seed_cache"](ttnn, torch, mesh, cache, writer, cleanup_errors=errors)
    require(count == 128 and not errors, "Incomplete seed or seed cleanup failure")
    return count


def terminal_bridge(bridge):
    # drain_cancelled already validated the role-specific terminal, and exits main normally.
    code = bridge.process.wait(timeout=150)
    bridge._record_exit(code, bridge.rows())
    require(code == 0, "Bridge terminal exit failed")
    if not bridge.process.stdin.closed:
        bridge.process.stdin.close()
    if not bridge.log.closed:
        bridge.log.close()
    return code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--plan-sha256", required=True)
    ap.add_argument("--role", choices=("source", "passive"), required=True)
    a = ap.parse_args()
    require(sha256(a.plan) == a.plan_sha256, "Plan changed")
    plan = json.loads(a.plan.read_bytes())
    validate_plan(plan)
    role = a.role
    peer = "passive" if role == "source" else "source"
    local = plan[role]
    require(
        socket.gethostname() == local["host"] and os.environ.get("SLURM_JOB_ID") == str(local["job_id"]),
        "Wrong assigned endpoint",
    )
    require(
        len(os.sched_getaffinity(0)) == 1 and os.environ.get("SLURM_CPUS_PER_TASK") == "1",
        "Single CPU inheritance required",
    )
    require(
        all(
            os.environ.get(k) == "1"
            for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        ),
        "Single-thread contract required",
    )
    require(
        os.environ.get("PREFILL_FABRIC_MODE") == "1d_ring"
        and os.environ.get("TT_METAL_SLOW_DISPATCH_MODE") not in ("1", "true"),
        "Wrong dispatch/fabric",
    )
    run = Path(plan["run_dir"])
    output = run / role
    output.mkdir(exist_ok=False)
    fd = int(os.environ["MIGRATION_NODE_LOCK_FD"])
    fs = os.fstat(fd)
    ps = Path(local["node_lock"]).stat()
    require((fs.st_dev, fs.st_ino) == (ps.st_dev, ps.st_ino), "Wrong inherited physical lock")
    require(
        "FLOCK ADVISORY WRITE" in " ".join(Path(f"/proc/self/fdinfo/{fd}").read_text().split()),
        "Physical lock is not held",
    )
    stop = {"requested": False}
    guard = PeerStopGuard(run, role, plan["run_nonce"], output)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.update(requested=True))

    def check():
        path = run / (role + "-stop-requested.json")
        if path.exists():
            v = json.loads(path.read_bytes())
            require(v.get("run_nonce") == plan["run_nonce"] and v.get("role") == role, "Foreign stop request")
            stop["requested"] = True
        require(not stop["requested"], "Cooperative stop requested")
        guard.check()

    owner = process_identity(os.getpid())
    report = dict(
        run_nonce=plan["run_nonce"],
        role=role,
        owner=owner,
        ok=False,
        errors=[],
        cleanup_errors=[],
        owner_cleanup_complete=False,
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        events=[],
        epochs={},
        synthetic_acks=0,
        real_layer_acks=0,
        runtime_h2d_tested=False,
        model_executed=False,
        decoder_tested=False,
        bytes_in_flight_at_cancel_proven=False,
    )
    report["jit_cache"] = prepare_jit_cache(output, os.environ)
    write_json(output / "jit-cache-before.json", report["jit_cache"])
    bootstrap = EtcdBootstrap(plan, role, output)
    mesh = cache = table = runtime = service = None
    ttnn = None
    states = {}
    for epoch in ("a", "b"):
        ep = epoch_plan(plan, epoch)
        out = Path(ep["run_dir"]) / role
        out.mkdir(parents=True, exist_ok=False)
        states[epoch] = dict(
            plan=ep, output=out, manager=None, bridge=None, manager_log=None, producer=None, router=None, safe=True
        )

    def publish(out, ep, name, **fields):
        write_json(out / (name + ".json"), dict(run_nonce=ep["run_nonce"], role=role, ok=True, **fields))

    def wait(ep, name, timeout=300):
        return wait_receipt(Path(ep["run_dir"]) / peer / (name + ".json"), ep["run_nonce"], timeout, check)

    def stop_native(state):
        manager = state["manager"]
        bridge = state["bridge"]
        out = state["output"]
        if bridge is not None and bridge.process.poll() is None:
            bridge.process.terminate()
            bridge.process.wait(timeout=10)
        if manager is not None and not state["safe"]:
            if manager.poll() is None:
                manager.send_signal(signal.SIGTERM)
            code = manager.wait(timeout=150)
            state["manager_log"].flush()
            require_clean_manager_exit(code, (out / "manager.log").read_text(errors="replace"))
            state["safe"] = True
        require(state["safe"], "Native manager stop remains ambiguous")

    def pair_stop(state):
        stop_native(state)
        return finish_native_pair(state["plan"], role, owner, state["manager"], state["output"])

    def identities(epoch):
        ep = states[epoch]["plan"]
        result = {k: {} for k in ("manager", "bridge")}
        for endpoint in ("source", "passive"):
            v = json.loads((Path(ep["run_dir"]) / endpoint / "processes.json").read_bytes())
            require(v["run_nonce"] == ep["run_nonce"] and v["role"] == endpoint, "Foreign epoch identities")
            for kind in result:
                result[kind][endpoint] = v[kind]
        return result

    try:
        check()
        for state in states.values():
            state["bridge_config"], network = prepare_bridge_config(state["plan"], role)
            write_json(state["output"] / "bridge-network.json", network)
        report["etcd_preflight"] = bootstrap.start(check)
        with (output / "native-environment.log").open("xb") as log:
            probe = subprocess.run(
                [sys.executable, "-B", plan["native_probe"]], stdout=log, stderr=subprocess.STDOUT, timeout=120
            )
        write_json(
            output / "native-environment.json",
            dict(actual_exit=probe.returncode, probe_sha256=sha256(plan["native_probe"])),
        )
        require(probe.returncode == 0, "Native environment failed")
        check()
        spec = json.loads(Path(plan["accepted_gate_spec"]).read_bytes())
        for path, digest in json.loads(Path(plan["accepted_source_pins"]).read_bytes()).items():
            require(sha256(path) == digest, "Accepted source changed: " + path)
        for name, digest in spec["checkpoint_metadata"].items():
            require(sha256(Path(plan["checkpoint"]) / name) == digest, "Checkpoint identity differs")
        prompts = prompt_pair(
            json.loads(Path(plan["input_ids"]).read_bytes()), json.loads(Path(plan["restart_input_ids"]).read_bytes())
        )
        sys.path.insert(0, spec["prepared_source"])
        os.environ["LLAMA31_8B_HF_MODEL"] = os.environ["PREFILL_HF_MODEL"] = plan["checkpoint"]
        import torch

        import ttnn
        from models.demos.common.prefill.adapter import PrefillRunParams
        from models.demos.common.prefill.runners.migration import export_device_map_to_file, serialize_device_map
        from models.demos.common.prefill.runners.runner_utils import build_h2d_service, open_mesh_device
        from models.demos.llama_3p1_8b_d_p.tt.kv_cache import write_kv_chunk
        from models.demos.llama_3p1_8b_d_p.tt.runners.adapters.llama_3p1_8b import Llama31PrefillAdapter
        from models.demos.llama_3p1_8b_d_p.tt.runners.kv_chunk_table import build_kv_chunk_address_table

        for fn in (build_kv_chunk_address_table, write_kv_chunk):
            require(
                str(Path(fn.__code__.co_filename).resolve()).startswith(spec["prepared_source"] + "/"),
                "Wrong production cache helper",
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
        require(mesh.get_num_devices() == 32, "Full32 required")
        cache = adapter.allocate_kv_cache(mesh_device=mesh, hf_config=None, params=params)
        require(cache.k.dtype == cache.v.dtype == ttnn.bfloat8_b, "BFP8 cache required")
        if role == "source":
            from models.demos.llama_3p1_8b_d_p.tt.input import pack_token_ids
            from models.demos.llama_3p1_8b_d_p.tt.tt_prefill_runtime import build_runtime

            for fn in (build_runtime, pack_token_ids):
                require(
                    str(Path(fn.__code__.co_filename).resolve()).startswith(spec["prepared_source"] + "/"),
                    "Wrong production runtime helper",
                )
            runtime = build_runtime(mesh, params=params, checkpoint_path=Path(plan["checkpoint"]))
            require(
                runtime.model.num_layers == len(runtime.model.layers) == 32
                and [x.layer_idx for x in runtime.model.layers] == list(range(32)),
                "Full32 real model required",
            )
            require(runtime.model.max_seq_len == cache.max_seq_len == 2048, "Runtime/cache capacity differs")
            report["model_executed"] = True
            runtime.compile(cache)
            report["warmup_full32_calls"] = 2
            mesh.clear_loaded_sub_device_manager()
            service = build_h2d_service(
                mesh,
                mesh_shape=(4, 8),
                chunk_size=1024,
                mapper_config=ttnn.MeshMapperConfig(placements=[ttnn.PlacementShard(0), ttnn.PlacementReplicate()]),
                worker_cores=ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0)),
                metadata_size_bytes=12,
            )
        report["initial_seed_calls"] = seeded_cache(
            plan, ttnn, torch, mesh, cache, write_kv_chunk, report["cleanup_errors"]
        )
        table = build_kv_chunk_address_table(mesh_device=mesh, kv_cache=cache, chunk_size=1024)
        require(geometry(table) == 2048, "Exact2K required")
        ttnn.experimental.disaggregation.export_to_protobuf_file(table, str(output / "table.pb"))
        export_device_map_to_file(mesh, (4, 8), str(output / "device-map.txt"))
        serialize_device_map(mesh, str(output / "device-map.json"))
        mapping = json.loads((output / "device-map.json").read_bytes())
        require(len(mapping) == len(set(mapping.values())) == 32, "Incomplete32 ASIC map")
        ns = device_namespace_receipt(ttnn, mesh, mapping)
        write_json(output / "device-namespaces.json", ns)
        write_json(output / "owner-loaded-libraries.json", mapped_libraries(os.getpid(), plan["manager_libraries"]))
        for state in states.values():
            for name in ("table.pb", "device-map.txt", "device-map.json"):
                shutil.copyfile(output / name, state["output"] / name)
        publish(
            output,
            plan,
            "allocation",
            owner=owner,
            bases=[int(cache.k.buffer_address()), int(cache.v.buffer_address())],
            table_sha256=sha256(output / "table.pb"),
        )
        other = wait(plan, "allocation", 900)
        require(other["table_sha256"] == sha256(run / peer / "table.pb"), "Peer table changed")
        for epoch, state in states.items():
            ep = state["plan"]
            out = state["output"]
            check()
            if epoch == "b":
                proofs = {
                    role: json.loads((states["a"]["output"] / "native-stopped.json").read_bytes()),
                    peer: peer_stopped(states["a"]["plan"], role),
                }
                require_epoch_b_allowed(proofs, role)
                if role == "passive":
                    report["restart_seed_calls"] = seeded_cache(
                        plan, ttnn, torch, mesh, cache, write_kv_chunk, report["cleanup_errors"]
                    )
                    publish(
                        out,
                        ep,
                        "sentinel-rewritten",
                        prior_native_stops={
                            k: sha256(Path(states["a"]["plan"]["run_dir"]) / k / "native-stopped.json")
                            for k in ("source", "passive")
                        },
                    )
                else:
                    wait(ep, "sentinel-rewritten")
                report["events"].append("sentinel-ready")
            cfg = dict(state["bridge_config"])
            env = manager_environment(ep, role, Path(ep["run_dir"]), out, ns["umd_device_ids"])
            expected_env = {
                k: v
                for k, v in env.items()
                if k.startswith("KV_MANAGER_") or k in {"KVM_ID", "PEERS", "ROLE", "LD_LIBRARY_PATH"}
            }
            write_json(out / "manager-environment.json", expected_env)
            publish(out, ep, "epoch-ready", owner=owner)
            wait(ep, "epoch-ready")
            check()
            state["manager_log"] = (out / "manager.log").open("xb")
            manager = subprocess.Popen(
                [plan["manager_binary"]], stdout=state["manager_log"], stderr=subprocess.STDOUT, env=env
            )
            state["manager"] = manager
            state["safe"] = False
            write_json(
                out / "manager-started.json",
                dict(
                    run_nonce=ep["run_nonce"],
                    role=role,
                    owner=owner,
                    endpoint=local,
                    manager=process_identity(manager.pid),
                ),
            )
            deadline = time.monotonic() + 180
            while True:
                check()
                require(manager.poll() is None, "Manager exited during startup")
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{local['health_port']}/health", timeout=2
                    ) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError):
                    pass
                require(time.monotonic() < deadline, "Manager readiness timed out")
                time.sleep(0.05)
            check_manager_ready(ep, role, 200, (out / "manager.log").read_text(errors="replace"))

            def observe(label):
                write_json(
                    out / (label + ".json"),
                    observe_manager(
                        manager,
                        os.getpid(),
                        {int(x) for x in mapping.values()},
                        plan["manager_libraries"],
                        expected_env,
                    ),
                )

            observe("ownership-before-transfer")
            publish(out, ep, "manager-ready")
            wait(ep, "manager-ready")
            if role == "source":
                from ttnn._experimental.layer_completion import LayerCompletionQueue, LayerCompletionRouter

                ring = "/llama_cancel_ring_" + ep["run_nonce"]
                ack = "/llama_cancel_acks_" + ep["run_nonce"]
                state["router"] = LayerCompletionRouter(
                    rank=0,
                    world_size=1,
                    master_rank=0,
                    ring_shm_name=ring,
                    scheduler_channel_shm_name=ack,
                    teardown_timeout_ms=30000,
                )
                state["producer"] = LayerCompletionQueue.connect(ring, connect_timeout_ms=30000)
                cfg["ack_shm"] = ack
            bridge = Bridge(plan[role + "_client"], cfg, out, env, check=check)
            state["bridge"] = bridge
            publish(
                out, ep, "processes", manager=process_identity(manager.pid), bridge=process_identity(bridge.process.pid)
            )
            wait(ep, "processes")
            report["events"].append("epoch-" + epoch + "-started")
            if role == "source":
                write_json(
                    out / "source-client-loaded-libraries.json",
                    mapped_libraries(bridge.process.pid, plan["manager_libraries"]),
                )

            def pages(label):
                # Runtime capture is invoked after its synchronize; delayed-passive
                # snapshots also synchronize before comparing all first1K pages.
                ttnn.synchronize_device(mesh)
                end = 1024 if label.startswith("delay-") else 32
                return selected_pages(table, out, "selected-" + label, 0 if role == "source" else 1, end)

            def produce(count):
                require(role == "source" and count == 32, "Only one full32 real call per epoch")
                ids = prompts[0][:1024] if epoch == "a" else prompts[1]
                request = 0 if epoch == "a" else 1

                def receive(token_ids, end):
                    host = (
                        pack_token_ids(token_ids, max_seq_len=2048, actual_start=0, actual_end=end)
                        .to(torch.uint32)
                        .reshape(4, 1, 256)
                        .contiguous()
                    )
                    service.forward_to_tensor_bytes(host.numpy(), metadata=struct.pack("<III", 0, 0, end))
                    tokens, metadata = ttnn.experimental.deepseek_prefill.inbound_socket_service_sync(
                        service, metadata_size_bytes=12
                    )
                    rows = [
                        ttnn.to_torch(x).view(torch.int32).flatten()[:3].tolist()
                        for x in ttnn.get_device_tensors(metadata)
                    ]
                    return dict(tokens=tokens, metadata=metadata, metadata_rows=rows, host=host)

                def borrowed(packet):
                    shards = ttnn.get_device_tensors(packet["tokens"])
                    require(len(shards) == 32, "Missing H2D shards")
                    for chip, shard in enumerate(shards):
                        require(
                            torch.equal(
                                ttnn.to_torch(shard).reshape(-1).to(torch.int64),
                                packet["host"][chip // 8].reshape(-1).to(torch.int64),
                            ),
                            "Borrowed H2D input changed",
                        )

                def capture(label):
                    # selected-before is the transfer oracle AFTER the current real
                    # writes, never the still-valid cache from the cancelled epoch.
                    return pages("prewrite" if label == "prewrite" else "before")

                def push(layer, request_id):
                    deadline = time.monotonic() + 30
                    check()
                    # Each epoch has a new router sequence, while the retained runtime
                    # keeps a monotonic worker request ID across both real calls.
                    while not state["producer"].try_push(
                        seq=layer, source_rank=0, layer_idx=layer, request_id=request_id
                    ):
                        check()
                        require(time.monotonic() < deadline, "Real ack backpressure")
                        time.sleep(0.001)

                row = produce_real_chunk(
                    runtime,
                    cache,
                    epoch=epoch,
                    nonce=ep["run_nonce"],
                    request_id=request,
                    ids=ids,
                    receive=receive,
                    check_borrowed=borrowed,
                    capture=capture,
                    push=push,
                )
                write_json(out / "real-runtime.json", row)
                report["real_layer_acks"] += 32
                report["runtime_h2d_tested"] = True
                return row

            send = lambda name, **fields: publish(out, ep, name, **fields)
            await_peer = lambda name: wait(ep, name)
            if epoch == "a":
                terminal = run_cancel_epoch(
                    role, bridge, send, await_peer, produce, check, tokens=prompts[0], page_receipt=pages
                )
                terminal["bridge_exit_code"] = terminal_bridge(bridge)
            else:
                old, new = identities("a"), identities("b")
                check_fresh_identities(old, new)

                def restart_identity():
                    return dict(
                        old_nonce=states["a"]["plan"]["run_nonce"],
                        new_nonce=ep["run_nonce"],
                        **{
                            f"{which}_{kind}_pids": {k: v["pid"] for k, v in data[kind].items()}
                            for which, data in [("old", old), ("new", new)]
                            for kind in ("manager", "bridge")
                        },
                    )

                terminal = run_restart_epoch(
                    role, bridge, send, await_peer, produce, pages, restart_identity, check, tokens=prompts[1]
                )
            publish(out, ep, "client-drained", terminal=terminal)
            report["events"].append("epoch-" + epoch + "-drained")
            wait(ep, "client-drained")
            observe("ownership-after-transfer")
            stop_native(state)
            report["events"].append("epoch-" + epoch + "-stopped")
            pair_stop(state)
            report["events"].append("epoch-" + epoch + "-pair-stopped")
            if epoch == "b" or role == "source":
                after = selected_pages(table, out, "selected-after-stop", 0 if role == "source" else 1)
                expected = json.loads(
                    (out / ("selected-before.json" if role == "source" else "selected-after.json")).read_bytes()
                )["groups"]
                require(after == expected, "Selected cache changed during native shutdown")
            cleanup_each(
                (
                    ("producer", lambda: state["producer"].shutdown() if state["producer"] is not None else None),
                    ("router", lambda: state["router"].stop() if state["router"] is not None else None),
                ),
                report["cleanup_errors"],
                [],
            )
            state["producer"] = state["router"] = None
            require(not report["cleanup_errors"], "Completion channel cleanup failed")
            report["epochs"][epoch] = dict(
                nonce=ep["run_nonce"],
                manager_exit=manager.returncode,
                terminal=terminal,
                processes=identities(epoch),
                native_stop_sha256=sha256(out / "native-stopped.json"),
            )
        report["ok"] = True
    except BaseException as error:
        report["errors"].append(repr(error))
        traceback.print_exc()
        write_json(output / "failure.json", dict(run_nonce=plan["run_nonce"], role=role, ok=False, error=repr(error)))
    finally:
        # Visit both epochs, including never-started ones. A peer that entered the next
        # epoch must stop too; the epoch-A proof cannot release this retained cache.
        try:
            for state in states.values():
                stop_native(state)
            for state in states.values():
                pair_stop(state)
        except BaseException as error:
            report["cleanup_errors"].append(repr(error))
            report["allocation_release_blocked"] = True
            report["recovery_handoff"] = dict(
                owner=owner,
                endpoints={k: plan[k] for k in ("source", "passive")},
                epochs={
                    k: dict(
                        plan=v["plan"],
                        manager_pid=None if v["manager"] is None else v["manager"].pid,
                        manager_exit=None if v["manager"] is None else v["manager"].poll(),
                        log=str(v["output"] / "manager.log"),
                    )
                    for k, v in states.items()
                },
                action="Root must prove native I/O stopped on BOTH endpoints across BOTH epochs; no automatic reset/kill; lease expiry is not safe retention",
            )
            try:
                write_json(output / "recovery-required.json", report)
            except BaseException:
                traceback.print_exc()
            try:
                print("ROOT_RECOVERY_REQUIRED both epochs/owners retained; no forced release", flush=True)
            except OSError:
                pass
            for state in states.values():
                await_pair_cleanup(state["output"] / "root-cleanup.json", owner, state["manager"], state["plan"], role)
            report["allocation_release_blocked"] = False
        attempts = []
        cleanup_each((("etcd.close", bootstrap.close),), report["cleanup_errors"], attempts)
        for state in states.values():
            cleanup_each(
                (
                    ("producer", lambda s=state: s["producer"].shutdown() if s["producer"] is not None else None),
                    ("router", lambda s=state: s["router"].stop() if s["router"] is not None else None),
                    ("manager.log", lambda s=state: s["manager_log"].close() if s["manager_log"] is not None else None),
                ),
                report["cleanup_errors"],
                attempts,
            )
        if report["ok"]:
            report["events"].append("cache-release")
            cleanup_each(
                (("event order", lambda: check_role_events(report["events"])),), report["cleanup_errors"], attempts
            )
        owned = dict(
            service=service,
            producer=None,
            router=None,
            saved=None,
            model=None if runtime is None else runtime.model,
            mesh=mesh,
            **{"cache.k": None if cache is None else cache.k, "cache.v": None if cache is None else cache.v},
        )
        service = runtime = cache = mesh = None
        attempts += cleanup_owner(owned, api=ttnn, collect=gc.collect, errors=report["cleanup_errors"])
        report["cleanup_attempts"] = attempts
        report["owner_cleanup_complete"] = not report["cleanup_errors"]
        elfs = sorted(Path(report["jit_cache"]["path"]).rglob("*.elf"))
        write_json(
            output / "jit-cache-after.json",
            dict(files={str(x): sha256(x) for x in elfs}, count=len(elfs), bytes=sum(x.stat().st_size for x in elfs)),
        )
        report["ok"] = report["ok"] and not report["errors"] and not report["cleanup_errors"]
        write_json(output / "result.json", report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
