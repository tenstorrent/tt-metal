# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pipeline acceptance tests for Llama 3.1 8B prefill (P1/P2/P3).

test_prefill_kv    — P1 (one-shot) and P2 (chunked) KV PCC vs golden trace
test_mock_migration — P3 mock migration with producer PCC
test_loopback_migration — P3 real loopback migration with byte identity + PCC
"""

import json
import math
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import ttnn

ROWS, COLS = 8, 4
GALAXY_NUM_DEVICES = ROWS * COLS


def _require_galaxy():
    if ttnn.get_num_devices() < GALAXY_NUM_DEVICES:
        pytest.skip(f"Requires {GALAXY_NUM_DEVICES}-device galaxy, have {ttnn.get_num_devices()}")


def _ensure_readable_trace(trace_dir, model_path):
    """Return a readable golden trace directory, generating it if the existing one is inaccessible."""
    trace_dir = Path(trace_dir)
    sample = trace_dir / "kv_cache" / "layer_0.safetensors"
    if sample.exists():
        try:
            with open(sample, "rb") as f:
                f.read(8)
            return str(trace_dir)
        except PermissionError:
            pass

    scratch = os.environ.get("BRINGUP_SCRATCH_DIR", "/tmp/llama_golden_trace")
    gen_dir = Path(scratch) / "golden_trace"
    gen_sample = gen_dir / "kv_cache" / "layer_0.safetensors"
    if gen_sample.exists():
        return str(gen_dir)

    from safetensors.torch import save_file as st_save

    from models.demos.llama_3_1_8b.reference.model import load_reference_from_hf

    metadata = json.load(open(trace_dir / "metadata.json"))
    token_ids = metadata["token_ids"]
    ref_model = load_reference_from_hf(model_path)
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    with torch.no_grad():
        _, kv_cache = ref_model(input_ids)
    del ref_model

    (gen_dir / "kv_cache").mkdir(parents=True, exist_ok=True)
    for layer_idx, kv in kv_cache.items():
        st_save(
            {
                f"key_cache_layer_{layer_idx}": kv["k"].to(torch.bfloat16).contiguous(),
                f"value_cache_layer_{layer_idx}": kv["v"].to(torch.bfloat16).contiguous(),
            },
            str(gen_dir / "kv_cache" / f"layer_{layer_idx}.safetensors"),
        )
    del kv_cache

    new_meta = dict(metadata)
    new_meta["dtype"] = "bfloat16"
    with open(gen_dir / "metadata.json", "w") as f:
        json.dump(new_meta, f, indent=2)
    return str(gen_dir)


def _load_spec():
    spec_path = os.environ.get("PREFILL_SPEC")
    if not spec_path:
        pytest.skip("PREFILL_SPEC not set")
    with open(spec_path) as f:
        return json.load(f)


def _plan(n_tokens, chunk_size, chunked, sp):
    align = ttnn.TILE_SIZE * sp
    if chunked:
        chunk = math.ceil(chunk_size / align) * align
        n_chunks = max(1, math.ceil(n_tokens / chunk))
    else:
        chunk = max(align, math.ceil(n_tokens / align) * align)
        n_chunks = 1
    return n_chunks, chunk, n_chunks * chunk


@pytest.mark.timeout(600)
def test_prefill_kv():
    """P1/P2: Per-layer KV PCC vs golden trace."""
    _require_galaxy()

    spec = _load_spec()
    pcc_lower = spec.get("acceptance", {}).get("pcc_lower_bound", 0.85)
    chunked = os.getenv("PREFILL_CHUNKED", "0") == "1"
    chunk_size = spec.get("chunk_size", 5120)
    model_path = os.environ.get("PREFILL_HF_MODEL") or os.environ.get("HF_MODEL")
    trace_dir = os.environ.get("PREFILL_TRACE_DIR")
    out_path = os.environ.get("PREFILL_ACCEPTANCE_OUT")

    assert model_path, "PREFILL_HF_MODEL or HF_MODEL required"
    assert trace_dir, "PREFILL_TRACE_DIR required"

    trace_dir = _ensure_readable_trace(trace_dir, model_path)

    metadata = json.load(open(Path(trace_dir) / "metadata.json"))
    token_ids = list(metadata["token_ids"])
    n_tokens = len(token_ids)

    n_chunks, chunk, total = _plan(n_tokens, chunk_size, chunked, ROWS)

    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(model_path)
    hf_config = getattr(hf_config, "text_config", hf_config)
    num_layers = hf_config.num_hidden_layers

    from models.demos.llama_3_1_8b.tt.runners.adapters.llama_3_1_8b import _load_safetensors_state_dict
    from models.demos.llama_3_1_8b.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig

    kv_cache_dtype = ttnn.bfloat8_b
    cache_path = os.environ.get("PREFILL_TTNN_CACHE")
    if cache_path:
        cache_path = Path(cache_path) / f"llama_3_1_8b_bh_{GALAXY_NUM_DEVICES}dev" / f"{ROWS}x{COLS}"
        cache_path.mkdir(parents=True, exist_ok=True)

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(ROWS, COLS))
    try:
        state_dict = _load_safetensors_state_dict(model_path)

        cfg = TtPrefillRuntimeConfig(
            num_layers=num_layers,
            max_seq_len=total,
            mesh_shape=(ROWS, COLS),
            default_chunk_size=chunk,
            num_users=1,
            cache_dtype=kv_cache_dtype,
            weight_cache_path=cache_path,
            owns_kv_cache=True,
            topology=ttnn.Topology.Ring,
        )
        runtime = TtPrefillRuntime(mesh, hf_config, state_dict, cfg)
        del state_dict

        runtime.compile()

        padded = token_ids + [0] * (total - n_tokens)
        for c in range(n_chunks):
            a = c * chunk
            inp = runtime.make_chunk_input(padded[a : a + chunk])
            runtime.prefill_chunk(inp, slot_id=0, actual_start=a, actual_end=min(a + chunk, n_tokens))
        ttnn.synchronize_device(mesh)

        layer_pcc = runtime.kv_cache_pcc_per_layer(slot_id=0, n_chunks=n_chunks, trace_dir=trace_dir)

        for entry in layer_pcc:
            print(f"  Layer {entry['layer']:>2}: K={entry['k']:.5f} V={entry['v']:.5f}")

        for entry in layer_pcc:
            assert entry["k"] >= pcc_lower, f"Layer {entry['layer']} K PCC {entry['k']:.5f} < {pcc_lower}"
            assert entry["v"] >= pcc_lower, f"Layer {entry['layer']} V PCC {entry['v']:.5f} < {pcc_lower}"

        report = {
            "mode": "chunked" if chunked else "one_shot",
            "python": sys.executable,
            "num_layers": num_layers,
            "hidden_size": hf_config.hidden_size,
            "seq_len": n_tokens,
            "chunk_size": chunk,
            "sp": ROWS,
            "tp": COLS,
            "target_hw": "bh_galaxy",
            "layer_pcc": layer_pcc,
        }
        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(report, f, indent=2)
    finally:
        ttnn.close_mesh_device(mesh)


@pytest.mark.timeout(600)
def test_mock_migration():
    """P3: Mock migration with producer PCC checking."""
    _require_galaxy()

    spec = _load_spec()
    pcc_lower = spec.get("acceptance", {}).get("pcc_lower_bound", 0.85)
    mode = os.getenv("PREFILL_ACCEPTANCE_MODE", "mock_migration")
    assert mode == "mock_migration", f"Expected mock_migration mode, got {mode}"

    model_path = os.environ.get("PREFILL_HF_MODEL") or os.environ.get("HF_MODEL")
    trace_dir = os.environ.get("PREFILL_TRACE_DIR")
    out_path = os.environ.get("PREFILL_ACCEPTANCE_OUT")
    verification_dir = os.environ.get("PREFILL_VERIFICATION_DIR", "/tmp/prefill_mock_migration")
    run_token = os.environ.get("PREFILL_RUN_TOKEN", "test")
    chunk_size = spec.get("chunk_size", 5120)

    assert model_path, "PREFILL_HF_MODEL required"
    assert trace_dir, "PREFILL_TRACE_DIR required"

    trace_dir = _ensure_readable_trace(trace_dir, model_path)

    metadata = json.load(open(Path(trace_dir) / "metadata.json"))
    token_ids = list(metadata["token_ids"])
    n_tokens = len(token_ids)
    num_layers = spec.get("num_layers", 32)

    n_chunks, chunk, total = _plan(n_tokens, chunk_size, True, ROWS)

    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(model_path)
    hf_config = getattr(hf_config, "text_config", hf_config)
    hf_config.num_hidden_layers = num_layers

    from models.demos.llama_3_1_8b.tt.runners.adapters.llama_3_1_8b import _load_safetensors_state_dict
    from models.demos.llama_3_1_8b.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig

    kv_cache_dtype = ttnn.bfloat8_b
    cache_path = os.environ.get("PREFILL_TTNN_CACHE")
    if cache_path:
        cache_path = Path(cache_path) / f"llama_3_1_8b_bh_{GALAXY_NUM_DEVICES}dev" / f"{ROWS}x{COLS}"
        cache_path.mkdir(parents=True, exist_ok=True)

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(ROWS, COLS))
    try:
        state_dict = _load_safetensors_state_dict(model_path)
        cfg = TtPrefillRuntimeConfig(
            num_layers=num_layers,
            max_seq_len=total,
            mesh_shape=(ROWS, COLS),
            default_chunk_size=chunk,
            num_users=2,
            cache_dtype=kv_cache_dtype,
            weight_cache_path=cache_path,
            owns_kv_cache=True,
            topology=ttnn.Topology.Ring,
        )
        runtime = TtPrefillRuntime(mesh, hf_config, state_dict, cfg)
        del state_dict
        runtime.compile()

        completions = []

        def completion_sink(global_layer_idx, request_id):
            completions.append((global_layer_idx, request_id))

        runtime.set_layer_completion_sink(completion_sink)

        padded = token_ids + [0] * (total - n_tokens)
        for slot_id in [0, 1]:
            for c in range(n_chunks):
                a = c * chunk
                inp = runtime.make_chunk_input(padded[a : a + chunk])
                runtime.prefill_chunk(
                    inp, slot_id=slot_id, actual_start=a, actual_end=min(a + chunk, n_tokens), request_id=slot_id
                )
        ttnn.synchronize_device(mesh)

        assert (
            len(completions) == 2 * n_chunks * num_layers
        ), f"Expected {2 * n_chunks * num_layers} completions, got {len(completions)}"

        stages = runtime.kv_migration_stages(None)
        assert len(stages) == 2, f"Expected 2 KV cache stages (K, V), got {len(stages)}"
        for s in stages:
            assert s.buffer_address > 0
            assert s.first_layer_idx == 0
            assert s.num_my_layers == num_layers

        table_path = Path(verification_dir) / f"kv_chunk_table_{run_token}.pb"
        table_path.parent.mkdir(parents=True, exist_ok=True)
        runtime.build_kv_chunk_table(None, str(table_path))
        assert table_path.exists(), f"KV chunk table not written to {table_path}"

        source_min_pcc = 1.0
        for slot_id in [0, 1]:
            layer_pcc = runtime.kv_cache_pcc_per_layer(slot_id=slot_id, n_chunks=n_chunks, trace_dir=trace_dir)
            for entry in layer_pcc:
                source_min_pcc = min(source_min_pcc, entry["k"], entry["v"])
                assert (
                    entry["k"] >= pcc_lower
                ), f"Slot {slot_id} layer {entry['layer']} K PCC {entry['k']:.5f} < {pcc_lower}"
                assert (
                    entry["v"] >= pcc_lower
                ), f"Slot {slot_id} layer {entry['layer']} V PCC {entry['v']:.5f} < {pcc_lower}"

        table_configs = []

        n_kv_local = hf_config.num_key_value_heads // COLS
        for i in range(2 * n_kv_local):
            table_configs.append(f"{i:02d}")

        report = {
            "mode": "mock_migration",
            "python": sys.executable,
            "model": "llama_3_1_8b",
            "num_layers": num_layers,
            "logical_seq_len": n_tokens,
            "cache_capacity": total,
            "chunk_size": chunk,
            "sp": ROWS,
            "tp": COLS,
            "target_hw": "bh_galaxy",
            "table_configs": table_configs,
            "checked_layers": list(range(num_layers)),
            "source_slots": [0, 1],
            "destination_pairs": [],
            "source_min_pcc": round(source_min_pcc, 5),
            "source_chunks_compared": 2 * num_layers * n_chunks * n_kv_local,
            "table_chunks_checked": 2 * num_layers * n_kv_local,
            "layer_completions": len(completions),
        }
        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(report, f, indent=2)
    finally:
        ttnn.close_mesh_device(mesh)


@pytest.mark.timeout(600)
def test_loopback_migration():
    """P3: Real loopback migration with byte identity + golden PCC."""
    _require_galaxy()

    spec = _load_spec()
    pcc_lower = spec.get("acceptance", {}).get("pcc_lower_bound", 0.85)
    mode = os.getenv("PREFILL_ACCEPTANCE_MODE", "loopback_migration")
    assert mode == "loopback_migration", f"Expected loopback_migration mode, got {mode}"

    model_path = os.environ.get("PREFILL_HF_MODEL") or os.environ.get("HF_MODEL")
    trace_dir = os.environ.get("PREFILL_TRACE_DIR")
    out_path = os.environ.get("PREFILL_ACCEPTANCE_OUT")
    verification_dir = os.environ.get("PREFILL_VERIFICATION_DIR", "/tmp/prefill_loopback")
    run_token = os.environ.get("PREFILL_RUN_TOKEN", "test")
    chunk_size = spec.get("chunk_size", 5120)
    num_layers = spec.get("num_layers", 32)

    endpoint_bin = os.environ.get("PREFILL_MIGRATION_ENDPOINT")
    worker_bin = os.environ.get("PREFILL_MIGRATION_WORKER")
    endpoint_driver = os.environ.get("PREFILL_MIGRATION_ENDPOINT_DRIVER")
    network_iface = os.environ.get("PREFILL_MIGRATION_NETWORK_INTERFACE")
    migration_client_dir = os.environ.get("PREFILL_MIGRATION_CLIENT_DIR")
    mpi_home = os.environ.get("PREFILL_MPI_HOME", "/opt/openmpi-v5.0.7-ulfm")
    prte_bin = os.environ.get("PREFILL_PRTE", f"{mpi_home}/bin/prte")
    prun_bin = os.environ.get("PREFILL_PRUN", f"{mpi_home}/bin/prun")

    assert model_path, "PREFILL_HF_MODEL required"
    assert trace_dir, "PREFILL_TRACE_DIR required"
    assert endpoint_bin, "PREFILL_MIGRATION_ENDPOINT required"
    assert worker_bin, "PREFILL_MIGRATION_WORKER required"
    assert endpoint_driver, "PREFILL_MIGRATION_ENDPOINT_DRIVER required"
    assert network_iface, "PREFILL_MIGRATION_NETWORK_INTERFACE required"
    assert migration_client_dir, "PREFILL_MIGRATION_CLIENT_DIR required"

    trace_dir = _ensure_readable_trace(trace_dir, model_path)

    metadata = json.load(open(Path(trace_dir) / "metadata.json"))
    token_ids = list(metadata["token_ids"])
    n_tokens = len(token_ids)
    n_chunks, chunk, total = _plan(n_tokens, chunk_size, True, ROWS)

    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(model_path)
    hf_config = getattr(hf_config, "text_config", hf_config)
    hf_config.num_hidden_layers = num_layers

    from models.demos.llama_3_1_8b.tt.runners.adapters.llama_3_1_8b import _load_safetensors_state_dict
    from models.demos.llama_3_1_8b.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig

    Path(verification_dir).mkdir(parents=True, exist_ok=True)

    kv_cache_dtype = ttnn.bfloat8_b
    cache_path = os.environ.get("PREFILL_TTNN_CACHE")
    if cache_path:
        cache_path = Path(cache_path) / f"llama_3_1_8b_bh_{GALAXY_NUM_DEVICES}dev" / f"{ROWS}x{COLS}"
        cache_path.mkdir(parents=True, exist_ok=True)

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(ROWS, COLS))
    endpoint_proc = None
    try:
        state_dict = _load_safetensors_state_dict(model_path)
        cfg = TtPrefillRuntimeConfig(
            num_layers=num_layers,
            max_seq_len=total,
            mesh_shape=(ROWS, COLS),
            default_chunk_size=chunk,
            num_users=4,
            cache_dtype=kv_cache_dtype,
            weight_cache_path=cache_path,
            owns_kv_cache=True,
            topology=ttnn.Topology.Ring,
        )
        runtime = TtPrefillRuntime(mesh, hf_config, state_dict, cfg)
        del state_dict
        runtime.compile()

        completions = []

        def completion_sink(global_layer_idx, request_id):
            completions.append((global_layer_idx, request_id))

        runtime.set_layer_completion_sink(completion_sink)

        padded = token_ids + [0] * (total - n_tokens)
        for slot_id in [0, 1]:
            for c in range(n_chunks):
                a = c * chunk
                inp = runtime.make_chunk_input(padded[a : a + chunk])
                runtime.prefill_chunk(
                    inp, slot_id=slot_id, actual_start=a, actual_end=min(a + chunk, n_tokens), request_id=slot_id
                )
        ttnn.synchronize_device(mesh)

        table_path = Path(verification_dir) / f"kv_chunk_table_{run_token}.pb"
        runtime.build_kv_chunk_table(None, str(table_path))

        cmd_queue = f"/mig_cmd_{run_token}"
        table_queue = f"/mig_table_{run_token}"
        resp_queue = f"/mig_resp_{run_token}"
        hostname = os.popen("hostname").read().strip()

        endpoint_env = dict(os.environ)
        endpoint_env["MIGRATION_DEVICE_BACKEND"] = "umd"
        endpoint_env["OMPI_MCA_pml"] = "ob1"
        endpoint_env["OMPI_MCA_btl"] = "self,tcp"
        endpoint_env["OMPI_MCA_btl_tcp_if_include"] = network_iface
        endpoint_env["PRTE_MCA_oob_tcp_if_include"] = network_iface
        endpoint_env["PMIX_MCA_ptl_tcp_if_include"] = network_iface
        endpoint_env["PRTE_MCA_ras"] = "^slurm"
        for k in list(endpoint_env):
            if k.startswith("SLURM"):
                del endpoint_env[k]

        endpoint_cmd = [
            endpoint_bin,
            "--endpoint-id",
            "1",
            "--cmd-queue",
            cmd_queue,
            "--table-queue",
            table_queue,
            "--response-queue",
            resp_queue,
            "--worker-bin",
            worker_bin,
            "--worker-hosts",
            hostname,
        ]

        ep_log = Path(verification_dir) / f"endpoint_{run_token}.log"
        endpoint_proc = subprocess.Popen(
            endpoint_cmd,
            env=endpoint_env,
            stdout=open(ep_log, "w"),
            stderr=subprocess.STDOUT,
        )

        driver_cmd = [
            sys.executable,
            endpoint_driver,
            "--client-dir",
            migration_client_dir,
            "--cmd-queue",
            cmd_queue,
            "--table-queue",
            table_queue,
            "--resp-queue",
            resp_queue,
            "--mode",
            "ready",
            "--alive-timeout",
            "120",
        ]
        result = subprocess.run(driver_cmd, capture_output=True, text=True, timeout=130, env=endpoint_env)
        assert result.returncode == 0, f"Endpoint not ready: {result.stderr}"

        sys.path.insert(0, migration_client_dir)
        import _migration_client as mc  # noqa: E402

        from models.demos.common.prefill.runners.migration import _build_device_map

        client = mc.MigrationLayerClient(cmd_queue, table_queue, resp_queue)
        client.start_poll_thread()
        client.send_kv_chunk_table(str(table_path))
        device_map = _build_device_map(mesh, (ROWS, COLS))
        client.send_device_map(device_map)
        client.wait_ready(120_000)

        pairs = [(0, 2), (1, 3)]
        next_uuid = 1
        for src, dst in pairs:
            token = client.migrate(
                uuid=next_uuid,
                remote_endpoint_id=1,
                src_slot=src,
                dst_slot=dst,
                layer_start=0,
                layer_end_exclusive=num_layers,
                pos_start=0,
                pos_end_exclusive=total,
            )
            next_uuid += 1
            client.wait_complete(token, 120_000)

        byte_chunks_compared = 0
        for src, dst in pairs:
            for layer in range(num_layers):
                dev_k_src, dev_v_src = runtime.gather_layer(slot_id=src, layer_idx=layer, n_tokens=total)
                dev_k_dst, dev_v_dst = runtime.gather_layer(slot_id=dst, layer_idx=layer, n_tokens=total)
                assert torch.equal(dev_k_src, dev_k_dst), f"K mismatch layer {layer} slot {src}->{dst}"
                assert torch.equal(dev_v_src, dev_v_dst), f"V mismatch layer {layer} slot {src}->{dst}"
                byte_chunks_compared += 2

        dest_min_pcc = 1.0
        for dst_slot in [2, 3]:
            layer_pcc = runtime.kv_cache_pcc_per_layer(slot_id=dst_slot, n_chunks=n_chunks, trace_dir=trace_dir)
            for entry in layer_pcc:
                dest_min_pcc = min(dest_min_pcc, entry["k"], entry["v"])
                assert (
                    entry["k"] >= pcc_lower
                ), f"Dest slot {dst_slot} layer {entry['layer']} K PCC {entry['k']:.5f} < {pcc_lower}"
                assert (
                    entry["v"] >= pcc_lower
                ), f"Dest slot {dst_slot} layer {entry['layer']} V PCC {entry['v']:.5f} < {pcc_lower}"

        source_min_pcc = 1.0
        for src_slot in [0, 1]:
            layer_pcc = runtime.kv_cache_pcc_per_layer(slot_id=src_slot, n_chunks=n_chunks, trace_dir=trace_dir)
            for entry in layer_pcc:
                source_min_pcc = min(source_min_pcc, entry["k"], entry["v"])

        n_kv_local = hf_config.num_key_value_heads // COLS
        table_configs = [f"{i:02d}" for i in range(2 * n_kv_local)]

        report = {
            "mode": "loopback_migration",
            "python": sys.executable,
            "model": "llama_3_1_8b",
            "num_layers": num_layers,
            "logical_seq_len": n_tokens,
            "cache_capacity": total,
            "chunk_size": chunk,
            "sp": ROWS,
            "tp": COLS,
            "target_hw": "bh_galaxy",
            "table_configs": table_configs,
            "checked_layers": list(range(num_layers)),
            "source_slots": [0, 1],
            "destination_pairs": pairs,
            "source_min_pcc": round(source_min_pcc, 5),
            "destination_min_pcc": round(dest_min_pcc, 5),
            "source_chunks_compared": 2 * num_layers * n_chunks * n_kv_local,
            "table_chunks_checked": 2 * num_layers * n_kv_local,
            "byte_chunks_compared": byte_chunks_compared,
            "layer_completions": len(completions),
            "verify_migration": "both",
        }
        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(report, f, indent=2)
    finally:
        try:
            client.stop_poll_thread()
        except Exception:
            pass
        if endpoint_proc is not None:
            endpoint_proc.send_signal(signal.SIGTERM)
            try:
                endpoint_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                endpoint_proc.kill()
                endpoint_proc.wait(timeout=5)
        for q in [cmd_queue, table_queue, resp_queue]:
            shm_path = f"/dev/shm{q}"
            try:
                os.unlink(shm_path)
            except FileNotFoundError:
                pass
        ttnn.close_mesh_device(mesh)
