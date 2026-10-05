"""Opt-in host-only serving ledger. No device operations or per-token file I/O.

The optional runner bridge supplies immutable submission IDs and reports output
completion after the existing event wait, host sampling, and output formatting.
Snapshots use a host thread and do not drain the serving queue.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(path)


def accurate_attention_groups(batch, limit):
    """Mirror the decoder's host-known grouping; no tensor/device access."""
    if not 1 <= batch <= 32 or limit not in (1, 2, 4, 8, 16, 32):
        raise ValueError("Unsupported accurate-attention batch/group size")
    groups = []
    start = 0
    while start < batch:
        count = 1 << (min(limit, batch - start).bit_length() - 1)
        groups.append({"row_start": start, "row_end": start + count, "batch": count})
        start += count
    return groups


def accurate_attention_geometry(groups, transport):
    geometry = []
    for group in groups:
        packed = transport == "packed_gqa_rows_v1" and group["batch"] > 1
        geometry.append(
            {
                **group,
                "logical_q_heads": 8,
                "physical_q_heads": 2 if packed else 8,
                "physical_q_rows": 32,
                "useful_rows_per_head": 4 if packed else 1,
                "query_buffer_factor": 1 if packed else 2 if group["batch"] > 8 else 1,
                "offset_mode": "raw_clamped" if packed else "aligned32_clamped",
                "mask_mode": "constant_raw_position" if packed else "causal_rows",
            }
        )
    return geometry


def _accurate_attention_identity(model):
    """Pin the already-loaded private binding and its FP32 compute kernel."""
    from .accurate_attention import _load

    binding, kernel_path = _load()  # Already cached by every decoder's construction.
    root = Path(__file__).resolve().parent / "accurate_attention"
    provenance_path = root / ".build/provenance.json"
    provenance = json.loads(provenance_path.read_text())
    limits = {int(getattr(layer, "accurate_attention_batch_size", 1)) for layer in model.layers}
    if len(limits) != 1 or next(iter(limits)) not in (1, 2, 4, 8, 16, 32):
        raise ValueError("All layers must expose one supported accurate-attention grouping")
    transports = {getattr(layer, "accurate_attention_transport", "per_row_v1") for layer in model.layers}
    if len(transports) != 1 or next(iter(transports)) not in ("per_row_v1", "grouped_rows_v1", "packed_gqa_rows_v1"):
        raise ValueError("All layers must expose one supported accurate-attention transport")
    grouped = next(iter(transports)) == "grouped_rows_v1"
    packed = next(iter(transports)) == "packed_gqa_rows_v1"
    library = Path(binding.__file__).resolve()
    if _hash(library) != provenance["library_sha256"]:
        raise RuntimeError("Loaded accurate-attention binding provenance changed")
    identity = {
        "accounting_schema_version": 4 if packed else 3 if grouped else 2,
        "transport": next(iter(transports)),
        "max_group_size": next(iter(limits)),
        "grouping": "largest_power_of_two_consecutive_rows",
        "query_concat": "absent" if grouped or packed else "materialized_dram_for_group_gt_1",
        "output_row_slice": (
            "packed_first4_rows" if packed else "absent" if grouped else "materialized_dram_for_group_gt_1"
        ),
        "query_buffer_factor": "1" if packed else "2_if_group_gt8_else1",
        "math_fidelity": "HiFi4",
        "fp32_denominator": True,
        "fp32_output_accumulator": True,
        "grid": [8, 8],
        "q_chunk_size": 32,
        "k_chunk_size": 128,
        "offset_reads_per_core": 2,
        "offset_read_bytes": 4,
        "library": {"path": str(library), "sha256": _hash(library)},
        "provenance": {"path": str(provenance_path), "sha256": _hash(provenance_path), "data": provenance},
        "compute_kernel": {"path": str(kernel_path), "sha256": _hash(kernel_path)},
        "wrapper": {"path": str(root / "__init__.py"), "sha256": _hash(root / "__init__.py")},
    }
    if packed:
        mask = provenance["packed_gqa_mask"]
        writer = Path(mask["writer_path"])
        if (
            mask["contract"] != "same_raw_position_for_all_query_rows_v1"
            or mask["required_position_residues"] != list(range(32))
            or _hash(writer) != mask["writer_sha256"]
        ):
            raise RuntimeError("Loaded packed GQA writer/mask provenance differs")
        identity.update(
            logical_q_heads_per_rank=8,
            physical_q_heads_per_packed_group=2,
            logical_kv_heads_per_rank=2,
            query_pack_factor=4,
            valid_packed_rows=4,
            single_row_group="legacy_q8_aligned_offset",
            packed_mask=mask,
            packed_writer={"path": str(writer), "sha256": _hash(writer)},
        )
    return identity


def runtime_identity(adapter):
    """Read immutable Python/allocation descriptors once, at model startup."""
    model = adapter.generator.model
    mesh = adapter.mesh
    modules = {}
    for name, module in list(sys.modules.items()):
        if name.startswith("models.demos.k2_horizon_7b_qb2.tt.") or name in (
            "vllm_tt_plugin.model_runner",
            "vllm_tt_plugin.async_decode",
        ):
            source = getattr(module, "__file__", None)
            if source and Path(source).is_file():
                modules[name] = {"path": str(Path(source).resolve()), "sha256": _hash(source)}
    selected = Path(__file__).resolve().parents[1] / "doc/datatype_sweep/selected_precision_config.json"
    grid = mesh.compute_with_storage_grid_size()
    dram_grid = mesh.dram_grid_size()
    from .model import HF_MODEL, HF_REVISION

    commits = {}
    for name in (adapter.__class__.__module__, "vllm_tt_plugin.model_runner"):
        source = modules.get(name, {}).get("path")
        if source:
            commits[name] = subprocess.check_output(
                ["git", "-C", str(Path(source).parent), "rev-parse", "HEAD"], text=True
            ).strip()

    allocations = []
    policies = []
    for index, layer in enumerate(model.layers):
        policies.append({"layer": index, "policy": dataclasses.asdict(layer.policy)})
        for role, variant, shape, dtype in layer.weight_allocations:
            allocations.append(
                {"layer": index, "role": role, "variant": variant, "per_rank_shape": list(shape), "dtype": dtype}
            )
    head = []
    for index, weight in enumerate(model.head_decode.output_weights):
        # Mesh tensor.shape is the local per-rank execution shape.
        head.append(
            {
                "part": index,
                "per_rank_shape": list(weight.shape),
                "padded_shape": list(weight.padded_shape),
                "dtype": str(weight.dtype),
            }
        )
    return {
        "schema_version": 1,
        "kind": "identity",
        "server_instance": uuid.uuid4().hex,
        "clock": "perf_counter_ns",
        "started_ns": time.perf_counter_ns(),
        "pid": os.getpid(),
        "max_num_seqs": adapter.max_batch_size,
        "layer_count": model.num_layers,
        "model": HF_MODEL,
        "model_revision": HF_REVISION,
        "tokenizer_revision": HF_REVISION,
        "imported_class": adapter.__class__.__module__ + "." + adapter.__class__.__qualname__,
        "source_commits": commits,
        "max_model_len": model.context,
        "module_files": modules,
        "mesh_shape": list(mesh.shape),
        "chip_ids": list(mesh.get_device_ids()),
        "grid": {"x": grid.x, "y": grid.y},
        "dram_grid": {"x": dram_grid.x, "y": dram_grid.y},
        "arch": str(mesh.arch()),
        "precision_config": model.precision_config,
        "precision_sha256": _hash(selected),
        "effective_layer_policies": policies,
        "weight_allocations": allocations,
        "head_allocations": head,
        "head_split_sizes": list(model.head_decode.config.output_split_sizes),
        "dimensions": {
            key: int(getattr(model.config, key))
            for key in (
                "hidden_size",
                "intermediate_size",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "vocab_size",
                "num_hidden_layers",
            )
        },
        "padded_vocab": model.padded_vocab,
        "tp": 4,
        "dp": 1,
        "allow_host_sampling": adapter.allow_host_sampling,
        "force_host_sampling": adapter.force_host_sampling,
        "accurate_attention": _accurate_attention_identity(model),
        "host_sampler": (
            type(adapter.host_sampler).__module__ + "." + type(adapter.host_sampler).__qualname__
            if getattr(adapter, "host_sampler", None) is not None
            else "upstream"
        ),
        "host_sampler_workers": (
            adapter.host_sampler.topk_topp_sampler.max_workers
            if getattr(adapter, "host_sampler", None) is not None
            else None
        ),
        "host_sampler_setup": (
            adapter.host_sampler.stats_snapshot() if getattr(adapter, "host_sampler", None) is not None else None
        ),
        "cpu_threads": {
            "torch_intraop": sys.modules["torch"].get_num_threads(),
            "torch_interop": sys.modules["torch"].get_num_interop_threads(),
            "omp_environment": os.environ.get("OMP_NUM_THREADS"),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
        },
    }


class PhaseRecorder:
    """A bounded in-memory ledger; overflow/pending submissions invalidate evidence."""

    def __init__(self, path, identity, *, limit=500000, host_sampler_stats=None):
        self.path = str(Path(path).resolve())
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.socket_path = self.path + ".sock"
        # Linux AF_UNIX paths have a 108-byte limit. Keep discovery in identity.
        if len(os.fsencode(self.socket_path)) >= 104:
            self.socket_path = "/tmp/k2-phase-" + hashlib.sha256(self.path.encode()).hexdigest()[:24] + ".sock"
        self.identity = dict(identity, phase_path=self.path, snapshot_socket=self.socket_path)
        self.host_sampler_stats = host_sampler_stats
        self.limit = limit
        self.lock = threading.Lock()
        self.local = threading.local()
        self.pending = {}
        self.records = []
        self.errors = []
        self.next_id = 1
        self.positions = None
        self.decode_request_ids = None
        self.layout_epoch = 0
        self.closed = False
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        if Path(self.socket_path).exists():
            raise RuntimeError(f"Benchmark snapshot socket already exists: {self.socket_path}")
        self.server.bind(self.socket_path)
        os.chmod(self.socket_path, 0o600)
        self.server.listen(2)
        self.server.settimeout(1)
        self.thread = threading.Thread(target=self._serve, name="k2-benchmark-snapshot", daemon=True)
        self.thread.start()
        self.snapshot()

    @classmethod
    def from_adapter(cls, adapter):
        path = os.environ.get("K2_BENCHMARK_PHASE_PATH")
        identity_path = os.environ.get("K2_BENCHMARK_IDENTITY_PATH")
        if not path and not identity_path:
            return None
        identity = runtime_identity(adapter)
        sampler = getattr(adapter, "host_sampler", None)
        recorder = cls(path, identity, host_sampler_stats=getattr(sampler, "stats_snapshot", None)) if path else None
        if identity_path:
            _atomic_json(identity_path, recorder.identity if recorder else identity)
        return recorder

    def entry(self):
        return time.perf_counter_ns()

    @property
    def current_id(self):
        return getattr(self.local, "step_id", None)

    def begin(self, *, entry_ns, request_ids, prompt_lengths, positions, chunk_ends, sampling_mode):
        """Called after normal runner input preparation; all arguments are host data."""
        positions = list(map(int, positions))
        ends = None if chunk_ends is None else list(map(int, chunk_ends))
        with self.lock:
            step_id = self.next_id
            self.next_id += 1
            self.pending[step_id] = {
                "schema_version": 1,
                "kind": "step",
                "step_id": step_id,
                "phase": "decode" if ends is None else "prefill",
                "entry_ns": int(entry_ns),
                "prepared_ns": time.perf_counter_ns(),
                "host_thread": threading.get_ident(),
                "request_ids": list(request_ids),
                "prompt_lengths": list(map(int, prompt_lengths)),
                "host_positions": positions,
                "positions": positions,
                "chunk_starts": positions if ends is not None else None,
                "chunk_ends": ends,
                "terminal_prefill": (
                    None if ends is None else [end >= int(length) for end, length in zip(ends, prompt_lengths)]
                ),
                "sampling_mode": sampling_mode,
                "wire_slots": self.identity["max_num_seqs"],
            }
        self.local.step_id = step_id
        return step_id

    def annotate(self, **fields):
        step_id = self.current_id
        if step_id is not None:
            with self.lock:
                self.pending[step_id].update(fields)

    def decode(self, *, positions, reload_inputs, generated_batch, table_capacity):
        with self.lock:
            current = self.pending.get(self.current_id)
            request_ids = list(current["request_ids"]) if current is not None else None
        if reload_inputs:
            self.positions = list(map(int, positions[:generated_batch]))
            self.decode_request_ids = request_ids
            self.layout_epoch += 1
        elif request_ids != self.decode_request_ids:
            with self.lock:
                self.errors.append("Decode request mapping changed without authoritative input reload")
        if self.positions is None or len(self.positions) != generated_batch:
            with self.lock:
                self.errors.append("Missing or inconsistent effective decode-position ledger")
            return
        effective = list(self.positions)
        threshold = 4096 * min(16, max(1, (64 // generated_batch) // 2))
        accurate = table_capacity > threshold
        limit = int(self.identity.get("accurate_attention", {}).get("max_group_size", 1))
        groups = accurate_attention_groups(generated_batch, limit) if accurate else []
        grouped = accurate and any(group["batch"] > 1 for group in groups)
        transport = self.identity.get("accurate_attention", {}).get("transport", "per_row_v1")
        geometry = accurate_attention_geometry(groups, transport)
        packed = grouped and transport == "packed_gqa_rows_v1"
        self.annotate(
            positions=effective,
            active_mask=[p >= 0 for p in effective],
            generated_batch=generated_batch,
            table_capacity=table_capacity,
            attention_branch=(
                "accurate_attention_packed_gqa"
                if packed
                else (
                    "accurate_attention_batched"
                    if grouped
                    else "accurate_attention_fallback"
                    if accurate
                    else "stock_paged_decode"
                )
            ),
            attention_groups=groups,
            attention_transport=transport if accurate else None,
            attention_query_buffer_factors=[group["query_buffer_factor"] for group in geometry],
            attention_group_geometry=geometry,
            attention_effective_fidelity="HiFi4" if accurate else None,
            attention_fp32_denominator=True if accurate else None,
            attention_fp32_output_accumulator=True if accurate else None,
            reset_inputs=bool(reload_inputs),
            layout_epoch=self.layout_epoch,
            physical_rows=32,
        )
        # decode() is called before submission; commit advancement only on a successful return.

    def submitted(self):
        self.annotate(generated_return_ns=time.perf_counter_ns())
        if self.current_id is not None:
            with self.lock:
                phase = self.pending[self.current_id]["phase"]
            if phase == "decode" and self.positions is not None:
                self.positions = [p + 1 if p >= 0 else -1 for p in self.positions]

    def complete(self, step_id, request_ids, output_lengths):
        now = time.perf_counter_ns()
        if step_id is None:
            return
        with self.lock:
            record = self.pending.pop(step_id, None)
            if record is None:
                self.errors.append(f"Unmatched/duplicate completed step {step_id}")
                return
            record.update(
                output_ready_ns=now,
                output_request_ids=list(request_ids),
                output_token_counts=list(map(int, output_lengths)),
            )
            if len(self.records) >= self.limit:
                if not self.errors or self.errors[-1] != "Recorder overflow":
                    self.errors.append("Recorder overflow")
            else:
                self.records.append(record)

    def finished(self, request_ids, timestamp_ns):
        if request_ids:
            with self.lock:
                if len(self.records) >= self.limit:
                    if not self.errors or self.errors[-1] != "Recorder overflow":
                        self.errors.append("Recorder overflow")
                else:
                    self.records.append(
                        {"kind": "requests_finished", "request_ids": list(request_ids), "timestamp_ns": timestamp_ns}
                    )

    def snapshot(self):
        with self.lock:
            records = list(self.records)
            status = {
                "kind": "status",
                "schema_version": 1,
                "count": len(records),
                "pending": [dict(record) for record in self.pending.values()],
                "errors": list(self.errors),
                "snapshot_ns": time.perf_counter_ns(),
            }
        if self.host_sampler_stats is not None:
            status["host_sampler_stats"] = self.host_sampler_stats()
        path = Path(self.path)
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("w") as out:
            for record in [self.identity, *records, status]:
                out.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        tmp.replace(path)
        return {"path": self.path, **status}

    def _serve(self):
        while not self.closed:
            try:
                connection, _ = self.server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with connection:
                connection.settimeout(10)
                try:
                    command = connection.recv(128).decode().strip()
                    reply = self.snapshot() if command == "snapshot" else {"error": "Expected snapshot"}
                    connection.sendall((json.dumps(reply) + "\n").encode())
                except Exception as exc:
                    with self.lock:
                        self.errors.append(f"Snapshot error: {exc!r}")

    def close(self):
        self.closed = True
        self.server.close()
        self.thread.join(timeout=2)
        self.snapshot()
        Path(self.socket_path).unlink(missing_ok=True)
