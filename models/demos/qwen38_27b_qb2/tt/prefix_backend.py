# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in scheduler/worker backend for complete text-prefix checkpoints.

Construction uses host metadata only. The plugin's KVConnector allocates all
private pages before bind/restore; no attention-only APC is enabled. An operator
must provision an AtomicDirectoryStore and a deployment/tenant namespace first.
This experimental synchronous backend captures prefill frontiers, not suspended
generations, and makes no SSD or serving-performance qualification claim.
"""

import hashlib
import json
import os
from dataclasses import asdict, replace
from pathlib import Path

from models.demos.qwen38_27b_qb2.tt.prefix_checkpoint import Checkpoint, Identity, Layout, digest, find_prefix
from models.demos.qwen38_27b_qb2.tt.prefix_serving import GeneratorPrefixDriver, PrefixServingCache
from models.demos.qwen38_27b_qb2.tt.prefix_storage import AtomicDirectoryStore


def weight_metadata(snapshot):
    """Bind local immutable-artifact metadata; this is not a weight-content hash.

    The supplied weights_revision is the operator's trusted artifact identity.
    File size/mtime and index/config hashes additionally invalidate ordinary
    local edits. Do not mutate a mounted weight artifact while serving.
    """
    snapshot = Path(snapshot).resolve()
    index = snapshot / "model.safetensors.index.json"
    paths = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    files = {}
    for name in paths:
        path = snapshot / name
        # Standard HF snapshots use symlinks into the local blob directory.
        # Trust the operator's immutable artifact, but reject path traversal in
        # its index rather than resolving a shard name outside that snapshot.
        if Path(name).name != name or not path.is_file():
            raise ValueError("Require weight shard names directly inside the immutable snapshot")
        stat = path.stat()
        files[name] = [stat.st_size, stat.st_mtime_ns]
    return {
        "index": hashlib.sha256(index.read_bytes()).hexdigest(),
        "config": hashlib.sha256((snapshot / "config.json").read_bytes()).hexdigest(),
        "files": files,
    }


def implementation_identity():
    root = Path(__file__).parent
    return digest(
        {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*"))
            if path.suffix in (".py", ".cpp", ".hpp", ".h")
        }
    )


def text_config(config):
    config = dict(config.get("text_config", config))
    # HF fills these host provenance fields differently in the scheduler and
    # model loader; all actual architecture/rotary settings remain in the key.
    for key in ("_name_or_path", "_commit_hash", "transformers_version", "dtype", "torch_dtype"):
        config.pop(key, None)
    return config


class Qwen38PrefixBackend:
    """Capability-selected adapter consumed by TTCompletePrefixConnector."""

    block_size = 32

    def __init__(self, settings, hf_config):
        required = {"root", "max_bytes", "namespace", "weights_path", "weights_revision", "precision"}
        if required - set(settings) or set(settings) - required - {"capture_interval", "batched_transfer"}:
            raise ValueError(
                f"Require backend settings {sorted(required)} plus optional capture_interval/batched_transfer"
            )
        self.config = text_config(hf_config)
        self.layout = Layout.qwen_tp4_bfp8(self.config["layer_types"])
        self.namespace_base = settings["namespace"]
        if not isinstance(self.namespace_base, str) or not self.namespace_base or len(self.namespace_base) > 1024:
            raise ValueError("An explicit bounded deployment/tenant cache namespace is required")
        revision = settings["weights_revision"]
        if not isinstance(revision, str) or not revision or len(revision) > 1024:
            raise ValueError("Require the immutable weight artifact revision")
        self.snapshot = Path(settings["weights_path"]).resolve()
        self.weight_metadata = weight_metadata(self.snapshot)
        self.precision = dict(settings["precision"])
        if any(
            self.precision.get(key) != value
            for key, value in {
                "kv_cache_dtype": "bfloat8_b",
                "recurrent_dtype": "float32",
                "convolution_dtype": "bfloat16",
            }.items()
        ):
            raise ValueError("Complete-prefix format requires BFP8 KV, FP32 recurrent state and BF16 conv")
        self.interval = settings.get("capture_interval", 4096)
        if type(self.interval) is not int or self.interval < 32 or self.interval % 32:
            raise ValueError("capture_interval must be a positive multiple of 32 tokens")
        self.batched = settings.get("batched_transfer", False)
        if type(self.batched) is not bool:
            raise ValueError("batched_transfer must be a boolean")
        root = Path(settings["root"])
        if not root.is_absolute():
            raise ValueError("Require an absolute operator-provisioned cache directory")
        self.store = AtomicDirectoryStore(root, max_bytes=settings["max_bytes"])
        self.identity = Identity(
            model_revision=digest({"revision": revision, "metadata": self.weight_metadata}),
            implementation_revision=implementation_identity(),
            execution_fingerprint=digest(
                {
                    "config": self.config,
                    "precision": self.precision,
                    "environment": {key: value for key, value in os.environ.items() if key.startswith("QWEN_")},
                }
            ),
            namespace=self.namespace_base,
        )
        self.coordinator = None
        self.requests = {}
        self.fingerprint = digest(asdict(self.identity))

    def namespace(self, cache_salt):
        if cache_salt is not None and (not isinstance(cache_salt, str) or len(cache_salt) > 4096):
            raise ValueError("Invalid request cache salt")
        # An unsalted request stays within the explicitly configured tenant.
        return digest({"tenant": self.namespace_base, "cache_salt": cache_salt})

    def _checkpoint(self, tokens, namespace, consumed):
        return Checkpoint.for_tokens(replace(self.identity, namespace=namespace), self.layout, tokens, consumed)

    def checkpoint(self, tokens, namespace, consumed):
        checkpoint = self._checkpoint(tokens, namespace, consumed)
        return {"key": checkpoint.key, "consumed": checkpoint.consumed}

    def lookup(self, tokens, namespace):
        frontiers = range((len(tokens) - 1) // self.interval * self.interval, 0, -self.interval)
        checkpoint = find_prefix(
            self.store, replace(self.identity, namespace=namespace), self.layout, tokens, frontiers
        )
        return None if checkpoint is None else {"key": checkpoint.key, "consumed": checkpoint.consumed}

    def should_capture(self, consumed):
        return consumed > 0 and consumed % self.interval == 0

    def validate_scheduler(self, scheduler):
        if not scheduler.enable_chunked_prefill or scheduler.long_prefill_token_threshold != self.interval:
            raise ValueError(
                "Prefix checkpoints require chunked prefill with long_prefill_token_threshold=capture_interval"
            )

    def bind_model(self, adapter):
        model = adapter.generator.model
        if self.coordinator is not None or getattr(adapter, "_complete_prefix_backend", None) is not None:
            raise ValueError("A serving model can bind only one complete-prefix backend")
        if (
            tuple(model.mesh.shape) != (1, 4)
            or list(model.layer_indices) != list(range(64))
            or adapter.cache is None
            or model.precision != self.precision
            or text_config(model.config.to_dict()) != self.config
            or model.snapshot.resolve() != self.snapshot
            or weight_metadata(model.snapshot) != self.weight_metadata
        ):
            raise ValueError("Loaded model/cache differs from the complete-prefix artifact identity")
        self.coordinator = PrefixServingCache(
            self.store,
            self.identity,
            self.layout,
            GeneratorPrefixDriver(adapter.generator, batched=self.batched),
            slots=adapter.cache.batch_size,
            num_pages=adapter.cache.num_pages,
        )
        adapter._complete_prefix_backend = self

    def execution(self):
        if self.coordinator is None:
            raise RuntimeError("Worker must bind its model before executing prefix operations")
        return self.coordinator.execution()

    def refresh_allocations(self, pages_by_request):
        """Track newly allocated decode pages before admitting another request."""
        for request_id, record in self.requests.items():
            if request_id not in pages_by_request:
                raise ValueError("A live hybrid lease lost its scheduler allocation without release")
            pages = pages_by_request[request_id]
            self.coordinator.update(record["handle"], tokens=record["tokens"], pages=pages)
            record["pages"] = tuple(pages)

    def prepare_request(self, request_id, *, slot, tokens, pages, namespace, start, checkpoint):
        record = self.requests.get(request_id)
        if record is None:
            handle = self.coordinator.admit(request_id, slot=slot, tokens=tokens, pages=pages, namespace=namespace)
            record = self.requests[request_id] = {
                "handle": handle,
                "slot": slot,
                "tokens": tuple(tokens),
                "pages": tuple(pages),
                "namespace": namespace,
            }
        else:
            if record["slot"] != slot or record["namespace"] != namespace:
                raise ValueError("A live request changed recurrent slot or tenant without a lifecycle transition")
            self.coordinator.update(record["handle"], tokens=tokens, pages=pages)
            record.update(tokens=tuple(tokens), pages=tuple(pages))
        handle = record["handle"]
        if checkpoint is not None:
            if self.checkpoint(tokens, namespace, start) != checkpoint:
                raise ValueError("Scheduler and worker complete-prefix identities differ")
            if self.coordinator.frontier(handle):
                raise ValueError("An external checkpoint can only load into a fresh private generation")
            restored = self.coordinator.restore_longest(handle, [start]) == start
            if not restored:
                # A damaged immutable file must not force repeated cold retries
                # forever. Evict this reconstructable key under the store's
                # lease; a later cold prefill can republish it. Device failures
                # raise before here and never become storage misses.
                try:
                    self.store.discard(checkpoint["key"])
                except OSError:
                    self.coordinator.stats["failed_storage_evictions"] += 1
            return restored
        if self.coordinator.frontier(handle) != start:
            raise ValueError("Scheduled continuation does not match the consumed hybrid frontier")
        return True

    def complete_request(self, request_id, consumed, *, capture):
        handle = self.requests[request_id]["handle"]
        self.coordinator.advance(handle, consumed)
        if capture:
            if not self.should_capture(consumed):
                raise ValueError("Capture frontier differs from the configured checkpoint policy")
            self.coordinator.capture(handle)

    def release_slot(self, slot):
        for request_id, record in list(self.requests.items()):
            if record["slot"] == slot:
                self.coordinator.release(record["handle"])
                del self.requests[request_id]

    def remap_slots(self, moves):
        remap = {slot: moves.get(slot, slot) for slot in range(self.coordinator.slots)}
        self.coordinator.remap_slots(remap)
        for record in self.requests.values():
            record["slot"] = remap[record["slot"]]
