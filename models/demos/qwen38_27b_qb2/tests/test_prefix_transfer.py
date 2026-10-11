# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Physical TP4 packed-byte offload gate; no numerical/serving claim.

Use synthetic opaque bytes on real Qwen tensor geometries, all four ranks,
two layer types, shuffled KV pages and nonzero recurrent/conv slots. This
quiescent test owns every allocation; its leases are NOT scheduler adapters.
"""

import hashlib
import os
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import save
from models.demos.qwen38_27b_qb2.tt.generator import configure_fabric
from models.demos.qwen38_27b_qb2.tt.prefix_checkpoint import (
    COMPLETE,
    MAX_CHUNK_BYTES,
    Checkpoint,
    Identity,
    Layout,
    capture,
    restore,
)
from models.demos.qwen38_27b_qb2.tt.prefix_storage import AtomicDirectoryStore
from models.demos.qwen38_27b_qb2.tt.prefix_transfer import PackedCacheTransfer


def raw_buffer(host, rank=0):
    return torch.from_dlpack(host.host_buffer().get_shard(ttnn.MeshCoordinate(0, rank)))


def allocate_pattern(mesh, shape, dtype, layout, seed):
    host = ttnn.allocate_tensor_on_host(shape, dtype, layout, mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    expected = []
    for rank in range(4):
        raw = raw_buffer(host, rank)
        assert raw.dtype == torch.uint8
        raw.copy_(
            torch.randint(0, 64, raw.shape, dtype=torch.uint8, generator=torch.Generator().manual_seed(seed + rank))
        )
        expected.append(raw.numpy().tobytes())
    device = ttnn.to_device(host, mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.synchronize_device(mesh)
    return device, expected


def download(tensor):
    host = ttnn.allocate_tensor_on_host(
        tensor.shape, tensor.dtype, tensor.layout, tensor.device(), memory_config=tensor.memory_config()
    )
    ttnn.copy_device_to_host_tensor(tensor, host, blocking=True)
    return [raw_buffer(host, rank).numpy().tobytes() for rank in range(4)]


class ExclusiveTestSource:
    def __init__(self, transport):
        self.transport = transport

    @contextmanager
    def freeze(self, checkpoint):
        assert checkpoint == self.transport.checkpoint
        self.transport.fence()
        try:
            yield self.transport
        finally:
            self.transport.close()


class ExclusiveTestTarget:
    def __init__(self, transport):
        self.transport = transport
        self.committed = False
        self.aborted = False

    @contextmanager
    def begin(self, checkpoint):
        assert checkpoint == self.transport.checkpoint
        try:
            yield self
            assert self.committed
        except BaseException:
            self.aborted = True
            self.transport.close(discard=True)
            raise
        else:
            self.transport.close()

    def write(self, segment, offset, data):
        self.transport.write(segment, offset, data)

    def commit(self, consumed):
        assert consumed == self.transport.checkpoint.consumed
        self.transport.fence()
        self.committed = True


def stats(transport):
    return {
        name: getattr(transport, name)
        for name in ("bytes_read", "bytes_written", "windows_read", "windows_written", "max_host_window_bytes")
    }


@pytest.mark.skipif(os.getenv("QWEN_PREFIX_TRANSFER") != "1", reason="explicit allocated TP4 transfer experiment")
def test_prefix_transfer():
    assert not any(os.getenv(k) for k in ("TT_METAL_SIMULATOR", "TT_METAL_SLOW_DISPATCH_MODE"))
    receipt = Path(os.environ["QWEN_PREFIX_TRANSFER_RECEIPT"])
    assert not receipt.exists(), "Preserve each independent attempt"
    torch.set_num_threads(4)
    report = dict(
        state="opening",
        passed=False,
        cleanup_completed=False,
        serving_enabled=False,
        numerical_continuation_tested=False,
        started_at=time.time(),
    )
    save(receipt, report)
    parent = mesh = None
    try:
        configure_fabric(topology=ttnn.Topology.Linear)
        parent = ttnn.open_mesh_device(ttnn.MeshShape(8, 4), trace_region_size=200000000)
        mesh = parent.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))
        report["device_ids"] = list(mesh.get_device_ids())
        assert len(set(report["device_ids"])) == 4
        geometry = Layout(("full_attention", "linear_attention"), 4, 32, 8704, 786432, 15360, "tp4-transfer-test-v1")
        checkpoint = Checkpoint.for_tokens(
            Identity("synthetic", "test", "opaque-bytes", "isolated"), geometry, list(range(129)), 128
        )
        tensors, before = {}, {}
        for name, shape, dtype, layout in (
            ("key", (16, 1, 32, 256), ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
            ("value", (16, 1, 32, 256), ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
            ("recurrent", (4, 12, 128, 128), ttnn.float32, ttnn.TILE_LAYOUT),
            ("conv", (4, 3, 2560), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
        ):
            tensors[name], before[name] = allocate_pattern(
                mesh, shape, dtype, layout, seed=20261010 + len(tensors) * 100
            )
        cache = SimpleNamespace(
            batch_size=4,
            num_pages=16,
            layers=(
                SimpleNamespace(key=tensors["key"], value=tensors["value"]),
                SimpleNamespace(recurrent=tensors["recurrent"], conv=tensors["conv"]),
            ),
        )
        pages = (7, 8, 2, 4)
        destination_pages = (10, 0, 11, 5)
        source = PackedCacheTransfer(mesh, cache, checkpoint, slot=1, pages=pages)
        target = ExclusiveTestTarget(PackedCacheTransfer(mesh, cache, checkpoint, slot=3, pages=destination_pages))
        root = receipt.parent / "checkpoint-store"
        store = AtomicDirectoryStore(root, max_bytes=2 * checkpoint.encoded_bytes, create=True)
        report.update(state="capture", payload_bytes=checkpoint.payload_bytes, encoded_bytes=checkpoint.encoded_bytes)
        save(receipt, report)
        started = time.monotonic()
        capture(store, checkpoint, ExclusiveTestSource(source), chunk_bytes=100003)
        report["capture_s"] = time.monotonic() - started
        # Reopen the on-disk store, rather than retaining an in-memory blob.
        store = AtomicDirectoryStore(root, max_bytes=2 * checkpoint.encoded_bytes)
        started = time.monotonic()
        restore(store, checkpoint, target, chunk_bytes=77777)
        report["restore_s"] = time.monotonic() - started
        assert target.committed and not target.aborted
        assert source.max_host_window_bytes <= MAX_CHUNK_BYTES
        assert target.transport.max_host_window_bytes <= MAX_CHUNK_BYTES
        report.update(state="checking_all_bytes", capture=stats(source), restore=stats(target.transport))
        save(receipt, report)
        hashes = {}
        after = {}
        for name, tensor in tensors.items():
            after[name] = download(tensor)
            for rank in range(4):
                expected = bytearray(before[name][rank])
                unit = geometry.kv_page_bytes if name in ("key", "value") else getattr(geometry, name + "_bytes")
                for src, dst in zip(pages, destination_pages) if name in ("key", "value") else ((1, 3),):
                    expected[dst * unit : (dst + 1) * unit] = before[name][rank][src * unit : (src + 1) * unit]
                assert after[name][rank] == expected, f"Changed neighbour or wrong destination: {name}, rank {rank}"
                hashes[f"{name}:{rank}"] = hashlib.sha256(after[name][rank]).hexdigest()
        # Corrupt the final checksum: earlier device writes must never publish.
        blob = root / (checkpoint.key + ".checkpoint")
        with blob.open("r+b") as stream:
            stream.seek(-len(COMPLETE) - 1, 2)
            offset = stream.tell()
            value = stream.read(1)
            stream.seek(offset)
            stream.write(bytes([value[0] ^ 1]))
        aborted = ExclusiveTestTarget(PackedCacheTransfer(mesh, cache, checkpoint, slot=2, pages=(12, 13, 14, 15)))
        try:
            restore(store, checkpoint, aborted, chunk_bytes=65537)
        except ValueError as error:
            assert "checksum" in str(error)
        else:
            raise AssertionError("Corrupt checkpoint was accepted")
        assert aborted.aborted and not aborted.committed
        # The aborted private destination is quarantined until teardown. Every
        # other byte, including the previously restored request, must survive.
        for name, tensor in tensors.items():
            observed = download(tensor)
            unit = geometry.kv_page_bytes if name in ("key", "value") else getattr(geometry, name + "_bytes")
            private = range(12, 16) if name in ("key", "value") else (2,)
            for rank in range(4):
                for index in range(len(observed[rank]) // unit):
                    if index not in private:
                        assert (
                            observed[rank][index * unit : (index + 1) * unit]
                            == after[name][rank][index * unit : (index + 1) * unit]
                        )
        report.update(
            state="completed",
            all_rank_bytes_exact=True,
            neighbours_unchanged=True,
            corrupt_restore_unpublished=True,
            hashes=hashes,
        )
    except BaseException as error:
        report.update(state="failed", error=type(error).__name__, detail=str(error)[:3000])
        raise
    finally:
        try:
            try:
                if mesh is not None:
                    ttnn.close_mesh_device(mesh)
            finally:
                if parent is not None:
                    ttnn.close_mesh_device(parent)
            report["cleanup_completed"] = parent is not None
            report["passed"] = report["state"] == "completed" and report["cleanup_completed"]
        finally:
            report["finished_at"] = time.time()
            save(receipt, report)
