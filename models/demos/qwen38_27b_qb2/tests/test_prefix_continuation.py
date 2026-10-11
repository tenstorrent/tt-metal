# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compare disk-restored prefixes with uninterrupted model execution on TP4.

This test owns all request slots/pages. It does not implement vLLM leases,
enable serving prefix hits, qualify AgentX, or replace independent evals.
"""

import hashlib
import os
import time
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.qwen38_27b_qb2.demo.galaxy_serving import model_source_hashes
from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import save
from models.demos.qwen38_27b_qb2.tests.test_prefix_transfer import ExclusiveTestTarget, stats
from models.demos.qwen38_27b_qb2.tt.generator import build_generator, configure_fabric
from models.demos.qwen38_27b_qb2.tt.prefix_checkpoint import Checkpoint, Identity, Layout, capture, digest, restore
from models.demos.qwen38_27b_qb2.tt.prefix_storage import AtomicDirectoryStore
from models.demos.qwen38_27b_qb2.tt.prefix_transfer import PackedCacheTransfer


class QuiescentModelSource:
    """Test-only execution lease; the single test thread owns the whole model."""

    def __init__(self, gen, transfer):
        self.gen, self.transfer = gen, transfer

    @contextmanager
    def freeze(self, checkpoint):
        assert checkpoint == self.transfer.checkpoint
        ttnn.synchronize_device(self.gen.mesh)
        self.gen.model.suspend_decode_bucket()
        self.transfer.fence()
        try:
            yield self.transfer
        finally:
            self.transfer.close()


def recurrent_hashes(transfer):
    """Inspect canonical state under the caller's exclusive, already-fenced lease."""
    result = {}
    for segment in transfer.checkpoint.segments():
        if segment.kind in ("recurrent", "conv"):
            result[f"{segment.rank}:{segment.layer}:{segment.kind}"] = hashlib.sha256(
                transfer.read(segment, 0, segment.size)
            ).hexdigest()
    transfer.close()
    return result


@pytest.mark.skipif(os.getenv("QWEN_PREFIX_CONTINUATION") != "1", reason="explicit allocated TP4 continuation test")
def test_prefix_continuation():
    assert not any(os.getenv(k) for k in ("TT_METAL_SIMULATOR", "TT_METAL_SLOW_DISPATCH_MODE"))
    assert os.environ["QWEN_DECODE_BUCKETS"] == "1"
    layers = int(os.environ["QWEN_PREFIX_LAYERS"])
    assert layers in (4, 64)
    receipt = Path(os.environ["QWEN_PREFIX_CONTINUATION_RECEIPT"])
    assert not receipt.exists()
    torch.set_num_threads(8)
    report = dict(
        state="opening",
        passed=False,
        cleanup_completed=False,
        layers=layers,
        serving_enabled=False,
        independent_eval=False,
        started_at=time.time(),
        cases=[],
    )
    save(receipt, report)
    parent = mesh = gen = None
    try:
        configure_fabric(topology=ttnn.Topology.Linear)
        parent = ttnn.open_mesh_device(ttnn.MeshShape(8, 4), trace_region_size=200000000)
        mesh = parent.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))
        report.update(state="loading", device_ids=list(mesh.get_device_ids()))
        save(receipt, report)
        source = Path(__file__).resolve().parents[1]
        policy = source / "config/precision_single_step_compact_gdn_bfp8_all.json"
        gen = build_generator(
            source,
            mesh,
            layer_indices=list(range(layers)),
            precision_config=policy,
            host_sampling=True,
            topology=ttnn.Topology.Linear,
        )
        report.update(state="prefill", source_sha256=model_source_hashes(source), precision=gen.model.precision)
        save(receipt, report)
        batch, capacity, prefix_length, suffix_length = 16, 4608, 4096, 32
        cache = gen.model.allocate_cache(batch_size=batch, capacity=capacity)
        table = (
            torch.randperm(cache.num_pages, generator=torch.Generator().manual_seed(20261010)).int().reshape(batch, -1)
        )
        gen.bind_cache(cache, table)
        gen.reset()
        layout = Layout(
            tuple(layer.kind for layer in gen.model.layers),
            4,
            32,
            8704,
            786432,
            15360,
            "tt-bh-tp4-bfp8-tile32-kv-fp32-tiled-gdn-bf16-row-conv-v1",
        )
        if layers == 64:
            assert layout == Layout.qwen_tp4_bfp8(layout.layer_types)
        identity = Identity(
            "local-pinned-checkpoint",
            digest(report["source_sha256"]),
            digest(report["precision"]),
            "exclusive-continuation-test",
        )
        seed_tokens = gen.tokenizer.encode("The cedar tree stands beside the river. Remember the word cedar. ")
        prefix = (seed_tokens * (1 + prefix_length // len(seed_tokens)))[:prefix_length]
        suffix = (gen.tokenizer.encode(" Tell me what you remember about the tree. ") * 8)[:suffix_length]
        consumed = prefix + suffix
        checkpoint = Checkpoint.for_tokens(identity, layout, consumed, prefix_length)
        store = AtomicDirectoryStore(
            receipt.parent / "checkpoint-store", max_bytes=4 * checkpoint.encoded_bytes, create=True
        )

        def transfer(cp, slot):
            return PackedCacheTransfer(mesh, cache, cp, slot=slot, pages=table[slot, : cp.consumed // 32].tolist())

        started = time.monotonic()
        result = gen.prefill_forward(
            torch.tensor([prefix]), page_table=table, kv_cache=cache, prompt_lens=[prefix_length], slots=[1]
        )
        del result
        ttnn.synchronize_device(mesh)
        report["prefix_prefill_s"] = time.monotonic() - started
        src = transfer(checkpoint, 1)
        started = time.monotonic()
        capture(store, checkpoint, QuiescentModelSource(gen, src))
        report["prefix_capture_s"] = time.monotonic() - started
        dst = ExclusiveTestTarget(transfer(checkpoint, 3))
        started = time.monotonic()
        restore(store, checkpoint, dst)
        report["prefix_restore_s"] = time.monotonic() - started
        assert dst.committed
        report.update(state="suffix_prefill", prefix_capture=stats(src), prefix_restore=stats(dst.transport))
        save(receipt, report)
        logits = []
        for slot in (1, 3):
            values = gen.prefill_forward(
                torch.tensor([suffix]),
                page_table=table,
                kv_cache=cache,
                prompt_lens=[suffix_length],
                slots=[slot],
                start_pos=[prefix_length],
            )
            logits.append(gen._host_logits(values[0]))
            del values
        assert torch.isfinite(logits[0]).all() and torch.equal(logits[0], logits[1])
        report["cases"].append(dict(name="suffix_prefill", logits_exact=True))
        del logits

        def decode(token1, token3, position1, position3):
            tokens = torch.zeros(batch, dtype=torch.int64)
            tokens[1], tokens[3] = token1, token3
            positions = torch.full((batch,), -1, dtype=torch.int32)
            positions[1], positions[3] = position1, position3
            return gen.decode_forward(tokens, positions, page_table=table, kv_cache=cache, active_slots=(1, 3))

        report.update(state="traced_decode")
        save(receipt, report)
        for step in range(32):
            token = seed_tokens[step % len(seed_tokens)]
            output = decode(token, token, len(consumed), len(consumed))
            assert torch.isfinite(output[[1, 3]]).all()
            assert torch.equal(output[1], output[3]), f"Restored decode differs at step {step}"
            consumed.append(token)
        report["cases"].append(dict(name="traced_decode", steps=32, logits_exact=True))
        assert len(consumed) % 32 == 0 and gen.model._resident_decode_valid
        resident_addresses = [
            tensor.buffer_address()
            for state in gen.model._resident_decode_bucket[2].layers
            for tensor in (state.recurrent, state.conv)
            if tensor is not None
        ]
        trace, trace_count = gen.trace, gen.counters["trace_captures"]
        decoded = Checkpoint.for_tokens(identity, layout, consumed, len(consumed))
        capture(store, decoded, QuiescentModelSource(gen, transfer(decoded, 1)))
        assert not gen.model._resident_decode_valid
        report.update(state="restore_with_live_trace")
        save(receipt, report)
        next_token = seed_tokens[0]
        reference = decode(next_token, next_token, len(consumed), len(consumed))[1].clone()
        ttnn.synchronize_device(mesh)
        gen.model.suspend_decode_bucket()
        ttnn.synchronize_device(mesh)
        neighbour = recurrent_hashes(transfer(decoded, 3))
        # Test-only withdrawal of slot 1: no other thread can observe it until
        # successful commit. Slot 3 remains resident and must survive unchanged.
        restored = ExclusiveTestTarget(transfer(decoded, 1))
        restore(store, decoded, restored)
        assert restored.committed
        assert recurrent_hashes(transfer(decoded, 3)) == neighbour
        replay = decode(next_token, next_token, len(consumed), len(consumed) + 1)[1]
        assert torch.isfinite(replay).all() and torch.equal(replay, reference)
        assert gen.trace == trace and gen.counters["trace_captures"] == trace_count
        assert resident_addresses == [
            tensor.buffer_address()
            for state in gen.model._resident_decode_bucket[2].layers
            for tensor in (state.recurrent, state.conv)
            if tensor is not None
        ]
        report["cases"].append(
            dict(
                name="post_decode_checkpoint",
                logits_exact=True,
                trace_preserved=True,
                resident_addresses_preserved=True,
                neighbour_unchanged=True,
            )
        )
        report.update(
            state="completed",
            prefix_tokens=prefix_length,
            suffix_tokens=suffix_length,
            decoded_checkpoint_tokens=len(consumed),
            checkpoint_payload_bytes=checkpoint.payload_bytes,
        )
    except BaseException as error:
        report.update(state="failed", error=type(error).__name__, detail=str(error)[:3000])
        raise
    finally:
        try:
            try:
                if gen is not None:
                    gen._release_traces()
            finally:
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
