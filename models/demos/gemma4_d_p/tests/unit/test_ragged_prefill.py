# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from models.demos.gemma4_d_p.tt.ragged_prefill import PrefillRequest, RaggedPrefillPlan, iter_request_batches
from models.demos.gemma4_d_p.tt.runners.runtime import Gemma4PrefillRuntime


def request(identity, slot, start, length):
    return PrefillRequest(identity, slot, start, tuple(range(1, length + 1)))


@pytest.mark.parametrize("lengths", [(1, 31), (32, 33), (1023, 1025), (8191, 8192), (8192, 8192), (1,)])
def test_absolute_positions_and_inverse_cp_tp_mapping(lengths):
    requests = tuple(request(i, i, i * 8192, n) for i, n in enumerate(lengths))
    plan = RaggedPrefillPlan.for_requests(requests)
    tokens = torch.tensor(plan.pack(requests))
    positions = torch.tensor(plan.pack(requests, positions=True))
    # These splits describe the embedding partition, TP norm/MLP row ownership,
    # TP all-gather and CP attention redistribution in that order.
    cp_rows = tokens.reshape(8, -1)
    tp_rows = cp_rows.reshape(8, 4, -1)
    assert tp_rows.shape[-1] % 32 == 0
    gathered = tp_rows.reshape(8, -1).flatten()
    for req, offset, size in zip(requests, plan.offsets, plan.segment_sizes):
        local = torch.nn.functional.pad(gathered[offset : offset + size], (0, 8192 - size)).reshape(8, 1024)
        assert local.flatten()[: len(req.token_ids)].tolist() == list(req.token_ids)
        assert positions[offset : offset + len(req.token_ids)].tolist() == list(range(req.actual_start, req.actual_end))
    outputs = plan.unpack(gathered.reshape(1, 1, -1, 1), requests)
    for req in requests:
        assert outputs[req.request_id].flatten().tolist() == list(req.token_ids)
    assert plan.packed_size < sum(lengths) + 32 * len(lengths) + 1024


def test_trace_shape_ignores_identity_slot_prefix_and_exact_length():
    first = (request(0, 0, 8192, 33), request(1, 1, 0, 1023))
    reused = (request(2, 5, 0, 63), request(3, 0, 16384, 1024))
    assert RaggedPrefillPlan.for_requests(first) == RaggedPrefillPlan.for_requests(reused)
    assert RaggedPrefillPlan.for_requests(first) != RaggedPrefillPlan.for_requests(first[:1])
    assert RaggedPrefillPlan.for_requests(first).pack(first, positions=True) != RaggedPrefillPlan.for_requests(
        reused
    ).pack(reused, positions=True)


def runtime():
    rt = object.__new__(Gemma4PrefillRuntime)
    rt.config = SimpleNamespace(num_users=3, chunk_size=8192, max_seq_len=32768)
    rt.model = SimpleNamespace(vocab_size=262144)
    rt.slot_ends = [8192, 16384, 33]
    rt.slot_requests = [10, 11, 12]
    return rt


def test_continuations_and_refill_at_boundary():
    rt = runtime()
    rt.validate_batch((request(10, 0, 8192, 33), request(13, 2, 0, 1025)))
    # Validation is atomic, including ownership; no slot is committed yet.
    assert rt.slot_ends == [8192, 16384, 33]
    assert rt.slot_requests == [10, 11, 12]


@pytest.mark.parametrize(
    "requests, message",
    [
        ((), "active requests"),
        ((request(10, 0, 8192, 1), request(11, 0, 0, 1)), "one chunk"),
        ((request(10, 0, 8192, 1), request(10, 1, 16384, 1)), "unique"),
        ((request(99, 0, 8192, 1),), "belongs"),
        ((request(10, 0, 16384, 1),), "expects"),
        ((request(12, 2, 33, 1),), "multiple"),
        ((request(10, 0, 8192, 8193),), "real tokens"),
        ((request(10, 0, 8192, 0),), "real tokens"),
        ((request(10, 3, 0, 1),), "outside"),
    ],
)
def test_invalid_batch_does_not_advance_any_slot(requests, message, expect_error):
    rt = runtime()
    with expect_error(ValueError, message):
        rt.validate_batch(requests)
    assert rt.slot_ends == [8192, 16384, 33]
    assert rt.slot_requests == [10, 11, 12]


def test_packing_bucket_mismatch(expect_error):
    plan = RaggedPrefillPlan((32,))
    with expect_error(ValueError, "occupancy"):
        plan.pack(())
    with expect_error(ValueError, "length"):
        plan.pack((request(0, 0, 0, 33),))


def test_scheduler_refills_finished_slots_and_preserves_chunk_order():
    prompts = [(1, range(8193)), (2, range(33)), (3, range(16385)), (4, range(32))]
    batches = list(iter_request_batches(prompts, num_slots=2))
    assert [[(r.request_id, r.slot_id, r.actual_start, len(r.token_ids)) for r in batch] for batch in batches] == [
        [(1, 0, 0, 8192), (2, 1, 0, 33)],
        [(1, 0, 8192, 1), (3, 1, 0, 8192)],
        [(4, 0, 0, 32), (3, 1, 8192, 8192)],
        [(3, 1, 16384, 1)],
    ]
    for identity, tokens in prompts:
        assert [token for batch in batches for r in batch if r.request_id == identity for token in r.token_ids] == list(
            tokens
        )


def test_cache_writes_receive_each_lanes_stable_valid_end(monkeypatch):
    import ttnn
    from models.demos.gemma4_d_p.tt.attention.ring_prefill import (
        write_chunk_to_global_ring_cache,
        write_chunk_to_sliding_ring_cache,
    )
    from models.demos.gemma4_d_p.tt.prefill_metadata import PrefillMetadata

    monkeypatch.setattr(PrefillMetadata, "_device_scalar", lambda _: object())
    monkeypatch.setattr(PrefillMetadata, "_host_scalar", lambda _, value: value)
    copies = {}
    monkeypatch.setattr(ttnn, "copy_host_to_device_tensor", lambda value, tensor: copies.update({tensor: value}))
    lanes = [PrefillMetadata(SimpleNamespace(cp_axis=0), clamp_valid=True) for _ in range(2)]
    writes = []
    monkeypatch.setattr(ttnn.experimental.deepseek_prefill, "update_padded_kv_cache", lambda **kw: writes.append(kw))
    tensor = SimpleNamespace(dtype=ttnn.bfloat8_b)
    addresses = [(m.slot_idx, m.kv_actual_global, m.valid_global) for m in lanes]
    for replay in range(3):
        for i, metadata in enumerate(lanes):
            metadata.update(
                slot_idx=(i + replay) % 3, kv_actual_global=replay * 8192, valid_global=replay * 8192 + i + 1
            )
            write_chunk_to_global_ring_cache(tensor, tensor, metadata.mesh_config, 0, prefill_metadata=metadata)
            write_chunk_to_sliding_ring_cache(
                tensor, tensor, tensor, tensor, metadata.mesh_config, 0, prefill_metadata=metadata
            )
            assert copies[metadata.valid_global] == replay * 8192 + i + 1
            for write in writes[-3:]:
                assert (write["slot_idx"], write["kv_actual_global"], write["valid_global"]) == addresses[i]
    assert len(writes) == 18


def test_migration_rows_keep_original_chunk_geometry():
    from models.demos.gemma4_d_p.tt.runners.kv_chunk_table import iter_cache_chunk_locations

    # Packing does not change the durable cache allocation or its address table.
    entries = iter_cache_chunk_locations(
        seq_len=32768,
        chunk_size=8192,
        cp=8,
        num_users=3,
        heads_per_device=4,
        local_head=2,
        num_banks=8,
        chunk_size_bytes=8704,
    )
    for rank, slot, position, bank, offset in entries:
        chunk, within = divmod(position, 8192)
        assert within // 1024 == rank
        row = chunk * 32 + (within % 1024) // 32
        shard = (slot * 4 + 2) * 128 + row
        assert bank == shard % 8
        assert offset == shard // 8 * 8704


def test_runtime_replays_metadata_then_releases_before_allocating_new_shape(monkeypatch, expect_error):
    import ttnn
    from models.demos.gemma4_d_p.tt.runners import ragged_runtime

    events = []

    class Variant:
        def __init__(self, rt, plan):
            self.plan = plan
            self.output = object()
            events.append(("allocate", plan))

        def stage(self, requests):
            self.requests = requests
            events.append(("stage", tuple((r.slot_id, r.actual_start, r.actual_end) for r in requests)))

        def run(self, **kwargs):
            events.append(("run", self.plan))
            return self.output

        def release(self):
            events.append(("release", self.plan))

    monkeypatch.setattr(ragged_runtime, "RaggedTraceVariant", Variant)
    monkeypatch.setattr(ttnn, "synchronize_device", lambda _: events.append(("sync",)))
    rt = runtime()
    rt.config.num_layers = 2
    rt.mesh_config = SimpleNamespace(cp_degree=8, tp_degree=4)
    rt.mesh_device = object()
    rt.trace_id = None
    rt.ragged_variants = {}
    rt.output_generation = 0
    rt.batch_result = None
    rt.next_completion_id = 0
    rt.d2h_service = None
    rt._check_cache = lambda _: None
    acks = []
    rt.layer_completion_sink = lambda layer, identity: acks.append((layer, identity))
    first = rt.prefill_batch((request(10, 0, 8192, 33), request(11, 1, 16384, 1023)), None)
    variant = next(iter(rt.ragged_variants.values()))
    retained_requests = first.requests
    second = rt.prefill_batch((request(20, 2, 0, 63), request(21, 0, 0, 1024)), None)
    assert next(iter(rt.ragged_variants.values())) is variant
    assert rt.slot_requests == [21, 11, 20]
    assert rt.slot_ends == [1024, 17407, 63]
    with expect_error(RuntimeError, "expired"):
        first.to_torch()
    assert [r.request_id for r in retained_requests] == [10, 11]
    assert [r.completion_id for r in second.requests] == [2, 3]
    assert acks == [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2), (0, 3), (1, 3)]
    rt.prefill_batch((request(22, 0, 0, 31),), None)
    old_release = events.index(("release", variant.plan))
    new_allocate = events.index(("allocate", RaggedPrefillPlan((32,))))
    assert old_release < new_allocate
    assert len(rt.ragged_variants) == 1
    rt.release_trace()
    assert not rt.ragged_variants


def test_full_extent_route_slice_does_not_alias_its_freed_source(monkeypatch):
    import ttnn
    from models.demos.gemma4_d_p.tt.ragged_prefill import RaggedAttentionLayout

    source = SimpleNamespace(shape=(1, 8, 8192, 256))
    owned = object()
    monkeypatch.setattr(ttnn, "clone", lambda value: owned if value is source else None)
    monkeypatch.setattr(ttnn, "slice", lambda *a, **kw: source)  # TTNN's full-extent shortcut.
    assert RaggedAttentionLayout.slice_rows(source, 0, 8192) is owned


@pytest.mark.parametrize("rows", [32, 256, 288, 2048])
def test_shared_tokenwise_slabs_preserve_rows_and_input_ownership(monkeypatch, rows):
    import ttnn
    from models.demos.gemma4_d_p.tt.ragged_prefill import map_packed_rows

    class Tensor:
        def __init__(self, values):
            self.values = values
            self.shape = values.shape
            self.freed = False

        def deallocate(self, force):
            self.freed = True

    monkeypatch.setattr(ttnn, "slice", lambda x, start, end: Tensor(x.values[:, :, start[2] : end[2], :].clone()))
    monkeypatch.setattr(ttnn, "concat", lambda xs, dim: Tensor(torch.cat([x.values for x in xs], dim)))
    source = Tensor(torch.arange(rows * 2).reshape(1, 1, rows, 2))
    slabs = []

    def operation(slab):
        assert not slab.freed and slab.shape[2] <= 256
        slabs.append(slab)
        return Tensor(slab.values * 3 + 1)

    output = map_packed_rows(source, 256, operation)
    torch.testing.assert_close(output.values, source.values * 3 + 1)
    assert not source.freed
    if rows > 256:
        assert all(slab.freed for slab in slabs)


@pytest.mark.parametrize("rows", [128, 256, 1024])
def test_packed_projection_keeps_reference_math_fidelity(monkeypatch, rows):
    import ttnn
    from models.demos.gemma4_d_p.tt.attention.operations import projection_matmul_configs

    monkeypatch.setattr(ttnn, "init_device_compute_kernel_config", lambda arch, **kw: kw)
    device = SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=11, y=10), arch=lambda: None)
    activations = SimpleNamespace(shape=(1, 1, rows, 5376), padded_shape=(1, 1, rows, 5376), device=lambda: device)
    weight = SimpleNamespace(padded_shape=(1, 1, 5376, 5376))
    program, compute = projection_matmul_configs(activations, weight, reference_rows=1024)
    assert program is not None
    assert compute["math_fidelity"] == ttnn.MathFidelity.LoFi
    assert compute["fp32_dest_acc_en"]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_tp_sum_keeps_small_partials_until_final_rounding(monkeypatch, dtype):
    import ttnn
    from models.demos.gemma4_d_p.tt.ccl import ccl_reduce_scatter_rows

    class Tensor:
        def __init__(self, values):
            self.values = values
            self.shape = values.shape
            self.dtype = ttnn.float32 if values.dtype == torch.float32 else ttnn.bfloat16
            self.freed = False

        def deallocate(self, force):
            assert not self.freed
            self.freed = True

    source = Tensor(torch.zeros(1, 1, 128, 32, dtype=dtype))
    gathered = Tensor(torch.tensor([4096, 1, -4096, 1], dtype=dtype).reshape(4, 1, 1, 1).expand(4, 1, 128, 32))
    monkeypatch.setattr(ttnn, "all_gather", lambda *a, **kw: gathered)
    monkeypatch.setattr(ttnn, "slice", lambda x, start, end: Tensor(x.values[start[0] : end[0]].clone()))
    monkeypatch.setattr(
        ttnn,
        "typecast",
        lambda x, target, **kw: Tensor(x.values.to(torch.float32 if target == ttnn.float32 else torch.bfloat16)),
    )

    def add(left, right):
        assert not left.freed and not right.freed
        assert left.values.dtype == right.values.dtype == torch.float32
        return Tensor(left.values + right.values)

    monkeypatch.setattr(ttnn, "add", add)
    monkeypatch.setattr(ttnn, "mesh_partition", lambda x, **kw: Tensor(x.values[:, :, :32].clone()))
    result = ccl_reduce_scatter_rows(source, SimpleNamespace(tp_degree=4, tp_axis=1), None, stable=True)
    torch.testing.assert_close(result.values, torch.full((1, 1, 32, 32), 2, dtype=dtype), rtol=0, atol=0)
    assert source.freed and gathered.freed and not result.freed


@pytest.mark.parametrize("chunk_size, rows, message", [(4096, 32, "cache chunk geometry"), (8192, 64, "rows")])
def test_model_rejects_ragged_geometry_before_cache_writes(chunk_size, rows, message, expect_error):
    from models.demos.gemma4_d_p.tt.model import Gemma4Model

    model = object.__new__(Gemma4Model)
    model.mesh_config = SimpleNamespace(cp_degree=8, tp_degree=4)
    model.prefill_chunk_size = 8192
    plan = RaggedPrefillPlan((32,), chunk_size=chunk_size)
    with expect_error(ValueError, message):
        model(
            SimpleNamespace(shape=(1, 1, rows, 128)),
            ragged_layout=SimpleNamespace(plan=plan),
            rope_positions=object(),
        )
