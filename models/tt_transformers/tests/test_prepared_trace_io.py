# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Generator boundaries with real device operations and allocation tracking."""

from types import SimpleNamespace

import pytest
import torch
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.trace_io import PreparedTraceIO

pytestmark = pytest.mark.skipif(
    not trace_allocation_tracker.TRACE_ALLOC_TRACKING,
    reason="requires TT_METAL_TRACE_ALLOC_TRACKING=1",
)

MESH_CONFIGS = [
    pytest.param(1, {"trace_region_size": 24 * 1024 * 1024}, id="single"),
    pytest.param(
        (1, 8),
        {"trace_region_size": 24 * 1024 * 1024, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING},
        id="t3k",
    ),
]


class _Model:
    sampling = None

    def __init__(self, mesh):
        self.mesh = mesh

    def host(self, value):
        return ttnn.from_torch(
            value.to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )

    def prepare_prefill_inputs_trace(self, value, **kwargs):
        return self.host(value), None, None, None, None, None

    def transform_and_embed_prefill_inputs_device(self, *inputs):
        return inputs

    def ttnn_prefill_forward(self, x, **kwargs):
        return ttnn.neg(x)

    def prepare_decode_inputs_host(self, tokens, current_pos, page_table=None):
        return self.host(tokens), None, None, None

    def ttnn_decode_forward(self, tokens, *args, **kwargs):
        return ttnn.neg(tokens), None


def _host(tensor):
    return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])


@pytest.mark.parametrize("mesh_device,device_params", MESH_CONFIGS, indirect=True)
@pytest.mark.parametrize("decode_first", [False, True])
def test_generator_outputs_survive_other_graphs(mesh_device, device_params, decode_first, expect_error):
    model = _Model(mesh_device)
    generator = Generator([model], [SimpleNamespace(mesh_device=mesh_device)], mesh_device)
    inputs = [torch.full((1, 1, n, 32), float(i + 1)) for i, n in enumerate((32, 64, 32))]
    prefills = [generator._prepare_trace_prefill(value, model_id=0) for value in inputs]
    decode = generator._prepare_decode_trace_text([inputs[0]], [torch.zeros(1)])
    # Compatible prefill shapes share storage; stateful decode inputs do not.
    assert prefills[0]["output"].buffer_unique_id() == prefills[2]["output"].buffer_unique_id()
    assert prefills[0]["device_inputs"][0].buffer_unique_id() == prefills[2]["device_inputs"][0].buffer_unique_id()
    assert prefills[0]["device_inputs"][0].buffer_unique_id() != decode["device_inputs"][0][0].buffer_unique_id()
    retained = ttnn.clone(prefills[0]["output"])
    ttnn.copy(prefills[0]["output"], retained)
    cache_entries = mesh_device.num_program_cache_entries()
    traces = []
    try:
        preparations = [("prefill", p) for p in prefills] + [("decode", decode)]
        if decode_first:
            preparations.reverse()
        for kind, prepared in preparations:
            if kind == "prefill":
                trace_id, output, *_ = generator._record_trace_prefill(prepared)
            else:
                ids, outputs, *_ = generator._record_decode_trace_text(prepared)
                trace_id, output = ids[0], outputs[0][0]
            traces.append((trace_id, output, prepared, kind))
        snapshots = []
        for step, (trace_id, output, prepared, kind) in enumerate(traces * 2, 1):
            target = prepared["device_inputs"][0] if kind == "prefill" else prepared["device_inputs"][0][0]
            value = torch.full(tuple(target.shape), float(step))
            ttnn.copy_host_to_device_tensor(model.host(value), target)
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            snapshots.append((output.cpu(blocking=False), -value))
            # Borrowed output is valid until another writer to its group runs.
            if tuple(output.shape) == tuple(retained.shape):
                ttnn.copy(output, retained)
                retained_expected = -value
        ttnn.synchronize_device(mesh_device)
        for snapshot, expected in snapshots:
            torch.testing.assert_close(_host(snapshot), expected.to(torch.bfloat16), rtol=0, atol=0)
        torch.testing.assert_close(_host(retained), retained_expected.to(torch.bfloat16), rtol=0, atol=0)
        assert mesh_device.num_program_cache_entries() == cache_entries
        with expect_error(RuntimeError, "preparation is closed"):
            generator._prepare_trace_prefill(torch.zeros(1, 1, 96, 32), model_id=0)
        with expect_error(RuntimeError, "preparation is closed"):
            generator._prepare_decode_trace_text([inputs[0]], [torch.zeros(1)], skip_precompile=True)
    finally:
        for trace_id, *_ in traces:
            ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.parametrize("mesh_device,device_params", MESH_CONFIGS[:1], indirect=True)
def test_prepared_io_distinguishes_mesh_placement(mesh_device, device_params):
    # Even on a one-chip mesh these have different topology metadata.
    value = torch.zeros(1, 1, 32, 32, dtype=torch.bfloat16)
    replicated = ttnn.from_torch(value, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))
    sharded = ttnn.from_torch(value, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1))
    registry = PreparedTraceIO()
    a = registry.prepare_inputs((replicated,), mesh_device, "input")[0]
    b = registry.prepare_inputs((sharded,), mesh_device, "input")[0]
    assert a.buffer_unique_id() != b.buffer_unique_id()


@pytest.mark.parametrize("mesh_device,device_params", MESH_CONFIGS[:1], indirect=True)
def test_capture_does_not_hide_model_survivors(mesh_device, device_params, monkeypatch, expect_error):
    model = _Model(mesh_device)
    generator = Generator([model], [SimpleNamespace(mesh_device=mesh_device)], mesh_device)
    prepared = generator._prepare_trace_prefill(torch.ones(1, 1, 32, 32), model_id=0)
    older_id, _, *_ = generator._record_trace_prefill(prepared)
    survivors = []
    forward = model.ttnn_prefill_forward

    def leak(*args, **kwargs):
        output = forward(*args, **kwargs)
        survivors.append(output)
        return output

    monkeypatch.setattr(model, "ttnn_prefill_forward", leak)
    newer_id = None
    try:
        newer_id, _, *_ = generator._record_trace_prefill(prepared)
        assert survivors[0].buffer_unique_id() in trace_allocation_tracker.get_unsafe_tracked_ids(mesh_device, older_id)
        with expect_error(RuntimeError, "still alive before trace replay"):
            ttnn.execute_trace(mesh_device, older_id, cq_id=0, blocking=True)
    finally:
        if newer_id is not None:
            ttnn.release_trace(mesh_device, newer_id)
        ttnn.release_trace(mesh_device, older_id)


@pytest.mark.parametrize("mesh_device,device_params", MESH_CONFIGS[:1], indirect=True)
def test_batched_tail_uses_runtime_bounds_after_capture(mesh_device, device_params):
    from models.tt_transformers.tt.model import Transformer

    model = object.__new__(Transformer)
    model.mesh_device = mesh_device
    for name in ("_tail_slice_start", "_tail_slice_end"):
        setattr(
            model,
            name,
            ttnn.from_torch(
                torch.zeros(4, dtype=torch.int32),
                device=mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            ),
        )
    values = torch.arange(2 * 1024 * 32, dtype=torch.float32).to(torch.bfloat16).reshape(2, 1, 1024, 32)
    hidden = ttnn.from_torch(
        values, device=mesh_device, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    )
    warm = model._slice_last_token_tile(hidden, 1023)
    scratch = ttnn.neg(hidden)
    del warm, scratch
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    scratch = ttnn.neg(hidden)
    del scratch
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    entries = mesh_device.num_program_cache_entries()
    try:
        for last in (0, 96, 776, 1023):
            tile = model._slice_last_token_tile(hidden, last)
            snapshot = tile.cpu(blocking=False)
            del tile
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            start = (last // 32) * 32
            torch.testing.assert_close(_host(snapshot), values[:, :, start : start + 32], atol=0, rtol=0)
        assert mesh_device.num_program_cache_entries() == entries
    finally:
        ttnn.release_trace(mesh_device, trace_id)
