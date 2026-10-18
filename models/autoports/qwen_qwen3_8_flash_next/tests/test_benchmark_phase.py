# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host evidence handshake and async-stale input position accounting."""

import json
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.autoports.qwen_qwen3_8_flash_next.tools.benchmark_phase import PhaseObserver, execution_metadata


@pytest.mark.parametrize("prefill_arrays", [False, True], ids=["python-ends", "numpy-prefill"])
def test_flush_and_stale_decode_positions(tmp_path, prefill_arrays):
    path = tmp_path / "phase.jsonl"
    model = SimpleNamespace(
        batch_size=32,
        context=262144,
        mesh=SimpleNamespace(get_device_ids=lambda: [0, 1, 2, 3]),
        generator=SimpleNamespace(
            model=SimpleNamespace(
                layer_indices=list(range(48)),
                layers=[SimpleNamespace(kind="indexed_attention", attention=SimpleNamespace(indexer_prefix_cap=True))]
                * 48,
            ),
            prefill_view=SimpleNamespace(runtime_summary=lambda: {"execution_batch": 1, "server_slots": 32}),
        ),
    )
    observer = PhaseObserver(path, model)
    scheduler = SimpleNamespace(num_scheduled_tokens={"request": 8}, finished_req_ids=set())
    model_input = SimpleNamespace(
        row_req_ids=["request"],
        input_positions=torch.tensor([4088]),
        prompt_lens=[4096],
        input_tokens=torch.zeros(32, 4096),
    )
    if prefill_arrays:
        # The real runner builds prefill starts and absolute ends from NumPy
        # scheduler buffers, while decode positions are Torch tensors.
        model_input.input_positions = np.array([4088], dtype=np.int64)
        model_input.prompt_lens = np.array([4096], dtype=np.int64)
        scheduler.num_scheduled_tokens["request"] = np.int64(8)
    phase = observer.begin(model_input, scheduler, time.monotonic_ns())
    phase.mark_submitted()
    chunks = [{"batch": 1, "logical_row": 0, "chunk_length": 8, "start_positions": [4088], "active": [True]}]
    observer.record_prefill_chunks(chunks)
    chunks[0]["start_positions"][0] = 0
    phase.complete()
    model_input.prompt_lens = None
    model_input.input_positions = torch.tensor([4096])
    scheduler.num_scheduled_tokens["request"] = 1
    for _ in range(2):
        phase = observer.begin(model_input, scheduler, time.monotonic_ns())
        phase.mark_submitted()
        phase.complete()
    request = path.with_name(path.name + ".flush")
    ack = path.with_name(path.name + ".flushed")
    request.write_text("test-flush")
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if ack.exists() and json.loads(ack.read_text())["token"] == "test-flush":
            break
        time.sleep(0.01)
    else:
        raise AssertionError("Live flush did not acknowledge")
    raw_events = [json.loads(line) for line in path.read_text().splitlines()]
    events = [e for e in raw_events if e["event"] != "submitted"]
    assert observer.error is None
    assert sum(e["event"] == "submitted" for e in raw_events) == 3
    assert events[0]["execution_version"] == "unqualified_pooled_gather"
    assert events[0]["indexer_prefix_cap"] is True
    assert events[1]["physical_chunks"][0]["start_positions"] == [4088]
    assert events[1]["prompt_ends"] == [4096]
    assert events[1]["scheduled_tokens"] == {"request": 8}
    assert [e["positions"] for e in events[1:]] == [[4088], [4096], [4097]]
    assert [e["suffix_lengths"] for e in events[1:]] == [[8], [1], [1]]
    assert all(e["start_ns"] <= e["submit_ns"] <= e["complete_ns"] for e in events[1:])
    assert events[-1]["previous_complete_ns"] == events[-2]["complete_ns"]


@pytest.mark.parametrize("batch,broadcast", [(1, False), (32, False), (32, True), (1, True)])
def test_broadcast_execution_version_attests_owner_and_view(batch, broadcast):
    layers = [
        SimpleNamespace(
            kind="indexed_attention" if i % 4 == 3 else "linear_attention",
            attention=SimpleNamespace(
                recurrence_broadcast=broadcast, _pooled_gather_partition=dict(kind="cyclic", work=8192, workers=110)
            ),
        )
        for i in range(48)
    ]
    model = SimpleNamespace(
        batch_size=batch,
        generator=SimpleNamespace(model=SimpleNamespace(layer_indices=list(range(48)), layers=layers)),
    )
    flags = [dict(layer=i, enabled=broadcast) for i in range(48) if i % 4 != 3]
    partitions = [dict(layer=i, kind="cyclic", work=8192, workers=110) for i in range(3, 48, 4)]
    view = dict(gdn_recurrence_broadcast=flags, pooled_gather_partitions=partitions) if batch == 32 else None
    result = execution_metadata(model, view, batch == 32)
    expected = (
        ("compact_view_prefix_broadcast_cyclic_v4" if broadcast else "unqualified_pooled_gather")
        if batch == 32
        else ("unqualified_pooled_gather" if broadcast else "native_b1_gdn_cyclic_v3")
    )
    assert result["execution_version"] == expected
    assert result["gdn_recurrence_broadcast"] == flags
    if batch == 32 and broadcast:
        view["gdn_recurrence_broadcast"] = flags[:-1]
        assert execution_metadata(model, view, True)["execution_version"] == "unqualified_pooled_gather"
        view["gdn_recurrence_broadcast"] = flags
        layers[0].attention.recurrence_broadcast = False
        assert execution_metadata(model, view, True)["execution_version"] == "unqualified_pooled_gather"


@pytest.mark.parametrize("batch", [1, 32])
def test_gather_partition_ready_requires_all_original_owner_and_view_layers(batch):
    layers = [
        SimpleNamespace(
            kind="indexed_attention" if i % 4 == 3 else "linear_attention",
            attention=SimpleNamespace(
                recurrence_broadcast=batch == 32, _pooled_gather_partition=dict(kind="cyclic", work=8192, workers=110)
            ),
        )
        for i in range(48)
    ]
    model = SimpleNamespace(
        batch_size=batch, generator=SimpleNamespace(model=SimpleNamespace(layer_indices=list(range(48)), layers=layers))
    )
    partitions = [dict(layer=i, kind="cyclic", work=8192, workers=110) for i in range(3, 48, 4)]
    view = (
        dict(
            gdn_recurrence_broadcast=[dict(layer=i, enabled=True) for i in range(48) if i % 4 != 3],
            pooled_gather_partitions=partitions,
        )
        if batch == 32
        else None
    )
    assert execution_metadata(model, view, batch == 32)["pooled_gather_partitions"] == partitions
    if view is not None:
        view["pooled_gather_partitions"] = partitions[:-1]
        assert execution_metadata(model, view, True)["execution_version"] == "unqualified_pooled_gather"
        view["pooled_gather_partitions"] = partitions
    del layers[3].attention._pooled_gather_partition
    assert execution_metadata(model, view, batch == 32)["execution_version"] == "unqualified_pooled_gather"


def test_decode_geometry_snapshot_precedes_async_submission_and_survives_next_step():
    import queue

    observer = object.__new__(PhaseObserver)
    observer.error = None
    observer.physical_batch = 32
    observer.device_ids = [0, 1, 2, 3]
    observer.sequence = 0
    observer.positions = {"one": 4096, "two": 5000}
    observer.pending_prefill = observer.pending_decode = None
    observer.records = queue.SimpleQueue()
    scheduler = SimpleNamespace(num_scheduled_tokens={"one": 1, "two": 1}, finished_req_ids=set())
    model_input = SimpleNamespace(
        row_req_ids=["one"], input_positions=torch.tensor([9]), prompt_lens=None, input_tokens=torch.zeros(32, 1)
    )
    first = observer.begin(model_input, scheduler, time.monotonic_ns())
    active = [True] + [False] * 31
    observer.record_decode_execution(4, active_rows=active)
    first.mark_submitted()
    active[1] = True
    model_input.row_req_ids = ["two"]
    second = observer.begin(model_input, scheduler, time.monotonic_ns())
    observer.record_decode_execution(32, active_rows=[False, True] + [False] * 30)
    second.mark_submitted()
    records = [observer.records.get(), observer.records.get()]
    assert [r["physical_batch"] for r in records] == [32, 32]
    assert [r["decode_execution_batch"] for r in records] == [4, 32]
    assert records[0]["decode_execution_positions"] == [4096, 0, 0, 0]
    assert records[1]["decode_execution_positions"] == [0, 5000] + [0] * 30
    assert records[0]["decode_active_rows"] == [True] + [False] * 31
    assert all(r["sampler_batch"] == 32 for r in records)


def test_native_b1_conv_metadata_keeps_original_path():
    from models.autoports.qwen_qwen3_8_flash_next.tools.benchmark_work import NATIVE_CONV_VERSION

    layers = [
        SimpleNamespace(
            kind="indexed_attention" if i % 4 == 3 else "linear_attention",
            attention=SimpleNamespace(
                recurrence_broadcast=False,
                _decode_conv_batch_packed=False,
                _pooled_gather_partition=dict(kind="cyclic", work=8192, workers=110),
            ),
        )
        for i in range(48)
    ]
    model = SimpleNamespace(
        batch_size=1, generator=SimpleNamespace(model=SimpleNamespace(layer_indices=list(range(48)), layers=layers))
    )
    assert execution_metadata(model, None, False)["execution_version"] == NATIVE_CONV_VERSION
    for layer in layers:
        if layer.kind == "linear_attention":
            layer.attention._decode_conv_history_packed = False
    assert execution_metadata(model, None, False)["execution_version"] == NATIVE_CONV_VERSION
    layers[0].attention._decode_conv_history_packed = True
    assert execution_metadata(model, None, False)["execution_version"] == "unqualified_history_conv"
    for layer in layers:
        if layer.kind == "linear_attention":
            del layer.attention._decode_conv_history_packed
    layers[0].attention._decode_conv_batch_packed = True
    assert execution_metadata(model, None, False)["execution_version"] == "unqualified_conv_pack"
