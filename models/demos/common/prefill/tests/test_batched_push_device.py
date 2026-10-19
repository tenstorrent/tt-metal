# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Packed H2D forward on a device with no model, run one entry at a time through prefill_chunk."""

import os

# The runner and producer read these at import.
os.environ.setdefault("PREFILL_SP", "4")
os.environ.setdefault("PREFILL_TP", "1")
os.environ.setdefault("PREFILL_CHUNK_SIZE", "256")
os.environ.setdefault("PREFILL_FORWARD_CHUNKS", "2")

import pytest
import torch

import ttnn
from models.demos.common.prefill.chunk_layout import rotate_chunk_tokens
from models.demos.common.prefill.forward_entries import (
    SHUTDOWN_WORD,
    PrefillEntry,
    metadata_size_bytes,
    pack_forward_metadata,
    pad_entries,
    sentinel_metadata,
)
from models.demos.common.prefill.runners import prefill_producer as producer
from models.demos.common.prefill.runners import prefill_runner as runner
from models.demos.common.prefill.runners.runner_utils import build_h2d_service, make_h2d_spec

SP, TP = int(os.environ["PREFILL_SP"]), int(os.environ["PREFILL_TP"])
C = int(os.environ["PREFILL_CHUNK_SIZE"])
F = int(os.environ["PREFILL_FORWARD_CHUNKS"])
W = C // SP
HIDDEN = 64
FILL = [1] * C


def _per_row_ids(t: ttnn.Tensor) -> list:
    devs = ttnn.get_device_tensors(t)
    assert len(devs) == SP * TP
    return [(i // TP, ttnn.to_torch(d).view(torch.int32).flatten().tolist()) for i, d in enumerate(devs)]


def _activation(mesh_device, full: torch.Tensor) -> ttnn.Tensor:
    return ttnn.from_torch(
        full,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.create_mesh_mapper(
            mesh_device, ttnn.MeshMapperConfig(placements=[ttnn.PlacementShard(2), ttnn.PlacementReplicate()])
        ),
    )


def _push(service, chunks, entries, metadata=None):
    rows = producer._h2d_forward_rows(chunks + [FILL] * (F - len(chunks)), [e.start for e in pad_entries(entries, F)])
    service.forward_to_tensor_bytes(rows, metadata=metadata or pack_forward_metadata(pad_entries(entries, F)))


class _Echo:
    def __init__(self):
        self.calls = []

    def prefill_chunk(
        self, inp, kv, *, slot_id, actual_start, actual_end, request_id=0, d2h_service=None, metadata_msg=None
    ):
        self.calls.append((slot_id, actual_start, actual_end, request_id))
        return inp


def _check_joined(out, full, n):
    for i, dev in enumerate(ttnn.get_device_tensors(out)):
        r = i // TP
        got = ttnn.to_torch(dev).float().reshape(F * W, HIDDEN)
        want = full[0, 0, r * F * W : r * F * W + n * W].to(torch.bfloat16).float()
        assert torch.equal(got[: n * W], want), f"chip {i} entries"
        assert torch.all(got[n * W :] == 0), f"chip {i} empty entries"


@pytest.mark.parametrize("mesh_device", [pytest.param((SP, TP), id=f"sp{SP}-tp{TP}")], indirect=True)
def test_packed_push_is_decoded_and_run_per_entry(mesh_device):
    producer._load_env_config()
    assert (runner.FORWARD_CHUNKS, runner.CHUNK_SIZE, runner._sp, runner.FORWARD_TOKENS) == (F, C, SP, F * C)
    assert runner.CHUNK_METADATA_SIZE_BYTES == producer.CHUNK_METADATA_SIZE_BYTES == metadata_size_bytes(F)

    service = build_h2d_service(
        mesh_device,
        global_spec=make_h2d_spec((SP, TP), runner.FORWARD_TOKENS, 0),
        mapper_config=runner.H2D_MAPPER_CONFIG,
        worker_cores=runner.SYNC_WORKER_CORES,
        metadata_size_bytes=runner.CHUNK_METADATA_SIZE_BYTES,
    )

    a, b = list(range(1000, 1000 + C)), list(range(2000, 2000 + C))
    chunks = [a, b][:F]
    entries = [PrefillEntry(1, 64, 64 + 40), PrefillEntry(2, 0, C)][:F]
    _push(service, chunks, entries)

    tt_ids, _mtp, meta, tt_md = runner._socket_next(service)
    assert not runner._is_shutdown_sentinel(meta) and not runner._is_warmup_sentinel(meta)
    assert meta["entries"] == pad_entries(entries, F)
    for k, (toks, e) in enumerate(zip(chunks, entries)):
        sl = runner._entry_rows(tt_ids, k, W)
        rot = rotate_chunk_tokens(toks, e.start, SP)
        for r, got in _per_row_ids(sl):
            assert got == rot[r * W : (r + 1) * W], f"entry {k} row {r}"
        if F > 1:
            ttnn.deallocate(sl)
    ttnn.deallocate(tt_ids)
    ttnn.deallocate(tt_md)

    if F > 1:
        full = torch.arange(SP * F * W * HIDDEN, dtype=torch.float32).reshape(1, 1, SP * F * W, HIDDEN) % 251
        echo = _Echo()
        out = runner._prefill_entries(echo, None, _activation(mesh_device, full), meta["entries"], 7)
        assert echo.calls == [(e.slot, e.start, e.end, 7) for e in entries]
        _check_joined(out, full, len(entries))
        ttnn.deallocate(out)

        # Partial forward: one entry, then empty ones.
        _push(service, [b], [PrefillEntry(3, 0, 100)])
        tt_ids, _mtp, meta, tt_md = runner._socket_next(service)
        assert meta["entries"] == pad_entries([PrefillEntry(3, 0, 100)], F)
        for k in range(1, F):
            pad = runner._entry_rows(tt_ids, k, W)
            for _r, got in _per_row_ids(pad):
                assert set(got) == {1}, f"entry {k} is fill"
            ttnn.deallocate(pad)
        ttnn.deallocate(tt_ids)
        ttnn.deallocate(tt_md)

        echo = _Echo()
        out = runner._prefill_entries(echo, None, _activation(mesh_device, full), meta["entries"], 0)
        assert echo.calls == [(3, 0, 100, 0)]
        _check_joined(out, full, 1)
        ttnn.deallocate(out)

    _push(service, [b], [PrefillEntry(0, 0, C)], metadata=sentinel_metadata(SHUTDOWN_WORD, F))
    tt_ids, _mtp, meta, tt_md = runner._socket_next(service)
    assert runner._is_shutdown_sentinel(meta)
    ttnn.deallocate(tt_ids)
    ttnn.deallocate(tt_md)
