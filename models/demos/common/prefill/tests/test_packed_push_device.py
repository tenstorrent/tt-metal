# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Packed H2D forward on a device with no model, run one chunk at a time through prefill_chunk."""

import os

# The runner reads these at import.
os.environ.setdefault("PREFILL_SP", "4")
os.environ.setdefault("PREFILL_TP", "1")
os.environ.setdefault("PREFILL_CHUNK_SIZE", "256")
os.environ.setdefault("PREFILL_FORWARD_TOKENS_BUDGET", "512")

import pytest
import torch

import ttnn
from models.demos.common.prefill.chunk_layout import pack_forward_tokens, rotate_chunk_tokens
from models.demos.common.prefill.prefill_chunk import (
    SHUTDOWN_WORD,
    PrefillChunk,
    metadata_size_bytes,
    pack_forward_metadata,
    pad_chunks,
    sentinel_metadata,
)
from models.demos.common.prefill.runners import prefill_runner as runner
from models.demos.common.prefill.runners.runner_utils import build_h2d_service, make_h2d_spec

SP, TP = int(os.environ["PREFILL_SP"]), int(os.environ["PREFILL_TP"])
C = int(os.environ["PREFILL_CHUNK_SIZE"])
F = int(os.environ["PREFILL_FORWARD_TOKENS_BUDGET"]) // C
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


def _push(service, tokens, chunks, metadata=None):
    starts = [c.actual_start for c in pad_chunks(chunks, F)]
    packed = pack_forward_tokens(tokens + [FILL] * (F - len(tokens)), starts, SP)
    rows = torch.tensor(packed, dtype=torch.int64).view(SP, 1, F * W).to(torch.uint32).contiguous().numpy()
    service.forward_to_tensor_bytes(rows, metadata=metadata or pack_forward_metadata(pad_chunks(chunks, F), F))


def _next(service):
    sockets = runner.SocketBuffers(metadata=None)
    tokens, _mtp, meta = runner._socket_next(service, sockets)
    return tokens, meta, sockets.metadata


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
        assert torch.equal(got[: n * W], want), f"chip {i} chunks"
        assert torch.all(got[n * W :] == 0), f"chip {i} empty chunks"


@pytest.mark.parametrize("mesh_device", [pytest.param((SP, TP), id=f"sp{SP}-tp{TP}")], indirect=True)
def test_packed_push_is_decoded_and_run_per_chunk(mesh_device):
    assert (runner.FORWARD_CHUNKS, runner.CHUNK_SIZE, runner._sp, runner.FORWARD_TOKENS_BUDGET) == (F, C, SP, F * C)
    assert runner.CHUNK_METADATA_SIZE_BYTES == metadata_size_bytes(F)

    service = build_h2d_service(
        mesh_device,
        global_spec=make_h2d_spec((SP, TP), runner.FORWARD_TOKENS_BUDGET, 0),
        mapper_config=runner.H2D_MAPPER_CONFIG,
        worker_cores=runner.SYNC_WORKER_CORES,
        metadata_size_bytes=runner.CHUNK_METADATA_SIZE_BYTES,
    )

    a, b = list(range(1000, 1000 + C)), list(range(2000, 2000 + C))
    tokens = [a, b][:F]
    chunks = [PrefillChunk(1, 64, 64 + 40), PrefillChunk(2, 0, C)][:F]
    _push(service, tokens, chunks)

    tt_ids, meta, tt_md = _next(service)
    assert not runner._is_shutdown_sentinel(meta) and not runner._is_warmup_sentinel(meta)
    assert meta["chunks"] == pad_chunks(chunks, F)
    for k, (toks, c) in enumerate(zip(tokens, chunks)):
        sl = runner._chunk_rows(tt_ids, k, W)
        rot = rotate_chunk_tokens(toks, c.actual_start, SP)
        for r, got in _per_row_ids(sl):
            assert got == rot[r * W : (r + 1) * W], f"chunk {k} row {r}"
        if F > 1:
            ttnn.deallocate(sl)
    ttnn.deallocate(tt_ids)
    ttnn.deallocate(tt_md)

    if F > 1:
        full = torch.arange(SP * F * W * HIDDEN, dtype=torch.float32).reshape(1, 1, SP * F * W, HIDDEN) % 251
        echo = _Echo()
        out = runner._prefill_chunks(echo, None, _activation(mesh_device, full), meta["chunks"], 7)
        assert echo.calls == [(c.slot_id, c.actual_start, c.actual_end, 7 + k) for k, c in enumerate(chunks)]
        _check_joined(out, full, len(chunks))
        ttnn.deallocate(out)

        # Partial forward: one chunk, then empty ones.
        _push(service, [b], [PrefillChunk(3, 0, 100)])
        tt_ids, meta, tt_md = _next(service)
        assert meta["chunks"] == pad_chunks([PrefillChunk(3, 0, 100)], F)
        for k in range(1, F):
            pad = runner._chunk_rows(tt_ids, k, W)
            for _r, got in _per_row_ids(pad):
                assert set(got) == {1}, f"chunk {k} is fill"
            ttnn.deallocate(pad)
        ttnn.deallocate(tt_ids)
        ttnn.deallocate(tt_md)

        echo = _Echo()
        out = runner._prefill_chunks(echo, None, _activation(mesh_device, full), meta["chunks"], 0)
        assert echo.calls == [(3, 0, 100, 0)]
        _check_joined(out, full, 1)
        ttnn.deallocate(out)

    _push(service, [b], [PrefillChunk(0, 0, C)], metadata=sentinel_metadata(SHUTDOWN_WORD, F))
    tt_ids, meta, tt_md = _next(service)
    assert runner._is_shutdown_sentinel(meta)
    ttnn.deallocate(tt_ids)
    ttnn.deallocate(tt_md)
