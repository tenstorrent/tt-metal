# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Batched H2D push on a device with no model."""

import os

# The runner and producer read these at import.
os.environ.setdefault("PREFILL_SP", "4")
os.environ.setdefault("PREFILL_TP", "1")
os.environ.setdefault("PREFILL_CHUNK_SIZE", "256")
os.environ.setdefault("PREFILL_MAX_CHUNKS_PER_BATCH", "2")

import pytest
import torch

import ttnn
from models.demos.common.prefill.batch_metadata import INVALID_SLOT, SHUTDOWN_WORD, decode, sentinel
from models.demos.common.prefill.chunk_layout import rotate_chunk_tokens
from models.demos.common.prefill.prefill_entry import PrefillEntry, pad_entries
from models.demos.common.prefill.runners import prefill_producer as producer
from models.demos.common.prefill.runners import prefill_runner as runner
from models.demos.common.prefill.runners.runner_utils import build_h2d_service, make_h2d_spec

SP, TP = int(os.environ["PREFILL_SP"]), int(os.environ["PREFILL_TP"])
C = int(os.environ["PREFILL_CHUNK_SIZE"])
K = int(os.environ["PREFILL_MAX_CHUNKS_PER_BATCH"])
W = C // SP
HIDDEN = 64


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


class _Echo:
    class config:
        is_last_rank = False

    def prefill_chunk(
        self, inp, kv, *, slot_id, actual_start, actual_end, request_id=0, d2h_service=None, metadata_msg=None
    ):
        self.calls = getattr(self, "calls", []) + [(slot_id, actual_start, actual_end, request_id)]
        return inp


class _Forward:
    def prefill_forward(self, inp, kv, entries, *, request_id, d2h_service, metadata_msg):
        self.got = (inp, entries, request_id, metadata_msg)
        return "activation"


def test_prefill_forward_gets_the_push_padded_to_k_entries():
    model, e = _Forward(), PrefillEntry(1, 0, C)
    assert runner._prefill_entries(model, None, "inp", [e], 7, metadata_msg="md") == "activation"
    assert model.got == ("inp", pad_entries([e], K), 7, "md")


@pytest.mark.parametrize("mesh_device", [pytest.param((SP, TP), id=f"sp{SP}-tp{TP}")], indirect=True)
def test_batched_push_is_decoded_and_sliced_per_record(mesh_device):
    producer._load_env_config()
    assert (runner.MAX_CHUNKS_PER_BATCH, runner.CHUNK_SIZE, runner._sp, runner.FORWARD_TOKENS) == (K, C, SP, K * C)
    assert runner.CHUNK_METADATA_SIZE_BYTES == producer.CHUNK_METADATA_SIZE_BYTES == (12 if K == 1 else 200)

    service = build_h2d_service(
        mesh_device,
        global_spec=make_h2d_spec((SP, TP), runner.FORWARD_TOKENS, 0),
        mapper_config=runner.H2D_MAPPER_CONFIG,
        worker_cores=runner.SYNC_WORKER_CORES,
        metadata_size_bytes=runner.CHUNK_METADATA_SIZE_BYTES,
    )

    try:
        a, b = list(range(1000, 1000 + C)), list(range(2000, 2000 + C))
        chunks = [(a, 49), (b, 0)][:K]
        records = [(1, 49, 49 + 40), (2, 0, C)][:K]
        service.forward_to_tensor_bytes(producer._h2d_rows_batch(chunks), metadata=producer._pack_metadata(*records))

        tt_ids, _mtp, words, tt_md = runner._socket_next(service)
        assert not runner._is_shutdown_sentinel(words) and not runner._is_warmup_sentinel(words)
        entries = [PrefillEntry(*r) for r in records]
        assert decode(words, K) == entries
        for k, (toks, start) in enumerate(chunks):
            sl = runner._record_rows(tt_ids, k, W)
            rot = rotate_chunk_tokens(toks, start, SP)
            for r, got in _per_row_ids(sl):
                assert got == rot[r * W : (r + 1) * W], f"record {k} row {r}"
            if K > 1:
                ttnn.deallocate(sl)
        ttnn.deallocate(tt_ids)

        if K > 1:
            full = torch.arange(SP * K * W * HIDDEN, dtype=torch.float32).reshape(1, 1, SP * K * W, HIDDEN) % 251
            n = len(entries)
            echo = _Echo()
            out = runner._prefill_entries(echo, None, _activation(mesh_device, full), entries, 7, metadata_msg=tt_md)
            assert [c[3] for c in echo.calls] == list(range(7, 7 + n))
            for i, dev in enumerate(ttnn.get_device_tensors(out)):
                r = i // TP
                got = ttnn.to_torch(dev).float().reshape(K * W, HIDDEN)
                want = full[0, 0, r * K * W : r * K * W + n * W].to(torch.bfloat16).float()
                assert torch.equal(got[: n * W], want), f"chip {i} records"
                assert torch.all(got[n * W :] == 0), f"chip {i} unfilled columns"
            ttnn.deallocate(out)

            # Partial push.
            service.forward_to_tensor_bytes(
                producer._h2d_rows_batch([(b, 0)]), metadata=producer._pack_metadata((3, 0, 100))
            )
            tt_ids, _mtp, words, tt_md2 = runner._socket_next(service)
            assert decode(words, K) == [PrefillEntry(3, 0, 100)]
            assert words[0] == 1 and (words[5] & 0xFFFFFFFF) == INVALID_SLOT
            for k in range(1, K):
                pad = runner._record_rows(tt_ids, k, W)
                for _r, got in _per_row_ids(pad):
                    assert set(got) == {-1}, f"column {k} is PAD"
                ttnn.deallocate(pad)
            ttnn.deallocate(tt_ids)

            out = runner._prefill_entries(
                _Echo(), None, _activation(mesh_device, full), entries[:1], 0, metadata_msg=tt_md2
            )
            for i, dev in enumerate(ttnn.get_device_tensors(out)):
                r = i // TP
                got = ttnn.to_torch(dev).float().reshape(K * W, HIDDEN)
                want = full[0, 0, r * K * W : r * K * W + W].to(torch.bfloat16).float()
                assert torch.equal(got[:W], want), f"chip {i} record 0"
                assert torch.all(got[W:] == 0), f"chip {i} zero columns"
            ttnn.deallocate(out)
            ttnn.deallocate(tt_md2)

        # Shutdown sentinel.
        service.forward_to_tensor_bytes(producer._h2d_rows_batch([(b, 0)]), metadata=sentinel(SHUTDOWN_WORD, K))
        tt_ids, _mtp, words, tt_md3 = runner._socket_next(service)
        assert runner._is_shutdown_sentinel(words)
        for t in (tt_ids, tt_md, tt_md3):
            ttnn.deallocate(t)
    finally:
        service = None
