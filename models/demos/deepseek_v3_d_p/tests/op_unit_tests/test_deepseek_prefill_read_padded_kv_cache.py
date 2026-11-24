# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Device test for ttnn.experimental.deepseek_prefill.read_padded_kv_cache, the read mirror of update_padded_kv_cache.

A chunk's TP-deduped write (``update_padded_kv_cache(cluster_axis=0, tp_axis=1)``) stores ``chunk / (sp * tp)`` rows
per chip at a per-chip offset. ``read_padded_kv_cache`` returns those rows for a layer range, and writing them into
a second cache with ``update_padded_kv_cache`` in complete-mesh mode (``cluster_axis=None``, one layer at a time by
``input_row_offset`` / ``input_rows``) at the same slot / start / end must reproduce the first cache bit for bit over
the read layers. This is how a pipeline stage forwards a chunk's stored KV rows to a downstream copy of the cache.

One schedule of chunks over two user slots covers: chunk-aligned starts, a 32-row-aligned start that is not
chunk-aligned (rotated per-chip offsets) and partial ends; an FP8 cache read as its bytes (UINT8 rows) and written
back; reads into a fresh output and into a caller's output; and fresh metadata tensors half way, so program-cache
hits must patch the new metadata and output addresses. The layer the read skips must stay untouched in the copy.
"""

from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, init_mla_kv_cache

NUM_USERS = 2
NUM_LAYERS = 3
FIRST_READ_LAYER, NUM_READ_LAYERS = 1, 2  # layer 0 is written but never read


def _chunk_and_schedule(chips: int) -> tuple[int, list]:
    """(chunk, schedule): a chunk of 4 tiles per chip, and (slot, start, end) chunks in execution order. Slot 1
    resumes at 96, 32-aligned but not chunk-aligned, so its chunks' per-chip offsets rotate; ends at 96 and inside
    a chunk are partial."""
    chunk = 4 * ttnn.TILE_SIZE * chips
    schedule = [
        (0, 0, chunk),
        (1, 0, 96),
        (0, chunk, 2 * chunk),
        (1, 96, 96 + chunk),
        (0, 2 * chunk, 2 * chunk + 3 * ttnn.TILE_SIZE + 5),
        (1, 96 + chunk, 96 + chunk + chunk // 2),
    ]
    return chunk, schedule


def _scalar(mesh_device, value: int) -> ttnn.Tensor:
    return ttnn.from_torch(
        torch.tensor([value], dtype=torch.int64).reshape(1, 1, 1, 1),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _set(dst: ttnn.Tensor, value: int) -> None:
    host = ttnn.from_torch(
        torch.tensor([value], dtype=torch.int64).reshape(tuple(dst.shape)),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(dst.device()),
    )
    ttnn.copy_host_to_device_tensor(host, dst)


def _chip_bytes(t: ttnn.Tensor) -> list[torch.Tensor]:
    """Each chip's host copy as raw bytes, in mesh order (single-device to_torch rejects FP8)."""
    mesh = t.device()
    whole = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
    return [c.contiguous().view(torch.uint8) for c in torch.chunk(whole, mesh.get_num_devices(), dim=0)]


def _fp8_cache(mesh_device, seq_len: int) -> ttnn.Tensor:
    """A TP-deduped scaled-FP8 KVPE cache with SP on mesh axis 0 and TP on axis 1, slots on DRAM bank 0."""
    return init_mla_kv_cache(
        cache_format=MlaKvCacheFormat.SCALED_FP8,
        hf_config=SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64),
        mesh_device=mesh_device,
        seq_len=seq_len,
        mesh_shape=list(mesh_device.shape),
        sp_axis=0,
        num_kvpe_cache_layers=NUM_LAYERS,
        num_users=NUM_USERS,
        tp_axis=1,
        bank_aligned=True,
    ).storage


@pytest.mark.parametrize(
    "mesh_device",
    [
        # SP 2 x TP 4: the LoudBox op leg.
        pytest.param(
            (2, 4), marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"), id="2x4"
        ),
        # SP 4 x TP 4 with SP on mesh axis 0: a Galaxy tray half (run_4x4_stage_tests.sh), the GLM-5.4 stage layout.
        pytest.param(
            (4, 4), marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 4), topology="mesh-4x4"), id="4x4-sp0"
        ),
    ],
    indirect=True,
)
@pytest.mark.timeout(600)
def test_read_padded_kv_cache_round_trip(mesh_device):
    if not is_blackhole():
        pytest.skip("FP8_E4M3 is Blackhole-only")
    mesh_device.enable_program_cache()
    sp, tp = tuple(mesh_device.shape)
    chunk, schedule = _chunk_and_schedule(sp * tp)
    rows_per_chip = chunk // (sp * tp)
    seq_len = 8192  # a multiple of the chunk and of 8192 (bank-aligned slots)
    assert seq_len % chunk == 0 and max(end for _, _, end in schedule) <= seq_len
    source, copy = _fp8_cache(mesh_device, seq_len), _fp8_cache(mesh_device, seq_len)
    width = int(source.shape[-1])
    assert source.dtype == ttnn.fp8_e4m3

    # Rows over SP, replicated over TP: the TP-deduped write keeps one copy of each row.
    rows_mapper = ttnn.create_mesh_mapper(
        mesh_device, ttnn.MeshMapperConfig(placements=[ttnn.PlacementShard(2), ttnn.PlacementReplicate()])
    )
    caller_output = ttnn.allocate_tensor_on_device(
        ttnn.TensorSpec(
            shape=ttnn.Shape([1, 1, NUM_READ_LAYERS * rows_per_chip, width]),
            dtype=ttnn.uint8,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            buffer_type=ttnn.BufferType.DRAM,
        ),
        mesh_device,
    )
    meta = [_scalar(mesh_device, 0) for _ in range(3)]  # slot, start, end
    generator = torch.Generator().manual_seed(27)
    warm_entries = None
    for n, (slot, start, end) in enumerate(schedule):
        if n == len(schedule) // 2:
            meta = [_scalar(mesh_device, 0) for _ in range(3)]  # fresh addresses on a program-cache hit
        for dst, value in zip(meta, (slot, start, end)):
            _set(dst, value)
        for layer in range(NUM_LAYERS):
            rows = torch.randn(1, 1, chunk, width, generator=generator).to(torch.bfloat16)
            bf16 = ttnn.from_torch(
                rows, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=rows_mapper
            )
            fp8 = ttnn.typecast(bf16, ttnn.fp8_e4m3)
            ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                source,
                fp8,
                meta[0],
                meta[1],
                layer_idx=layer,
                num_layers=NUM_LAYERS,
                cluster_axis=0,
                valid_global=meta[2],
                tp_axis=1,
            )
            ttnn.deallocate(bf16)
            ttnn.deallocate(fp8)
        fresh_output = n % 2 == 1
        read = ttnn.experimental.deepseek_prefill.read_padded_kv_cache(
            source,
            meta[0],
            meta[1],
            first_layer=FIRST_READ_LAYER,
            num_read_layers=NUM_READ_LAYERS,
            num_layers=NUM_LAYERS,
            chunk_size_global=chunk,
            cluster_axis=0,
            tp_axis=1,
            output=None if fresh_output else caller_output,
        )
        assert read.dtype == ttnn.uint8, f"an FP8 cache is read as its bytes, got {read.dtype}"
        assert list(read.shape) == [1, 1, NUM_READ_LAYERS * rows_per_chip, width]
        # Complete-mesh mode checks that the rows are distributed like the cache's sequence dim.
        read.update_tensor_topology(copy.tensor_topology())
        for k in range(NUM_READ_LAYERS):
            ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                copy,
                read,
                meta[0],
                meta[1],
                layer_idx=FIRST_READ_LAYER + k,
                num_layers=NUM_LAYERS,
                cluster_axis=None,
                valid_global=meta[2],
                input_row_offset=k * rows_per_chip,
                input_rows=rows_per_chip,
            )
        if fresh_output:
            ttnn.deallocate(read)
        if n == 1:
            warm_entries = mesh_device.num_program_cache_entries()  # both output modes compiled
    ttnn.synchronize_device(mesh_device)
    assert mesh_device.num_program_cache_entries() == warm_entries, "a later chunk compiled a new program"

    for chip, (a, b) in enumerate(zip(_chip_bytes(source), _chip_bytes(copy))):
        for entry in range(NUM_USERS * NUM_LAYERS):
            layer = entry % NUM_LAYERS
            if FIRST_READ_LAYER <= layer < FIRST_READ_LAYER + NUM_READ_LAYERS:
                assert torch.count_nonzero(a[entry]) > 0, f"chip {chip} entry {entry}: nothing written"
                assert torch.equal(a[entry], b[entry]), f"chip {chip} entry {entry}: copy differs from source"
            else:
                assert torch.count_nonzero(b[entry]) == 0, f"chip {chip} entry {entry}: unread layer was written"
