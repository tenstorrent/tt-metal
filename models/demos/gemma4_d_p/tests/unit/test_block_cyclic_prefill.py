# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Block-cyclic request mapping and trace replay, without model weights."""

from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.gemma4_d_p.config import MeshConfig
from models.demos.gemma4_d_p.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gemma4_d_p.tt.attention import ring_prefill
from models.demos.gemma4_d_p.tt.attention.sliding_chunk import SlidingChunk
from models.demos.gemma4_d_p.tt.ccl import CCLManager
from models.demos.gemma4_d_p.tt.model import Gemma4Model, _cp_chunk_major_row_order
from models.demos.gemma4_d_p.tt.prefill_metadata import PrefillMetadata, chunk_positions
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("cp", [4, 8])
@pytest.mark.parametrize("start", [0, 32, 1024, 1056, 7008, 8192, 14368, 32736])
def test_chunk_mapping_and_sliding_groups(cp, start, monkeypatch):
    chunk, capacity = 8192, 32768
    local = chunk // cp
    positions = chunk_positions(start, chunk, cp)
    # Enumerate ownership from absolute tokens, independently of the writer formula.
    tokens = torch.arange(start, start + chunk)
    expected = torch.stack([tokens[(tokens // local) % cp == rank] for rank in range(cp)])
    torch.testing.assert_close(positions, expected)

    monkeypatch.setattr(SlidingChunk, "_stage", lambda self, name, values, seq_dim=None: values)
    sliding = SlidingChunk(SimpleNamespace(cp_degree=cp), chunk, capacity)
    sliding.update(start, positions)
    restore = sliding.output_indices.reshape(cp, local)
    for rank in range(cp):
        q = positions[rank]
        results = []
        for origin, indices in zip(sliding.group_starts, sliding.q_indices):
            idx = indices.reshape(cp, local)[rank]
            # Encode both the supplied Q identity and the absolute SDPA row.
            results.append(torch.stack((q[idx], origin.item() + rank * local + torch.arange(local)), dim=-1))
        restored = torch.cat(results)[restore[rank]]
        valid = positions[rank] < capacity
        torch.testing.assert_close(restored[valid, 0], q[valid])
        torch.testing.assert_close(restored[valid, 1], q[valid])


@pytest.mark.parametrize(
    "slot,start,end,match",
    [
        (-1, 0, 32, "slot_idx"),
        (2, 0, 32, "slot_idx"),
        (0, -32, 32, "actual_start"),
        (0, 7000, 9000, "32-token aligned"),
        (0, 32, 32, "actual_start < actual_end"),
        (0, 32, 8225, "actual_start < actual_end"),
        (0, 16352, 16385, "actual_start < actual_end"),
    ],
)
def test_invalid_request_rejected_before_device_writes(slot, start, end, match, expect_error):
    metadata = object.__new__(PrefillMetadata)
    metadata.num_users, metadata.chunk_size, metadata.max_seq_len = 2, 8192, 16384
    with expect_error(ValueError, match):
        metadata.update(slot_idx=slot, actual_start=start, actual_end=end)


@pytest.mark.parametrize("global_cache", [False, True])
@pytest.mark.parametrize("device_metadata", [False, True])
def test_cache_writer_passes_valid_end(monkeypatch, global_cache, device_metadata):
    calls = []
    monkeypatch.setattr(
        ttnn.experimental.deepseek_prefill, "update_padded_kv_cache", lambda **kwargs: calls.append(kwargs)
    )
    tensor = SimpleNamespace(dtype=ttnn.bfloat8_b)
    metadata = SimpleNamespace(slot_idx=object(), kv_actual_global=object(), actual_end=object())
    kwargs = dict(
        mesh_config=SimpleNamespace(cp_axis=0),
        kv_actual_global=7008,
        actual_end=9000,
        prefill_metadata=metadata if device_metadata else None,
    )
    if global_cache:
        ring_prefill.write_chunk_to_global_ring_cache(tensor, tensor, **kwargs)
    else:
        ring_prefill.write_chunk_to_sliding_ring_cache(tensor, tensor, tensor, tensor, **kwargs)
    assert len(calls) == (1 if global_cache else 2)
    for call in calls:
        assert call["valid_global"] == (metadata.actual_end if device_metadata else 9000)
        assert call["kv_actual_global"] == (metadata.kv_actual_global if device_metadata else 7008)


def test_model_stages_supplied_bounds():
    calls = []
    model = object.__new__(Gemma4Model)
    model.mesh_config = SimpleNamespace(cp_degree=8)
    model.prefill_chunk_size, model.max_seq_len = 8192, 16384
    model._prefill_metadata_external = False
    model._rope_prefill_positions = None
    model.prefill_metadata = SimpleNamespace(update=lambda **kwargs: calls.append(kwargs))
    model.layers = []
    hidden = SimpleNamespace(shape=(1, 1, 1024, 64))
    assert model(hidden, user_id=1, actual_start=7008, actual_end=9000) is hidden
    assert calls == [dict(slot_idx=1, actual_start=7008, actual_end=9000)]


@parametrize_mesh_with_fabric(device_params_extra={"trace_region_size": 32 * 1024 * 1024})
@pytest.mark.parametrize("global_cache", [False, True], ids=["sliding", "global"])
def test_device_block_cyclic_cache_attention_replay(mesh_device, global_cache):
    """Replay with changed users/offsets; compare cache, RoPE and real attention rows."""
    torch.manual_seed(42)
    mesh_config = MeshConfig(mesh_device)
    cp, tp = mesh_config.cp_degree, mesh_config.tp_degree
    chunk, capacity, users = 8192, 16384, 2
    local = chunk // cp
    heads = 8
    kv_heads = 1 if global_cache else 4
    width = 512 if global_cache else 256
    cache_width = 640 if global_cache else width
    metadata = PrefillMetadata(mesh_config, chunk, capacity, users)
    ccl = CCLManager(mesh_config)
    mapper = mesh_config.shard_mapper(mesh_dims=(2, None))

    def host_tensor(x, dtype=ttnn.bfloat16):
        return ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)

    def put(x, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG):
        return ttnn.to_device(host_tensor(x, dtype), mesh_device, memory_config=memory_config)

    order = _cp_chunk_major_row_order(capacity, cp, chunk)
    # All positions are initialized, including poisoned future rows. Only causal real rows may contribute.
    expected_k = torch.randn(users, kv_heads, capacity, cache_width).bfloat16() * 0.1
    expected_v = None if global_cache else torch.randn_like(expected_k)
    memcfg = ring_prefill.migration_ring_memory_config(mesh_device, cache_width)
    cache_k = put(expected_k[:, :, order], ttnn.bfloat8_b, memcfg)
    cache_v = None if global_cache else put(expected_v[:, :, order], ttnn.bfloat8_b, memcfg)
    tt_q = put(torch.zeros(1, heads, chunk, width, dtype=torch.bfloat16))
    tt_k = put(torch.zeros(1, kv_heads, chunk, cache_width, dtype=torch.bfloat16))
    tt_v = None if global_cache else put(torch.zeros(1, kv_heads, chunk, width, dtype=torch.bfloat16))
    rope_table = torch.sin(torch.arange(capacity).float().unsqueeze(1) / torch.arange(1, 33).float()).bfloat16()
    tt_rope = ttnn.from_torch(
        rope_table,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    def forward():
        rope = ttnn.embedding(metadata.positions, tt_rope, layout=ttnn.TILE_LAYOUT)
        kwargs = dict(mesh_config=mesh_config, kv_actual_global=0, prefill_metadata=metadata)
        if global_cache:
            ring_prefill.write_chunk_to_global_ring_cache(cache_k, tt_k, **kwargs)
        else:
            ring_prefill.write_chunk_to_sliding_ring_cache(cache_k, cache_v, tt_k, tt_v, **kwargs)
        args = dict(
            mesh_config=mesh_config,
            ccl_manager=ccl,
            prefill_metadata=metadata,
            num_local_kv_heads=kv_heads,
            max_seq_len=capacity,
            logical_n=capacity,
            kv_actual_global=0,
            scale=width**-0.5,
            compute_kernel_config=ttnn.init_device_compute_kernel_config(
                mesh_device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            ),
        )
        if global_cache:
            out = ring_prefill.global_ring_prefill_attention(tt_q, cache_k, **args)
        else:
            out = ring_prefill.sliding_ring_prefill_attention(
                tt_q,
                cache_k,
                cache_v,
                head_dim=width,
                sliding_window_size=1024,
                **args,
            )
        return out, rope

    # Capture one graph, then change every runtime scalar without recompiling it.
    compiled = forward()
    ttnn.synchronize_device(mesh_device)
    for tensor in compiled:
        tensor.deallocate(True)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    output, rope_output = forward()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    # Warmup wrote zero K/V into user 0's first chunk.
    expected_k[0, :, :chunk] = 0
    if expected_v is not None:
        expected_v[0, :, :chunk] = 0

    previous = {}
    for label, cache in (("k", cache_k), ("v", cache_v)):
        if cache is not None:
            previous[label] = [ttnn.to_torch(shard).float() for shard in ttnn.get_device_tensors(cache)[::tp]]
    buffer_addresses = {key: value.buffer_address() for key, value in metadata._buffers.items()}
    try:
        for slot, start, end in [(0, 0, 1056), (1, 1056, 9000), (0, 7008, 9000), (1, 8192, 12001), (0, 15392, 16381)]:
            positions = chunk_positions(start, chunk, cp)
            flat = positions.flatten()
            q = torch.randn(1, heads, chunk, width).bfloat16() * 0.1
            k = torch.randn(1, kv_heads, chunk, cache_width).bfloat16() * 0.1
            v = None if global_cache else torch.randn(1, kv_heads, chunk, width).bfloat16()
            metadata.update(slot_idx=slot, actual_start=start, actual_end=end)
            for src, dst in ((q, tt_q), (k, tt_k), (v, tt_v)):
                if src is not None:
                    ttnn.copy_host_to_device_tensor(host_tensor(src), dst)
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            assert buffer_addresses == {key: value.buffer_address() for key, value in metadata._buffers.items()}
            written = flat < ((end + 31) // 32 * 32)
            expected_k[slot, :, flat[written]] = k[0, :, written]
            if v is not None:
                expected_v[slot, :, flat[written]] = v[0, :, written]

            output_shards = ttnn.get_device_tensors(output)
            rope_shards = ttnn.get_device_tensors(rope_output)
            k_shards = ttnn.get_device_tensors(cache_k)
            v_shards = None if global_cache else ttnn.get_device_tensors(cache_v)
            for rank in range(cp):
                device_index = rank * tp
                rows = positions[rank]
                actual_rope = ttnn.to_torch(rope_shards[device_index]).reshape(local, 32)
                safe = rows.masked_fill(rows >= capacity, 0)
                torch.testing.assert_close(actual_rope, rope_table[safe], rtol=0, atol=0)
                actual_k = ttnn.to_torch(k_shards[device_index]).float()
                cache_rows = order.reshape(cp, -1)[rank]
                actual_v = None if v_shards is None else ttnn.to_torch(v_shards[device_index]).float()
                for label, expected_cache, actual_cache in (("k", expected_k, actual_k), ("v", expected_v, actual_v)):
                    if actual_cache is None:
                        continue
                    assert_with_pcc(expected_cache[:, :, cache_rows].float(), actual_cache, 0.999)
                    unchanged = (cache_rows < start) | (cache_rows >= (end + 31) // 32 * 32)
                    torch.testing.assert_close(
                        actual_cache[slot, :, unchanged], previous[label][rank][slot, :, unchanged], rtol=0, atol=0
                    )
                    torch.testing.assert_close(actual_cache[1 - slot], previous[label][rank][1 - slot], rtol=0, atol=0)
                    previous[label][rank] = actual_cache
                valid = torch.where(rows < end)[0]
                if not len(valid):
                    continue
                sample = valid[torch.linspace(0, len(valid) - 1, min(16, len(valid))).long()]
                query = q[:, :, rank * local + sample].float()
                keys = expected_k[slot : slot + 1, :, :end, :width].float()
                values = (
                    expected_k[slot : slot + 1, :, :end, 128:] if global_cache else expected_v[slot : slot + 1, :, :end]
                ).float()
                keys = keys.repeat_interleave(heads // kv_heads, dim=1)
                values = values.repeat_interleave(heads // kv_heads, dim=1)
                mask = torch.arange(end)[None, :] <= rows[sample, None]
                if not global_cache:
                    mask &= torch.arange(end)[None, :] > rows[sample, None] - 1024
                expected = torch.nn.functional.scaled_dot_product_attention(query, keys, values, attn_mask=mask)
                actual = ttnn.to_torch(output_shards[device_index])[:, :, sample].float()
                assert_with_pcc(expected, actual, 0.995)
    finally:
        ttnn.release_trace(mesh_device, trace_id)
