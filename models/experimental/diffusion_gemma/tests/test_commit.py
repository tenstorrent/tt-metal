# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Critical device regressions for commit-time KV writes and traced fresh-tail reads."""

import os
from contextlib import contextmanager

import pytest
import torch

import ttnn
from models.experimental.diffusion_gemma.tt.commit_batched import _write_canvas_kv_contiguous


_needs_device = pytest.mark.skipif(
    os.environ.get("DG_RUN_DEVICE") != "1",
    reason="set DG_RUN_DEVICE=1 to run on a Tenstorrent device",
)
TRACE_REGION = 64 << 20


@pytest.fixture(scope="module")
def mesh():
    """Use one session so trace-region configuration is consistent on every chip."""
    shape = os.environ.get("MESH_DEVICE", "P150x4")
    rows, columns = (1, 4) if shape == "P150x4" else (int(shape.split("x")[0]), int(shape.split("x")[1]))
    device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(rows, columns),
        trace_region_size=TRACE_REGION,
    )
    try:
        yield device
    finally:
        ttnn.close_mesh_device(device)


@pytest.fixture(scope="module")
def device(mesh):
    if mesh.get_num_devices() == 1:
        return mesh
    return mesh.create_submesh(ttnn.MeshShape(1, 1))


def _to_device(device, host, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        host,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _write(
    device,
    *,
    mode,
    cache_host,
    canvas_k_host,
    canvas_v_host,
    start_pos,
    canvas_len,
):
    k_cache = _to_device(device, cache_host)
    v_cache = _to_device(device, cache_host)
    canvas_k = _to_device(device, canvas_k_host)
    canvas_v = _to_device(device, canvas_v_host)
    _write_canvas_kv_contiguous(
        k_cache,
        v_cache,
        canvas_k,
        canvas_v,
        start_pos=start_pos,
        canvas_len=canvas_len,
        mesh_device=device,
        write_mode=mode,
    )
    out = ttnn.to_torch(k_cache), ttnn.to_torch(v_cache)
    for tensor in (k_cache, v_cache, canvas_k, canvas_v):
        tensor.deallocate(True)
    return out


# Representative production geometries: sliding KV, full-attention head width,
# and a write ending exactly at max_seq.
GEOMETRIES = [
    pytest.param(2, 256, 1024, 512, 256, id="sliding"),
    pytest.param(1, 512, 1024, 512, 256, id="full-attention"),
    pytest.param(2, 256, 1024, 768, 256, id="last-block"),
]


@_needs_device
@pytest.mark.parametrize(
    "num_kv_heads,head_dim,max_seq,start_pos,canvas_len",
    GEOMETRIES,
)
def test_fill_write_is_bit_identical_to_per_position(
    device,
    num_kv_heads,
    head_dim,
    max_seq,
    start_pos,
    canvas_len,
):
    torch.manual_seed(0)
    cache_host = torch.randn(1, num_kv_heads, max_seq, head_dim).bfloat16()
    canvas_k_host = torch.randn(1, num_kv_heads, canvas_len, head_dim).bfloat16()
    canvas_v_host = torch.randn(1, num_kv_heads, canvas_len, head_dim).bfloat16()
    kwargs = {
        "cache_host": cache_host,
        "canvas_k_host": canvas_k_host,
        "canvas_v_host": canvas_v_host,
        "start_pos": start_pos,
        "canvas_len": canvas_len,
    }

    ref_k, ref_v = _write(device, mode="position", **kwargs)
    fill_k, fill_v = _write(device, mode="fill", **kwargs)

    assert torch.equal(ref_k, fill_k)
    assert torch.equal(ref_v, fill_v)
    for got, canvas in ((fill_k, canvas_k_host), (fill_v, canvas_v_host)):
        expected = cache_host.clone()
        expected[:, :, start_pos : start_pos + canvas_len, :] = canvas
        assert torch.equal(got, expected)


@_needs_device
def test_consecutive_blocks_preserve_each_other(device):
    num_kv_heads, head_dim, max_seq, canvas_len = 2, 256, 1024, 256
    torch.manual_seed(3)
    cache_host = torch.randn(1, num_kv_heads, max_seq, head_dim).bfloat16()
    blocks = [
        (256, torch.randn(1, num_kv_heads, canvas_len, head_dim).bfloat16()),
        (512, torch.randn(1, num_kv_heads, canvas_len, head_dim).bfloat16()),
    ]

    k_cache = _to_device(device, cache_host)
    v_cache = _to_device(device, cache_host)
    expected = cache_host.clone()
    for start_pos, canvas_host in blocks:
        canvas = _to_device(device, canvas_host)
        _write_canvas_kv_contiguous(
            k_cache,
            v_cache,
            canvas,
            canvas,
            start_pos=start_pos,
            canvas_len=canvas_len,
            mesh_device=device,
            write_mode="fill",
        )
        canvas.deallocate(True)
        expected[:, :, start_pos : start_pos + canvas_len, :] = canvas_host

    assert torch.equal(ttnn.to_torch(k_cache), expected)
    assert torch.equal(ttnn.to_torch(v_cache), expected)


P_MAX = 128
CANVAS = 32
NUM_KV_HEADS = 2
HEAD_DIM = 32


@contextmanager
def _capture(mesh_device, *, cq_id=0):
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=cq_id)
    try:
        yield trace_id
    except BaseException:
        try:
            ttnn.end_trace_capture(mesh_device, trace_id, cq_id=cq_id)
            ttnn.release_trace(mesh_device, trace_id)
        except BaseException:
            pass
        raise
    else:
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=cq_id)


def _replicate(mesh_device):
    if mesh_device.get_num_devices() > 1:
        return ttnn.ReplicateTensorToMesh(mesh_device)
    return None


def _to_mesh_replicated(value, mesh_device):
    return ttnn.from_torch(
        value,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=_replicate(mesh_device),
    )


def _first_shard(tensor, mesh_device):
    out = ttnn.to_torch(
        tensor,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )
    return out[:1] if out.shape[0] > 1 else out


def _make_workspace(mesh_device, prefix):
    host = torch.zeros(
        1,
        NUM_KV_HEADS,
        P_MAX + CANVAS,
        HEAD_DIM,
        dtype=torch.bfloat16,
    )
    host[:, :, :P_MAX, :] = prefix
    return _to_mesh_replicated(host, mesh_device)


@_needs_device
def test_traced_fill_then_read_sees_each_replays_own_write(mesh):
    """A replayed trace must observe the tail written by that same replay."""
    prefix = torch.randn(1, NUM_KV_HEADS, P_MAX, HEAD_DIM).to(torch.bfloat16)
    cache = _make_workspace(mesh, prefix)
    cache_address = cache.buffer_address()
    canvas_a = torch.randn(1, NUM_KV_HEADS, CANVAS, HEAD_DIM).to(torch.bfloat16)
    canvas_b = torch.randn(1, NUM_KV_HEADS, CANVAS, HEAD_DIM).to(torch.bfloat16)
    canvas_buffer = _to_mesh_replicated(canvas_a, mesh)

    ttnn.fill_cache(cache, canvas_buffer, 0, update_idx=P_MAX)
    ttnn.clone(cache).deallocate(True)
    with _capture(mesh) as trace_id:
        ttnn.fill_cache(cache, canvas_buffer, 0, update_idx=P_MAX)
        out_buffer = ttnn.clone(cache)

    results = {}
    for name, canvas in (("a", canvas_a), ("b", canvas_b)):
        fresh = _to_mesh_replicated(canvas, mesh)
        ttnn.copy(fresh, canvas_buffer)
        fresh.deallocate(True)
        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
        results[name] = _first_shard(out_buffer, mesh).to(torch.float32).clone()
    ttnn.release_trace(mesh, trace_id)

    for name, canvas in (("a", canvas_a), ("b", canvas_b)):
        assert torch.equal(results[name][:, :, :P_MAX, :], prefix.float())
        assert torch.equal(results[name][:, :, P_MAX:, :], canvas.float())
    assert not torch.equal(results["a"], results["b"])
    assert cache.buffer_address() == cache_address


@_needs_device
def test_traced_fill_then_sdpa_reads_the_fresh_tail(mesh):
    """Pin fresh traced writes through the production SDPA consumer."""
    torch.manual_seed(0)
    prefix = torch.randn(1, NUM_KV_HEADS, P_MAX, HEAD_DIM).to(torch.bfloat16)
    cache_k = _make_workspace(mesh, prefix)
    cache_v = _make_workspace(mesh, prefix)
    query = torch.randn(1, NUM_KV_HEADS, CANVAS, HEAD_DIM).to(torch.bfloat16)
    tt_query = _to_mesh_replicated(query, mesh)
    canvas_a = torch.randn(1, NUM_KV_HEADS, CANVAS, HEAD_DIM).to(torch.bfloat16)
    canvas_b = torch.randn(1, NUM_KV_HEADS, CANVAS, HEAD_DIM).to(torch.bfloat16)
    canvas_buffer_k = _to_mesh_replicated(canvas_a, mesh)
    canvas_buffer_v = _to_mesh_replicated(canvas_a, mesh)
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 1),
        q_chunk_size=CANVAS,
        k_chunk_size=32,
        exp_approx_mode=False,
    )

    def fill_and_attend():
        ttnn.fill_cache(cache_k, canvas_buffer_k, 0, update_idx=P_MAX)
        ttnn.fill_cache(cache_v, canvas_buffer_v, 0, update_idx=P_MAX)
        return ttnn.transformer.scaled_dot_product_attention(
            tt_query,
            cache_k,
            cache_v,
            is_causal=False,
            scale=1.0,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=program_config,
        )

    fill_and_attend().deallocate(True)
    with _capture(mesh) as trace_id:
        sdpa_out = fill_and_attend()

    outputs = {}
    for name, canvas in (("a", canvas_a), ("b", canvas_b)):
        for buffer in (canvas_buffer_k, canvas_buffer_v):
            fresh = _to_mesh_replicated(canvas, mesh)
            ttnn.copy(fresh, buffer)
            fresh.deallocate(True)
        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
        outputs[name] = _first_shard(sdpa_out, mesh).to(torch.float32).clone()
    ttnn.release_trace(mesh, trace_id)

    for name, canvas in (("a", canvas_a), ("b", canvas_b)):
        kv_reference = torch.cat([prefix, canvas], dim=2).float()
        reference = torch.nn.functional.scaled_dot_product_attention(
            query.float(),
            kv_reference,
            kv_reference,
            scale=1.0,
        )
        pcc = torch.corrcoef(torch.stack([outputs[name].flatten(), reference.flatten()]))[0, 1].item()
        assert pcc > 0.99
    assert not torch.equal(outputs["a"], outputs["b"])
