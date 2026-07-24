# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""On-device equivalence gate for the batched commit's KV write (#47557).

The batched commit appends a 256-token canvas into the frozen contiguous KV cache at
absolute positions ``[start_pos, start_pos+256)``. Two mechanisms exist
(``tt/commit_batched.py::_write_canvas_kv_contiguous``):

* ``"position"`` — the device-proven reference: 256 x (slice + reshard +
  ``paged_update_cache``) per K/V, ~1536 dispatches per layer;
* ``"fill"`` — the default: ONE ``ttnn.fill_cache`` per K/V at the tile-aligned
  ``update_idx=start_pos``, 2 dispatches per layer.

Because ``start_pos`` and ``canvas_len`` are both multiples of 32, the write span is
tile-aligned and FILL is a pure tile copy, so the two mechanisms must agree
**bit-for-bit** — not within a PCC. This asserts exactly that, over the WHOLE cache
tensor (so a disturbed frozen prefix ``[0, start_pos)`` or tail
``[start_pos+256, max_seq)`` fails too), against a torch oracle, and it pins the
fallback behaviour for geometries ``fill`` cannot serve.

Checkpoint-free (raw tensors + the write helper only) — runs in seconds.

Run on QB2:
  DG_RUN_DEVICE=1 pytest models/experimental/diffusion_gemma/tests/test_device_commit_kv_write.py
"""

import os

import pytest
import torch

import ttnn
from models.experimental.diffusion_gemma.tt.commit_batched import (
    _fill_write_unsupported_reason,
    _read_cache_kv,
    _write_canvas_kv_contiguous,
)

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("DG_RUN_DEVICE") != "1",
        reason="set DG_RUN_DEVICE=1 to run on a Tenstorrent device",
    ),
    # One device open/teardown for the whole module: repeated per-test CreateDevice on
    # QB2 can hang an active-erisc core (see test_device_bidirectional_sdpa.py).
    pytest.mark.use_module_device,
]


def _to_dev(device, host, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        host,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _write(device, *, mode, cache_host, canvas_k_host, canvas_v_host, start_pos, canvas_len, canvas_dtype):
    """Run one write mode into a fresh copy of ``cache_host``; return (k, v) host caches."""
    k_cache = _to_dev(device, cache_host)
    v_cache = _to_dev(device, cache_host)
    canvas_k = _to_dev(device, canvas_k_host, dtype=canvas_dtype)
    canvas_v = _to_dev(device, canvas_v_host, dtype=canvas_dtype)
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
    out = (ttnn.to_torch(k_cache), ttnn.to_torch(v_cache))
    for t in (k_cache, v_cache, canvas_k, canvas_v):
        t.deallocate(True)
    return out


# (nkv, head_dim, max_seq, start_pos, canvas_len). The 26B-A4B contiguous commit runs
# nkv_local in {1, 2, 8} (kv_replicated / TP-split sliding-vs-full), head_dim in
# {256, 512}, canvas_len 256, start_pos a multiple of 32.
GEOMETRIES = [
    (1, 256, 1024, 512, 256),
    (2, 256, 1024, 512, 256),  # 26B-A4B sliding layers on a 1x4 mesh
    (1, 512, 1024, 512, 256),  # 26B-A4B full-attention layers (kv_replicated, head_dim 512)
    (8, 256, 1024, 512, 256),
    (2, 512, 1024, 256, 256),
    (2, 256, 1024, 0, 256),  # first block: writes at offset 0
    (2, 256, 1024, 768, 256),  # last block: write ends exactly at max_seq
    (2, 256, 1024, 32, 32),  # single-tile canvas
    (8, 256, 2048, 1024, 256),
]


@pytest.mark.parametrize("nkv,head_dim,max_seq,start_pos,canvas_len", GEOMETRIES)
def test_fill_write_is_bit_identical_to_per_position(device, nkv, head_dim, max_seq, start_pos, canvas_len):
    torch.manual_seed(0)
    # A non-zero cache so a disturbed prefix/tail cannot hide behind zeros.
    cache_host = torch.randn(1, nkv, max_seq, head_dim).bfloat16()
    canvas_k_host = torch.randn(1, nkv, canvas_len, head_dim).bfloat16()
    canvas_v_host = torch.randn(1, nkv, canvas_len, head_dim).bfloat16()
    kw = dict(
        cache_host=cache_host,
        canvas_k_host=canvas_k_host,
        canvas_v_host=canvas_v_host,
        start_pos=start_pos,
        canvas_len=canvas_len,
        canvas_dtype=ttnn.bfloat16,
    )

    assert (
        _fill_write_unsupported_reason(
            _to_dev(device, cache_host),
            _to_dev(device, cache_host),
            _to_dev(device, canvas_k_host),
            _to_dev(device, canvas_v_host),
            start_pos=start_pos,
            canvas_len=canvas_len,
            mesh_device=device,
        )
        is None
    ), "fill must be supported at the DiffusionGemma commit geometry"

    ref_k, ref_v = _write(device, mode="position", **kw)
    fill_k, fill_v = _write(device, mode="fill", **kw)

    # 1. the two mechanisms agree over the WHOLE cache, exactly.
    assert torch.equal(ref_k, fill_k), f"K differs, max_abs={(ref_k - fill_k).abs().max()}"
    assert torch.equal(ref_v, fill_v), f"V differs, max_abs={(ref_v - fill_v).abs().max()}"

    # 2. and both equal the torch oracle: canvas in the span, cache elsewhere.
    for name, got, canvas in (("K", fill_k, canvas_k_host), ("V", fill_v, canvas_v_host)):
        want = cache_host.clone()
        want[:, :, start_pos : start_pos + canvas_len, :] = canvas
        assert torch.equal(got, want), f"{name} != oracle, {int((got != want).sum())} elements wrong"


def test_consecutive_blocks_preserve_each_other(device):
    """Two blocks in a row, as the generation loop does it.

    Block 2's write must leave block 1's committed K/V intact — the failure mode a
    single whole-canvas write could introduce (wrong offset, or a whole-slot fill) that
    a single-block test cannot see.
    """
    nkv, head_dim, max_seq, canvas_len = 2, 256, 1024, 256
    torch.manual_seed(3)
    cache_host = torch.randn(1, nkv, max_seq, head_dim).bfloat16()
    blocks = [
        (256, torch.randn(1, nkv, canvas_len, head_dim).bfloat16()),
        (512, torch.randn(1, nkv, canvas_len, head_dim).bfloat16()),
    ]

    k_cache = _to_dev(device, cache_host)
    v_cache = _to_dev(device, cache_host)
    want = cache_host.clone()
    for start_pos, canvas_host in blocks:
        canvas = _to_dev(device, canvas_host)
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
        want[:, :, start_pos : start_pos + canvas_len, :] = canvas_host

    for name, cache in (("K", k_cache), ("V", v_cache)):
        got = ttnn.to_torch(cache)
        assert torch.equal(got, want), f"{name} != oracle after 2 blocks, {int((got != want).sum())} wrong"


def test_fill_falls_back_and_stays_correct_on_dtype_mismatch(device):
    """FILL refuses to convert dtypes; the guard must catch it and still write correctly."""
    nkv, head_dim, max_seq, start_pos, canvas_len = 2, 256, 1024, 512, 256
    torch.manual_seed(1)
    cache_host = torch.randn(1, nkv, max_seq, head_dim).bfloat16()
    canvas_k_host = torch.randn(1, nkv, canvas_len, head_dim)
    canvas_v_host = torch.randn(1, nkv, canvas_len, head_dim)
    kw = dict(
        cache_host=cache_host,
        canvas_k_host=canvas_k_host,
        canvas_v_host=canvas_v_host,
        start_pos=start_pos,
        canvas_len=canvas_len,
        canvas_dtype=ttnn.float32,  # cache is bfloat16 -> FILL's dtype equality fails
    )

    reason = _fill_write_unsupported_reason(
        _to_dev(device, cache_host),
        _to_dev(device, cache_host),
        _to_dev(device, canvas_k_host, dtype=ttnn.float32),
        _to_dev(device, canvas_v_host, dtype=ttnn.float32),
        start_pos=start_pos,
        canvas_len=canvas_len,
        mesh_device=device,
    )
    assert reason is not None and "dtype" in reason

    ref_k, ref_v = _write(device, mode="position", **kw)
    fell_back_k, fell_back_v = _write(device, mode="fill", **kw)
    assert torch.equal(ref_k, fell_back_k)
    assert torch.equal(ref_v, fell_back_v)


def test_cache_read_at_max_seq_does_not_alias_the_cache(device):
    """The commit reads back ``[0, start_pos+C)`` and deallocates the result.

    When the committed block ends exactly at ``max_seq`` that read is a FULL-span slice,
    which ttnn short-circuits to an alias of the input — so deallocating it would free the
    KV cache itself. ``_read_cache_kv`` must hand back a distinct buffer.
    """
    nkv, head_dim, max_seq = 2, 256, 1024
    torch.manual_seed(4)
    cache_host = torch.randn(1, nkv, max_seq, head_dim).bfloat16()
    k_cache = _to_dev(device, cache_host)
    v_cache = _to_dev(device, cache_host)

    full_k, full_v = _read_cache_kv((k_cache, v_cache), end_pos=max_seq)
    assert full_k is not k_cache and full_v is not v_cache
    assert torch.equal(ttnn.to_torch(full_k), cache_host)
    full_k.deallocate(True)
    full_v.deallocate(True)

    # The cache must have survived the caller's deallocate and still be readable.
    assert torch.equal(ttnn.to_torch(k_cache), cache_host)
    assert torch.equal(ttnn.to_torch(v_cache), cache_host)


def test_fill_guard_rejects_head_boundary_spill(device):
    """The one FILL hazard the op does not self-check, and the fallback that covers it.

    ``fill_cache``'s program factory splits (nkv * C/32) tile-rows over the core grid and
    each core writes its rows contiguously from a single ``cache_start_id``, assuming no
    core's range crosses a kv-head boundary. Once the rows exceed the core count (and the
    input does not span the whole cache) that assumption breaks and the op silently writes
    rows to the wrong head — device-confirmed here: with nkv=8, C=1024, max_seq=2048 a raw
    ``ttnn.fill_cache`` corrupts the cache, so the guard must reject it and fall back.
    """
    nkv, head_dim, max_seq, start_pos, canvas_len = 8, 256, 2048, 512, 1024
    grid = device.compute_with_storage_grid_size()
    rows = nkv * (canvas_len // ttnn.TILE_SIZE)
    assert rows > grid.x * grid.y, f"geometry no longer spills on this grid ({rows} rows, {grid.x * grid.y} cores)"

    torch.manual_seed(2)
    cache_host = torch.randn(1, nkv, max_seq, head_dim).bfloat16()
    canvas_host = torch.randn(1, nkv, canvas_len, head_dim).bfloat16()
    want = cache_host.clone()
    want[:, :, start_pos : start_pos + canvas_len, :] = canvas_host

    reason = _fill_write_unsupported_reason(
        _to_dev(device, cache_host),
        _to_dev(device, cache_host),
        _to_dev(device, canvas_host),
        _to_dev(device, canvas_host),
        start_pos=start_pos,
        canvas_len=canvas_len,
        mesh_device=device,
    )
    assert reason is not None and "spill" in reason

    # The raw op really does corrupt this geometry (guard is not superstition).
    raw_cache = _to_dev(device, cache_host)
    ttnn.fill_cache(raw_cache, _to_dev(device, canvas_host), 0, update_idx=start_pos)
    assert not torch.equal(ttnn.to_torch(raw_cache), want), (
        "ttnn.fill_cache no longer spills across kv-head boundaries here — the factory may "
        "have been fixed; re-check the guard in _fill_write_unsupported_reason"
    )
    raw_cache.deallocate(True)

    # ...and the guarded write is correct anyway, via the per-position fallback.
    got_k, got_v = _write(
        device,
        mode="fill",
        cache_host=cache_host,
        canvas_k_host=canvas_host,
        canvas_v_host=canvas_host,
        start_pos=start_pos,
        canvas_len=canvas_len,
        canvas_dtype=ttnn.bfloat16,
    )
    assert torch.equal(got_k, want)
    assert torch.equal(got_v, want)
