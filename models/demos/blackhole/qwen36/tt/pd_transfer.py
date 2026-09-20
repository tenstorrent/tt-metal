# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefill/decode disaggregation: host-staged export/import of one request's state (TP model).

A request's state on the Qwen3.x hybrid model is (a) its paged-KV blocks in the 16 full-attention
layers and (b) its Gated DeltaNet state in the 48 linear-attention layers: the fp32 recurrent state
`[Nv/TP, Dk, Dv]` and the K causal-conv taps `[1, qkv_dim_tp]` per layer per chip. Both halves of a
prefill/decode split run the same TP sharding, so everything moves per device (mesh dim 0 = device)
with no reshuffle.

Prefill side: `prefill_paged_slots` snapshots each request's GDN state to the host anyway (the served
plain path writes the decode slot from that snapshot); with `model.pd_gdn_capture` set to a dict it
also parks the snapshot under the request's slot, and `model.pd_skip_gdn_slot_write` skips the slot
write (a pure-prefill instance never decodes). `export_kv_blocks` reads the request's blocks out of the
paged caches.

Decode side: `import_kv_blocks` fills the request's blocks via `paged_fill_cache` over its page-table
row; `import_gdn_slot` writes the snapshot into the request's decode slot through `_write_gdn_slot`
(the same path the served prefill uses). The decode instance then continues the request with one
ordinary decode step for the last prompt token (the prefill side computed h(N-1)).
"""

from __future__ import annotations

import os
import time

import torch
from loguru import logger

import ttnn


def coalesce_runs(block_ids):
    """Consecutive runs of block ids, ascending or descending, as (lo, hi, descending) with [lo, hi) the
    contiguous region; order of runs = order of block_ids. vLLM's free-block queue hands out descending
    sequences after frees, so both directions matter: [3,4,5,9,10,7] -> [(3,6,F),(9,11,F),(7,8,F)];
    [17,16,15,2,3] -> [(15,18,T),(2,4,F)]."""
    runs = []  # [lo, hi, direction] direction: 0 unknown (single), +1 asc, -1 desc
    for b in block_ids:
        b = int(b)
        if runs:
            lo, hi, d = runs[-1]
            if d >= 0 and b == hi:
                runs[-1] = [lo, hi + 1, 1]
                continue
            if d <= 0 and b == lo - 1:
                runs[-1] = [lo - 1, hi, -1]
                continue
        runs.append([b, b + 1, 0])
    return [(lo, hi, d < 0) for lo, hi, d in runs]


def _reorder_runs(host: torch.Tensor, runs) -> torch.Tensor:
    """Put a device-read tensor whose dim 0 is the concatenation of the runs' contiguous regions into
    block_ids order: descending runs are read ascending on device and flipped here."""
    if not any(desc for _, _, desc in runs):
        return host
    parts, off = [], 0
    for lo, hi, desc in runs:
        n = hi - lo
        part = host[off : off + n]
        parts.append(torch.flip(part, dims=[0]) if desc else part)
        off += n
    return torch.cat(parts, dim=0)


def _device_convert() -> bool:
    """QWEN36_PD_DEVICE_CONVERT=0 falls back to host-side tilize/untilize + dtype conversion."""
    return os.environ.get("QWEN36_PD_DEVICE_CONVERT", "1") != "0"


def _upload(model, host: torch.Tensor, dtype, mapper):
    """Host tensor -> device tensor of `dtype` in TILE layout. With device conversion (default) the host
    transfer is a row-major memcpy in the host tensor's own dtype (bf16/fp32) and tilize + typecast run on
    device; otherwise from_torch tilizes/packs on the host."""
    mesh = model.mesh_device
    if not _device_convert():
        return ttnn.from_torch(
            host,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
    host_dtype = {torch.bfloat16: ttnn.bfloat16, torch.float32: ttnn.float32}.get(host.dtype)
    if host_dtype is None:
        host = host.to(torch.bfloat16)
        host_dtype = ttnn.bfloat16
    rm = ttnn.from_torch(
        host,
        dtype=host_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )
    t = ttnn.to_layout(rm, ttnn.TILE_LAYOUT)
    ttnn.deallocate(rm)
    if t.dtype != dtype:
        tc = ttnn.typecast(t, dtype)
        ttnn.deallocate(t)
        t = tc
    return t


def _attention_layers(model):
    return [layer.attention for layer in model.layers if layer.is_full_attention]


def pad_block_ids(block_ids, n, num_blocks):
    """Extend `block_ids` to `n` entries with other VALID block ids (their rows are dropped on the host).
    A single monotonic run is extended in its own direction so it stays one run (the single-slice path);
    anything else repeats the last block. Returns (padded_ids, offset) with the real rows at
    [offset, offset + len(block_ids))."""
    ids = [int(b) for b in block_ids]
    k = n - len(ids)
    if k <= 0:
        return ids, 0
    runs = coalesce_runs(ids)
    if len(runs) == 1:
        lo, hi, desc = runs[0]
        if not desc and hi + k <= num_blocks:
            return ids + list(range(hi, hi + k)), 0
        if desc and lo - k >= 0:
            return ids + list(range(lo - 1, lo - 1 - k, -1)), 0
        if not desc and lo - k >= 0:  # top of the pool: extend below, real rows follow the pad rows
            return list(range(lo - k, lo)) + ids, k
        if desc and hi + k <= num_blocks:
            return list(range(hi + k - 1, hi - 1, -1)) + ids, k
    return ids + [ids[-1]] * k, 0


def export_kv_blocks(model, block_ids):
    """Read one request's paged-KV blocks off the device.

    Returns, per full-attention layer, a `(k, v)` pair of host tensors shaped
    `[n_blocks, n_dev * n_local_kv_heads, block_size, head_dim]` in torch.bfloat16 (a bf8 cache reads
    back through bf16 exactly), block order = `block_ids` order.

    One device read for the whole request: the blocks of all 32 cache tensors are concatenated on device
    into a single tensor, converted (bfp8 -> bf16, untilize) once, and read once with a dim-0 mesh
    composer (mesh reads cost ~60 ms fixed each and composing along a middle dim runs at ~0.1 GiB/s, so
    32 separate reads took seconds; one dim-0 read runs at ~1.2 GiB/s).

    Program shapes depend on the block count, so the block list is padded to a power-of-two bucket
    (export_warmup compiles every bucket at startup) and read by one of two fixed-shape paths:
    `runs`  -- the list is one contiguous run: one slice per cache, concat of 32;
    `blocks` -- anything else: one single-block slice per (cache, block), a concat of `bucket` per
    cache, then a concat of 32. Slice offsets are runtime arguments, so neither path compiles per
    block id. (`ttnn.gather` was tried for fragmented lists: it reads the WHOLE cache per call, ~110 ms
    x 32 caches = 3.5 s per request on a 500 MB cache, and returned wrong rows for a bfp8 cache.)
    """
    t0 = time.perf_counter()
    layers = _attention_layers(model)
    n_dev = model.num_devices
    n_real = len(block_ids)
    block_ids = [int(b) for b in block_ids]
    n = export_bucket(n_real) if os.environ.get("QWEN36_PD_EXPORT_BUCKETS", "1") == "1" else n_real
    cache0 = layers[0].paged_k
    num_blocks, nkv, blk, hd = cache0.shape
    block_ids, real_off = pad_block_ids(block_ids, n, int(num_blocks))
    runs = coalesce_runs(block_ids)
    caches = [c for att in layers for c in (att.paged_k, att.paged_v)]
    mode = os.environ.get("QWEN36_PD_EXPORT", "auto")
    if mode == "auto":
        mode = "runs" if len(runs) == 1 else "blocks"
    parts = []
    if mode == "runs":
        for cache in caches:
            for lo, hi, _ in runs:
                parts.append(ttnn.slice(cache, (lo, 0, 0, 0), (hi, nkv, blk, hd)))
    elif mode == "blocks":
        for cache in caches:
            singles = [ttnn.slice(cache, (b, 0, 0, 0), (b + 1, nkv, blk, hd)) for b in block_ids]
            if len(singles) > 1:
                cat = ttnn.concat(singles, dim=0)
                for s_ in singles:
                    ttnn.deallocate(s_)
                parts.append(cat)
            else:
                parts.append(singles[0])
    else:
        raise ValueError(f"QWEN36_PD_EXPORT={mode!r}: expected auto, runs or blocks")
    big = ttnn.concat(parts, dim=0) if len(parts) > 1 else parts[0]  # per device [32*n, nkv, blk, hd]
    if len(parts) > 1:
        for p_ in parts:
            ttnn.deallocate(p_)
    if _device_convert():
        if big.dtype != ttnn.bfloat16:
            b16 = ttnn.typecast(big, ttnn.bfloat16)
            ttnn.deallocate(big)
            big = b16
        rm = ttnn.to_layout(big, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(big)
        big = rm
    t1 = time.perf_counter()
    host = ttnn.to_torch(big, mesh_composer=ttnn.ConcatMeshToTensor(model.mesh_device, dim=0)).to(torch.bfloat16)
    ttnn.deallocate(big)
    t2 = time.perf_counter()
    # host: [n_dev * 32 * n, nkv, blk, hd] -> [n_dev, 32, n, nkv, blk, hd]
    host = host.view(n_dev, len(caches), n, nkv, blk, hd)
    out = []
    for li in range(len(layers)):
        pair = []
        for j in (0, 1):
            t = host[:, 2 * li + j]  # [n_dev, n, nkv, blk, hd]
            t = t.permute(1, 0, 2, 3, 4).reshape(n, n_dev * nkv, blk, hd)
            if mode == "runs":
                t = _reorder_runs(t, runs)
            pair.append(t[real_off : real_off + n_real].contiguous())
        out.append((pair[0], pair[1]))
    logger.debug(
        f"[pd] exported {n_real} KV blocks (bucket {n}) x {len(out)} layers in {1e3 * (time.perf_counter() - t0):.1f} ms "
        f"({len(runs)} run(s), {mode}; device {1e3 * (t1 - t0):.1f} ms, read {1e3 * (t2 - t1):.1f} ms, "
        f"host {1e3 * (time.perf_counter() - t2):.1f} ms)"
    )
    return out


_EXPORT_BUCKETS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


def export_bucket(n_blocks: int) -> int:
    for b in _EXPORT_BUCKETS:
        if b >= n_blocks:
            return b
    return n_blocks  # beyond the pre-warmed buckets: exact count (compiles on first use)


def export_warmup(model, max_bucket: int = 256):
    """Compile the export programs for every bucket up to max_bucket (~2 s per bucket and path)."""
    t0 = time.perf_counter()
    num_blocks = int(_attention_layers(model)[0].paged_k.shape[0])
    for b in _EXPORT_BUCKETS:
        if b > max_bucket:
            break
        # both fixed-shape paths: one contiguous run (single slice per cache) and the per-block path
        export_kv_blocks(model, list(range(1, 1 + b)))
        if b > 1:
            export_kv_blocks(model, [1] * b)
    logger.info(f"[pd] export warm-up: buckets <= {max_bucket} in {time.perf_counter() - t0:.1f} s")


def import_kv_blocks(model, block_ids, kv):
    """Write `kv` (the `export_kv_blocks` layout) into this instance's paged caches at `block_ids`."""
    t0 = time.perf_counter()
    n_dev = model.num_devices
    n_blocks = len(block_ids)
    pt = torch.tensor([list(int(b) for b in block_ids)], dtype=torch.int32)  # [1, n_blocks]
    page_table_tt = ttnn.from_torch(
        pt,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=model.mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(model.mesh_device),
    )
    mapper = ttnn.ShardTensorToMesh(model.mesh_device, dim=0)
    layers = _attention_layers(model)
    assert len(kv) == len(layers), f"{len(kv)} KV layer pairs for {len(layers)} attention layers"
    for att, (k_host, v_host) in zip(layers, kv):
        for host, cache in ((k_host, att.paged_k), (v_host, att.paged_v)):
            nb, ndn, blk, hd = host.shape
            assert nb == n_blocks, f"{nb} blocks in payload vs {n_blocks} block ids"
            nkv = ndn // n_dev
            # [n_blocks, n_dev*nkv, blk, hd] -> [n_dev, nkv, n_blocks*blk, hd] (one [1, nkv, T, hd] fill per device)
            x = host.view(nb, n_dev, nkv, blk, hd).permute(1, 2, 0, 3, 4).reshape(n_dev, nkv, nb * blk, hd)
            xt = _upload(model, x.contiguous(), cache.dtype, mapper)
            ttnn.experimental.paged_fill_cache(cache, xt, page_table_tt, batch_idx=0)
            ttnn.deallocate(xt)
    ttnn.deallocate(page_table_tt)
    logger.debug(
        f"[pd] imported {n_blocks} KV blocks x {len(layers)} layers in {1e3 * (time.perf_counter() - t0):.1f} ms"
    )


def import_gdn_slot(model, slot, rec_snap, conv_snap, mode=None):
    """Write one request's GDN snapshot (`prefill_paged_slots` capture layout: rec_snap[li] host
    `[n_dev, Nv, Dk, Dv]`, conv_snap[li][m] host `[n_dev, 1, D]`) into decode `slot`.

    mode "trace" (default) = TracedGdnImporter: host memcpy into fixed row-major staging tensors + one replayed
    per-slot trace (tilize, fill_cache rows, tap row writes, packed-history row write for all layers).
    mode "host" = the served prefill's own slot write (`_write_gdn_slot`: one from_torch + slice/concat/copy
    row write per tensor, ~4 s for 48 layers); "fillcache" (default, QWEN36_PD_GDN_IMPORT) = one batched
    upload of all layers, then per layer an in-place `ttnn.fill_cache` for the recurrent state row and the
    slice/concat/copy row write for the K conv taps (the `_write_index` form the host path uses -- a masked
    `where` does not land the row on TILE-padded taps), then the per-slot packed-history repack.
    """
    mode = mode or os.environ.get("QWEN36_PD_GDN_IMPORT", "trace")
    t0 = time.perf_counter()
    if mode == "host":
        model._write_gdn_slot(int(slot), rec_snap, conv_snap)
    elif mode == "trace":
        get_traced_importer(model).import_slot(int(slot), rec_snap, conv_snap)
    else:
        _import_gdn_slot_fillcache(model, int(slot), rec_snap, conv_snap)
    logger.debug(f"[pd] imported GDN state into slot {slot} ({mode}) in {1e3 * (time.perf_counter() - t0):.1f} ms")


def _import_gdn_slot_fillcache(model, slot, rec_snap, conv_snap):
    mesh = model.mesh_device
    n_dev = model.num_devices
    mapper = ttnn.ShardTensorToMesh(mesh, dim=0)
    dn_layers = [layer.attention for layer in model.layers if not layer.is_full_attention]
    L = len(dn_layers)
    assert len(rec_snap) == L and len(conv_snap) == L, f"snapshot has {len(rec_snap)} GDN layers, model {L}"
    K = dn_layers[0].K
    dn0 = dn_layers[0]
    # ---- recurrent state: [L][n_dev, Nv, Dk, Dv] -> device-major [n_dev*L, Nv, Dk, Dv], one upload ----
    rec_all = torch.stack(rec_snap, dim=1)  # [n_dev, L, Nv, Dk, Dv]
    rec_all = rec_all.reshape(n_dev * L, *rec_all.shape[2:]).contiguous()
    rec_dev = _upload(model, rec_all, dn0.rec_state.dtype, mapper)  # per device [L, Nv, Dk, Dv]
    # ---- conv taps: [L][K][n_dev, 1, C] -> [n_dev*L*K, 1, C], one upload ----
    taps = torch.stack([torch.stack(conv_snap[li], dim=1) for li in range(L)], dim=1)  # [n_dev, L, K, 1, C]
    taps = taps.reshape(n_dev * L * K, 1, taps.shape[-1]).contiguous()
    taps_dev = _upload(model, taps, dn0.conv_states[0].dtype, mapper)  # per device [L*K, 1, C]
    for li, dn in enumerate(dn_layers):
        rec_l = dn._slice_along(rec_dev, 0, li, li + 1)  # [1, Nv, Dk, Dv]
        if rec_l.dtype != dn.rec_state.dtype:
            rec_c = ttnn.typecast(rec_l, dn.rec_state.dtype)
            ttnn.deallocate(rec_l)
            rec_l = rec_c
        ttnn.fill_cache(dn.rec_state, rec_l, slot)  # in place: rec_state[slot] = rec_l
        ttnn.deallocate(rec_l)
        for m in range(dn.K):
            c = dn._slice_along(taps_dev, 0, li * K + m, li * K + m + 1)  # [1, 1, C]
            if c.dtype != dn.conv_states[m].dtype:
                c_c = ttnn.typecast(c, dn.conv_states[m].dtype)
                ttnn.deallocate(c)
                c = c_c
            dn._write_index(dn.conv_states[m], c, slot, dim=1)  # consumes c
        if dn.conv_hist_packed is not None and dn._hist_packed_valid:
            dn._sync_conv_hist_packed(slot=slot)
        else:
            dn._sync_conv_hist_packed()
    ttnn.deallocate(rec_dev)
    ttnn.deallocate(taps_dev)


def gdn_state_nbytes(rec_snap, conv_snap):
    n = sum(t.numel() * t.element_size() for t in rec_snap)
    n += sum(c.numel() * c.element_size() for taps in conv_snap for c in taps)
    return n


def kv_nbytes(kv):
    return sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in kv)


# --------------------------------------------------------------------------------------
# traced per-slot GDN import
# --------------------------------------------------------------------------------------


class TracedGdnImporter:
    """Write a request's GDN snapshot into decode slot ``slot`` with one trace replay.

    The eager writes (48 layers x (recurrent row + K tap rows + packed-history row)) are ~1000 small ops and
    dispatch-bound (~0.4 s). Here the host copies the snapshot into fixed ROW_MAJOR staging tensors (rec fp32,
    taps bf16, packed history bf16 -- the packed row is built on the host with the layer's own
    ``_pack_head_tiles``, parity ``slot & 1``), and a per-slot trace does tilize + all the row writes on device.
    Traces are captured lazily per slot (or up front via ``precapture``); slot indices are baked into them.
    """

    def __init__(self, model):
        self.model = model
        self.mesh = model.mesh_device
        self.n_dev = model.num_devices
        self.dn = [layer.attention for layer in model.layers if not layer.is_full_attention]
        dn0 = self.dn[0]
        self.L, self.K = len(self.dn), dn0.K
        rec_shape = tuple(dn0.rec_state.shape)  # [B, Nv, Dk, Dv]
        self.Nv, self.Dk, self.Dv = rec_shape[1], rec_shape[2], rec_shape[3]
        self.C = int(dn0.conv_states[0].shape[-1])
        self.rec_dtype = dn0.rec_state.dtype
        self.with_hist = dn0.conv_hist_packed is not None
        mapper = ttnn.ShardTensorToMesh(self.mesh, dim=0)

        def stage(shape, torch_dtype, tt_dtype):
            return ttnn.from_torch(
                torch.zeros(*shape, dtype=torch_dtype),
                dtype=tt_dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )

        rec_torch = torch.float32 if self.rec_dtype == ttnn.float32 else torch.bfloat16
        self.rec_rm = stage((self.n_dev * self.L, self.Nv, self.Dk, self.Dv), rec_torch, self.rec_dtype)
        self.taps_rm = stage((self.n_dev * self.L * self.K, 1, self.C), torch.bfloat16, ttnn.bfloat16)
        self.hist_rm = (
            stage((self.n_dev * self.L, self.Nv, 4, 32, 32), torch.bfloat16, ttnn.bfloat16) if self.with_hist else None
        )
        self.traces: dict[int, int] = {}
        self.mapper = mapper
        logger.info(
            f"[pd] TracedGdnImporter: L={self.L} K={self.K} rec={self.rec_dtype} hist={'on' if self.with_hist else 'off'}; "
            f"staging {self.n_dev * self.L * self.Nv * self.Dk * self.Dv * (4 if rec_torch == torch.float32 else 2) / 2**20:.0f} MiB rec"
        )

    # -- device body (trace-capturable: no host transfers) --
    def _body(self, slot: int):
        rec_t = ttnn.to_layout(self.rec_rm, ttnn.TILE_LAYOUT)
        taps_t = ttnn.to_layout(self.taps_rm, ttnn.TILE_LAYOUT)
        hist_t = ttnn.to_layout(self.hist_rm, ttnn.TILE_LAYOUT) if self.with_hist else None
        K = self.K
        for li, dn in enumerate(self.dn):
            rec_l = dn._slice_along(rec_t, 0, li, li + 1)
            ttnn.fill_cache(dn.rec_state, rec_l, slot)
            ttnn.deallocate(rec_l)
            for m in range(K):
                c = dn._slice_along(taps_t, 0, li * K + m, li * K + m + 1)
                dn._write_index(dn.conv_states[m], c, slot, dim=1)  # consumes c
            if hist_t is not None and dn.conv_hist_packed is not None:
                h = dn._slice_along(hist_t, 0, li, li + 1)
                dn._write_index(dn.conv_hist_packed, h, slot, dim=0)  # consumes h
        ttnn.deallocate(rec_t)
        ttnn.deallocate(taps_t)
        if hist_t is not None:
            ttnn.deallocate(hist_t)

    def capture(self, slot: int):
        if slot in self.traces:
            return
        t0 = time.perf_counter()
        self._body(slot)  # compile pass (also a harmless write of the staged data into the slot)
        ttnn.synchronize_device(self.mesh)
        tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        self._body(slot)
        ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
        ttnn.synchronize_device(self.mesh)
        self.traces[slot] = tid
        logger.info(f"[pd] captured GDN import trace for slot {slot} in {1e3 * (time.perf_counter() - t0):.0f} ms")

    def precapture(self, slots):
        for s in slots:
            self.capture(int(s))

    # -- host side --
    def _gather_index(self):
        """[Nv, 3*Dk] channel indices: head h reads its q chunk (kv head hk), k chunk and v chunk of a [C] row --
        the vectorized form of _pack_head_tiles' per-head concat."""
        if getattr(self, "_gidx", None) is None:
            dn0 = self.dn[0]
            Nv, Nk, Dk, Dv = dn0.Nv, dn0.Nk, dn0.Dk, dn0.Dv
            rf, kd = Nv // Nk, Nk * Dk
            rows = []
            for h in range(Nv):
                hk = h // rf
                rows.append(
                    torch.cat(
                        [
                            torch.arange(hk * Dk, (hk + 1) * Dk),
                            torch.arange(kd + hk * Dk, kd + (hk + 1) * Dk),
                            torch.arange(2 * kd + h * Dv, 2 * kd + (h + 1) * Dv),
                        ]
                    )
                )
            self._gidx = torch.stack(rows)  # [Nv, 3*Dk]
            self._n_chunks = self._gidx.shape[1] // 32
            self._verified_pack = False
        return self._gidx

    def _host_hist(self, conv_snap, slot):
        """Packed history rows for all layers/devices at parity slot & 1, vectorized (torch ops only)."""
        gidx = self._gather_index()
        par = slot & 1
        n = self._n_chunks
        # taps [L, K, n_dev, C] -> gather channels -> [L, K, n_dev, Nv, n, 32]
        taps = torch.stack(
            [torch.stack([conv_snap[li][j].reshape(self.n_dev, -1) for j in range(self.K)]) for li in range(self.L)]
        )
        g = taps[..., gidx].to(torch.bfloat16).reshape(self.L, self.K, self.n_dev, self.Nv, n, 32)
        out = torch.zeros(self.n_dev, self.L, self.Nv, 4, 32, 32, dtype=torch.bfloat16)
        out[..., par : 2 * n + par : 2, :] = g.permute(2, 0, 3, 1, 4, 5)  # [n_dev, L, Nv, K, n, 32]
        if not self._verified_pack:
            # one-time check against the layer's own scalar packer
            ref = self.dn[0]._pack_head_tiles([conv_snap[0][j][0].reshape(-1) for j in range(self.K)], parity=par)
            if not torch.equal(ref, out[0, 0]):
                raise RuntimeError("vectorized packed-history layout differs from _pack_head_tiles")
            self._verified_pack = True
        return out.reshape(self.n_dev * self.L, self.Nv, 4, 32, 32).contiguous()

    def _upload(self, host, dst):
        h = ttnn.from_torch(host, dtype=dst.dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=None, mesh_mapper=self.mapper)
        ttnn.copy_host_to_device_tensor(h, dst)
        return h  # keep alive until the replay is synchronized

    def import_slot(self, slot: int, rec_snap, conv_snap):
        t0 = time.perf_counter()
        rec_all = torch.stack(rec_snap, dim=1).reshape(self.n_dev * self.L, self.Nv, self.Dk, self.Dv).contiguous()
        taps = torch.stack([torch.stack(conv_snap[li], dim=1) for li in range(self.L)], dim=1)
        taps = taps.reshape(self.n_dev * self.L * self.K, 1, self.C).contiguous()
        refs = [self._upload(rec_all, self.rec_rm), self._upload(taps, self.taps_rm)]
        if self.with_hist:
            refs.append(self._upload(self._host_hist(conv_snap, slot), self.hist_rm))
        t1 = time.perf_counter()
        if slot not in self.traces:
            self.capture(slot)
        ttnn.execute_trace(self.mesh, self.traces[slot], cq_id=0, blocking=False)
        ttnn.synchronize_device(self.mesh)
        refs.clear()
        for dn in self.dn:
            dn._hist_packed_valid = dn._hist_packed_valid if dn.conv_hist_packed is None else True
        logger.debug(
            f"[pd] traced GDN import slot {slot}: host+upload {1e3 * (t1 - t0):.1f} ms, replay {1e3 * (time.perf_counter() - t1):.1f} ms"
        )


def get_traced_importer(model) -> "TracedGdnImporter":
    imp = getattr(model, "pd_gdn_importer", None)
    if imp is None:
        imp = model.pd_gdn_importer = TracedGdnImporter(model)
    return imp


def verify_gdn_slot(model, slot, rec_snap, conv_snap, tag=""):
    """Read back decode `slot` (recurrent state, conv taps, packed history) and compare with the snapshot."""
    comp = ttnn.ConcatMeshToTensor(model.mesh_device, dim=0)
    dn_layers = [layer.attention for layer in model.layers if not layer.is_full_attention]
    n_dev = model.num_devices
    worst_rec, worst_tap, worst_hist = 0.0, 0.0, 0.0
    for li in (0, len(dn_layers) // 2, len(dn_layers) - 1):
        dn = dn_layers[li]
        r = dn._slice_along(dn.rec_state, 0, slot, slot + 1)
        got = ttnn.to_torch(r, mesh_composer=comp).float()  # [n_dev, Nv, Dk, Dv]
        ttnn.deallocate(r)
        worst_rec = max(worst_rec, float((got - rec_snap[li].float()).abs().max()))
        for m in range(dn.K):
            c = dn._slice_along(dn.conv_states[m], 1, slot, slot + 1)
            gotc = ttnn.to_torch(c, mesh_composer=comp).float().reshape(n_dev, -1)  # [n_dev, C]
            ttnn.deallocate(c)
            worst_tap = max(worst_tap, float((gotc - conv_snap[li][m].float().reshape(n_dev, -1)).abs().max()))
        if dn.conv_hist_packed is not None:
            h = dn._slice_along(dn.conv_hist_packed, 0, slot, slot + 1)
            goth = ttnn.to_torch(h, mesh_composer=comp).to(torch.bfloat16)  # [n_dev, Nv, 4, 32, 32]
            ttnn.deallocate(h)
            ref = torch.stack(
                [
                    dn._pack_head_tiles([conv_snap[li][j][d].reshape(-1) for j in range(dn.K)], parity=slot & 1)
                    for d in range(n_dev)
                ]
            )
            worst_hist = max(worst_hist, float((goth.float() - ref.float()).abs().max()))
    logger.info(
        f"[pd] VERIFY slot {slot} {tag}: max|rec diff| {worst_rec:.3g}, max|tap diff| {worst_tap:.3g}, max|packed-hist diff| {worst_hist:.3g}"
    )
