# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Collectives for a 1x2 Blackhole mesh whose two chips are joined only by PCIe (desktop-0 chips 4 and 5).

Decode-time collectives run on device through ``ttnn.experimental.pcie_exchange`` (each core writes its L1
shard straight into the peer's L1 over the PCIe peer-to-peer windows, ~1 us + 28 GB/s). Anything the op does
not cover (prefill-sized DRAM tensors) falls back to a host round trip, which is correct but slow and must
never sit inside a trace.

Enable with ``TT_PCIE_P2P_CHIPS=4,5``; conftest's ``mesh_device`` fixture then opens those chips as an
isolated mesh and TT_CCL attaches a ``PcieCCL`` as ``tt_ccl.pcie``.
"""
import os

import torch
from loguru import logger

import ttnn

ALL_REDUCE = 0
ALL_GATHER = 1


def _volume(shape):
    v = 1
    for d in shape:
        v *= d
    return v


def pcie_p2p_chips():
    v = os.environ.get("TT_PCIE_P2P_CHIPS")
    return [int(c) for c in v.split(",")] if v else None


class PcieCCL:
    def __init__(self, mesh_device, num_window_cores=64):
        self.mesh_device = mesh_device
        self.num_devices = mesh_device.get_num_devices()
        assert self.num_devices == 2, f"PcieCCL supports exactly 2 devices, mesh has {self.num_devices}"
        self.grid = mesh_device.compute_with_storage_grid_size()
        # Row-major first N cores of the compute grid: the residual / matmul-output layout and the core walk
        # the op's all-gather output uses.
        self.cores = [ttnn.CoreCoord(i % self.grid.x, i // self.grid.x) for i in range(num_window_cores)]
        ttnn._ttnn.pcie_p2p.setup(mesh_device, self.cores)
        self._exchange = {}
        # Persistent control tiles (flag / sequence / staging) for the per-call prefill landing buffers. They must
        # be zeroed exactly once, outside any trace: the peer may write its flag into them before this device's op
        # starts, so a per-call zero-fill races it (seen as watchdog timeouts under traced prefill).
        self._ctrl = {n: self._make_ctrl(n) for n in (64, 32)}
        # Decode all-reduce payload in bfp8 (half the link bytes; both devices sum the same two rounded operands).
        self.bfp8_reduce = os.environ.get("TT_PCIE_BFP8_REDUCE", "1") == "1"  # PCC unchanged, 13.8 -> 8.8 us/op
        self.host_fallbacks = 0
        logger.info(
            f"PcieCCL: PCIe P2P windows on {num_window_cores} cores of {self.grid}, devices {mesh_device.get_device_ids()}"
        )

    # ---------------------------------------------------------------- layouts
    def width_sharded(self, num_cores, shard_w):
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.num_cores_to_corerangeset(num_cores, self.grid, True),
                [32, shard_w],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

    @staticmethod
    def _op_capable(t):
        if not (t.is_sharded() and t.layout == ttnn.TILE_LAYOUT):
            return False
        mc = t.memory_config()
        if mc.buffer_type != ttnn.BufferType.L1 or mc.memory_layout != ttnn.TensorMemoryLayout.WIDTH_SHARDED:
            return False
        ss = mc.shard_spec
        rows = t.padded_shape[-2]
        return ss.shape[0] == rows and (_volume(t.padded_shape) // t.padded_shape[-1]) == rows

    def _make_ctrl(self, n_cores):
        grid = ttnn.num_cores_to_corerangeset(n_cores, self.grid, True)
        mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, [32, 32], ttnn.ShardOrientation.ROW_MAJOR),
        )
        return ttnn.moreh_full(
            [1, 1, 32, n_cores * 32],
            0.0,
            self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mc,
        )

    def _landing_for(self, t):
        """Per-call landing buffer (two shards wide, no control tile). It must NOT be written by this device at all:
        under trace replay the peer can already be pushing its payload into it while this device is still an op
        behind, so even a device-side zero-fill right before the exchange races the peer (seen as wrong prefill
        results with traced prefill). A pure allocation has no device write and is legal inside trace capture."""
        ss = t.memory_config().shard_spec
        tile_bytes = {
            ttnn.bfloat16: 2048,
            ttnn.bfloat8_b: 1088,
            ttnn.bfloat4_b: 576,
            ttnn.uint32: 4096,
            ttnn.int32: 4096,
            ttnn.float32: 4096,
        }[t.dtype]
        shard_bytes = (ss.shape[0] // 32) * (ss.shape[1] // 32) * tile_bytes
        xb_tiles = -(-(2 * shard_bytes) // 2048)
        n = ss.grid.num_cores()
        mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(ss.grid, [32, xb_tiles * 32], ttnn.ShardOrientation.ROW_MAJOR),
        )
        return ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, 32, n * xb_tiles * 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh_device, mc
        )

    def _exchange_for(self, t, persistent=True):
        """Landing buffer for tensors with this shard grid and shard byte size (cached when persistent)."""
        mc = t.memory_config()
        ss = mc.shard_spec
        if t.layout == ttnn.TILE_LAYOUT:
            tile_bytes = {
                ttnn.bfloat16: 2048,
                ttnn.bfloat8_b: 1088,
                ttnn.bfloat4_b: 576,
                ttnn.uint32: 4096,
                ttnn.int32: 4096,
                ttnn.float32: 4096,
            }[t.dtype]
            shard_bytes = (ss.shape[0] // 32) * (ss.shape[1] // 32) * tile_bytes
        else:
            elem = {ttnn.bfloat16: 2, ttnn.uint32: 4, ttnn.int32: 4, ttnn.float32: 4}[t.dtype]
            shard_bytes = ss.shape[0] * ss.shape[1] * elem
        key = (str(ss.grid), shard_bytes)
        xb = self._exchange.get(key) if persistent else None
        if xb is None:
            xb_tiles = -(-(2 * shard_bytes + 2048) // 2048)
            shard_w = xb_tiles * 32
            n = ss.grid.num_cores()
            mc = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(ss.grid, [32, shard_w], ttnn.ShardOrientation.ROW_MAJOR),
            )
            # Control words (flag, sequence) must start at zero on both devices. moreh_full is a device kernel, so
            # it is legal inside trace capture (ttnn.zeros / ttnn.full / from_torch are host writes and are not).
            xb = ttnn.moreh_full(
                [1, 1, 32, n * shard_w],
                0.0,
                self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=mc,
            )
            if persistent:
                self._exchange[key] = xb
                logger.info(f"PcieCCL: exchange buffer for {n} cores x {shard_bytes} B shards ({xb_tiles} tiles/core)")
        return xb

    # Prefill-sized landing buffers (2 x 64 KiB + control per core) are allocated per call and freed right after
    # the exchange: prefill ops (the interleaved norm) need nearly all of L1 between ops, and a persistent 130 KiB
    # buffer per core clashes with their static circular buffers.
    PREFILL_ROWS_PER_PIECE = 512

    def release_prefill_buffers(self):
        """Kept for the decode-switch hook; prefill buffers are no longer persistent."""
        return

    def _exchange_dram(self, t, mode, n_cores):
        """Prefill collectives on DRAM-interleaved TILE tensors: view as [rows, W], cut into <= 512-row pieces,
        width-shard each piece over n_cores window cores (one [rows, W/n] shard per core), run the L1 exchange op,
        and put the result back in the input's DRAM layout."""
        shape = list(t.shape)
        w = t.padded_shape[-1]
        rows = _volume(t.padded_shape) // w
        x = ttnn.reshape(t, [1, 1, rows, w]) if list(t.padded_shape) != [1, 1, rows, w] else t
        assert w % (n_cores * 32) == 0, f"PcieCCL: width {w} does not split over {n_cores} cores"
        step = self.PREFILL_ROWS_PER_PIECE
        pieces = []
        for r0 in range(0, rows, step):
            r1 = min(rows, r0 + step)
            piece = ttnn.slice(x, [0, 0, r0, 0], [1, 1, r1, w]) if (r0, r1) != (0, rows) else x
            sh = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.num_cores_to_corerangeset(n_cores, self.grid, True),
                    [r1 - r0, w // n_cores],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )
            ps = ttnn.to_memory_config(piece, sh)
            if piece is not x:
                ttnn.deallocate(piece)
            xb = self._landing_for(ps)
            out_sh = ttnn.experimental.pcie_exchange(ps, xb, mode, ctrl=self._ctrl[n_cores])
            ttnn.deallocate(ps)
            ttnn.deallocate(xb)
            out = ttnn.to_memory_config(out_sh, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(out_sh)
            pieces.append(out)
        # x is a reshape view sharing t's buffer: never deallocate it (that would free the caller's input).
        res = pieces[0] if len(pieces) == 1 else ttnn.concat(pieces, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if len(pieces) > 1:
            for pc in pieces:
                ttnn.deallocate(pc)
        if mode == ALL_GATHER:
            shape[-1] = shape[-1] * self.num_devices
        return ttnn.reshape(res, shape) if list(res.shape) != shape else res

    @staticmethod
    def _dram_tile(t):
        mc = t.memory_config()
        return (
            t.layout == ttnn.TILE_LAYOUT
            and not t.is_sharded()
            and mc.buffer_type == ttnn.BufferType.DRAM
            and t.padded_shape[-2] % 32 == 0
        )

    # ------------------------------------------------------------- collectives
    def all_reduce(self, t):
        """Sum over the two devices; output replicated with the input's layout."""
        if self._op_capable(t) and t.dtype in (ttnn.bfloat16, ttnn.bfloat8_b):
            return ttnn.experimental.pcie_exchange(
                t, self._exchange_for(t), ALL_REDUCE, bfp8_payload=self.bfp8_reduce and t.dtype == ttnn.bfloat16
            )
        if self._dram_tile(t) and t.dtype in (ttnn.bfloat16, ttnn.bfloat8_b) and t.padded_shape[-1] % (64 * 32) == 0:
            return self._exchange_dram(t, ALL_REDUCE, 64)
        return self._host_all_reduce(t)

    def all_gather(self, t, dim=-1):
        """Concatenate the two devices' tensors along the last dim; output replicated."""
        if dim not in (-1, 3, len(t.shape) - 1):
            return self._host_all_gather(t, dim)
        if self._op_capable(t):
            return ttnn.experimental.pcie_exchange(t, self._exchange_for(t), ALL_GATHER)
        # Prefill-sized DRAM tensors (e.g. the column-parallel embedding [1, S, dim/2]): 32 input cores -> 64 output.
        if self._dram_tile(t) and t.padded_shape[-2] > 32 and t.padded_shape[-1] % (32 * 32) == 0:
            return self._exchange_dram(t, ALL_GATHER, 32)
        # Decode-sized tensor not yet in an op layout: put it on the first 32 cores (or 1 core if narrow).
        if (
            t.layout == ttnn.TILE_LAYOUT
            and t.padded_shape[-2] == 32
            and _volume(t.padded_shape) == 32 * t.padded_shape[-1]
        ):
            w = t.padded_shape[-1]
            for n in (32, 16, 8, 4, 2, 1):
                if w % (n * 32) == 0:
                    ts = ttnn.to_memory_config(t, self.width_sharded(n, w // n))
                    out = ttnn.experimental.pcie_exchange(ts, self._exchange_for(ts), ALL_GATHER)
                    ttnn.deallocate(ts)
                    return out
        return self._host_all_gather(t, dim)

    def argmax_exchange(self, logits, tt_out_tok=None, valid_rows=None):
        """Greedy token over the column-parallel lm_head: local max/argmax, exchange 2 x [32,1], pick on device.

        logits: per-device [1, 1, 32, V_local] bf16 TILE (device d owns global columns [d*V_local, (d+1)*V_local)).
        valid_rows: only the first B rows carry users; the row-major argmax runs on those and the padded rows
        come back as token 0 (as the single-device fast path does). Returns uint32 tokens shaped like tt_out_tok
        (copied into it when given).
        """
        v_local = logits.padded_shape[-1]
        rows = logits.padded_shape[-2]
        b = valid_rows if valid_rows is not None and 0 < valid_rows < rows else rows
        if b <= self.ROW_GATHER_MAX_ROWS:
            return self._argmax_row_gather(logits, tt_out_tok, b, rows, v_local)
        vals = ttnn.max(logits, dim=-1, keepdim=True)  # [1,1,32,1] TILE, same dtype as the logits
        if vals.dtype != ttnn.bfloat16:  # prefill logits are bfp8; the exchange moves bf16/32-bit types only
            vals_bf = ttnn.typecast(vals, ttnn.bfloat16)
            ttnn.deallocate(vals)
            vals = vals_bf
        rm = ttnn.untilize(logits, use_multicore=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if b < rows:
            rm_b = ttnn.slice(rm, [0, 0, 0, 0], [1, 1, b, v_local], memory_config=rm.memory_config())
            ttnn.deallocate(rm)
            rm = rm_b
        idx = ttnn.argmax(rm, dim=-1, keepdim=True)  # [1,1,b,1] uint32 row-major
        ttnn.deallocate(rm)
        idx_t = ttnn.to_layout(idx, ttnn.TILE_LAYOUT)  # padded to a [32,32] tile
        ttnn.deallocate(idx)
        vals_g = self.all_gather(vals)  # [1,1,32,64]: column 32*d = device d's max (L1 width-sharded, 1 core)
        idx_g = self.all_gather(idx_t)  # [1,1,b,64]: column 32*d = device d's local argmax
        ttnn.deallocate(vals)
        ttnn.deallocate(idx_t)
        # Pick on device with elementwise ops only (argmax needs interleaved inputs; gt/typecast do not care).
        vals_i = ttnn.to_memory_config(vals_g, ttnn.L1_MEMORY_CONFIG)
        idx_i = ttnn.typecast(ttnn.to_memory_config(idx_g, ttnn.L1_MEMORY_CONFIG), ttnn.int32)
        ttnn.deallocate(vals_g)
        ttnn.deallocate(idx_g)
        # The gather concatenates whole tiles: device d's [32, 1] column sits at column 32 * d.
        v0 = ttnn.slice(vals_i, [0, 0, 0, 0], [1, 1, b, 1])
        v1 = ttnn.slice(vals_i, [0, 0, 0, 32], [1, 1, b, 33])
        sel_i = ttnn.typecast(ttnn.gt(v1, v0), ttnn.int32)  # 1 where device 1 holds the max, else 0
        i0 = ttnn.slice(idx_i, [0, 0, 0, 0], [1, 1, b, 1])
        i1 = ttnn.slice(idx_i, [0, 0, 0, 32], [1, 1, b, 33])
        # token = i0 + sel * (i1 + V_local - i0)
        diff = ttnn.subtract(ttnn.add(i1, v_local), i0)
        tok = ttnn.add(i0, ttnn.multiply(sel_i, diff))
        for x in (vals_i, idx_i, v0, v1, sel_i, i0, i1, diff):
            ttnn.deallocate(x)
        tok_u = ttnn.typecast(tok, ttnn.uint32)
        ttnn.deallocate(tok)
        tok_rm = ttnn.to_layout(tok_u, ttnn.ROW_MAJOR_LAYOUT)  # [1,1,b,1]
        ttnn.deallocate(tok_u)
        tok_rm = ttnn.reshape(tok_rm, [1, 1, b])
        if b < rows:
            pad = self._pad_tokens(rows - b, tok_rm)
            full = ttnn.concat([tok_rm, pad], dim=-1, memory_config=tok_rm.memory_config())
            ttnn.deallocate(tok_rm)
            tok_rm = full
        if tt_out_tok is not None:
            ttnn.copy(ttnn.reshape(tok_rm, tt_out_tok.shape), tt_out_tok)
            ttnn.deallocate(tok_rm)
            return tt_out_tok
        return tok_rm

    # Small batches: ship the valid logit rows themselves (128 KiB per row per device) and argmax locally.
    ROW_GATHER_MAX_ROWS = 8

    def _argmax_row_gather(self, logits, tt_out_tok, b, rows, v_local):
        """Untilize, keep the b valid rows, gather them row-major over PCIe (b x 128 KiB) and run one argmax over
        the full vocab: the global index comes out directly, no max/select arithmetic."""
        rm = ttnn.untilize(logits, use_multicore=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if b < rows:
            rows_t = ttnn.slice(rm, [0, 0, 0, 0], [1, 1, b, v_local], memory_config=rm.memory_config())
            ttnn.deallocate(rm)
        else:
            rows_t = rm
        # Row-major shards need 16 B aligned rows; 32 B keeps every NoC write aligned. 64128 cols -> 8 cores.
        n = 1
        for cand in (32, 16, 8, 4, 2):
            if v_local % cand == 0 and ((v_local // cand) * 2) % 32 == 0:
                n = cand
                break
        sh = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.num_cores_to_corerangeset(n, self.grid, True), [b, v_local // n], ttnn.ShardOrientation.ROW_MAJOR
            ),
        )
        rows_sh = ttnn.to_memory_config(rows_t, sh)
        ttnn.deallocate(rows_t)
        gathered = ttnn.experimental.pcie_exchange(rows_sh, self._exchange_for(rows_sh), ALL_GATHER)  # [1,1,b,2V] RM
        ttnn.deallocate(rows_sh)
        g_il = ttnn.to_memory_config(gathered, ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(gathered)
        tok = ttnn.argmax(g_il, dim=-1, keepdim=False)  # [1,1,b] uint32: global vocab index
        ttnn.deallocate(g_il)
        if b < rows:
            pad = self._pad_tokens(rows - b, tok)
            full = ttnn.concat([tok, pad], dim=-1, memory_config=tok.memory_config())
            ttnn.deallocate(tok)
            tok = full
        if tt_out_tok is not None:
            ttnn.copy(ttnn.reshape(tok, tt_out_tok.shape), tt_out_tok)
            ttnn.deallocate(tok)
            return tt_out_tok
        return tok

    def _pad_tokens(self, n, like):
        key = ("pad", n)
        pad = self._exchange.get(key)
        if pad is None:
            pad = ttnn.from_torch(
                torch.zeros(1, 1, n, dtype=torch.int32),
                dtype=like.dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh_device,
                memory_config=like.memory_config(),
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            self._exchange[key] = pad
        return pad

    # ----------------------------------------------------------- host fallback
    def _host_all_reduce(self, t):
        self.host_fallbacks += 1
        logger.warning(f"PcieCCL: host all-reduce fallback for {t.shape} {t.dtype} {t.memory_config()}")
        host = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0))
        n = self.num_devices
        host = host.view(n, *host.shape[1:]) if host.shape[0] == n * t.shape[0] else host
        summed = host.float().sum(0, keepdim=True).to(host.dtype)
        return ttnn.from_torch(
            summed,
            dtype=t.dtype,
            layout=t.layout,
            device=self.mesh_device,
            memory_config=t.memory_config(),
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _host_all_gather(self, t, dim):
        self.host_fallbacks += 1
        logger.warning(f"PcieCCL: host all-gather fallback for {t.shape} {t.dtype} dim {dim}")
        host = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=dim))
        return ttnn.from_torch(
            host,
            dtype=t.dtype,
            layout=t.layout,
            device=self.mesh_device,
            memory_config=t.memory_config(),
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
