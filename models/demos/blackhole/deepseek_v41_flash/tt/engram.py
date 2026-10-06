# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Engram (layers 1 and 14) with the heavy part on the device.

The n-gram table lookup depends only on token ids, so the host does it (``HostEngramRows``: hash ids + the lazily read
rows of the ~100 GB table) and uploads ``rows`` [T,1,1,Kin] bf16 once per step. The device does the rest of
``Engram.forward`` of the checkpoint's model.py:

    kv    = rows @ wkv^T                           [T,1,1,(hc+1)*D]    -> key [T,1,hc,D], value [T,1,1,D]
    rstd  = rsqrt(mean(h^2)+eps) * rsqrt(mean(key^2)+eps)              per (token, hc copy)
    dot   = sum(h * (q*k_weight) * key) * rstd * D^-0.5
    gate  = sigmoid(copysign(sqrt(max(|dot|, 1e-6)), dot))
    out   = h + gate * value
"""

import os

import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference import ref_kernels
from models.demos.blackhole.deepseek_v41_flash.tt.host_model import HostEngram
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import _Shards


class _MappedTable:
    """The fp8 rows + e8m0 scales of one Engram table, memory-mapped straight from the safetensors file so a lookup is a
    vectorised gather (the safetensors ``get_slice`` path costs ~18 us per row in Python)."""

    def __init__(self, sh: _Shards, layer_id: int):
        import json
        import struct

        import numpy as np

        wkey, skey = f"layers.{layer_id}.engram.embed.weight", f"layers.{layer_id}.engram.embed.scale"
        path = os.path.join(sh.dir, sh.index[wkey])
        with open(path, "rb") as f:
            (hlen,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(hlen))
        base = 8 + hlen

        def view(key, dtype):
            m = header[key]
            a, b = m["data_offsets"]
            raw = np.memmap(path, dtype=np.uint8, mode="r", offset=base + a, shape=(b - a,))
            return torch.from_numpy(raw).view(dtype).reshape(m["shape"])

        self.w = view(wkey, torch.float8_e4m3fn)  # [N, 256]
        self.s = view(skey, torch.float8_e8m0fnu)  # [N, 8]
        self._shapes = (self.w.shape, self.s.shape)
        self._regions = [
            (path, base + header[k]["data_offsets"][0], header[k]["data_offsets"][1] - header[k]["data_offsets"][0])
            for k in (wkey, skey)
        ]
        self.fd = os.open(path, os.O_RDONLY)
        self.woff, self.soff = self._regions[0][1], self._regions[1][1]

    def prefetch(self, threads=16, chunk=256 << 20):
        """Read the whole table once so it sits in the host page cache (the lookups are random 256-byte reads: ~1 ms each from
        cold NFS, ~0.3 us from RAM). ~100 GB per layer; blocks until done."""
        from concurrent.futures import ThreadPoolExecutor

        def read(job):
            path, off, n = job
            with open(path, "rb", buffering=0) as f:
                f.seek(off)
                left = n
                while left > 0:
                    left -= len(f.read(min(chunk, left)))

        jobs = [
            (path, off + i, min(chunk * 4, n - i)) for path, off, n in self._regions for i in range(0, n, chunk * 4)
        ]
        with ThreadPoolExecutor(threads) as ex:
            list(ex.map(read, jobs))

    def rows(self, idx: torch.Tensor) -> torch.Tensor:
        flat = idx.reshape(-1)
        v = self.w[flat].float().unflatten(-1, (-1, 32)) * self.s[flat].float().unsqueeze(-1)
        return v.flatten(-2).to(torch.bfloat16).reshape(*idx.shape, -1)

    in_ram = False

    def load_ram(self, threads=16, chunk=1 << 30, huge=True):
        """Copy the whole table (fp8 rows + e8m0 scales, ~100 GB) into this process's memory. The page cache cannot be relied on to
        keep it (the host's RAM is nearly all page cache: a prefetch evicts part of itself), and random 256-byte reads over NFS cost
        ~60 us each; from RAM a lookup is ~0.3 us."""
        from concurrent.futures import ThreadPoolExecutor

        out = []
        for (path, off, n), dtype, shape in (
            (self._regions[0], torch.float8_e4m3fn, self._shapes[0]),
            (self._regions[1], torch.float8_e8m0fnu, self._shapes[1]),
        ):
            buf = torch.empty(n, dtype=torch.uint8)
            if (
                huge
            ):  # THP is in 'madvise' mode on these hosts: ask for 2 MB pages before the first touch (fewer TLB misses on random rows)
                import ctypes
                import ctypes.util

                libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
                a, al = buf.data_ptr(), 2 << 20
                st = (a + al - 1) // al * al
                libc.madvise(ctypes.c_void_p(st), ctypes.c_size_t((a + n - st) // al * al), 14)  # MADV_HUGEPAGE
            mv = memoryview(buf.numpy())

            def read(i, path=path, off=off, n=n, mv=mv):
                size = min(chunk, n - i)
                view = mv[i : i + size]
                with open(path, "rb", buffering=0) as f:
                    f.seek(off + i)
                    got = 0
                    while got < size:
                        k = f.readinto(view[got:])
                        if not k:
                            raise IOError(f"short read of {path} at {off + i + got}")
                        got += k

            with ThreadPoolExecutor(threads) as ex:
                list(ex.map(read, range(0, n, chunk)))
            out.append(buf.view(dtype).reshape(shape))
        self.w, self.s = out
        self.in_ram = True

    def read_task(self, k):
        """One row's raw bytes (fp8 row, e8m0 scales): two positional reads, GIL-free so a thread pool overlaps the NFS latency."""
        return os.pread(self.fd, 256, self.woff + k * 256), os.pread(self.fd, 8, self.soff + k * 8)

    @staticmethod
    def dequant(raw, shape):
        w = (
            torch.frombuffer(bytearray(b"".join(r[0] for r in raw)), dtype=torch.uint8)
            .view(torch.float8_e4m3fn)
            .reshape(len(raw), 256)
        )
        s = (
            torch.frombuffer(bytearray(b"".join(r[1] for r in raw)), dtype=torch.uint8)
            .view(torch.float8_e8m0fnu)
            .reshape(len(raw), 8)
        )
        v = w.float().unflatten(-1, (-1, 32)) * s.float().unsqueeze(-1)
        return v.flatten(-2).to(torch.bfloat16).reshape(*shape, -1)


class HostEngramRows:
    """Token ids -> the rows each Engram layer reads, [B, 1, Kin] bf16 (24 rows of the table per token and layer)."""

    def __init__(self, layer_ids=(1, 14), max_batch_size=16, max_seq_len=256):
        self.engram = HostEngram(layer_ids, max_batch_size, max_seq_len)
        sh = _Shards()
        self.tables = {lid: _MappedTable(sh, lid) for lid in layer_ids}

    def prefetch(self, **kw):
        """Warm the page cache for all tables (see ``_MappedTable.prefetch``)."""
        for t in self.tables.values():
            t.prefetch(**kw)

    def load_ram(self, **kw):
        """Hold all tables in this process's memory (~100 GB per layer): lookups never touch NFS again."""
        for t in self.tables.values():
            t.load_ram(**kw)

    def hashes(self, input_ids, start_pos):
        return self.engram.hashes(input_ids, start_pos)

    @torch.no_grad()
    def rows(self, layer_id, hashes):
        e = self.engram.mods[layer_id]
        ids = hashes[:, :, e.layer_hash_index, :]  # [B, L, n_hash_cols]
        return self.tables[layer_id].rows(ids).flatten(-2)  # [B, L, Kin]

    _pool = None

    @torch.no_grad()
    def rows_all(self, hashes, layer_ids=None):
        """Rows of every Engram layer at once: all the random 256-byte reads of all layers go through ONE thread pool
        (~10 ms for 2 layers x 16 users x 24 rows over NFS, vs ~48 ms per layer for the memory-mapped gather)."""
        from concurrent.futures import ThreadPoolExecutor

        if all(t.in_ram for t in self.tables.values()):
            return {lid: self.rows(lid, hashes) for lid in (layer_ids or list(self.tables))}
        if HostEngramRows._pool is None:
            HostEngramRows._pool = ThreadPoolExecutor(128)
        layer_ids = layer_ids or list(self.tables)
        tasks, spans, shapes = [], {}, {}
        for lid in layer_ids:
            e = self.engram.mods[lid]
            ids = hashes[:, :, e.layer_hash_index, :]
            flat = ids.reshape(-1).tolist()
            spans[lid] = (len(tasks), len(tasks) + len(flat))
            shapes[lid] = tuple(ids.shape)
            tasks += [(self.tables[lid], k) for k in flat]
        raw = list(HostEngramRows._pool.map(lambda t: t[0].read_task(t[1]), tasks))
        return {lid: _MappedTable.dequant(raw[a:b], shapes[lid]).flatten(-2) for lid, (a, b) in spans.items()}

    @torch.no_grad()
    def rows_reference(self, layer_id, hashes):
        """The slow per-row path of HostEngram (for tests)."""
        e = self.engram.mods[layer_id]
        return e.embed(hashes[:, :, e.layer_hash_index, :]).flatten(-2)


class DSV41DeviceEngram:
    def __init__(
        self, mesh_device, layer_id, shards=None, hc=4, dim=5120, eps=1e-6, users_per_row=4, mesh_config=None, ccl=None
    ):
        """``mesh_config`` + ``ccl``: shard the wkv projection (167 MB bfp8) over the mesh columns and all-gather the 205 KB result
        (each device then reads 21 MB, ~0.1 ms, instead of ~0.7 ms); without them the weight is replicated."""
        sh = shards or _Shards()
        p = f"layers.{layer_id}.engram."
        self.md, self.hc, self.dim, self.eps = mesh_device, hc, dim, eps
        wkv = ref_kernels.dequant_fp8_weight(sh.get(p + "wkv.weight"), sh.get(p + "wkv.scale"), 32)  # [(hc+1)*D, Kin]
        rep = ttnn.ReplicateTensorToMesh(mesh_device)
        up = lambda t, dt, lay=ttnn.TILE_LAYOUT: ttnn.from_torch(
            t, device=mesh_device, dtype=dt, layout=lay, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep
        )
        self.kin = wkv.shape[1]
        self.mesh_config, self.ccl = mesh_config, ccl
        w_t = wkv.t().contiguous().reshape(1, 1, self.kin, -1).to(torch.bfloat16)
        if mesh_config is None:
            self.wkv_T = up(w_t, ttnn.bfloat8_b)
        else:  # output columns split over the mesh columns (25600 / 8 = 3200 = 100 tiles each)
            rows_, cols_ = tuple(mesh_device.shape)
            assert w_t.shape[-1] % (32 * cols_) == 0
            self.wkv_T = ttnn.from_torch(
                w_t,
                device=mesh_device,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 3), mesh_shape=(rows_, cols_)),
            )
        self.weight = up(
            (sh.get(p + "q_weight").float() * sh.get(p + "k_weight").float()).reshape(1, 1, hc, dim), ttnn.float32
        )
        self.ckc = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True
        )
        # forward_v2 constants: D^-0.5 folded into the weight; fp32 HiFi4 for the norms
        self._w_scaled = up(
            (sh.get(p + "q_weight").float() * sh.get(p + "k_weight").float() * dim**-0.5).reshape(1, 1, hc, dim),
            ttnn.float32,
        )
        self._w_scaled_torch = (sh.get(p + "q_weight").float() * sh.get(p + "k_weight").float() * dim**-0.5).reshape(
            hc, dim
        )
        self._w_rows = (
            {}
        )  # T -> weight tiled to [1,1,T*hc,D] (built on first use of that T: call once before trace capture)
        self.ckc_norm = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
        )
        self._rep = rep
        self._ones_col = ttnn.from_torch(
            torch.ones(1, 1, dim, 1),
            device=mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )

    def forward(self, x, rows):
        """x [T,1,hc,D] fp32 streams; rows [T,1,1,Kin] bf16 (tile) -> streams [T,1,hc,D] fp32."""
        T, hc, D = x.shape[0], self.hc, self.dim
        kv = ttnn.matmul(
            rows, self.wkv_T, compute_kernel_config=self.ckc, dtype=ttnn.bfloat16, core_grid=ttnn.CoreGrid(y=8, x=8)
        )  # [T,1,1,(hc+1)D]; default config: 1.9 ms, 8x8 grid: 0.7 ms
        if self.mesh_config is not None:  # each column computed its 1/8 of the outputs: gather them back
            kv = self.mesh_config.allgather(kv, self.ccl, axis=1, dim=3)
        kv = ttnn.typecast(kv, ttnn.float32)
        key = ttnn.reshape(kv[:, :, :, : hc * D], [T, 1, hc, D])
        value = kv[:, :, :, hc * D :]  # [T,1,1,D]
        ms = lambda t: ttnn.mean(ttnn.multiply(t, t), dim=-1, keepdim=True)
        rstd = ttnn.multiply(ttnn.rsqrt(ttnn.add(ms(x), self.eps)), ttnn.rsqrt(ttnn.add(ms(key), self.eps)))
        dot = ttnn.multiply(
            ttnn.sum(ttnn.multiply(ttnn.multiply(x, self.weight), key), dim=-1, keepdim=True), D**-0.5
        )
        dot = ttnn.multiply(dot, rstd)  # [T,1,hc,1]
        mag = ttnn.sqrt(ttnn.clamp(ttnn.abs(dot), min=1e-6))
        gate = ttnn.sigmoid(ttnn.multiply(mag, ttnn.sign(dot)))  # copysign(sqrt(max(|dot|, eps)), dot)
        return ttnn.add(x, ttnn.multiply(gate, value))

    def _weight_rows(self, T):
        if T not in self._w_rows:
            self._w_rows[T] = ttnn.from_torch(
                self._w_scaled_torch.repeat(T, 1).reshape(1, 1, T * self.hc, self.dim),
                device=self.md,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=self._rep,
            )
        return self._w_rows[T]

    def forward_v2(self, x, rows, compact=True, rm_key=True, fuse_out=True, mode="matmul"):
        """Same maths as ``forward`` (fp32, PCC >= 0.99999) in ~12 ops instead of ~30:

        * the kv split happens in ROW_MAJOR (slice + view) and, with ``compact``, key / x are viewed as [1,1,T*hc,D] so the T*hc
          = 16 rows fill half a tile instead of 4 tiles' first rows (4 rows pad to 32: 8x fewer tiles to read / reduce);
        * rstd_x * rstd_k is two fused ``rms_norm`` ops (x*rsqrt(mean(x^2)+eps)), and D^-0.5 sits in the weight constant, so
          ``dot = sum(xn * w' * kn)`` is 2 multiplies + 1 sum;
        * copysign(sqrt(max(|dot|,1e-6)), dot) and the sigmoid are ONE binary op (lhs: abs/max/sqrt, rhs: sign, post: sigmoid);
        * ``x + gate * value`` is one addcmul (``fuse_out``).
        Call once eagerly before trace capture (the compact weight rows for this T are uploaded on first use).
        """
        T, hc, D = x.shape[0], self.hc, self.dim
        R = T * hc
        U, UT = ttnn.UnaryWithParam, ttnn.UnaryOpType
        # rows: [1,1,T,Kin] (one M=T matmul: 73 us vs 113/439/868 us at T=4/16/32 for T batched M=1 matmuls) or the legacy [T,1,1,Kin]
        if mode == "matmul" and rows.shape[2] != T:
            rows = ttnn.reshape(rows, [1, 1, T, rows.shape[-1]])
        elif mode != "matmul" and rows.shape[2] == T and T > 1:
            rows = ttnn.reshape(rows, [T, 1, 1, rows.shape[-1]])
        kv = ttnn.matmul(
            rows, self.wkv_T, compute_kernel_config=self.ckc, dtype=ttnn.bfloat16, core_grid=ttnn.CoreGrid(y=8, x=8)
        )  # [1,1,T,(hc+1)D] (matmul mode) / [T,1,1,(hc+1)D]
        if self.mesh_config is not None:
            kv = self.mesh_config.allgather(kv, self.ccl, axis=1, dim=3)
        if mode == "matmul":
            return self._gate_matmul(x, kv)
        if rm_key:
            kv = ttnn.to_layout(kv, ttnn.ROW_MAJOR_LAYOUT)
            key = ttnn.reshape(kv[:, :, :, : hc * D], [1, 1, R, D] if compact else [T, 1, hc, D])
            value = kv[:, :, :, hc * D :]
            key = ttnn.typecast(ttnn.to_layout(key, ttnn.TILE_LAYOUT), ttnn.float32)
            value = ttnn.typecast(ttnn.to_layout(value, ttnn.TILE_LAYOUT), ttnn.float32)
        else:
            kv = ttnn.typecast(kv, ttnn.float32)
            key = ttnn.reshape(kv[:, :, :, : hc * D], [1, 1, R, D] if compact else [T, 1, hc, D])
            value = kv[:, :, :, hc * D :]
        xr = ttnn.reshape(x, [1, 1, R, D]) if compact else x
        w = self._weight_rows(T) if compact else self._w_scaled
        xn = ttnn.rms_norm(xr, epsilon=self.eps, compute_kernel_config=self.ckc_norm)
        kn = ttnn.rms_norm(key, epsilon=self.eps, compute_kernel_config=self.ckc_norm)
        dot = ttnn.sum(ttnn.multiply(ttnn.multiply(xn, w), kn), dim=-1, keepdim=True)  # [.., 1] fp32
        gate = ttnn.multiply(
            dot,
            dot,
            input_tensor_a_activations=[U(UT.ABS), U(UT.MAXIMUM, 1e-6), U(UT.SQRT)],
            input_tensor_b_activations=[U(UT.SIGN)],
            activations=[U(UT.SIGMOID)],
        )
        if compact:
            gate = ttnn.reshape(gate, [T, 1, hc, 1])
        if fuse_out:
            return ttnn.addcmul(x, gate, value)
        return ttnn.add(x, ttnn.multiply(gate, value))

    def forward_v2_own(self, x, rows):
        """Column-split Engram: ``x`` [T,1,hc,D] is THIS column's own chunk, ``rows`` [1,1,n*T,Kin] the rows of the n = #columns chunks of the group
        (chunk j of the group belongs to column j; replicated on every device). The kv projection is sharded over the columns, so ONE matmul over all
        n*T tokens gives each column its 1/n slice of every chunk's kv; an all_to_all (hidden shards -> own tokens) hands every column the FULL kv of
        its own chunk, and the gating runs on the own T tokens only. Same maths as ``forward_v2`` on the gathered chunks (bit-identical), without the
        all_gather of x, the n separate calls and the reduce_scatter of the n identical copies."""
        kv = ttnn.matmul(
            rows, self.wkv_T, compute_kernel_config=self.ckc, dtype=ttnn.bfloat16, core_grid=ttnn.CoreGrid(y=8, x=8)
        )  # [1,1,n*T,(hc+1)D/n]
        kv = ttnn.experimental.all_to_all_async_generic(
            kv, in_dim=3, out_dim=2, num_links=self.ccl.num_links, topology=ttnn.Topology.Ring, cluster_axis=1
        )  # [1,1,T,(hc+1)D]
        return self._gate_matmul(x, kv)

    def _gate_matmul(self, x, kv):
        """kv: [1,1,T,(hc+1)D] (row t = token t)."""
        """Everything after the kv matmul in the compact [1,1,T*hc,D] layout, reductions over D as matmuls with a ones column
        (``rms_norm`` / ``ttnn.sum`` run one tile row on ONE core: 160 serial tiles, 44-68 us each; the matmul with a [D,1] ones
        spreads nothing either but streams K 3x faster), the three products read straight from the elementwise ops."""
        T, hc, D = x.shape[0], self.hc, self.dim
        R = T * hc
        U, UT = ttnn.UnaryWithParam, ttnn.UnaryOpType
        kv = ttnn.to_layout(kv, ttnn.ROW_MAJOR_LAYOUT)
        key = ttnn.typecast(
            ttnn.to_layout(ttnn.reshape(kv[:, :, :, : hc * D], [1, 1, R, D]), ttnn.TILE_LAYOUT), ttnn.float32
        )
        vrep = ttnn.repeat(
            ttnn.reshape(kv[:, :, :, hc * D :], [T, 1, 1, D]), ttnn.Shape([1, 1, hc, 1])
        )  # [T,1,hc,D]: each token's value for its hc streams
        value = ttnn.typecast(ttnn.to_layout(ttnn.reshape(vrep, [1, 1, R, D]), ttnn.TILE_LAYOUT), ttnn.float32)
        xr = ttnn.reshape(x, [1, 1, R, D])
        w = self._weight_rows(T)
        mm = lambda t: ttnn.matmul(
            t,
            self._ones_col,
            compute_kernel_config=self.ckc_norm,
            dtype=ttnn.float32,
            core_grid=ttnn.CoreGrid(y=2, x=8),
        )
        a = mm(ttnn.multiply(xr, xr))  # [1,1,R,1] sum x^2
        b = mm(ttnn.multiply(key, key))
        c = mm(ttnn.multiply(ttnn.multiply(xr, w), key))  # sum x*w*key*D^-0.5
        pre = [U(UT.MUL_UNARY_SFPU, 1.0 / D), U(UT.ADD_UNARY_SFPU, self.eps), U(UT.RSQRT)]
        rstd = ttnn.multiply(
            a, b, input_tensor_a_activations=pre, input_tensor_b_activations=pre
        )  # rsqrt(ms_x+eps)*rsqrt(ms_k+eps)
        dot = ttnn.multiply(c, rstd)
        gate = ttnn.multiply(
            dot,
            dot,
            input_tensor_a_activations=[U(UT.ABS), U(UT.MAXIMUM, 1e-6), U(UT.SQRT)],
            input_tensor_b_activations=[U(UT.SIGN)],
            activations=[U(UT.SIGMOID)],
        )
        out = ttnn.addcmul(xr, gate, value)  # [1,1,R,D]
        return ttnn.reshape(out, [T, 1, hc, D])
