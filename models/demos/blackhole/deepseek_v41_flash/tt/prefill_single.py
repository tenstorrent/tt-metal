# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""SINGLE-TRACE chunked prefill (env ``DSV41_PF_SINGLE=1``, default off; needs ``DSV41_PF_PAGED=read`` and the column-sharded indexer).

ONE chunk trace is captured for the build's maximum context (Wmax) and replayed for every chunk of every prompt, whatever its length, with a cost that follows the chunk's POSITION
(not Wmax). The S_pad dependent pieces of the old chunk forward (tt/prefill_sparse.py, tt/prefill_dyn.py) are replaced by position-indexed state read through device tensors:

  * index keys: the key slab of an index-key owner is no longer a FIFO whose newest chunk sits at the END (every chunk shifted the whole slab; the score op saw all L slots, a frozen host
    scalar ``kv_len``). Layout per mesh column c (global slot p lives on column p % cols at local slot p // cols, as the sharded FIFO):

        global slots [0, Cc)           SCRATCH: the CURRENT chunk's Cc = C / ratio entries (rewritten every replay, causally masked by a constant staircase),
        global slots [Cc, Cc + Lmax)   HISTORY: entry j at slot Cc + j (appended at a tensor-driven offset by ``key_append``, tt/paged_kernels/key_append.cpp)

    so at chunk start s0 the valid keys are the PREFIX [0, Cc + s0 / ratio) of the slab (all history entries are visible to every query of the chunk: no per-query mask, only the staircase
    on the scratch part) and its length is a device tensor (``nvalid``): ``indexer_score_dsa(kv_len_tensor=nvalid)`` (C++ change: branch ssinghal/dsv4p1-chunk1-cpp) scores and writes only that
    prefix, ``topk_large_indices(valid_length_tensor=nvalid)`` ranks only it. No O(L) mask, shift or layout conversion is left in the chunk.
  * the selected ids are slots of that layout; ``prefill_id_translate_single.cpp`` maps them to pool rows (the latents live in the paged pool, DSV41_PF_PAGED=read).
  * hand-off to decode: entry j sits at slot Cc + j (``DSV41Model._export_keys_users``).

Per replay the host refreshes (``SingleCtx.update``, one small upload each): ``nvalid[r]`` (local valid slots), ``kofs[r]`` (destination row block of the append per (mesh row, user); 0 = scratch for
users that are not part of the replay), and the pool translation word (tt/prefill_paged.py).
"""

import os

import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt import paged_ops as P
from models.demos.blackhole.deepseek_v41_flash.tt.h2d import h2d

IDIM = 128
CAPTURING = [
    False
]  # set by the chunk-trace capture sites (tt/prefill_model.py): nothing may synchronize inside a capture
TILE_BYTES = 2048  # bf16 32x32 tile


def enabled():
    return os.environ.get("DSV41_PF_SINGLE", "0") == "1"


def kv_len_tensor_available():
    """True when the loaded ttnn build has the ``kv_len_tensor`` argument of ``ttnn.experimental.indexer_score_dsa`` (C++ change of this release, ttnn/cpp/.../indexer_score; the shared build of the
    previous release lacks it). Checked on the nanobind signature, no device needed."""
    try:
        return "kv_len_tensor" in (ttnn.experimental.indexer_score_dsa.__doc__ or "")
    except Exception:  # noqa: BLE001
        return False


def resolve_server_default(chunk, max_ctx):
    """vLLM server: decide ONCE, before the model is built, whether the server runs the single chunk trace (DSV41_PF_SINGLE=1) and pin the answer in the environment (modes are never flipped at run time).
    Default ON when the paged read mode with the bf16 pool is active and the kv_len_tensor op is available; ``DSV41_PF_SINGLE=0`` keeps the per-S_pad chunk traces (+ DSV41_PF_SMALL_BUCKET contexts).
    ``DSV41_PF_SINGLE=1`` with a missing prerequisite falls back the same way (log line). -> (enabled, reason)."""
    want = os.environ.get("DSV41_PF_SINGLE")
    if want == "0":
        return False, "DSV41_PF_SINGLE=0"
    why = None
    if os.environ.get("DSV41_PF_PAGED", "read") != "read":
        why = f"DSV41_PF_PAGED={os.environ.get('DSV41_PF_PAGED')} (the single trace needs read mode)"
    elif os.environ.get("DSV41_PF_PAGED_FP8", "0") == "1":
        why = "DSV41_PF_PAGED_FP8=1 (fp8 pool)"
    elif os.environ.get("DSV41_PREFILL_DYN", "1") == "0":
        why = "DSV41_PREFILL_DYN=0"
    elif int(chunk) % 512:
        why = f"chunk {chunk} is not a multiple of 512 (tile split of the per-column key slab)"
    elif int(max_ctx) < 16384:
        why = f"max context {max_ctx} < 16384 (a column slab needs >= 512 local key slots for the top-k)"
    elif not kv_len_tensor_available():
        why = "the loaded ttnn build lacks indexer_score_dsa(kv_len_tensor=...) (build the C++ of branch ssinghal/dsv4p1-release: build_local/build_local.py)"
    if why:
        os.environ["DSV41_PF_SINGLE"] = "0"
        return False, why
    os.environ["DSV41_PF_SINGLE"] = "1"
    return True, "read mode + bf16 pool + kv_len_tensor op available"


def use_kv_tensor():
    """DSV41_PF_SINGLE_KVT=0 (debug): score with the host scalar kv_len = the whole slab (no C++ change needed; the top-k stays bounded by the tensor): the cost then follows Wmax again."""
    return os.environ.get("DSV41_PF_SINGLE_KVT", "1") == "1"


def shard_cols_single(md, C, r):
    """mesh columns the single-trace slab of ratio ``r`` is sharded over, 0 if the geometry does not allow it (chunk entries must split into whole 32-row tiles per column)."""
    cols = int(md.shape[1])
    Cc = C // r
    if cols < 2 or Cc % (cols * 32):
        return 0
    return cols


def _acc(t):
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


def key_append(src, kofs, dst, users, nrb, dst_rows):
    """dst[u, :, kofs[u] * 32 : kofs[u] * 32 + nrb * 32, :] = src[u] for every user u (IN PLACE, tile layout, one core).
    src [U,1,nrb*32,128] bf16 tile, kofs [rows*U, 32] uint32 RM (per device [U, 32]), dst [U,1,dst_rows,128] bf16 tile.
    """
    ct = [users, nrb * 4, (dst_rows // 32) * 4, TILE_BYTES, 0] + _acc(src) + _acc(kofs) + _acc(dst)
    md = dst.device()
    core = ttnn.CoreCoord(0, 0)
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    total = 128 + TILE_BYTES
    cb = ttnn.CBDescriptor(
        total_size=total,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.int32, page_size=total)],
    )
    rt = ttnn.RuntimeArgs()
    rt[core.x][core.y] = []
    k = ttnn.KernelDescriptor(
        kernel_source=f"{P.KDIR}/key_append.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=ct,
        runtime_args=rt,
        common_runtime_args=[src.buffer_address(), kofs.buffer_address(), dst.buffer_address()],
        config=ttnn.ReaderConfigDescriptor(),
    )
    prog = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[cb])
    prog.custom_program_hash = (0x9B1 << 40) | (hash(tuple(ct)) & ((1 << 40) - 1))
    ttnn.generic_op([src, kofs, dst], prog)
    del md


class SingleCtx:
    """Per DynCtx persistent tensors of the single-trace indexer (one entry per compress ratio that has an index-key owner)."""

    def __init__(self, md, ctx, U):
        self.md, self.ctx, self.U = md, ctx, U
        self.rows, self.cols = tuple(md.shape)
        self.nvalid, self.kofs = {}, {}
        self._rep = ttnn.ReplicateTensorToMesh(md)
        self._shard = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(self.rows, self.cols))

    def ensure(self, r):
        """tensors of ratio r (created once, before any trace capture)"""
        if r in self.nvalid:
            return
        md, U = self.md, self.U
        self.nvalid[r] = ttnn.from_torch(
            torch.zeros(1, 1, 1, 1, dtype=torch.int32),
            device=md,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self._rep,
        )
        self.kofs[r] = ttnn.from_torch(
            torch.zeros(self.rows * U, 32, dtype=torch.int32),
            device=md,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self._shard,
        )

    def values(self, r, s0, active):
        """(local valid slots, append row block per (row, user)) of ratio r for a replay at chunk start s0; ``active`` bool [rows * U]"""
        cols, C = self.cols, self.ctx.C
        Cc8 = (C // r) // cols
        nloc = Cc8 + (s0 // r) // cols
        assert nloc % 32 == 0, (r, s0, nloc)
        ko = torch.where(
            active, torch.full((self.rows * self.U,), nloc // 32), torch.zeros(self.rows * self.U, dtype=torch.long)
        )
        return nloc, ko

    def update(self, ctx, s0):
        """per-replay refresh (registered as a DynCtx upd hook)"""
        act = getattr(ctx, "active", None)
        act = torch.ones(self.rows * self.U, dtype=torch.bool) if act is None else torch.as_tensor(act).bool()
        for r in self.nvalid:
            nloc, ko = self.values(r, s0, act)
            h2d(
                ttnn.from_torch(
                    torch.full((1, 1, 1, 1), nloc, dtype=torch.int32),
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=self._rep,
                ),
                self.nvalid[r],
            )
            kk = torch.zeros(self.rows * self.U, 32, dtype=torch.int32)
            kk[:, 0] = ko.to(torch.int32)
            h2d(
                ttnn.from_torch(kk, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self._shard),
                self.kofs[r],
            )


def translate_ids_single(ids, pt, off, *, ratio, src_off, C, ccur, n_cores=64):
    """``prefill_paged.translate_ids`` for the single-trace slot layout (ids: slots [0, ccur) = the current chunk's entries off[0] + s, slots >= ccur = entry s - ccur)."""
    md = ids.device()
    U = int(ids.shape[0])
    NID = int(ids.shape[-1])
    rows = U * C
    n = min(n_cores, rows)
    while rows % n or C % (rows // n) and (rows // n) % C:
        n -= 1
    rpc = rows // n
    assert C % rpc == 0 or rpc % C == 0
    assert rpc <= C, "a core handles rows of one user only"
    maxp = int(pt.shape[-1])
    ids2 = ttnn.reshape(ids, [rows, NID])
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([rows, NID]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, md, ttnn.DRAM_MEMORY_CONFIG
    )
    cores = [ttnn.CoreCoord(i % 8, i // 8) for i in range(n)]
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    total = ((maxp * 4 + 63) // 64) * 64 + 64 + 2 * NID * 4
    total = (total + 63) // 64 * 64
    cb = ttnn.CBDescriptor(
        total_size=total,
        core_ranges=core_set,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.int32, page_size=total)],
    )
    ct = [rpc, C, ratio, src_off, maxp, NID, P.PAGE_TOKENS, P.PAGE_ROWS, 0, ccur]
    ct += _acc(ids2) + _acc(pt) + _acc(off) + _acc(out)
    rt = ttnn.RuntimeArgs()
    for i, c in enumerate(cores):
        rt[c.x][c.y] = [i]
    k = ttnn.KernelDescriptor(
        kernel_source=f"{P.KDIR}/prefill_id_translate_single.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=ct,
        runtime_args=rt,
        common_runtime_args=[ids2.buffer_address(), pt.buffer_address(), off.buffer_address(), out.buffer_address()],
        config=ttnn.ReaderConfigDescriptor(),
    )
    prog = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[cb])
    prog.custom_program_hash = (0x9B2 << 40) | (hash(tuple(ct)) & ((1 << 40) - 1))
    res = ttnn.generic_op([ids2, pt, off, out], prog)
    return ttnn.reshape(res, [U, 1, C, NID])


class Ticker:
    """DSV41_PF_SINGLE_PROF=1: per-step device time (synchronize after every step) of the single-trace indexer in the EAGER passes (compile pass), printed once per call; a no-op otherwise
    (inside a trace capture nothing may synchronize)."""

    on = os.environ.get("DSV41_PF_SINGLE_PROF", "0") == "1"

    def __init__(self, md, name):
        import time

        self.md, self.name, self.t = md, name, None
        self.acc = {}
        self.act = Ticker.on and not CAPTURING[0]
        self._time = time
        if self.act:
            ttnn.synchronize_device(md)
            self.t = time.perf_counter()

    def tick(self, label):
        if not self.act:
            return
        ttnn.synchronize_device(self.md)
        t = self._time.perf_counter()
        self.acc[label] = self.acc.get(label, 0.0) + t - self.t
        self.t = t

    def report(self):
        if self.act:
            print(
                "PF_SINGLE_PROF " + self.name + ": " + ", ".join(f"{k} {v * 1e3:.2f} ms" for k, v in self.acc.items()),
                flush=True,
            )
