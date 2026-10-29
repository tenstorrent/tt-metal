# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mesh plan, CCL and weight helpers shared by the MiMo TTNN modules.

Parallel plan (Blackhole Galaxy, 4x8):

* TP=8 along mesh axis 1. One replica is one Galaxy row (a 1x8 submesh). Every collective in a
  replica runs on ``cluster_axis=1``, so it stays inside the row's ring.
* DP=4 along mesh axis 0. The four rows are independent replicas with their own weights, KV cache
  and users (see ``tt/generator.py``). No collective crosses rows.

Per chip, both attention layer types see the same shape: 8 of the 64 Q heads and 1 KV head (SWA
layers have 8 KV heads, one per chip; global layers have 4, so chips 2i and 2i+1 both hold KV head i).
Routed experts are tensor-parallel too: every chip holds all 256 experts with 1/8 of
``moe_intermediate_size`` (2048 / 8 = 256), and the expert output is all-reduced across the row.
"""

import contextlib
import fcntl
import math
import os

import torch

import ttnn

TILE = ttnn.TILE_SIZE


class MeshPlan:
    """TP on ``tp_axis`` of a (sub)mesh. For a 1x8 replica ``dp == 1``; a 4x8 parent has ``dp == 4``."""

    def __init__(self, mesh_device, tp_axis=1, num_links=2, topology=ttnn.Topology.Ring):
        self.mesh_device = mesh_device
        self.mesh_shape = tuple(mesh_device.shape)
        self.tp_axis = tp_axis
        self.tp = self.mesh_shape[tp_axis]
        self.num_links = num_links
        self.topology = topology if self.tp > 2 else ttnn.Topology.Linear
        # ``prefill_reduce``: how all_reduce reduces a prefill pass's rows (see there).
        self.reduce_mode = None
        self.sum_config = compute_config(ttnn.MathFidelity.HiFi4, fp32_acc=True, packer_l1_acc=False)

    def shard(self, dim):
        """Shard tensor ``dim`` across the TP axis, replicate across the other axis."""
        dims = [None, None]
        dims[self.tp_axis] = dim
        return ttnn.ShardTensor2dMesh(self.mesh_device, dims=tuple(dims), mesh_shape=self.mesh_shape)

    def replicate(self):
        return ttnn.ReplicateTensorToMesh(self.mesh_device)

    @contextlib.contextmanager
    def prefill_reduce(self, users, padded):
        """Inside: ``all_reduce`` of a prefill pass's rows [1, 1, users * padded, hidden] (``users`` chunks padded to
        ``padded``, user after user) gives every user's rows the sum the same chunk would get alone, bit for bit, so a
        user's prefill does not depend on what it was batched with. ``ttnn.all_reduce``'s order of adding a row's chips
        changes with the rows around it. For padded >= 96 reducing the pass as a batch of ``users`` [1, 1, padded,
        hidden] tensors already matches each chunk alone (checked for 2-4 users, padded 96-512); shorter chunks
        all-gather the chips and sum them in chip order in fp32, alone or batched (8x the input's memory while it
        runs)."""
        assert self.reduce_mode is None, "prefill_reduce does not nest"
        if padded <= 64:
            self.reduce_mode = ("ordered", users, padded)
        elif users > 1:
            self.reduce_mode = ("users", users, padded)
        try:
            yield
        finally:
            self.reduce_mode = None

    def all_reduce(self, x, memory_config=None):
        if self.tp == 1:
            return x
        mode = self.reduce_mode
        if mode is not None and mode[0] == "users":
            users, padded = mode[1], mode[2]
            assert x.shape[-2] == users * padded, (x.shape, mode)
            batch = ttnn.reshape(x, (users, 1, padded, x.shape[-1]))  # the same tiles, user after user
            out = ttnn.all_reduce(
                batch,
                cluster_axis=self.tp_axis,
                num_links=self.num_links,
                topology=self.topology,
                memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
            )
            x.deallocate(True)
            return ttnn.reshape(out, (1, 1, users * padded, out.shape[-1]))
        if mode is not None and mode[0] == "ordered":
            gathered = ttnn.all_gather(x, dim=0, cluster_axis=self.tp_axis, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            x.deallocate(True)
            out = ttnn.sum(
                gathered,
                dim=0,
                keepdim=True,
                memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.sum_config,
            )
            gathered.deallocate(True)
            return out
        out = ttnn.all_reduce(
            x,
            cluster_axis=self.tp_axis,
            num_links=self.num_links,
            topology=self.topology,
            memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
        )
        x.deallocate(True)
        return out

    def all_gather(self, x, dim, memory_config=None):
        if self.tp == 1:
            return x
        # ttnn.all_gather picks its links and topology itself; passing num_links / topology only logs a deprecation
        # warning on every call (thousands of lines per serving run).
        return ttnn.all_gather(
            x, dim=dim, cluster_axis=self.tp_axis, memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG
        )

    def to_torch_tp_concat(self, x, dim):
        """Host readback of a TP-sharded tensor from the first replica row: concat the TP shards on ``dim``."""
        shards = ttnn.get_device_tensors(x)
        row = [ttnn.to_torch(shards[i]) for i in range(self.tp)]
        return torch.cat(row, dim=dim)

    def to_torch_replicated(self, x):
        return ttnn.to_torch(ttnn.get_device_tensors(x)[0])


def cache_name(cache_dir, *parts):
    if cache_dir is None:
        return None
    return os.path.join(cache_dir, "_".join(str(p) for p in parts))


@contextlib.contextmanager
def cache_fill(cache_dir, tag):
    """Yields whether ``tag``'s tensors must be converted (its ``complete`` marker is missing; always without a
    ``cache_dir``), and writes the marker when the block finishes.

    Several processes may build one cache at once (``serving/serve_rows.sh``: one vLLM engine per Galaxy row, all on
    one ``MIMO_TT_CACHE``), and ``ttnn.as_tensor`` writes a cache file in place. So a cold tag is filled under an
    exclusive lock on ``<tag>_lock``: one process converts and writes it, the others wait, then find it complete and
    load it. A complete tag takes no lock."""
    if cache_dir is None:
        yield True
        return
    done = cache_name(cache_dir, tag, "complete")
    if os.path.exists(done):
        yield False
        return
    with open(cache_name(cache_dir, tag, "lock"), "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # released when the file closes, also when the block raises
        need = not os.path.exists(done)
        yield need
        if need:
            open(done, "w").close()


def cache_exists(cache_file, dtype, layout=ttnn.TILE_LAYOUT):
    """True when ``as_weight`` would load ``cache_file`` without needing the host tensor."""
    if cache_file is None:
        return False
    return os.path.exists(f"{cache_file}_dtype_{dtype.name}_layout_{layout.name}.tensorbin")


BFP_MANTISSA_BITS = {ttnn.bfloat8_b: 7, ttnn.bfloat4_b: 3}


def bfp_project_blocks(x, bits, offsets=(0, -1, 1)):
    """Round [n, 16] fp32 blocks onto the BFP grid with ``bits`` mantissa bits; each block takes the shared exponent
    ``e + offset`` (``e``: its largest element exponent, the stock packer's) of least squared error, ties to the
    earlier offset. Returns fp32 values the stock packer reproduces exactly (no signed zeros)."""
    max_code, scale_offset = float((1 << bits) - 1), 126 + bits
    a = x.abs()
    e = ((x.view(torch.int32) >> 23) & 0xFF).amax(-1, keepdim=True)
    best_err = best_step = None
    for offset in offsets:
        cand = e + offset
        # Power-of-two scaling and round-half-even are exact in fp32 for these magnitudes.
        step = torch.ldexp(torch.ones_like(a[:, :1]), (cand - scale_offset).float())
        err = (torch.round(a / step).clamp_(max=max_code) * step - a).square_().sum(-1, keepdim=True)
        if offset:
            err = torch.where((cand >= bits) & (cand <= 254), err, torch.full_like(err, float("inf")))
        if best_err is None:
            best_err, best_step = err, step
        else:
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_step = torch.where(better, step, best_step)
    q = torch.round(a / best_step).clamp_(max=max_code) * best_step
    return torch.where(q > 0, torch.copysign(q, x), torch.zeros_like(x))


def bfp_mse_prepare(w, dtype, chunk=1 << 24):
    """Values on the ``dtype`` (bfloat8_b / bfloat4_b) grid with each block's shared exponent chosen for least error.

    ``w`` [..., N] is in its stored layout: the 16-value blocks of a tile face row run along the last dim, so blocks
    are 16 consecutive columns. The stock host packer gives every block the largest element exponent ``e`` and
    rounds; here each block takes the best of ``{e, e - 1, e + 1}`` (``e - 1`` clips the largest elements) under
    squared error, with ties to ``e``. The result is exactly representable in bf16, and the stock packer reproduces
    it bit for bit, so the device format, memory and speed are unchanged (tt-quant's ``mse`` policy,
    github.com/tenstorrent/tt-quant). On MiMo's MXFP4 experts in bfloat4_b it raises the weight SQNR from ~18.3 to
    ~20.9 dB.
    """
    bits = BFP_MANTISSA_BITS[dtype]
    shape = w.shape
    assert shape[-1] % 16 == 0, shape
    flat = w.reshape(-1, 16)
    out = torch.empty(flat.shape, dtype=torch.bfloat16)
    rows = chunk // 16
    for s in range(0, flat.shape[0], rows):
        out[s : s + rows] = bfp_project_blocks(flat[s : s + rows].float(), bits).to(torch.bfloat16)
    return out.reshape(shape)


def as_weight(torch_tensor, plan, mapper, dtype, cache_file, layout=ttnn.TILE_LAYOUT, memory_config=None):
    """``ttnn.as_tensor`` that loads from the cache when ``torch_tensor`` is None and the cache is warm."""
    return ttnn.as_tensor(
        torch_tensor,
        device=plan.mesh_device,
        dtype=dtype,
        layout=layout,
        mesh_mapper=mapper,
        memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=cache_file,
    )


def compute_config(fidelity=ttnn.MathFidelity.HiFi2, fp32_acc=True, packer_l1_acc=True):
    # WormholeComputeKernelConfig is the compute config type for Blackhole too.
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=packer_l1_acc,
    )


def mcast_1d_config(grid, m, n, k, in0_block_w=4, out_subblock_w=4):
    """1D in0-mcast matmul config over ``grid`` (x, y) cores; N split across cores. Used by sparse_matmul."""
    core_x, core_y = grid
    num_cores = core_x * core_y
    nt = math.ceil(n / TILE)
    kt = math.ceil(k / TILE)
    per_core_n = math.ceil(nt / num_cores)
    while kt % in0_block_w:
        in0_block_w -= 1
    out_subblock_w = min(out_subblock_w, per_core_n)
    while per_core_n % out_subblock_w:
        out_subblock_w -= 1
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        out_block_h=1,
        out_block_w=out_subblock_w,
        per_core_M=max(TILE, m) // TILE,
        per_core_N=per_core_n,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


def decode_linear_config(device, k, n, block_tiles=64):
    """1D in0-mcast config for a decode matmul [1, 1, 32, k] x [k, n] with a DRAM-interleaved weight.

    N is split over an exactly filled rectangle of the compute grid, fewest tiles per core first. K blocks are
    wide (about ``block_tiles`` weight tiles per core per block): the weight reader waits on one read barrier
    per block, so with narrow K blocks these M = 32 matmuls pay a DRAM round trip every few K tiles."""
    assert k % TILE == 0 and n % TILE == 0, (k, n)
    kt, nt = k // TILE, n // TILE
    grid = device.compute_with_storage_grid_size()
    for per_core_n in range(1, nt + 1):
        if nt % per_core_n:
            continue
        cores = nt // per_core_n
        rows = [y for y in range(grid.y, 0, -1) if cores % y == 0 and cores // y <= grid.x]
        if rows:
            core_x, core_y = cores // rows[0], rows[0]
            break
    in0_block_w = max(1, min(kt, block_tiles // per_core_n))
    while kt % in0_block_w:
        in0_block_w -= 1
    out_subblock_w = min(per_core_n, 4)
    while per_core_n % out_subblock_w:
        out_subblock_w -= 1
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        out_block_h=1,
        out_block_w=per_core_n,
        per_core_M=1,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


def pad_to(n, multiple):
    return ((n + multiple - 1) // multiple) * multiple


class DramShardedMatmul:
    """Decode matmul [1, 1, 32, K] x [K, N] with the weight width-sharded over the DRAM banks.

    One worker core per DRAM bank reads only its own bank, which for these M = 32 decode matmuls is
    2-5x faster than a DRAM-interleaved weight (BH, measured: router 50 -> 11 us, SWA qkv 72 -> 34 us).
    N is padded per chip to a multiple of 32 x banks; ``__call__`` returns the padded width and callers
    slice when it matters. Prefill keeps using ``ttnn.linear`` on ``interleaved(weight)``.
    """

    def __init__(self, plan, k, n, out_dtype=ttnn.bfloat16, compute_kernel_config=None):
        device = plan.mesh_device
        self.banks = banks = device.dram_grid_size().x
        self.k, self.n = k, n
        self.n_pad = math.ceil(n / (TILE * banks)) * TILE * banks
        assert k % (TILE * banks) == 0, (k, banks)
        dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
        self.weight_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(dram_grid, (k, self.n_pad // banks), ttnn.ShardOrientation.ROW_MAJOR),
        )
        cores = ttnn.num_cores_to_corerangeset(banks, device.compute_with_storage_grid_size(), row_wise=True)

        def l1_width(width):
            return ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(cores, (TILE, width), ttnn.ShardOrientation.ROW_MAJOR),
            )

        self.in_memory_config = l1_width(k // banks)
        self.out_memory_config = l1_width(self.n_pad // banks)
        self.program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=k // TILE // banks,
            per_core_M=1,
            per_core_N=self.n_pad // TILE // banks,
            fused_activation=None,
        )
        self.out_dtype = out_dtype
        self.compute_kernel_config = compute_kernel_config or compute_config(ttnn.MathFidelity.HiFi2)

    def pad_chip_blocks(self, w, tp):
        """Host [K, tp * n] (chip blocks side by side) -> [K, tp * n_pad], zero columns at the end of each block."""
        blocks = w.reshape(w.shape[0], tp, self.n)
        return torch.nn.functional.pad(blocks, (0, self.n_pad - self.n)).reshape(w.shape[0], tp * self.n_pad)

    def matmul_sharded(self, x_sharded, weight, memory_config=None):
        """x already in ``in_memory_config`` -> [1, 1, 32, n_pad] (``memory_config``, default DRAM interleaved).

        The caller keeps ownership of ``x_sharded``."""
        out = ttnn.matmul(
            x_sharded,
            weight,
            program_config=self.program_config,
            memory_config=self.out_memory_config,
            dtype=self.out_dtype,
            compute_kernel_config=self.compute_kernel_config,
        )
        result = ttnn.to_memory_config(out, memory_config or ttnn.DRAM_MEMORY_CONFIG)
        out.deallocate(True)
        return result

    def __call__(self, x, weight, memory_config=None):
        """x [1, 1, 32, K] (interleaved) -> [1, 1, 32, n_pad] (``memory_config``, default DRAM interleaved)."""
        x_sharded = ttnn.to_memory_config(x, self.in_memory_config)
        result = self.matmul_sharded(x_sharded, weight, memory_config)
        x_sharded.deallocate(True)
        return result
