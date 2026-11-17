# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the weight-streaming expert (ws_expert) generic_op programs and benches."""

import os

import torch
import ttnn

KDIR = "models/demos/ws_expert/kernels"
TILE_BYTES = {
    ttnn.bfloat16: 2048,
    ttnn.bfloat8_b: 1088,
    ttnn.bfloat4_b: 576,
    ttnn.float32: 4096,
}
PEAK_CYC_PER_TILE_LOFI = 16  # 2 * 32^3 FLOP / 4096 FLOP a cycle


def kpath(name):
    return os.path.join(KDIR, name)


def crs(cores):
    """CoreRangeSet of single-core ranges for a list of (x, y)."""
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])


def cb(index, dtype, n_tiles, core_ranges, page=None):
    page = page or TILE_BYTES[dtype]
    return ttnn.CBDescriptor(
        total_size=n_tiles * page,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
    )


# Device-side zeros (traces: no host writes inside a captured layer). With DEVICE_ZERO set, zeroed_l1 allocates
# without a write and queues the uint32 tensors (the programs' state words: flags, counters, tables, stamps);
# zero_pending() then zeroes every queued tensor with one device program, before the program that uses them. The other
# formats hold data a program writes before it reads it (rings, staging), left unzeroed. Unset, zeroed_l1 writes the
# zeros from the host as before.
DEVICE_ZERO = False
_pending = []
_queue = [True]  # (self_zeroing(): the programs built inside clear their own state words; nothing queued)
L1_LOG = os.environ.get("WS_L1_LOG", "0") == "1"  # (log every zeroed_l1 / stamps allocation: address, bytes a core)


def l1_log(msg):
    if L1_LOG:
        print(f"WSL1 {msg}", flush=True)


def zeroed_l1(device, cores, shard_shape, dtype, layout, torch_dtype=torch.float32):
    """A HEIGHT_SHARDED L1 tensor of shard_shape a core on cores, zeroed (from the host, or queued for
    zero_pending())."""
    mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(crs(cores), list(shard_shape), ttnn.ShardOrientation.ROW_MAJOR),
    )
    shape = [len(cores) * shard_shape[0], shard_shape[1]]
    if DEVICE_ZERO:
        t = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dtype, layout, device, mem)
        if dtype == ttnn.uint32 and _queue[-1]:
            _pending.append((device, list(cores), t))
    else:
        t = ttnn.from_torch(
            torch.zeros(shape, dtype=torch_dtype), dtype=dtype, layout=layout, device=device, memory_config=mem
        )
    l1_log(
        f"alloc {t.buffer_address()} shard {list(shard_shape)} {dtype} cores {len(cores)} first {cores[0]} "
        f"device_zero {DEVICE_ZERO}"
    )
    return t


def drop_pending():
    """Forget the queued tensors without zeroing them (their zeros come from an earlier zero_pending)."""
    _pending.clear()
    return 0


class self_zeroing:
    """Context: the programs built inside clear their own state words at the start of each launch, so DEVICE_ZERO
    queues nothing for them."""

    def __enter__(self):
        _queue.append(False)

    def __exit__(self, *a):
        _queue.pop()


def zero_pending():
    """Zero every tensor zeroed_l1 queued since the last call: one program over the union of their cores (per core
    its tensors' shards, the same address on every chip). Returns the most bytes a core zeroes."""
    if not _pending:
        return 0
    device = _pending[0][0]
    ranges = {}
    for _, cores, t in _pending:
        spec = t.memory_config().shard_spec
        nbytes = spec.shape[0] * spec.shape[1] * 4
        for c in cores:
            ranges.setdefault(c, []).extend([t.buffer_address(), nbytes])
    # (the range count a compile-time arg, every core's pairs padded to it: one cached program a count, its runtime
    # args always the same length)
    n = max(len(r) // 2 for r in ranges.values())
    rt = ttnn.RuntimeArgs()
    for (x, y), r in ranges.items():
        rt[x][y] = r + [0] * (2 * n - len(r))
    k = ttnn.KernelDescriptor(
        kernel_source=kpath("wsz_zero.cpp"),
        core_ranges=crs(list(ranges)),
        compile_time_args=[n],
        runtime_args=rt,
        config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0),
    )
    R, Cn = tuple(device.shape)
    mpd = ttnn.MeshProgramDescriptor()
    mpd[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(R - 1, Cn - 1))] = ttnn.ProgramDescriptor(
        kernels=[k], semaphores=[], cbs=[]
    )
    io = [t for _, _, t in _pending]
    _pending.clear()
    ttnn.generic_op(io if len(io) >= 2 else io * 2, mpd)
    return max(sum(r[1::2]) for r in ranges.values())


def stamps_tensor(device, cores, words=64):
    """An L1 HEIGHT_SHARDED uint32 [len(cores) * 8, words] tensor: 8 x words stamp words per core (same address)."""
    return zeroed_l1(device, cores, [8, words], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32)


def read_stamps(st, n_cores, words=64):
    t = ttnn.to_torch(st).to(torch.int64) & 0xFFFFFFFF
    return t.reshape(n_cores, 8 * words)


def to_dev(t, dtype, device, mem=None):
    return ttnn.from_torch(
        t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem or ttnn.DRAM_MEMORY_CONFIG
    )


def quantize(t, dtype):
    """Round-trip t through a ttnn dtype on the host (what the device sees)."""
    if dtype == ttnn.bfloat16:
        return t.to(torch.bfloat16).float()
    return ttnn.to_torch(ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT)).float()


def pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def aiclk_mhz(dev_id=0):
    try:
        with open(f"/sys/class/tenstorrent/tenstorrent!{dev_id}/tt_aiclk") as f:
            return int(f.read().strip())
    except OSError:
        return -1
