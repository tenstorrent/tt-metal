# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Vocab-sharded embedding table of the fused decode path at TP > 1 (fused_decode/config.py embed_vshard_on).

The BF16 table (gpt-oss: 201088 x 2880, 1.16 GB) is replicated on every device otherwise. Here device d holds rows
[d * Vd, (d + 1) * Vd) (Vd = ceil(vocab / tp)) plus one zero row (local index Vd): 0.29 GB per device.

- Decode (fused_decode/inputs.py, EMB_VSHARD): each device writes its row of the token (or the zero row) as the flat
  partial of an all-reduce, which the first layer's input boundary performs (site "entry") with a zero residual.
- Prefill (vshard_embedding): VocabLocalize maps the token ids to local ids (owned: id - d * Vd, else Vd), ttnn.embedding
  gathers the local rows (zero rows for the other devices' tokens), and an all-reduce over the TP devices sums them.
Both are exact: every element is one nonzero BF16 term plus zeros.
"""

import os
from pathlib import Path

import torch

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"


def vshard_rows(vocab, tp):
    """Rows per device (Vd); local row Vd is the zero row."""
    return -(-vocab // tp)


def vshard_table(weight, tp):
    """[vocab, hidden] -> [1, 1, tp * (Vd + 1), hidden]: device d's rows [d Vd, (d + 1) Vd) then a zero row (shard on
    dim 2 over the TP devices)."""
    V, H = weight.shape
    vd = vshard_rows(V, tp)
    parts = []
    for d in range(tp):
        p = torch.zeros(vd + 1, H, dtype=weight.dtype)
        lo, hi = d * vd, min((d + 1) * vd, V)
        p[: hi - lo] = weight[lo:hi]
        parts.append(p)
    return torch.cat(parts, 0).reshape(1, 1, tp * (vd + 1), H)


def vshard_table_from_cache(replicated_name, vshard_name, vocab, tp):
    """The vocab-sharded host table from the replicated table's cache file when only that one is cached (a warm cache
    built before the sharded table existed), else None (ttnn.as_tensor then loads or the caller builds it)."""
    suffix = "_dtype_BFLOAT16_layout_ROW_MAJOR.tensorbin"
    if replicated_name is None or os.path.isfile(f"{vshard_name}{suffix}"):
        return None
    path = f"{replicated_name}{suffix}"
    if not os.path.isfile(path):
        return None
    host = ttnn.load_tensor(path)
    try:
        w = ttnn.to_torch(host)
    except Exception:  # noqa: BLE001  (a multi-device host tensor: every shard is the same table)
        w = ttnn.to_torch(ttnn.get_device_tensors(host)[0])
    return vshard_table(w.reshape(vocab, -1), tp)


def vshard_offsets(mesh_device, vocab, tp):
    """Per device a [1, 1, 1, 32] UINT32 row-major tensor whose entry 0 is the device's first row id d * Vd."""
    vd = vshard_rows(vocab, tp)
    t = torch.zeros(1, 1, 1, 32 * tp, dtype=torch.int64)
    for d in range(tp):
        t[0, 0, 0, 32 * d] = d * vd
    return ttnn.from_torch(
        t,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=(None, 3)),
    )


class VocabLocalize:
    """[1, 1, 1, S] UINT32 row-major token ids -> local ids of this device's vocab shard (kernels/vocab_localize.cpp, one
    core): id - d * Vd when the id is in the shard, else Vd (the zero row)."""

    CHUNK = 1024  # ids per read

    def __init__(self, mesh_device, offsets, vd):
        self.mesh_device, self.offsets, self.vd = mesh_device, offsets, vd
        self.core = ttnn.CoreCoord(0, 0)
        self.cores = ttnn.CoreRangeSet([ttnn.CoreRange(self.core, self.core)])

    def __call__(self, tokens):
        S = tokens.shape[-1]
        assert tokens.dtype == ttnn.uint32 and tokens.layout == ttnn.ROW_MAJOR_LAYOUT and S % 16 == 0, tokens
        out = ttnn.empty(
            list(tokens.shape),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=tokens.memory_config(),
        )
        ct = [0, S, self.CHUNK, self.vd]
        for t in (tokens, self.offsets, out):
            ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        rt = ttnn.RuntimeArgs()
        rt[self.core.x][self.core.y] = [tokens.buffer_address(), self.offsets.buffer_address(), out.buffer_address()]
        kernel = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "vocab_localize.cpp"),
            core_ranges=self.cores,
            compile_time_args=ct,
            runtime_args=rt,
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0),
        )
        scratch = 128 + 4 * self.CHUNK
        cb = ttnn.CBDescriptor(
            total_size=scratch,
            core_ranges=self.cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=scratch)],
        )
        ttnn.generic_op([tokens, self.offsets, out], ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[cb]))
        return out


def vshard_embedding(localize, tokens, table, tp_axis, num_links, dtype=ttnn.bfloat8_b):
    """Prefill embedding from the vocab-sharded table: local ids, the local rows (zeros for foreign ids), all-reduce
    over the TP devices; [1, 1, S, hidden] in `dtype`."""
    local = localize(tokens)
    rows = ttnn.embedding(local, table, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    local.deallocate(True)
    if len(rows.shape) == 3:
        rows = ttnn.unsqueeze_to_4D(rows)
    out = ttnn.all_reduce(rows, num_links=num_links, topology=ttnn.Topology.Ring, cluster_axis=tp_axis)
    rows.deallocate(True)
    if dtype != ttnn.bfloat16:
        typed = ttnn.typecast(out, dtype)
        out.deallocate(True)
        out = typed
    return out
