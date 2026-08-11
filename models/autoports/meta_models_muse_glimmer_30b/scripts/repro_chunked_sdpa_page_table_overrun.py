# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Standalone reproducer: ``chunked_scaled_dot_product_attention`` overruns the page table.

The op validates only ``kv_length >= q_len + chunk_start_idx``
(``sdpa_device_operation.cpp``), but its program factory rounds the K extent up to the
K chunk size::

    padded_Sk    = ceil(Sk / k_chunk_size) * k_chunk_size     # sdpa_program_factory.cpp
    k_num_chunks = padded_Sk / k_chunk_size

and the reader consumes one page-table block id per ``block_size`` tokens of ``padded_Sk``.
When ``padded_Sk`` exceeds ``page_table.shape[1] * block_size`` the reader walks past the
row's last valid block id, into the page-table stick's 32-byte alignment padding, and uses
whatever integers are there as physical block ids.

That padding is zero when the page table came from ``ttnn.from_torch`` (which writes the
whole aligned page), which is why the overrun is invisible in most tests. When the page
table is a device tensor produced by ``ttnn.slice`` + ``ttnn.concat`` — the gather any
non-identity KV-slot order needs — the padding holds stale DRAM, so the block ids are
arbitrary and the op reads K/V from outside the cache buffer.

Geometry below is the one that hit the Muse-Glimmer decoder: prefix 8192, 64-token tail
chunk, ``block_size=128``, 65-block page table (8320 tokens), ``k_chunk_size=256``
(``padded_Sk`` = 8448 > 8320).

Observed symptoms of the overrun, both from reading DRAM outside the cache buffer:

* garbage output — PCC -0.0018 against the golden, ``max|out|`` 2.4e37 (measured while
  diagnosing, with one set of padding contents);
* a **device hang** — the run below wedged for >900 s with no output and had to be killed
  (measured with a different set of padding contents; the device recovered on process exit,
  `tt-smi -ls --local` and a mesh smoke were clean afterwards).

Both overrunning calls can hang, so **neither** runs by default — including the
``from_torch`` one. A ``from_torch`` page table only *usually* hides the overrun (it writes
the whole aligned page, so the padding reads as block id 0, which lands back inside the
cache buffer); that is luck about the padding contents, not safety. Only the in-range call
is run by default, next to a static table of which geometries overrun.

Run::

    python models/autoports/meta_models_muse_glimmer_30b/scripts/repro_chunked_sdpa_page_table_overrun.py
    # opt in to the out-of-bounds call, which may hang the device and need `tt-smi -r`:
    python .../repro_chunked_sdpa_page_table_overrun.py --include-hang-case
"""

from __future__ import annotations

import sys

import torch

import ttnn

BLOCK, KV_HEADS, HEAD_DIM, HEADS = 128, 2, 128, 32
PREFIX, Q_LEN = 8192, 64
SEQ = PREFIX + Q_LEN  # 8256
POOL_ROWS, BLOCKS_PER_SEQ = 4, 65
NUM_BLOCKS = BLOCKS_PER_SEQ * POOL_ROWS + 5
ROWS = [3, 1]  # non-identity slot order -> needs a gather
CAPACITY = BLOCKS_PER_SEQ * BLOCK


def main() -> None:
    torch.manual_seed(0)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    mapper = ttnn.ReplicateTensorToMesh(mesh)

    def to_dev(tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(
            tensor, dtype=dtype, layout=layout, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper
        )

    generator = torch.Generator().manual_seed(1717)
    page_table = torch.randperm(NUM_BLOCKS, generator=generator)[: BLOCKS_PER_SEQ * POOL_ROWS]
    page_table = page_table.reshape(POOL_ROWS, BLOCKS_PER_SEQ).to(torch.int32)

    # Paged K/V: zeros everywhere, only the two used slots written, only over [0, SEQ).
    k_pool = torch.zeros(NUM_BLOCKS, KV_HEADS, BLOCK, HEAD_DIM)
    v_pool = torch.zeros_like(k_pool)
    k_seq, v_seq = {}, {}
    for row in ROWS:
        k_seq[row] = torch.randn(KV_HEADS, SEQ, HEAD_DIM) * 0.7
        v_seq[row] = torch.randn(KV_HEADS, SEQ, HEAD_DIM) * 0.7
        for logical in range((SEQ + BLOCK - 1) // BLOCK):
            lo, hi = logical * BLOCK, min((logical + 1) * BLOCK, SEQ)
            physical = int(page_table[row, logical])
            k_pool[physical, :, : hi - lo] = k_seq[row][:, lo:hi]
            v_pool[physical, :, : hi - lo] = v_seq[row][:, lo:hi]
    k_cache, v_cache = to_dev(k_pool), to_dev(v_pool)

    q_torch = torch.randn(len(ROWS), HEADS, Q_LEN, HEAD_DIM) * 6.0
    q = to_dev(q_torch)
    pool_tt = to_dev(page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)

    kernel_config = ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    grid = mesh.compute_with_storage_grid_size()

    def golden():
        outs = []
        for index, row in enumerate(ROWS):
            k = k_seq[row].repeat_interleave(HEADS // KV_HEADS, 0)
            v = v_seq[row].repeat_interleave(HEADS // KV_HEADS, 0)
            scores = (q_torch[index].float() @ k.transpose(-1, -2)) / (HEAD_DIM**0.5)
            mask = torch.ones(Q_LEN, SEQ, dtype=torch.bool).tril(diagonal=PREFIX)
            outs.append(scores.masked_fill(~mask, float("-inf")).softmax(-1) @ v)
        return torch.stack(outs)

    reference = golden()

    def pcc(a, b):
        a, b = a.reshape(-1).double(), b.reshape(-1).double()
        return float(torch.corrcoef(torch.stack([a, b]))[0, 1])

    def run(k_chunk, gathered):
        if gathered:  # how a non-identity slot order builds its page table
            slices = [ttnn.slice(pool_tt, [r, 0], [r + 1, BLOCKS_PER_SEQ]) for r in ROWS]
            page_table_tt = ttnn.concat(slices, dim=0)
        else:
            page_table_tt = to_dev(page_table[ROWS].contiguous(), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        assert torch.equal(ttnn.to_torch(page_table_tt).to(torch.int32), page_table[ROWS].contiguous())
        out = ttnn.transformer.chunked_scaled_dot_product_attention(
            q,
            k_cache,
            v_cache,
            page_table_tt,
            PREFIX,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=grid, q_chunk_size=64, k_chunk_size=k_chunk, exp_approx_mode=False
            ),
            compute_kernel_config=kernel_config,
        )
        result = ttnn.to_torch(out).float()
        padded_sk = -(-SEQ // k_chunk) * k_chunk
        print(
            f"k_chunk={k_chunk:3d} padded_Sk={padded_sk} capacity={CAPACITY} "
            f"{'OVERRUN' if padded_sk > CAPACITY else 'in-range'}  "
            f"page_table={'slice+concat' if gathered else 'from_torch '}  "
            f"-> pcc {pcc(reference, torch.nan_to_num(result)):9.6f} "
            f"max|out| {float(torch.nan_to_num(result).abs().max()):.4g}"
        )
        out.deallocate(True)

    print(f"geometry: SEQ={SEQ}, page table {BLOCKS_PER_SEQ} blocks x {BLOCK} = {CAPACITY} tokens")
    for k_chunk in (32, 64, 128, 256, 512):
        padded_sk = -(-SEQ // k_chunk) * k_chunk
        print(
            f"  k_chunk={k_chunk:3d} -> padded_Sk={padded_sk:5d} "
            f"{'OVERRUN (reads past the page table)' if padded_sk > CAPACITY else 'in range'}"
        )

    run(128, True)  # padded_Sk 8320 == capacity -> correct; this is what the layer now picks
    if "--include-hang-case" in sys.argv:
        # Both of these read past the page table. Either can return garbage (measured: PCC
        # -0.0018, max|out| 2.4e37) or hang the device (measured: >900 s with no output,
        # recovered on process exit). Opt-in only.
        run(256, True)
        run(256, False)
    else:
        print("skipping both out-of-bounds calls (they may hang the device); " "pass --include-hang-case to run them")

    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
