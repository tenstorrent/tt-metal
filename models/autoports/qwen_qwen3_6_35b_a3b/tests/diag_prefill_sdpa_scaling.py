# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""How prefill SDPA cost grows with `chunk_start_idx` (README section 5).

Section 5 splits the ~12.8 s that a 262143-token `full` prefill costs above the position-independent
extrapolation into device attention and per-chunk program creation. That split needs a model of how
chunk *i* scales, and every profiled prefill in the stage is a *single* chunk at `abs_pos = 0`, so the
model was never measured -- review round 9 pointed out that the assumed `(i+1)x` ignores causality.

Two candidate models, for `q_chunk_size = k_chunk_size = sdpa_chunk = 128` and
`prefill_chunk_size = 2048` (16 q-chunks per call):

* **rectangular** `Sk = chunk_start_idx + Sq`, so chunk *i* does `(i+1)x` chunk 0's work;
* **causal**, which is what the op actually runs (`sdpa.cpp` passes `is_causal=true` for the chunked
  entry point, and the K loop is bounded by the diagonal in
  `kernels/dataflow/reader_interleaved.cpp` and `kernels/compute/compute_common.hpp`): q-chunk `j` of
  call `i` walks `16i + j + 1` k-chunks, so the call does `sum_j (16i + j + 1) = 256i + 136`
  against chunk 0's 136, i.e. `(1.88i + 1)x`.

They differ by 1.87x in the total over 128 chunks, which is the whole disagreement about whether the
residual belongs to attention or to program creation. This measures the op directly -- no model code,
no profiler -- at the real prefill shapes.

    python models/autoports/qwen_qwen3_6_35b_a3b/tests/diag_prefill_sdpa_scaling.py
"""

import time

import torch

import ttnn

NH, NKV, HD, BLOCK = 16, 2, 256, 64
SQ = 2048  # prefill_chunk_size
CHUNK = 128  # DecoderConfig.sdpa_chunk == PREFILL_ALIGN
#: chunk index i, i.e. chunk_start_idx = i * SQ. Kept small so the k/v allocation stays modest.
INDICES = [0, 1, 2, 3, 7, 15]
ITERS = 10


def k_chunk_iterations(i, sq=SQ, chunk=CHUNK):
    """Causal model: sum over this call's q-chunks of how many k-chunks each one walks."""
    per_call = sq // chunk
    return sum(per_call * i + j + 1 for j in range(per_call))


def main():
    torch.set_num_threads(16)
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    grid = device.compute_with_storage_grid_size()
    hifi = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    pcfg = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
        q_chunk_size=CHUNK,
        k_chunk_size=CHUNK,
        exp_approx_mode=False,
    )
    try:
        gen = torch.Generator().manual_seed(11)
        q = ttnn.from_torch(
            torch.randn(1, NH, SQ, HD, generator=gen).to(torch.bfloat16).float(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        base, base_iters = None, k_chunk_iterations(0)
        print(f"SCALE  q_chunk=k_chunk={CHUNK}  Sq={SQ}  ({SQ // CHUNK} q-chunks per call)")
        print(f"SCALE  {'i':>3} {'Sk':>8} {'ms':>9} {'measured':>9} {'causal':>8} {'rect':>8}")
        for i in INDICES:
            sk = i * SQ + SQ
            # Paged K/V and a page table, exactly as `_full_attention_prefill` calls it.
            blocks = sk // BLOCK
            kv = [
                ttnn.from_torch(
                    torch.randn(blocks, NKV, BLOCK, HD, generator=gen).to(torch.bfloat16).float(),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                for _ in range(2)
            ]
            page_table = ttnn.from_torch(
                torch.arange(blocks).reshape(1, blocks).int(),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            kv.append(page_table)
            call = lambda: ttnn.transformer.chunked_scaled_dot_product_attention(  # noqa: E731
                input_tensor_q=q,
                input_tensor_k=kv[0],
                input_tensor_v=kv[1],
                page_table_tensor=page_table,
                chunk_start_idx=i * SQ,
                scale=HD**-0.5,
                program_config=pcfg,
                compute_kernel_config=hifi,
            )
            try:
                ttnn.deallocate(call())
                ttnn.synchronize_device(device)
                start = time.perf_counter()
                for _ in range(ITERS):
                    ttnn.deallocate(call())
                ttnn.synchronize_device(device)
                ms = (time.perf_counter() - start) * 1000.0 / ITERS
                base = base or ms
                print(
                    f"SCALE  {i:>3} {sk:>8} {ms:>9.3f} {ms / base:>9.2f}x "
                    f"{k_chunk_iterations(i) / base_iters:>7.2f}x {i + 1:>7.2f}x"
                )
            except Exception as exc:  # noqa: BLE001 - diagnostic reports, never raises
                print(f"SCALE  {i:>3} {sk:>8}  ERR:{type(exc).__name__}: {exc}")
            for t in kv:
                ttnn.deallocate(t)
        total_causal = sum(k_chunk_iterations(i) for i in range(128)) / base_iters
        print(f"SCALE  sum over 128 chunks: causal {total_causal:.0f}x chunk 0, rectangular {128 * 129 / 2:.0f}x")
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
