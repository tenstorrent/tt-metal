# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""QKV projection, head split and RoPE in one program.

Replaces minimal_matmul (QKV) followed by fused_heads_rope. Separately, the matmul writes the
fused [B * S, 3072] QKV activation and the head op reads it back, rotates Q and K and writes
the three head tensors: one extra DRAM round trip over the whole activation and a second
launch. Here the matmul's output stage rotates Q and K tiles while they are still in L1
and its writer puts every tile straight at its place in the head layout.

The weight keeps HF's row order (no Meta pair interleave), so rotate_half pairs dim j with
j + 32: with head_dim 64 a head is two tiles (lo, hi) and
    lo' = lo * cos - hi * sin,   hi' = hi * cos + lo * sin
with one cos, sin and -sin tile (the 32 frequencies) per 32-token sequence tile; each output tile is
two multiplies, the second accumulating into DST.
"""

import os

import torch

import ttnn
from models.demos.blackhole.llama_nemotron_rerank_1b.tt.custom_ops import _mm_grid as mg
from models.demos.blackhole.llama_nemotron_rerank_1b.tt.custom_ops._common import TILE_BYTES

_HERE = os.path.dirname(os.path.abspath(__file__))
WRITER_KERNEL = os.path.join(_HERE, "kernels", "writer_qkv_rope.cpp")
COMPUTE_KERNEL = os.path.join(_HERE, "kernels", "compute_qkv_rope.cpp")
TILE = 32


def make_rope_table(cos: torch.Tensor, sin: torch.Tensor, device):
    """cos / sin ``[S, 32]`` (one column per frequency) -> ``[1, 1, S, 96]`` bf16 TILE: per
    sequence tile s, pages 3s, 3s + 1, 3s + 2 are the cos, sin and -sin tiles."""
    cs = torch.cat([cos, sin, -sin], dim=-1).to(torch.float32).reshape(1, 1, cos.shape[0], 3 * TILE)
    return ttnn.from_torch(cs, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


def matmul_qkv_rope(
    x: ttnn.Tensor,
    w: ttnn.Tensor,
    rope_table: ttnn.Tensor,
    *,
    batch: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int = 64,
    block=(8, 8, 4, 1, 4),
    grid=None,
    fidelity=ttnn.MathFidelity.HiFi2,
    out_dtype=ttnn.bfloat16,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    rope_fidelity=None,
    defines=(),
    fp32_acc=True,
    in0_mcast=False,
    in0_depth=2,
    in1_depth=2,
):
    """x ``[1, 1, B * S, K]``, w ``[K, (n_heads + 2 * n_kv_heads) * head_dim]`` (Q, K, V column blocks)
    -> q ``[B, n_heads, S, head_dim]``, k and v ``[B, n_kv_heads, S, head_dim]``; RoPE applied to q, k.

    ``block`` = (M_block, K_block, N_block, subblock_h, subblock_w) in tiles; N_block must be even
    and so must each core's N range (whole heads)."""
    device = x.device()
    if head_dim != 2 * TILE:
        raise ValueError("the rotation pairs the two tiles of a 64-wide head")
    shape = list(x.padded_shape)
    M = 1
    for d in shape[:-1]:
        M *= int(d)
    M //= TILE
    K = shape[-1] // TILE
    N = int(w.padded_shape[-1]) // TILE
    S = M * TILE // batch
    St = S // TILE
    head_tiles = head_dim // TILE
    if N != (n_heads + 2 * n_kv_heads) * head_tiles or St * batch != M:
        raise ValueError(f"shapes do not match: M {M} tiles, N {N} tiles, batch {batch}")
    gxy = grid or device.compute_with_storage_grid_size()
    gx, gy = int(gxy.x), int(gxy.y)
    mb, kb, nb, sh, sw = block
    g = mg.Grid(gx, gy, M, K, N, mb, kb, nb)
    if nb % 2 or g.N_per_core % 2:
        raise ValueError(f"N block {nb} and per-core N {g.N_per_core} must be even (whole heads)")
    if mb % sh or nb % sw or sh * sw > (4 if fp32_acc else 8):
        raise ValueError(f"bad subblock {sh}x{sw} for block {mb}x{nb}")

    q = ttnn.allocate_tensor_on_device(
        ttnn.Shape([batch, n_heads, S, head_dim]), out_dtype, ttnn.TILE_LAYOUT, device, memory_config
    )
    k = ttnn.allocate_tensor_on_device(
        ttnn.Shape([batch, n_kv_heads, S, head_dim]), out_dtype, ttnn.TILE_LAYOUT, device, memory_config
    )
    v = ttnn.allocate_tensor_on_device(
        ttnn.Shape([batch, n_kv_heads, S, head_dim]), out_dtype, ttnn.TILE_LAYOUT, device, memory_config
    )

    all_cores = g.all_cores()
    cbs = [
        mg.cb(0, mb * kb * in0_depth, x.dtype, all_cores),
        mg.cb(1, kb * nb * in1_depth, w.dtype, all_cores),
        mg.cb(2, mb * nb * 2, out_dtype, all_cores),
        mg.cb(3, mb * nb, ttnn.float32, all_cores),
        mg.cb(4, 6 * mb, ttnn.bfloat16, all_cores),
    ]
    semaphores = mg.matmul_semaphores(all_cores)
    kernels = mg.in0_mcast_kernels(device, g, x) if in0_mcast else mg.in0_kernels(device, g, x, q)

    prefix = [g.M, g.padded_M, g.K, g.K, g.N, gy * g.N_per_core, mb, kb, nb, g.M_blocks_per_core]
    prefix += [g.N_blocks_per_core, TILE_BYTES[w.dtype], TILE_BYTES[out_dtype], 3, 4, 5]
    tail = [St, head_tiles, n_heads, n_kv_heads] + mg.accessor_args(w, rope_table, q, k, v)
    send_ct = prefix + [1] + tail
    recv_ct = prefix + [0] + tail
    rope_tiles = (n_heads + n_kv_heads) * head_tiles
    compute_ct = [g.K_blocks, mb, kb, nb, g.M_blocks_per_core, g.N_blocks_per_core, sh, sw, rope_tiles]

    addrs = [
        w.buffer_address(),
        rope_table.buffer_address(),
        q.buffer_address(),
        k.buffer_address(),
        v.buffer_address(),
    ]
    send_rt, recv_rt, compute_rt = [], [], []
    for xx in range(gx):
        for yy in range(gy):
            m0, m1 = g.m_range(xx, yy)
            n0, n1 = g.n_range(xx, yy)
            rt = addrs + mg.in1_chain_args(device, g, xx, yy) + [m0, m1, n0, n1, g.defer_write_k_block(xx, yy)]
            (send_rt if xx == 0 else recv_rt).append(((xx, yy), rt))
            compute_rt.append(((xx, yy), [m0, m1, n0, n1]))

    def dm(ct, rt, core_set):
        return ttnn.KernelDescriptor(
            kernel_source=WRITER_KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_set,
            compile_time_args=ct,
            runtime_args=rt,
            defines=[],
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=mg.IN1_NOC),
            compiler_include_paths=[mg.KERNEL_ROOT],
        )

    kernels.append(dm(send_ct, send_rt, g.in1_sender_cores()))
    if gx > 1:
        kernels.append(dm(recv_ct, recv_rt, g.in1_receiver_cores()))
    kernels.append(
        ttnn.KernelDescriptor(
            kernel_source=COMPUTE_KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_cores,
            compile_time_args=compute_ct,
            runtime_args=compute_rt,
            defines=list(defines) + ([("ROPE_FIDELITY", rope_fidelity)] if rope_fidelity else []),
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=fidelity,
                math_approx_mode=False,
                fp32_dest_acc_en=fp32_acc,
                dst_full_sync_en=False,
            ),
        )
    )
    program = ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=semaphores)
    ttnn.generic_op([x, w, rope_table, q, k, v], program)
    return q, k, v
