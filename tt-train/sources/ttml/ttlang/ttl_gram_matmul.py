# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys

import torch
import ttnn
import ttl

TILE = 32
GRID_DIM = 10
kRtol = 1e-2
# For size like 320x320 15 is enough, the error increase with the size of the matrix.
kAtol = 200


@ttl.operation(grid=(GRID_DIM + 1, GRID_DIM), fp32_dest_acc_en=True)
def gram_matmul(x, out):
    m_tiles = x.shape[0] // TILE
    k_tiles = x.shape[1] // TILE
    if m_tiles % GRID_DIM != 0:
        raise ValueError(f"kernel expects M tiles divisible by GRID_DIM={GRID_DIM}, got {m_tiles}")
    n = m_tiles // GRID_DIM
    if k_tiles != GRID_DIM * n:
        raise ValueError(f"kernel expects K={GRID_DIM * n} tiles ({GRID_DIM * n * TILE} cols), got {k_tiles}")
    half_k = k_tiles // 2
    tile11 = (1, 1)

    row_upper_net = ttl.PipeNet([ttl.Pipe((0, y), (slice(y, GRID_DIM), y)) for y in range(GRID_DIM)])
    row_lower_net = ttl.PipeNet([ttl.Pipe((0, y), (slice(0, y), y)) for y in range(1, GRID_DIM)])

    col_upper_net = ttl.PipeNet([ttl.Pipe((x, 0), (x, slice(0, x + 1))) for x in range(GRID_DIM)])
    col_lower_net = ttl.PipeNet([ttl.Pipe((x, 0), (x, slice(x + 1, GRID_DIM))) for x in range(0, GRID_DIM - 1)])

    upper_to_lower_net = ttl.PipeNet(
        [ttl.Pipe((y, x), (x, y)) for x in range(GRID_DIM - 1) for y in range(x + 1, GRID_DIM)]
    )

    helper_row_even_net = ttl.PipeNet([ttl.Pipe((0, y), (GRID_DIM, y)) for y in range(GRID_DIM)])
    helper_col_even_net = ttl.PipeNet([ttl.Pipe((y, 0), (GRID_DIM, y)) for y in range(GRID_DIM)])
    helper_to_diag_net = ttl.PipeNet([ttl.Pipe((GRID_DIM, y), (y, y)) for y in range(GRID_DIM)])

    row_k_cb = ttl.make_dataflow_buffer_like(x, shape=tile11, block_count=1)
    col_k_cb = ttl.make_dataflow_buffer_like(x, shape=tile11, block_count=1)
    row_send_cb = ttl.make_dataflow_buffer_like(x, shape=tile11, block_count=2)
    col_send_cb = ttl.make_dataflow_buffer_like(x, shape=tile11, block_count=2)
    acc_cb = ttl.make_dataflow_buffer_like(out, shape=tile11, block_count=2)

    local_cb = ttl.make_dataflow_buffer_like(out, shape=tile11, block_count=2)
    transpose_send_cb = ttl.make_dataflow_buffer_like(out, shape=tile11, block_count=2)
    transpose_b_cb = ttl.make_dataflow_buffer_like(x, shape=tile11, block_count=2)
    peer_cb = ttl.make_dataflow_buffer_like(out, shape=tile11, block_count=1)
    diag_even_cb = ttl.make_dataflow_buffer_like(out, shape=tile11, block_count=1)
    out_cb = ttl.make_dataflow_buffer_like(out, shape=tile11, block_count=2)

    @ttl.compute()
    def compute():
        core_x, core_y = ttl.node(dims=2)
        for _row_local in range(n):
            for _col_local in range(n):
                with acc_cb.reserve() as z:
                    z.store(ttl.math.fill(z, 0.0))

                if core_x == GRID_DIM:
                    # Helper column computes diagonal-even partials only.
                    for _ in range(half_k):
                        with col_k_cb.wait() as b_blk, transpose_b_cb.reserve() as b_transposed:
                            b_transposed.store(ttl.math.transpose(b_blk))
                        with (
                            row_k_cb.wait() as a_blk,
                            transpose_b_cb.wait() as b_transposed,
                            acc_cb.wait() as acc,
                        ):
                            with acc_cb.reserve() as next_acc:
                                next_acc.store(acc + (a_blk @ b_transposed))
                    with acc_cb.wait() as acc, transpose_send_cb.reserve() as tsb:
                        tsb.store(ttl.math.transpose(acc))
                else:
                    for _ in range(half_k):
                        with col_k_cb.wait() as b_blk, transpose_b_cb.reserve() as b_transposed:
                            b_transposed.store(ttl.math.transpose(b_blk))
                        with (
                            row_k_cb.wait() as a_blk,
                            transpose_b_cb.wait() as b_transposed,
                            acc_cb.wait() as acc,
                        ):
                            with acc_cb.reserve() as next_acc:
                                next_acc.store(acc + (a_blk @ b_transposed))

                    if core_y > core_x:
                        with (
                            acc_cb.wait() as acc,
                            local_cb.reserve() as loc,
                        ):
                            loc.store(acc)
                        with local_cb.wait() as loc, peer_cb.wait() as peer, out_cb.reserve() as o:
                            o.store(loc + peer)
                    elif core_y < core_x:
                        with (
                            acc_cb.wait() as acc,
                            local_cb.reserve() as loc,
                            transpose_send_cb.reserve() as tsb,
                        ):
                            loc.store(acc)
                            tsb.store(ttl.math.transpose(acc))
                        with local_cb.wait() as loc, out_cb.reserve() as o:
                            o.store(loc)
                    else:
                        with acc_cb.wait() as acc, local_cb.reserve() as loc:
                            loc.store(acc)
                        with local_cb.wait() as loc, diag_even_cb.wait() as even_part, out_cb.reserve() as o:
                            o.store(loc + even_part)

    @ttl.datamovement()
    def dm_read():
        core_x, core_y = ttl.node(dims=2)
        for row_local in range(n):
            for col_local in range(n):
                for k_idx in range(k_tiles):
                    is_even = (k_idx % 2) == 0

                    if core_x == 0:
                        row_idx = core_y * n + row_local
                        with row_send_cb.reserve() as sb:
                            tx = ttl.copy(x[row_idx : row_idx + 1, k_idx : k_idx + 1], sb)
                            tx.wait()
                        src = row_send_cb.wait()

                        def send_row(pipe):
                            tx = ttl.copy(src, pipe)
                            tx.wait()

                        if not is_even:
                            row_upper_net.if_src(send_row)
                        else:
                            row_lower_net.if_src(send_row)
                            helper_row_even_net.if_src(send_row)

                    def recv_row(pipe):
                        with row_k_cb.reserve() as rb:
                            tx = ttl.copy(pipe, rb)
                            tx.wait()

                    if not is_even:
                        row_upper_net.if_dst(recv_row)
                    else:
                        row_lower_net.if_dst(recv_row)

                    if core_y == 0 and core_x < GRID_DIM:
                        col_idx = core_x * n + col_local
                        with col_send_cb.reserve() as sb:
                            tx = ttl.copy(x[col_idx : col_idx + 1, k_idx : k_idx + 1], sb)
                            tx.wait()
                        src = col_send_cb.wait()

                        def send_col(pipe):
                            tx = ttl.copy(src, pipe)
                            tx.wait()

                        if not is_even:
                            col_upper_net.if_src(send_col)
                        else:
                            col_lower_net.if_src(send_col)
                            helper_col_even_net.if_src(send_col)

                    def recv_col(pipe):
                        with col_k_cb.reserve() as cb:
                            tx = ttl.copy(pipe, cb)
                            tx.wait()

                    if not is_even:
                        col_upper_net.if_dst(recv_col)
                    else:
                        col_lower_net.if_dst(recv_col)
                        if core_x == GRID_DIM:
                            helper_row_even_net.if_dst(recv_row)
                            helper_col_even_net.if_dst(recv_col)

                if core_x == GRID_DIM:
                    with transpose_send_cb.wait() as src:

                        def send_diag_even(pipe):
                            tx = ttl.copy(src, pipe)
                            tx.wait()

                        helper_to_diag_net.if_src(send_diag_even)
                elif core_y > core_x:

                    def recv_from_upper(pipe):
                        with peer_cb.reserve() as pb:
                            tx = ttl.copy(pipe, pb)
                            tx.wait()

                    upper_to_lower_net.if_dst(recv_from_upper)
                elif core_y < core_x:
                    with transpose_send_cb.wait() as src:

                        def send_xpose(pipe):
                            tx = ttl.copy(src, pipe)
                            tx.wait()

                        upper_to_lower_net.if_src(send_xpose)
                elif core_x == core_y:

                    def recv_diag_even(pipe):
                        with diag_even_cb.reserve() as db:
                            tx = ttl.copy(pipe, db)
                            tx.wait()

                    helper_to_diag_net.if_dst(recv_diag_even)

    @ttl.datamovement()
    def dm_write():
        core_x, core_y = ttl.node(dims=2)
        if core_x != GRID_DIM:
            for row_local in range(n):
                for col_local in range(n):
                    row_idx = core_y * n + row_local
                    col_idx = core_x * n + col_local
                    with out_cb.wait() as blk:
                        tx = ttl.copy(blk, out[row_idx : row_idx + 1, col_idx : col_idx + 1])
                        tx.wait()


def run_case(device, n: int) -> None:
    if n <= 0:
        raise ValueError(f"N must be positive, got {n}")
    torch.manual_seed(0)

    m = GRID_DIM * TILE * n
    k = GRID_DIM * TILE * n
    x_torch = torch.empty(m, k, dtype=torch.float32)
    x_torch.uniform_(-1.0, 1.0)
    out_torch = torch.zeros(m, m, dtype=torch.float32)

    x_tt = ttnn.from_torch(
        x_torch,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out_tt = ttnn.from_torch(
        out_torch,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    gram_matmul(x_tt, out_tt)
    ttnn.synchronize_device(device)

    got = ttnn.to_torch(out_tt).to(torch.float32)
    ref = x_torch @ x_torch.T
    upper_mask = torch.triu(torch.ones_like(got, dtype=torch.bool), diagonal=1)

    got = got.masked_fill(upper_mask, 0.0)
    ref = ref.masked_fill(upper_mask, 0.0)

    torch.testing.assert_close(got, ref, rtol=kRtol, atol=kAtol)
    print("PASS: gram_matmul")


if __name__ == "__main__":
    default_size = ttnn.device.get_max_worker_l1_unreserved_size()
    device = ttnn.open_device(device_id=0, worker_l1_size=default_size - 90112)
    try:
        run_case(device, 1)  # 320x320  kAtol=8 is needed
        run_case(device, 10)  # 3200x3200  kAtol=80 is needed
        run_case(device, 26)  # 8320x8320  kAtol=190 is needed
    finally:
        ttnn.close_device(device)
