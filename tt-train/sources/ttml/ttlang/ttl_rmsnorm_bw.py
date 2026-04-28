"""Column-split rmsnorm backward with gather+mcast for scale all-reduce.
"""
from __future__ import annotations

import torch
import ttnn
import ttl

TILE = 32
# BH p110
GRID_X_MAX = 11
GRID_Y_MAX = 10


def pick_config(ht: int, wt: int, grid_x: int = GRID_X_MAX, grid_y: int = GRID_Y_MAX):
    """Choose a 2D worker grid and per-core tile counts so the logical tensor fits when padded.

    ``ht`` / ``wt``: tensor height and width in **tiles** (``rows // TILE``, ``cols // TILE``).
    ``grid_x`` / ``grid_y``: upper bounds on how many cores to use along width / height (defaults ``GRID_X_MAX``, ``GRID_Y_MAX``).

    Returns ``(col_cores, row_cores, cpc, rpc, ht_padded, wt_padded)`` where ``col_cores`` / ``row_cores`` are
    the chosen mesh size, ``cpc`` / ``rpc`` are tile-columns / tile-rows per core, and the padded tile sizes are
    ``ht_padded = row_cores * rpc`` and ``wt_padded = col_cores * cpc`` (ceil division of ``ht``/``wt`` over the grid).
    """
    col_cores = min(grid_x, wt)
    cpc = -(-wt // col_cores)
    wt_padded = col_cores * cpc
    row_cores = min(grid_y, ht)
    rpc = -(-ht // row_cores)
    ht_padded = row_cores * rpc
    return col_cores, row_cores, cpc, rpc, ht_padded, wt_padded


def make_kernel_for_shape(rows: int, cols: int, block_count: int = 2, n_row_strip: int = 1):
    """Build a TTL backward kernel for a logical matrix shape and return it plus the tiling tuple.

    ``rows`` / ``cols``: element height and width of the (unpadded) problem; converted to tile counts internally.
    ``block_count``: minimum circular-buffer depth passed through to ``make_kernel`` (see ``bc`` there).
    ``n_row_strip``: how many row strips each core processes per full ``rpc`` tile-rows; must divide ``rpc``.

    Returns ``(kernel, (col_cores, row_cores, cpc, rpc, ht_padded, wt_padded))`` where the tuple matches
    ``pick_config`` and ``ht_padded`` / ``wt_padded`` are padded sizes in **tiles** (multiply by ``TILE`` for elements).
    """
    ht = rows // TILE
    wt = cols // TILE
    col_cores, row_cores, cpc, rpc, ht_padded, wt_padded = pick_config(ht, wt)
    k = make_kernel(col_cores, row_cores, cpc, rpc, n_row_strip, cols, block_count)
    return k, (col_cores, row_cores, cpc, rpc, ht_padded, wt_padded)


def make_kernel(
    col_cores: int, row_cores: int, cpc: int, rpc: int, n_row_strip: int, elem_c: int, bc: int = 2
):
    """ Return a column-split RMSNorm backward kernel.

    ``col_cores`` / ``row_cores``: Core counts in the horizontal / vertical dimensions of the grid.
    ``cpc``: tile-columns per core; logical width in tiles is ``wt == col_cores * cpc``.
    ``rpc``: tile-rows per core; logical height in tiles is ``ht == row_cores * rpc``.
    ``n_row_strip``: number of outer strip iterations per core along rows; must divide ``rpc``;
        each strip uses ``rb = rpc // n_row_strip`` tile-rows (``block_shape`` first dim).
    ``elem_c``: full row width in **elements** (not padded); ``inv_c = 1 / elem_c`` for the backward mean term.
    ``bc``: minimum dataflow buffer depth;``.
    """
    assert rpc % n_row_strip == 0
    # rb: how many tile-rows each core processes per outer strip.
    rb = rpc // n_row_strip

    @ttl.operation(grid=(col_cores, row_cores), fp32_dest_acc_en=True)
    def rmsnorm_bw_colsplit(
        inp, gamma, rms, dL, dL_da_out, dL_dg_out
    ):
        ht = inp.shape[0] // TILE
        wt = inp.shape[1] // TILE
        if wt != col_cores * cpc:
            raise ValueError(f"wt={wt} != col_cores*cpc ({col_cores}*{cpc})")
        if ht != row_cores * rpc:
            raise ValueError(f"ht={ht} != row_cores*rpc ({row_cores}*{rpc})")

        inv_c = 1.0 / elem_c

        # gather_net: each non-zero column sends its (rb,1) partial sum to column 0 on the same row y
        # so column 0 can reduce the full-row scale.
        gather_net = ttl.PipeNet([
            ttl.Pipe((x, y), (0, y))
            for x in range(1, col_cores)
            for y in range(row_cores)
        ])
        # bcast_net: column 0 sends the reduced scale back to columns 1..col_cores-1 on the same row y.
        bcast_net = ttl.PipeNet([
            ttl.Pipe((0, y), (slice(1, col_cores), y))
            for y in range(row_cores)
        ])

        block_shape = (rb, cpc)
        row_shape = (rb, 1)
        scalar_shape = (1, 1)

        # Per-core tile blocks (rb x cpc) staged from device tensors for compute.
        inp_cb = ttl.make_dataflow_buffer_like(inp, shape=block_shape, block_count=bc)
        gam_cb = ttl.make_dataflow_buffer_like(gamma, shape=block_shape, block_count=bc)
        rms_cb = ttl.make_dataflow_buffer_like(rms, shape=block_shape, block_count=bc)
        dL_cb = ttl.make_dataflow_buffer_like(dL, shape=block_shape, block_count=bc)

        # 1/rms for the current tile block (matches dtype/layout of upstream dL staging).
        rms_recip_cb = ttl.make_dataflow_buffer_like(dL, shape=block_shape, block_count=bc)
        # gained = recip * gamma * dL (numerator of dL/dx before the mean-correction term).
        gained_cb = ttl.make_dataflow_buffer_like(inp, shape=block_shape, block_count=bc)

        # Constant 1/C tile block for the backward mean over logical width (not pad width).
        inv_c_cb = ttl.make_dataflow_buffer_like(inp, shape=block_shape, block_count=bc)
        # Unity (1x1) reduce target passed into ttl.math.reduce_sum(..., dims=[1]).
        scaler_cb = ttl.make_dataflow_buffer_like(inp, shape=scalar_shape, block_count=bc)

        # Per-column partial sum over width: reduce_sum(iv * gained, dims=[1]) -> (rb, 1).
        partial_cb = ttl.make_dataflow_buffer_like(inp, shape=row_shape, block_count=bc)
        # Column 0: one (rb,1) slot per remote column for gather recv before accumulation.
        recv_cb = ttl.make_dataflow_buffer_like(inp, shape=row_shape, block_count=col_cores)
        # Column 0: full reduced scale after recv chain, then forwarded via bcast pipes.
        sum_cb = ttl.make_dataflow_buffer_like(inp, shape=row_shape, block_count=bc)
        # Columns > 0: receives the broadcast scale tile from column 0.
        bcast_cb = ttl.make_dataflow_buffer_like(inp, shape=row_shape, block_count=bc)

        # Compute outputs staged before DM writeback to dL_da_out / dL_dg_out slices.
        out_da_cb = ttl.make_dataflow_buffer_like(dL_da_out, shape=block_shape, block_count=bc)
        out_dg_cb = ttl.make_dataflow_buffer_like(dL_dg_out, shape=block_shape, block_count=bc)

        @ttl.compute()
        def compute():
            node_col, _ = ttl.node(dims=2)

            with scaler_cb.reserve() as sc:
                sc.store(ttl.math.fill(sc, 1.0))
            scaler = scaler_cb.wait()
            with inv_c_cb.reserve() as ic:
                ic.store(ttl.math.fill(ic, inv_c))
            icb = inv_c_cb.wait()

            for _ri in range(n_row_strip):
                iv = inp_cb.wait()
                rms = rms_cb.wait()
                dLw = dL_cb.wait()
                gamm = gam_cb.wait()
        
                with rms_recip_cb.reserve() as rms_recip:
                    rms_recip.store(ttl.math.recip(rms))
                recip = rms_recip_cb.wait()
                with gained_cb.reserve() as gc:
                    gc.store(recip * gamm * dLw)
                gained = gained_cb.wait()

                with partial_cb.reserve() as pc:
                    pc.store(ttl.math.reduce_sum(iv * gained, scaler, dims=[1]))
                if node_col == 0:
                    scale = partial_cb.wait()
                    for _ in range(col_cores - 1):
                        scale = scale + recv_cb.wait()

                    with sum_cb.reserve() as sc:
                        sc.store(scale)

                    out_da_cb.reserve().store(gained - ttl.math.broadcast(scale, iv, dims=[1]) * iv * recip * recip * icb)
                    out_dg_cb.reserve().store(iv * recip * dLw)
                else:
                    out_da_cb.reserve().store(gained - ttl.math.broadcast(bcast_cb.wait(), iv, dims=[1]) * iv * recip * recip * icb)
                    out_dg_cb.reserve().store(iv * recip * dLw)

        @ttl.datamovement()
        def dm_read():
            node_col, node_row = ttl.node(dims=2)
            col_start = node_col * cpc
            row_start = node_row * rpc

            for ri in range(n_row_strip):
                r0 = row_start + ri * rb
                i_blk = inp_cb.reserve()
                g_blk = gam_cb.reserve()
                r_blk = rms_cb.reserve()
                d_blk = dL_cb.reserve()
                i_tx = ttl.copy(inp[r0 : r0 + rb, col_start : col_start + cpc], i_blk)
                g_tx = ttl.copy(gamma[r0 : r0 + rb, col_start : col_start + cpc], g_blk)
                r_tx = ttl.copy(rms[r0 : r0 + rb, col_start : col_start + cpc], r_blk)
                d_tx = ttl.copy(dL[r0 : r0 + rb, col_start : col_start + cpc], d_blk)
                i_tx.wait()
                g_tx.wait()
                r_tx.wait()
                d_tx.wait()

                if node_col > 0:
                    blk = partial_cb.wait()
                    def send(pipe):
                        tx = ttl.copy(blk, pipe)
                        tx.wait()
                    gather_net.if_src(send)

                    def recv(pipe):
                        b = bcast_cb.reserve()
                        tx = ttl.copy(pipe, b)
                        tx.wait()
                    bcast_net.if_dst(recv)
                else:
                    def recv(pipe):
                        b = recv_cb.reserve()
                        tx = ttl.copy(pipe, b)
                        tx.wait()
                    gather_net.if_dst(recv)

                    blk = sum_cb.wait()
                    def send(pipe):
                        tx = ttl.copy(blk, pipe)
                        tx.wait()
                    bcast_net.if_src(send)

        @ttl.datamovement()
        def dm_write():
            node_col, node_row = ttl.node(dims=2)
            col_start = node_col * cpc
            row_start = node_row * rpc
            for ri in range(n_row_strip):
                r0 = row_start + ri * rb
                da_blk = out_da_cb.wait()
                da_tx = ttl.copy(da_blk, dL_da_out[r0 : r0 + rb, col_start : col_start + cpc])
                da_tx.wait()
                dg_blk = out_dg_cb.wait()
                dg_tx = ttl.copy(dg_blk, dL_dg_out[r0 : r0 + rb, col_start : col_start + cpc])
                dg_tx.wait()

    return rmsnorm_bw_colsplit


def _to_dev(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _pad(t, rows_padded, cols_padded):
    rows, cols = t.shape
    pad_r = rows_padded - rows
    pad_c = cols_padded - cols
    if pad_r == 0 and pad_c == 0:
        return t
    return torch.nn.functional.pad(t, (0, pad_c, 0, pad_r), value=0.0)


