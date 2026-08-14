# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Optional tt-lang (ttl) fused kernels for the DiT example.

Provides a fused adaLN modulate as a ttml autograd Function:

    forward:  y = x * (1 + scale) + shift
    backward: dx = dy * (1 + scale)
              dscale = sum_T(dy * x);  dshift = sum_T(dy)

where scale/shift live at tile-aligned offsets inside one packed modulation
tensor mod [B, 1, 1, n*D] (the canonical DiT "one linear, chunked" form —
the kernel reads the offsets natively, so no autograd split is needed).

Measured on Blackhole p150: 1.52x vs the pre-sliced ttnn mul+add pair and
2.56x vs the full ttnn chain, with gradients matching torch autograd
(see tt-lang spike logs). Kernels are compiled per (B, T, D, offsets) shape
on first use (~30-60 s JIT, then cached).

Requires the tt-lang "light" wheel (internal index; TTNN_DEP_MODE=external)
installed into the tt-train python env. Everything degrades gracefully:
`is_available()` gates usage and the model falls back to composed ttnn ops.
"""

from __future__ import annotations

TILE = 32

try:
    import ttl  # noqa: F401

    _TTL_AVAILABLE = True
except ImportError:
    _TTL_AVAILABLE = False


def is_available() -> bool:
    return _TTL_AVAILABLE


_FWD_CACHE: dict = {}
_BWD_CACHE: dict = {}


def _make_modulate_fwd(B, Tt, Dt, so_t, sh_t):
    import ttl
    import ttnn

    n_blocks = B * Tt

    @ttl.operation(grid="full")
    def modulate_fwd(x: ttnn.Tensor, mod: ttnn.Tensor, out: ttnn.Tensor) -> None:
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        cores = grid_cols * grid_rows
        bpc = -(-n_blocks // cores)

        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, Dt), block_count=2)
        s_dfb = ttl.make_dataflow_buffer_like(mod, shape=(1, Dt), block_count=2)
        h_dfb = ttl.make_dataflow_buffer_like(mod, shape=(1, Dt), block_count=2)
        o_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, Dt), block_count=2)

        @ttl.compute()
        def compute():
            col_c, row_c = ttl.node(dims=2)
            cid = col_c * grid_rows + row_c
            for i in range(bpc):
                blk = cid * bpc + i
                if blk < n_blocks:
                    with (
                        x_dfb.wait() as xb,
                        s_dfb.wait() as sb,
                        h_dfb.wait() as hb,
                        o_dfb.reserve() as ob,
                    ):
                        sbc = ttl.block.broadcast(sb, dims=[0], shape=xb.shape)
                        hbc = ttl.block.broadcast(hb, dims=[0], shape=xb.shape)
                        ob.store(ttl.add(ttl.add(ttl.mul(xb, sbc), xb), hbc))

        @ttl.datamovement()
        def read():
            col_c, row_c = ttl.node(dims=2)
            cid = col_c * grid_rows + row_c
            for i in range(bpc):
                blk = cid * bpc + i
                if blk < n_blocks:
                    b, rt = blk // Tt, blk % Tt
                    with x_dfb.reserve() as xb, s_dfb.reserve() as sb, h_dfb.reserve() as hb:
                        tx = ttl.copy(x[b, rt:rt + 1, 0:Dt], xb)
                        ts = ttl.copy(mod[b, 0:1, so_t:so_t + Dt], sb)
                        th = ttl.copy(mod[b, 0:1, sh_t:sh_t + Dt], hb)
                        tx.wait(); ts.wait(); th.wait()

        @ttl.datamovement()
        def write():
            col_c, row_c = ttl.node(dims=2)
            cid = col_c * grid_rows + row_c
            for i in range(bpc):
                blk = cid * bpc + i
                if blk < n_blocks:
                    b, rt = blk // Tt, blk % Tt
                    with o_dfb.wait() as ob:
                        ttl.copy(ob, out[b, rt:rt + 1, 0:Dt]).wait()

    return modulate_fwd


def _make_modulate_bwd(B, Tt, Dt, so_t, sh_t):
    import ttl
    import ttnn

    @ttl.operation(grid="full")
    def modulate_bwd(dy: ttnn.Tensor, x: ttnn.Tensor, mod: ttnn.Tensor,
                     dx: ttnn.Tensor, dmod: ttnn.Tensor) -> None:
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        cores = grid_cols * grid_rows
        bpc = -(-B // cores)

        dy_dfb = ttl.make_dataflow_buffer_like(dy, shape=(1, Dt), block_count=2)
        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, Dt), block_count=2)
        s_dfb = ttl.make_dataflow_buffer_like(mod, shape=(1, Dt), block_count=2)
        dx_dfb = ttl.make_dataflow_buffer_like(dx, shape=(1, Dt), block_count=2)
        accs_dfb = ttl.make_dataflow_buffer_like(dy, shape=(1, Dt), block_count=2)
        acch_dfb = ttl.make_dataflow_buffer_like(dy, shape=(1, Dt), block_count=2)
        ds_dfb = ttl.make_dataflow_buffer_like(dmod, shape=(1, Dt), block_count=2)
        dh_dfb = ttl.make_dataflow_buffer_like(dmod, shape=(1, Dt), block_count=2)

        @ttl.compute()
        def compute():
            col_c, row_c = ttl.node(dims=2)
            cid = col_c * grid_rows + row_c
            for i in range(bpc):
                b = cid * bpc + i
                if b < B:
                    with s_dfb.wait() as scale:
                        for rt in range(Tt):
                            with dy_dfb.wait() as dyb, x_dfb.wait() as xb:
                                with dx_dfb.reserve() as dxw:
                                    sbc = ttl.block.broadcast(scale, dims=[0], shape=dyb.shape)
                                    dxw.store(ttl.add(ttl.mul(dyb, sbc), dyb))
                                if Tt == 1:
                                    with ds_dfb.reserve() as dsw:
                                        dsw.store(ttl.math.reduce_sum(ttl.mul(dyb, xb), dims=[0]))
                                    with dh_dfb.reserve() as dhw:
                                        dhw.store(ttl.math.reduce_sum(dyb, dims=[0]))
                                elif rt == 0:
                                    with accs_dfb.reserve() as aw:
                                        aw.store(ttl.math.reduce_sum(ttl.mul(dyb, xb), dims=[0]))
                                    with acch_dfb.reserve() as bw:
                                        bw.store(ttl.math.reduce_sum(dyb, dims=[0]))
                                elif rt < Tt - 1:
                                    with accs_dfb.wait() as aold, accs_dfb.reserve() as anew:
                                        anew.store(ttl.add(aold, ttl.math.reduce_sum(ttl.mul(dyb, xb), dims=[0])))
                                    with acch_dfb.wait() as bold, acch_dfb.reserve() as bnew:
                                        bnew.store(ttl.add(bold, ttl.math.reduce_sum(dyb, dims=[0])))
                                else:
                                    with accs_dfb.wait() as aold, ds_dfb.reserve() as dsw:
                                        dsw.store(ttl.add(aold, ttl.math.reduce_sum(ttl.mul(dyb, xb), dims=[0])))
                                    with acch_dfb.wait() as bold, dh_dfb.reserve() as dhw:
                                        dhw.store(ttl.add(bold, ttl.math.reduce_sum(dyb, dims=[0])))

        @ttl.datamovement()
        def read():
            col_c, row_c = ttl.node(dims=2)
            cid = col_c * grid_rows + row_c
            for i in range(bpc):
                b = cid * bpc + i
                if b < B:
                    with s_dfb.reserve() as sb:
                        ttl.copy(mod[b, 0:1, so_t:so_t + Dt], sb).wait()
                    for rt in range(Tt):
                        with dy_dfb.reserve() as dyb, x_dfb.reserve() as xb:
                            t1 = ttl.copy(dy[b, rt:rt + 1, 0:Dt], dyb)
                            t2 = ttl.copy(x[b, rt:rt + 1, 0:Dt], xb)
                            t1.wait(); t2.wait()

        @ttl.datamovement()
        def write():
            col_c, row_c = ttl.node(dims=2)
            cid = col_c * grid_rows + row_c
            for i in range(bpc):
                b = cid * bpc + i
                if b < B:
                    for rt in range(Tt):
                        with dx_dfb.wait() as dxb:
                            ttl.copy(dxb, dx[b, rt:rt + 1, 0:Dt]).wait()
                    with ds_dfb.wait() as dsb:
                        ttl.copy(dsb, dmod[b, 0:1, so_t:so_t + Dt]).wait()
                    with dh_dfb.wait() as dhb:
                        ttl.copy(dhb, dmod[b, 0:1, sh_t:sh_t + Dt]).wait()

    return modulate_bwd


def _get_fwd(key):
    if key not in _FWD_CACHE:
        _FWD_CACHE[key] = _make_modulate_fwd(*key)
    return _FWD_CACHE[key]


def _get_bwd(key):
    if key not in _BWD_CACHE:
        _BWD_CACHE[key] = _make_modulate_bwd(*key)
    return _BWD_CACHE[key]


def make_fused_modulate():
    """Returns the FusedAdaLNModulate autograd Function class (requires ttl)."""
    import ttml
    import ttnn

    class FusedAdaLNModulate(ttml.autograd.Function):
        """y = x * (1 + mod[so:so+D]) + mod[sh:sh+D]; offsets in tiles.

        x: [B,1,T,D]; mod: [B,1,1,n*D]. Both bf16 TILE. dmod is written only
        at this call's offsets; multiple calls sharing one mod tensor
        accumulate their grads through autograd's add_grad.
        """

        @staticmethod
        def forward(ctx, x, mod, so_t, sh_t):
            vx, vm = x.get_value(), mod.get_value()
            Bc, _, Tc, Dc = (int(s) for s in vx.shape)
            ctx.key = (Bc, Tc // TILE, Dc // TILE, so_t, sh_t)
            ctx.mod_shape = [int(s) for s in vm.shape]
            x3 = ttnn.reshape(vx, [Bc, Tc, Dc])
            m3 = ttnn.reshape(vm, [Bc, 1, ctx.mod_shape[-1]])
            out3 = ttnn.zeros([Bc, Tc, Dc], dtype=vx.dtype, layout=vx.layout, device=vx.device())
            _get_fwd(ctx.key)(x3, m3, out3)
            ctx.save_for_backward(x3, m3)
            return ttnn.reshape(out3, [Bc, 1, Tc, Dc])

        @staticmethod
        def backward(ctx, grad_output):
            Bc, Tt, Dt, so_t, sh_t = ctx.key
            x3, m3 = ctx.saved_tensors
            dy3 = ttnn.reshape(grad_output, [Bc, Tt * TILE, Dt * TILE])
            dx3 = ttnn.zeros([Bc, Tt * TILE, Dt * TILE], dtype=dy3.dtype, layout=dy3.layout, device=dy3.device())
            dmod3 = ttnn.zeros([Bc, 1, ctx.mod_shape[-1]], dtype=dy3.dtype, layout=dy3.layout, device=dy3.device())
            _get_bwd(ctx.key)(dy3, x3, m3, dx3, dmod3)
            dx = ttnn.reshape(dx3, [Bc, 1, Tt * TILE, Dt * TILE])
            dmod = ttnn.reshape(dmod3, ctx.mod_shape)
            return dx, dmod

    return FusedAdaLNModulate
