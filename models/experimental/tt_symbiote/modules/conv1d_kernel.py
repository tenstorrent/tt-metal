# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Fused depthwise causal-conv1d-update + silu tt-lang kernel.

Replaces the chain of ~13 TTNN ops in `ttnn_causal_conv1d_update_step`
(4× shift copies + 4× weight muls + 3× adds + bias add + silu) with a
single tt-lang dispatch.

Math (per channel `c`):
    new_slot0[c] = old_slot1[c]   (shift)
    new_slot1[c] = old_slot2[c]
    new_slot2[c] = old_slot3[c]
    new_slot3[c] = x[c]
    y[c] = silu(w0[c]*new_slot0[c] + w1[c]*new_slot1[c]
              + w2[c]*new_slot2[c] + w3[c]*new_slot3[c]
              + bias[c])

Equivalent to torch_causal_conv1d_update on a single token with kernel
size K=4. Channels are fully independent, so the kernel parallelises
across tile-columns of D.

Old `slot0` is intentionally never read: its storage is reused to hold
the post-shift value (= old slot1). This saves one CB and one DRAM read.
"""

import ttl


TILE = 32
CONV_KERNEL_SIZE = 4  # K, fixed for Qwen3.5 / Qwen3.6 GatedDeltaNet


def _make_conv1d_update_kernel(conv_dim, grid_x=8, grid_y=8):
    """Build a fused conv1d_update kernel specialised to `conv_dim`.

    Args:
        conv_dim: number of channels = key_dim*2 + value_dim. Must be
            divisible by `grid_x * grid_y * TILE` so tiles distribute
            evenly across cores.
        grid_x, grid_y: kernel grid. Default (8, 8) = 64 cores, the
            full Blackhole / Wormhole compute grid.

    Returns:
        A ttl.operation that, when called with
        `(x, slot0, slot1, slot2, slot3, w0, w1, w2, w3, bias, y_out)`
        mutates `slot0..slot3` in place and writes `y_out`.
    """
    num_tiles = conv_dim // TILE
    num_cores = grid_x * grid_y
    if num_tiles % num_cores != 0:
        raise ValueError(
            f"conv_dim={conv_dim} (={num_tiles} tiles) not evenly "
            f"divisible by grid {grid_x}x{grid_y} ({num_cores} cores)"
        )
    tiles_per_core = num_tiles // num_cores

    @ttl.operation(
        grid=(grid_x, grid_y),
        # fp32_dest_acc_en is OFF on purpose. Enabling it makes each DST tile
        # take 2 physical register slots (fp32 vs bf16); with only 4 physical
        # DST registers we then can't hold even (running_y + one mul product
        # + silu working tile) without spilling. Conv1d update is short
        # enough that bf16 accumulation is fine.
        fp32_dest_acc_en=False,
        # No --ttl-maximize-dst: we fan out waited CB values into multiple
        # output CBs (slot updates + y); aggressive DST reuse can cause
        # the same kind of garbled output we saw in gdn_kernel earlier.
        options="",
    )
    def conv1d_update(
        x,
        slot0,
        slot1,
        slot2,
        slot3,
        w0,
        w1,
        w2,
        w3,
        bias,
        y_out,
    ):
        # Per-tile input CBs. slot0 is NOT read -- its post-shift value
        # equals pre-shift slot1, which we already have via si1.
        si1 = ttl.make_dataflow_buffer_like(slot1, shape=(1, 1), block_count=2)
        si2 = ttl.make_dataflow_buffer_like(slot2, shape=(1, 1), block_count=2)
        si3 = ttl.make_dataflow_buffer_like(slot3, shape=(1, 1), block_count=2)
        xi = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        wi0 = ttl.make_dataflow_buffer_like(w0, shape=(1, 1), block_count=2)
        wi1 = ttl.make_dataflow_buffer_like(w1, shape=(1, 1), block_count=2)
        wi2 = ttl.make_dataflow_buffer_like(w2, shape=(1, 1), block_count=2)
        wi3 = ttl.make_dataflow_buffer_like(w3, shape=(1, 1), block_count=2)
        bi = ttl.make_dataflow_buffer_like(bias, shape=(1, 1), block_count=2)

        # Output CBs (one per output tensor that compute writes).
        yo = ttl.make_dataflow_buffer_like(y_out, shape=(1, 1), block_count=2)
        so0 = ttl.make_dataflow_buffer_like(slot0, shape=(1, 1), block_count=2)
        so1 = ttl.make_dataflow_buffer_like(slot1, shape=(1, 1), block_count=2)
        so2 = ttl.make_dataflow_buffer_like(slot2, shape=(1, 1), block_count=2)
        so3 = ttl.make_dataflow_buffer_like(slot3, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            for _ in range(tiles_per_core):
                with (
                    si1.wait() as s1,
                    si2.wait() as s2,
                    si3.wait() as s3,
                    xi.wait() as xx,
                    wi0.wait() as ww0,
                    wi1.wait() as ww1,
                    wi2.wait() as ww2,
                    wi3.wait() as ww3,
                    bi.wait() as bb,
                ):
                    # Mutate slots first (CB->CB copies of waited values; no DST).
                    # Keeps DST registers free for the arithmetic below.
                    # slot0 <- pre-shift slot1, slot1 <- pre-shift slot2,
                    # slot2 <- pre-shift slot3, slot3 <- x.
                    with so0.reserve() as sb:
                        sb.store(s1)
                    with so1.reserve() as sb:
                        sb.store(s2)
                    with so2.reserve() as sb:
                        sb.store(s3)
                    with so3.reserve() as sb:
                        sb.store(xx)
                    # y = silu(w0*s1 + w1*s2 + w2*s3 + w3*x + bias), built up
                    # one binary op at a time so we never have more than two
                    # DST tiles live simultaneously (the running sum + one
                    # product). The fused single-expression form exhausted the
                    # 4 DST registers available on Blackhole.
                    y = ww0 * s1
                    y = y + ww1 * s2
                    y = y + ww2 * s3
                    y = y + ww3 * xx
                    y = y + bb
                    y = ttl.math.silu(y)
                    with yo.reserve() as yb:
                        yb.store(y)

        @ttl.datamovement()
        def dm_read():
            nx, ny = ttl.node(dims=2)
            base = (ny * grid_x + nx) * tiles_per_core
            for i in range(tiles_per_core):
                t = base + i
                with si1.reserve() as blk:
                    ttl.copy(slot1[0, t], blk).wait()
                with si2.reserve() as blk:
                    ttl.copy(slot2[0, t], blk).wait()
                with si3.reserve() as blk:
                    ttl.copy(slot3[0, t], blk).wait()
                with xi.reserve() as blk:
                    ttl.copy(x[0, t], blk).wait()
                with wi0.reserve() as blk:
                    ttl.copy(w0[0, t], blk).wait()
                with wi1.reserve() as blk:
                    ttl.copy(w1[0, t], blk).wait()
                with wi2.reserve() as blk:
                    ttl.copy(w2[0, t], blk).wait()
                with wi3.reserve() as blk:
                    ttl.copy(w3[0, t], blk).wait()
                with bi.reserve() as blk:
                    ttl.copy(bias[0, t], blk).wait()

        @ttl.datamovement()
        def dm_write():
            nx, ny = ttl.node(dims=2)
            base = (ny * grid_x + nx) * tiles_per_core
            for i in range(tiles_per_core):
                t = base + i
                with yo.wait() as blk:
                    ttl.copy(blk, y_out[0, t]).wait()
                with so0.wait() as blk:
                    ttl.copy(blk, slot0[0, t]).wait()
                with so1.wait() as blk:
                    ttl.copy(blk, slot1[0, t]).wait()
                with so2.wait() as blk:
                    ttl.copy(blk, slot2[0, t]).wait()
                with so3.wait() as blk:
                    ttl.copy(blk, slot3[0, t]).wait()

    return conv1d_update


# Per-conv_dim kernel cache. Each call to `fused_conv1d_update_step`
# below picks the right kernel based on the input dim.
_KERNEL_BY_DIM = {}


def fused_conv1d_update_step(x, slots, weights_per_k, bias, y_out):
    """Drop-in replacement for ttnn_decode_ops.ttnn_causal_conv1d_update_step.

    Args:
        x: ttnn.Tensor [1, conv_dim] - new input row (replicated).
        slots: list of K=4 ttnn.Tensors [1, conv_dim] - persistent shift
            register. `slots[0]` oldest, `slots[K-1]` newest. Mutated
            in place.
        weights_per_k: list of K=4 ttnn.Tensors [1, conv_dim] - depthwise
            conv weights per lag.
        bias: ttnn.Tensor [1, conv_dim].
        y_out: ttnn.Tensor [1, conv_dim] - pre-allocated output buffer.
            The kernel writes silu(conv_output) here.

    Returns:
        y_out (the same tensor passed in, now containing the output).
    """
    if len(slots) != CONV_KERNEL_SIZE or len(weights_per_k) != CONV_KERNEL_SIZE:
        raise ValueError(f"fused_conv1d_update_step requires K={CONV_KERNEL_SIZE} slots & weights")
    conv_dim = int(x.shape[-1])
    if conv_dim not in _KERNEL_BY_DIM:
        _KERNEL_BY_DIM[conv_dim] = _make_conv1d_update_kernel(conv_dim)
    kernel_fn = _KERNEL_BY_DIM[conv_dim]
    kernel_fn(
        x,
        slots[0],
        slots[1],
        slots[2],
        slots[3],
        weights_per_k[0],
        weights_per_k[1],
        weights_per_k[2],
        weights_per_k[3],
        bias,
        y_out,
    )
    return y_out
