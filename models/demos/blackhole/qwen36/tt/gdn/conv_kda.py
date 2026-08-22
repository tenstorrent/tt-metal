# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused decode conv1d+SiLU via the KDA qkv_causal_conv1d_silu kernel.

Replaces the composite decode shift-register (K copies + mul + K-1 mac + silu
+ 3 slices) with one fused op. The KDA kernel is single-stream causal over its
sequence dim, while GDN decode is B independent single-token streams, so the
per-user tap windows are laid out user-major as one [1, K*B, C] sequence: user
b occupies rows [K*b, K*b+K) oldest-to-newest, and only output row K*b + K-1
(the row whose K-tap window sits entirely inside user b's block) is real. The
valid rows are extracted with a constant one-hot select matmul; the other rows
mix adjacent users and are discarded.

Tap mapping is identity: conv_taps[j] multiplies the token j-(K-1) steps in the
past in both the composite MAC and the KDA kernel (tap0 = oldest). SiLU is
applied after the tap sum in both, and splitting silu(conv) at the q/k/v
channel boundaries equals silu of the splits, so q_width/k_width/v_width give
the same q/k/v the composite slices produce.

All per-step work is device-only (trace-safe); the one-hot select and history
placeholder are prebuilt host-side in build_buffers.
"""
import torch

import ttnn

_KDA_TAP_COUNT = 4  # the KDA kernel is hardwired to a 4-tap window


def default_channel_chunk(channels):
    """Largest tile-multiple divisor of `channels` up to 128. More blocks =
    more cores for the tiny decode T; 32 always divides a tile-aligned C."""
    for c in (128, 96, 64, 32):
        if channels % c == 0:
            return c
    return channels


class KDAConvBuffers:
    """Persistent per-layer device state for the KDA decode conv.

    win:  [B, K, C] ROW_MAJOR bf16 — per-user tap windows (dim1 oldest->newest).
          Canonical decode conv state while the KDA path is active; the
          composite conv_states list is only a prefill-side source to sync from.
    hist: [1, K-1, C] ROW_MAJOR zeros — satisfies the kernel's history input;
          only user 0's discarded rows ever read it.
    sel:  [1, B, K*B] TILE one-hot — row b picks sequence row K*b + K-1.
    """

    def __init__(self, mesh, batch, kernel_size, channels, channel_chunk=None):
        assert kernel_size == _KDA_TAP_COUNT, f"KDA conv kernel is 4-tap, got K={kernel_size}"
        assert (kernel_size * batch) % 32 == 0, f"K*B={kernel_size * batch} must be tile-aligned for the KDA kernel"
        assert channels % 32 == 0, f"channels={channels} must be tile-aligned"
        self.B = batch
        self.K = kernel_size
        self.C = channels
        self.chunk = channel_chunk if channel_chunk is not None else default_channel_chunk(channels)

        def rep(t, layout):
            kw = {"mesh_mapper": ttnn.ReplicateTensorToMesh(mesh)} if isinstance(mesh, ttnn.MeshDevice) else {}
            return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=layout, device=mesh, **kw)

        self.win = rep(torch.zeros(batch, kernel_size, channels, dtype=torch.bfloat16), ttnn.ROW_MAJOR_LAYOUT)
        self.zero_win = rep(torch.zeros(batch, kernel_size, channels, dtype=torch.bfloat16), ttnn.ROW_MAJOR_LAYOUT)
        self.hist = rep(torch.zeros(1, kernel_size - 1, channels, dtype=torch.bfloat16), ttnn.ROW_MAJOR_LAYOUT)
        sel = torch.zeros(1, batch, kernel_size * batch, dtype=torch.bfloat16)
        for b in range(batch):
            sel[0, b, kernel_size * b + kernel_size - 1] = 1.0
        self.sel = rep(sel, ttnn.TILE_LAYOUT)
        # HiFi4 + fp32 acc keeps the one-hot select exactly value-preserving.
        self.select_cfg = ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        )

    def deallocate(self):
        for t in (self.win, self.zero_win, self.hist, self.sel):
            ttnn.deallocate(t)

    def reset(self):
        """Zero the tap windows in place (trace-address preserving)."""
        ttnn.copy(self.zero_win, self.win)


def _dealloc_unless_alias(view, base):
    """Free `view` only when reshape materialized a copy (a view shares base's buffer)."""
    if view.buffer_address() != base.buffer_address():
        ttnn.deallocate(view)


def rebuild_window(bufs, conv_states):
    """Sync the KDA windows from the composite conv_states list (st[0] oldest),
    in place. Called wherever prefill writes conv_states so decode continues
    from the produced shift register; device-only, so safe inside a prefill trace."""
    B, K, C = bufs.B, bufs.K, bufs.C
    cols = []
    for m in range(K):
        rm = ttnn.to_layout(conv_states[m], ttnn.ROW_MAJOR_LAYOUT)  # [1, B, C]
        col = ttnn.reshape(rm, (B, 1, C))
        cols.append((rm, col))
    new = ttnn.concat([c for (_, c) in cols], dim=1)  # [B, K, C]
    ttnn.copy(new, bufs.win)
    ttnn.deallocate(new)
    for rm, col in cols:
        _dealloc_unless_alias(col, rm)
        ttnn.deallocate(rm)


def write_window_slot(bufs, convs, slot):
    """Write one user's K taps (convs[m] [1,1,C], m oldest->newest) into window
    row-block `slot`, preserving the other users. Does not consume convs."""
    B, K, C = bufs.B, bufs.K, bufs.C
    rows = []
    for c in convs:
        rm = ttnn.to_layout(c, ttnn.ROW_MAJOR_LAYOUT)
        rows.append((rm, ttnn.reshape(rm, (1, 1, C))))
    user = ttnn.concat([r for (_, r) in rows], dim=1)  # [1, K, C]
    parts = []
    if slot > 0:
        parts.append(ttnn.slice(bufs.win, (0, 0, 0), (slot, K, C)))
    parts.append(user)
    if slot < B - 1:
        parts.append(ttnn.slice(bufs.win, (slot + 1, 0, 0), (B, K, C)))
    new = ttnn.concat(parts, dim=0)
    ttnn.copy(new, bufs.win)
    ttnn.deallocate(new)
    for p in parts:
        ttnn.deallocate(p)
    for rm, r in rows:
        # `user` consumed the reshaped rows; free the row-major copies.
        if r is not rm and r.buffer_address() != rm.buffer_address():
            ttnn.deallocate(r)
        ttnn.deallocate(rm)


def gather_window_slots(bufs, idx):
    """Window row-block i takes the block previously at idx[i] (vLLM batch condense).
    `new` is fully materialized before the copy, so self-gather is safe."""
    B, K, C = bufs.B, bufs.K, bufs.C
    rows = [ttnn.slice(bufs.win, (i, 0, 0), (i + 1, K, C)) for i in idx]
    new = ttnn.concat(rows, dim=0)
    ttnn.copy(new, bufs.win)
    ttnn.deallocate(new)
    for r in rows:
        ttnn.deallocate(r)


def decode_conv(bufs, qkv, taps, q_width, v_width, memory_config=None):
    """One decode step: shift every user's window, append qkv, run the fused
    conv+SiLU+split, select the valid row per user.

    qkv:  [1, B, C] TILE bf16 (full batch width — caller pads bucketed decode).
    taps: the 4 composite conv taps ([1,1,C] TILE bf16), oldest-token first.
    Returns (q [1,B,q_width], k [1,B,q_width], v [1,B,v_width]) TILE, window
    updated in place. Device-only (trace-safe).
    """
    B, K, C = bufs.B, bufs.K, bufs.C
    mc = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    # Shift: window rows 1..K-1 move down, the new token lands in row K-1.
    x = ttnn.to_layout(qkv, ttnn.ROW_MAJOR_LAYOUT)  # [1, B, C]
    x_col = ttnn.reshape(x, (B, 1, C))
    shifted = ttnn.slice(bufs.win, (0, 1, 0), (B, K, C))  # [B, K-1, C]
    new_win = ttnn.concat([shifted, x_col], dim=1)  # [B, K, C]
    ttnn.copy(new_win, bufs.win)
    ttnn.deallocate(new_win)
    ttnn.deallocate(shifted)
    _dealloc_unless_alias(x_col, x)
    ttnn.deallocate(x)

    # User-major windows as one causal sequence; only rows K*b+K-1 are valid.
    seq = ttnn.reshape(bufs.win, (1, K * B, C))
    q4, k4, v4 = ttnn.experimental.kda.qkv_causal_conv1d_silu(
        seq,
        bufs.hist,
        taps[0],
        taps[1],
        taps[2],
        taps[3],
        q_width,
        q_width,
        v_width,
        program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=bufs.chunk),
        memory_config=mc,
    )
    _dealloc_unless_alias(seq, bufs.win)

    q = ttnn.matmul(bufs.sel, q4, memory_config=mc, compute_kernel_config=bufs.select_cfg)
    k = ttnn.matmul(bufs.sel, k4, memory_config=mc, compute_kernel_config=bufs.select_cfg)
    v = ttnn.matmul(bufs.sel, v4, memory_config=mc, compute_kernel_config=bufs.select_cfg)
    ttnn.deallocate(q4)
    ttnn.deallocate(k4)
    ttnn.deallocate(v4)
    return q, k, v
