# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill weights rebuilt on device from the fused decode stream copies.

In the batch-1 fused model the DRAM-streaming decode copies of the dense projections are the only stored copies: QKV,
o_proj and router (fused_decode/stream.py stream_linear_layout) and the LM head (fused_decode/terminal.py, the same
layout per device). Prefill rebuilds the normal [K, N] weight from a stream when it needs it and frees it after use.

Stream layout ([rows, banks * 32] tiles, one 32-wide column per DRAM bank): bank b holds the slots
[b * C, (b + 1) * C) (C = ceil(N / 32 / banks)), each a column of kt = K / 32 + 1 tiles (K / 32 weight tiles, then the
bias tile), and slot s holds output column tile order[s] (identity unless an order is given). So stream tile
(c * kt + k, b) is weight tile (k, order[b * C + c]).

Re-layout of the tile grid [C, kt, banks] to [kt, banks * C] (slot order), whole 32x32 tiles only:
  copy to DRAM interleaved in bf16              [1, 1, C * kt * 32, B * 32]
  view [1, C, kt * 32, B * 32], transpose(-2, -1) -> [1, C, B * 32, kt * 32]
  view [C, B, 32, kt * 32], permute(1, 0, 2, 3)   -> [B, C, 32, kt * 32]
  view [1, 1, B * C * 32, kt * 32], transpose      -> [1, 1, kt * 32, B * C * 32]: tile (k, s) = slot s, K tile k
  slices of the K weight rows and of the slot runs in output column order, concat, optional zero columns, cast.
Exactness: bf8 -> bf16 is exact (a bf8 value has at most 7 significant bits). The transposes run in bf16 because a bf8
transpose would regroup the 16-value shared exponents; the two in-tile transposes cancel. Every 32x32 tile then holds
its original values in its original place, so the final cast to bf8 sees the original 16-value groups and reproduces
the stored values (each value is representable at the group's shared exponent, which can only be the original one or
smaller). No step rounds.

The index math (LinearStreamPlan, relayout, vocab_resplit_pieces) does not import ttnn: tests/unit/
test_stream_relayout.py runs it with torch on the real stream_linear_layout. The ttnn backend (rebuild_linear,
resplit_vocab) imports ttnn when called.
"""

TILE = 32


def column_runs(slot_of_column):
    """[slot of output column tile n] -> [(first slot, count)]: maximal runs of consecutive slots holding consecutive
    output columns, in output column order."""
    runs = []
    for s in slot_of_column:
        if runs and runs[-1][0] + runs[-1][1] == s:
            runs[-1] = (runs[-1][0], runs[-1][1] + 1)
        else:
            runs.append((s, 1))
    return runs


class LinearStreamPlan:
    """Shapes and column runs of one stream_linear_layout weight: k x n weights over `banks` DRAM banks, order = the
    output column tile of each slot (stream_linear_layout `order`, length C * banks), None for the identity."""

    def __init__(self, k, n, banks, order=None):
        assert k % TILE == 0 and n % TILE == 0, f"stream weights are whole tiles: {k} x {n}"
        self.k, self.n, self.banks = k, n, banks
        self.n_tiles = n // TILE
        self.C = -(-self.n_tiles // banks)
        self.kt = k // TILE + 1
        self.slots = self.C * banks
        self.rows = self.C * self.kt * TILE  # fused_decode/stream.py linear_stream_rows
        order = list(range(self.slots)) if order is None else [int(c) for c in order]
        assert sorted(order) == list(range(self.slots)), f"order is not a permutation of the {self.slots} slots"
        slot = [0] * self.slots
        for s, col in enumerate(order):
            slot[col] = s
        self.identity = order == list(range(self.slots))
        self.runs = column_runs(slot[: self.n_tiles])

    def bank_groups(self, groups):
        """[(first bank, end bank)]: `groups` contiguous bank ranges (identity order only: each range then holds a
        contiguous range of output columns) that hold at least one real column."""
        if groups <= 1:
            return [(0, self.banks)]
        assert self.identity, "bank groups need the identity slot order"
        size = -(-self.banks // groups)
        out = [(b, min(b + size, self.banks)) for b in range(0, self.banks, size)]
        return [(b0, b1) for b0, b1 in out if b0 * self.C < self.n_tiles]

    def group_runs(self, b0, b1):
        """Column runs of the banks [b0, b1), as (first slot within the group, count)."""
        if (b0, b1) == (0, self.banks):
            return self.runs
        lo, hi = b0 * self.C, min(b1 * self.C, self.n_tiles)
        return [(0, hi - lo)]


def _slot_major(x, plan, g, ops):
    """[1, 1, C * kt * 32, g * 32] (g banks of the stream, work dtype) -> [1, 1, kt * 32, g * C * 32]: tile (k, s) =
    K tile k of the group's slot s (s = bank * C + c)."""
    C, kt = plan.C, plan.kt
    if C == 1:  # one slot per bank: already slot-major
        return x
    x = ops.reshape(x, [1, C, kt * TILE, g * TILE])
    y = ops.transpose(x)
    ops.free(x)
    y = ops.reshape(y, [C, g, TILE, kt * TILE])
    z = ops.swap01(y)
    ops.free(y)
    z = ops.reshape(z, [1, 1, g * C * TILE, kt * TILE])
    w = ops.transpose(z)
    ops.free(z)
    return w


def _concat(pieces, ops):
    if len(pieces) == 1:
        return pieces[0]
    out = ops.concat(pieces)
    for p in pieces:
        ops.free(p)
    return out


def relayout(stream, plan, ops, out_dtype, bank_groups=1, pad_n=None):
    """The stream of one device -> its [1, 1, K, N] weight (N = pad_n with zero columns after the real ones, if
    given), through `ops` (TtnnOps on device; tests: a torch backend). bank_groups > 1 (identity order only) rebuilds
    the columns of that many bank ranges one at a time, so the bf16 intermediates are about 1 / bank_groups of the
    whole (the LM head). Frees every intermediate; the stream is only read."""
    groups = plan.bank_groups(bank_groups)
    width = pad_n if pad_n is not None else plan.n
    assert width >= plan.n and width % TILE == 0, f"pad_n {pad_n} < N {plan.n} or not whole tiles"
    shared = ops.load(stream, work=False) if len(groups) > 1 else None
    outs, done = [], 0
    for i, (b0, b1) in enumerate(groups):
        if shared is None:
            x = ops.load(stream, work=True)
        else:
            part = ops.slice(shared, [0, 0, 0, b0 * TILE], [1, 1, plan.rows, b1 * TILE])
            x = ops.cast(part, None)
            if x is not part:
                ops.free(part)
        x = _slot_major(x, plan, b1 - b0, ops)
        pieces = [ops.slice(x, [0, 0, 0, s * TILE], [1, 1, plan.k, (s + n) * TILE]) for s, n in plan.group_runs(b0, b1)]
        if all(p is not x for p in pieces):
            ops.free(x)
        y = _concat(pieces, ops)
        cols = sum(n for _, n in plan.group_runs(b0, b1)) * TILE
        done += cols
        if i + 1 == len(groups) and width > done:
            z = ops.pad_cols(y, width - done)
            ops.free(y)
            y = z
        z = ops.cast(y, out_dtype)
        if z is not y:
            ops.free(y)
        outs.append(z)
    if shared is not None:
        ops.free(shared)
    return _concat(outs, ops)


def vocab_resplit_pieces(tp, vd, per_device):
    """Column ranges of the all-gathered decode-split logits ([.., tp * per_device]: device d's block [d * per_device,
    (d + 1) * per_device) holds vocab ids d * vd + j for j < vd and zeros after) whose concatenation is the prefill
    split's [.., tp * per_device] logits (vocab id v at column v, zeros after the tp * vd ids): every device's vd real
    columns in device order, then every device's zero tail."""
    assert vd <= per_device and vd % TILE == 0 and per_device % TILE == 0, (vd, per_device)
    real = [(d * per_device, d * per_device + vd) for d in range(tp)]
    tails = [(d * per_device + vd, (d + 1) * per_device) for d in range(tp) if vd < per_device]
    return real + tails


class TtnnOps:
    """relayout's device backend: DRAM interleaved tile tensors, bf16 work dtype."""

    def __init__(self):
        import ttnn

        self.ttnn = ttnn
        self.dram = ttnn.DRAM_MEMORY_CONFIG

    def load(self, stream, work):
        """The width-sharded stream -> an interleaved copy (work: in bf16). Never aliases the stream."""
        assert stream.memory_config().is_sharded(), "the decode stream copies are DRAM width-sharded"
        # (DRAM-sharded -> interleaved goes through ttnn's copy op, which converts tiles exactly bf8 -> bf16)
        if work:
            return self.ttnn.to_memory_config(stream, self.dram, dtype=self.ttnn.bfloat16)
        return self.ttnn.to_memory_config(stream, self.dram)

    def cast(self, x, dtype):
        dtype = self.ttnn.bfloat16 if dtype is None else dtype
        return x if x.dtype == dtype else self.ttnn.typecast(x, dtype, memory_config=self.dram)

    def reshape(self, x, shape):
        return self.ttnn.reshape(x, shape)  # a view: the last dim and tile-aligned second-last dims are kept

    def transpose(self, x):
        return self.ttnn.transpose(x, -2, -1, memory_config=self.dram)

    def swap01(self, x):
        return self.ttnn.permute(x, (1, 0, 2, 3), memory_config=self.dram)  # whole tiles (transpose CN)

    def slice(self, x, starts, ends):
        if list(starts) == [0] * len(starts) and list(ends) == [x.shape[i] for i in range(len(ends))]:
            return x  # (ttnn.slice of the whole tensor aliases its input)
        return self.ttnn.slice(x, starts, ends, memory_config=self.dram)

    def concat(self, pieces):
        return self.ttnn.concat(pieces, dim=3, memory_config=self.dram)

    def pad_cols(self, x, cols):
        return self.ttnn.pad(x, [(0, 0), (0, 0), (0, 0), (0, cols)], 0.0)

    def free(self, x):
        self.ttnn.deallocate(x)


def rebuild_linear(stream, plan, dtype, bank_groups=1, pad_n=None):
    """Device: a decode stream tensor (per device [1, 1, plan.rows, banks * 32]) -> a new DRAM interleaved
    [1, 1, K, N or pad_n] weight in `dtype` (bfloat8_b: the stream's own values; bfloat16: the same values). The caller
    frees it after use."""
    import ttnn

    assert dtype in (ttnn.bfloat8_b, ttnn.bfloat16), f"{dtype}: a rebuilt stream weight is bf8 or bf16 (exact)"
    assert stream.dtype in (ttnn.bfloat8_b, ttnn.bfloat16), f"stream dtype {stream.dtype}"
    assert stream.shape[-2] == plan.rows and stream.shape[-1] == plan.banks * TILE, (tuple(stream.shape), plan.rows)
    return relayout(stream, plan, TtnnOps(), dtype, bank_groups=bank_groups, pad_n=pad_n)


def resplit_vocab(logits, tp, vd, per_device, cluster_axis):
    """Device: LM-head logits in the decode vocab split (device d: ids [d * vd, (d + 1) * vd) in its first vd columns,
    zeros up to per_device) -> the prefill split (device d: ids [d * per_device, (d + 1) * per_device), zeros past the
    tp * vd ids), [.., per_device] per device. Tile-aligned moves only (all_gather, slices, concat, mesh_partition):
    the values are unchanged. Frees `logits`."""
    import ttnn

    dram = ttnn.DRAM_MEMORY_CONFIG
    rank = len(logits.shape)
    lead = [logits.shape[i] for i in range(rank - 1)]
    assert logits.shape[-1] == per_device, f"logits width {logits.shape[-1]} != {per_device}"
    gathered = ttnn.all_gather(logits, rank - 1, cluster_axis=cluster_axis, memory_config=dram)
    ttnn.deallocate(logits)
    pieces = [
        ttnn.slice(gathered, [0] * (rank - 1) + [a], lead + [b], memory_config=dram)
        for a, b in vocab_resplit_pieces(tp, vd, per_device)
    ]
    ttnn.deallocate(gathered)
    full = ttnn.concat(pieces, dim=rank - 1, memory_config=dram)
    for p in pieces:
        ttnn.deallocate(p)
    out = ttnn.mesh_partition(full, rank - 1, cluster_axis=cluster_axis, memory_config=dram)
    ttnn.deallocate(full)
    return out
