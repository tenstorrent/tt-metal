# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Inkling-2-Mini residual-path helpers for batch-1 decode on one card: row ops (RMS norms, block-state
stacks, depth attention), the cross-pass messages (MessageAttnRes) and HF cache indexing. All on device, no host
reads, static shapes: trace-safe. The per-layer block attention residual, message gates and short convs run as
the fused kernels in tt/fused/.

Row form: a per-token vector is a [1, 1, 1, C] TILE tensor ("row"); r vectors = [1, 1, r, C].
RMS norms run width-sharded over 18 cores (~10 us incl. reshards vs 25 us on one core).

Block-state stacks are kept already RMS-normalised and split along C into SPLIT chunks: [1, SPLIT, 32, C/SPLIT],
row k of every chunk = state k (rows >= depth zero). The C-long dot products of attention over depth then run as
SPLIT small batched matmuls + one sum (no fp32 0/1 row-move matmuls, no single-core K=C matmul).

MessageAttnRes takes a weights dict keyed by cache names (torch or host ttnn tensors):
    msg_q                                 [1,12,32,C] fp32, rows 0-3 of loop layer l = q, k, v, mlp
"""


import torch

import ttnn

TILE = 32
MAX_DEPTH = 5  # 4 sealed block states + 1 partial
NUM_LOOP_LAYERS = 12
NUM_CACHE_SETS = 48  # 24 layers + 12 loop layers x 2 extra passes (HF cache index)
SPLIT = 9  # C chunks of the depth-attention dot products (C=2304 -> 256 = 8 tiles)
RMS_GRID = (9, 2)  # width-sharded RMS norm: 18 cores (as tt/moe.py)


def compute_config(device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


def to_device(t, device, dtype=ttnn.float32):
    """Host torch tensor -> 4D TILE device tensor in DRAM."""
    while t.dim() < 4:
        t = t.unsqueeze(0)
    return ttnn.from_torch(
        t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _torch(w):
    return ttnn.to_torch(w) if isinstance(w, ttnn.Tensor) else w


def _pad_rows(t, rows=TILE):
    """[..., r, C] -> [..., rows, C] zero padded."""
    return torch.nn.functional.pad(t, (0, 0, 0, rows - t.shape[-2]))


def _f32(x):
    return x if x.dtype == ttnn.float32 else ttnn.typecast(x, ttnn.float32)


def split_rows(t):
    """[r, C] torch -> [1, SPLIT, r, C/SPLIT] (chunk b = columns b*C/SPLIT ...)."""
    r, c = t.shape
    return t.reshape(r, SPLIT, c // SPLIT).permute(1, 0, 2).unsqueeze(0).contiguous()


def split_k_weight(w, device, dtype=ttnn.bfloat16):
    """[K, N] torch -> [1, SPLIT, K/SPLIT, N] device weight for split_k_matmul."""
    k, n = w.shape
    return to_device(
        w.reshape(1, SPLIT, k // SPLIT, n).to(torch.bfloat16 if dtype == ttnn.bfloat16 else w.dtype), device, dtype
    )


def split_k_matmul(x, w, ckc):
    """x [1,1,r,K] @ w (split_k_weight) as SPLIT batched matmuls + fp32 sum: a [r,K]x[K,N<=6 tiles] matmul would
    run K serially on N cores (55 us at K=4608; split: 18-29 us). Returns [1,1,r,N] fp32."""
    r, k = x.shape[-2], x.shape[-1]
    if r == 1:
        xs = ttnn.reshape(x, [1, SPLIT, 1, k // SPLIT])
    else:
        xs = ttnn.permute(ttnn.reshape(x, [1, r, SPLIT, k // SPLIT]), (0, 2, 1, 3))
    return ttnn.sum(ttnn.matmul(xs, w, dtype=ttnn.float32, compute_kernel_config=ckc), dim=1, keepdim=True)


def cache_index(layer, loop_pass=1):
    """HF cache index of (layer, pass): L on pass 1, 24 + (p-2)*12 + (L-6) on passes 2-3 of loop layers."""
    return layer if loop_pass == 1 else 24 + (loop_pass - 2) * NUM_LOOP_LAYERS + (layer - 6)


class RowOps:
    """Shared per-device constants and helpers of the row-form residual path."""

    def __init__(self, device, c=2304):
        self.device = device
        self.c = c
        self.ckc = compute_config(device)
        self._sel = {}
        self.onehot = []  # [1,1,32,1]: 1 at row k (stack row placement)
        for k in range(TILE):
            o = torch.zeros(1, 1, TILE, 1)
            o[..., k, 0] = 1
            self.onehot.append(to_device(o, device))
        self.depth_mask = {}  # additive softmax mask over depth columns (keep cols < d), one full tile: the fused
        for d in range(1, MAX_DEPTH + 1):  # scale_mask_softmax applies it to every tile row of the logits
            m = torch.full((TILE, TILE), -1e9)
            m[:, :d] = 0
            self.depth_mask[d] = to_device(m, device)
        self.rms_shard = ttnn.create_sharded_memory_config(
            (TILE, c // (RMS_GRID[0] * RMS_GRID[1])),
            core_grid=ttnn.CoreGrid(y=RMS_GRID[1], x=RMS_GRID[0]),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        bw = c // (RMS_GRID[0] * RMS_GRID[1]) // TILE
        self.rms_program = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=RMS_GRID,
            subblock_w=max(d for d in range(1, 5) if bw % d == 0),
            block_h=1,
            block_w=bw,
            inplace=False,
        )

    def selector(self, rows_out, rows_in, pairs):
        """0/1 matrix [rows_out, rows_in] with 1 at each (out_row, in_row); out = sel @ x."""
        key = (rows_out, rows_in, tuple(pairs))
        if key not in self._sel:
            s = torch.zeros(rows_out, rows_in)
            for o, i in pairs:
                s[o, i] = 1
            self._sel[key] = to_device(s, self.device)
        return self._sel[key]

    def mm(self, a, b, **kw):
        return ttnn.matmul(a, b, dtype=ttnn.float32, compute_kernel_config=self.ckc, **kw)

    def rms(self, x, weight=None, eps=1e-6, dtype=ttnn.float32):
        """RMSNorm over C of x [1,1,r<=32,C] (fp32 math; bf16 input is cast while resharding). Interleaved DRAM out."""
        xs = ttnn.to_memory_config(x, self.rms_shard, dtype=ttnn.float32)
        y = ttnn.rms_norm(
            xs,
            weight=weight,
            epsilon=eps,
            program_config=self.rms_program,
            compute_kernel_config=self.ckc,
            memory_config=self.rms_shard,
        )
        ttnn.deallocate(xs)
        out = ttnn.to_memory_config(y, ttnn.DRAM_MEMORY_CONFIG, dtype=dtype)
        ttnn.deallocate(y)
        return out

    def seal(self, stack, h, k):
        """Normalised split stack with rms(h) (h: row) in row k (row k of stack must be zero). stack=None starts
        an empty stack. Returns a new tensor; the old one stays valid (passes 2-3 restart from the pre-loop stack)."""
        n = ttnn.reshape(self.rms(h), [1, SPLIT, 1, self.c // SPLIT])
        if stack is None:
            return ttnn.multiply(self.onehot[k], n, dtype=ttnn.float32)
        return ttnn.addcmul(stack, self.onehot[k], n)

    def attend(self, q, stack, mask, nq):
        """softmax(q . n^T + mask) . n over stack rows; q [1,SPLIT,nq,C/SPLIT] pre-scaled. Returns [1,1,nq,C] fp32."""
        part = self.mm(q, stack, transpose_b=True)  # [1,SPLIT,nq,32]
        logits = ttnn.sum(part, dim=1, keepdim=True)
        w = ttnn.scale_mask_softmax(logits, None, mask, compute_kernel_config=self.ckc, numeric_stable=True)
        out = self.mm(w, stack)  # [1,SPLIT,nq,C/SPLIT]
        if nq > 1:
            out = ttnn.permute(out, (0, 2, 1, 3))
        return ttnn.reshape(out, [1, 1, nq, self.c])


class MessageAttnRes:
    """Inkling2LoopMessageAttentionResidual for one whole pass. Keys = [input of L12 (block state), output of
    L17] of the previous pass, shared by all 12 loop layers: one [48, C] query attend per pass."""

    def __init__(self, rows: RowOps, weights, name="msg_q"):
        q = _torch(weights[name]).float().reshape(NUM_LOOP_LAYERS, TILE, -1)[:, :4]  # [12, 4, C]
        c = q.shape[-1]
        self.rows = rows
        self.nq = 4 * NUM_LOOP_LAYERS
        self.q = to_device(split_rows(q.reshape(-1, c) * c**-0.5), rows.device)  # row 4l+j
        self.mask = rows.depth_mask[2]
        for l in range(NUM_LOOP_LAYERS):  # selectors exist before any trace capture
            self.qkv_sel(l), self.mlp_sel(l)

    def qkv_sel(self, l):
        return self.rows.selector(3, self.nq, [(j, 4 * l + j) for j in range(3)])

    def mlp_sel(self, l):
        return self.rows.selector(1, self.nq, [(0, 4 * l + 3)])

    def all_messages(self, h12_in, h17_out):
        """Rows of the previous pass -> [1,1,48,C] fp32, row 4l+j = message j (q, k, v, mlp) of loop layer l."""
        r = self.rows
        return r.attend(self.q, r.seal(r.seal(None, h12_in, 0), h17_out, 1), self.mask, self.nq)
