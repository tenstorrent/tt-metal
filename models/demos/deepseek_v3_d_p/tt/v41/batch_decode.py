# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 batched decode: one step for every user of the mesh (harness; eager, or traced once).

``TtV41BatchDecoder`` runs ``TtV41Decoder``'s step on a tile whose rows are different users: row ``u`` of the chips
of mesh row ``r`` is slot ``r * users_per_rank + u`` of the :class:`V41DecodePool`, so the SP ranks decode different
users (TP splits hidden and heads as before) and the per-step host inputs are per rank: the slots' tokens, RoPE rows
at their positions, the indexer masks' visible counts, the ring slot and the compressed rows the step writes.

Per step and layer the graph is the single-token one with the position-dependent pieces made per row:

* the window row of every user goes to its layer ring at slot ``step % 128`` (``scatter_rows``; the window index
  rows are a constant per layer, the ring being the window);
* a ratio-2 source pools every row with the pool's carry buffers and writes the carry back unconditionally (the odd
  rows' pooled result is their group's row, the even rows' is junk the scatter skips: the step's write indices are
  the sentinel for rows that complete no group), a ratio-1 source writes every active row; both go straight to the
  user's row ``position // ratio`` (no scratch row, no copy after the step);
* an index source scores the query tile against each slot's index keys (``cache_batch_idx``), takes row ``u`` of
  slot ``u``'s selection, and the attention reads the pool with per-row offsets;
* the MoE sees ``users_per_rank`` real rows per chip (``padding_config``), so only they are dispatched.

Users join from a prefill state (``join``: the pool copies the state's caches into the slot) and leave by being
forgotten; a free slot's rows still run (its inputs are zeros, its writes go nowhere). The step's graph is the same
for every position, so a single capture replays every step (``trace=True``); with ``V41_DECODE_C1_MOE=1`` a second
capture serves the steps with exactly one live row (the MoEs on that row broadcast to every chip,
``tt/v41/moe_c1.py``). Limits as ``TtV41Decoder``: text only, positions from ``SLIDING_WINDOW - 1`` to below
``max_seq_len``, compressed rows within the candidate budget.
"""

import os
import time
from dataclasses import dataclass

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.mhc.tt_mhc import mhc_expand
from models.demos.deepseek_v3_d_p.tt.v41 import attention as attention_module
from models.demos.deepseek_v3_d_p.tt.v41.attention import TOPK_ALIGN, TtV41Attention
from models.demos.deepseek_v3_d_p.tt.v41.cache import V41PrefillState
from models.demos.deepseek_v3_d_p.tt.v41.decode import MOE_ROWS, ROWS, TtV41Decoder
from models.demos.deepseek_v3_d_p.tt.v41.decode_norm import DECODE_NORM, DecodeRmsNorm
from models.demos.deepseek_v3_d_p.tt.v41.flash_sdpa import FlashSdpa
from models.demos.deepseek_v3_d_p.tt.v41.head_layout import o_heads, q_heads
from models.demos.deepseek_v3_d_p.tt.v41.indexer import CandidateBlocks, DecodePins, TtV41Indexer
from models.demos.deepseek_v3_d_p.tt.v41.indexer_kernels import range_scores
from models.demos.deepseek_v3_d_p.tt.v41.layout import TP_AXIS
from models.demos.deepseek_v3_d_p.tt.v41.moe import TtV41Moe
from models.demos.deepseek_v3_d_p.tt.v41.moe_c1 import _SENTINEL as C1_ONEHOT_SENTINEL
from models.demos.deepseek_v3_d_p.tt.v41.moe_c1 import C1Moe
from models.demos.deepseek_v3_d_p.tt.v41.pool import RING, V41DecodePool, scatter_rows
from models.demos.deepseek_v3_d_p.tt.v41.qdq import fp4_e4m3_qdq, fp8_qdq
from models.demos.deepseek_v3_d_p.tt.v41.rope import RopeTables, cos_sin
from models.demos.deepseek_v3_d_p.tt.v41.transformer import TtV41Transformer
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat
from models.demos.deepseek_v3_d_p.utils.sub_device_trace import SubDeviceTraceController

NEG_INF = float("-inf")
JOIN_MIN_WIDTH = 8192  # the narrowest join index (tokens); doubles up to the pool's max_seq_len

# the c1 index path's one-hot matmuls: exact row picks (HiFi4, fp32 accumulation)
_C1_HIFI4 = ttnn.types.BlackholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False, math_approx_mode=False
)


def c1_place(topk, put_row):
    """The live user's selection ``topk`` [1, 1, 1, K] uint32 row-major into its row of the tile, the other rows the
    sentinel -> [1, 1, ROWS, K] uint32 row-major. ``put_row`` is the step's ``c1put`` page ([1, 1, 1, ROWS] uint32
    row-major: word u is 0 on the live rank's row u, the one-hot sentinel elsewhere; ``C1Moe.host_rows``). The row
    is repeated over the tile and masked by the live row's column in fp32 (ids below 2^24 and the 0 / 1 mask are
    exact there; a matmul is not: the FPU keeps ~10 bits of an fp32 input), 1 added first so the masked rows and
    the sentinel tail read -1 after it is taken back."""
    rows = topk.shape[2]
    col = ttnn.to_layout(ttnn.reshape(put_row, (1, 1, put_row.shape[3], 1)), ttnn.TILE_LAYOUT)  # [ROWS, 1]
    live = ttnn.typecast(ttnn.eq(col, 0), ttnn.float32)  # 1.0 at the live row
    tile = ttnn.to_layout(ttnn.concat([topk] * (put_row.shape[3] // rows), dim=2), ttnn.TILE_LAYOUT)
    ids = ttnn.typecast(ttnn.typecast(tile, ttnn.int32), ttnn.float32)
    placed = ttnn.subtract(ttnn.multiply(ttnn.add(ids, 1.0), live), 1.0)
    return ttnn.to_layout(ttnn.typecast(ttnn.typecast(placed, ttnn.int32), ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)


NEG_SIN = os.environ.get("V41_DECODE_NEG_SIN", "layer")
assert NEG_SIN in ("layer", "once"), NEG_SIN


@dataclass
class _User:
    position: int  # the position the next token goes to
    engram_history: torch.Tensor | None = None


class StepPacker:
    """The step's small per-rank inputs packed into few host tensors (``V41_DECODE_HOST_INPUTS=packed``): every ``from_torch`` with a mesh mapper and every ``copy_host_to_device_tensor`` cost
    0.2 to 0.4 ms for 32 chips whatever the size, and the step had 19 of each (11 + 6 ms a step at two rows).
    ``pack`` builds one uint32 row-major tensor of the integer rows (``[sp, 1, R, ROWS]``: token, ring, the
    per-ratio newest and comp pages, the sampler's k), one bf16 row-major tensor of the sampler's p and
    temperature, one bf16 tiled tensor of the six RoPE rows side by side (``[sp, 1, ROWS, 6 * rope_dim]``) and,
    when the step masks, one fp32 tiled tensor of the per-ratio visible counts as columns; ``unpack`` slices the
    device buffers back into the tensors the graph reads, keyed as before (a dozen tiny ops inside the capture).
    The row and column orders are fixed at the first ``pack``."""

    INTS = "ints"
    FLOATS = "floats"
    ROPE = "rope"
    VISIBLE = "visible"
    ONE = "one"

    def __init__(self, pool, sp: int, mode: str = "packed"):
        """``mode`` ``packed``: the three host tensors above (four when the step masks); ``one``
        (``V41_DECODE_HOST_INPUTS=one``): the integer rows, the sampler's rows and the RoPE
        rows in one uint32 row-major tensor of one row per rank (``[sp, 1, 1, (ints + floats) * ROWS + ROWS * 6 *
        rope_dim]``, one page; a bf16 value travels as its fp32 bit pattern, ``bits << 16``) built from per-chip
        host shards (``ttnn.from_host_shards``) and uploaded once: a ``from_torch`` with a mesh mapper costs 0.4 to
        0.5 ms a call for 32 chips whatever the size, and a ``copy_host_to_device_tensor`` pays per page (the
        14-row integer tensor 1.66 ms against 0.2 for the one-row ones). ``unpack`` then
        bitcasts the bf16 rows (``ttnn.bitcast`` to fp32, ``typecast`` to bf16: exact, the low half is zero) and
        reshapes and tilizes the RoPE block on device, a dozen tiny ops inside the capture as before plus ten. The
        visible counts (the masks' input, not served) stay a tensor of their own."""
        assert mode in ("packed", "one"), mode
        self.pool, self.sp, self.mode = pool, sp, mode
        self.int_rows: list[str] = []
        self.float_rows: list[str] = []
        self.rope_cols: list[str] = []
        self.visible_cols: list[str] = []
        self.wide_cols: list[str] = []
        self.wide_widths: list[int] = []
        self.rope_width = 0
        self.one_host: torch.Tensor | None = None  # the "one" mode's host rows, reused every step

    @staticmethod
    def _order(have: list[str], keys) -> list[str]:
        keys = list(keys)
        if not have:
            have.extend(keys)
        assert have == keys, (have, keys)
        return have

    def pack(self, ints: dict, floats: dict, rope: dict, visible: dict, wide: dict | None = None) -> dict:
        """``ints`` name -> int32 ``[sp, ROWS]``; ``floats`` name -> fp32 ``[sp, ROWS]``; ``rope`` name -> (cos, sin)
        bf16 ``[sp * ROWS, rope_dim]``; ``visible`` name -> fp32 ``[sp, ROWS]``; ``wide`` name -> fp32
        ``[sp * ROWS, w]`` (``w`` a multiple of 32; tiled columns after the visible ones in the same tensor) -> the
        host tensors (per rank)."""
        sp, pool = self.sp, self.pool
        out = {}
        if wide:
            cols = self._order(self.wide_cols, wide)
            self.wide_widths = [wide[k].shape[1] for k in cols]
            assert all(w % 32 == 0 for w in self.wide_widths), self.wide_widths
            extra = torch.cat([wide[k].float().view(sp, 1, ROWS, -1) for k in cols], dim=3)
        else:
            extra = None
        if self.mode == "one":
            out[self.ONE] = self._pack_one(ints, floats, rope)
            if visible or extra is not None:
                cols = self._order(self.visible_cols, visible)
                packed = torch.zeros(sp, 1, ROWS, 32, dtype=torch.float32)
                for j, k in enumerate(cols):
                    packed[:, 0, :, j] = visible[k].float()
                if extra is not None:
                    packed = torch.cat([packed, extra], dim=3)
                out[self.VISIBLE] = pool.per_rank(packed.contiguous(), ttnn.float32, ttnn.TILE_LAYOUT, device=False)
            return out
        rows = self._order(self.int_rows, ints)
        packed = torch.stack([ints[k].to(torch.int32) for k in rows], dim=1).view(sp, 1, len(rows), ROWS)
        out[self.INTS] = pool.per_rank(packed.contiguous(), ttnn.uint32, device=False)
        if floats:
            rows = self._order(self.float_rows, floats)
            packed = torch.stack([floats[k].float() for k in rows], dim=1).view(sp, 1, len(rows), ROWS)
            out[self.FLOATS] = pool.per_rank(packed.contiguous(), ttnn.bfloat16, device=False)
        cols = self._order(self.rope_cols, rope)
        packed = torch.cat([t for k in cols for t in rope[k]], dim=1).view(sp, 1, ROWS, -1)
        out[self.ROPE] = pool.per_rank(packed.contiguous(), ttnn.bfloat16, ttnn.TILE_LAYOUT, device=False)
        if visible or extra is not None:
            cols = self._order(self.visible_cols, visible)
            packed = torch.zeros(sp, 1, ROWS, 32, dtype=torch.float32)
            for j, k in enumerate(cols):
                packed[:, 0, :, j] = visible[k].float()
            if extra is not None:
                packed = torch.cat([packed, extra], dim=3)
            out[self.VISIBLE] = pool.per_rank(packed.contiguous(), ttnn.float32, ttnn.TILE_LAYOUT, device=False)
        return out

    @staticmethod
    def _bf16_bits(t: torch.Tensor) -> torch.Tensor:
        """bf16 values -> their fp32 bit patterns as int32 (the bf16 bits in the high half)."""
        return t.to(torch.bfloat16).contiguous().view(torch.int16).to(torch.int32) << 16

    def _pack_one(self, ints: dict, floats: dict, rope: dict) -> ttnn.Tensor:
        """The "one" mode's host tensor, one row (one page) per rank: int row i at columns ``[i * ROWS, (i + 1) *
        ROWS)``, then the float rows (bf16 bits << 16), then the RoPE block (``[ROWS, 2 * len(rope) * rope_dim]``
        bf16 flattened row-major, bits << 16)."""
        sp, tp = self.sp, self.pool.tp
        rows_i = self._order(self.int_rows, ints)
        rows_f = self._order(self.float_rows, floats)
        cols = self._order(self.rope_cols, rope)
        cat = torch.cat([t for k in cols for t in rope[k]], dim=1)  # [sp * ROWS, 2 * len(cols) * rope_dim] bf16
        width = cat.shape[1]
        if self.rope_width == 0:
            self.rope_width = width
        assert width == self.rope_width, (width, self.rope_width)
        ni, nf = len(rows_i), len(rows_f)
        base = (ni + nf) * ROWS
        if self.one_host is None:
            self.one_host = torch.zeros(sp, 1, 1, base + ROWS * width, dtype=torch.int32)
        host = self.one_host
        assert host.shape[3] == base + ROWS * width, (host.shape, ni, nf, width)
        host[:, 0, 0, : ni * ROWS] = torch.stack([ints[k].to(torch.int32) for k in rows_i], dim=1).view(sp, -1)
        for j, k in enumerate(rows_f):
            host[:, 0, 0, (ni + j) * ROWS : (ni + j + 1) * ROWS] = self._bf16_bits(floats[k]).view(sp, ROWS)
        host[:, 0, 0, base:] = self._bf16_bits(cat).view(sp, ROWS * width)
        parts = []
        for r in range(sp):
            shard = ttnn.Tensor(host[r : r + 1], ttnn.uint32)  # [1, 1, 1, L]: this rank's, on every TP chip
            parts.extend([shard] * tp)
        return ttnn.from_host_shards(parts, ttnn.MeshShape(sp, tp))

    def _unpack_one(self, buffers: dict) -> dict:
        """The "one" mode's device buffer -> the graph's tensors (as ``unpack`` keys them): the int rows sliced,
        the float rows sliced, bitcast to fp32 and typecast to bf16 (exact: the low half is zero), the RoPE block
        sliced, reshaped, bitcast, typecast and tilized, then its pairs sliced."""
        out = {}
        one = buffers[self.ONE]
        ni, nf, width = len(self.int_rows), len(self.float_rows), self.rope_width
        for i, k in enumerate(self.int_rows):
            t = ttnn.slice(one, [0, 0, 0, i * ROWS], [1, 1, 1, (i + 1) * ROWS])
            out[k] = ttnn.reshape(t, (ROWS,)) if k == "k" else t
        for j, k in enumerate(self.float_rows):
            t = ttnn.slice(one, [0, 0, 0, (ni + j) * ROWS], [1, 1, 1, (ni + j + 1) * ROWS])
            out[k] = ttnn.reshape(ttnn.typecast(ttnn.bitcast(t, ttnn.float32), ttnn.bfloat16), (ROWS,))
        base = (ni + nf) * ROWS
        rope = ttnn.reshape(ttnn.slice(one, [0, 0, 0, base], [1, 1, 1, base + ROWS * width]), (1, 1, ROWS, width))
        rope = ttnn.tilize(ttnn.typecast(ttnn.bitcast(rope, ttnn.float32), ttnn.bfloat16))
        pair = width // (2 * len(self.rope_cols))
        for i, k in enumerate(self.rope_cols):
            out[k] = tuple(
                ttnn.slice(rope, [0, 0, 0, (2 * i + j) * pair], [1, 1, ROWS, (2 * i + j + 1) * pair]) for j in range(2)
            )
        self._unpack_visible(buffers, out)
        return out

    def _unpack_visible(self, buffers: dict, out: dict) -> None:
        if self.visible_cols or self.wide_cols:
            visible = buffers[self.VISIBLE]
            for j, k in enumerate(self.visible_cols):
                out[k] = ttnn.slice(visible, [0, 0, 0, j], [1, 1, ROWS, j + 1])
            start = 32
            for k, w in zip(self.wide_cols, self.wide_widths):
                out[k] = ttnn.slice(visible, [0, 0, 0, start], [1, 1, ROWS, start + w])
                start += w

    def unpack(self, buffers: dict) -> dict:
        """The device buffers of ``pack`` -> the graph's tensors by name: the int and float rows as ``[1, 1, 1,
        ROWS]`` (the sampler's as ``[ROWS]``), the RoPE pairs as tiled ``[1, 1, ROWS, rope_dim]``, the visible
        counts as tiled ``[1, 1, ROWS, 1]``."""
        if self.mode == "one":
            return self._unpack_one(buffers)
        out = {}
        ints = buffers[self.INTS]
        for i, k in enumerate(self.int_rows):
            t = ttnn.slice(ints, [0, 0, i, 0], [1, 1, i + 1, ROWS])
            out[k] = ttnn.reshape(t, (ROWS,)) if k == "k" else t
        if self.float_rows:
            floats = buffers[self.FLOATS]
            for i, k in enumerate(self.float_rows):
                out[k] = ttnn.reshape(ttnn.slice(floats, [0, 0, i, 0], [1, 1, i + 1, ROWS]), (ROWS,))
        rope = buffers[self.ROPE]
        width = rope.shape[3] // (2 * len(self.rope_cols))
        for i, k in enumerate(self.rope_cols):
            out[k] = tuple(
                ttnn.slice(rope, [0, 0, 0, (2 * i + j) * width], [1, 1, ROWS, (2 * i + j + 1) * width])
                for j in range(2)
            )
        self._unpack_visible(buffers, out)
        return out


class DeviceSampler:
    """The step's sampling on device: each TP chip's top-32 of its vocabulary shard, gathered over TP into every
    row's top-128 candidates, ``ttnn.sampling`` with per-row top-k (1..32), top-p and temperature, the sampled ids
    gathered over SP. The cores' random generators are seeded once (``ttnn.manual_seed``; random seeds, or the
    per-row seeds derived from ``seed`` / ``V41_SAMPLE_SEED`` so that two processes draw the same stream); the captured sampling (seed None) leaves them running, so replays draw fresh numbers.
    The seeds are replicated over the mesh: the same row index on every rank draws the same uniform."""

    SEED = "V41_SAMPLE_SEED"  # an integer: reproducible draws (unset or 0: random seeds)

    def __init__(self, mesh_device, config, ccl, gather_sp, seed: int | None = None):
        self.mesh_device, self.ccl, self.gather_sp = mesh_device, ccl, gather_sp
        self.seed = int(os.environ.get(self.SEED, "0")) if seed is None else seed
        self.sp, self.tp = tuple(mesh_device.shape)
        per_chip = config.VOCAB_SIZE // self.tp
        offsets = torch.arange(self.tp, dtype=torch.int32).view(self.tp, 1, 1, 1).expand(self.tp, 1, ROWS, 32)
        self.vocab_offsets = ttnn.from_torch(
            (offsets * per_chip).contiguous(),
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, (self.sp, self.tp), dims=(None, 0)),
        )
        self.seeded = False
        self.debug = None  # a dict: the last call keeps clones of its logits, candidates and ids (tests)

    def host_inputs(self, k: torch.Tensor, p: torch.Tensor, temp: torch.Tensor) -> dict:
        """Per-row (rank-major, ``sp * ROWS``) top-k (1..32), top-p and inverse temperature -> the host tensors
        of ``__call__``'s buffers (``k``, ``p``, ``temp``: one ``[ROWS]`` row-major tensor per rank)."""
        shard = ttnn.ShardTensor2dMesh(self.mesh_device, (self.sp, self.tp), dims=(0, None))
        return {
            name: ttnn.from_torch(host, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=shard)
            for name, host, dtype in (("k", k, ttnn.uint32), ("p", p, ttnn.bfloat16), ("temp", temp, ttnn.bfloat16))
        }

    def candidates(self, logits):
        """``logits`` ``[1, 1, ROWS, vocab/tp]`` fp32 tiled per chip (consumed) -> (values ``[1, 1, ROWS, 128]``
        bf16 tiled, indices ``[1, 1, ROWS, 128]`` int32 row-major): every row's top-128 candidates, the same on the
        TP chips of a rank."""
        values, indices = ttnn.topk(ttnn.typecast(logits, ttnn.bfloat16), 32, dim=-1, largest=True, sorted=True)
        ttnn.deallocate(logits)
        indices = ttnn.add(ttnn.typecast(indices, ttnn.int32), self.vocab_offsets)
        values = self.ccl.tp_all_gather(values, dim=3)
        indices = ttnn.to_layout(self.ccl.tp_all_gather(indices, dim=3), ttnn.ROW_MAJOR_LAYOUT)
        return values, indices

    def __call__(self, logits, buffers: dict):
        """``logits`` ``[1, 1, ROWS, vocab/tp]`` fp32 tiled per chip (consumed) -> the sampled ids of every rank's
        rows on every chip (``to_host`` reads them)."""
        return self.gather(self.sample(logits, buffers))

    def sample(self, logits, buffers: dict):
        """``logits`` (consumed) -> this rank's sampled ids ``[1, 1, 1, ROWS]`` uint32 row-major (the same on the
        rank's TP chips)."""
        if self.debug is not None:
            self.debug["logits"] = ttnn.clone(logits)
        values, indices = self.candidates(logits)
        if self.debug is not None:
            self.debug["values"], self.debug["indices"] = values, indices
        if not self.seeded:
            if self.seed:
                seeds = torch.randint(
                    1, 2**31 - 1, (ROWS,), dtype=torch.int32, generator=torch.Generator().manual_seed(self.seed)
                )
            else:
                seeds = torch.randint(1, 2**31 - 1, (ROWS,), dtype=torch.int32)
            mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)
            ttnn.manual_seed(
                seeds=ttnn.from_torch(seeds, device=self.mesh_device, dtype=ttnn.uint32, mesh_mapper=mapper),
                user_ids=ttnn.from_torch(
                    torch.arange(ROWS, dtype=torch.int32),
                    device=self.mesh_device,
                    dtype=ttnn.uint32,
                    mesh_mapper=mapper,
                ),
            )
            self.seeded = True
        ids = ttnn.sampling(values, indices, buffers["k"], buffers["p"], buffers["temp"])
        if self.debug is not None:
            self.debug["ids"] = ids
        return ids

    def gather(self, row):
        """A rank's row ``[1, 1, 1, W]`` uint32 row-major (``W`` a multiple of 32) -> one tile per rank gathered over
        SP: ``[1, 1, sp * 32, W]`` uint32 tiled, rows ``[r * 32, (r + 1) * 32)`` rank r's row, every row the same
        (the 128-byte row of ids gathered by all_gather_async arrived on the other ranks' chips as stale memory)."""
        tile = ttnn.to_layout(ttnn.concat([row] * ROWS, dim=2), ttnn.TILE_LAYOUT)
        return self.gather_sp(tile, dim=2)

    @staticmethod
    def to_host(gathered) -> torch.Tensor:
        """``__call__``'s result -> ``[sp * ROWS]`` int64 ids, ``r * ROWS + u`` rank r's row u."""
        return DeviceSampler.rows_to_host(gathered).reshape(-1)

    @staticmethod
    def rows_to_host(gathered) -> torch.Tensor:
        """``gather``'s result -> ``[sp, W]`` int64: rank r's row."""
        t = ttnn.to_torch(ttnn.get_device_tensors(gathered)[0])[0, 0].to(torch.int64)  # [sp * 32, W]
        return t.view(-1, ROWS, t.shape[1])[:, 0, :]


class TtV41BatchDecoder(TtV41Decoder):
    POOL_RING_SLOTS = RING  # the pool's ring slots per layer (a subclass with verify rows takes more)
    C1_ROWS = 1  # the most live rows a c1 MoE call takes (the verify rows' subclass: 1 + K)

    def __init__(
        self,
        model: TtV41Transformer,
        weight_cache_path,
        users_per_rank: int = 4,
        routed_expert_weights_dtype=None,
        trace: bool = False,
        warmup: int = 2,
        max_seq_len: int | None = None,
        sample_on_device: bool = False,
    ):
        """``users_per_rank`` slots per mesh row (``sp * users_per_rank`` users); ``max_seq_len`` of the pool
        (default the model's). ``trace``: capture the step once after ``warmup`` eager steps and replay it.
        ``sample_on_device``: the step ends in ``ttnn.sampling`` over each row's top-128 logits (the four TP chips'
        top-32 each) with per-row top-k, top-p and temperature, and ``decode`` returns token ids instead of logits
        (``sampling`` keyword: ``{slot: (k, p, temperature)}``; a temperature of 0 is argmax: k 1)."""
        self.users = users_per_rank  # before the decoder builds the MoEs: their decode routing takes the real rows
        super().__init__(model, weight_cache_path, routed_expert_weights_dtype, trace=trace, warmup=warmup)
        self.pool = V41DecodePool(
            self.mesh_device,
            self.config,
            max_seq_len or model.max_seq_len,
            model.layers,
            users_per_rank,
            model.kv_format,
            ring_slots=self.POOL_RING_SLOTS,
        )
        self.slots: dict[int, _User] = {}
        # the MoE's per-rank [real rows, right padding]: the slots' rows of every rank
        self.padding_config = (
            None
            if os.environ.get("V41_BATCH_MOE_PAD", "1") == "0"
            else ttnn.from_torch(
                torch.tensor([[users_per_rank, 0]] * self.sp, dtype=torch.int32),
                device=self.mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(0, None), mesh_shape=self.mesh_device.shape),
            )
        )
        # per ratio: fp32 column indices [1, 1, ROWS, cap + 32] of an index source's score (the mask compares them
        # with the step's per-row visible counts); the sentinel tail of the pad rows' selection
        self.iota = {}
        for src, slot in self.pool.slots.items():
            if src < 0:
                continue
            ratio = self.config.compress_ratio(src)
            if ratio not in self.iota:
                cols = torch.arange(slot.cap + 32, dtype=torch.float32).view(1, 1, 1, -1).expand(1, 1, ROWS, -1)
                self.iota[ratio] = self._replicated(cols.contiguous(), ttnn.float32)
        self.pad_topk = self._replicated(
            torch.full((1, 1, ROWS - self.users, self.config.INDEX_TOPK), -1, dtype=torch.int32),
            ttnn.uint32,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        self.sample_on_device = sample_on_device
        # the single-query scorer: every densely scored source scores one user's query against
        # the visible prefix of its tiled pool (range_scores, the bound read on device, the keys split over the cores
        # as rows of a split layout) and selects by merging the rows' top-k (select_split); no 32-row dense score,
        # no mask round-trip, no per-step mask build
        self.single_query = os.environ.get("V41_DECODE_SINGLE_QUERY", "0") == "1"
        # V41_DECODE_SDPA=local: the SDPA of both captures on the chip's own heads;
        # ``flash``: those heads through the flash-decode program (its constants allocated
        # here, before any capture; one merge slot a core: the c1 capture's live rows are 1 + K <= 8)
        self.sdpa_local = attention_module._SDPA in ("local", "flash")
        self.flash = None
        if attention_module._SDPA == "flash":
            attn0 = model.blocks[0].attn
            self.flash = FlashSdpa(
                self.mesh_device,
                heads=attn0.sdpa_heads,
                rows=ROWS,
                width=RING + self.config.INDEX_TOPK,
                v_dim=attn0.head_dim,
                real_heads=attn0.local_heads,
                max_live=8,
            )
        self.sampler = (
            DeviceSampler(self.mesh_device, self.config, model.blocks[0].attn.ccl, self.pool.gather_sp)
            if sample_on_device
            else None
        )
        # the step's captures by variant: "full" (the 32-row tile), and with V41_DECODE_C1_MOE=1 "c1" (one live
        # row: the MoEs on that row broadcast to every chip); each variant runs ``warmup``
        # eager steps before its capture (a capture cannot load binaries)
        self.captures: dict[str, object] = {}
        self.eager_counts: dict[str, int] = {}
        self.last_variant = "full"
        self.c1 = os.environ.get("V41_DECODE_C1_MOE", "0") == "1"
        if self.c1:
            assert MOE_ROWS == ROWS, "V41_DECODE_C1_MOE=1 needs V41_DECODE_ROUTING=1 (the 32-row MoE tile)"
            ccl = model.blocks[0].attn.ccl
            self.c1_moes = [C1Moe(moe, ccl, self.pool, rows=self.C1_ROWS) for moe in self.moes]
            self.c1_table = C1Moe.identity_table(self.mesh_device)
        # V41_DECODE_C1_INDEX=1: in the c1 capture the index sources select for the live
        # user alone (``_index_c1``): its user index, newest row and block page are per-step host rows (``c1usr``,
        # ``c1pin`` per ratio, ``c1all``), read on device by the scorers, so the capture carries no slot; the
        # selection is placed into the tile's row by the ``c1put`` page (``c1_place``), the other rows sentinels
        self.c1_index = os.environ.get("V41_DECODE_C1_INDEX", "0") == "1"
        if self.c1_index:
            assert (
                self.c1 and self.single_query
            ), "V41_DECODE_C1_INDEX=1 needs V41_DECODE_C1_MOE=1 and V41_DECODE_SINGLE_QUERY=1"
            self.c1_caps: dict[int, int] = {}
            for src, slot in self.pool.slots.items():
                if src < 0:
                    continue
                ratio = self.config.compress_ratio(src)
                assert self.c1_caps.setdefault(ratio, slot.cap) == slot.cap, (ratio, slot.cap, self.c1_caps)
        # V41_DECODE_NORM=sharded: the sublayer norms of the decode tile on a core grid
        # instead of the interleaved ops' one core (DecodeRmsNorm; the prefill keeps the block's norms)
        self.decode_norms: dict[int, tuple] = {}
        if DECODE_NORM == "sharded":
            ccl = model.blocks[0].attn.ccl
            width = self.config.EMB_SIZE // self.tp
            for layer, block in zip(model.layers, model.blocks, strict=False):
                self.decode_norms[layer] = (
                    DecodeRmsNorm(block.attn_norm, width, ccl.tp_all_gather),
                    DecodeRmsNorm(block.ffn_norm, width, ccl.tp_all_gather),
                )
        # V41_DECODE_HOST_INPUTS=packed: the RoPE rows from host tables by position, the
        # Engram rows gathered for the chips that hold a live row only, the small inputs packed into four
        # uploads (StepPacker) and sliced apart in the graph; the default builds and uploads every input apart
        self.host_inputs_mode = os.environ.get("V41_DECODE_HOST_INPUTS", "")
        assert self.host_inputs_mode in ("", "packed", "one"), self.host_inputs_mode
        self.packer = StepPacker(self.pool, self.sp, self.host_inputs_mode) if self.host_inputs_mode else None
        # V41_DECODE_REPLAY_SYNC=0: no device sync between the staging and the replay (the
        # same command queue orders them; the sync was a round trip for the timing split's sake)
        self.replay_sync = os.environ.get("V41_DECODE_REPLAY_SYNC", "1") == "1"
        self.next_histories: dict[int, torch.Tensor] = {}  # slot -> its Engram history after this step's token
        self.rope_tables = RopeTables(self.config, self.pool.max_seq_len) if self.packer else None
        self.inputs = self.buffers  # what the graph reads: the buffers, or their unpacked slices (``_forward``)
        # a join's compressed index is as wide as the prompt's bucket, not the pool: at a 1M pool
        # the full-width index was ~200 MB of host sharding and copies per join, 65 ms on the path
        # between two prefills at c32; captures are per (state, slot, width)
        self.join_widths = []
        w = JOIN_MIN_WIDTH
        while w < self.pool.max_seq_len:
            self.join_widths.append(w)
            w *= 2
        self.join_widths.append(self.pool.max_seq_len)
        self.join_traces: dict[tuple[int, int, int], SubDeviceTraceController] = {}
        self.join_buffers: dict[int, dict] = {}  # width -> the index buffers
        self.last_join: dict = {}

    @property
    def capture(self):
        """The 32-row step's capture (None until captured)."""
        return self.captures.get("full")

    @property
    def eager_count(self) -> int:
        return self.eager_counts.get("full", 0)

    @eager_count.setter
    def eager_count(self, value: int) -> None:
        self.eager_counts["full"] = value

    def variant(self, tokens: dict) -> str:
        """The step variant for ``tokens``: "c1" with the c1 MoE on and exactly one slot stepping, else "full"."""
        return "c1" if self.c1 and len(tokens) == 1 else "full"

    # --- users -------------------------------------------------------------------------------------------------
    @property
    def num_slots(self) -> int:
        return self.sp * self.users

    def join(self, slot: int, tokens: torch.Tensor, state: V41PrefillState) -> None:
        """Slot ``slot`` takes over ``state`` after ``prefill(tokens)`` filled it: its caches move into the pool
        (``V41DecodePool.join_ops``; replayed when ``capture_join`` captured this state and slot, else eager), its
        Engram history is seeded from the last compressed ids."""
        assert 0 <= slot < self.num_slots, slot
        assert state.start >= self.config.SLIDING_WINDOW - 1, f"a prompt of {state.start}: decode needs a full window"
        user = _User(state.start)
        if self.model.engram:
            hash_ = self.model.engram_hash
            n = hash_.layout.max_ngram_size - 1
            user.engram_history = hash_.token_map[tokens[-n:].to(torch.int64)].clone()
        t0 = time.time()
        width = self.join_width(state.start)
        inputs = self.pool.join_inputs(slot, state, width)
        trace = self.join_traces.get((id(state), slot, width))
        if trace is None:
            idx = {k: ttnn.to_device(v, self.mesh_device) for k, v in inputs.items()}
            self.pool.join_ops(slot, state, idx)
            for t in idx.values():
                ttnn.deallocate(t)
            mode = "eager"
        else:
            self._stage_join(inputs, width)
            trace.replay(blocking=False)
            mode = "replay"
        self.last_join = {"mode": mode, "s": round(time.time() - t0, 4), "width": width}
        self.slots[slot] = user

    def join_width(self, start: int) -> int:
        """The join index width (tokens) of a prompt of ``start`` tokens: the smallest of ``join_widths`` at or
        above it."""
        for w in self.join_widths:
            if start <= w:
                return w
        raise AssertionError(f"a prompt of {start} tokens beyond the pool's {self.pool.max_seq_len}")

    def _stage_join(self, inputs: dict, width: int) -> None:
        buffers = self.join_buffers.get(width)
        if buffers is None:
            self.join_buffers[width] = {k: ttnn.to_device(v, self.mesh_device) for k, v in inputs.items()}
            return
        for k, v in inputs.items():
            ttnn.copy_host_to_device_tensor(v, buffers[k])

    def capture_join(self, state: V41PrefillState, slot: int, width: int | None = None) -> None:
        """Capture the join of ``state`` into ``slot`` at index ``width`` (default: every width of
        ``join_widths``; their tensors are persistent, the index buffers are uploaded per join), so later joins
        of the triple replay it: a join is ~60 small ops, 20 to 80 ms issued eagerly."""
        if width is None:
            for w in self.join_widths:
                self.capture_join(state, slot, w)
            return
        key = (id(state), slot, width)
        if key in self.join_traces:
            return
        self._stage_join(self.pool.join_inputs(slot, state, width), width)
        # each width's scatter is its own program (the row count is a runtime argument of a generic op): compile
        # it with one eager join before the capture, which cannot load binaries
        self.pool.join_ops(slot, state, self.join_buffers[width])
        controller = SubDeviceTraceController(self.mesh_device)
        ttnn.synchronize_device(self.mesh_device)
        controller.begin_capture()
        try:
            self.pool.join_ops(slot, state, self.join_buffers[width])
        finally:
            controller.end_capture()
        self.join_traces[key] = controller

    def leave(self, slot: int) -> None:
        self.slots.pop(slot, None)

    def position(self, slot: int) -> int:
        return self.slots[slot].position

    # --- host inputs ---------------------------------------------------------------------------------------------
    def _host_inputs(self, tokens: dict[int, int], sampling: dict | None = None) -> dict:
        """The step's host tensors (per rank, device layouts), keyed as the device buffers: ``tokens`` slot ->
        token of the slots that step (the others get zeros and write nothing); ``sampling`` slot -> (k, p,
        temperature) when sampling on device (other slots: argmax). Under ``V41_DECODE_HOST_INPUTS=packed`` the
        keys are the ``StepPacker``'s (plus ``engram``) and ``_forward`` unpacks them."""
        cfg, sp, users = self.config, self.sp, self.users
        tick = self._tick("in")
        packed = self.packer is not None
        step = self.pool.step
        ids = torch.zeros(sp, 1, ROWS, dtype=torch.int32)
        positions = torch.zeros(sp, ROWS, dtype=torch.int64)
        active = torch.zeros(sp, ROWS, dtype=torch.bool)
        for slot, token in tokens.items():
            r, u = divmod(slot, users)
            ids[r, 0, u] = token
            positions[r, u] = self.slots[slot].position
            active[r, u] = True
        tick("rows")
        ints, floats, rope, visible_cols = {"token": ids.view(sp, ROWS)}, {}, {}, {}
        inputs = {} if packed else {"token": self.pool.per_rank(ids, ttnn.uint32, device=False)}
        tick("token")
        if self.c1:
            live = divmod(next(iter(tokens)), users) if len(tokens) == 1 else None
            for k, v in C1Moe.host_rows(sp, ROWS, live).items():
                if packed:
                    ints[k] = v
                else:
                    inputs[k] = self.pool.per_rank(v.view(sp, 1, 1, ROWS), ttnn.uint32, device=False)
            if self.c1_index:
                # every word the live user's index on the live rank (the one-hot of every row: the live query
                # repeated over the tile), the identity table's zero row elsewhere
                all_ = torch.full((sp, ROWS), C1_ONEHOT_SENTINEL, dtype=torch.int32)
                if live is not None:
                    all_[live[0], :] = live[1]
                if packed:
                    ints["c1all"] = all_
                else:
                    inputs["c1all"] = self.pool.per_rank(all_.view(sp, 1, 1, ROWS), ttnn.uint32, device=False)
            tick("c1")
        if self.sample_on_device:
            k = torch.ones(sp * ROWS, dtype=torch.int32)
            p = torch.zeros(sp * ROWS, dtype=torch.float32)
            temp = torch.ones(sp * ROWS, dtype=torch.float32)
            for slot, (sk, sp_, st) in (sampling or {}).items():
                r, u = divmod(slot, users)
                i = r * ROWS + u
                if st <= 0:
                    continue  # argmax
                k[i], p[i], temp[i] = max(1, min(int(sk), 32)) if sk > 0 else 32, float(sp_), 1.0 / float(st)
            if packed:
                ints["k"], floats["p"], floats["temp"] = k.view(sp, ROWS), p.view(sp, ROWS), temp.view(sp, ROWS)
            else:
                inputs.update(self.sampler.host_inputs(k, p, temp))
            tick("sampling")

        def rope_rows(compressed: bool, pos: torch.Tensor):
            if packed:
                out = self.rope_tables.rows(compressed, pos)
                tick("rope_cos_sin")
                return out
            out = []
            tables = cos_sin(cfg, compressed, pos.flatten())
            tick("rope_cos_sin")
            for t in tables:
                out.append(self.pool.per_rank(t.view(sp, 1, ROWS, -1), ttnn.bfloat16, ttnn.TILE_LAYOUT, device=False))
            tick("rope_tensors")
            return tuple(out)

        rope["rope0"], rope["ropec"] = rope_rows(False, positions), rope_rows(True, positions)
        rope["ropeg"] = rope_rows(True, positions // 2 * 2)
        if not packed:
            inputs.update(rope)
        ring = torch.full((sp, ROWS), step % RING, dtype=torch.int32)
        if packed:
            ints["ring"] = ring
        else:
            inputs["ring"] = self._replicated(
                ring[:1].view(1, 1, 1, ROWS), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device=False
            )
        tick("ring")
        for ratio in sorted(self.iota):
            visible = torch.where(active, (positions + 1) // ratio, torch.zeros_like(positions)).float()
            # the selection's per-row pins: each row's newest compressed row, the sentinel for an idle or empty row
            newest = torch.where(visible > 0, visible - 1, torch.full_like(visible, -1)).to(torch.int32)
            writes = active & (positions % ratio == ratio - 1)
            comp = torch.where(writes, positions // ratio, torch.full_like(positions, -1)).to(
                torch.int32
            )  # -1: sentinel
            if self.c1_index:
                # the c1 capture's index pages: ``c1usr`` word 0 the live user (0 when none: row 0 is read, nothing
                # is visible), word 1 the page of its block 0, word 2 its newest row (the sentinel for none);
                # ``c1pin`` its newest in every word (the candidate scorer's per-row page)
                usr = torch.full((sp, ROWS), -1, dtype=torch.int32)
                usr[:, :2] = 0
                pin = torch.full((sp, ROWS), -1, dtype=torch.int32)
                if live is not None:
                    r, u = live
                    rows_u = self.c1_caps[ratio] + 32
                    usr[r, 0], usr[r, 1], usr[r, 2] = u, u * rows_u // cfg.CANDIDATE_BLOCK_SIZE, int(newest[r, u])
                    pin[r, :] = int(newest[r, u])
                if packed:
                    ints[f"c1usr{ratio}"], ints[f"c1pin{ratio}"] = usr, pin
                else:
                    inputs[f"c1usr{ratio}"] = self.pool.per_rank(usr.view(sp, 1, 1, ROWS), ttnn.uint32, device=False)
                    inputs[f"c1pin{ratio}"] = self.pool.per_rank(pin.view(sp, 1, 1, ROWS), ttnn.uint32, device=False)
            if packed:
                ints[f"newest{ratio}"], ints[f"comp{ratio}"] = newest, comp
                if not self.single_query:  # the masks are the only reader
                    visible_cols[f"visible{ratio}"] = visible
            else:
                inputs[f"visible{ratio}"] = self.pool.per_rank(
                    visible.view(sp, 1, ROWS, 1), ttnn.float32, ttnn.TILE_LAYOUT, device=False
                )
                inputs[f"newest{ratio}"] = self.pool.per_rank(newest.view(sp, 1, 1, ROWS), ttnn.uint32, device=False)
                inputs[f"comp{ratio}"] = self.pool.per_rank(comp.view(sp, 1, 1, ROWS), ttnn.uint32, device=False)
            tick(f"ratio{ratio}")
        if packed:
            inputs.update(self.packer.pack(ints, floats, rope, visible_cols))
            tick("pack")
        if self.model.engram:
            hash_ = self.model.engram_hash
            order = list(tokens)
            if order:
                ids_batch, histories = hash_.step(
                    torch.tensor([tokens[s] for s in order]),
                    torch.stack([self.slots[s].engram_history for s in order]),
                )  # [U, n_layers, cols]; the histories after these tokens (``decode`` keeps them once the step ran)
                self.next_histories = {slot: histories[k] for k, slot in enumerate(order)}
            tick("engram_hash")
            inputs["engram"] = {}
            # packed: the rows of the chips that hold a live row only; the others read the all-zero shard (their
            # rows are idle: nothing reads what they compute)
            live = active.flatten().nonzero().flatten() if packed else None
            for l, e in self.model.engram.items():
                ids = torch.zeros(sp * ROWS, hash_.n_hash_cols, dtype=torch.int64)
                for k, slot in enumerate(order):
                    r, u = divmod(slot, users)
                    ids[r * ROWS + u] = ids_batch[k, hash_.layer_index(l)]
                tick(f"engram{l}_ids")
                inputs["engram"][l] = e.prepare_host(ids, sp * ROWS, live_rows=live)
                tick(f"engram{l}_rows")
        return inputs

    # --- one step ----------------------------------------------------------------------------------------------
    def decode(self, tokens: dict[int, int], sampling: dict | None = None) -> dict:
        """Append ``tokens`` (slot -> token) at the slots' positions -> slot -> fp32 logits ``[vocab]`` of the next
        position (or, sampling on device, slot -> the sampled token id); the slots advance by one, the pool's step
        too."""
        for slot in tokens:
            user = self.slots[slot]
            assert user.position < self.pool.max_seq_len, f"slot {slot} at {user.position}: the pool is full"
        self.items = {}
        t0 = time.time()
        inputs = self._host_inputs(tokens, sampling)
        t1 = time.time()
        self._stage(inputs)
        t2 = time.time()
        variant = self.variant(tokens)
        self.last_variant = variant
        c1 = variant == "c1"
        if not self.trace:
            mode = "eager"
        elif variant in self.captures:
            mode = "replay"
        elif self.eager_counts.get(variant, 0) >= self.warmup:
            mode = "capture"
        else:
            mode = "eager"
        self.last = {"mode": mode, "inputs_s": round(t1 - t0, 4), "stage_s": round(t2 - t1, 4), "variant": variant}
        if self.input_timing:
            self.last["items"] = dict(self.items)
        if mode == "eager":
            self.eager_counts[variant] = self.eager_counts.get(variant, 0) + 1
            logits = self._forward(self.on_block, c1=c1)
        else:
            if mode == "capture":
                self.captures[variant] = self._capture_step(c1=c1)
            capture = self.captures[variant]
            if self.replay_sync:
                ttnn.synchronize_device(self.mesh_device)
            t3 = time.time()
            capture.controller.replay(blocking=False)
            if self.stage_probe == "pre":  # the probe copy queued behind the replay
                self.probe_copy()
            t4 = time.time()
            ttnn.synchronize_device(self.mesh_device)
            t5 = time.time()
            logits = capture.logits
            self.last.update(sync_s=round(t3 - t2, 4), device_s=round(t5 - t3, 4), issue_s=round(t4 - t3, 4))
            if self.on_block is not None:
                for layer, x in capture.blocks:
                    self.on_block(layer, x)
        t6 = time.time()
        out = self._tokens_to_host(logits, tokens) if self.sample_on_device else self._logits_to_host(logits, tokens)
        if mode == "eager":
            ttnn.deallocate(logits)
        self.last["read_s"] = round(time.time() - t6, 4)
        if self.model.engram and tokens:
            # the histories ``_host_inputs`` computed with the step's ids (one hash a step, not two)
            assert set(self.next_histories) == set(tokens), (set(self.next_histories), set(tokens))
            for slot, history in self.next_histories.items():
                self.slots[slot].engram_history = history
            self.next_histories = {}
        for slot in tokens:
            self.slots[slot].position += 1
        self.pool.step += 1
        return out

    def _tokens_to_host(self, ids, tokens: dict[int, int]) -> dict[int, int]:
        """The gathered sampled ids -> slot -> token id."""
        row = DeviceSampler.to_host(ids)
        return {slot: int(row[(slot // self.users) * ROWS + slot % self.users]) for slot in tokens}

    def _logits_to_host(self, logits, tokens: dict[int, int]) -> dict[int, torch.Tensor]:
        """The gathered ``[1, 1, slots, vocab]`` fp32 logits (the same on every chip) -> slot -> ``[vocab]``."""
        rows = ttnn.to_torch(ttnn.get_device_tensors(logits)[0])[0, 0].float()
        return {slot: rows[slot] for slot in tokens}

    def replay_device_time(self, repeats: int = 3) -> dict:
        """Device time of one replay of the last stepped variant's capture without host work -> {"step": min
        seconds}."""
        capture = self.captures.get(self.last_variant)
        if capture is None:
            return {}
        ttnn.synchronize_device(self.mesh_device)
        best = None
        for _ in range(repeats):
            t0 = time.time()
            capture.controller.replay(blocking=False)
            ttnn.synchronize_device(self.mesh_device)
            dt = time.time() - t0
            best = dt if best is None else min(best, dt)
        return {"step": round(best, 4)}

    def _capture_step(self, c1: bool = False):
        from models.demos.deepseek_v3_d_p.tt.v41.decode import _Trace

        controller = SubDeviceTraceController(self.mesh_device)
        for moe in self.moes:
            moe.set_trace_controller(controller)
        blocks = []
        ttnn.synchronize_device(self.mesh_device)
        controller.begin_capture()
        try:
            logits = self._forward(lambda layer, x: blocks.append((layer, x)), c1=c1)
        finally:
            controller.end_capture()
            for moe in self.moes:
                moe.set_trace_controller(None)
        return _Trace(controller, logits, 0, blocks)

    def release(self) -> None:
        for capture in self.captures.values():
            capture.controller.release()
        self.captures = {}
        for controller in self.join_traces.values():
            controller.release()
        self.join_traces = {}

    @property
    def trace_segments(self) -> int:
        capture = self.captures.get(self.last_variant)
        return capture.controller.num_segments if capture is not None else 0

    def _forward(self, on_block, c1: bool = False):
        """The step's device graph on the staged buffers -> the logits tile ``[1, 1, ROWS, vocab/tp]`` per chip
        (row u: slot u of the chip's rank); the pool is updated in place. ``c1``: the MoEs on the one live row
        (``C1Moe``; the idle rows' MoE output zero)."""
        model = self.model
        if self.packer is not None:
            self.inputs = dict(self.packer.unpack(self.buffers), engram=self.buffers.get("engram", {}))
            self.inputs["token"] = ttnn.reshape(self.inputs["token"], (1, 1, ROWS))
        else:
            self.inputs = self.buffers
        self._neg_sin = {}  # rope key -> -sin, computed once a step (V41_DECODE_NEG_SIN=once)
        masks = {ratio: None if self.single_query else self._mask(ratio) for ratio in self.iota}
        selection = {}
        h = model.embedding(self.inputs["token"])
        x = mhc_expand(ttnn.typecast(h, ttnn.float32), self.config.HC_MULT)
        pre = self.pre_mix
        if c1:
            sel, put = C1Moe.onehots(self.inputs["c1sel"], self.inputs["c1put"], self.c1_table)
            moes = self.c1_moes
            if self.c1_index:
                sel_all, _ = C1Moe.onehots(self.inputs["c1all"], self.inputs["c1all"], self.c1_table)
                selection["c1"] = sel_all
            if self.flash is not None:
                # the flash SDPA's live page: ``c1put`` (word u 0 on the live rank's live row, the one-hot sentinel
                # elsewhere); the 32-row capture keeps sparse_sdpa
                selection["flash_live"] = self.inputs["c1put"]
        else:
            moes = self.moes
        for layer, block, moe in zip(model.layers, model.blocks, moes):
            if layer in model.engram:
                x = model.engram[layer](x, self.inputs["engram"][layer])
            attn_norm, ffn_norm = self.decode_norms.get(layer, (block.attn_norm, block.ffn_norm))
            x, pre = block.residual(
                x,
                pre,
                attention=lambda h, a=block.attn, n=attn_norm: self._attention(a, n(h), masks, selection),
                ffn=(
                    (lambda h, m=moe, n=ffn_norm: m(n(h), sel, put))
                    if c1
                    else (lambda h, m=moe, n=ffn_norm: self._ffn(m, n(h)))
                ),
            )
            if on_block is not None:
                on_block(layer, x)
        final = ttnn.typecast(model.blocks[-1].residual.final_collapse(x, pre), ttnn.bfloat16)
        logits, _ = model.head(final, 0)  # [1, 1, ROWS, vocab/tp] fp32 per chip, row u slot u of the chip's rank
        ccl = model.blocks[0].attn.ccl
        if self.sample_on_device:
            return self.sampler(logits, self.inputs)  # the slots' sampled ids (DeviceSampler.to_host)
        # the slots' rows, gathered over TP (the vocabulary) and SP (the ranks): one [1, 1, slots, vocab] read
        rows = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(logits)
        if self.users < ROWS:
            part = ttnn.slice(rows, [0, 0, 0, 0], [1, 1, self.users, rows.shape[3]])
            ttnn.deallocate(rows)
            rows = part
        # the ranks first (pages of vocab/tp fp32; a full-vocabulary row is beyond L1 as an input page)
        rows = self.pool.gather_sp(rows, dim=2)
        return ccl.tp_all_gather(rows, dim=3)

    def _mask(self, ratio: int):
        """Additive ``[1, 1, ROWS, cap + 32]`` bf16: row u -inf at the columns at or past its visible count."""
        beyond = ttnn.ge(self.iota[ratio], self.inputs[f"visible{ratio}"])
        return ttnn.typecast(ttnn.where(beyond, NEG_INF, 0.0), ttnn.bfloat16)

    def moe_real_rows(self) -> int:
        """The slots' rows of the MoE tile: the first ``users_per_rank`` of every chip (``padding_config``)."""
        return self.users

    def _ffn(self, moe: TtV41Moe, h):
        """The decode MoE on the tile repeated to MOE_ROWS rows, the slots' rows real -> ROWS rows."""
        x = h if MOE_ROWS == ROWS else ttnn.concat([h] * (MOE_ROWS // ROWS), dim=2)
        if self.padding_config is None:
            out = moe(x)
        else:
            out = moe(x, actual_isl=self.users, padding_config=self.padding_config)
        return out if MOE_ROWS == ROWS else ttnn.slice(out, [0, 0, 0, 0], [1, 1, ROWS, out.shape[3]])

    # --- attention ------------------------------------------------------------------------------------------------
    def _attention(self, attn: TtV41Attention, x, masks: dict, selection: dict):
        """``TtV41Attention.forward`` for the tile of users (``x`` [1, 1, ROWS, hidden/tp] bf16)."""
        layer, pool, cfg = attn.layer, self.pool, self.config
        rope_key = "ropec" if attn.ratio else "rope0"
        cos, sin = self.inputs[rope_key]
        xq = fp8_qdq(x)
        qr = ttnn.rms_norm(attn.ccl.tp_all_reduce(attn._dense(xq, "wq_a")), weight=attn.q_norm, epsilon=attn.eps)
        q = attn._dense(fp8_qdq(qr), "wq_b")  # [1, 1, ROWS, H/tp * head_dim]
        kv = ttnn.rms_norm(attn.ccl.tp_all_reduce(attn._dense(xq, "wkv")), weight=attn.kv_norm, epsilon=attn.eps)
        kv_format = pool.kv_format if attn.ratio else MlaKvCacheFormat.BF16_RM
        reference_values = kv_format == MlaKvCacheFormat.BF16_RM
        kv = attn._rope(kv, cos, sin)
        kv_tensor = pool.kv_tensor(layer)
        rows = self._encode(fp8_qdq(kv) if reference_values else kv, kv_format)
        scatter_rows(
            rows,
            kv_tensor,
            self.inputs["ring"],
            dst_base=pool.ring_row(layer),
            stride=pool.slot_rows(layer),
            rows=self.users,
        )
        ttnn.deallocate(rows)

        compressed = None
        if attn.ratio:
            ratio, src = attn.ratio, cfg.kv_source(layer)
            if attn.is_kv_source:
                latent = self._compress(attn.compressor, x, layer)
                c_rope = self.inputs["ropeg" if ratio == 2 else "ropec"]  # the group's first position
                keys = attn.index_keys(latent, c_rope)
                comp_kv = attn._rope(latent, *c_rope)
                rows = self._encode(fp4_e4m3_qdq(comp_kv) if reference_values else comp_kv, kv_format)
                idx, n = self.inputs[f"comp{ratio}"], self.users
                scatter_rows(rows, pool.kv[src], idx, dst_base=pool.comp_row(src), stride=pool.slot_rows(layer), rows=n)
                ttnn.deallocate(rows)
                keys = ttnn.to_layout(keys, ttnn.ROW_MAJOR_LAYOUT)
                pool.write_index_keys(src, keys, idx, stride=pool.slots[src].cap + 32, rows=n)
                ttnn.deallocate(keys)
            if attn.is_index_source and "c1" in selection:
                compressed, candidates = self._index_c1(
                    attn.indexer, x, qr, src, selection.get("candidates"), selection["c1"]
                )
                selection["topk"] = compressed
                if candidates is not None:
                    selection["candidates"] = candidates
            elif attn.is_index_source:
                compressed, candidates = self._index(
                    attn.indexer, x, qr, src, masks[ratio], selection.get("candidates")
                )
                selection["topk"] = compressed
                if candidates is not None:
                    selection["candidates"] = candidates
            else:
                compressed = selection["topk"]
        rows = self._index_rows(layer, compressed)

        k_chunk = next(c for c in (128, 64, 32) if rows.shape[-1] % c == 0)
        if self.sdpa_local and self.tp > 1:
            # this chip's heads alone (padded with zero heads to sparse_sdpa's 32; their output rows are never read):
            # no all-gather of the queries, half the SDPA, no partition of the output; with
            # ``flash`` the live rows' keys split over the grid
            qh = q_heads(q, cos, sin, attn.trans_mat, attn.local_heads, attn.rope_dim, out_heads=attn.sdpa_heads)
            if self.flash is not None and "flash_live" in selection:
                out = self.flash(
                    qh,
                    kv_tensor,
                    rows,
                    kv_format == MlaKvCacheFormat.SCALED_FP8,
                    attn.scale,
                    attn.sink_local,
                    live=selection["flash_live"],
                )
            else:
                out = ttnn.transformer.sparse_sdpa(
                    qh,
                    kv_tensor,
                    rows,
                    attn.head_dim,
                    kv_format=kv_format.sparse_sdpa_format,
                    scale=attn.scale,
                    k_chunk_size=k_chunk,
                    attention_sink=attn.sink_local,
                )
            out = o_heads(
                out,
                cos,
                self._neg(rope_key, sin),
                attn.trans_mat,
                attn.o_groups // self.tp,
                attn.rope_dim,
                heads=attn.local_heads,
            )
            return attn._o_proj(out)
        q = attn.ccl.tp_all_gather(q, dim=3)
        out = ttnn.transformer.sparse_sdpa(
            q_heads(q, cos, sin, attn.trans_mat, attn.heads, attn.rope_dim),
            kv_tensor,
            rows,
            attn.head_dim,
            kv_format=kv_format.sparse_sdpa_format,
            scale=attn.scale,
            k_chunk_size=k_chunk,
            attention_sink=attn.sink,
        )
        out = o_heads(out, cos, self._neg(rope_key, sin), attn.trans_mat, attn.o_groups, attn.rope_dim)
        if self.tp > 1:
            out = ttnn.mesh_partition(out, dim=1, cluster_axis=TP_AXIS)
        return attn._o_proj(out)

    def _neg(self, key: str, sin):
        """``-sin`` for ``o_heads``' inverse RoPE: one ``ttnn.neg`` a step per RoPE table under
        ``V41_DECODE_NEG_SIN=once`` (the same ``sin`` serves every layer of the step; 40 one-tile ops a step
        otherwise), one a layer otherwise."""
        if NEG_SIN != "once":
            return ttnn.neg(sin)
        out = self._neg_sin.get(key)
        if out is None:
            out = self._neg_sin[key] = ttnn.neg(sin)
        return out

    def _encode(self, rows, fmt: MlaKvCacheFormat):
        """Logical ``[1, 1, ROWS, 512]`` rows -> the row-major rows of ``fmt`` (as ``V41PrefillState._encode``)."""
        rows = ttnn.to_layout(rows, ttnn.ROW_MAJOR_LAYOUT)
        if fmt == MlaKvCacheFormat.BF16_RM:
            return rows
        latent, scales = ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(
            rows, round_scale_to_power_of_two=True
        )
        packed = ttnn.experimental.deepseek_prefill.pack_scaled_fp8_kv_cache(latent, scales)
        ttnn.deallocate(latent)
        ttnn.deallocate(scales)
        return packed

    def _compress(self, compressor, x, layer: int):
        """The reference ``Compressor`` for every row: ratio 1 the row's latent; ratio 2 the pool of the pool's
        carry (the previous step's projections) and this step's, the carry then replaced by this step's."""
        if compressor.ratio == 1:
            kv = compressor._project(x, compressor.wkv, ttnn.bfloat16)
            return ttnn.rms_norm(kv, weight=compressor.norm, epsilon=compressor.eps)
        xf = ttnn.typecast(x, ttnn.float32)
        kv = compressor._project(xf, compressor.wkv, ttnn.float32)
        score = compressor._project(xf, compressor.wgate, ttnn.float32)
        k0, s0 = self.pool.carry[layer]
        m = ttnn.maximum(s0, score)
        e0, e1 = ttnn.exp(ttnn.subtract(s0, m)), ttnn.exp(ttnn.subtract(score, m))
        pooled = ttnn.divide(ttnn.add(ttnn.multiply(k0, e0), ttnn.multiply(kv, e1)), ttnn.add(e0, e1))
        ttnn.copy(kv, k0)
        ttnn.copy(score, s0)
        return ttnn.rms_norm(ttnn.typecast(pooled, ttnn.bfloat16), weight=compressor.norm, epsilon=compressor.eps)

    def _index(self, indexer: TtV41Indexer, x, qr, source: int, mask, candidates: dict | None):
        """The indexer of every slot's user over its index keys -> (top-k rows ``[1, 1, ROWS, INDEX_TOPK]`` uint32
        row-major, row u slot u's selection (the pad rows all sentinels); the candidate source's published
        ``CandidateBlocks`` per user, else None). The selection is the prefill's (``TtV41Indexer.select``) with the
        tile's per-row pins (``DecodePins``): the candidate source publishes each user's candidate blocks (none
        below CANDIDATE_TOPK_BLOCKS blocks of capacity: every block is a candidate), and a candidate index source
        selects inside its user's published blocks (``candidates[u]``)."""
        cfg, pool = self.config, self.pool
        cap = pool.slots[source].cap
        assert (cap + 32) % 32 == 0, cap  # the score's width, a segment_max input
        pins = DecodePins(self.inputs[f"newest{cfg.compress_ratio(source)}"])
        cos, sin = self.inputs["ropec"]
        q = ttnn.linear(fp8_qdq(qr), indexer.wq_b, compute_kernel_config=indexer.compute_kernel_config)
        q, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            q, num_heads=indexer.heads, num_kv_heads=0, transpose_k_heads=False
        )
        q = indexer.qdq(indexer.rope(q, cos, sin))
        w = indexer.ccl.tp_all_reduce(ttnn.linear(x, indexer.weights_proj))
        k = pool.index_k[source]  # [users, 1, cap + 32, idim], tiled by every write: no conversion of the history
        config = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=min(256, cap + 32), head_group_size=0)
        parts, published = [], {} if indexer.is_candidate_source else None
        blocks_of = pool.index_k_blocks.get(source)
        in_blocks = (
            blocks_of is not None and candidates is not None and any(c.ids is not None for c in candidates.values())
        )
        single = self.single_query
        q_rm, w_rm = indexer.decode_query(q, w) if in_blocks or single else (None, None)
        for u in range(self.users):
            # Row u is the user's; the other rows' selections (and pins) are other users' queries against this
            # user's keys, discarded
            mine = candidates[u] if candidates is not None else None
            if single:
                # the user's own pin page [1, 1, 1, 1]: the split selection pins by it, the candidate scorer bounds
                pin_u = ttnn.slice(pins.newest, [0, 0, 0, u], [1, 1, 1, u + 1])
            if in_blocks and mine.ids is not None:
                # a candidate index source over published blocks: score the user's candidate rows only, from the
                # pool's block pages (the kernel masks the rows past the user's pin)
                base = u * (cap + 32) // cfg.CANDIDATE_BLOCK_SIZE
                topk = indexer.select_user_in_blocks(
                    q_rm, w_rm, blocks_of, pin_u if single else pins.newest, mine.ids, u, cap, base, cfg.INDEX_TOPK
                )
                parts.append(topk)
                continue
            if single:
                # the single-query scorer: the user's query row against the visible prefix of its tiled keys, the
                # bound read from the newest page on device, -inf past it (the mask's value), the keys split over
                # the rows of the split layout (one core each); the selection merges the rows' top-k
                score = range_scores(q_rm, w_rm, k, pins.newest, u, cap + 32)
                topk, blocks = indexer.select_split(score, pin_u, cap + 32, cfg.INDEX_TOPK, mine)
                if published is not None:
                    published[u] = blocks
                parts.append(topk)
                continue
            # every row sees every key (the causal window starts past the slot's rows); the mask then hides the
            # rows past each user's visible count, and the pad tile
            score = ttnn.experimental.indexer_score_dsa(
                q, k, w, chunk_start_idx=cap, seq_shard_axes=[], program_config=config, cache_batch_idx=u
            )
            score = ttnn.to_layout(ttnn.add(ttnn.to_layout(score, ttnn.TILE_LAYOUT), mask), ttnn.ROW_MAJOR_LAYOUT)
            topk, blocks = indexer.select(score, pins, 0, cfg.INDEX_TOPK, mine)
            if published is not None:
                published[u] = blocks
            parts.append(ttnn.slice(topk, [0, 0, u, 0], [1, 1, u + 1, topk.shape[3]]))
        if self.users < ROWS:
            parts.append(self.pad_topk)
        return ttnn.concat(parts, dim=2), published

    def _index_c1(self, indexer: TtV41Indexer, x, qr, source: int, candidates: CandidateBlocks | None, sel_all):
        """``_index`` for the c1 capture (``V41_DECODE_C1_INDEX=1``): the live user's selection alone, the user read
        on device from the step's ``c1usr`` page (``range_scores(dyn=True)``, ``select_live_in_blocks``), placed into
        its row of the tile by the ``c1put`` page (``c1_place``; the other rows sentinels) -> (the top-k tile
        ``[1, 1, ROWS, INDEX_TOPK]`` uint32 row-major, the candidate source's published ``CandidateBlocks`` of the
        live user or None). ``sel_all``: the one-hot that repeats the live row over the tile (``c1all``)."""
        cfg, pool = self.config, self.pool
        cap = pool.slots[source].cap
        ratio = cfg.compress_ratio(source)
        usr, pin_n = self.inputs[f"c1usr{ratio}"], self.inputs[f"c1pin{ratio}"]
        cos, sin = self.inputs["ropec"]
        q = ttnn.linear(fp8_qdq(qr), indexer.wq_b, compute_kernel_config=indexer.compute_kernel_config)
        q, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            q, num_heads=indexer.heads, num_kv_heads=0, transpose_k_heads=False
        )
        q = indexer.qdq(indexer.rope(q, cos, sin))
        w = indexer.ccl.tp_all_reduce(ttnn.linear(x, indexer.weights_proj))
        blocks_of = pool.index_k_blocks.get(source)
        if blocks_of is not None and candidates is not None and candidates.ids is not None:
            # a candidate index source over the live user's published blocks: its query and head weights in every
            # row of the tile by the one-hot matmul (exact: one 1.0 a row), the scorer's per-row pin page the
            # ``c1pin`` row, its block 0 read by the scorer from ``c1usr``
            q_n = ttnn.to_layout(
                ttnn.matmul(sel_all, ttnn.experimental.nlp_concat_heads(q), compute_kernel_config=_C1_HIFI4),
                ttnn.ROW_MAJOR_LAYOUT,
            )
            w_n = ttnn.to_layout(ttnn.matmul(sel_all, w, compute_kernel_config=_C1_HIFI4), ttnn.ROW_MAJOR_LAYOUT)
            topk = indexer.select_live_in_blocks(
                q_n, w_n, blocks_of, pin_n, candidates.ids, usr, cap + 32, cfg.INDEX_TOPK
            )
            published = None
        else:
            q_rm, w_rm = indexer.decode_query(q, w)
            score = range_scores(q_rm, w_rm, pool.index_k[source], usr, 0, cap + 32, dyn=True)
            pin_u = ttnn.slice(usr, [0, 0, 0, 2], [1, 1, 1, 3])
            topk, published = indexer.select_split(score, pin_u, cap + 32, cfg.INDEX_TOPK, candidates)
        return c1_place(topk, self.inputs["c1put"]), published

    def _index_rows(self, layer: int, compressed):
        """``sparse_sdpa`` index rows ``[1, 1, ROWS, K]`` uint32 row-major: the layer's window ring, then the selected
        compressed rows of each row's user (offset to its slot) with their sentinel tail."""
        pool = self.pool
        window = pool.window_rows(layer)
        if compressed is None:
            return window
        src = self.config.kv_source(layer)
        comp = ttnn.typecast(ttnn.to_layout(compressed, ttnn.TILE_LAYOUT), ttnn.int32)
        comp = ttnn.where(ttnn.eq(comp, -1), -1, ttnn.add(comp, pool.user_offsets(src)))
        comp = ttnn.to_layout(ttnn.typecast(comp, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
        rows = ttnn.concat([window, comp], dim=-1)
        assert rows.shape[3] % TOPK_ALIGN == 0, tuple(rows.shape)
        return rows
