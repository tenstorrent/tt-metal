# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 attention for one prefill chunk (graph nodes B4-B16; beads F3, F7).

q = wq_b(q_norm(wq_a(x))) with RoPE on the trailing 64 channels (no per-head norm); window KV
kv_norm(wkv(x)), RoPE, FP8 QDQ (BF16 KV format; the SCALED_FP8 format encodes the unrounded KV, epic KV
FORMAT; likewise the compressed KV's FP4 QDQ). Compressed layers attend over their KV source's compressed rows too:
a KV source runs its compressor, index keys and compressed-KV write; an index source runs its indexer
(the candidate source publishes candidate blocks); consumers reuse the top-k their index source published
this chunk. One ``sparse_sdpa`` over the layer's KV tensor (``cache.V41PrefillState``: window region + the
source's compressed rows) with the per-head sink, inverse RoPE, grouped low-rank output projection. Position-dependent
inputs (RoPE, window rows) are device slices of ``state.tables`` (``cache.V41ChunkTables``): the forward uploads
nothing, so it can be traced.

Every quantized GEMM input gets the reference's FP8 activation QDQ (dev-spec D-I). Layout: input/output
``[1, 1, S/sp, hidden/tp]``; the attention itself runs on a sequence shard of all heads (head->sequence
all-to-all over TP), which is also the query layout of the indexer's selection. The queries stay in the
projection's seq-major layout through that all-to-all; one fused op (``head_layout.q_heads``) splits them into
sparse_sdpa's row-major heads with the RoPE tail, and one (``head_layout.o_heads``) takes the output back to the
tiled head groups of ``wo_a`` with the inverse RoPE (no create / concat-heads or RoPE-tail glue ops).
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.tt.mla.rope import get_rot_transformation_mat
from models.demos.deepseek_v3_d_p.tt.v41.cache import V41PrefillState
from models.demos.deepseek_v3_d_p.tt.v41.ccl import V41Collectives
from models.demos.deepseek_v3_d_p.tt.v41.compressor import TtV41Compressor
from models.demos.deepseek_v3_d_p.tt.v41.head_layout import o_heads, q_heads
from models.demos.deepseek_v3_d_p.tt.v41.indexer import TtV41Indexer, TtV41IndexKeys
from models.demos.deepseek_v3_d_p.tt.v41.layout import TP_AXIS
from models.demos.deepseek_v3_d_p.tt.v41.qdq import fp4_e4m3_qdq, fp8_qdq
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat

TOPK_ALIGN = 32  # topk_large_indices needs a multiple of 16; sparse_sdpa k chunks a multiple of 32


def _round_up(x: int, m: int) -> int:
    return -(-x // m) * m


# Dense projections on a 2D-multicast program config (bead 8y7.9.7): ttnn's default config for these shapes is 1.4-2x
# slower at chunk 5120. The K block is per projection: bf16 partial sums round more with wider blocks, so each takes the
# widest that keeps the default's accuracy (tests/v41/test_v41_attention_matmuls.py sweeps and checks them).
MATMUL_GRID = (11, 10)  # as tt/mla/mla_config.py
DENSE_IN0_BLOCK_W = {"wq_a": 5, "wkv": 5, "wq_b": 5, "wo_a": 8}
DENSE_L1_BUDGET = 1 << 20  # bytes of output block + double-buffered in0 / in1 blocks per core
# ttnn.linear without a program config runs bf16 x bfp8 at HiFi2 with packer L1 accumulation; a program config alone
# would select LoFi. Every projection runs this explicit config (the default's numerics).
DENSE_COMPUTE_CONFIG = ttnn.types.BlackholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
)


def dense_program_config(m: int, k: int, n: int, in0_block_w: int):
    """2D-multicast config of a [m, k] x [k, n] bf16 x bfp8 matmul (per batch) on ``MATMUL_GRID``, or None (ttnn's
    default) when the blocks do not fit ``DENSE_L1_BUDGET`` or ``in0_block_w`` does not divide K."""
    tile = 32
    mt, kt, nt = m // tile, k // tile, n // tile
    gx, gy = MATMUL_GRID
    per_core_m, per_core_n = -(-mt // gy), -(-nt // gx)
    if kt % in0_block_w:
        return None
    l1 = per_core_m * per_core_n * 2048 + 2 * per_core_m * in0_block_w * 2048 + 2 * in0_block_w * per_core_n * 1088
    if l1 > DENSE_L1_BUDGET:
        return None
    sub_w = max(w for w in range(1, 9) if per_core_n % w == 0)
    sub_h = max(h for h in range(1, 9) if per_core_m % h == 0 and h * sub_w <= 8)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=MATMUL_GRID,
        in0_block_w=in0_block_w,
        out_subblock_h=sub_h,
        out_subblock_w=sub_w,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fuse_batch=False,
        fused_activation=None,
    )


class TtV41Attention(LightweightModule):
    def __init__(
        self,
        mesh_device,
        config,
        layer: int,
        weights: dict,
        weights_dtype=ttnn.bfloat8_b,
        compute_kernel_config=None,
    ):
        """``weights`` (torch, checkpoint orientation [out, in], FP8 already dequantized): wq_a, q_norm, wq_b,
        wkv, kv_norm, wo_a, wo_b, attn_sink; plus ``compressor`` {wkv, norm[, wgate]} for KV sources and
        ``indexer`` {wq_b, weights_proj[, wk, k_norm]} for index sources (``tt/v41/weights.py`` keys)."""
        self.mesh_device, self.config, self.layer = mesh_device, config, layer
        self.ratio = config.compress_ratio(layer)
        self.heads, self.head_dim = config.NUM_ATTENTION_HEADS, config.HEAD_DIM
        self.rope_dim, self.window = config.QK_ROPE_HEAD_DIM, config.SLIDING_WINDOW
        self.o_groups, self.eps = config.O_GROUPS, config.RMS_NORM_EPS
        self.scale = self.head_dim**-0.5
        self.sp, self.tp = mesh_device.shape
        self.compute_kernel_config = compute_kernel_config or DENSE_COMPUTE_CONFIG
        self.ccl = V41Collectives(mesh_device)
        shape = tuple(mesh_device.shape)
        tp_mapper = lambda dim: ttnn.ShardTensor2dMesh(mesh_device, shape, dims=(None, dim))
        replicate = ttnn.ReplicateTensorToMesh(mesh_device)

        def linear_weight(w, tp_dim):
            t = w.detach().to(torch.bfloat16).transpose(-2, -1).contiguous()[None, None]
            return ttnn.from_torch(
                t, device=mesh_device, dtype=weights_dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=tp_mapper(tp_dim)
            )

        def row(w):
            return ttnn.from_torch(
                w.detach().to(torch.bfloat16).reshape(1, 1, 1, -1),
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=replicate,
            )

        self.wq_a = linear_weight(weights["wq_a"], 2)  # row-parallel over hidden
        self.wq_b = linear_weight(weights["wq_b"], 3)  # column-parallel over heads
        self.wkv = linear_weight(weights["wkv"], 2)
        self.q_norm, self.kv_norm = row(weights["q_norm"]), row(weights["kv_norm"])
        in_per_group = self.heads * self.head_dim // self.o_groups
        o_a = weights["wo_a"].detach().to(torch.bfloat16).view(self.o_groups, -1, in_per_group)
        self.wo_a = ttnn.from_torch(
            o_a.transpose(1, 2).unsqueeze(0).contiguous(),
            device=mesh_device,
            dtype=weights_dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=tp_mapper(1),
        )
        self.wo_b = linear_weight(weights["wo_b"], 2)
        # sparse_sdpa multiplies the sink by the softmax scale; the reference adds it unscaled
        sink = (weights["attn_sink"].detach().float() / self.scale).reshape(1, 1, 1, self.heads).to(torch.bfloat16)
        self.sink = ttnn.from_torch(
            sink, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=replicate
        )
        self.trans_mat = ttnn.from_torch(
            get_rot_transformation_mat(),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=replicate,
        )
        # column iota of the index-row compaction (``_compact_leading``): the widest row is window + top-k
        self.compact_rows = _round_up(self.window - 1, 32)  # queries that can lack window rows
        width = _round_up(self.window + (config.INDEX_TOPK if self.ratio else 0), TOPK_ALIGN)
        self.column_iota = ttnn.from_torch(
            torch.arange(width, dtype=torch.int32).expand(self.compact_rows, width).reshape(1, 1, -1, width),
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=replicate,
        )

        self.is_kv_source = layer in config.KV_SOURCE_LAYERS
        self.is_index_source = layer in config.INDEX_SOURCE_LAYERS
        self.compressor = (
            TtV41Compressor(mesh_device, config, layer, weights["compressor"]) if self.is_kv_source else None
        )
        self.index_keys = TtV41IndexKeys(mesh_device, config, layer, weights["indexer"]) if self.is_kv_source else None
        self.indexer = TtV41Indexer(mesh_device, config, layer, weights["indexer"]) if self.is_index_source else None

    def _dense(self, x, name: str):
        """``x @ self.<name>`` on its tuned program config."""
        w = getattr(self, name)
        config = dense_program_config(x.shape[-2], w.shape[-2], w.shape[-1], DENSE_IN0_BLOCK_W[name])
        return ttnn.linear(x, w, program_config=config, compute_kernel_config=self.compute_kernel_config)

    def _rope(self, t, cos, sin, inverse=False):
        b, h, s, d = t.shape
        nope = ttnn.slice(t, [0, 0, 0, 0], [b, h, s, d - self.rope_dim])
        rope = ttnn.slice(t, [0, 0, 0, d - self.rope_dim], [b, h, s, d])
        rope = ttnn.experimental.rotary_embedding_llama(
            rope, cos, ttnn.neg(sin) if inverse else sin, self.trans_mat, is_decode_mode=False
        )
        return ttnn.concat([nope, rope], dim=-1)

    def _index_rows(self, window, window_base: int, compressed, start: int):
        """``window``: the chunk's per-chip window rows (``V41ChunkTables.window_rows``) -> per-chip
        [1, 1, S/(sp*tp), K] uint32 rows into the KV tensor, valid first, sentinel tail.

        Both parts are valid-first except the window of a query with fewer than ``window`` earlier tokens: its
        missing rows are a leading run of -1 (``_window_table``); the top-k has a sentinel tail. So [window | top-k]
        is valid-first for every query at or after position window - 1; the earlier ones are compacted."""
        if compressed is None:
            rows = window
        else:
            comp = ttnn.typecast(ttnn.to_layout(compressed, ttnn.TILE_LAYOUT), ttnn.int32)
            comp = ttnn.where(ttnn.eq(comp, -1), -1, ttnn.add(comp, window_base))
            rows = ttnn.concat([window, comp], dim=-1)
        width = rows.shape[-1]
        k = _round_up(width, TOPK_ALIGN)
        if k != width:
            rows = ttnn.pad(rows, [(0, 0), (0, 0), (0, 0), (0, k - width)], -1)
        if start < self.window - 1:
            rows = self._compact_leading(rows, window)
        return ttnn.to_layout(ttnn.typecast(rows, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)

    def _compact_leading(self, rows, window):
        """Rotate each of the chip's first ``compact_rows`` query rows left by its count of missing window rows (the
        leading -1 run): valid rows first, the -1 run after the top-k's sentinel tail. Later queries (all window rows
        present) rotate by 0; a chip's rows are contiguous query positions, so no later chip needs more rows."""
        q_rows, k = rows.shape[2], rows.shape[3]
        n = min(self.compact_rows, q_rows)
        head_window = ttnn.slice(window, [0, 0, 0, 0], [1, 1, n, window.shape[3]])
        missing = ttnn.sum(ttnn.typecast(ttnn.eq(head_window, -1), ttnn.float32), dim=-1, keepdim=True)
        shift = ttnn.add(ttnn.slice(self.column_iota, [0, 0, 0, 0], [1, 1, n, k]), ttnn.typecast(missing, ttnn.int32))
        # sentinel columns past the end keep every shifted index in range
        head = ttnn.pad(
            ttnn.slice(rows, [0, 0, 0, 0], [1, 1, n, k]), [(0, 0), (0, 0), (0, 0), (0, window.shape[3])], -1
        )
        head = ttnn.gather(head, -1, ttnn.typecast(shift, ttnn.uint32))
        if n == q_rows:
            return head
        return ttnn.concat([head, ttnn.slice(rows, [0, 0, n, 0], [1, 1, q_rows, k])], dim=2)

    # --- forward -------------------------------------------------------------------------------------
    def forward(self, x, state: V41PrefillState, length: int):
        """x [1, 1, S/sp, hidden/tp] bf16 (after attn_norm) for the chunk at ``state.start`` with ``length``
        valid tokens -> [1, 1, S/sp, hidden/tp] bf16. Writes this layer's window carry and, for a KV source,
        its compressed rows and carry; an index source publishes its selection in ``state.selection``."""
        start = state.start
        tables = state.tables
        cos, sin = tables.rope(self.ratio > 0, 1, start)
        xq = fp8_qdq(x)

        qr = ttnn.rms_norm(
            self.ccl.tp_all_reduce(self._dense(xq, "wq_a")),
            weight=self.q_norm,
            epsilon=self.eps,
        )
        q = self._dense(fp8_qdq(qr), "wq_b")  # [1, 1, S/sp, H/tp * head_dim], this chip's heads side by side

        kv = ttnn.rms_norm(
            self.ccl.tp_all_reduce(self._dense(xq, "wkv")),
            weight=self.kv_norm,
            epsilon=self.eps,
        )
        # stage 1 (BF16) stores the reference's QDQ values; SCALED_FP8 quantizes the unrounded KV itself
        kv_format = state.layer_kv_format(self.layer)
        reference_values = kv_format == MlaKvCacheFormat.BF16_RM
        kv = self._rope(kv, cos, sin)
        kv_tensor = state.write_window(self.layer, fp8_qdq(kv) if reference_values else kv)

        compressed = None
        if self.ratio:
            if self.is_kv_source:
                latent, carry = self.compressor(x)
                state.set_compressor_carry(self.layer, carry, length)
                c_rope = tables.rope(True, self.ratio, start)  # the compressed rows' group-first positions
                keys = self.index_keys(latent, c_rope)
                comp_kv = self._rope(latent, *c_rope)
                state.write_compressed(self.layer, fp4_e4m3_qdq(comp_kv) if reference_values else comp_kv, keys, length)
            if self.is_index_source:
                src = self.config.kv_source(self.layer)
                compressed, candidates = self.indexer(
                    x, qr, state.index_k[src], tables, start, length, state.selection.get("candidates")
                )
                state.selection["topk"] = compressed
                if candidates is not None:
                    state.selection["candidates"] = candidates
            else:
                compressed = state.selection["topk"]
        rows = self._index_rows(tables.window_rows(start), state.geometry.window_rows, compressed, start)

        # sparse_sdpa needs >= 32 heads per chip: attend on a sequence shard of all heads (head->sequence: the chips'
        # head columns side by side, in head order); RoPE and its inverse run on that shard (the chip's rows of cos /
        # sin), fused into the head layout ops around the row-major, head-major sparse_sdpa operands
        head_to_seq = self.tp > 1
        if head_to_seq:
            q = self.ccl.tp_all_to_all(q, in_dim=3, out_dim=2)
            cos, sin = (ttnn.mesh_partition(t, dim=2, cluster_axis=TP_AXIS) for t in (cos, sin))
        attn = ttnn.transformer.sparse_sdpa(
            q_heads(q, cos, sin, self.trans_mat, self.heads, self.rope_dim),
            kv_tensor,
            rows,
            self.head_dim,
            kv_format=kv_format.sparse_sdpa_format,
            scale=self.scale,
            k_chunk_size=next(c for c in (128, 64, 32) if rows.shape[-1] % c == 0),
            attention_sink=self.sink,
        )
        attn = o_heads(attn, cos, ttnn.neg(sin), self.trans_mat, self.o_groups, self.rope_dim)  # [1, G, S', G heads]
        if head_to_seq:
            attn = self.ccl.tp_all_to_all(attn, in_dim=2, out_dim=1)
        state.update_window_carry(self.layer, length)
        return self._o_proj(attn)

    def _o_proj(self, attn):
        """[1, G/tp, S/sp, (H/G) * head_dim] (each group's heads side by side) -> [1, 1, S/sp, hidden/tp]."""
        grouped = ttnn.experimental.nlp_concat_heads(
            self._dense(attn, "wo_a")
        )  # [1, g, S, rank] -> [1, 1, S, g * rank]
        out = ttnn.linear(fp8_qdq(grouped), self.wo_b, compute_kernel_config=self.compute_kernel_config)
        return self.ccl.tp_reduce_scatter(out)
