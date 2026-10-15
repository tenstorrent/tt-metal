# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Data-parallel prefill with TP decode (Blackhole 1x4, GPT_OSS_DP_PREFILL=1, default off).

Each of the N = 4 devices prefills its own user with a full-width copy of every layer (no TP collective inside a
layer); each layer's K and V are redistributed into the TP paged KV cache, so decode is unchanged.

User -> device mapping: step s (of ceil(B / 4)) prefills users 4s .. 4s + 3, user 4s + c on device c. A batch that is
not a multiple of 4 pads its last step with a copy of a real user's tokens whose page-table row is all -1 (writes
nothing) and whose logits are not read.

Data shapes per step (L = padded prompt length, the same for the 4 users of a step; H = 2880):
  tokens   host [4, L] -> ShardTensor2dMesh(dims=(None, 0)) -> per device [1, 1, 1, L] uint32
  embed    [1, 1, L, H] bf8 (the TP model's replicated embedding table)
  per layer:
    input norm (TP layer's replicated RMSNorm)                      [1, 1, L, H] bf8
    QKV   x @ wqkv [H, 5120] bf8 (replicated, full width) + bias   [1, 1, L, 5120] bf16
    heads Q [1, 64, L, 64], K / V [1, 8, L, 64]; RoPE on Q, K; K / V typecast bf8 (the cache dtype)
    K/V handoff (each of K, V): all_to_all_async_generic(in_dim=0, out_dim=1, cluster_axis=1)
        [1, 8, L, 64] -> [4, 2, L, 64]: device d receives kv heads 2d, 2d + 1 (its TP cache heads); batch c comes
        from device c (user 4s + c). One paged_fill_cache into the TP layer's cache with the step's page table
        [4, L / block] (replicated, row c = user 4s + c) and batch_idx_tensor 0..3.
    SDPA  causal (+ sliding window on even layers), sinks [1, 64, 1, 1], the TP layer's program configs
    o_proj exact (default, GPT_OSS_DP_PREFILL_EXACT=1): the 4 TP partials (16-head group d @ TP shard d [1024, H],
          TP's config) [4, 1, L, H] bf8 -> tp_exact_sum -> [1, 1, L, H] bf8, bit-identical to the TP all-reduce;
          local (=0): heads [1, 1, L, 4096] @ o_proj [4096, H] (replicated) + bias -> [1, 1, L, H] bf8, no collective
    residual add -> bf16; post-attention norm; router (the TP MLP's routed router: same slot ids)
    MoE   DPRoutedPrefillExperts: all 32 experts per device, local routing; exact: the 4 TP-device partials
          [4, 1, L, H] (call_groups) -> tp_exact_sum; local: one combine, no collective -> [1, 1, L, H] bf8
    residual add -> bf8
  tp_exact_sum ([4, 1, L, H] partials of user c on device c): all_to_all (in_dim=2, out_dim=0) -> [1, 1, 4L, H]
    (device d: partial d of the 4 users); per_user_reduce(scatter=True) -> [1, 1, L, H] (rows d L / 4 .. of each
    user); [1, 4, L / 4, H] view, all_to_all (in_dim=2, out_dim=1) -> [1, 1, L, H] (user c's rows on device c).
  last token: one-hot [1, 1, 32, L] selection matmul per device puts user u's last-token row at row u of a
    [1, 1, 32, H] tile, summed over the steps; after the last step one all_reduce over the mesh row, the TP final
    norm and the TP (column-parallel) LM head on those 32 rows: [1, 1, 32, vocab / 4] per device.

Weights: attention is re-assembled from the TP layer's device tensors (head shards read back, concatenated per Q / K /
V, re-uploaded replicated; bf8 -> bf8 is exact because the shards move in whole tiles); the routed experts are the
TP routed cache's bytes regrouped on host (experts/prefill_routed_dp.py). No weight is converted again and no cache
file is written. Embedding, norms, router, final norm, LM head and RoPE are shared with the TP model.
"""

import math
import os
from dataclasses import dataclass

import torch
from loguru import logger

import ttnn

from .batched_prefill import batched_reduce_links, fill_batch_idx, per_user_reduce
from .experts.prefill_routed_dp import DPRoutedPrefillExperts

DP_PREFILL_ENV = "GPT_OSS_DP_PREFILL"
DP_PREFILL_MESH = (1, 4)
# Longest per-user prompt the DP stack runs (DecoderLayer switches to an in-place bf8 residual above 32K).
DP_PREFILL_MAX_SEQ = 32 * 1024
# "1" (default): the o_proj and MoE sums of every layer exactly as the TP batch-1 prefill computes them
# (tp_exact_sum: 2 all_to_all + the TP reduce-scatter per sum), so each user's KV pages and logits are bit-identical to
# its TP batch-1 prefill. "0": local sums (one o_proj matmul, one MoE combine, no collective): faster, but the
# rounding differs from TP's and grows through the layers like any other TP variant's (dp/README.md).
DP_PREFILL_EXACT_ENV = "GPT_OSS_DP_PREFILL_EXACT"


def dp_prefill_exact():
    return os.getenv(DP_PREFILL_EXACT_ENV, "1") == "1"


def dp_prefill_enabled():
    return os.getenv(DP_PREFILL_ENV, "0") == "1"


def dp_prefill_mesh_supported(mesh_device):
    return ttnn.device.is_blackhole(mesh_device) and tuple(mesh_device.shape) == DP_PREFILL_MESH


@dataclass(frozen=True)
class DPAttentionWeights:
    """Full-width attention weights, replicated on every device."""

    wqkv: ttnn.Tensor  # [1, 1, H, (Hq + 2 Hkv) * D]: all Q heads, then all K, then all V
    wqkv_bias: ttnn.Tensor
    o_proj: ttnn.Tensor  # [Hq * D, H] (no CCL padding columns)
    o_proj_bias: ttnn.Tensor  # [H]
    sinks: ttnn.Tensor  # [1, Hq, 1, 1], pre-divided by the scale like the TP sinks
    # TP-exact o_proj (DP_PREFILL_EXACT): TP device d's unpadded o_proj shard [Hq * D / TP, H] and bias shard (the
    # real bias on group 0, zeros elsewhere), replicated.
    o_proj_groups: tuple = ()
    o_bias_groups: tuple = ()


def dp_attention_weights(tp_weights, config, mesh_device):
    """DPAttentionWeights from a TP layer's AttentionWeights (attention/weights.py layouts):
    wqkv / wqkv_bias column-parallel, device i = [Q heads of i | K heads of i | V heads of i]; o_proj row-parallel
    [K / TP, padded H] (real columns first); o_proj_bias on device 0 only (padded); sinks sharded over heads."""
    tp = mesh_device.shape[1]
    hd = config.head_dim
    q_local = config.num_heads * hd // tp
    kv_local = config.num_kv_heads * hd // tp
    hidden = config.hidden_size

    def shards(t):
        return [ttnn.to_torch(s) for s in ttnn.get_device_tensors(t)]

    def qkv_full(parts):
        q = torch.cat([p[..., :q_local] for p in parts], dim=-1)
        k = torch.cat([p[..., q_local : q_local + kv_local] for p in parts], dim=-1)
        v = torch.cat([p[..., q_local + kv_local : q_local + 2 * kv_local] for p in parts], dim=-1)
        return torch.cat([q, k, v], dim=-1)

    rep = ttnn.ReplicateTensorToMesh(mesh_device)

    def put(t, dtype):
        return ttnn.from_torch(
            t,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )

    o_parts = [p[..., :hidden] for p in shards(tp_weights.o_proj)]
    b_parts = [b[..., :hidden] for b in shards(tp_weights.o_proj_bias)]
    return DPAttentionWeights(
        wqkv=put(qkv_full(shards(tp_weights.wqkv)), tp_weights.wqkv.dtype),
        wqkv_bias=put(qkv_full(shards(tp_weights.wqkv_bias)), tp_weights.wqkv_bias.dtype),
        o_proj=put(torch.cat(o_parts, dim=-2), tp_weights.o_proj.dtype),
        o_proj_bias=put(b_parts[0], tp_weights.o_proj_bias.dtype),
        sinks=put(torch.cat(shards(tp_weights.sinks), dim=1), tp_weights.sinks.dtype),
        o_proj_groups=tuple(put(p, tp_weights.o_proj.dtype) for p in o_parts),
        o_bias_groups=tuple(put(b, tp_weights.o_proj_bias.dtype) for b in b_parts),
    )


def tp_exact_sum(groups, num_links):
    """The TP reduction of this device's user, bit for bit.

    groups: [N, 1, L, H] bf8 per device (device c = user c of the step): row block d is the partial sum TP device d
    computes for that user. Returns [1, 1, L, H]: the user's rows exactly as the TP reduce-scatter / all-reduce of that
    user alone gives them. On-device sums cannot reproduce it (the ring reduce-scatter's per-hop rounding and order
    match no stock add chain; dp/README.md), so the TP collective itself runs, on all 4 users at once:
      all_to_all (in_dim=2, out_dim=0): device d gets partial d of every user, [1, 1, N L, H] stacked by user;
      per_user_reduce(scatter=True): device d gets rows d L / N .. of every user (bit-identical to each user's
        batch-1 ttnn.reduce_scatter, the half ttnn.all_reduce runs first: tests/unit/test_batched_prefill.py);
      all_to_all (in_dim=2, out_dim=1) of the [1, N, L / N, H] view: device c gets user c's N slices in order."""
    n = groups.shape[0]
    seq_len, hidden = groups.shape[-2], groups.shape[-1]
    by_group = ttnn.experimental.all_to_all_async_generic(
        groups, in_dim=2, out_dim=0, num_links=num_links, memory_config=ttnn.DRAM_MEMORY_CONFIG, cluster_axis=1
    )
    groups.deallocate(True)
    rows = per_user_reduce(by_group, n, 1, num_links, scatter=True)
    by_group.deallocate(True)
    rows_v = ttnn.reshape(rows, (1, n, seq_len // n, hidden))
    full = ttnn.experimental.all_to_all_async_generic(
        rows_v, in_dim=2, out_dim=1, num_links=num_links, memory_config=ttnn.DRAM_MEMORY_CONFIG, cluster_axis=1
    )
    rows_v.deallocate(True)
    rows.deallocate(True)
    return full


def dp_proj_program_config(mesh_device, role, x, weight, in0_dtype=None):
    """2D multicast config of a full-width DP projection (role "qkv": [L, 2880] x [2880, 5120], "o": [L, 4096] x
    [4096, 2880]) with the TP role's grid columns and K block (attention/config.py PREFILL_PROJ_2D), N per core
    scaled to the full width; out_block_h / per_core_M chosen by the TP rule (prefill_proj_program_config).

    The same K block means each output element accumulates its K tiles in the same order as the TP projection: the
    DP QKV output is then the TP one (the 4 TP shards side by side). The default ttnn config rounds Q differently
    (Q PCC 0.99993 vs TP), which the peaked softmax of later layers amplifies (layer-23 attention output 0.997 vs TP).
    None (ttnn default) when the shape or dtypes do not fit."""
    from .attention.config import PREFILL_PROJ_2D, PREFILL_PROJ_L1_BYTES

    if not ttnn.device.is_blackhole(mesh_device):
        return None
    shape = x.padded_shape
    if any(shape[i] != 1 for i in range(len(shape) - 2)):
        return None
    in0_dtype = x.dtype if in0_dtype is None else in0_dtype
    if in0_dtype != ttnn.bfloat8_b or weight.dtype != ttnn.bfloat8_b:
        return None
    grid = mesh_device.compute_with_storage_grid_size()
    m_tiles = shape[-2] // ttnn.TILE_SIZE
    k_tiles = shape[-1] // ttnn.TILE_SIZE
    n_tiles = weight.padded_shape[-1] // ttnn.TILE_SIZE
    gx, _, kb = PREFILL_PROJ_2D[role]
    if gx > grid.x or n_tiles % gx or k_tiles % kb:
        return None
    pcn = n_tiles // gx
    out_tile = 2048 if role == "qkv" else 1088  # QKV writes bf16, o_proj bf8

    def block_h(pcm):
        for obh in range(min(pcm, 32), 0, -1):
            l1 = obh * kb * 2 * 1088 + kb * pcn * 2 * 1088 + obh * pcn * (out_tile + 2048)
            if pcm % obh == 0 and l1 <= PREFILL_PROJ_L1_BYTES:
                return obh
        return 1

    pcm = -(-m_tiles // grid.y)
    obh = block_h(pcm)
    if obh < 12 and pcm > 12:
        pcm, obh = max(((p, block_h(p)) for p in range(pcm, pcm + 9)), key=lambda t: (t[1], -t[0]))
    gy = -(-m_tiles // pcm)
    sw = next(w for w in (4, 2, 1) if pcn % w == 0)
    sh = next(h for h in (4, 2, 1) if obh % h == 0 and h * sw <= 8)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        in0_block_w=kb,
        out_subblock_h=sh,
        out_subblock_w=sw,
        out_block_h=obh,
        out_block_w=pcn,
        per_core_M=pcm,
        per_core_N=pcn,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )


def kv_handoff(k, v, kv_cache, page_table, batch_idx, num_links):
    """Write this step's K and V ([1, Hkv, L, D] per device, device c = user c of the step, cache dtype) into the TP
    paged cache: all_to_all to [N, Hkv / N, L, D] (device d: its TP kv heads of all N users, batch c from device c),
    then one batched paged fill through page_table [N, blocks] (row c = user of device c; -1 rows write nothing)."""
    k_cache, v_cache = kv_cache
    page_len = page_table.shape[-1] * k_cache.shape[2]
    for src, cache in ((k, k_cache), (v, v_cache)):
        recv = ttnn.experimental.all_to_all_async_generic(
            src, in_dim=0, out_dim=1, num_links=num_links, memory_config=ttnn.DRAM_MEMORY_CONFIG, cluster_axis=1
        )
        fill = recv[:, :, :page_len, :] if page_len < recv.shape[2] else recv
        ttnn.experimental.paged_fill_cache(cache, fill, page_table, batch_idx_tensor=batch_idx)
        if fill is not recv:
            fill.deallocate(True)
        recv.deallocate(True)


def dp_attention_prefill(
    x, rope_mats, weights: DPAttentionWeights, attn, kv_cache, page_table, batch_idx, num_links, exact=False
):
    """Full-width prefill attention of this device's user. x: [1, 1, L, H] normed input (consumed). attn: the TP
    layer's Attention (config, program config, RoPE transformation matrix). Returns [1, 1, L, H] bf8.
    exact: the output projection and its sum exactly as the TP attention computes them (tp_exact_sum)."""
    config = attn.config
    pc = attn.program_config
    mesh_device = attn.mesh_device
    seq_len = x.shape[-2]
    qkv = ttnn.linear(
        x,
        weights.wqkv,
        bias=weights.wqkv_bias,
        dtype=ttnn.bfloat16,
        program_config=dp_proj_program_config(mesh_device, "qkv", x, weights.wqkv),
    )
    x.deallocate(True)
    q, k, v = ttnn.experimental.nlp_create_qkv_heads(
        qkv,
        num_heads=config.num_heads,
        num_kv_heads=config.num_kv_heads,
        transpose_k_heads=False,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    qkv.deallocate(True)
    tmat = attn.transformation_mats["prefill"]
    q_r = ttnn.experimental.rotary_embedding_llama(q, rope_mats[0], rope_mats[1], tmat, is_decode_mode=False)
    k_r = ttnn.experimental.rotary_embedding_llama(k, rope_mats[0], rope_mats[1], tmat, is_decode_mode=False)
    q.deallocate(True)
    k.deallocate(True)
    k_c = ttnn.typecast(k_r, kv_cache[0].dtype)
    v_c = ttnn.typecast(v, kv_cache[1].dtype)
    k_r.deallocate(True)
    v.deallocate(True)

    kv_handoff(k_c, v_c, kv_cache, page_table, batch_idx, num_links)

    out = ttnn.transformer.scaled_dot_product_attention(
        q_r,
        k_c,
        v_c,
        is_causal=True,
        sliding_window_size=config.sliding_window,
        program_config=pc.get_prefill_sdpa_config(mesh_device, seq_len, config.sliding_window),
        compute_kernel_config=pc.get_prefill_sdpa_compute_kernel_config(mesh_device),
        attention_sink=weights.sinks,
    )
    q_r.deallocate(True)
    k_c.deallocate(True)
    v_c.deallocate(True)
    heads = ttnn.experimental.nlp_concat_heads(out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out.deallocate(True)
    heads_b8 = ttnn.typecast(heads, ttnn.bfloat8_b)
    heads.deallocate(True)
    if exact:
        # TP's o_proj per head group (its weight shard, bias shard and program config), then TP's reduction.
        groups = len(weights.o_proj_groups)
        width = heads_b8.shape[-1] // groups
        parts = []
        for d in range(groups):
            hd = ttnn.slice(heads_b8, (0, 0, 0, d * width), (1, 1, seq_len, (d + 1) * width))
            w_d = weights.o_proj_groups[d]
            parts.append(
                ttnn.linear(
                    hd,
                    w_d,
                    bias=weights.o_bias_groups[d],
                    dtype=ttnn.bfloat8_b,
                    program_config=pc.get_prefill_proj_config(
                        mesh_device, "o", hd, w_d, groups, in0_dtype=ttnn.bfloat8_b
                    ),
                )
            )
            hd.deallocate(True)
        heads_b8.deallocate(True)
        stacked = ttnn.concat(parts, dim=0)
        for t in parts:
            t.deallocate(True)
        return tp_exact_sum(stacked, num_links)
    o = ttnn.linear(
        heads_b8,
        weights.o_proj,
        bias=weights.o_proj_bias,
        dtype=ttnn.bfloat8_b,
        program_config=dp_proj_program_config(mesh_device, "o", heads_b8, weights.o_proj),
    )
    heads_b8.deallocate(True)
    return o


class DPDecoderLayer:
    """One decoder layer of the DP stack: full-width attention + K/V handoff, all-local routed MoE. Shares the TP
    layer's norms, router and attention configs."""

    def __init__(self, tp_layer, experts_state_dict=None):
        self.tp = tp_layer
        attn = tp_layer.self_attn
        assert tp_layer.mlp.routed_prefill is not None, "the DP prefill stack needs the TP routed prefill experts"
        self.attn_weights = dp_attention_weights(attn.weights, attn.config, attn.mesh_device)
        self.experts = DPRoutedPrefillExperts(tp_layer.mlp.routed_prefill, experts_state_dict)
        self.router = tp_layer.mlp.routed_router

    def __call__(self, x, rope_mats, kv_cache, page_table, batch_idx, num_links, expert_input_dtype=None, exact=False):
        """x: [1, 1, L, H] bf8 residual of this device's user (consumed). expert_input_dtype: dtype the experts read
        the post-attention norm output in (None: as produced), mirroring the TP path the same length takes.
        exact: the attention and MoE sums exactly as TP computes them (tp_exact_sum)."""
        residual = x
        h = self.tp.input_layernorm(residual)
        attn_out = dp_attention_prefill(
            h, rope_mats, self.attn_weights, self.tp.self_attn, kv_cache, page_table, batch_idx, num_links, exact
        )
        # As DecoderLayer: the bf8 attention output is added into a bf16 residual (L <= 32K).
        hidden = ttnn.add(residual, attn_out, dtype=ttnn.bfloat16)
        attn_out.deallocate(True)
        residual.deallocate(True)
        h2 = self.tp.post_attention_layernorm(hidden)
        idx, w = self.router(h2, True)
        x_e = (
            h2
            if expert_input_dtype is None or h2.dtype == expert_input_dtype
            else ttnn.typecast(h2, expert_input_dtype)
        )
        moe = tp_exact_sum(self.experts.call_groups(x_e, idx, w), num_links) if exact else self.experts(x_e, idx, w)
        ttnn.deallocate(idx)
        ttnn.deallocate(w)
        if x_e is not h2:
            x_e.deallocate(True)
        h2.deallocate(True)
        out = ttnn.add(hidden, moe, output_tensor=moe)
        hidden.deallocate(True)
        return out


class DPPrefillStack:
    """The 24-layer DP prefill stack of a TP Model (built by Model when GPT_OSS_DP_PREFILL=1)."""

    def __init__(self, model, state_dict=None):
        mesh_device = model.mesh_device
        assert dp_prefill_mesh_supported(mesh_device), f"DP prefill runs on Blackhole {DP_PREFILL_MESH} only"
        self.model = model
        self.mesh_device = mesh_device
        self.num_devices = mesh_device.shape[1]
        self.layers = []
        for i, layer in enumerate(model.layers):
            experts_sd = None
            if state_dict:
                prefix = f"model.layers.{i}.mlp.experts."
                experts_sd = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
            self.layers.append(DPDecoderLayer(layer, experts_sd))
        self.num_links = batched_reduce_links(model.ccl_manager, 1)
        self.batch_idx = fill_batch_idx(model.ccl_manager, self.num_devices)
        logger.info(f"DP prefill stack built: {len(self.layers)} layers, a2a links {self.num_links}")

    def _rope(self, seq_len):
        model = self.model
        rope = model._prefill_rope_slices.get(seq_len)
        if rope is None:
            rope = [
                model.rope_setup.cos_matrix_prefill[:, :, :seq_len, :],
                model.rope_setup.sin_matrix_prefill[:, :, :seq_len, :],
            ]
            model._prefill_rope_slices[seq_len] = rope
        return rope

    def _expert_input_dtype(self, seq_len):
        """bf8 where the TP batch-1 prefill of this length runs sequence-parallel (its experts read the bf8-gathered
        norm rows, MLP.prefill_sequence_parallel), else None (the norm output as produced)."""
        from .mlp import PREFILL_SP_EXPERT_INPUT_DTYPE
        from .model import PREFILL_SP_ENV, PREFILL_SP_MIN_SEQ

        if os.getenv(PREFILL_SP_ENV, "1") == "1" and seq_len >= PREFILL_SP_MIN_SEQ:
            return PREFILL_SP_EXPERT_INPUT_DTYPE
        return None

    def step_users(self, step, num_users):
        """(user ids of devices 0..N-1 in this step, which of them are real)."""
        users = [step * self.num_devices + c for c in range(self.num_devices)]
        return users, [u < num_users for u in users]

    def forward_step(self, tokens, page_table, kv_cache, seq_len, exact=None):
        """One step: tokens [N, L] host (row c = user of device c), page_table [N, blocks] host. Returns the final
        residual [1, 1, L, H] bf8 per device."""
        mesh_device = self.mesh_device
        n = self.num_devices
        tt_tok = ttnn.from_torch(
            tokens.reshape(n, 1, 1, seq_len).to(torch.int32),
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 0), mesh_shape=mesh_device.shape),
        )
        h = ttnn.embedding(tt_tok, self.model.embedding_weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
        tt_tok.deallocate(True)
        if len(h.shape) == 3:
            h = ttnn.unsqueeze_to_4D(h)
        tt_pt = ttnn.from_torch(
            page_table.to(torch.int32),
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        rope = self._rope(seq_len)
        x_dtype = self._expert_input_dtype(seq_len)
        exact = dp_prefill_exact() if exact is None else exact
        assert not exact or seq_len % (ttnn.TILE_SIZE * self.num_devices) == 0, f"exact DP prefill needs L % 128 == 0"
        for i, layer in enumerate(self.layers):
            h = layer(
                h, rope, kv_cache[i], tt_pt, self.batch_idx, self.num_links, expert_input_dtype=x_dtype, exact=exact
            )
        tt_pt.deallocate(True)
        return h

    def prefill(self, tokens, page_table, kv_cache, prompt_lens=None, page_table_plant=None):
        """DP prefill of B users (B <= 32 is the intended use; any B works).

        tokens: [B, L] host, every row padded to the same L (L % 64 == 0, 128 <= L <= 32K); page_table: [B, >= blocks]
        host, row u = user u's pages (the TP cache layout); kv_cache: per layer [k, v] of the TP model; prompt_lens:
        real lengths (default L). Returns last-token logits [B, vocab] (float, host).

        page_table_plant (tests only): fn(step, page_table [N, blocks]) -> page table actually used for that step."""
        num_users, seq_len = tokens.shape
        n = self.num_devices
        assert seq_len % 64 == 0 and 128 <= seq_len <= DP_PREFILL_MAX_SEQ, f"DP prefill length {seq_len}"
        prompt_lens = [seq_len] * num_users if prompt_lens is None else [int(p) for p in prompt_lens]
        assert len(prompt_lens) == num_users and all(0 < p <= seq_len for p in prompt_lens), prompt_lens
        block = kv_cache[0][0].shape[2]
        blocks = math.ceil(seq_len / block)
        pt = page_table[:num_users, :blocks].clone().to(torch.int32)
        for u, p in enumerate(prompt_lens):
            # Only the blocks of the real prompt belong to this request (as the batch-1 generator path).
            pt[u, math.ceil(p / block) :] = -1
        out_rows = -(-num_users // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        mesh_device = self.mesh_device
        acc = None
        for step in range(-(-num_users // n)):
            users, real = self.step_users(step, num_users)
            first_real = users[0]
            tok_s = torch.stack([tokens[u if r else first_real] for u, r in zip(users, real)])
            pt_s = torch.stack([pt[u] if r else torch.full_like(pt[0], -1) for u, r in zip(users, real)])
            if page_table_plant is not None:
                pt_s = page_table_plant(step, pt_s)
            h = self.forward_step(tok_s, pt_s, kv_cache, seq_len)
            # Last-token rows: device c puts its user's last-token row at row u of a [1, 1, out_rows, H] tile.
            sel = torch.zeros(n, 1, out_rows, seq_len, dtype=torch.float32)
            for c, (u, r) in enumerate(zip(users, real)):
                if r:
                    sel[c, 0, u, prompt_lens[u] - 1] = 1.0
            sel_tt = ttnn.from_torch(
                sel,
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
            )
            picked = ttnn.matmul(
                sel_tt,
                h,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    math_approx_mode=False,
                    fp32_dest_acc_en=True,
                    packer_l1_acc=False,
                ),
            )
            sel_tt.deallocate(True)
            h.deallocate(True)
            if acc is None:
                acc = picked
            else:
                acc = ttnn.add(acc, picked, output_tensor=acc)
                picked.deallocate(True)
        rows = ttnn.all_reduce(acc, topology=ttnn.Topology.Ring, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        acc.deallocate(True)
        # The TP head reads the bf8 last-token rows (last_token_rows keeps the residual dtype); these rows are bf8
        # values, so the cast back is exact.
        rows_b8 = ttnn.typecast(rows, ttnn.bfloat8_b)
        rows.deallocate(True)
        normed = self.model.norm(rows_b8)
        rows_b8.deallocate(True)
        logits = ttnn.matmul(normed, self.model.lm_head_weight, dtype=ttnn.bfloat8_b)
        normed.deallocate(True)
        out = self.model.process_output_prefill_rows(logits, list(range(num_users)))
        return torch.cat([o.reshape(1, -1) for o in out], dim=0).float()
