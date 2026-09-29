# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused GDN decode step: QWEN36_GDN_DECODE_FUSED=2 (single device, B = 1). DEFAULT ON (unset = "2"; user
decision 2026-09-25, INT-2); QWEN36_GDN_DECODE_FUSED=0 restores the composite decode path and BF16 GDN state.

One GDN decode layer becomes: mega in-proj linear -> ttnn.experimental.kda.gdn_decode_step (fused-conv
mode: 4-tap causal conv + SiLU, beta/decay gates, L2 norms, delta-rule recurrence, gated RMSNorm, silu(z)
output gate, in-place history shift) -> out-proj linear. It replaces the ~70-op composite glue of
gated_deltanet_forward_ttnn's T == 1 path.

State the op needs (all allocated outside any trace, addresses fixed):
  * recurrent_state: FP32 [1, Nv, Dk, Dv], updated in place by the op. With the flag on, every GDN
    recurrent-state buffer is allocated FP32 (external buffers, reset zeros, eager init), so prefill
    writes FP32 directly and no state typecast runs per chunk.
  * conv_hist: packed BF16 TILE [1, Nv, 4, 32, 32]; tile (h, s) row 2c = 32-channel chunk c of head h's
    [q_h | k_h | v_h] in history slot s (slot 3 newest, slot 0 dead). The op shifts it in place. It is
    rebuilt ON DEVICE from fused_conv_state [1, 3, C] (the prefill output) after the last prefill writer,
    after a state restore and on reset -- see Qwen36Model._gdn_refresh_conv_hist. The op never updates
    fused_conv_state, so fused_conv_state is stale during fused decode (nothing reads it there).
  * conv_taps: packed BF16 TILE [Nv, 4, 32, 32] (both row parities), built once on the host at init.

Supported shapes: rf = Nv / Nk = 1, Dk == Dv, 2 * Nv <= 32, qkvz tile aligned (Qwen3.5-2B: Nv = Nk = 16,
Dk = Dv = 128). Other configs warn and keep the composite path.

QWEN36_GDN_CONV_REPACK selects the conv_hist rebuild: "batched" (one op chain over all GDN
layers + one slice/copy per layer), "gather" (R12: one ttnn.embedding per layer against a shared
index table, built once, in place of the per-layer slice/copy -- see repack_conv_hist_gather()), or
"perlayer" (default: the 8-op chain per layer).
"""
import os

import torch
from loguru import logger

import ttnn

TILE = 32


DECODE_FUSED_DEFAULT = "2"  # user decision 2026-09-25: fused GDN decode on by default; "0" = composite path


def decode_fused_enabled():
    return os.environ.get("QWEN36_GDN_DECODE_FUSED", DECODE_FUSED_DEFAULT) == "2"  # any other value = composite


def repack_variant():
    v = os.environ.get("QWEN36_GDN_CONV_REPACK", "perlayer")
    assert v in ("batched", "perlayer", "gather"), f"QWEN36_GDN_CONV_REPACK must be batched|perlayer|gather (got {v!r})"
    return v


def fused_supported(cfg, weights):
    """(ok, reason) for the op contract + this module's packing (rf = 1, Dk == Dv)."""
    Nv, Nk, Dk, Dv = cfg.num_v_heads, cfg.num_heads, cfg.head_k_dim, cfg.head_v_dim
    if Nv != Nk:
        return False, f"needs Nv == Nk (rf = 1), got Nv={Nv} Nk={Nk}"
    if Dk != Dv or Dk % TILE:
        return False, f"needs Dk == Dv (tile aligned), got Dk={Dk} Dv={Dv}"
    if 2 * Nv > TILE:
        return False, f"needs 2*Nv <= 32 (a|b in one tile), got Nv={Nv}"
    if 2 * ((2 * Dk + Dv) // TILE) > TILE:
        return False, "packed head row needs <= 16 chunks"
    if cfg.conv_kernel_size != 4:
        return False, f"needs conv_kernel_size == 4, got {cfg.conv_kernel_size}"
    if weights.mega_fused_weight is None or weights.fused_conv_bias_dev is not None:
        return False, "needs the mega-fused [q|k|v|z|a|b] in-proj weight and no conv bias"
    qkvz = 2 * Nk * Dk + 2 * Nv * Dv
    if weights.mega_qkv_dim + weights.mega_g_dim != qkvz or qkvz % TILE:
        return False, f"mega layout mismatch (qkv {weights.mega_qkv_dim} + g {weights.mega_g_dim} != {qkvz})"
    return True, ""


def _pack_rows_host(rows, Nv, Dk, both_parity):
    """4 x [C] rows (slot 0 oldest) -> [Nv, 4, 32, 32] bf16; tile (h, s) row 2c (+1) = chunk c of head h."""
    nck = 3 * Dk // TILE  # chunks per head ([q | k | v], rf = 1, Dk == Dv)
    x = torch.stack([r.reshape(-1).to(torch.bfloat16) for r in rows])  # [4, C]
    x = x.reshape(4, 3, Nv, Dk).permute(2, 0, 1, 3).reshape(Nv, 4, nck, TILE)
    out = torch.zeros(Nv, 4, TILE, TILE, dtype=torch.bfloat16)
    out[:, :, 0 : 2 * nck : 2, :] = x
    if both_parity:
        out[:, :, 1 : 2 * nck : 2, :] = x
    return out


def pack_conv_taps(cfg, fused_conv_weight_taps, device):
    """Host pack (once, at init) of the 4 x [1, 1, C] conv taps (tap 3 = newest token)."""
    taps = [ttnn.to_torch(t).reshape(-1) for t in fused_conv_weight_taps]
    packed = _pack_rows_host(taps, cfg.num_v_heads, cfg.head_k_dim, both_parity=True)
    return ttnn.from_torch(
        packed, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def alloc_conv_hist(cfg, device):
    return ttnn.from_torch(
        torch.zeros(1, cfg.num_v_heads, 4, TILE, TILE, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _pack_chain(x, lead, Nv, Dk):
    """ROW_MAJOR [lead..., 3, C] -> TILE [lead * Nv, 4, 32, 32]-shaped packed history (slot 0 = 0)."""
    nck = 3 * Dk // TILE
    n = 1
    for d in lead:
        n *= d
    if n == 1:
        x = ttnn.reshape(x, [3, 3, Nv, Dk])  # slot, {q,k,v}, head, channel
        x = ttnn.permute(x, (2, 0, 1, 3))  # [Nv, 3, 3, Dk]
    else:
        x = ttnn.reshape(x, [n, 3, 3, Nv, Dk])
        x = ttnn.permute(x, (0, 3, 1, 2, 4))  # [n, Nv, 3, 3, Dk]
    x = ttnn.reshape(x, [n * Nv, 3, nck, TILE])  # chunk c = 32 channels
    # one pad: dead slot 0 in front, chunks nck -> 16, columns 32 -> 64 (row 2c + 1 = 0)
    x = ttnn.pad(x, [(0, 0), (1, 0), (0, TILE // 2 - nck), (0, TILE)], value=0.0)  # [n*Nv, 4, 16, 64]
    x = ttnn.reshape(x, [n, Nv, 4, TILE, TILE])
    return ttnn.to_layout(x, ttnn.TILE_LAYOUT)


def repack_conv_hist(fcs, conv_hist, cfg):
    """Per layer, 8 device ops, no host reads (trace safe): fused_conv_state [1, 3, C] -> conv_hist in place."""
    x = ttnn.to_layout(fcs, ttnn.ROW_MAJOR_LAYOUT)
    x = _pack_chain(x, [1], cfg.num_v_heads, cfg.head_k_dim)
    ttnn.copy(x, conv_hist)
    ttnn.deallocate(x)


def repack_conv_hist_batched(pairs, cfg):
    """All layers at once: concat -> one pack chain -> per layer slice + copy (1 + 7 + 2n device ops)."""
    if len(pairs) == 1:
        return repack_conv_hist(pairs[0][0], pairs[0][1], cfg)
    n = len(pairs)
    x = ttnn.concat([f for f, _ in pairs], dim=0)  # [n, 3, C] TILE
    xr = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.deallocate(x)
    packed = _pack_chain(xr, [n], cfg.num_v_heads, cfg.head_k_dim)  # [n, Nv, 4, 32, 32] TILE
    Nv = cfg.num_v_heads
    for i, (_, hist) in enumerate(pairs):
        s = ttnn.slice(packed, [i, 0, 0, 0, 0], [i + 1, Nv, 4, TILE, TILE])
        ttnn.copy(s, hist)
        ttnn.deallocate(s)
    ttnn.deallocate(packed)


# R12 (analysis_50ms/R_small_items_spec.md): one shared index table + one ttnn.embedding per layer,
# in place of repack_conv_hist_batched's per-layer slice + copy. Keyed by (layer count, Nv, Dk) --
# not by device, since this module already assumes a single device throughout (see
# Qwen36Model._gdn_refresh_conv_hist: `if self.num_devices > 1: return`) -- so it is built once per
# shape (before any trace capture, on the first repack call) and reused: same discipline as conv_hist
# / conv_taps, fixed addresses, no host reads on later calls.
_gather_cache = {}


def _gather_cache_key(cfg, n_layers):
    return (n_layers, cfg.num_v_heads, cfg.head_k_dim)


def _build_gather_cache(fcs0, cfg, n_layers, device):
    """One-time host + device build of the gather index tensors and the shared all-zero fcs block.

    Table row for (layer l, head h, slot s, tile-row r) of conv_hist: with c = r // 2,
      l * rows_per_layer + (s - 1) * slot_stride + (c // cph) * (Nv * cph) + h * cph + (c % cph)
    matching _pack_chain's layout (row 2c of tile (h, s) = channel-chunk c of head h, slot s; q chunks
    0 : cph, k chunks cph : 2*cph, v chunks 2*cph : 3*cph). Dead slot 0, odd r, and r >= 2 * nck (the
    q|k|v-to-16-chunk pad) instead point at a shared all-zero block appended after the last real
    layer's rows (pad_token = n_layers * rows_per_layer): any row in that block works, since it is
    all zero, so one padding_idx value covers every padded position.
    """
    Nv, Dk = cfg.num_v_heads, cfg.head_k_dim
    cph = Dk // TILE  # channel-chunks per head per q/k/v section (4 for Dk = 128)
    nck = 3 * cph  # channel-chunks per head across q|k|v (12)
    C = 3 * Nv * Dk  # fused_conv_state's raw per-slot channel width (6144 for Nv=16, Dk=128)
    rows_per_layer = 3 * C // TILE  # a layer's [3, C] block flattened to rows of width TILE (576)
    slot_stride = C // TILE  # rows contributed by one slot (192)
    pad_token = n_layers * rows_per_layer
    out_rows = Nv * 4 * TILE  # conv_hist flattened to [1, 1, out_rows, TILE] (2048)

    idx_dev = []
    for l in range(n_layers):
        idx = torch.full((1, out_rows), pad_token, dtype=torch.int64)
        for h in range(Nv):
            for s in range(1, 4):  # s == 0 (dead slot) stays padded
                for r in range(0, 2 * nck, 2):  # odd r and r >= 2 * nck stay padded
                    c = r // 2
                    group = (c // cph) * (Nv * cph) + h * cph + (c % cph)
                    idx[0, h * (4 * TILE) + s * TILE + r] = l * rows_per_layer + (s - 1) * slot_stride + group
        idx_dev.append(ttnn.from_torch(idx, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device))

    return {
        "idx": idx_dev,
        "zero_fcs": ttnn.zeros_like(fcs0),
        "pad_token": pad_token,
        "rows_per_layer": rows_per_layer,
        "out_rows": out_rows,
    }


def repack_conv_hist_gather(pairs, cfg, device):
    """All layers at once: concat (+ shared zero block) -> untilize -> one reshape -> one
    ttnn.embedding per layer straight into conv_hist (3 + n device ops, vs batched's 3 + 2n)."""
    n = len(pairs)
    if n == 0:
        return
    key = _gather_cache_key(cfg, n)
    cache = _gather_cache.get(key)
    if cache is None:
        cache = _build_gather_cache(pairs[0][0], cfg, n, device)
        _gather_cache[key] = cache

    t = ttnn.concat([f for f, _ in pairs] + [cache["zero_fcs"]], dim=0)  # [n + 1, 3, C] TILE
    tr = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.deallocate(t)
    tbl = ttnn.reshape(tr, [(n + 1) * cache["rows_per_layer"], TILE])
    for i, (_, hist) in enumerate(pairs):
        out = ttnn.reshape(hist, [1, 1, cache["out_rows"], TILE])  # metadata only (same tile order)
        ttnn.embedding(cache["idx"][i], tbl, layout=ttnn.TILE_LAYOUT, padding_idx=cache["pad_token"], output_tensor=out)
    ttnn.deallocate(tbl)


def _l1_debug(gdn):
    """QWEN36_GDN_DECODE_FUSED_L1DBG=1: print the L1 allocator view right before the op (eager runs)."""
    try:
        mv = ttnn.get_memory_view(gdn.device, ttnn.BufferType.L1)
        base = ttnn.get_allocator_base_address(gdn.device, ttnn.BufferType.L1)
        logger.info(
            f"[GDN_FUSED_L1] banks={mv.num_banks} per_bank total={mv.total_bytes_per_bank} "
            f"allocated={mv.total_bytes_allocated_per_bank} free={mv.total_bytes_free_per_bank} "
            f"largest_free={mv.largest_contiguous_bytes_free_per_bank} alloc_base={base} "
            f"(op DFBs need 698368 B per core on 16 cores)"
        )
    except Exception as e:  # noqa: BLE001
        logger.info(f"[GDN_FUSED_L1] memory view unavailable: {e!r}")


def fused_decode_forward(gdn, x):
    """x [1, 1, hidden] -> [1, 1, hidden]: mega linear -> gdn_decode_step -> out-proj (same linears,
    kernel configs, program configs and L1 outputs as the composite T == 1 path)."""
    w = gdn.weights
    cfg = gdn.cfg
    L1 = ttnn.L1_MEMORY_CONFIG
    ckc = gdn.compute_kernel_config_decode
    pc = getattr(gdn, "_decode_progcfg_fn", None)
    if gdn.conv_hist is None:  # eager-only fallback; the traced flows allocate + fill it up front
        gdn.refresh_conv_hist()
    mega = ttnn.linear(
        x,
        w.mega_fused_weight,
        memory_config=L1,
        compute_kernel_config=ckc,
        program_config=pc(x.shape[-1], w.mega_fused_weight.shape[-1]) if pc is not None else None,
    )
    if os.environ.get("QWEN36_GDN_DECODE_FUSED_L1DBG") == "1":
        _l1_debug(gdn)
    o = ttnn.experimental.kda.gdn_decode_step(
        mega,
        w.dt_bias,
        w.A_neg,
        gdn.recurrent_state,
        w.o_norm_weight,
        cfg.num_v_heads,
        cfg.num_heads,
        cfg.head_k_dim,
        cfg.head_v_dim,
        scale=cfg.head_k_dim**-0.5,
        l2_epsilon=1e-6,
        norm_epsilon=gdn.norm_eps,
        memory_config=L1,
        output_dtype=ttnn.bfloat16,
        conv_hist=gdn.conv_hist,
        conv_taps=gdn.conv_taps_packed,
        qkvz_dim=w.mega_qkv_dim + w.mega_g_dim,
    )
    ttnn.deallocate(mega)
    y = ttnn.linear(
        o,
        w.o_proj_weight,
        memory_config=L1,
        compute_kernel_config=ckc,
        program_config=pc(o.shape[-1], w.o_proj_weight.shape[-1]) if pc is not None else None,
    )
    ttnn.deallocate(o)
    return y


def fused_decode_applicable(gdn, x):
    rs = gdn.recurrent_state
    return (
        gdn._decode_fused
        and x.shape[0] == 1
        and x.shape[1] == 1
        and rs is not None
        and rs.dtype == ttnn.float32
        and rs.shape[0] == 1
    )


def warn_unsupported(reason):
    logger.warning(f"QWEN36_GDN_DECODE_FUSED=2 ignored for this GDN layer: {reason}")
