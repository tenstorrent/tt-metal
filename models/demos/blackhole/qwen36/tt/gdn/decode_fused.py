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
layers + one slice/copy per layer) or "perlayer" (default: the 8-op chain per layer).
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
    assert v in ("batched", "perlayer"), f"QWEN36_GDN_CONV_REPACK must be batched|perlayer (got {v!r})"
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
