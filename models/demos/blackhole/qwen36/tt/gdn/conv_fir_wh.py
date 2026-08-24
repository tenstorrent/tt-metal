# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Wormhole variant of the shared GDN depthwise causal conv1d FIR.

WHY THIS FILE EXISTS
--------------------
``_causal_conv1d_fir`` in
``models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_deltanet.py`` builds its padded
input ``x_padded = concat([carry, x], dim=1)`` in TILE layout, then reads K shifted windows
``x_padded[:, k:k+T]`` out of it. Those windows start at rows 1..K-1 of the sequence dimension and
so are NEVER tile-aligned, and ttnn has no sub-tile row-shift on TILE data (``ttnn.roll`` falls back
to untilize/roll/tilize unless the shift is a multiple of 32). Each window therefore untilizes the
WHOLE ``[B, (K-1)+T, D]`` tensor independently.

Measured on an N300 at T=2048, D=4096 (single-layer GDN prefill, tt-perf-report):

    concat prologue      untilize 229us + concat 201us + tilize 253us
    taps 1..3            3 x (untilize 264us + slice 212us + tilize 190us)
    ---------------------------------------------------------------------
    UntilizeWithUnpadding totalled 1,033us of a 21,669us layer, ~800us of it redundant

Building ``x_padded`` in ROW_MAJOR off ONE untilize and letting each tap tilize only its own slice
is the same arithmetic in the same order (multiply / addcmul against the same taps), and measured
UntilizeWithUnpadding 1,033us -> 15us. Layer total 19,941us -> 19,144us on the GDN prefill profile.

The layout of ``x_padded`` is a local inside that function, so there is no narrow seam to override
-- hence this copy, kept in this model's folder so the shared module is not edited (other models
import that function too).

BLACKHOLE IS NEVER AFFECTED
---------------------------
``causal_conv1d_fir_dispatch`` delegates to the original upstream function whenever
``is_blackhole()``, so Blackhole always executes upstream code, not this copy, even as upstream
evolves. Wormhole is the only caller of the body below.

MAINTENANCE
-----------
The body below is a verbatim copy of upstream at the commit this was written against, with two
changes, each marked in-line:

1. "THE ONE CHANGE vs upstream" -- x_padded is built in ROW_MAJOR instead of TILE, and
   correspondingly each tap tilizes its own slice instead of untilizing the whole tensor.
2. "SECOND CHANGE vs upstream" -- the last tap reuses ``x`` directly instead of slicing
   x_padded[K-1 : K-1+T], which is the same rows by construction. Saves a slice + a tilize.

``_check_fork_is_current()`` asserts the upstream anchor still exists so drift fails loudly instead
of silently running stale code.
"""
import inspect

import torch

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.experimental.gated_attention_gated_deltanet.tt import ttnn_gated_deltanet as _shared
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_deltanet import _causal_conv1d_decode_t1

# The TILE-layout concat this file exists to replace. If upstream restructures it (e.g. adopts the
# ROW_MAJOR form), this copy is stale -- fail loudly, not run old logic.
_UPSTREAM_ANCHOR = "x_padded = ttnn.concat([pad, x], dim=1, memory_config=mc)"

# Largest per-tap tilized window (bytes) allowed in L1; above this the tap
# falls back to the caller's memory config. 16MB == one [1, 2048, 4096] bf16
# window, the production chunk-outer prefill length. Sized to the transient
# window ALONE -- freed inside the loop iteration, never coexisting with the
# (contrast the 8MB threshold on the long-lived norm tensors in gdn/tp.py).
_TAP_L1_BUDGET = 16 << 20


def _check_fork_is_current():
    try:
        upstream_src = inspect.getsource(_shared._causal_conv1d_fir)
    except (OSError, TypeError):  # source unavailable (zipimport etc.) -- skip the check
        return
    if _UPSTREAM_ANCHOR not in upstream_src:
        raise RuntimeError(
            "models/demos/blackhole/qwen36/tt/gdn/conv_fir_wh.py is a copy of _causal_conv1d_fir() "
            "from ttnn_gated_deltanet.py, and upstream has changed: the anchor "
            f"{_UPSTREAM_ANCHOR!r} is gone. Re-copy that function into this file and re-apply the "
            "single ROW_MAJOR x_padded edit (marked 'THE ONE CHANGE vs upstream')."
        )


def causal_conv1d_fir_dispatch(
    x,
    weight,
    bias,
    kernel_size,
    device,
    memory_config=None,
    conv_state=None,
    weight_taps=None,
    bias_dev=None,
    valid_len=None,
    model_args=None,
):
    """Drop-in for the shared ``_causal_conv1d_fir``; upstream on Blackhole, ROW_MAJOR fork on WH.

    model_args: the caller's ModelArgs (gdn/tp.py passes self.args), used to scope the fork to
    wh_9b_n300. None (default, for any caller that hasn't been updated) falls back to the previous
    is_blackhole()-only decision.

    DELIBERATE narrowing (Wormhole gating audit, item 1): this used to be is_blackhole()-gated
    (Wormhole-wide), so T3K and N150 got the ROW_MAJOR fork too. Narrowed to wh_9b_n300 on purpose
    -- the fork is a pure perf optimization (same taps, same accumulation order, only the
    intermediate layout changes; see this file's module docstring), so T3K/N150 falling back to
    the upstream TILE-layout path is a measured perf regression there, not a correctness one.
    Accepted per explicit instruction -- don't revert this to is_blackhole() without re-measuring
    on T3K/N150.
    """
    _use_wh = tpc.wh_9b_n300(model_args) if model_args is not None else (not is_blackhole())
    if not _use_wh:
        return _shared._causal_conv1d_fir(
            x,
            weight,
            bias,
            kernel_size,
            device,
            memory_config=memory_config,
            conv_state=conv_state,
            weight_taps=weight_taps,
            bias_dev=bias_dev,
            valid_len=valid_len,
        )
    _check_fork_is_current()
    return _causal_conv1d_fir_wh(
        x,
        weight,
        bias,
        kernel_size,
        device,
        memory_config=memory_config,
        conv_state=conv_state,
        weight_taps=weight_taps,
        bias_dev=bias_dev,
        valid_len=valid_len,
    )


def _causal_conv1d_fir_wh(
    x,
    weight,
    bias,
    kernel_size,
    device,
    memory_config=None,
    conv_state=None,
    weight_taps=None,
    bias_dev=None,
    valid_len=None,
):
    """Depthwise causal conv1d + SiLU via K shifted multiply-accumulate slices.

    x [B,T,D]; conv_state [B,K-1,D] or list of [B,1,D]; weight_taps/bias_dev optional.
    Returns output [B,T,D], new_state [B,K-1,D].
    """
    mc = memory_config
    B, T, D = x.shape[0], x.shape[1], x.shape[2]

    # Fast path: T=1 decode with state + pre-sliced taps
    if T == 1 and conv_state is not None and weight_taps is not None:
        return _causal_conv1d_decode_t1(
            x, conv_state, kernel_size, device, memory_config=mc, weight_taps=weight_taps, bias_dev=bias_dev
        )

    # ---- THE ONE CHANGE vs upstream: build x_padded in ROW_MAJOR, not TILE. ----
    # Upstream concatenates in TILE, which makes each of the K shifted windows below untilize the
    # whole [B,(K-1)+T,D] tensor (they start at rows 1..K-1, never tile-aligned). One untilize here
    # instead of the concat's tilize + 3 full untilizes; each tap tilizes only
    # slice. Same ops, same order, same taps -- just fewer whole-tensor relayouts.
    _rm = ttnn.ROW_MAJOR_LAYOUT
    x_rm = ttnn.to_layout(x, _rm, memory_config=mc)
    if conv_state is not None:
        # conv_state is the previous chunk's last K-1 rows, in TILE; only K-1 rows, so cheap.
        cs_rm = ttnn.to_layout(conv_state, _rm, memory_config=mc)
        x_padded = ttnn.concat([cs_rm, x_rm], dim=1, memory_config=mc)
        if cs_rm is not conv_state:
            ttnn.deallocate(cs_rm)
    else:
        pad = ttnn.zeros(
            [B, kernel_size - 1, D],
            device=device,
            dtype=ttnn.bfloat16,
            layout=_rm,
            memory_config=mc,
        )
        x_padded = ttnn.concat([pad, x_rm], dim=1, memory_config=mc)
        ttnn.deallocate(pad)
    if x_rm is not x:
        ttnn.deallocate(x_rm)
    # ---- end of the change; everything below is upstream. ----

    # new_state: last K-1 tokens; land in DRAM (carry alive across downstream kernel CBs).
    total_len = (kernel_size - 1) + T
    if valid_len is None:
        new_state = x_padded[:, total_len - (kernel_size - 1) :, :]
        # to_layout then to_memory_config: slice keeps L1 if memory_config passed to to_layout
        new_state = ttnn.to_layout(new_state, ttnn.TILE_LAYOUT)
        new_state = ttnn.to_memory_config(new_state, ttnn.DRAM_MEMORY_CONFIG)
    else:
        # Fixed-bucket masking: x is right-padded to a bucket length T but only the first
        # valid_len positions are real; the decode conv window must come from the real tail
        # x[valid_len-(K-1):valid_len], i.e. x_padded[:, valid_len : valid_len+(K-1)] (x[i]
        # is at x_padded index (K-1)+i). A static slice there would compile a new program per
        # valid_len value — defeating the bounded-program goal — so select those rows with a
        # one-hot matmul instead: the program depends only on shapes (fixed per bucket), and
        # only the one-hot VALUES depend on valid_len.
        # valid_len may be a scalar (all B rows) or a per-row list/tuple of
        # B (batched prefill: each user's own real length picks that user's decode conv window).
        sel = torch.zeros(B, kernel_size - 1, total_len, dtype=torch.float32)
        if isinstance(valid_len, (list, tuple)):
            for bi in range(B):
                for j in range(kernel_size - 1):
                    sel[bi, j, int(valid_len[bi]) + j] = 1.0
        else:
            for j in range(kernel_size - 1):
                sel[:, j, valid_len + j] = 1.0
        sel_tt = ttnn.from_torch(sel, dtype=x_padded.dtype, layout=ttnn.TILE_LAYOUT, device=device)
        xp = ttnn.to_layout(x_padded, ttnn.TILE_LAYOUT)
        # cross-chunk carry -> DRAM
        new_state = ttnn.matmul(sel_tt, xp, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(sel_tt)

    # Precompute weight taps if not provided
    if weight_taps is None:
        weight_torch = ttnn.to_torch(weight)
        weight_taps = []
        for k in range(kernel_size):
            w_k = weight_torch[:, 0, k].reshape(1, 1, D).contiguous()
            weight_taps.append(
                ttnn.from_torch(w_k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
            )

    total_len = (kernel_size - 1) + T
    _dram = ttnn.DRAM_MEMORY_CONFIG
    # Depthwise K-tap FIR via multiply + addcmul. x_padded is ROW_MAJOR, so each
    # tap tilizes its own slice rather than untilizing the whole tensor.
    #
    # SECOND CHANGE vs upstream: the LAST tap needs no slice and no tilize.
    #   x_padded[K-1 : K-1+T] == x
    # Its window is `x` by construction (x_padded is [carry(K-1) | x(T)]), and
    # `x` is still live in TILE layout -- what we untilized to build x_padded.
    # Feed it straight in. At T=2048, D=4096 on N300 that drops a 211us slice
    # ~400us/layer. Accumulation order is untouched (still k = 0,1,..,K-1), so the result is unchanged.
    # THIRD CHANGE vs upstream: land each tap's tilized window in L1, not `mc`.
    # The window is transient — produced, consumed by the multiply/addcmul, and freed inside one
    # iteration, so unlike the long-lived tensors that forced DRAM here it never
    # chunk kernel's circular buffers. Measured at T=2048, D=4096 on N300: Tilize 565->454us (it
    # writes L1 instead of DRAM) and the addcmuls 948->843us (they read L1), together -206us.
    # Guarded by size: only when one window fits _TAP_L1_BUDGET, else use `mc`.
    # that covers chunks up to T=2048 (the production chunk-outer length) and declines gracefully above.
    _tap_mc = ttnn.L1_MEMORY_CONFIG if (T * D * 2) <= _TAP_L1_BUDGET else mc
    out = None
    for k in range(kernel_size):
        if k == kernel_size - 1:
            x_slice = x  # == x_padded[K-1 : K-1+T], already TILE
        else:
            # The ROW_MAJOR window stays in `mc` (DRAM). MEASURED: putting it in L1 too
            # explicit ttnn.slice with memory_config) makes the slice 262us cheaper but the tilize 249us
            # dearer (an L1->L1 tilize beats DRAM->L1 here): net -38us, i.e. noise.
            # Not worth the extra slice+deallocate, so only the tilize OUTPUT is L1 (see _tap_mc).
            x_slice = ttnn.to_layout(x_padded[:, k : k + T], ttnn.TILE_LAYOUT, memory_config=_tap_mc)
        if out is None:
            out = ttnn.multiply(x_slice, weight_taps[k], memory_config=mc)
        else:
            out = ttnn.addcmul(out, x_slice, weight_taps[k], memory_config=mc)

    # Bias (+ fused SiLU if bias present) else standalone SiLU. Output in DRAM.
    _silu = [ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)]
    if bias_dev is not None:
        return ttnn.add(out, bias_dev, activations=_silu, memory_config=_dram), new_state
    if bias is not None:
        bias_torch = ttnn.to_torch(bias).reshape(1, 1, D).contiguous()
        bias_dev_tmp = ttnn.from_torch(
            bias_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc
        )
        return ttnn.add(out, bias_dev_tmp, activations=_silu, memory_config=_dram), new_state
    # Conv output in DRAM (feeds gated_delta_attn_seq; MAC still ran in L1 when mc=L1)
    return ttnn.silu(out, memory_config=_dram), new_state
