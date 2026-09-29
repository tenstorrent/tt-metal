# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native depthwise causal conv1d + SiLU for single-device GDN chunk prefill.

Port of TPGatedDeltaNet._conv1d_prefill: prepend the K-1 carry rows on host (native padding only zeros,
which cannot inject cross-chunk carry), run ttnn.conv1d HEIGHT_SHARDED with the L1-full slice config in
channel chunks (depthwise is per-channel independent, so chunk-then-concat is exact), interleave, tilize,
SiLU. Replaces the FIR fallback (4 shifted slices, 3 full re-tilizes, 1 mul + 3 addcmul) for widths > 2048.

F2 (2026-09-20): rewritten glue path to remove the tilize/untilize churn measured around the F1 conv
path (see gdn/weights.py precompute + profiling notes): exactly ONE full-size relayout (x -> ROW_MAJOR
L1) instead of several round trips through TILE/DRAM, and — when the channel split lines up with
Q/K/V exactly (n_cc=3, chunk width == q_dim == k_dim == v_dim) — the 3 conv outputs are returned
directly as (q, k, v), skipping the concat this file used to do and the post-conv split the caller
used to do (ttnn_gated_deltanet.py). SiLU is applied on the still-sharded conv output when possible
(one fewer op than SiLU-after-interleave); see QWEN36_GDN_CONV_SILU_SHARDED below if that ever needs
to be disabled for a new shape/dtype combination.
"""
import os

import ttnn

_L1 = ttnn.L1_MEMORY_CONFIG
_DRAM = ttnn.DRAM_MEMORY_CONFIG


def make_native_conv1d_fn(
    device,
    w1d_chunks,
    kernel_size,
    q_dim=None,
    k_dim=None,
    v_dim=None,
    max_channels_per_call=3072,
    t3_max=0,
    xin_l1_max_t=2048,
):
    """Build a chunk-prefill depthwise conv1d+SiLU callable for one GDN layer.

    w1d_chunks: dict {n_cc: [chunk tensors]} — HOST ttnn tensors [cw, 1, K] bf16 ROW_MAJOR, one
    list per channel-chunk-count this layer might use (e.g. `GDNWeights.fused_conv_w1d_chunks`;
    cw = C // n_cc within each list). A plain list is also accepted for back-compat (treated as
    the single entry {len(list): list}). Depthwise conv is per-channel-independent, so running
    each chunk through its own HEIGHT_SHARDED ttnn.conv1d call and concatenating the outputs is
    numerically exact — the split only exists so a single call's L1_FULL-sliced CBs fit L1 (the
    fused Q|K|V channel count overflows a single call on Blackhole).

    q_dim, k_dim, v_dim: per-stream channel widths. When a chunk width (n_cc=3) equals all three,
    the 3 conv outputs ARE q, k, v in order (fused conv weight is built q|k|v-concatenated) —
    fn() then returns them as a tuple instead of concatenating + making the caller re-split them.

    t3_max: largest T for which the n_cc=3 fast path is used (must fit L1 at that T — measured
    empirically, see scripts/step1/test_conv_native.py); n_cc=2 (or whichever wider default chunk
    width was precomputed) is used above t3_max. Default 0 (never use n_cc=3): measured on P150
    with C=6144 (cw=2048), the "Statically allocated circular buffers ... grow to 1684544 B ...
    beyond max L1 size of 1572864 B" overflow happens at EVERY T tested (512, 1024, 2048) with the
    identical byte counts each time — HEIGHT_SHARDED grows the core grid with T instead of growing
    per-core row count, so the per-core CB footprint (and thus the L1 fit) is driven by channel
    width alone, not by T. So there is no T at which n_cc=3 fits on this device/shape; only
    QWEN36_GDN_CONV_CHUNKS=3 (explicit override, for experimentation) or a smaller/differently
    architected shard config could revisit this. QWEN36_GDN_CONV_CHUNKS=<n> overrides n_cc
    unconditionally, at every T (must be a key in w1d_chunks).

    xin_l1_max_t: largest T for which the row-major glue tensors (x_rm/head/xin and the per-chunk
    conv1d input slices) are kept L1-resident; above it they fall back to DRAM (see fn() for why —
    an L1-resident xin at large Lin clashes with the conv1d's own sharded CBs). Default 2048 (F10A,
    step1 task G3 — raised from the legacy 1024 default for the T=2048 production chunk size; retest
    at T=2048/cw=3072 before trusting this, the code path changed since the original 1024 measurement
    in scripts/step1/baseline_short.py). QWEN36_GDN_CONV_XIN_L1_MAX_T overrides; "0" restores the
    legacy 1024 default exactly (not a literal 0, which would disable L1 glue placement below 1024
    too).

    Returns fn(x, conv_state, qkv_out_mc=None) -> (out, new_state):
      x: [1, T, C] TILE (any memory) — the fused Q|K|V projection output for one prefill chunk.
      conv_state: [1, K-1, C] TILE (any memory) cross-chunk carry, or None (first chunk — zero-padded).
      qkv_out_mc: step2 (2026-09-23) — optional override for the memory_config of `out`/(q, k, v).
        Default None keeps the legacy behavior (always L1-interleaved, _L1, regardless of T).
        Added so conv1d_kda.py's fused-op wrapper can force this function's OUTPUT to DRAM under
        QWEN36_GDN_FLA_INPUTS_DRAM when it uses this fn as ITS fallback (masked-tail valid_len, a
        non-tile-aligned/short T, or a channel-width mismatch — see conv1d_kda.fn's guard). Before
        this parameter existed, that fallback always left q/k/v in L1 even when T was well above
        xin_l1_max_t: e.g. model.prefill_paged's exact-length chunked reference
        (prefill_layer_chunked, chunk_size=2048) produces a non-tile-aligned remainder chunk (T not
        a multiple of 32) that hits this exact fallback; the resulting L1-resident q/k/v raised the
        allocator's L1 high-water mark just enough for a LATER, unrelated fused-KDA call's static
        CBs to clash by as little as 1024 B ("Statically allocated circular buffers ... clash with
        L1 buffers" — see qwen35_2b_handoff HANDOFF_2026-09-21.md, F13 row).
      out: SiLU(depthwise_conv1d(x)) [1, T, C] TILE at qkv_out_mc (default L1-interleaved) — OR,
        when the n_cc=3 fast path fires, a tuple (q, k, v) of SiLU(depthwise_conv1d(x)) slices,
        each TILE at qkv_out_mc.
      new_state: last K-1 rows of x (next chunk's carry) [1, K-1, C] TILE DRAM (unchanged contract
        — unaffected by qkv_out_mc, which only redirects the q/k/v OUTPUT).
    """
    K = kernel_size
    _xin_l1_max_t_env = int(os.environ.get("QWEN36_GDN_CONV_XIN_L1_MAX_T", str(xin_l1_max_t)))
    xin_l1_max_t = 1024 if _xin_l1_max_t_env == 0 else _xin_l1_max_t_env

    chunks_by_ncc = dict(w1d_chunks) if isinstance(w1d_chunks, dict) else {len(w1d_chunks): list(w1d_chunks)}
    assert chunks_by_ncc, "make_native_conv1d_fn needs at least one channel-chunk-width entry"

    Cs = {sum(w.shape[0] for w in chunks) for chunks in chunks_by_ncc.values()}
    assert len(Cs) == 1, f"all n_cc variants must cover the same total channel width C, got {Cs}"
    C = Cs.pop()
    for n_cc, chunks in chunks_by_ncc.items():
        assert C % n_cc == 0, f"GDN conv channels {C} not divisible by n_cc {n_cc}"
        cw = C // n_cc
        assert all(w.shape[0] == cw for w in chunks), f"n_cc={n_cc}: all channel chunks must be equal width"
        assert (
            cw <= max_channels_per_call
        ), f"channel chunk {cw} (n_cc={n_cc}) exceeds max_channels_per_call {max_channels_per_call}"

    override = os.environ.get("QWEN36_GDN_CONV_CHUNKS")
    override_n_cc = None
    if override is not None:
        override_n_cc = int(override)
        assert (
            override_n_cc in chunks_by_ncc
        ), f"QWEN36_GDN_CONV_CHUNKS={override!r} has no precomputed chunk list (have {list(chunks_by_ncc)})"

    # n_cc=3 qkv-tuple fast path: only valid when the 3 equal chunks line up exactly with Q/K/V.
    qkv_tuple_ncc = None
    if 3 in chunks_by_ncc:
        cw3 = C // 3
        if cw3 == q_dim == k_dim == v_dim:
            qkv_tuple_ncc = 3

    # Default changed 2026-09-20 (step1 Task C A/B): measured on P150 T=512, the HEIGHT_SHARDED
    # SiLU costs 94us per 3072-wide half (188us/GDN layer) vs 13.4us per half (26.7us/GDN layer)
    # after sharded_to_interleaved -- ~7x slower sharded, ~2.9ms slower over the whole T=512 eager
    # prefill (35.047ms -> 32.126ms total device kernel time). Default is now 0 (SiLU after
    # sharded_to_interleaved); QWEN36_GDN_CONV_SILU_SHARDED=1 restores the sharded SiLU for A/B.
    silu_on_sharded = os.environ.get("QWEN36_GDN_CONV_SILU_SHARDED", "0") != "0"

    cc = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    # Needs l1_small_size on the device (prefill/demo set 24576); matches the validated TP config.
    conv_cfg = ttnn.Conv1dConfig(
        weights_dtype=ttnn.bfloat16,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    )
    # Prepared weights depend on (n_cc, Lin) — a layer sees several T values in eager use (e.g.
    # different prefill chunk sizes) and may switch n_cc across calls (see _select_ncc), so both
    # must be part of the cache key.
    _wprep = {}

    def _prepare(n_cc, Lin):
        key = (n_cc, Lin)
        if key not in _wprep:
            cw = C // n_cc
            _wprep[key] = [
                ttnn.prepare_conv_weights(
                    weight_tensor=w,
                    input_memory_config=_DRAM,
                    input_layout=ttnn.ROW_MAJOR_LAYOUT,
                    weights_format="OIHW",
                    in_channels=cw,
                    out_channels=cw,
                    batch_size=1,
                    input_height=1,
                    input_width=Lin,
                    kernel_size=(1, K),
                    stride=(1, 1),
                    padding=(0, 0),
                    dilation=(1, 1),
                    has_bias=False,
                    groups=cw,
                    device=device,
                    input_dtype=ttnn.bfloat16,
                    conv_config=conv_cfg,
                    compute_config=cc,
                )
                for w in chunks_by_ncc[n_cc]
            ]
        return _wprep[key]

    def _select_ncc(T):
        if override_n_cc is not None:
            return override_n_cc
        if qkv_tuple_ncc is not None and T <= t3_max:
            return qkv_tuple_ncc
        others = [n for n in chunks_by_ncc if n != qkv_tuple_ncc]
        return min(others) if others else qkv_tuple_ncc

    def _conv_chunks(xin, Lin, T, n_cc, glue_mc, out_mc):
        cw = C // n_cc
        wpreps = _prepare(n_cc, Lin)
        outs = []
        for i, wprep in enumerate(wpreps):
            xin_i = (
                xin
                if n_cc == 1
                else ttnn.slice(xin, (0, 0, 0, i * cw), (1, Lin, 1, (i + 1) * cw), memory_config=glue_mc)
            )
            out_i = ttnn.conv1d(
                input_tensor=xin_i,
                weight_tensor=wprep,
                device=device,
                in_channels=cw,
                out_channels=cw,
                batch_size=1,
                input_length=Lin,
                kernel_size=K,
                stride=1,
                padding=0,
                dilation=1,
                groups=cw,
                dtype=ttnn.bfloat16,
                conv_config=conv_cfg,
                compute_config=cc,
                # L1_FULL slice: keep the conv in L1 instead of DRAM-width-slicing. The DRAM-slice path
                # does host reads that begin_trace_capture rejects; L1_FULL is trace-safe.
                slice_config=ttnn.Conv2dL1FullSliceConfig,
                return_output_dim=False,
                return_weights_and_bias=False,
            )
            if n_cc > 1:
                ttnn.deallocate(xin_i)
            if silu_on_sharded:
                # SiLU directly on the HEIGHT_SHARDED TILE conv output — one fewer op than
                # SiLU-after-sharded_to_interleaved. Falls back below (QWEN36_GDN_CONV_SILU_SHARDED=0)
                # if this ever regresses PCC or errors for a new shape/dtype combination.
                out_i = ttnn.silu(out_i)
                out_i = ttnn.sharded_to_interleaved(out_i, out_mc)
            else:
                out_i = ttnn.sharded_to_interleaved(out_i, out_mc)
                out_i = ttnn.silu(out_i, memory_config=out_mc)
            outs.append(ttnn.reshape(out_i, (1, T, cw)))
        return outs

    # F6-C (QWEN36_GDN_CONV_TILED_SPLIT): channel-split x (and the conv_state carry) while still
    # TILE, THEN untilize each half separately — instead of untilizing the full width once and
    # slicing the two ROW_MAJOR halves out of the concatenated [1,Lin,1,C] xin (measured 89 us
    # EACH on P150 at T=2048, cw=3072). Only engaged when there is actually more than one channel
    # chunk and T is tile-aligned (T % 32 == 0); every other case (n_cc==1, or an odd T such as a
    # masked-bucket tail) falls back to the legacy full-untilize-then-slice path in fn() below,
    # unchanged. Default OFF (phase-2 measurement, 2026-09-20): at T=2048, x lives in DRAM
    # (glue_mc; xin_l1_max_t default 1024), so the "cheap ~13us TILE slice" assumption doesn't
    # hold there — the per-half TILE slice measured 65-67us (DRAM-bandwidth-bound, not the L1 case
    # the 13us figure came from), leaving only a small theoretical net win (~50us/layer from the
    # smaller paired untilizes) that did not show up in val_b at T=2048/4096 (both runs were
    # slightly SLOWER with this on than off, consistently in both trials — see F6 handoff report).
    # Set QWEN36_GDN_CONV_TILED_SPLIT=1 to re-enable / re-evaluate at a different xin_l1_max_t.
    _tiled_split_enabled = os.environ.get("QWEN36_GDN_CONV_TILED_SPLIT", "0") != "0"

    def _new_state_from_tile(x, T):
        """[1,K-1,C] TILE DRAM carry state, straight off the TILE x (T tile-aligned): one cheap
        tile-aligned [1,32,C] slice, a tiny untilize, a tiny row slice, and a tiny re-tilize —
        avoids the full-width untilize of x done by the legacy path just to get 3 rows out of it."""
        last_tile = ttnn.slice(x, (0, T - 32, 0), (1, T, C))  # [1,32,C] TILE, tile-aligned
        last_tile_rm = ttnn.to_layout(last_tile, ttnn.ROW_MAJOR_LAYOUT, memory_config=_L1)
        ttnn.deallocate(last_tile)
        tail_rm = ttnn.slice(last_tile_rm, (0, 32 - (K - 1), 0), (1, 32, C), memory_config=_L1)
        ttnn.deallocate(last_tile_rm)
        new_state = ttnn.to_layout(tail_rm, ttnn.TILE_LAYOUT, memory_config=_DRAM)
        ttnn.deallocate(tail_rm)
        return new_state

    def _conv_chunks_tiled_split(x, conv_state, Lin, T, n_cc, glue_mc, out_mc):
        """Same result as `_conv_chunks(xin, ...)` (per-channel-chunk conv+SiLU outputs, each
        [1,T,cw] TILE), but builds each chunk's [1,Lin,1,cw] conv1d input directly from
        TILE-sliced-then-untilized halves of x/conv_state instead of slicing it out of one big
        pre-built ROW_MAJOR xin. n_cc must be > 1 here (the caller only takes this path then)."""
        cw = C // n_cc
        wpreps = _prepare(n_cc, Lin)
        outs = []
        for i, wprep in enumerate(wpreps):
            x_half = ttnn.slice(x, (0, 0, i * cw), (1, T, (i + 1) * cw))  # TILE, tile-aligned
            x_half_rm = ttnn.to_layout(x_half, ttnn.ROW_MAJOR_LAYOUT, memory_config=glue_mc)
            ttnn.deallocate(x_half)

            if conv_state is None:
                head_half_rm = ttnn.zeros(
                    [1, K - 1, cw],
                    device=device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=glue_mc,
                )
            else:
                head_half = ttnn.slice(conv_state, (0, 0, i * cw), (1, K - 1, (i + 1) * cw))  # TILE, tiny
                head_half_rm = ttnn.to_layout(head_half, ttnn.ROW_MAJOR_LAYOUT, memory_config=glue_mc)
                ttnn.deallocate(head_half)

            xin_i = ttnn.concat([head_half_rm, x_half_rm], dim=1, memory_config=glue_mc)
            ttnn.deallocate(head_half_rm)
            ttnn.deallocate(x_half_rm)
            xin_i = ttnn.reshape(xin_i, (1, Lin, 1, cw))

            out_i = ttnn.conv1d(
                input_tensor=xin_i,
                weight_tensor=wprep,
                device=device,
                in_channels=cw,
                out_channels=cw,
                batch_size=1,
                input_length=Lin,
                kernel_size=K,
                stride=1,
                padding=0,
                dilation=1,
                groups=cw,
                dtype=ttnn.bfloat16,
                conv_config=conv_cfg,
                compute_config=cc,
                slice_config=ttnn.Conv2dL1FullSliceConfig,
                return_output_dim=False,
                return_weights_and_bias=False,
            )
            ttnn.deallocate(xin_i)
            if silu_on_sharded:
                out_i = ttnn.silu(out_i)
                out_i = ttnn.sharded_to_interleaved(out_i, out_mc)
            else:
                out_i = ttnn.sharded_to_interleaved(out_i, out_mc)
                out_i = ttnn.silu(out_i, memory_config=out_mc)
            outs.append(ttnn.reshape(out_i, (1, T, cw)))
        return outs

    def fn(x, conv_state, qkv_out_mc=None):
        B, T, C_in = x.shape[0], x.shape[1], x.shape[2]
        assert B == 1, "native_conv1d_fn is single-device (B=1) only"
        assert C_in == C, f"channel mismatch: got {C_in}, expected {C}"
        Lin = (K - 1) + T
        n_cc = _select_ncc(T)
        # Above xin_l1_max_t, an L1-resident xin (full [1,Lin,1,C]) clashes with the conv1d's own
        # HEIGHT_SHARDED L1_FULL-sliced CBs (measured: T<=1024 fits alongside them at cw=3072,
        # T=2048 throws "Statically allocated circular buffers ... clash with L1 buffers" — the
        # persisting L1 buffer's footprint grows with Lin while the CB budget doesn't shrink to
        # compensate). Falls back to DRAM there, same as the pre-F2 code; layout is still ROW_MAJOR
        # either way, so the tilize/untilize savings from steps 1-4 hold regardless of T.
        glue_mc = _L1 if T <= xin_l1_max_t else _DRAM
        # q/k/v OUTPUT placement only (see fn's qkv_out_mc docstring above): default (None) keeps
        # the legacy _L1-always behavior; conv1d_kda.py's fallback passes its own DRAM-under-flag
        # decision here instead.
        out_mc = _L1 if qkv_out_mc is None else qkv_out_mc

        if _tiled_split_enabled and n_cc > 1 and T >= 32 and T % 32 == 0:
            # F6-C tiled-split path: no full-width untilize, no big-xin channel slice — see
            # _conv_chunks_tiled_split / _new_state_from_tile above. Requires T tile-aligned (the
            # masked-bucket / odd-length tail, e.g. T=594, falls through to the legacy path below).
            new_state = _new_state_from_tile(x, T)
            conv_outs = _conv_chunks_tiled_split(x, conv_state, Lin, T, n_cc, glue_mc, out_mc)
        else:
            # 1. The ONE full-size relayout: TILE (any memory) -> ROW_MAJOR (L1 or DRAM, see above).
            x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=glue_mc)

            # 2. new_state (tiny): last K-1 rows of x, row-major -> TILE DRAM (external contract
            # unchanged: callers still get [1, K-1, C] TILE DRAM regardless of the n_cc chosen here).
            new_state_rm = ttnn.slice(x_rm, (0, T - (K - 1), 0), (1, T, C), memory_config=_L1)
            new_state = ttnn.to_layout(new_state_rm, ttnn.TILE_LAYOUT, memory_config=_DRAM)
            ttnn.deallocate(new_state_rm)

            # 3. head (tiny): zero pad (first chunk) or carried state, row-major, same mc as x_rm.
            if conv_state is None:
                head = ttnn.zeros(
                    [1, K - 1, C],
                    device=device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=glue_mc,
                )
            else:
                head = ttnn.to_layout(conv_state, ttnn.ROW_MAJOR_LAYOUT, memory_config=glue_mc)

            # 4. xin: one row-major concat (L1 or DRAM). No tilize/untilize from here until the conv output.
            xin = ttnn.concat([head, x_rm], dim=1, memory_config=glue_mc)
            ttnn.deallocate(head)
            ttnn.deallocate(x_rm)
            xin = ttnn.reshape(xin, (1, Lin, 1, C))

            # 5. Per channel-chunk conv1d + SiLU (glue tensors in L1 or DRAM per glue_mc; conv itself
            # is always L1_FULL-sliced HEIGHT_SHARDED regardless).
            conv_outs = _conv_chunks(xin, Lin, T, n_cc, glue_mc, out_mc)
            ttnn.deallocate(xin)

        # 6. n_cc==qkv_tuple_ncc: the chunks ARE q, k, v (fused weight order is q|k|v) — return
        # them directly instead of concatenating just to have the caller re-split them.
        if n_cc == qkv_tuple_ncc:
            q, k, v = conv_outs
            return (q, k, v), new_state
        out = conv_outs[0] if n_cc == 1 else ttnn.concat(conv_outs, dim=-1, memory_config=out_mc)
        return out, new_state

    if os.environ.get("QWEN36_GDN_CONV_LEGACY") == "1":
        return _make_legacy_fn(device, chunks_by_ncc, override_n_cc, K, C, max_channels_per_call, cc, conv_cfg)
    return fn


def _make_legacy_fn(device, chunks_by_ncc, override_n_cc, K, C, max_channels_per_call, cc, conv_cfg):
    """Pre-F2 implementation, reachable via QWEN36_GDN_CONV_LEGACY=1 for A/B comparison only.

    Always uses the "default" (non-3, concat-everything) n_cc — the tuple fast path never existed
    in this path. Picks whichever non-3 n_cc got precomputed (falls back to n_cc=3 if that's the
    only entry available), or the QWEN36_GDN_CONV_CHUNKS override if one was given.
    """
    if override_n_cc is not None:
        n_cc = override_n_cc
    else:
        non_three = [n for n in chunks_by_ncc if n != 3]
        n_cc = min(non_three) if non_three else next(iter(chunks_by_ncc))
    w1d_chunks = chunks_by_ncc[n_cc]
    cw = C // n_cc
    assert cw <= max_channels_per_call, f"channel chunk {cw} exceeds max_channels_per_call {max_channels_per_call}"

    _dram = _DRAM
    _wprep = {}

    def fn(x, conv_state, qkv_out_mc=None):
        # qkv_out_mc accepted (and ignored) only for call-signature parity with the non-legacy
        # fn() above (conv1d_kda.py's fallback always passes it by keyword): this legacy path's
        # output is unconditionally DRAM (_dram) already, below, so there is nothing to override.
        B, T, C_in = x.shape[0], x.shape[1], x.shape[2]
        assert B == 1, "native_conv1d_fn is single-device (B=1) only"
        assert C_in == C, f"channel mismatch: got {C_in}, expected {C}"
        Lin = (K - 1) + T
        # new_state: last K-1 real input tokens (for the next chunk's carry), TILE/DRAM.
        new_state = ttnn.slice(x, (0, T - (K - 1), 0), (1, T, C))
        new_state = ttnn.to_memory_config(ttnn.to_layout(new_state, ttnn.TILE_LAYOUT), _dram)
        if conv_state is None:
            pad = ttnn.zeros(
                [1, K - 1, C], device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=_dram
            )
            xin = ttnn.concat([pad, x], dim=1, memory_config=_dram)
            ttnn.deallocate(pad)
        else:
            xin = ttnn.concat([conv_state, x], dim=1, memory_config=_dram)
        xin = ttnn.to_layout(xin, ttnn.ROW_MAJOR_LAYOUT, memory_config=_dram)
        xin = ttnn.reshape(xin, (1, Lin, 1, C))

        if Lin not in _wprep:
            _wprep[Lin] = [
                ttnn.prepare_conv_weights(
                    weight_tensor=w,
                    input_memory_config=_dram,
                    input_layout=ttnn.ROW_MAJOR_LAYOUT,
                    weights_format="OIHW",
                    in_channels=cw,
                    out_channels=cw,
                    batch_size=1,
                    input_height=1,
                    input_width=Lin,
                    kernel_size=(1, K),
                    stride=(1, 1),
                    padding=(0, 0),
                    dilation=(1, 1),
                    has_bias=False,
                    groups=cw,
                    device=device,
                    input_dtype=ttnn.bfloat16,
                    conv_config=conv_cfg,
                    compute_config=cc,
                )
                for w in w1d_chunks
            ]
        wpreps = _wprep[Lin]

        conv_outs = []
        for i, wprep in enumerate(wpreps):
            xin_i = xin if n_cc == 1 else ttnn.slice(xin, (0, 0, 0, i * cw), (1, Lin, 1, (i + 1) * cw))
            out_i = ttnn.conv1d(
                input_tensor=xin_i,
                weight_tensor=wprep,
                device=device,
                in_channels=cw,
                out_channels=cw,
                batch_size=1,
                input_length=Lin,
                kernel_size=K,
                stride=1,
                padding=0,
                dilation=1,
                groups=cw,
                dtype=ttnn.bfloat16,
                conv_config=conv_cfg,
                compute_config=cc,
                slice_config=ttnn.Conv2dL1FullSliceConfig,
                return_output_dim=False,
                return_weights_and_bias=False,
            )
            if n_cc > 1:
                ttnn.deallocate(xin_i)
            conv_outs.append(ttnn.reshape(ttnn.sharded_to_interleaved(out_i, _dram), (1, T, cw)))
        ttnn.deallocate(xin)
        out = conv_outs[0] if n_cc == 1 else ttnn.concat(conv_outs, dim=-1, memory_config=_dram)
        out = ttnn.to_layout(out, ttnn.TILE_LAYOUT, memory_config=_dram)
        return ttnn.silu(out, memory_config=_dram), new_state

    return fn
