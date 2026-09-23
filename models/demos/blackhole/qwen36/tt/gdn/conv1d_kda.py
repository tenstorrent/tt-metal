# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused-op depthwise causal conv1d + SiLU for GDN chunk prefill (T>1), via
`ttnn.experimental.kda.qkv_causal_conv1d_silu`.

Replaces conv1d_native.py's per-channel-chunk HEIGHT_SHARDED ttnn.conv1d loop (slice ->
interleaved_to_sharded -> conv1d -> sharded_to_interleaved -> silu, x2 chunks) + trailing
concat + 3 slices with ONE fused device op that does the 4-tap depthwise conv, SiLU, and the
q/k/v split together. Keeps the same ROW_MAJOR glue (`x_rm`) and `new_state` production as
conv1d_native.py so decode's `restore_split_conv_from_fused` / traced in-place state writes
(gdn/state.py, gdn/decode.py) keep working unchanged.

Recon: /local/ttuser/atupe/qwen35_2b_handoff/scripts/step2/kda_conv_probe.py and the conv_kda
recon note this integration follows (op signature/constraints, reference math, tap-order
convention, channel_chunk_size legality/L1-footprint sweep).

The op does not return an updated history (the caller owns history updates — see the op's own
docstring, ttnn.experimental.kda.qkv_causal_conv1d_silu), so `new_state` is still computed here
exactly as conv1d_native.py does it: the last `kernel_size - 1` rows of `x`, row-major-sliced off
the same `x_rm` used as the op's `input`, then tilized back to TILE/DRAM (the persistent
`fused_conv_state` buffer's format).
"""
import os

import ttnn

_L1 = ttnn.L1_MEMORY_CONFIG
_DRAM = ttnn.DRAM_MEMORY_CONFIG

# QWEN36_GDN_FLA_INPUTS_DRAM (step2, 2026-09-22; default "1" when QWEN_GDN_PATH=="fused", else
# "0"): forces this op's q/k/v OUTPUT to DRAM-interleaved even when T <= xin_l1_max_t would
# otherwise put them in L1 (glue_mc). Why: q/k/v are the ttnn.transformer.chunk_gated_delta_rule
# ("the FLA op") call's inputs, produced here and consumed there with no op in between -- so
# whatever memory_config they get here is exactly what is L1-resident (persistent, not a CB) when
# the FLA op's own program is built. At T=2048 the FLA op's internal QWEN_GDN_PATH=fused C++
# dispatch statically allocates circular buffers that end at 1,213,440 B/core (of a 1,436,672 B/
# bank Blackhole P150a L1 ceiling -- 223,232 B free above the CBs); q+k+v alone at T=2048
# ([1,2048,2048] bf16 TILE each, ~8 MB/tensor = ~64.5 KB/bank average across 130 banks, ~193.6 KB/
# bank for all three) very nearly exhausts that headroom on its own, and the small mc_small
# beta/g/a_biased/sp chain (also moved to DRAM by this same flag -- see
# ttnn_gated_deltanet.py's `_use_chunk_fn` block) tips a few banks over, producing "Statically
# allocated circular buffers ... clash with L1 buffers". Moving q/k/v to DRAM here removes the
# dominant term; x_rm/history_rm (the conv glue, freed inside this function before returning) and
# new_state (already DRAM, see step (d) below) are UNCHANGED -- this flag only redirects the op's
# OUTPUT. QWEN_GDN_PATH itself only selects the FLA op's internal C++ dispatch (fused/phased/mono)
# and is unrelated to this Python-level memory placement; QWEN36_GDN_FLA_INPUTS_DRAM defaults on
# only when that path is selected because "fused" is the dispatch whose CBs are large enough to
# clash in the first place (phased/mono have smaller static CBs and fit with q/k/v in L1). Set
# explicitly to override either way. No effect on math (memory placement only).
#
# step2 (2026-09-23): this flag also governs the OUTPUT of fn()'s fallback to native_fn
# (conv1d_native.py -- masked-tail valid_len, a non-tile-aligned/short T, or a channel-width
# mismatch; see fn()'s guard below), via the `qkv_out_mc` kwarg threaded down to it. Before that,
# native_fn's own output was unconditionally L1 regardless of T -- e.g. model.prefill_paged's
# exact-length chunked reference (prefill_layer_chunked, chunk_size=2048) produces a non-tile-
# aligned remainder chunk that hits this fallback via T % 32 != 0 (not valid_len), and its
# leftover L1-resident q/k/v raised the L1 allocator's high-water mark just enough for a LATER,
# unrelated fused-KDA call's static CBs to clash by as little as 1024 B -- the exact
# "chunk1_tail1500_b2048" failure this flag's fallback plumbing fixes (test_prefill.py -k tail).
_FLA_INPUTS_DRAM_DEFAULT = "1" if os.environ.get("QWEN_GDN_PATH") == "fused" else "0"

# Measured (kda_conv_probe.py K1-K5 sweep, T=2048/Q=K=V=2048/C=6144 on P150): channel_chunk_size
# must be tile-aligned and evenly divide C=6144. 2048 (block_ct=64) is ~5 KB/core over the
# Blackhole P150a per-bank L1 ceiling (see project memory: 130 banks x 1,436,672 B) before any
# other resident buffers are counted, so it is likely to overflow; 768 (block_ct=24, 8 blocks)
# leaves a comfortable margin and was the fastest legal width measured in the probe. Override with
# QWEN36_GDN_CONV_KDA_CCS for a different device/shape.
_DEFAULT_CCS = 768


def make_kda_conv1d_fn(
    device,
    weight_taps,
    kernel_size,
    q_dim,
    k_dim,
    v_dim,
    native_fn,
    xin_l1_max_t=2048,
    channel_chunk_size=_DEFAULT_CCS,
):
    """Build a chunk-prefill fused causal-conv1d+SiLU callable for one GDN layer.

    weight_taps: list of `kernel_size` device tensors [1, 1, C] TILE bf16, DRAM-interleaved, tap
    order oldest (t-K+1) .. current (t) — `GDNWeights.fused_conv_weight_taps`, already in the
    exact format/order `ttnn.experimental.kda.qkv_causal_conv1d_silu` requires (same convention;
    see the recon's tap-ordering note). The op is compiled for exactly 4 taps.

    native_fn: the already-built `conv1d_native.make_native_conv1d_fn(...)` callable
    `fn(x, conv_state) -> (out_or_(q,k,v)_tuple, new_state)`, used as the fallback for shapes this
    fused path does not (or should not) handle: masked-tail (`valid_len is not None`), a
    non-tile-aligned or T<32 sequence length, or a channel-width mismatch. Required (not
    optional): the guards below always have somewhere safe to fall back to.

    xin_l1_max_t: same env-driven threshold as conv1d_native.py's `xin_l1_max_t` (largest T for
    which the ROW_MAJOR glue tensors — here, `x_rm`/`history_rm` and the fused op's output — stay
    L1-resident; above it everything falls back to DRAM). Kept identical to conv1d_native.py's
    default (2048) and env override (`QWEN36_GDN_CONV_XIN_L1_MAX_T`; "0" restores the legacy 1024
    default) so both conv paths pick the same L1/DRAM split for a given T.

    channel_chunk_size: `program_config.channel_chunk_size` for the fused op (logical channels;
    must be tile-aligned and evenly divide `q_dim + k_dim + v_dim`). Default 768 (see
    `_DEFAULT_CCS`). QWEN36_GDN_CONV_KDA_CCS overrides.

    QWEN36_GDN_CONV_KDA_FP32ACC (default "0", read here): the op's `compute_kernel_config=None`
    default resolves to HiFi4/math_approx_mode=False/fp32_dest_acc_en=False/packer_l1_acc=False
    (confirmed by the op's own test,
    test_qkv_causal_conv1d_silu_default_compute_config_matches_explicit_defaults). Read directly
    from validate_on_program_cache_miss (qkv_causal_conv1d_silu_device_operation.cpp) and
    check_compute_config (kda_factory_utils.cpp): only `math_approx_mode=true` and
    `packer_l1_acc=true` are TT_FATAL-rejected; `fp32_dest_acc_en` is NOT restricted, so "1" builds
    a legal explicit `ttnn.WormholeComputeKernelConfig(math_fidelity=HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=False)` (accumulate in fp32 through the conv+SiLU, for
    parity with the native HEIGHT_SHARDED conv1d path's own `cc`, which also sets
    fp32_dest_acc_en=True — see conv1d_native.py:118-120 — though that path's packer_l1_acc=True
    is NOT reproducible here, the op rejects it).

    Measured (step2 conv-kda accuracy experiment, 2026-09-22): fp32acc=1 raises the ISOLATED
    op-level PCC vs the native path (V1: q/k/v PCC 0.9999866 -> 0.9999949, random single-layer
    inputs) but REGRESSES the whole-model logits comparison (V2, vs logits_step2_default_fp32off,
    T=512/1024) at T=1024: PCC 0.999782 -> 0.999665 (T=512 only marginally improves: 0.999785 ->
    0.999793). Whole-model PCC is the metric that matters (it is what the 0.9998 gate is measured
    against), so fp32acc=0 (op default, NOT fp32_dest_acc_en=True) is the better configuration and
    is the DEFAULT here, despite losing the isolated-op-level comparison. Set to "1" only to
    re-run/re-examine the fp32-accumulate variant.

    Returns fn(x, conv_state, valid_len=None) -> ((q, k, v), new_state) — the SAME call contract
    as conv1d_native.make_native_conv1d_fn's fn (ttnn_gated_deltanet.py:636 calls it positionally
    as `native_conv1d_fn(qkv, fused_conv_state)`; `valid_len` is accepted only as an optional
    defensive kwarg — the current call site never passes it, since
    ttnn_gated_deltanet.py:629 already gates dispatch into `native_conv1d_fn` on
    `valid_len is None`):
      x: [1, T, C] TILE (any memory) — the fused Q|K|V projection output for one prefill chunk.
      conv_state: [1, K-1, C] TILE (any memory) cross-chunk carry, or None (first chunk).
      (q, k, v): SiLU(depthwise_conv1d(x)) split into q_dim/k_dim/v_dim-wide TILE tensors,
        L1-interleaved when T <= xin_l1_max_t else DRAM-interleaved (matches conv1d_native.py's
        `glue_mc` convention for its own q/k/v output) — UNLESS QWEN36_GDN_FLA_INPUTS_DRAM is on
        (default when QWEN_GDN_PATH=="fused"), in which case q/k/v are always DRAM-interleaved
        regardless of T (see the flag's docstring at the top of this module).
      new_state: last K-1 rows of x (next chunk's carry) [1, K-1, C] TILE DRAM (unchanged
        contract — bit-identical production recipe to conv1d_native.py's own new_state).
    """
    K = kernel_size
    assert len(weight_taps) == K, f"got {len(weight_taps)} weight taps for kernel_size={K}"
    assert K == 4, f"ttnn.experimental.kda.qkv_causal_conv1d_silu is compiled for exactly 4 taps, got kernel_size={K}"
    C = q_dim + k_dim + v_dim
    for name, d in (("q_dim", q_dim), ("k_dim", k_dim), ("v_dim", v_dim)):
        assert d > 0 and d % 32 == 0, f"{name}={d} must be positive and tile-aligned for the KDA op"

    _xin_l1_max_t_env = int(os.environ.get("QWEN36_GDN_CONV_XIN_L1_MAX_T", str(xin_l1_max_t)))
    xin_l1_max_t = 1024 if _xin_l1_max_t_env == 0 else _xin_l1_max_t_env

    # See QWEN36_GDN_FLA_INPUTS_DRAM docstring above (module level, next to _FLA_INPUTS_DRAM_DEFAULT).
    _fla_inputs_dram = os.environ.get("QWEN36_GDN_FLA_INPUTS_DRAM", _FLA_INPUTS_DRAM_DEFAULT) != "0"

    ccs = int(os.environ.get("QWEN36_GDN_CONV_KDA_CCS", str(channel_chunk_size)))
    assert ccs > 0 and ccs % 32 == 0 and ccs <= C and C % ccs == 0, (
        f"QWEN36_GDN_CONV_KDA_CCS={ccs} illegal for channels={C}: must be >0, tile-aligned (%32==0), "
        f"<= {C}, and evenly divide {C}"
    )
    program_config = ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=ccs)
    tap0, tap1, tap2, tap3 = weight_taps

    # See QWEN36_GDN_CONV_KDA_FP32ACC docstring above: fp32_dest_acc_en is unrestricted by the
    # op's validate() (only math_approx_mode/packer_l1_acc are rejected), so this accumulate-in-
    # fp32 config is legal to pass here.
    _fp32acc_enabled = os.environ.get("QWEN36_GDN_CONV_KDA_FP32ACC", "0") != "0"
    _kda_ckc = (
        ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        if _fp32acc_enabled
        else None
    )

    def fn(x, conv_state, valid_len=None):
        B, T, C_in = x.shape[0], x.shape[1], x.shape[2]
        assert B == 1, "kda_conv1d_fn is single-device (B=1) only"

        # Above xin_l1_max_t, keep the glue tensors (and the fused op's output, unless
        # QWEN36_GDN_FLA_INPUTS_DRAM overrides just the output below) in DRAM instead of L1 — same
        # threshold/rationale as conv1d_native.py's glue_mc. Computed BEFORE the fallback guard
        # below (not just in the fused branch) so qkv_out_mc is ready to hand to native_fn too —
        # see the guard's own comment for why that matters.
        glue_mc = _L1 if T <= xin_l1_max_t else _DRAM

        # q/k/v OUTPUT placement only — x_rm/history_rm above/below stay on glue_mc either way.
        # See QWEN36_GDN_FLA_INPUTS_DRAM's docstring (module top): this decouples the FLA op's
        # persistent L1-alive inputs from the conv glue's own L1 threshold when T <= xin_l1_max_t.
        qkv_out_mc = _DRAM if _fla_inputs_dram else glue_mc

        # Guard: fall back to the native (HEIGHT_SHARDED conv1d) path for anything the fused op
        # does not cover here — masked-tail valid_len (not implemented in either path, see module
        # docstring), a non-tile-aligned/too-short T, or an unexpected channel width. `qkv_out_mc`
        # is passed through so this fallback's q/k/v output honors QWEN36_GDN_FLA_INPUTS_DRAM too
        # (added step2 2026-09-23): native_fn's own default output placement is unconditionally L1
        # regardless of T, which left a large L1-resident tensor alive on e.g. the non-tile-aligned
        # remainder chunk model.prefill_paged's exact-length chunked reference produces (T not a
        # multiple of 32 hits this same guard via T % 32 != 0, not just valid_len) — that leftover
        # raised the L1 allocator's high-water mark just enough for a LATER, unrelated fused-KDA
        # call's static CBs to clash by as little as 1024 B. See conv1d_native.py's `qkv_out_mc`
        # docstring for the full mechanism.
        if valid_len is not None or C_in != C or T < 32 or T % 32 != 0:
            return native_fn(x, conv_state, qkv_out_mc=qkv_out_mc)

        # a) The one full-size relayout: TILE (any memory) -> ROW_MAJOR (L1 or DRAM per glue_mc).
        # Same op/mc convention as conv1d_native.fn's step 1 — the op's `input` requires
        # interleaved ROW_MAJOR bf16, which glue_mc (L1 or DRAM, both interleaved) already is.
        x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=glue_mc)
        if tuple(x_rm.shape) != (1, T, C):
            x_rm = ttnn.reshape(x_rm, (1, T, C))

        # b) history: zero pad (first chunk) or the carried fused conv state, row-major, same mc
        # as x_rm. Zero-pad path reuses the exact op conv1d_native.fn already uses for this case
        # (no new cost); the carried-state path is the same untilize conv1d_native.fn already does
        # for its `head` (chunk >= 2), just consumed directly by the fused op instead of a concat.
        if conv_state is None:
            history_rm = ttnn.zeros(
                [1, K - 1, C],
                device=device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=glue_mc,
            )
        else:
            history_rm = ttnn.to_layout(conv_state, ttnn.ROW_MAJOR_LAYOUT, memory_config=glue_mc)

        # c) Fused 4-tap depthwise conv + SiLU + q/k/v split in ONE device op — replaces the
        # entire per-channel-chunk HEIGHT_SHARDED conv1d loop + trailing concat/slice. No bias add
        # needed: the op has none, and this model's GDN has no conv1d bias (see gdn/weights.py /
        # conv1d_native.py's fused_conv_bias_dev comment). compute_kernel_config=_kda_ckc: either
        # None (the op's HiFi4/fp32_dest_acc_en=False default) or the explicit fp32-accumulate
        # config, per QWEN36_GDN_CONV_KDA_FP32ACC (see docstring) — either way, NOT
        # conv1d_native.py's own `cc` (packer_l1_acc=True is rejected by this op).
        q, k, v = ttnn.experimental.kda.qkv_causal_conv1d_silu(
            x_rm,
            history_rm,
            tap0,
            tap1,
            tap2,
            tap3,
            q_dim,
            k_dim,
            v_dim,
            program_config=program_config,
            memory_config=qkv_out_mc,
            compute_kernel_config=_kda_ckc,
        )
        ttnn.deallocate(history_rm)

        # d) new_state (unchanged contract): last K-1 rows of x_rm, row-major -> TILE DRAM. The
        # fused op does not return an updated history (caller owns history updates), so this exact
        # pair — bit-identical to conv1d_native.fn's own new_state production — must remain,
        # reusing the already-computed x_rm (no extra untilize). Any use_inplace_state copy-back
        # into the persistent fused_conv_state buffer happens in the caller (gdn/decode.py), which
        # only needs a valid TILE tensor of the right shape/dtype back — unaffected by which conv
        # path produced it.
        new_state_rm = ttnn.slice(x_rm, (0, T - (K - 1), 0), (1, T, C), memory_config=_L1)
        new_state = ttnn.to_layout(new_state_rm, ttnn.TILE_LAYOUT, memory_config=_DRAM)
        ttnn.deallocate(new_state_rm)

        # e) Done with x_rm.
        ttnn.deallocate(x_rm)

        return (q, k, v), new_state

    return fn
