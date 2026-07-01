# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN port of Kokoro ``TextEncoder`` (embedding + Conv1d/LayerNorm + BiLSTM).

Reference: ``models.experimental.kokoro.reference.modules.TextEncoder``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn

import ttnn

from .tt_conv import TTConv1dParams, tt_conv1d_nlc, tt_weight_norm_materialize
from .tt_lstm import TTLSTMParams, build_fused_recurrent_weight, preprocess_tt_lstm_1layer, tt_bilstm_nlc


@dataclass(frozen=True)
class TTTextEncoderConvLNBlockParams:
    """Weights for one Conv1d + channel LayerNorm stage (dropout omitted at inference)."""

    conv: TTConv1dParams
    ln_weight: ttnn.Tensor
    ln_bias: ttnn.Tensor


@dataclass(frozen=True)
class TTTextEncoderParams:
    """Preprocessed TTNN parameters for :class:`TTTextEncoder`."""

    embedding_weight: ttnn.Tensor
    blocks: tuple[TTTextEncoderConvLNBlockParams, ...]
    lstm_fwd: TTLSTMParams
    lstm_rev: TTLSTMParams
    # Block-diagonal recurrent weight fusing both BiLSTM directions into one matmul/step
    # (None for a unidirectional LSTM). Halves per-step matmul/activation/elementwise ops on
    # the unpadded path; bit-exact at bf16 state (see ``build_fused_recurrent_weight``).
    lstm_w_h_block: Optional[ttnn.Tensor] = None
    ln_eps: float = 1e-5


def preprocess_tt_text_encoder(
    text_encoder: nn.Module,
    device: ttnn.Device,
    *,
    weights_dtype=ttnn.bfloat16,
) -> TTTextEncoderParams:
    """Upload PyTorch ``TextEncoder`` weights to device for :class:`TTTextEncoder`."""
    emb_w = ttnn.from_torch(
        text_encoder.embedding.weight.detach().cpu(),
        dtype=ttnn.bfloat16,  # ttnn.embedding requires BF16 weights on device
        # Store ROW_MAJOR: ttnn.embedding gathers rows in row-major, so a TILE-layout table is
        # untilized to row-major on every forward (the 17µs UntilizeWithUnpadding before the
        # embedding). Doing the (un)tilize once here at preprocess removes it from the hot path.
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        # Stage the (constant) gather table in L1, interleaved. The embedding sweep
        # (perf/test_embedding_text_encoder_perf_sweep.py) found an L1-resident table+indices is the
        # base of every PCC-passing speedup (the op forbids *sharded* weights, so interleaved L1 is the
        # ceiling). Table is VOCAB×C bf16 (~178×512 ≈ 182 KiB) — a one-time residency cost that removes
        # a DRAM read from every forward's gather. Pairs with the height-sharded output below.
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    block_params: list[TTTextEncoderConvLNBlockParams] = []

    for block in text_encoder.cnn:
        conv = block[0]
        ln = block[1]

        if hasattr(conv, "weight_v") and hasattr(conv, "weight_g"):
            w = tt_weight_norm_materialize(conv.weight_v.detach().cpu(), conv.weight_g.detach().cpu())
        else:
            # PyTorch ``nn.utils.parametrizations.weight_norm`` exposes materialized ``weight``.
            w = conv.weight.detach().cpu()
        b = conv.bias.detach().cpu() if conv.bias is not None else None

        w_tt = ttnn.from_torch(w, dtype=weights_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
        b_tt = (
            ttnn.from_torch(b.reshape(1, 1, 1, -1), dtype=weights_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
            if b is not None
            else None
        )
        conv_p = TTConv1dParams(
            weight=w_tt,
            bias=b_tt,
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size[0],
            stride=conv.stride[0],
            padding=conv.padding[0],
            groups=conv.groups,
        )

        ln_w = ttnn.from_torch(
            ln.gamma.detach().cpu(),
            dtype=weights_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ln_b = ttnn.from_torch(
            ln.beta.detach().cpu(),
            dtype=weights_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        block_params.append(TTTextEncoderConvLNBlockParams(conv=conv_p, ln_weight=ln_w, ln_bias=ln_b))

    fwd, rev = preprocess_tt_lstm_1layer(text_encoder.lstm, device, weights_dtype=weights_dtype)
    assert rev is not None, "TextEncoder expects a bidirectional LSTM"
    lstm_w_h_block = build_fused_recurrent_weight(text_encoder.lstm, device, weights_dtype=weights_dtype)

    # Pre-stage the (constant) BiLSTM matmul weights to L1 once, here, instead of copying them
    # DRAM->L1 on every forward inside the fused loop (the 3 large CopyDeviceOperations before the
    # LSTM matmuls: w_h_block ~10µs + the two w_x ~4µs each). The fused path uses L1 in1 for its
    # gate-precompute + recurrent matmuls; making the weights L1-resident gives that same speedup
    # while removing the per-forward staging copy. tt_bilstm_nlc detects already-L1 weights and skips
    # both the copy and the end-of-forward dealloc, so these persist for the encoder's lifetime.
    # Footprint is small and interleaved (w_h_block [2H,8H] 2 MiB + 2× w_x [H,4H] 1 MiB = ~4 MiB
    # spread across L1 banks, ~31 KiB/core), and during the forward they were already L1-resident
    # (transiently) so the forward's peak L1 is unchanged — only the post-forward residency differs.
    w_x_l1 = lambda p: TTLSTMParams(
        w_x=ttnn.to_memory_config(p.w_x, ttnn.L1_MEMORY_CONFIG), w_h=p.w_h, b=p.b, hidden_size=p.hidden_size
    )
    fwd, rev = w_x_l1(fwd), w_x_l1(rev)
    lstm_w_h_block = ttnn.to_memory_config(lstm_w_h_block, ttnn.L1_MEMORY_CONFIG)

    return TTTextEncoderParams(
        embedding_weight=emb_w,
        blocks=tuple(block_params),
        lstm_fwd=fwd,
        lstm_rev=rev,
        lstm_w_h_block=lstm_w_h_block,
        ln_eps=1e-5,
    )


def _mask_keep_flat(text_mask: torch.Tensor, *, device: ttnn.Device) -> ttnn.Tensor:
    """Batch-flattened keep-mask ``[1, 1, B*T, 1]`` (``text_mask[b,t] == True`` => padded, as in the
    reference ``masked_fill_``). Shaped to match the CNN's ``[1, 1, B*T, C]`` activations."""
    B, T = text_mask.shape
    keep = (~text_mask).to(torch.float32).reshape(1, 1, B * T, 1)
    return ttnn.from_torch(keep, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


def _fused_recurrent_program_config(hidden_size: int, device: ttnn.Device):
    """Tuned program config for the per-step fused recurrent matmul ``[B, 2H] @ [2H, 8H]``.

    The matmul sweep (``perf/test_recurrent_matmul_sweep.py``) found a 1D mcast config
    (8x8 grid, ``in0_block_w=8``, ``per_core_M=per_core_N=1``, width-sharded output) is the fastest
    option for the H=256 shape: 3.73 µs vs ~10 µs for the default config and 5.52 µs for the old
    ``in0_block_w=4``, with PCC unchanged (the matmul is bit-exact across layouts; only the
    tiling/mcast schedule changes). Interleaved in0 is required: a sharded-in0 winner would need an
    InterleavedToSharded per step inside the host-driven loop, a net regression on this
    dispatch-bound model.

    ``per_core_M=1`` holds for any ``B<=32`` (Mt=1). Returns ``None`` (use the default config) unless
    the shape matches the swept/validated H (``8H`` tiles == 64, evenly mapped onto an 8x8 grid), so
    other-width BiLSTMs are unaffected.
    """
    if device.arch() != ttnn.device.Arch.BLACKHOLE:
        return None
    n_tiles = (8 * hidden_size) // 32  # 8H output cols in tiles
    k_tiles = (2 * hidden_size) // 32  # 2H contraction in tiles
    if n_tiles != 64 or k_tiles % 4 != 0:  # validated shape only (H=256 -> Nt=64, Kt=16)
        return None
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),  # 64 cores -> one 8H tile each
        # in0_block_w=8 (half of Kt=16, two K-steps) is the sweep winner for [32,512]@[512,2048] at
        # LoFi width-sharded output: 3.73µs vs the old ibw=4's 5.52µs (-32%), ibw=16 (single K-step)
        # ties at 3.77µs and ibw=2 blows up to 9.5µs. See perf/test_recurrent_matmul_sweep.py.
        in0_block_w=8,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=1,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


def _layernorm_prog_from_sharded(x: ttnn.Tensor):
    """LayerNorm program config matching ``x``'s own block-sharded layout, or ``None``.

    The batched CNN conv emits its output **block-sharded in L1** (e.g. the ``B*T``×512 stage: grid
    8×3, shard ``[32, 64]`` => block_h=1, block_w=2). Feeding that straight into a sharded LayerNorm
    (program config derived here from the shard spec) lets the LN read it in place — dropping BOTH the
    conv's ``ShardedToInterleaved`` and the ``InterleavedToSharded`` a fixed-config sharded LN would
    need (two reshards/CNN stage). The layout sweep (``perf/test_layernorm_text_encoder_perf_sweep.py``)
    measured this 8×3 layout at ~7.8µs vs ~12.4µs DRAM (1.6×) — 0.6µs slower than the standalone 4×3
    optimum (7.2µs) but with zero reshards, a net win. Returns ``None`` (caller keeps the DRAM LN) when
    ``x`` isn't block-sharded or doesn't tile cleanly.
    """
    TILE = 32
    mc = x.memory_config()
    if mc.memory_layout != ttnn.TensorMemoryLayout.BLOCK_SHARDED:
        return None
    ss = mc.shard_spec
    if ss is None:
        return None
    shard_h, shard_w = int(ss.shape[0]), int(ss.shape[1])
    if shard_h % TILE or shard_w % TILE or shard_h == 0 or shard_w == 0:
        return None
    block_h, block_w = shard_h // TILE, shard_w // TILE
    # The grid (and thus block_h/gy) is set by the conv on the *tile-padded* row count, so read it from
    # the shard's own bounding box rather than the logical shape (e.g. B*T=120 pads to 128 => grid 8x4).
    # The LN program config grid must match the shard grid exactly, anchored at the origin.
    bb = ss.grid.bounding_box()
    if bb.start.x != 0 or bb.start.y != 0:
        return None
    gx, gy = bb.end.x + 1, bb.end.y + 1
    subblock_w = next((d for d in range(min(block_w, 4), 0, -1) if block_w % d == 0), 1)
    return ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        subblock_w=subblock_w,
        block_h=block_h,
        block_w=block_w,
        inplace=False,
    )


def _maybe_interleaved(x: ttnn.Tensor) -> ttnn.Tensor:
    if ttnn.is_tensor_storage_on_device(x) and x.is_sharded():
        return ttnn.sharded_to_interleaved(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return x


class TTTextEncoderConvLNBlock:
    """One depth-wise CNN stack: Conv1d → LayerNorm → LeakyReLU (matches reference ``Sequential``)."""

    def __init__(
        self,
        *,
        device: ttnn.Device,
        params: TTTextEncoderConvLNBlockParams,
        ln_eps: float,
        compute_kernel_config,
        conv_compute_kernel_config=None,
    ) -> None:
        self.device = device
        self.params = params
        self.ln_eps = ln_eps
        self.compute_kernel_config = compute_kernel_config
        # Conv may run at a different (lower) math fidelity than the LN; defaults to the shared config.
        self.conv_compute_kernel_config = conv_compute_kernel_config or compute_kernel_config

    def forward(
        self,
        x_flat: ttnn.Tensor,
        mask_keep: Optional[ttnn.Tensor],
        *,
        batch: int,
        seq: int,
        keep_sharded: bool = False,
    ) -> ttnn.Tensor:
        """One CNN stage on batch-flattened activations ``[1, 1, batch*seq, C]`` (same shape out).

        The conv emits its output **block-sharded in L1**; when that layout is a valid sharded-LN input
        the channel-LayerNorm + LeakyReLU run on it in place (no reshard either side). ``ttnn.conv1d``
        also *accepts* a block-sharded input, so with ``keep_sharded`` the LeakyReLU output is left
        sharded and handed straight to the next stage's conv — dropping the ShardedToInterleaved (here)
        and the InterleavedToSharded (next conv's input) that a DRAM hand-off would cost. Set it only
        when the consumer is another CNN conv and no mask multiply intervenes; otherwise (last stage, or
        the padded mask path) the output is re-interleaved to DRAM. Falls back to a DRAM LayerNorm when
        the conv output isn't block-sharded (non-BH / odd shapes).
        """
        x = tt_conv1d_nlc(
            x_nlc=x_flat,
            params=self.params.conv,
            device=self.device,
            compute_config=self.conv_compute_kernel_config,
            # Blackhole runs the whole B-batch in one conv instead of one-conv-per-item (the split
            # is a Wormhole-only correctness workaround), halving conv/halo/shard dispatch. Flattened
            # in/out so the CNN stack never pays the [1,B*L,C]->[B,L,C] split between stages.
            batched_shape=(batch, seq),
            # Keep the conv's native block-sharded L1 output so the LayerNorm reads it in place.
            output_sharded=True,
        )
        ln_prog = _layernorm_prog_from_sharded(x)
        if ln_prog is not None:
            # Conv output is block-sharded L1: normalize + activate on-core (no reshard).
            ln_mem = x.memory_config()
            x = ttnn.layer_norm(
                x,
                weight=self.params.ln_weight,
                bias=self.params.ln_bias,
                epsilon=self.ln_eps,
                compute_kernel_config=self.compute_kernel_config,
                memory_config=ln_mem,
                program_config=ln_prog,
            )
            x = ttnn.leaky_relu(x, negative_slope=0.2, memory_config=ln_mem)
            # Hand the sharded LeakyReLU output straight to the next conv (no mask, not the last stage).
            if keep_sharded and mask_keep is None:
                return x
            # Otherwise re-interleave to DRAM for the reshape→LSTM (last stage) or the mask multiply.
            x = ttnn.sharded_to_interleaved(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            x = _maybe_interleaved(x)
            x = ttnn.layer_norm(
                x,
                weight=self.params.ln_weight,
                bias=self.params.ln_bias,
                epsilon=self.ln_eps,
                compute_kernel_config=self.compute_kernel_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            x = ttnn.leaky_relu(x, negative_slope=0.2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # No padding (mask_keep is None) -> the reference ``masked_fill_`` is a no-op, so skip the
        # all-ones multiply (the full-length/single-utterance path; saves one BinaryNg per block).
        if mask_keep is None:
            return x
        return ttnn.multiply(x, mask_keep, memory_config=ttnn.DRAM_MEMORY_CONFIG)


class TTTextEncoder:
    """TTNN ``TextEncoder``: embedding → masked CNN stages → packed-length BiLSTM → ``[B, C, T]``."""

    def __init__(self, device: ttnn.Device, params: TTTextEncoderParams) -> None:
        self.device = device
        self.params = params
        # LayerNorm + the (numerically-touchier) BiLSTM matmuls run at HiFi3.
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi3,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )
        # The CNN conv runs at LoFi: a fidelity sweep (perf/test_conv_text_encoder_perf_sweep.py,
        # KOKORO_CONV_FIDELITY_ONLY=1) swept LoFi->HiFi4 x fp32_dest_acc_en on/off at the production conv
        # shape (96x2560x512, block-sharded 24c, double-buffered). The conv inputs are bf16, so PCC is
        # already saturated at LoFi — isolated conv PCC 0.99988 (LoFi) vs 0.99998 (HiFi2/3/4), all far
        # above the bar — while LoFi is the fastest PCC-passing config: 29.7µs vs HiFi2 33.3µs (-11%) and
        # HiFi3 41.8µs (-29%) per conv. fp32_dest_acc_en stays True (it costs ~0 here, 29.69 vs 29.90µs,
        # and keeps the high PCC; fp32acc=False drops LoFi to 0.99965). Kept separate from the main config
        # so the BiLSTM matmuls (whose output feeds the decoder) stay at HiFi3.
        self.conv_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )
        # LoFi + bf16 dest-acc for the BiLSTM matmuls (bf16 weights). The recurrent-matmul sweep found
        # fp32_dest_acc_en=False shaves the per-step [B,2H]@[2H,8H] matmul 3.74->3.68µs and the
        # gate-precompute 7.97->7.86µs (reorders unchanged); the DST accumulates in bf16 instead of
        # fp32. It's a real precision drop but the tolerant ASR TextEncoder absorbs it (full-seq PCC
        # 0.99930 unchanged to 4 decimals). TextEncoder-only config — the F0-sensitive prosody/duration
        # BiLSTMs keep their own fp32-acc config, so this never touches the F0 path.
        self.lstm_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
        )
        self._cnn_blocks = tuple(
            TTTextEncoderConvLNBlock(
                device=device,
                params=bp,
                ln_eps=params.ln_eps,
                compute_kernel_config=self.compute_kernel_config,
                conv_compute_kernel_config=self.conv_compute_kernel_config,
            )
            for bp in params.blocks
        )
        self._recurrent_program_config = _fused_recurrent_program_config(params.lstm_fwd.hidden_size, device)

    def forward(
        self,
        input_ids: torch.LongTensor,
        input_lengths: torch.LongTensor,
        text_mask: Optional[torch.Tensor] = None,
        *,
        mask_keep_float: Optional[torch.Tensor] = None,
    ) -> ttnn.Tensor:
        """
        Args:
            input_ids: ``[B, T]`` token indices (CPU or CUDA tensor; copied to device).
            input_lengths: ``[B]`` valid length per row (CPU long, as in reference).
            text_mask: ``[B, T]`` bool, ``True`` where positions are masked out.
            mask_keep_float: optional pre-computed ``[B, T, 1]`` float32 keep mask
                (``1.0`` where real, ``0.0`` where padded). When provided, the mask is
                uploaded directly with no torch ops inside forward. When ``None``,
                computed from ``text_mask`` for backward compatibility.

        Returns:
            TTNN tensor ``[B, C, T]`` (channels = ``2 * lstm_hidden``), layout TILE.
        """
        dev = self.device
        B, T = input_ids.shape

        # Run the whole CNN stack batch-flattened in the conv's native rank-4 shape: feed ids as
        # [1, 1, B*T] so the embedding lands as [1, 1, B*T, C] directly and the batched conv chains
        # conv→LN→activation with no per-stage reshape (each ReshapeView is a ~5µs dispatch). The
        # only un-flatten ([1, 1, B*T, C] -> [B, T, C]) happens once, right before the LSTM.
        # (Embedding/CNN ops are per-(row, channel), so folding B into the length dim is exact.)
        tt_ids = ttnn.from_torch(
            input_ids.reshape(1, 1, B * T),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=dev,
            # L1-resident indices — the second half of the embedding sweep's L1-input/L1-weight win.
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        # Write the gather output to interleaved L1 (not DRAM). The embedding sweep
        # (perf/test_embedding_text_encoder_perf_sweep.py) found HEIGHT-sharding the output is the
        # fastest gather *in isolation* (N=512: 2.24µs vs 7.99µs DRAM), but block 0's ttnn.conv1d runs
        # with the L1_FULL slice config and cannot size its halo CBs from a *sharded* input — it
        # overflows L1 (5.8 MiB > 1.5 MiB). Re-interleaving the sharded gather before the conv would
        # add a ShardedToInterleaved, a dispatched op this dispatch-bound model can't amortize (cf. the
        # rejected width-sharded recurrent matmul). Interleaved L1 keeps the whole gather→conv handoff
        # L1-resident with zero added ops — the safe, conv-compatible slice of the sweep's win.
        x = ttnn.embedding(
            tt_ids, self.params.embedding_weight, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
        )  # [1, 1, B*T, C]
        ttnn.deallocate(tt_ids)

        # When there's no padding the keep-mask is all-ones and every ``x * mask_keep`` is the
        # identity (matching the reference ``masked_fill_`` no-op). Detect that on the host and
        # skip building/uploading the mask and all 5 multiplies (embedding, 3 CNN blocks, post-
        # LSTM) entirely — bit-exact for the full-length / single-utterance path. The mask matches
        # the flattened CNN activations ([1, 1, B*T, 1]).
        if mask_keep_float is not None:
            needs_mask = bool((mask_keep_float < 1.0).any())
            mask_keep = (
                ttnn.from_torch(
                    mask_keep_float.reshape(1, 1, B * T, 1),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=dev,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                if needs_mask
                else None
            )
        elif text_mask is not None and bool(text_mask.any()):
            mask_keep = _mask_keep_flat(text_mask, device=dev)
        else:
            # No padding (text_mask all-False or absent): keep-mask is all-ones -> identity.
            mask_keep = None

        if mask_keep is not None:
            # The mask multiply pairs the gather output with a DRAM-interleaved mask, so re-interleave
            # the (height-sharded) embedding output first; the conv reshards its input anyway.
            x = ttnn.multiply(_maybe_interleaved(x), mask_keep, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Chain the CNN blocks keeping activations block-sharded in L1 between stages: each conv accepts
        # the previous LeakyReLU's sharded output directly, so only the last stage re-interleaves to DRAM
        # for the reshape→LSTM (the masked path also re-interleaves, gated inside the block).
        last = len(self._cnn_blocks) - 1
        for i, blk in enumerate(self._cnn_blocks):
            x = blk.forward(x, mask_keep, batch=B, seq=T, keep_sharded=(i != last))

        # Single un-flatten [1, 1, B*T, C] -> [B, T, C] for the per-batch BiLSTM recurrence.
        x = ttnn.reshape(x, [B, T, x.shape[-1]], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if mask_keep is not None:
            # Post-LSTM masking needs the per-batch [B, T, 1] shape; reshape the flattened mask once
            # (padded path only — None in the common full-length case).
            mask_keep_bt = ttnn.reshape(mask_keep, [B, T, 1], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(mask_keep)
            mask_keep = mask_keep_bt

        lengths_list: Sequence[int] = input_lengths.detach().cpu().tolist()
        x = tt_bilstm_nlc(
            x_nlc=x,
            fwd=self.params.lstm_fwd,
            rev=self.params.lstm_rev,
            compute_kernel_config=self.lstm_compute_kernel_config,
            sequence_lengths=lengths_list,
            w_h_block=self.params.lstm_w_h_block,
            # TextEncoder is the one LSTM that tolerates the gate-sum rounding change; fold the
            # per-step gates_x add into the recurrent matmul bias (one fewer BinaryNg/step).
            fold_gates_bias=True,
            # Tuned 1D mcast config for the per-step recurrent matmul (interleaved, loop-safe).
            recurrent_program_config=self._recurrent_program_config,
            # Fuse the per-step cell-state update f*c + tanh(g)*i into one ttnn.mac (one fewer
            # BinaryNg/step). TextEncoder-only — the shared F0-feeding prosody/duration BiLSTMs
            # reject the MAC accumulation-rounding change (see _lstm_step_fused).
            fuse_cell_math=True,
        )

        if mask_keep is not None:
            x = ttnn.multiply(x, mask_keep, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(mask_keep)

        return ttnn.permute(x, (0, 2, 1))

    __call__ = forward
