# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from loguru import logger

import ttnn

from ...layers.embeddings import Embedding
from ...layers.linear import ColParallelLinear, RowParallelLinear
from ...layers.module import Module, ModuleList
from ...layers.normalization import RMSNorm
from ...parallel.config import EncoderParallelConfig
from ...parallel.manager import CCLManager
from ...utils import tensor

if TYPE_CHECKING:
    from collections.abc import Sequence

# Prompt sequence-length bucket. Prompt seq lens are rounded UP to a multiple of this
# (to a tile multiple below it) so only a handful of distinct shapes ever reach the device,
# minimizing JIT recompiles while bounding padding waste. This is purely that bucketing
# tradeoff — SDPA pads+masks a partial last chunk internally, so nothing needs to be aligned
# to it. (Also reused as the flash-SDPA q/k chunk size, an independent perf-tiling knob.)
SEQ_BUCKET_SIZE = 128


@dataclass
class Qwen3VlContext:
    device: ttnn.MeshDevice
    tp_axis: int | None
    ccl_manager: CCLManager | None
    fsdp_mesh_axis: int | None = None
    # When set, every decoder linear (qkv/o/gate/up/down projections) is built with this compute
    # kernel config instead of the tt_dit-wide default from `linear_compute_config`. See
    # `Qwen3VlTextEncoder`'s `high_fidelity_linears`.
    linear_compute_kernel_config: object | None = None
    # Sequence parallelism: shard the sequence on this axis and attend across shards with causal ring
    # attention. Only the causal path is supported. Composes with FSDP on the same axis: FSDP shards
    # WEIGHTS (all-gathered to full immediately before use, so every device on the axis applies
    # identical weights) while SP shards ACTIVATION rows -- the two never interact, the classical
    # FSDP-over-the-data-axis arrangement. They do share the axis's link bandwidth, so their
    # collectives serialize; that is a perf tradeoff, not a correctness constraint.
    # Default None keeps every existing caller on the TP-only path, byte-for-byte.
    #
    # The ring uses contiguous shards (`is_balanced=False`). The zigzag-balanced layout was measured
    # (test_ring_causal_sdpa.py::balanced) and gave ~0% at the block level and ~2.5% on attention
    # alone -- the block is matmul-bound and those shards shrink /sp either way -- so it is not wired
    # in; it would only cost a stricter alignment (2*sp*32) and a caller-side reorder for no gain.
    sp_axis: int | None = None

    def __post_init__(self) -> None:
        if self.sp_axis is None:
            return
        if self.sp_axis == self.tp_axis:
            raise ValueError(f"sp_axis ({self.sp_axis}) must differ from tp_axis ({self.tp_axis})")
        if self.ccl_manager is None:
            raise ValueError("decoder SP needs a ccl_manager for the ring all-gather")


def vision_token_runs(input_ids: torch.Tensor, image_token_id: int | Sequence[int]) -> list[tuple[int, int]]:
    """`(start, length)` of each contiguous run of a vision pad token in a single sequence.

    Qwen3-VL presentations never interleave a vision block with anything: MiniMax-H3 emits a
    `"<Picture i>: "` label, then `<|vision_start|>`, then one run of `<|image_pad|>`, then
    `<|vision_end|>`. So one image is one run, and the merged vision tokens map onto the runs in order.

    `image_token_id` may be several ids, which `ref2va` needs: its presentation mixes
    `<|image_pad|>` runs (one per image reference) with `<|video_pad|>` runs (one per merged frame
    pair of a video reference). Runs come back in sequence order regardless of pad id, in one
    interleaved list, because `_scatter_rows` consumes the tower's rows in run order and the caller
    assembles that output in presentation order.

    A run boundary is a change of token, so two adjacent runs of different pad ids are two runs. That
    does not arise in a MiniMax-H3 presentation, where a `"<{t} seconds>"` label separates two video
    blocks, but merging them would mis-slice.
    """
    pad_ids = {int(image_token_id)} if isinstance(image_token_id, int) else {int(i) for i in image_token_id}
    if input_ids.ndim == 2:
        if input_ids.shape[0] != 1:
            msg = f"expected a single sequence, got batch {input_ids.shape[0]}"
            raise ValueError(msg)
        input_ids = input_ids[0]
    runs: list[tuple[int, int]] = []
    start, current = None, None
    for index, token in enumerate(input_ids.tolist()):
        if token in pad_ids:
            if start is None:
                start, current = index, token
            elif token != current:
                # A different pad id: close the run and open a new one at this row.
                runs.append((start, index - start))
                start, current = index, token
        elif start is not None:
            runs.append((start, index - start))
            start, current = None, None
    if start is not None:
        runs.append((start, len(input_ids) - start))
    return runs


def _scatter_rows(base: ttnn.Tensor, values: ttnn.Tensor, runs: Sequence[tuple[int, int]], *, add: bool) -> ttnn.Tensor:
    """Write `values` into the row ranges `runs` of `base`, either replacing or adding.

    Done by slicing and concatenating rather than a masked scatter: the vision positions are contiguous
    runs, so this is exact, and row slicing on the sequence axis is already how this module trims its
    padding. `values` rows are consumed in run order.
    """
    # The vision tower emits `(rows, hidden)` while the sequence buffer is `(batch, seq, hidden)`;
    # normalize so the slicing below is rank-agnostic.
    if len(values.shape) == 2:
        values = ttnn.reshape(values, (1, values.shape[0], values.shape[1]))

    total = sum(length for _, length in runs)
    if values.shape[-2] != total:
        msg = f"runs cover {total} rows but values has {values.shape[-2]}"
        raise ValueError(msg)

    pieces: list[ttnn.Tensor] = []
    cursor = taken = 0
    for start, length in runs:
        if start < cursor:
            msg = f"runs must be sorted and disjoint; {start} overlaps {cursor}"
            raise ValueError(msg)
        if start > cursor:
            pieces.append(base[:, cursor:start, :])
        chunk = values[:, taken : taken + length, :]
        pieces.append(ttnn.add(base[:, start : start + length, :], chunk) if add else chunk)
        cursor, taken = start + length, taken + length
    if cursor < base.shape[-2]:
        pieces.append(base[:, cursor:, :])
    return ttnn.concat(pieces, dim=-2) if len(pieces) > 1 else pieces[0]


# adapted from https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L769
class Qwen3VlTextEncoder(Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        rms_norm_eps: float,
        rope_theta: float,
        mrope_section: Sequence[int],
        head_dim: int | None = None,
        activation_layers: Sequence[int] | None = None,
        device: ttnn.MeshDevice,
        parallel_config: EncoderParallelConfig | None = None,
        ccl_manager: CCLManager | None = None,
        is_fsdp: bool = False,
        high_fidelity_linears: bool = False,
    ) -> None:
        super().__init__()

        # Qwen3-VL declares `head_dim` explicitly and it is NOT always `hidden_size // num_heads`:
        # Qwen3-VL-8B happens to satisfy that (4096 // 32 == 128), but MiniMax-H3's conditioner does
        # not (5120 // 64 == 80, while the checkpoint's head_dim is 128, giving an inner dimension of
        # 8192 wider than the residual stream). Pass the config's value; the derivation is only a
        # fallback for callers whose checkpoint omits it.
        if head_dim is None:
            if hidden_size % num_attention_heads != 0:
                msg = (
                    f"cannot derive head_dim: hidden_size {hidden_size} is not divisible by "
                    f"num_attention_heads {num_attention_heads}. Pass `head_dim` explicitly."
                )
                raise ValueError(msg)
            head_dim = hidden_size // num_attention_heads

        # Sequence parallelism: shard the sequence on the SP axis (the non-TP axis) and ring the
        # causal attention over it. Composes with FSDP on the same axis (weights vs activation rows;
        # see Qwen3VlContext).
        sp_axis = None
        if parallel_config is not None and parallel_config.sequence_parallel is not None:
            sp_axis = parallel_config.sequence_parallel.mesh_axis

        # FSDP: For encoders, we can only use FSDP if there's a separate axis from TP.
        # Since the encoder runs on a submesh (e.g., 1x4), we need to check if the other axis
        # has size > 1. If the mesh is 1xN, FSDP can't be enabled because there's no second axis.
        fsdp_mesh_axis = None
        if is_fsdp and parallel_config is not None:
            tp_axis = parallel_config.tensor_parallel.mesh_axis
            # Check if there's a different axis that can be used for FSDP
            other_axis = 1 - tp_axis  # If TP is on axis 1, check axis 0; if TP is on axis 0, check axis 1
            if device.shape[other_axis] > 1:
                fsdp_mesh_axis = other_axis
            if fsdp_mesh_axis is None:
                logger.warning(
                    f"Qwen3-VL: FSDP was requested but is disabled — no mesh axis other than the "
                    f"tensor-parallel axis has size > 1 (mesh shape {tuple(device.shape)})."
                )

        # `high_fidelity_linears` builds every decoder linear at HiFi4 instead of the tt_dit-wide
        # HiFi2 default. Everything else (fp32 DEST accumulate, packer L1 accumulate, non-approx)
        # matches the default. Off by default: one model's measurement is not evidence about the
        # others sharing this module.
        linear_compute_kernel_config = None
        if high_fidelity_linears:
            linear_compute_kernel_config = ttnn.init_device_compute_kernel_config(
                device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )

        ctx = Qwen3VlContext(
            device=device,
            tp_axis=parallel_config.tensor_parallel.mesh_axis if parallel_config is not None else None,
            ccl_manager=ccl_manager,
            fsdp_mesh_axis=fsdp_mesh_axis,
            linear_compute_kernel_config=linear_compute_kernel_config,
            sp_axis=sp_axis,
        )

        if ctx.tp_axis is not None and ctx.ccl_manager is None:
            msg = "ccl_manager must be provided if tensor parallelism is used"
            raise ValueError(msg)

        self.embed_tokens = Embedding(vocab_size, hidden_size, device=ctx.device, mesh_axis=ctx.tp_axis)
        self.layers = ModuleList(
            Qwen3VlDecoderLayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=hidden_act,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                rms_norm_eps=rms_norm_eps,
                head_dim=head_dim,
                ctx=ctx,
            )
            for _ in range(num_hidden_layers)
        )
        self.norm = Qwen3VlRmsNorm(hidden_size, eps=rms_norm_eps, ctx=ctx)

        self._device = ctx.device
        self._tp_axis = ctx.tp_axis
        self._ccl_manager = ctx.ccl_manager
        self._sp_axis = ctx.sp_axis
        self._sp_factor = device.shape[ctx.sp_axis] if ctx.sp_axis is not None else 1
        self._mrope_section = mrope_section
        # See Qwen3VlAttention: head_dim is not derivable from hidden_size / num_heads for every
        # Qwen3-VL checkpoint. The rope tables are built at this width, so it must agree.
        self._head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self._rope_theta = rope_theta
        # When set, forward returns the raw hidden states after each of these decoder layers
        # (no final norm) — the Ideogram4 multi-layer feature tap. Otherwise returns the
        # final normalized hidden state.
        self._activation_layers = tuple(activation_layers) if activation_layers is not None else None

    def forward(
        self,
        input_ids: ttnn.Tensor,
        *,
        attention_mask: ttnn.Tensor | None = None,
        pos_embeds: tuple[ttnn.Tensor, ttnn.Tensor],
        vision_embeds: ttnn.Tensor | None = None,
        vision_runs: Sequence[tuple[int, int]] | None = None,
        deepstack_embeds: Sequence[ttnn.Tensor] | None = None,
    ) -> list[ttnn.Tensor]:
        """Args beyond the text path, all optional so text-only callers are unaffected:

        vision_embeds: the vision tower's merged tokens, which *replace* the embeddings of the
            `<|image_pad|>` rows named by `vision_runs` (see [`vision_token_runs`]).
        deepstack_embeds: one feature per entry of the tower's `deepstack_visual_indexes`, *added* to
            the vision rows after decoder layers `0 .. len(deepstack_embeds) - 1`. The reference keys
            these off the list length, not the vision layer indexes they came from.
        """
        if (vision_embeds is None) != (vision_runs is None):
            msg = "vision_embeds and vision_runs must be passed together"
            raise ValueError(msg)
        if deepstack_embeds and vision_runs is None:
            msg = "deepstack_embeds needs vision_runs to know which rows to add to"
            raise ValueError(msg)
        if self._sp_axis is not None and attention_mask is not None:
            # SP routes attention through the causal ring (`is_causal=True`), which admits no explicit
            # bias. The MiniMax-H3 conditioner passes a single un-padded presentation with no mask, so
            # this only fires if a future caller wants masked SP.
            msg = "sequence-parallel decoder supports only the causal (no-mask) path"
            raise ValueError(msg)
        batch_size, seq_len = input_ids.shape

        if attention_mask is not None:
            if seq_len < SEQ_BUCKET_SIZE:
                # short prompt: bucket to the tile multiple (fine granularity, little waste)
                padded_seq_len = -(-seq_len // 32) * 32
            else:
                # longer prompt: bucket to a multiple of SEQ_BUCKET_SIZE (fewer JIT compiles)
                padded_seq_len = -(-seq_len // SEQ_BUCKET_SIZE) * SEQ_BUCKET_SIZE

            input_ids = ttnn.pad(input_ids, [(0, padded_seq_len - seq_len)], value=0)
            pos_embeds = tuple(ttnn.pad(x, [(0, padded_seq_len - seq_len), (0, 0)], value=0) for x in pos_embeds)

            assert attention_mask.shape == (batch_size, seq_len)
            attention_mask = ttnn.pad(attention_mask, [(0, padded_seq_len - seq_len)], value=0)
            attention_bias = prepare_attention_bias(attention_mask)
        elif self._sp_axis is not None:
            # SP needs each shard tile-aligned: pad the sequence to a multiple of sp*32. The tail pad
            # rows are harmless under causal attention (real rows never attend forward into them) and
            # are sliced off after the final gather. Same pad idiom as the masked path above.
            padded_seq_len = -(-seq_len // (self._sp_factor * 32)) * (self._sp_factor * 32)
            if padded_seq_len != seq_len:
                input_ids = ttnn.pad(input_ids, [(0, padded_seq_len - seq_len)], value=0)
                pos_embeds = tuple(ttnn.pad(x, [(0, padded_seq_len - seq_len), (0, 0)], value=0) for x in pos_embeds)
            attention_bias = None
        else:
            # padding is only required by `ttnn.transformer.scaled_dot_product_attention` when using
            # an attention mask
            padded_seq_len = seq_len

            attention_bias = None

        del attention_mask

        input_embeds = self.embed_tokens.forward(input_ids)

        if self._tp_axis is not None:
            input_embeds = self._ccl_manager.all_gather_persistent_buffer(
                input_embeds, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True
            )
            # clone to move out of persistent buffer
            input_embeds = ttnn.clone(input_embeds)

        if vision_embeds is not None:
            input_embeds = _scatter_rows(input_embeds, vision_embeds, vision_runs, add=False)

        # Sequence parallelism: everything above ran on the full (replicated) sequence -- including the
        # vision scatter -- so no per-shard offset logic is needed. Now SP-shard the stream (and the
        # rotary tables) on the sequence, and pre-build the deepstack adds the same way: scatter each
        # feature into a zero base on the full sequence, then shard it, so the mid-stack deepstack step
        # becomes a plain sharded add rather than a shard-aware scatter.
        deepstack_sharded: list[ttnn.Tensor] | None = None
        if self._sp_axis is not None:
            if deepstack_embeds:
                zero_base = ttnn.zeros_like(input_embeds)
                deepstack_sharded = [
                    ttnn.mesh_partition(
                        _scatter_rows(zero_base, ds, vision_runs, add=False), dim=1, cluster_axis=self._sp_axis
                    )
                    for ds in deepstack_embeds
                ]
            input_embeds = ttnn.mesh_partition(input_embeds, dim=1, cluster_axis=self._sp_axis)
            pos_embeds = tuple(ttnn.mesh_partition(x, dim=2, cluster_axis=self._sp_axis) for x in pos_embeds)

        hidden_states = input_embeds
        captured: list[ttnn.Tensor] = []

        for layer_idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer.forward(
                hidden_states,
                attention_bias=attention_bias,
                pos_embeds=pos_embeds,
            )
            # Vision also enters here, not only at the embeddings: the tower's intermediate features are
            # added to the vision rows of the first few layers. Under SP the add is a plain sharded add
            # against the pre-sharded deepstack tensor (built above); otherwise a row-range scatter-add.
            if deepstack_embeds and layer_idx < len(deepstack_embeds):
                if self._sp_axis is not None:
                    hidden_states = ttnn.add(hidden_states, deepstack_sharded[layer_idx])
                else:
                    hidden_states = _scatter_rows(hidden_states, deepstack_embeds[layer_idx], vision_runs, add=True)
            if self._activation_layers is not None and layer_idx in self._activation_layers:
                captured.append(hidden_states)

        if self._activation_layers is None:
            # default: final normalized hidden state
            captured = [self.norm.forward(hidden_states)]

        if self._sp_axis is not None:
            # Gather the sequence back so the return matches the TP-only contract (full, replicated).
            captured = [
                self._ccl_manager.all_gather_persistent_buffer(x, dim=1, mesh_axis=self._sp_axis, use_hyperparams=True)
                for x in captured
            ]

        if padded_seq_len != seq_len:
            captured = [x[:, :seq_len, :] for x in captured]

        return captured

    def create_rope_tensors(
        self, batch_size: int, sequence_length: int, attention_mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return create_rope_tensors(
            batch_size,
            sequence_length,
            attention_mask,
            self._head_dim,
            self._rope_theta,
            self._mrope_section,
        )


# adapted from https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L684
class Qwen3VlDecoderLayer(Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        intermediate_size: int,
        hidden_act: str,
        rms_norm_eps: float,
        head_dim: int | None = None,
        ctx: Qwen3VlContext,
    ) -> None:
        super().__init__()

        self.self_attn = Qwen3VlAttention(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            rms_norm_eps=rms_norm_eps,
            head_dim=head_dim,
            ctx=ctx,
        )
        self.mlp = Qwen3VlMlp(
            hidden_size=hidden_size, intermediate_size=intermediate_size, hidden_act=hidden_act, ctx=ctx
        )
        self.input_layernorm = Qwen3VlRmsNorm(hidden_size, eps=rms_norm_eps, ctx=ctx)
        self.post_attention_layernorm = Qwen3VlRmsNorm(hidden_size, eps=rms_norm_eps, ctx=ctx)

    def forward(
        self,
        x: ttnn.Tensor,
        *,
        attention_bias: ttnn.Tensor | None = None,
        pos_embeds: tuple[ttnn.Tensor, ttnn.Tensor],
    ) -> ttnn.Tensor:
        residual = x
        x = self.input_layernorm.forward(x)
        x = self.self_attn.forward(x, attention_bias=attention_bias, pos_embeds=pos_embeds)
        x = x + residual

        residual = x
        x = self.post_attention_layernorm.forward(x)
        x = self.mlp.forward(x)
        return x + residual


# adapted from https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L590
class Qwen3VlAttention(Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: int,
        rms_norm_eps: float,
        head_dim: int | None = None,
        ctx: Qwen3VlContext,
    ) -> None:
        super().__init__()

        if ctx.tp_axis is not None:
            assert ctx.ccl_manager is not None

        # Qwen3-VL does not require `head_dim == hidden_size // num_heads`, and the larger
        # checkpoints do not satisfy it: the 32B-class model MiniMax-H3 conditions on has
        # hidden_size 5120 with 64 heads of 128, so `q_proj` is [8192, 5120] rather than square.
        # `num_heads * head_dim` need not equal `hidden_size`: the projections below are sized from
        # `head_dim` throughout, so a q/k/v inner dimension wider (or narrower) than the residual
        # stream is supported. Passing head_dim explicitly is therefore required for those
        # checkpoints; the derivation stays the default because it is correct for the 8B
        # (4096 / 32 = 128) that Ideogram-4 loads.
        if head_dim is None:
            if hidden_size % num_heads != 0:
                msg = f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
                raise ValueError(msg)
            head_dim = hidden_size // num_heads

        # Qwen3 (unlike Qwen2.5) applies per-head RMSNorm to q and k (over head_dim),
        # after the qkv projection / head split and before RoPE.
        self.q_norm = Qwen3VlRmsNorm(head_dim, eps=rms_norm_eps, ctx=ctx)
        self.k_norm = Qwen3VlRmsNorm(head_dim, eps=rms_norm_eps, ctx=ctx)

        tp_factor = ctx.device.shape[ctx.tp_axis] if ctx.tp_axis is not None else 1
        group_count = num_key_value_heads
        group_size = num_heads // num_key_value_heads

        opt_group_count, opt_group_size, split_factor = optimal_groups(group_count, group_size, tp_factor)
        padded_heads = opt_group_count * opt_group_size

        self.qkv_proj = ColParallelLinear(
            hidden_size,
            (padded_heads + 2 * opt_group_count) * head_dim,
            bias=False,  # Qwen3 attention has no qkv bias (Qwen2.5 did)
            mesh_device=ctx.device,
            mesh_axis=ctx.tp_axis,
            fsdp_mesh_axis=ctx.fsdp_mesh_axis,
            ccl_manager=ctx.ccl_manager,
            compute_kernel_config=ctx.linear_compute_kernel_config,
        )
        self.o_proj = ColParallelLinear(
            padded_heads * head_dim,
            hidden_size,
            bias=False,
            mesh_device=ctx.device,
            mesh_axis=ctx.tp_axis,
            fsdp_mesh_axis=ctx.fsdp_mesh_axis,
            ccl_manager=ctx.ccl_manager,
            compute_kernel_config=ctx.linear_compute_kernel_config,
        )

        self._sdpa_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            # packer_l1_acc=True,
        )

        self._head_dim = head_dim
        self._group_count = group_count
        self._group_size = group_size
        self._num_local_heads = padded_heads // tp_factor
        self._num_local_kv_heads = opt_group_count // tp_factor
        self._group_size_padding = opt_group_size * split_factor - group_size
        self._group_count_padding = opt_group_count - group_count * split_factor
        self._split_factor = split_factor
        self._tp_axis = ctx.tp_axis
        self._tp_factor = tp_factor
        self._device = ctx.device
        self._ccl_manager = ctx.ccl_manager
        self._sp_axis = ctx.sp_axis
        self._sp_factor = ctx.device.shape[ctx.sp_axis] if ctx.sp_axis is not None else 1

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        def _prepare_qkv(q: ttnn.Tensor, k: ttnn.Tensor, v: ttnn.Tensor) -> ttnn.Tensor:
            q = q.unflatten(0, [self._group_count, self._group_size, self._head_dim])
            k = k.unflatten(0, [self._group_count, 1, self._head_dim])
            v = v.unflatten(0, [self._group_count, 1, self._head_dim])

            # pad group size
            q = _pad(q, self._group_size_padding, dim=1)

            # split groups
            s = self._split_factor
            q = q.flatten(0, 1).unflatten(0, [self._group_count * s, -1])
            k = k.repeat_interleave(s, dim=0)
            v = v.repeat_interleave(s, dim=0)

            # pad group count
            q = _pad(q, self._group_count_padding, dim=0)
            k = _pad(k, self._group_count_padding, dim=0)
            v = _pad(v, self._group_count_padding, dim=0)

            # fuse
            q = q.flatten(0, 1).unflatten(0, [self._tp_factor, self._num_local_heads])
            k = k.flatten(0, 1).unflatten(0, [self._tp_factor, self._num_local_kv_heads])
            v = v.flatten(0, 1).unflatten(0, [self._tp_factor, self._num_local_kv_heads])

            return torch.cat([q, k, v], dim=1).flatten(0, 2)

        if "q_proj.weight" in state and "k_proj.weight" in state and "v_proj.weight" in state:
            state["qkv_proj.weight"] = _prepare_qkv(
                state.pop("q_proj.weight"), state.pop("k_proj.weight"), state.pop("v_proj.weight")
            )

        if "q_proj.bias" in state and "k_proj.bias" in state and "v_proj.bias" in state:
            state["qkv_proj.bias"] = _prepare_qkv(
                state.pop("q_proj.bias"), state.pop("k_proj.bias"), state.pop("v_proj.bias")
            )

        if "o_proj.weight" in state:
            o = state["o_proj.weight"]

            o = o.unflatten(1, [self._group_count, self._group_size, self._head_dim])

            # pad group size
            o = _pad(o, self._group_size_padding, dim=2)

            # split groups
            o = o.flatten(1, 2).unflatten(1, [self._group_count * self._split_factor, -1])

            # pad group count
            o = _pad(o, self._group_count_padding, dim=1)

            state["o_proj.weight"] = o.flatten(1, 3)

    def forward(
        self,
        x: ttnn.Tensor,
        *,
        attention_bias: ttnn.Tensor | None,
        pos_embeds: tuple[ttnn.Tensor, ttnn.Tensor],
    ) -> ttnn.Tensor:
        x = self.qkv_proj.forward(x)

        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            ttnn.unsqueeze(x, 1),
            num_heads=self._num_local_heads,
            num_kv_heads=self._num_local_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        q = self.q_norm.forward(q)
        k = self.k_norm.forward(k)

        cos, sin = pos_embeds
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        if self._sp_axis is not None:
            # Sequence sharded on the SP axis: attend across shards with causal ring attention. Only
            # the causal path is supported here -- an explicit bias would need per-shard slicing.
            assert attention_bias is None, "SP decoder attention supports only the causal path (attention_bias=None)"
            x = self._ring_attention(q, k, v)
        else:
            x = ttnn.transformer.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_bias,
                is_causal=attention_bias is None,
                program_config=self._sdpa_program_config(q.shape[2]),
                compute_kernel_config=self._sdpa_compute_kernel_config,
            )

        x = ttnn.transformer.concatenate_heads(x)

        if self._tp_axis is not None:
            x = self._ccl_manager.all_gather_persistent_buffer(x, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True)

        x = self.o_proj.forward(x)

        if self._tp_axis is not None:
            x = self._ccl_manager.all_gather_persistent_buffer(x, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True)

        return x

    def _sdpa_program_config(self, seq_len: int) -> ttnn.SDPAProgramConfig:
        grid_size = self._device.compute_with_storage_grid_size()

        seq_len = -(-seq_len // 32) * 32
        chunk_size = min(seq_len, SEQ_BUCKET_SIZE)  # flash q/k tiling; SDPA handles a partial last chunk

        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid_size,
            q_chunk_size=chunk_size,
            # k 512: fewer streaming-softmax rescales (each rounds bf16; see the tower's
            # _windowed_program_config for the sweep). Fidelity-neutral here -- the causal decoder's
            # attention error was already sub-noise -- but ~2-5 % faster (fewer K iterations).
            k_chunk_size=min(seq_len, 512),
            exp_approx_mode=False,
        )

    def _ring_program_config(self, local_seq_len: int) -> tuple[ttnn.SDPAProgramConfig, tuple[int, int]]:
        """Flash tiling for the ring, sized from the LOCAL shard, reserving the last core column for the
        CCL workers (mirrors `vision_qwen3vl.py::_ring_program_config`). Returns the config and the
        worker grid so the caller can point `ccl_core_grid_offset` at the reserved column."""
        full_grid = self._device.compute_with_storage_grid_size()
        worker_grid = (full_grid.x - 1, full_grid.y)
        chunk = min(-(-local_seq_len // 32) * 32, SEQ_BUCKET_SIZE)
        cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=worker_grid,
            q_chunk_size=chunk,
            # k 512 within each ring shard (self-capping when the shard is shorter, e.g. any
            # sp32 config or short prompts): same rationale and measurement as the causal path.
            k_chunk_size=min(-(-local_seq_len // 32) * 32, 512),
            exp_approx_mode=False,
        )
        return cfg, worker_grid

    def _ring_attention(self, q: ttnn.Tensor, k: ttnn.Tensor, v: ttnn.Tensor) -> ttnn.Tensor:
        """Causal attention over a sequence sharded on the SP axis (FSDP-off configs only).

        `ring_joint_scaled_dot_product_attention(is_causal=True)` gathers k/v around the SP ring while
        streaming the softmax, so no device materializes the full `s x s` score matrix; the joint slots
        are zero-width to keep it pure self-attention. Contiguous shards (`is_balanced=False`): correct
        but not load-balanced (the causal-load zigzag is a later perf refinement). Validated at this
        head geometry by `tests/encoders/qwen3vl/test_ring_causal_sdpa.py`.
        """
        sp_axis, ccl = self._sp_axis, self._ccl_manager
        local_seq_len = q.shape[2]
        pc, worker_grid = self._ring_program_config(local_seq_len)
        empty_q = tensor.bf16_tensor(torch.zeros(1, self._num_local_heads, 0, self._head_dim), device=self._device)
        empty_kv = tensor.bf16_tensor(torch.zeros(1, self._num_local_kv_heads, 0, self._head_dim), device=self._device)
        attn, _joint, _lse = ttnn.transformer.ring_joint_scaled_dot_product_attention(
            q,
            k,
            v,
            empty_q,
            empty_kv,
            empty_kv,
            persistent_output_buffer_k=ccl.get_ag_ping_pong_buffer(k.shape, 2, sp_axis, dtype=k.dtype),
            persistent_output_buffer_v=ccl.get_ag_ping_pong_buffer(v.shape, 2, sp_axis, dtype=v.dtype),
            joint_strategy="rear",
            logical_n=local_seq_len * self._sp_factor,
            program_config=pc,
            compute_kernel_config=self._sdpa_compute_kernel_config,
            dim=2,
            scale=self._head_dim**-0.5,
            multi_device_global_semaphore=ccl.get_ag_ping_pong_semaphore(sp_axis),
            num_links=ccl.num_links,
            cluster_axis=sp_axis,
            mesh_device=self._device,
            topology=ccl.topology,
            subdevice_id=ccl.ccl_sub_device_id,
            ccl_core_grid_offset=(worker_grid[0], 0),
            use_column_major_ccl=True,
            is_causal=True,
            is_balanced=False,
        )
        return attn


# adapted from https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L529
class Qwen3VlMlp(Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        ctx: Qwen3VlContext,
    ) -> None:
        super().__init__()

        if ctx.tp_axis is not None:
            assert ctx.ccl_manager is not None

        # intermediate_size is much greater than hidden_size
        self.gate_proj = ColParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
            mesh_device=ctx.device,
            mesh_axis=ctx.tp_axis,
            fsdp_mesh_axis=ctx.fsdp_mesh_axis,
            ccl_manager=ctx.ccl_manager,
            compute_kernel_config=ctx.linear_compute_kernel_config,
        )
        self.up_proj = ColParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
            mesh_device=ctx.device,
            mesh_axis=ctx.tp_axis,
            fsdp_mesh_axis=ctx.fsdp_mesh_axis,
            ccl_manager=ctx.ccl_manager,
            compute_kernel_config=ctx.linear_compute_kernel_config,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            mesh_device=ctx.device,
            mesh_axis=ctx.tp_axis,
            fsdp_mesh_axis=ctx.fsdp_mesh_axis,
            ccl_manager=ctx.ccl_manager,
            compute_kernel_config=ctx.linear_compute_kernel_config,
        )

        if hidden_act != "silu":
            msg = f"unsupported activation function: {hidden_act}"
            raise ValueError(msg)
        self.act_fn = ttnn.silu

        self._ccl_manager = ctx.ccl_manager
        self._tp_axis = ctx.tp_axis

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = self.act_fn(self.gate_proj.forward(x)) * self.up_proj.forward(x)
        x = self.down_proj(x)

        if self._tp_axis is not None:
            x = self._ccl_manager.all_gather_persistent_buffer(x, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True)

        return x


class Qwen3VlRmsNorm(RMSNorm):
    def __init__(self, size: int, *, eps: float, ctx: Qwen3VlContext) -> None:
        super().__init__(size, norm_eps=eps, bias=False, mesh_device=ctx.device)

        self._compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return super().forward(x, compute_kernel_config=self._compute_kernel_config)


def _apply_rope(x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
    # Half-split RoPE (rotate_half), matching the HF reference layout directly. Holds PCC >=0.99
    # (real weights) and 0.9899-0.9953 per tapped layer (random init, bf16 error over 36 layers).
    #
    # The denoiser instead permutes Q/K + cos/sin to the interleaved (adjacent-pair) layout to use
    # the fused ttnn.experimental.rotary_embedding_llama (rope_halfsplit_to_interleaved*); that
    # conversion is numerically neutral there. Not adopted here: it needs a head_dim channel
    # permutation baked into the qkv weights, and the encoder RoPE is a negligible slice of latency
    # (the encoder is ~1.5% of end-to-end), so the fused op buys ~nothing. If it is ever adopted,
    # cos/sin must be converted to adjacent-pair along with the rotate -- mixing the two layouts
    # regresses PCC silently.
    return x * cos + _rotate_half(x) * sin


def _rotate_half(x: ttnn.Tensor) -> ttnn.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return ttnn.concat([ttnn.neg(x2), x1], dim=-1)


def optimal_groups(group_count: int, group_size: int, device_count: int) -> tuple[int, int, int]:
    # In order to distribute heads evenly on devices, three operations are possibly performed:
    # 1. Pad to increase group size.
    # 2. Pad to increase group count (= number of key/value heads).
    # 3. Split groups into smaller groups defined by a split factor.
    # For a particular split factor, padding sizes follow from the requirements that the padded
    # group size must be divisible by this factor and the new group count must be divisible by the
    # device count. We choose this factor such that memory requirements are minimized.

    best_split_factor = 1
    best_size = math.inf
    best_group_count = group_count
    best_group_size = group_size

    for s in range(1, group_size + 1):
        new_group_size = -(-group_size // s)  # = ceil(group_size / s)
        new_group_count = -(-group_count * s // device_count) * device_count

        # query heads + 2 * key/value heads
        size = new_group_size * new_group_count + 2 * new_group_count

        if size < best_size:
            best_size = size
            best_split_factor = s
            best_group_count = new_group_count
            best_group_size = new_group_size

    return best_group_count, best_group_size, best_split_factor


def _pad(t: torch.Tensor, amount: int, *, dim: int) -> torch.Tensor:
    """Pad tensor with `amount` zeros on the end of dimension `dim`."""
    padding = [0] * (2 * t.ndim)
    padding[-(dim * 2 + 1)] = amount
    return torch.nn.functional.pad(t, padding)


def prepare_attention_bias(attention_mask: ttnn.Tensor) -> ttnn.Tensor:
    batch_size, seq_len = attention_mask.shape

    # convert to causal attention mask
    attention_mask = attention_mask.reshape([batch_size, 1, 1, seq_len])
    attention_mask = ttnn.expand(attention_mask, [-1, -1, seq_len, -1])
    attention_mask = tensor.tril(attention_mask)  # tt_dit util: tracing-safe + mask-cached (cf. qwen25vl)

    attention_mask = (attention_mask - 1.0) * math.inf

    return ttnn.clone(attention_mask, dtype=ttnn.bfloat4_b)


# adapted from https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L491
# and https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L545
def vision_position_ids(
    start_position: int,
    grid_thw: Sequence[int] | torch.Tensor,
    *,
    temp_merge_size: int = 1,
    spatial_merge_size: int = 1,
    time_interval: int = 1,
) -> torch.Tensor:
    """The `(3, num_vision_tokens)` M-RoPE grid of one image or video block, offset by `start_position`.

    The `(t, h, w)` axes carry *different* positions here, unlike a text run where all three share the
    token index. That divergence is what makes the interleaved layout observable -- see
    [`create_rope_tensors`].

    The repeat patterns below are load-bearing and mirror the reference exactly; `start_position` is
    added to the temporal axis only *after* `time_interval` is applied, and order matters.
    """
    llm_grid_t = int(grid_thw[0]) // temp_merge_size
    llm_grid_h = int(grid_thw[1]) // spatial_merge_size
    llm_grid_w = int(grid_thw[2]) // spatial_merge_size

    position_temporal = torch.arange(llm_grid_t) * time_interval
    position_width = torch.arange(llm_grid_w) + start_position
    position_height = torch.arange(llm_grid_h) + start_position

    position_width = position_width.repeat(llm_grid_h * llm_grid_t)
    position_height = position_height.repeat_interleave(llm_grid_w).repeat(llm_grid_t)
    position_temporal = position_temporal.repeat_interleave(llm_grid_h * llm_grid_w) + start_position

    return torch.stack([position_temporal, position_height, position_width], dim=0)


def mrope_position_ids(
    mm_token_type_ids: torch.Tensor,
    *,
    image_grid_thw: torch.Tensor | None = None,
    video_grid_thw: torch.Tensor | None = None,
    spatial_merge_size: int = 2,
) -> torch.Tensor:
    """The `(3, batch_size, sequence_length)` M-RoPE grid of a multimodal prompt.

    The sequence is walked as runs of one modality, which is what `mm_token_type_ids` (0 text, 1
    image, 2 video -- as produced by `Qwen3VLProcessor.create_mm_token_type_ids`) encodes. A text run
    advances all three axes together by its length; a vision run consumes the next entry of the
    matching `*_grid_thw` and advances the clock by `max(h, w) // spatial_merge_size`.

    Only the position grid is returned. The reference also produces `mrope_position_deltas`, which
    exists to re-base positions for cached incremental decoding; the conditioner runs a single
    prefill with `use_cache=False` and never needs it.

    Padding is not handled: this takes one unpadded sequence per batch item, which is what the
    conditioner feeds (its attention mask is all ones).
    """
    if video_grid_thw is not None:
        # Timestamps separate the frames of a video, so each frame is its own grid.
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0).clone()
        video_grid_thw[:, 0] = 1

    grid_iters = {
        1: iter(image_grid_thw) if image_grid_thw is not None else None,
        2: iter(video_grid_thw) if video_grid_thw is not None else None,
    }

    batch_size, sequence_length = mm_token_type_ids.shape
    position_ids = torch.zeros(3, batch_size, sequence_length, dtype=torch.long)
    for batch_idx in range(batch_size):
        current_pos = 0
        runs = []
        for modality, group in itertools.groupby(enumerate(mm_token_type_ids[batch_idx].tolist()), lambda x: x[1]):
            group = list(group)
            runs.append((modality, group[0][0], group[-1][0] + 1))

        segments = []
        for modality, start_idx, end_idx in runs:
            if modality == 0:
                text_len = end_idx - start_idx
                segments.append(torch.arange(text_len).view(1, -1).expand(3, -1) + current_pos)
                current_pos += text_len
            else:
                if grid_iters[modality] is None:
                    msg = f"mm_token_type_ids contains modality {modality} but no matching grid was passed"
                    raise ValueError(msg)
                grid_thw = next(grid_iters[modality])
                segments.append(vision_position_ids(current_pos, grid_thw, spatial_merge_size=spatial_merge_size))
                current_pos += max(int(grid_thw[1]), int(grid_thw[2])) // spatial_merge_size
        position_ids[:, batch_idx] = torch.cat(segments, dim=1).reshape(3, -1)

    return position_ids


def _apply_interleaved_mrope(freqs: torch.Tensor, mrope_section: Sequence[int]) -> torch.Tensor:
    """Reorganize `(3, batch, seq, head_dim // 2)` freqs from chunked `[TTT..HHH..WWW]` to interleaved
    `[THWTHW..TT]`, preserving frequency continuity. Returns `(batch, seq, head_dim // 2)`."""
    freqs_t = freqs[0]
    for dim, offset in enumerate((1, 2), start=1):  # H, W
        length = mrope_section[dim] * 3
        idx = slice(offset, length, 3)
        freqs_t[..., idx] = freqs[dim, ..., idx]
    return freqs_t


# adapted from https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L491
# and https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L545
def create_rope_tensors(
    batch_size: int,
    sequence_length: int,
    attention_mask: torch.Tensor | None,
    head_dim: int,
    rope_theta: float,
    mrope_section: Sequence[int],
    position_ids: torch.Tensor | None = None,
    interleaved: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(cos, sin)` of shape `(batch, 1, seq, head_dim)` for the decoder's rotary embedding.

    Args:
        position_ids: `(3, batch, sequence_length)` from [`mrope_position_ids`]. Defaults to the token
            index shared by all three axes, which is what a text-only prompt needs.
        interleaved: whether the checkpoint declares `rope_scaling.mrope_interleaved`. The two layouts
            assign the same *frequency* to each output slot and differ only in which axis's position
            feeds it, so they coincide exactly while all three axes agree -- i.e. for a text-only
            prompt, where this flag is a no-op. It becomes observable as soon as `position_ids`
            carries a vision run.
    """
    if position_ids is not None:
        assert position_ids.shape == (
            3,
            batch_size,
            sequence_length,
        ), f"position_ids must be (3, {batch_size}, {sequence_length}), got {tuple(position_ids.shape)}"
    elif attention_mask is not None:
        assert attention_mask.shape == (batch_size, sequence_length)

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
    else:
        position_ids = torch.arange(sequence_length).view(1, 1, -1).expand(3, batch_size, -1)

    # `theta ** -x` rather than the reference's `1 / theta ** x`: mathematically identical, one fp32
    # ulp apart (~1.2e-7 relative), and *not* invisible after the bf16 cast the caller applies -- a few
    # entries land on the other side of a rounding boundary. Do not switch forms: existing callers
    # depend on this one bit-for-bit, and its 1-ulp divergence from the reference is absorbed in the
    # measured PCC.
    inv_freq = rope_theta ** (-torch.arange(0, head_dim, 2, dtype=torch.int64).to(dtype=torch.float) / head_dim)
    inv_freq_expanded = inv_freq[None, None, :, None].float().expand(3, batch_size, -1, 1)
    position_ids_expanded = position_ids[:, :, None, :].float()
    freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)

    if interleaved:
        # The axis selection happens on the freqs, before the duplication and the cos/sin.
        freqs = _apply_interleaved_mrope(freqs, mrope_section)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().unsqueeze(1), emb.sin().unsqueeze(1)

    # Chunked: select per contiguous section, after the cos/sin.
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()

    s = list(mrope_section) * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(s, dim=-1))], dim=-1).unsqueeze(1)
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(s, dim=-1))], dim=-1).unsqueeze(1)

    return cos, sin
