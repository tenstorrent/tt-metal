# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN denoise-forward helpers for DiffusionGemma canvas attention.

Canvas queries attend to the frozen prompt prefix plus the current canvas as an
all-attend rectangle. The canonical path is maskless non-causal SDPA; explicit
masks are kept only for op A/B tests and non-canonical experiments. Sampling,
self-conditioning, and the multi-layer generation loop live in later W3/W4
helpers; this module owns the real denoise attention wiring.
"""

from __future__ import annotations

import os

import torch
import ttnn
from loguru import logger

from models.experimental.diffusion_gemma.reference.attention_mask import (
    build_canvas_denoise_mask,
    build_canvas_reveal_denoise_mask,
    build_canvas_reveal_denoise_window_mask,
)
from models.experimental.diffusion_gemma.tt.ccl import replicate_mapper as _replicate_mapper
from models.experimental.diffusion_gemma.tt.concat_moe import concat_experts_forward
from models.experimental.diffusion_gemma.tt.diffusion_attention import denoise_attention
from models.experimental.diffusion_gemma.tt.expert_operations import shared_mlp_forward
from models.experimental.diffusion_gemma.tt.self_conditioning import (
    _rms_norm_dram,
    build_self_conditioning,
    build_self_conditioning_embedding_weight,
)
from models.experimental.diffusion_gemma.weight_mapping import GEMMA4_LM_PREFIX, remap_state_dict

NEG = -1.0e9
TILE_SIZE = getattr(ttnn, "TILE_SIZE", 32)


def default_self_conditioning_compute_kernel_config():
    """Select HiFi4/fp32 accumulation for the moderate-vocab full-softmax path.

    The production 262144-vocab path uses ordered online chunks in
    ``tt.self_conditioning`` and does not forward this matmul config.
    """
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def build_device_canvas_denoise_mask(
    mesh_device,
    *,
    prompt_len: int,
    canvas_len: int,
    layer_type: str | None = None,
    sliding_window: int | None = None,
    dtype=ttnn.bfloat16,
):
    """Build a `[1, 1, C, P+C]` denoise mask on device."""
    mask = build_canvas_denoise_mask(
        prompt_len,
        canvas_len,
        layer_type=layer_type,
        sliding_window=sliding_window,
        local_window=False,
        neg_inf=NEG,
        dtype=torch.float32,
    ).view(1, 1, canvas_len, prompt_len + canvas_len)
    return ttnn.from_torch(
        mask,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        mesh_mapper=_replicate_mapper(mesh_device),
    )


def build_device_canvas_reveal_mask(
    mesh_device,
    *,
    prompt_len: int,
    canvas_len: int,
    p_max: int,
    layer_type: str | None = None,
    sliding_window: int | None = None,
    enforce_sliding_window: bool = False,
    hidden_prefix_span: tuple[int, int] | None = None,
    dtype=ttnn.bfloat16,
):
    """Build a constant-shape ``[1, 1, C, p_max + C]`` reveal mask on device (Phase 1).

    Content reveals committed prefix ``[0:prompt_len]`` + all canvas, hides the uncommitted
    tail ``[prompt_len:p_max]`` with ``NEG``. ``enforce_sliding_window=False`` matches the
    maskless all-attend production path; ``True`` additionally applies HF's bidirectional
    window.
    """
    mask = build_canvas_reveal_denoise_mask(
        prompt_len,
        canvas_len,
        p_max,
        layer_type=layer_type,
        sliding_window=sliding_window,
        enforce_sliding_window=enforce_sliding_window,
        hidden_prefix_span=hidden_prefix_span,
        neg_inf=NEG,
        dtype=torch.float32,
    ).view(1, 1, canvas_len, p_max + canvas_len)
    return ttnn.from_torch(
        mask,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        mesh_mapper=_replicate_mapper(mesh_device),
    )


def build_device_canvas_reveal_window_mask(
    mesh_device,
    *,
    prompt_len: int,
    canvas_len: int,
    span: int,
    lo: int,
    sliding_window: int,
    hidden_prefix_span: tuple[int, int] | None = None,
    dtype=ttnn.bfloat16,
):
    """Device ``[1, 1, C, span + C]`` mask for a bounded sliding-layer read (see the reference)."""
    mask = build_canvas_reveal_denoise_window_mask(
        prompt_len,
        canvas_len,
        span,
        lo,
        sliding_window=sliding_window,
        hidden_prefix_span=hidden_prefix_span,
        neg_inf=NEG,
        dtype=torch.float32,
    ).view(1, 1, canvas_len, span + canvas_len)
    return ttnn.from_torch(
        mask,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        mesh_mapper=_replicate_mapper(mesh_device),
    )


def sliding_read_span(sliding_window: int, p_max: int) -> int:
    """Tile-aligned rows a sliding layer must read to cover HF's retained window.

    HF retains ``sliding_window - 1`` tokens, which is not tile-aligned, so read one whole
    tile more and let the mask drop the extra column(s). Never exceed ``p_max``. This is the
    perf half of the sliding-window work (#51080): a sliding layer only needs the
    ``sliding_window - 1`` most recent committed tokens, so it need not read the whole
    ``p_max`` prefix.

    The bounded read is output-identical to a full-span read only while the retention mask is
    enforced, which is why it is gated on :func:`denoise_sliding_window_enabled` at its one
    decision point in ``tt/traced_denoise.py`` rather than on a flag of its own. Without
    retention a bounded read would silently CHANGE visibility instead of implementing it;
    :meth:`DenoiseLogitsAdapter.prepare_reveal_mask_buffers` independently raises on a span
    without ``enforce_window``.

    Output-identical is not memory-free: the block-resident window buffers add per-chip DRAM
    residency (masks are keyed by layer TYPE, so there is ONE sliding mask, not one per layer);
    the win is the transient per-step read, not residency.
    """
    needed = int(sliding_window)  # (sliding_window - 1) rounded up to the next tile == sliding_window when W%32==0
    span = ((needed + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
    return min(span, int(p_max))


def sliding_read_offset(prompt_len: int, span: int, p_max: int) -> int:
    """Tile-aligned start of the bounded window: the last ``span`` committed rows.

    ``prompt_len`` is always a 32-multiple on the commit path, so this stays tile-aligned (the
    slice op requires it). Clamped so the read never runs past the reveal span.
    """
    lo = max(0, int(prompt_len) - int(span))
    lo = min(lo, max(0, int(p_max) - int(span)))
    if lo % ttnn.TILE_SIZE != 0:
        raise ValueError(f"bounded sliding read offset must be tile aligned, got {lo}")
    return lo


def hide_prefill_pads_enabled() -> bool:
    """Whether the reveal mask hides the prefill pad slots (block-0 fidelity). Default ON.

    ``generate._pad_prompt_tokens_for_prefill`` right-pads the prompt to a tile multiple and prefill
    writes K/V for those pad tokens, while the reveal predicate is evaluated with the PADDED length --
    so without this gate the canvas attends up to 31 garbage keys sitting IMMEDIATELY before it,
    corrupting the block-0 template prefix the first block bootstraps from.

    Decision-changing for every prompt whose length is not a 32-multiple; prompts that ARE aligned
    have no pad slots, so the mask is unchanged there. Set DG_DENOISE_HIDE_PREFILL_PADS=0 for the
    old maskless behaviour.

    Composes with the bounded sliding read: the bounded builder takes the same absolute-position
    span, because its key axis already carries absolute positions and needs no column arithmetic.
    In a sliding layer the predicate is self-retiring: the pads sit at the start of the prompt, so
    once the window has scrolled past them it is a no-op and the mask returns to its
    prompt_len-independent steady state. The full-attention layers read the whole p_max prefix and
    keep hiding the pads for the whole run.
    """
    return os.environ.get("DG_DENOISE_HIDE_PREFILL_PADS", "1").lower() in ("1", "true", "yes", "on")


def prefill_pad_span(true_prompt_len: int | None, padded_prompt_len: int | None):
    """Prefix slots holding prefill pad tokens, ``[true_prompt_len, padded_prompt_len)``.

    Returns ``None`` when there is nothing to hide -- the gate is off, the true length is unknown, or
    the prompt was already tile-aligned -- so callers can pass the result straight through.
    """
    if not hide_prefill_pads_enabled() or true_prompt_len is None or padded_prompt_len is None:
        return None
    true_len, padded = int(true_prompt_len), int(padded_prompt_len)
    if true_len <= 0 or true_len > padded:
        raise ValueError(f"true_prompt_len {true_len} must be in (0, padded_prompt_len {padded}]")
    return None if true_len == padded else (true_len, padded)


def denoise_sliding_window_enabled() -> bool:
    """Whether denoise applies HF's sliding-layer key retention (#51080). Default ON.

    HF's sliding layers hold only the last ``sliding_window - 1`` committed tokens, so a maskless
    all-attend denoise attends keys HF does not have, for every committed prefix past the window.
    Enabling this makes TT match HF; below the window the mask never binds and the result is
    bit-identical, so the change is confined to longer prefixes. HF caches exactly
    ``sliding_window - 1`` committed keys on sliding layers and the full prompt on full layers.

    Set ``DG_DENOISE_SLIDING_WINDOW=0`` to restore the old maskless behaviour.
    """
    return os.environ.get("DG_DENOISE_SLIDING_WINDOW", "1").lower() in ("1", "true", "yes", "on")


def _layer_type_for_denoise(tt_model, layer_idx: int) -> str | None:
    layer_types = getattr(getattr(tt_model, "hf_config", None), "layer_types", None)
    if layer_types is not None:
        return layer_types[layer_idx]
    attn_config = getattr(getattr(tt_model.layers[layer_idx], "self_attn", None), "config", None)
    return getattr(attn_config, "layer_type", None)


def _sliding_window_for_denoise(tt_model, layer_idx: int) -> int | None:
    # Validation-only override. The real window masks too few keys to reliably flip an argmax,
    # so an end-to-end output A/B cannot prove the plumbing is live; forcing a small window
    # makes the effect unmistakable.
    override = os.environ.get("DG_DENOISE_SLIDING_WINDOW_OVERRIDE", "").strip()
    if override:
        value = int(override)
        if value <= 0:
            raise ValueError(f"DG_DENOISE_SLIDING_WINDOW_OVERRIDE must be positive, got {value}")
        return value
    attn_config = getattr(getattr(tt_model.layers[layer_idx], "self_attn", None), "config", None)
    window = getattr(attn_config, "sliding_window", None)
    if window is not None:
        return window
    return getattr(getattr(tt_model, "hf_config", None), "sliding_window", None)


def _sliding_layer_needs_denoise_mask(prompt_len: int, canvas_len: int, sliding_window: int) -> bool:
    """Whether a sliding layer needs a mask at all for this committed prefix length.

    HF's sliding cache retains the last ``sliding_window - 1`` committed tokens and the canvas is
    always fully visible, so a mask is needed exactly when some committed position has been
    evicted: ``prompt_len > sliding_window - 1``. ``canvas_len`` does not enter the predicate
    (see reference/attention_mask.py and #51080).
    """
    del canvas_len  # not part of the predicate; kept for call-site compatibility
    return prompt_len > sliding_window - 1


def _build_denoise_attn_mask_for_layer(
    tt_model,
    layer_idx: int,
    *,
    prompt_len: int,
    canvas_len: int,
    use_explicit_sliding_mask: bool = False,
    mask_builder=build_device_canvas_denoise_mask,
):
    if not use_explicit_sliding_mask:
        return None

    layer_type = _layer_type_for_denoise(tt_model, layer_idx)
    if layer_type != "sliding_attention":
        return None

    sliding_window = _sliding_window_for_denoise(tt_model, layer_idx)
    if sliding_window is None or sliding_window <= 0:
        raise ValueError(f"sliding_attention layer {layer_idx} requires a positive sliding_window")
    if not _sliding_layer_needs_denoise_mask(prompt_len, canvas_len, sliding_window):
        return None

    if bool(getattr(tt_model, "_dg_model_owned_hybrid_kv", False)):
        span = min(int(sliding_window), int(prompt_len))
        lo = max(0, int(prompt_len) - span)
        return build_device_canvas_reveal_window_mask(
            tt_model.mesh_device,
            prompt_len=prompt_len,
            canvas_len=canvas_len,
            span=span,
            lo=lo,
            sliding_window=sliding_window,
        )

    return mask_builder(
        tt_model.mesh_device,
        prompt_len=prompt_len,
        canvas_len=canvas_len,
        layer_type="sliding_attention",
        sliding_window=sliding_window,
    )


def _deallocate_optional_tensor(tensor) -> None:
    if tensor is not None and hasattr(tensor, "deallocate"):
        tensor.deallocate(True)


def denoise_attention_forward(
    tt_model,
    *,
    layer_idx: int,
    prompt_hidden=None,
    prompt_kv=None,
    canvas_hidden,
    attn_mask=None,
    q_rope_offset: int | None = None,
):
    """Run one DiffusionGemma denoise attention layer on canvas hidden states.

    Args:
        tt_model: `Gemma4Model` carrying the reused DiffusionGemma decoder weights.
        layer_idx: decoder layer to run.
        prompt_hidden: frozen prompt hidden states `[1, 1, P, H]` on device.
        prompt_kv: optional frozen projected prompt `(K, V)` heads. This is the
            cache-shaped input used by the eventual paged encoder KV read path.
        canvas_hidden: current canvas hidden states `[1, 1, C, H]` on device.
        attn_mask: optional prebuilt `[1, 1, C, P+C]` additive mask on device.
            Leave unset for the canonical all-attend denoise path.

    Returns:
        The attention output for the canvas positions `[1, 1, C, H]`.
    """
    if (prompt_hidden is None) == (prompt_kv is None):
        raise ValueError("pass exactly one of prompt_hidden or prompt_kv")
    prompt_len = prompt_kv[0].shape[-2] if prompt_kv is not None else prompt_hidden.shape[-2]
    canvas_len = canvas_hidden.shape[-2]
    q_rope_offset = prompt_len if q_rope_offset is None else q_rope_offset
    kv_hidden = None if prompt_kv is not None else ttnn.concat([prompt_hidden, canvas_hidden], dim=2)
    out = denoise_attention(
        tt_model.layers[layer_idx].self_attn,
        canvas_hidden,
        rope_mats=tt_model._get_rope_mats(layer_idx, seq_len=q_rope_offset + canvas_len),
        attn_mask=attn_mask,
        kv_hidden_states=kv_hidden,
        prefix_kv=prompt_kv,
        q_rope_offset=q_rope_offset,
    )
    if kv_hidden is not None:
        kv_hidden.deallocate(True)
    return out


def _prompt_source_len(prompt_source):
    return prompt_source[0].shape[-2] if isinstance(prompt_source, (tuple, list)) else prompt_source.shape[-2]


def _prompt_source_is_owned(prompt_source_fn, layer_idx: int) -> bool:
    """Whether the per-layer prompt source must be deallocated by the caller.

    Ownership can differ PER LAYER once bounded sliding spans are active: a windowed layer gets a
    persistent block-resident buffer (never free it) while a full-attention layer may get either a
    borrowed cache view (never free) or an owned clone (must free). Prefer the per-layer query and
    fall back to the whole-source flag, then to the historic owned contract.
    """
    per_layer = getattr(prompt_source_fn, "owns_result_for", None)
    if callable(per_layer):
        return bool(per_layer(layer_idx))
    return bool(getattr(prompt_source_fn, "owns_result", True))


def _deallocate_prompt_source(prompt_source) -> None:
    if isinstance(prompt_source, (tuple, list)):
        for tensor in prompt_source:
            _deallocate_optional_tensor(tensor)
    else:
        _deallocate_optional_tensor(prompt_source)


# ---------------------------------------------------------------------------------------------
# Full-canvas RMSNorm: ONE width-sharded rms_norm over the whole canvas instead of per-32-row
# chunks. gemma4 RMSNorm's width-sharded fast path only takes 32-row slices, and ``norm.forward``
# on the full canvas falls to the slow plain-interleaved path; the chunked route pays extra
# slices, a DRAM concat, extra sharded-norm launches, and I2S/S2I round-trips per norm call.
#
# RMSNorm normalizes each ROW independently over the hidden width, so one full-canvas op is
# per-row equivalent to the 32-row chunks -- the cross-core width reduction is per-row regardless
# of block_h. This path runs one width-sharded rms_norm (reusing ``norm.tt_weight``) and hands the
# output straight back, dropping the slice/concat glue. With fp32 accumulation
# (_norm_compute_kernel_config) the full-canvas and chunked shapes are bit-identical.
# ---------------------------------------------------------------------------------------------

_NORM_FULLCANVAS_CFG_CACHE = {}


def _build_fullcanvas_norm_cfg(mesh, seq_len, hidden_size):
    """Width-sharded rms_norm config for the FULL canvas (``seq_len`` rows, ``block_h=seq_len//32``).

    Largest core grid whose count divides the hidden tile-cols (mirrors gemma4
    RMSNorm._build_sharded_cfg, but block_h>1 for the whole canvas). Returns
    ``(input_memcfg, program_config)`` or ``None`` if no usable grid divides the width.
    """
    if hidden_size % TILE_SIZE != 0 or seq_len % TILE_SIZE != 0:
        return None
    key = (id(mesh), seq_len, hidden_size)
    cached = _NORM_FULLCANVAS_CFG_CACHE.get(key)
    if cached is not None:
        return cached
    tiles = hidden_size // TILE_SIZE
    grid = mesh.compute_with_storage_grid_size()
    best = None  # (num_cores, gx, gy)
    for gy in range(1, grid.y + 1):
        for gx in range(1, grid.x + 1):
            n = gx * gy
            if tiles % n == 0 and (best is None or n > best[0]):
                best = (n, gx, gy)
    if best is None or best[0] == 1:
        _NORM_FULLCANVAS_CFG_CACHE[key] = None
        return None
    num_cores, gx, gy = best
    block_w = tiles // num_cores
    subblock_w = 4
    while subblock_w > 1 and block_w % subblock_w != 0:
        subblock_w -= 1
    input_memcfg = ttnn.create_sharded_memory_config(
        shape=(seq_len, hidden_size // num_cores),
        core_grid=ttnn.CoreGrid(x=gx, y=gy),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[gx, gy],
        subblock_w=subblock_w,
        block_h=seq_len // TILE_SIZE,
        block_w=block_w,
        inplace=False,
    )
    cfg = (input_memcfg, program_config)
    _NORM_FULLCANVAS_CFG_CACHE[key] = cfg
    return cfg


_NORM_ACC_CFG_CACHE = {}


def _norm_compute_kernel_config(device):
    """fp32 destination accumulation for the DG-owned RMSNorms.

    ttnn's rmsnorm default is HiFi4 with **fp32_acc = false**, so each per-core partial sum of
    squares is rounded to bf16 before the cross-core combine. fp32 partials remove the dependence
    on the combine TREE: the full-canvas norm and the 32-row chunks become bit-identical, which is
    what lets the full-canvas shape be a pure perf choice rather than a decision change.

    Cost is negligible here because these configs land on block_w=1 / subblock_w=1, so halving
    DST capacity has nothing to take away; do not generalise that to wide output blocks without
    measuring.
    """
    key = id(device)
    cfg = _NORM_ACC_CFG_CACHE.get(key)
    if cfg is None:
        cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,  # ttnn's own rmsnorm default
            math_approx_mode=True,  # unchanged; only the accumulator moves
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        _NORM_ACC_CFG_CACHE[key] = cfg
    return cfg


def _sharded_rms_norm_rows(norm, hidden_states, rows):
    """One width-sharded rms_norm over ``rows`` at a time, DG-owned, with fp32 accumulation.

    Both the full-canvas (rows = canvas) and the chunked (rows = 32) shapes go through here, so the
    only difference between them is ``block_h`` -- and with fp32 accumulation that difference is
    bit-identical. The 32-row config this builds is the same one gemma4's
    ``RMSNorm._build_sharded_cfg`` picks (same grid search, same block_w, block_h=1), so routing the
    chunks here instead of through ``norm.forward`` changes the accumulator and nothing else.

    Returns ``None`` when no width-sharded config fits, so the caller can fall back.
    """
    cfg = _build_fullcanvas_norm_cfg(hidden_states.device(), rows, hidden_states.shape[-1])
    if cfg is None:
        return None
    input_memcfg, program_config = cfg
    weight = getattr(norm, "tt_weight", None)
    ckcfg = _norm_compute_kernel_config(hidden_states.device())
    seq_len = hidden_states.shape[-2]

    outs = []
    for start in range(0, seq_len, rows):
        piece = hidden_states
        if rows != seq_len:
            piece = ttnn.slice(
                hidden_states,
                [0, 0, start, 0],
                [hidden_states.shape[0], hidden_states.shape[1], start + rows, hidden_states.shape[3]],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        x_sh = ttnn.to_memory_config(piece, input_memcfg)
        out_sh = ttnn.rms_norm(
            x_sh,
            weight=weight,
            epsilon=norm.eps,
            program_config=program_config,
            memory_config=input_memcfg,
            compute_kernel_config=ckcfg,
        )
        outs.append(ttnn.sharded_to_interleaved(out_sh, ttnn.DRAM_MEMORY_CONFIG))
        x_sh.deallocate(True)
        out_sh.deallocate(True)
        if piece is not hidden_states:
            piece.deallocate(True)

    if len(outs) == 1:
        return outs[0]
    out = ttnn.concat(outs, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for part in outs:
        part.deallocate(True)
    return out


def _chunked_norm_forward(norm, hidden_states, *, chunk_size: int = 32):
    # A SCALELESS norm keeps its own path, and this test has to come FIRST: the full-canvas path
    # below is written for the WEIGHTED norms and, ordered first, would capture the MoE router's
    # weightless norm and silently swap its reduction topology (a structurally different
    # summation tree, not just a block_h difference).
    if getattr(norm, "with_scale", True) is False and getattr(norm, "tt_weight", None) is None:
        return _rms_norm_dram(
            hidden_states,
            epsilon=norm.eps,
            chunk_size=chunk_size,
            compute_kernel_config=_norm_compute_kernel_config(hidden_states.device()),
        )
    seq_len = hidden_states.shape[-2]
    if seq_len <= chunk_size:
        return norm.forward(hidden_states)

    # ONE width-sharded norm over the whole canvas, not per-32-row chunks; with the fp32
    # accumulator this path uses, the two shapes are bit-identical.
    out = _sharded_rms_norm_rows(norm, hidden_states, seq_len)
    if out is not None:
        return out

    # No width-sharded config fits this width: fall back to gemma4's per-chunk norm (bf16 accumulate).
    chunks = []
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk = ttnn.slice(
            hidden_states,
            [0, 0, start, 0],
            [hidden_states.shape[0], hidden_states.shape[1], end, hidden_states.shape[3]],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        chunks.append(norm.forward(chunk))
        chunk.deallocate(True)
    out = ttnn.concat(chunks, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for chunk in chunks:
        chunk.deallocate(True)
    return out


def _denoise_router_forward(router, hidden_states):
    normed = _chunked_norm_forward(router.norm, hidden_states)
    scaled = ttnn.mul(normed, router.scale)
    normed.deallocate(True)
    scaled = ttnn.mul(scaled, router.scalar_root_size)

    expert_scores = ttnn.linear(scaled, router.proj_weight)
    scaled.deallocate(True)

    router_probs = ttnn.softmax(expert_scores, dim=-1)
    expert_scores.deallocate(True)

    top_k_values, top_k_indices = ttnn.topk(router_probs, k=router.top_k, dim=-1)
    top_k_sum = ttnn.sum(top_k_values, dim=-1, keepdim=True)
    top_k_values = ttnn.div(top_k_values, top_k_sum)
    top_k_sum.deallocate(True)

    dense_routing = ttnn.scatter(
        ttnn.zeros_like(router_probs),
        dim=-1,
        index=top_k_indices,
        src=top_k_values,
    )
    router_probs.deallocate(True)
    top_k_values.deallocate(True)
    top_k_indices.deallocate(True)

    if router.per_expert_scale is not None:
        dense_routing = ttnn.mul(dense_routing, router.per_expert_scale)

    return dense_routing


def _denoise_moe_forward(moe, router_input, expert_input):
    """Router + concat-experts MoE. The single denoise (and batched-commit) MoE body.

    Prefill is a different MoE: it keeps the ragged top-8 path in tt/sparse_moe.py.
    """
    dense_routing = _denoise_router_forward(moe.router, router_input)
    out = concat_experts_forward(moe.experts, expert_input, dense_routing)
    dense_routing.deallocate(True)
    return out


def _denoise_shared_mlp_forward(mlp, hidden_states):
    return shared_mlp_forward(mlp, hidden_states)


# ---------------------------------------------------------------------------------------------
# DG_SKIP — in-graph component zeroing, for pricing a component's TRACED cost (measurement only).
#
# A serial per-op profile does not price a traced step: under trace the host dispatch is overlapped,
# so an op's standalone time can be almost entirely hidden. The only way to learn what a component
# actually contributes to the traced step is to remove it and re-measure. ``DG_SKIP`` replaces a
# component with a shape-preserving ``ttnn.mul(x, 0.0)`` at its seam, so the op graph keeps the same
# shapes and the rest of the step is untouched.
#
# ``DG_SKIP="attn,shared,moe"`` — comma-separated, validated (an unknown token raises). Default
# empty (nothing skipped). See ``_SKIP_TOKENS`` for the authoritative list:
#   attn / shared / moe   the denoise attention, shared MLP, and router+expert path
#
# Only the DENOISE layer body is ablatable; there is deliberately no token for self-conditioning
# or for the commit body. If the commit body ever needs pricing, add the token AND the seam in
# ``commit_batched.py`` in the same change.
#
# The output of a skipped run is garbage BY CONSTRUCTION — never feed a DG_SKIP run into a
# committed_sha256 comparison or a quality gate. Note also that zeroing the MoE feeds an all-zero
# hidden into later routers, so a downstream component's cost under DG_SKIP is only meaningful if
# that cost is value-independent.
# ---------------------------------------------------------------------------------------------


# Every token DG_SKIP honours. Validated on read: a silently ignored typo (``DG_SKIP=moe1``)
# would measure the UNABLATED step and report it as an ablation.
_SKIP_TOKENS = frozenset(
    {
        "attn",  # denoise attention: QKV, RoPE, SDPA, o_proj, all-reduce
        "shared",  # denoise shared MLP
        "moe",  # denoise router + expert path
    }
)


def _skip_components() -> frozenset:
    raw = os.environ.get("DG_SKIP", "")
    tokens = frozenset(part.strip().lower() for part in raw.split(",") if part.strip())
    unknown = tokens - _SKIP_TOKENS
    if unknown:
        raise ValueError(
            f"DG_SKIP has unknown component(s) {sorted(unknown)}; valid tokens are "
            f"{sorted(_SKIP_TOKENS)}. Refusing to run: an ignored token would silently measure the "
            "unablated step and report it as an ablation."
        )
    return tokens


def _zeros_like_via_mul(tensor):
    """Shape/layout/memory-config-preserving zero, without a ``ttnn.full`` (trace-safe)."""
    return ttnn.mul(tensor, 0.0)


def _denoise_layer_forward(
    tt_model, layer_idx, hidden_states, prompt_source, attn_mask, q_rope_offset, canvas_rope_provider=None
):
    layer = tt_model.layers[layer_idx]
    skip = _skip_components()
    residual = hidden_states
    normed = _chunked_norm_forward(layer.input_layernorm, hidden_states)
    prefix_kv = prompt_source if isinstance(prompt_source, (tuple, list)) else None
    kv_hidden = None if prefix_kv is not None or "attn" in skip else ttnn.concat([prompt_source, normed], dim=2)
    # Cross-block-trace-reusable RoPE: when a canvas_rope_provider is supplied it returns a
    # CONSTANT-SHAPE [1,1,C,head_dim] buffer already holding cos/sin for the absolute canvas
    # positions [start_pos:start_pos+C] (updated per block OUTSIDE the trace); applying it at
    # start_offset=0 is bit-identical to slicing the growing [:, :, :start_pos+C, :] cache at
    # start_offset=start_pos (RoPE cos/sin depend only on absolute position). This keeps the
    # captured trace's RoPE tensor addresses/shapes fixed across blocks. Only valid on the
    # prefix_kv path (Q and canvas-K are both C rows); the kv_hidden recompute path (K spans the
    # full prompt+canvas) must keep the growing slice. See #47465.
    if canvas_rope_provider is not None:
        if prefix_kv is None:
            raise ValueError("canvas_rope_provider requires the prefix_kv denoise path")
        rope_mats = canvas_rope_provider(layer_idx)
        rope_offset = 0
    else:
        rope_mats = tt_model._get_rope_mats(layer_idx, seq_len=q_rope_offset + hidden_states.shape[-2])
        rope_offset = q_rope_offset
    if "attn" in skip:
        attn_output = _zeros_like_via_mul(normed)  # DG_SKIP: price attention's traced contribution
    else:
        attn_output = denoise_attention(
            layer.self_attn,
            normed,
            rope_mats=rope_mats,
            attn_mask=attn_mask,
            kv_hidden_states=kv_hidden,
            prefix_kv=prefix_kv,
            q_rope_offset=rope_offset,
        )
    if kv_hidden is not None:
        kv_hidden.deallocate(True)

    attn_output = _chunked_norm_forward(layer.post_attention_layernorm, attn_output)
    hidden_states = ttnn.add(residual, attn_output)
    residual.deallocate(True)
    attn_output.deallocate(True)

    residual = hidden_states
    normed = _chunked_norm_forward(layer.pre_feedforward_layernorm, hidden_states)
    if "shared" in skip:
        mlp_output = _zeros_like_via_mul(normed)  # DG_SKIP: price the shared MLP
    else:
        mlp_output = _denoise_shared_mlp_forward(layer.shared_mlp, normed)
    normed.deallocate(True)

    if layer.enable_moe_block:
        mlp_normed = _chunked_norm_forward(layer.post_feedforward_layernorm_1, mlp_output)
        mlp_output.deallocate(True)
        expert_input = _chunked_norm_forward(layer.pre_feedforward_layernorm_2, residual)
        if "moe" in skip:
            expert_output = _zeros_like_via_mul(expert_input)  # DG_SKIP: price router + experts
        else:
            expert_output = _denoise_moe_forward(layer.moe, residual, expert_input)
        expert_input.deallocate(True)
        expert_normed = _chunked_norm_forward(layer.post_feedforward_layernorm_2, expert_output)
        expert_output.deallocate(True)
        hidden_states = ttnn.add(mlp_normed, expert_normed)
        mlp_normed.deallocate(True)
        expert_normed.deallocate(True)
    else:
        hidden_states = mlp_output

    hidden_states = _chunked_norm_forward(layer.post_feedforward_layernorm, hidden_states)
    combined = ttnn.add(residual, hidden_states)
    residual.deallocate(True)
    hidden_states.deallocate(True)
    if layer.layer_scalar != 1.0:
        scaled = ttnn.mul(combined, layer.layer_scalar)
        combined.deallocate(True)
        combined = scaled
    return combined


def denoise_hidden_forward(
    tt_model,
    *,
    prompt_hidden_by_layer,
    canvas_hidden,
    q_rope_offset: int | None = None,
    prompt_len: int | None = None,
    use_explicit_sliding_mask: bool = False,
    mask_builder=build_device_canvas_denoise_mask,
    canvas_rope_provider=None,
    reveal_mask_provider=None,
):
    """Run the DiffusionGemma denoise backbone to final hidden states.

    ``prompt_hidden_by_layer`` provides the frozen encoder-side attention source
    for each decoder layer. Entries can be either `[1, 1, P, H]` hidden tensors
    (legacy shim), projected `(K, V)` prompt heads, or a callable that lazily
    returns a per-layer prompt source. The production path is maskless all-attend;
    set ``use_explicit_sliding_mask`` only for HF-geometry A/B tests.
    """
    prompt_source_fn = prompt_hidden_by_layer if callable(prompt_hidden_by_layer) else None
    if prompt_source_fn is None and len(prompt_hidden_by_layer) != len(tt_model.layers):
        raise ValueError(
            f"prompt_hidden_by_layer has {len(prompt_hidden_by_layer)} entries but model has {len(tt_model.layers)} layers"
        )
    if prompt_len is None:
        if prompt_source_fn is not None:
            raise ValueError("prompt_len is required when prompt_hidden_by_layer is callable")
        prompt_len = _prompt_source_len(prompt_hidden_by_layer[0])

    hidden_states = canvas_hidden
    q_rope_offset = prompt_len if q_rope_offset is None else q_rope_offset
    for layer_idx in range(len(tt_model.layers)):
        if reveal_mask_provider is not None:
            # Paged-prefix Phase 1: persistent per-block reveal mask (hides the uncommitted
            # tail of the fixed p_max prefix read). Owned by the adapter — NOT deallocated here.
            attn_mask = reveal_mask_provider(layer_idx)
        else:
            attn_mask = _build_denoise_attn_mask_for_layer(
                tt_model,
                layer_idx,
                prompt_len=prompt_len,
                canvas_len=hidden_states.shape[-2],
                use_explicit_sliding_mask=use_explicit_sliding_mask,
                mask_builder=mask_builder,
            )
        prompt_source = (
            prompt_source_fn(layer_idx) if prompt_source_fn is not None else prompt_hidden_by_layer[layer_idx]
        )
        try:
            hidden_states = _denoise_layer_forward(
                tt_model,
                layer_idx,
                hidden_states,
                prompt_source,
                attn_mask,
                q_rope_offset,
                canvas_rope_provider=canvas_rope_provider,
            )
        finally:
            # A prompt source may hand back the model-owned KV cache itself rather than a copy
            # (MutablePrefixKVReader.owns_result False); freeing that would destroy the cache.
            # Same discipline as the reveal mask below, which is provider-owned and never freed
            # here. Default True so any other callable keeps the historic owned contract.
            if prompt_source_fn is not None and _prompt_source_is_owned(prompt_source_fn, layer_idx):
                _deallocate_prompt_source(prompt_source)
            if reveal_mask_provider is None:
                _deallocate_optional_tensor(attn_mask)
    final_hidden = _chunked_norm_forward(tt_model.norm, hidden_states)
    hidden_states.deallocate(True)
    return final_hidden


def denoise_logits_forward(
    tt_model,
    *,
    prompt_hidden_by_layer,
    canvas_hidden,
    q_rope_offset: int | None = None,
    prompt_len: int | None = None,
    use_explicit_sliding_mask: bool = False,
    canvas_rope_provider=None,
    reveal_mask_provider=None,
):
    """Run a short-prompt DiffusionGemma denoise logits forward.

    The returned logits cover all canvas positions, which the diffusion sampler
    consumes each denoise step.
    """
    hidden_states = denoise_hidden_forward(
        tt_model,
        prompt_hidden_by_layer=prompt_hidden_by_layer,
        canvas_hidden=canvas_hidden,
        q_rope_offset=q_rope_offset,
        prompt_len=prompt_len,
        use_explicit_sliding_mask=use_explicit_sliding_mask,
        canvas_rope_provider=canvas_rope_provider,
        reveal_mask_provider=reveal_mask_provider,
    )
    return _apply_denoise_lm_head(hidden_states, tt_model)


def _apply_denoise_lm_head(hidden_states, tt_model):
    """Project denoise hidden states without the shared plain all-gather.

    ``create_tt_model_dg`` currently returns the shared Gemma4Model in the
    default precision mode. Its ``_apply_lm_head`` ends in ``ttnn.all_gather``;
    captured denoise traces then replay that broadcast with op-owned
    semaphores, and a later replay can wedge all devices in
    ``broadcast_rm_writer::noc_semaphore_wait``. Keep the shared projection and
    softcap exactly, but route the TP gather through DG's pre-created
    semaphores.
    """
    from models.demos.gemma4.tt.model import _get_lm_head_program_config

    if tt_model.lm_head_weight is not None:
        lm_head_pc = _get_lm_head_program_config(
            tt_model.mesh_device,
            m=hidden_states.shape[2],
            k=tt_model.hidden_size,
            n=tt_model.lm_head_weight.shape[-1],
        )
        logits = ttnn.linear(hidden_states, tt_model.lm_head_weight, program_config=lm_head_pc)
        hidden_states.deallocate(True)
    else:
        logits = hidden_states

    if tt_model.final_logit_softcapping and tt_model.final_logit_softcapping > 0:
        cap = tt_model.final_logit_softcapping
        logits = ttnn.mul(logits, 1.0 / cap)
        logits = ttnn.tanh(logits)
        logits = ttnn.mul(logits, cap)

    if tt_model.mesh_config is not None and tt_model.mesh_config.tp > 1 and tt_model.lm_head_weight is not None:
        from models.experimental.diffusion_gemma.tt.ccl import ccl_allgather

        logits = ccl_allgather(logits, tt_model.mesh_config, tt_model.ccl_manager)
    return logits


def read_prompt_kv_cache_slice(kv_cache, *, prompt_len: int, seq_len_start: int = 0, borrow_full_span: bool = False):
    """Read a frozen prompt K/V prefix from a contiguous Gemma4 KV cache.

    This is the non-paged cache adapter for W2: it reads the encoder-written K/V
    heads from `[B, heads, max_seq, head_dim]` cache tensors and returns the
    cache-shaped `(K, V)` prefix accepted by denoise attention. The underlying
    TTNN slice op requires full tiles along sequence, so bounds must be 32-aligned.

    ``borrow_full_span`` returns the model-owned cache tensors THEMSELVES for a read that
    spans the entire cache seq dim, instead of cloning them. The caller must then not
    deallocate the result — see :attr:`MutablePrefixKVReader.owns_result`, which is what
    ``denoise_hidden_forward`` consults. Default off so every existing caller keeps getting
    an owned copy.
    """
    seq_len_end = seq_len_start + prompt_len
    if seq_len_start % ttnn.TILE_SIZE != 0 or seq_len_end % ttnn.TILE_SIZE != 0:
        raise ValueError("KV cache slice bounds must be multiples of 32")
    k_cache, v_cache = kv_cache
    # A slice spanning the ENTIRE cache seq dim (the reveal-mask fixed p_max == cache_len read)
    # ALIASES the cache buffer, so it must never be sliced-then-deallocated: freeing the alias
    # frees the model-owned cache ("Input Tensor is not allocated" on the next block). The free
    # is the caller's, at ``denoise_hidden_forward``'s per-layer ``finally`` — denoise_attention
    # itself never deallocates the prefix, it only frees its own ``to_memory_config`` temporaries.
    # So there are two safe options for the full-span read: hand back a clone the caller may
    # free, or hand back the cache itself and have the caller skip the free. The clone costs a
    # whole-cache copy per layer per step of data that is invariant across a block's 48 steps,
    # which is why borrowing exists. Partial slices already produce an independent copy.
    if seq_len_start == 0 and seq_len_end == int(k_cache.shape[2]):
        if borrow_full_span:
            return (k_cache, v_cache)
        return (ttnn.clone(k_cache), ttnn.clone(v_cache))
    starts = [0, 0, seq_len_start, 0]
    k_ends = [k_cache.shape[0], k_cache.shape[1], seq_len_end, k_cache.shape[3]]
    v_ends = [v_cache.shape[0], v_cache.shape[1], seq_len_end, v_cache.shape[3]]
    return (
        ttnn.slice(k_cache, starts, k_ends, memory_config=ttnn.DRAM_MEMORY_CONFIG),
        ttnn.slice(v_cache, starts, v_ends, memory_config=ttnn.DRAM_MEMORY_CONFIG),
    )


def hybrid_cache_sequence_segments(
    *, seq_len_start: int, prompt_len: int, capacity: int
) -> tuple[tuple[int, int], ...]:
    """Physical circular-cache segments for one absolute logical span.

    Returned bounds are half-open and ordered chronologically.  The model-owned
    hybrid path only calls this for a sliding layer, whose paged kernels map
    absolute positions through ``position % sliding_window``.
    """

    seq_len_start = int(seq_len_start)
    prompt_len = int(prompt_len)
    capacity = int(capacity)
    if seq_len_start < 0 or prompt_len <= 0 or capacity <= 0:
        raise ValueError(
            f"invalid hybrid cache span: start={seq_len_start} prompt_len={prompt_len} capacity={capacity}"
        )
    if prompt_len > capacity:
        raise ValueError(f"hybrid sliding read {prompt_len} exceeds physical capacity {capacity}")
    start = seq_len_start % capacity
    end = start + prompt_len
    if end <= capacity:
        return ((start, end),)
    return ((start, capacity), (0, end - capacity))


def _paged_cache_to_sequence(cache):
    """``[blocks, heads, block, dim] -> [1, heads, blocks*block, dim]``."""

    heads = int(cache.shape[1])
    span = int(cache.shape[0]) * int(cache.shape[2])
    head_dim = int(cache.shape[3])
    head_major = ttnn.permute(cache, (1, 0, 2, 3))
    return ttnn.reshape(head_major, (1, heads, span, head_dim))


def _read_hybrid_sliding_cache(cache, *, prompt_len: int, seq_len_start: int, capacity: int):
    sequence = _paged_cache_to_sequence(cache)
    segments = hybrid_cache_sequence_segments(
        seq_len_start=seq_len_start,
        prompt_len=prompt_len,
        capacity=capacity,
    )
    if len(segments) == 1 and segments[0] == (0, capacity):
        return sequence

    slices = []
    for start, end in segments:
        slices.append(
            ttnn.slice(
                sequence,
                [0, 0, start, 0],
                [sequence.shape[0], sequence.shape[1], end, sequence.shape[3]],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        )
    sequence.deallocate(True)
    if len(slices) == 1:
        return slices[0]
    out = ttnn.concat(slices, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for tensor in slices:
        tensor.deallocate(True)
    return out


def read_prompt_kv_model_owned_hybrid(
    tt_model,
    *,
    prompt_len: int,
    seq_len_start: int = 0,
    layer_idx: int,
    borrow_full_span: bool = False,
):
    """Read one layer from DG's identity-mapped model-owned hybrid cache."""

    sliding_layers = getattr(tt_model, "_dg_hybrid_sliding_layers", frozenset())
    if layer_idx not in sliding_layers:
        views = getattr(tt_model, "_dg_hybrid_full_cache_views", {})
        if layer_idx not in views:
            raise RuntimeError(f"hybrid full-attention cache view missing for layer {layer_idx}")
        return read_prompt_kv_cache_slice(
            views[layer_idx],
            prompt_len=prompt_len,
            seq_len_start=seq_len_start,
            borrow_full_span=borrow_full_span,
        )

    capacity = int(getattr(tt_model, "_dg_hybrid_sliding_window", 0))
    if prompt_len > capacity:
        seq_len_start += prompt_len - capacity
        prompt_len = capacity
    k_cache, v_cache = tt_model.tt_kv_cache[layer_idx]
    return (
        _read_hybrid_sliding_cache(
            k_cache,
            prompt_len=prompt_len,
            seq_len_start=seq_len_start,
            capacity=capacity,
        ),
        _read_hybrid_sliding_cache(
            v_cache,
            prompt_len=prompt_len,
            seq_len_start=seq_len_start,
            capacity=capacity,
        ),
    )


def read_prompt_kv_cache_by_layer(
    tt_model,
    *,
    prompt_len: int,
    seq_len_start: int = 0,
    layer_idx: int | None = None,
    read_fn=read_prompt_kv_cache_slice,
    borrow_full_span: bool = False,
):
    """Read frozen prompt K/V prefixes from every layer's Gemma4 KV cache.

    This is the production-shaped prompt source for the denoise adapter: one
    `(K, V)` tuple per decoder layer, rather than the early single-layer test
    shim. The returned tensors are owned by the caller unless ``borrow_full_span``
    is set (see :func:`read_prompt_kv_cache_slice`).
    """
    if len(tt_model.tt_kv_cache) != len(tt_model.layers):
        raise ValueError(
            f"tt_kv_cache has {len(tt_model.tt_kv_cache)} layers but model has {len(tt_model.layers)} layers"
        )
    hybrid = bool(getattr(tt_model, "_dg_model_owned_hybrid_kv", False))
    if hybrid and read_fn is read_prompt_kv_cache_slice:
        if layer_idx is None:
            return [
                read_prompt_kv_model_owned_hybrid(
                    tt_model,
                    prompt_len=prompt_len,
                    seq_len_start=seq_len_start,
                    layer_idx=i,
                    borrow_full_span=borrow_full_span,
                )
                for i in range(len(tt_model.layers))
            ]
        return read_prompt_kv_model_owned_hybrid(
            tt_model,
            prompt_len=prompt_len,
            seq_len_start=seq_len_start,
            layer_idx=layer_idx,
            borrow_full_span=borrow_full_span,
        )

    # Only forward the kwarg when borrowing is actually requested: ``read_fn`` is a documented
    # injection point and existing test doubles / reference/replay_hf_tt.py monkeypatches accept only
    # (kv_cache, *, prompt_len, seq_len_start).
    extra = {"borrow_full_span": True} if borrow_full_span else {}
    if layer_idx is not None:
        return read_fn(tt_model.tt_kv_cache[layer_idx], prompt_len=prompt_len, seq_len_start=seq_len_start, **extra)
    return [
        read_fn(kv_cache, prompt_len=prompt_len, seq_len_start=seq_len_start, **extra)
        for kv_cache in tt_model.tt_kv_cache
    ]


class MutablePrefixKVReader:
    """Lazy per-layer contiguous-cache reader with a commit-advanced prefix span."""

    def __init__(
        self,
        tt_model,
        *,
        prompt_len: int,
        seq_len_start: int = 0,
        read_fn=read_prompt_kv_cache_by_layer,
        borrow_full_span: bool = False,
    ):
        self.tt_model = tt_model
        self.prompt_len = int(prompt_len)
        self.seq_len_start = int(seq_len_start)
        self.read_fn = read_fn
        # When the read covers the whole cache seq dim, hand back the model-owned cache tensors
        # instead of cloning them (~2 whole-cache copies per layer per step of block-invariant
        # data). Default off; the traced up-front path opts in once the fixed span is bound.
        self.borrow_full_span = bool(borrow_full_span)
        # Paged-prefix Phase 1 (reveal-mask): a CONSTANT read span decouples the traced
        # slice shape from the growing committed ``prompt_len``. When set, ``__call__`` always
        # reads ``read_span`` rows (so the trace never invalidates on prefix growth), and the
        # committed ``prompt_len`` only drives the reveal mask content + canvas RoPE anchor.
        self.read_span = None
        # Bounded per-layer spans (#51080 item 3 perf half). ``window_layers`` maps layer_idx ->
        # span for layers that read only a bounded window instead of the full ``read_span``.
        # Because the window OFFSET slides with the committed prompt_len and a slice offset is
        # baked into a captured trace, those reads cannot happen inside the trace: each windowed
        # layer gets a persistent block-resident buffer, allocated BEFORE capture and refreshed
        # (contents only, address stable) once per block from outside any trace.
        self.window_layers: dict[int, int] = {}
        self._window_bufs: dict[int, tuple] = {}

    def set_read_span(self, p_max: int) -> None:
        p_max = int(p_max)
        if p_max % ttnn.TILE_SIZE != 0:
            raise ValueError(f"reveal-mask read span must be tile aligned, got {p_max}")
        if p_max < self.prompt_len:
            raise ValueError(f"read span {p_max} < committed prompt_len {self.prompt_len}")
        self.read_span = p_max

    @property
    def owns_result(self) -> bool:
        """Whether the caller must deallocate what :meth:`__call__` returns, for a NON-windowed layer.

        ``False`` only when the read spans the whole cache seq dim AND borrowing was requested —
        in that case ``__call__`` returns the model-owned cache tensors themselves and freeing
        them would destroy the cache. Windowed layers are always non-owned (they return persistent
        buffers); use :meth:`owns_result_for` when per-layer resolution matters.
        """
        if not self.borrow_full_span:
            return True
        n = self.read_span if self.read_span is not None else self.prompt_len
        if self.seq_len_start != 0:
            return True
        caches = getattr(self.tt_model, "tt_kv_cache", None)
        if not caches:
            return True
        spans = {int(k_cache.shape[2]) for k_cache, _v_cache in caches}
        return not (len(spans) == 1 and n == next(iter(spans)))

    def owns_result_for(self, layer_idx: int) -> bool:
        """Per-layer ownership. Windowed layers hand back persistent buffers we must NOT free."""
        if layer_idx in self._window_bufs:
            return False
        if bool(getattr(self.tt_model, "_dg_model_owned_hybrid_kv", False)):
            if not self.borrow_full_span or self.seq_len_start != 0:
                return True
            if layer_idx in getattr(self.tt_model, "_dg_hybrid_sliding_layers", frozenset()):
                return True
            n = self.read_span if self.read_span is not None else self.prompt_len
            view = getattr(self.tt_model, "_dg_hybrid_full_cache_views", {}).get(layer_idx)
            return view is None or int(view[0].shape[2]) != int(n)
        return self.owns_result

    # --- bounded per-layer window buffers -------------------------------------------------
    def prepare_window_buffers(self, window_layers: dict) -> None:
        """Allocate the block-resident K/V buffers OUTSIDE any trace (pre-capture).

        One persistent pair per listed layer. Their ADDRESSES are what the captured trace bakes;
        only their contents change per block (see :meth:`refresh_windows`).

        Shape ``[1, nkv, span, hd]`` — just the bounded prefix window; the canvas is still
        concatenated per step by ``denoise_attention``.
        """
        self.release_window_buffers()
        self.window_layers = {int(k): int(v) for k, v in dict(window_layers).items()}
        if not self.window_layers:
            return
        # FAIL LOUD on prefill-from-non-zero. The bounded read derives its absolute offset as
        # ``sliding_read_offset(prompt_len, span, p_max)`` and passes that straight through as
        # ``seq_len_start`` (below, and again in refresh_windows) -- it does NOT add
        # ``self.seq_len_start`` the way the unbounded read does, so at a non-zero base the sliding
        # layers would read the wrong rows with no error anywhere. Fix is to fold seq_len_start into
        # the offset in BOTH places and delete this guard, with a test at a non-zero base.
        if self.seq_len_start != 0:
            raise NotImplementedError(
                f"bounded sliding read does not support a non-zero prefix base yet "
                f"(seq_len_start={self.seq_len_start}); its offset is computed from prompt_len alone, "
                f"so it would read the wrong absolute rows. Fold seq_len_start into "
                f"sliding_read_offset before enabling prefill-from-non-zero."
            )
        p_max = self.read_span if self.read_span is not None else self.prompt_len
        mesh_device = self.tt_model.mesh_device
        for layer_idx, span in sorted(self.window_layers.items()):
            lo = sliding_read_offset(self.prompt_len, span, p_max)
            # The reader returns owned tensors for both a contiguous partial slice and a
            # paged circular-window gather, so they can serve directly as persistent buffers.
            self._window_bufs[layer_idx] = self.read_fn(
                self.tt_model,
                prompt_len=span,
                seq_len_start=lo,
                layer_idx=layer_idx,
            )

    def refresh_windows(self, prompt_len: int) -> None:
        """Refresh bounded-window buffer CONTENTS for a block (OUTSIDE any trace).

        Addresses are untouched, so every captured trace stays valid; only the slice offset moves.
        """
        if not self._window_bufs:
            return
        p_max = self.read_span if self.read_span is not None else int(prompt_len)
        mesh_device = self.tt_model.mesh_device
        for layer_idx, span in sorted(self.window_layers.items()):
            lo = sliding_read_offset(int(prompt_len), span, p_max)
            k_buf, v_buf = self._window_bufs[layer_idx]
            k_src, v_src = self.read_fn(
                self.tt_model,
                prompt_len=span,
                seq_len_start=lo,
                layer_idx=layer_idx,
            )
            try:
                ttnn.copy(k_src, k_buf)
                ttnn.copy(v_src, v_buf)
            finally:
                k_src.deallocate(True)
                v_src.deallocate(True)

    def release_window_buffers(self) -> None:
        for layer_idx, pair in getattr(self, "_window_bufs", {}).items():
            for tensor in pair:
                try:
                    _deallocate_optional_tensor(tensor)
                except BaseException as cleanup_error:
                    logger.error(f"failed to release window buffer for layer {layer_idx}: {cleanup_error}")
        self._window_bufs = {}
        self.window_layers = {}

    def __call__(self, layer_idx: int):
        buffered = self._window_bufs.get(layer_idx)
        if buffered is not None:
            return buffered
        n = self.read_span if self.read_span is not None else self.prompt_len
        extra = {"borrow_full_span": True} if self.borrow_full_span else {}
        return self.read_fn(
            self.tt_model,
            prompt_len=n,
            seq_len_start=self.seq_len_start,
            layer_idx=layer_idx,
            **extra,
        )

    def set_prompt_len(self, prompt_len: int) -> None:
        prompt_len = int(prompt_len)
        if prompt_len < self.prompt_len:
            raise ValueError(f"frozen prefix cannot shrink: {self.prompt_len} -> {prompt_len}")
        if prompt_len % ttnn.TILE_SIZE != 0:
            raise ValueError(f"frozen prefix length must be tile aligned, got {prompt_len}")
        if self.read_span is not None and prompt_len > self.read_span:
            raise ValueError(f"committed prompt_len {prompt_len} exceeds reveal read span {self.read_span}")
        self.prompt_len = prompt_len

    def reset_prompt_len(self, prompt_len: int) -> None:
        """Reset the committed prefix at a request boundary on a fixed-span reader.

        Unlike :meth:`set_prompt_len`, this permits shrinking because a new request has
        overwritten the model-owned cache head. It is intentionally unavailable without a
        reveal-mask ``read_span``: shrinking a shape-baked prefix would replay stale prompt KV.
        """
        prompt_len = int(prompt_len)
        if self.read_span is None:
            raise RuntimeError("request-boundary prefix reset requires a fixed reveal-mask read span")
        if prompt_len < 0:
            raise ValueError(f"frozen prefix length must be non-negative, got {prompt_len}")
        if prompt_len % ttnn.TILE_SIZE != 0:
            raise ValueError(f"frozen prefix length must be tile aligned, got {prompt_len}")
        if prompt_len > self.read_span:
            raise ValueError(f"committed prompt_len {prompt_len} exceeds reveal read span {self.read_span}")
        self.prompt_len = prompt_len


def embed_canvas_tokens(tt_model, canvas_tokens):
    """Embed device canvas token ids into `[1, 1, C, H]` TILE hidden states."""
    if canvas_tokens.shape[0] != 1:
        raise ValueError("embed_canvas_tokens currently supports batch=1")
    if len(canvas_tokens.shape) == 4 and canvas_tokens.shape[-1] == 1:
        canvas_len = canvas_tokens.shape[-2]
        token_ids = ttnn.reshape(canvas_tokens, (canvas_tokens.shape[0], canvas_len))
        token_ids = ttnn.to_layout(token_ids, ttnn.ROW_MAJOR_LAYOUT)
    else:
        canvas_len = canvas_tokens.shape[-1]
        token_ids = canvas_tokens
    canvas_hidden = tt_model.embed_tokens(token_ids)
    if token_ids is not canvas_tokens:
        token_ids.deallocate(True)
    if len(canvas_hidden.shape) == 3:
        canvas_hidden = ttnn.reshape(canvas_hidden, (1, 1, canvas_len, tt_model.hidden_size))
    elif canvas_hidden.shape[-2] != canvas_len:
        canvas_hidden = ttnn.reshape(canvas_hidden, (1, 1, canvas_len, tt_model.hidden_size))
    return ttnn.to_layout(canvas_hidden, ttnn.TILE_LAYOUT)


def denoise_logits_from_tokens(
    tt_model,
    *,
    prompt_hidden_by_layer,
    canvas_tokens,
    self_conditioning=None,
    prev_logits=None,
    q_rope_offset: int | None = None,
    prompt_len: int | None = None,
    use_explicit_sliding_mask: bool = False,
    self_conditioning_embedding_weight=None,
    self_conditioning_compute_kernel_config=None,
    self_conditioning_temperature: float = 1.0,
):
    """Embed canvas token ids, optionally self-condition, then run denoise logits."""
    canvas_hidden = embed_canvas_tokens(tt_model, canvas_tokens)
    if self_conditioning is not None:
        conditioned = self_conditioning.condition(
            canvas_hidden,
            prev_logits,
            self_conditioning_embedding_weight,
            compute_kernel_config=self_conditioning_compute_kernel_config,
            temperature=self_conditioning_temperature,
        )
        canvas_hidden.deallocate(True)
        canvas_hidden = conditioned
    return denoise_logits_forward(
        tt_model,
        prompt_hidden_by_layer=prompt_hidden_by_layer,
        canvas_hidden=canvas_hidden,
        q_rope_offset=q_rope_offset,
        prompt_len=prompt_len,
        use_explicit_sliding_mask=use_explicit_sliding_mask,
    )


class DenoiseLogitsAdapter:
    """Stateful W2 logits callback for the W3 denoise controller.

    The controller calls ``logits_fn(canvas_tokens, step)``. This adapter turns
    that narrow callback into the real W2 path: token embedding, optional
    self-conditioning from the previous step's logits, and denoise logits forward.

    **Trace-safe self-conditioning** (``prepare_trace_safe_self_conditioning``).
    The default eager path threads the self-cond signal across steps as the
    previous step's *full* ``[1,1,C,vocab]`` logits, freshly allocated every step
    (``self.prev_logits``). A fixed Metal trace bakes in buffer addresses and one
    fixed unrolled graph, and this chained fresh-alloc cross-step state does not
    replay bit-exactly. The trace-safe variant carries the cross-step state as the
    small ``[1,1,C,hidden]`` soft-embedding **signal** in a persistent,
    preallocated buffer updated
    **in-place** each step (``ttnn.copy``), so its device address is fixed. Step 0
    still uses the ``condition(None)`` branch (``post_norm(embed)``) and only
    *writes* the buffer; steps 1+ *read* the buffer written by the immediately
    preceding step. This is bit-exact to the eager path (same ``soft_embedding``
    math, just computed at the producer step's end and copied) and carries no
    stale cross-block state (step 0 never reads the buffer). See #47465.
    """

    def __init__(
        self,
        tt_model,
        *,
        prompt_hidden_by_layer,
        self_conditioning=None,
        self_conditioning_embedding_weight=None,
        self_conditioning_compute_kernel_config=None,
        q_rope_offset: int | None = None,
        prompt_len: int | None = None,
        true_prompt_len: int | None = None,
        max_denoise_steps: int | None = None,
        temperature_start: float = 0.8,
        temperature_end: float = 0.4,
        logits_from_tokens=denoise_logits_from_tokens,
    ):
        self.tt_model = tt_model
        self.prompt_hidden_by_layer = prompt_hidden_by_layer
        self.prompt_len = prompt_len
        disable_self_conditioning = os.environ.get("DG_DISABLE_SELF_CONDITIONING", "0") == "1"
        self.self_conditioning = None if disable_self_conditioning else self_conditioning
        self.self_conditioning_embedding_weight = (
            None if disable_self_conditioning else self_conditioning_embedding_weight
        )
        self.self_conditioning_compute_kernel_config = self_conditioning_compute_kernel_config
        self.q_rope_offset = q_rope_offset
        self.logits_from_tokens = logits_from_tokens
        self.max_denoise_steps = max_denoise_steps
        self.temperature_start = float(temperature_start)
        self.temperature_end = float(temperature_end)
        self.prev_logits = None
        # Trace-safe self-conditioning: persistent [1,1,C,hidden] signal buffer(s)
        # (for the single-step traced loop; KV-cache-style cross-replay state).
        self.trace_safe_self_conditioning = False
        self.signal_buf = None
        # Ping-pong (double-buffered) signal: read buffer != write buffer within a step.
        # Bit-identical on device to the in-place default; kept opt-in (default off) as a
        # verified-equivalent option. The shipped traced path uses the in-place default. See #47465.
        self.signal_ping_pong = False
        self.signal_buf_b = None
        # Cross-block-trace-reusable canvas RoPE (constant-shape per-layer-type buffers).
        self.use_canvas_rope = False
        self._canvas_rope_bufs = {}
        # Paged-prefix Phase 1 reveal mask (constant-shape [1,1,C,p_max+C] written input,
        # content refreshed per block OUTSIDE capture to reveal committed prefix / hide tail).
        self.use_reveal_mask = False
        self._reveal_mask_buf = None
        self._reveal_mask_bufs = {}
        self._reveal_sliding_span = None
        self._reveal_p_max = None
        self._reveal_canvas_len = None
        self._reveal_enforce_window = False
        # Prefill pad slots to hide, in ABSOLUTE prefix positions, so this stays fixed for the whole
        # request while prompt_len grows by a canvas per committed block.
        self._reveal_pad_span = prefill_pad_span(true_prompt_len, prompt_len)

    def prepare_trace_safe_self_conditioning(self, *, canvas_len: int, dtype=ttnn.bfloat16, ping_pong: bool = False):
        """Preallocate the persistent in-place self-cond signal buffer OUTSIDE any trace.

        Intended for the **single-step traced loop** (``tt/denoise_loop.py``): one
        denoise step is captured as a Metal trace and replayed once per step, with
        the self-cond signal carried across replays in this persistent buffer,
        updated in-place each step — exactly the KV-cache pattern that a traced
        decode uses. Uniform-graph: **every** step runs
        ``forward(embed, signal_buf)``; step 0 reads the zeroed buffer, which is
        bit-exact to the ``condition(None)`` (=``post_norm(embed)``) branch because
        ``forward`` with a zero signal has ``pre_norm(0)=0`` → gate/up/down all zero →
        ``post_norm(embed + 0)``. ``reset_signal_buffer`` must be called before each
        block's step 0 to re-zero. See #47465.
        """
        self.signal_ping_pong = ping_pong
        if self.self_conditioning is None:
            self.trace_safe_self_conditioning = True
            self.signal_buf = None
            self.signal_buf_b = None
            return
        hidden_size = self.self_conditioning.hidden_size
        if self.signal_buf is not None:
            self.signal_buf.deallocate(True)
        if self.signal_buf_b is not None:
            self.signal_buf_b.deallocate(True)
            self.signal_buf_b = None

        def _zeros():
            return ttnn.zeros(
                [1, 1, canvas_len, hidden_size],
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.tt_model.mesh_device,
            )

        self.signal_buf = _zeros()
        if ping_pong:
            self.signal_buf_b = _zeros()
        self.trace_safe_self_conditioning = True

    def reset_signal_buffer(self):
        """Zero the persistent signal buffer (call before each block's step 0).

        Outside any trace (a fill is a WRITE, forbidden in capture). Step 0 then
        reads zeros -> ``forward(embed, 0) == post_norm(embed)``, matching the eager
        step-0 ``condition(None)`` branch bit-for-bit.
        """
        if self.signal_buf is not None:
            ttnn.mul(self.signal_buf, 0.0, output_tensor=self.signal_buf)
        if self.signal_buf_b is not None:
            ttnn.mul(self.signal_buf_b, 0.0, output_tensor=self.signal_buf_b)

    def _signal_read_write_bufs(self, step: int):
        """Return (read_buf, write_buf) for this step's self-cond signal.

        In-place (default): read == write == signal_buf. Ping-pong: even steps read
        buffer A and write buffer B, odd steps swap — so step N reads exactly the
        buffer step N-1 wrote, and no step reads+writes the same address in-place.
        """
        if not self.signal_ping_pong:
            return self.signal_buf, self.signal_buf
        if step % 2 == 0:
            return self.signal_buf, self.signal_buf_b
        return self.signal_buf_b, self.signal_buf

    # --- Cross-block-trace-reusable canvas RoPE (second single-step-trace blocker) ---
    #
    # The single-step trace also bakes in the RoPE cos/sin tensors used inside denoise
    # attention. The eager path passes ``_get_rope_mats(seq_len=start_pos+C)`` (a GROWING
    # slice) and applies it at ``start_offset=start_pos`` — so a trace captured on block N
    # is only valid for block N's start_pos. These methods preallocate a CONSTANT-SHAPE
    # ``[1,1,C,head_dim]`` canvas RoPE buffer per layer-type whose CONTENT (the cos/sin for
    # absolute positions ``[start_pos:start_pos+C]``) is refreshed per block OUTSIDE the trace;
    # ``_denoise_layer_forward`` then applies it at ``start_offset=0``. Because RoPE cos/sin
    # depend only on absolute position, that is bit-identical to the growing-slice path, and
    # the trace's RoPE tensor addresses/shapes stay fixed across blocks. See #47465.

    def prepare_canvas_rope_buffers(self, *, canvas_len: int):
        """Preallocate per-layer-type constant-shape canvas RoPE buffers OUTSIDE any trace.

        The buffers are cloned from the FIRST-``canvas_len`` slice of each layer-type's real
        RoPE cache, so their dtype / layout / memory_config match the cache exactly (a later
        ``ttnn.copy`` into them from an offset slice is then a same-spec device copy). Content
        is overwritten per block by ``update_canvas_rope_buffers``.
        """
        tt_model = self.tt_model
        self._canvas_rope_len = canvas_len
        self._canvas_rope_bufs = {}
        layer_types = tt_model.hf_config.layer_types
        for layer_idx in range(len(tt_model.layers)):
            layer_type = layer_types[layer_idx]
            if layer_type in self._canvas_rope_bufs:
                continue
            cos_full, sin_full = tt_model._get_rope_mats(layer_idx)
            bufs = []
            for full in (cos_full, sin_full):
                head = ttnn.slice(full, [0, 0, 0, 0], [full.shape[0], full.shape[1], canvas_len, full.shape[3]])
                bufs.append(ttnn.clone(head))
                head.deallocate(True)
            self._canvas_rope_bufs[layer_type] = (bufs[0], bufs[1])
        self.use_canvas_rope = True

    def update_canvas_rope_buffers(self, start_pos: int):
        """Refresh canvas RoPE buffer content for a block's absolute ``start_pos`` (OUTSIDE trace)."""
        if not getattr(self, "use_canvas_rope", False):
            return
        if start_pos % 32 != 0:
            raise ValueError(f"canvas RoPE start_pos must be a 32-tile multiple, got {start_pos}")
        tt_model = self.tt_model
        C = self._canvas_rope_len
        layer_types = tt_model.hf_config.layer_types
        done = set()
        for layer_idx in range(len(tt_model.layers)):
            layer_type = layer_types[layer_idx]
            if layer_type in done:
                continue
            cos_full, sin_full = tt_model._get_rope_mats(layer_idx)
            cos_buf, sin_buf = self._canvas_rope_bufs[layer_type]
            for full, buf in ((cos_full, cos_buf), (sin_full, sin_buf)):
                sliced = ttnn.slice(
                    full,
                    [0, 0, start_pos, 0],
                    [full.shape[0], full.shape[1], start_pos + C, full.shape[3]],
                )
                ttnn.copy(sliced, buf)
                sliced.deallocate(True)
            done.add(layer_type)

    def _canvas_rope_provider(self, layer_idx):
        layer_type = self.tt_model.hf_config.layer_types[layer_idx]
        return self._canvas_rope_bufs[layer_type]

    def release_canvas_rope_buffers(self):
        try:
            for layer_type, pair in getattr(self, "_canvas_rope_bufs", {}).items():
                for kind, tensor in zip(("cos", "sin"), pair):
                    try:
                        tensor.deallocate(True)
                    except BaseException as cleanup_error:
                        logger.error(f"failed to release canvas RoPE {layer_type}.{kind}: {cleanup_error}")
        finally:
            self._canvas_rope_bufs = {}
            self.use_canvas_rope = False

    # --- Paged-prefix Phase 1 reveal mask (constant-shape written input) ------------
    # A single persistent [1,1,C,p_max+C] additive mask shared by all 30 layers, allocated
    # BEFORE begin_trace_capture and refreshed per block OUTSIDE capture. Phase 1 reveals the
    # committed prefix [0:prompt_len] + all canvas (all-attend, matching the maskless path)
    # and hides the uncommitted tail [prompt_len:p_max] with NEG. Paired with a fixed p_max
    # prefix read (MutablePrefixKVReader.set_read_span) so the traced graph is shape-invariant
    # → capture-once/replay-many.

    def _reveal_mask_layer_types(self) -> tuple[str, ...]:
        """Distinct layer types needing their own reveal mask, in a stable order.

        Only ``sliding_attention`` layers get a different mask, and only when the sliding
        window is enforced. Otherwise one shared ``full_attention`` mask serves every layer,
        exactly as before.
        """
        if not self._reveal_enforce_window:
            return ("full_attention",)
        tt_model = self.tt_model
        types = []
        for layer_idx in range(len(tt_model.layers)):
            layer_type = _layer_type_for_denoise(tt_model, layer_idx)
            key = "sliding_attention" if layer_type == "sliding_attention" else "full_attention"
            if key not in types:
                types.append(key)
        return tuple(types)

    def _sliding_span(self) -> int | None:
        """Bounded span for sliding layers, or None when they read the full p_max."""
        return getattr(self, "_reveal_sliding_span", None)

    def _build_reveal_mask_device(self, prompt_len: int, layer_type: str = "full_attention"):
        sliding_window = None
        if layer_type == "sliding_attention":
            sliding_window = _sliding_window_for_denoise(self.tt_model, self._sliding_reference_layer_idx())
            span = self._sliding_span()
            if span is not None:
                # Bounded read: prefix column r maps to absolute lo + r, so the mask must be
                # built for (span, lo) rather than the full p_max. lo is recomputed here from the
                # same helper the reader uses, keeping mask and read in lockstep.
                lo = sliding_read_offset(int(prompt_len), span, self._reveal_p_max)
                return build_device_canvas_reveal_window_mask(
                    self.tt_model.mesh_device,
                    prompt_len=int(prompt_len),
                    canvas_len=self._reveal_canvas_len,
                    span=span,
                    lo=lo,
                    sliding_window=sliding_window,
                    # Absolute-position span; a no-op once the window has scrolled past the pads.
                    hidden_prefix_span=getattr(self, "_reveal_pad_span", None),
                )
        return build_device_canvas_reveal_mask(
            self.tt_model.mesh_device,
            prompt_len=prompt_len,
            canvas_len=self._reveal_canvas_len,
            p_max=self._reveal_p_max,
            layer_type=layer_type,
            sliding_window=sliding_window,
            enforce_sliding_window=self._reveal_enforce_window and layer_type == "sliding_attention",
            hidden_prefix_span=getattr(self, "_reveal_pad_span", None),
        )

    def _sliding_reference_layer_idx(self) -> int:
        """Index of any sliding layer (they share one window), for reading the window size."""
        tt_model = self.tt_model
        for layer_idx in range(len(tt_model.layers)):
            if _layer_type_for_denoise(tt_model, layer_idx) == "sliding_attention":
                return layer_idx
        return 0

    def prepare_reveal_mask_buffers(
        self,
        *,
        canvas_len: int,
        p_max: int,
        prompt_len: int,
        enforce_window: bool = False,
        sliding_span: int | None = None,
    ):
        """Preallocate the persistent reveal mask(s) OUTSIDE any trace.

        One buffer per distinct layer type. Without ``sliding_span`` they all share the same
        ``[1, 1, C, p_max + C]`` shape, so enforcing the window changes mask CONTENT only. With
        ``sliding_span`` the sliding-layer mask is instead ``[1, 1, C, sliding_span + C]``, matching
        the bounded read those layers now perform — a per-layer-type SHAPE difference, which is
        fine because a trace bakes each layer's own program. Mirrors the canvas-RoPE discipline.
        """
        if p_max % ttnn.TILE_SIZE != 0:
            raise ValueError(f"reveal p_max must be tile aligned, got {p_max}")
        if sliding_span is not None:
            if not enforce_window:
                raise ValueError("a bounded sliding span requires the retention mask (enforce_window)")
            if sliding_span % ttnn.TILE_SIZE != 0 or sliding_span <= 0:
                raise ValueError(f"sliding span must be a positive tile multiple, got {sliding_span}")
            if sliding_span > int(p_max):
                raise ValueError(f"sliding span {sliding_span} exceeds reveal span {p_max}")
        self._reveal_canvas_len = int(canvas_len)
        self._reveal_p_max = int(p_max)
        self._reveal_enforce_window = bool(enforce_window)
        self._reveal_sliding_span = None if sliding_span is None else int(sliding_span)
        self.release_reveal_mask_buffers()
        self._reveal_mask_bufs = {
            layer_type: self._build_reveal_mask_device(int(prompt_len), layer_type)
            for layer_type in self._reveal_mask_layer_types()
        }
        # Back-compat handle for callers/tests that expect the single-mask attribute.
        self._reveal_mask_buf = self._reveal_mask_bufs.get("full_attention")
        self.use_reveal_mask = True

    def update_reveal_mask_buffer(self, prompt_len: int):
        """Refresh reveal mask CONTENT for a block's committed ``prompt_len`` (OUTSIDE trace)."""
        if not self.use_reveal_mask:
            return
        if prompt_len % ttnn.TILE_SIZE != 0:
            raise ValueError(f"reveal prompt_len must be a 32-tile multiple, got {prompt_len}")
        for layer_type, buf in self._reveal_mask_bufs.items():
            fresh = self._build_reveal_mask_device(int(prompt_len), layer_type)
            ttnn.copy(fresh, buf)
            fresh.deallocate(True)

    def _reveal_mask_provider(self, layer_idx):
        bufs = self._reveal_mask_bufs
        if len(bufs) == 1:
            return next(iter(bufs.values()))
        layer_type = _layer_type_for_denoise(self.tt_model, layer_idx)
        key = "sliding_attention" if layer_type == "sliding_attention" else "full_attention"
        return bufs[key]

    def release_reveal_mask_buffers(self):
        for layer_type, buf in getattr(self, "_reveal_mask_bufs", {}).items():
            try:
                if buf is not None:
                    buf.deallocate(True)
            except BaseException as cleanup_error:
                logger.error(f"failed to release reveal mask {layer_type}: {cleanup_error}")
        self._reveal_mask_bufs = {}
        self._reveal_mask_buf = None
        self.use_reveal_mask = False

    def _temperature_at_step(self, step: int) -> float:
        if self.max_denoise_steps is None:
            return 1.0
        from models.experimental.diffusion_gemma.reference.sampling import temperature_at_step

        return temperature_at_step(
            step,
            self.max_denoise_steps,
            self.temperature_start,
            self.temperature_end,
        )

    def _trace_safe_call(self, canvas_tokens, step: int):
        tt_model = self.tt_model
        read_buf, write_buf = self._signal_read_write_bufs(step)
        canvas_hidden = embed_canvas_tokens(tt_model, canvas_tokens)
        if self.self_conditioning is None:
            conditioned = canvas_hidden
        else:
            # Uniform: forward over the persistent signal read buffer (zeroed for step 0).
            conditioned = self.self_conditioning.forward(canvas_hidden, read_buf)
            canvas_hidden.deallocate(True)
        logits = denoise_logits_forward(
            tt_model,
            prompt_hidden_by_layer=self.prompt_hidden_by_layer,
            canvas_hidden=conditioned,
            q_rope_offset=self.q_rope_offset,
            prompt_len=self.prompt_len,
            canvas_rope_provider=self._canvas_rope_provider if self.use_canvas_rope else None,
            reveal_mask_provider=self._reveal_mask_provider if self.use_reveal_mask else None,
        )
        if conditioned is not canvas_hidden:
            conditioned.deallocate(True)
        if self.self_conditioning is not None:
            # Update the persistent signal buffer in-place for the next step (logits
            # is fully consumed within this step: soft_embedding here + the loop's
            # decision path). Across single-step trace replays the buffer persists.
            new_signal = self.self_conditioning.soft_embedding(
                logits,
                self.self_conditioning_embedding_weight,
                compute_kernel_config=self.self_conditioning_compute_kernel_config,
                temperature=self._temperature_at_step(step),
            )
            ttnn.copy(new_signal, write_buf)
            new_signal.deallocate(True)
        return logits

    def __call__(self, canvas_tokens, step: int):
        if self.trace_safe_self_conditioning:
            return self._trace_safe_call(canvas_tokens, step)
        old_prev_logits = self.prev_logits
        logits = self.logits_from_tokens(
            self.tt_model,
            prompt_hidden_by_layer=self.prompt_hidden_by_layer,
            canvas_tokens=canvas_tokens,
            self_conditioning=self.self_conditioning,
            prev_logits=old_prev_logits,
            q_rope_offset=self.q_rope_offset,
            prompt_len=self.prompt_len,
            self_conditioning_embedding_weight=self.self_conditioning_embedding_weight,
            self_conditioning_compute_kernel_config=self.self_conditioning_compute_kernel_config,
            self_conditioning_temperature=(self._temperature_at_step(step - 1) if old_prev_logits is not None else 1.0),
        )
        self.prev_logits = logits
        if old_prev_logits is not None:
            old_prev_logits.deallocate(True)
        return logits

    def owns_logits(self, logits) -> bool:
        """Return True when ``logits`` is retained for next-step self-conditioning."""
        return self.prev_logits is logits

    def rebind_prompt(self, prompt_len: int, *, true_prompt_len: int | None = None) -> None:
        """Bind a startup-captured reveal-mask trace to a newly prefetched request.

        The model-owned KV cache head has already been overwritten by the request prefill.
        Only persistent buffer contents and scalar position state change here; every address
        baked into the trace remains stable. Rebinding a prefix-shape-baked adapter is rejected
        because it would silently replay another request's prompt.
        """
        if not getattr(self, "use_reveal_mask", False):
            raise RuntimeError("prompt rebind requires a captured up-front reveal-mask adapter")
        resetter = getattr(self.prompt_hidden_by_layer, "reset_prompt_len", None)
        if not callable(resetter):
            raise RuntimeError("prompt rebind requires a MutablePrefixKVReader prefix source")

        prompt_len = int(prompt_len)
        canvas_len = int(getattr(self, "_canvas_rope_len", 0) or 0)
        p_max = int(getattr(self, "_reveal_p_max", 0) or 0)
        if canvas_len and p_max and prompt_len + canvas_len > p_max:
            raise ValueError(
                "request exceeds the up-front capture context: "
                f"{prompt_len} + {canvas_len} = {prompt_len + canvas_len} > {p_max}"
            )

        resetter(prompt_len)
        self.prompt_len = prompt_len
        self.q_rope_offset = prompt_len
        # A new request has its own pad count, so this must be recomputed BEFORE the mask refresh
        # below rather than carried over from the previous request.
        self._reveal_pad_span = prefill_pad_span(true_prompt_len, prompt_len)
        self.update_reveal_mask_buffer(prompt_len)
        self.update_canvas_rope_buffers(prompt_len)
        # A new request overwrote the cache head, so the bounded-window buffers hold the previous
        # request's KV until re-filled. Same contract as the mask/RoPE refreshes above.
        self._refresh_prefix_windows(prompt_len)

    def advance_prefix_after_commit(self, next_pos: int) -> bool:
        """Expose newly committed KV to later denoise blocks.

        Returns ``True`` for the mutable contiguous-cache reader used by generation
        and serving. Static prompt-hidden test adapters return ``False``.
        """
        setter = getattr(self.prompt_hidden_by_layer, "set_prompt_len", None)
        if not callable(setter):
            return False
        setter(next_pos)
        self.prompt_len = int(next_pos)
        self.q_rope_offset = int(next_pos)
        # Reveal-mask capture-once: the read span stays fixed at p_max; growth is exposed by
        # revealing the newly committed prefix in the persistent mask (refreshed OUTSIDE any
        # trace, before the next block's replay). The controller demotes its recapture guard.
        if getattr(self, "use_reveal_mask", False):
            self.update_reveal_mask_buffer(int(next_pos))
        # Bounded sliding spans: the window OFFSET slides with the committed prefix, so the
        # block-resident buffers must be re-filled here (outside any trace) in lockstep with the
        # mask above, which is rebuilt from the same sliding_read_offset().
        self._refresh_prefix_windows(int(next_pos))
        return True

    def _refresh_prefix_windows(self, prompt_len: int) -> None:
        refresher = getattr(self.prompt_hidden_by_layer, "refresh_windows", None)
        if callable(refresher):
            refresher(int(prompt_len))

    def reset(self):
        """Release eager and trace-persistent adapter state for request teardown."""
        try:
            if self.prev_logits is not None:
                try:
                    self.prev_logits.deallocate(True)
                except BaseException as cleanup_error:
                    logger.error(f"failed to release previous denoise logits: {cleanup_error}")
            for name in ("signal_buf", "signal_buf_b"):
                tensor = getattr(self, name, None)
                if tensor is not None:
                    try:
                        tensor.deallocate(True)
                    except BaseException as cleanup_error:
                        logger.error(f"failed to release trace self-conditioning {name}: {cleanup_error}")
            self.release_canvas_rope_buffers()
            self.release_reveal_mask_buffers()
            release_prefix_windows = getattr(self.prompt_hidden_by_layer, "release_window_buffers", None)
            if callable(release_prefix_windows):
                release_prefix_windows()
        finally:
            self.prev_logits = None
            self.signal_buf = None
            self.signal_buf_b = None
            self.trace_safe_self_conditioning = False
            self.signal_ping_pong = False


def make_denoise_logits_adapter_from_kv_cache(
    tt_model,
    *,
    prompt_len: int,
    seq_len_start: int = 0,
    self_conditioning=None,
    self_conditioning_embedding_weight=None,
    self_conditioning_compute_kernel_config=None,
    q_rope_offset: int | None = None,
    true_prompt_len: int | None = None,
    max_denoise_steps: int | None = None,
    temperature_start: float = 0.8,
    temperature_end: float = 0.4,
    read_prompt_kv_fn=read_prompt_kv_cache_by_layer,
    adapter_cls=DenoiseLogitsAdapter,
):
    """Build a denoise logits adapter from the model's per-layer prompt KV cache."""

    prompt_kv_for_layer = MutablePrefixKVReader(
        tt_model,
        prompt_len=prompt_len,
        seq_len_start=seq_len_start,
        read_fn=read_prompt_kv_fn,
    )

    return adapter_cls(
        tt_model,
        prompt_hidden_by_layer=prompt_kv_for_layer,
        self_conditioning=self_conditioning,
        self_conditioning_embedding_weight=self_conditioning_embedding_weight,
        self_conditioning_compute_kernel_config=self_conditioning_compute_kernel_config,
        q_rope_offset=prompt_len if q_rope_offset is None else q_rope_offset,
        prompt_len=prompt_len,
        true_prompt_len=true_prompt_len,
        max_denoise_steps=max_denoise_steps,
        temperature_start=temperature_start,
        temperature_end=temperature_end,
    )


def make_denoise_logits_adapter_from_checkpoint_state(
    tt_model,
    *,
    prompt_len: int,
    self_conditioning_state,
    embedding_weight,
    config=None,
    hidden_size: int | None = None,
    intermediate_size: int | None = None,
    eps: float | None = None,
    seq_len_start: int = 0,
    q_rope_offset: int | None = None,
    true_prompt_len: int | None = None,
    self_conditioning_dtype=ttnn.bfloat16,
    self_conditioning_compute_kernel_config=None,
    max_denoise_steps: int | None = None,
    temperature_start: float = 0.8,
    temperature_end: float = 0.4,
    default_compute_kernel_config_fn=default_self_conditioning_compute_kernel_config,
    self_conditioning_builder=build_self_conditioning,
    embedding_weight_builder=build_self_conditioning_embedding_weight,
    adapter_builder=make_denoise_logits_adapter_from_kv_cache,
):
    """Build the full denoise logits adapter from remapped real-checkpoint pieces."""
    self_conditioning = self_conditioning_builder(
        tt_model.mesh_device,
        self_conditioning_state,
        config=config,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        eps=eps,
        dtype=self_conditioning_dtype,
    )
    if config is not None and hidden_size is None:
        hidden_size = config["hidden_size"] if isinstance(config, dict) else config.hidden_size
    embedding_weight_tt = embedding_weight_builder(
        tt_model.mesh_device,
        embedding_weight,
        hidden_size=hidden_size,
        dtype=self_conditioning_dtype,
    )
    if self_conditioning_compute_kernel_config is None:
        self_conditioning_compute_kernel_config = default_compute_kernel_config_fn()
    return adapter_builder(
        tt_model,
        prompt_len=prompt_len,
        seq_len_start=seq_len_start,
        self_conditioning=self_conditioning,
        self_conditioning_embedding_weight=embedding_weight_tt,
        self_conditioning_compute_kernel_config=self_conditioning_compute_kernel_config,
        q_rope_offset=q_rope_offset,
        true_prompt_len=true_prompt_len,
        max_denoise_steps=max_denoise_steps,
        temperature_start=temperature_start,
        temperature_end=temperature_end,
    )


def make_denoise_logits_adapter_from_remapped_state(
    tt_model,
    *,
    prompt_len: int,
    backbone_state,
    self_conditioning_state,
    embedding_key: str = GEMMA4_LM_PREFIX + "embed_tokens.weight",
    checkpoint_adapter_builder=make_denoise_logits_adapter_from_checkpoint_state,
    **kwargs,
):
    """Build a denoise adapter from ``weight_mapping.remap_state_dict`` outputs."""
    if embedding_key not in backbone_state:
        raise ValueError(f"missing tied embedding weight in backbone_state: {embedding_key}")
    return checkpoint_adapter_builder(
        tt_model,
        prompt_len=prompt_len,
        self_conditioning_state=self_conditioning_state,
        embedding_weight=backbone_state[embedding_key],
        **kwargs,
    )


def make_generation_logits_fn_builder_from_remapped_state(
    *,
    backbone_state,
    self_conditioning_state,
    adapter_builder=make_denoise_logits_adapter_from_remapped_state,
    **adapter_kwargs,
):
    """Return a ``tt.generate`` post-prefill builder for remapped checkpoint state."""

    def logits_fn_builder(
        tt_model,
        *,
        prompt_tokens=None,
        prompt_len: int,
        page_table=None,
        page_tables_per_layer=None,
    ):
        del page_table, page_tables_per_layer
        # prompt_tokens is the UNPADDED prompt, so its length is the true one; prompt_len here is the
        # tile-aligned cache_len that prefill wrote. Callers/tests may pass a sentinel for
        # prompt_tokens, so ask for a shape instead of assuming one.
        prompt_shape = getattr(prompt_tokens, "shape", None)
        true_prompt_len = int(prompt_shape[-1]) if prompt_shape is not None else None
        return adapter_builder(
            tt_model,
            prompt_len=prompt_len,
            true_prompt_len=true_prompt_len,
            backbone_state=backbone_state,
            self_conditioning_state=self_conditioning_state,
            **adapter_kwargs,
        )

    return logits_fn_builder


def make_generation_logits_fn_builder_from_checkpoint_state(
    dg_state_dict,
    *,
    remap_fn=remap_state_dict,
    remapped_builder=make_generation_logits_fn_builder_from_remapped_state,
    **adapter_kwargs,
):
    """Return a generation logits builder directly from raw DiffusionGemma state."""
    backbone_state, self_conditioning_state, _ignored_keys = remap_fn(dg_state_dict)
    return remapped_builder(
        backbone_state=backbone_state,
        self_conditioning_state=self_conditioning_state,
        **adapter_kwargs,
    )
