# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Denoise pipeline stage + builders + streamed driver for the pi0.5 streamed-denoise port.

VENDORED from ``tt_symbiote.models.pipelined_pi05.denoise_pipeline`` with:
  * imports rewired to the native infra (._d2d_bridge, ._d2d_pipeline, ._transport,
    ._module, ._device) and vendored modeling (.modeling.*, .denoise_block).
  * hop + velocity_wrap transports -> SplitSocketTransport (the native split-API subclass).
  * env flags BAKED (plan §9): PI05_VLM_KV_BF16 -> bfloat8_b; PI05_PERF_AH dropped
    (perf_action_horizon takes the real default; perf_suffix_len kept as a tile-math helper).
  * NEW thin ``build_expert_only_pipeline`` (plan §7.6) for the byte-faithful run_expert_chain.
ZERO tt_symbiote imports.
"""
from __future__ import annotations

import inspect

import torch
import ttnn

from ._device import set_device
from ._module import DeviceArch, StatelessTTNNModule, run_on_devices
from ._trace import trace_enabled
from .denoise_block import TTNNPi05DenoiseExpertBlock
from .mesh_carve import carve_four_submeshes
from .modeling.common import precompute_freqs_cis_meta
from .modeling.gemma import TTNNPi05AdaRMSGemmaBlock, _linear_weight_to_tt
from .modeling.suffix import TTNNPi05SuffixEmbedding

TT_METAL_COMMIT = "58672b47cfd304195798bcf34d44f5dbcbcf5189"

_L1 = ttnn.L1_MEMORY_CONFIG
_DRAM = ttnn.DRAM_MEMORY_CONFIG

__all__ = [
    "carve_four_submeshes",
    "TTNNPi05DenoisePipelineStage",
    "build_denoise_pipeline",
    "build_n_stage_pipeline",
    "build_single_stage_reference",
    "build_expert_only_pipeline",
    "TTNNPi05DenoiseExpertBlock",
    "perf_action_horizon",
    "perf_suffix_len",
    "build_denoise_loop_pipeline",
    "TTNNPi05DenoiseStreamedPipeline",
    "euler_schedule",
]


def perf_action_horizon(default: int = 50) -> int:
    # PI05_PERF_AH dropped (plan §9): callers pass the real config.action_horizon. The default
    # mirrors the SuffixConfig default; this helper is retained only for API parity.
    return int(default)


def perf_suffix_len(action_horizon: int) -> int:
    return ((action_horizon + 31) // 32) * 32


def _d2d_transport_modules():
    """Lazy import: socket transport ops are only needed for multi-stage pipelines."""
    from ._d2d_bridge import D2DBridge
    from ._d2d_pipeline import Pipeline
    from ._transport import SplitSocketTransport

    return D2DBridge, Pipeline, SplitSocketTransport


# ----------------------------------------------------------------------------- fill_cache shim
_FILL_CACHE_SHIM_ATTR = "_pi05_pipeline_fill_cache_shim"


def _install_fill_cache_shim():
    """Install an update_idx-capable ttnn.fill_cache shim IFF native fill_cache lacks update_idx.

    Load-bearing ONLY for the static-KV reference/PCC builders; the streamed concat-KV hot
    path never calls fill_cache. Idempotent.
    """
    if getattr(ttnn, _FILL_CACHE_SHIM_ATTR, False):
        return
    _native_fill_cache = ttnn.fill_cache
    try:
        params = inspect.signature(_native_fill_cache).parameters
        # Only treat an EXPLICITLY-named update_idx parameter as native support. The ttnn op
        # decorator wraps the C++ op as ``(*function_args, **function_kwargs)`` -- a VAR_KEYWORD
        # that does NOT actually accept update_idx -- so a VAR_KEYWORD presence must NOT short-
        # circuit the shim (the C++ fill_cache has no update_idx; it would TypeError at call).
        has_update_idx = "update_idx" in params
    except (TypeError, ValueError):
        has_update_idx = False  # builtins with no introspectable signature -> install shim
    if has_update_idx:
        setattr(ttnn, _FILL_CACHE_SHIM_ATTR, True)
        return

    def _fill_cache_compat(cache_tensor, input_tensor, batch_idx, *, update_idx=0):
        if update_idx == 0:
            return _native_fill_cache(cache_tensor, input_tensor, batch_idx)
        s = input_tensor.shape[-2]
        hd = input_tensor.shape[-1]
        for i in range(s):
            row = ttnn.slice(input_tensor, [0, 0, i, 0], [1, 1, i + 1, hd])
            ttnn.update_cache(cache_tensor, row, update_idx + i, batch_offset=batch_idx)
            ttnn.deallocate(row)
        return cache_tensor

    ttnn.fill_cache = _fill_cache_compat
    setattr(ttnn, _FILL_CACHE_SHIM_ATTR, True)


# Install at module import so the static-KV reference/PCC builders see the update_idx-capable
# fill_cache even when imported directly (not only via the package __init__). Idempotent.
_install_fill_cache_shim()


def _kv_dtype() -> "ttnn.DataType":
    # PI05_VLM_KV_BF16 dropped -> bfloat8_b (concat-KV loop default). bf16 PCC-recovery flip
    # documented in PORT_NOTES.
    return ttnn.bfloat8_b


def _bind_prefix_kv(entry, mesh, dtype, memcfg):
    """Resolve one prefix-KV tensor to (dtype, TILE, memcfg) resident on `mesh`.

    Host torch tensor -> from_torch (legacy path). Device tensor already on `mesh` -> relaid out
    on-device, no host round-trip (zero-copy when already in target format). Device tensor on
    another mesh -> relocated via host (to_torch -> from_torch), matching the legacy behaviour."""
    if not isinstance(entry, ttnn.Tensor):
        return ttnn.from_torch(entry, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=memcfg)
    try:
        co_resident = entry.device() is mesh
    except Exception:
        co_resident = False
    if not co_resident:
        return ttnn.from_torch(
            ttnn.to_torch(entry), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=memcfg
        )
    t = entry if entry.layout == ttnn.TILE_LAYOUT else ttnn.to_layout(entry, ttnn.TILE_LAYOUT)
    if t.dtype != dtype:
        return ttnn.typecast(t, dtype, memory_config=memcfg)
    return ttnn.to_memory_config(t, memcfg)


def _to_dram(tensors):
    out = []
    for t in tensors:
        # Preserve sharded tensors as-is (e.g. the fused-residual gate must stay resident
        # width-sharded across the down base cores; moving it to interleaved DRAM would both break
        # the matmul_decode fused-residual contract and add a per-forward reshard). Only the plain
        # interleaved modulation tensors are parked in DRAM.
        if t.memory_config().is_sharded():
            out.append(t)
            continue
        d = ttnn.to_memory_config(t, _DRAM)
        ttnn.deallocate(t)
        out.append(d)
    return tuple(out)


def _slice_rope(cos, sin, seq_len, offset):
    hd = cos.shape[-1]
    c = ttnn.slice(cos, [0, 0, offset, 0], [1, 1, offset + seq_len, hd])
    s = ttnn.slice(sin, [0, 0, offset, 0], [1, 1, offset + seq_len, hd])
    return c, s


@trace_enabled
class TTNNPi05DenoisePipelineStage(StatelessTTNNModule):
    def __init__(
        self,
        *,
        blocks,
        expert_config,
        suffix=None,
        is_first=False,
        is_last=False,
        max_seq_len,
        rope_base,
        eps_expert,
        expert_width,
        prefix_len,
        suffix_len,
        position_offset,
        action_horizon,
        use_concat_kv=False,
    ):
        super().__init__()
        self._bypass_tensor_wrapping = True
        self._use_concat_kv = use_concat_kv
        self._prefix_kv = None
        self.blocks = list(blocks)
        self.suffix = suffix
        self._is_first = is_first
        self._is_last = is_last
        self._expert_config = expert_config
        self._max_seq_len = max_seq_len
        self._rope_base = rope_base
        self._eps_expert = eps_expert
        self._expert_width = expert_width
        self._prefix_len = prefix_len
        self._suffix_len = suffix_len
        self._position_offset = position_offset
        self._action_horizon = action_horizon
        # Resident sliced RoPE cos/sin. The slice window (position_offset + suffix seq_len) is fixed
        # for this stage across all denoise steps, so it is computed once on the first forward and
        # held; the per-forward ttnn.slice pair then drops out of the captured trace.
        self._cos_sin_resident = None
        self._raw_final_norm_mod_w = None
        self._raw_final_norm_mod_b = None
        self._precomputed_block_mods = None
        self._precomputed_final_mod = None
        self._attention_mask = None
        self._tt_final_mod_w = None
        self._tt_final_mod_b = None
        self._tt_expert_norm_ones = None
        self.tt_cos_expert = None
        self.tt_sin_expert = None

    def preprocess_weights_impl(self):
        if self._is_last and self._raw_final_norm_mod_w is not None:
            self._tt_final_mod_w = _linear_weight_to_tt(self._raw_final_norm_mod_w, dtype=ttnn.bfloat16)
            self._tt_final_mod_b = (
                ttnn.from_torch(
                    self._raw_final_norm_mod_b.reshape(1, -1).contiguous(),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                )
                if self._raw_final_norm_mod_b is not None
                else None
            )
            self._tt_expert_norm_ones = ttnn.from_torch(
                torch.ones(1, self._expert_width), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            )

    def move_weights_to_device_impl(self):
        dev = self.device
        self.tt_cos_expert, self.tt_sin_expert = precompute_freqs_cis_meta(
            self._expert_config.head_dim, self._max_seq_len, dev, self._rope_base
        )
        if self._is_last and self._tt_final_mod_w is not None:
            self._tt_final_mod_w = ttnn.to_device(self._tt_final_mod_w, dev, memory_config=_DRAM)
            if self._tt_final_mod_b is not None:
                self._tt_final_mod_b = ttnn.to_device(self._tt_final_mod_b, dev, memory_config=_DRAM)
            self._tt_expert_norm_ones = ttnn.to_device(self._tt_expert_norm_ones, dev, memory_config=_DRAM)

    def _precompute_final_mod(self, cond_dev):
        m = ttnn.linear(cond_dev, self._tt_final_mod_w, bias=self._tt_final_mod_b, memory_config=_L1)
        b = m.shape[0]
        W = self._expert_width
        scale = ttnn.reshape(ttnn.slice(m, [0, 0], [b, W]), (b, 1, W))
        shift = ttnn.reshape(ttnn.slice(m, [0, W], [b, 2 * W]), (b, 1, W))
        ttnn.deallocate(m)
        scale1 = ttnn.add(scale, 1.0, memory_config=_L1)
        ttnn.deallocate(scale)
        return _to_dram((scale1, shift))

    def _ada_rms_norm_no_gate(self, x, precomputed):
        scale1, shift = precomputed
        normed = ttnn.rms_norm(x, weight=self._tt_expert_norm_ones, epsilon=self._eps_expert, memory_config=_L1)
        out = ttnn.multiply(normed, scale1, memory_config=_L1)
        out = ttnn.add(out, shift, memory_config=_L1)
        ttnn.deallocate(normed)
        return out

    @run_on_devices(DeviceArch.P150, DeviceArch.BHGLX)
    def forward(self, x):
        if self._is_first:
            h = self.suffix.embed_actions(x)
        else:
            h = x
        s = h.shape[-2]
        if self._cos_sin_resident is None:
            self._cos_sin_resident = _slice_rope(self.tt_cos_expert, self.tt_sin_expert, s, self._position_offset)
        cos, sin = self._cos_sin_resident
        for i, block in enumerate(self.blocks):
            block_mod = self._precomputed_block_mods[i]
            if self._use_concat_kv:
                pk, pv = self._prefix_kv[i]
                h, _ = block(h, cos, sin, None, self._attention_mask, (pk, pv), False, precomputed_mod=block_mod)
            else:
                h, _ = block(h, cos, sin, None, self._attention_mask, None, False, precomputed_mod=block_mod)
        if self._is_last and self._precomputed_final_mod is not None:
            h = self._ada_rms_norm_no_gate(h, self._precomputed_final_mod)
            h = self.suffix.project_output(h)
        if h.memory_config().buffer_type != ttnn.BufferType.L1:
            h = ttnn.to_memory_config(h, _L1)
        return h


def _bind_stage_runtime(
    stages,
    submeshes_n,
    bounds,
    *,
    config,
    suffix_config,
    adarms_cond_torch,
    prefix_kv_cache,
    prefix_len,
    suffix_len,
    attention_mask_torch,
    bind_mods=True,
):
    kvd = _kv_dtype()
    for k, (lo, hi) in enumerate(bounds):
        st, mesh = stages[k], submeshes_n[k]
        if st._use_concat_kv:
            kvd_concat = _kv_dtype()
            st._prefix_kv = []
            for j, blk in enumerate(st.blocks):
                pk, pv = prefix_kv_cache[lo + j]
                pk_dev = _bind_prefix_kv(pk, mesh, kvd_concat, _L1)
                pv_dev = _bind_prefix_kv(pv, mesh, kvd_concat, _L1)
                st._prefix_kv.append((pk_dev, pv_dev))
        else:
            for j, blk in enumerate(st.blocks):
                pk, pv = prefix_kv_cache[lo + j]
                blk.init_static_kv(prefix_len, suffix_len)
                k_dev = _bind_prefix_kv(pk, mesh, kvd, _DRAM)
                v_dev = _bind_prefix_kv(pv, mesh, kvd, _DRAM)
                blk.fill_static_prefix(k_dev, v_dev)
                ttnn.deallocate(k_dev)
                ttnn.deallocate(v_dev)
        if bind_mods:
            cond_dev = ttnn.from_torch(adarms_cond_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
            st._precomputed_block_mods = [_to_dram(blk.precompute_mods(cond_dev)) for blk in st.blocks]
            if st._is_last and st._tt_final_mod_w is not None:
                st._precomputed_final_mod = st._precompute_final_mod(cond_dev)
            ttnn.deallocate(cond_dev)
        st._attention_mask = ttnn.from_torch(
            attention_mask_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=_DRAM
        )


def _build_stages(
    reference_blocks,
    reference_final_mod_w,
    reference_final_mod_b,
    reference_suffix,
    config,
    suffix_config,
    *,
    prefix_len,
    suffix_len,
    position_offset,
    splits,
    block_cls,
    use_concat_kv,
    expert_only=False,
):
    """Construct the N stages (shared by build_n_stage_pipeline and build_expert_only_pipeline)."""
    ec = config.expert_config
    n = len(splits)
    assert n >= 2 and sum(splits) == len(reference_blocks) == 18
    assert prefix_len % 32 == 0 and suffix_len % 32 == 0 and suffix_len <= 96

    tt_blocks = [block_cls.from_torch(b, ec) for b in reference_blocks]
    bounds, acc = [], 0
    for sp in splits:
        bounds.append((acc, acc + sp))
        acc += sp
    common = dict(
        expert_config=ec,
        max_seq_len=config.max_seq_len,
        rope_base=ec.rope_base,
        eps_expert=ec.rms_norm_eps,
        expert_width=ec.width,
        prefix_len=prefix_len,
        suffix_len=suffix_len,
        position_offset=position_offset,
        action_horizon=suffix_config.action_horizon,
        use_concat_kv=use_concat_kv,
    )
    if expert_only:
        # Bare pipeline: every stage skips embed (is_first=False) AND skips final-norm+project
        # (is_last=False, suffix=None) -> raw post-expert hidden out (plan §7.6).
        stages = [
            TTNNPi05DenoisePipelineStage(blocks=tt_blocks[lo:hi], suffix=None, is_first=False, is_last=False, **common)
            for (lo, hi) in bounds
        ]
        return stages, bounds
    suffix0 = TTNNPi05SuffixEmbedding.from_torch(reference_suffix, suffix_config)
    suffixN = TTNNPi05SuffixEmbedding.from_torch(reference_suffix, suffix_config)
    stages = []
    for k, (lo, hi) in enumerate(bounds):
        stages.append(
            TTNNPi05DenoisePipelineStage(
                blocks=tt_blocks[lo:hi],
                suffix=(suffix0 if k == 0 else (suffixN if k == n - 1 else None)),
                is_first=(k == 0),
                is_last=(k == n - 1),
                **common,
            )
        )
    stages[-1]._raw_final_norm_mod_w = reference_final_mod_w
    stages[-1]._raw_final_norm_mod_b = reference_final_mod_b
    return stages, bounds


def build_n_stage_pipeline(
    reference_blocks,
    reference_final_mod_w,
    reference_final_mod_b,
    reference_suffix,
    config,
    suffix_config,
    parent_mesh,
    *,
    adarms_cond_torch,
    prefix_kv_cache,
    prefix_len,
    suffix_len=64,
    attention_mask_torch,
    position_offset,
    splits,
    submeshes=None,
    block_cls=TTNNPi05AdaRMSGemmaBlock,
    use_concat_kv=False,
) -> Pipeline:
    assert attention_mask_torch is not None, "phantom-suffix mask is REQUIRED"
    n = len(splits)
    stages, bounds = _build_stages(
        reference_blocks,
        reference_final_mod_w,
        reference_final_mod_b,
        reference_suffix,
        config,
        suffix_config,
        prefix_len=prefix_len,
        suffix_len=suffix_len,
        position_offset=position_offset,
        splits=splits,
        block_cls=block_cls,
        use_concat_kv=use_concat_kv,
    )
    sm = submeshes if submeshes is not None else carve_four_submeshes(parent_mesh)
    submeshes_n = sm[:n]
    for st, mesh in zip(stages, submeshes_n):
        set_device(st, mesh)
    _bind_stage_runtime(
        stages,
        submeshes_n,
        bounds,
        config=config,
        suffix_config=suffix_config,
        adarms_cond_torch=adarms_cond_torch,
        prefix_kv_cache=prefix_kv_cache,
        prefix_len=prefix_len,
        suffix_len=suffix_len,
        attention_mask_torch=attention_mask_torch,
    )
    D2DBridge, Pipeline, SplitSocketTransport = _d2d_transport_modules()
    bridges = [
        D2DBridge(stages[i], stages[i + 1], transport=SplitSocketTransport(), tag=f"hop{i}") for i in range(n - 1)
    ]
    return Pipeline(bridges, sync_on_return=True)


def build_expert_only_pipeline(
    reference_blocks,
    reference_suffix,
    config,
    suffix_config,
    parent_mesh,
    *,
    adarms_cond_torch,
    prefix_kv_cache,
    prefix_len,
    suffix_len=64,
    attention_mask_torch,
    position_offset,
    splits,
    submeshes=None,
    block_cls=TTNNPi05DenoiseExpertBlock,
    use_concat_kv=True,
    bind_mods=True,
) -> Pipeline:
    """Bare 18-layer expert chain (NO embed, NO final-norm/project) -- byte-faithful Galaxy
    run_expert_chain semantics (plan §7.6). Input = already-embedded suffix hidden; output =
    raw post-expert hidden on the last stage. Set ``bind_mods=False`` to bind KV + mask only
    and (re-)precompute mods externally per call (the eager per-step path)."""
    assert attention_mask_torch is not None, "phantom-suffix mask is REQUIRED"
    n = len(splits)
    stages, bounds = _build_stages(
        reference_blocks,
        None,
        None,
        reference_suffix,
        config,
        suffix_config,
        prefix_len=prefix_len,
        suffix_len=suffix_len,
        position_offset=position_offset,
        splits=splits,
        block_cls=block_cls,
        use_concat_kv=use_concat_kv,
        expert_only=True,
    )
    sm = submeshes if submeshes is not None else carve_four_submeshes(parent_mesh)
    submeshes_n = sm[:n]
    for st, mesh in zip(stages, submeshes_n):
        set_device(st, mesh)
    _bind_stage_runtime(
        stages,
        submeshes_n,
        bounds,
        config=config,
        suffix_config=suffix_config,
        adarms_cond_torch=adarms_cond_torch,
        prefix_kv_cache=prefix_kv_cache,
        prefix_len=prefix_len,
        suffix_len=suffix_len,
        attention_mask_torch=attention_mask_torch,
        bind_mods=bind_mods,
    )
    D2DBridge, Pipeline, SplitSocketTransport = _d2d_transport_modules()
    bridges = [
        D2DBridge(stages[i], stages[i + 1], transport=SplitSocketTransport(), tag=f"hop{i}") for i in range(n - 1)
    ]
    return Pipeline(bridges, sync_on_return=True)


def build_denoise_pipeline(
    reference_blocks,
    reference_final_mod_w,
    reference_final_mod_b,
    reference_suffix,
    config,
    suffix_config,
    parent_mesh,
    *,
    adarms_cond_torch,
    prefix_kv_cache,
    prefix_len,
    suffix_len=64,
    attention_mask_torch,
    position_offset,
    splits=(5, 5, 4, 4),
    submeshes=None,
    block_cls=TTNNPi05AdaRMSGemmaBlock,
    use_concat_kv=False,
) -> Pipeline:
    assert len(splits) == 4
    return build_n_stage_pipeline(
        reference_blocks,
        reference_final_mod_w,
        reference_final_mod_b,
        reference_suffix,
        config,
        suffix_config,
        parent_mesh,
        adarms_cond_torch=adarms_cond_torch,
        prefix_kv_cache=prefix_kv_cache,
        prefix_len=prefix_len,
        suffix_len=suffix_len,
        attention_mask_torch=attention_mask_torch,
        position_offset=position_offset,
        splits=splits,
        submeshes=submeshes,
        block_cls=block_cls,
        use_concat_kv=use_concat_kv,
    )


def build_single_stage_reference(
    reference_blocks,
    reference_final_mod_w,
    reference_final_mod_b,
    reference_suffix,
    config,
    suffix_config,
    submesh,
    *,
    adarms_cond_torch,
    prefix_kv_cache,
    prefix_len,
    suffix_len=64,
    attention_mask_torch,
    position_offset,
    block_cls=TTNNPi05AdaRMSGemmaBlock,
    use_concat_kv=False,
) -> TTNNPi05DenoisePipelineStage:
    ec = config.expert_config
    assert len(reference_blocks) == 18
    assert prefix_len % 32 == 0 and suffix_len % 32 == 0 and suffix_len <= 96
    assert attention_mask_torch is not None
    tt_blocks = [block_cls.from_torch(b, ec) for b in reference_blocks]
    suffix = TTNNPi05SuffixEmbedding.from_torch(reference_suffix, suffix_config)
    stage = TTNNPi05DenoisePipelineStage(
        blocks=tt_blocks,
        suffix=suffix,
        is_first=True,
        is_last=True,
        expert_config=ec,
        max_seq_len=config.max_seq_len,
        rope_base=ec.rope_base,
        eps_expert=ec.rms_norm_eps,
        expert_width=ec.width,
        prefix_len=prefix_len,
        suffix_len=suffix_len,
        position_offset=position_offset,
        action_horizon=suffix_config.action_horizon,
        use_concat_kv=use_concat_kv,
    )
    stage._raw_final_norm_mod_w = reference_final_mod_w
    stage._raw_final_norm_mod_b = reference_final_mod_b
    set_device(stage, submesh)
    kvd = _kv_dtype()
    if use_concat_kv:
        kvd_concat = _kv_dtype()
        stage._prefix_kv = []
        for j, blk in enumerate(stage.blocks):
            pk, pv = prefix_kv_cache[j]
            pk_dev = _bind_prefix_kv(pk, submesh, kvd_concat, _L1)
            pv_dev = _bind_prefix_kv(pv, submesh, kvd_concat, _L1)
            stage._prefix_kv.append((pk_dev, pv_dev))
    else:
        for j, blk in enumerate(stage.blocks):
            pk, pv = prefix_kv_cache[j]
            blk.init_static_kv(prefix_len, suffix_len)
            k_dev = _bind_prefix_kv(pk, submesh, kvd, _DRAM)
            v_dev = _bind_prefix_kv(pv, submesh, kvd, _DRAM)
            blk.fill_static_prefix(k_dev, v_dev)
            ttnn.deallocate(k_dev)
            ttnn.deallocate(v_dev)
    cond_dev = ttnn.from_torch(adarms_cond_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=submesh)
    stage._precomputed_block_mods = [_to_dram(blk.precompute_mods(cond_dev)) for blk in stage.blocks]
    stage._precomputed_final_mod = stage._precompute_final_mod(cond_dev)
    ttnn.deallocate(cond_dev)
    stage._attention_mask = ttnn.from_torch(
        attention_mask_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=submesh, memory_config=_DRAM
    )
    return stage


def euler_schedule(num_steps):
    timesteps = [1.0 - i / num_steps for i in range(num_steps + 1)]
    dts = [timesteps[i + 1] - timesteps[i] for i in range(num_steps)]
    return timesteps, dts


def _as_l1_dev(out):
    src = out if isinstance(out, ttnn.Tensor) else getattr(out, "ttnn_tensor", out)
    if src.memory_config().buffer_type != ttnn.BufferType.L1:
        src = ttnn.to_memory_config(src, _L1)
    return src


def _rewrap(template, recv):
    if isinstance(template, ttnn.Tensor):
        return recv
    try:
        return type(template)(recv)
    except Exception:
        return recv


class TTNNPi05DenoiseStreamedPipeline:
    def __init__(
        self, pipeline, *, num_steps, action_horizon, per_step_block_mods, per_step_final_mod, dts, drain="all"
    ):
        self._pipe = pipeline
        self._stages = pipeline.stages
        self._meshes = pipeline.meshes
        self._bridges = [pipeline._hop_in[i] for i in range(1, len(self._stages))]
        self._n = num_steps
        self._ah = action_horizon
        self._block_mods = per_step_block_mods
        self._final_mod = per_step_final_mod
        self._dts = dts
        self._drain = drain
        self._stage0_mesh = self._meshes[0]
        self._last_mesh = self._meshes[-1]
        _, Pipeline, SplitSocketTransport = _d2d_transport_modules()
        self._wrap_tp = SplitSocketTransport()
        Pipeline._track_transport(self._wrap_tp)
        self._wrap_tag = "velocity_wrap"
        self._x_t = None
        self._hop_sock = None
        self._wrap_sock = None
        self._loop_tids = None
        # Drill-down instrumentation (plan iter-3 §6.3): None-guarded, eager-only. When
        # _hop_dump is a dict, _emit_step appends ttnn.to_torch snapshots keyed by tap name.
        # NEVER wired into capture_loop -> the captured/replayed op count is unchanged.
        self._hop_dump = None

    def _dump(self, key, ten):
        if self._hop_dump is not None:
            self._hop_dump.setdefault(key, []).append(ttnn.to_torch(ten))

    def _set_step_mods(self, i):
        for k, st in enumerate(self._stages):
            st._precomputed_block_mods = self._block_mods[i][k]
            if st._is_last:
                st._precomputed_final_mod = self._final_mod[i]

    def _emit_step(self, i):
        self._set_step_mods(i)
        self._dump("T_xt_in", self._x_t)
        x_bf16 = ttnn.typecast(self._x_t, ttnn.bfloat16, memory_config=_L1)
        self._dump("T0_xbf16", x_bf16)
        out0 = self._stages[0].forward(x_bf16)
        ttnn.deallocate(x_bf16)
        self._dump("T1_out0", out0)
        self._bridges[0].transport.send_only(_as_l1_dev(out0), self._hop_sock[1]["ss"])
        out = out0
        for s in range(1, len(self._stages)):
            sk = self._hop_sock[s]
            self._bridges[s - 1].transport.recv_only(sk["buf"], sk["rs"])
            self._dump(f"T_recv_s{s}", sk["buf"])
            out = self._stages[s].forward(_rewrap(out, sk["buf"]))
            self._dump(f"T_stageout_s{s}", out)
            if s < len(self._stages) - 1:
                self._bridges[s].transport.send_only(_as_l1_dev(out), self._hop_sock[s + 1]["ss"])
        velocity = out
        self._dump("T5_velocity", velocity)
        vsrc = _as_l1_dev(velocity)
        self._wrap_tp.send_only(vsrc, self._wrap_sock["ss"])
        self._wrap_tp.recv_only(self._wrap_sock["buf"], self._wrap_sock["rs"])
        recv = self._wrap_sock["buf"]
        self._dump("T6_wraprecv", recv)
        v_fp32 = ttnn.typecast(recv, ttnn.float32, memory_config=_L1)
        self._dump("T6b_vfp32", v_fp32)
        v_scaled = ttnn.multiply(v_fp32, self._dts[i], memory_config=_L1)
        ttnn.deallocate(v_fp32)
        ttnn.add(self._x_t, v_scaled, output_tensor=self._x_t)
        ttnn.deallocate(v_scaled)
        self._dump("T7_xt_out", self._x_t)

    def _warmup_caches(self, x_t_init):
        self._x_t = ttnn.from_torch(
            x_t_init, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self._stage0_mesh, memory_config=_L1
        )
        self._hop_sock = [None]
        x_bf16 = ttnn.typecast(self._x_t, ttnn.bfloat16, memory_config=_L1)
        out = self._stages[0].forward(x_bf16)
        ttnn.deallocate(x_bf16)
        for s in range(1, len(self._stages)):
            b = self._bridges[s - 1]
            src = _as_l1_dev(out)
            ss, rs, buf = b.transport.prepare(src, b.mesh_b, tag=b.tag)
            b.transport.send_only(src, ss)
            b.transport.recv_only(buf, rs)
            self._hop_sock.append({"ss": ss, "rs": rs, "buf": buf})
            out = self._stages[s].forward(_rewrap(out, buf))
        velocity = out
        vsrc = _as_l1_dev(velocity)
        ws, wr, wbuf = self._wrap_tp.prepare(vsrc, self._stage0_mesh, tag=self._wrap_tag)
        self._wrap_tp.send_only(vsrc, ws)
        self._wrap_tp.recv_only(wbuf, wr)
        self._wrap_sock = {"ss": ws, "rs": wr, "buf": wbuf}
        v_fp32 = ttnn.typecast(wbuf, ttnn.float32, memory_config=_L1)
        v_scaled = ttnn.multiply(v_fp32, self._dts[0], memory_config=_L1)
        ttnn.deallocate(v_fp32)
        ttnn.add(self._x_t, v_scaled, output_tensor=self._x_t)
        ttnn.deallocate(v_scaled)
        for m in self._pipe._distinct_meshes(self._meshes):
            ttnn.synchronize_device(m)

    def stream_euler(self, x_t_init, *, capture=True):
        self._warmup_caches(x_t_init)
        ttnn.copy(
            ttnn.from_torch(
                x_t_init, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self._stage0_mesh, memory_config=_L1
            ),
            self._x_t,
        )
        if not capture:
            for i in range(self._n):
                self._emit_step(i)
                for m in self._pipe._distinct_meshes(self._meshes):
                    ttnn.synchronize_device(m)
        else:
            self._loop_tids = self._pipe.capture_loop(self._meshes, self._emit_step, self._n)
            drain_mesh = self._stage0_mesh if self._drain == "stage0" else None
            self._pipe.replay_loop(self._loop_tids, drain=self._drain, drain_mesh=drain_mesh)
        for m in self._pipe._distinct_meshes(self._meshes):
            ttnn.synchronize_device(m)
        host = ttnn.to_torch(self._x_t)
        return host[:, : self._ah, :]

    def replay(self):
        assert self._loop_tids is not None, "call stream_euler(capture=True) first"
        # Single drain: the velocity_wrap socket lands the final x_t on stage0 and the readback
        # reads stage0, so syncing stage0 ALONE transitively gates the full critical path.
        self._pipe.replay_loop(self._loop_tids, drain="stage0", drain_mesh=self._stage0_mesh)
        return ttnn.to_torch(self._x_t)[:, : self._ah, :]

    def close(self):
        self._pipe.release_loop(self._loop_tids)
        self._loop_tids = None
        try:
            self._wrap_tp.close()
        except Exception:
            pass
        # Release the inter-stage hop SocketTransports (tracked on the Pipeline class) so they are
        # unbound from the submeshes before the next multi-chip open. release_loop drops only the
        # loop traces; the wrap transport is closed above. Idempotent (re-callers no-op). The carved
        # submeshes are closed by the caller's harness close_parent() (single closer -> no double-close).
        try:
            _, Pipeline, _ = _d2d_transport_modules()
            Pipeline.release_all()
        except Exception:
            pass


def build_denoise_loop_pipeline(
    reference_blocks,
    reference_final_mod_w,
    reference_final_mod_b,
    reference_suffix,
    config,
    suffix_config,
    parent_mesh,
    *,
    adarms_cond_per_step,
    prefix_kv_cache,
    prefix_len,
    suffix_len,
    attention_mask_torch,
    position_offset,
    num_steps,
    action_horizon,
    splits=(5, 5, 4, 4),
    submeshes=None,
    block_cls=TTNNPi05DenoiseExpertBlock,
    use_concat_kv=True,
    drain="all",
):
    assert len(adarms_cond_per_step) == num_steps, "per-step adarms_cond REQUIRED (len == num_steps)"
    pipe = build_n_stage_pipeline(
        reference_blocks,
        reference_final_mod_w,
        reference_final_mod_b,
        reference_suffix,
        config,
        suffix_config,
        parent_mesh,
        adarms_cond_torch=adarms_cond_per_step[0],
        prefix_kv_cache=prefix_kv_cache,
        prefix_len=prefix_len,
        suffix_len=suffix_len,
        attention_mask_torch=attention_mask_torch,
        position_offset=position_offset,
        splits=splits,
        submeshes=submeshes,
        block_cls=block_cls,
        use_concat_kv=use_concat_kv,
    )
    stages = pipe.stages
    meshes = pipe.meshes
    per_step_block_mods = []
    per_step_final_mod = []
    for i in range(num_steps):
        step_block = []
        step_final = None
        for k, st in enumerate(stages):
            cond_dev = ttnn.from_torch(
                adarms_cond_per_step[i], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=meshes[k]
            )
            step_block.append([_to_dram(blk.precompute_mods(cond_dev)) for blk in st.blocks])
            if st._is_last:
                step_final = st._precompute_final_mod(cond_dev)
            ttnn.deallocate(cond_dev)
        per_step_block_mods.append(step_block)
        per_step_final_mod.append(step_final)
    _, dts = euler_schedule(num_steps)
    return TTNNPi05DenoiseStreamedPipeline(
        pipe,
        num_steps=num_steps,
        action_horizon=action_horizon,
        per_step_block_mods=per_step_block_mods,
        per_step_final_mod=per_step_final_mod,
        dts=dts,
        drain=drain,
    )
