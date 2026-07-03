# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""CFG-parallel native-trunk Cosmos3-I2V pipeline.

Splits the mesh along its smaller axis into two submeshes; runs the cond
trunk on submesh A and the uncond trunk on submesh B concurrently from
two Python threads. CFG combine + scheduler.step happen on host on the
two host-resident velocity tensors, so no mesh-socket combiner is needed
(the existing native proxy already brings trunk outputs back to host).

Falls back to the single-submesh `build_cosmos3_i2v_native_pipeline` when
the mesh has no spare axis (LoudBox 1x8) or cfg_parallel is explicitly
disabled.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

import ttnn
from models.tt_dit.experimental.cosmos3_i2v.model_config import HF_REPO
from models.tt_dit.experimental.cosmos3_i2v.pipelines.pipeline_cosmos3_native import (
    NativeLayerProxy,
    build_cosmos3_i2v_native_pipeline,
)

if TYPE_CHECKING:
    from models.tt_dit.experimental.cosmos3_i2v.model.transformer import Cosmos3OmniTransformer as NativeTransformer


class DualSubmeshProxy(nn.Module):
    """One HF-layer proxy backed by two trunks on two submeshes.

    Routes each call to trunk A or B based on a thread-local flag set by
    `Cosmos3OmniPipelineCFG._dispatch_cond_uncond`. Two concurrent threads
    therefore drive two submeshes in parallel through the same shared
    `pipe.transformer` module.
    """

    def __init__(
        self,
        trunk_a: NativeTransformer,
        mesh_a: ttnn.MeshDevice,
        trunk_b: NativeTransformer,
        mesh_b: ttnn.MeshDevice,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "_tls", threading.local())
        object.__setattr__(self, "_proxy_a", NativeLayerProxy(trunk_a, mesh_a))
        object.__setattr__(self, "_proxy_b", NativeLayerProxy(trunk_b, mesh_b))

    def get_proxy(self, which: str) -> NativeLayerProxy:
        return self._proxy_a if which == "a" else self._proxy_b

    def set_active(self, which: str) -> None:
        # `which` is "a" or "b". The caller sets this once per thread before
        # entering pipe.transformer(...); it persists for the duration of the call.
        self._tls.which = which

    def forward(
        self,
        und_seq: torch.Tensor,
        gen_seq: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        which = getattr(self._tls, "which", None)
        if which is None:
            msg = "DualSubmeshProxy: thread-local routing not set; call set_active() first"
            raise RuntimeError(msg)
        proxy = self._proxy_a if which == "a" else self._proxy_b
        return proxy(und_seq, gen_seq, rotary_emb, **kwargs)


def _build_second_trunk(
    submesh: ttnn.MeshDevice,
    *,
    hf_config: Any,
    num_links: int | None,
    trunk_weight_dtype: ttnn.DataType,
    cache_namespace: str = "cosmos3-i2v",
    enable_device_proj_out: bool = False,
    enable_device_proj_in: bool = False,
):
    """Build a NativeTransformer on `submesh` and load weights from the on-disk cache.

    Requires the cache to already exist for this (parallel_config, mesh_shape, dtype)
    triple — populated by the first build_cosmos3_i2v_native_pipeline call. If the
    cache is missing (TT_DIT_CACHE_DIR unset or fresh), this raises and the caller
    must fall back to a path that supplies a torch state_dict.
    """
    from models.tt_dit.experimental.cosmos3_i2v.model.transformer import Cosmos3OmniTransformer as NativeTransformer
    from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
    from models.tt_dit.parallel.manager import CCLManager
    from models.tt_dit.utils import cache

    mesh_shape = tuple(submesh.shape)
    tp_axis = max(range(len(mesh_shape)), key=lambda i: mesh_shape[i])
    tp_factor = mesh_shape[tp_axis]
    sp_axis = 1 - tp_axis if len(mesh_shape) == 2 else 0
    sp_factor = mesh_shape[sp_axis] if len(mesh_shape) == 2 else 1

    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(1, 0),
        sequence_parallel=ParallelFactor(sp_factor, sp_axis),
        tensor_parallel=ParallelFactor(tp_factor, tp_axis),
    )
    if num_links is None:
        num_links = 2 if ttnn.device.is_blackhole() else (4 if mesh_shape == (4, 8) else 1)
    ccl_manager = (
        CCLManager(mesh_device=submesh, num_links=num_links, topology=ttnn.Topology.Linear)
        if tp_factor > 1 or sp_factor > 1
        else None
    )

    trunk = NativeTransformer(
        hidden_size=hf_config.hidden_size,
        head_dim=hf_config.head_dim,
        num_attention_heads=hf_config.num_attention_heads,
        num_key_value_heads=hf_config.num_key_value_heads,
        intermediate_size=hf_config.intermediate_size,
        num_hidden_layers=hf_config.num_hidden_layers,
        patch_latent_dim=hf_config.patch_latent_dim if (enable_device_proj_in or enable_device_proj_out) else None,
        enable_proj_in=enable_device_proj_in,
        enable_proj_out=enable_device_proj_out,
        attention_bias=getattr(hf_config, "attention_bias", False),
        rms_norm_eps=hf_config.rms_norm_eps,
        mesh_device=submesh,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
        dtype=trunk_weight_dtype,
    )

    if enable_device_proj_in and enable_device_proj_out:
        _subfolder = "transformer-native-proj-in-out"
    elif enable_device_proj_out:
        _subfolder = "transformer-native-proj-out"
    else:
        _subfolder = "transformer-native"
    cache.load_model(
        trunk,
        model_name=cache_namespace,
        subfolder=_subfolder,
        parallel_config=parallel_config,
        mesh_shape=mesh_shape,
        dtype="bf16" if trunk_weight_dtype == ttnn.bfloat16 else "bfp8",
        get_torch_state_dict=None,
    )
    return trunk


def _make_cfg_pipeline_class():
    from models.tt_dit.experimental.cosmos3_i2v.reference.pipeline_cosmos3_omni import Cosmos3OmniPipeline

    class Cosmos3OmniPipelineCFG(Cosmos3OmniPipeline):
        """Cosmos3OmniPipeline with cfg-parallel dual-submesh trunk dispatch."""

        # Set by the builder after construction.
        _dual_proxy: DualSubmeshProxy

        def __call__(self, *args, **kwargs):
            # Both caches below are keyed by id() of a per-generation object (input_ids /
            # the rotary tuple). Within one generation that id is stable; across
            # generations a freed object's id gets reused, so a stale entry is a false
            # hit — and a smaller prior gen's rotary then fails a larger one
            # (cos_seq_len < seq_len). Reset both per call.
            cache = getattr(self.transformer, "_static_pre_cache", None)
            if cache is not None:
                cache.clear()
            for proxy in (getattr(self._dual_proxy, "_proxy_a", None), getattr(self._dual_proxy, "_proxy_b", None)):
                if proxy is not None:
                    object.__setattr__(proxy, "_rotary_cache_key", None)
                    object.__setattr__(proxy, "_rotary_cache_value", None)
            return super().__call__(*args, **kwargs)

        def _dispatch_cond_uncond(
            self,
            *,
            cond_packed_static: dict,
            uncond_packed_static: dict,
            vision_tokens: torch.Tensor,
            sound_tokens: torch.Tensor | None,
            action_tokens: torch.Tensor | None,
            vision_timesteps: torch.Tensor,
            sound_timesteps: torch.Tensor | None,
            action_timesteps: torch.Tensor | None,
            action_domain_id: Any,
            vision_condition_mask: torch.Tensor,
            sound_condition_mask: torch.Tensor | None,
            action_condition_mask: torch.Tensor | None,
            raw_action_dim_resolved: int | None,
        ) -> tuple[
            tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None],
            tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None],
        ]:
            # If CFG is off, no point in running two submeshes — fall back to base.
            if not self.do_classifier_free_guidance:
                return super()._dispatch_cond_uncond(
                    cond_packed_static=cond_packed_static,
                    uncond_packed_static=uncond_packed_static,
                    vision_tokens=vision_tokens,
                    sound_tokens=sound_tokens,
                    action_tokens=action_tokens,
                    vision_timesteps=vision_timesteps,
                    sound_timesteps=sound_timesteps,
                    action_timesteps=action_timesteps,
                    action_domain_id=action_domain_id,
                    vision_condition_mask=vision_condition_mask,
                    sound_condition_mask=sound_condition_mask,
                    action_condition_mask=action_condition_mask,
                    raw_action_dim_resolved=raw_action_dim_resolved,
                )

            trunk_kwargs = dict(
                vision_tokens=vision_tokens,
                sound_tokens=sound_tokens,
                action_tokens=action_tokens,
                vision_timesteps=vision_timesteps,
                sound_timesteps=sound_timesteps,
                action_timesteps=action_timesteps,
                action_domain_id=action_domain_id,
            )
            mask_kwargs = dict(
                vision_condition_mask=[vision_condition_mask],
                sound_condition_mask=[sound_condition_mask] if sound_condition_mask is not None else None,
                action_condition_mask=[action_condition_mask] if action_condition_mask is not None else None,
                raw_action_dim=raw_action_dim_resolved,
            )

            proxy = self._dual_proxy

            import os as _os
            import time as _time

            _timing = _os.environ.get("TT_COSMOS3_TIMING") in ("1", "true", "True")
            _step = getattr(self, "_timing_step_idx", 0)
            if _timing:
                object.__setattr__(self, "_timing_step_idx", _step + 1)
                _t0 = _time.perf_counter()

            def _run(which: str, packed_static: dict):
                proxy.set_active(which)
                return self._run_trunk(packed_static, **trunk_kwargs)

            if getattr(self, "_serial_dispatch", False):
                cond_preds_v, cond_preds_s, cond_preds_a = _run("a", cond_packed_static)
                uncond_preds_v, uncond_preds_s, uncond_preds_a = _run("b", uncond_packed_static)
            else:
                with ThreadPoolExecutor(max_workers=2) as ex:
                    cond_future = ex.submit(_run, "a", cond_packed_static)
                    uncond_future = ex.submit(_run, "b", uncond_packed_static)
                    cond_preds_v, cond_preds_s, cond_preds_a = cond_future.result()
                    uncond_preds_v, uncond_preds_s, uncond_preds_a = uncond_future.result()

            if _timing:
                _t_after_join = _time.perf_counter()
                print(
                    f"[timing] step={_step} dispatch_both_threads={(_t_after_join - _t0) * 1000:.1f}ms",
                    flush=True,
                )

            cond_v = self._mask_velocity_predictions(
                cond_preds_v, cond_preds_s, preds_action=cond_preds_a, **mask_kwargs
            )
            uncond_v = self._mask_velocity_predictions(
                uncond_preds_v, uncond_preds_s, preds_action=uncond_preds_a, **mask_kwargs
            )

            if _timing:
                _t_end = _time.perf_counter()
                print(
                    f"[timing] step={_step} mask_velocity={(_t_end - _t_after_join) * 1000:.1f}ms "
                    f"total_dispatch={(_t_end - _t0) * 1000:.1f}ms",
                    flush=True,
                )
            return cond_v, uncond_v

    return Cosmos3OmniPipelineCFG


def build_cosmos3_i2v_native_cfg_pipeline(
    device: ttnn.MeshDevice,
    *,
    dtype: torch.dtype | None = None,
    hf_repo: str = HF_REPO,
    enable_vae_tiling: bool = False,
    num_links: int | None = None,
    trunk_weight_dtype: ttnn.DataType = ttnn.bfloat16,
    use_tt_vae: bool = True,
    vae_encoder_t_chunk_size: int | None = None,
    vae_decoder_t_chunk_size: int | None = None,
    flow_shift: float = 6.0,
    cfg_parallel: bool = True,
    serial_dispatch: bool = False,
    cache_namespace: str = "cosmos3-i2v",
):
    """Build the cfg-parallel native-trunk Cosmos3-I2V pipeline.

    When the mesh has a spare axis (smaller axis > 1) and `cfg_parallel=True`,
    the trunk is replicated across two submeshes and cond/uncond are dispatched
    concurrently. Otherwise this returns the standard single-submesh pipeline.
    """
    mesh_shape = tuple(device.shape)
    can_split = cfg_parallel and len(mesh_shape) == 2 and min(mesh_shape) >= 2

    if not can_split:
        print(
            f"[cfg-pipeline] mesh={mesh_shape} has no spare axis or cfg_parallel disabled; "
            f"falling back to single-submesh build.",
            flush=True,
        )
        return build_cosmos3_i2v_native_pipeline(
            device,
            dtype=dtype,
            hf_repo=hf_repo,
            enable_vae_tiling=enable_vae_tiling,
            num_links=num_links,
            trunk_weight_dtype=trunk_weight_dtype,
            use_tt_vae=use_tt_vae,
            vae_encoder_t_chunk_size=vae_encoder_t_chunk_size,
            vae_decoder_t_chunk_size=vae_decoder_t_chunk_size,
            flow_shift=flow_shift,
            cache_namespace=cache_namespace,
        )

    # Split along the smaller axis (default): each submesh keeps the TP axis intact.
    # `TT_COSMOS3_CFG_SPLIT_LARGER=1` flips to the larger axis, trading TP for SP
    # per submesh. On 4x8 BH Galaxy this turns dual 2x8 (tp=8, sp=2) into dual 4x4
    # (tp=4, sp=4) — doubles SP at the cost of halving TP.
    import os as _os

    if _os.environ.get("TT_COSMOS3_CFG_SPLIT_LARGER") in ("1", "true", "True"):
        cfg_axis = max(range(2), key=lambda i: mesh_shape[i])
    else:
        cfg_axis = min(range(2), key=lambda i: mesh_shape[i] if mesh_shape[i] > 0 else 9999)
    if mesh_shape[cfg_axis] % 2 != 0:
        msg = f"cfg-parallel requires even count on the smaller axis; got mesh={mesh_shape}"
        raise ValueError(msg)
    submesh_shape = list(mesh_shape)
    submesh_shape[cfg_axis] //= 2
    submeshes = device.create_submeshes(ttnn.MeshShape(*submesh_shape))
    if len(submeshes) < 2:
        msg = f"expected 2 submeshes from {mesh_shape}; got {len(submeshes)}"
        raise ValueError(msg)
    submesh_a, submesh_b = submeshes[0], submeshes[1]
    print(
        f"[cfg-pipeline] split mesh {mesh_shape} along axis {cfg_axis} into "
        f"two submeshes of shape {tuple(submesh_a.shape)}",
        flush=True,
    )

    # Build the per-submesh trunk via the existing single-mesh builder, then
    # transplant its trunks onto a DualSubmeshProxy. We invoke the underlying
    # builder twice to get two trunks with weights loaded; the second call
    # reuses tt_dit's tensor cache (same parallel_config + submesh shape).
    pipe_a = build_cosmos3_i2v_native_pipeline(
        submesh_a,
        dtype=dtype,
        hf_repo=hf_repo,
        enable_vae_tiling=enable_vae_tiling,
        num_links=num_links,
        trunk_weight_dtype=trunk_weight_dtype,
        use_tt_vae=use_tt_vae,
        vae_encoder_t_chunk_size=vae_encoder_t_chunk_size,
        vae_decoder_t_chunk_size=vae_decoder_t_chunk_size,
        flow_shift=flow_shift,
        cache_namespace=cache_namespace,
        enable_device_proj_out=True,
        enable_device_proj_in=False,
    )
    trunk_a = pipe_a.transformer.layers[0]._native_trunk

    # Build trunk_b directly on submesh_b. Same parallel_config + mesh_shape as
    # trunk_a, so the tt_dit cache populated by the first build is hit on disk
    # and we skip a second HF from_pretrained + state_dict pump.
    trunk_b = _build_second_trunk(
        submesh_b,
        hf_config=pipe_a.transformer.config,
        num_links=num_links,
        trunk_weight_dtype=trunk_weight_dtype,
        cache_namespace=cache_namespace,
        enable_device_proj_out=True,
        enable_device_proj_in=False,
    )

    dual_proxy = DualSubmeshProxy(trunk_a, submesh_a, trunk_b, submesh_b)
    pipe_a.transformer.layers = nn.ModuleList([dual_proxy])

    cfg_cls = _make_cfg_pipeline_class()
    pipe_a.__class__ = cfg_cls
    object.__setattr__(pipe_a, "_dual_proxy", dual_proxy)
    object.__setattr__(pipe_a, "_serial_dispatch", serial_dispatch)

    _install_device_proj_out_forward(pipe_a)
    _install_timing_hooks(pipe_a, dual_proxy)
    return pipe_a


def _install_device_proj_out_forward(pipe) -> None:
    """Replace `pipe.transformer.forward` with a variant that consumes the new
    proxy contract: gen_out comes back already projected (shape `[N_gen,
    patch_latent_dim]`) from the device-side proj_out. The original HF forward
    cannot run with this contract because (a) it concats und_out [text_len,
    hidden] with gen_out [N_gen, patch_latent_dim] (mismatched last-dim) and
    (b) it then applies a host-side proj_out we've already done on device.

    Pure-I2V only: sound/action paths are not supported on this trunk; asserts
    on entry. Falls back to nothing fancy — copies the HF pre-block verbatim
    up to the layers loop, calls the proxy, slices gen_out at the noisy
    indices, runs unpatchify on host, returns `(preds_vision, None, None)`.
    """

    transformer = pipe.transformer

    # Per (input_ids, position_ids) cache for everything that's constant across denoise
    # steps for one of (cond, uncond). Two entries total: one per pass.
    object.__setattr__(transformer, "_static_pre_cache", {})

    import os as _os
    import time as _time
    import types

    _hoist_timing_enabled = _os.environ.get("TT_COSMOS3_TIMING") in ("1", "true", "True")

    def _native_forward(
        self,
        input_ids,
        text_indexes,
        position_ids,
        und_len,
        sequence_length,
        vision_tokens,
        vision_token_shapes,
        vision_sequence_indexes,
        vision_mse_loss_indexes,
        vision_timesteps,
        vision_noisy_frame_indexes,
        sound_tokens=None,
        sound_token_shapes=None,
        sound_sequence_indexes=None,
        sound_mse_loss_indexes=None,
        sound_timesteps=None,
        sound_noisy_frame_indexes=None,
        action_tokens=None,
        action_token_shapes=None,
        action_sequence_indexes=None,
        action_mse_loss_indexes=None,
        action_timesteps=None,
        action_noisy_frame_indexes=None,
        action_domain_ids=None,
    ):
        if sound_tokens is not None or action_tokens is not None:
            msg = "_native_forward (device proj_out path) only supports pure-I2V; sound/action not implemented"
            raise NotImplementedError(msg)

        t_section = _time.perf_counter() if _hoist_timing_enabled else 0.0

        # Cache invariants per pass: text embed, mRoPE cos/sin (+ und/gen slices),
        # post-trunk gen-relative indices, and the und_seq itself (= text embedding
        # since text_indexes = arange(und_len)). Avoids re-running rotary_emb on the
        # full joint sequence every step and skips the giant joint hidden_states
        # allocation entirely.
        cache_key = id(input_ids)
        cached = self._static_pre_cache.get(cache_key)
        if cached is None:
            packed_text_embedding = self.embed_tokens(input_ids)
            cos_full, sin_full = self.rotary_emb(
                position_ids=position_ids.unsqueeze(0) if position_ids.ndim == 1 else position_ids.unsqueeze(1),
                device=packed_text_embedding.device,
                dtype=packed_text_embedding.dtype,
            )
            cos_full = cos_full.squeeze(0)
            sin_full = sin_full.squeeze(0)
            # text_indexes is arange(und_len) (see _prepare_text_segment), so the text
            # embedding fills positions [0, und_len) of the joint sequence and equals
            # und_seq verbatim.
            und_seq_cached = packed_text_embedding
            cached = {
                "target_dtype": packed_text_embedding.dtype,
                "und_seq": und_seq_cached,
                "rotary_emb": (cos_full[:und_len], sin_full[:und_len], cos_full[und_len:], sin_full[und_len:]),
                "gen_indices": vision_mse_loss_indexes - und_len,
                "gen_len": int(sequence_length) - int(und_len),
                "hidden_size": self.config.hidden_size,
            }
            self._static_pre_cache[cache_key] = cached

        target_dtype = cached["target_dtype"]
        und_seq = cached["und_seq"]
        rotary_emb = cached["rotary_emb"]
        gen_indices = cached["gen_indices"]
        gen_len = cached["gen_len"]
        hidden_size = cached["hidden_size"]

        if _hoist_timing_enabled:
            t_after_cache = _time.perf_counter()

        # Per-step: patchify + host proj_in + time embed + noisy-token scatter.
        # proj_in stays on host until the Phase B PCC regression (0.83) on the
        # ttnn.multiply broadcast semantics is resolved (see git history).
        packed_tokens_vision, original_latent_shapes = self._patchify_and_pack_latents(vision_tokens)
        t_after_patchify = _time.perf_counter() if _hoist_timing_enabled else 0.0
        packed_tokens_vision = self.proj_in(packed_tokens_vision)
        t_after_projin = _time.perf_counter() if _hoist_timing_enabled else 0.0
        # vision_timesteps is torch.full((N,), t) — N identical scalars. time_embedder
        # is row-wise, so running it on all N rows gives N identical outputs. Compute
        # once on the scalar, expand the (1, D) result to (N, D) as a view; downstream
        # scatter_add reads it row-wise so the non-contiguous broadcast is fine.
        scalar_timestep = vision_timesteps[:1] * self.config.timestep_scale
        single_embed = self.time_embedder(self.time_proj(scalar_timestep)).to(target_dtype)
        t_after_timeembed = _time.perf_counter() if _hoist_timing_enabled else 0.0

        # Noisy tokens are packed after the conditioning frame tokens; add the embed
        # in-place on the noisy slice to avoid allocating any temporary tensor.
        # _native_forward is pure-I2V only (sound/action raise NotImplementedError above),
        # so conditioning tokens are always the first n_clean rows of packed_tokens_vision.
        import math as _math

        _n_noisy = sum(
            ni.shape[0] * _math.prod(ts[1:]) for ni, ts in zip(vision_noisy_frame_indexes, vision_token_shapes)
        )
        _n_clean = packed_tokens_vision.shape[0] - _n_noisy
        packed_tokens_vision[_n_clean:].add_(single_embed)

        if _hoist_timing_enabled:
            t_after_vision_pre = _time.perf_counter()
            print(
                f"[timing] vpre patchify={(t_after_patchify - t_after_cache) * 1000:.1f}ms "
                f"projin={(t_after_projin - t_after_patchify) * 1000:.1f}ms "
                f"timeembed={(t_after_timeembed - t_after_projin) * 1000:.1f}ms "
                f"applytsembed={(t_after_vision_pre - t_after_timeembed) * 1000:.1f}ms",
                flush=True,
            )

        # pure-I2V: vision_sequence_indexes = arange(und_len, und_len + gen_len),
        # so vision_sequence_indexes - und_len = arange(gen_len). The scatter is a
        # full copy; use packed_tokens_vision directly as gen_seq.
        gen_seq = packed_tokens_vision

        if _hoist_timing_enabled:
            t_after_gen_build = _time.perf_counter()

        # LAYERS: trunk runs the 64-layer stack + on-device proj_out, returning
        # preds-packed [N_gen, patch_latent_dim].
        proxy = self.layers[0]
        _und_out, gen_out_proj = proxy(und_seq, gen_seq, rotary_emb)

        if _hoist_timing_enabled:
            t_after_layers = _time.perf_counter()

        preds_vision_packed = gen_out_proj[gen_indices]
        preds_vision = self._unpatchify_and_unpack_latents(
            preds_vision_packed,
            token_shapes_vision=vision_token_shapes,
            noisy_frame_indexes_vision=vision_noisy_frame_indexes,
            original_latent_shapes=original_latent_shapes,
        )

        if _hoist_timing_enabled:
            import threading as _threading

            t_end = _time.perf_counter()
            tid = _threading.get_ident() & 0xFFFF
            print(
                f"[timing] fwd tid={tid:04x} "
                f"cache={(t_after_cache - t_section) * 1000:.1f}ms "
                f"vision_pre={(t_after_vision_pre - t_after_cache) * 1000:.1f}ms "
                f"gen_build={(t_after_gen_build - t_after_vision_pre) * 1000:.1f}ms "
                f"layers={(t_after_layers - t_after_gen_build) * 1000:.1f}ms "
                f"post={(t_end - t_after_layers) * 1000:.1f}ms",
                flush=True,
            )

        return preds_vision, None, None

    transformer.forward = types.MethodType(_native_forward, transformer)
    print("[cfg-pipeline] installed device-proj_out transformer.forward (with static-pre cache)", flush=True)


def _install_timing_hooks(pipe, dual_proxy: DualSubmeshProxy) -> None:
    """Wrap HF transformer.forward and DualSubmeshProxy.forward to attribute the
    per-step host gap (HF pre/post vs trunk vs sync). No-op unless
    `TT_COSMOS3_TIMING=1`."""
    import os
    import threading
    import time

    if os.environ.get("TT_COSMOS3_TIMING") not in ("1", "true", "True"):
        return

    _orig_transformer_forward = pipe.transformer.forward

    def _timed_transformer_forward(*args, **kwargs):
        tid = threading.get_ident() & 0xFFFF
        t0 = time.perf_counter()
        out = _orig_transformer_forward(*args, **kwargs)
        dt = (time.perf_counter() - t0) * 1000
        print(f"[timing] hf_transformer tid={tid:04x} total={dt:.1f}ms", flush=True)
        return out

    pipe.transformer.forward = _timed_transformer_forward

    _orig_proxy_forward = dual_proxy.forward

    def _timed_proxy_forward(*args, **kwargs):
        tid = threading.get_ident() & 0xFFFF
        t0 = time.perf_counter()
        out = _orig_proxy_forward(*args, **kwargs)
        dt = (time.perf_counter() - t0) * 1000
        print(f"[timing] dual_proxy tid={tid:04x} layers_call={dt:.1f}ms", flush=True)
        return out

    object.__setattr__(dual_proxy, "forward", _timed_proxy_forward)
    print("[timing] hooks installed (TT_COSMOS3_TIMING=1)", flush=True)
