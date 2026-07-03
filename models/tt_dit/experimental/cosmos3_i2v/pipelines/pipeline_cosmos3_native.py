# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native-trunk Cosmos3-I2V pipeline factory (Stage B+C of the Read B pivot).

Replaces the tt-symbiote-shimmed decoder stack with a single call into
the native `Cosmos3OmniTransformer` (model/transformer.py). Everything
*outside* the decoder stack (text embedding, patchify+pack, time
embedder, proj_in, proj_out, mRoPE cos/sin, VAE, scheduler, tokenizer,
unpatchify) stays as host PyTorch using the existing
`Cosmos3OmniPipeline` from `reference/pipeline_cosmos3_omni.py`. Only
the 64-layer for-loop + the two final RMSNorms are swapped.

How it's done:
  1. Load the vendored `Cosmos3OmniPipeline` normally — gets the full
     HF transformer with host weights, plus VAE, scheduler, tokenizer.
  2. Apply the same UniPC `set_begin_index(0)` post-`set_timesteps`
     fix the old pipeline factory uses.
  3. Build a native `Cosmos3OmniTransformer` on the device mesh.
  4. Load the native trunk's weights from
     `hf_transformer.state_dict()` with `strict=False` — picks up
     `layers.{i}.<...>`, `norm.weight`, `norm_moe_gen.weight` and
     ignores all the heads (embed_tokens, proj_in, proj_out, time_*,
     lm_head, audio_*, action_*) which the host pipeline still owns.
  5. Replace `hf_transformer.layers` with a `ModuleList` of one
     `NativeLayerProxy`. The proxy's `forward(und_seq, gen_seq,
     rotary_emb)` converts the inputs from host torch to TILE ttnn on
     the mesh, calls the native trunk (which runs all 64 layers + the
     two final norms on device in one call), and converts the outputs
     back to host torch. The HF forward's `for decoder_layer in
     self.layers:` now runs exactly once.
  6. Replace `hf_transformer.norm` and `hf_transformer.norm_moe_gen`
     with `nn.Identity` — the native trunk already applied them.

Memory: the HF transformer's layer + norm weights stay alive in host
RAM after step 5/6 (Python doesn't free them automatically because
self.layers is just rebound), but they're no longer reachable through
the proxy's forward path. We could `del` them to recover ~100 GB host
RAM at the cost of preventing fallback; that's a separate optimization.

Constraint: `tp_factor` (mesh's larger axis) must divide
`num_key_value_heads=8` (so tp ∈ {1, 2, 4, 8}). LoudBox 1x8 ✓,
BH Galaxy 4x8 with TP=8 on axis 1 ✓.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

import ttnn
from models.tt_dit.experimental.cosmos3_i2v.model_config import HF_REPO
from models.tt_dit.experimental.cosmos3_i2v.pipelines.cosmos3_prompt import install_json_prompt_parsing

if TYPE_CHECKING:
    from models.tt_dit.experimental.cosmos3_i2v.model.transformer import Cosmos3OmniTransformer as NativeTransformer


class NativeLayerProxy(nn.Module):
    """Replaces the HF transformer's `layers` ModuleList with a single one-call shim.

    The HF `Cosmos3OmniTransformer.forward` runs:
        for decoder_layer in self.layers:
            und_seq, gen_seq = decoder_layer(und_seq, gen_seq, rotary_emb)
        und_out = self.norm(und_seq)
        gen_out = self.norm_moe_gen(gen_seq)

    With this proxy registered as the only entry in `self.layers` (and
    `self.norm` / `self.norm_moe_gen` replaced with `nn.Identity`), that
    loop executes once and the entire 64-layer stack + final norms run
    on device in a single native trunk call.

    Input contract: und_seq / gen_seq are 2D `[N, hidden_size]` torch
    tensors. rotary_emb is a 4-tuple of 2D `[N, head_dim]` torch tensors
    (cos_und, sin_und, cos_gen, sin_gen). Outputs match input shapes.
    """

    def __init__(self, native_trunk: NativeTransformer, mesh_device: ttnn.MeshDevice) -> None:
        super().__init__()
        # Hold a non-Module reference so torch.nn doesn't try to introspect the tt-dit Module
        # (which has different attribute layout). The native trunk lives in its own world.
        object.__setattr__(self, "_native_trunk", native_trunk)
        object.__setattr__(self, "_mesh_device", mesh_device)
        from models.tt_dit.experimental.cosmos3_i2v.model.attention import sp_ring_enabled

        cfg_sp_factor = native_trunk.parallel_config.sequence_parallel.factor
        # Effective sp_factor for the proxy's scatter path follows the same env gate as
        # the attention's SP branch. If SP is disabled, treat gen as replicated.
        sp_factor = cfg_sp_factor if sp_ring_enabled() else 1
        sp_axis = native_trunk.parallel_config.sequence_parallel.mesh_axis
        object.__setattr__(self, "_sp_factor", sp_factor)
        object.__setattr__(self, "_sp_axis", sp_axis)
        # When sp>1 the gen seq must divide evenly across sp_axis AND each per-chip slice
        # must be a multiple of k_chunk_size for the ring SDPA op. k_chunk_size=128 is
        # set in the attention module — keep them in sync.
        object.__setattr__(self, "_gen_seq_multiple", 128 * sp_factor if sp_factor > 1 else 1)
        # When the trunk has a device-side proj_out, gen_out comes back with the
        # patch_latent_dim last-dim (e.g. 192) instead of hidden_size (5120). The
        # download path needs the right reshape width and the consumer needs to
        # know we're returning preds-packed-shape, not gen-shape.
        gen_out_last_dim = native_trunk.proj_out.out_features if native_trunk.proj_out is not None else None
        object.__setattr__(self, "_gen_out_last_dim", gen_out_last_dim)
        # Lazy tracer — captured on first forward, replayed on subsequent calls.
        # Opt out with TT_COSMOS3_DISABLE_TRACE=1 if the trace region overflows or a
        # captured op trips a replay invariant.
        object.__setattr__(self, "_trunk_tracer", None)
        # Single-entry cache for the rotary (cos/sin × und/gen) tensors uploaded each
        # forward. The rotary tuple lives in the pipeline's static_pre cache and is
        # reused across every denoise step, so caching by tuple identity holds across
        # all 50 steps. A new generation builds a new tuple → key miss → rebuild
        # (old ttnn tensors fall out of scope and Python GC deallocates them).
        object.__setattr__(self, "_rotary_cache_key", None)
        object.__setattr__(self, "_rotary_cache_value", None)

    @staticmethod
    def _to_tile_ttnn(x: torch.Tensor, mesh_device: ttnn.MeshDevice) -> ttnn.Tensor:
        """Host torch [N, D] (or [N, H]) → replicated TILE ttnn [1, 1, N, D]."""
        # Reshape to [1, 1, N, last] so the native trunk gets the contract it expects.
        # ttnn.from_torch can't reshape into 4D from 2D directly; do it host-side.
        if x.ndim == 2:
            n, d = x.shape
            host = x.detach().to(torch.bfloat16).contiguous().view(1, 1, n, d)
        elif x.ndim == 3:
            b, n, d = x.shape
            host = x.detach().to(torch.bfloat16).contiguous().view(1, b, n, d)
        elif x.ndim == 4:
            host = x.detach().to(torch.bfloat16).contiguous()
        else:
            msg = f"NativeLayerProxy._to_tile_ttnn expected rank-2/3/4 input, got rank {x.ndim}"
            raise ValueError(msg)
        return ttnn.from_torch(
            host,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    @staticmethod
    def _to_tile_ttnn_sharded(
        x: torch.Tensor, mesh_device: ttnn.MeshDevice, sp_axis: int, gen_seq_multiple: int
    ) -> tuple[ttnn.Tensor, int]:
        """Host torch [N, D] → sp-sharded TILE ttnn `[1, 1, N_padded/sp, D]`, plus pad_n.

        Pads N to a multiple of `gen_seq_multiple` (= k_chunk_size * sp_factor) so the
        per-chip slice after scatter is divisible by k_chunk_size — required by the ring
        SDPA op. Padding rows are zeros; the gen pathway's contribution from padded gen
        Q is junk and discarded by the host-side reshape on return. Padded gen K rows
        produce zero attention scores against any Q (dot product of zero K row = 0); the
        zero score is non-zero after softmax but only contaminates the OUTPUT of padded
        Q rows, which we throw away. Real Q rows attend to real K rows (the leading
        N_gen) plus the joint und K (separately handled via the joint -1e4 pad).
        """
        if x.ndim != 2:
            msg = f"NativeLayerProxy._to_tile_ttnn_sharded expects rank-2 input, got rank {x.ndim}"
            raise ValueError(msg)
        n, d = x.shape
        pad_n = (-n) % gen_seq_multiple
        if pad_n > 0:
            x = torch.cat([x, x.new_zeros(pad_n, d)], dim=0)
        host = x.detach().to(torch.bfloat16).contiguous().view(1, 1, n + pad_n, d)
        mesh_shape = tuple(mesh_device.shape)
        if len(mesh_shape) != 2:
            msg = f"sp sharding expects 2D mesh, got {mesh_shape}"
            raise ValueError(msg)
        dims = (2 if sp_axis == 0 else None, 2 if sp_axis == 1 else None)
        mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=dims)
        tensor = ttnn.from_torch(
            host,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        return tensor, pad_n

    @staticmethod
    def _from_replicated_ttnn(x_tt: ttnn.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
        """Replicated TILE ttnn `[1, 1, N, D]` → host torch with the given target shape.

        The native trunk's outputs are replicated across the mesh (after the
        internal all-gather), so device 0's slice carries the full tensor.
        """
        full = ttnn.to_torch(ttnn.get_device_tensors(x_tt)[0])
        return full.reshape(target_shape)

    def forward(
        self,
        und_seq: torch.Tensor,
        gen_seq: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        *,
        time_embed: torch.Tensor | None = None,
        noisy_mask_gen: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        import os as _os
        import threading as _threading
        import time as _time

        _timing = _os.environ.get("TT_COSMOS3_TIMING") in ("1", "true", "True")
        _tid = _threading.get_ident() & 0xFFFF if _timing else 0
        _t0 = _time.perf_counter() if _timing else 0.0

        cos_und, sin_und, cos_gen, sin_gen = rotary_emb
        mesh_device = self._mesh_device

        und_shape = tuple(und_seq.shape)
        gen_shape = tuple(gen_seq.shape)

        und_tt = self._to_tile_ttnn(und_seq, mesh_device)

        sp_factor = self._sp_factor
        sp_axis = self._sp_axis
        gen_seq_multiple = self._gen_seq_multiple

        # The ring SDPA masks the padded gen boundary only at k_chunk granularity, so a
        # gen sequence whose length isn't a multiple of k_chunk_size corrupts attention
        # (the boundary chunk mixes real and padded rows) and the decoded video is noise.
        # gen_seq = latent_t * patch_h * patch_w, so this rejects num_frames / resolution
        # combinations that don't land on the boundary rather than emit silent garbage.
        if sp_factor > 1:
            k_chunk = gen_seq_multiple // sp_factor
            if gen_seq.shape[0] % k_chunk != 0:
                msg = (
                    f"gen sequence length {gen_seq.shape[0]} is not a multiple of k_chunk_size "
                    f"{k_chunk}; the sp={sp_factor} ring SDPA would corrupt attention (noise). "
                    f"Pick a num_frames/resolution whose vision-token count is a multiple of {k_chunk} "
                    f"(e.g. at 720x1280: 61, 125, 189 frames)."
                )
                raise ValueError(msg)

        if sp_factor > 1:
            gen_tt, pad_n_gen = self._to_tile_ttnn_sharded(gen_seq, mesh_device, sp_axis, gen_seq_multiple)
        else:
            gen_tt = self._to_tile_ttnn(gen_seq, mesh_device)
            pad_n_gen = 0
        logical_n_gen = gen_seq.shape[0]

        # Cache the rotary uploads: cos_und/sin_und/cos_gen/sin_gen are derived from
        # position_ids in the pipeline's static_pre cache, so the same 4-tuple feeds
        # every denoise step. Skip the 4 re-uploads on cache hit.
        rotary_key = id(rotary_emb)
        cached = self._rotary_cache_value if self._rotary_cache_key == rotary_key else None
        if cached is None:
            cos_und_tt = self._to_tile_ttnn(cos_und, mesh_device)
            sin_und_tt = self._to_tile_ttnn(sin_und, mesh_device)
            if sp_factor > 1:
                cos_gen_tt, _ = self._to_tile_ttnn_sharded(cos_gen, mesh_device, sp_axis, gen_seq_multiple)
                sin_gen_tt, _ = self._to_tile_ttnn_sharded(sin_gen, mesh_device, sp_axis, gen_seq_multiple)
            else:
                cos_gen_tt = self._to_tile_ttnn(cos_gen, mesh_device)
                sin_gen_tt = self._to_tile_ttnn(sin_gen, mesh_device)
            object.__setattr__(self, "_rotary_cache_value", (cos_und_tt, sin_und_tt, cos_gen_tt, sin_gen_tt))
            object.__setattr__(self, "_rotary_cache_key", rotary_key)
        else:
            cos_und_tt, sin_und_tt, cos_gen_tt, sin_gen_tt = cached

        # When proj_in lives on device, gen_seq enters as raw patches and the trunk
        # also broadcast-adds the timestep embedding under noisy_mask. Upload both
        # (time_embed = single [hidden] vector replicated; noisy_mask = [N_gen, 1]
        # sharded the same way as gen) so the trace replay sees consistent inputs.
        time_embed_tt = None
        noisy_mask_tt = None
        if self._native_trunk.proj_in is not None:
            if time_embed is None or noisy_mask_gen is None:
                msg = "device proj_in path requires both time_embed and noisy_mask_gen"
                raise ValueError(msg)
            time_embed_tt = self._to_tile_ttnn(time_embed, mesh_device)
            if sp_factor > 1:
                noisy_mask_tt, _ = self._to_tile_ttnn_sharded(
                    noisy_mask_gen, mesh_device, self._sp_axis, self._gen_seq_multiple
                )
            else:
                noisy_mask_tt = self._to_tile_ttnn(noisy_mask_gen, mesh_device)

        if _timing:
            _t_uploads_done = _time.perf_counter()
            print(f"[timing] proxy tid={_tid:04x} uploads={(_t_uploads_done - _t0) * 1000:.1f}ms", flush=True)

        trunk_kwargs = {"logical_n_gen": logical_n_gen}
        if time_embed_tt is not None:
            trunk_kwargs["time_embed"] = time_embed_tt
            trunk_kwargs["noisy_mask_gen"] = noisy_mask_tt

        if _os.environ.get("TT_COSMOS3_DISABLE_TRACE") in (None, "", "0", "false", "False"):
            if self._trunk_tracer is None:
                from models.tt_dit.utils.tracing import Tracer

                object.__setattr__(
                    self,
                    "_trunk_tracer",
                    Tracer(self._native_trunk, device=mesh_device, prep_run=True),
                )
            und_out_tt, gen_out_tt = self._trunk_tracer(
                und_tt, gen_tt, cos_und_tt, sin_und_tt, cos_gen_tt, sin_gen_tt, **trunk_kwargs
            )
        else:
            und_out_tt, gen_out_tt = self._native_trunk(
                und_tt, gen_tt, cos_und_tt, sin_und_tt, cos_gen_tt, sin_gen_tt, **trunk_kwargs
            )

        if _timing:
            _t_trunk_done = _time.perf_counter()
            print(f"[timing] proxy tid={_tid:04x} trunk={(_t_trunk_done - _t_uploads_done) * 1000:.1f}ms", flush=True)

        # und_out is unused by I2V (only gen velocity feeds the scheduler — see
        # `learn-cosmos3-i2v-trunk-roundtrips.md`). Skip the download. Return a
        # zero-element placeholder so the legacy caller signature is preserved;
        # callers that need und_out must reinstate this download.
        und_out = und_seq.new_empty((0, und_seq.shape[-1]))

        # gen output last-dim differs from input: hidden_size on the legacy path
        # (no device proj_out), patch_latent_dim when proj_out is on device.
        gen_out_last = self._gen_out_last_dim if self._gen_out_last_dim is not None else gen_shape[-1]
        gen_out_shape = (gen_shape[0], gen_out_last)
        if pad_n_gen > 0:
            # Read full padded gen_out, slice off the pad rows host-side.
            gen_full = ttnn.to_torch(ttnn.get_device_tensors(gen_out_tt)[0]).reshape(-1, gen_out_last)
            gen_out = gen_full[: gen_shape[0]].to(gen_seq.dtype).to(gen_seq.device)
        else:
            gen_out = self._from_replicated_ttnn(gen_out_tt, gen_out_shape).to(gen_seq.dtype).to(gen_seq.device)

        if _timing:
            _t_end = _time.perf_counter()
            print(
                f"[timing] proxy tid={_tid:04x} downloads={(_t_end - _t_trunk_done) * 1000:.1f}ms "
                f"total_proxy={(_t_end - _t0) * 1000:.1f}ms",
                flush=True,
            )

        # Diagnostic: dump per-call (und_in, gen_in, und_out, gen_out) to
        # `${TT_COSMOS3_DUMP_TRUNK_DIR}/call{N}.pt`. Enables direct A/B between
        # SP and no-SP runs at the same step. No-op when env unset.
        import os as _os

        dump_dir = _os.environ.get("TT_COSMOS3_DUMP_TRUNK_DIR")
        if dump_dir:
            _os.makedirs(dump_dir, exist_ok=True)
            call_idx = getattr(self, "_dump_call_idx", 0)
            object.__setattr__(self, "_dump_call_idx", call_idx + 1)
            torch.save(
                {
                    "und_in": und_seq.detach().cpu(),
                    "gen_in": gen_seq.detach().cpu(),
                    "und_out": und_out.detach().cpu(),
                    "gen_out": gen_out.detach().cpu(),
                },
                f"{dump_dir}/call{call_idx:03d}.pt",
            )

        return und_out, gen_out


def build_cosmos3_i2v_native_pipeline(
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
    cache_namespace: str = "cosmos3-i2v",
    enable_device_proj_out: bool = False,
    enable_device_proj_in: bool = False,
):
    """Construct the native-trunk Cosmos3 I2V pipeline.

    Args:
        device: Open ttnn MeshDevice (e.g. 1x8 LoudBox or 4x8 BH Galaxy).
        dtype: torch dtype for the host pieces (default bfloat16).
        hf_repo: HF model id (defaults to the project HF_REPO).
        enable_vae_tiling: enable VAE tiled decode (helps memory at high res).
        num_links: ccl_manager num_links (1 is fine on LoudBox; tune for Galaxy).
        use_tt_vae: monkey-patch the host VAE's encode/decode with TT-NN
            adapters. Disable to fall back to host PyTorch for VAE (useful for
            bisecting VAE vs trunk regressions).
        vae_encoder_t_chunk_size: temporal chunk size for the TT encoder.
            None = full-T single pass (fastest, most memory). 4+ chunks
            (must be >=4 per WanEncoder). Only consulted when use_tt_vae=True.
        vae_decoder_t_chunk_size: temporal chunk size for the TT decoder.
            None = full-T. 1+ chunks. Only consulted when use_tt_vae=True.
    """
    from models.experimental.tt_symbiote.utils.device_management import (  # noqa: F401  reused for cache compatibility
        set_device,
    )
    from models.tt_dit.experimental.cosmos3_i2v.model.transformer import Cosmos3OmniTransformer as NativeTransformer
    from models.tt_dit.experimental.cosmos3_i2v.reference.autoencoder_cosmos3_audio import Cosmos3AVAEAudioTokenizer
    from models.tt_dit.experimental.cosmos3_i2v.reference.pipeline_cosmos3_omni import Cosmos3OmniPipeline
    from models.tt_dit.experimental.cosmos3_i2v.reference.transformer_cosmos3 import Cosmos3OmniTransformer
    from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
    from models.tt_dit.parallel.manager import CCLManager

    if dtype is None:
        dtype = torch.bfloat16

    # Auto-pick num_links based on platform + mesh shape, mirroring
    # `models/tt_dit/pipelines/wan/pipeline_wan.py` presets:
    #   - BH: 2 for all mesh shapes
    #   - WH 4x8: 4
    #   - WH everything else: 1
    if num_links is None:
        mesh_shape_ = tuple(device.shape)
        if ttnn.device.is_blackhole():
            num_links = 2
        elif mesh_shape_ == (4, 8):
            num_links = 4
        else:
            num_links = 1
        print(f"[native-pipeline] auto-picked num_links={num_links} for mesh={mesh_shape_}", flush=True)

    # Step 1: load the full HF pipeline (host PyTorch). Same diffusers-vendoring shim the
    # tt-symbiote pipeline uses to satisfy `from_pretrained`'s class lookup on the pinned
    # diffusers 0.35.1 (Cosmos3OmniTransformer / Cosmos3OmniPipeline are on diffusers/main).
    import diffusers as _diffusers

    _diffusers.Cosmos3OmniTransformer = Cosmos3OmniTransformer
    _diffusers.Cosmos3OmniPipeline = Cosmos3OmniPipeline
    _diffusers.Cosmos3AVAEAudioTokenizer = Cosmos3AVAEAudioTokenizer

    pipe = Cosmos3OmniPipeline.from_pretrained(hf_repo, torch_dtype=dtype, enable_safety_checker=False)

    # VAE tiling is BROKEN in diffusers 0.35.1 AND main for the Wan/Cosmos3 patch_size config:
    #   - tiled_encode: forgets to call patchify() before encoder → channel mismatch
    #     (encoder.conv_in expects 12 channels for patch_size=(1,2,2), gets 3 RGB).
    #     Fixed by monkey-patch below.
    #   - tiled_decode: temporal-causal `avg_shortcut` in `up_block` fires with mismatched
    #     temporal sizes at non-trivial latent T values (e.g. T=21 from 81 input frames).
    #     `RuntimeError: size of tensor a (2) must match tensor b (4) at dim 2`. The
    #     causal-temporal feature cache + first_chunk flag interaction is the culprit.
    #     We don't patch this one — the temporal chunking logic is non-trivial.
    #
    # Workaround: default-disable VAE tiling. For 720x720x81 the non-tiled VAE decode peak
    # is ~5-8 GB host memory, which fits on Galaxy hosts. Pass enable_vae_tiling=True only
    # if you've patched tiled_decode upstream too.
    #
    # The tiled_encode patchify monkey-patch below stays in place either way; harmless
    # when tiling is disabled and protects against accidental re-enables.
    if hasattr(pipe, "vae"):
        from diffusers.models.autoencoders.autoencoder_kl_wan import patchify as _wan_patchify

        _orig_tiled_encode = pipe.vae.tiled_encode
        _vae_patch_size = pipe.vae.config.patch_size

        def _tiled_encode_with_patchify(x):
            if _vae_patch_size is not None and _vae_patch_size != 1:
                x = _wan_patchify(x, patch_size=_vae_patch_size)
            return _orig_tiled_encode(x)

        pipe.vae.tiled_encode = _tiled_encode_with_patchify

        if enable_vae_tiling:
            pipe.vae.enable_tiling()
        elif hasattr(pipe.vae, "disable_tiling"):
            pipe.vae.disable_tiling()

    # Step 1.5: monkey-patch pipe.vae.encode / pipe.vae.decode to run on TT-NN.
    #
    # The reference pipeline (Cosmos3OmniPipeline) calls these via:
    #   - `_encode_video`: `retrieve_latents(self.vae.encode(x), sample_mode="argmax")`
    #     → expects an object with .latent_dist.mode() returning raw mu BCTHW.
    #     Normalization (latents_mean/std) is applied by the caller, not vae.encode.
    #   - end-of-__call__: `self.vae.decode(z_raw).sample` → expects an object
    #     with .sample being BCTHW float video. Denormalization is done by the
    #     caller before this call.
    #
    # Both adapters share `pipe.vae` so we only carry one host copy of the ~5GB
    # AutoencoderKLWan weights (the TT modules pull state_dict from there).
    if use_tt_vae and hasattr(pipe, "vae"):
        from models.tt_dit.experimental.cosmos3_i2v.tokenizer.vae_cosmos3 import (
            Cosmos3VAEDecoderAdapter,
            Cosmos3VAEEncoderAdapter,
        )
        from models.tt_dit.parallel.config import VaeHWParallelConfig
        from models.tt_dit.parallel.manager import CCLManager as _CCLManager

        mesh_shape_vae = tuple(device.shape)
        tp_axis_vae = max(range(len(mesh_shape_vae)), key=lambda i: mesh_shape_vae[i])
        tp_factor_vae = mesh_shape_vae[tp_axis_vae]
        sp_axis_vae = 1 - tp_axis_vae if len(mesh_shape_vae) == 2 else 0
        sp_factor_vae = mesh_shape_vae[sp_axis_vae] if len(mesh_shape_vae) == 2 else 1

        vae_parallel_config = VaeHWParallelConfig.from_tuples(
            height=(tp_factor_vae, tp_axis_vae),
            width=(sp_factor_vae, sp_axis_vae),
        )
        # VAE uses its own CCL manager with Linear topology (per pipeline_wan.py:303).
        vae_ccl_manager = _CCLManager(
            mesh_device=device,
            num_links=num_links,
            topology=ttnn.Topology.Linear,
        )

        print(
            f"[native-pipeline] TT VAE: mesh={mesh_shape_vae} "
            f"vae_parallel=(h={tp_factor_vae}@axis{tp_axis_vae}, w={sp_factor_vae}@axis{sp_axis_vae}) "
            f"encoder_t_chunk={vae_encoder_t_chunk_size} decoder_t_chunk={vae_decoder_t_chunk_size}",
            flush=True,
        )

        tt_vae_encoder = Cosmos3VAEEncoderAdapter(
            checkpoint_name=hf_repo,
            parallel_config=vae_parallel_config,
            ccl_manager=vae_ccl_manager,
            encoder_t_chunk_size=vae_encoder_t_chunk_size,
            vae_dtype=ttnn.bfloat16,
            torch_vae=pipe.vae,
        )

        # Decoder wants height/width/num_frames up front to size its caches.
        # We don't know these until call-time (image size is a runtime arg),
        # so we lazy-initialize on first decode().
        _vae_decoder_holder: dict = {"adapter": None}

        def _make_tt_decoder(height: int, width: int, num_frames: int):
            if _vae_decoder_holder["adapter"] is not None:
                return _vae_decoder_holder["adapter"]
            print(
                f"[native-pipeline] TT VAE decoder: lazy-init H={height} W={width} F={num_frames}",
                flush=True,
            )
            adapter = Cosmos3VAEDecoderAdapter(
                checkpoint_name=hf_repo,
                parallel_config=vae_parallel_config,
                ccl_manager=vae_ccl_manager,
                height=height,
                width=width,
                num_frames=num_frames,
                vae_t_chunk_size=vae_decoder_t_chunk_size,
                vae_dtype=ttnn.bfloat16,
                torch_vae=pipe.vae,
            )
            _vae_decoder_holder["adapter"] = adapter
            return adapter

        # Diffusers-API shims so the reference pipeline can call vae.encode / vae.decode
        # unchanged.
        class _LatentDistShim:
            def __init__(self, mu: torch.Tensor) -> None:
                self._mu = mu

            def mode(self) -> torch.Tensor:
                return self._mu

            def sample(self, generator=None) -> torch.Tensor:  # noqa: ARG002
                # Cosmos3 uses argmax (= mode) so this is fine for inference.
                return self._mu

        class _EncOutShim:
            def __init__(self, mu: torch.Tensor) -> None:
                self.latent_dist = _LatentDistShim(mu)
                self.latents = mu

        class _DecOutShim:
            def __init__(self, sample: torch.Tensor) -> None:
                self.sample = sample

        def _tt_vae_encode(x: torch.Tensor, return_dict: bool = True):  # noqa: ARG001
            raw_mu = tt_vae_encoder.encode(x)
            return _EncOutShim(raw_mu)

        def _tt_vae_decode(z: torch.Tensor, return_dict: bool = True):  # noqa: ARG001
            # z shape is (B, z_dim, T_lat, H_lat, W_lat). Derive pixel dims for
            # the lazy-init decoder.
            sf_s = int(pipe.vae.config.scale_factor_spatial)
            sf_t = int(pipe.vae.config.scale_factor_temporal)
            B, C, T_lat, H_lat, W_lat = z.shape
            height_px = H_lat * sf_s
            width_px = W_lat * sf_s
            num_frames_px = (T_lat - 1) * sf_t + 1
            dec = _make_tt_decoder(height_px, width_px, num_frames_px)
            sample = dec.decode(z, output_type="pt")
            return _DecOutShim(sample)

        pipe.vae.encode = _tt_vae_encode
        pipe.vae.decode = _tt_vae_decode

        # Stash the adapter on pipe for tests/diagnostics.
        pipe._tt_vae_encoder = tt_vae_encoder
        pipe._tt_vae_decoder_factory = _make_tt_decoder

    if hasattr(pipe, "vae"):
        import os as _os
        import time as _time

        try:
            import psutil as _psutil
        except ImportError:
            _psutil = None

        try:
            _vae_param = next(pipe.vae.parameters())
            _vae_device = _vae_param.device
            _vae_dtype = _vae_param.dtype
        except StopIteration:
            _vae_device = "?"
            _vae_dtype = "?"
        print(
            f"[native-pipeline] host VAE device={_vae_device} dtype={_vae_dtype} "
            f"torch_threads={torch.get_num_threads()} cpu_count={_os.cpu_count()} "
            f"training={pipe.vae.training} grad_enabled={torch.is_grad_enabled()}",
            flush=True,
        )

        pipe.vae.eval()
        if _vae_dtype != torch.bfloat16:
            pipe.vae.to(torch.bfloat16)
            print(f"[native-pipeline] host VAE cast {_vae_dtype} -> bfloat16", flush=True)

        _orig_decode = pipe.vae.decode

        def _instrumented_decode(z, *a, **kw):
            rss_before = _psutil.Process().memory_info().rss / 1e9 if _psutil else -1.0
            t0 = _time.perf_counter()
            vae_param_dtype = next(pipe.vae.parameters()).dtype
            if z.dtype != vae_param_dtype:
                z = z.to(vae_param_dtype)
            out = _orig_decode(z, *a, **kw)
            dt = _time.perf_counter() - t0
            rss_after = _psutil.Process().memory_info().rss / 1e9 if _psutil else -1.0
            sample = getattr(out, "sample", out)
            print(
                f"[vae-decode] z={tuple(z.shape)} {z.dtype} -> "
                f"{tuple(sample.shape)} {sample.dtype} "
                f"wall={dt:.1f}s rss_before={rss_before:.1f}GB rss_after={rss_after:.1f}GB",
                flush=True,
            )
            return out

        pipe.vae.decode = _instrumented_decode

    # Override flow_shift for I2V inference. Hub scheduler_config.json defaults
    # to 1.0, but the Cosmos3 paper Table 21 specifies shift=5 for
    # Cosmos3-Super-Image2Video. The shift σ' = s·σ / (1 + (s-1)·σ) biases
    # sigmas toward low-noise refinement.
    if hasattr(pipe.scheduler, "register_to_config"):
        pipe.scheduler.register_to_config(flow_shift=float(flow_shift))
    else:
        pipe.scheduler.config.flow_shift = float(flow_shift)

    # Step 2: UniPC scheduler fix — re-apply set_begin_index(0) after every set_timesteps()
    # call. set_timesteps resets _begin_index = None, wiping out any build-time fix.
    # (Same monkey-patch pattern as the tt-symbiote factory.)
    if hasattr(pipe.scheduler, "set_begin_index"):
        _orig_set_timesteps = pipe.scheduler.set_timesteps

        def _set_timesteps_keep_begin_index_zero(*args, **kwargs):
            _orig_set_timesteps(*args, **kwargs)
            pipe.scheduler.set_begin_index(0)

        pipe.scheduler.set_timesteps = _set_timesteps_keep_begin_index_zero
        pipe.scheduler.set_begin_index(0)

    # Diagnostic: log scheduler.step's inputs and outputs to pinpoint where NaN comes from.
    # The proxy diagnostic showed call 2's gen_seq (= scheduler's output from step 0) is
    # 100% NaN even though call 0's trunk output was clean. The bug is somewhere in the
    # host chain: trunk_out -> proj_out -> unpatchify -> CFG -> scheduler.step -> new_latents.
    # Logging here lets us see whether velocity (input to step) is already NaN or whether
    # step itself produces NaN from clean velocity.
    # Build sigmas the way diffusers `main` does for use_flow_sigmas=True, then let the
    # ORIGINAL diffusers UniPC step() run (predictor + corrector, bh2). Two prior solver
    # variants are now retired:
    #
    #   - Diffusers 0.35.1 UniPC.set_timesteps with use_karras_sigmas + use_flow_sigmas
    #     produces sigmas in the wrong scale (~200) and karras-clustered duplicate
    #     timesteps. So we still bypass set_timesteps and build sigmas ourselves.
    #
    #   - We previously replaced UniPC.step with Adams-Bashforth 2 because raw UniPC.step
    #     NaNs on this config: `_sigma_to_alpha_sigma_t` returns alpha_t = 1 - sigma, and at
    #     sigma[0] = 1.0 that's exactly 0 — then `lambda = log(alpha) - log(sigma)` is -inf.
    #     AB-2 (predictor-only) avoided the log entirely but lacks UniPC's corrector step,
    #     which is the most plausible source of the temporal high-frequency wobble in the
    #     output video.
    #
    # Fix: diffusers main itself guards this with `if |sigmas[0] - 1| < 1e-6: sigmas[0] -=
    # 1e-6`  (scheduling_unipc_multistep.py, around line 437). Apply that same eps shift,
    # then UniPC's bh2 predictor+corrector runs end-to-end without NaN.
    _orig_set_timesteps_reset = pipe.scheduler.set_timesteps

    def _set_timesteps_flow_unipc(num_inference_steps=None, device=None, *args, **kwargs):
        N = num_inference_steps if num_inference_steps is not None else pipe.scheduler.num_inference_steps
        cfg = pipe.scheduler.config
        num_train = cfg.num_train_timesteps

        # Diffusers main flow path: linspace(1, 1/num_train, N+1)[:-1] → N sigmas
        sigmas_np = np.linspace(1.0, 1.0 / float(num_train), N + 1)[:-1]
        # Apply flow_shift (identity at shift=1.0, but keep the line to mirror main)
        shift = float(cfg.flow_shift)
        sigmas_np = shift * sigmas_np / (1.0 + (shift - 1.0) * sigmas_np)
        # Guard against log(0) in multistep_uni_p_bh_update: avoid sigmas[0] == 1.0 exactly.
        eps = 1e-6
        if abs(sigmas_np[0] - 1.0) < eps:
            sigmas_np[0] -= eps
        timesteps_np = sigmas_np * float(num_train)
        # final_sigmas_type: ref config says "zero". Append 0 so the last step's sigma_next
        # is 0 (alpha_t = 1, sigma_t = 0). UniPC's bh_update handles this in the limit; if
        # it produces NaN in practice, swap to "sigma_min" (sigmas_np[-1]).
        if cfg.final_sigmas_type == "sigma_min":
            sigma_last = float(sigmas_np[-1])
        else:
            sigma_last = 0.0
        sigmas_full = np.concatenate([sigmas_np, [sigma_last]]).astype(np.float32)

        sigmas_arr = torch.from_numpy(sigmas_full)
        timesteps_arr = torch.from_numpy(timesteps_np.astype(np.float32))

        target_device = device or (pipe.scheduler.timesteps.device if hasattr(pipe.scheduler, "timesteps") else "cpu")
        try:
            # diffusers keeps sigmas on CPU intentionally (see set_timesteps); only move timesteps.
            timesteps_arr = timesteps_arr.to(target_device)
        except Exception:
            pass

        pipe.scheduler.sigmas = sigmas_arr
        pipe.scheduler.timesteps = timesteps_arr
        pipe.scheduler.num_inference_steps = N

        # Reset UniPC's internal multistep state so the predictor history starts clean.
        pipe.scheduler.model_outputs = [None] * cfg.solver_order
        pipe.scheduler.lower_order_nums = 0
        pipe.scheduler.last_sample = None
        pipe.scheduler._step_index = None
        pipe.scheduler._begin_index = None

        print(
            f"[native-pipeline] flow sigmas (unipc-bh2): N={N} "
            f"sigmas.shape={tuple(sigmas_arr.shape)} "
            f"first3={[float(s) for s in sigmas_arr[:3]]} "
            f"last3={[float(s) for s in sigmas_arr[-3:]]} | "
            f"timesteps first3={[float(t) for t in timesteps_arr[:3]]} "
            f"last3={[float(t) for t in timesteps_arr[-3:]]}",
            flush=True,
        )

    pipe.scheduler.set_timesteps = _set_timesteps_flow_unipc

    # Step 3: pull config off the loaded HF transformer + sanity-check the GQA constraint.
    config = pipe.transformer.config
    mesh_shape = tuple(device.shape)
    tp_axis = max(range(len(mesh_shape)), key=lambda i: mesh_shape[i])
    tp_factor = mesh_shape[tp_axis]
    sp_axis = 1 - tp_axis if len(mesh_shape) == 2 else 0
    sp_factor = mesh_shape[sp_axis] if len(mesh_shape) == 2 else 1

    if config.num_key_value_heads % tp_factor != 0:
        msg = (
            f"native trunk: tp_factor={tp_factor} doesn't divide num_key_value_heads="
            f"{config.num_key_value_heads}. Pick a mesh whose larger axis divides "
            f"{config.num_key_value_heads}."
        )
        raise ValueError(msg)

    # Step 4: build the native trunk on the mesh.
    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(1, 0),
        sequence_parallel=ParallelFactor(sp_factor, sp_axis),
        tensor_parallel=ParallelFactor(tp_factor, tp_axis),
    )
    ccl_manager = (
        CCLManager(mesh_device=device, num_links=num_links, topology=ttnn.Topology.Linear)
        if tp_factor > 1 or sp_factor > 1
        else None
    )

    print(
        f"[native-pipeline] trunk weight dtype = {trunk_weight_dtype} "
        f"({'bfp8: ~halved weight footprint + faster math fidelity tier' if trunk_weight_dtype == ttnn.bfloat8_b else 'bfloat16: highest accuracy'})",
        flush=True,
    )
    native_trunk = NativeTransformer(
        hidden_size=config.hidden_size,
        head_dim=config.head_dim,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        patch_latent_dim=config.patch_latent_dim if (enable_device_proj_in or enable_device_proj_out) else None,
        enable_proj_in=enable_device_proj_in,
        enable_proj_out=enable_device_proj_out,
        attention_bias=getattr(config, "attention_bias", False),
        rms_norm_eps=config.rms_norm_eps,
        mesh_device=device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
        dtype=trunk_weight_dtype,
    )

    # Step 5: load weights via the tt-dit cache layer.
    #
    # The native trunk owns:
    #   - layers.{0..N-1}.<all the decoder-layer keys>
    #   - norm.weight
    #   - norm_moe_gen.weight
    #
    # The HF state dict carries 10 more keys (embed_tokens, lm_head, proj_in/out,
    # time_embedder.linear_1/2, all with .weight and .bias where applicable) which the host
    # HF transformer keeps for the pre/post pieces. `cache.load_model` calls
    # `load_torch_state_dict` with strict=True on cache miss, so we filter the HF state dict
    # down to just what our native trunk expects.
    #
    # The cache layer:
    #   - First call: writes per-Parameter .tensorbin files to
    #     `$TT_DIT_CACHE_DIR/cosmos3-i2v/transformer-native/<config_key>/`.
    #     `config_key` includes parallel_config, mesh_shape, dtype, so different topologies
    #     get distinct directories.
    #   - Later calls with matching config: skip the ~150 s of HF state-dict pumping and
    #     ttnn.from_torch conversion, just ttnn.load_tensor each Parameter directly.
    #
    # If TT_DIT_CACHE_DIR isn't set: falls back to the load_torch_state_dict path with a
    # logger.info hint that caching is available.
    #
    # The existing tt-symbiote factory uses subfolder="transformer"; we deliberately pick
    # a different one so the two on-disk layouts can coexist.
    from models.tt_dit.utils import cache

    expected_top_level = {"layers", "norm", "norm_moe_gen"}
    if enable_device_proj_out:
        expected_top_level.add("proj_out")
    if enable_device_proj_in:
        expected_top_level.add("proj_in")

    def _native_trunk_filtered_state_dict() -> dict:
        """Filter HF state_dict to just the keys the native trunk's children consume.

        Keys are matched by top-level segment: anything starting with `layers.`,
        `norm.`, `norm_moe_gen.` is kept; everything else (embed_tokens / lm_head /
        proj_in / proj_out / time_embedder / rotary_emb / audio_* / action_*) is
        dropped because the host HF transformer still owns those.
        """
        state = pipe.transformer.state_dict()
        return {k: v for k, v in state.items() if k.split(".", 1)[0] in expected_top_level}

    print(
        f"[native-pipeline] hf_transformer.state_dict() has {len(pipe.transformer.state_dict())} "
        f"keys; passing filtered subset to cache.load_model.",
        flush=True,
    )

    if enable_device_proj_in and enable_device_proj_out:
        _cache_subfolder = "transformer-native-proj-in-out"
    elif enable_device_proj_out:
        _cache_subfolder = "transformer-native-proj-out"
    else:
        _cache_subfolder = "transformer-native"
    cache.load_model(
        native_trunk,
        model_name=cache_namespace,
        subfolder=_cache_subfolder,
        parallel_config=parallel_config,
        mesh_shape=mesh_shape,
        dtype="bf16" if trunk_weight_dtype == ttnn.bfloat16 else "bfp8",
        get_torch_state_dict=_native_trunk_filtered_state_dict,
    )

    # Step 6: monkey-patch the HF transformer to use the native trunk for the decoder stack.
    pipe.transformer.layers = nn.ModuleList([NativeLayerProxy(native_trunk, device)])
    pipe.transformer.norm = nn.Identity()
    pipe.transformer.norm_moe_gen = nn.Identity()

    # JSON-object prompts carry the generation specs inline and must be reformatted
    # (not tokenized as free text). The native-cfg builder reuses this builder, so
    # installing here covers both pipelines.
    install_json_prompt_parsing(pipe)

    torch.set_grad_enabled(False)
    return pipe


def _release_native_trunks(pipe) -> None:
    layers = getattr(pipe.transformer, "layers", None)
    if not layers:
        return
    proxy = layers[0]
    trunks = []
    if hasattr(proxy, "_proxy_a") and hasattr(proxy, "_proxy_b"):
        trunks.append(proxy._proxy_a._native_trunk)
        trunks.append(proxy._proxy_b._native_trunk)
    elif hasattr(proxy, "_native_trunk"):
        trunks.append(proxy._native_trunk)
    for trunk in trunks:
        if hasattr(trunk, "deallocate_weights"):
            trunk.deallocate_weights()


def _make_release_callback(num_steps: int):
    # Free the on-device trunk weights after the final denoise step so the VAE
    # decode has DRAM headroom.
    def cb(pipe_ref, step, timestep, callback_kwargs):  # noqa: ARG001
        if step == num_steps - 1:
            _release_native_trunks(pipe_ref)
        return callback_kwargs

    return cb
