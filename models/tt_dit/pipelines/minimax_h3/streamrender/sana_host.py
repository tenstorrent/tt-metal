# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Run Sana's StreamRender-H3 session logic on the host with the DiT on TT.

Everything except the 50-layer DiT stays Sana's own code from a checkout of `models/streamrender_h3`: window
planning, packing, prefix bookkeeping, the x0 wrapper, the streaming TAE encoder and the causal decoder (both on
CPU here). The DiT is built on the meta device, so it passes Sana's model-type checks while holding no weights,
and its `forward` is the TT adapter.

The Qwen context depends only on the prompt and the picture, so it is computed once with the text encoder and
cached on disk; a session start never needs the 63 GB encoder resident.
"""

from __future__ import annotations

import copy
import hashlib
import io
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

import ttnn

from .dit_adapter import PrefixBank, PrefixCapture, StreamRenderDiT


def mount_sana(root: Path) -> None:
    """Make Sana's runtime (`render`, `runtime`) and its vendored packages importable."""
    for directory in (root, root / "render/vendor", root / "render/decoder"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    from dev.yanzuolu.utils import flex_attn

    # The bootstrap's dense window would otherwise build a torch BlockMask nobody reads: attention runs on TT.
    flex_attn.FLEX_FLASH_ATTN_AVAILABLE = True


def _cpu_model_config(model_config: Any) -> Any:
    from dev.yanzuolu.common.config import CfgNode

    config = CfgNode(copy.deepcopy(dict(model_config)))
    config["placement"] = CfgNode({"device": "cpu"})
    runtime = config.get("runtime")
    if runtime is not None:
        runtime["param_dtype"] = "float32"
        runtime.pop("plugins", None)
    return config


def _build_text_encoder(path: str) -> torch.nn.Module:
    """Sana's truncated Qwen3-VL encoder, loaded from the HF checkpoint.

    Sana's build leaves a meta shell for its DCP plugin to fill; this deployment has the HF checkpoint instead.
    The checkpoint has 64 language layers and an `lm_head`; only the first `MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER`
    layers are built, so those are the only expected unused keys.
    """
    from dev.yanzuolu.projects.minimax_h3_videoref.modeling.ref2va_encoder import (
        MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER,
        MiniMaxH3Ref2VAEncoder,
    )
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration

    config = AutoConfig.from_pretrained(path, local_files_only=True)
    config.text_config.num_hidden_layers = MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER
    config.text_config.use_cache = False
    # The checkpoint is laid out for the generation head (`model.*`, `lm_head`); loading the bare model class
    # does not strip that prefix on every transformers version.
    full, info = Qwen3VLForConditionalGeneration.from_pretrained(
        path, config=config, dtype=torch.bfloat16, local_files_only=True, output_loading_info=True
    )
    if info["missing_keys"]:
        raise RuntimeError(
            f"Qwen3-VL encoder is missing {len(info['missing_keys'])} weights: {info['missing_keys'][:5]}"
        )
    return MiniMaxH3Ref2VAEncoder(full.model).eval()


class _ReferenceDecoderStream:
    """`StreamingCausalViT`'s interface over the torch reference `CausalViTDecoder.decode_chunk`.

    The deployment stream needs FlashAttention-4; the reference is documented as decoding identically chunk by
    chunk (same windows, 4-latent K/V cache and frame trim), with plain SDPA.
    """

    def __init__(self, vit: torch.nn.Module) -> None:
        self.vit = vit
        self.state = None

    def init_state(self) -> "_ReferenceDecoderStream":
        self.state = self.vit.init_state()
        return self

    @torch.no_grad()
    def decode_chunk(self, z: torch.Tensor) -> torch.Tensor:
        return self.vit.decode_chunk(z, self.state)


def _pixel_revert(processor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel `(scale, bias)` of the codec's pixel revert, read off its transform at 0 and 1."""
    zero = processor.transform_rev(torch.zeros(1, 3, 1, 1)).reshape(3)
    one = processor.transform_rev(torch.ones(1, 3, 1, 1)).reshape(3)
    return one - zero, zero


def _rgb_stream_class():
    """Sana's causal ViT stream for a decoder that already emits uint8 RGB: no float revert on the host."""
    from causal_vit_codec import _CausalViTStream

    class RGBStream(_CausalViTStream):
        def _decode_pair(self, z: torch.Tensor) -> torch.Tensor:
            codec = self.codec
            if self.fast is None:
                self.fast = codec.fast_for(z.shape[-2], z.shape[-1], z.device)
                self.fast.init_state()
            pixels = self.fast.decode_chunk(z)
            self.t += codec.chunk
            return pixels.transpose(1, 2)

    return RGBStream


def build_host_models(cfg: Any, dit: StreamRenderDiT, *, text_encoder: bool, mesh_device=None) -> dict[str, Any]:
    """Sana's model dict with the DiT replaced by `dit`; the video decoder runs on `mesh_device` when given."""
    from dev.yanzuolu.common.model import build_model
    from dev.yanzuolu.projects.minimax_h3.modeling.build import MiniMaxH3CausalX0DiT

    backbone_cfg = cfg.models.backbone
    with torch.device("meta"):
        backbone = MiniMaxH3CausalX0DiT(**dict(backbone_cfg.args))
    backbone.eval()
    backbone.dit.forward = dit

    models = {"backbone": backbone}
    decoder_cfg = _cpu_model_config(cfg.models.video_decoder)
    decoder_cfg.args["cuda_graph"] = False
    for name, config in (("video_vae", _cpu_model_config(cfg.models.video_vae)), ("video_decoder", decoder_cfg)):
        models[name] = build_model(name, config)["model"].eval()
    decoder = models["video_decoder"]
    if mesh_device is None:
        decoder.fast_for = lambda H, W, device: _ReferenceDecoderStream(decoder.vit)
    else:
        from ....parallel.manager import CCLManager
        from .causal_decoder import CausalViTDecoderTT

        # Windows shard over the mesh's second axis (SP=8 on a 4x8 galaxy); the first axis replicates.
        ccl_manager = CCLManager(mesh_device=mesh_device, num_links=2, topology=ttnn.Topology.Ring)
        tt_decoders: dict[tuple[int, int], CausalViTDecoderTT] = {}

        def fast_for(H: int, W: int, device: torch.device) -> CausalViTDecoderTT:
            # One TT decoder per latent size; each stream resets it through `init_state`.
            if (H, W) not in tt_decoders:
                tt_decoder = CausalViTDecoderTT(
                    decoder.vit,
                    height=H,
                    width=W,
                    mesh_device=mesh_device,
                    ccl_manager=ccl_manager,
                    sp_axis=1,
                    rgb=_pixel_revert(decoder.processor),
                )
                tt_decoder.load_reference_weights()
                tt_decoders[(H, W)] = tt_decoder
            return tt_decoders[(H, W)]

        decoder.fast_for = fast_for
        decoder.tt_decoders = tt_decoders
        decoder.create_stream = lambda: _rgb_stream_class()(decoder)
    # Untrained on purpose: the streaming decoder only reads the BigVGAN graph to size its audio context, and
    # the runtime publishes video alone, so these weights never touch an output.
    from dev.yanzuolu.projects.minimax_h3.modeling.build import MiniMaxH3AudioVAE

    models["audio_vae"] = MiniMaxH3AudioVAE(**dict(cfg.models.audio_vae.args)).eval()
    if text_encoder:
        models["text_encoder"] = _build_text_encoder(cfg.models.text_encoder.args.pretrained_model_name_or_path)
    return models


def make_meta_class():
    """`render.pipeline.OnlineMeta` with the prefix bank held on TT."""
    from dev.yanzuolu.projects.minimax_h3.modeling.packed_tokens import minimax_h3_unpatchify_video_tokens
    from dev.yanzuolu.projects.minimax_h3.modeling.streaming_prefix import build_streaming_prefix_forward
    from render.pipeline import OnlineMeta

    class TTOnlineMeta(OnlineMeta):
        qwen_cache_dir: Path | None = None

        @staticmethod
        def _sp_size() -> int:
            return 1

        def _validation_qwen_contexts(self, models, prompts, caption_segments, pictures):
            if self.qwen_cache_dir is None:
                return super()._validation_qwen_contexts(models, prompts, caption_segments, pictures)
            digest = hashlib.sha256()
            for prompt, picture in zip(prompts, pictures, strict=True):
                digest.update(str(prompt).encode())
                buffer = io.BytesIO()
                picture["image"].save(buffer, format="PNG")
                digest.update(buffer.getvalue())
            path = self.qwen_cache_dir / f"{digest.hexdigest()[:16]}.pt"
            if path.is_file():
                return torch.load(path, weights_only=False)
            if "text_encoder" not in models:
                raise FileNotFoundError(f"no cached Qwen context at {path}; run once with the text encoder")
            contexts = super()._validation_qwen_contexts(models, prompts, caption_segments, pictures)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(contexts, path)
            return contexts

        def _streaming_prefix_prefill(
            self,
            model,
            inputs,
            *,
            video_xts,
            audio_xts,
            video_timesteps,
            audio_timesteps,
            active=None,
            prefix_masks=None,
        ):
            self.prefills += 1
            if inputs.batch_size != 1:
                raise ValueError("the TT prefix bank serves one stream")
            packed = build_streaming_prefix_forward(
                inputs,
                self._streaming_kwargs(
                    model,
                    inputs,
                    video_xts=video_xts,
                    audio_xts=audio_xts,
                    video_timesteps=video_timesteps,
                    audio_timesteps=audio_timesteps,
                ),
                part="prefix",
                sp_size=1,
                active=active,
                prefix_masks=prefix_masks,
            )
            capture = PrefixCapture()
            packed.kwargs.update(past_key_values=capture, update_past_key_values=True)
            model(**packed.kwargs)
            return [capture.bank]

        def _streaming_prefix_forward(
            self,
            model,
            inputs,
            *,
            video_xts,
            audio_xts,
            video_timesteps,
            audio_timesteps,
            caches=None,
            prefix_masks=None,
        ):
            if caches is None:
                return super()._streaming_prefix_forward(
                    model,
                    inputs,
                    video_xts=video_xts,
                    audio_xts=audio_xts,
                    video_timesteps=video_timesteps,
                    audio_timesteps=audio_timesteps,
                    caches=None,
                    prefix_masks=prefix_masks,
                )
            self.nfe += 1
            (bank,) = caches
            if not isinstance(bank, PrefixBank):
                raise TypeError("a continuation needs the TT prefix bank of its first continuation")
            packed = build_streaming_prefix_forward(
                inputs,
                self._streaming_kwargs(
                    model,
                    inputs,
                    video_xts=video_xts,
                    audio_xts=audio_xts,
                    video_timesteps=video_timesteps,
                    audio_timesteps=audio_timesteps,
                ),
                part="suffix",
                sp_size=1,
                prefix_masks=prefix_masks,
            )
            packed.kwargs.update(past_key_values=bank, update_past_key_values=False)
            video_logits, audio_logits = packed.restore_outputs(model(**packed.kwargs))
            (plan,), (pack,) = inputs.plans, inputs.packs
            channels, _, height, width = plan["target_video_shape"]
            frames = int(pack["noisy_video_frames"])
            video = (
                minimax_h3_unpatchify_video_tokens(
                    video_logits,
                    latent_shape=(frames, height // 2, width // 2, channels),
                    patch_size=(1, 2, 2),
                )[0]
                if frames
                else video_logits.new_empty((channels, 0, height, width))
            )
            audio = audio_logits[: int(pack["audio_rows"])].index_select(0, pack["audio_noisy_sel"])
            audio = audio.reshape(2, int(pack["noisy_audio_frames"]), plan["target_audio_shape"][1]).permute(0, 2, 1)
            return [video], [audio]

    return TTOnlineMeta


def make_pipeline_class():
    """`render.pipeline.StreamingPipeline` timed on the host and releasing TT banks between sessions."""
    from render.pipeline import StreamingPipeline

    class TTStreamingPipeline(StreamingPipeline):
        def __init__(self, config, models, settings, meta_class):
            # The parent builds `OnlineMeta`; ours differs only in the overrides above.
            import render.pipeline as sana_pipeline

            original = sana_pipeline.OnlineMeta
            sana_pipeline.OnlineMeta = meta_class
            try:
                super().__init__(config, models, settings)
            finally:
                sana_pipeline.OnlineMeta = original

        def release_banks(self) -> None:
            # Every trace goes before the session's banks are freed and the next ones allocated: DiT traces read
            # the banks by address, and a replay may overwrite anything allocated after its capture.
            release = getattr(self.models["backbone"].dit.forward, "release_traces", None)
            if release is not None:
                release()
            for tt_decoder in getattr(self.models["video_decoder"], "tt_decoders", {}).values():
                tt_decoder.release_trace()
            for state in self.states or ():
                for bank in state.prefix_banks.values():
                    bank.release()
                state.prefix_banks.clear()

        def start(self, *, prompt, picture, seed):
            self.release_banks()
            super().start(prompt=prompt, picture=picture, seed=seed)
            self.plan_reset()

        def plan_reset(self) -> None:
            self._planned = self.states[0].next_video

        def plan_next(self) -> tuple[int, int]:
            """`(latents, reference frames)` of the next chunk not yet planned; advances the plan.

            The window schedule depends only on the stream position, so a producer can plan, collect and encode
            ahead of the chunk the device is rendering.
            """
            from dev.yanzuolu.projects.minimax_h3.data.streaming import streaming_window_stop

            state = self.states[0]
            stop = streaming_window_stop(
                self._planned, state.streaming_config, video_temporal_mapping=self.meta.video_temporal_mapping
            )
            boundary = self.meta.video_temporal_mapping.decode_timeline.boundary
            counts = stop - self._planned, boundary(stop) - boundary(self._planned)
            self._planned = stop
            return counts

        @torch.inference_mode()
        def encode(self, rgb: torch.Tensor, count: int) -> torch.Tensor:
            """Reference pixels `(3, frames, h, w)` in [0, 1] to `count` normalized TAE latents. Calls must come in
            stream order; one thread may run them while another renders earlier chunks."""
            reference = self.settings["reference"]
            if rgb.shape[0] != 3 or tuple(rgb.shape[2:]) != (reference["height"], reference["width"]):
                raise ValueError(f"Reference chunk geometry mismatch: {tuple(rgb.shape)}")
            codec = self.models["video_vae"]
            value = rgb.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
            pieces = []
            for i in range(rgb.shape[1]):
                piece = self.encoder.encode(value[:, i : i + 1])
                if piece is not None:
                    pieces.append(piece)
            encoded = torch.cat(pieces, 1)[0].permute(1, 0, 2, 3).contiguous()
            encoded = (
                (encoded - codec.latents_mean.to(encoded).view(-1, 1, 1, 1))
                / codec.latents_std.to(encoded).view(-1, 1, 1, 1)
            ).contiguous()
            if encoded.shape[1] != count:
                raise RuntimeError(f"Expected {count} reference latents, got {encoded.shape[1]}")
            return encoded

        @torch.inference_mode()
        def render_encoded(self, rgb: torch.Tensor, encoded: torch.Tensor):
            count = encoded.shape[1]
            expected, _ = self.next_counts()
            if count != expected:
                raise RuntimeError(f"chunk of {count} latents arrived where the stream expects {expected}")
            marks = [time.perf_counter()]
            self.meta.nfe = self.meta.prefills = 0
            chunk = self.meta.step(
                self.models["backbone"],
                self.states,
                [encoded],
                video_counts=[count],
                is_last=[False],
                models=self.models,
                reference_pixels=[rgb],
            )
            marks.append(time.perf_counter())
            output = []
            for i in range(chunk.video[0].shape[1]):
                publication = self.decoder.push(video=chunk.video[0][:, i : i + 1])
                if publication.video is not None:
                    output.append(publication.video)
            marks.append(time.perf_counter())
            result = None if not output else torch.cat(output, dim=2)
            metrics = {
                "round": self.round,
                "native_latents": count,
                "reference_frames": int(rgb.shape[1]),
                "nfe": self.meta.nfe,
                "prefix_prefills": self.meta.prefills,
                "output_frames": 0 if result is None else int(result.shape[2]),
                "render_ms": (marks[1] - marks[0]) * 1000,
                "decode_ms": (marks[2] - marks[1]) * 1000,
                "rgb_ready_wall_ms": (marks[2] - marks[0]) * 1000,
                "publication_interval_ms": None if self.last_ready is None else (marks[2] - self.last_ready) * 1000,
            }
            self.last_ready = marks[2]
            self.round += 1
            return result, metrics

    return TTStreamingPipeline


def qwen_cache_dir() -> Path:
    return Path(os.environ.get("STREAMRENDER_QWEN_CACHE", Path.home() / ".cache/streamrender-qwen"))
