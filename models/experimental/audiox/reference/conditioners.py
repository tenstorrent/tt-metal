# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""AudioX conditioner stack (reference / CPU).

The AudioX HF config (HKUSTAudio/AudioX) wires three cross-attention
conditioners — CLIP (video), T5 (text), AudioAutoencoder (audio) — into a
``MultiConditioner`` dispatcher. Conditioners run **once per generation**,
producing tensors of shape [B, S_k, 768] that get concatenated along the
sequence dim into ``cross_attn_cond`` for the DiT (which runs ~50 steps × 24
blocks per call). Conditioner compute is <0.1% of total — porting them to
TTNN would not move the needle, so this layer stays on CPU and reuses the
HuggingFace models directly. The TTNN side ingests the concatenated tensor
via ``ttnn.from_torch`` at the DiT boundary.

CLIP + AudioAutoencoder both have a learned ``empty_*_feat`` parameter and
a zero-input fast-path: when the corresponding modality is absent (the
common text-to-audio case), the empty-feat tensor is returned directly and
the heavy encoder is skipped entirely. The full encoder paths only run when
real video / audio prompts are supplied.
"""

import typing as tp

import torch
from torch import nn

from models.experimental.audiox.reference.temptransformer import SATransformer


class Conditioner(nn.Module):
    """Base class: optional output projection so each conditioner emits the
    shared ``cond_token_dim`` (768 for AudioX) regardless of its native dim."""

    def __init__(self, dim: int, output_dim: int, project_out: bool = False):
        super().__init__()
        self.dim = dim
        self.output_dim = output_dim
        self.proj_out = nn.Linear(dim, output_dim) if (dim != output_dim or project_out) else nn.Identity()

    def forward(self, x: tp.Any, device: tp.Any) -> tp.Any:
        raise NotImplementedError()


class T5Conditioner(Conditioner):
    """Wraps a HuggingFace T5 encoder + an output projection. AudioX uses
    ``t5-base`` (dim 768) with ``max_length=128``, padded to fixed length so
    the cross-attn sequence shape is constant across generations.

    The HF model is held outside ``self`` (via ``__dict__``) so it does not
    appear in ``state_dict`` — matches upstream and keeps checkpoint loading
    of the surrounding pipeline clean."""

    T5_MODELS = ("t5-small", "t5-base", "t5-large", "google/flan-t5-base", "google/flan-t5-large")
    T5_MODEL_DIMS = {
        "t5-small": 512,
        "t5-base": 768,
        "t5-large": 1024,
        "google/flan-t5-base": 768,
        "google/flan-t5-large": 1024,
    }

    def __init__(
        self,
        output_dim: int,
        t5_model_name: str = "t5-base",
        max_length: int = 128,
        project_out: bool = False,
    ):
        assert t5_model_name in self.T5_MODELS, f"Unknown T5 model: {t5_model_name}"
        super().__init__(self.T5_MODEL_DIMS[t5_model_name], output_dim, project_out=project_out)

        from transformers import AutoTokenizer, T5EncoderModel

        self.t5_model_name = t5_model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
        # Held outside state_dict; loaded weights stay on the HF model side.
        self.__dict__["model"] = T5EncoderModel.from_pretrained(t5_model_name).eval().requires_grad_(False)

    def forward(self, texts: tp.List[str], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device).to(torch.bool)

        self.model.to(device)
        self.proj_out.to(device)

        with torch.no_grad():
            embeddings = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        embeddings = self.proj_out(embeddings.float())
        # Zero-out padded positions so they contribute nothing through cross-attn.
        embeddings = embeddings * attention_mask.unsqueeze(-1).float()
        return embeddings, attention_mask


class MultiConditioner(nn.Module):
    """Dispatcher: maps a list of per-sample metadata dicts into a dict of
    (cond_tensor, mask) tuples keyed by conditioner id. Mirrors upstream
    ``audiox/models/conditioners.py:MultiConditioner`` for inference (no
    drop-out, no negative-prompt handling here — the diffusion wrapper owns
    that)."""

    def __init__(
        self,
        conditioners: tp.Dict[str, Conditioner],
        default_keys: tp.Optional[tp.Dict[str, str]] = None,
    ):
        super().__init__()
        self.conditioners = nn.ModuleDict(conditioners)
        self.default_keys = default_keys or {}

    def forward(
        self,
        batch_metadata: tp.List[tp.Dict[str, tp.Any]],
        device: tp.Union[torch.device, str],
    ) -> tp.Dict[str, tp.Any]:
        output = {}
        for key, conditioner in self.conditioners.items():
            inputs = []
            for sample in batch_metadata:
                ck = key if key in sample else self.default_keys.get(key, key)
                if ck not in sample:
                    raise ValueError(f"Conditioner key '{ck}' not found in batch metadata")
                value = sample[ck]
                # Upstream unwraps single-element list/tuple but keeps multi-element ones.
                if isinstance(value, (list, tuple)) and len(value) == 1:
                    value = value[0]
                inputs.append(value)
            output[key] = conditioner(inputs, device)
        return output


class CLIPConditioner(Conditioner):
    """Mirrors AudioX's ``CLIPWithSyncWithEmptyFeatureConditioner``.

    AudioX wires this with ``clip-vit-base-patch32``, ``video_fps=5``,
    ``duration=10`` and a 4-block / 16-head SA temporal transformer. The
    visual encoder produces 50 patch tokens per frame; ``Temp_pos_embedding``
    adds a position embedding across the ``duration*fps = 50`` frames before
    the temporal transformer mixes them.

    For text-to-audio (the demo target) the video tensor is all-zero — the
    ``is_zero`` shortcut returns ``empty_visual_feat`` directly and the
    HF visual encoder is never called. The full encoder path stays here so
    video-to-audio generation works once a real video tensor is passed in.

    The HF visual encoder is held outside ``self`` so it doesn't appear in
    ``state_dict`` (mirrors upstream and keeps checkpoint loading clean)."""

    def __init__(
        self,
        output_dim: int,
        clip_model_name: str = "clip-vit-base-patch32",
        video_fps: int = 5,
        duration: int = 10,
        out_features: int = 128,
        temporal_dim: int = 768,
        sa_depth: int = 4,
        sa_heads: int = 16,
        sa_dim_head: int = 64,
        sa_hidden_scale: int = 4,
        project_out: bool = False,
    ):
        super().__init__(dim=temporal_dim, output_dim=output_dim, project_out=project_out)

        # Patch tokens per frame for ViT-B/32 (49 patches + 1 CLS).
        tokens_per_frame = 50
        in_features = tokens_per_frame * video_fps * duration
        num_frames = duration * video_fps

        self.clip_model_name = clip_model_name
        self.in_features = in_features
        self.out_features = out_features

        self.empty_visual_feat = nn.Parameter(torch.zeros(1, out_features, temporal_dim))
        self.proj = nn.Linear(in_features, out_features)
        self.proj_sync = nn.Linear(240, out_features)
        self.sync_weight = nn.Parameter(torch.zeros(()))
        self.Temp_pos_embedding = nn.Parameter(torch.randn(1, num_frames, temporal_dim))
        self.Temp_transformer = SATransformer(
            dim=temporal_dim,
            depth=sa_depth,
            heads=sa_heads,
            dim_head=sa_dim_head,
            mlp_dim=temporal_dim * sa_hidden_scale,
        )

        # Held outside state_dict; loaded weights stay on the HF model side.
        self.__dict__["visual_encoder_model"] = None  # lazy: only built when a non-empty video is seen.

        # CLIP normalization stats (used only on the non-empty path).
        self.register_buffer("clip_mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer("clip_std", torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))

    def _ensure_visual_encoder(self):
        if self.visual_encoder_model is None:
            from transformers import CLIPVisionModelWithProjection

            hf_id = "openai/" + self.clip_model_name
            self.__dict__["visual_encoder_model"] = (
                CLIPVisionModelWithProjection.from_pretrained(hf_id, use_safetensors=True).eval().requires_grad_(False)
            )

    def forward(
        self, video_inputs: tp.List[tp.Any], device: tp.Union[torch.device, str]
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        # Each sample is either a tensor [1, T, C, H, W] or a dict with
        # ``video_tensors`` (and optional ``video_sync_frames``). Cat across
        # the leading 1-dim to form a single batched tensor.
        if isinstance(video_inputs[0], dict):
            video_tensors = [item["video_tensors"] for item in video_inputs]
            sync_frames_list = [item.get("video_sync_frames") for item in video_inputs]
            if all(s is not None for s in sync_frames_list):
                video_sync_frames = torch.cat(sync_frames_list, dim=0).to(device)
            else:
                video_sync_frames = None
        else:
            video_tensors = video_inputs
            video_sync_frames = None

        videos = torch.cat(video_tensors, dim=0).to(device)
        batch, time_length = videos.shape[0], videos.shape[1]
        is_zero = torch.all(videos == 0, dim=tuple(range(1, videos.ndim)))

        if video_sync_frames is None:
            video_sync_frames = torch.zeros(batch, 240, 768, device=device)

        # Fast-path: every sample is empty -> skip CLIP entirely. AudioX's
        # text-to-audio demo always hits this branch.
        if bool(is_zero.all()):
            video_hidden = self.empty_visual_feat.to(device).expand(batch, -1, -1)
            return video_hidden, torch.ones(batch, 1, device=device)

        # Full path: flatten time into batch, run CLIP, mix temporally.
        self._ensure_visual_encoder()
        self.visual_encoder_model.to(device)
        self.proj.to(device)
        self.proj_sync.to(device)

        frames = videos.reshape(batch * time_length, *videos.shape[2:])
        frames = (frames / 255.0 - self.clip_mean) / self.clip_std
        with torch.no_grad():
            video_hidden = self.visual_encoder_model(pixel_values=frames).last_hidden_state

        # [(B*T), Q, H] -> [(B*Q), T, H] for temporal attention across frames.
        bt, q, h = video_hidden.shape
        video_hidden = (
            video_hidden.reshape(batch, time_length, q, h).permute(0, 2, 1, 3).reshape(batch * q, time_length, h)
        )
        video_hidden = video_hidden + self.Temp_pos_embedding
        video_hidden = self.Temp_transformer(video_hidden)
        # [(B*Q), T, H] -> [B, T*Q, H]
        video_hidden = (
            video_hidden.reshape(batch, q, time_length, h).permute(0, 2, 1, 3).reshape(batch, time_length * q, h)
        )

        video_hidden = self.proj(video_hidden.reshape(-1, self.in_features))
        video_hidden = video_hidden.reshape(batch, self.out_features, -1)

        sync = self.proj_sync(video_sync_frames.reshape(-1, 240)).reshape(batch, self.out_features, -1)
        video_hidden = video_hidden + self.sync_weight * sync

        # Per-sample empty replacement (mixed batches).
        empty = self.empty_visual_feat.to(device).expand(batch, -1, -1)
        video_hidden = torch.where(is_zero.view(batch, 1, 1), empty, video_hidden)

        return video_hidden, torch.ones(batch, 1, device=device)


class AudioAutoencoderConditioner(Conditioner):
    """Mirrors AudioX's ``AudioAutoencoderConditioner``.

    Real audio prompts run through ``pretransform.encode`` (the Oobleck VAE
    encoder) and then ``proj_out`` to ``output_dim``. Zero audio (text-to-
    audio) returns the learned ``empty_audio_feat`` directly. Caller supplies
    the pretransform — for the text-to-audio demo any object with the right
    duck-typed shape works since the encode path never runs.

    Upstream's mask shape ``[B, output_dim]`` (rather than ``[B, S]``) is
    preserved verbatim — it's an upstream quirk but we never consume it as a
    sequence mask in the AudioX cross-attn path."""

    def __init__(self, pretransform, output_dim: int, empty_seq_len: int = 215):
        super().__init__(dim=pretransform.encoded_channels, output_dim=output_dim)
        self.pretransform = pretransform
        self.empty_audio_feat = nn.Parameter(torch.zeros(1, empty_seq_len, output_dim))

    def forward(self, audio: tp.Any, device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(audio, (list, tuple)):
            audio = torch.cat(audio, dim=0)
        audio = audio.to(device)
        is_zero = torch.all(audio == 0, dim=tuple(range(1, audio.ndim)))

        if bool(is_zero.all()):
            batch = audio.shape[0]
            latents = self.empty_audio_feat.to(device).expand(batch, -1, -1)
            return latents, torch.ones(batch, latents.shape[2], device=device)

        self.pretransform.to(device)
        self.proj_out.to(device)
        latents = self.pretransform.encode(audio).permute(0, 2, 1)
        latents = self.proj_out(latents)

        # Per-sample empty replacement only when the batch is mixed zero/non-zero.
        # The upstream config sets ``latent_seq_len=215`` to match the encoder's
        # output length, but only the where-path requires that match.
        if bool(is_zero.any()):
            empty = self.empty_audio_feat.to(device).expand(latents.shape[0], -1, -1)
            latents = torch.where(is_zero.view(-1, 1, 1), empty, latents)
        return latents, torch.ones(latents.shape[0], latents.shape[2], device=device)
