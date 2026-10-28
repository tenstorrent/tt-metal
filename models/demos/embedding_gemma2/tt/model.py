# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""EmbeddingGemma2Model: device forward of the full multimodal model (HF `EmbeddingGemma2Model` +
sentence-transformers Pooling(mean) + Normalize, optional MRL truncation).

    text   = embed_tokens(ids) * sqrt(512)                          [B,1,S,512]  (placeholders -> pad id)
    soft_i = VisionModel(image inputs)                              [Bi,1,Si,512]
    soft_v = VisionModel(video-frame inputs)                        [F,1,Sv,512]
    soft_a = AudioModel(audio inputs)                               [Ba,1,Sa,512]
    table  = concat(rows(text), rows(soft_i), rows(soft_v), rows(soft_a))      [R, 512] (ROW_MAJOR)
    embeds = embedding(gather, table)                               [B,1,S,512]  (the HF masked_scatter)
    ple    = PLE(embeds);  h = 24 encoder layers(embeds, ple, masks(keep))
    last   = norm(h) @ W_proj                                        [B,1,S,768]
    emb    = L2normalize(masked_mean(last, keep))                    [B,1,1,768]
    emb_d  = L2normalize(emb[..., :d])                               (truncate_dim d, optional)

The merge is one `ttnn.embedding` lookup into a table that is built on device from the text
embeddings and the tower outputs. `gather[b, s]` (host-built from the token ids, uploaded with the
other inputs) is `b*S + s` for a text or padding position, and the table row of the matching soft
token for an `<|image|>`, `<|video|>` or `<|audio|>` position. HF semantics
(modeling_embedding_gemma2.py, transformers 5.20.2): placeholder ids are looked up as the pad id
(:778-779), the soft tokens replace those rows after the sqrt(512) scale (:795, :814, :835,
masked_scatter in flattened batch order), and PLE runs on the merged embeddings (:527). Masks and
positions do not depend on the modality (:530, :540-541). doc/full_model/README.md has the details.

Contract (forward)
    `FullDeviceInputs` built by `EmbeddingGemma2.prepare_batch` and uploaded (see embedding_model.py):
        input_ids [B,S] uint32 RM, keep [B,1,1,S] bf16 TILE, gather [B,S] uint32 RM (None: text only),
        image / video: `VisionDeviceInputs` (or None), audio: `AudioDeviceInputs` (or None).
    S is a text bucket prepared with `prepare(key)`; audio shapes must be prepared the same way.
    Returns the normalized embedding [B,1,1,d] fp32 (d = truncate_dim or 768). forward() issues
    device ops only (no host round trip), so the whole multimodal forward can be captured in one trace.
"""

from __future__ import annotations

from dataclasses import dataclass

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.embedding_gemma2.tt.audio.audio_model import AudioDeviceInputs, AudioModel
from models.demos.embedding_gemma2.tt.pooler import l2_normalize
from models.demos.embedding_gemma2.tt.text_model import TextModel
from models.demos.embedding_gemma2.tt.vision.vision_model import VisionDeviceInputs, VisionModel


@dataclass(frozen=True)
class ShapeKey:
    """Physical shape of one forward; one trace per key.

    batch, seq_len: text rows and text bucket. image / video / audio: (count, bucket) of the tower
    batch (images, video frames, audio clips) or None. truncate_dim: MRL dim or None (768).
    """

    batch: int
    seq_len: int
    image: tuple[int, int] | None = None
    video: tuple[int, int] | None = None
    audio: tuple[int, int] | None = None
    truncate_dim: int | None = None

    @property
    def has_media(self) -> bool:
        return self.image is not None or self.video is not None or self.audio is not None

    def __str__(self) -> str:
        parts = [f"b{self.batch}_s{self.seq_len}"]
        for name in ("image", "video", "audio"):
            v = getattr(self, name)
            if v is not None:
                parts.append(f"{name}{v[0]}x{v[1]}")
        if self.truncate_dim is not None:
            parts.append(f"d{self.truncate_dim}")
        return "_".join(parts)


@dataclass
class FullDeviceInputs:
    input_ids: ttnn.Tensor
    keep: ttnn.Tensor
    gather: ttnn.Tensor | None = None
    image: VisionDeviceInputs | None = None
    video: VisionDeviceInputs | None = None
    audio: AudioDeviceInputs | None = None

    def tensors(self) -> tuple[ttnn.Tensor, ...]:
        """Every persistent input tensor, in a fixed order (trace refresh)."""
        out = [self.input_ids, self.keep]
        if self.gather is not None:
            out.append(self.gather)
        for m in (self.image, self.video, self.audio):
            if m is not None:
                out.extend(m.tensors())
        return tuple(out)


class EmbeddingGemma2Model(LightweightModule):
    def __init__(
        self,
        device,
        text: TextModel,
        vision: VisionModel | None = None,
        audio: AudioModel | None = None,
    ):
        super().__init__()
        self.device = device
        self.text = text
        self.vision = vision
        self.audio = audio
        self.hidden_size = text.config.hidden_size

    # ---------------------------------------------------------------- setup
    def prepare(self, key: ShapeKey, inputs: FullDeviceInputs) -> None:
        """Setup-time per-shape work (text band mask upload, audio conv weight layout)."""
        self.text.prepare(key.seq_len)
        if key.audio is not None:
            if self.audio is None or inputs.audio is None:
                raise ValueError("audio shape without an audio tower / audio inputs")
            self.audio.prepare(inputs.audio)

    # ---------------------------------------------------------------- forward
    def _rows(self, t: ttnn.Tensor) -> ttnn.Tensor:
        B, _, S, H = t.shape
        return ttnn.reshape(t, (1, 1, B * S, H))

    def merge(self, x: FullDeviceInputs) -> ttnn.Tensor:
        """Text embeddings with the tower soft tokens placed into their placeholder rows. [B,1,S,512] bf16."""
        text = self.text.embedding(x.input_ids)  # [B,1,S,H]
        if x.gather is None:
            return text
        B, _, S, H = text.shape
        parts = [self._rows(text)]
        owned = [text]
        for inputs, tower in ((x.image, self.vision), (x.video, self.vision), (x.audio, self.audio)):
            if inputs is None:
                continue
            if tower is None:
                raise ValueError("media inputs given but the matching tower is not loaded")
            soft, aux = tower(inputs)
            ttnn.deallocate(aux)
            parts.append(self._rows(soft))
            owned.append(soft)
        table = ttnn.concat(parts, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1,1,R,H] TILE
        for t in owned:
            ttnn.deallocate(t)
        table_rm = ttnn.to_layout(table, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(table)
        R = table_rm.shape[2]
        table_rm = ttnn.reshape(table_rm, (R, H))
        merged = ttnn.embedding(
            x.gather,
            table_rm,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(table_rm)
        return ttnn.reshape(merged, (B, 1, S, H))

    def encoder(self, embeds: ttnn.Tensor, keep: ttnn.Tensor) -> ttnn.Tensor:
        """PLE + 24 encoder layers from merged input embeddings (TextModel.forward_hidden after the lookup)."""
        tm = self.text
        return tm.encode_embeds(embeds, tm.masks(keep))

    def forward(self, x: FullDeviceInputs, truncate_dim: int | None = None) -> ttnn.Tensor:
        embeds = self.merge(x)
        h = self.encoder(embeds, x.keep)
        last = self.text.head(h)
        ttnn.deallocate(h)
        emb = self.text.pooler(last, x.keep)  # [B,1,1,768] fp32, unit norm
        ttnn.deallocate(last)
        if truncate_dim is None or truncate_dim == emb.shape[-1]:
            return emb
        B = emb.shape[0]
        cut = ttnn.slice(emb, (0, 0, 0, 0), (B, 1, 1, truncate_dim))
        ttnn.deallocate(emb)
        out = l2_normalize(cut)
        ttnn.deallocate(cut)
        return out
