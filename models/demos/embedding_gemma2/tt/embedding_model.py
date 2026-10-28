# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""EmbeddingGemma2: public host API of the full multimodal model (input formatting, batching,
bucketing, upload, trace capture/replay, readback) around the device-only `EmbeddingGemma2Model`.

    model = EmbeddingGemma2.from_pretrained(device)
    emb = model.encode(["task: search result | query: northern lights",
                        {"image": pil_image},
                        {"text": "Someone says:", "audio": {"array": wave_16k, "sampling_rate": 16000}},
                        {"text": "A dashboard.", "video": {"array": frames, "video_metadata": {"fps": 1.0}}}])
    emb.shape  # [4, 768], unit L2 norm; truncate_dim=256 gives [4, 256] re-normalized (MRL)

Input formatting is the sentence-transformers 6.x path for this checkpoint (`modules.json` Transformer
with the "message" modality; `base/modality.py` `to_message`, `base/modules/transformer.py`
`preprocess` / `_apply_chat_template`): every input becomes one user message whose content lists
the dict's modalities in key order, a prompt becomes a system message, and the HF processor renders
it with `chat_template.jinja`, expands each placeholder to `<|image>` + n x `<|image|>` + `<image|>`
(video: one such block per frame; audio: `<|audio>` + n x `<|audio|>` + `<audio|>`) and tokenizes
with BOS/EOS. The token ids therefore equal the ids ST passes to HF (checked per item against an
ST dump in tests/pcc/test_full_model.py).

Host work (all before upload): the processor (tokenizer, image resize/patchify, video frame
patchify, audio log-mel), padding to the buckets, the per-tower inputs (VisionEncoder /
AudioEncoder.prepare_inputs), and the merge index `gather` built from the token ids. Everything after
upload, including both towers and the soft-token merge, runs on device in one forward
(`EmbeddingGemma2Model.forward`), eagerly or as one trace replay per `ShapeKey`.

Batching: items with the same media structure (number of images, video frames and audio clips, and
the audio token bucket) share one forward; text-only items batch up to `batch_size`. A batch runs at
the smallest text bucket >= its longest item (32 ... max_seq_len, power of two). Images run in the
2560-patch bucket, video frames in the 1280-patch bucket, audio clips in token buckets 32 ... 768.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.embedding_gemma2.tt.audio.audio_encoder import AudioEncoder
from models.demos.embedding_gemma2.tt.audio.audio_model import AudioDeviceInputs
from models.demos.embedding_gemma2.tt.config import bucket_for
from models.demos.embedding_gemma2.tt.model import EmbeddingGemma2Model, FullDeviceInputs, ShapeKey
from models.demos.embedding_gemma2.tt.text_encoder import TextEncoder
from models.demos.embedding_gemma2.tt.vision.vision_encoder import VisionEncoder
from models.demos.embedding_gemma2.tt.vision.vision_model import VisionDeviceInputs
from models.demos.embedding_gemma2.tt.weights import resolve_snapshot

MODALITIES = ("text", "image", "audio", "video")
MRL_DIMS = (768, 512, 256, 128)


# ---------------------------------------------------------------------- input formatting (ST)
def to_messages(item, prompt: str | None = None) -> tuple[list[dict], dict]:
    """One ST input -> (messages, extra processor kwargs). Mirrors ST `parse_inputs` + `to_message`
    + `prepend_prompt_to_messages` (structured format)."""
    from PIL.Image import Image as PILImage

    if isinstance(item, str):
        item = {"text": item}
    elif isinstance(item, PILImage):
        item = {"image": item}
    if not isinstance(item, dict) or not item:
        raise TypeError(f"an input is a str, a PIL image or a dict with keys in {MODALITIES}, got {type(item)}")
    extra = {"audio": {}, "video": {}}
    content = []
    for modality, value in item.items():
        if modality not in MODALITIES:
            raise KeyError(f"unknown modality {modality!r}; expected {MODALITIES}")
        if modality == "text":
            values = [value]
        else:
            values = list(value) if isinstance(value, (list, tuple)) else [value]
        for v in values:
            if modality == "audio" and isinstance(v, dict):
                if "sampling_rate" in v:
                    sr = extra["audio"].setdefault("sampling_rate", v["sampling_rate"])
                    if sr != v["sampling_rate"]:
                        raise ValueError("audio clips of one item must share a sampling rate")
                v = v["array"]
            elif modality == "video":
                meta = None
                if isinstance(v, dict):
                    meta = v.get("video_metadata")
                    v = v["array"]
                extra["video"].setdefault("video_metadata", []).append(meta)
            content.append({"type": modality, modality: v})
    messages = [{"role": "user", "content": content}]
    if prompt:
        messages = [{"role": "system", "content": [{"type": "text", "text": prompt}]}] + messages
    meta = extra["video"].get("video_metadata")
    if meta is not None and all(m is None for m in meta):
        del extra["video"]["video_metadata"]
    return messages, extra


def process_item(processor, item, prompt: str | None = None):
    """Run the HF processor exactly as ST 6.x does for one input. Returns the processor BatchFeature
    (input_ids [1, L], attention_mask, and pixel_values / pixel_values_videos / input_features ...)."""
    messages, extra = to_messages(item, prompt)
    return processor.apply_chat_template(
        [messages],
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={
            "text_kwargs": {"padding": True, "truncation": "longest_first"},
            "images_kwargs": {},
            "audio_kwargs": {"padding": True, **extra["audio"]},
            "videos_kwargs": dict(extra["video"]),
            "common_kwargs": {"return_tensors": "pt"},
        },
        load_audio_from_video=False,
    )


@dataclass
class ProcessedItem:
    features: dict  # processor output for this one item
    input_ids: torch.Tensor  # [L] int64
    num_images: int
    num_frames: int
    num_audio: int
    audio_tokens: int  # max valid audio tokens over the item's clips (0 if none)

    @property
    def length(self) -> int:
        return int(self.input_ids.numel())


@dataclass
class FullHostInputs:
    tensors: FullDeviceInputs  # host (not on device) ttnn tensors
    key: ShapeKey
    lengths: list[int] = field(default_factory=list)


@dataclass
class _Trace:
    trace_id: int
    inputs: FullDeviceInputs
    output: ttnn.Tensor


class EmbeddingGemma2(LightweightModule):
    def __init__(
        self,
        device,
        model: EmbeddingGemma2Model,
        text: TextEncoder,
        vision: VisionEncoder | None,
        audio: AudioEncoder | None,
        snapshot: Path,
        token_ids: dict[str, int],
    ):
        super().__init__()
        self.device = device
        self.model = model
        self.text = text
        self.vision = vision
        self.audio = audio
        self.snapshot = snapshot
        self.token_ids = token_ids
        self.max_seq_len = text.config.max_seq_len
        self.pad_token_id = text.config.pad_token_id
        self._processor = None
        self._prompts = None
        self._traces: dict[ShapeKey, _Trace] = {}
        self._compiled: set[ShapeKey] = set()
        self._prepared: set[ShapeKey] = set()

    @classmethod
    def from_pretrained(
        cls,
        device,
        model_id_or_path: str = "google/embeddinggemma-2",
        *,
        max_seq_len: int = 8192,
        vision: bool = True,
        audio: bool = True,
        cache_dir: Path | None = None,
        precision=None,
    ) -> "EmbeddingGemma2":
        """Load the text tower and (optionally) the vision and audio towers. Open the device with a
        trace region (the tests use 256 MiB for the full-model traces). `precision`: the precision
        policy of all three towers (None: the selected config, see tt/precision.py)."""
        import json

        from models.demos.embedding_gemma2.tt.precision import resolve_precision

        precision = resolve_precision(precision)
        snapshot = resolve_snapshot(model_id_or_path)
        text = TextEncoder.from_pretrained(
            device, str(snapshot), max_seq_len=max_seq_len, cache_dir=cache_dir, precision=precision
        )
        venc = (
            VisionEncoder.from_pretrained(device, str(snapshot), cache_dir=cache_dir, precision=precision)
            if vision
            else None
        )
        aenc = (
            AudioEncoder.from_pretrained(device, str(snapshot), cache_dir=cache_dir, precision=precision)
            if audio
            else None
        )
        cfg = json.loads((snapshot / "config.json").read_text())
        token_ids = {"image": cfg["image_token_id"], "video": cfg["video_token_id"], "audio": cfg["audio_token_id"]}
        model = EmbeddingGemma2Model(
            device, text.model, venc.model if venc is not None else None, aenc.model if aenc is not None else None
        )
        return cls(device, model, text, venc, aenc, snapshot, token_ids)

    # ---------------------------------------------------------------- host: formatting
    @property
    def processor(self):
        if self._processor is None:
            from transformers import AutoProcessor

            self._processor = AutoProcessor.from_pretrained(str(self.snapshot))
            if self._processor.video_token_id != self.token_ids["video"]:
                raise RuntimeError("processor <|video|> id differs from config.video_token_id")
        return self._processor

    @property
    def prompts(self) -> dict[str, str]:
        """`config_sentence_transformers.json` prompts (prompt_name -> prefix)."""
        if self._prompts is None:
            import json

            self._prompts = json.loads((self.snapshot / "config_sentence_transformers.json").read_text())["prompts"]
        return self._prompts

    def process(self, item, prompt: str | None = None) -> ProcessedItem:
        f = process_item(self.processor, item, prompt)
        ids = f["input_ids"][0]
        if "attention_mask" in f and not bool(f["attention_mask"].all()):
            raise ValueError("a single processed item has padding")
        n_img = int(f["pixel_values"].shape[0]) if "pixel_values" in f else 0
        n_frames = int(f["pixel_values_videos"].shape[0]) if "pixel_values_videos" in f else 0
        n_audio, a_tok = 0, 0
        if "input_features" in f:
            if self.audio is None:
                raise ValueError("audio input but the audio tower is not loaded")
            n_audio = int(f["input_features"].shape[0])
            m = f["input_features_mask"].bool()
            a_tok = int(m[:, ::2][:, ::2].sum(-1).max())
        if (n_img or n_frames) and self.vision is None:
            raise ValueError("image/video input but the vision tower is not loaded")
        if ids.numel() > self.max_seq_len:
            raise ValueError(f"item has {ids.numel()} tokens > max_seq_len {self.max_seq_len}")
        return ProcessedItem(dict(f), ids.long(), n_img, n_frames, n_audio, a_tok)

    # ---------------------------------------------------------------- host: batch inputs
    def batch_key(self, items: list[ProcessedItem], truncate_dim: int | None = None) -> ShapeKey:
        from models.demos.embedding_gemma2.tt.audio.config import token_bucket_for
        from models.demos.embedding_gemma2.tt.vision.config import patch_bucket_for

        B = len(items)
        S = bucket_for(max(it.length for it in items), self.max_seq_len)
        n_img = sum(it.num_images for it in items)
        n_frames = sum(it.num_frames for it in items)
        n_audio = sum(it.num_audio for it in items)
        image = (n_img, patch_bucket_for(items[0].features["pixel_values"].shape[1])) if n_img else None
        video = (n_frames, patch_bucket_for(items[0].features["pixel_values_videos"].shape[1])) if n_frames else None
        audio = (n_audio, token_bucket_for(max(it.audio_tokens for it in items))) if n_audio else None
        td = None if truncate_dim in (None, 768) else int(truncate_dim)
        return ShapeKey(B, S, image, video, audio, td)

    def prepare_batch(self, items: list[ProcessedItem], truncate_dim: int | None = None) -> FullHostInputs:
        """Processed items -> host ttnn tensors for one forward (no device work)."""
        key = self.batch_key(items, truncate_dim)
        B, S = key.batch, key.seq_len
        ids = torch.full((B, S), self.pad_token_id, dtype=torch.int64)
        keep = torch.zeros((B, 1, 1, S), dtype=torch.float32)
        for b, it in enumerate(items):
            ids[b, : it.length] = it.input_ids
            keep[b, 0, 0, : it.length] = 1.0

        def cat(name):
            return torch.cat([it.features[name] for it in items if name in it.features], dim=0)

        image = video = audio = None
        img_counts = vid_counts = aud_counts = None
        if key.image is not None:
            h = self.vision.prepare_inputs(cat("pixel_values").float(), cat("image_position_ids").long())
            image, img_counts = h.tensors, h.num_soft_tokens
            assert (h.batch, h.seq_len) == key.image
        if key.video is not None:
            h = self.vision.prepare_inputs(cat("pixel_values_videos").float(), cat("video_position_ids").long())
            video, vid_counts = h.tensors, h.num_soft_tokens
            assert (h.batch, h.seq_len) == key.video
        if key.audio is not None:
            feats = [it.features["input_features"] for it in items if "input_features" in it.features]
            masks = [it.features["input_features_mask"] for it in items if "input_features" in it.features]
            T = max(f.shape[1] for f in feats)
            F = feats[0].shape[2]
            fpad = torch.cat([torch.nn.functional.pad(f.float(), (0, 0, 0, T - f.shape[1])) for f in feats])
            mpad = torch.cat([torch.nn.functional.pad(m.bool(), (0, T - m.shape[1])) for m in masks])
            h = self.audio.prepare_inputs(fpad.reshape(-1, T, F), mpad)
            if h.seq_len < key.audio[1]:
                raise AssertionError("audio bucket mismatch")
            audio, aud_counts = h.tensors, h.num_tokens
            key = ShapeKey(key.batch, key.seq_len, key.image, key.video, (h.batch, h.seq_len), key.truncate_dim)

        gather = None
        if key.has_media:
            gather = self.gather_index(ids, key, img_counts, vid_counts, aud_counts)
        for tok in self.token_ids.values():
            ids[ids == tok] = self.pad_token_id  # HF: placeholder ids are looked up as pad (rows replaced)

        def rm_u32(t):
            return ttnn.from_torch(t.to(torch.int32).contiguous(), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)

        tensors = FullDeviceInputs(
            input_ids=rm_u32(ids),
            keep=ttnn.from_torch(keep, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
            gather=rm_u32(gather) if gather is not None else None,
            image=image,
            video=video,
            audio=audio,
        )
        return FullHostInputs(tensors, key, [it.length for it in items])

    def gather_index(self, ids: torch.Tensor, key: ShapeKey, img_counts, vid_counts, aud_counts) -> torch.Tensor:
        """[B, S] table row per position: b*S+s for text/padding, the soft-token row for placeholders.

        Table rows: [text B*S | image soft Bi*Si_out | video soft F*Sv_out | audio soft Ba*Sa].
        Placeholders are filled in flattened (row-major) order, as HF `masked_scatter` does.
        """
        B, S = key.batch, key.seq_len
        gather = torch.arange(B * S, dtype=torch.int64).reshape(B, S)
        offset = B * S
        sources = []
        if key.image is not None:
            rows = self.vision.config.pooled_rows(key.image[1])
            sources.append(("image", offset, rows, img_counts))
            offset += key.image[0] * rows
        if key.video is not None:
            rows = self.vision.config.pooled_rows(key.video[1])
            sources.append(("video", offset, rows, vid_counts))
            offset += key.video[0] * rows
        if key.audio is not None:
            rows = key.audio[1]
            sources.append(("audio", offset, rows, aud_counts))
            offset += key.audio[0] * rows
        for name, base, rows, counts in sources:
            src = torch.cat([base + j * rows + torch.arange(n) for j, n in enumerate(counts)])
            pos = (ids == self.token_ids[name]).reshape(-1).nonzero().flatten()
            if pos.numel() != src.numel():
                raise ValueError(f"{name}: {pos.numel()} placeholder tokens but {src.numel()} soft tokens")
            gather.view(-1)[pos] = src
        return gather

    # ---------------------------------------------------------------- device
    def to_device(self, host: FullHostInputs) -> FullDeviceInputs:
        dram = ttnn.DRAM_MEMORY_CONFIG

        def up(t):
            return ttnn.to_device(t, self.device, memory_config=dram) if t is not None else None

        x = host.tensors
        return FullDeviceInputs(
            input_ids=up(x.input_ids),
            keep=up(x.keep),
            gather=up(x.gather),
            image=VisionDeviceInputs(*(up(t) for t in x.image.tensors())) if x.image is not None else None,
            video=VisionDeviceInputs(*(up(t) for t in x.video.tensors())) if x.video is not None else None,
            audio=AudioDeviceInputs(*(up(t) for t in x.audio.tensors())) if x.audio is not None else None,
        )

    def prepare(self, key: ShapeKey, x: FullDeviceInputs) -> None:
        if key in self._prepared:
            return
        if self._traces:
            raise RuntimeError(f"shape {key} was not prepared before the traces were captured; use capture_all()")
        self.model.prepare(key, x)
        self._prepared.add(key)

    def forward(self, x: FullDeviceInputs, key: ShapeKey) -> ttnn.Tensor:
        """Device-only forward on already uploaded inputs (the measured unit)."""
        return self.model(x, truncate_dim=key.truncate_dim)

    @staticmethod
    def read_embedding(emb: ttnn.Tensor, batch: int) -> torch.Tensor:
        return ttnn.to_torch(emb).float().reshape(batch, -1)

    def run(self, host: FullHostInputs, *, traced: bool = False) -> torch.Tensor:
        """One forward for prepared host inputs; returns [B, d] float32 normalized embeddings."""
        key = host.key
        if traced:
            out = self.replay(host)
        else:
            if self._traces and key not in self._compiled:
                raise RuntimeError(
                    f"eager shape {key} was not compiled before the traces were captured; its new "
                    "program-cache buffers could overlap trace buffers. Include it in capture_all()."
                )
            x = self.to_device(host)
            self.prepare(key, x)
            out = self.forward(x, key)
            self._compiled.add(key)
        emb = self.read_embedding(out, key.batch)
        if not traced:
            ttnn.deallocate(out)
        return emb

    # ---------------------------------------------------------------- public API
    def plan(self, items, *, prompt: str | None = None, prompt_name: str | None = None, batch_size: int = 8):
        """Process the inputs and group them into batches. Returns (processed items, list of index lists)."""
        if prompt_name is not None:
            if prompt is not None:
                raise ValueError("pass prompt or prompt_name, not both")
            prompt = self.prompts[prompt_name]
        if not isinstance(items, (list, tuple)):
            items = [items]
        processed = [self.process(it, prompt) for it in items]
        groups: dict[tuple, list[int]] = {}
        for i, p in enumerate(processed):
            sig = (p.num_images, p.num_frames, p.num_audio, self.batch_key([p]).audio)
            groups.setdefault(sig, []).append(i)
        batches = []
        for idx in groups.values():
            # sort by length inside a group so a batch's bucket is as small as possible
            idx = sorted(idx, key=lambda i: processed[i].length)
            batches.extend(idx[j : j + batch_size] for j in range(0, len(idx), batch_size))
        return processed, batches

    def encode(
        self,
        items,
        *,
        prompt: str | None = None,
        prompt_name: str | None = None,
        batch_size: int = 8,
        truncate_dim: int | None = None,
        traced: bool = False,
    ) -> torch.Tensor:
        """Normalized embeddings [N, d] (d = truncate_dim or 768) for a list of inputs.

        An input is a str (text), a PIL image, or a dict with keys "text", "image", "audio", "video"
        (sentence-transformers format; values may be lists). Audio: {"array": wave, "sampling_rate":
        16000} or a 16 kHz array. Video: {"array": frames [F,H,W,3] uint8, "video_metadata": {"fps":
        ...}}. `prompt` / `prompt_name` (config_sentence_transformers.json) prefix every input.
        truncate_dim in (512, 256, 128): MRL truncation + re-normalization on device.
        traced=True replays captured traces (`capture_all`) and fails for an uncaptured shape.
        """
        if truncate_dim is not None and truncate_dim not in MRL_DIMS:
            raise ValueError(f"truncate_dim must be one of {MRL_DIMS}")
        processed, batches = self.plan(items, prompt=prompt, prompt_name=prompt_name, batch_size=batch_size)
        d = truncate_dim or 768
        out = torch.zeros((len(processed), d), dtype=torch.float32)
        for idx in batches:
            host = self.prepare_batch([processed[i] for i in idx], truncate_dim)
            out[idx] = self.run(host, traced=traced)
        return out

    def shape_keys(self, items, *, prompt=None, prompt_name=None, batch_size: int = 8, truncate_dim=None):
        """The ShapeKeys `encode` would run for these inputs (pass them to `capture_all`)."""
        processed, batches = self.plan(items, prompt=prompt, prompt_name=prompt_name, batch_size=batch_size)
        return sorted({self.prepare_batch([processed[i] for i in idx], truncate_dim).key for idx in batches}, key=str)

    # ---------------------------------------------------------------- trace
    def dummy_host(self, key: ShapeKey) -> FullHostInputs:
        """Valid inputs of shape `key` for allocation / compile / capture (text rows read only text)."""
        B, S = key.batch, key.seq_len

        def rm_u32(t):
            return ttnn.from_torch(t.to(torch.int32).contiguous(), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)

        ids = torch.full((B, S), 2, dtype=torch.int64)
        keep = torch.ones((B, 1, 1, S), dtype=torch.float32)
        image = self.vision._dummy_host(*key.image).tensors if key.image is not None else None
        video = self.vision._dummy_host(*key.video).tensors if key.video is not None else None
        audio = self.audio._dummy_host(*key.audio).tensors if key.audio is not None else None
        gather = torch.arange(B * S).reshape(B, S) if key.has_media else None
        tensors = FullDeviceInputs(
            input_ids=rm_u32(ids),
            keep=ttnn.from_torch(keep, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
            gather=rm_u32(gather) if gather is not None else None,
            image=image,
            video=video,
            audio=audio,
        )
        return FullHostInputs(tensors, key, [S] * B)

    def capture_all(self, keys) -> None:
        """Capture one trace per ShapeKey, in three phases (as Text/Vision/AudioEncoder.capture_all).

        Any device allocation that outlives a capture and is made after it can land in that trace's
        buffers and be overwritten on replay (program-cache buffers of a newly compiled program
        included). So, before the first capture: (1) allocate every persistent input and run the
        per-shape setup (text band masks, audio conv weight layout), (2) compile every shape with
        one eager pass. Only then capture. A trace output is rewritten by its own replay; read it
        before replaying another trace. Call once per session; eager runs afterwards are allowed only
        for the shapes listed here.
        """
        if self._traces:
            raise RuntimeError("capture_all() may run once, before any other trace is captured")
        keys = sorted(set(keys), key=str)
        inputs = {k: self.to_device(self.dummy_host(k)) for k in keys}
        for k in keys:
            self.prepare(k, inputs[k])
        for k in keys:
            out = self.forward(inputs[k], k)
            ttnn.synchronize_device(self.device)
            ttnn.deallocate(out)
            self._compiled.add(k)
        from ttnn.tools.trace_allocation_tracker import acknowledge_corruptible

        for k in keys:
            trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
            out = self.forward(inputs[k], k)
            ttnn.end_trace_capture(self.device, trace_id, cq_id=0)
            ttnn.synchronize_device(self.device)
            acknowledge_corruptible(out)
            self._traces[k] = _Trace(trace_id, inputs[k], out)

    @property
    def captured(self) -> list[ShapeKey]:
        return sorted(self._traces, key=str)

    def replay(self, host: FullHostInputs, *, blocking: bool = False) -> ttnn.Tensor:
        if host.key not in self._traces:
            raise KeyError(f"no trace for {host.key}; captured: {[str(k) for k in self.captured]}")
        tr = self._traces[host.key]
        for src, dst in zip(host.tensors.tensors(), tr.inputs.tensors()):
            ttnn.copy_host_to_device_tensor(src, dst)
        ttnn.execute_trace(self.device, tr.trace_id, cq_id=0, blocking=blocking)
        return tr.output

    def execute_captured(self, key: ShapeKey, *, blocking: bool = False) -> ttnn.Tensor:
        """Replay a captured trace on whatever its persistent inputs hold (perf loop helper)."""
        tr = self._traces[key]
        ttnn.execute_trace(self.device, tr.trace_id, cq_id=0, blocking=blocking)
        return tr.output

    def release_traces(self) -> None:
        for tr in self._traces.values():
            ttnn.release_trace(self.device, tr.trace_id)
        self._traces.clear()
