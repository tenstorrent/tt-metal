# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The full pplx-decider text model on one device: embedding -> 64 decoder layers -> decision head.

Reproduces ``DecisionModel.predict`` (snapshot ``source/src/autojev/model.py:202-224``) at batch 1:
``Qwen3_5Model(...).last_hidden_state[:, -1]`` -> ``readout`` -> mask by option count ->
``/ temperature`` -> softmax.

Length contract (person decision, see ``doc/context_contract.json``): the prompt is right-padded to
the next bucket of 128 / 1024 / 2048 / 4096 / 8192 and the head reads the hidden state of the last
REAL token. Both mixers are causal, so the padding cannot change any real token's hidden state.
Prompts longer than 8192 tokens are rejected, as the app does.

All 64 layers are resident in device DRAM at once. Weights are converted from the snapshot one
layer at a time (the host never holds the 54 GB checkpoint) and persisted with the ttnn disk
weight cache (``LazyWeight.cache_dir_weight_name``). A cached load builds the weight bundles from
``meta`` tensors (shapes only, read from the safetensors headers), so it reads no HF weights.

Between the token upload and the probability readback the forward issues TTNN device ops only.

Images (``vision=True``): the 12A vision tower (``tt/vision``, BF16) is resident next to the text
stack. ``prepare_images`` is per-request input prep on host - 3D position ids
(``rope.get_rope_index``) -> per-request cos/sin (``PplxRequestRotary``), the splice index list,
and the vision tower inputs - and uploads them. On device, ``embed_with_images`` gathers the
embedding rows into ``[text embeddings ; image features]`` with one ``ttnn.embedding`` over the
splice index (HF ``inputs_embeds.masked_scatter(input_ids == image_token_id, image_features)``),
so the 5120-wide embeddings never leave the device. Then the same 64 layers and head run, with
the request rotary in the 16 full-attention layers. Text-only requests take the unchanged path.

Usage::

    model = PplxDeciderModel.from_snapshot(device)
    tokens, last_index = model.upload_tokens(input_ids)       # host -> device, padded to the bucket
    probs, logits, _ = model(tokens, last_index, count)        # device tensors [1, 1, 256] fp32

    model = PplxDeciderModel.from_snapshot(device, vision=True)
    tokens, last_index, images = model.prepare_images(processor_outputs)   # input prep + upload
    probs, logits, _ = model(tokens, last_index, count, images=images)
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from loguru import logger
from safetensors import safe_open

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.modules.lazy_weight import LazyWeight
from models.demos.pplx_decider_v1_27b.tt.decoder import DecoderLayerConfig, PplxDecoderLayer
from models.demos.pplx_decider_v1_27b.tt.embedding import EmbeddingConfig, PplxEmbedding
from models.demos.pplx_decider_v1_27b.tt.head import PADDED_OPTIONS, DecisionHeadConfig, PplxDecisionHead
from models.demos.pplx_decider_v1_27b.tt.model_config import APP_MAX_LENGTH, PplxDeciderArgs
from models.demos.pplx_decider_v1_27b.tt.optimizations import Optimizations, PrecisionPolicy
from models.demos.pplx_decider_v1_27b.tt.rope import PplxRequestRotary, PplxRotary, get_rope_index
from models.demos.pplx_decider_v1_27b.tt.weight_adapter import (
    build_decoder_layer_weights,
    build_embedding_weight,
    build_readout_weight,
    zero_centred_norm,
)

BUCKETS = (128, 1024, 2048, 4096, 8192)
DEFAULT_CACHE_ROOT = Path(
    os.environ.get("PPLX_DECIDER_WEIGHT_CACHE", "/local/ttuser/gtobar/artifacts/pplx_decider/weight_cache")
)
TEXT_PREFIX = "language_model."
IMAGE_TOKEN_ID = 248056  # config.image_token_id (<|image_pad|>), checked against the config at load
_ST_DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}


def bucket_for(length: int) -> int:
    """Smallest prefill bucket holding ``length`` tokens; longer prompts are rejected like the app."""
    if length < 1:
        raise ValueError("A prompt needs at least one token")
    for bucket in BUCKETS:
        if length <= bucket:
            return bucket
    raise ValueError(f"Question branch exceeds the {APP_MAX_LENGTH}-token limit; no input was truncated.")


def dram_view(device) -> dict:
    """Allocated / free device DRAM from the allocator (all banks)."""
    view = ttnn.get_memory_view(device, ttnn.BufferType.DRAM)
    banks = view.num_banks
    return dict(
        total_gib=banks * view.total_bytes_per_bank / 2**30,
        allocated_gib=banks * view.total_bytes_allocated_per_bank / 2**30,
        free_gib=banks * view.total_bytes_free_per_bank / 2**30,
        largest_free_per_bank_mib=view.largest_contiguous_bytes_free_per_bank / 2**20,
    )


# ----------------------------------------------------------------------------------------------
# LazyWeight bookkeeping: cache names, cache-hit check, dropping host sources after upload
# ----------------------------------------------------------------------------------------------


@dataclass
class CachedWeight(LazyWeight):
    """``LazyWeight`` whose cache file name does not depend on the process-local device id.

    ``LazyWeight`` puts ``device.id()`` in the fingerprint. That id is a per-process counter (the
    first device opened in a process is 1, a second open in the same process is 2), so the same
    weight cached by one process missed in the next device open (measured: 0/66 hits on a second
    open in one process). The cache file holds the unsharded host tensor of a replicated weight,
    which is device independent, so the id is dropped from the name.
    """

    def _get_fingerprint(self) -> str:
        return re.sub(r"_device_[^_]+$", "", super()._get_fingerprint())


def _bind(obj, device, cache_dir: Path | None, name: str):
    """Copy of a weight bundle (dataclasses / tuples of LazyWeight) bound to ``device``, with a
    cache name per weight when ``cache_dir`` is set."""
    if isinstance(obj, LazyWeight):
        cache = (cache_dir, name) if cache_dir is not None else None
        values = {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj) if f.init}
        return CachedWeight(**{**values, "device": device, "cache_dir_weight_name": cache})
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        changes = {
            f.name: _bind(getattr(obj, f.name), device, cache_dir, f"{name}.{f.name}") for f in dataclasses.fields(obj)
        }
        return dataclasses.replace(obj, **changes)
    if isinstance(obj, tuple):
        return tuple(_bind(v, device, cache_dir, f"{name}.{i}") for i, v in enumerate(obj))
    return obj


def _lazy_weights(root) -> list[LazyWeight]:
    """Every LazyWeight reachable from a module tree (module attributes, config dataclasses, containers)."""
    found, seen, stack = [], set(), [root]
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, LazyWeight):
            found.append(obj)
        elif isinstance(obj, (list, tuple)):
            stack.extend(obj)
        elif isinstance(obj, dict):
            stack.extend(obj.values())
        elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            stack.extend(getattr(obj, f.name) for f in dataclasses.fields(obj))
        elif isinstance(obj, LightweightModule):
            stack.extend(vars(obj).values())
    return found


def _device_weights(root) -> list[LazyWeight]:
    """The resolved (device-bound) LazyWeights of a module tree: the ones that get uploaded."""
    return [w for w in _lazy_weights(root) if None not in (w.device, w.layout, w.memory_config)]


def _all_cached(root) -> bool:
    weights = _device_weights(root)
    return bool(weights) and all(
        w.cache_dir_weight_name is not None and w._get_cache_fill_path(*w.cache_dir_weight_name).exists()
        for w in weights
    )


def _release_sources(root) -> None:
    """Swap every host source tensor for a shape-only meta tensor once the weights are on device."""
    for w in _lazy_weights(root):
        if isinstance(w.source, torch.Tensor) and w.source.device.type != "meta":
            w.source = torch.empty(w.source.shape, dtype=w.source.dtype, device="meta")


def _meta_tensors(path: Path, keys: list[str], strip: str) -> dict[str, torch.Tensor]:
    """Shape/dtype-only tensors from safetensors headers: no weight bytes are read."""
    out = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in keys:
            sl = f.get_slice(key)
            out[key[len(strip) :]] = torch.empty(sl.get_shape(), dtype=_ST_DTYPES[sl.get_dtype()], device="meta")
    return out


class _SnapshotWeights:
    """Real or meta (shape-only) tensors from the snapshot, by group."""

    def __init__(self, reader):
        self.reader = reader

    def _meta_prefix(self, prefix: str) -> dict[str, torch.Tensor]:
        by_shard: dict[str, list[str]] = {}
        for key, shard in self.reader.weight_map.items():
            if key.startswith(prefix):
                by_shard.setdefault(shard, []).append(key)
        out = {}
        for shard, keys in by_shard.items():
            out.update(_meta_tensors(self.reader.path / shard, keys, prefix))
        return out

    def layer(self, idx: int, meta: bool) -> dict[str, torch.Tensor]:
        prefix = f"{TEXT_PREFIX}layers.{idx}."
        return self._meta_prefix(prefix) if meta else self.reader.layer_state_dict(idx)

    def embedding(self, meta: bool) -> torch.Tensor:
        return self._meta_prefix(TEXT_PREFIX + "embed_tokens.")["weight"] if meta else self.reader.embedding_weight()

    def final_norm(self, meta: bool) -> torch.Tensor:
        return self._meta_prefix(TEXT_PREFIX + "norm.")["weight"] if meta else self.reader.final_norm_weight()

    def readout(self, meta: bool) -> torch.Tensor:
        if meta:
            return _meta_tensors(self.reader.path / "readout.safetensors", ["weight"], "")["weight"]
        return self.reader.readout_weight()


# ----------------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------------


@dataclass
class LoadReport:
    """Setup measurements recorded while building the model (seconds, GiB)."""

    seconds: dict = field(default_factory=dict)
    cache_hits: dict = field(default_factory=dict)
    dram: dict = field(default_factory=dict)
    layer_weight_gib: dict = field(default_factory=dict)  # kind -> list of per-layer DRAM deltas

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class ModelConfig:
    args: PplxDeciderArgs
    optimizations: Optimizations
    temperature: float
    pad_token_id: int
    layer_ids: tuple[int, ...]
    cache_dir: Path | None


@dataclass
class ImageInputs:
    """Device inputs of the image part of one request (built by ``PplxDeciderModel.prepare_images``).

    ``splice_index[0, s]`` is the row of ``[text embeddings (bucket rows) ; image features]`` that
    token ``s`` takes: ``s`` for text and padding tokens, ``bucket + k`` for the k-th image token.
    """

    vision: list  # VisionInputs per image, prompt order
    splice_index: ttnn.Tensor  # [1, bucket] uint32 ROW_MAJOR
    rotary: PplxRequestRotary  # 3D mRoPE cos/sin, [1, 1, bucket, 256] each
    position_ids: torch.Tensor  # host [3, real_len] int64 (HF get_rope_index), for tests / logs
    num_image_tokens: int

    def deallocate(self) -> None:
        for v in self.vision:
            v.deallocate()
        ttnn.deallocate(self.splice_index)
        self.rotary.deallocate()


class PplxDeciderModel(LightweightModule):
    """Embedding + decoder stack + decision head (+ optional vision tower), all resident on one device."""

    def __init__(
        self, config: ModelConfig, embedding, layers, rotary, head, load_report: LoadReport | None = None, vision=None
    ):
        super().__init__()
        self.config = config
        self.mesh_device = config.optimizations.mesh_device
        self.embedding = embedding
        self.layers = layers
        self.rotary = rotary
        self.head = head
        self.vision = vision
        self.load_report = load_report or LoadReport()

    # -- construction ---------------------------------------------------------------------------
    @classmethod
    def from_snapshot(
        cls,
        mesh_device,
        reader=None,
        *,
        policy: PrecisionPolicy | None = None,
        layer_ids=None,
        cache_dir: Path | str | None = "default",
        prefill_chunk: int = 2048,
        vision: bool = False,
    ) -> "PplxDeciderModel":
        """Build and upload every weight. ``layer_ids`` (default: all 64) allows a reduced stack for debugging.

        ``vision=True`` also loads the BF16 vision tower (``PplxVisionTower``, ~0.96 GiB) for image requests.

        ``policy`` defaults to ``PrecisionPolicy.default()`` (the stage-8 selected precision config).
        ``cache_dir="default"`` uses ``DEFAULT_CACHE_ROOT/<revision>/weights``, shared by every policy:
        each cache file name carries the weight's dtype, shape and layout (``LazyWeight`` fingerprint),
        so a policy reuses the files of the groups whose dtype it shares. ``None`` disables the cache.
        """
        from models.demos.pplx_decider_v1_27b.reference.hf_reference import REVISION, SnapshotReader

        reader = reader or SnapshotReader()
        args = PplxDeciderArgs.from_hf_config(reader.text_config)
        opts = Optimizations.build(
            mesh_device,
            policy=policy or PrecisionPolicy.default(),
            max_seq_len=args.max_seq_len,
            prefill_chunk=prefill_chunk,
        )
        logger.info(f"precision policy {opts.policy.name}: {json.dumps(opts.policy.describe())}")
        if cache_dir == "default":
            cache_dir = DEFAULT_CACHE_ROOT / REVISION[:12] / "weights"
        cache_dir = Path(cache_dir) if cache_dir is not None else None
        layer_ids = tuple(range(args.num_hidden_layers) if layer_ids is None else layer_ids)
        source = _SnapshotWeights(reader)
        report = LoadReport()
        t_start = time.perf_counter()
        report.dram["empty"] = dram_view(mesh_device)

        def build(name, make):
            """Build with meta sources; if any weight misses the cache, rebuild from the real tensors."""
            start = time.perf_counter()
            module = make(meta=cache_dir is not None)
            hit = cache_dir is not None and _all_cached(module)
            if not hit:
                module = make(meta=False)
            for sub in _loadable(module):
                sub.load_device_weights()
            _release_sources(module)
            report.cache_hits[name] = hit
            return module, time.perf_counter() - start

        def cached(bundle, name):
            return _bind(bundle, mesh_device, cache_dir, name)

        policy_ = opts.policy
        embedding, report.seconds["embedding"] = build(
            "embedding",
            lambda meta: PplxEmbedding.from_config(
                EmbeddingConfig(
                    weight=cached(build_embedding_weight(source.embedding(meta), policy_), "embedding"),
                    mesh_device=mesh_device,
                )
            ),
        )
        report.dram["after_embedding"] = dram_view(mesh_device)
        rotary = PplxRotary(args.rotary_dim, args.rope_theta, args.max_seq_len, mesh_device, head_dim=args.head_dim)

        layers, layer_seconds = [], []
        for idx in layer_ids:
            before = dram_view(mesh_device)["allocated_gib"]
            layer, seconds = build(
                f"L{idx:02d}",
                lambda meta, idx=idx: PplxDecoderLayer.from_config(
                    DecoderLayerConfig(
                        weights=cached(
                            build_decoder_layer_weights(source.layer(idx, meta), args, idx, policy_), f"L{idx:02d}"
                        ),
                        args=args,
                        layer_idx=idx,
                        optimizations=opts,
                    )
                ),
            )
            layers.append(layer)
            layer_seconds.append(seconds)
            report.layer_weight_gib.setdefault(layer.kind, []).append(dram_view(mesh_device)["allocated_gib"] - before)
            logger.info(
                f"layer {idx:2d} ({layer.kind}) on device in {seconds:.1f}s (cache hit {report.cache_hits[f'L{idx:02d}']})"
            )
        report.seconds["layers"] = sum(layer_seconds)

        head, report.seconds["head"] = build(
            "head",
            lambda meta: PplxDecisionHead.from_config(
                DecisionHeadConfig(
                    norm_weight=cached(zero_centred_norm(source.final_norm(meta), policy_).weight, "final_norm"),
                    readout_weight=cached(
                        build_readout_weight(source.readout(meta), policy_, pad_to=PADDED_OPTIONS), "readout"
                    ),
                    eps=args.rms_norm_eps,
                    temperature=reader.temperature,
                    linear=opts.linear,
                    norm=opts.norm,
                    mesh_device=mesh_device,
                )
            ),
        )
        tower = None
        if vision:
            from models.demos.pplx_decider_v1_27b.tt.vision.tower import PplxVisionTower

            start = time.perf_counter()
            tower = PplxVisionTower.from_snapshot(mesh_device, reader)
            report.seconds["vision"] = time.perf_counter() - start
            full_config = json.loads((reader.path / "config.json").read_text())
            if full_config.get("image_token_id") != IMAGE_TOKEN_ID:
                raise ValueError(f"config image_token_id {full_config.get('image_token_id')} != {IMAGE_TOKEN_ID}")
        ttnn.synchronize_device(mesh_device)
        report.seconds["total"] = time.perf_counter() - t_start
        report.dram["after_load"] = dram_view(mesh_device)
        config = ModelConfig(
            args=args,
            optimizations=opts,
            temperature=reader.temperature,
            pad_token_id=_pad_token_id(reader),
            layer_ids=layer_ids,
            cache_dir=cache_dir,
        )
        logger.info(f"model loaded: {json.dumps(report.seconds)}; DRAM {report.dram['after_load']}")
        return cls(config, embedding, layers, rotary, head, report, vision=tower)

    # -- runtime ----------------------------------------------------------------------------------
    def upload_tokens(self, input_ids, bucket: int | None = None) -> tuple[ttnn.Tensor, int]:
        """Right-pad ``input_ids`` to its bucket (or ``bucket``) and upload [1, bucket] uint32.

        Returns the device tensor and the index of the last real token.
        """
        ids = [int(t) for t in input_ids]
        bucket = bucket or bucket_for(len(ids))
        if bucket not in BUCKETS or len(ids) > bucket:
            raise ValueError(f"{len(ids)} tokens do not fit bucket {bucket} (buckets {BUCKETS})")
        padded = torch.full((1, bucket), self.config.pad_token_id, dtype=torch.int32)
        padded[0, : len(ids)] = torch.tensor(ids, dtype=torch.int32)
        tokens = ttnn.from_torch(
            padded,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return tokens, len(ids) - 1

    # -- image requests ---------------------------------------------------------------------------
    def prepare_images(self, encoded: dict, bucket: int | None = None) -> tuple[ttnn.Tensor, int, ImageInputs | None]:
        """Per-request input prep for a processor output (``input_ids``, ``mm_token_type_ids``,
        ``pixel_values``, ``image_grid_thw``; batch 1, unpadded) and its upload.

        Host work (like tokenization): 3D position ids (``get_rope_index``) -> per-request cos/sin,
        the splice index list, the per-image vision inputs (``PplxVisionTower.prepare_inputs``).
        Returns ``(tokens, last_index, images)``; ``images`` is None for a prompt without images,
        which then runs the text-only path unchanged.
        """
        ids = torch.as_tensor(encoded["input_ids"]).reshape(-1)
        if "pixel_values" not in encoded or encoded.get("image_grid_thw") is None:
            if bool((ids == IMAGE_TOKEN_ID).any()):
                raise ValueError("Prompt has image tokens but no pixel_values / image_grid_thw")
            tokens, last_index = self.upload_tokens(ids.tolist(), bucket)
            return tokens, last_index, None
        if self.vision is None:
            raise ValueError("Image request on a model built without the vision tower (from_snapshot(vision=True))")
        types = torch.as_tensor(encoded["mm_token_type_ids"]).reshape(-1)
        is_image = ids == IMAGE_TOKEN_ID
        if not torch.equal(is_image, types == 1) or bool((types > 1).any()):
            raise ValueError("mm_token_type_ids must mark exactly the image tokens (no video)")
        grids = torch.as_tensor(encoded["image_grid_thw"]).reshape(-1, 3)
        patches = [int(t * h * w) for t, h, w in grids.tolist()]
        pixel_values = torch.as_tensor(encoded["pixel_values"])
        if pixel_values.shape[0] != sum(patches):
            raise ValueError(f"pixel_values has {pixel_values.shape[0]} rows, grids need {sum(patches)}")
        merge = self.vision.config.args.merge_unit
        num_image_tokens = int(is_image.sum())
        if num_image_tokens != sum(p // merge for p in patches):
            raise ValueError(f"{num_image_tokens} image tokens for {sum(patches) // merge} image features")

        tokens, last_index = self.upload_tokens(ids.tolist(), bucket)
        bucket = tokens.shape[-1]
        a = self.config.args
        position_ids = get_rope_index(ids, types, grids, spatial_merge_size=self.vision.config.args.spatial_merge_size)
        rotary = PplxRequestRotary.from_position_ids(
            position_ids,
            bucket,
            rotary_dim=a.rotary_dim,
            theta=a.rope_theta,
            head_dim=a.head_dim,
            mesh_device=self.mesh_device,
        )
        vision = [
            self.vision.prepare_inputs(pv, grid) for pv, grid in zip(torch.split(pixel_values, patches, dim=0), grids)
        ]
        index = torch.arange(bucket, dtype=torch.int32)
        index[: ids.shape[0]][is_image] = bucket + torch.arange(num_image_tokens, dtype=torch.int32)
        splice_index = ttnn.from_torch(
            index.reshape(1, bucket),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        images = ImageInputs(
            vision=vision,
            splice_index=splice_index,
            rotary=rotary,
            position_ids=position_ids,
            num_image_tokens=num_image_tokens,
        )
        return tokens, last_index, images

    def image_features(self, images: ImageInputs) -> list[ttnn.Tensor]:
        """Vision tower per image: [1, 1, n_i / 4, 5120] BF16 TILE each (device only)."""
        return [self.vision(v) for v in images.vision]

    def splice(self, tokens: ttnn.Tensor, images: ImageInputs, features: list[ttnn.Tensor]) -> ttnn.Tensor:
        """Text embeddings with the image-token rows replaced by the image features, on device.

        ``table = [embedding(tokens) ; features...]`` (ROW_MAJOR, bucket + N rows) and one
        ``ttnn.embedding`` gather over ``splice_index`` -> [1, bucket, 5120] BF16 TILE. Both steps
        copy rows only, so text rows equal the text-only embedding bit for bit and image rows equal
        the vision features. Frees ``features``.
        """
        self.embedding.load_device_weights()
        weight = self.embedding.weight
        hidden, bucket = weight.shape[-1], tokens.shape[-1]
        text = ttnn.embedding(tokens, weight, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        parts = [ttnn.reshape(text, [bucket, hidden])]
        for f in features:
            rows = ttnn.to_layout(f, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(f)
            parts.append(ttnn.reshape(rows, [rows.shape[-2], hidden]))
        table = ttnn.concat(parts, dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for p in parts:
            ttnn.deallocate(p)
        x = ttnn.embedding(images.splice_index, table, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(table)
        return ttnn.reshape(x, [1, bucket, hidden])

    def embed_with_images(self, tokens: ttnn.Tensor, images: ImageInputs) -> ttnn.Tensor:
        """Vision tower -> splice: the ``inputs_embeds`` of an image request [1, bucket, 5120] BF16 TILE."""
        return self.splice(tokens, images, self.image_features(images))

    # -- forward ----------------------------------------------------------------------------------
    def forward(
        self,
        tokens: ttnn.Tensor,
        last_index: int,
        count: int,
        *,
        images: ImageInputs | None = None,
        embeds: ttnn.Tensor | None = None,
        return_hidden: bool = False,
        collect_layer_hidden: bool = False,
    ):
        """tokens [1, bucket] uint32 on device -> (probs, logits, extras), device tensors.

        probs / logits are [1, 1, 256] fp32 (probs zero from ``count`` on). ``extras`` is a dict that
        holds, on request, ``final_hidden`` (final-normed last real token [1, 1, 5120]) and
        ``layer_hidden`` (the last real token's residual after every layer, list of [1, 1, 5120]).

        ``images`` (from ``prepare_images``): run the vision tower, splice its features into the
        embeddings and use the request's 3D-mRoPE tables. ``embeds``: an already spliced
        ``embed_with_images`` output to start from (consumed; for per-phase timing and tests).
        Without either, this is the text-only path.
        """
        hidden_size = self.config.args.hidden_size
        if images is None:
            x = self.embedding(tokens)
            rotary = self.rotary
        else:
            x = embeds if embeds is not None else self.embed_with_images(tokens, images)
            rotary = images.rotary
        layer_hidden = []
        for layer in self.layers:
            y = layer(x, rotary)
            ttnn.deallocate(x)
            x = y
            if collect_layer_hidden:
                layer_hidden.append(ttnn.slice(x, [0, last_index, 0], [1, last_index + 1, hidden_size]))
        probs, logits, final_hidden = self.head(x, last_index, count, return_hidden=return_hidden)
        ttnn.deallocate(x)
        extras = {}
        if return_hidden:
            extras["final_hidden"] = final_hidden
        if collect_layer_hidden:
            extras["layer_hidden"] = layer_hidden
        return probs, logits, extras

    def decide(self, input_ids, count: int, *, bucket: int | None = None) -> list[float]:
        """Probabilities of the ``count`` options: upload -> device forward -> one readback."""
        tokens, last_index = self.upload_tokens(input_ids, bucket)
        probs, logits, _ = self(tokens, last_index, count)
        out = ttnn.to_torch(probs).reshape(-1)[:count].tolist()
        for t in (tokens, probs, logits):
            ttnn.deallocate(t)
        return out

    def decide_encoded(self, encoded: dict, count: int, *, bucket: int | None = None) -> list[float]:
        """``decide`` for a processor output that may hold images: input prep -> device forward -> one readback."""
        tokens, last_index, images = self.prepare_images(encoded, bucket)
        probs, logits, _ = self(tokens, last_index, count, images=images)
        out = ttnn.to_torch(probs).reshape(-1)[:count].tolist()
        for t in (tokens, probs, logits):
            ttnn.deallocate(t)
        if images is not None:
            images.deallocate()
        return out


def _loadable(module) -> list:
    """Sub-modules with weights to upload (decoder layers have no loader of their own)."""
    if isinstance(module, PplxDecoderLayer):
        return [module.input_norm, module.post_norm, module.mlp, module.mixer]
    return [module]


def _pad_token_id(reader) -> int:
    """Any id works for right padding (causal model); use the tokenizer's pad id, else 0."""
    try:
        cfg = json.loads((reader.path / "tokenizer_config.json").read_text())
        token = cfg.get("pad_token")
        vocab = json.loads((reader.path / "tokenizer.json").read_text())
        added = {t["content"]: t["id"] for t in vocab.get("added_tokens", [])}
        return int(added.get(token, 0))
    except (OSError, ValueError, KeyError, TypeError):
        return 0
