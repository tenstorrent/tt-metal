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

Usage::

    model = PplxDeciderModel.from_snapshot(device)
    tokens, last_index = model.upload_tokens(input_ids)       # host -> device, padded to the bucket
    probs, logits, _ = model(tokens, last_index, count)        # device tensors [1, 1, 256] fp32
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
from models.demos.pplx_decider_v1_27b.tt.rope import PplxRotary
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


class PplxDeciderModel(LightweightModule):
    """Embedding + decoder stack + decision head, all resident on one device."""

    def __init__(self, config: ModelConfig, embedding, layers, rotary, head, load_report: LoadReport | None = None):
        super().__init__()
        self.config = config
        self.mesh_device = config.optimizations.mesh_device
        self.embedding = embedding
        self.layers = layers
        self.rotary = rotary
        self.head = head
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
    ) -> "PplxDeciderModel":
        """Build and upload every weight. ``layer_ids`` (default: all 64) allows a reduced stack for debugging.

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
        return cls(config, embedding, layers, rotary, head, report)

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

    def forward(
        self,
        tokens: ttnn.Tensor,
        last_index: int,
        count: int,
        *,
        return_hidden: bool = False,
        collect_layer_hidden: bool = False,
    ):
        """tokens [1, bucket] uint32 on device -> (probs, logits, extras), device tensors.

        probs / logits are [1, 1, 256] fp32 (probs zero from ``count`` on). ``extras`` is a dict that
        holds, on request, ``final_hidden`` (final-normed last real token [1, 1, 5120]) and
        ``layer_hidden`` (the last real token's residual after every layer, list of [1, 1, 5120]).
        """
        hidden_size = self.config.args.hidden_size
        x = self.embedding(tokens)
        layer_hidden = []
        for layer in self.layers:
            y = layer(x, self.rotary)
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
