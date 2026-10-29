# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 prefill transformer and chunk driver (bead F9; dev-spec §4b).

Per request: allocate ``V41PrefillState``; for each chunk of ``chunk`` tokens (the last one padded, graph.md
rule 8): copy its host-prepared inputs (token ids; Engram hash ids -> packed rows / row ids) into the request's
fixed device input buffers, embed, expand to ``hc_mult`` streams, one-hot pre-mix, run the layers in order (DSpark taps at its
target layers), seed the state's DSpark window rings, advance the state. In the chunks holding scored
positions (the last real token by default): collapse the streams with the last block's pre, final norm, LM
head on those rows.

The layer list is any execution-ordered subset of V4.1 layers whose sources are present (a full model is
layers 0..39). Engram layers (1, 14) apply their n-gram update to the streams before their block; the hash
depends only on tokens and carries the last tokens across chunks, so the host prepares chunk i + 1 while the device
runs chunk i (bead 8y7.17). The device forward of a text chunk makes no host transfers (position tables in the state,
DSpark RoPE tables uploaded per request, the initial pre-mix built once), so ``V41PrefillTrace`` captures the same
chunk loop and replays it (bead 8y7.9.2); the untraced ``prefill`` is that loop without the capture.

Image spans (bead 10.2, graph V5; reference ``Transformer.forward`` / ``merge_image_embeddings``): with
``token_types`` (``image_processor`` TEXT / IMAGE_START / IMAGE / IMAGE_NEW_LINE / IMAGE_END per position) and one
aligner output per span (``TtV41Vision``), the first chunk's embedding rows of each span are overwritten: the
delimiters with the learned ``image_start`` / ``image_end`` / ``image_newline`` rows, the IMAGE slots with the
aligner rows in reading order. On device: a per-request table ``[delimiters | image rows]`` (each chip its TP slice)
is gathered by a per-position row index and selected into the embedding by the image mask (exact copies). Image
spans must lie in the first chunk (the reference asserts it). Image tokens route with each gate's ``bias_vl``
(``TtV41Moe``) and take no part in Engram n-grams and get no Engram update (hash and gate mask).
"""

import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable

import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.reference.deepseek_v41.image_processor import (
    IMAGE,
    IMAGE_END,
    IMAGE_NEW_LINE,
    IMAGE_START,
    TEXT,
)
from models.demos.deepseek_v3_d_p.tests.v41.weight_cache import routed_expert_dtype
from models.demos.deepseek_v3_d_p.tt.mhc.tt_mhc import mhc_expand
from models.demos.deepseek_v3_d_p.tt.v41.block import TtV41Block
from models.demos.deepseek_v3_d_p.tt.v41.cache import PACK, V41PrefillState, check_pack, pack_enabled, pack_segments
from models.demos.deepseek_v3_d_p.tt.v41.dspark import TtV41DSpark
from models.demos.deepseek_v3_d_p.tt.v41.engram import EngramChunkInputs
from models.demos.deepseek_v3_d_p.tt.v41.head import TP_AXIS, TtV41Embedding, TtV41Head
from models.demos.deepseek_v3_d_p.tt.v41.mhc import initial_pre_mix
from models.demos.deepseek_v3_d_p.tt.v41.weights import begin_layer, complete_layer
from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import init_checker
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat
from models.demos.deepseek_v3_d_p.utils.sub_device_trace import SubDeviceTraceController

# merge-table rows of the span delimiters (checkpoint names), in table order; the image rows follow them
_DELIMITERS = ("image_start", "image_end", "image_newline")
_DELIMITER_ROW = {IMAGE_START: 0, IMAGE_END: 1, IMAGE_NEW_LINE: 2}


PREFILL_GENERIC = "TT_V41_PREFILL_GENERIC"  # "1": prefill states are position-generic (V41GenericPrefillTrace)
# chunks of host inputs prepared ahead of the device by a thread (0: in the loop, as before). The replay issue of a
# captured chunk blocks when the command queue is full, so without the thread the Engram hashing and table gathers
# of the next chunk (~90 ms a chunk at 40 layers) serialize behind it (17.7 s of a 61.7 s 1M
# prefill); a thread prepares them while the main thread waits in the issue.
PREFILL_PREFETCH = "TT_V41_PREFILL_PREFETCH"
# "1": a chunked prefill call that ends before the prompt's end waits for the device and reads its
# logits back as the final call does (the earlier synchronous behaviour). Default "0": it returns
# once its replays are issued (no logits: the plugin discards an intermediate chunk's), so the next
# call's host work (the scheduler, the stage of the next chunk) overlaps the device; the next
# chunk's copies and replay are ordered behind this chunk's on the command queue.
PREFILL_DRAIN = "TT_V41_PREFILL_DRAIN"
# "1": a generic chunk builds each index mask once (per ratio and score width, row-major) and its index sources
# add it to the score kernel's row-major output directly; default "0": every index source
# builds its own mask and tilizes the score around the add.
PREFILL_MASK_ONCE = "TT_V41_PREFILL_MASK_ONCE"
# "1": the chunk loop's host-to-device copies in one dispatch round each: ``reset_state``'s
# zero-fills of the window carries (and DSpark rings) as one ``ttnn.copy_host_to_device_tensors``, and a chunk's
# staging (the token ids, the Engram inputs and a generic state's position inputs) as one. Every copy is a round
# of the 32 per-device dispatch workers (one wake-up of the slowest), so the
# one-chunk call at 1K paid ~40 rounds in ``reset`` (9 ms) and ~15 in ``stage`` (6 ms) of its ~35 ms of host. The bytes and the buffers are the same: bit for bit. Default "0": one copy per tensor, as before.
PREFILL_HOST_BATCH = "TT_V41_PREFILL_HOST_BATCH"


def prefill_host_batch() -> bool:
    return os.environ.get(PREFILL_HOST_BATCH, "0") == "1"


# "1": ``reset_state`` zeroes the window carries and DSpark rings with device-to-device copies from one persistent
# all-zero device tensor per shape (``ttnn.copy``), instead of uploading ~6 MB of zero pages to every chip from the
# host. The upload was 9 ms a request as ~40 single copies, 3 ms as
# one batch on the dispatch workers, and 23 ms as one batch under ``TT_METAL_MESH_WRITE_INLINE=1``
# (the served catalog now: the batch runs on the calling thread, device after device). Zeros are zeros:
# bit for bit. Default "0": the host upload (batched or not by TT_V41_PREFILL_HOST_BATCH).
PREFILL_RESET_DEVICE = "TT_V41_PREFILL_RESET_DEVICE"


def prefill_reset_device() -> bool:
    return os.environ.get(PREFILL_RESET_DEVICE, "0") == "1"


def prefill_prefetch() -> int:
    return int(os.environ.get(PREFILL_PREFETCH, "2"))


def prefill_drain() -> bool:
    return os.environ.get(PREFILL_DRAIN, "0") == "1"


def prefill_mask_once() -> bool:
    return os.environ.get(PREFILL_MASK_ONCE, "0") == "1"


def prefill_generic() -> bool:
    return os.environ.get(PREFILL_GENERIC, "0") == "1"


@dataclass
class _Request:
    tokens: torch.Tensor  # [S]
    first_scored: int  # first prompt position whose logits are returned
    token_types: torch.Tensor | None  # [S] image_processor types; None = text only
    image_features: list | None


@dataclass
class _HostChunk:
    """One chunk's host inputs (token-only: prepared one chunk ahead of the device)."""

    start: int
    length: int
    token_ids: ttnn.Tensor  # host [1, 1, chunk/sp] uint32
    engram: dict[int, EngramChunkInputs]  # host tensors
    types: torch.Tensor | None  # the first chunk's token types padded with TEXT, when it holds image spans
    tables: list | None = None  # generic states: ``V41ChunkTables.prepare_chunk_host`` (host tensors, buffers)
    cost: dict | None = None  # host seconds of making this chunk: hash, gather (Engram), tables
    segments: tuple | None = None  # a packed chunk's layout, ((offset, length), ...) (TT_V41_PACK); None: a prompt


def image_spans(token_types: torch.Tensor) -> list[tuple[int, int]]:
    """(start, end) of each image span of ``token_types`` [S] (IMAGE_START .. IMAGE_END, end exclusive)."""
    starts = (token_types == IMAGE_START).nonzero().flatten().tolist()
    ends = (token_types == IMAGE_END).nonzero().flatten().tolist()
    spans = [(s, e + 1) for s, e in zip(starts, ends)]
    image = token_types != TEXT
    covered = torch.zeros_like(image)
    for s, e in spans:
        covered[s:e] = True
    assert (
        len(starts) == len(ends) and all(s < e for s, e in spans) and torch.equal(covered, image)
    ), "image positions must form IMAGE_START .. IMAGE_END spans"
    return spans


class TtV41Transformer(LightweightModule):
    def __init__(
        self,
        mesh_device,
        config,
        layers: list[int],
        layer_weights: Callable[[int, bool], dict],
        embed_weight: torch.Tensor,
        norm_weight: torch.Tensor,
        head_weight: torch.Tensor,
        max_seq_len: int,
        chunk: int,
        dspark_weights: dict | None = None,
        engram: dict | None = None,
        engram_hash=None,
        image_embeds: dict | None = None,
        weight_cache_path=None,
        routed_expert_weights_dtype=None,
        kv_format: MlaKvCacheFormat | None = None,
    ):
        """``layer_weights(layer, include_moe)`` returns that block's ``TtV41Block`` weights (``weights.load_layer``
        for a checkpoint), without the MoE entries when ``include_moe`` is False; it is called once per layer, so
        host copies are dropped as blocks are built. ``weight_cache_path``: device MoE tensors are converted once
        and loaded from there afterwards (a marker records each completed layer; key the directory by weight
        identity, e.g. checkpoint revision + dtype + mesh). ``image_embeds``: the learned span delimiters
        ``image_start`` / ``image_end`` / ``image_newline`` (``[hidden]`` bf16, checkpoint names) for prompts with
        images; each layer's ``gate_bias_vl`` comes with its ``layer_weights``."""
        self.mesh_device, self.config = mesh_device, config
        # the routed experts' dtype: the argument, else TT_V41_EXPERT_DTYPE (tests/v41/weight_cache.py; the cache
        # directory is keyed by it, so pass the same to weight_cache_dir)
        if routed_expert_weights_dtype is None:
            routed_expert_weights_dtype = routed_expert_dtype()
        self.routed_expert_weights_dtype = routed_expert_weights_dtype
        self.layers, self.chunk, self.max_seq_len = list(layers), chunk, max_seq_len
        # the compressed layers' KV row format of every state this model creates (``new_state`` / ``prefill``):
        # the argument, else ``TT_V41_KV_FORMAT`` (``bf16_rm``, the default, or ``scaled_fp8``)
        self.kv_format = kv_format or MlaKvCacheFormat(os.environ.get("TT_V41_KV_FORMAT", "bf16_rm"))
        self._zero_hosts = {}  # reset_state's zero carries, per spec
        self._zero_devices: dict = {}  # reset_state's zero carries on device, per spec (TT_V41_PREFILL_RESET_DEVICE)
        assert self.layers == sorted(set(self.layers)), "layers must be in execution order"
        needed = [l for l in self.layers if l in config.ENGRAM_LAYER_IDS]
        self.engram = dict(engram or {})
        assert sorted(self.engram) == needed, f"Engram modules for layers {needed} required, got {sorted(self.engram)}"
        assert not needed or engram_hash is not None, "Engram layers need the n-gram hasher"
        self.engram_hash = engram_hash
        self.embedding = TtV41Embedding(mesh_device, config, embed_weight)
        self.image_delimiters = None
        if image_embeds is not None:
            rows = torch.stack([image_embeds[k] for k in _DELIMITERS]).to(torch.bfloat16)
            assert rows.shape == (len(_DELIMITERS), config.EMB_SIZE), rows.shape
            dims = [None, None]
            dims[TP_AXIS] = 3
            self.image_delimiters = ttnn.from_torch(
                rows[None, None],
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, tuple(mesh_device.shape), dims=tuple(dims)),
            )
        self.blocks = []
        if weight_cache_path is not None:
            weight_cache_path.mkdir(parents=True, exist_ok=True)
            init_checker(weight_cache_path)
        for layer in self.layers:
            cached = weight_cache_path is not None and begin_layer(weight_cache_path, layer)
            self.blocks.append(
                TtV41Block(
                    mesh_device,
                    config,
                    layer,
                    layer_weights(layer, not cached),
                    chunk,
                    routed_expert_weights_dtype=routed_expert_weights_dtype,
                    weight_cache_path=weight_cache_path,
                )
            )
            if weight_cache_path is not None:
                complete_layer(weight_cache_path, layer)
        self.head = TtV41Head(mesh_device, config, norm_weight, head_weight)
        self.dspark = None
        if dspark_weights is not None:
            missing = set(config.DSPARK_TARGET_LAYER_IDS) - set(self.layers)
            assert not missing, f"DSpark needs the taps of layers {sorted(missing)}"
            self.dspark = TtV41DSpark(mesh_device, config, dspark_weights)
        self.pre_mix = initial_pre_mix(mesh_device, config, chunk)  # every chunk's first pre-mix (read only)
        self.weight_cache_path = weight_cache_path

    def with_chunk(self, chunk: int, max_seq_len: int) -> "TtV41Transformer":
        """A view of this model for prefill chunks of ``chunk`` tokens (``max_seq_len`` its states' context): the
        same weights and modules except what the chunk sizes, which is ``pre_mix`` and each block's MoE (``TtMoe``
        is built for a sequence length: gate bias rows, dispatch and combine buffers). The blocks are shallow copies
        with a new ``TtV41Moe`` of the chunk, built as the decoder builds its 32-row MoEs (``TtV41Decoder._decode_moe``):
        the routed and shared experts' weights are this model's device tensors (injected under ``mock.patch``:
        ``TtV41Moe.__init__`` is part of the weight cache's key), the gate loads from the weight cache; the shared
        expert keeps the prefill's overlap with the dispatch. Prefill a chunk of this model before one of the view: the
        shared expert's reduce-scatter intermediate is one buffer per mesh, allocated at the first input's shape
        (``TtCCL.get_shared_rs_intermediate``) and the Linear path tolerates a smaller input, not a larger one.
        Short prompts in a small chunk: a 1,028-token prompt in a 5,120-row chunk paid the
        padded rows in every op but the MoE's."""
        import copy
        from unittest import mock

        from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
        from models.demos.deepseek_v3_d_p.tt.moe.tt_shared_expert import TtSharedExpert
        from models.demos.deepseek_v3_d_p.tt.v41.moe import TtV41Moe

        assert self.weight_cache_path is not None, "a chunk view builds its MoE gates from the weight cache"
        assert chunk < self.chunk and max_seq_len % chunk == 0, (chunk, self.chunk, max_seq_len)
        view = copy.copy(self)
        view.chunk, view.max_seq_len = chunk, max_seq_len
        view.pre_mix = initial_pre_mix(self.mesh_device, self.config, chunk)
        view.blocks = []
        for layer, block in zip(self.layers, self.blocks):
            source = block.ffn.moe.routed_expert
            routed = (source.gate_projs, source.up_projs, source.down_projs)
            dense = block.ffn.moe.shared_expert
            shared = {"gate": dense.gate_proj, "up": dense.up_proj, "down": dense.down_proj}
            with (
                mock.patch.object(
                    TtRoutedExpert, "_convert_and_cache_expert_weights", staticmethod(lambda *a, **k: routed)
                ),
                mock.patch.object(TtSharedExpert, "_convert_and_cache_weights", staticmethod(lambda *a, **k: shared)),
            ):
                moe = TtV41Moe(
                    self.mesh_device,
                    self.config,
                    layer,
                    {},
                    chunk,
                    routed_expert_weights_dtype=self.routed_expert_weights_dtype,
                    weight_cache_path=self.weight_cache_path,
                )
            copied = copy.copy(block)
            copied.ffn = moe
            view.blocks.append(copied)
        return view

    def _token_ids(self, tokens: torch.Tensor, device=True):
        """[chunk] int -> [1, 1, chunk/sp] uint32 SP-sharded, replicated over TP (a host tensor unless ``device``)."""
        return ttnn.from_torch(
            tokens.to(torch.int32).view(1, 1, -1),
            device=self.mesh_device if device else None,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, tuple(self.mesh_device.shape), dims=(2, None)),
        )

    def _column(self, values: torch.Tensor) -> ttnn.Tensor:
        """[chunk] -> [1, 1, chunk/sp, 1] bf16 TILE, SP-sharded, replicated over TP."""
        return ttnn.from_torch(
            values.to(torch.bfloat16).view(1, 1, -1, 1),
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, tuple(self.mesh_device.shape), dims=(2, None)),
        )

    def _merge_images(self, h: ttnn.Tensor, types: torch.Tensor, image_features: list, image_rows: ttnn.Tensor):
        """Overwrite the image spans of the first chunk's embedding ``h`` [1, 1, chunk/sp, hidden/tp] bf16 (``types``
        [chunk], padded with TEXT): delimiters and aligner rows (``image_features[i]`` [1, 1, T_i, hidden] replicated,
        span i's IMAGE slots in reading order) gathered from one merge table, selected where ``image_rows`` is 1."""
        spans = image_spans(types)
        assert len(spans) == len(image_features), f"{len(spans)} image spans, {len(image_features)} feature sets"
        index = torch.zeros(self.chunk, dtype=torch.int64)
        for kind, row in _DELIMITER_ROW.items():
            index[types == kind] = row
        offset, parts = len(_DELIMITERS), [self.image_delimiters]
        for (s, e), features in zip(spans, image_features):
            slots = (types[s:e] == IMAGE).nonzero().flatten() + s
            assert tuple(features.shape) == (
                1,
                1,
                slots.numel(),
                self.config.EMB_SIZE,
            ), f"span at {s}: {slots.numel()} IMAGE slots, features {tuple(features.shape)}"
            index[slots] = offset + torch.arange(slots.numel())
            offset += slots.numel()
            local = ttnn.mesh_partition(features, dim=3, cluster_axis=TP_AXIS)  # this chip's hidden slice
            parts.append(ttnn.to_layout(local, ttnn.ROW_MAJOR_LAYOUT))
        table = ttnn.concat(parts, dim=2)
        table = ttnn.reshape(table, (table.shape[2], table.shape[3]))
        ids = self._token_ids(index)
        merged = ttnn.embedding(ids, table, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        merged = ttnn.reshape(merged, tuple(h.shape))
        out = ttnn.where(image_rows, merged, h)
        for t in parts[1:] + [table, ids, merged, h]:
            ttnn.deallocate(t)
        return out

    # --- the chunk loop: host inputs (tokens only, one chunk ahead) -> fixed device buffers -> device forward ------
    def _request(self, tokens, logit_positions=1, token_types=None, image_features=None) -> _Request:
        total = int(tokens.numel())
        assert 0 < total <= self.max_seq_len, f"prompt of {total} tokens, max {self.max_seq_len}"
        assert 0 < logit_positions <= total, f"logit_positions {logit_positions} outside the {total}-token prompt"
        if token_types is not None:
            assert token_types.shape == tokens.shape, f"token_types {tuple(token_types.shape)} vs {tuple(tokens.shape)}"
            assert not (token_types[self.chunk :] != TEXT).any(), "image spans must lie in the first chunk"
            if not (token_types != TEXT).any():
                token_types = None  # text only
        assert (token_types is None) == (not image_features), "image features need token types with image spans"
        assert token_types is None or self.image_delimiters is not None, "prompts with images need image_embeds"
        return _Request(tokens, total - logit_positions, token_types, image_features)

    def _new_state(
        self, generic: bool = False, max_seq_len: int | None = None, pack_segments: int = 0
    ) -> V41PrefillState:
        """``generic``: a state whose chunk graphs are position-generic (``V41GenericPrefillTrace``); its chunk
        inputs are host-filled per chunk (``_run`` does it), so the untraced prefill runs through it as well.
        ``max_seq_len``: the state's context (default the model's; a packed state needs the first score bucket
        only); ``pack_segments``: its segment carries allocated now (``V41PrefillState``)."""
        state = V41PrefillState(
            self.mesh_device,
            self.config,
            self.max_seq_len if max_seq_len is None else max_seq_len,
            self.chunk,
            self.layers,
            kv_format=self.kv_format,
            generic=generic,
            tiled_index_k=generic and prefill_mask_once(),
            dspark=self.dspark is not None,
            pack_segments=pack_segments,
        )
        if self.dspark is not None:
            state.dspark_rings = self.dspark.new_rings()
        return state

    def _host_chunks(self, request: _Request, first: int = 0, tables=None):
        """Each chunk's host inputs in order from the chunk at ``first`` (a multiple of the chunk): token ids and
        (Engram layers) the n-gram hash and the packed rows / row ids of its lookups, and (``tables``, a generic
        state's ``V41ChunkTables``) the chunk's position inputs. Tokens-only, so the loop prepares chunk i + 1
        while the device runs chunk i. Resuming at ``first`` seeds the Engram history from the tokens before it
        (what hashing them would have left)."""
        tokens, total = request.tokens, int(request.tokens.numel())
        assert first % self.chunk == 0 and 0 <= first < total, (first, total)
        history = None
        if self.engram:
            n = self.engram_hash.layout.max_ngram_size - 1
            history = self.engram_hash.token_map[tokens[max(0, first - n) : first].to(torch.int64)]
        for start in range(first, total, self.chunk):
            length = min(self.chunk, total - start)
            chunk_tokens = torch.zeros(self.chunk, dtype=torch.int64)
            chunk_tokens[:length] = tokens[start : start + length]
            # image spans lie in the first chunk: the other chunks run the text path
            types = request.token_types[:length] if request.token_types is not None and start == 0 else None
            text = None if types is None else types == TEXT
            engram = {}
            cost = {"hash": 0.0, "gather": 0.0, "tables": 0.0}
            if self.engram:
                t0 = time.perf_counter()
                ids, history = self.engram_hash(tokens[start : start + length], history, text)
                cost["hash"] = time.perf_counter() - t0
                t0 = time.perf_counter()
                engram = {
                    l: e.prepare_host(ids[:, self.engram_hash.layer_index(l)], self.chunk, text)
                    for l, e in self.engram.items()
                }
                cost["gather"] = time.perf_counter() - t0
            padded = None
            if types is not None:
                padded = torch.full((self.chunk,), TEXT, dtype=torch.int64)
                padded[:length] = types
            position = None
            if tables is not None:
                t0 = time.perf_counter()
                position = tables.prepare_chunk_host(start)
                cost["tables"] = time.perf_counter() - t0
            yield _HostChunk(start, length, self._token_ids(chunk_tokens, device=False), engram, padded, position, cost)

    def _stage(self, host: _HostChunk, buffers: dict, tables=None):
        """Write a chunk's host inputs into the request's device input buffers (allocated by its first chunk, so
        every chunk's forward reads fixed addresses) -> (token ids, Engram inputs) on device. ``tables`` (a generic
        state's ``V41ChunkTables``): the chunk's position inputs are filled too (``prepare_chunk``), in the same
        dispatch round as the rest with ``TT_V41_PREFILL_HOST_BATCH=1``. A packed chunk's layout (``host.segments``,
        ``TT_V41_PACK``) goes to ``prepare_chunk`` with it."""
        if "token_ids" not in buffers:
            buffers["token_ids"] = ttnn.to_device(host.token_ids, self.mesh_device)
            buffers["engram"] = {l: e.to_device(self.mesh_device) for l, e in host.engram.items()}
            if tables is not None:
                tables.prepare_chunk(host.start, host.length, host.tables, segments=host.segments)
            return buffers["token_ids"], dict(buffers["engram"])
        if not prefill_host_batch():
            ttnn.copy_host_to_device_tensor(host.token_ids, buffers["token_ids"])
            engram = {l: e.copy_into(buffers["engram"][l]) for l, e in host.engram.items()}
            if tables is not None:
                tables.prepare_chunk(host.start, host.length, host.tables, segments=host.segments)
            return buffers["token_ids"], engram
        pairs = [(host.token_ids, buffers["token_ids"])]
        engram = {}
        for l, e in host.engram.items():
            pairs += e.copy_pairs(buffers["engram"][l])
            engram[l] = e.device_inputs(buffers["engram"][l])
        if tables is not None:
            tables.prepare_chunk(host.start, host.length, host.tables, pairs=pairs, segments=host.segments)
        ttnn.copy_host_to_device_tensors([h for h, _ in pairs], [d for _, d in pairs])
        return buffers["token_ids"], engram

    def _chunk_forward(
        self, state, host: _HostChunk, token_ids, engram_inputs, image_features, rope, first_scored, on_block
    ):
        """One chunk on device -> its scored rows' device logits, [(logits, sp_row, chunk row of its first row, chunk
        rows read from it)] (``TtV41Head.scored_rows``: a tile, or only the scored rows of a partly scored tile). A text chunk without ``on_block`` makes no host transfers (the image merge uploads its
        inputs)."""
        start, length = host.start, host.length
        h = self.embedding(token_ids)
        image_rows = None
        if host.types is not None:
            image_rows = self._column(host.types != TEXT)
            h = self._merge_images(h, host.types, image_features, image_rows)
        x = mhc_expand(ttnn.typecast(h, ttnn.float32), self.config.HC_MULT)
        pre = self.pre_mix
        taps = []
        mask_once = state.generic and prefill_mask_once()
        if mask_once:
            state.tables.begin_chunk_masks()
        for layer, block in zip(self.layers, self.blocks):
            if layer in self.engram:
                x = self.engram[layer](x, engram_inputs[layer])
            if self.dspark is not None and layer in self.config.DSPARK_TARGET_LAYER_IDS:
                taps.append(self.dspark.tap(x))
            x, pre = block(x, pre, state, length, image_rows)
            if on_block is not None:
                on_block(layer, x, start, length)
        if mask_once:
            state.tables.end_chunk_masks()
        if image_rows is not None:
            ttnn.deallocate(image_rows)
        if self.dspark is not None:
            self.dspark.seed(taps, start, length, state.dspark_rings, rope)
        scored = []
        if host.segments is not None:  # packed: each segment's last row (in segment order)
            rows = [offset + n - 1 for offset, n in host.segments]
        else:
            rows = list(range(max(first_scored - start, 0), length)) if start + length > first_scored else []
        if rows:
            final = ttnn.typecast(self.blocks[-1].residual.final_collapse(x, pre), ttnn.bfloat16)
            while rows:  # one head projection per distinct 32-row tile (a tile never spans chips)
                logits, (sp_row, offset) = self.head(final, rows[0])
                first = rows[0] - offset
                tile = [r for r in rows if r < first + ttnn.TILE_SIZE]
                logits = self.head.scored_rows(logits, tile[0] - first, len(tile))
                scored.append((logits, sp_row, tile[0], tile))
                rows = rows[len(tile) :]
        return scored

    def _run(
        self,
        request: _Request,
        state,
        buffers: dict,
        step: Callable,
        on_block=None,
        timings: dict | None = None,
        start: int = 0,
        end: int | None = None,
        resume: dict | None = None,
    ) -> list:
        """The chunk loop of one request from position ``start`` (0, or a multiple of the chunk the state is at:
        a prefill resumed after its earlier chunks) to ``end`` (the prompt's end, or a chunk boundary before it:
        one call of a chunked prefill). ``step(c, forward)`` runs chunk ``c``'s device forward: ``forward()``
        (untraced or captured) or a replay of its capture. Returns every chunk's scored device logits.
        ``timings`` accumulates the host seconds of each phase (``prepare``: the chunks' host inputs, ``stage``:
        their copies to the device, ``step``: the device call, untraced or a replay issued without waiting).
        ``resume`` carries the host-input pipeline across the calls of one chunked prefill (the prompt's
        remaining chunks are prepared ahead while this call's run): pass the dict the previous call filled."""
        total = int(request.tokens.numel())
        end = total if end is None else end
        assert start < end <= total and (end == total or end % self.chunk == 0), (start, end, total)
        assert state.start == start, f"the state is at {state.start}, the chunk loop starts at {start}"
        if self.dspark is not None and not state.generic and "dspark_rope" not in buffers:
            buffers["dspark_rope"] = [
                self.dspark.seed_rope(s, min(self.chunk, total - s)) for s in range(0, total, self.chunk)
            ]
        t = timings if timings is not None else {}
        scored = []
        t0 = time.perf_counter()
        if resume is not None and resume.get("next") == start and resume.get("host") is not None:
            chunks, host = resume["chunks"], resume["host"]  # the pipeline of the previous call, a chunk ahead
        else:
            tables = state.tables if state.generic else None
            chunks = self._prefetched(self._host_chunks(request, start, tables), prefill_prefetch())
            host = next(chunks)
        t["prepare"] = t.get("prepare", 0.0) + time.perf_counter() - t0
        chunk_log = os.environ.get("TT_V41_PREFILL_CHUNK_LOG", "0") == "1"
        for c in range(start // self.chunk, -(-end // self.chunk)):
            if host.cost is not None:  # the thread's (or the loop's) seconds making this chunk's host inputs
                for k, v in host.cost.items():
                    t[k] = t.get(k, 0.0) + v
            t0 = time.perf_counter()
            token_ids, engram_inputs = self._stage(host, buffers, state.tables if state.generic else None)
            t["stage"] = t.get("stage", 0.0) + time.perf_counter() - t0
            if self.dspark is None:
                rope = None
            elif state.generic:  # the seed rows' pair is a host-filled chunk input (prepare_chunk, this chunk)
                rope = state.tables.dspark_rope(host.length)
            else:
                rope = buffers["dspark_rope"][c]
            forward = lambda: self._chunk_forward(
                state, host, token_ids, engram_inputs, request.image_features, rope, request.first_scored, on_block
            )
            t0 = time.perf_counter()
            scored += step(c, forward)
            if chunk_log:  # the chunk's wall time, device included: a measurement, not the serving path (it
                # waits for the device before preparing the next chunk's host inputs)
                ttnn.synchronize_device(self.mesh_device)
                logger.info(
                    f"prefill chunk {c} at {host.start} ({host.length} tokens): {time.perf_counter() - t0:.3f} s"
                )
            t["step"] = t.get("step", 0.0) + time.perf_counter() - t0
            length = host.length
            t0 = time.perf_counter()
            host = next(chunks, None)  # the next chunk's host inputs, while the device runs this chunk
            t["prepare"] += time.perf_counter() - t0
            state.advance(length)
        if resume is not None:
            resume.update(chunks=chunks, host=host, next=end)
        return scored

    @staticmethod
    def _prefetched(chunks, ahead: int):
        """``chunks`` (the host-input generator) run ``ahead`` items ahead on a thread; 0 returns it as is. The
        items are host-only (torch and host ttnn tensors), so the thread needs no device."""
        if ahead <= 0:
            return chunks
        q: queue.Queue = queue.Queue(maxsize=ahead)
        done = object()

        def worker():
            try:
                for item in chunks:
                    q.put(item)
                q.put(done)
            except BaseException as e:  # surfaced in the consumer
                q.put(e)

        threading.Thread(target=worker, daemon=True, name="v41-prefill-prefetch").start()

        def consume():
            while True:
                item = q.get()
                if item is done:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item

        return consume()

    def _logits(self, scored: list) -> torch.Tensor:
        """fp32 [scored rows, vocab] from the device logits of ``_run`` (in prompt order)."""
        rows = []
        for logits, sp_row, first, tile in scored:
            host = self.head._tile_to_host(logits, sp_row)
            rows.append(host[[r - first for r in tile]])
        return torch.cat(rows)

    def prefill(
        self,
        tokens: torch.Tensor,
        logit_positions: int = 1,
        on_block: Callable | None = None,
        token_types: torch.Tensor | None = None,
        image_features: list | None = None,
    ):
        """tokens [S] (S <= max_seq_len) -> (fp32 logits [logit_positions, vocab] of the last ``logit_positions``
        prompt positions, state); the state holds the caches, carries and (with DSpark) the seeded DSpark
        window rings. Generation needs the last position only; more positions score the prompt itself.
        ``on_block(layer, streams, start, length)`` observes each block's output streams per chunk (accuracy
        gates on free-running per-layer drift). A prompt with images gives ``token_types`` [S] (``image_processor``
        types; image positions carry the image token id in ``tokens``) and ``image_features``: per image span in
        prompt order its aligner output ``[1, 1, T_i, hidden]`` bf16 replicated (``TtV41Vision``), which the caller
        owns; all spans must lie in the first chunk. ``V41PrefillTrace`` runs the same chunk loop traced."""
        request = self._request(tokens, logit_positions, token_types, image_features)
        # ``TT_V41_PREFILL_GENERIC=1``: the position-generic graphs untraced (what ``V41GenericPrefillTrace`` replays),
        # so accuracy tests measure the served graph
        state = self._new_state(generic=prefill_generic() and token_types is None)
        scored = self._run(request, state, {}, lambda c, forward: forward(), on_block)
        return self._logits(scored), state

    def prefill_into(
        self,
        tokens: torch.Tensor,
        state: V41PrefillState,
        logit_positions: int = 1,
        start: int = 0,
        end: int | None = None,
        resume: dict | None = None,
    ) -> torch.Tensor:
        """``prefill`` of a text prompt into an existing ``state`` (one of ``new_state()``, or one a previous request
        used: it is reset to what a fresh request starts from, as ``V41PrefillTrace.run`` does), so a request pool can
        keep its states, and the decode captures made on them, across requests. ``start`` > 0 resumes the prompt's
        prefill at that chunk boundary instead (``tokens`` the whole prefix; the state at ``start`` from the earlier
        chunks, not reset); ``end`` stops after the chunk ending there (``_run``: ``resume`` carries the host
        pipeline to the next call). Returns the logits of the last ``logit_positions`` positions before ``end``."""
        end = int(tokens.numel()) if end is None else end
        request = self._request(tokens, logit_positions)
        request = _Request(request.tokens, end - logit_positions, request.token_types, request.image_features)
        if start == 0:
            self.reset_state(state)
        scored = self._run(request, state, {}, lambda c, forward: forward(), start=start, end=end, resume=resume)
        return self._logits(scored)

    def new_state(
        self, generic: bool = False, max_seq_len: int | None = None, pack_segments: int = 0
    ) -> V41PrefillState:
        return self._new_state(generic, max_seq_len, pack_segments)

    # --- packed chunks (TT_V41_PACK) ----------------------------------------------
    def _host_pack(self, prompts: list[torch.Tensor]) -> _HostChunk:
        """The host inputs of one chunk holding ``prompts`` concatenated from row 0: the token ids, each prompt's
        Engram hash from an empty history (an n-gram never spans two prompts) and the layout."""
        segments = pack_segments([int(t.numel()) for t in prompts])
        check_pack(segments, self.chunk, {self.config.compress_ratio(l) for l in self.layers})
        tokens = torch.cat([t.to(torch.int64) for t in prompts])
        total = int(tokens.numel())
        chunk_tokens = torch.zeros(self.chunk, dtype=torch.int64)
        chunk_tokens[:total] = tokens
        engram = {}
        cost = {"hash": 0.0, "gather": 0.0, "tables": 0.0}
        if self.engram:
            t0 = time.perf_counter()
            ids = torch.cat([self.engram_hash(t, self.engram_hash.new_history())[0] for t in prompts])
            cost["hash"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            engram = {
                l: e.prepare_host(ids[:, self.engram_hash.layer_index(l)], self.chunk, None)
                for l, e in self.engram.items()
            }
            cost["gather"] = time.perf_counter() - t0
        return _HostChunk(0, total, self._token_ids(chunk_tokens, device=False), engram, None, None, cost, segments)

    def prefill_packed(
        self,
        prompts: list[torch.Tensor],
        state: V41PrefillState,
        buffers: dict | None = None,
        on_block: Callable | None = None,
        timings: dict | None = None,
        step: Callable | None = None,
    ) -> tuple[torch.Tensor, tuple]:
        """Prefill several whole text prompts (each ``[S_i]``, ratio-aligned lengths whose sum fits one chunk) in one
        chunk of ``state`` (generic, reset first), each prompt isolated from the others (``V41ChunkTables``'s packed
        fills: its own positions, window rows, index-mask bounds, Engram history, window carries and scored row)
        -> (fp32 logits ``[len(prompts), vocab]`` of each prompt's last position, the layout ``((offset, length),
        ...)``). After the call the state holds every prompt's rows at its segment's offset (compressed and
        index-K rows at ``offset / ratio``, the window scratch at ``WINDOW_SLOT + offset``), each segment's window
        carries in ``state.segment_carry[g]`` and ``state.start`` at the pack's end; a segment joins a decode slot
        from those rows (the pool's segment join). ``step(forward)`` runs the chunk's device forward (default
        ``forward()``; a test's capture hook). Eager only (captures are)."""
        assert pack_enabled(), f"{PACK}=1 is required"
        assert state.generic, "packed chunks run on generic states"
        buffers = {} if buffers is None else buffers
        t = timings if timings is not None else {}
        t0 = time.perf_counter()
        self.reset_state(state)
        t["reset"] = t.get("reset", 0.0) + time.perf_counter() - t0
        t0 = time.perf_counter()
        host = self._host_pack(prompts)
        t["prepare"] = t.get("prepare", 0.0) + time.perf_counter() - t0
        for k, v in host.cost.items():
            t[k] = t.get(k, 0.0) + v
        t0 = time.perf_counter()
        token_ids, engram_inputs = self._stage(host, buffers)
        state.tables.prepare_chunk(0, host.length, None, segments=host.segments)
        t["stage"] = t.get("stage", 0.0) + time.perf_counter() - t0
        t0 = time.perf_counter()
        forward = lambda: self._chunk_forward(state, host, token_ids, engram_inputs, None, None, 0, on_block)
        scored = forward() if step is None else step(forward)
        t["step"] = t.get("step", 0.0) + time.perf_counter() - t0
        state.advance(host.length)
        t0 = time.perf_counter()
        logits = self._logits(scored)
        t["logits"] = t.get("logits", 0.0) + time.perf_counter() - t0
        return logits, host.segments

    def reset_state(self, state: V41PrefillState) -> None:
        """``state`` as a fresh request starts from, where the first chunk reads it: position 0, zero window carries
        (the window slot rows before position 0) and DSpark rings, no compressor carry, no Engram history."""
        state.start = 0
        state.selection = {}
        state.engram_history = None
        for layer in state.compressor_carry:
            state.compressor_carry[layer] = None
        targets = list(state.window_carry.values()) + list(state.dspark_rings or ())
        if prefill_reset_device():
            for t in targets:
                ttnn.copy(self._zeros_device(t), t)
        elif prefill_host_batch() and targets:
            ttnn.copy_host_to_device_tensors([self._zeros(t) for t in targets], targets)
        else:
            for t in targets:
                ttnn.copy_host_to_device_tensor(self._zeros(t), t)

    def _zeros_device(self, t: ttnn.Tensor) -> ttnn.Tensor:
        """A persistent all-zero device tensor of ``t``'s shape, dtype, layout and memory config (one per spec,
        allocated on first use: before any capture, since ``reset_state`` runs for the warm-up's first prompt)."""
        key = (tuple(t.shape), t.dtype, t.layout, t.memory_config())
        if key not in self._zero_devices:
            self._zero_devices[key] = ttnn.to_device(self._zeros(t), self.mesh_device, memory_config=t.memory_config())
        return self._zero_devices[key]

    def _zeros(self, t: ttnn.Tensor) -> ttnn.Tensor:
        """A replicated all-zero host tensor of ``t``'s shape, dtype and layout (cached per spec)."""
        key = (tuple(t.shape), t.dtype, t.layout)
        if key not in self._zero_hosts:
            self._zero_hosts[key] = ttnn.from_torch(
                torch.zeros(key[0]),
                dtype=t.dtype,
                layout=t.layout,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        return self._zero_hosts[key]


class V41PrefillTrace:
    """Trace perf mode of ``TtV41Transformer.prefill`` for text prompts of one length (dev-spec D-G (b)).

    Construction runs the chunk loop twice untraced (compile, then steady-state resources) and captures it once:
    one ``SubDeviceTraceController`` per chunk (the MoE's sub-device switches split each capture), recorded against
    one request state and the request's device input buffers. ``run(tokens)`` replays it for a prompt of the same
    length: per chunk the host copies the token ids and Engram inputs into the fixed buffers, replays the chunk
    without waiting, and prepares the next chunk's host inputs while the device runs. The replay writes the same
    state (caches, carries, DSpark rings; returned by ``run``) as the untraced prefill, bit-identically.
    Device tensors allocated after the capture may overlap its freed intermediates: only host work may run
    between replays; ``release()`` before other device work."""

    def __init__(
        self,
        model: TtV41Transformer,
        tokens: torch.Tensor,
        logit_positions: int = 1,
        state: V41PrefillState | None = None,
        buffers: dict | None = None,
        warmup: int = 2,
    ):
        """``tokens``: a prompt of the traced length (warm-up and capture run it; Engram row subsets must hold its
        rows). ``state``: capture into this state (a request pool's slot, so its decode captures and this one share
        the slot's cache tensors) instead of a new one; the warm-up runs go through it too, and ``run`` resets it.
        ``buffers``: the fixed device input buffers, shared between traces of one model (allocated by the first
        ``_stage``; per-chunk shapes, so one set serves every length). ``warmup``: untraced runs before the capture
        (2: compile, then steady-state persistent resources; fewer when the length ran untraced already)."""
        self.model, self.total, self.logit_positions = model, int(tokens.numel()), logit_positions
        request = model._request(tokens, logit_positions)
        self.buffers = {} if buffers is None else buffers
        self.state = model._new_state() if state is None else state
        for _ in range(warmup):
            model.reset_state(self.state)
            model._run(request, self.state, self.buffers, lambda c, forward: forward())
        ttnn.synchronize_device(model.mesh_device)
        model.reset_state(self.state)
        self.controllers = []
        self.scored = model._run(request, self.state, self.buffers, self._capture)
        # the capture's own outputs that the chunk loop records on the state from inside the captured forward:
        # the compressor carry slices of a trailing odd token (``set_compressor_carry``). A replay rewrites them at
        # the same addresses but runs no host bookkeeping, so ``run`` restores the handles after the replay.
        self.carry = dict(self.state.compressor_carry)

    def _capture(self, c: int, forward: Callable):
        controller = SubDeviceTraceController(self.model.mesh_device)
        moes = [block.ffn for block in self.model.blocks]
        for moe in moes:
            moe.set_trace_controller(controller)
        ttnn.synchronize_device(self.model.mesh_device)
        controller.begin_capture()
        try:
            scored = forward()
        finally:
            controller.end_capture()
            for moe in moes:
                moe.set_trace_controller(None)
        self.controllers.append(controller)
        return scored

    @property
    def num_segments(self) -> int:
        return sum(c.num_segments for c in self.controllers)

    def run(self, tokens: torch.Tensor):
        """tokens [S] of the traced length -> (fp32 logits [logit_positions, vocab], state) as ``prefill``."""
        assert int(tokens.numel()) == self.total, f"traced for {self.total} tokens, got {tokens.numel()}"
        model = self.model
        request = model._request(tokens, self.logit_positions)
        t0 = time.perf_counter()
        model.reset_state(self.state)
        self.last = {"reset": time.perf_counter() - t0}
        model._run(
            request,
            self.state,
            self.buffers,
            lambda c, forward: self.controllers[c].replay(blocking=False) or [],
            timings=self.last,
        )
        t0 = time.perf_counter()
        ttnn.synchronize_device(model.mesh_device)
        self.last["wait"] = time.perf_counter() - t0
        self.state.compressor_carry.update(self.carry)
        t0 = time.perf_counter()
        logits = model._logits(self.scored)
        self.last["logits"] = time.perf_counter() - t0
        return logits, self.state

    def release(self) -> None:
        for controller in self.controllers:
            controller.release()
        self.controllers = []


class V41PackTrace:
    """Trace perf mode of ``TtV41Transformer.prefill_packed`` for one layout: the packed
    chunk of ``k`` whole prompts of the layout's lengths, captured once on a packed state and replayed for every
    pack of that layout. The capture bakes the layout (its window table, the carry buffers' addresses, the scored
    rows); the token ids, the Engram inputs and the layout's position fills are host copies into the state's fixed
    buffers before each replay (``prepare_chunk``), as ``V41PrefillTrace``. Construction runs the chunk ``warmup``
    times eagerly (the compile, then the steady-state resources: the layout's tables, the pack total's MoE padding
    config, the segment carries), then captures it; the device logits of the ``k`` scored rows are read after the
    replay. Device tensors allocated after the capture may overlap its freed intermediates (``release()`` before
    other long-lived device work; the allocation rule for what the served warm-up allocates first)."""

    def __init__(
        self, model: TtV41Transformer, layout, state: V41PrefillState, buffers: dict | None = None, warmup: int = 2
    ):
        assert state.generic, "a generic state (new_state(generic=True))"
        self.model, self.state = model, state
        self.layout = tuple((int(o), int(n)) for o, n in layout)
        self.lengths = [n for _, n in self.layout]
        self.buffers = {} if buffers is None else buffers
        prompts = [torch.randint(1, model.config.VOCAB_SIZE, (n,), dtype=torch.int64) for n in self.lengths]
        t0 = time.perf_counter()
        for _ in range(warmup):
            model.prefill_packed(prompts, state, self.buffers)
        ttnn.synchronize_device(model.mesh_device)
        self.controller = None
        self.scored = None
        _, segments = model.prefill_packed(prompts, state, self.buffers, step=self._capture)
        assert segments == self.layout, (segments, self.layout)
        self.last: dict = {}
        logger.info(
            f"packed prefill: layout {self.layout} captured in {time.perf_counter() - t0:.1f} s "
            f"({self.controller.num_segments} segments)"
        )

    def _capture(self, forward: Callable):
        controller = SubDeviceTraceController(self.model.mesh_device)
        moes = [block.ffn for block in self.model.blocks]
        for moe in moes:
            moe.set_trace_controller(controller)
        ttnn.synchronize_device(self.model.mesh_device)
        controller.begin_capture()
        try:
            self.scored = forward()
        finally:
            controller.end_capture()
            for moe in moes:
                moe.set_trace_controller(None)
        self.controller = controller
        return self.scored

    @property
    def num_segments(self) -> int:
        return self.controller.num_segments

    def run(self, prompts: list[torch.Tensor]) -> tuple[torch.Tensor, tuple]:
        """``prompts`` of the layout's lengths -> (fp32 logits ``[k, vocab]``, the layout) as ``prefill_packed``;
        ``self.last`` the phases (``wait``: the replay's device time from its issue)."""
        assert [int(t.numel()) for t in prompts] == self.lengths, ([int(t.numel()) for t in prompts], self.lengths)
        self.last = {}

        def step(forward: Callable):
            self.controller.replay(blocking=False)
            t0 = time.perf_counter()
            ttnn.synchronize_device(self.model.mesh_device)
            self.last["wait"] = time.perf_counter() - t0
            return self.scored

        return self.model.prefill_packed(prompts, self.state, self.buffers, timings=self.last, step=step)

    def release(self) -> None:
        if self.controller is not None:
            self.controller.release()
            self.controller = None


class V41GenericPrefillTrace:
    """Trace perf mode of the chunk loop for text prompts of any length: position-generic captures.

    A chunk's graph on a generic state (``TtV41Transformer.new_state(generic=True)``) depends on its start only
    through host-filled device inputs (``V41ChunkTables.prepare_chunk``: RoPE rows, the pin page, the indexer
    masks' visible counts, the cache writes' row indices), and on its visible token count only through the
    indexer's score width, which is rounded up to a bucket (``V41ChunkTables.buckets``: 8K, 16K, ..., max_seq).
    So one capture per bucket serves every full chunk whose visible count falls in it; chunk 0 has its own
    capture (its window table and leading-row compaction differ). Every capture scores its chunk's last row (one
    head projection), so a prompt whose length is a multiple of the chunk replays its final chunk too and reads
    the logits from the capture's output; a partial final chunk runs untraced (its valid length, compressor carry
    and scored rows depend on the prompt), as does the final chunk when more than one position is scored. Captures are made on
    synthetic tokens at the last chunk start of each bucket without running the chunks before it: the state's
    contents do not matter to a capture, only its addresses. The replay writes the state exactly as the
    untraced prefill does (the inputs are the same values at the same addresses).

    ``run(tokens)`` resets the state and runs the loop: per chunk the host copies the token ids and Engram inputs
    into the fixed buffers, fills the chunk's position inputs, and replays its bucket's capture without waiting
    (the final chunk runs untraced after them). As ``V41PrefillTrace``, device tensors allocated after the
    captures may overlap their freed intermediates; ``release()`` before other long-lived device work."""

    def __init__(self, model: TtV41Transformer, state: V41PrefillState, buffers: dict | None = None, warmup: int = 2):
        assert state.generic, "a generic state (new_state(generic=True))"
        self.model, self.state, self.chunk = model, state, model.chunk
        self.buffers = {} if buffers is None else buffers
        self.controllers: dict = {}  # bucket (full chunks) or (bucket, valid length) (final chunks) -> capture
        self.scored: dict = {}  # per capture: the last valid row's device logits (fixed addresses)
        self.carry_after: dict = {}  # per final capture: the compressor-carry host state its replay leaves
        self.tokens = torch.randint(1, model.config.VOCAB_SIZE, (self.chunk,), dtype=torch.int64)
        self.captured_at: dict = {}
        self.warmup = warmup
        for key in [0] + list(state.tables.buckets):
            # the last full chunk (at a chunk boundary, not chunk 0) whose visible count falls in the bucket
            start = 0 if key == 0 else (key - self.chunk) // self.chunk * self.chunk
            if key and (start < self.chunk or state.tables.bucket(start, self.chunk) != key):
                continue  # no full chunk after chunk 0 falls in this bucket
            self._capture_at(key, start, self.chunk)
        self.last: dict = {}

    def _capture_at(self, key, start: int, length: int) -> None:
        """Warm up and capture the chunk at ``start`` with ``length`` valid tokens under ``key``."""
        t0 = time.perf_counter()
        for _ in range(self.warmup):
            self._one_chunk(start, length, lambda forward: forward())
        self.scored[key] = self._one_chunk(start, length, self._capture(key))
        self.captured_at[key] = start
        self.carry_after[key] = dict(self.state.compressor_carry)
        logger.info(
            f"generic prefill: {key} captured at chunk start {start} ({length} tokens) in "
            f"{time.perf_counter() - t0:.1f} s ({self.controllers[key].num_segments} segments)"
        )

    def final_key(self, total: int):
        """The key of the final-chunk capture a prompt of ``total`` tokens would replay (a partial final chunk, or
        a full one: the full-chunk capture serves that), and that chunk's start."""
        start = (total - 1) // self.chunk * self.chunk
        length = total - start
        if length == self.chunk or start == 0:
            return None, start
        return (self.state.tables.bucket(start, length), length), start

    def capture_final(self, total: int) -> None:
        """Capture the final chunk of prompts of ``total`` tokens (``total`` past the first chunk, not a multiple
        of it): generic in the chunk's start within its bucket, exact in its valid length (which sets the rows
        the chunk writes, the carry and the scored row). Replayed for every prompt whose final chunk has that
        bucket and length."""
        key, start = self.final_key(total)
        assert key is not None, f"{total} tokens: the final chunk is a full chunk, or the only chunk"
        if key not in self.controllers:
            self._capture_at(key, start, key[1])

    def _one_chunk(self, start: int, length: int, step: Callable):
        """Run one chunk of ``length`` synthetic tokens at ``start`` (a full chunk, or a partial final one) through
        the state: ``step(forward)``; the chunk's last valid row is scored."""
        model, state = self.model, self.state
        model.reset_state(state)
        state.start = start
        request = _Request(self.tokens[:length], length - 1, None, None)  # the chunk's last valid row scored
        host = next(model._host_chunks(request))
        host = _HostChunk(start, host.length, host.token_ids, host.engram, host.types)
        token_ids, engram_inputs = model._stage(host, self.buffers)
        state.tables.prepare_chunk(start, length)
        first_scored = start + length - 1
        rope = state.tables.dspark_rope(length) if model.dspark is not None else None  # the seed rows' pair
        out = step(lambda: model._chunk_forward(state, host, token_ids, engram_inputs, None, rope, first_scored, None))
        ttnn.synchronize_device(model.mesh_device)
        return out

    def _capture(self, key: int):
        def capture(forward: Callable):
            controller = SubDeviceTraceController(self.model.mesh_device)
            moes = [block.ffn for block in self.model.blocks]
            for moe in moes:
                moe.set_trace_controller(controller)
            ttnn.synchronize_device(self.model.mesh_device)
            controller.begin_capture()
            try:
                out = forward()
            finally:
                controller.end_capture()
                for moe in moes:
                    moe.set_trace_controller(None)
            self.controllers[key] = controller
            return out

        return capture

    @property
    def num_segments(self) -> int:
        return sum(c.num_segments for c in self.controllers.values())

    def key_of(self, c: int, total: int, logit_positions: int = 1):
        """The capture a prompt of ``total`` tokens replays for chunk ``c``: 0, a bucket, a (bucket, length) of a
        final chunk, or None (untraced: a final chunk without a capture, or one scoring more than its last row)."""
        start = c * self.chunk
        final = start + self.chunk >= total
        if final and logit_positions != 1:
            return None
        if start + self.chunk > total:  # a partial final chunk
            key, _ = self.final_key(total)
        else:
            key = 0 if c == 0 else self.state.tables.bucket(start, self.chunk)
        return key if key in self.controllers else None

    def run(self, tokens: torch.Tensor, logit_positions: int = 1, start: int = 0, end: int | None = None, resume=None):
        """tokens [S] (the whole prompt) -> (fp32 logits [logit_positions, vocab] of the position before ``end``,
        state) as ``prefill``; ``self.last`` the phases. ``start`` > 0 resumes the prompt at that chunk boundary
        (as ``prefill_into``); ``end`` (a chunk boundary, default the prompt's end) stops after that chunk, the
        next chunks' host inputs prepared ahead through ``resume`` (``TtV41Transformer._run``). A call with
        ``end`` before the prompt's end returns ``(None, state)`` without waiting for the device unless
        ``TT_V41_PREFILL_DRAIN=1`` (``PREFILL_DRAIN``)."""
        model, state = self.model, self.state
        total = int(tokens.numel())
        end = total if end is None else end
        assert logit_positions <= total - (total - 1) // self.chunk * self.chunk, "scored rows lie in the final chunk"
        request = _Request(tokens, end - logit_positions, None, None)
        t0 = time.perf_counter()
        if start == 0:
            model.reset_state(state)
        self.last = {"reset": time.perf_counter() - t0, "replayed": 0, "eager": 0}

        def step(c: int, forward: Callable):
            key = self.key_of(c, total, logit_positions)
            if key is None:
                self.last["eager"] += 1
                return forward()
            self.last["replayed"] += 1
            self.controllers[key].replay(blocking=False)
            if (c + 1) * self.chunk >= end:  # this call's last chunk: its logits, and the carry its replay leaves
                state.compressor_carry.update(self.carry_after[key])
                return self.scored[key]
            return []

        scored = model._run(request, state, self.buffers, step, timings=self.last, start=start, end=end, resume=resume)
        if end < total and not prefill_drain():  # an intermediate call: its replays are issued, not awaited
            self.last["wait"] = self.last["readback"] = 0.0
            return None, state
        t0 = time.perf_counter()
        ttnn.synchronize_device(model.mesh_device)
        self.last["wait"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        logits = model._logits(scored)
        self.last["readback"] = time.perf_counter() - t0
        return logits, state

    def release(self) -> None:
        for controller in self.controllers.values():
            controller.release()
        self.controllers = {}
