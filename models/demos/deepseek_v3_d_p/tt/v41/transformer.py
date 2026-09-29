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

from dataclasses import dataclass
from typing import Callable

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.reference.deepseek_v41.image_processor import (
    IMAGE,
    IMAGE_END,
    IMAGE_NEW_LINE,
    IMAGE_START,
    TEXT,
)
from models.demos.deepseek_v3_d_p.tt.mhc.tt_mhc import mhc_expand
from models.demos.deepseek_v3_d_p.tt.v41.block import TtV41Block
from models.demos.deepseek_v3_d_p.tt.v41.cache import V41PrefillState
from models.demos.deepseek_v3_d_p.tt.v41.dspark import TtV41DSpark
from models.demos.deepseek_v3_d_p.tt.v41.engram import EngramChunkInputs
from models.demos.deepseek_v3_d_p.tt.v41.head import TP_AXIS, TtV41Embedding, TtV41Head
from models.demos.deepseek_v3_d_p.tt.v41.mhc import initial_pre_mix
from models.demos.deepseek_v3_d_p.tt.v41.weights import begin_layer, complete_layer
from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import init_checker
from models.demos.deepseek_v3_d_p.utils.sub_device_trace import SubDeviceTraceController

# merge-table rows of the span delimiters (checkpoint names), in table order; the image rows follow them
_DELIMITERS = ("image_start", "image_end", "image_newline")
_DELIMITER_ROW = {IMAGE_START: 0, IMAGE_END: 1, IMAGE_NEW_LINE: 2}


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
        routed_expert_weights_dtype=ttnn.bfloat8_b,
    ):
        """``layer_weights(layer, include_moe)`` returns that block's ``TtV41Block`` weights (``weights.load_layer``
        for a checkpoint), without the MoE entries when ``include_moe`` is False; it is called once per layer, so
        host copies are dropped as blocks are built. ``weight_cache_path``: device MoE tensors are converted once
        and loaded from there afterwards (a marker records each completed layer; key the directory by weight
        identity, e.g. checkpoint revision + dtype + mesh). ``image_embeds``: the learned span delimiters
        ``image_start`` / ``image_end`` / ``image_newline`` (``[hidden]`` bf16, checkpoint names) for prompts with
        images; each layer's ``gate_bias_vl`` comes with its ``layer_weights``."""
        self.mesh_device, self.config = mesh_device, config
        self.layers, self.chunk, self.max_seq_len = list(layers), chunk, max_seq_len
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

    def _new_state(self) -> V41PrefillState:
        state = V41PrefillState(self.mesh_device, self.config, self.max_seq_len, self.chunk, self.layers)
        if self.dspark is not None:
            state.dspark_rings = self.dspark.new_rings()
        return state

    def _host_chunks(self, request: _Request):
        """Each chunk's host inputs in order: token ids and (Engram layers) the n-gram hash and the packed rows /
        row ids of its lookups. Tokens-only, so the loop prepares chunk i + 1 while the device runs chunk i."""
        tokens, total = request.tokens, int(request.tokens.numel())
        history = self.engram_hash.new_history() if self.engram else None
        for start in range(0, total, self.chunk):
            length = min(self.chunk, total - start)
            chunk_tokens = torch.zeros(self.chunk, dtype=torch.int64)
            chunk_tokens[:length] = tokens[start : start + length]
            # image spans lie in the first chunk: the other chunks run the text path
            types = request.token_types[:length] if request.token_types is not None and start == 0 else None
            text = None if types is None else types == TEXT
            engram = {}
            if self.engram:
                ids, history = self.engram_hash(tokens[start : start + length], history, text)
                engram = {
                    l: e.prepare_host(ids[:, self.engram_hash.layer_index(l)], self.chunk, text)
                    for l, e in self.engram.items()
                }
            padded = None
            if types is not None:
                padded = torch.full((self.chunk,), TEXT, dtype=torch.int64)
                padded[:length] = types
            yield _HostChunk(start, length, self._token_ids(chunk_tokens, device=False), engram, padded)

    def _stage(self, host: _HostChunk, buffers: dict):
        """Write a chunk's host inputs into the request's device input buffers (allocated by its first chunk, so
        every chunk's forward reads fixed addresses) -> (token ids, Engram inputs) on device."""
        if "token_ids" not in buffers:
            buffers["token_ids"] = ttnn.to_device(host.token_ids, self.mesh_device)
            buffers["engram"] = {l: e.to_device(self.mesh_device) for l, e in host.engram.items()}
            return buffers["token_ids"], dict(buffers["engram"])
        ttnn.copy_host_to_device_tensor(host.token_ids, buffers["token_ids"])
        return buffers["token_ids"], {l: e.copy_into(buffers["engram"][l]) for l, e in host.engram.items()}

    def _chunk_forward(
        self, state, host: _HostChunk, token_ids, engram_inputs, image_features, rope, first_scored, on_block
    ):
        """One chunk on device -> its scored rows' device logits, [(logits, sp_row, first row of the tile, chunk rows
        read from it)]. A text chunk without ``on_block`` makes no host transfers (the image merge uploads its
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
        for layer, block in zip(self.layers, self.blocks):
            if layer in self.engram:
                x = self.engram[layer](x, engram_inputs[layer])
            if self.dspark is not None and layer in self.config.DSPARK_TARGET_LAYER_IDS:
                taps.append(self.dspark.tap(x))
            x, pre = block(x, pre, state, length, image_rows)
            if on_block is not None:
                on_block(layer, x, start, length)
        if image_rows is not None:
            ttnn.deallocate(image_rows)
        if self.dspark is not None:
            self.dspark.seed(taps, start, length, state.dspark_rings, rope)
        scored = []
        if start + length > first_scored:
            final = ttnn.typecast(self.blocks[-1].residual.final_collapse(x, pre), ttnn.bfloat16)
            rows = list(range(max(first_scored - start, 0), length))
            while rows:  # one head projection per distinct 32-row tile (a tile never spans chips)
                logits, (sp_row, offset) = self.head(final, rows[0])
                first = rows[0] - offset
                tile = [r for r in rows if r < first + ttnn.TILE_SIZE]
                scored.append((logits, sp_row, first, tile))
                rows = rows[len(tile) :]
        return scored

    def _run(self, request: _Request, state, buffers: dict, step: Callable, on_block=None) -> list:
        """The chunk loop of one request. ``step(c, forward)`` runs chunk ``c``'s device forward: ``forward()``
        (untraced or captured) or a replay of its capture. Returns every chunk's scored device logits."""
        total = int(request.tokens.numel())
        if self.dspark is not None and "dspark_rope" not in buffers:
            buffers["dspark_rope"] = [
                self.dspark.seed_rope(s, min(self.chunk, total - s)) for s in range(0, total, self.chunk)
            ]
        chunks, scored = self._host_chunks(request), []
        host = next(chunks)
        for c in range(-(-total // self.chunk)):
            token_ids, engram_inputs = self._stage(host, buffers)
            rope = buffers["dspark_rope"][c] if self.dspark is not None else None
            forward = lambda: self._chunk_forward(
                state, host, token_ids, engram_inputs, request.image_features, rope, request.first_scored, on_block
            )
            scored += step(c, forward)
            length = host.length
            host = next(chunks, None)  # the next chunk's host inputs, while the device runs this chunk
            state.advance(length)
        return scored

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
        state = self._new_state()
        scored = self._run(request, state, {}, lambda c, forward: forward(), on_block)
        return self._logits(scored), state


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

    def __init__(self, model: TtV41Transformer, tokens: torch.Tensor, logit_positions: int = 1):
        """``tokens``: a prompt of the traced length (warm-up and capture run it; Engram row subsets must hold its
        rows)."""
        self.model, self.total, self.logit_positions = model, int(tokens.numel()), logit_positions
        request = model._request(tokens, logit_positions)
        self.buffers = {}
        for _ in range(2):  # compile, then steady-state persistent resources (semaphores, constants)
            model._run(request, model._new_state(), self.buffers, lambda c, forward: forward())
        ttnn.synchronize_device(model.mesh_device)
        self.state, self.controllers = model._new_state(), []
        self._zero_hosts = {}
        self.scored = model._run(request, self.state, self.buffers, self._capture)

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
        self.state.start = 0
        # the state a fresh request starts from, where the first chunk reads it: zero window carries (the window
        # slot rows before position 0) and DSpark rings (slots a short prompt does not reach)
        for t in list(self.state.window_carry.values()) + list(self.state.dspark_rings or ()):
            ttnn.copy_host_to_device_tensor(self._zeros(t), t)
        model._run(
            request, self.state, self.buffers, lambda c, forward: self.controllers[c].replay(blocking=False) or []
        )
        ttnn.synchronize_device(model.mesh_device)
        return model._logits(self.scored), self.state

    def _zeros(self, t: ttnn.Tensor) -> ttnn.Tensor:
        """A replicated all-zero host tensor of ``t``'s shape, dtype and layout (cached per spec)."""
        key = (tuple(t.shape), t.dtype, t.layout)
        if key not in self._zero_hosts:
            self._zero_hosts[key] = ttnn.from_torch(
                torch.zeros(key[0]),
                dtype=t.dtype,
                layout=t.layout,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.model.mesh_device),
            )
        return self._zero_hosts[key]

    def release(self) -> None:
        for controller in self.controllers:
            controller.release()
        self.controllers = []
