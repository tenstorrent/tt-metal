# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 prefill transformer and chunk driver (bead F9; dev-spec §4b).

Per request: allocate ``V41PrefillState``; for each chunk of ``chunk`` tokens (the last one padded, graph.md
rule 8): embed, expand to ``hc_mult`` streams, one-hot pre-mix, run the layers in order (DSpark taps at its
target layers), seed the state's DSpark window rings, advance the state. In the chunks holding scored
positions (the last real token by default): collapse the streams with the last block's pre, final norm, LM
head on those rows.

The layer list is any execution-ordered subset of V4.1 layers whose sources are present (a full model is
layers 0..39). Engram layers (1, 14) apply their n-gram update to the streams before their block; the hash
depends only on tokens and carries the last tokens across chunks.

Image spans (bead 10.2, graph V5; reference ``Transformer.forward`` / ``merge_image_embeddings``): with
``token_types`` (``image_processor`` TEXT / IMAGE_START / IMAGE / IMAGE_NEW_LINE / IMAGE_END per position) and one
aligner output per span (``TtV41Vision``), the first chunk's embedding rows of each span are overwritten: the
delimiters with the learned ``image_start`` / ``image_end`` / ``image_newline`` rows, the IMAGE slots with the
aligner rows in reading order. On device: a per-request table ``[delimiters | image rows]`` (each chip its TP slice)
is gathered by a per-position row index and selected into the embedding by the image mask (exact copies). Image
spans must lie in the first chunk (the reference asserts it). Image tokens route with each gate's ``bias_vl``
(``TtV41Moe``) and take no part in Engram n-grams and get no Engram update (hash and gate mask).
"""

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
from models.demos.deepseek_v3_d_p.tt.v41.head import TP_AXIS, TtV41Embedding, TtV41Head
from models.demos.deepseek_v3_d_p.tt.v41.mhc import initial_pre_mix
from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import init_checker

# merge-table rows of the span delimiters (checkpoint names), in table order; the image rows follow them
_DELIMITERS = ("image_start", "image_end", "image_newline")
_DELIMITER_ROW = {IMAGE_START: 0, IMAGE_END: 1, IMAGE_NEW_LINE: 2}


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
            marker = None if weight_cache_path is None else weight_cache_path / f"layer_{layer}.complete"
            cached = marker is not None and marker.exists()
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
            if marker is not None:
                marker.touch()
        self.head = TtV41Head(mesh_device, config, norm_weight, head_weight)
        self.dspark = None
        if dspark_weights is not None:
            missing = set(config.DSPARK_TARGET_LAYER_IDS) - set(self.layers)
            assert not missing, f"DSpark needs the taps of layers {sorted(missing)}"
            self.dspark = TtV41DSpark(mesh_device, config, dspark_weights)

    def _token_ids(self, tokens: torch.Tensor):
        """[chunk] int -> [1, 1, chunk/sp] uint32 SP-sharded, replicated over TP."""
        return ttnn.from_torch(
            tokens.to(torch.int32).view(1, 1, -1),
            device=self.mesh_device,
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
        owns; all spans must lie in the first chunk."""
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
        first_scored = total - logit_positions
        state = V41PrefillState(self.mesh_device, self.config, self.max_seq_len, self.chunk, self.layers)
        if self.dspark is not None:
            state.dspark_rings = self.dspark.new_rings()
        logits = []
        history = self.engram_hash.new_history() if self.engram else None
        while state.start < total:
            start = state.start
            length = min(self.chunk, total - start)
            chunk_tokens = torch.zeros(self.chunk, dtype=torch.int64)
            chunk_tokens[:length] = tokens[start : start + length]
            # image spans lie in the first chunk: the other chunks run the text path
            types = token_types[:length] if token_types is not None and start == 0 else None
            text = None if types is None else types == TEXT
            engram_inputs = {}
            if self.engram:
                ids, history = self.engram_hash(tokens[start : start + length], history, text)
                engram_inputs = {
                    l: e.prepare(ids[:, self.engram_hash.layer_index(l)], self.chunk, text)
                    for l, e in self.engram.items()
                }
            h = self.embedding(self._token_ids(chunk_tokens))
            image_rows = None
            if types is not None:
                padded = torch.full((self.chunk,), TEXT, dtype=torch.int64)
                padded[:length] = types
                image_rows = self._column(padded != TEXT)
                h = self._merge_images(h, padded, image_features, image_rows)
            x = mhc_expand(ttnn.typecast(h, ttnn.float32), self.config.HC_MULT)
            pre = initial_pre_mix(self.mesh_device, self.config, self.chunk)
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
                self.dspark.seed(taps, start, length, state.dspark_rings)
            if start + length > first_scored:
                final = ttnn.typecast(self.blocks[-1].residual.final_collapse(x, pre), ttnn.bfloat16)
                rows = list(range(max(first_scored - start, 0), length))
                logits.append(self.head.rows_to_host(final, rows))
            state.advance(length)
        return torch.cat(logits), state
