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
depends only on tokens and carries the last tokens across chunks. Text only (no image mask yet, 10.2).
"""

from typing import Callable

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.tt.mhc.tt_mhc import mhc_expand
from models.demos.deepseek_v3_d_p.tt.v41.block import TtV41Block
from models.demos.deepseek_v3_d_p.tt.v41.cache import V41PrefillState
from models.demos.deepseek_v3_d_p.tt.v41.dspark import TtV41DSpark
from models.demos.deepseek_v3_d_p.tt.v41.head import TtV41Embedding, TtV41Head
from models.demos.deepseek_v3_d_p.tt.v41.mhc import initial_pre_mix
from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import init_checker


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
        weight_cache_path=None,
        topology=ttnn.Topology.Linear,
        routed_expert_weights_dtype=ttnn.bfloat8_b,
    ):
        """``layer_weights(layer, include_moe)`` returns that block's ``TtV41Block`` weights (``weights.load_layer``
        for a checkpoint), without the MoE entries when ``include_moe`` is False; it is called once per layer, so
        host copies are dropped as blocks are built. ``weight_cache_path``: device MoE tensors are converted once
        and loaded from there afterwards (a marker records each completed layer; key the directory by weight
        identity, e.g. checkpoint revision + dtype + mesh)."""
        self.mesh_device, self.config = mesh_device, config
        self.layers, self.chunk, self.max_seq_len = list(layers), chunk, max_seq_len
        assert self.layers == sorted(set(self.layers)), "layers must be in execution order"
        needed = [l for l in self.layers if l in config.ENGRAM_LAYER_IDS]
        self.engram = dict(engram or {})
        assert sorted(self.engram) == needed, f"Engram modules for layers {needed} required, got {sorted(self.engram)}"
        assert not needed or engram_hash is not None, "Engram layers need the n-gram hasher"
        self.engram_hash = engram_hash
        self.embedding = TtV41Embedding(mesh_device, config, embed_weight)
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
                    topology=topology,
                    routed_expert_weights_dtype=routed_expert_weights_dtype,
                    weight_cache_path=weight_cache_path,
                )
            )
            if marker is not None:
                marker.touch()
        self.head = TtV41Head(mesh_device, config, norm_weight, head_weight, topology=topology)
        self.dspark = None
        if dspark_weights is not None:
            missing = set(config.DSPARK_TARGET_LAYER_IDS) - set(self.layers)
            assert not missing, f"DSpark needs the taps of layers {sorted(missing)}"
            self.dspark = TtV41DSpark(mesh_device, config, dspark_weights, topology)

    def _token_ids(self, tokens: torch.Tensor):
        """[chunk] int -> [1, 1, chunk/sp] uint32 SP-sharded, replicated over TP."""
        return ttnn.from_torch(
            tokens.to(torch.int32).view(1, 1, -1),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, tuple(self.mesh_device.shape), dims=(2, None)),
        )

    def prefill(self, tokens: torch.Tensor, logit_positions: int = 1):
        """tokens [S] (S <= max_seq_len) -> (fp32 logits [logit_positions, vocab] of the last ``logit_positions``
        prompt positions, state); the state holds the caches, carries and (with DSpark) the seeded DSpark
        window rings. Generation needs the last position only; more positions score the prompt itself."""
        total = int(tokens.numel())
        assert 0 < total <= self.max_seq_len, f"prompt of {total} tokens, max {self.max_seq_len}"
        assert 0 < logit_positions <= total, f"logit_positions {logit_positions} outside the {total}-token prompt"
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
            engram_inputs = {}
            if self.engram:
                ids, history = self.engram_hash(tokens[start : start + length], history)
                engram_inputs = {
                    l: e.prepare(ids[:, self.engram_hash.layer_index(l)], self.chunk) for l, e in self.engram.items()
                }
            h = self.embedding(self._token_ids(chunk_tokens))
            x = mhc_expand(ttnn.typecast(h, ttnn.float32), self.config.HC_MULT)
            pre = initial_pre_mix(self.mesh_device, self.config, self.chunk)
            taps = []
            for layer, block in zip(self.layers, self.blocks):
                if layer in self.engram:
                    x = self.engram[layer](x, engram_inputs[layer])
                if self.dspark is not None and layer in self.config.DSPARK_TARGET_LAYER_IDS:
                    taps.append(self.dspark.tap(x))
                x, pre = block(x, pre, state, length)
            if self.dspark is not None:
                self.dspark.seed(taps, start, length, state.dspark_rings)
            if start + length > first_scored:
                final = ttnn.typecast(self.blocks[-1].residual.final_collapse(x, pre), ttnn.bfloat16)
                rows = list(range(max(first_scored - start, 0), length))
                logits.append(self.head.rows_to_host(final, rows))
            state.advance(length)
        return torch.cat(logits), state
