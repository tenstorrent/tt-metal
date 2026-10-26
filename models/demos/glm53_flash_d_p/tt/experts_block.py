# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3 MoE layers on the MoE block (models/demos/moe_block, GLM_EXPERTS_MODE=block): router, shared expert and
routed experts in one call, on the split residual as it is (MoEOptions.residual "owners": each chip's own S/n rows at
full width, row-major chip order - the block's token owners).

    x [S/n, H] -> rows gathered over each mesh row (TP), router partial logits on the chip's H/TP columns,
    reduce-scattered to the owners, top-k; the shared expert (TP-split) on the gathered rows, reduce-scattered to the
    owners -> the routing gather, planner, dispatch, flat experts + combine -> the owners' sums + the shared expert's
    -> [S/n, H]

One block per (mesh, chunk) serves every layer: built at the first call of that chunk (after the model's other ops
placed their L1: the exchange's arena address is baked into its kernels); each layer holds its own device weights
(MoELayer: router, shared expert, the flat experts from GLM's flat weight cache) and passes them with each call - one
weight set a layer, a plan per chunk (MoELayer.at).
"""

from __future__ import annotations

import os

import ttnn
from models.demos.glm53_flash_d_p.tt.experts_ag import flat_cache_dir

_BLOCKS = {}  # (mesh, chunk) -> MoEBlock
_DONE = {}  # mesh -> the done words every layer's flat expert shares


def _model():
    from models.demos.moe_block.moe_block import GLM_53_FLASH

    return GLM_53_FLASH


class TtExpertsBlock:
    """One GLM MoE layer on the shared MoE block: __call__(x) -> routed + shared [1, 1, S/n, H]."""

    routes_itself = True  # (the model's router step is a no-op: the block routes)

    def __init__(self, mesh, layer: int, loader, cfg, max_chunk: int, weights_dtype=ttnn.bfloat4_b):
        from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import FlatRoutedExpert

        from models.demos.glm53_flash_d_p.reference.weights import PREFIX
        from models.demos.moe_block.moe_block import (
            Geometry,
            MoEDtypes,
            MoELayer,
            MoEWeights,
            RoutedExperts,
            fuse_gate_up,
        )

        assert weights_dtype == ttnn.bfloat4_b, "GLM_EXPERTS_MODE=block: bfp4 routed experts"
        m = _model()
        assert (cfg.n_routed_experts, cfg.num_experts_per_tok, cfg.hidden_size, cfg.moe_intermediate_size) == (
            m.num_experts,
            m.top_k,
            m.emb_dim,
            m.routed_expert_hidden_dim,
        ), "GLM_EXPERTS_MODE=block: the block's GLM_53_FLASH dims"
        self.mesh, self.layer, self.chunk = mesh, layer, max_chunk
        self.links = int(os.environ.get("GLM_MOE_LINKS", "2"))
        self.g = Geometry.of(mesh, m, max_chunk, self.links, zigzag=False)
        self.dtypes = MoEDtypes()
        p = f"{PREFIX}layers.{layer}.mlp."
        w = lambda n: loader.weight(p + n).float()  # noqa: E731

        def host():  # (the router and shared expert, read when a head lays them out: once a head)
            return MoEWeights(
                router={"weight": w("gate.weight"), "bias": w("gate.e_score_correction_bias")},
                routed=None,
                shared={
                    "gate_up": fuse_gate_up(w("shared_experts.gate_proj.weight"), w("shared_experts.up_proj.weight")),
                    "down": w("shared_experts.down_proj.weight"),
                },
                latent=None,
            )

        fake = getattr(loader, "fake", False)  # (uninitialised routed experts: perf runs)
        prefix = None
        if not fake and os.environ.get("GLM_EXPERTS_CACHE", "1") != "0":
            d = flat_cache_dir(mesh)
            d.mkdir(parents=True, exist_ok=True)
            prefix = str(d / f"layer_{layer}")
        gids = RoutedExperts.gids_of(self.g)

        def routed_weights():  # (read on a weight cache miss only)
            t = lambda x: x.T.contiguous()  # noqa: E731
            q = f"{p}experts."
            return [
                [tuple(t(loader.weight(f"{q}{e}.{n}_proj.weight")) for n in ("gate", "up", "down")) for e in gl]
                for gl in gids
            ]

        flat = FlatRoutedExpert(
            mesh,
            "fake" if fake else routed_weights,
            m=RoutedExperts.cap_of(self.g),
            H=m.emb_dim,
            I=m.routed_expert_hidden_dim,
            gids=gids,
            n_global=m.num_experts,
            wdtype="bf4",
            act="clamped_silu",
            pin=1,
            cache_prefix=prefix,
            layout=RoutedExperts.layout_of(self.g, m),
        )
        # (one done-words buffer for every layer: the layers run one at a time and each launch leaves it zeroed)
        shared = _DONE.get(id(mesh))
        if shared is None or shared[0] is not mesh:
            _DONE[id(mesh)] = (mesh, flat.done)
        else:
            ttnn.deallocate(flat.done)
            flat.done = shared[1]
        self.weights = MoELayer.build(mesh, self.g, m, host, self.dtypes, routed_op=flat)
        self.at_chunk = {max_chunk: (self.g, self.weights)}
        self.shared = True  # (the model's shared-expert step folds into this call)

    def _layer(self, chunk):
        """(geometry, this layer's weights) at this chunk."""
        hit = self.at_chunk.get(chunk)
        if hit is None:
            from models.demos.moe_block.moe_block import Geometry

            g = Geometry.of(self.mesh, _model(), chunk, self.links, zigzag=False)
            hit = self.at_chunk[chunk] = (g, self.weights.at(self.mesh, g, _model()))
        return hit

    def _block(self, chunk, layer):
        from models.demos.moe_block.moe_block import MoEBlock, MoEOptions, WeightCache

        key = (id(self.mesh), chunk)
        hit = _BLOCKS.get(key)
        if hit is None or hit.mesh_device is not self.mesh:
            hit = MoEBlock(
                self.mesh,
                _model(),
                chunk,
                layer,
                WeightCache(path=None, layer=self.layer),
                num_links=self.links,
                dtypes=self.dtypes,
                options=MoEOptions(residual="owners"),
            )
            _BLOCKS[key] = hit
        return hit

    def __call__(self, x):
        """x [1, 1, S/n, H] (this chip's own rows, the split layout) -> routed + shared [1, 1, S/n, H] bf16 TILE."""
        chunk = int(x.shape[-2]) * self.mesh.get_num_devices()
        _, layer = self._layer(chunk)
        tmp = None
        if x.dtype != ttnn.bfloat16:
            x = tmp = ttnn.typecast(x, ttnn.bfloat16)
        y = self._block(chunk, layer)(x, layer=layer)
        if tmp is not None:
            ttnn.deallocate(tmp)
        return y


def build_experts_block(mesh, loader, cfg, layer: int, max_chunk: int, weights_dtype=ttnn.bfloat4_b):
    return TtExpertsBlock(mesh, layer, loader, cfg, max_chunk, weights_dtype=weights_dtype)
