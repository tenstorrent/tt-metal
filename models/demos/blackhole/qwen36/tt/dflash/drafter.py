# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""M8: the DFlash drafter -- context encoder, 5 decoder layers, final norm.

    target_hidden -> fc -> hidden_norm ----.
                                            \\  (context K/V, per layer)
    [anchor, MASK x15] embeddings -----------+--> 5 layers --> norm --> hidden

Returns the ``[1, 1, block, 5120]`` hidden state. Turning that into candidate tokens needs
the **target's** LM head over the last ``block - 1`` rows -- the drafter ships neither an
embedding table nor an LM head, so both ends of it belong to the target and stay outside
this module.

One forward pass drafts the whole block; there is no iterative denoising loop.

Nothing in ``forward`` touches host. The RoPE table is built once at construction and
sliced on device; masks are built once per distinct context length and cached (see
``_mask_for``). Weights are loaded at construction.
"""

from __future__ import annotations

import ttnn
from models.demos.blackhole.qwen36.tt.dflash import mask as dflash_mask
from models.demos.blackhole.qwen36.tt.dflash.ccl import ccl_topology
from models.demos.blackhole.qwen36.tt.dflash.config import DFlashDrafterConfig
from models.demos.blackhole.qwen36.tt.dflash.encoder import DFlashContextEncoder
from models.demos.blackhole.qwen36.tt.dflash.layer import DFlashLayer
from models.demos.blackhole.qwen36.tt.dflash.rms_norm import rms_norm
from models.demos.blackhole.qwen36.tt.dflash.rope import DFlashRoPE
from models.demos.blackhole.qwen36.tt.dflash.weights import DFlashWeights, load_weights
from models.tt_transformers.tt.ccl import TT_CCL

_MC = ttnn.DRAM_MEMORY_CONFIG


class DFlashDrafter:
    """The whole drafter on one mesh."""

    def __init__(
        self,
        mesh_device,
        cfg: DFlashDrafterConfig,
        max_position: int,
        weights: DFlashWeights | None = None,
        checkpoint_path: str | None = None,
        tt_ccl=None,
        topology=None,
    ):
        self.mesh_device = mesh_device
        self.cfg = cfg
        self.tt_ccl = tt_ccl if tt_ccl is not None else TT_CCL(mesh_device)
        self.topology = topology if topology is not None else ccl_topology(mesh_device)

        self.weights = weights if weights is not None else load_weights(mesh_device, cfg, checkpoint_path)
        self.encoder = DFlashContextEncoder(mesh_device, cfg, self.weights, self.tt_ccl, self.topology)
        self.layers = [
            DFlashLayer(mesh_device, cfg, self.weights.layers[i], i, self.tt_ccl, self.topology)
            for i in range(cfg.num_hidden_layers)
        ]
        self.rope = DFlashRoPE(mesh_device, cfg, max_position=max_position)

        # Masks depend only on (layer kind, ctx_len, block_len), so they are pure functions
        # of shape. Cached per key and built on first use; a production loop should pre-warm
        # the context buckets (config.CONTEXT_BUCKETS) at startup so steady-state drafting
        # never builds one -- the inference path must not touch host.
        self._masks: dict[tuple[bool, int, int], object] = {}

    def prewarm_masks(self, ctx_lens, block_len: int | None = None) -> None:
        """Build and cache masks for the given context lengths ahead of time."""
        block_len = block_len or self.cfg.block_size
        for ctx_len in ctx_lens:
            for layer_idx in range(self.cfg.num_hidden_layers):
                self._mask_for(layer_idx, ctx_len, block_len)

    def _mask_for(self, layer_idx: int, ctx_len: int, block_len: int):
        # Keyed on the layer *kind*, not the index: all 4 sliding layers share one mask.
        key = (self.cfg.is_sliding(layer_idx), ctx_len, block_len)
        if key not in self._masks:
            self._masks[key] = dflash_mask.to_device(
                self.cfg, layer_idx, self.mesh_device, ctx_len=ctx_len, block_len=block_len
            )
        return self._masks[key]

    def forward(self, target_hidden, noise_embedding, ctx_start: int = 0):
        """Draft one block.

        Args:
            target_hidden: ``[1, 1, ctx, 25600/tp]``, TP-sharded on the tap axis -- the
                target's residual stream at ``target_layer_ids``, permuted to the device tap
                layout (``weights.fc_input_permutation``).
            noise_embedding: ``[1, 1, block, 5120]`` replicated -- the target's embeddings of
                ``[anchor, MASK x (block-1)]``.
            ctx_start: absolute position of the first context token. RoPE is absolute, so
                this is what lets a cached/truncated context keep correct positions.

        Returns:
            ``[1, 1, block, 5120]`` replicated hidden. Feed the last ``block - 1`` rows to
            the target's LM head to get candidates.
        """
        cfg = self.cfg
        ctx_len = target_hidden.shape[-2]
        block_len = noise_embedding.shape[-2]

        # Shared by every layer: computed once, not per layer.
        encoded = self.encoder.forward(target_hidden)

        # q takes the TAIL of the position range, k takes all of it (see rope.py).
        cos_k, sin_k = self.rope.tables_for(ctx_start, ctx_len + block_len)
        cos_q, sin_q = self.rope.tables_for(ctx_start + ctx_len, block_len)

        x = noise_embedding
        for layer_idx, layer in enumerate(self.layers):
            ctx_k, ctx_v = layer.project_context(encoded)
            attn_mask = self._mask_for(layer_idx, ctx_len, block_len)
            x = layer.forward(x, ctx_k, ctx_v, self.rope, cos_q, sin_q, cos_k, sin_k, attn_mask)
            ttnn.deallocate(ctx_k)
            ttnn.deallocate(ctx_v)

        ttnn.deallocate(encoded)
        return rms_norm(x, self.weights.norm, cfg.rms_norm_eps, memory_config=_MC)
