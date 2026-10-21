# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""fp32 reference of a MiMo MTP layer (model.mtp.layers.i), as vLLM's ``MiMoV2MTP`` runs it.

HF's modeling code ignores the MTP weights, so vLLM's implementation is the reference: at position t the layer takes
the target's final normed hidden state at t and the embedding of the token at t + 1, projects
cat(enorm(embedding), hnorm(hidden)) with ``eh_proj``, runs a decoder block (SWA attention with sinks, a dense MLP)
and normalizes with its own ``final_layernorm``; the target's LM head turns that into the token at t + 2. Further
draft steps feed the layer its own output as the hidden state. The embedding and LM head are the target's.
"""

import torch

from .model import ReferenceDecoderLayer, load_layer_weights, rms_norm


def load_mtp_weights(ckpt, cfg, i):
    w = load_layer_weights(ckpt, cfg, cfg.mtp_layer_index(i))
    w["head"] = ckpt.mtp_head(i)
    return w


class ReferenceMTP:
    def __init__(self, cfg, i=0, ckpt=None, weights=None):
        """MTP layer ``i`` from ``ckpt``, or prepared ``weights`` (load_mtp_weights' keys) for synthetic tests."""
        self.cfg = cfg
        if weights is None:
            weights = load_mtp_weights(ckpt, cfg, i)
        self.head = weights["head"]
        self.layer = ReferenceDecoderLayer(cfg, cfg.mtp_layer_index(i), weights=weights)

    def __call__(self, hidden, embeds, positions, history=None):
        """hidden [S, hidden]: the target's (or this layer's) final normed hidden at ``positions`` [S]; embeds
        [S, hidden]: the embeddings of the tokens one position later. Returns the layer's final normed output
        [S, hidden]; its LM head logits predict the tokens two positions later."""
        eps = self.cfg.layernorm_epsilon
        e = rms_norm(embeds.float(), self.head["enorm"], eps)
        h = rms_norm(hidden.float(), self.head["hnorm"], eps)
        x = torch.cat((e, h), dim=-1) @ self.head["eh_proj"].T
        y = self.layer(x, positions, history)
        return rms_norm(y, self.head["final_layernorm"], eps)
