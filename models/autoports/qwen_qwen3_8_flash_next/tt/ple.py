# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device PLE arithmetic after the device n-gram/packed lookup boundary.

The decoder must run its device hash and packed lookup before invoking this
component. Supplying host-precomputed embeddings is a diagnostic boundary only.
Inputs: hidden [1,1,T,hc_count*hidden_size], embeddings [1,1,T,ple_embed_dim],
conv_state [1,1,9,hc_count*hidden_size] storing the previous normalized gated
values. The request owns conv_state; fresh requests initialize it to zero.
This component updates that existing state allocation in place and accepts any
positive logical T, including single-token decode. It contains no host reads or
runtime torch. Device parity/trace validation must be performed by its caller.
"""

import math

import ttnn

from .decoder_components import linear, rms


class PLE:
    def __init__(self, loader, config):
        self.width = config.hidden_size
        self.streams = config.hc_count
        self.eps = config.rms_norm_eps
        self.dilation = config.ngram_size
        self.kernel = config.ple_conv_kernel_size
        self.state_len = (self.kernel - 1) * self.dilation
        self.compute = loader.compute
        self.key = loader.matrix("ple.key_proj.weight")
        self.value = loader.matrix("ple.value_proj.weight")
        self.norm_key = loader.norm("ple.norm_key.weight", groups=self.streams)
        self.norm_query = loader.norm("ple.norm_query.weight", groups=self.streams)
        self.norm_conv = loader.norm("ple.norm_conv.weight", groups=self.streams)
        conv = loader.state_dict["ple.conv1d.weight"]
        self.conv = [loader.tensor(conv[:, 0, tap].reshape(1, 1, 1, -1)) for tap in range(self.kernel)]

    def _group(self, value):
        return ttnn.permute(ttnn.reshape(value, (1, value.shape[-2], self.streams, self.width)), (0, 2, 1, 3))

    def _flatten(self, value):
        return ttnn.reshape(ttnn.permute(value, (0, 2, 1, 3)), (1, 1, value.shape[-2], self.streams * self.width))

    def __call__(self, hidden, embeddings, *, conv_state, conv_mask=None):
        tokens = hidden.shape[-2]
        key = rms(self._group(linear(embeddings, self.key, self.compute)), self.norm_key, self.eps)
        query = rms(self._group(hidden), self.norm_query, self.eps)
        value = linear(embeddings, self.value, self.compute)
        # Match HF: its multiplication/reduction is BF16, nonlinear gate is
        # signed sqrt(|gate|). clamp applies to magnitude, sign(0) remains 0.
        gate = ttnn.multiply(ttnn.sum(ttnn.multiply(key, query), dim=-1, keepdim=True), 1 / math.sqrt(self.width))
        gate = ttnn.multiply(ttnn.sqrt(ttnn.clamp(ttnn.abs(gate), min=1e-6)), ttnn.sign(gate))
        gated = ttnn.multiply(ttnn.sigmoid(gate), value)
        normalized = self._flatten(rms(gated, self.norm_conv, self.eps))
        gated = self._flatten(gated)
        if conv_mask is not None:
            gated = ttnn.multiply(gated, conv_mask)
            normalized = ttnn.multiply(normalized, conv_mask)
        history = ttnn.concat((conv_state, normalized), dim=2)
        convolution = None
        for tap, weight in enumerate(self.conv):
            start = tap * self.dilation
            contribution = ttnn.multiply(history[:, :, start : start + tokens, :], weight)
            convolution = contribution if convolution is None else ttnn.add(convolution, contribution)
        ttnn.copy(history[:, :, -self.state_len :, :], conv_state)
        return ttnn.add(gated, ttnn.silu(convolution))
