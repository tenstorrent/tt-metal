# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""TTNN ModernBERT masked-LM head.

    logits = decoder(norm(gelu(dense(last_hidden_state))))

`dense` and `norm` are bias-free (classifier_bias=False, norm_bias=False). The
`decoder` is the only layer in the model carrying a bias (decoder_bias=True); the
paper describes it as offsetting the downsides of weight tying, tie_word_embeddings
being True so decoder.weight is the token embedding matrix.

GELU runs in exact (erf) mode to match nn.GELU's default. tanh is interchangeable
here - it costs 5.9e-05 of MLM logit PCC (0.99369 vs 0.99363, top-1 8/8 either way)
- but this is a single op against the encoder's 22, so approximating it buys
nothing measurable.
"""

import ttnn
from models.experimental.modernbert.tt.model_config import compute_kernel_config


class TtnnModernBertPredictionHead:
    def __init__(self, parameters, config):
        self.dense = parameters["dense"]
        self.norm = parameters["norm"]
        self.eps = config.norm_eps
        self.compute_kernel_config = compute_kernel_config()

    def __call__(self, hidden_states):
        x = ttnn.linear(hidden_states, self.dense, compute_kernel_config=self.compute_kernel_config)
        activated = ttnn.gelu(x, fast_and_approximate_mode=False)
        ttnn.deallocate(x)
        out = ttnn.layer_norm(activated, weight=self.norm, epsilon=self.eps)
        ttnn.deallocate(activated)
        return out


class TtnnModernBertLMHead:
    """Prediction head followed by the tied decoder projection to vocab size."""

    def __init__(self, parameters, config):
        self.head = TtnnModernBertPredictionHead(parameters["head"], config)
        self.decoder_weight = parameters["decoder"]["weight"]
        self.decoder_bias = parameters["decoder"]["bias"]
        self.compute_kernel_config = compute_kernel_config()

    def __call__(self, last_hidden_state):
        """Returns logits of shape (B, S, vocab_size)."""
        pooled = self.head(last_hidden_state)
        logits = ttnn.linear(
            pooled,
            self.decoder_weight,
            bias=self.decoder_bias,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(pooled)
        return logits
