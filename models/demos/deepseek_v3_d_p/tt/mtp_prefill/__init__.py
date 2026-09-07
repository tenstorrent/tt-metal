# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tenstorrent device modules for GLM-5.2 Multi-Token Prediction (MTP) during PREFILL.

An MTP module predicts token ``t_{p+k}`` from position ``p``. It is a fused input projection over
the shifted token embedding and the previous level's hidden state, followed by one *complete* GLM
decoder layer:

    x^k = eh_proj( cat[ enorm(embed(t_{p+k})) , hnorm(h^{k-1}[p]) ] )    <- TtFusedMTP
    h^k = GLM_decoder_layer(x^k)                                          <- TtPrefillBlock
    out = shared_head.norm(h^k)                                           <- TtMTPModule

``TtFusedMTP`` owns only the input-side projection; ``TtMTPModule`` wraps it together with one
``TtPrefillBlock`` and the output-side ``shared_head.norm``; ``TtMTPPredictor`` replays one such
module across ``num_levels`` levels, each writing its own KV cache slot.

GLM-5.2 uses the DeepSeek-V3 paper scheme: K levels predicted at ONE position, K KV caches, ONE
shared set of weight modules. It is NOT EAGLE-style autoregressive drafting. ``num_nextn_predict_layers``
in the checkpoint config counts weight *modules* (1), not prediction *levels* — see
``MTPConfig.num_weight_modules`` vs ``num_levels`` in ``mtp_config.py``.

The MTP weights live on layer ``num_hidden_layers`` (78), which is a full-size 256-expert MoE
decoder layer carrying its own MLA *and* indexer weights. See issue #53533 / tt-blaze#1674.
"""
