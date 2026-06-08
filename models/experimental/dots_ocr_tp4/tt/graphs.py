# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``@trace_enabled`` graph wrappers for the dots.ocr TP4 text path.

These are the **trace boundaries**: invoking ``graph(...)`` routes through the
TTNNModule framework's ``TracedRun`` (1st call warms, 2nd captures, 3rd+ replays),
so the whole prefill / decode forward is captured as one ttnn trace and replayed
per token — removing host dispatch. The framework auto-allocates persistent input
buffers and ``ttnn.copy``s the (embedding, cache_position) tensors in on each
replay; the paged-cache object and the integer ``token_index`` pass through.

The graphs reference the already-built ``DotsOCRPrefillModelTP4`` (body + head);
their own weight lifecycle is a no-op (weights live on the referenced model). The
head uses the **distributed argmax** path (``forward_token_dist``) and returns the
per-chip ``(max_value, local_index)`` candidates as device tensors — the cheap
host combine happens OUTSIDE the trace (no host readback inside a captured region).
"""

import ttnn

from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.core.run_config import trace_enabled


def _update_kv_seq_len(text_model, past_key_value, seq_len):
    if past_key_value is None or not hasattr(past_key_value, "update_seq_length"):
        return
    for layer in text_model.layers:
        past_key_value.update_seq_length(layer_idx=layer.self_attn.layer_idx, seq_len=seq_len)


@trace_enabled
class DotsOCRPrefillGraphTP4(TTNNModule):
    """Traced prefill: decoder body (fills the paged KV cache) + distributed-argmax
    head at the real last position. Captured per input-embedding shape."""

    def preprocess_weights_impl(self):
        return self

    def move_weights_to_device_impl(self):
        return self

    def __init__(self, text_model):
        super().__init__()
        self._text = text_model

    def forward(self, embeds, past_key_value, token_index):
        hidden = self._text.forward(embeds, past_key_value=past_key_value)
        return self._text.head.forward_token_dist(hidden, token_index=token_index)

    def post_trace_execute(self, func_args, func_kwargs, result):
        seq_len = int(func_args[0].shape[-2])
        _update_kv_seq_len(self._text, func_kwargs.get("past_key_value"), seq_len)


@trace_enabled
class DotsOCRDecodeGraphTP4(TTNNModule):
    """Traced single-token decode: paged body + distributed-argmax head.

    ``decode_embed`` is the host-embedded ``[1, 1, H]`` replicated token embedding
    (tp4 embeds on host) and ``cache_position`` is the new token's int32 position;
    both are framework-buffered tensor args, so the same captured trace replays for
    every step with the new values copied in."""

    def preprocess_weights_impl(self):
        return self

    def move_weights_to_device_impl(self):
        return self

    def __init__(self, text_model):
        super().__init__()
        self._text = text_model

    def forward(self, decode_embed, cache_position, past_key_value):
        hidden = self._text.forward(decode_embed, past_key_value=past_key_value, cache_position=cache_position)
        return self._text.head.forward_token_dist(hidden, token_index=None)

    def post_trace_execute(self, func_args, func_kwargs, result):
        _update_kv_seq_len(self._text, func_kwargs.get("past_key_value"), 1)


@trace_enabled
class DotsOCRVisionGraphTP4(TTNNModule):
    """Traced vision trunk (42 blocks + post-trunk norm + patch merger).

    ``patch_embed`` runs OUTSIDE the trace (variable per image); this graph takes
    its output ``x`` (the only framework-buffered tensor input). The 2D RoPE is
    **prebuilt once** for the fixed 88x128 bucket and stored persistently on the
    graph, so the captured block loop reads stable rope buffers — avoiding the
    fresh per-call ``rope.build`` allocation (and the host ``grid_thw``) that would
    otherwise break trace capture/replay. Mirrors
    ``vision_tower_post_patch_embed_tp4_bh`` op-for-op with the rope hoisted out."""

    def preprocess_weights_impl(self):
        return self

    def move_weights_to_device_impl(self):
        return self

    def __init__(self, vision_tower, rot_mats, cu_seqlens):
        super().__init__()
        self._vt = vision_tower
        self._rot_mats = rot_mats  # prebuilt (fixed bucket), persistent -> trace-safe
        self._cu_seqlens = cu_seqlens

    def forward(self, x):
        from models.experimental.tt_symbiote.modules.vision_tp4_bh import (
            ensure_l1_tensor,
            vision_patch_merger_tp4_bh_forward,
        )

        x = ensure_l1_tensor(x, dtype=ttnn.bfloat8_b)
        if len(x.shape) == 3:
            x = ttnn.reshape(x, (1, 1, int(x.shape[1]), int(x.shape[2])), memory_config=ttnn.L1_MEMORY_CONFIG)
        for block in self._vt.blocks:
            x = block.forward(x, rot_mats=self._rot_mats, cu_seqlens=self._cu_seqlens)
        if self._vt.post_trunk_norm is not None:
            x = self._vt.post_trunk_norm(x, output_l1=True)
        if self._vt.patch_merger is not None:
            x = vision_patch_merger_tp4_bh_forward(self._vt.patch_merger, x)
        return ensure_l1_tensor(x)
