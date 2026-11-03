# SPDX-License-Identifier: Apache-2.0
"""One-row decode graph for a lone front-packed user of the fixed32 server.

It executes the standalone B1 decode code path (singleton norms, BF16 HC
partials, column-parallel output projections, no per-lane loops) against the owner's public state: request banks, paged QSA
caches, the PLE cache, the owner's sampler and output buffers. Only the extra
B1 weight representations and private B1 scratch are allocated here.

QWEN_FLASH_DECODE_B1=0 disables the view (base behaviour: the B4 view).
QWEN_FLASH_DECODE_B1_BF4=1 adds the standalone B1 BF4 QSA decode projection
(no measured gain at one QSA layer, ~10x larger logit KL), so BF8 is default.
"""

import os

import ttnn

from .decode_view import CompactDecodeView, _decoder, _ops
from .prefill_view import _clone


def enabled():
    return os.environ.get("QWEN_FLASH_DECODE_B1", "1") != "0"


def _column_weight(projection, ops):
    # Row shards hold W[:, r*1536:(r+1)*1536].T, so their concatenation is W.T.
    # BFP8 shares exponents per 16 output columns of one input row in both
    # layouts (640 = 40*16), so re-quantizing the column split is lossless.
    full = ttnn.to_torch(projection.row_weight, mesh_composer=ttnn.ConcatMeshToTensor(ops.mesh, dim=0))
    full = full.reshape(-1, full.shape[-1])
    return ops.shards([v.unsqueeze(0).unsqueeze(0) for v in full.chunk(4, -1)], ttnn.bfloat8_b)


def _model(owner, pool):
    if not owner.serving_pool or owner.batch_size != 32:
        raise ValueError("The one-row decode view requires a fixed32 serving owner")
    bf4 = os.environ.get("QWEN_FLASH_DECODE_B1_BF4", "0") == "1"
    result = _clone(owner, _ops(owner.ops))
    result.batch_size = 1
    result.terminal = _clone(owner.terminal, result.ops)
    result.layers = []
    for layer in owner.layers:
        view = _decoder(layer, 1, pool)
        attention = view.attention
        attention.output_projection.column_weight = _column_weight(attention.output_projection, view.ops)
        if layer.kind != "linear_attention":
            attention.decode_projection = (
                ttnn.typecast(layer.attention.packed_projection, ttnn.bfloat4_b)
                if bf4
                else layer.attention.packed_projection
            )
        result.layers.append(view)
    result.cache = result.export_cache()
    result._bound_cache = result.cache
    for original, view in zip(owner.layers, result.layers):
        public = (
            ("recurrent_bank", "conv_bank")
            if original.kind == "linear_attention"
            else ("packed_cache", "raw_key_cache", "pooled_cache")
        )
        for name in public:
            if getattr(original.attention, name) is not getattr(view.attention, name):
                raise RuntimeError(f"One-row decode lost public ownership of {name}")
        if original.ple is not None and original.ple_cache is not view.ple_cache:
            raise RuntimeError("One-row decode lost public PLE ownership")
        if original.attention.output_projection.column_weight is not None:
            raise RuntimeError("One-row decode expects a row-only fixed32 owner")
    if result.precision_policy is not owner.precision_policy or result.lm_head is not owner.lm_head:
        raise RuntimeError("One-row decode changed immutable owner policy or weights")
    return result


class SingleRowDecodeView(CompactDecodeView):
    """B1 execution of the compact view's input/logit bridge."""

    def __init__(self, generator, pool):
        self.generator = generator
        self.batch_size = 1
        self.model = _model(generator.model, pool)
        if "logit_rows" not in pool:
            pool["logit_rows"] = ttnn.zeros(
                tuple(generator.tt_logits.shape),
                dtype=generator.tt_logits.dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=generator.mesh,
            )
        self.rows = pool["logit_rows"]
