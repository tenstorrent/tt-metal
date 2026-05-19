# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Auto-infer transformer-block boundaries from the Tracy op stream.

The point of this module is to produce per-block utilization-vs-runtime plots
WITHOUT requiring the demo to emit `tracy.block_scope` signposts (which
`simple_text_demo.py` does not today).

Strategy (in order of preference; first one that succeeds wins):

  1. Anchor-op segmentation. Each transformer layer fires exactly one of a
     small set of "anchor" ops (attention SDPA / kv-cache update / concat-
     heads-decode / paged-update-cache). We count anchor occurrences, expect
     them to be a multiple of `num_hidden_layers`, and use the count to
     stamp every row between anchor_i and anchor_(i+1) with layer `i % N`.

  2. Even-segmentation fallback. If no anchor is reliable, sort rows by
     `global_call_count` and divide into N equal-size chunks. Less precise
     (prefill ops contaminate the bins), but always populates something
     reasonable for the chart so the framework degrades gracefully.

  3. No-op. If `num_hidden_layers` is unknown (HF probe failed at collect
     time), every row keeps its current `block_path = "root"` and the chart
     falls back to the existing "no signposts" message.

Both strategies are model-agnostic. They infer structure from the op stream
alone — same logic works for Qwen, Llama, Mistral, Gemma, etc.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple


# Ops that fire exactly once per transformer layer in `tt_transformers`.
# Order matters: the first anchor whose count is a positive multiple of
# `num_hidden_layers` wins. SDPA-decode is the most reliable because every
# layer's attention emits exactly one.
ANCHOR_OPS_BY_PRIORITY: Tuple[str, ...] = (
    "ttnn.transformer.sdpa_decode",
    "ttnn.transformer.scaled_dot_product_attention_decode",
    "ttnn.scaled_dot_product_attention_decode",
    "ttnn.experimental.paged_update_cache",
    "ttnn.experimental.nlp_concat_heads_decode",
    "ttnn.nlp_concat_heads_decode",
)


def _normalize(op_code: str) -> str:
    """Drop the leading namespace difference between Tracy CSV op codes
    (which may use `ttnn::experimental::foo`) and Python op names
    (`ttnn.experimental.foo`)."""
    return op_code.replace("::", ".")


def _find_anchor_op(op_codes: List[str], num_hidden_layers: int) -> Optional[str]:
    """Return the first anchor whose occurrence count is a positive multiple
    of `num_hidden_layers`. None if no anchor is suitable."""
    if num_hidden_layers <= 0:
        return None
    seen = [_normalize(c) for c in op_codes]
    for anchor in ANCHOR_OPS_BY_PRIORITY:
        # Allow substring match: Tracy sometimes prefixes with `ttnn.` and
        # sometimes with the module path. Substring is good enough because
        # the anchors are unambiguous.
        count = sum(1 for code in seen if anchor in code)
        if count > 0 and count % num_hidden_layers == 0:
            return anchor
    return None


def infer_block_paths(
    op_codes: List[str],
    num_hidden_layers: Optional[int],
) -> List[str]:
    """For a time-ordered list of op codes, return a parallel list of
    `block_path` strings.

    The output is the same length as the input. Rows belonging to layer i
    get `block_path = f"decoder.layers.{i:02d}"`. Rows that couldn't be
    classified (e.g. lead-in / tear-down ops before the first anchor) get
    `block_path = "root"` so the existing aggregation continues to work.
    """
    n = len(op_codes)
    if n == 0:
        return []

    if not num_hidden_layers or num_hidden_layers <= 0:
        return ["root"] * n

    # Strategy 1: anchor-op segmentation.
    anchor = _find_anchor_op(op_codes, num_hidden_layers)
    if anchor is not None:
        return _segment_by_anchor(op_codes, anchor, num_hidden_layers)

    # Strategy 2: even-size fallback over the entire op stream.
    return _segment_evenly(n, num_hidden_layers)


def _segment_by_anchor(op_codes: List[str], anchor: str, num_hidden_layers: int) -> List[str]:
    """Stamp each row's block_path using anchor-op positions as boundaries.

    Pre-anchor rows (model warmup / first prefill block before any layer
    has fired) get `"root"`. Subsequent rows are labeled by the running
    anchor-encounter modulo `num_hidden_layers`.
    """
    out: List[str] = []
    layer_idx = -1  # bumps to 0 on the first anchor encounter
    for code in op_codes:
        if anchor in _normalize(code):
            layer_idx += 1
            out.append(_layer_label(layer_idx, num_hidden_layers))
        else:
            if layer_idx < 0:
                out.append("root")
            else:
                out.append(_layer_label(layer_idx, num_hidden_layers))
    return out


def _segment_evenly(n: int, num_hidden_layers: int) -> List[str]:
    """Equal-size fallback bucketing. Always produces N populated layers."""
    if num_hidden_layers <= 0 or n <= 0:
        return ["root"] * n
    chunk = max(1, n // num_hidden_layers)
    out: List[str] = []
    for i in range(n):
        layer = min(i // chunk, num_hidden_layers - 1)
        out.append(_layer_label(layer, num_hidden_layers))
    return out


def _layer_label(layer_idx: int, num_hidden_layers: int) -> str:
    """Two-digit zero-padded labels so a 100-layer model sorts correctly."""
    width = max(2, len(str(max(0, num_hidden_layers - 1))))
    return f"decoder.layers.{layer_idx % num_hidden_layers:0{width}d}"


def summarize_block_inference(block_paths: List[str], num_hidden_layers: Optional[int]) -> Dict[str, object]:
    """One-line stats so the report can print 'inferred 28 blocks from N rows'."""
    populated = {b for b in block_paths if b != "root"}
    return {
        "num_hidden_layers_from_config": num_hidden_layers,
        "num_blocks_populated": len(populated),
        "num_rows_in_root": sum(1 for b in block_paths if b == "root"),
        "num_rows_in_blocks": sum(1 for b in block_paths if b != "root"),
    }
