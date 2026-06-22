# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-layer completion sink for pipelined (multi-rank) prefill.

The runtime fires its `on_layer_complete(layer_idx)` callback once per layer, after
the KV cache write + pad-zero (see TtMLA). For a pipeline we cannot inject the
scheduler's counter channel directly from every rank — only the master rank may, and
only in global layer order. So each rank instead pushes a full completion message into
its host-local `ttnn.layer_completion.LayerCompletionQueue`; the
`ttnn.layer_completion.LayerCompletionRouter` forwards it to the master rank, which
re-emits completions strictly in ascending `seq` into the counter channel the scheduler
already connects to.

This module is intentionally dependency-light (no model/ttnn imports at module scope)
so the seq logic can be unit-tested in isolation.
"""

from typing import Callable


def build_layer_completion_sink(
    producer,
    *,
    source_rank: int,
    num_layers: int,
    get_request_id: Callable[[], int],
) -> Callable[[int], None]:
    """Build the per-layer callback the runtime fires once per layer.

    Computes a globally-dense ordering key and pushes a full completion into
    `producer` (a connected `ttnn.layer_completion.LayerCompletionQueue`).

    seq = request_id * num_layers + layer_idx

    `layer_idx` MUST be the GLOBAL layer index (it is, on a pipeline-sliced
    TtPrefillRuntime: the transformer builds blocks with
    layer_idx = first_layer_idx + local_idx, and MLA fires
    on_layer_complete(self.layer_idx)). `num_layers` MUST be the GLOBAL model layer
    total (NUM_LAYERS), NOT this rank's slice count — so that for chunk `c` the union
    of every rank's emitted layer indices tiles [c*num_layers, (c+1)*num_layers) with
    no gaps or collisions, and the master's reorder buffer drains a single dense cursor.

    `get_request_id` returns the current chunk index (a per-rank counter that aligns
    across ranks because chunks flow FIFO in identical order through every stage, and
    the warm-up chunk is processed before this sink is registered, so every rank starts
    the counter at 0 on the first real chunk).

    Args:
        producer: connected LayerCompletionQueue (the host-local ring).
        source_rank: this rank's world rank (diagnostic in the payload).
        num_layers: GLOBAL model layer total (the seq stride per chunk).
        get_request_id: zero-arg callable returning the current chunk index.
    """

    def on_layer_complete(layer_idx: int) -> None:
        request_id = get_request_id()
        seq = request_id * num_layers + layer_idx
        # Ring is sized well above in-flight depth; a full ring means the router thread
        # has stalled — surface it rather than drop a completion silently.
        if not producer.try_push(seq=seq, source_rank=source_rank, layer_idx=layer_idx, request_id=request_id):
            raise RuntimeError(f"layer-completion ring full (seq={seq}); router not draining")

    return on_layer_complete
