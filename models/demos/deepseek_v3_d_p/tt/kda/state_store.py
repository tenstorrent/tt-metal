# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-layer ownership of the carries a KDA prefill stream walks forward.

``ttKDA.forward`` reads a state and hands back its replacement, retaining nothing, so
something outside the layer has to hold them. That something is this: the KDA counterpart
of ``MlaKvCache``. A KDA layer writes no KV slab, and these carries are what a later chunk
continues the recurrence from and what a chunked prefill eventually hands to decode.

Keyed by GLOBAL layer index, matching the block's own ``layer_idx``, so a pipeline rank
holding a layer slice needs no second numbering.
"""

from __future__ import annotations

import ttnn
from models.demos.deepseek_v3_d_p.tt.kda.kda import KdaState, ttKDA


class KdaStateStore:
    """The live KDA carries for one prefill stream, one entry per KDA layer."""

    def __init__(self, layers: dict[int, ttKDA], batch_size: int = 1) -> None:
        self._layers = dict(layers)
        self._batch_size = batch_size
        self._states: dict[int, KdaState] = {}
        self.reset()

    def reset(self) -> None:
        """Zero every carry, ending whatever stream was in flight.

        A carry summarizes the whole prefix behind it, so a new request must not start from
        the previous one's. ``allocate_state`` zeros, which is where a stream begins.
        """
        self.deallocate()
        self._states = {idx: layer.allocate_state(self._batch_size) for idx, layer in self._layers.items()}

    def get(self, layer_idx: int) -> KdaState:
        return self._states[layer_idx]

    def replace(self, layer_idx: int, state: KdaState) -> None:
        """Install a layer's returned carry and free the one it supersedes.

        Freeing is safe because ``ttKDA.forward`` only reads the state it was given: nothing
        reachable from the superseded carry is aliased into its replacement.
        """
        previous = self._states.get(layer_idx)
        self._states[layer_idx] = state
        if previous is not None:
            ttnn.deallocate(previous.recurrent)
            ttnn.deallocate(previous.convolution)

    def deallocate(self) -> None:
        for state in self._states.values():
            ttnn.deallocate(state.recurrent)
            ttnn.deallocate(state.convolution)
        self._states = {}
