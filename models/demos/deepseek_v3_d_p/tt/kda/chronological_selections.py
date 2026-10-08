# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Semantic selections using device-derived chronology; no host offset readback."""

from dataclasses import dataclass, field

import ttnn
from models.demos.deepseek_v3_d_p.tt.kda.config import KDA_DISTRIBUTED_WORKING_MEMORY_CONFIG

_layout = ttnn._ttnn.operations.experimental.kda._selection_layout


@dataclass(frozen=True)
class ChronologicalSelections:
    """Select convolution history and the final recurrent state.

    The private UINT32 row-major device table has shape (6, 8) per device and
    contains selection instructions, not activations or states. Each aligned
    record has eight words: history records use three row indices, and the
    final-state selection uses consecutive start/end records with four
    coordinates each (exclusive end).

    Records describe outgoing/predecessor/final history, the final recurrent
    state, then the local final history.
    Native ``chronological_selections`` produces the table; its shared layout
    definitions are exposed privately through ``_selection_layout``.
    """

    _selection_records: ttnn.Tensor
    # Index rows sliced out of the table, by (record, count). Every KDA layer of a chunk reads the same rows, so a
    # table shared across layers slices each row once per chunk instead of once per layer. They live in DRAM: a
    # shared table keeps them for the whole chunk, and long-lived L1 buffers clash with later programs' static
    # circular buffers.
    _indices_cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def select_outgoing_history(self, projected_qkv: ttnn.Tensor) -> ttnn.Tensor:
        """Three local tokens preceding the next physical rank's segment."""
        return self._select_rows(projected_qkv, _layout.OUTGOING_HISTORY)

    def select_predecessor_history(self, gathered_history: ttnn.Tensor) -> ttnn.Tensor:
        """The preceding physical rank's history from the gathered candidates."""
        return self._select_rows(gathered_history, _layout.PREDECESSOR_HISTORY)

    def select_local_final_history(self, projected_qkv: ttnn.Tensor) -> ttnn.Tensor:
        """Last three locally valid rows; ignored when this rank is empty."""
        return self._select_rows(projected_qkv, _layout.LOCAL_FINAL_HISTORY)

    def select_final_history(self, candidates: ttnn.Tensor) -> ttnn.Tensor:
        """History at the logical sequence end, replicated for the next call."""
        return self._select_rows(candidates, _layout.FINAL_HISTORY)

    def select_final_state(self, device_finals: ttnn.Tensor, prefix_state: ttnn.Tensor) -> ttnn.Tensor:
        """Replacement recurrent state, shaped [1, batch_heads, key_dim, value_dim]."""
        batch_heads, key_dim, value_dim = prefix_state.shape
        device_finals = ttnn.reshape(device_finals, (-1, batch_heads, key_dim, value_dim))
        prefix_state = ttnn.reshape(prefix_state, (1, batch_heads, key_dim, value_dim))
        candidates = ttnn.concat(
            [device_finals, prefix_state], dim=0, memory_config=KDA_DISTRIBUTED_WORKING_MEMORY_CONFIG
        )
        return self._select_block(candidates, _layout.FINAL_STATE)

    def _indices(self, record_index: int, count: int) -> ttnn.Tensor:
        key = (record_index, count)
        if key not in self._indices_cache:
            self._indices_cache[key] = ttnn.reshape(
                ttnn.slice(
                    self._selection_records,
                    (record_index, 0),
                    (record_index + 1, count),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                ),
                (count,),
            )
        return self._indices_cache[key]

    def _select_block(self, tensor: ttnn.Tensor, record_index: int) -> ttnn.Tensor:
        return ttnn.slice(
            tensor,
            self._indices(record_index, _layout.SLICE_RANK),
            self._indices(record_index + 1, _layout.SLICE_RANK),
            slice_dim=0,
            num_devices=tensor.shape[0],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _select_rows(self, tensor: ttnn.Tensor, record_index: int) -> ttnn.Tensor:
        width = tensor.shape[-1]
        table = ttnn.reshape(tensor, (-1, width))
        selected = ttnn.embedding(
            self._indices(record_index, _layout.HISTORY_ROWS),
            table,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return ttnn.reshape(selected, (1, _layout.HISTORY_ROWS, width))
