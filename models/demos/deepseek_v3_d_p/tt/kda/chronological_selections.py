# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Semantic selections using device-derived chronology; no host offset readback."""

from dataclasses import dataclass

import ttnn
from models.demos.deepseek_v3_d_p.tt.kda.config import KDA_DISTRIBUTED_WORKING_MEMORY_CONFIG

_layout = ttnn._ttnn.operations.experimental.kda._selection_layout


@dataclass(frozen=True)
class ChronologicalSelections:
    """Select convolution history, affine transforms, and recurrent states.

    The private UINT32 row-major device table has shape (8 + 2 * SP size, 8)
    per device and contains selection instructions, not activations or states.
    Each aligned record has eight words: history
    records use three row indices, and recurrence selections use consecutive
    start/end records with four coordinates each (exclusive end).

    Records describe outgoing/predecessor/final history, local entry/final
    recurrent state, then one affine-transform bounds pair per SP step.
    Native ``chronological_selections`` produces the table; its shared layout
    definitions are exposed privately through ``_selection_layout``.
    """

    _selection_records: ttnn.Tensor

    def select_outgoing_history(self, projected_qkv: ttnn.Tensor) -> ttnn.Tensor:
        """Three local tokens preceding the next physical rank's segment."""
        return self._select_rows(projected_qkv, _layout.OUTGOING_HISTORY)

    def select_predecessor_history(self, gathered_history: ttnn.Tensor) -> ttnn.Tensor:
        """The preceding physical rank's history from the gathered candidates."""
        return self._select_rows(gathered_history, _layout.PREDECESSOR_HISTORY)

    def select_local_final_history(self, projected_qkv: ttnn.Tensor, sp_size: int) -> ttnn.Tensor:
        """Last three locally valid rows; ignored when this rank is empty."""
        return self._select_rows(projected_qkv, _layout.local_final_history(sp_size))

    def select_final_history(self, candidates: ttnn.Tensor) -> ttnn.Tensor:
        """History at the logical sequence end, replicated for the next call."""
        return self._select_rows(candidates, _layout.FINAL_HISTORY)

    def select_affine_transform(
        self, gathered: ttnn.Tensor, step: int, *, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG
    ) -> ttnn.Tensor:
        """Physical device transform at the requested chronological step."""
        return self._select_block(gathered, _layout.affine_transform(step), memory_config=memory_config)

    def select_local_entry_state(
        self, chronological_entries: ttnn.Tensor, *, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG
    ) -> ttnn.Tensor:
        """This device's initial recurrent state from chronological entry states."""
        return self._select_block(chronological_entries, _layout.LOCAL_ENTRY_STATE, memory_config=memory_config)

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
        return ttnn.reshape(
            ttnn.slice(
                self._selection_records,
                (record_index, 0),
                (record_index + 1, count),
                memory_config=ttnn.L1_MEMORY_CONFIG,
            ),
            (count,),
        )

    def _select_block(
        self, tensor: ttnn.Tensor, record_index: int, *, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG
    ) -> ttnn.Tensor:
        return ttnn.slice(
            tensor,
            self._indices(record_index, _layout.SLICE_RANK),
            self._indices(record_index + 1, _layout.SLICE_RANK),
            slice_dim=0,
            num_devices=tensor.shape[0],
            memory_config=memory_config,
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
