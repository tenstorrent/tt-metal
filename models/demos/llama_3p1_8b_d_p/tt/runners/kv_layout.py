# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host address arithmetic for the packed SP4/TP8 BF8_B prefill cache."""

from dataclasses import dataclass

from models.demos.llama_3p1_8b_d_p.reference.llama_3p1_8b_config import Llama31_8BConfig as Model
from models.demos.llama_3p1_8b_d_p.tt.prefill_geometry import DEFAULT_MAX_SEQ_LEN
from models.demos.llama_3p1_8b_d_p.tt.prefill_geometry import PREFILL_LAYOUT as layout

CONFIG_NAMES = tuple(f"{kv}_h{head}" for kv in ("k", "v") for head in range(Model.NUM_KEY_VALUE_HEADS))


def integer(name, value, minimum=0):
    if type(value) is not int:
        raise TypeError(f"{name} must be a Python int")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


@dataclass(frozen=True)
class PrefillKVLayout:
    num_banks: int
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN
    num_slots: int = layout.num_users
    num_layers: int = Model.NUM_LAYERS
    chunk_size: int = layout.chunk_size
    sp: int = layout.sp
    tp: int = layout.tp
    head_dim: int = Model.HEAD_DIM

    def __post_init__(self):
        for name in ("max_seq_len", "num_slots", "num_layers", "num_banks", "chunk_size"):
            integer(name, getattr(self, name), 1)
        if (self.sp, self.tp, self.head_dim) != (layout.sp, layout.tp, Model.HEAD_DIM):
            raise ValueError(f"Llama source layout requires SP{layout.sp}/TP{layout.tp} and head_dim={Model.HEAD_DIM}")
        if self.chunk_size % (self.sp * 32) or self.max_seq_len % self.chunk_size:
            raise ValueError("chunk_size must divide max_seq_len and contain whole tiles per SP row")

    @property
    def config_names(self):
        return CONFIG_NAMES

    @property
    def chunk_size_bytes(self):
        return 4352

    def locate(self, config, layer, position, slot, base_addr):
        """Return (mesh coordinate, DRAM bank, absolute bank byte offset)."""
        for name, value, extent in (
            ("config", config, 16),
            ("layer", layer, self.num_layers),
            ("position", position, self.max_seq_len),
            ("slot", slot, self.num_slots),
        ):
            integer(name, value)
            if value >= extent:
                raise ValueError(f"{name} must be below {extent}")
        integer("base_addr", base_addr)
        if position % 32:
            raise ValueError("position must be tile aligned")
        width = self.chunk_size // self.sp
        row = position % self.chunk_size // width
        local_position = position // self.chunk_size * width + position % width
        page = ((slot * self.num_layers + layer) * (self.max_seq_len // self.sp) + local_position) // 32
        bank = page % self.num_banks
        offset = base_addr + page // self.num_banks * self.chunk_size_bytes
        if offset + self.chunk_size_bytes > 1 << 32:
            raise ValueError("DRAM offset exceeds the 32-bit NoC address field")
        return (row, config % self.tp), bank, offset
