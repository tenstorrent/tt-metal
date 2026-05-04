# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""GPT-OSS expert program configurations."""

import math
from dataclasses import dataclass

from models.demos.gpt_oss.tt.experts.config import ProgramConfig


def _largest_divisor_at_most(value: int, cap: int) -> int:
    """Largest positive divisor of `value` that does not exceed `cap`.

    The matmul kernel requires Kt % in0_block_w == 0 (see
    ttnn/cpp/ttnn/operations/matmul/device/factory/*.cpp). When K per device isn't
    a multiple of the hand-tuned default (e.g. gpt-oss-120b on BH T3K 4-chip gives
    Kt=23, which is prime), we must fall back to a divisor of Kt.
    """
    cap = max(1, min(cap, value))
    for d in range(cap, 0, -1):
        if value % d == 0:
            return d
    return 1


@dataclass
class GPTOSSProgramConfig(ProgramConfig):
    """
    GPT-OSS expert configuration.

    Optimized for: hidden=2088, intermediate=360
    """

    # Decode
    decode_gate_up_cores: tuple[int, int] = (3, 4)
    decode_gate_up_in0_block_w: int = 30
    decode_down_cores: tuple[int, int] = (5, 6)
    decode_down_in0_block_w: int = 12

    # Prefill
    prefill_gate_up_cores: tuple[int, int] = (3, 4)
    prefill_gate_up_in0_block_w: int = 30
    prefill_down_cores: tuple[int, int] = (5, 6)
    prefill_down_in0_block_w: int = 12

    # Memory
    sequence_chunk_size: int = 4 * 1024
    base_down_split_size: int = 1024

    @classmethod
    def for_model(cls, hidden_size: int, intermediate_size: int, tp: int) -> "GPTOSSProgramConfig":
        """Build a config whose `in0_block_w` values actually divide Kt on the active mesh.

        The defaults above are hand-tuned for tp=8 (gpt-oss-120b on Galaxy/LoudBox/BH T3K 8-chip).
        On BH T3K 4-chip (tp=4), intermediate_per_device=720 -> Kt=23 (prime), so `in0_block_w=12`
        fails `Kt % in0_block_w == 0`. We keep gate_up's K (hidden_size) as the source for its
        block width and derive down's K from `intermediate_size / tp`.
        """
        gate_up_kt = math.ceil(hidden_size / 32)
        down_kt = math.ceil(math.ceil(intermediate_size / tp) / 32)
        return cls(
            decode_gate_up_in0_block_w=_largest_divisor_at_most(gate_up_kt, cls.decode_gate_up_in0_block_w),
            prefill_gate_up_in0_block_w=_largest_divisor_at_most(gate_up_kt, cls.prefill_gate_up_in0_block_w),
            decode_down_in0_block_w=_largest_divisor_at_most(down_kt, cls.decode_down_in0_block_w),
            prefill_down_in0_block_w=_largest_divisor_at_most(down_kt, cls.prefill_down_in0_block_w),
        )
