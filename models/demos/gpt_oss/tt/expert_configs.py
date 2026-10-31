# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""GPT-OSS expert program configurations."""

from dataclasses import dataclass

from models.demos.gpt_oss.tt import opt_switches
from models.demos.gpt_oss.tt.experts.config import ProgramConfig


def _exp_expert_grids():
    """GPT_OSS_OPT_EXPERT_GRID: override the expert matmul core
    grids, e.g. "dgu=11x3,ddn=10x9,pgu=11x3,pdn=9x5" (decode / prefill gate|up and down; each key optional). Only N is
    split differently, K blocking is unchanged. Read at call time."""
    spec = opt_switches.value("EXPERT_GRID")
    keys = {
        "dgu": "decode_gate_up_cores",
        "ddn": "decode_down_cores",
        "pgu": "prefill_gate_up_cores",
        "pdn": "prefill_down_cores",
    }
    out = {}
    for item in filter(None, spec.split(",")):
        k, v = item.split("=")
        x, y = v.lower().split("x")
        out[keys[k.strip()]] = (int(x), int(y))
    return out


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

    def _config(self, kind, m, n, k):
        cores = getattr(self, f"{kind}_cores")
        override = _exp_expert_grids().get(f"{kind}_cores")
        if override is not None:
            # The sparse matmul needs exactly one N block per grid core (a rectangle of ceil(Nt / per_core_N) cores);
            # an override that does not give that (e.g. Nt = 23, prime, on 33 cores) keeps the default grid.
            nt = -(-n // 32)
            per_core_n = -(-nt // (override[0] * override[1]))
            if -(-nt // per_core_n) == override[0] * override[1]:
                cores = override
        subblock_w = getattr(self, f"{kind}_subblock_w")
        if kind.startswith("decode") and opt_switches.on("DECODE_SUBBLOCK"):
            # GPT_OSS_OPT_DECODE_SUBBLOCK: one-tile-row decode products in [1, per_core_N] output subblocks
            # (each unpacked in0 tile feeds per_core_N products); _build_matmul_config snaps it to a divisor.
            nt = -(-n // 32)
            subblock_w = -(-nt // (cores[0] * cores[1]))
        return self._build_matmul_config(
            cores,
            m,
            n,
            in0_block_w=getattr(self, f"{kind}_in0_block_w"),
            out_subblock_w=subblock_w,
            k=k,
        )

    def get_decode_gate_up_config(self, m, n, k=None):
        return self._config("decode_gate_up", m, n, k)

    def get_decode_down_config(self, m, n, k=None):
        return self._config("decode_down", m, n, k)

    def get_prefill_gate_up_config(self, m, n, k=None):
        return self._config("prefill_gate_up", m, n, k)

    def get_prefill_down_config(self, m, n, k=None):
        return self._config("prefill_down", m, n, k)
