# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Core-utilization optimizations of the generic GPT-OSS decode and prefill paths (measured on gpt-oss-120b, Blackhole
1x4). Each one is on by default on Blackhole and off elsewhere (not yet validated on Wormhole meshes).

Read at call time, so a process can A/B them by setting GPT_OSS_OPT_<NAME>=0 / 1; GPT_OSS_OPT_DEFAULTS=0 turns every
default off (main's behavior).
    DECODE_TOPK       decode MoE over the top-k routed experts only (experts/decode.py)
    DECODE_AR         one-op decode TP all-reduce (experts/operations.py)
    DECODE_NORM       30-core sharded RMSNorm for one-tile-row activations (rms_norm.py)
    DECODE_ROUTER     one-block decode router logits matmul (topk.py)
    DECODE_LMHEAD_DRAM decode LM head reads its input from DRAM (model.py)
    DECODE_DSMM       DRAM-sharded decode QKV / o_proj (attention/__init__.py, decode.py)
    DECODE_SUBBLOCK   [1, per_core_N] output subblocks in the decode expert matmuls (expert_configs.py)
    PREFILL_MBLOCK    prefill experts without per-token-tile weight re-reads (experts/prefill.py)
    PREFILL_FUSEBIAS  + per-expert bias fused into the prefill expert matmuls
    PREFILL_IN0L1     + prefill expert input L1-interleaved (S <= 2048)
    PREFILL_SUBBLOCK  + up-to-8-tile output subblocks in the M-blocked prefill expert matmuls
    EXPERT_GRID=spec  expert matmul core grids, e.g. "pdn=10x9" (expert_configs.py)
"""

import functools
import os

PREFIX = "GPT_OSS_OPT_"

DEFAULTS = {
    "DECODE_TOPK": "1",
    "DECODE_AR": "1",
    "DECODE_NORM": "1",
    "DECODE_ROUTER": "1",
    "DECODE_LMHEAD_DRAM": "1",
    "DECODE_DSMM": "1",
    "DECODE_SUBBLOCK": "1",
    "PREFILL_MBLOCK": "1",
    "PREFILL_FUSEBIAS": "1",
    "PREFILL_IN0L1": "1",
    "PREFILL_SUBBLOCK": "1",
    "EXPERT_GRID": "pdn=10x9",
}


@functools.cache
def _is_blackhole():
    import ttnn

    return "blackhole" in ttnn.get_arch_name()


def _defaults_apply():
    return os.environ.get(PREFIX + "DEFAULTS", "1") != "0" and _is_blackhole()


def value(name, default=""):
    v = os.environ.get(PREFIX + name)
    if v is not None:
        return v
    return DEFAULTS.get(name, default) if _defaults_apply() else default


def on(name):
    return value(name, "0") == "1"
