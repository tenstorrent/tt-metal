# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Identity and algebra for the bounded C12 experiment. No device imports.

The native path is deliberately restricted to one verified main-vocoder module.
This module proves no native arithmetic, capacity, replay or performance property.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

SCHEMA = "ltx-c12-phase-major-v1"
MODULE_PATH = "vocoder.resblocks.16.convs2.0"
PINNED_SOURCE = "b9f8587ce6c681f380e341753c6f0ea7f5ee441d"
CONFIG = {
    "resblock_kernel_sizes": [3, 7, 11],
    "upsample_rates": [5, 2, 2, 2, 2, 2],
    "upsample_kernel_sizes": [11, 4, 4, 4, 4, 4],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "upsample_initial_channel": 1536,
    "resblock": "AMP1",
    "activation": "snakebeta",
}
PRECISION = {
    "dtype": "float32",
    "math_fidelity": "HiFi4",
    "math_approx_mode": False,
    "fp32_dest_acc_en": True,
    "packer_l1_acc": True,
    "split_mode": "off",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class Cell:
    mode: str

    def __post_init__(self):
        if self.mode not in ("A", "B", "C", "D"):
            raise ValueError("C12 cell must be A, B, C or D")

    @property
    def pack(self):
        return 1 if self.mode in "AB" else 4

    @property
    def time_block(self):
        return 4 if self.mode in "AC" else 32

    def identity(self):
        return {
            "schema": SCHEMA,
            "mode": self.mode,
            "module_path": MODULE_PATH,
            "original": {"Cin": 24, "Cout": 24, "K": 7, "dilation": 1, "stride": 1},
            "pack": self.pack,
            "weight_layout": "phase*C+channel; Cout,Cin,K,1,1; prepare_conv3d",
            "blocking": {
                "C_in_block": 32,
                "C_out_block": 32,
                "T_out_block": self.time_block,
                "H_out_block": 1,
                "W_out_block": 1,
            },
            "precision": dict(PRECISION),
        }


def parse_mode(value):
    if value in (None, "", "off", "0"):
        return None, False
    automatic = value.startswith("auto:")
    return Cell(value[5:] if automatic else value), automatic


def checkpoint_reason(config, evidence):
    if not evidence or not all(evidence.get(k) for k in ("source_id", "weight_sha256", "config_sha256")):
        return "missing checkpoint evidence"
    if evidence["config_sha256"] != digest(config):
        return "checkpoint configuration identity mismatch"
    for key, expected in CONFIG.items():
        if key not in config:
            return f"missing checkpoint configuration: {key}"
        if config[key] != expected:
            return f"unsupported checkpoint configuration: {key}"
    if evidence.get("module_path") != MODULE_PATH or evidence.get("weight_shape") != [24, 24, 7]:
        return "checkpoint module/weight identity mismatch"
    if evidence.get("bias_shape") not in (None, [24]):
        return "checkpoint bias shape mismatch"
    if evidence.get("bias_shape") is not None and not evidence.get("bias_sha256"):
        return "missing checkpoint bias evidence"
    return None


def shape_reason(cell, shape, *, starts, time_factor):
    if len(shape) != 3 or shape[0] != 1 or shape[2] != 24:
        return "requires logical batch1/C24 BTC input"
    local_t = shape[1]
    if local_t < 1 or time_factor < 2 or len(starts) != time_factor:
        return "requires positive localT and one sharded time axis"
    if tuple(starts) != tuple(i * local_t for i in range(time_factor)):
        return "requires contiguous equal global shards starting at zero"
    if cell.pack == 4:
        if local_t % 4 or any(s % 4 for s in starts):
            return "pack4 requires localT and global shard starts divisible by4"
        if local_t // 4 < 32:
            return "pack4 requires at least32 packed rows"
    return None


def weight_map(pack):
    """(dst output, dst input, packed tap, original output/input/tap)."""
    if pack not in (1, 4):
        raise ValueError("C12 supports pack1/pack4 only")
    half = (3 + pack - 1) // pack
    for phase in range(pack):
        for tap in range(7):
            shift, source_phase = divmod(phase + tap - 3, pack)
            for out_channel in range(24):
                for in_channel in range(24):
                    yield (
                        phase * 24 + out_channel,
                        source_phase * 24 + in_channel,
                        shift + half,
                        out_channel,
                        in_channel,
                        tap,
                    )


def transform_weight(weight, bias, pack):
    """Analytic host preparation; accepts Torch tensors, preserving their dtype.

    CPU FP64 references may use this helper. Runtime loading separately enforces FP32.
    """
    if tuple(weight.shape) != (24, 24, 7) or (bias is not None and tuple(bias.shape) != (24,)):
        raise ValueError("C12 requires C24/C24/K7 and optional C24 bias")
    half = (3 + pack - 1) // pack
    result = weight.new_zeros((pack * 24, pack * 24, 2 * half + 1))
    for out_idx, in_idx, tap, out_channel, in_channel, old_tap in weight_map(pack):
        result[out_idx, in_idx, tap] = weight[out_channel, in_channel, old_tap]
    return result, None if bias is None else bias.repeat(pack)


def cache_suffix(identity):
    return "" if identity is None else "_c12_" + digest(identity)


def claim_cache_root(root, identity):
    """Comparison cells require a fresh owned root and process, never a shared cache.

    The exclusive marker survives process exit: reusing a prior comparison root fails.
    """
    path = Path(root).resolve()
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise ValueError("C12 requires an empty, separate TT_DIT_CACHE_DIR per comparison process")
    with (path / "c12-session.json").open("x") as stream:
        json.dump({"pid": os.getpid(), "identity": identity}, stream, indent=2)
    return str(path)


def trace_key(shape, identity):
    return tuple(shape) if identity is None else (tuple(shape), digest(identity))
