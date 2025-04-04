# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import dataclass

import ttnn


@dataclass
class ConvArgsOverride:
    act_block_h: int
    enable_split_reader: bool
    enable_act_double_buffer: bool
    deallocate_activation: bool
    reshard_if_not_optimal: bool
    shard_layout: ttnn.TensorMemoryLayout
    transpose_shards: bool

    @staticmethod
    def from_json(data: dict) -> "ConvArgsOverride":
        data["shard_layout"] = ttnn.TensorMemoryLayout.from_json(data["shard_layout"])

        return ConvArgsOverride(**data)


@dataclass
class Downsample1ConvArgsOverride:
    c1: ConvArgsOverride
    c2: ConvArgsOverride
    c3: ConvArgsOverride
    c4: ConvArgsOverride
    c5: ConvArgsOverride
    c6: ConvArgsOverride
    c7: ConvArgsOverride
    c8: ConvArgsOverride

    @staticmethod
    def from_json(data: dict) -> "Downsample1ConvArgsOverride":
        data["c1"] = ConvArgsOverride.from_json(data["c1"])
        data["c2"] = ConvArgsOverride.from_json(data["c2"])
        data["c3"] = ConvArgsOverride.from_json(data["c3"])
        data["c4"] = ConvArgsOverride.from_json(data["c4"])
        data["c5"] = ConvArgsOverride.from_json(data["c5"])
        data["c6"] = ConvArgsOverride.from_json(data["c6"])
        data["c7"] = ConvArgsOverride.from_json(data["c7"])
        data["c8"] = ConvArgsOverride.from_json(data["c8"])

        return Downsample1ConvArgsOverride(**data)

    @staticmethod
    def from_file(file_path: str) -> "Downsample1ConvArgsOverride":
        with open(file_path, "r") as f:
            conv_args_overrides = Downsample1ConvArgsOverride.from_json(json.load(f))


@dataclass
class Downsample2ConvArgsOverride:
    c1: ConvArgsOverride
    c2: ConvArgsOverride
    c3: ConvArgsOverride
    c4: ConvArgsOverride
    c5: ConvArgsOverride
    res0: ConvArgsOverride
    res3: ConvArgsOverride


def apply_overrides(conv_args, conv_args_overrides):
    for key, value in conv_args_overrides.__dict__.items():
        if hasattr(conv_args, key):
            getattr(conv_args, key).update(value.__dict__)
        else:
            raise ValueError(f"Invalid key '{key}' in conv_args overrides")
