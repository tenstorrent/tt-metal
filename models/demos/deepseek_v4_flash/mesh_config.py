# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Mode(Enum):
    DECODE = "decode"
    PREFILL = "prefill"


@dataclass(frozen=True)
class ModeConfig:
    tp: int
    ep: int = 1
    sp: int = 1

    def __post_init__(self) -> None:
        if self.tp < 1 or self.ep < 1 or self.sp < 1:
            raise ValueError(f"Parallelism values must be >= 1: tp={self.tp}, ep={self.ep}, sp={self.sp}")


class MeshConfig:
    def __init__(
        self,
        mesh_shape: tuple[int, int],
        *,
        decode: ModeConfig,
        prefill: ModeConfig | None = None,
        tp_axis: int = 1,
    ) -> None:
        if len(mesh_shape) != 2:
            raise ValueError(f"mesh_shape must be (rows, cols), got {mesh_shape}")
        if tp_axis not in (0, 1):
            raise ValueError(f"tp_axis must be 0 or 1, got {tp_axis}")
        self.mesh_shape = tuple(int(dim) for dim in mesh_shape)
        self.tp_axis = tp_axis
        self.ep_axis = 0 if tp_axis == 1 else 1
        self.sp_axis = self.ep_axis
        self.total_devices = self.mesh_shape[0] * self.mesh_shape[1]
        self.decode = decode
        self.prefill = prefill or ModeConfig(tp=decode.tp, ep=1, sp=self.mesh_shape[self.sp_axis])
        self._validate_config(self.decode, Mode.DECODE)
        self._validate_config(self.prefill, Mode.PREFILL)

    def get_config(self, mode: Mode | str) -> ModeConfig:
        mode = Mode(mode) if isinstance(mode, str) else mode
        return self.decode if mode is Mode.DECODE else self.prefill

    def data_parallel(self, mode: Mode | str) -> int:
        config = self.get_config(mode)
        return self.total_devices // (config.tp * config.ep)

    def shard_size(self, total_size: int, mode: Mode | str = Mode.DECODE) -> int:
        config = self.get_config(mode)
        if total_size % config.tp != 0:
            raise ValueError(f"Cannot shard size {total_size} across TP={config.tp}")
        return total_size // config.tp

    def to_manifest_dict(self) -> dict:
        return {
            "mesh_shape": list(self.mesh_shape),
            "tp_axis": self.tp_axis,
            "decode": _mode_to_dict(self.decode, self.data_parallel(Mode.DECODE)),
            "prefill": _mode_to_dict(self.prefill, self.data_parallel(Mode.PREFILL)),
        }

    def _validate_config(self, config: ModeConfig, mode: Mode) -> None:
        if config.tp > self.mesh_shape[self.tp_axis]:
            raise ValueError(
                f"{mode.value}: TP({config.tp}) > mesh axis {self.tp_axis} size {self.mesh_shape[self.tp_axis]}"
            )
        if config.ep > self.mesh_shape[self.ep_axis]:
            raise ValueError(
                f"{mode.value}: EP({config.ep}) > mesh axis {self.ep_axis} size {self.mesh_shape[self.ep_axis]}"
            )
        if config.sp > self.mesh_shape[self.sp_axis]:
            raise ValueError(
                f"{mode.value}: SP({config.sp}) > mesh axis {self.sp_axis} size {self.mesh_shape[self.sp_axis]}"
            )
        if self.total_devices % (config.tp * config.ep) != 0:
            raise ValueError(
                f"{mode.value}: TP({config.tp}) * EP({config.ep}) must divide total_devices({self.total_devices})"
            )

    def __repr__(self) -> str:
        decode_dp = self.data_parallel(Mode.DECODE)
        prefill_dp = self.data_parallel(Mode.PREFILL)
        decode = f"decode[TP={self.decode.tp}, EP={self.decode.ep}, SP={self.decode.sp}, DP={decode_dp}]"
        prefill = f"prefill[TP={self.prefill.tp}, EP={self.prefill.ep}, SP={self.prefill.sp}, DP={prefill_dp}]"
        return f"MeshConfig({self.mesh_shape}, {decode}, {prefill})"


def mesh_1x8() -> MeshConfig:
    return MeshConfig((1, 8), decode=ModeConfig(tp=8, ep=1), prefill=ModeConfig(tp=8, ep=1, sp=1))


def mesh_2x4() -> MeshConfig:
    return MeshConfig((2, 4), decode=ModeConfig(tp=4, ep=2), prefill=ModeConfig(tp=4, ep=1, sp=2))


def mesh_for_shape(mesh_shape: tuple[int, int]) -> MeshConfig:
    if mesh_shape == (1, 8):
        return mesh_1x8()
    if mesh_shape == (2, 4):
        return mesh_2x4()
    rows, cols = mesh_shape
    return MeshConfig(mesh_shape, decode=ModeConfig(tp=cols, ep=rows), prefill=ModeConfig(tp=cols, ep=1, sp=rows))


def _mode_to_dict(config: ModeConfig, dp: int) -> dict:
    return {"tp": config.tp, "ep": config.ep, "sp": config.sp, "dp": dp}
