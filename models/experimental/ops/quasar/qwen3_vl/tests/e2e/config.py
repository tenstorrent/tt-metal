# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Run configuration for the Qwen3-VL Quasar e2e test."""
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

HF_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"


def parse_grid(s):
    if s in (None, ""):
        return None
    m = re.fullmatch(r"(\d+)x(\d+)", str(s))
    if not m:
        raise ValueError(f"grid must look like 3x2, got {s!r}")
    return int(m.group(1)), int(m.group(2))


def _csv(s):
    return tuple(x.strip() for x in str(s or "").split(",") if x.strip())


@dataclass(frozen=True)
class RunConfig:
    size: str
    vision_layers: int
    text_layers: int
    decode_steps: int
    deepstack_at: int | None
    kv_blocks: int | None
    host_ops: tuple
    disable_wa: tuple
    quasar_config: bool
    expect_grid: tuple | None
    run_dir: Path

    @classmethod
    def from_options(cls, getoption: Callable[[str], object]) -> "RunConfig":
        ds = getoption("--qwen-deepstack-at")
        kv = getoption("--qwen-kv-blocks")
        return cls(
            size=str(getoption("--qwen-size")),
            vision_layers=int(getoption("--qwen-vision-layers")),
            text_layers=int(getoption("--qwen-text-layers")),
            decode_steps=int(getoption("--qwen-decode-steps")),
            deepstack_at=None if ds is None else int(ds),
            kv_blocks=None if kv is None else int(kv),
            host_ops=_csv(getoption("--qwen-host-ops")),
            disable_wa=_csv(getoption("--qwen-disable-wa")),
            quasar_config=bool(getoption("--qwen-quasar-config")),
            expect_grid=parse_grid(getoption("--qwen-expect-grid")),
            run_dir=Path(str(getoption("--qwen-run-dir"))),
        )

    def cache_key(self, grid) -> str:
        ds = "real" if self.deepstack_at is None else f"at{self.deepstack_at}"
        policy = "qcfg_bf16" if self.quasar_config else "native"
        from models.experimental.ops.quasar.qwen3_vl.tt import quasar_config

        layout = f"_w{quasar_config.WEIGHT_LAYOUT_VERSION}" if self.quasar_config else ""
        return f"{policy}{layout}_g{grid[0]}x{grid[1]}_v{self.vision_layers}_t{self.text_layers}_ds{ds}"
