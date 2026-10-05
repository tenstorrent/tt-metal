# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Every KDA model shape x LoudBox layout resolves to an explicit program configuration."""

from collections.abc import Callable
from dataclasses import dataclass, replace

import pytest

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.glm_5_3_flash_config import glm_5_3_flash_kda_config
from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import kimi_k3_kda_config
from models.demos.deepseek_v3_d_p.tests.kda.head_slice import galaxy_chip_head_slice_config
from models.demos.deepseek_v3_d_p.tt.kda.config import (
    KDAProgramConfig,
    glm_5_3_flash_program_config,
    kimi_k3_program_config,
    tuned_projection_matmul_configs,
)
from models.demos.deepseek_v3_d_p.tt.kda.recurrence import KDARecurrence


@dataclass(frozen=True)
class _Layout:
    """LoudBox proxy of Galaxy SP8xTP4 5120-token prefill; SP on mesh axis 0, TP on axis 1."""

    mesh_shape: tuple[int, int]
    sequence_length: int
    # LB-B runs on each chip, without TP, the heads one Galaxy TP4 chip owns.
    galaxy_chip_head_slice: bool

    def chip_config(self, config: KDAConfig) -> KDAConfig:
        layer_config = galaxy_chip_head_slice_config(config) if self.galaxy_chip_head_slice else config
        heads, tp = layer_config.num_heads, self.mesh_shape[1]
        assert heads % tp == 0
        return replace(layer_config, num_heads=heads // tp)

    @property
    def local_rows(self) -> int:
        return self.sequence_length // self.mesh_shape[0]


_LAYOUTS = {
    "LB-A": _Layout(mesh_shape=(2, 4), sequence_length=1280, galaxy_chip_head_slice=False),
    "LB-B": _Layout(mesh_shape=(8, 1), sequence_length=5120, galaxy_chip_head_slice=True),
}
_MODELS: dict[str, tuple[Callable[[], KDAConfig], Callable[..., KDAProgramConfig], int, bool]] = {
    # model: (config builder, program config, heads per chip, tuned projection schedules)
    "kimi_k3": (kimi_k3_kda_config, kimi_k3_program_config, 24, True),
    "glm_5_3_flash": (glm_5_3_flash_kda_config, glm_5_3_flash_program_config, 16, False),
}
_CASES = [pytest.param(model, layout, id=f"{model}-{layout}") for model in _MODELS for layout in _LAYOUTS]
# Galaxy Blackhole worker grid the tuned projection schedules were measured on.
_GALAXY_GRID = ttnn.CoreCoord(12, 10)


def _resolve(model: str, layout_name: str) -> tuple[KDAConfig, _Layout, KDAProgramConfig]:
    build_config, build_program_config, _, _ = _MODELS[model]
    layout = _LAYOUTS[layout_name]
    program_config = build_program_config(active_seq_len_local=layout.local_rows, tp_ccl_topology=ttnn.Topology.Linear)
    return layout.chip_config(build_config()), layout, program_config


@pytest.mark.parametrize("model, layout_name", _CASES)
def test_loudbox_program_config_is_explicit(model: str, layout_name: str) -> None:
    chip_config, layout, program_config = _resolve(model, layout_name)
    _, _, chip_heads, tuned = _MODELS[model]

    # Both layouts run Galaxy's per-chip work: 640 local rows and a quarter of the heads.
    assert (layout.local_rows, chip_config.num_heads) == (640, chip_heads)
    assert program_config.recurrence.local_scan_strategy == "grouped"
    assert program_config.recurrence.summary_group_chunks == 20  # 640 rows = 20 chunks of 32
    assert program_config.qkv_channel_chunk_size == 512
    assert program_config.gated_rms_output_dtype == ttnn.bfloat16
    assert program_config.output_projection_math_fidelity == ttnn.MathFidelity.HiFi2
    assert program_config.tuned_projection_matmuls is tuned
    if tuned:
        tuned_projection_matmul_configs(_GALAXY_GRID, layout.local_rows, chip_config.v_dim, chip_config.hidden_size)


def test_glm_program_config_rejects_unconfigured_local_length(expect_error) -> None:
    with expect_error(ValueError, "no tuned GLM-5.3-Flash"):
        glm_5_3_flash_program_config(active_seq_len_local=1280, tp_ccl_topology=ttnn.Topology.Linear)


def test_tuned_projection_schedules_reject_misfit(expect_error) -> None:
    # 672 rows = 21 row tiles over 10 grid rows -> 3 tiles per core, which the tuned blocking cannot hold.
    with expect_error(ValueError, "do not fit"):
        tuned_projection_matmul_configs(_GALAXY_GRID, 672, 3072, 7168)


@run_for_blackhole()
@pytest.mark.parametrize("model, layout_name", _CASES)
def test_loudbox_program_config_fits_device(device: ttnn.Device, model: str, layout_name: str) -> None:
    """The per-chip program resolves on this device's worker grid: tuned schedules and recurrence owners."""
    chip_config, layout, program_config = _resolve(model, layout_name)
    grid = device.compute_with_storage_grid_size()
    if program_config.tuned_projection_matmuls:
        tuned_projection_matmul_configs(grid, layout.local_rows, chip_config.v_dim, chip_config.hidden_size)
    KDARecurrence(
        device,
        program_config.recurrence,
        sequence_parallel_axis=0,
        local_rows=layout.local_rows,
        heads=chip_config.num_heads,
        key_dim=chip_config.head_k_dim,
        value_dim=chip_config.head_v_dim,
    )
    print(f"{model} {layout_name}: worker grid {grid.x}x{grid.y}, {chip_config.num_heads} heads/chip")
