# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn
from models.tt_dit.tests.models.minimax_h3.tools import sweep_tap_filter_configs as sweep


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
@pytest.mark.parametrize("failed_trial", [False, True])
def test_eight_shapes_release_l1_small_between_trials(mesh_device, monkeypatch, failed_trial):
    mesh_device.disable_and_clear_program_cache()
    mesh_device.enable_program_cache()
    baseline = ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1_SMALL).total_bytes_allocated_per_bank
    if failed_trial:
        original = sweep._run_formulation
        first = True

        def fail_after_allocation(*args, **kwargs):
            nonlocal first
            out = original(*args, **kwargs)
            if first:
                first = False
                ttnn.deallocate(out)
                raise RuntimeError("injected failure after allocating cached reader indices")
            return out

        monkeypatch.setattr(sweep, "_run_formulation", fail_after_allocation)

    # More than the old six-shape workaround, with distinct program-cache keys.
    for index in range(8):
        result = sweep.sweep_shape(mesh_device, (1, 166 + 32 * index, 64, 7, 1), max_slices=2, repeat=1, time_mac=False)
        direct = result["formulations"]["direct"]
        assert direct["auto"]["fits"], direct
        assert direct["auto"]["max_abs_err"] < 0.03
        assert direct["best_seconds"] is not None
        used = ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1_SMALL).total_bytes_allocated_per_bank
        assert used == baseline, f"shape {index} retained {used - baseline} L1_SMALL bytes per bank"
