# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Prefill block-level device-perf tests with real pretrained weights.

- `block_8x4_layer0_dense`: full MLA + dense FFN block on an 8x4 galaxy mesh. Measures
  MLA, dense FFN, norms, and residual adds. Gives a baseline without the MoE path.
- `block_8x4_layer3_moe`: full MLA + MoE block on an 8x4 galaxy mesh (gate + dispatch +
  routed experts + combine + shared expert + reduce). Subtract the dense-layer number to
  isolate MoE overhead.
- `block_2x4_layer3_moe`: full MLA + MoE block on a 2x4 2-link mesh, using the smaller
  `isl_6k4` configuration for real-weights perf coverage on that topology.

All three run 1 iteration with `skip_reference=True` (no torch reference, bfp8/bfp4
production dtypes). The 8x4 cases run at `isl_total=25*1024` (`isl_25k`), while the 2x4
case runs at `isl_total=6*1024+512` (`isl_6k4`). Requires the appropriate hardware for the
selected mesh topology and `DEEPSEEK_V3_HF_MODEL` pointing to the pretrained checkpoint.
"""

import os
import re

import pytest

from models.demos.deepseek_v3_d_p.utils.perf_utils import run_model_device_perf_test_with_merge

_TEST_PATH = "models/demos/deepseek_v3_d_p/tests/test_prefill_block_loop.py"

# Optional ISL sweep override: set DS_PERF_ISL=isl_5k (or isl_1k/isl_6k4/isl_12k8/isl_25k) to
# retarget EVERY perf entry below to that ISL instead of its native one, to measure device perf at
# that length. 5120 (5*1024) fits L1 on all meshes (8x4=640/chip, 4x4=1280, 2x4=2560). The per-entry
# expected_ns stays calibrated for the native ISL, so the threshold check will FAIL under the
# override — that's expected; the run still surfaces the measured perf. Unset = native ISL + check.
_ISL_OVERRIDE = os.environ.get("DS_PERF_ISL")

# 4x4 sub-torus carving: 16 of the galaxy's 32 chips + the Ring-4-on-Y graph descriptor.
# Applied via run_model_device_perf_test_with_merge(extra_env=...) (which sets os.environ
# around the subprocess) rather than prefixed into the pytest command — tracy's `-m` flag
# mis-parses leading KEY=VAL tokens as module names (see perf_utils docstring).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([".."] * 5)))
_SUBTORUS_Y4_ENV = {
    "TT_VISIBLE_DEVICES": "2,3,6,7,10,11,14,15,18,19,22,23,26,27,30,31",
    "TT_MESH_GRAPH_DESC_PATH": os.path.join(
        _REPO_ROOT,
        "tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_subtorus_y4_graph_descriptor.textproto",
    ),
}


@pytest.mark.parametrize(
    "command, expected_device_perf_ns_per_iteration, subdir, model_name, num_iterations, batch_size, margin, comments",
    [
        # FABRIC_1D baselines — tightened with `and not fabric2d-` to exclude the new
        # 2D parametrize ids in test_prefill_block_loop.py (substring `mesh-8x4`/`mesh-2x4`
        # would otherwise match both 1D and 2D variants).
        (
            f"pytest {_TEST_PATH} -k 'mesh-8x4 and layer0 and gate_device and no_ref and isl_25k and not fabric2d-'",
            19_378_393,  # Recalibrated 2026-06-10 on bh-glx-110-c08u02 (deepseek_v3_d_p perf run); FABRIC_1D.
            "deepseek_v3_prefill_block",
            "deepseek_v3_prefill_block_8x4_layer0_dense",
            1,
            1,
            0.03,
            "glx_8x4_layer0_dense_real_weights",
        ),
        (
            f"pytest {_TEST_PATH} -k 'mesh-8x4 and layer3 and gate_device and no_ref and isl_25k and not fabric2d-'",
            71_888_628,  # Recalibrated 2026-06-10 on bh-glx-110-c08u02 (deepseek_v3_d_p perf run); FABRIC_1D.
            "deepseek_v3_prefill_block",
            "deepseek_v3_prefill_block_8x4_layer3_moe",
            1,
            1,
            0.03,
            "glx_8x4_layer3_moe_real_weights",
        ),
        (
            f"pytest {_TEST_PATH} -k 'mesh-2x4-2link and layer3 and gate_device and no_ref and isl_6k4'",
            58_750_603,  # Recalibrated 2026-06-10 on BH LoudBox 2x4; FABRIC_1D.
            "deepseek_v3_prefill_block",
            "deepseek_v3_prefill_block_2x4_layer3_moe",
            1,
            1,
            0.03,
            "2x4_layer3_moe_real_weights_2link",
        ),
        # FABRIC_2D variants. The layer3 MoE entry has been calibrated against a real run on
        # BH Galaxy (~149.6 ms measured); the other two are still uncalibrated placeholders
        # with a wide margin so the first run will pass and surface the measured number.
        (
            f"pytest {_TEST_PATH} -k 'fabric2d-mesh-8x4 and layer0 and gate_device and no_ref and isl_25k'",
            25_862_584,  # Recalibrated 2026-06-10 on bh-glx-110-c08u02 (with FABRIC_2D init flush=false change).
            "deepseek_v3_prefill_block",
            "deepseek_v3_prefill_block_8x4_layer0_dense_fabric2d",
            1,
            1,
            0.03,
            "glx_8x4_layer0_dense_real_weights_fabric2d",
        ),
        (
            f"pytest {_TEST_PATH} -k 'fabric2d-mesh-8x4 and layer3 and gate_device and no_ref and isl_25k'",
            87_100_959,  # Recalibrated 2026-06-10 on bh-glx-110-c08u02 (with FABRIC_2D init flush=false change).
            "deepseek_v3_prefill_block",
            "deepseek_v3_prefill_block_8x4_layer3_moe_fabric2d",
            1,
            1,
            0.03,
            "glx_8x4_layer3_moe_real_weights_fabric2d",
        ),
        (
            f"pytest {_TEST_PATH} -k 'fabric2d-mesh-2x4 and layer3 and gate_device and no_ref and isl_6k4'",
            72_463_075,  # Recalibrated 2026-06-10 on BH LoudBox 2x4; FABRIC_2D.
            "deepseek_v3_prefill_block",
            "deepseek_v3_prefill_block_2x4_layer3_moe_fabric2d",
            1,
            1,
            0.03,
            "2x4_layer3_moe_real_weights_fabric2d",
        ),
        # FABRIC_2D_TORUS_Y variants — single-galaxy 8x4 with the SP axis (dim 0) closed into a ring
        # (Ring on SP-axis dispatch/combine, Linear on TP-axis collectives). Only the 8x4 cases are
        # meaningful: TORUS_Y wraps the SP axis, and on a 2x4 mesh that axis is 2-wide so the ring
        # degenerates to Linear. Uncalibrated placeholders (estimated from the fabric2d siblings)
        # with a wide margin so the first run passes and surfaces the real measured number; recalibrate
        # the expected_ns and tighten the margin afterwards.
        (
            f"pytest {_TEST_PATH} -k 'fabric2d-torus-y-8x4 and layer0 and gate_device and no_ref and isl_25k'",
            25_862_584,  # UNCALIBRATED placeholder (copied from fabric2d-mesh-8x4 layer0); recalibrate.
            "deepseek_v3_prefill_block",
            "deepseek_v3_prefill_block_8x4_layer0_dense_torus_y",
            1,
            1,
            0.5,
            "glx_8x4_layer0_dense_real_weights_torus_y",
        ),
        (
            f"pytest {_TEST_PATH} -k 'fabric2d-torus-y-8x4 and layer3 and gate_device and no_ref and isl_25k'",
            87_100_959,  # UNCALIBRATED placeholder (copied from fabric2d-mesh-8x4 layer3); recalibrate.
            "deepseek_v3_prefill_block",
            "deepseek_v3_prefill_block_8x4_layer3_moe_torus_y",
            1,
            1,
            0.5,
            "glx_8x4_layer3_moe_real_weights_torus_y",
        ),
        # FABRIC_2D_TORUS_Y on a 4x4 sub-torus (16 chips, Ring-4 on the SP axis). Must be run with
        # TT_VISIBLE_DEVICES (16 chips) + TT_MESH_GRAPH_DESC_PATH=...subtorus_y4... (applied via
        # extra_env below). Uses isl_12k8 (12800 = half of 25k) so the 4-wide SP axis shards to
        # 3200 tokens/chip — the same per-chip load as the 8x4 at isl_25k, which fits L1. isl_25k
        # here would be 6400/chip and overflow L1 (MoE circular buffers exceed the 1.5 MB budget).
        # Uncalibrated placeholders with a wide margin; recalibrate after the first measured run.
        (
            f"pytest {_TEST_PATH} -k 'fabric2d-torus-y-4x4 and layer0 and gate_device and no_ref and isl_12k8'",
            25_862_584,  # UNCALIBRATED placeholder (8x4 layer0, same 3200/chip load); recalibrate for 4x4.
            "deepseek_v3_prefill_block",
            "deepseek_v3_prefill_block_4x4_layer0_dense_torus_y",
            1,
            1,
            0.5,
            "subtorus_4x4_layer0_dense_real_weights_torus_y_isl12k8",
        ),
        (
            f"pytest {_TEST_PATH} -k 'fabric2d-torus-y-4x4 and layer3 and gate_device and no_ref and isl_12k8'",
            87_100_959,  # UNCALIBRATED placeholder (8x4 layer3, same 3200/chip load); recalibrate for 4x4.
            "deepseek_v3_prefill_block",
            "deepseek_v3_prefill_block_4x4_layer3_moe_torus_y",
            1,
            1,
            0.5,
            "subtorus_4x4_layer3_moe_real_weights_torus_y_isl12k8",
        ),
        # Explicit 4x4 sub-torus MoE scenario at 8 experts/chip. On the (4,4) mesh the loop test
        # auto-halves the routed experts to 128 (128/16=8 per chip, matching the 8x4) and FORCES the
        # HOST gate, because the device grouped-gate kernels hard-require exactly 256 experts. The
        # gate_device id is forced to HOST_ALL internally; the gate_host param is skipped to dedupe.
        # (Same selecting filter as the layer3 entry above, which now also runs 128/host — this row
        # is kept as a first-class, clearly-labeled perf data point for the 8-experts/chip config.)
        (
            f"pytest {_TEST_PATH} -k 'fabric2d-torus-y-4x4 and layer3 and gate_device and no_ref and isl_12k8'",
            87_100_959,  # UNCALIBRATED placeholder; gate runs on HOST here, so recalibrate vs the device-gate rows.
            "deepseek_v3_prefill_block",
            "deepseek_v3_prefill_block_4x4_layer3_moe_torus_y_128ec_hostgate",
            1,
            1,
            0.5,
            "subtorus_4x4_layer3_moe_128experts_8perchip_hostgate_isl12k8",
        ),
        # 4x4 sub-torus at isl_2k56 (2560 = half of 5k -> 640 tokens/chip on the 4-wide SP axis).
        # Gate behavior follows the 4x4 default: 128 experts + HOST gate, unless DS_4X4_FULL_EXPERTS=1
        # is set (then 256 experts + the device gate, which fits L1 easily at 640 tokens/chip).
        (
            f"pytest {_TEST_PATH} -k 'fabric2d-torus-y-4x4 and layer3 and gate_device and no_ref and isl_2k56'",
            45_000_000,  # UNCALIBRATED placeholder (640 tokens/chip); recalibrate after first run.
            "deepseek_v3_prefill_block",
            "deepseek_v3_prefill_block_4x4_layer3_moe_torus_y_isl2k56",
            1,
            1,
            0.5,
            "subtorus_4x4_layer3_moe_torus_y_isl2k56_640perchip",
        ),
        # Same 4x4 isl_2k56 (640 tokens/chip), but the 128-expert / HOST-gate variant (8 experts/chip).
        # Run WITHOUT DS_4X4_FULL_EXPERTS so the loop test auto-halves to 128 and forces HOST_ALL
        # (the device grouped-gate kernels require exactly 256). Counterpart to the entry above, which
        # — with DS_4X4_FULL_EXPERTS=1 — runs the full 256 experts on the device gate.
        (
            f"pytest {_TEST_PATH} -k 'fabric2d-torus-y-4x4 and layer3 and gate_device and no_ref and isl_2k56'",
            45_000_000,  # UNCALIBRATED placeholder; gate runs on HOST here. Recalibrate after first run.
            "deepseek_v3_prefill_block",
            "deepseek_v3_prefill_block_4x4_layer3_moe_torus_y_isl2k56_128ec_hostgate",
            1,
            1,
            0.5,
            "subtorus_4x4_layer3_moe_128experts_8perchip_hostgate_isl2k56",
        ),
    ],
    ids=[
        "block_8x4_layer0_dense",
        "block_8x4_layer3_moe",
        "block_2x4_layer3_moe",
        "block_8x4_layer0_dense_fabric2d",
        "block_8x4_layer3_moe_fabric2d",
        "block_2x4_layer3_moe_fabric2d",
        "block_8x4_layer0_dense_torus_y",
        "block_8x4_layer3_moe_torus_y",
        "block_4x4_layer0_dense_torus_y",
        "block_4x4_layer3_moe_torus_y",
        "block_4x4_layer3_moe_torus_y_128ec_hostgate",
        "block_4x4_layer3_moe_torus_y_isl2k56",
        "block_4x4_layer3_moe_torus_y_isl2k56_128ec_hostgate",
    ],
)
@pytest.mark.timeout(0)
def test_deepseek_v3_prefill_block_perf(
    command,
    expected_device_perf_ns_per_iteration,
    subdir,
    model_name,
    num_iterations,
    batch_size,
    margin,
    comments,
):
    # 4x4 sub-torus variants need 16 specific chips carved out + the Ring-4-on-Y descriptor.
    # These must reach the worker via os.environ (extra_env), not the command string.
    extra_env = None
    if "_4x4_" in model_name:
        extra_env = dict(_SUBTORUS_Y4_ENV)
        # The dedicated host-gate entries must ALWAYS run 128 experts + host gate, even if a stray
        # DS_4X4_FULL_EXPERTS is exported in the shell (e.g. left over from a device-gate run).
        # Clear it for the worker so the loop test halves to 128 and forces HOST_ALL.
        if "128ec_hostgate" in model_name:
            extra_env["DS_4X4_FULL_EXPERTS"] = ""

    # DS_PERF_ISL sweep: retarget this entry's ISL to measure perf at a different length. The
    # per-entry expected_ns is calibrated for the native ISL, so the threshold check will FAIL —
    # that's fine, the run still surfaces the measured device perf at the new length.
    if _ISL_OVERRIDE:
        command = re.sub(r"isl_\w+", _ISL_OVERRIDE, command)

    run_model_device_perf_test_with_merge(
        command=command,
        expected_device_perf_ns_per_iteration=expected_device_perf_ns_per_iteration,
        subdir=subdir,
        model_name=model_name,
        num_iterations=num_iterations,
        batch_size=batch_size,
        margin=margin,
        comments=comments,
        extra_env=extra_env,
    )
