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

# 4x4 sub-torus carving: 16 of the galaxy's 32 chips + the Ring-4-on-Y graph descriptor.
# Applied via run_model_device_perf_test_with_merge(extra_env=...) (which sets os.environ
# around the subprocess) rather than prefixed into the pytest command — tracy's `-m` flag
# mis-parses leading KEY=VAL tokens as module names (see perf_utils docstring).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([".."] * 5)))
_SUBTORUS_Y4_ENV = {
    # The 16 PCIe device indices forming the inner 4x4 sub-torus carved from the 8x4 galaxy.
    # These MUST match the chip->(tray, asic) pinnings in
    # single_bh_galaxy_subtorus_y4_pinned_graph_descriptor.textproto: if the descriptor's pinned
    # set changes, regenerate this list from it (do not edit independently).
    "TT_VISIBLE_DEVICES": "2,3,6,7,10,11,14,15,18,19,22,23,26,27,30,31",
    "TT_MESH_GRAPH_DESC_PATH": os.path.join(
        _REPO_ROOT,
        "models/demos/deepseek_v3_d_p/experimental_descriptors/single_bh_galaxy_subtorus_y4_graph_descriptor.textproto",
    ),
}
# Same 16-chip carve, X-ring ([LINE,RING]) and XY-ring ([RING,RING]) descriptors.
_SUBTORUS_X4_ENV = {
    "TT_VISIBLE_DEVICES": _SUBTORUS_Y4_ENV["TT_VISIBLE_DEVICES"],
    "TT_MESH_GRAPH_DESC_PATH": os.path.join(
        _REPO_ROOT,
        "models/demos/deepseek_v3_d_p/experimental_descriptors/single_bh_galaxy_subtorus_x4_graph_descriptor.textproto",
    ),
}
_SUBTORUS_XY4_ENV = {
    "TT_VISIBLE_DEVICES": _SUBTORUS_Y4_ENV["TT_VISIBLE_DEVICES"],
    "TT_MESH_GRAPH_DESC_PATH": os.path.join(
        _REPO_ROOT,
        "models/demos/deepseek_v3_d_p/experimental_descriptors/single_bh_galaxy_subtorus_xy4_graph_descriptor.textproto",
    ),
}

# The 4x4 sub-torus (16 chips) can only measure the 128-expert HOST-gate MoE path: the loop test
# halves the routed experts 256 -> 128 on the (4,4) mesh (8/chip), and the device grouped-gate
# kernel (deepseek_grouped_gate / moe_grouped_topk) hard-requires exactly 256 experts — so the
# device gate is unavailable at 128, and the 256-expert device gate stalls on the ring on the
# sub-torus (the reason the host gate exists for 4x4). The HOST gate's device-perf number is
# dominated by host round-trip stalls and no longer tracks the 2026-07-01 device-kernel baseline
# (measured ~9x over on current main), so the margin check is not meaningful. Skip until the device
# gate is viable on the 4x4 sub-torus, or the baseline is recalibrated for the host-gate path.
_SUBTORUS_4X4_HOSTGATE_SKIP = pytest.mark.skip(
    reason="4x4 sub-torus MoE runs only the 128-expert HOST gate (device gate needs 256 experts and "
    "stalls on the ring on the sub-torus); host-gate device-perf is host-stall-dominated and does not "
    "track the device-kernel baseline (~9x over on current main). Recalibrate/re-enable when the "
    "device gate is viable on the 4x4 sub-torus."
)


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
            50_810_785,  # Re-centered 2026-07-20 for two stacked speedups now in the tree -- BOTH
            # the in-place direct-write change (drop the separate output buffer + per-layer fill;
            # measured 50.61 ms alone) AND #47536 (update_padded_kv_cache RM/fp8; measured 51.29 ms
            # alone). The combined 2x4-2link number can't be measured on the galaxy, so the target is
            # the midpoint of the plausible combined band [48.91, 50.61] ms; margin 0.03 ->
            # [48.27, 51.25] ms brackets both speedups stacking and either alone. Was 53_000_000.
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
            76_706_230,  # Recalibrated 2026-07-05 (perf improvement, was 87_100_959).
            "deepseek_v3_prefill_block",
            "deepseek_v3_prefill_block_8x4_layer3_moe_fabric2d",
            1,
            1,
            0.03,
            "glx_8x4_layer3_moe_real_weights_fabric2d",
        ),
        (
            f"pytest {_TEST_PATH} -k 'fabric2d-mesh-2x4 and layer3 and gate_device and no_ref and isl_6k4'",
            65_161_594,  # Re-centered 2026-06-25 for two stacked speedups now in the tree -- BOTH
            # the in-place direct-write change (measured 64.30 ms alone) AND #47536
            # (update_padded_kv_cache RM/fp8; measured 64.80 ms alone). The combined 2x4-2link number
            # can't be measured on the galaxy, so the target is the midpoint of the plausible combined
            # band [62.10, 64.30] ms; margin 0.03 -> [61.30, 65.10] ms brackets both speedups stacking
            # and either alone. Was 67_000_000.
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
        # degenerates to Linear. All entries calibrated 2026-06-26 on the 110-c78 BH galaxy at the
        # standard 0.03 margin (real weights).
        (
            f"pytest {_TEST_PATH} -k 'torus-y-8x4 and layer0 and gate_device and no_ref and isl_25k'",
            25_236_993,  # Recalibrated 2026-06-26 on 110-c78 BH galaxy; FABRIC_2D_TORUS_Y Ring-8 (layer0 dense, isl_25k).
            "deepseek_v3_prefill_block",
            "deepseek_v3_prefill_block_8x4_layer0_dense_torus_y",
            1,
            1,
            0.03,
            "glx_8x4_layer0_dense_real_weights_torus_y",
        ),
        (
            f"pytest {_TEST_PATH} -k 'torus-y-8x4 and layer3 and gate_device and no_ref and isl_25k'",
            67_193_413,  # Recalibrated 2026-06-26 on 110-c78 BH galaxy; FABRIC_2D_TORUS_Y Ring-8 (layer3 MoE, device gate).
            "deepseek_v3_prefill_block",
            "deepseek_v3_prefill_block_8x4_layer3_moe_torus_y",
            1,
            1,
            0.03,
            "glx_8x4_layer3_moe_real_weights_torus_y",
        ),
        # FABRIC_2D_TORUS_Y on a 4x4 sub-torus (16 chips, Ring-4 on the SP axis). Must be run with
        # TT_VISIBLE_DEVICES (16 chips) + TT_MESH_GRAPH_DESC_PATH=...subtorus_y4... (applied via
        # extra_env below). Uses isl_12k8 (12800 = half of 25k) so the 4-wide SP axis shards to
        # 3200 tokens/chip — the same per-chip load as the 8x4 at isl_25k, which fits L1. isl_25k
        # here would be 6400/chip and overflow L1 (MoE circular buffers exceed the 1.5 MB budget).
        # The layer0 dense entry below is live (calibrated 2026-06-26 on 110-c78, Ring-4 SP axis, 0.03
        # margin). The two layer3 MoE entries are SKIPPED (_SUBTORUS_4X4_HOSTGATE_SKIP): they run the
        # 128-expert HOST gate whose device-perf no longer tracks the baseline — same reason as the
        # torus-x/xy 4x4 block below.
        #
        # Gate: on the (4,4) mesh the loop test auto-halves the routed experts to 128 (128/16 = 8
        # per chip, matching the 8x4 at 256/32) and runs the HOST gate, because the device
        # grouped-gate kernels hard-require exactly 256 experts. So these layer3 entries measure the
        # 128-expert / HOST-gate path (the working 4x4 scenario). The 256-expert device gate can be
        # forced with DS_4X4_FULL_EXPERTS=1, but that path currently stalls on the ring (the reason
        # moe-gate_host_all was added) and is deliberately NOT exercised here.
        (
            f"pytest {_TEST_PATH} -k 'torus-y-4x4 and layer0 and gate_device and no_ref and isl_12k8'",
            17_978_418,  # Recalibrated 2026-06-26 on 110-c78 BH galaxy; FABRIC_2D_TORUS_Y Ring-4 (layer0 dense, 3200 tok/chip).
            "deepseek_v3_prefill_block",
            "deepseek_v3_prefill_block_4x4_layer0_dense_torus_y",
            1,
            1,
            0.03,
            "subtorus_4x4_layer0_dense_real_weights_torus_y_isl12k8",
        ),
        pytest.param(
            f"pytest {_TEST_PATH} -k 'torus-y-4x4 and layer3 and gate_device and no_ref and isl_12k8'",
            56_528_886,  # Recalibrated 2026-06-26 on 110-c78 BH galaxy; FABRIC_2D_TORUS_Y Ring-4 (layer3 MoE, 128 experts / HOST gate).
            "deepseek_v3_prefill_block",
            "deepseek_v3_prefill_block_4x4_layer3_moe_torus_y",  # 128 experts / HOST gate (see note above)
            1,
            1,
            0.03,
            "subtorus_4x4_layer3_moe_128experts_8perchip_hostgate_isl12k8",
            marks=_SUBTORUS_4X4_HOSTGATE_SKIP,  # host-gate perf no longer tracks the baseline; see marker note
        ),
        # 4x4 sub-torus at isl_2k56 (2560 = half of 5k -> 640 tokens/chip on the 4-wide SP axis).
        # Same 128-expert / HOST-gate behavior as the isl_12k8 layer3 entry above (see note).
        pytest.param(
            f"pytest {_TEST_PATH} -k 'torus-y-4x4 and layer3 and gate_device and no_ref and isl_2k56'",
            15_570_232,  # Recalibrated 2026-06-26 on 110-c78 BH galaxy; FABRIC_2D_TORUS_Y Ring-4 (layer3 MoE, 640 tok/chip, HOST gate).
            "deepseek_v3_prefill_block",
            "deepseek_v3_prefill_block_4x4_layer3_moe_torus_y_isl2k56",  # 128 experts / HOST gate
            1,
            1,
            0.03,
            "subtorus_4x4_layer3_moe_128experts_8perchip_hostgate_isl2k56",
            marks=_SUBTORUS_4X4_HOSTGATE_SKIP,  # host-gate perf no longer tracks the baseline; see marker note
        ),
        # FABRIC_2D_TORUS_X (X/TP-axis ring) — production full-galaxy case ([LINE,RING] pipeline
        # descriptors): Ring on the TP-axis collectives (RMS-norm, MLA, dense-FFN, shared-expert,
        # gate), Linear on the SP-axis MoE dispatch/combine. Baselines calibrated 2026-07-01 on the
        # 110-c910 BH galaxy at the standard 0.03 margin (real weights).
        (
            f"pytest {_TEST_PATH} -k 'torus-x-8x4 and layer0 and gate_device and no_ref and isl_25k'",
            22_656_265,  # Calibrated 2026-07-01 on 110-c910 BH galaxy; TORUS_X Ring (layer0 dense, isl_25k). Faster than torus_y (TP collectives ring).
            "deepseek_v3_prefill_block",
            "deepseek_v3_prefill_block_8x4_layer0_dense_torus_x",
            1,
            1,
            0.03,
            "glx_8x4_layer0_dense_real_weights_torus_x",
        ),
        (
            f"pytest {_TEST_PATH} -k 'torus-x-8x4 and layer3 and gate_device and no_ref and isl_25k'",
            75_154_221,  # Calibrated 2026-07-01 on 110-c910 BH galaxy; TORUS_X (layer3 MoE, isl_25k). Slower than torus_y: SP dispatch/combine run Linear (X-ring doesn't wrap the SP axis).
            "deepseek_v3_prefill_block",
            "deepseek_v3_prefill_block_8x4_layer3_moe_torus_x",
            1,
            1,
            0.03,
            "glx_8x4_layer3_moe_real_weights_torus_x",
        ),
        # FABRIC_2D_TORUS_XY (both axes ring) on the full 8x4 galaxy.
        (
            f"pytest {_TEST_PATH} -k 'torus-xy-8x4 and layer0 and gate_device and no_ref and isl_25k'",
            18_157_603,  # Calibrated 2026-07-01 on 110-c910 BH galaxy; TORUS_XY (layer0 dense, isl_25k). Fastest dense: both axes ring, incl. the SP-axis ring-attention SDPA.
            "deepseek_v3_prefill_block",
            "deepseek_v3_prefill_block_8x4_layer0_dense_torus_xy",
            1,
            1,
            0.03,
            "glx_8x4_layer0_dense_real_weights_torus_xy",
        ),
        (
            f"pytest {_TEST_PATH} -k 'torus-xy-8x4 and layer3 and gate_device and no_ref and isl_25k'",
            60_634_662,  # Calibrated 2026-07-01 on 110-c910 BH galaxy; TORUS_XY (layer3 MoE, isl_25k). Fastest MoE: SP dispatch/combine AND TP collectives all ring.
            "deepseek_v3_prefill_block",
            "deepseek_v3_prefill_block_8x4_layer3_moe_torus_xy",
            1,
            1,
            0.03,
            "glx_8x4_layer3_moe_real_weights_torus_xy",
        ),
        # 4x4 sub-torus (16-chip carve) at isl_12k8 — 128-expert / HOST-gate path (loop test halves
        # experts on 4x4), same methodology as the torus_y 4x4 entries. Needs the carve env.
        # SKIPPED: host-gate device-perf doesn't track the device-kernel baseline; see
        # _SUBTORUS_4X4_HOSTGATE_SKIP above. (The torus_y 4x4 entries share this host-gate path.)
        pytest.param(
            f"pytest {_TEST_PATH} -k 'torus-x-4x4 and layer3 and gate_device and no_ref and isl_12k8'",
            54_804_819,  # Calibrated 2026-07-01 on 110-c910 BH galaxy; TORUS_X Ring-4 (layer3 MoE, 128 experts / HOST gate, isl_12k8).
            "deepseek_v3_prefill_block",
            "deepseek_v3_prefill_block_4x4_layer3_moe_torus_x",
            1,
            1,
            0.03,
            "subtorus_4x4_layer3_moe_128experts_8perchip_hostgate_isl12k8_torus_x",
            marks=_SUBTORUS_4X4_HOSTGATE_SKIP,
        ),
        pytest.param(
            f"pytest {_TEST_PATH} -k 'torus-xy-4x4 and layer3 and gate_device and no_ref and isl_12k8'",
            52_978_544,  # Calibrated 2026-07-01 on 110-c910 BH galaxy; TORUS_XY Ring-4 (layer3 MoE, 128 experts / HOST gate, isl_12k8).
            "deepseek_v3_prefill_block",
            "deepseek_v3_prefill_block_4x4_layer3_moe_torus_xy",
            1,
            1,
            0.03,
            "subtorus_4x4_layer3_moe_128experts_8perchip_hostgate_isl12k8_torus_xy",
            marks=_SUBTORUS_4X4_HOSTGATE_SKIP,
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
        "block_4x4_layer3_moe_torus_y_isl2k56",
        "block_8x4_layer0_dense_torus_x",
        "block_8x4_layer3_moe_torus_x",
        "block_8x4_layer0_dense_torus_xy",
        "block_8x4_layer3_moe_torus_xy",
        "block_4x4_layer3_moe_torus_x",
        "block_4x4_layer3_moe_torus_xy",
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
    # Subtorus/torus perf entries spawn a child pytest on a FABRIC_2D_TORUS_Y fabric, which needs
    # wrap cabling absent on CI runners -> would hang on bh-glx. Skip on CI; these run only on a
    # subtorus-wired host (CI unset). Mirrors the collection-time guard in the deepseek conftest.
    if "torus" in model_name and (os.getenv("CI") == "true" or "TT_GH_CI_INFRA" in os.environ):
        pytest.skip("subtorus/torus perf entry: no wrap-cabled CI runner (would hang on bh-glx)")

    # Force both MoE overlaps (shared/dispatch and routed/combine) off for every case, so the
    # device-perf baselines are measured with overlaps disabled (honored in test_prefill_block_loop.py).
    extra_env = {"TT_DS_MOE_DISABLE_OVERLAP": "1"}

    # 4x4 sub-torus variants need 16 specific chips carved out + the Ring-4-on-Y descriptor.
    # These must reach the worker via os.environ (extra_env), not the command string.
    if "_4x4_" in model_name:
        # Pick the carve descriptor by the exact `_torus_<axis>` token. A regex on the delimited
        # token (not a raw substring) avoids the "torus_x" ⊂ "torus_xy" trap and tolerates a trailing
        # suffix like `_isl2k56` (present on some torus_y ids). Alternation tries `xy` before `x`.
        _carve_env_by_axis = {"x": _SUBTORUS_X4_ENV, "xy": _SUBTORUS_XY4_ENV, "y": _SUBTORUS_Y4_ENV}
        m = re.search(r"_torus_(xy|x|y)(?:_|$)", model_name)
        assert m is not None, f"4x4 perf entry {model_name!r} has no _torus_<axis> token"
        # update() (not reassignment) so the TT_DS_MOE_DISABLE_OVERLAP flag set above is preserved.
        extra_env.update(_carve_env_by_axis[m.group(1)])
        # 4x4 layer3 entries measure the 128-expert / HOST-gate path. Clear DS_4X4_FULL_EXPERTS for
        # the worker so the result is deterministic regardless of any value left in the launching
        # shell — the loop test then halves to 128 experts and forces HOST_ALL. (Forcing 256 experts
        # on the device gate currently stalls on the ring; see the parametrize note above.)
        extra_env["DS_4X4_FULL_EXPERTS"] = ""

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
