# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Match WHISPER_L1_SMALL_SIZE in ttnn_optimized_functional_whisper (avoid importing that module here).
_WHISPER_L1_SMALL_SIZE = 1024

# Trace region for test_encoder (encoder capture/replay). Moderate size to avoid OOM vs 100M default.
_ENCODER_TEST_TRACE_REGION_SIZE = 16_000_000


def pytest_generate_tests(metafunc):
    """test_encoder always uses fused trace path; reserve a trace region on the device."""
    if metafunc.definition is None or metafunc.definition.name != "test_encoder":
        return
    metafunc.parametrize(
        "device_params",
        [{"l1_small_size": _WHISPER_L1_SMALL_SIZE, "trace_region_size": _ENCODER_TEST_TRACE_REGION_SIZE}],
        indirect=True,
        scope="function",
    )
