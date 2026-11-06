# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "falcon40b_config(decode_configs): skip unsupported Falcon memory/mode/device-count parameter combinations",
    )


def pytest_collection_modifyitems(items):
    # These checks span separate parametrizations. Mark items without changing
    # their IDs or setting up mesh_device; actual grid checks stay in the tests.
    for item in items:
        marker = item.get_closest_marker("falcon40b_config")
        if marker is None:
            continue

        params = item.callspec.params
        llm_mode = params["llm_mode"]
        num_devices = params["num_devices"]
        if "memcfg" in params:
            # The end-to-end tests require eight chips in BOTH modes.
            unsupported_prefill = (llm_mode == "prefill" and params["memcfg"] != "DRAM") or num_devices != 8
            unsupported_decode = llm_mode == "decode" and params["memcfg"] != "SHARDED"
        else:
            model_config_str = params["model_config_str"]
            unsupported_prefill = llm_mode == "prefill" and (
                model_config_str not in ("BFLOAT8_B-DRAM", "BFLOAT16-DRAM") or num_devices != 8
            )
            decode_configs = marker.kwargs.get("decode_configs", ("BFLOAT8_B-SHARDED", "BFLOAT16-SHARDED"))
            unsupported_decode = llm_mode == "decode" and model_config_str not in decode_configs

        if unsupported_prefill:
            item.add_marker(pytest.mark.skip(reason="Prefill is only supported for DRAM memory config and 8 chips!"))
        elif unsupported_decode:
            item.add_marker(pytest.mark.skip(reason="Decode is only supported for SHARDED memory config!"))
