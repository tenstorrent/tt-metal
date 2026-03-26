# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from models.tt_transformers.demo import trace_region_config


def _make_request(data_parallel=1):
    return SimpleNamespace(node=SimpleNamespace(callspec=SimpleNamespace(params={"data_parallel": data_parallel})))


def test_get_supported_trace_region_size_for_llama_32_90b_t3k(monkeypatch):
    monkeypatch.setenv("HF_MODEL", "meta-llama/Llama-3.2-90B-Vision-Instruct")
    monkeypatch.setenv("MESH_DEVICE", "T3K")
    monkeypatch.setattr(trace_region_config, "get_mesh_device_name", lambda num_devices, mesh_device_name: "T3K")

    assert trace_region_config.get_supported_trace_region_size(_make_request(), (1, 8)) == 20_000_000
