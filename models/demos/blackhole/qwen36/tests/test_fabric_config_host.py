# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no device): the fabric packet payload that Qwen36MTPContractForCausalLM declares to vllm-tt-plugin.

* model_capabilities["fabric_config"] holds only a router config with an 8192 B max packet payload, no fabric type.
* vllm-tt-plugin's set_fabric (worker.py) passes that router config to ttnn.set_fabric_config and keeps the fabric type
  of the launch's override_tt_config (FABRIC_1D or FABRIC_1D_RING), or the hardware default without one.
"""

import enum
from types import SimpleNamespace

import pytest

import ttnn

qwen36_vllm = pytest.importorskip("models.demos.blackhole.qwen36.tt.qwen36_vllm")
MTP = qwen36_vllm.Qwen36MTPContractForCausalLM


def test_the_mtp_class_declares_an_8192_byte_fabric_packet_payload():
    fabric = MTP.model_capabilities["fabric_config"]
    assert set(fabric) == {"router_config"}
    assert fabric["router_config"].max_packet_payload_size_bytes == 8192


class _FabricConfig(enum.Enum):
    DISABLED = "disabled"
    FABRIC_1D = "1d"
    FABRIC_1D_RING = "1d_ring"
    FABRIC_2D_TORUS_XY = "2d_torus_xy"


@pytest.mark.parametrize(
    "tt_config, fabric_type",
    [
        ({"fabric_config": "FABRIC_1D"}, _FabricConfig.FABRIC_1D),
        ({"fabric_config": "FABRIC_1D_RING"}, _FabricConfig.FABRIC_1D_RING),
        (None, _FabricConfig.FABRIC_1D),
    ],
)
def test_the_plugin_keeps_the_launch_fabric_type_and_takes_the_payload(monkeypatch, tt_config, fabric_type):
    worker = pytest.importorskip("vllm_tt_plugin.worker")
    calls = []
    monkeypatch.setattr(ttnn, "FabricConfig", _FabricConfig)
    monkeypatch.setattr(
        ttnn,
        "cluster",
        SimpleNamespace(
            get_cluster_type=lambda: "P150_X8",
            ClusterType=SimpleNamespace(BLACKHOLE_GALAXY="BLACKHOLE_GALAXY", GALAXY="GALAXY"),
        ),
    )
    monkeypatch.setattr(ttnn, "set_fabric_config", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(worker, "ttnn", ttnn)

    router = MTP.model_capabilities["fabric_config"]["router_config"]
    worker.set_fabric(tt_config, 8, MTP.model_capabilities["fabric_config"])

    assert len(calls) == 1
    assert calls[0]["config"] is fabric_type
    assert calls[0]["router_config"] is router
