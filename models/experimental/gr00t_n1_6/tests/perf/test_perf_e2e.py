# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end performance benchmark for GR00T N1.6 on Blackhole.

Measures latency for:
- Vision encoding (SigLIP2 + pixel shuffle + connector)
- Flow matching (4 Euler steps with 32-layer DiT)
- Full pipeline

Run:
    cd /home/ttuser/experiments/pi0/tt-metal
    export TT_METAL_HOME=$(pwd) ARCH_NAME=blackhole
    pytest models/experimental/gr00t_n1_6/tests/perf/test_perf_e2e.py -svv
"""

import sys
import time
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))


@pytest.fixture(scope="module")
def tt_device():
    import ttnn

    device = ttnn.open_device(device_id=0)
    yield device
    ttnn.close_device(device)


@pytest.fixture(scope="module")
def weight_loader():
    from models.experimental.gr00t_n1_6.common.weight_loader import Gr00tN16WeightLoader

    loader = Gr00tN16WeightLoader()
    loader.load()
    return loader


@pytest.fixture(scope="module")
def config():
    from models.experimental.gr00t_n1_6.common.configs import Gr00tN16Config

    return Gr00tN16Config.default()


@pytest.fixture(scope="module")
def model(config, weight_loader, tt_device):
    from models.experimental.gr00t_n1_6.tt.ttnn_groot_n16_model import Gr00tN16ModelTTNN

    return Gr00tN16ModelTTNN(config, weight_loader, tt_device)


class TestVisionPerf:
    """Vision encoding performance benchmarks."""

    def test_vision_latency(self, model):
        """Measure vision encoding latency (target: <30ms)."""
        pixel_values = torch.randn(1, 3, 224, 224)
        model.encode_vision(pixel_values)  # warmup

        times = []
        for _ in range(10):
            t0 = time.time()
            model.encode_vision(pixel_values)
            times.append(time.time() - t0)

        avg_ms = sum(times) / len(times) * 1000
        min_ms = min(times) * 1000
        max_ms = max(times) * 1000

        print(f"\n  Vision encoding latency:")
        print(f"    Average: {avg_ms:.1f}ms")
        print(f"    Min: {min_ms:.1f}ms")
        print(f"    Max: {max_ms:.1f}ms")

        assert avg_ms < 100, f"Vision encoding too slow: {avg_ms:.1f}ms (threshold: 100ms)"


class TestFlowMatchingPerf:
    """Flow matching performance benchmarks."""

    def test_flow_matching_latency(self, model, config, tt_device):
        """Measure flow matching latency for 4 Euler steps (target: <150ms)."""
        from models.experimental.gr00t_n1_6.tt.ttnn_common import to_tt_tensor

        backbone = to_tt_tensor(torch.randn(1, 64, 2048), tt_device)
        state = torch.randn(1, config.embodiment.max_state_dim)

        # Warmup
        model.run_flow_matching(backbone, state, embodiment_id=0)

        times = []
        for _ in range(5):
            b = to_tt_tensor(torch.randn(1, 64, 2048), tt_device)
            t0 = time.time()
            model.run_flow_matching(b, state, embodiment_id=0)
            times.append(time.time() - t0)

        avg_ms = sum(times) / len(times) * 1000
        min_ms = min(times) * 1000

        print(f"\n  Flow matching latency (4 steps):")
        print(f"    Average: {avg_ms:.1f}ms")
        print(f"    Min: {min_ms:.1f}ms")
        print(f"    Per step: {avg_ms/4:.1f}ms")

        assert avg_ms < 500, f"Flow matching too slow: {avg_ms:.1f}ms (threshold: 500ms)"


class TestEndToEndPerf:
    """Full pipeline performance benchmarks."""

    def test_e2e_latency(self, model, config, tt_device):
        """Measure full E2E latency: vision + flow matching (target: <200ms)."""
        import ttnn
        from models.experimental.gr00t_n1_6.tt.ttnn_common import to_tt_tensor

        pixel_values = torch.randn(1, 3, 224, 224)
        state = torch.randn(1, config.embodiment.max_state_dim)

        # Warmup
        img = model.encode_vision(pixel_values)
        backbone = to_tt_tensor(ttnn.to_torch(img), tt_device)
        model.run_flow_matching(backbone, state, embodiment_id=0)

        # Measure
        times = []
        for _ in range(5):
            t0 = time.time()
            img = model.encode_vision(pixel_values)
            backbone = to_tt_tensor(ttnn.to_torch(img), tt_device)
            actions = model.run_flow_matching(backbone, state, embodiment_id=0)
            times.append(time.time() - t0)

        avg_ms = sum(times) / len(times) * 1000
        min_ms = min(times) * 1000
        hz = 1000.0 / avg_ms

        print(f"\n  End-to-end performance:")
        print(f"    Average: {avg_ms:.1f}ms")
        print(f"    Min: {min_ms:.1f}ms")
        print(f"    Throughput: {hz:.1f} Hz")
        print(f"    Actions shape: {actions.shape}")

        assert avg_ms < 500, f"E2E too slow: {avg_ms:.1f}ms (threshold: 500ms)"

    def test_e2e_throughput_report(self, model, config, tt_device):
        """Generate throughput report with breakdown."""
        import ttnn
        from models.experimental.gr00t_n1_6.tt.ttnn_common import to_tt_tensor

        pixel_values = torch.randn(1, 3, 224, 224)
        state = torch.randn(1, config.embodiment.max_state_dim)

        # Warmup
        img = model.encode_vision(pixel_values)
        backbone = to_tt_tensor(ttnn.to_torch(img), tt_device)
        model.run_flow_matching(backbone, state, embodiment_id=0)

        # Measure components separately
        n_runs = 5

        # Vision
        vision_times = []
        for _ in range(n_runs):
            t0 = time.time()
            img = model.encode_vision(pixel_values)
            vision_times.append(time.time() - t0)

        # Flow matching
        flow_times = []
        for _ in range(n_runs):
            b = to_tt_tensor(torch.randn(1, 64, 2048), tt_device)
            t0 = time.time()
            model.run_flow_matching(b, state, embodiment_id=0)
            flow_times.append(time.time() - t0)

        vision_ms = sum(vision_times) / len(vision_times) * 1000
        flow_ms = sum(flow_times) / len(flow_times) * 1000
        total_ms = vision_ms + flow_ms

        print(f"\n  {'='*50}")
        print(f"  GR00T N1.6 Performance Report (Blackhole p150a)")
        print(f"  {'='*50}")
        print(f"  Vision (SigLIP2 + connector): {vision_ms:.1f}ms")
        print(f"  Flow matching (4 steps):      {flow_ms:.1f}ms")
        print(f"  {'─'*50}")
        print(f"  Total:                        {total_ms:.1f}ms")
        print(f"  Throughput:                   {1000/total_ms:.1f} Hz")
        print(f"  {'='*50}")
