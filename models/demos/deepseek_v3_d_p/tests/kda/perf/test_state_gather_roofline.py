# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hand-checked values for the KDA state all-gather roofline (host only)."""

import pytest

from models.demos.deepseek_v3_d_p.tests.kda.perf.state_gather_roofline import gather_roofline


def test_galaxy_recurrent_shard_is_fabric_bound():
    # [1, 3, 128, 128] FP32 = 196608 B per rank, SP ring of 8, 2 links, 2D fabric.
    roofline = gather_roofline(shard_bytes=196608, num_ranks=8, num_links=2, wraps=True, fabric_2d=True)
    assert roofline.received_bytes == 1376256  # 7 * 196608
    assert roofline.inbound_links == 4  # 2 links x 2 ring directions
    assert roofline.hops == 4
    assert roofline.fabric_ns == pytest.approx(13762.56)  # 1376256 B / 100 GB/s
    assert roofline.latency_ns == pytest.approx(2476.0)  # 4 x 619 ns
    assert roofline.dram_ns == pytest.approx(3072.0)  # 8 x 196608 B / 512 GB/s
    assert roofline.ideal_ns == pytest.approx(13762.56)
    assert roofline.bound == "fabric"
    assert roofline.utilization(27525.12) == pytest.approx(0.5)
    assert roofline.effective_gb_per_s(27525.12) == pytest.approx(50.0)


def test_galaxy_convolution_shard_is_latency_bound():
    # 3 branches x 3 heads x 2 halves x 384 B = 6912 B per rank.
    roofline = gather_roofline(shard_bytes=6912, num_ranks=8, num_links=2, wraps=True, fabric_2d=True)
    assert roofline.fabric_ns == pytest.approx(483.84)  # 48384 B / 100 GB/s
    assert roofline.dram_ns == pytest.approx(108.0)  # 55296 B / 512 GB/s
    assert roofline.ideal_ns == pytest.approx(2476.0)
    assert roofline.bound == "latency"


def test_line_uses_one_direction_and_full_diameter():
    roofline = gather_roofline(shard_bytes=1000, num_ranks=4, num_links=1, wraps=False, fabric_2d=False)
    assert roofline.inbound_links == 1
    assert roofline.hops == 3
    assert roofline.fabric_ns == pytest.approx(120.0)  # 3000 B / 25 GB/s
    assert roofline.latency_ns == pytest.approx(1545.0)  # 3 x 515 ns
    assert roofline.dram_ns == pytest.approx(7.8125)  # 4000 B / 512 GB/s


def test_utilization_is_not_capped():
    roofline = gather_roofline(shard_bytes=6912, num_ranks=8, num_links=2, wraps=True, fabric_2d=True)
    assert roofline.utilization(1238.0) == pytest.approx(2.0)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        (dict(shard_bytes=0, num_ranks=8, num_links=2), "shard_bytes must be positive"),
        (dict(shard_bytes=1, num_ranks=1, num_links=2), "at least 2 ranks"),
        (dict(shard_bytes=1, num_ranks=8, num_links=0), "num_links must be at least 1"),
    ],
    ids=["empty-shard", "one-rank", "no-links"],
)
def test_invalid_inputs_are_rejected(kwargs, message, expect_error):
    with expect_error(ValueError, message):
        gather_roofline(wraps=True, fabric_2d=True, **kwargs)


def test_nonpositive_measurement_is_rejected(expect_error):
    roofline = gather_roofline(shard_bytes=1, num_ranks=2, num_links=1, wraps=False, fabric_2d=True)
    with expect_error(ValueError, "measured time must be positive"):
        roofline.utilization(0.0)
