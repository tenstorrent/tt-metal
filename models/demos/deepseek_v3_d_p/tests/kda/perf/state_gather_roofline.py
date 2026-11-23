# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Theoretical lower bound for an all-gather of the KDA carry over one mesh axis.

The same model as ``ttnn.all_gather``'s ``create_op_performance_model``
(``ttnn/cpp/ttnn/operations/ccl/all_gather/device/all_gather_device_operation.cpp``) for a one-axis gather, written out
so variants without a C++ model (``high_bw_all_gather``) are judged against the same bound:

* fabric: the worst-connected rank still receives the (N-1) shards it lacks, ``(N-1) * S`` bytes, over its inbound
  links ``K = num_links * (2 if the axis wraps else 1)``, each at the per-link bandwidth;
* latency: the farthest shard crosses the axis diameter, ``N // 2`` hops on a ring and ``N - 1`` on a line, at the
  per-hop fabric latency;
* DRAM: each rank reads its shard ``S`` and writes the gathered ``N * S``, at the aggregate DRAM bandwidth.

The three overlap, so ``ideal = max(fabric, latency, dram)``. The C++ DRAM term uses an empirical per-transaction
table instead of a single aggregate bandwidth; here DRAM never binds for the KDA carry (it is under a quarter of the
fabric term), so the choice does not move the ideal.

Capability constants (Blackhole), with their sources:

* link bandwidth 25 GB/s per link per direction: ``ccl_common.cpp`` ``lookup_fabric_link_bw`` (hardware 50 GB/s,
  "currently limited to half"; ``BLACKHOLE_LINK_HARDWARE_GB_PER_S`` keeps the hardware figure for what-if reports);
* per-hop latency 619 ns on a 2D fabric, 515 ns on a 1D fabric: ``ccl_common.cpp`` ``lookup_fabric_hop_latency_ns``,
  measured p50 first-packet forwarding cost, so an empirical estimate, not a hardware floor;
* DRAM 512 GB/s aggregate: Blackhole p150 product specification (8 GDDR6 channels), not an in-repo constant.
"""

from __future__ import annotations

from dataclasses import dataclass

BLACKHOLE_LINK_GB_PER_S = 25.0
BLACKHOLE_LINK_HARDWARE_GB_PER_S = 50.0
BLACKHOLE_HOP_LATENCY_NS_2D = 619.0
BLACKHOLE_HOP_LATENCY_NS_1D = 515.0
BLACKHOLE_DRAM_GB_PER_S = 512.0


@dataclass(frozen=True)
class GatherRoofline:
    """Ideal time of one all-gather and its independently inspectable terms, in ns (bytes / (GB/s) = ns)."""

    shard_bytes: int
    num_ranks: int
    inbound_links: int
    hops: int
    fabric_ns: float
    latency_ns: float
    dram_ns: float

    @property
    def received_bytes(self) -> int:
        """Bytes the worst-connected rank must receive: (N-1) * S."""
        return (self.num_ranks - 1) * self.shard_bytes

    @property
    def ideal_ns(self) -> float:
        return max(self.fabric_ns, self.latency_ns, self.dram_ns)

    @property
    def bound(self) -> str:
        terms = {"fabric": self.fabric_ns, "latency": self.latency_ns, "dram": self.dram_ns}
        return max(terms, key=terms.get)

    def utilization(self, measured_ns: float) -> float:
        """ideal / measured, uncapped."""
        if measured_ns <= 0:
            raise ValueError(f"measured time must be positive, got {measured_ns}")
        return self.ideal_ns / measured_ns

    def effective_gb_per_s(self, measured_ns: float) -> float:
        """(N-1) * S / measured."""
        if measured_ns <= 0:
            raise ValueError(f"measured time must be positive, got {measured_ns}")
        return self.received_bytes / measured_ns


def gather_roofline(
    *,
    shard_bytes: int,
    num_ranks: int,
    num_links: int,
    wraps: bool,
    fabric_2d: bool,
    link_gb_per_s: float = BLACKHOLE_LINK_GB_PER_S,
    dram_gb_per_s: float = BLACKHOLE_DRAM_GB_PER_S,
) -> GatherRoofline:
    """Roofline of an all-gather of ``shard_bytes`` per rank over a ``num_ranks`` axis (Blackhole constants)."""
    if shard_bytes <= 0:
        raise ValueError(f"shard_bytes must be positive, got {shard_bytes}")
    if num_ranks < 2:
        raise ValueError(f"an all-gather needs at least 2 ranks, got {num_ranks}")
    if num_links < 1:
        raise ValueError(f"num_links must be at least 1, got {num_links}")
    if link_gb_per_s <= 0 or dram_gb_per_s <= 0:
        raise ValueError("bandwidths must be positive")
    inbound_links = num_links * (2 if wraps else 1)
    hops = num_ranks // 2 if wraps else num_ranks - 1
    hop_latency_ns = BLACKHOLE_HOP_LATENCY_NS_2D if fabric_2d else BLACKHOLE_HOP_LATENCY_NS_1D
    received = (num_ranks - 1) * shard_bytes
    return GatherRoofline(
        shard_bytes=shard_bytes,
        num_ranks=num_ranks,
        inbound_links=inbound_links,
        hops=hops,
        fabric_ns=received / (inbound_links * link_gb_per_s),
        latency_ns=hops * hop_latency_ns,
        dram_ns=max(shard_bytes, num_ranks * shard_bytes) / dram_gb_per_s,
    )
