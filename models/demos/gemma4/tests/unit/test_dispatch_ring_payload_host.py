# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only: the ring transport's fabric payload contract (dispatch_fabric2d sends one whole bf16 token page plus
64 B of forwarding metadata per packet). The fabric default is 4352 B on the PR base and on main alike, so the
26B's 2816-wide hidden (5696 B needed) is refused by the planner with the router config that works (6528 B, six
bfp8-tile units, within the Blackhole maximum); the ring unit tests open the mesh with it. No device, no conftest."""

from models.demos.gemma4.tt.experts.dispatch_prefill import (
    FABRIC_DEFAULT_PAYLOAD_BYTES,
    FABRIC_MAX_PAYLOAD_BYTES_BLACKHOLE,
    RING_FWD_EXTRA_BYTES,
    ring_fabric_payload_bytes,
    ring_fabric_router_config,
    ring_payload_required_bytes,
    ring_token_bytes,
    ring_transport_reason,
)

HIDDEN_26B = 2816  # google/gemma-4-26B-A4B-it text hidden_size


def test_26b_token_needs_more_than_the_fabric_default_on_both_trees():
    assert ring_token_bytes(HIDDEN_26B) == 5632 and RING_FWD_EXTRA_BYTES == 64
    assert ring_payload_required_bytes(HIDDEN_26B) == 5696
    assert FABRIC_DEFAULT_PAYLOAD_BYTES == 4352  # the PR base and main: 4 bfp8 tiles, static_assert in the builder
    reason = ring_transport_reason(HIDDEN_26B, FABRIC_DEFAULT_PAYLOAD_BYTES)
    assert reason is not None and "5696" in reason and "4352" in reason and "6528" in reason, reason


def test_router_config_for_the_26b_is_the_smallest_fitting_unit():
    assert ring_fabric_payload_bytes(HIDDEN_26B) == 6528  # 6 x 1088 >= 5696
    assert ring_fabric_payload_bytes(HIDDEN_26B) % 64 == 0  # the fabric's L1 alignment check
    assert ring_fabric_payload_bytes(HIDDEN_26B) <= FABRIC_MAX_PAYLOAD_BYTES_BLACKHOLE
    assert ring_transport_reason(HIDDEN_26B, ring_fabric_payload_bytes(HIDDEN_26B)) is None
    cfg = ring_fabric_router_config(HIDDEN_26B)
    assert cfg.max_packet_payload_size_bytes == 6528


def test_small_tokens_keep_the_default():
    assert ring_fabric_payload_bytes(1152) == FABRIC_DEFAULT_PAYLOAD_BYTES  # 2304 + 64 fits
    assert ring_transport_reason(2144, FABRIC_DEFAULT_PAYLOAD_BYTES) is None  # 4288 + 64 = 4352 exactly
    assert ring_transport_reason(2176, FABRIC_DEFAULT_PAYLOAD_BYTES) is not None  # one tile over
