#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Command choreography for a future paired native cancellation/restart device gate."""

from cancel_restart_contract import CAPACITY, CHUNK, LAYERS, check_cancelled_pair, check_restart
from delayed_peer import passive_arm, source_before_arm


def run_cancel_epoch(role, bridge, publish, wait_peer, produce, check, *, tokens, page_receipt):
    uuid = 700
    check()
    if role == "passive":
        passive_arm(bridge, page_receipt, publish, wait_peer, check)
        wait_peer("chunk-issued")
        terminal = bridge.snapshot_until(
            lambda row: any(
                item.get("uuid") == uuid and item.get("cancelled") is True
                for item in row.get("inbound_generations", [])
            )
        )
        publish("cancel-terminal", snapshot=terminal)
        source = wait_peer("cancel-terminal")["snapshot"]
        check_cancelled_pair(source, terminal, uuid)
    else:
        reply = bridge.rpc("register", request_id=11, uuid=uuid, tokens=tokens, **{"from": 0, "to": CAPACITY})
        if reply.get("slot") != 0:
            raise RuntimeError("source cancellation used the wrong slot")
        bridge.rpc("prepare", slot=0, request_id=11, begin=0, end=CHUNK)
        source_before_arm(bridge, produce, publish, wait_peer, check)
        issued = bridge.snapshot_until(
            lambda row: row.get("acks") == LAYERS
            and sum(item.get("op") == "layer" for item in row.get("calls", [])) == LAYERS
            and not any(item.get("op") == "seal" for item in row.get("calls", []))
        )
        publish("chunk-issued", snapshot=issued)
        bridge.rpc("cancel", uuid=uuid)
        terminal = bridge.snapshot_until(
            lambda row: any(
                item.get("uuid") == uuid and item.get("cancelled") is True for item in row.get("generations", [])
            )
        )
        publish("cancel-terminal", snapshot=terminal)
        passive = wait_peer("cancel-terminal")["snapshot"]
        check_cancelled_pair(terminal, passive, uuid)
    check()
    return bridge.rpc("drain_cancelled", uuid=uuid)


def run_restart_epoch(role, bridge, publish, wait_peer, produce, page_receipt, restart_identity, check, *, tokens):
    uuid = 701
    check()
    if role == "passive":
        before = page_receipt("before")
        bridge.rpc("expect", uuid=uuid, slot=1, **{"from": 0, "to": 32})
        publish("restart-armed", pages=before)
        wait_peer("restart-issued")
        terminal = bridge.snapshot_until(
            lambda row: any(
                item.get("uuid") == uuid and item.get("complete") is True for item in row.get("inbound_generations", [])
            )
        )
        source = wait_peer("restart-terminal")
        receipt = dict(source["receipt"])
        receipt["passive_sentinel"] = before
        receipt["passive_after"] = page_receipt("after")
        check_restart(receipt)
        publish("restart-verified", snapshot=terminal)
    else:
        wait_peer("restart-armed")
        reply = bridge.rpc("register", request_id=12, uuid=uuid, tokens=tokens, **{"from": 0, "to": 32})
        if reply.get("slot") != 0:
            raise RuntimeError("restart source used the wrong slot")
        bridge.rpc("prepare", slot=0, request_id=12, begin=0, end=32)
        runtime_receipt = produce(LAYERS)
        before = runtime_receipt["source_after"]
        terminal = bridge.snapshot_until(
            lambda row: any(
                item.get("uuid") == uuid and item.get("successful") is True for item in row.get("generations", [])
            )
        )
        calls = terminal["calls"]
        receipt = {
            "calls": calls,
            "source_before": before,
            "selected_groups": sorted(before),
            "synthetic_acks": 0,
            "real_layer_acks": LAYERS,
            "runtime_h2d_tested": True,
            "runtime_receipt": runtime_receipt,
            **restart_identity(),
        }
        publish("restart-issued", snapshot=terminal)
        publish("restart-terminal", snapshot=terminal, receipt=receipt)
        wait_peer("restart-verified")
    check()
    return bridge.drain()
