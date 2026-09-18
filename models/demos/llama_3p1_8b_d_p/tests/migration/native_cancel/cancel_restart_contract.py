#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Pure host validators for the intentional cancel and fresh-manager restart gate."""

LAYERS = 32
CHUNK = 1024
CAPACITY = 2048
CONFIGS = 16
OK = 0
INTERNAL = 6


def require(condition, message):
    if not condition:
        raise ValueError(message)


def one(items, message):
    require(len(items) == 1, message)
    return items[0]


def check_cancelled_pair(source, passive, uuid):
    generations = [row for row in source.get("generations", []) if row.get("uuid") == uuid]
    generation = one(generations, "source cancelled generation missing or duplicated")
    require(len(source.get("generations", [])) == 1, "source cancellation must be the only generation")
    require(
        generation.get("chunk_count") == 1
        and generation.get("prompt_len") == CAPACITY
        and generation.get("from") == 0
        and generation.get("to") == CAPACITY,
        "source cancelled geometry differs",
    )
    require(
        generation.get("cancelled") is True
        and generation.get("successful") is False
        and generation.get("terminal_kind") == "PREFILL_DONE"
        and generation.get("terminal_position") == CHUNK,
        "source cancellation terminal differs",
    )
    require(source.get("acks") == LAYERS and source.get("successful") is False, "source ack total differs")

    calls = source.get("calls", [])
    register = one(
        [row for row in calls if row.get("op") == "register" and row.get("uuid") == uuid],
        "source register missing or duplicated",
    )
    transfer = register.get("transfer")
    selected = [row for row in calls if row.get("transfer") == transfer]
    expected_ops = ["register", "peer_ready"] + ["layer"] * LAYERS + ["cancel", "completion"]
    require([row.get("op") for row in selected] == expected_ops, "source layer or seal work appeared after cancel")
    ready = selected[1]
    require(
        ready.get("from") == 0 and ready.get("to") == CAPACITY,
        "source peer-ready interval differs",
    )
    layers = selected[2 : 2 + LAYERS]
    require(
        [(row.get("layer"), row.get("from"), row.get("to")) for row in layers]
        == [(layer, 0, CHUNK) for layer in range(LAYERS)],
        "source first-chunk layer inventory differs",
    )
    completion = selected[-1]
    require(
        completion.get("status") == OK and completion.get("tokens") == 0,
        "source completion must retain local OK/0 cancellation semantics",
    )
    slot = one(
        [row for row in source.get("slots", []) if row.get("slot") == generation.get("slot")],
        "source slot evidence missing",
    )
    require(
        slot.get("position") == CHUNK
        and slot.get("pins") == 0
        and slot.get("in_flight") == 0
        and slot.get("pending") == 0,
        "source claims did not drain",
    )

    inbound = one(
        [row for row in passive.get("inbound_generations", []) if row.get("uuid") == uuid],
        "passive cancelled generation missing or duplicated",
    )
    require(len(passive.get("inbound_generations", [])) == 1, "passive cancellation must be the only generation")
    require(
        inbound.get("from") == 0
        and inbound.get("to") == CAPACITY
        and inbound.get("complete") is True
        and inbound.get("retired") is False
        and inbound.get("cancelled") is True,
        "passive cancelled state differs",
    )
    passive_completion = one(
        [row for row in passive.get("calls", []) if row.get("op") == "completion"],
        "passive completion missing or duplicated",
    )
    # Passive snapshots expose terminal state per inbound generation and callback;
    # the aggregate successful field belongs only to source snapshots.
    require(
        passive_completion.get("transfer") == inbound.get("transfer")
        and passive_completion.get("status") == INTERNAL
        and passive_completion.get("tokens") == 0
        and inbound.get("status") == INTERNAL
        and inbound.get("tokens") == 0,
        "passive must retain INTERNAL/0 failed-landing semantics",
    )


def check_restart(receipt):
    require(
        receipt.get("new_nonce") and receipt.get("new_nonce") != receipt.get("old_nonce"),
        "restart did not use a fresh nonce",
    )
    for process in ("manager", "bridge"):
        old_pids = set(receipt.get(f"old_{process}_pids", {}).values())
        new_pids = set(receipt.get(f"new_{process}_pids", {}).values())
        require(
            len(old_pids) == len(new_pids) == 2 and not old_pids.intersection(new_pids),
            f"restart did not use fresh exact {process} identities",
        )
    groups = receipt.get("selected_groups", [])
    require(
        len(groups) == LAYERS * CONFIGS and len(groups) == len(set(groups)),
        "restart selected-group inventory differs",
    )
    source = receipt.get("source_before", {})
    sentinel = receipt.get("passive_sentinel", {})
    after = receipt.get("passive_after", {})
    require(set(source) == set(sentinel) == set(after) == set(groups), "restart page groups differ")
    require(all(source[key] != sentinel[key] for key in groups), "restart sentinel was not distinct in every group")
    require(all(source[key] == after[key] for key in groups), "restart after bytes differ from source")

    calls = receipt.get("calls", [])
    register = one([row for row in calls if row.get("op") == "register"], "restart register missing or duplicated")
    transfer = register.get("transfer")
    selected = [row for row in calls if row.get("transfer") == transfer]
    expected_ops = ["register", "peer_ready"] + ["layer"] * LAYERS + ["seal", "completion"]
    require([row.get("op") for row in selected] == expected_ops, "restart seal or command inventory differs")
    require(
        [(row.get("layer"), row.get("from"), row.get("to")) for row in selected[2 : 2 + LAYERS]]
        == [(layer, 0, 32) for layer in range(LAYERS)],
        "restart layer inventory differs",
    )
    completion = selected[-1]
    require(completion.get("status") == OK and completion.get("tokens") == 32, "restart completion is not exact OK/32")
    require(
        receipt.get("synthetic_acks") == 0 and receipt.get("real_layer_acks") == LAYERS,
        "restart real ack count differs",
    )
    require(receipt.get("runtime_h2d_tested") is True, "restart must use real runtime and H2D")


def check_lifecycle(events):
    require(len(events) == len(set(events)), "lifecycle evidence is duplicated")
    position = {event: index for index, event in enumerate(events)}
    required = [
        "cancel",
        "drain-control-source",
        "drain-control-passive",
        "manager-stopped-source-epoch-a",
        "manager-stopped-passive-epoch-a",
        "rewrite-passive-sentinel",
        "restart-source",
        "restart-passive",
        "manager-stopped-source-epoch-b",
        "manager-stopped-passive-epoch-b",
        "owner-cleanup",
    ]
    require(all(event in position for event in required), "lifecycle evidence is incomplete")
    cancel = position["cancel"]
    for role in ("source", "passive"):
        drain = position[f"drain-control-{role}"]
        stopped = position[f"manager-stopped-{role}-epoch-a"]
        require(cancel < drain < stopped, f"{role} epoch-A manager stopped before its drain")
    stopped_a = max(position["manager-stopped-source-epoch-a"], position["manager-stopped-passive-epoch-a"])
    for event in events[cancel + 1 : stopped_a]:
        require(event not in {"layer", "seal"}, "layer or seal work appeared after cancel")
    rewrite = position["rewrite-passive-sentinel"]
    require(
        stopped_a < rewrite < position["restart-source"] and rewrite < position["restart-passive"],
        "both epoch-A managers must stop before sentinel rewrite and restart",
    )
    for role in ("source", "passive"):
        restart = position[f"restart-{role}"]
        stopped = position[f"manager-stopped-{role}-epoch-b"]
        require(restart < stopped, f"{role} epoch-B manager stopped before its restart")
    stopped_b = max(position["manager-stopped-source-epoch-b"], position["manager-stopped-passive-epoch-b"])
    require(stopped_b < position["owner-cleanup"], "both epoch-B managers must stop before owner cleanup")
