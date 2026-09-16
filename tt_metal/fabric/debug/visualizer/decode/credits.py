# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Normalize per-channel credits and a viewer-only stall score."""

from __future__ import annotations

import re
from typing import Any

from .merge import endpoint_key
from .structs import CONNECTION_STATE

_SENDER = re.compile(r"^(?:credits\.)?sender\.(\d+)\.(.+)$")
_RECEIVER_PKTS = re.compile(r"^(?:credits\.)?receiver\.(\d+)\.pkts_sent$")
_RECEIVER_RING = re.compile(r"^receiver\.(\d+)\.ring$")
_DOWNSTREAM = re.compile(r"^credits\.downstream\.vc(\d+)\.edge(\d+)\.free_slots$")
_TO_SENDER_ACK = "credits.to_sender_ack"
_TO_SENDER_COMPLETED = "credits.to_sender_completion"


def _counter_at(counters: Any, index: int) -> Any:
    if not isinstance(counters, list) or index >= len(counters):
        return None
    return counters[index]


def _vc_for_flat(counts: list[int], index: int) -> int | None:
    remaining = index
    for vc, count in enumerate(counts):
        if remaining < count:
            return vc
        remaining -= count
    return None


def _uses_counters(instance: dict[str, Any] | None, vc: int | None) -> bool:
    if instance is None or vc is None:
        return False
    plan = instance.get("credit_plan") or {}
    return bool(plan.get(f"vc{vc}_uses_counters"))


def _stream_post(region: dict[str, Any]) -> tuple[int | None, str]:
    if region["status"] == "torn":
        return None, "torn"
    value = region.get("value")
    if region["status"] != "ok" or not isinstance(value, dict) or value.get("post") is None:
        return None, region["status"] if region["status"] != "ok" else "unknown"
    if value.get("torn"):
        return None, "torn"
    return int(value["post"]), "ok"


def _occupancy(depth: int | None, used: int | None, status: str) -> tuple[int | None, str]:
    if depth is None or used is None or status != "ok":
        return None, "unknown" if status == "ok" else status
    occupied = used
    if occupied < 0 or (depth is not None and occupied > depth):
        return occupied, "inconsistent"
    return occupied, "ok"


def decode_channels(router: dict[str, Any]) -> dict[str, Any]:
    """Build the channels block and fill ring occupancy from stream counts."""

    instance = router.get("instance") or {}
    sender_counts = [int(value) for value in instance.get("sender_channels_per_vc") or []]
    worker = int(instance.get("worker_sender_channel", 0))
    warnings = router.setdefault("warnings", [])
    regions = {region["id"]: region for region in router.get("regions", [])}

    senders: dict[int, dict[str, Any]] = {}
    receivers: dict[int, dict[str, Any]] = {}
    downstream: list[dict[str, Any]] = []
    to_sender_ack = (regions.get(_TO_SENDER_ACK) or {}).get("value") or {}
    to_sender_completed = (regions.get(_TO_SENDER_COMPLETED) or {}).get("value") or {}
    ack_counters = to_sender_ack.get("counters")
    completed_counters = to_sender_completed.get("counters")

    def sender(index: int) -> dict[str, Any]:
        if index not in senders:
            vc = _vc_for_flat(sender_counts, index)
            senders[index] = {
                "index": index,
                "vc": vc,
                "role": "worker" if index == worker else "upstream",
                "depth": None,
                "free_slots": None,
                "occupied": None,
                "acked_pending": None,
                "completed_pending": None,
                "credit_backing": "counter" if _uses_counters(instance, vc) else "stream_reg",
                "connection": {"raw": None, "name": None},
                "torn": False,
                "status": "ok",
            }
        return senders[index]

    def receiver(index: int) -> dict[str, Any]:
        if index not in receivers:
            receivers[index] = {
                "index": index,
                "vc": _vc_for_flat(
                    [int(value) for value in instance.get("receiver_channels_per_vc") or []],
                    index,
                ),
                "depth": None,
                "pkts_pending": None,
                "torn": False,
                "status": "ok",
            }
        return receivers[index]

    for region in router.get("regions", []):
        if not region.get("enabled"):
            continue
        match = _SENDER.match(region["id"])
        if match:
            channel = sender(int(match.group(1)))
            suffix = match.group(2)
            if suffix == "ring":
                channel["depth"] = int(region.get("count", 0))
            elif suffix == "free_slots":
                value, status = _stream_post(region)
                channel["free_slots"] = value
                channel["torn"] = status == "torn" or channel["torn"]
                if status != "ok":
                    channel["status"] = status
            elif suffix == "credits.acked":
                value, status = _stream_post(region)
                if channel["credit_backing"] == "stream_reg":
                    channel["acked_pending"] = value
                    if status == "torn":
                        channel["torn"] = True
            elif suffix == "credits.completed":
                value, status = _stream_post(region)
                if channel["credit_backing"] == "stream_reg":
                    channel["completed_pending"] = value
                    if status == "torn":
                        channel["torn"] = True
            elif suffix == "control.connection_sem":
                word = (region.get("value") or {}).get("word")
                channel["connection"] = {"raw": word, "name": CONNECTION_STATE.get(word)}
            continue

        match = _RECEIVER_RING.match(region["id"])
        if match:
            receiver(int(match.group(1)))["depth"] = int(region.get("count", 0))
            continue
        match = _RECEIVER_PKTS.match(region["id"])
        if match:
            channel = receiver(int(match.group(1)))
            value, status = _stream_post(region)
            channel["pkts_pending"] = value
            channel["torn"] = status == "torn"
            if status != "ok":
                channel["status"] = status
            continue
        match = _DOWNSTREAM.match(region["id"])
        if match:
            value, status = _stream_post(region)
            downstream.append(
                {
                    "vc": int(match.group(1)),
                    "edge": int(match.group(2)),
                    "free_slots": value,
                    "depth": None,
                    "torn": status == "torn",
                    "status": status,
                }
            )

    for channel in senders.values():
        if channel["credit_backing"] == "counter":
            channel["acked_pending"] = None
            channel["completed_pending"] = None
            channel["counters"] = {
                "to_sender_ack": _counter_at(ack_counters, channel["index"]),
                "to_sender_completion": _counter_at(completed_counters, channel["index"]),
            }
        used = None if channel["depth"] is None or channel["free_slots"] is None else channel["depth"] - channel["free_slots"]
        occupied, occupancy_status = _occupancy(channel["depth"], used, channel["status"])
        channel["occupied"] = occupied
        if occupancy_status == "inconsistent":
            channel["status"] = "inconsistent"
            warnings.append(f"sender.{channel['index']} occupancy {occupied} is outside depth {channel['depth']}")

    for channel in receivers.values():
        occupied, occupancy_status = _occupancy(channel["depth"], channel["pkts_pending"], channel["status"])
        if occupancy_status == "inconsistent":
            channel["status"] = "inconsistent"
            channel["pkts_pending"] = occupied
            warnings.append(
                f"receiver.{channel['index']} pkts_pending {occupied} is outside depth {channel['depth']}"
            )

    rings_by_id = {ring["id"]: ring for ring in router.get("rings", [])}
    for index, channel in senders.items():
        ring = rings_by_id.get(f"sender.{index}.ring")
        if ring is None:
            continue
        ring["occupied_count"] = channel["occupied"] if channel["occupied"] is not None and channel["occupied"] >= 0 else None
        if channel["status"] == "torn":
            ring["occupancy_source"] = None
            ring["occupancy_status"] = "unknown"
        elif channel["occupied"] is None:
            ring["occupancy_source"] = None
            ring["occupancy_status"] = "unknown"
        else:
            ring["occupancy_source"] = "stream"
            ring["occupancy_status"] = "inconsistent" if channel["status"] == "inconsistent" else "ok"

    for index, channel in receivers.items():
        ring = rings_by_id.get(f"receiver.{index}.ring")
        if ring is None:
            continue
        pending = channel["pkts_pending"]
        ring["occupied_count"] = pending if pending is not None and pending >= 0 else None
        if channel["status"] == "torn" or pending is None:
            ring["occupancy_source"] = None
            ring["occupancy_status"] = "unknown"
        else:
            ring["occupancy_source"] = "stream"
            ring["occupancy_status"] = "inconsistent" if channel["status"] == "inconsistent" else "ok"

    return {
        "senders": [senders[index] for index in sorted(senders)],
        "receivers": [receivers[index] for index in sorted(receivers)],
        "downstream": sorted(downstream, key=lambda item: (item["vc"], item["edge"])),
    }


def stall_score(router: dict[str, Any]) -> float | None:
    """Heuristic occupancy colouring. Idle-healthy and hung-backpressure can look identical."""

    if router["capture"]["status"] != "ok":
        return None
    channels = router.get("channels") or {}
    scores: list[float] = []
    usable = False
    for channel in channels.get("senders", []):
        if channel["status"] in {"torn", "unknown", "inconsistent"}:
            continue
        usable = True
        depth = channel.get("depth")
        occupied = channel.get("occupied")
        if depth and occupied is not None and occupied >= 0:
            scores.append(min(occupied / depth, 1.0))
    for channel in channels.get("receivers", []):
        if channel["status"] in {"torn", "unknown", "inconsistent"}:
            continue
        usable = True
        depth = channel.get("depth")
        pending = channel.get("pkts_pending")
        if depth and pending is not None and pending >= 0:
            scores.append(min(pending / depth, 1.0))
    for channel in channels.get("downstream", []):
        if channel["status"] in {"torn", "unknown", "inconsistent"}:
            continue
        usable = True
        if channel.get("free_slots") == 0:
            scores.append(1.0)
        elif isinstance(channel.get("free_slots"), int) and channel["free_slots"] > 0:
            scores.append(0.0)
    if not usable:
        return None
    return max(scores) if scores else 0.0


def annotate_links(decoded: dict[str, Any]) -> None:
    """Copy each router's stall score onto every outgoing topology edge."""

    owners = {endpoint_key(router["id"]): router for router in decoded["routers"]}
    for link in decoded["topology"]["links"]:
        owner = owners.get(endpoint_key(link["src"]))
        if owner is None:
            link["stall_score"] = None
            link["status"] = "not_captured"
        else:
            link["stall_score"] = owner.get("stall_score")
            link["status"] = owner["capture"]["status"]
