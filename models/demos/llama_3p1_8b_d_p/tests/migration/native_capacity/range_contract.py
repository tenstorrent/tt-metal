"""Import-free six-generation receipts. Token endpoints are not byte/page counts."""

import json
from pathlib import Path

from check_scenario import validate
from runner_support import require, sha256


def scenario(path):
    value = json.loads(Path(path).read_bytes())
    validate(value)
    return value


def rpc(bridge, command):
    command = dict(command)
    op = command.pop("op")
    command.pop("id", None)
    return bridge.rpc(op, **command)


def generation(value, phase, role):
    key = "generations" if role == "source" else "inbound_generations"
    uuid = phase["source_command"]["uuid"]
    rows = [r for r in value.get(key, []) if r.get("uuid") == uuid]
    require(len(rows) == 1, "missing/duplicate generation")
    row = rows[0]
    spec = phase["source_command"]
    slot = phase[role + "_slot"] if role == "source" else phase["destination_slot"]
    require(
        (row["slot"], row["from"], row["to"], row["selected_token_count"])
        == (slot, spec["from"], spec["to"], spec["to"] - spec["from"]),
        "generation range/slot differs",
    )
    if role == "source":
        require(
            row["successful"] is True
            and row["request_id"] == spec["request_id"]
            and row["prompt_len"] == phase["valid_prompt_tokens"]
            and row["reused"] == phase["reused_tokens"]
            and row["chunk_count"] == len(phase["compute_calls"]),
            "source generation not complete",
        )
        slots = [r for r in value["slots"] if r["slot"] == slot]
        require(
            len(slots) == 1
            and slots[0]["position"] == phase["valid_prompt_tokens"]
            and slots[0]["request_id"] == spec["request_id"]
            and slots[0]["pins"] == slots[0]["in_flight"] == slots[0]["pending"] == 0
            and slots[0]["idle_resident"] is True
            and slots[0]["evict_pending"] is False,
            "source generation not drained",
        )
        calls = value["calls"]
        registers = [r for r in calls if r["op"] == "register" and r["uuid"] == uuid]
        require(
            len(registers) == 1 and registers[0]["src"] == slot and registers[0]["from"] == phase["reused_tokens"],
            "wrong source registration identity",
        )
        own = [r for r in calls if r["transfer"] == registers[0]["transfer"]]
        ready = [r for r in own if r["op"] == "peer_ready"]
        done = [r for r in own if r["op"] == "completion"]
        require(
            len(ready) == len(done) == 1
            and (ready[0]["src"], ready[0]["dst"], ready[0]["from"], ready[0]["to"])
            == (slot, phase["destination_slot"], spec["from"], spec["to"]),
            "wrong readiness mapping",
        )
        require(
            done[0]["status"] == 0
            and done[0]["src"] == slot
            and done[0]["tokens"] == done[0]["completion_end"] == spec["to"],
            "source completion must report exclusive endpoint",
        )
        expected = []
        for index, call in enumerate(phase["compute_calls"]):
            for layer in range(32):
                ranges = ([(0, phase["reused_tokens"])] if index == 0 and phase["reused_tokens"] else []) + [
                    (call["begin"], call["end"])
                ]
                for begin, end in ranges:
                    begin = max(begin, spec["from"])
                    end = min(end, spec["to"])
                    if begin < end:
                        expected.append((slot, phase["destination_slot"], layer, begin, end))
        actual = [(r["src"], r["dst"], r["layer"], r["from"], r["to"]) for r in own if r["op"] == "layer"]
        require(actual == expected, "missing/duplicate/reordered native layer ranges")
        require(
            [r["op"] for r in own] == ["register", "peer_ready"] + ["layer"] * len(expected) + ["seal", "completion"],
            "registration/readiness/seal/completion order differs",
        )
    else:
        require(
            row["complete"] is True
            and row["status"] == 0
            and row["tokens"] == row["completion_end"] == spec["to"]
            and row["expected_reused"] == row["peer_reused"] == phase["passive_command"]["expected_reused"]
            and row["peer_reused_known"] is True,
            "passive endpoint/reuse metadata differs",
        )
    return row


def phase_receipt(value, phase, nonce, role):
    require(
        value.get("run_nonce") == nonce
        and value.get("role") == role
        and value.get("ok") is True
        and value.get("uuid") == phase["source_command"]["uuid"],
        "stale/wrong phase receipt",
    )
    return value


def snapshot_receipt(value, phase, nonce, role):
    path = Path(value["receipt_path"])
    require(sha256(path) == value["receipt_sha256"], "snapshot receipt hash differs")
    loaded = json.loads(path.read_bytes())
    require(
        all(
            loaded.get(k) == v
            for k, v in value.items()
            if k not in ("receipt_path", "receipt_sha256", "snapshot_complete_ns")
        ),
        "snapshot receipt contents differ",
    )
    identity = loaded["identity"]
    require(
        identity["run_nonce"] == nonce
        and identity["role"] == role
        and identity["uuid"] == phase["source_command"]["uuid"],
        "wrong snapshot identity",
    )
    return loaded


class AckRecorder:
    def __init__(self, calls):
        self.calls = calls
        self.active = None
        self.completed_ns = None
        self.rows = []

    def begin(self, request_id, slot, start, end, started_ns):
        require(
            self.active is None and request_id == len(self.rows) // 32 and request_id < len(self.calls),
            "wrong call ordinal",
        )
        call = self.calls[request_id]
        require((slot, start, end) == (call["slot"], call["begin"], call["end"]), "wrong call range")
        self.active = (request_id, slot, start, end, started_ns)
        self.completed_ns = None

    def synchronized(self, when_ns):
        if self.active is not None:
            require(when_ns >= self.active[4], "sync predates call")
            self.completed_ns = when_ns

    def ack(self, layer, request_id, when_ns):
        require(self.active is not None and self.completed_ns is not None, "ack before successful sync")
        active, slot, start, end, _ = self.active
        require(
            request_id == active
            and layer == len(self.rows) - active * 32
            and 0 <= layer < 32
            and when_ns >= self.completed_ns,
            "wrong callback ordering",
        )
        self.rows.append(
            dict(
                request_id=request_id,
                slot=slot,
                start=start,
                end=end,
                layer=layer,
                synchronized_ns=self.completed_ns,
                ack_ns=when_ns,
            )
        )

    def finish(self, routed):
        require(
            self.active is not None and len(self.rows) == (self.active[0] + 1) * 32 and routed == 32, "wrong ack count"
        )
        self.active = None
