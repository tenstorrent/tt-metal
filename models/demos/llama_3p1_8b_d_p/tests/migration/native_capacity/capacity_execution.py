"""Exact real-prefill capacity cases, without tensor or native imports."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

CAPACITIES = (4096, 8192, 16384, 32768, 65536)
PAGE = 4352


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def capacity(value):
    require(type(value) is int and value in CAPACITIES, "Only requested4K..64K capacities are allowed")
    return value


@dataclass(frozen=True)
class Call:
    name: str
    prompt: str
    slot: int
    begin: int
    end: int


def make_cases(value, manifest_path, manifest_sha256):
    value = capacity(value)
    path = Path(manifest_path)
    require(hashlib.sha256(path.read_bytes()).hexdigest() == manifest_sha256, "Book manifest changed")
    manifest = json.loads(path.read_bytes())
    require(manifest["validation_passed"] is True, "Unvalidated tokens")
    fixtures = {}
    phases = []
    for slot in (0, 1):
        row = [x for x in manifest["fixtures"] if x["slot"] == slot and x["context_length"] == value]
        require(len(row) == 1, "Missing or duplicate token fixture")
        f = path.parent / row[0]["token_ids_file"]
        raw = f.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == row[0]["token_ids_sha256"], "Token bytes changed")
        tokens = json.loads(raw)
        require(
            isinstance(tokens, list)
            and len(tokens) == value
            and all(type(x) is int and 0 <= x < 128256 for x in tokens),
            "Invalid token inventory",
        )
        end = value - slot * 32
        begin = end - 1024
        name = "slot" + str(slot)
        fixtures[name] = tokens[:end]
        uuid = 700000 + value * 2 + slot
        rid = slot + 101
        calls = [dict(slot=slot, request_id=rid, begin=a, end=min(a + 1024, end)) for a in range(0, end, 1024)]
        phases.append(
            dict(
                name=name,
                fixture=name,
                source_slot=slot,
                destination_slot=1 - slot,
                valid_prompt_tokens=end,
                reused_tokens=0,
                compute_calls=calls,
                source_command=dict(
                    op="register", request_id=rid, uuid=uuid, tokens=fixtures[name], **{"from": begin, "to": end}
                ),
                passive_command=dict(
                    op="expect", uuid=uuid, slot=1 - slot, expected_reused=0, **{"from": begin, "to": end}
                ),
                retire_source_after_verified_landing=dict(op="retire", slot=slot, uuid=uuid),
                retire_destination_after_verified_readback=dict(op="retire", slot=1 - slot, uuid=uuid),
            )
        )
    require(fixtures["slot0"][: value - 32] != fixtures["slot1"], "Distinct prompts required")
    return dict(capacity=value, phases=phases), fixtures


def selected_keys(phase):
    command = phase["source_command"]
    return [
        (c, phase["source_slot"], layer, pos)
        for c in range(16)
        for layer in range(32)
        for pos in range(command["from"], command["to"], 32)
    ]


def destination_keys(phase):
    return [(c, phase["destination_slot"], l, p) for c, _, l, p in selected_keys(phase)]


def sentinel_keys(value, phase, phases=None):
    """Per-phase adjacent/outside/other-slot samples; optionally exclude all final landings."""
    value = capacity(value)
    spec = phase["source_command"]
    dst = phase["destination_slot"]
    positions = {0, 32, (value // 2) // 32 * 32, spec["from"] - 32, spec["to"], spec["to"] - 32}
    positions = {x for x in positions if 0 <= x < value}
    forbidden = set(destination_keys(phase))
    if phases is not None:
        forbidden = set(k for other in phases for k in destination_keys(other))
    return sorted(
        {
            (c, s, l, p)
            for c in range(16)
            for s in (0, 1)
            for l in range(32)
            for p in positions
            if (c, s, l, p) not in forbidden
        }
    )


def resources(value):
    value = capacity(value)
    phases = [
        dict(
            source_slot=s, destination_slot=1 - s, source_command={"from": value - s * 32 - 1024, "to": value - s * 32}
        )
        for s in (0, 1)
    ]
    sentinel_pages = 2 * sum(len(sentinel_keys(value, p)) for p in phases) + 2 * len(
        {k for p in phases for k in sentinel_keys(value, p, phases)}
    )
    return dict(
        capacity=value,
        requests=2,
        full32_chunk_calls=2 * (value // 1024),
        compile_warmup_full32_calls=2,
        capacity_warmup_full32_calls=2 * (value // 1024),
        total_full32_calls=2 + 4 * (value // 1024),
        routed_layer_acks=64 * (value // 1024),
        native_layer_commands=96,
        native_audit_calls=104,
        table_entries_per_endpoint=32 * value,
        packed_cache_bytes_per_endpoint=32 * value * PAGE,
        packed_cache_bytes_per_chip=value * PAGE,
        source_and_passive_host_fp32_cache_seed_payload_bytes_each=16384 * value,
        source_two_persistent_bfp8_gather_buffers_bytes_per_chip=272 * value,
        sentinel_read_pages=sentinel_pages,
        sentinel_read_bytes=sentinel_pages * PAGE,
        selected_pages=2 * 16 * 32 * 32,
        selected_bytes=2 * 16 * 32 * 32 * PAGE,
        selected_packed_reads_including_pre_request_change_check_bytes=6 * 2 * 16 * 32 * 32 * PAGE,
        selected_disk_bytes=2 * 16 * 32 * 32 * PAGE,
        scope="real H2D/model producer and exact selected native transport; no golden/performance claim",
    )
