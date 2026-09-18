"""A bounded named barrier holds expect until real completed source work is observed.

PrefillReader::step queues consumed layer acks and retires the local chunk without
an announce; try_drain refuses all native sends until arm_burst sets a destination.
"""

from runtime_cancel import require

DELAY_KEYS = {f"{c}:{l}:{x}" for c in range(16) for l in range(32) for x in range(0, 1024, 32)}


def check_unarmed(row):
    require(row.get("acks") == 32 and row.get("retired") is True, "first chunk has not retired locally")
    calls = row.get("calls", [])
    require(
        len(calls) == 1
        and calls[0].get("op") == "register"
        and calls[0].get("uuid") == 700
        and calls[0].get("src") == 0,
        "native work escaped the delayed expect barrier",
    )
    generations = row.get("generations", [])
    require(
        len(generations) == 1
        and all(
            generations[0].get(k) == v
            for k, v in dict(
                uuid=700, request_id=11, slot=0, prompt_len=2048, chunk_count=1, **{"from": 0, "to": 2048}
            ).items()
        ),
        "wrong delayed source generation",
    )
    slots = [x for x in row.get("slots", []) if x.get("slot") == 0]
    require(
        len(slots) == 1
        and all(slots[0].get(k) == v for k, v in dict(position=1024, pins=1, in_flight=0, pending=0).items()),
        "unarmed source lifetime changed",
    )
    return row


def check_delayed_passive(before, after, snapshot):
    require(set(before) == set(after) == DELAY_KEYS and before == after, "unarmed passive sentinel changed")
    require(
        snapshot.get("calls") == [] and snapshot.get("inbound_generations") == [], "passive armed before delayed check"
    )


def source_before_arm(bridge, produce, publish, wait_peer, check):
    wait_peer("delay-held")
    check()
    produced = produce(32)
    snapshot = bridge.snapshot_until(lambda row: row.get("acks") == 32 and row.get("retired") is True)
    check_unarmed(snapshot)
    publish("unarmed-complete", snapshot=snapshot)
    wait_peer("delay-verified")
    check()
    snapshot = bridge.rpc("snapshot")
    check_unarmed(snapshot)
    publish("arm-permitted", snapshot=snapshot)
    wait_peer("cancel-armed")
    return produced


def passive_arm(bridge, pages, publish, wait_peer, check):
    before = pages("delay-before")
    publish("delay-held", pages=before)
    source = wait_peer("unarmed-complete")
    check_unarmed(source["snapshot"])
    check()
    after = pages("delay-after")
    snapshot = bridge.rpc("snapshot")
    check_delayed_passive(before, after, snapshot)
    publish("delay-verified", pages=after, snapshot=snapshot)
    source = wait_peer("arm-permitted")
    check_unarmed(source["snapshot"])
    check()
    bridge.rpc("expect", uuid=700, slot=1, **{"from": 0, "to": 2048})
    publish("cancel-armed")
