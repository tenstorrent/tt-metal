"""Two high-end real-prefill transfers, with exact readback before retirement."""

from range_contract import generation, phase_receipt, rpc
from runner_support import require


def run_phases(
    role,
    doc,
    nonce,
    bridge,
    publish,
    wait_peer,
    run_call,
    verify_destination,
    check,
    before_destination,
    verify_source,
    before_source,
    phase_timeout,
):
    results = []
    ordinal = 0
    for index, phase in enumerate(doc["phases"]):
        check()
        uuid = phase["source_command"]["uuid"]
        prefix = f"phase-{uuid}-"

        def send(label, **fields):
            publish(prefix + label, uuid=uuid, **fields)

        def wait(label, peer):
            return phase_receipt(wait_peer(prefix + label), phase, nonce, peer)

        if role == "source":
            before_source(phase)
            reply = rpc(bridge, phase["source_command"])
            require(reply["slot"] == phase["source_slot"], "source admitted the wrong physical slot")
            if phase["reused_tokens"]:
                require(reply["reused"] == phase["reused_tokens"], "source reused-prefix length differs")
            send("registered")
            wait("armed", "passive")
            last = None
            for call in phase["compute_calls"]:
                rpc(bridge, dict(op="prepare", **call))
                last = run_call(phase, call, ordinal)
                ordinal += 1
            terminal = bridge.snapshot_until(
                lambda v: any(r.get("uuid") == uuid and r.get("successful") is True for r in v.get("generations", [])),
                timeout=phase_timeout,
            )
            generation(terminal, phase, "source")
            require(terminal["acks"] == ordinal * 32, "cumulative routed count differs")
            source_unchanged = verify_source(phase, last)
            send("source-done", terminal=terminal, snapshot=last, call_count=ordinal, source_unchanged=source_unchanged)
            verified = wait("bytes-verified", "passive")
            require(verified["source_sha256"] == last["selected"]["sha256"], "readback used wrong source snapshot")
            rpc(bridge, phase["retire_source_after_verified_landing"])
            retired = bridge.rpc("snapshot")
            require(generation(retired, phase, "source")["retired"] is True, "source did not retire")
            send("retired", terminal=retired)
            wait("retired", "passive")
            results.append(dict(uuid=uuid, terminal=terminal, readback=verified, retired=retired))
        else:
            wait("registered", "source")
            before_destination(phase)
            rpc(bridge, phase["passive_command"])
            send("armed")
            terminal = bridge.snapshot_until(
                lambda v: any(
                    r.get("uuid") == uuid and r.get("complete") is True for r in v.get("inbound_generations", [])
                ),
                timeout=phase_timeout,
            )
            generation(terminal, phase, "passive")
            source = wait("source-done", "source")
            generation(source["terminal"], phase, "source")
            verified = verify_destination(phase, source["snapshot"])
            send("bytes-verified", **verified)
            rpc(bridge, phase["retire_destination_after_verified_readback"])
            retired = bridge.rpc("snapshot")
            require(generation(retired, phase, "passive")["retired"] is True, "passive did not retire")
            send("retired", terminal=retired)
            wait("retired", "source")
            results.append(dict(uuid=uuid, terminal=terminal, readback=verified, retired=retired))
    require(
        len(results) == 2 and (role != "source" or ordinal == 2 * (doc["capacity"] // 1024)),
        "incomplete phase inventory",
    )
    return results, bridge.rpc("snapshot")
