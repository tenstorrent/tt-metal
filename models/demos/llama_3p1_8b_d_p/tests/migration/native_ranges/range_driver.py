"""Six sequential native bursts, with verified readback before request retirement."""
from range_contract import generation, phase_receipt, rpc
from runner_support import require


def run_phases(role, doc, nonce, bridge, publish, wait_peer, run_call, verify_destination, check):
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
            if index == 4:
                # A has already retired and B has completed before the public Evictor reclaims A's old slot.
                rpc(bridge, doc["between_B_and_C"])
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
                lambda v: any(r.get("uuid") == uuid and r.get("successful") is True for r in v.get("generations", []))
            )
            generation(terminal, phase, "source")
            require(terminal["acks"] == ordinal * 32, "cumulative routed count differs")
            send("source-done", terminal=terminal, snapshot=last, call_count=ordinal)
            verified = wait("bytes-verified", "passive")
            require(verified["source_snapshot_sha256"] == last["receipt_sha256"], "readback used wrong source snapshot")
            rpc(bridge, phase["retire_source_after_verified_landing"])
            retired = bridge.rpc("snapshot")
            require(generation(retired, phase, "source")["retired"] is True, "source did not retire")
            send("retired", terminal=retired)
            wait("retired", "passive")
            results.append(dict(uuid=uuid, terminal=terminal, readback=verified, retired=retired))
        else:
            wait("registered", "source")
            rpc(bridge, phase["passive_command"])
            send("armed")
            terminal = bridge.snapshot_until(
                lambda v: any(
                    r.get("uuid") == uuid and r.get("complete") is True for r in v.get("inbound_generations", [])
                )
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
    require(len(results) == 6 and (role != "source" or ordinal == 7), "incomplete phase inventory")
    return results, bridge.rpc("snapshot")
