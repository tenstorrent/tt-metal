"""Borrowed-input runtime call, import-free so failure ordering can be tested."""
from edge_checks import check_metadata


def execute_call(runtime, cache, recorder, bridge, call, ordinal, packet, check_input, check):
    check()
    check_metadata(packet["metadata_rows"], call)
    recorder.begin(ordinal, call.slot, call.begin, call.end, packet["began_ns"])
    check()
    runtime.prefill_chunk(
        packet["tokens"],
        cache,
        slot_id=call.slot,
        actual_start=call.begin,
        actual_end=call.end,
        request_id=ordinal,
        metadata_msg=packet["metadata"],
    )
    check()
    state = bridge.snapshot_until(lambda v: v["retired"] and v["acks"] == (ordinal + 1) * 32)
    recorder.finish(32)
    check_input(packet, call)
    return state
