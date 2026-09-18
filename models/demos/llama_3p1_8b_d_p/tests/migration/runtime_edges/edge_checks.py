"""Pure cache-effect checks for the five-call runtime gate; no device imports."""
import math

from edge_coverage import PAGE_BYTES, RUNTIME_CALLS, pages


def seed_value(config, slot, layer):
    value = (1 + slot * 32 + layer) / 16
    return value if config < 8 else -value / 2


def check_values(rows, *, valid_rows=32, seed=None, forbidden_seed=None):
    if type(valid_rows) is not int or not 0 <= valid_rows <= 32:
        raise ValueError("invalid semantic row count")
    if len(rows) != 32 or any(len(row) != 128 for row in rows):
        raise ValueError("decoded page must be32x128")
    for row, values in enumerate(rows):
        if row < valid_rows:
            if not any(value != 0 for value in values):
                raise ValueError("all-zero valid cache row")
            if forbidden_seed is not None and all(value == forbidden_seed for value in values):
                raise ValueError("valid cache row retained seed sentinel")
        for value in values:
            if not math.isfinite(value):
                raise ValueError("nonfinite cache value")
            if seed is not None and value != seed:
                raise ValueError("nonzero seed differs")
            if row >= valid_rows and value != 0:
                raise ValueError("cache padding is nonzero")
    return dict(valid_values=valid_rows * 128, padding_values=(32 - valid_rows) * 128)


class EffectCheck:
    """Check complete streamed inventory, valid/pad values and all untouched bytes."""

    def __init__(self, call, decode):
        self.call, self.decode = call, decode
        self.positions = set(pages(call.begin, call.end))
        self.count = self.changed = self.untouched = self.valid_values = self.padding_values = 0
        self.groups = {}

    def accept(self, key, before, after):
        c, s, l, p = key
        if not (0 <= c < 16 and s in (0, 1) and 0 <= l < 32 and 0 <= p < 2048 and p % 32 == 0):
            raise ValueError("invalid page key")
        ordinal = ((s * 16 + c) * 32 + l) * 64 + p // 32
        # Snapshot order is slot, config, layer, position. No missing or duplicate page is permitted.
        if ordinal != self.count:
            raise ValueError("missing, duplicate or out-of-order page")
        if any(not isinstance(raw, bytes) or len(raw) != PAGE_BYTES for raw in (before, after)):
            raise ValueError("invalid packed page")
        selected = s == self.call.slot and p in self.positions
        if selected:
            if before == after:
                raise ValueError("selected page retained the old contents")
            result = check_values(
                self.decode(after), valid_rows=min(32, self.call.end - p), forbidden_seed=seed_value(c, s, l)
            )
            self.valid_values += result["valid_values"]
            self.padding_values += result["padding_values"]
            self.changed += 1
            self.groups[c, l] = self.groups.get((c, l), 0) + 1
        else:
            if before != after:
                raise ValueError("untouched cache page changed")
            self.untouched += 1
        self.count += 1

    def finish(self):
        expected = len(self.positions) * 16 * 32
        if self.count != 65536 or self.changed != expected or len(self.groups) != 512:
            raise ValueError("incomplete cache-effect coverage")
        if any(n != len(self.positions) for n in self.groups.values()):
            raise ValueError("missing layer/config page coverage")
        return dict(
            pages=self.count,
            changed_pages=self.changed,
            untouched_pages=self.untouched,
            configs=16,
            layers=32,
            slots=2,
            valid_values=self.valid_values,
            padding_values=self.padding_values,
            semantic_valid_end=self.call.end,
            packed_end=max(self.positions) + 32,
            golden_comparison=False,
            structural_write_checked=True,
            decoded_checks="finite nonzero valid rows replace seed sentinel; exact zero padding; no numerical golden",
        )


def check_metadata(actual, call):
    expected = [call.slot, call.begin, call.end]
    if len(actual) != 32 or any(row != expected for row in actual):
        raise ValueError("H2D metadata differs on one or more of32 chips")


def drive_requests(runtime, cache, recorder, fixtures, receive, check_input, consume, record):
    """Borrow inputs once. A completed prior runtime call is the only reuse claim here."""
    for request, call in enumerate(RUNTIME_CALLS):
        packet = None
        try:
            packet = receive(call, fixtures[call.prompt][call.begin : call.end])
            check_metadata(packet["metadata_rows"], call)
            recorder.begin(request, call.slot, call.begin, call.end, packet["began_ns"])
            runtime.prefill_chunk(
                packet["tokens"],
                cache,
                slot_id=call.slot,
                actual_start=call.begin,
                actual_end=call.end,
                request_id=request,
                metadata_msg=packet["metadata"],
            )
            routed = consume()
            recorder.finish(routed)
            check_input(packet, call)
            record(request, call, routed)
        finally:
            # Borrowed tensors are service-owned. Do not deallocate them here.
            packet = None
