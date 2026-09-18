from edge_coverage import RUNTIME_CALLS

REQUESTS = tuple((c.slot, c.begin, c.end) for c in RUNTIME_CALLS)


class AckRecorder:
    def __init__(self):
        self.active = None
        self.completed_ns = None
        self.rows = []

    def begin(self, request_id, slot, start, end, started_ns):
        if self.active is not None or request_id != len(self.rows) // 32 or (slot, start, end) != REQUESTS[request_id]:
            raise ValueError("Unexpected request/slot sequence")
        self.active = (request_id, slot, start, end, started_ns)
        self.completed_ns = None

    def synchronized(self, when_ns):
        if self.active is not None:
            if when_ns < self.active[4]:
                raise ValueError("Completion predates request")
            self.completed_ns = when_ns

    def ack(self, layer, request_id, when_ns):
        if self.active is None or self.completed_ns is None:
            raise ValueError("Acknowledgment without completed model synchronization")
        active, slot, start, end, _ = self.active
        if (
            request_id != active
            or len(self.rows) >= (active + 1) * 32
            or layer != len(self.rows) - active * 32
            or when_ns < self.completed_ns
        ):
            raise ValueError("Wrong/out-of-order request layer acknowledgment")
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

    def finish(self, consumed):
        if self.active is None or len(self.rows) != (self.active[0] + 1) * 32 or consumed != 32:
            raise ValueError("Expected exactly32 local and routed acknowledgments")
        self.active = None
