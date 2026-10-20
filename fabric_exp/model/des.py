#!/usr/bin/env python3
"""Event-driven model of one fabric link (both directions) for a placement of pieces on cores.

Each core runs its pieces in a polling loop, paying the catalog's costs; packets move through sender rings, the
wire (serialized per direction, shared by data, pointer updates and acks), receiver slots (governed by credits),
NoC ports and handoff queues between cores. Where the static model sums per-packet costs, this one also captures
credit loops, latency, batching dynamics and burstiness.

usage:
  python3 fabric_exp/model/des.py --validate           # compare with hardware measurements
  python3 fabric_exp/model/des.py --calibrate          # fit the one-way Ethernet latency to the slot sweep
"""
import argparse
import copy
import heapq
import itertools

from fabric_model import (
    ERISC_ONLY,
    Options,
    _cost,
    baseline_placement,
    core_scale,
    credit_frame_cycles,
    data_wire_cycles,
    header_bytes,
    load_catalog,
    noc_cycles_per_byte,
    pieces_for,
    small_frame_cycles,
    summary_doorbell_producer,
    tx_scan,
    wire_cycles,
)

HANDOFF_LATENCY = 10  # estimated: L1 descriptor + stream-register doorbell between the two ERISCs
WORKER_COST = 40  # estimated: worker kernel cycles per packet in benchmark mode (credit-only producer)
WORKER_POLL = 10


class Engine:
    def __init__(self):
        self.now = 0.0
        self._q = []
        self._seq = itertools.count()

    def start(self, gen, at=0.0):
        heapq.heappush(self._q, (at, next(self._seq), gen))

    def run(self, until):
        while self._q:
            t, _, gen = heapq.heappop(self._q)
            if t > until:
                break
            self.now = t
            try:
                d = next(gen)
            except StopIteration:
                continue
            heapq.heappush(self._q, (t + d, next(self._seq), gen))


class Counter:
    """Integer whose increments become visible at given times."""

    def __init__(self, value=0):
        self.value_now = value
        self._pending = []

    def add(self, t, n):
        heapq.heappush(self._pending, (t, n))

    def value(self, now):
        while self._pending and self._pending[0][0] <= now:
            self.value_now += heapq.heappop(self._pending)[1]
        return self.value_now

    def take(self, now, n):
        self.value(now)
        self.value_now -= n


class Queue:
    """Items that become visible at given times; a blocked item can be put back at the head."""

    def __init__(self):
        self._h = []
        self._front = []
        self._seq = itertools.count()

    def put(self, t, item):
        heapq.heappush(self._h, (t, next(self._seq), item))

    def put_front(self, item):
        self._front.append(item)

    def ready(self, now):
        return len(self._front) + sum(1 for t, _, _ in self._h if t <= now)

    def pop(self):
        return self._front.pop() if self._front else heapq.heappop(self._h)[2]

    def empty(self):
        return not self._h and not self._front


class Pipe:
    """Serialized resource (a wire direction or a NoC port)."""

    def __init__(self):
        self.free_at = 0.0

    def xfer(self, now, cycles):
        start = max(now, self.free_at)
        self.free_at = start + cycles
        return self.free_at


class Packet:
    __slots__ = ("flow", "sent", "decoded", "at", "blocked", "consumer", "local_done", "burst")

    def __init__(self, flow, burst=None):
        self.flow = flow
        self.burst = burst  # index of the burst the packet belongs to (bursty traffic only)
        self.sent = 0.0  # time the Ethernet slot's last NoC read completes (slot can retire)
        self.decoded = False  # decoded once already (a retry may reuse it)
        self.at = None  # DRISC holding a pushed copy of the packet
        self.blocked = False  # admission failed: retry from the head of the same queue
        self.consumer = 0  # cycles the receiving core spends picking up the handoff
        self.local_done = False  # local delivery already issued (a retry must not repeat it)


class Endpoint:
    def __init__(self, name, cores, sender_slots, receiver_slots):
        self.name = name
        self.peer = None
        self.sender_ring = Queue()
        self.worker_credits = Counter(sender_slots)
        self.remote_credits = Counter(receiver_slots)
        self.completions = Counter(0)
        self.rx_ring = Queue()
        self.retire = Queue()
        self.wire_out = Pipe()
        eth_port = Pipe()  # the ERISCs share the Ethernet tile's NoC port(s)
        self.noc = {c: (eth_port if c.startswith("E") else Pipe()) for c in cores}
        self.noc_local = {c: Pipe() for c in cores if not c.startswith("E")}  # a split DRISC's local-delivery port
        self.inbox = {}  # (core, stage) -> Queue of handed-off packets
        self.staging = {}  # DRISC -> Counter of free staging slots (as seen by the pushing ERISC)
        self.delivered = 0
        self.retries = 0
        self.busy = {c: 0.0 for c in cores}
        self.tokens = {}  # core -> per-output Counter of downstream slots, as that core sees them
        self.ds_pipe = []  # per output: the downstream's drain (its outgoing link share)
        self.visits = 0  # sender visits that sent something, and the packets they sent
        self.sent_in_visits = 0
        self.burst_start = {}  # burst index -> start time (bursty traffic)
        self.burst_done = {}  # burst index -> last delivery time at the peer
        self.burst_count = {}  # burst index -> packets delivered


def simulate(
    cat,
    scen_name,
    placement,
    cores,
    opts=None,
    *,
    sender_slots=32,
    receiver_slots=32,
    latency=None,
    cycles=400000,
    warmup=60000,
    staging_slots=8,
    downstream=None,
    stats=None,
    burst=None,
):
    """Returns (cycles per packet per direction, per-core fraction of time spent in piece work, excluding polls).

    downstream: the routers that forwarded packets go to, per endpoint: {"slots": downstream sender-channel slots,
      "factor": drain interval as a multiple of the wire time (>1: the downstream link is shared), "stall": (period,
      length) periodic downstream stalls, "hold": cycles a downstream slot stays busy after draining (its own
      Ethernet round trip), "congested": endpoints the factor and stalls apply to}.
    stats: optional dict, filled with per-endpoint delivery intervals, admission retries per packet and per-core work.
    burst: (packets, period, worker cycles per packet): instead of streaming, each worker starts a burst of `packets`
      every `period` cycles, producing one packet per `worker cycles` while it has credits. stats then also gets the
      mean burst time (burst start to the last packet's delivery) and the mean packets sent per sender visit.
    """
    opts = opts or Options()
    scen = cat["scenarios"][scen_name]
    path = scen["path"]
    P, polls, conn = cat["pieces"], cat["polls"], cat["connectors"]
    assert not opts.tx_scan_on, "the event-driven model does not model moving the sender-channel scan (static only)"
    assert (
        scen.get("tx_rate", 1.0) == scen.get("rx_rate", 1.0) == 1.0
    ), "the event-driven model assumes symmetric traffic"
    assert not (opts.tx_delay or opts.rx_delay), "delay emulation is static only"
    kind = {c.name: c.kind for c in cores}
    names = pieces_for(scen)
    lat = latency if latency is not None else cat["link"]["one_way_latency_cycles"]
    noc_lat = cat["noc"]["write_latency_cycles"]
    # Cycles per byte on each core's NoC port; both ERISCs share their Ethernet tile's port (see Endpoint), and a
    # DRISC shared by several routers carries all of their traffic.
    port = {c.name: noc_cycles_per_byte(cat, c.kind, opts) for c in cores}
    scale = {c.name: core_scale(cat, c.kind, opts) for c in cores}
    k = opts.k_drain
    payload, header = opts.payload, header_bytes(cat, scen)
    wire_data = data_wire_cycles(cat, payload, header)  # payload frames, without the pointer update
    ptr_wire = small_frame_cycles(cat)  # pointer update: a minimum-size register write
    credit_wire = credit_frame_cycles(cat, scen)  # credit: an L1 write of the counter arrays

    send = P["tx_send"]
    send_work = _cost(send, path) - send["batch"]
    ptr_cost = send["batch"]
    comp_cost = _cost(P["tx_completion"], path)
    # The router batches credits only on the speedy path; mirror its cadence (erisc_datamover_builder.cpp).
    comp_min = min(2, max(1, sender_slots // 2)) if path == "speedy" else 1
    b_ack = opts.b_ack or (4 if path == "speedy" else 1)
    b_ack = max(1, min(b_ack, receiver_slots // 2))
    ack_cost = _cost(P["rx_ack"], path)
    scan, _ = tx_scan(cat, scen, opts)
    rx = scen["rx"]
    fwd = rx["forwards"]
    adm = P["rx_admit"]
    check, reserve = adm["check_per_output"], adm["reserve_per_output"]
    has_admit = "rx_admit" in names
    decode_only = rx["decode"] - (check * fwd if has_admit else 0)
    stage_cost = {"rx_decode": decode_only}
    if has_admit:
        stage_cost["rx_admit"] = (check + reserve) * fwd
    if "rx_local" in names:
        stage_cost["rx_local"] = _cost(P["rx_local"], rx["local"])
    if "rx_forward" in names:
        per_copy = _cost(P["rx_forward"], "any") - (reserve if has_admit else 0)
        if kind[placement["rx_forward"]] == "drisc" and cat["core_types"]["drisc"].get("forward_per_copy") is not None:
            per_copy = cat["core_types"]["drisc"]["forward_per_copy"]
        per_copy += summary_doorbell_producer(cat, opts)
        stage_cost["rx_forward"] = per_copy * fwd
    # Local delivery holds no downstream credit, so it needs only the decode; admission gates the forwards.
    chain = [s for s in ("rx_decode", "rx_local", "rx_admit", "rx_forward") if s in names]
    payload_stages = ("rx_local", "rx_forward")
    front, ack_core, tx_core, comp_core = (placement[p] for p in ("rx_poll", "rx_ack", "tx_send", "tx_completion"))
    adm_on_drisc = has_admit and kind[placement["rx_admit"]] == "drisc"
    relay = adm_on_drisc and not opts.drisc_credits_direct
    credit_home = "drisc" if adm_on_drisc and opts.drisc_credits_direct else "eth"  # where downstream credits land
    for p in names:
        assert p not in ERISC_ONLY or kind[placement[p]] == "erisc", p
    ds = {"slots": 16, "factor": 1.0, "stall": None, "hold": 2 * lat + 200, "congested": ("A", "B")}
    ds.update(downstream or {})

    sim = Engine()
    eps = [
        Endpoint("A", [c.name for c in cores], sender_slots, receiver_slots),
        Endpoint("B", [c.name for c in cores], sender_slots, receiver_slots),
    ]
    eps[0].peer, eps[1].peer = eps[1], eps[0]
    for ep in eps:
        for c in cores:
            if c.kind == "drisc":
                ep.staging[c.name] = Counter(staging_slots)
        # Downstream credits land in Ethernet-tile stream registers (relayed to an admitting DRISC), or straight in
        # the DRISC's when downstream routers return them there.
        ep.tokens["eth"] = [Counter(0 if adm_on_drisc else ds["slots"]) for _ in range(fwd)]
        ep.tokens["drisc"] = [Counter(ds["slots"] if adm_on_drisc else 0) for _ in range(fwd)]
        ep.ds_pipe = [Pipe() for _ in range(fwd)]
        congested = ep.name in ds["congested"]
        ep.ds_interval = wire_cycles(cat, payload, header, opts.b_ptr) * (ds["factor"] if congested else 1.0)
        ep.ds_stall = ds["stall"] if congested else None

    def work(ep, core, d):
        if sim.now >= warmup:
            ep.busy[core] += d * scale[core]
        return d

    def bursty_worker(ep):
        size, period, per_pkt = burst
        b = 0
        while True:
            start = b * period
            if sim.now < start:
                yield start - sim.now
            ep.burst_start[b] = sim.now
            for _ in range(size):
                while ep.worker_credits.value(sim.now) <= 0:
                    yield WORKER_POLL
                ep.worker_credits.take(sim.now, 1)
                yield per_pkt
                ep.sender_ring.put(sim.now + noc_lat, Packet(ep.name, b))
            b += 1

    def worker(ep):
        while True:
            if ep.worker_credits.value(sim.now) > 0:
                ep.worker_credits.take(sim.now, 1)
                yield WORKER_COST
                ep.sender_ring.put(sim.now + noc_lat, Packet(ep.name))
            else:
                yield WORKER_POLL

    # --- TX pieces
    def tx_send(ep, core):
        yield scan
        n = min(k, ep.sender_ring.ready(sim.now), ep.remote_credits.value(sim.now))
        if n and sim.now >= warmup:
            ep.visits += 1
            ep.sent_in_visits += n
        pending = []
        for i in range(n):
            pkt = ep.sender_ring.pop()
            ep.remote_credits.take(sim.now, 1)
            yield work(ep, core, send_work)
            ep.wire_out.xfer(sim.now, wire_data)
            pending.append(pkt)
            if len(pending) == opts.b_ptr or i == n - 1:
                yield work(ep, core, ptr_cost)
                arrival = ep.wire_out.xfer(sim.now, ptr_wire) + lat
                for p in pending:
                    ep.peer.rx_ring.put(arrival, p)
                pending = []

    def tx_completion(ep, core):
        separate = core != tx_core
        if separate:
            yield polls["handoff"]
        n = ep.completions.value(sim.now)
        if n < comp_min:
            return
        ep.completions.take(sim.now, n)
        yield work(ep, core, comp_cost)
        if separate:
            c = conn["erisc_to_erisc"]
            yield work(ep, core, c["producer"] + c["producer_per_batch"])
            ep.remote_credits.add(sim.now + HANDOFF_LATENCY, n)
        else:
            ep.remote_credits.add(sim.now, n)
        ep.worker_credits.add(sim.now + noc_lat, n)

    # --- RX pieces: a chain of stages; consecutive stages on the same core run inline
    def to_ack(ep, core, pkt):
        """Hand the Ethernet slot's retirement to the ack core."""
        if core == ack_core:
            ep.retire.put(pkt.sent, pkt)
        else:
            c = conn["erisc_to_erisc"]
            yield work(ep, core, c["producer"] + c["producer_per_batch"] / k)
            ep.retire.put(max(pkt.sent, sim.now) + HANDOFF_LATENCY, pkt)

    def downstream_free(ep, o, arrive):
        """When downstream output o has drained a packet that arrived at `arrive`."""
        start = max(arrive, ep.ds_pipe[o].free_at)
        if ep.ds_stall:
            period, length = ep.ds_stall
            if start % period < length:
                start += length - start % period
        ep.ds_pipe[o].free_at = start + ep.ds_interval
        return ep.ds_pipe[o].free_at

    def staging_target(idx):
        """DRISC that will need the packet in its L1 after chain position idx, if any."""
        for s in chain[idx:]:
            if s in payload_stages and kind[placement[s]] == "drisc":
                return placement[s]
        return None

    def run_chain(ep, core, pkt, idx):
        while idx < len(chain):
            stage = chain[idx]
            dst = placement[stage]
            if dst != core:
                yield from handoff(ep, core, dst, pkt, idx)
                return
            if stage == "rx_local" and pkt.local_done:
                idx += 1
                continue
            if stage == "rx_decode":
                if not (pkt.decoded and opts.cache_decode):
                    yield work(ep, core, decode_only)
                pkt.decoded = True
            elif stage == "rx_admit":
                # All-or-nothing: a slot on every output, plus the DRISC staging slot when the packet moves to a
                # DRISC after admission. Otherwise the packet stays at the head and is examined again next visit.
                view = ep.tokens["drisc" if kind[core] == "drisc" else "eth"]
                stage_to = staging_target(idx + 1)
                need_staging = stage_to is not None and pkt.at != stage_to
                if any(c.value(sim.now) < 1 for c in view) or (
                    need_staging and ep.staging[stage_to].value(sim.now) < 1
                ):
                    yield work(ep, core, check)
                    pkt.blocked = True
                    ep.retries += sim.now >= warmup
                    return
                for c in view:
                    c.take(sim.now, 1)
                if need_staging:
                    ep.staging[stage_to].take(sim.now, 1)
                    pkt.at = "reserved:" + stage_to
                yield work(ep, core, stage_cost[stage])
            else:
                yield work(ep, core, stage_cost[stage])
                copies = 1 if stage == "rx_local" else fwd
                pkt.local_done |= stage == "rx_local"
                # A split DRISC sends forwards and local deliveries through separate NoC ports.
                pipe = (
                    ep.noc_local[core]
                    if opts.drisc_split and kind[core] == "drisc" and stage == "rx_local"
                    else ep.noc[core]
                )
                done = pipe.xfer(sim.now, copies * payload * port[core])
                if stage == "rx_forward":  # each downstream frees its slot after draining and its own round trip
                    for o in range(fwd):
                        ep.tokens[credit_home][o].add(downstream_free(ep, o, done + noc_lat) + ds["hold"] + noc_lat, 1)
                if kind[core] == "erisc":
                    pkt.sent = max(pkt.sent, done)
                elif idx == len(chain) - 1 or placement[chain[idx + 1]] != core:
                    # DRISC staging slot frees once its writes are sent; credit back to the pushing ERISC
                    c = conn["drisc_to_erisc"]
                    yield work(ep, core, c["producer_per_batch"] / max(k, 4))
                    ep.staging[core].add(done + noc_lat, 1)
            idx += 1
        ep.delivered += sim.now >= warmup
        if pkt.burst is not None:
            src = ep.peer
            src.burst_done[pkt.burst] = max(src.burst_done.get(pkt.burst, 0.0), sim.now)
            src.burst_count[pkt.burst] = src.burst_count.get(pkt.burst, 0) + 1
        if kind[core] == "erisc":
            yield from to_ack(ep, core, pkt)

    def handoff(ep, src, dst, pkt, idx):
        if kind[src] == "erisc" and kind[dst] == "erisc":
            c = conn["erisc_to_erisc"]
            yield work(ep, src, c["producer"] + c["producer_per_batch"] / k)
            pkt.consumer = c["consumer"]
            ep.inbox[(dst, idx)].put(sim.now + HANDOFF_LATENCY, pkt)
        elif kind[dst] == "drisc":
            needs_payload = staging_target(idx) == dst
            if needs_payload and pkt.at != dst:
                # Push the packet into the DRISC's L1 (the decision rides along). Its staging slot was reserved at
                # admission, or is taken now; with none free the packet waits at the head.
                if pkt.at != "reserved:" + dst:
                    if ep.staging[dst].value(sim.now) < 1:
                        pkt.blocked = True
                        return
                    ep.staging[dst].take(sim.now, 1)
                c = conn["erisc_to_drisc_payload"]
                yield work(ep, src, c["producer"] + c["producer_per_batch"] / k)
                done = ep.noc[src].xfer(sim.now, (payload + header) * port[src])
                pkt.at = dst
                pkt.consumer = c["consumer"]
                ep.inbox[(dst, idx)].put(done + noc_lat, pkt)
                pkt.sent = max(pkt.sent, done)  # the Ethernet slot retires once the push has been read out
                yield from to_ack(ep, src, pkt)
            elif pkt.at == dst:  # already in the DRISC's L1: grant it
                c = conn["erisc_to_drisc_grant"]
                yield work(ep, src, c["producer"] + c["producer_per_batch"] / k)
                pkt.consumer = c["consumer"]
                ep.inbox[(dst, idx)].put(sim.now + noc_lat, pkt)
            else:  # header only; the payload stays on the Ethernet core
                c = conn["erisc_to_drisc_header"]
                yield work(ep, src, c["producer"] + c["producer_per_batch"] / k)
                done = ep.noc[src].xfer(sim.now, header * port[src])
                pkt.consumer = c["consumer"]
                ep.inbox[(dst, idx)].put(done + noc_lat, pkt)
        else:  # DRISC -> ERISC: an admission request (decoded outputs), or the decision back
            request = has_admit and placement["rx_admit"] == dst and "rx_admit" in chain[idx:]
            c = conn["drisc_to_erisc_request" if request else "drisc_to_erisc"]
            yield work(ep, src, c["producer"] + c["producer_per_batch"] / k)
            pkt.consumer = c["consumer"]
            ep.inbox[(dst, idx)].put(sim.now + noc_lat, pkt)

    def drain(ep, core, q, idx):
        """Process up to k packets from q in order; a blocked packet goes back to the head and ends the visit."""
        n = min(k, q.ready(sim.now))
        for _ in range(n):
            pkt = q.pop()
            if pkt.consumer:
                yield work(ep, core, pkt.consumer)
                pkt.consumer = 0
            yield from run_chain(ep, core, pkt, idx)
            if pkt.blocked:
                pkt.blocked = False
                q.put_front(pkt)
                break

    def inbox_task(ep, core, idx):
        yield polls["handoff"]
        yield from drain(ep, core, ep.inbox[(core, idx)], idx)

    def rx_poll(ep, core):
        yield polls["rx_ring"]
        yield from drain(ep, core, ep.rx_ring, 0)

    def credit_relay(ep, core):
        """Forward downstream credits from the Ethernet-tile stream registers to the admitting DRISC."""
        yield polls["handoff"]
        c = conn["credit_relay"]
        for o in range(fwd):
            n = ep.tokens["eth"][o].value(sim.now)
            if n >= opts.relay_batch:
                ep.tokens["eth"][o].take(sim.now, n)
                yield work(ep, core, c["producer"] + c["producer_per_batch"])
                ep.tokens["drisc"][o].add(sim.now + noc_lat, n)

    def rx_ack(ep, core):
        yield polls["retire"]
        n = ep.retire.ready(sim.now)
        idle = ep.rx_ring.empty() and ep.retire.ready(sim.now) == len(ep.retire._h)
        if n >= b_ack or (n > 0 and idle):
            for _ in range(n):
                ep.retire.pop()
            yield work(ep, core, ack_cost)
            arrival = ep.wire_out.xfer(sim.now, credit_wire) + lat
            ep.peer.completions.add(arrival, n)

    # --- assemble core loops
    for ep in eps:
        # handoff inboxes: one per (core, chain index) whose predecessor sits elsewhere
        prev = front
        for idx, stage in enumerate(chain):
            dst = placement[stage]
            if dst != prev:
                ep.inbox[(dst, idx)] = Queue()
            prev = dst
        loops = {c.name: [] for c in cores}
        loops[tx_core].append(tx_send)
        loops[comp_core].append(tx_completion)
        loops[front].append(rx_poll)
        for core, idx in ep.inbox:
            loops[core].append(lambda ep, core, idx=idx: inbox_task(ep, core, idx))
        loops[ack_core].append(rx_ack)
        if relay:
            loops[front].append(credit_relay)

        def core_loop(ep, core, tasks, loop_cost):
            # A slower or shared core (a DRISC with alpha or share above 1) stretches every step it takes.
            while True:
                for t in tasks:
                    for d in t(ep, core):
                        yield d * scale[core]
                yield loop_cost * scale[core]

        for c in cores:
            if loops[c.name]:
                sim.start(core_loop(ep, c.name, loops[c.name], cat["core_types"][c.kind]["loop"]))
        sim.start(bursty_worker(ep) if burst else worker(ep))

    sim.run(cycles)
    measured = cycles - warmup
    per_dir = [measured / max(ep.delivered, 1) for ep in eps]
    interval = sum(per_dir) / len(per_dir)
    util = {c.name: sum(ep.busy[c.name] for ep in eps) / (len(eps) * measured) for c in cores}
    if stats is not None:
        for ep, iv in zip(eps, per_dir):
            stats[ep.name] = {
                "interval": iv,
                "retries_per_pkt": ep.retries / max(ep.delivered, 1),
                "busy": {c.name: ep.busy[c.name] / measured for c in cores},
                "pkts_per_send_visit": ep.sent_in_visits / max(ep.visits, 1),
            }
            if burst:
                done = [
                    ep.burst_done[b] - ep.burst_start[b]
                    for b in ep.burst_start
                    if ep.burst_start[b] >= warmup and ep.burst_count.get(b, 0) == burst[0]
                ]
                stats[ep.name]["burst_time"] = sum(done) / max(len(done), 1)
                stats[ep.name]["bursts"] = len(done)
    return interval, util


# --- validation against hardware --------------------------------------------------------------------------------
SLOT_CASES = [  # speedy neighbor exchange, 2 KiB (RESULTS §5): (sender slots, receiver slots, measured)
    (32, 32, 123.7),
    (8, 8, 243.9),
    (4, 4, 430.7),
    (2, 4, 820.9),
]
SIZE_CASES = [(1024, 123.6), (2048, 123.7), (4096, 124.5), (5440, 154.2), (7616, 213.1), (8704, 243.9)]  # §10
GENERIC_CASES = [  # (label, scenario, rx overrides, options, measured) RESULTS §8
    ("2D 1-hop E/W write", "2d_single_ew", {}, Options(), 223.0),
    ("2D 1-hop E/W write, 2 ch", "2d_single_ew", {}, Options(sender_channels=2), 168.1),
    ("2D 1-hop E/W scatter, 2 ch", "2d_single_ew", {"local": "scatter"}, Options(sender_channels=2), 191.0),
    ("2D 1-hop N/S write, 1 ch", "2d_single_ns", {}, Options(sender_channels=1), 152.4),
    ("2D 1-hop N/S scatter, 1 ch", "2d_single_ns", {"local": "scatter"}, Options(sender_channels=1), 206.4),
    ("2D 3-hop middle, 2 ch", "2d_3hop_middle", {}, Options(sender_channels=2), 188.7),
    # 4 KiB, keepalive 1000 (RESULTS §16; the one-way keepalive floor is not modelled here)
    ("2D 1-hop E/W write, 2 ch, 4 KiB", "2d_single_ew", {}, Options(sender_channels=2, payload=4096), 169.8),
    ("2D 1-hop N/S write, 1 ch, 4 KiB", "2d_single_ns", {}, Options(sender_channels=1, payload=4096), 152.5),
    ("2D 3-hop middle, 2 ch, 4 KiB", "2d_3hop_middle", {}, Options(sender_channels=2, payload=4096), 186.5),
]


def slot_error(cat, latency, cores):
    errs = []
    for s, r, measured in SLOT_CASES:
        iv, _ = simulate(
            cat,
            "nbr_speedy",
            baseline_placement(cat["scenarios"]["nbr_speedy"]),
            cores,
            Options(payload=2048),
            sender_slots=s,
            receiver_slots=r,
            latency=latency,
        )
        errs.append((iv / measured - 1) ** 2)
    return sum(errs) / len(errs)


def main():
    from fabric_model import make_cores

    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    args = ap.parse_args()
    cat = load_catalog()
    cores = make_cores()
    if args.calibrate:
        best = min(range(100, 1501, 50), key=lambda L: slot_error(cat, L, cores))
        print(f"best one-way latency: {best} cycles (rms error {slot_error(cat, best, cores) ** 0.5 * 100:.1f}%)")
        return
    if args.validate:
        base = baseline_placement(cat["scenarios"]["nbr_speedy"])
        print(f"{'configuration':38s} {'measured':>8s} {'model':>7s} {'error':>6s}   piece-work fraction per core")
        rows = []
        for s, r, m in SLOT_CASES:
            iv, u = simulate(cat, "nbr_speedy", base, cores, Options(payload=2048), sender_slots=s, receiver_slots=r)
            rows.append((f"speedy 2 KiB, slots {s}/{r}", m, iv, u))
        for size, m in SIZE_CASES:
            iv, u = simulate(cat, "nbr_speedy", base, cores, Options(payload=size))
            rows.append((f"speedy {size} B, slots 32/32", m, iv, u))
        for label, scen, over, opts, m in GENERIC_CASES:
            c2 = copy.deepcopy(cat)
            c2["scenarios"][scen]["rx"].update(over)
            iv, u = simulate(
                c2, scen, baseline_placement(c2["scenarios"][scen]), cores, opts, sender_slots=16, receiver_slots=16
            )
            rows.append((label, m, iv, u))
        errs = []
        for label, m, iv, u in rows:
            errs.append(abs(iv / m - 1))
            print(
                f"{label:38s} {m:8.1f} {iv:7.1f} {100 * (iv / m - 1):+5.1f}%   "
                + "  ".join(f"{c}={v * 100:.0f}%" for c, v in u.items())
            )
        print(f"\nmean |error| {100 * sum(errs) / len(errs):.1f}%, max {100 * max(errs):.1f}%")


if __name__ == "__main__":
    main()
