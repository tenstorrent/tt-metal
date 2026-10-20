#!/usr/bin/env python3
"""Static ("lego") model of the fabric router.

A router's per-packet work is split into pieces (catalog.yaml). A placement puts every piece on exactly one core.
Each core pays, per packet: the work of its pieces, its per-iteration overhead (loop + one poll per watched queue)
divided by the packets it moves per iteration, and connector costs wherever adjacent pieces sit on different cores.
The bottleneck core, the wire and the NoC ports bound the per-packet interval.

By default traffic is bidirectional and symmetric (as in most benchmarks): per interval the router sends one packet
and receives one. A scenario can set `tx_rate` and `rx_rate` (packets sent and received per unit of the interval)
when a router sends and receives at different rates, as in 1D multicast; loads and the interval are then per unit.
`reverse_data: false` marks a one-way benchmark: the link carries data in one direction only, so credits do not share
the data direction's wire, and the TX queue keepalive bounds the data direction (link.keepalive_cycles).
"""
import itertools
import math
import os
from dataclasses import dataclass, field

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))

# Pieces that must run on an Ethernet core: they drive TX queues or poll Ethernet-core registers.
ERISC_ONLY = {"tx_send", "tx_completion", "rx_poll", "rx_ack"}
TX_PIECES = {"tx_send", "tx_completion"}
PAYLOAD_PIECES = ("rx_local", "rx_forward")


def load_catalog(path=None):
    with open(path or os.path.join(HERE, "catalog.yaml")) as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class Core:
    name: str
    kind: str  # "erisc" or "drisc"


def make_cores(num_drisc=0):
    return [Core("E0", "erisc"), Core("E1", "erisc")] + [Core(f"D{i}", "drisc") for i in range(num_drisc)]


@dataclass
class Options:
    payload: int = 2048
    sender_channels: int | None = None  # override the scenario's channel count (e.g. trimming)
    summary_doorbell: bool = False  # TX core reads one summary register instead of scanning every channel
    idle_poll: str = "scan"  # sender-channel polling: scan (today: three checks per channel per visit),
    # lean (packet check only, completions only with packets in flight, lazy
    # connection check), fused (lean, with every packet check in one block),
    # fused_overlap (fused, with the reads in flight together)
    b_ptr: int = 1  # packets per Ethernet pointer update
    b_ack: int | None = None  # packets per completion ack (None: the path's default)
    k_drain: int = 1  # packets handled per queue visit when the queue is deep
    noc_ports: int | None = None  # NoCs used for payload copies by the Ethernet tile and the DRISC (None: catalog)
    cache_decode: bool = False  # keep a blocked packet's decoded action instead of decoding it again on retry
    blocked_per_pkt: float | None = None  # admission failures per packet (None: the scenario's measured rate)
    relay_batch: int = 1  # credits forwarded per relay when a DRISC admits (credit relay)
    drisc_credits_direct: bool = False  # downstream routers return credits to the DRISC's stream registers (protocol
    # change) instead of the Ethernet tile's, so a DRISC admits without a relay
    drisc_split: bool = False  # a DRISC sends forwards on one NoC and local delivery on the other (needs a
    # DRISC whose NIUs are both in stream mode); each NoC port carries one kind
    drisc_share: int | None = None  # routers served by one DRISC (None: catalog)
    tx_scan_on: str | None = None  # core that scans the sender channels (None: the sending core); another ERISC
    # hands each packet it finds to the sender through their shared L1
    keepalive: int | None = None  # TX queue keepalive period in cycles (None: catalog; 8000 is the firmware's)
    tx_delay: float = 0.0  # experiment emulation (TT_FABRIC_EXP_TX_DELAY): cycles spun per sent packet
    rx_delay: float = 0.0  #   ... (TT_FABRIC_EXP_RX_DELAY): per received packet, on the arrival ERISC


@dataclass
class Result:
    interval: float
    bottleneck: str
    loads: dict
    breakdown: dict = field(default_factory=dict)
    noc_bytes: dict = field(default_factory=dict)
    wire: float = 0.0
    valid: bool = True
    why_invalid: str = ""
    floors: dict = field(default_factory=dict)  # wire, keepalive and NoC-port bounds on the interval

    def gbps(self, cat, payload):
        return cat["clock_hz"] * payload / self.interval / 1e9


def frames(cat, nbytes):
    """(bytes on the wire, frame count) for one TT-link L1 write of `nbytes`: the TX queue splits it into frames of at
    most max_frame_payload_bytes, each with framing overhead and at least the minimum frame size."""
    link = cat["link"]
    max_f, over, min_f = link["max_frame_payload_bytes"], link["frame_overhead_bytes"], link["min_frame_wire_bytes"]
    n = max(1, math.ceil(nbytes / max_f))
    last = nbytes - (n - 1) * max_f
    return (n - 1) * (max_f + over) + max(last + over, min_f), n


def header_bytes(cat, scen):
    """Fabric packet header size: the scenario's, else the path's (2D generic 96 B, speedy 48 B)."""
    return scen.get("header_bytes", cat["link"]["header_bytes"][scen["path"]])


def _header_or_2d(cat, header):
    """The given header size, else the 2D (generic path) header."""
    return cat["link"]["header_bytes"]["generic"] if header is None else header


def small_frame_cycles(cat):
    """Wire cycles of a pointer update (a remote register write): one minimum-size frame."""
    return cat["link"]["min_frame_wire_bytes"] / cat["link"]["line_rate_bytes_per_cycle"]


def credit_frame_cycles(cat, scen=None, path="generic"):
    """Wire cycles of one credit: an L1 write of the receiver's counter arrays (the scenario's size, else the path's;
    2D 96 B -> 152 B on the wire, 1D 32 B -> 88 B)."""
    sizes = cat["link"]["credit_bytes"]
    nbytes = scen.get("credit_bytes", sizes[scen["path"]]) if scen is not None else sizes[path]
    return frames(cat, nbytes)[0] / cat["link"]["line_rate_bytes_per_cycle"]


def data_wire_cycles(cat, payload, header=None):
    """Wire cycles of one packet's payload and header (default: a 2D packet's), without its pointer update."""
    nbytes, _ = frames(cat, payload + _header_or_2d(cat, header))
    return nbytes / cat["link"]["line_rate_bytes_per_cycle"]


def wire_cycles(cat, payload, header=None, b_ptr=1):
    """Wire cycles per packet in the data direction: payload frames plus one pointer update per b_ptr packets.
    The default header is a 2D packet's, so wire_cycles(cat, payload) is the 2D line-rate interval."""
    return data_wire_cycles(cat, payload, header) + small_frame_cycles(cat) / b_ptr


def keepalive_floor(cat, payload, header=None, b_ptr=1, keepalive=None):
    """Cycles per packet on a link with no data coming back: the sender's acknowledgements arrive only in the peer's
    keepalive, so at most seq_window_frames frames go out per keepalive period (RESULTS §15-16)."""
    link = cat["link"]
    period = link["keepalive_cycles"] if keepalive is None else keepalive
    _, n = frames(cat, payload + _header_or_2d(cat, header))
    return (n + 1.0 / b_ptr) * period / link["seq_window_frames"]


def target_interval(cat, payload, gbps_payload):
    """Cycles per packet needed for `gbps_payload` gigabytes/s of payload."""
    return cat["clock_hz"] * payload / (gbps_payload * 1e9)


def pieces_for(scen):
    names = ["tx_send", "tx_completion", "rx_poll", "rx_decode"]
    if scen["rx"]["forwards"] > 0:
        names.append("rx_admit")  # downstream credit check + slot reservation; local delivery needs no credit
    if scen["rx"]["local"] != "none":
        names.append("rx_local")
    if scen["rx"]["forwards"] > 0:
        names.append("rx_forward")
    names.append("rx_ack")
    return names


def baseline_placement(scen):
    """Today's router: every TX piece on ERISC0, every RX piece on ERISC1."""
    return {p: ("E0" if p in TX_PIECES else "E1") for p in pieces_for(scen)}


def egress_of(cat, kind):
    """NoC port bytes per cycle for a core type."""
    return cat["core_types"][kind].get("egress_bytes_per_cycle", cat["noc"]["egress_bytes_per_cycle"])


def noc_ports(cat, kind, opts):
    """NoC ports carrying payload copies: per Ethernet tile (shared by both ERISCs) or per DRISC."""
    if opts.noc_ports:
        return opts.noc_ports
    return cat["noc"]["eth_tile_noc_ports"] if kind == "erisc" else cat["core_types"][kind].get("noc_ports", 1)


def noc_cycles_per_byte(cat, kind, opts):
    """Port time per payload byte, assuming copies spread evenly over the ports; a shared DRISC carries every router.

    With opts.drisc_split a DRISC's forwards and local deliveries each have their own NoC port, so a copy's port time
    is that of one port.
    """
    ports = 1 if kind == "drisc" and opts.drisc_split else noc_ports(cat, kind, opts)
    return core_share(cat, kind, opts) / (egress_of(cat, kind) * ports)


def core_share(cat, kind, opts=None):
    if kind == "drisc" and opts is not None and opts.drisc_share:
        return opts.drisc_share
    return cat["core_types"][kind].get("share", 1)


def core_scale(cat, kind, opts=None):
    """Time on a core per unit of ERISC-calibrated work: slower core (alpha) times routers sharing it."""
    return cat["core_types"][kind].get("alpha", 1.0) * core_share(cat, kind, opts)


def _cost(piece, key):
    c = piece["cost"]
    return c.get(key, c.get("any"))


def tx_scan(cat, scen, opts):
    """Sender-channel polling per loop iteration on the sending core: (cycles, label)."""
    pc = cat["polls"]
    channels = opts.sender_channels or scen["sender_channels"]
    active = min(scen["active_channels"], channels)
    idle = channels - active
    if opts.idle_poll == "scan":
        per_active = per_idle = pc["sender_channel"]
        fused = 0.0
    else:
        parts = pc["sender_channel_parts"]
        conn = parts["connection"] / pc["connection_check_every"]
        fused = 0.0
        if opts.idle_poll == "lean":
            packet = parts["packet"]
        else:
            packet = pc["fused_per_channel"]["overlapped" if opts.idle_poll == "fused_overlap" else "blocking"]
            fused = pc["fused_fixed"]
        # A channel with nothing in flight cannot receive completions.
        per_active = packet + parts["completions"] + conn
        per_idle = packet + conn
    cycles = fused + per_active * active + per_idle * idle
    label = f"{opts.idle_poll} {channels} sender channels"
    if opts.summary_doorbell:
        doorbell = pc["summary_doorbell"] + fused + per_active * active
        if doorbell < cycles:
            cycles, label = doorbell, "summary doorbell + active channels"
    return cycles, label


def summary_doorbell_producer(cat, opts):
    """Extra cycles per packet a producer pays to ring the summary doorbell of the sender channel it writes into."""
    return cat["polls"]["summary_doorbell_producer"] if opts.summary_doorbell else 0.0


def evaluate(cat, scen_name, placement, cores, opts=None):
    opts = opts or Options()
    scen = cat["scenarios"][scen_name]
    path = scen["path"]
    P = cat["pieces"]
    conn = cat["connectors"]
    polls_cat = cat["polls"]
    kind = {c.name: c.kind for c in cores}
    names = pieces_for(scen)
    k = opts.k_drain
    # Packets sent and received per unit of the interval (1 and 1 for symmetric traffic).
    tx_rate, rx_rate = scen.get("tx_rate", 1.0), scen.get("rx_rate", 1.0)

    work = {c.name: {} for c in cores}  # core -> label -> cycles per packet
    polls = {c.name: {} for c in cores}  # core -> label -> cycles per iteration
    streams = {c.name: set() for c in cores}
    noc = {c.name: 0.0 for c in cores}  # bytes per packet leaving each core's NoC port
    noc_in = {c.name: 0.0 for c in cores}
    noc_piece = {c.name: {} for c in cores}  # bytes per packet each payload piece sends from the core

    def add_work(core, label, cycles):
        work[core][label] = work[core].get(label, 0.0) + cycles

    def add_poll(core, label, cycles):
        polls[core][label] = polls[core].get(label, 0.0) + cycles

    def connect(name, src, dst, rate=1.0, batch=1):
        """Connector `name` from core src to core dst carrying `rate` items per packet in batches of `batch`."""
        c = conn[name]
        add_work(src, f"{name}(out)", rate * (c["producer"] + c["producer_per_batch"] / batch))
        add_work(dst, f"{name}(in)", rate * c["consumer"])
        add_poll(dst, f"poll {name}", polls_cat[c["poll"]])
        streams[dst].add("rx")

    # --- validity: ERISC-only pieces
    for p in names:
        if p in ERISC_ONLY and kind[placement[p]] != "erisc":
            return Result(math.inf, "", {}, valid=False, why_invalid=f"{p} must run on an ERISC")
    front = placement["rx_poll"]  # ERISC that sees arrivals; the payload sits in its (shared) L1
    dec = placement["rx_decode"]
    payload_cores = {placement[p] for p in PAYLOAD_PIECES if p in names}

    # --- TX pieces
    tx = placement["tx_send"]
    send = P["tx_send"]
    add_work(tx, "tx_send", tx_rate * (_cost(send, path) - send["batch"] + send["batch"] / opts.b_ptr))
    if opts.tx_delay:
        add_work(tx, "tx delay (experiment)", tx_rate * opts.tx_delay)
    scan_cycles, scan_label = tx_scan(cat, scen, opts)
    scan_core = opts.tx_scan_on or tx
    if scan_core != tx and kind.get(scan_core) != "erisc":
        return Result(math.inf, "", {}, valid=False, why_invalid="the sender channels are scanned on an ERISC")
    if scen.get("tx_extra"):
        add_work(scan_core, "connection checks", tx_rate * scen["tx_extra"])
    add_poll(scan_core, scan_label, scan_cycles)
    streams[scan_core].add("tx")
    streams[tx].add("tx")
    if scan_core != tx:
        # The scanning ERISC hands each packet it finds to the sender: descriptor in the shared L1 + doorbell.
        connect("erisc_to_erisc", scan_core, tx, rate=tx_rate, batch=k)

    comp_core = placement["tx_completion"]
    comp = P["tx_completion"]
    # Measured completion checks per packet, at the path's default ack batching; batching acks further (b_ack)
    # delivers completions in proportionally fewer, larger groups. The speedy path handles at least two completions
    # per check (erisc_datamover_builder.cpp), so it never checks more than every other packet.
    default_b_ack = 4 if path == "speedy" else 1
    comp_rate = comp["per_packet"][path] * default_b_ack / (opts.b_ack or default_b_ack)
    if path == "speedy":
        comp_rate = min(comp_rate, 0.5)
    add_work(comp_core, "tx_completion", tx_rate * _cost(comp, path) * comp_rate)
    streams[comp_core].add("tx")
    if comp_core != tx:
        add_poll(comp_core, "poll completions", polls_cat["handoff"])
        connect("erisc_to_erisc", comp_core, tx, rate=tx_rate * comp_rate)

    # --- RX pieces: arrival (front) -> decode -> admit (packets with forwards) -> local delivery / forwards
    rx = scen["rx"]
    fwd = rx["forwards"]
    adm = P["rx_admit"]
    has_admit = "rx_admit" in names
    adm_core = placement["rx_admit"] if has_admit else dec
    add_poll(front, "poll rx ring", polls_cat["rx_ring"])
    streams[front].add("rx")
    if opts.rx_delay:
        add_work(front, "rx delay (experiment)", rx_rate * opts.rx_delay)
    decode_only = rx["decode"] - (adm["check_per_output"] * fwd if has_admit else 0)
    add_work(dec, "rx_decode", rx_rate * decode_only)
    streams[dec].add("rx")
    if has_admit:
        add_work(adm_core, "rx_admit", rx_rate * (adm["check_per_output"] + adm["reserve_per_output"]) * fwd)
        streams[adm_core].add("rx")
        # A packet whose outputs lack space is examined again on every visit: the credit check repeats, and so does
        # the decode when it runs on the same core and its result is not kept.
        blocked = opts.blocked_per_pkt if opts.blocked_per_pkt is not None else scen.get("blocked_per_pkt", 0.0)
        retry = adm["check_per_output"] * fwd + (decode_only if dec == adm_core and not opts.cache_decode else 0)
        if blocked:
            add_work(adm_core, "admission retries", rx_rate * blocked * retry)
        if kind[adm_core] == "drisc" and not opts.drisc_credits_direct:
            # downstream credits land in Ethernet-tile stream registers: relay them
            connect("credit_relay", front, adm_core, rate=rx_rate * fwd, batch=opts.relay_batch)

    header = header_bytes(cat, scen)
    payload_driscs = {q for q in payload_cores if kind[q] == "drisc"}
    pusher = {}  # DRISC -> ERISC that pushed the packet into its L1

    def message(src, dst, request=False):
        """Pass the packet (or its decision) from src to dst; returns False for an unsupported pattern."""
        if src == dst:
            return True
        if kind[src] == "erisc" and kind[dst] == "erisc":
            connect("erisc_to_erisc", src, dst, rate=rx_rate, batch=k)
        elif kind[src] == "erisc":
            if dst in payload_driscs and dst not in pusher:  # push the packet; the decision rides along
                connect("erisc_to_drisc_payload", src, dst, rate=rx_rate, batch=k)
                pusher[dst] = src
                noc[src] += rx_rate * (opts.payload + header)
                noc_in[dst] += rx_rate * (opts.payload + header)
            elif dst in pusher:  # packet already there: grant it
                connect("erisc_to_drisc_grant", src, dst, rate=rx_rate, batch=k)
            else:  # header only
                connect("erisc_to_drisc_header", src, dst, rate=rx_rate, batch=k)
                noc[src] += rx_rate * header
                noc_in[dst] += rx_rate * header
        elif kind[dst] == "erisc":
            connect("drisc_to_erisc_request" if request else "drisc_to_erisc", src, dst, rate=rx_rate, batch=k)
        else:
            return False
        return True

    ok = message(front, dec)
    if has_admit:
        ok = ok and message(dec, adm_core, request=kind[dec] == "drisc" and kind[adm_core] == "erisc")
    # Local delivery holds no downstream credit, so it needs only the decode (an admission request to the same core
    # already carries it); forwards need admission.
    if "rx_local" in names and not (has_admit and placement["rx_local"] == adm_core):
        ok = ok and message(dec, placement["rx_local"])
    if "rx_forward" in names:
        ok = ok and message(adm_core, placement["rx_forward"])
    if not ok:
        return Result(math.inf, "", {}, valid=False, why_invalid="handoff between two DRISCs")

    for p in PAYLOAD_PIECES:
        if p not in names:
            continue
        q = placement[p]
        if p == "rx_local":
            cycles, copies = _cost(P[p], rx["local"]), 1
        else:  # the reservation part of a forward is counted in rx_admit
            per_copy = _cost(P[p], "any") - adm["reserve_per_output"]
            if kind[q] == "drisc" and cat["core_types"]["drisc"].get("forward_per_copy") is not None:
                per_copy = cat["core_types"]["drisc"]["forward_per_copy"]
            # Each copy lands in the next router's sender channel, whose summary doorbell the forwarder rings.
            per_copy += summary_doorbell_producer(cat, opts)
            cycles, copies = per_copy * fwd, fwd
        add_work(q, p, rx_rate * cycles)
        streams[q].add("rx")
        noc[q] += rx_rate * opts.payload * copies
        noc_piece[q][p] = noc_piece[q].get(p, 0.0) + rx_rate * opts.payload * copies

    ack_core = placement["rx_ack"]
    ack = P["rx_ack"]
    b_ack = opts.b_ack or (4 if path == "speedy" else 1)
    add_work(ack_core, "rx_ack", rx_rate * _cost(ack, path) / b_ack)
    add_poll(ack_core, "poll retire", polls_cat["retire"])
    streams[ack_core].add("rx")
    # Ethernet-slot retirement happens where the slot is last read: ERISCs issuing payload writes, or the ERISC
    # that pushed the packet to a DRISC once the push is sent. Other cores hand retirement over to the ack core.
    origins = {q for q in payload_cores if kind[q] == "erisc"} | set(pusher.values())
    if not origins:
        origins.add(dec if kind[dec] == "erisc" else front)
    for o in origins - {ack_core}:
        connect("erisc_to_erisc", o, ack_core, rate=rx_rate, batch=b_ack)
    for d, src in pusher.items():  # DRISC staging-slot credits back to the pushing ERISC
        connect("drisc_to_erisc", d, src, rate=rx_rate, batch=max(k, 4))

    # --- per-core totals
    clock_loop = {c.name: cat["core_types"][c.kind]["loop"] for c in cores}
    ppi = {"tx": scen["pkts_per_iter"] * k, "rx": k}
    side_rate = {"tx": tx_rate, "rx": rx_rate}
    loads, breakdown = {}, {}
    for c in cores:
        name = c.name
        if not work[name] and not polls[name]:
            continue
        # Loop iterations per unit: enough for the busier of the streams the core serves.
        iters = max(side_rate[s] / ppi[s] for s in streams[name]) if streams[name] else 1.0
        overhead = (clock_loop[name] + sum(polls[name].values())) * iters
        loads[name] = sum(work[name].values()) + overhead
        bd = dict(work[name])
        bd[f"loop + polls ({', '.join(polls[name])})"] = overhead
        breakdown[name] = bd

    # A DRISC may run slower than an ERISC (alpha) and may serve several routers (share): scale its whole load.
    for c in cores:
        if c.name in loads:
            scale = core_scale(cat, c.kind, opts)
            loads[c.name] *= scale
            breakdown[c.name] = {lbl: v * scale for lbl, v in breakdown[c.name].items()}

    # Busier link direction. Each direction carries one side's packets and pointer updates, plus the credits for the
    # packets coming the other way; a one-way benchmark has only the data direction, bounded by the keepalive.
    data_wire, credit_wire = wire_cycles(cat, opts.payload, header, opts.b_ptr), credit_frame_cycles(cat, scen) / b_ack
    if scen.get("reverse_data", True):
        wire = max(tx_rate * data_wire + rx_rate * credit_wire, rx_rate * data_wire + tx_rate * credit_wire)
        keepalive = 0.0
    else:
        wire = max(tx_rate, rx_rate) * data_wire
        keepalive = max(tx_rate, rx_rate) * keepalive_floor(cat, opts.payload, header, opts.b_ptr, opts.keepalive)
    # Busiest NoC port direction: both ERISCs send through their Ethernet tile's ports; each DRISC has its own.
    eth = [c for c in cores if c.kind == "erisc"]
    port_bytes = [(max(sum(noc[c.name] for c in eth), sum(noc_in[c.name] for c in eth)), "erisc")]
    for c in cores:
        if c.kind == "erisc":
            continue
        if opts.drisc_split:  # forwards and local deliveries on separate ports; the pushes arrive on one of them
            out = max(noc_piece[c.name].values(), default=0.0)
        else:
            out = noc[c.name]
        port_bytes.append((max(out, noc_in[c.name]), c.kind))
    noc_floor = max(b * noc_cycles_per_byte(cat, kind, opts) for b, kind in port_bytes)
    bottleneck_core = max(loads, key=loads.get)
    interval, bottleneck = loads[bottleneck_core], bottleneck_core
    if wire > interval:
        interval, bottleneck = wire, "wire"
    if keepalive > interval:
        interval, bottleneck = keepalive, "keepalive"
    if noc_floor > interval:
        interval, bottleneck = noc_floor, "noc port"
    return Result(
        interval,
        bottleneck,
        loads,
        breakdown,
        noc,
        wire,
        floors={"wire": wire, "keepalive": keepalive, "noc port": noc_floor},
    )


def slots_needed(cat, scen_name, opts, interval):
    """Rough slot depth for which the credit loop does not bound `interval` (verify with des.py).

    Loop: data and ack each cross the link, the worker credit crosses the NoC twice, a few stage services, plus the
    wait to fill each batch (pointer updates, acks).
    """
    import des  # worker cost lives with the event-driven model

    lat, noc = cat["link"]["one_way_latency_cycles"], cat["noc"]["write_latency_cycles"]
    b_ack = opts.b_ack or (4 if cat["scenarios"][scen_name]["path"] == "speedy" else 1)
    loop = 2 * lat + 2 * noc + des.WORKER_COST + interval * (3 + (opts.b_ptr - 1) + (b_ack - 1))
    return math.ceil(loop / interval)


def placements(cat, scen_name, cores):
    """Every placement of the scenario's pieces on `cores`, with interchangeable cores taken once.

    The two ERISCs are identical, so tx_send is pinned to E0; DRISCs are numbered in order of first use. Unless the
    catalog's rules allow it, receive pieces stay off E0 (the sending ERISC, which runs SC0).
    """
    names = pieces_for(cat["scenarios"][scen_name])
    erisc = [c.name for c in cores if c.kind == "erisc"]
    allc = [c.name for c in cores]
    # Keep the sending ERISC (SC0) free of receive work unless the catalog allows it.
    rx_cores = allc if cat.get("rules", {}).get("rx_on_tx_erisc", True) else [c for c in allc if c != erisc[0]]
    rx_eriscs = [c for c in erisc if c in rx_cores]

    def choice(p):
        if p == "tx_send":
            return [erisc[0]]
        if p.startswith("rx_"):
            return rx_eriscs if p in ERISC_ONLY else rx_cores
        return erisc if p in ERISC_ONLY else allc

    choices = [choice(p) for p in names]
    driscs = [c.name for c in cores if c.kind == "drisc"]
    for combo in itertools.product(*choices):
        used = [c for c in combo if c in driscs]
        seen = []
        for c in used:
            if c not in seen:
                seen.append(c)
        if seen != driscs[: len(seen)]:  # canonical DRISC numbering
            continue
        yield dict(zip(names, combo))


def search(cat, scen_name, cores, opts=None, top=5):
    results = []
    for pl in placements(cat, scen_name, cores):
        r = evaluate(cat, scen_name, pl, cores, opts)
        if r.valid:
            results.append((r.interval, sorted(r.loads.values(), reverse=True), pl, r))
    results.sort(key=lambda x: (x[0], x[1]))
    return [(pl, r) for _, _, pl, r in results[:top]]


def describe(pl):
    by_core = {}
    for piece, core in pl.items():
        by_core.setdefault(core, []).append(piece)
    return "  ".join(f"{c}: {'+'.join(ps)}" for c, ps in sorted(by_core.items()))
