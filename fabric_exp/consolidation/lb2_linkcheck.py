#!/usr/bin/env python3
"""lb2_linkcheck.py: the lb2 harness's link gate (LB2_HARNESS.md, "Link gate"). It gates only on the Ethernet links
the fabric uses: the chip-to-chip pairs in the node's host-check cluster descriptor (`ethernet_connections`). It fails
closed.

  lb2_linkcheck.py live BASELINE_DESC OUTPREFIX
      On the node, read-only (the devices are opened, nothing is written to them):
      - writes the current UMD cluster descriptor (as topo_dump.py does) to OUTPREFIX_cluster_desc.yaml;
      - writes every Ethernet core's boot status, with its channel, to OUTPREFIX_eth.txt.
  lb2_linkcheck.py replay BASELINE_DESC NOW_DESC NOW_ETH
      Offline: the gate on saved files. NOW_ETH is a read_eth_speed.py readout ("chip N eth (x, y): port_status=..."),
      whose chip N is taken as descriptor chip N. NOW_DESC may be the baseline's when the run saved no descriptor;
      then gate A is not tested, and the gate says so.

The gate, in order. Any failure fails the gate (exit 1); exit 0 only when every step passes:
  A. Link set, mapping-free. The current descriptor's ethernet_connections, as unordered (chip, chan) pairs with chips
     named by unique id, equal the baseline's exactly. A missing pair is a fabric link that did not come up; an extra
     pair is unexpected wiring.
  B. Channel to core, per chip. Each connection end's channel goes to one Ethernet core in two ways that must agree:
     - live, UMD's own soc descriptor of that device (get_eth_cores_for_channels);
     - in both modes, the descriptor: the arch file's eth order (blackhole_140_arch.yaml) with that chip's
       eth_harvesting_mask positions removed.
     An end that maps to no core, more than one core, or two different cores fails the gate.
  C. State. Every connection end's core reads port UP (1), train PASS (2), speed 400.
Every other Ethernet core (ports outside ethernet_connections, including links to other hosts in
ethernet_connections_to_remote_devices) is listed as information and never gates.
"""
import collections
import os
import re
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ARCH = os.path.join(HERE, "../../tt_metal/soc_descriptors/blackhole_140_arch.yaml")
GOOD = (1, 2, 400)
LINE = re.compile(r"chip (\d+) eth \(\s*(\d+),\s*(\d+)\): port_status=(\d+) train_status=(\d+) train_speed=(\d+)")


def arch_eth_order():
    d = yaml.safe_load(open(ARCH))
    return [tuple(int(v) for v in s.split("-")) for s in d["eth"]]


def uid_of(desc):
    """chip id -> the chip's unique id (discovery may number chips differently from run to run); the chip id itself
    when the descriptor has none."""
    u = desc.get("chip_unique_ids") or {}
    return {int(c): int(u.get(c, u.get(str(c), c))) for c in desc["arch"]}


def links_of(desc):
    """The fabric links as unordered pairs of (chip unique id, channel)."""
    uid = uid_of(desc)
    pairs = set()
    for a, b in desc["ethernet_connections"]:
        pairs.add(frozenset([(uid[int(a["chip"])], int(a["chan"])), (uid[int(b["chip"])], int(b["chan"]))]))
    return pairs


def remote_ends(desc):
    return {(int(a["chip"]), int(a["chan"])) for a, _b in desc.get("ethernet_connections_to_remote_devices") or []}


def descriptor_mapping(desc):
    """chip -> {chan: (x, y)}: the arch file's eth order with the chip's harvested positions removed."""
    order = arch_eth_order()
    out = {}
    for chip, h in (desc.get("harvesting") or {}).items():
        mask = int(h.get("eth_harvesting_mask", 0))
        kept = [xy for i, xy in enumerate(order) if not (mask >> i) & 1]
        out[int(chip)] = {ch: xy for ch, xy in enumerate(kept)}
    return out


def gate(base, now, readout, umd_map, now_is_baseline=False):
    """base / now: descriptors (dicts); readout: chip -> {(x, y): (port, train, speed)}; umd_map: chip -> {chan: set
    of (x, y)} or None (replay). Prints the decision with its lists; returns True when the gate passes."""
    ok = True
    lb, ln = links_of(base), links_of(now)
    chip_now = {u: c for c, u in uid_of(now).items()}
    if not set(u for p in lb for u, _ch in p) <= set(chip_now):
        print(f"link gate: FAIL (chips of the baseline missing now: unique ids "
              f"{sorted(set(u for p in lb for u, _ch in p) - set(chip_now))})")
        return False
    ends = sorted({(chip_now[u], ch) for p in lb for u, ch in p})  # in the current descriptor's chip ids
    missing = sorted(sorted((chip_now.get(u, u), ch) for u, ch in p) for p in lb - ln)
    extra = sorted(sorted((chip_now.get(u, u), ch) for u, ch in p) for p in ln - lb)
    tag = " (no current descriptor saved: the baseline's used; A not tested)" if now_is_baseline else ""
    print(f"link gate A: {len(lb)} fabric links ({len(ends)} ends) in the baseline, {len(ln)} now{tag}; "
          f"missing {missing or 'none'}; extra {extra or 'none'}")
    ok &= not missing and not extra
    dmap = descriptor_mapping(now if not now_is_baseline else base)
    core_of, bad_map = {}, []
    for chip, chan in ends:
        d = dmap.get(chip, {}).get(chan)
        u = None if umd_map is None else umd_map.get(chip, {}).get(chan, set())
        if d is None or (u is not None and (len(u) != 1 or next(iter(u)) != d)):
            bad_map.append(f"chip {chip} chan {chan}: descriptor {d}, UMD {sorted(u) if u is not None else '-'}")
        else:
            core_of[(chip, chan)] = d
    src = "UMD and the descriptor agree" if umd_map is not None else "descriptor only (replay)"
    print(f"link gate B: {len(core_of)} of {len(ends)} ends mapped to one core ({src}); unmapped or disagreeing: "
          f"{bad_map or 'none'}")
    ok &= not bad_map
    not_good = []
    for (chip, chan), xy in sorted(core_of.items()):
        st = readout.get(chip, {}).get(xy)
        if st != GOOD:
            not_good.append(f"chip {chip} chan {chan} eth {xy}: {st}")
    print(f"link gate C: {len(core_of) - len(not_good)} of {len(core_of)} fabric link ends UP/PASS/400; not: "
          f"{not_good or 'none'}")
    ok &= not not_good and len(core_of) == len(ends)
    # information only: every other Ethernet core
    fabric_cores = {(c, xy) for (c, _ch), xy in core_of.items()}
    rem = remote_ends(now if not now_is_baseline else base)
    chan_of = {c: {xy: ch for ch, xy in m.items()} for c, m in dmap.items()}
    info, unused, up = [], 0, 0
    for chip in sorted(readout):
        for xy, st in sorted(readout[chip].items()):
            if (chip, xy) in fabric_cores:
                continue
            ch = chan_of.get(chip, {}).get(xy)
            if st == GOOD:
                up += 1
            elif st[0] == 3:  # port UNUSED
                unused += 1
            else:
                kind = "a link to another host" if (chip, ch) in rem else "not a fabric link"
                info.append(f"   port down ({kind}): chip {chip} chan {ch} eth {xy}: port {st[0]} train {st[1]} "
                            f"speed {st[2]}")
    print(f"link gate info (never gating): {sum(len(v) for v in readout.values()) - len(fabric_cores)} other "
          f"Ethernet cores: {up} UP/PASS/400, {unused} unused, {len(info)} down or training:")
    for l in info:
        print(l)
    print(f"link gate: {'PASS' if ok else 'FAIL'}")
    return ok


def parse_readout(path):
    out = collections.defaultdict(dict)
    for l in open(path):
        m = LINE.match(l)
        if m:
            out[int(m.group(1))][(int(m.group(2)), int(m.group(3)))] = tuple(map(int, m.group(4, 5, 6)))
    return out


def live(base_path, prefix):
    import tt_umd

    base = yaml.safe_load(open(base_path))
    desc = tt_umd.TopologyDiscovery.create_cluster_descriptor()
    desc.serialize_to_file(prefix + "_cluster_desc.yaml")
    now = yaml.safe_load(open(prefix + "_cluster_desc.yaml"))
    # descriptor chip -> PCI device number: by PCI BDF, else by PCI bus id; a chip matched by neither, or two chips on
    # one device, fail the gate (fail closed)
    norm = lambda b: str(b).lower().split(":", 1)[-1] if str(b).count(":") == 2 else str(b).lower()
    infos = tt_umd.PCIDevice.enumerate_devices_info()
    dev_of_bdf = {norm(info.pci_bdf): n for n, info in infos.items()}
    dev_of_bus = {int(info.pci_bus): n for n, info in infos.items()}
    bdfs, buses = desc.get_chip_pci_bdfs(), desc.get_chip_to_bus_id()
    dev_of_chip = {}
    for chip in sorted(int(c) for c in now["arch"]):
        n = dev_of_bdf.get(norm(bdfs.get(chip, ""))) if bdfs.get(chip) else None
        if n is None and chip in buses:
            n = dev_of_bus.get(int(buses[chip]))
        dev_of_chip[chip] = n
    if None in dev_of_chip.values() or len(set(dev_of_chip.values())) != len(dev_of_chip):
        print(f"link gate: FAIL (descriptor chip -> PCI device not one-to-one: {dev_of_chip}; BDFs {bdfs}; "
              f"devices {{n: info.pci_bdf}} = { {n: i.pci_bdf for n, i in infos.items()} })")
        return False
    print(f"link gate: descriptor chip -> PCI device {dev_of_chip}")
    boot = 0x7CC00
    readout, umd_map, lines = collections.defaultdict(dict), {}, []
    for chip in sorted(int(c) for c in now["arch"]):
        n = dev_of_chip[chip]
        dev = tt_umd.TTDevice.create(n)
        dev.init_tt_device()
        soc = dev.get_soc_descriptor()
        umd_map[chip] = {}
        for ch in range(16):
            try:
                cores = soc.get_eth_cores_for_channels({ch}, tt_umd.CoordSystem.NOC0)
            except Exception:  # not a channel of this chip
                continue
            umd_map[chip][ch] = {(c.x, c.y) for c in cores}
        chan_of = {next(iter(v)): ch for ch, v in umd_map[chip].items() if len(v) == 1}
        for core in soc.get_cores(tt_umd.CoreType.ETH, tt_umd.CoordSystem.NOC0):
            st = tuple(dev.noc_read32(core.x, core.y, boot + off) for off in (0x04, 0x08, 0x0C))
            readout[chip][(core.x, core.y)] = st
            lines.append(f"chip {chip} eth ({core.x:2d},{core.y:2d}): port_status={st[0]} train_status={st[1]} "
                         f"train_speed={st[2]} chan={chan_of.get((core.x, core.y))} pci_dev={n}")
    open(prefix + "_eth.txt", "w").write("\n".join(lines) + "\n")
    return gate(base, now, readout, umd_map)


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "live":
        try:
            ok = live(sys.argv[2], sys.argv[3])
        except Exception as e:  # fail closed
            print(f"link gate: FAIL (error: {type(e).__name__}: {e})")
            ok = False
    elif len(sys.argv) == 5 and sys.argv[1] == "replay":
        base, now = yaml.safe_load(open(sys.argv[2])), yaml.safe_load(open(sys.argv[3]))
        ok = gate(base, now, parse_readout(sys.argv[4]), None, now_is_baseline=sys.argv[2] == sys.argv[3])
    else:
        print(__doc__)
        sys.exit(2)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
