#!/usr/bin/env python3
"""bnd_ethniu.py OUT.json [--chips 0,1,..] [--pc-reads N]: read-only snapshot of every connected Ethernet core's NoC
NIU counters and ERISC PCs (V4_BOUNDS.md, the inline-write check). Run after the test process has exited (a hung
router keeps running). Per chip and Ethernet core: NOC0 and NOC1 NIU status counters 0..63 (NOC_STATUS(i) = NoC
register block + 0x200 + 4i: NIU_MST_WR_ACK_RECEIVED 0x1, NONPOSTED_WR_REQ_SENT 0xA, POSTED_WR_REQ_SENT 0xB,
REQS_OUTSTANDING_ID(id) 0x10+id, WRITE_REQS_OUTGOING_ID(id) 0x20+id, ...), and N reads of each ERISC's PC over the
debug bus (pc_sampler.py's method: one select write, one read; the core is not halted). The device is opened as
hang_capture.py / pc_sampler.py open it (tt-exalens, safe mode). PCs are load addresses; map them with the launch
message (pc_sampler.core_layout) kept in the output."""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pc_sampler import core_layout  # noqa: E402

NIU = {0: 0xFFB20000, 1: 0xFFB30000}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--chips", default="")
    ap.add_argument("--pc-reads", type=int, default=16)
    a = ap.parse_args()
    os.environ.pop("TT_VISIBLE_DEVICES", None)
    from ttexalens.tt_exalens_init import init_ttexalens
    from ttexalens.context import NocId

    ctx = init_ttexalens(noc_id=NocId.NOC0, safe_mode=True)
    conns = ctx.cluster_descriptor.get_ethernet_connections()
    want = {int(c) for c in a.chips.split(",")} if a.chips else None
    out = {"cores": [], "errors": []}
    for chip_id, dev in sorted(ctx.devices.items()):
        if want is not None and chip_id not in want:
            continue
        active = set(conns.get(chip_id, {}).keys())
        for loc in dev.get_block_locations("eth"):
            chan = int(loc.to("logical")[0][1])
            if chan not in active:
                continue
            rec = {"chip": chip_id, "chan": chan, "niu": {}, "pc": {}}
            try:
                rec["layout"] = core_layout(loc)
            except Exception as ex:  # noqa: BLE001
                rec["layout"] = {"error": f"{type(ex).__name__}: {ex}"}
            for noc, base in NIU.items():
                vals = []
                for i in range(64):
                    try:
                        vals.append(loc.noc_read32(base + 0x200 + 4 * i))
                    except Exception as ex:  # noqa: BLE001
                        vals.append(None)
                        out["errors"].append(f"{chip_id}/{chan} noc{noc} {i}: {type(ex).__name__}: {str(ex)[:100]}")
                rec["niu"][noc] = vals
            bus = dev.get_block(loc).debug_bus
            for e in (0, 1):
                try:
                    sig = bus.get_signal_description(f"erisc{e}_pc")
                    config = (1 << 29) | (sig.rd_sel << 25) | (sig.daisy_sel << 16) | sig.sig_sel
                    pcs = []
                    for _ in range(a.pc_reads):
                        loc.noc_write32(bus._control_register_address, config)
                        pcs.append(loc.noc_read32(bus._data_register_address) & sig.mask)
                    rec["pc"][e] = pcs
                except Exception as ex:  # noqa: BLE001
                    out["errors"].append(f"{chip_id}/{chan} erisc{e} pc: {type(ex).__name__}: {str(ex)[:100]}")
            out["cores"].append(rec)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"ethniu: {len(out['cores'])} cores, {len(out['errors'])} errors -> {a.out}")


if __name__ == "__main__":
    main()
