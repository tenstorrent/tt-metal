#!/usr/bin/env python3
"""pc_sampler.py OUT_PREFIX --start-log LOG [--start-pattern P] [--stop-pattern P] [--max-s S] [--chips 0,1]
[--eriscs 0,1] [--only-loaded]: statistical PC sampling of the fabric routers' ERISCs while a test runs (RESULTS §98, phase/cost
analysis trial).

Each Blackhole Ethernet core has a debug bus: an observation mux read through two tile registers,
RISCV_DEBUG_REG_DBG_BUS_CNTL_REG (select a signal) and RISCV_DEBUG_REG_DBG_RD_DATA (its current value). The signals
erisc0_pc / erisc1_pc are the PC of the instruction in each ERISC's decode/execute stage (tt-exalens's get_pc uses the
same path). Reading them does not halt, step or otherwise touch the core: one NoC write (the select) and one NoC read
per sample, from the host over PCIe.

The device is opened as hang_capture.py opens it (tt-exalens, safe mode: no reset, no firmware load, no core init),
and only after the test has opened it and logged START_PATTERN, so the test's own device init is never overlapped.
Targets: every ERISC of every connected Ethernet core of the chosen chips, sampled round-robin until STOP_PATTERN
appears in the log or --max-s passes. Before sampling, each core's launch message (mailbox, as hang_decode.py reads it)
and the first bytes of each ERISC's kernel image are read once (plain L1 reads), so pcs_report.py can tell which build
a core runs and where it was loaded: kernels run from the kernel config window (config base + text offset), not from
their link address. Output: OUT_PREFIX.bin (records of float64 t, uint16 target, uint32 pc) and OUT_PREFIX.json
(targets with load addresses and image heads, rates, errors)."""
import argparse
import collections
import json
import os
import struct
import sys
import time

MAILBOX = 0x100  # MEM_AERISC_MAILBOX_BASE
LAUNCH_SIZE = 144  # sizeof(launch_msg_t)
ACTIVE_ETH = 1  # ProgrammableCoreType::ACTIVE_ETH
IMAGE_HEAD = 256  # bytes of each kernel image kept for matching


def u32(b, o):
    return struct.unpack_from("<I", b, o)[0]


def core_layout(loc):
    """The current launch message's kernel config base and per-ERISC text offset/size, and each image's head."""
    mb = bytearray(16 + 8 * LAUNCH_SIZE)
    loc.noc_read(MAILBOX, mb)
    rd = u32(mb, 12)
    rd = rd if rd < 8 else 0
    lb = 16 + LAUNCH_SIZE * rd
    kcb = u32(mb, lb + 4 * ACTIVE_ETH)
    out = {"rd_ptr": rd, "kcb": kcb, "erisc": {}}
    for p in (0, 1):
        off, size = u32(mb, lb + 52 + 4 * p), u32(mb, lb + 72 + 4 * p)
        head = bytearray(min(IMAGE_HEAD, size)) if size else bytearray()
        if head:
            loc.noc_read(kcb + off, head)
        out["erisc"][p] = {"text_off": off, "text_size": size, "load": kcb + off, "head": head.hex()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--start-log", required=True)
    ap.add_argument("--start-pattern", default="Running Test:")
    ap.add_argument("--stop-pattern", default="Tests: ")
    ap.add_argument("--start-delay-s", type=float, default=0.0)
    ap.add_argument("--max-s", type=float, default=30.0)
    ap.add_argument("--wait-s", type=float, default=300.0)
    ap.add_argument("--chips", default="")
    ap.add_argument("--eriscs", default="0,1")
    ap.add_argument("--only-loaded", action="store_true",
                    help="only cores whose launch message holds an ERISC0 kernel when sampling starts (the routers)")
    ap.add_argument("--tensix", default="",
                    help="Tensix cores to sample instead of the Ethernet cores: a file of 'chip x y' lines (noc0)")
    ap.add_argument("--tensix-riscs", default="brisc,ncrisc", help="Tensix RISCs to sample (debug-bus <risc>_pc)")
    ap.add_argument("--cores", default="", help="mixed targets: a file of 'chip block x y risc[,risc]' lines (see above)")
    ap.add_argument("--snap-s", type=float, default=0.05, help="Tensix mode: mailbox-head snapshot period (s)")
    args = ap.parse_args()

    def log_has(pat):
        try:
            with open(args.start_log, errors="replace") as f:
                return pat in f.read()
        except FileNotFoundError:
            return False

    t0 = time.time()
    while not log_has(args.start_pattern):
        if time.time() - t0 > args.wait_s:
            print("sampler: start pattern never appeared", flush=True)
            return 2
        time.sleep(0.05)
    time.sleep(args.start_delay_s)
    print(f"sampler: start seen after {time.time() - t0:.1f} s; opening the device", flush=True)

    os.environ.pop("TT_VISIBLE_DEVICES", None)
    from ttexalens.tt_exalens_init import init_ttexalens
    from ttexalens.context import NocId

    t_open = time.time()
    ctx = init_ttexalens(noc_id=NocId.NOC0, safe_mode=True)
    print(f"sampler: device open {time.time() - t_open:.2f} s", flush=True)

    conns = ctx.cluster_descriptor.get_ethernet_connections()
    want = {int(c) for c in args.chips.split(",")} if args.chips else None
    eriscs = [int(e) for e in args.eriscs.split(",")]
    targets = []  # (chip, chan, erisc, loc, ctrl_addr, data_addr, config, mask)
    errors = {}
    layouts = {}  # (chip, chan) -> core_layout
    tensix = collections.defaultdict(set)
    if args.tensix:
        for line in open(args.tensix):
            if line.strip() and not line.startswith("#"):
                c, x, y = (int(v) for v in line.split()[:3])
                tensix[c].add((x, y))
    # --cores FILE: lines 'chip block x y risc[,risc]' (block: functional_workers | eth | dram; x y in noc0, or '* *' for
    # every block of that type on the chip; risc names as in the debug-bus signal <risc>_pc: brisc, ncrisc, erisc0,
    # erisc1, drisc). Mixed block types on one timeline (relays next to the senders).
    extra = collections.defaultdict(list)
    if args.cores:
        for line in open(args.cores):
            if line.strip() and not line.startswith("#"):
                f = line.split()
                extra[int(f[0])].append((f[1], f[2], f[3], f[4].split(",")))
    for chip_id, dev in sorted(ctx.devices.items()):
        if want is not None and chip_id not in want:
            continue
        for block, xs, ys, riscs in extra.get(chip_id, []):
            for loc in dev.get_block_locations(block):
                xy = tuple(int(v) for v in loc.to("noc0"))
                if xs != "*" and xy != (int(xs), int(ys)):
                    continue
                bus = dev.get_block(loc).debug_bus
                for r in riscs:
                    try:
                        sig = bus.get_signal_description(f"{r}_pc")
                    except Exception as ex:  # noqa: BLE001
                        errors[f"{chip_id}/{block}/{xy}/{r}: no signal: {ex}"] = 1
                        continue
                    config = (1 << 29) | (sig.rd_sel << 25) | (sig.daisy_sel << 16) | sig.sig_sel
                    tag = f"{block[0]}{xy[0]}-{xy[1]}"
                    targets.append((chip_id, tag, r, loc, bus._control_register_address, bus._data_register_address,
                                    config, sig.mask))
        if args.cores and not args.tensix:
            continue
        if args.tensix:
            # Tensix mode: the listed workers' RISC PCs (same debug-bus read as the ERISCs); the launch message is the
            # Tensix mailbox's, read by pcs tools from the kernel cache instead, so no layout here.
            for loc in dev.get_block_locations("functional_workers"):
                xy = tuple(int(v) for v in loc.to("noc0"))
                if xy not in tensix.get(chip_id, ()):
                    continue
                bus = dev.get_block(loc).debug_bus
                for r in args.tensix_riscs.split(","):
                    sig = bus.get_signal_description(f"{r}_pc")
                    config = (1 << 29) | (sig.rd_sel << 25) | (sig.daisy_sel << 16) | sig.sig_sel
                    targets.append((chip_id, f"{xy[0]}-{xy[1]}", r, loc, bus._control_register_address,
                                    bus._data_register_address, config, sig.mask))
            continue
        active = set(conns.get(chip_id, {}).keys())
        for loc in dev.get_block_locations("eth"):
            chan = int(loc.to("logical")[0][1])
            if chan not in active:
                continue
            bus = dev.get_block(loc).debug_bus
            try:
                layouts[(chip_id, chan)] = core_layout(loc)
            except Exception as ex:  # noqa: BLE001
                layouts[(chip_id, chan)] = {"error": f"{type(ex).__name__}: {ex}"}
            if args.only_loaded and not layouts[(chip_id, chan)].get("erisc", {}).get(0, {}).get("text_size"):
                del layouts[(chip_id, chan)]
                continue
            for e in eriscs:
                sig = bus.get_signal_description(f"erisc{e}_pc")
                config = (1 << 29) | (sig.rd_sel << 25) | (sig.daisy_sel << 16) | sig.sig_sel
                targets.append((chip_id, chan, e, loc, bus._control_register_address, bus._data_register_address,
                                config, sig.mask))
    print(f"sampler: {len(targets)} targets", flush=True)

    recs = bytearray()
    n = 0
    # Tensix mode: every --snap-s, each sampled core's mailbox head (launch ring + go messages, as hang_workers.py reads
    # it), so the analysis can tell which kernel image (load address) each sample ran in.
    snap_cores = []
    if args.tensix or args.cores:
        seen = set()
        for t in targets:
            if str(t[1])[0] in "f0123456789" and (args.tensix or str(t[1]).startswith("f")) and (t[0], t[1]) not in seen:
                seen.add((t[0], t[1]))
                snap_cores.append((t[0], t[1], t[3]))
    snaps = bytearray()
    SNAP_BASE, SNAP_SIZE = 0x60, 1488
    last_snap = 0.0
    t_start = time.time()
    t_start_epoch = t_start  # sample t is seconds after this wall-clock time (to line samples up with log timestamps)
    stop_check = t_start
    while True:
        for i, (chip_id, chan, e, loc, ctrl, data, config, mask) in enumerate(targets):
            try:
                loc.noc_write32(ctrl, config)
                pc = loc.noc_read32(data) & mask
                recs += struct.pack("<dHI", time.time() - t_start, i, pc)
                n += 1
            except Exception as ex:  # noqa: BLE001
                k = f"{chip_id}/{chan}/{e}: {type(ex).__name__}: {str(ex)[:120]}"
                errors[k] = errors.get(k, 0) + 1
        now = time.time()
        if snap_cores and now - last_snap > args.snap_s:
            last_snap = now
            for k, (c, xy, loc) in enumerate(snap_cores):
                buf = bytearray(SNAP_SIZE)
                try:
                    loc.noc_read(SNAP_BASE, buf)
                    snaps += struct.pack("<dH", time.time() - t_start, k) + bytes(buf)
                except Exception as ex:  # noqa: BLE001
                    kk = f"{c}/{xy} snap: {type(ex).__name__}: {str(ex)[:120]}"
                    errors[kk] = errors.get(kk, 0) + 1
        if now - t_start > args.max_s:
            print("sampler: max time reached", flush=True)
            break
        if now - stop_check > 0.25:
            stop_check = now
            if log_has(args.stop_pattern):
                print("sampler: stop pattern seen", flush=True)
                break
    dur = time.time() - t_start
    # Read the layouts again: the routers may have launched after the first read; their launch message stays in L1.
    layouts_end = {}
    for key in (layouts if not args.tensix else []):
        loc = next(t[3] for t in targets if (t[0], t[1]) == key)
        try:
            layouts_end[key] = core_layout(loc)
        except Exception as ex:  # noqa: BLE001
            layouts_end[key] = {"error": f"{type(ex).__name__}: {ex}"}
    with open(args.out + ".bin", "wb") as f:
        f.write(recs)
    if snap_cores:
        with open(args.out + ".snap.bin", "wb") as f:
            f.write(snaps)
    meta = {
        "targets": [{"chip": t[0], "chan": t[1], "erisc": t[2]} for t in targets],
        "layouts": {f"{c}/{ch}": v for (c, ch), v in layouts.items()},
        "layouts_end": {f"{c}/{ch}": v for (c, ch), v in layouts_end.items()},
        "samples": n,
        "snap_cores": [{"chip": c, "core": xy} for c, xy, _ in snap_cores],
        "t_start_epoch": t_start_epoch,
        "duration_s": round(dur, 3),
        "rate_per_s": round(n / dur, 1) if dur else None,
        "errors": errors,
    }
    with open(args.out + ".json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"sampler: {n} samples in {dur:.2f} s ({meta['rate_per_s']}/s), {sum(errors.values())} read errors",
          flush=True)
    try:
        from ttexalens.tt_exalens_init import cleanup_global_context
        cleanup_global_context()
    except Exception:  # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
