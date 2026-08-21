#!/usr/bin/env python3
"""
DIAGNOSTIC 2 -- confirm the troublesome state actually occurred, end to end.

Takes the JSON from l1_cache_audit.py and, for each corrupt byte, establishes the full chain
so there is no doubt about what happened or where:

  1. ON-DISK ELF    the reference byte
  2. DRAM           locates the kernels_buffer copy by content and reads the same byte.
                    DRAM correct + cache wrong  ==  the error entered on the DRAM->L1 relay.
                    Also runs the cross-port oracle there (3 aliased NOC ports x N reads) to
                    prove the DRAM read path is not the culprit.
  3. PREFETCH L1    the corrupt byte, re-read through several independent paths and widths to
                    rule out a readback artifact
  4. WORKER L1      how many worker cores received the corruption (the multicast fan-out)
  5. DECODE         disassembles the containing word before and after, and resolves it to a
                    source location -- i.e. whether this byte is lethal or silent

READ-ONLY throughout. Safe against a live hang.
"""
import argparse, glob, json, os, subprocess, sys

PAGE, NBANKS = 2048, 8
OBJDUMP = None
A2L = None


def _tool(name):
    for base in ("runtime/sfpi/compiler/bin", "/opt/tenstorrent/sfpi/compiler/bin"):
        p = os.path.join(base, f"riscv-tt-elf-{name}")
        if os.path.exists(p):
            return p
    return None


def decode(word):
    """Return the mnemonic for a 32-bit word, or None if the target cannot decode it."""
    if not OBJDUMP:
        return None
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(word.to_bytes(4, "little") + b"\0\0\0\0")
        path = f.name
    try:
        out = subprocess.run(
            [OBJDUMP, "-D", "-b", "binary", "-m", "riscv:rv32", path], capture_output=True, text=True, timeout=20
        ).stdout
    finally:
        os.unlink(path)
    for line in out.splitlines():
        if not line.strip().startswith("0:"):
            continue
        # objdump emits:  "   0:\t280007b7 \tlui\ta5,0x28000"
        # everything from the third tab-separated field on is the instruction; an encoding the
        # target cannot decode comes back as ".insn 4, 0x...", which is the illegal-instruction
        # signal and must NOT be mistaken for a mnemonic.
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        txt = " ".join(x.strip() for x in parts[2:]).strip()
        return None if txt.startswith(".insn") else txt
    return None


def find_dram(ctx, dev, dev_id, anchor, anchor_off, top, low, chunk):
    """Locate an interleaved DRAM buffer by content. Returns (bank, addr_of_blob_start)."""
    from ttexalens.coordinate import OnChipCoordinate
    from ttexalens import tt_exalens_lib as lib

    for ch in range(NBANKS):
        loc = OnChipCoordinate.create(f"d{ch},0", dev)
        addr = top - chunk
        while addr >= low:
            buf = bytes(lib.read_from_device(loc, addr, device_id=dev_id, num_bytes=chunk, context=ctx))
            pos = buf.find(anchor)
            if pos >= 0:
                return ch, addr + pos - anchor_off
            addr -= chunk
    return None, None


def main():
    global OBJDUMP, A2L
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-json", required=True, help="output of l1_cache_audit.py --json")
    ap.add_argument("--worker-scan", type=int, default=160, help="worker cores to poll per device")
    ap.add_argument("--dram-top", type=lambda s: int(s, 0), default=0xFF000000)
    ap.add_argument("--dram-low", type=lambda s: int(s, 0), default=0xFC000000)
    ap.add_argument("--dram-chunk", type=lambda s: int(s, 0), default=2 << 20)
    ap.add_argument("--oracle-reps", type=int, default=20)
    ap.add_argument("--skip-dram", action="store_true")
    a = ap.parse_args()

    OBJDUMP, A2L = _tool("objdump"), _tool("addr2line")
    from ttexalens.tt_exalens_init import init_ttexalens
    from ttexalens.coordinate import OnChipCoordinate
    from ttexalens.memory_access import create_l1_memory_access
    from ttexalens import tt_exalens_lib as lib
    from ttexalens.elf import ElfFile

    rep = json.load(open(a.audit_json))
    if not rep["findings"]:
        print("audit reported no findings -- nothing to confirm")
        return 0

    ctx = init_ttexalens()
    by = {d.id: d for d in ctx.devices.values()}
    HDR = 0x20

    for f in rep["findings"]:
        dev_id, core = f["device"], f["core"]
        dev = by[dev_id]
        text = bytes(ElfFile.from_bytes(open(f["elf"], "rb").read()).get_section_by_name(".text").data)
        for d in f["diffs"]:
            boff, toff = d["blob_off"], d["text_off"]
            print("=" * 78)
            print(f"device {dev_id}  core {core}  {f['elf'].split('/kernels/')[-1]}")
            print(
                f"  .text+0x{toff:05x}   expected {d['expected']:02x}   "
                f"observed {d['observed']:02x}   xor {d['xor']:02x}   bit(s) {d['bits']}"
            )

            # ---- 1. reference ------------------------------------------------------
            w0 = boff & ~3
            exp_word = int.from_bytes(text[HDR + w0 : HDR + w0 + 4], "little")
            bad_word = exp_word ^ (d["xor"] << (8 * (boff - w0)))
            print(f"\n  [1] ON-DISK ELF   word 0x{exp_word:08x}   {decode(exp_word) or 'UNDECODABLE'}")

            # ---- 2. DRAM ----------------------------------------------------------
            if not a.skip_dram:
                anchor_off = max(0, boff - 0x28)
                anchor = text[HDR + anchor_off : HDR + anchor_off + 0x28]
                bank, base = find_dram(ctx, dev, dev_id, anchor, anchor_off, a.dram_top, a.dram_low, a.dram_chunk)
                if bank is None:
                    print("  [2] DRAM          kernels_buffer not found in the scanned range")
                else:
                    vals, xdiff = {}, 0
                    for sub in range(3):
                        try:
                            loc = OnChipCoordinate.create(f"d{bank},{sub}", dev)
                        except Exception:
                            continue
                        bufs = [
                            bytes(
                                lib.read_from_device(
                                    loc, base + boff - 64, device_id=dev_id, num_bytes=128, context=ctx
                                )
                            )
                            for _ in range(a.oracle_reps)
                        ]
                        vals[loc.to_str("noc0")] = (len(set(bufs)), bufs[0][64])
                    names = list(vals)
                    print(
                        f"  [2] DRAM          bank {bank} @ 0x{base + boff:08x}  "
                        f"byte = {vals[names[0]][1]:02x}  "
                        f"{'CORRECT' if vals[names[0]][1] == d['expected'] else 'ALSO WRONG'}"
                    )
                    for n, (nd, v) in vals.items():
                        print(
                            f"                    port {n:6s}: {nd} distinct value(s) over "
                            f"{a.oracle_reps} reads, byte {v:02x}"
                        )
                    if len({v for _, v in vals.values()}) == 1:
                        print(
                            f"                    cross-port oracle: all {len(vals)} ports AGREE "
                            f"-> DRAM read path is not the fault"
                        )

            # ---- 3. prefetch/dispatch L1, many read paths --------------------------
            l1 = f["l1_base"] + boff
            mem = create_l1_memory_access(OnChipCoordinate.create(core, dev, "noc0"))
            reads = {}
            for label, off, n, idx in (
                ("read(4)", l1, 4, 0),
                ("read(64)@-32", l1 - 32, 64, 32),
                ("read(4096)@-2048", l1 - 2048, 4096, 2048),
            ):
                b = bytearray(n)
                mem.read(off, b)
                reads[label] = b[idx]
            wl = lib.read_words_from_device(
                OnChipCoordinate.create(core, dev, "noc0"), l1 & ~3, device_id=dev_id, word_count=1, context=ctx
            )[0]
            reads["read_words"] = (wl >> (8 * (l1 & 3))) & 0xFF
            uniq = set(reads.values())
            print(f"\n  [3] {core} L1 0x{l1:06x}   " + "  ".join(f"{k}={v:02x}" for k, v in reads.items()))
            print(
                f"                    {len(uniq)} distinct value(s) across "
                f"{len(reads)} independent read paths"
                f"{'  -> NOT a readback artifact' if len(uniq) == 1 else '  -> UNSTABLE'}"
            )

            # ---- 4. worker fan-out ------------------------------------------------
            hits = miss = 0
            probe = text[HDR + max(0, boff - 0x28) : HDR + max(0, boff - 0x28) + 0x28]
            poff = boff - max(0, boff - 0x28)
            for x in range(1, 17):
                for y in range(2, 12):
                    if hits + miss >= a.worker_scan:
                        break
                    try:
                        m = create_l1_memory_access(OnChipCoordinate.create(f"{x}-{y}", dev, "noc0"))
                        buf = bytearray(0x180000)
                        m.read(0, buf)
                        buf = bytes(buf)
                    except Exception:
                        continue
                    p = buf.find(probe)
                    if p < 0:
                        continue
                    if buf[p + poff] == d["observed"]:
                        hits += 1
                    else:
                        miss += 1
            print(
                f"\n  [4] fan-out       {hits} core(s) hold the CORRUPT byte, "
                f"{miss} hold the correct byte  "
                f"(includes the {core} entry itself and the peer dispatch core)"
            )

            # ---- 5. what the corruption does --------------------------------------
            dec_bad = decode(bad_word)
            print(
                f"\n  [5] EXECUTED      word 0x{bad_word:08x}   " f"{dec_bad or 'UNDECODABLE -- illegal instruction'}"
            )
            if A2L:
                vaddr = ElfFile.from_bytes(open(f["elf"], "rb").read()).get_section_by_name(".text").address + toff
                out = subprocess.run(
                    [A2L, "-f", "-C", "-i", "-e", f["elf"], hex(vaddr)], capture_output=True, text=True, timeout=30
                ).stdout.strip()
                if out:
                    print(f"      at .text vaddr 0x{vaddr:x}:")
                    for line in out.splitlines()[:8]:
                        print(f"        {line}")
            verdict = (
                "LETHAL -- the core traps on an undecodable instruction"
                if dec_bad is None
                else f"SILENT-OR-WRONG -- still decodes ({dec_bad}), so it executes with wrong effect"
            )
            print(f"\n  VERDICT: {verdict}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
