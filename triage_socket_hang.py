#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
triage_socket_hang.py — ONE command: from a tt-triage dump, find the wedged D2D pipeline hop AND
read the two live counters (each end on its own host) to give the lost-forward vs lost-ack verdict.
Nothing passed manually.

  python3 triage_socket_hang.py <dump.txt> \
      --rank-binding rev_c_rank_binding_superpod.yaml \
      --rankfile rev_c_rank_file_superpod [--page-size <bytes>]

What it does, end to end:
  1. Parse the dump's dump_callstacks section -> per-stage relay states (R=socket_reserve_pages,
     W=socket_wait_for_pages) -> the BACKED_UP->STARVED boundary = the hung hop (stage s -> s+1).
  2. Pull the exact stuck d2d_exchange relay on each end: sender = the R relay on stage s,
     receiver = the W relay on stage r; capture its (dev, core).
  3. Resolve host (rankfile) + exalens chip id (rank_binding TT_VISIBLE_DEVICES[dev]) per end.
  4. ssh to each host, run dump_socket_counters.py --scan --end {sender|receiver} there
     (the --scan recovers config_addr from L1 — no addresses supplied), read S_acked / R_sent.
  5. D = (R_sent - S_acked) mod 2^32 : D~=0 => LOST FORWARD ; D~=fifo => LOST BACKWARD ACK.

Run it from a host with agent-forwarded ssh to the pod (e.g. the head node). Assumes the shared-NFS
checkout path is the same on every host (BLAZE_DIR below).
"""
import argparse
import re
import subprocess
import sys

BLAZE_DIR = "/data/aho/tt-blaze2"
MASK = 0xFFFFFFFF

HDR = re.compile(r'^(\d+),"(\d+-\d+) \((\d+),(\d+)\)",([a-z0-9]+),(\d+),([A-Za-z0-9_]+),')
RANKM = re.compile(r'^\s*\[rank (\d+)\]')
SECTION = re.compile(r'^[A-Za-z_]+\.py:')
FRAME0 = re.compile(r'#0\s+0x[0-9A-Fa-f]+ in\s+([^\s(]+)')
RESERVE = "socket_reserve_pages"
WAIT = "socket_wait_for_pages"


def _state(rec):
    m = FRAME0.search(rec)
    if not m:
        return None
    fn = m.group(1)
    if RESERVE in fn:
        return "R"
    if WAIT in fn:  # matches socket_wait_for_pages and ..._with_termination
        return "W"
    return None


def parse_relays(dump):
    """rank -> list of dicts {dev, core, risc, state} for d2d_exchange relays that are R or W."""
    relays = {}
    in_cs = False
    rank = None
    rec = None
    hdr = None

    def flush():
        if rec is None or hdr is None or rank is None:
            return
        if hdr[6] != "d2d_exchange":   # kernel name column; the pipeline relay (no hash suffix)
            return
        st = _state(rec)
        if st:
            relays.setdefault(rank, []).append(
                {"dev": int(hdr[0]), "core": f"{hdr[2]},{hdr[3]}", "risc": hdr[4], "state": st})

    for line in open(dump, errors="replace"):
        if SECTION.match(line):
            in_cs = line.startswith("dump_callstacks.py:")
            flush(); rec = None; hdr = None
            continue
        if not in_cs:
            continue
        rm = RANKM.match(line)
        if rm:
            flush(); rec = None; hdr = None
            rank = int(rm.group(1))
            continue
        hm = HDR.match(line)
        if hm:
            flush()
            hdr = hm.groups()
            rec = line
        elif rec is not None:
            rec += line
    flush()
    return relays


def find_hop(relays, n_stages):
    """Return (sender_stage, recv_stage) at the single BACKED_UP->STARVED leading edge (ring order)."""
    state = {}
    for r, lst in relays.items():
        nR = sum(1 for x in lst if x["state"] == "R")
        nW = sum(1 for x in lst if x["state"] == "W")
        state[r] = "R" if nR > nW else "W" if nW > nR else "."
    edges = [(i, (i + 1) % n_stages) for i in range(n_stages)
             if state.get(i) == "R" and state.get((i + 1) % n_stages) == "W"]
    return edges, state


def load_rankfile(path):
    m = {}
    for line in open(path, errors="replace"):
        rm = re.match(r'rank (\d+)=(\S+)', line)
        if rm:
            m[int(rm.group(1))] = rm.group(2)
    return m


def load_binding(path):
    """rank -> list[int visible devices]; handles quoted (sc16) and unquoted (rev_c) forms."""
    m = {}
    cur = None
    for line in open(path, errors="replace"):
        rm = re.match(r'\s*-\s*rank:\s*(\d+)', line)
        if rm:
            cur = int(rm.group(1))
        dm = re.search(r'TT_VISIBLE_DEVICES:\s*"?([0-9,\s]+?)"?\s*$', line)
        if dm and cur is not None:
            m[cur] = [int(x) for x in dm.group(1).replace(",", " ").split()]
    return m


# Self-contained remote read: shipped over ssh stdin to `python3 -` on the endpoint host, so this
# file has NO dependency on dump_socket_counters.py. Only needs ttexalens (from the tt-metal venv).
_READ_IMPORTS = (
    "import sys\n"
    "from ttexalens.tt_exalens_init import init_ttexalens\n"
    "from ttexalens.tt_exalens_lib import read_words_from_device\n"
    "WORD=4; L1_TOP=0x180000\n"
)
_READ_BODY = r'''
ctx=init_ttexalens(safe_mode=False)
def rw(a,n=1): return read_words_from_device(CORE,a,device_id=CHIP,word_count=n,context=ctx,safe_mode=False)
def window():
    w=[]; a=LO
    while a<HI:
        n=min(8192,(HI-a)//WORD)
        if n<=0: break
        w+=read_words_from_device(CORE,a,device_id=CHIP,word_count=n,context=ctx,safe_mode=False); a+=n*WORD
    return w
def align(v,a): return (v+a-1)//a*a
w=window(); cands=[]
for i in range(len(w)-10):
    base=LO+i*WORD
    if END=="sender":
        num_ds,fifo,is_d2h,dfa=w[i+1],w[i+5],w[i+6],w[i+4]
        if 1<=num_ds<=16 and is_d2h==0 and 0<fifo<=0x200000 and fifo%16==0 and 0x1000<=dfa<L1_TOP and (not PAGE or fifo%PAGE==0):
            cands.append((base,fifo,num_ds))
    else:
        rp,fifo,is_h2d,fa=w[i+1],w[i+3],w[i+5],w[i+2]
        if is_h2d==0 and 0<fifo<=0x200000 and fifo%16==0 and rp<=fifo and 0x1000<=fa<L1_TOP and (not PAGE or fifo%PAGE==0):
            cands.append((base,fifo,rp))
if not cands:
    print("RESULT_ERR end=%s no_socket_md_found" % END); sys.exit(3)
base,fifo,_=cands[0]
if END=="sender":
    S=rw(base+align(28,16))[0]
    print("RESULT end=sender config=0x%x S_acked=%d fifo_total=%d" % (base,S,fifo))
else:
    R=rw(base)[0]
    print("RESULT end=receiver config=0x%x R_sent=%d fifo_total=%d" % (base,R,fifo))
'''


def read_end(host, end, chip, core, page_size, scan_lo=0x150000, scan_hi=0x17FF00):
    """ssh to `host`, ship an embedded ttexalens scan+read to python3 -, return parsed RESULT dict."""
    snippet = (_READ_IMPORTS
               + f"END={end!r}; CHIP={chip}; CORE={core!r}; PAGE={page_size or 0}; LO={scan_lo}; HI={scan_hi}\n"
               + _READ_BODY)
    remote = f"cd {BLAZE_DIR} && source env.sh >/dev/null 2>&1 && python3 -"
    print(f"\n[{end}] ssh {host}: shipping embedded scan+read for chip {chip} core {core} ...")
    out = subprocess.run(["ssh", "-A", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", host, remote],
                         input=snippet, capture_output=True, text=True, timeout=180)
    if out.stdout.strip():
        sys.stdout.write(out.stdout)
    if out.returncode != 0 and out.stderr.strip():
        sys.stderr.write(out.stderr[-1500:])
    m = re.search(r"^RESULT .*$", out.stdout, re.M)
    if not m:
        print(f"  !! no RESULT from {end} read (scan found no socket_md, or ttexalens/ssh error — see above).")
        return None
    return dict(tok.split("=", 1) for tok in m.group(0).split()[1:])


# Eth PHY/link stats are HW/firmware-populated at fixed L1 addresses on each active eth core,
# independent of any telemetry flag. Blackhole addresses (from the tt-metal HAL):
ETH_RETRAIN = 0x7CE00     # uint32
ETH_CORR_CW = 0x7CE90     # uint64 (corrected FEC codewords; nonzero is NORMAL — FEC working)
ETH_UNCORR_CW = 0x7CE98   # uint64 (uncorrected codewords; nonzero == actual data loss on the link)
_ETH_READ_BODY = r'''
from ttexalens.tt_exalens_init import init_ttexalens
from ttexalens.tt_exalens_lib import read_word_from_device, read_words_from_device
ctx=init_ttexalens(safe_mode=False)
dev=ctx.devices[CHIP]
locs=[]
for bt in ("active_eth","eth"):
    try: locs=dev.get_block_locations(bt); break
    except Exception: pass
for loc in locs:
    try:
        rt=read_word_from_device(loc,0x7CE00,device_id=CHIP,context=ctx,safe_mode=False)
        cw=read_words_from_device(loc,0x7CE90,device_id=CHIP,word_count=2,context=ctx,safe_mode=False)
        uw=read_words_from_device(loc,0x7CE98,device_id=CHIP,word_count=2,context=ctx,safe_mode=False)
        try: cs=loc.to_str("noc0")
        except Exception: cs=str(loc)
        print("ETHSTAT chip=%d core=%s retrain=%d corr_cw=%d uncorr_cw=%d" % (
            CHIP, cs, rt, (cw[1]<<32)|cw[0], (uw[1]<<32)|uw[0]))
    except Exception as e:
        pass
'''


def read_eth_stats(host, chip):
    """ssh to `host`, enumerate active eth cores on `chip`, read retrain/corr_cw/uncorr_cw. Returns list of dicts."""
    snippet = f"CHIP={chip}\n" + _ETH_READ_BODY
    remote = f"cd {BLAZE_DIR} && source env.sh >/dev/null 2>&1 && python3 -"
    out = subprocess.run(["ssh", "-A", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", host, remote],
                         input=snippet, capture_output=True, text=True, timeout=180)
    rows = [dict(t.split("=", 1) for t in m.group(0).split()[1:])
            for m in re.finditer(r"^ETHSTAT .*$", out.stdout, re.M)]
    if not rows and out.stderr.strip():
        sys.stderr.write(out.stderr[-800:])
    return rows


def main():
    p = argparse.ArgumentParser(description="Auto-locate + counter-verdict a D2D pipeline-socket hang from a triage dump.")
    p.add_argument("dump")
    p.add_argument("--rank-binding", required=True)
    p.add_argument("--rankfile", required=True)
    p.add_argument("--page-size", type=lambda x: int(x, 0), default=None)
    p.add_argument("--n-stages", type=int, default=64)
    p.add_argument("--dry-run", action="store_true",
                   help="resolve hop + endpoints and print the per-end read commands, but don't ssh/read (offline validation)")
    args = p.parse_args()

    relays = parse_relays(args.dump)
    if not relays:
        print("No d2d_exchange relays found in the dump's dump_callstacks section.")
        return 1
    edges, state = find_hop(relays, args.n_stages)

    print("Per-stage relay state (R=backed-up/reserve, W=starved/wait, .=none):")
    row = ""
    for s in range(args.n_stages):
        row += f"{state.get(s, '.'):<3}{s:<3} "
        if (s + 1) % 8 == 0:
            print("  " + row.rstrip()); row = ""
    if not edges:
        print("\nNo BACKED_UP->STARVED boundary found — pipeline may be healthy/drained or dump partial.")
        return 1
    s_stage, r_stage = edges[0]
    print(f"\n=== WEDGED HOP: stage {s_stage} -> stage {r_stage} ===")
    if len(edges) > 1:
        print(f"  (note: {len(edges)} boundaries found: {edges}; using the first)")

    rankfile = load_rankfile(args.rankfile)
    binding = load_binding(args.rank_binding)

    def endpoint(stage, want):
        cands = [x for x in relays.get(stage, []) if x["state"] == want]
        cands.sort(key=lambda x: (x["risc"] != "brisc",))  # prefer brisc (the socket relay)
        return cands[0] if cands else None

    s_ep = endpoint(s_stage, "R")
    r_ep = endpoint(r_stage, "W")
    if not s_ep or not r_ep:
        print(f"  !! could not find both endpoints (sender R on {s_stage}: {s_ep}, recv W on {r_stage}: {r_ep}).")
        return 1

    def resolve(stage, ep):
        host = rankfile.get(stage)
        devs = binding.get(stage, [])
        chip = devs[ep["dev"]] if 0 <= ep["dev"] < len(devs) else None
        return host, chip

    s_host, s_chip = resolve(s_stage, s_ep)
    r_host, r_chip = resolve(r_stage, r_ep)
    print(f"  sender   (reserve): stage {s_stage} dev {s_ep['dev']} core {s_ep['core']} -> host {s_host} chip {s_chip}")
    print(f"  receiver (wait)   : stage {r_stage} dev {r_ep['dev']} core {r_ep['core']} -> host {r_host} chip {r_chip}")
    if None in (s_host, s_chip, r_host, r_chip):
        print("  !! host/chip resolution failed (check --rankfile / --rank-binding). "
              "You can still run dump_socket_counters.py --scan manually with these dev/core values.")
        return 1

    if args.dry_run:
        print("\n[dry-run] would ship an embedded ttexalens scan+read (no external script) via ssh to:")
        print(f"  sender  : ssh -A {s_host}  -> scan chip {s_chip} core {s_ep['core']} for sender_socket_md -> S_acked")
        print(f"  receiver: ssh -A {r_host}  -> scan chip {r_chip} core {r_ep['core']} for receiver_socket_md -> R_sent")
        print(f"  eth     : ssh -A {s_host} & {r_host} -> read retrain/corr_cw/uncorr_cw on active eth cores of chips {s_chip} & {r_chip}")
        print("  (then D = R_sent - S_acked verdict + eth physical-loss check; run without --dry-run to execute.)")
        return 0

    s_res = read_end(s_host, "sender", s_chip, s_ep["core"], args.page_size)
    r_res = read_end(r_host, "receiver", r_chip, r_ep["core"], args.page_size)
    if not s_res or not r_res:
        return 2

    S_acked = int(s_res["S_acked"]); R_sent = int(r_res["R_sent"])
    fifo = int(s_res.get("fifo_total", 0)) or int(r_res.get("fifo_total", 0))
    D = (R_sent - S_acked) & MASK
    tol = args.page_size or 0
    print("\n=== VERDICT ===")
    print(f"  hop stage {s_stage}->{r_stage}   R_sent={R_sent}  S_acked={S_acked}  D={D}  fifo_total={fifo}")
    if D <= tol:
        print("  LOST FORWARD PAGE/NOTIFY (D≈0): sender ran a full fifo ahead of what the receiver saw.")
    elif fifo and fifo - tol <= D <= fifo + tol:
        print("  LOST BACKWARD CREDIT/ACK (D≈fifo): data reached the receiver but acks never got back to the sender.")
    else:
        print("  AMBIGUOUS: D is neither ≈0 nor ≈fifo (partial loss, wrong page-size, or a stale/mis-scanned read).")

    # ---- eth PHY/link stats on both endpoint chips (physical-drop vs software-race discriminator) ----
    print("\n=== ETH LINK STATS (active eth cores on both endpoint chips) ===")
    eth_bad = False
    for label, host, chip in (("sender", s_host, s_chip), ("receiver", r_host, r_chip)):
        rows = read_eth_stats(host, chip)
        if not rows:
            print(f"  {label} {host} chip {chip}: (no active eth cores read)")
            continue
        for rr in rows:
            u = int(rr.get("uncorr_cw", "0")); rt = int(rr.get("retrain", "0"))
            flag = "   <== UNCORRECTED CODEWORDS = data loss on this link" if u > 0 else ""
            eth_bad = eth_bad or u > 0
            print(f"  {label} {host} chip {chip} core {rr['core']}: retrain={rt} corr_cw={rr.get('corr_cw')} uncorr_cw={u}{flag}")
    print("\n  (all active eth cores on each chip are listed; the 30→31 link is whichever core faces the peer chip.")
    print("   corr_cw>0 is normal FEC; uncorr_cw>0 or climbing retrains = real physical loss.)")
    print("  -> " + ("eth shows DATA LOSS: favors a dropped fabric packet as the desync trigger."
                     if eth_bad else
                     "eth counters CLEAN: favors a software credit/accounting race (no physical loss recorded)."))
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
