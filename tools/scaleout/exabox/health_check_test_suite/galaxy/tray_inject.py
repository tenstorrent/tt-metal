# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Test-only fault injection for timing the numeric and memory checks.

Off unless TRAY_INJECT is set; run.sh, every mode and the bundle never set it. Only the checks' compare functions read
it, through want(): with TRAY_INJECT unset want() returns None and every compare runs on the data the device returned,
unchanged, so no check is skipped or shortened. When it is set, run.sh prints
'TRAY_INJECT active: not a verdict' and the run's verdict is INCOMPLETE, never PASS.

TRAY_INJECT holds one or more specs separated by spaces or ';', each a comma list of key=value fields:
    check=<G3|G3b|N1|N2|N2m|N3|N4|N5|N7|DIDT|mix|C1|all>,bdf=<BDF>[,core=<x>-<y>][,after=<k>][,bit=<b>]
  check  which compare function corrupts its read-back data (all: every one; C1: the SPI image checksum computed
         from the chip's boot-fs read, fleet/fw_reads.py; N2m: the semaphore count
         N2's multicast leg recorded, noc_a2a.py).
  bdf    the target chip ('05:00.0' or '0000:05:00.0').
  core   the logical core x-y to corrupt where the check judges per core (default: the check's first core).
  after  corrupt only from the check's k-th compare on this chip (0-based, default 0): a fault that appears mid-run.
  bit    which bit of the element to flip (default: the second-highest bit, a gross error any compare must catch;
         bit=0 is a 1-ulp error, to probe a tolerance).
The corruption is applied to the host-side copy of the read-back, so it changes nothing on the chip. Each first hit per
check and chip prints one line to stderr, 'TRAY_INJECT applied: check=.. bdf=.. core=.. n=.. t=<epoch s>', the start of the
detection time measured to the check's first FAIL line naming that BDF, tray and core."""
import os
import sys
import time

CHECKS = ("G3", "G3b", "N1", "N2", "N2m", "N3", "N4", "N5", "N7", "DIDT", "mix", "C1")
_SEEN = set()
_N = {}


def _bdf(b):
    b = str(b or "").strip().lower()
    return b if b.count(":") >= 2 else "0000:" + b


def parse(text):
    """Specs from a TRAY_INJECT value; a bad field raises ValueError so a typo never runs as a clean run."""
    out = []
    for tok in (text or "").replace(";", " ").split():
        d = {}
        for f in tok.split(","):
            if "=" not in f:
                raise ValueError(f"TRAY_INJECT: field {f!r} is not key=value")
            k, v = f.split("=", 1)
            d[k.strip()] = v.strip()
        bad = set(d) - {"check", "bdf", "core", "after", "bit"}
        if bad:
            raise ValueError(f"TRAY_INJECT: unknown field(s) {sorted(bad)}")
        c = d.get("check", "")
        if c != "all" and c not in CHECKS:
            raise ValueError(f"TRAY_INJECT: check {c!r} is not one of {', '.join(CHECKS)} or all")
        if not d.get("bdf"):
            raise ValueError("TRAY_INJECT: bdf= is required")
        s = dict(
            check=c,
            bdf=_bdf(d["bdf"]),
            core=None,
            after=int(d.get("after", 0) or 0),
            bit=int(d["bit"]) if d.get("bit", "") != "" else None,
        )
        if d.get("core"):
            x, y = d["core"].replace(",", "-").split("-")
            s["core"] = (int(x), int(y))
        out.append(s)
    return out


def specs():
    return parse(os.environ.get("TRAY_INJECT", ""))


def active():
    return bool(os.environ.get("TRAY_INJECT", "").strip())


def want(check, bdf, n=None):
    """The spec that makes this compare call (check, chip bdf) corrupt its read-back, or None. n is the check's compare
    count on this chip (0-based); without one, want() counts calls per (check, chip) itself."""
    if not active():
        return None
    b = _bdf(bdf)
    if n is None:
        n = _N.get((check, b), 0)
        _N[(check, b)] = n + 1
    for s in specs():
        if s["check"] in (check, "all") and s["bdf"] == b and n >= s["after"]:
            return dict(s, n=n, check=check)
    return None


def applied(s, core=None, where=""):
    """Prints the one 'TRAY_INJECT applied' line per (check, chip): the time detection is measured from."""
    k = (s["check"], s["bdf"])
    if k in _SEEN:
        return
    _SEEN.add(k)
    c = core if core is not None else s.get("core")
    cs = f" core={c[0]}-{c[1]}" if c else ""
    print(
        f"TRAY_INJECT applied: check={s['check']} bdf={s['bdf']}{cs}{' ' + where if where else ''} n={s['n']} "
        f"t={time.time():.1f}",
        file=sys.stderr,
        flush=True,
    )


def _bit(s, nbits):
    b = s.get("bit")
    return (nbits - 2) if b is None else int(b) % nbits


def flip_int(s, v, nbits=32):
    """v with the spec's bit flipped (a word, a CRC, a count)."""
    return int(v) ^ (1 << _bit(s, nbits))


def flip_bytes(s, data, off=0):
    """A copy of bytes/bytearray with one bit of byte off flipped."""
    b = bytearray(data)
    b[off % len(b)] ^= 1 << _bit(s, 8)
    return bytes(b) if isinstance(data, bytes) else b


def flip_tensor(s, t, idx=0):
    """A copy of torch tensor t with one bit flipped in the raw bits of flat element idx (any dtype)."""
    import torch

    t = t.detach().clone().contiguous()
    flat = t.reshape(-1)
    nb = flat.element_size() * 8
    raw = flat.view({8: torch.uint8, 16: torch.int16, 32: torch.int32, 64: torch.int64}[nb])
    i = idx % raw.numel()
    raw[i] = raw[i] ^ torch.tensor(1 << _bit(s, nb), dtype=torch.int64).to(raw.dtype)
    return t


def flip_np(s, a, idx=0):
    """A copy of numpy array a with one bit flipped in flat element idx."""
    a = a.copy()
    flat = a.reshape(-1)
    raw = flat.view({1: "u1", 2: "u2", 4: "u4", 8: "u8"}[flat.itemsize])
    i = idx % raw.size
    raw[i] ^= raw.dtype.type(1 << _bit(s, flat.itemsize * 8))
    return a


if __name__ == "__main__":  # run.sh: validate TRAY_INJECT before any device step (exit 2 and the reason on a bad spec)
    try:
        for x in specs():
            print(
                f"TRAY_INJECT spec: check={x['check']} bdf={x['bdf']} core={x['core']} after={x['after']} bit={x['bit']}"
            )
    except ValueError as e:
        print(e)
        sys.exit(2)
