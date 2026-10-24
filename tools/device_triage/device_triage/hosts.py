# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host and chip identity.

Hosts are short lowercase names (`bh-glx-120-c08u02`). A chip is named by its
physical position `u<ubb>c<asic>` (UBB tray 1-4, ASIC location 1-8), the same
form kmd_triage prints. Driver ordinals (`device 17`, `/dev/tenstorrent/3`) are
probe order, not position, and are kept as `dev<N>` until a run's BDF list can
resolve them.
"""

import re
from collections.abc import Iterable

from .config import CHIPS_PER_GALAXY, CHIPS_PER_UBB

EXABOX_HOST_RE = re.compile(r"(?<![\w-])(bh-glx-\d{3}-[a-z]\d{2}u\d{2})(?![\w])", re.I)
# Hosts outside the exabox naming that still show up in tickets.
OTHER_HOST_RE = re.compile(r"(?<![\w-])(bh-glx6u-\d+|qb2-\d{3}-[a-z]\d{2}t\d{2}|g\d{2}blx\d{2})(?![\w-])", re.I)
SHORT_HOST_RE = re.compile(r"(?<![\w-])([a-d])(\d{2})u(\d{2})(?![\w])", re.I)
# "120A", "120-A", "110 b", "120-D-Quad4" name a superpod row.
SUPERPOD_ROW_RE = re.compile(r"(?<![\w-])(110|120)[\s-]?([a-d])(?![\w])", re.I)
SUPERPOD_RE = re.compile(r"(?<![\w.-])(?:bh-glx-|superpod\s*|sp\s*)?(110|120)(?![\w.])", re.I)
# "D06_07_08_09u14": one row, several racks, one slot.
QUAD_RE = re.compile(r"(?<![\w])([a-d])(\d{2})((?:_\d{2})+)u(\d{2})(?![\w])", re.I)

BDF_UBB_RE = re.compile(r"(0000:[0-9a-f]{2}:00\.\d)\s*\(UBB(\d)/U(\d)\)", re.I)
BDF_UC_RE = re.compile(r"(0000:[0-9a-f]{2}:00\.\d)\s+u(\d)c(\d)\b", re.I)
BDF_UBB_CHIP_RE = re.compile(r"(0000:[0-9a-f]{2}:00\.\d)\),\s*ubb:\s*(\d),\s*chip:\s*(\d)", re.I)
UC_RE = re.compile(r"(?<![\w])u(\d)c(\d)(?![\w])", re.I)
BDF_RE = re.compile(r"\b(0000:[0-9a-f]{2}:00\.\d)\b", re.I)
ENDPOINT_RE = re.compile(
    r"hostname='([^']+)',\s*tray_id=(\d+),\s*(?:asic_channel=AsicChannel\{asic_location=(\d+),\s*channel_id=(\d+)\}"
    r"|[^}]*?port_type=(\w+),\s*port_id=(\d+))"
)


def normalize_host(name: str | None) -> str | None:
    if not name:
        return None
    host = name.strip().lower().split(":", 1)[0].split(".", 1)[0]
    return host or None


def find_hosts(text: str) -> set[str]:
    return {m.group(1).lower() for m in EXABOX_HOST_RE.finditer(text)} | {
        m.group(1).lower() for m in OTHER_HOST_RE.finditer(text)
    }


def ticket_hosts(text: str, known_hosts: Iterable[str]) -> dict[str, str]:
    """Hosts a ticket talks about -> how they were found (full, quad, short, short-ambiguous).

    Short names (`a02u08`) only count with a superpod in sight ("120A a02u08");
    without one they map to every known host with that suffix, flagged ambiguous.
    """
    known = set(known_hosts)
    found = {h: "full" for h in find_hosts(text)}
    pods = {m.group(1) for m in SUPERPOD_ROW_RE.finditer(text)} | {m.group(1) for m in SUPERPOD_RE.finditer(text)}
    by_suffix: dict[str, list[str]] = {}
    for h in known:
        m = EXABOX_HOST_RE.fullmatch(h)
        if m:
            by_suffix.setdefault(h.rsplit("-", 1)[1], []).append(h)

    def add_short(suffix: str, how: str) -> None:
        candidates = [h for h in by_suffix.get(suffix, []) if not pods or h.split("-")[2] in pods]
        if not candidates and pods:
            candidates = [f"bh-glx-{p}-{suffix}" for p in sorted(pods)]
        tag = how if len(candidates) == 1 else f"{how}-ambiguous"
        for h in candidates:
            found.setdefault(h, tag)

    for m in QUAD_RE.finditer(text):
        row, first, rest, slot = m.group(1).lower(), m.group(2), m.group(3), m.group(4)
        for rack in [first, *rest.strip("_").split("_")]:
            add_short(f"{row}{rack}u{slot}", "quad")
    for m in SHORT_HOST_RE.finditer(text):
        add_short(m.group(0).lower(), "short")
    return found


def chip_label(ubb: int | str, asic: int | str) -> str:
    return f"u{int(ubb)}c{int(asic)}"


def bdf_chip_pairs(text: str) -> list[tuple[str, str]]:
    """(bdf, chip) pairs stated in a details string or a triage table."""
    pairs = [(m.group(1).lower(), chip_label(m.group(2), m.group(3))) for m in BDF_UBB_RE.finditer(text)]
    pairs += [(m.group(1).lower(), chip_label(m.group(2), m.group(3))) for m in BDF_UC_RE.finditer(text)]
    pairs += [(m.group(1).lower(), chip_label(m.group(2), m.group(3))) for m in BDF_UBB_CHIP_RE.finditer(text)]
    return pairs


def endpoint_chips(text: str) -> list[tuple[str, str]]:
    """(host, chip-or-tray) for every PhysicalChannel/PortEndpoint in a validation log line."""
    out = []
    for m in ENDPOINT_RE.finditer(text):
        host = normalize_host(m.group(1))
        tray = m.group(2)
        out.append((host, chip_label(tray, m.group(3)) if m.group(3) else f"u{int(tray)}"))
    return out


def scope_of(chips: Iterable[str]) -> str:
    """chip | tray | host for a set of affected chips (`u4c1`) or trays (`u4`)."""
    chips = set(chips)
    if not chips:
        return "unknown"
    if len(chips) >= CHIPS_PER_GALAXY - 2:
        return "host"
    trays = {c.split("c")[0] for c in chips}
    if len(trays) == 1 and (len(chips) >= CHIPS_PER_UBB - 2 or all("c" not in c for c in chips)):
        return "tray"
    return "chip"


def summarize_chips(weights: dict[str, float], keep: float = 0.4) -> str:
    """Strongest chips as text; a tray counts when its label or most of its chips carry weight."""
    if not weights:
        return ""
    top = max(weights.values())
    trays: dict[str, float] = {}
    for label, w in weights.items():
        m = re.fullmatch(r"(u\d)(?:c\d)?", label)
        if m:
            trays[m.group(1)] = trays.get(m.group(1), 0.0) + (w if label == m.group(1) else w / CHIPS_PER_UBB)
    shown_trays = sorted(
        t
        for t, w in trays.items()
        if weights.get(t, 0) >= keep * top
        or sum(1 for c in weights if c.startswith(t + "c") and weights[c] >= keep * top) >= 6
    )
    if len(shown_trays) == 4:
        return "all"
    chips = sorted(
        c
        for c, w in weights.items()
        if w >= keep * top
        and c not in shown_trays
        and not any(c.startswith(t + "c") for t in shown_trays)
        and not re.fullmatch(r"u\d", c)
    )
    return ",".join(shown_trays + chips[:8]) + (f",+{len(chips) - 8}" if len(chips) > 8 else "")


def compress_chips(chips: Iterable[str]) -> str:
    """Render a chip set compactly: whole trays as `u4`, all 32 as `all`."""
    chips = set(chips)
    if len(chips) >= CHIPS_PER_GALAXY:
        return "all"
    trays: dict[str, list[str]] = {}
    for c in chips:
        tray, _, asic = c.partition("c")
        trays.setdefault(tray, []).append(asic)
    parts = []
    for tray in sorted(trays):
        asics = trays[tray]
        if "" in asics or len(asics) >= CHIPS_PER_UBB:
            parts.append(tray)
        else:
            parts.extend(f"{tray}c{a}" for a in sorted(asics, key=int))
    return ",".join(parts)


TICKET_CHIP_PATTERNS = (
    (re.compile(r"(?<![\w])u(\d)c(\d)(?![\w])", re.I), lambda m: chip_label(m.group(1), m.group(2))),
    (re.compile(r"\bUBB\s*#?(\d)\s*/\s*U(\d)\b", re.I), lambda m: chip_label(m.group(1), m.group(2))),
    (re.compile(r"\b(?:UBB|tray)\s*#?\s*(\d)\b", re.I), lambda m: f"u{m.group(1)}"),
    (re.compile(r"\b(?:device|dev|pci)\s*[#:]?\s*(\d{1,2})\b", re.I), lambda m: f"dev{int(m.group(1))}"),
    (re.compile(r"\bchip\s*#?\s*(\d{1,2})\b", re.I), lambda m: f"chip{int(m.group(1))}"),
    (re.compile(r"\basic\s*#?\s*(\d{1,2})\b", re.I), lambda m: f"asic{int(m.group(1))}"),
)


def ticket_chip_refs(text: str) -> set[str]:
    refs = set()
    for pattern, label in TICKET_CHIP_PATTERNS:
        refs.update(label(m) for m in pattern.finditer(text))
    return refs
