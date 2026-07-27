# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The ONE device-recovery primitive. Every reset in this tool goes through `recover()`.

Four modules had grown their own reset path -- perf_mcp (MCP server), run.py (orchestrator
watchdog), probes (profiler layer) and cli (planner). They shared no code, so each independently
re-derived the same three decisions, and each got the same three wrong:

  WHICH board?  Every path inferred the target from the `--devices` flag: `single` -> chip 0 ->
                board 0,1. That flag is INTENT (use one chip), not PLACEMENT, and the runtime is
                free to place the mesh anywhere visible. On 2026-07-27 it had placed it on chip 3,
                so every reset for eleven hours hit a healthy board while the error text repeated
                "Read 0xffffffff over PCIe ID 3". The evidence was in the failure the whole time.

  WHETHER?      Recovery was gated behind counters held in module globals -- in the MCP server,
                which the client kills whenever a call runs long, and a wedged device is exactly
                what makes a call run long. A counter that zeroes on restart never reaches its
                threshold, so the two-strike gate never fired once.

  DID IT WORK?  Resets returned None, or a status string nobody parsed, or an exit code. Nothing
                asked the device whether it had actually come back, so a failed reset was
                indistinguishable from a successful one and the loop retried forever.

Only ONE thing legitimately differs per caller: how the reset is ISSUED (run.py's board-aware
`_reset_devices`, probes' galaxy-aware arg sets, a plain `tt-smi -r`). So `recover()` takes that as
a callable and owns everything else -- target order, verification, durable counters, escalation.
Adding a fifth caller means passing a reset function, not re-deciding any of the above.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

TT_SMI = shutil.which("tt-smi") or "/home/ttuser/.tenstorrent-venv/bin/tt-smi"

DEAD_BOARD_SIGS = ("0xffffffff", "board should be reset", "pcie link", "device hang", "hang detected")

HEALTH_TIMEOUT_S = float(os.environ.get("TT_RECOVERY_HEALTH_TIMEOUT_S", "45") or "45")
RESET_FAIL_LIMIT = int(
    os.environ.get("TT_RECOVERY_FAIL_LIMIT", os.environ.get("PERF_MCP_RESET_FAIL_LIMIT", "3")) or "3"
)


def is_dead_board(text) -> bool:
    """Is this the UNAMBIGUOUS 'the card stopped answering' signature?

    A PCIe read of 0xffffffff is all-ones: the bus reporting that nobody replied. There is nothing
    to disambiguate, so waiting for a second occurrence before acting only guarantees being down
    twice. Counters belong on flaky symptoms, not definitive ones.
    """
    s = (str(text) or "").lower()
    return any(sig in s for sig in DEAD_BOARD_SIGS)


def dead_chip_from_error(text):
    """THE EVIDENCE: the chip id the runtime named in the failure, or None.

    tt-metal reports "Read 0xffffffff over PCIe ID 3" -- it says which chip died. Read the id
    rather than infer it from a flag that describes intent.
    """
    m = re.search(r"pcie\s*(?:id|device)?\s*[:#]?\s*(\d+)", str(text or ""), re.I)
    if m:
        try:
            return int(m.group(1))
        except (TypeError, ValueError):
            return None
    return None


def device_is_healthy(timeout_s: float = HEALTH_TIMEOUT_S) -> bool:
    """Does the device answer at all? BOUNDED, because tt-smi hangs on a dead card -- an unbounded
    (or 420 s) health probe is what let the MCP client kill the server mid-reset, which erased the
    crash counter and stopped recovery ever firing."""
    try:
        r = subprocess.run([TT_SMI, "-s"], capture_output=True, text=True, timeout=timeout_s)
        return r.returncode == 0 and "board_type" in (r.stdout or "")
    except Exception:  # noqa: BLE001
        return False


def state_path() -> Path:
    """Where the durable counters live, KEYED per run so two concurrent optimize runs do not share
    a crash history."""
    override = os.environ.get("TT_RECOVERY_STATE")
    if override:
        return Path(override)
    model = os.environ.get("PERF_MCP_MODEL") or os.environ.get("TT_HW_PLANNER_MODEL") or "model"
    task = os.environ.get("PERF_MCP_TASK", "main")
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", "%s_%s" % (model, task))
    return Path(tempfile.gettempdir()) / ("tt_device_recovery_%s.json" % safe)


class Counter:
    """A counter that OUTLIVES THE PROCESS COUNTING IT.

    Durability is the whole point: these gate recovery, and the process holding them is the one the
    device fault gets killed in. Keeps the ``["n"]`` mapping interface of the module-level dicts it
    replaces, so existing call sites read unchanged.
    """

    __slots__ = ("_field",)

    def __init__(self, field: str):
        self._field = field

    def _load(self) -> dict:
        try:
            return json.loads(state_path().read_text())
        except Exception:  # noqa: BLE001
            return {}

    def __getitem__(self, key: str) -> int:
        try:
            return int(self._load().get(self._field, 0))
        except (TypeError, ValueError):
            return 0

    def __setitem__(self, key: str, value: int) -> None:
        state = self._load()
        state[self._field] = int(value)
        try:
            p = state_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(state))
        except Exception:  # noqa: BLE001
            pass


CONSEC_CRASH = Counter("consec_crash")
RESET_FAILS = Counter("reset_fails")


BOARD_MAP_FILE = Path(tempfile.gettempdir()) / "perf_mcp_board_topology.json"


def read_board_topology():
    """Live-read chip-index -> its board's PCI-resettable chips from tt-smi -s.

    Chips sharing a board_id are one board; a WHOLE board is reset by resetting every chip on it
    that has its own PCI bus_id. An n300's remote chip has no bus_id, so its board is {local}; a
    p300c's two ASICs are each PCIe endpoints, so its board is BOTH -- resetting only one half-resets
    the board and breaks enumeration. Returns {str(chip): [resettable chips]} or None.
    """
    try:
        r = subprocess.run([TT_SMI, "-s"], capture_output=True, text=True, timeout=120)
        di = (json.loads(r.stdout) or {}).get("device_info") or []
    except Exception:  # noqa: BLE001
        return None
    board_of = {}
    resettable_of_board = {}
    for i, dev in enumerate(di):
        bi = dev.get("board_info") or {}
        bid = bi.get("board_id")
        board_of[i] = bid
        bus = bi.get("bus_id")
        if bid is not None and bus and bus != "N/A":
            resettable_of_board.setdefault(bid, []).append(i)
    m = {str(i): sorted(resettable_of_board.get(board_of.get(i), [])) for i in board_of}
    m = {k: v for k, v in m.items() if v}
    return m or None


def board_map():
    """The reset map, preferring the copy captured while the board was HEALTHY -- a live read of a
    wedged card returns nothing, which is exactly when the map is needed."""
    try:
        m = json.loads(BOARD_MAP_FILE.read_text())
        if m:
            return m
    except Exception:  # noqa: BLE001
        pass
    return read_board_topology()


def expand_to_boards(chip_ids):
    """Chip ids -> the whole board(s) they live on, as a comma list; None if no topology is known.

    THE DEFAULT, not an opt-in. Reading the right chip out of an error only helps if the reset that
    follows covers its whole board, and a caller that forgets to expand issues a half-board reset --
    which wedges device-open. Making every caller remember is how the original defect spread across
    four modules in the first place.
    """
    m = board_map()
    if not m:
        return None
    targets = set()
    for c in chip_ids:
        v = m.get(str(c))
        if isinstance(v, list):
            targets.update(int(x) for x in v)
        elif v is not None:
            targets.add(int(v))
    return ",".join(str(x) for x in sorted(targets)) if targets else None


def expand_spec(spec):
    """Widen a device spec to WHOLE BOARDS, or None when that cannot be guaranteed.

    None means "do not reset this target" -- the caller falls through to the next one, and the last
    one is always every board. A reset that covers too much is recoverable; a reset that covers half
    a p300c leaves the untouched ASIC's clock arbiter inconsistent and wedges device-open, so an
    unverifiable target must widen, never narrow.
    """
    d = (spec or "").strip().lower()
    if d in ("", "all"):
        return "all"
    if d == "single":
        ids = [0]
    else:
        parts = [x.strip() for x in d.split(",") if x.strip()]
        if not all(x.isdigit() for x in parts):
            return None
        ids = [int(x) for x in parts]
    return expand_to_boards(ids)


def targets_for(error_text: str = "", config_target: str = "", expand=None) -> list:
    """The reset targets to try, in order -- evidence first, the config guess retained as the
    fallback it always should have been:

        1. the chip id named in the error   (observed)
        2. the --devices-derived board      (inferred; kept, but no longer the only source)
        3. every board                      (last resort)

    EVERY target is a whole board or every board. A bare chip id is never emitted, even though the
    error names one: a single-chip `-r 3` half-resets a p300c and wedges device-open. The earlier
    version fell back to the bare chip when the topology was unknown -- which is the common case
    during a real recovery, because the topology is live-read from the card that just died. When a
    target cannot be widened with confidence it is DROPPED, and the caller falls through to `all`.
    """
    out = []
    chip = dead_chip_from_error(error_text)
    if chip is not None:
        try:
            board = (expand or expand_to_boards)([chip])
        except Exception:  # noqa: BLE001
            board = None
        if board and str(board) not in out:
            out.append(str(board))
    cfg = expand_spec(config_target)
    if cfg and cfg not in out:
        out.append(cfg)
    if "all" not in out:
        out.append("all")
    return out


def recover(where: str, reset, error_text: str = "", config_target: str = "", log=None, expand=None) -> bool:
    """Reset the device and REPORT WHETHER IT CAME BACK. True only on a VERIFIED-healthy device.

    ``reset`` is a callable taking the target spec -- the only per-caller part. Everything else
    (which board, how many tries, whether it worked, when to give up) is decided here so no caller
    can decide it differently.
    """
    targets = targets_for(error_text, config_target, expand=expand)
    for tgt in targets:
        try:
            reset(tgt)
        except Exception as exc:  # noqa: BLE001
            if log:
                log("reset raised at %s (target=%s): %s" % (where, tgt, exc))
            continue
        if device_is_healthy():
            RESET_FAILS["n"] = 0
            if log:
                log("device recovered at %s via target=%s" % (where, tgt))
            return True
    RESET_FAILS["n"] = RESET_FAILS["n"] + 1
    if log:
        log(
            "RESET FAILED at %s (attempt %d/%d); targets tried: %s"
            % (where, RESET_FAILS["n"], RESET_FAIL_LIMIT, ", ".join(targets))
        )
    return False


def recovery_exhausted() -> bool:
    """Have resets failed enough times that the run should STOP rather than poll a dead board?"""
    return RESET_FAILS["n"] >= RESET_FAIL_LIMIT


def note_crash(where: str, reset, error_text: str = "", config_target: str = "", log=None, expand=None) -> bool:
    """Record a crash and recover when the evidence justifies it.

    A dead-board signature recovers IMMEDIATELY -- it is unambiguous. Ambiguous crashes keep the
    two-strike counter, because a single odd failure is not evidence of a wedge. The counter clears
    only on a verified-healthy device, so a failed reset never looks like a successful one.
    """
    if is_dead_board(error_text):
        ok = recover(where, reset, error_text, config_target, log, expand)
        if ok:
            CONSEC_CRASH["n"] = 0
        return ok
    CONSEC_CRASH["n"] = CONSEC_CRASH["n"] + 1
    if CONSEC_CRASH["n"] >= 2:
        ok = recover(where, reset, error_text, config_target, log, expand)
        if ok:
            CONSEC_CRASH["n"] = 0
        return ok
    return False


def note_ok() -> None:
    CONSEC_CRASH["n"] = 0
