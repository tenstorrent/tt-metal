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
import subprocess
import time
from pathlib import Path

# ONE state directory for every durable temp artifact -- see cc_optimize/tmpstate.py.
# agent/state_dir.py loads cc_optimize/tmpstate.py by path, once, for the four modules that need it.
from .state_dir import state_dir


from .probes import tt_smi_bin as _tt_smi_bin

# Resolved at import for a module-level constant, but through the ONE resolver -- see
# agent.probes.tt_smi_bin for why a bare "tt-smi" is not safe from a non-interactive launch.
TT_SMI = _tt_smi_bin()

DEAD_BOARD_SIGS = (
    "0xffffffff",
    "board should be reset",
    "pcie link",
    "device hang",
    "hang detected",
    # UMD's generic device-init failure (tt_device_error.cpp): "Firmware startup error on device N
    # at core X over NOC0: scratch_status=..., postcode=...". The card did not come up at all, which
    # is as definitive as an all-ones read -- and it is what a wedged Wormhole board on this host
    # actually printed, while matching none of the signatures above. Arch-independent: the same UMD
    # template serves every device.
    "firmware startup error",
    # A multi-chip ETH fabric that stopped advancing: "Timed out waiting for ETH heartbeat on
    # device ASIC ID: N, ETH core eX-Y (NOC0) to advance. Stuck at 0x...". run.py already calls
    # this a "heartbeat-stuck wedge" when it warns that resetting chip 0 alone can CAUSE one, so
    # the condition is known to the tool -- it just was not on this list, and a board in it fails
    # every retry identically until someone resets it by hand.
    "eth heartbeat",
    # UMD's NOC hang check (tt_device_error.cpp, NocHangError): "NOC0 is hung on PCIe device ID 9."
    # Raised at device-open while the NOC is unreachable, so every retry fails the same way within
    # seconds. Seen 2026-09-25 on a WH Galaxy after an e2e run hung mid-collective: no signature
    # matched, the temperature veto cancelled every reset (the ARC was still publishing), and each
    # emit-e2e round failed on the same chip until it was stopped by hand.
    "is hung on pcie device",
)

# THE KERNEL'S VERDICT that a reset cannot help. `tt-smi -r` talks to the card OVER PCIe and asks its
# board-management firmware to cycle power; when that firmware is the thing refusing, the request has
# nowhere to land. The driver logs exactly that, and it is the only signal available that separates
# "wedged, reset it" from "wedged, the host must reboot":
#
#   tenstorrent tenstorrent!2: Failed to set initial power state: -22
#
# Observed on this box 2026-08-05 13:36 through 2026-08-06 12:07 -- 714 occurrences, one per open
# attempt, spanning a reboot-less day in which 34 resets changed nothing. Nothing else in the kernel
# log marked it: no thermal trip, no PCIe AER, no OOM, no hung task.
UNRESETTABLE_KERNEL_SIGS = ("failed to set initial power state",)
KERNEL_LOG_LINES = int(os.environ.get("TT_RECOVERY_DMESG_LINES", "400") or "400")


def _kernel_tail() -> str:
    """The recent kernel log, or "" when it cannot be read.

    BOUNDED and best-effort: recovery must never depend on dmesg being readable, and a host that
    restricts it (dmesg_restrict=1, no journal) simply falls back to counting failures.
    """
    # THIS BOOT ONLY. `journalctl -k` without -b returns the last N kernel lines across EVERY boot,
    # so a fault from a previous boot is read as a live one. Observed on this box: chips 2 and 3 died,
    # the host was rebooted, all four came back healthy at 1e52 -- and board_needs_host_reboot() still
    # answered True, because yesterday's 714 "Failed to set initial power state" lines were still in
    # the journal. That refuses recovery on a working board, which is worse than the unbounded
    # retrying this check was added to stop. dmesg is inherently current-boot; journalctl needs -b.
    for cmd in (["dmesg", "--ctime"], ["journalctl", "-k", "-b", "--no-pager", "-n", str(KERNEL_LOG_LINES)]):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if r.returncode == 0 and r.stdout:
                return "\n".join((r.stdout or "").splitlines()[-KERNEL_LOG_LINES:])
        except Exception:  # noqa: BLE001
            continue
    return ""


def board_needs_host_reboot(kernel_text: str = None) -> bool:
    """After resets have FAILED, does the kernel say why -- a fault no PCIe reset can clear?

    Read only as an explanation, never as a gate. The message fires whenever a device is OPENED while
    its ARC is not ready, so it is transient by nature: a wedged board produced 714 of them across a
    day on this box, and a healthy run produced 4 in an hour while continuing to optimize normally.
    Gating on it would declare a working board dead at the first fault.

    So the count decides WHETHER to stop, and this decides WHAT TO TELL THE OPERATOR -- "reboot the
    host", which is actionable, instead of "unrecoverable after N attempts", which is not. Run 39 sat
    dead until morning for want of exactly that sentence.
    """
    txt = (_kernel_tail() if kernel_text is None else str(kernel_text or "")).lower()
    return any(sig in txt for sig in UNRESETTABLE_KERNEL_SIGS)


def stale_precompiled_firmware(tt_metal_root) -> str | None:
    """Is the pre-compiled firmware OLDER than the tt-metal library that will load it?

    THE ONE CHECK THAT WOULD HAVE SAVED 2026-09-22. Every device open failed with "Device 0 init:
    failed to initialize FW! Try resetting the board", then NOC0 hung and the ARC dropped off the
    bus -- the exact shape of a wedged board. Resets, a warm host reboot and a day were spent on it.
    The board was fine: profiler_heal had relinked libtt_metal.so after an upstream sync changed the
    Blackhole L1 routing-table layout, while tt_metal/pre-compiled/ (which the runtime prefers over a
    JIT build) still held the previous day's firmware. Old-layout firmware under a new-layout runtime
    never reports FW init done.

    Pure mtime comparison, no device access, so it is safe to call from any verdict path. Returns an
    actionable sentence when the pre-compiled tree is older than the newest libtt_metal.so under
    build*/; None when there is nothing to compare or nothing is stale.
    """
    try:
        root = Path(str(tt_metal_root))
        pre = root / "tt_metal" / "pre-compiled"
        if not pre.is_dir():
            return None
        elfs = list(pre.rglob("*.elf"))
        if not elfs:
            return None
        fw_newest = max(f.stat().st_mtime for f in elfs)
        libs = [q for q in root.glob("build*/tt_metal/libtt_metal.so")] + [
            q for q in root.glob("build*/lib/libtt_metal.so")
        ]
        if not libs:
            return None
        lib = max(libs, key=lambda q: q.stat().st_mtime)
        if lib.stat().st_mtime <= fw_newest:
            return None
        return (
            "STALE PRE-COMPILED FIRMWARE, not a board fault: %s is newer than every firmware image in "
            "%s. The runtime loads that firmware in preference to a JIT build, and firmware built "
            "against a different L1 layout never finishes FW init (all cores time out, NOC0 hangs, "
            "the ARC drops off the bus). Fix: `ninja -C %s precompile-fw` (or remove %s), then "
            "`tt-smi -r` the chips and re-run." % (lib, pre, lib.parent.parent, pre)
        )
    except Exception:  # noqa: BLE001
        return None


HEALTH_TIMEOUT_S = float(os.environ.get("TT_RECOVERY_HEALTH_TIMEOUT_S", "45") or "45")
RESET_FAIL_LIMIT = int(
    os.environ.get("TT_RECOVERY_FAIL_LIMIT", os.environ.get("PERF_MCP_RESET_FAIL_LIMIT", "3")) or "3"
)


# WHERE THE FAILURE HAPPENED, FOR WHEN NOTHING IT SAYS IS RECOGNISED.
#
# DEAD_BOARD_SIGS is an allowlist of PROSE, and the runtime keeps inventing new prose. Four entries
# above were each added by hand after a run had already been lost to the wording they match, and the
# list was still one short every time:
#
#   "Timed out waiting for ETH heartbeat ... Stuck at 0xaabb0024"   -> added by hand
#   "Firmware startup error on device N at core X over NOC0"        -> added by hand
#   "NOC0 is hung on PCIe device ID 9."                             -> added by hand
#   "Timeout waiting for Ethernet core service remote IO request."  -> NOT matched; 2026-09-25 the
#       profiler died on it 18 times in a row, each attempt reporting "no ops_perf_results_*.csv",
#       no reset was ever issued because this returned False, and the run spent its whole budget
#       recording nothing.
#
# An allowlist cannot be completed by adding to it, so this asks a question that does not depend on
# the wording: WHERE did the failure happen. A test whose SETUP raised a hard runtime fault never
# reached its body -- and setup, for every test this tool runs, is bringing the device up. A board
# that cannot be opened is dead whatever the runtime chose to call it.
#
# Deliberately conservative in three ways: it needs pytest's own setup-error report (so an ordinary
# failure inside a test body can never reach it), it needs a HARD fault class (an ImportError or a
# missing-file error at setup is a host problem and stays unmatched), and it only ever ADDS to the
# allowlist -- every signature that matched before still matches first, unchanged.
_SETUP_FAILURE_REPORT = "error at setup of"  # pytest's own report line; not our vocabulary
_HARD_FAULT_RE = re.compile(r"\b(RuntimeError|TT_FATAL|TT_THROW|TT_ASSERT)\b", re.IGNORECASE)


def is_device_bringup_failure(text) -> bool:
    """Did this failure happen while the DEVICE was being brought up, whatever it was called?

    Structural, not lexical: pytest reports a fixture failure as "ERROR at setup of <node>", and
    the only thing a perf/PCC test does in setup is open the mesh. A hard runtime fault there means
    the device did not come up, so the board needs recovery before any retry can differ.
    """
    s = str(text) or ""
    if _SETUP_FAILURE_REPORT not in s.lower():
        return False
    return bool(_HARD_FAULT_RE.search(s))


def is_dead_board(text) -> bool:
    """Is this the UNAMBIGUOUS 'the card stopped answering' signature?

    A PCIe read of 0xffffffff is all-ones: the bus reporting that nobody replied. There is nothing
    to disambiguate, so waiting for a second occurrence before acting only guarantees being down
    twice. Counters belong on flaky symptoms, not definitive ones.

    A recognised signature answers first; a failure at device BRING-UP answers when none does --
    see is_device_bringup_failure for why an allowlist of prose cannot be completed by extending it.
    """
    s = (str(text) or "").lower()
    if any(sig in s for sig in DEAD_BOARD_SIGS):
        return True
    return is_device_bringup_failure(text)


def dead_chip_from_error(text):
    """THE EVIDENCE: the chip id the runtime named in the failure, or None.

    tt-metal reports "Read 0xffffffff over PCIe ID 3" -- it says which chip died. Read the id
    rather than infer it from a flag that describes intent. UMD's NOC hang says "on PCIe device
    ID 9", so "device" and "id" may both appear.
    """
    m = re.search(r"pcie\s*(?:device\s*)?(?:id)?\s*[:#]?\s*(\d+)", str(text or ""), re.I)
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
    return state_dir() / ("tt_device_recovery_%s.json" % safe)


def _run_stamp() -> str:
    """Which run these counters belong to, or "" when nothing said.

    THE LIFETIME MATTERS AS MUCH AS THE VALUE. state_path() keys the file by (model, task), which
    outlives the run -- so "resets have stopped working" was inherited by every later run on that
    model. Harmless while nothing read the count; once recover() began REFUSING at the limit it became
    a latch, and run 39's dead board left reset_fails=34 in a file that survived the board being
    fixed, a host reboot, and a fresh run on healthy hardware -- which then halted before its first
    round with all four chips idling at 45C.

    "Resets are not working" is a statement about THIS run against THIS board. A new run re-establishes
    it in three attempts if it is still true. board_needs_host_reboot then explains the failure --
    "reboot the host" rather than "unrecoverable" -- but it does not gate, because the message it
    reads is transient and would condemn a working board.
    """
    return str(os.environ.get("PERF_MCP_RUN_ID") or "").strip()


def stamp_run() -> str:
    """One id for this run, set once and inherited by every child. Every entry point that can reset
    a device calls this before its first device work.

    The recovery counters are scoped to it: "resets have stopped working" is a fact about THIS run
    against THIS board, and carrying it into the next run is what turned a limit into a latch (run 39
    left reset_fails=34 in a (model, task)-keyed file that survived the board being fixed and a host
    reboot). Only optimize used to stamp its run, so every other stage counted under the empty stamp:
    on 2026-09-25 an emit-e2e run's three failed resets left reset_fails=3 there, and every later
    emit-e2e refused to reset at all. Never overwritten, so a supervisor restart does not silently get
    a fresh budget.
    """
    cur = _run_stamp()
    if not cur:
        cur = "%d_%d" % (int(time.time()), os.getpid())
        os.environ["PERF_MCP_RUN_ID"] = cur
    return cur


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
            d = json.loads(state_path().read_text())
        except Exception:  # noqa: BLE001
            return {}
        # A COUNT FROM ANOTHER RUN IS NOT EVIDENCE ABOUT THIS ONE. Rather than deleting the file --
        # which would lose the record a post-mortem reads -- a stamp mismatch simply reads as zero.
        if isinstance(d, dict) and str(d.get("run") or "") != _run_stamp():
            return {}
        return d if isinstance(d, dict) else {}

    def __getitem__(self, key: str) -> int:
        try:
            return int(self._load().get(self._field, 0))
        except (TypeError, ValueError):
            return 0

    def __setitem__(self, key: str, value: int) -> None:
        state = self._load()
        state["run"] = _run_stamp()
        state[self._field] = int(value)
        try:
            p = state_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(state))
        except Exception:  # noqa: BLE001
            pass


CONSEC_CRASH = Counter("consec_crash")
RESET_FAILS = Counter("reset_fails")


def board_map_file():
    """Resolved per call: a module constant freezes the path at import, before any redirect."""
    return state_dir() / "perf_mcp_board_topology.json"


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
        m = json.loads(board_map_file().read_text())
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


def _protected_pids() -> set:
    """This process and every ancestor -- the one set a reclaim must never kill.

    Killing an ancestor kills the orchestrator or the supervisor that would do the recovering, so a
    self-hold is handled by exiting to the supervisor (which reclaims from outside), never by the
    holder shooting itself. Walks /proc rather than psutil so it works on a bare host, and is bounded
    at 64 hops so a malformed /proc cannot spin it.
    """
    protected, pid = set(), os.getpid()
    for _ in range(64):
        if pid <= 1:
            break
        protected.add(pid)
        try:
            pid = int(open("/proc/%d/stat" % pid).read().split()[3])
        except Exception:  # noqa: BLE001
            break
    return protected


def _live_temps() -> list:
    """Plausible per-chip die temperatures, straight from the driver. [] when none answer.

    Bounds-checked at the source: an ARC that is not running publishes all-ones (65535999 = 65535.999
    C), which is a successful read of a value that is not a temperature, so the VALUE is the only
    thing that separates a reading from silence."""
    try:
        from .probes import board_telemetry

        return list(board_telemetry()[0] or [])
    except Exception:  # noqa: BLE001 -- no telemetry available is itself the wedge signal
        return []


def _board_needs_reset() -> bool:
    """True when at least one chip cannot report a die temperature -- the evidence a reset is for.

    THREE STATES, NOT TWO, and the middle one is why this is not simply "does anything answer".

        every chip reports        nothing is wedged      -> skip; a reset here is pure risk
        some chips report         partly down            -> reset; the silent ones need it
        no chip reports           fully wedged           -> reset

    "Some" must reset. Right now this board has two chips at 55-57C and two publishing all-ones, and
    those two do not come back on their own -- skipping because a majority answered would leave half
    a board permanently dead.

    Costs a file read per chip and never opens the device, so unlike tt-smi it still answers while
    the board is saturated -- which matters, because the moment a reset decision is made is the
    moment the board is least able to answer an expensive question about itself.
    """
    try:
        from .probes import board_telemetry

        live, dead = board_telemetry()
        return bool(dead) or not live
    except Exception:  # noqa: BLE001 -- cannot tell: fall back to the old unconditional behaviour
        return True


def board_needs_reset() -> bool:
    """Public name for the reset precondition, so every executor can ask it.

    THE GUARD WAS UNREACHABLE FROM THE PATHS THAT NEEDED IT. `_board_needs_reset` was private and
    called only from `recover()`, while cli and trace_gate ran `tt-smi -r` directly -- so the rule
    written after the 2026-08-17 incident ("a board that is alive and healthy no longer gets reset
    at all") protected one of three reset paths. On 2026-09-22 the same failure repeated through
    cli's path: a single-chip run, a four-chip reset issued while every chip was answering 61-77C,
    and the last two chips in the list did not come back.
    """
    return _board_needs_reset()


def device_holders() -> set:
    """Pids holding /dev/tenstorrent open, except this process and its ancestors. Best-effort: no
    `fuser` or an unreadable node yields fewer holders, never an exception. The one scan both the
    reaper below and the agent runners' leftover wait (cc_harness) use."""
    import glob as _glob
    import subprocess as _sp

    holders = set()
    for node in _glob.glob("/dev/tenstorrent/*"):
        try:
            r = _sp.run(["fuser", node], capture_output=True, text=True, timeout=30)
            holders.update(int(t) for t in (r.stdout + " " + r.stderr).split() if t.strip().isdigit())
        except Exception:  # noqa: BLE001
            pass
    return holders - _protected_pids()


def reap_device_holders() -> list:
    """SIGKILL every process holding /dev/tenstorrent except this one and its ancestors.

    Returns the pids killed, so the caller can say what it did rather than claiming it silently.

    BEST-EFFORT AT EVERY STEP. No `fuser`, an unreadable /proc, a holder that exits between the scan
    and the kill -- all of it degrades to "reaped fewer than there were", never to an exception. This
    runs on the recovery path, where the board is already in trouble; a reclaim that can raise would
    turn a recoverable wedge into a dead run.
    """
    import signal as _signal

    killed = []
    for pid in sorted(device_holders()):
        try:
            os.kill(pid, _signal.SIGKILL)
            killed.append(pid)
        except Exception:  # noqa: BLE001
            pass
    return killed


def recover(where: str, reset, error_text: str = "", config_target: str = "", log=None, expand=None) -> bool:
    """Reset the device and REPORT WHETHER IT CAME BACK. True only on a VERIFIED-healthy device.

    ``reset`` is a callable taking the target spec -- the only per-caller part. Everything else
    (which board, how many tries, whether it worked, when to give up) is decided here so no caller
    can decide it differently.
    """
    # GIVING UP IS DECIDED HERE, with everything else. This docstring has always promised that "how
    # many tries... [is] decided here so no caller can decide it differently" -- and the counting was,
    # but the STOPPING was not: recovery_exhausted() had exactly one consumer in the tool
    # (termination_check, on the profile-raises path), while the four call sites that actually reset
    # -- probes, run.py, perf_mcp._recover_device, and note_crash's dead-board branch -- consulted
    # nothing. So RESET_FAILS climbed to 34 against a limit of 3 and no one stopped: ~100 minutes of
    # resets on a board that could not come back, then hours of dead run. A limit counted in one place
    # and enforced in none is not a limit.
    if recovery_exhausted():
        if log:
            log(
                "recovery EXHAUSTED at %s (%d failures >= limit %d) -- not resetting again"
                % (where, RESET_FAILS["n"], RESET_FAIL_LIMIT)
            )
        return False
    # THE KERNEL LINE EXPLAINS A FAILURE; IT DOES NOT PREDICT ONE. An earlier revision treated
    # "Failed to set initial power state" as proof that no reset could work and refused in ZERO
    # attempts. That is wrong: the message fires whenever a device is OPENED while its ARC is not
    # ready, which is transient. Observed on this box -- a wedged board produced 714 of them across a
    # day, and a perfectly healthy run produced 4 in an hour while continuing to optimize. Gating on
    # it would have declared a working board dead on the first fault.
    #
    # So resets are tried, bounded by the limit, and the kernel log is consulted only AFTER they have
    # failed -- to say WHY, which is the difference between "unrecoverable" and "reboot the host".
    # RECLAIM BEFORE RESET, AT EVERY RECOVERY POINT. A reset clears the CHIP; it does nothing about
    # the process that was mid-transfer when the chip went, and that process keeps its device handles
    # open against a device whose state has just been destroyed. It cannot make progress and it
    # cannot be reset out of the way.
    #
    # Observed on Voxtral, 2026-08-11: a perf-test build blocked in ttnn.from_torch at 18:12:51 and
    # was still holding 8 /dev/tenstorrent fds at 19:37 -- 85 minutes, 91 minutes of CPU across 65
    # threads, no log output after the first second. Three perf-test regenerations reset the chip and
    # each one then fought that orphan for the device; all three reported "device wedged on a
    # non-capturable step". Killing it by hand let the run walk straight through to Step 9.
    #
    # The reap already existed -- in run.py's _reclaim_device, which its own docstring calls "the
    # UNIVERSAL device reclaim used at EVERY recovery point". It was wired into three of the eight
    # reset sites. The other five (perf_test_gen x2, perf_mcp x2, probes) reset without it, and those
    # are exactly the paths a wedged perf-test build takes. A policy that holds at three of eight
    # call sites is not a policy, so it moves HERE, into the one primitive every reset routes
    # through, and the callers stop deciding.
    reaped = reap_device_holders()
    if reaped and log:
        log("reclaimed %d device holder(s) before reset at %s: %s" % (len(reaped), where, reaped))
    # A RESET IS FOR A WEDGE, AND FOR NOTHING ELSE.
    #
    # Resetting a board that is still answering is not a neutral act -- it is how this board was
    # broken twice on 2026-08-17. `tt-smi -r 0,1,2,3` halts each chip and brings it back in turn, so
    # a sequence that errors partway leaves the chips it already halted DOWN, with no firmware and no
    # telemetry. Measured: the run used ONE chip (MESH_DEVICE=P150), the recovery reset FOUR, the
    # sequence returned rc=1, and the two dead afterwards were the last two in that list -- devices
    # the run had never touched.
    #
    # And the board was alive when it was reset. All four chips were reporting 97-102C through sysfs
    # at that moment. The reset went ahead because the liveness question was put to tt-smi, which has
    # to OPEN the device to answer and therefore hangs exactly when the board is busy or still held:
    #
    #     tt-smi -s          opens the device; ~0.27 s idle, HANGS under load
    #     sysfs temp1_input  a file read; 0.0003 s, answered throughout
    #
    # So the cheap signal is asked first, and it can only ever CANCEL a reset, never cause one. A
    # chip that reports a plausible die temperature has a running ARC, which is the thing a reset
    # exists to restore -- there is nothing to restore. Holders are reaped above regardless, because
    # a leaked process is worth clearing whether or not the board is sick.
    #
    # Deliberately conservative in the skip direction: a board that is alive-but-unusable loses a
    # reset it might have wanted, and gets one on the next attempt once its telemetry goes. A board
    # that is alive and healthy no longer gets reset at all, which is the failure that cost four
    # chips today.
    # EVIDENCE OUTRANKS TELEMETRY, and only evidence does.
    #
    # The temperature veto reads the ARC: a chip that reports a plausible die temperature has a
    # running ARC, "which is the thing a reset exists to restore". That holds for the fault it was
    # written for -- 2026-08-17, a reset fired by a TIMEOUT, with no failure signature behind it.
    #
    # It does not hold for a stuck ETH fabric. The ARC keeps running and every chip keeps publishing
    # its temperature while the fabric is dead, so the veto cancels the reset, recover() returns
    # True, and the caller is told the board came back when nothing was done. Measured 2026-09-25 on
    # this box: the coverage probe failed with "Timed out waiting for ETH heartbeat ... Stuck at
    # 0xaabb0024", the reclaim reported "no reset issued", the retry hit the identical wedge, and
    # `tt-smi -s` answered normally throughout. The veto's own escape hatch -- "gets one on the next
    # attempt once its telemetry goes" -- never opens for this fault, because the telemetry never
    # goes.
    #
    # So a failure whose OWN TEXT carries a dead-board signature is not vetoed. That is a narrow
    # door: is_dead_board matches specific runtime faults, never a slow op or a plain timeout, so
    # the 2026-08-17 path (no signature in its output) still cancels exactly as before. Absent
    # evidence the telemetry veto is unchanged and still has the last word.
    if not is_dead_board(error_text) and not _board_needs_reset():
        if log:
            log(
                "reset SKIPPED at %s: every chip reports a die temperature (%s), so nothing is "
                "wedged -- resetting a live board is what took two chips down on 2026-08-17"
                % (where, ", ".join("%.1fC" % t for t in _live_temps()[:4]) or "n/a")
            )
        return True
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


# One post-reap retry per run. Not a second chance at the limit -- a first chance under different
# conditions, and only when the conditions demonstrably changed.
_POST_REAP_RETRY = Counter("post_reap_retry")


def retry_once_after_reaping(where: str, reset, log=None) -> bool:
    """After recovery is exhausted, reap device holders and -- only if any were found -- try ONE more
    verified reset. True when the board came back.

    VOXTRAL RUN 18. The run halted with "a board-management fault no PCIe reset can clear -- REBOOT
    THE HOST" while the reclaim reported `killed holders none`. The supervisor then printed "the
    attempt left 2 process(es) running after exiting (1047857, 1245943) -- killing them before going
    on", and a plain `tt-smi -r` brought all four p300c back in ninety seconds. The board was never
    unresettable; it was held. The reset attempts that "failed" ran while something still had the
    device open, and the reaping that would have freed it happened after the verdict.

    WHY THIS IS NOT A LOOSENED LIMIT. "A limit counted in one place and enforced in none is not a
    limit" -- RESET_FAILS once climbed to 34 against a limit of 3, and that must not come back. So
    this fires at most ONCE per run, and only when reaping actually killed something: a holder that
    existed is new evidence about why the earlier resets failed, and a retry against a changed world
    is a different experiment. Reap nothing and this returns False without touching the device,
    because nothing changed and repeating an experiment unchanged is what the limit exists to stop.

    Deliberately NOT gated on the kernel signature. board_needs_host_reboot reads a message that
    "fires whenever a device is OPENED while its ARC is not ready" -- which is precisely what a stale
    holder causes, so the fault this recovers from is the one that message is most likely to be
    reporting.
    """
    if _POST_REAP_RETRY["n"]:
        return False
    _POST_REAP_RETRY["n"] = 1
    try:
        killed = reap_device_holders()
    except Exception:  # noqa: BLE001
        killed = []
    if not killed:
        if log:
            log("post-reap retry: no device holders found -- nothing changed, not resetting again")
        return False
    if log:
        log(
            "post-reap retry: killed %d stale device holder(s) %s -- the earlier resets ran while the "
            "device was held, so trying ONE more" % (len(killed), killed)
        )
    RESET_FAILS["n"] = 0
    return recover(where, reset, "post-reap retry", "", log, None)


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
    """The device worked: clear the CRASH STREAK.

    Deliberately NOT the reset count. That is a within-run backstop against retrying forever, and a
    board that alternates working and wedging would clear it on every good measurement -- restoring
    exactly the unbounded retrying the limit exists to stop. The reset count is scoped to the run
    instead (see _run_stamp), which is the honest lifetime for "have resets stopped working here".
    """
    CONSEC_CRASH["n"] = 0
