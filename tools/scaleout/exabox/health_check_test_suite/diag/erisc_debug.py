# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Phase 9: the ERISC debug dump (the ETH base firmware's state on every ETH core).

For each ETH core it reads, with tt-exalens: whether the FW sections in L1 match
the release ELF, the ERISC0 PC and its function, and the ETH, SerDes and MAC/PCS
postcodes in ``boot_results``. It only reads: the PC comes over the debug bus
while the core keeps running. The reads follow bh-erisc's
src/debug/erisc-debug.py; keep the postcode names in step with it.

It decodes through the ELF of the ETH FW release each chip runs. Nothing here
downloads one: the ELFs are provisioned on the host, in ``--erisc-elf-dir``. A
chip whose ELF is not there is not inspected, and the phase names the ELF it
needs.

The dump runs as a child process, ``python -m diag.erisc_debug``, under the
python the ETH status read runs tt-triage with (tools/triage/requirements.txt
brings tt-exalens), so a crash or hang in it costs this phase and not the report.

Forensics for whoever picks up the ticket, so it never gates.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

from .common import SUITE_DIR, emit_result, emit_running, log, run_with_timeout, write_command_log
from .eth_status import pcie_gate_check, read_pcie_functions, resolve_triage_python
from .model import PASS, SKIP, WARN, Check, Phase

# Where the ETH FW release ELFs (eth_init-vX.Y.Z.elf) are provisioned on the host.
DEFAULT_ERISC_ELF_DIR = Path("/opt/tenstorrent/eth-fw-elf")

# A backstop, not the expected duration: four section compares, a PC read and a
# boot_results read per ETH core.
ERISC_DEBUG_TIMEOUT_S = 600


# ─────────────────────────────────────────────────────────────────────────────
# The phase
# ─────────────────────────────────────────────────────────────────────────────


def summarize_erisc_debug(result: dict) -> tuple[str, str, dict]:
    """(status, details, data) from the dump's JSON. Pure, so a stored dump is graded the same."""
    cores = result.get("cores") or []
    read = [c for c in cores if not c.get("error")]
    unreadable = [f"dev {c['device']} {c['location']}: {c['error']}" for c in cores if c.get("error")]
    differing = [f"dev {c['device']} {c['location']}" for c in read if c.get("failed_sections")]
    missing = [
        f"ETH FW {e['eth_fw'] or 'unknown'} on {len(e['devices'])} device(s)"
        + (f" needs {e['needed']}" if e["needed"] else "")
        + f": {e['reason']}"
        for e in result.get("elfs") or []
        if not e.get("path")
    ]
    elfs = sorted(os.path.basename(e["path"]) for e in result.get("elfs") or [] if e.get("path"))
    data = {
        "elf_dir": result.get("elf_dir"),
        "elfs": result.get("elfs") or [],
        "missing_elfs": missing,
        "cores_read": len(read),
        "cores_unreadable": unreadable,
        "sections_differ": differing,
        "postcodes": {
            kind: dict(Counter(c["postcodes"][kind]["name"] for c in read)) for kind in ("eth", "serdes", "macpcs")
        },
        "dump_error": result.get("error"),
    }

    if not read:
        reasons = missing + unreadable[:3] + ([result["error"]] if result.get("error") else [])
        return SKIP, "no ETH core read: " + ("; ".join(reasons) or "no device found"), data

    devices = {c["device"] for c in read}
    details = (
        f"{len(read)} ETH cores read on {len(devices)} device(s) with {', '.join(elfs)}; "
        f"{len(differing)} with FW sections in L1 that differ from the ELF"
    )
    gaps = missing + ([f"{len(unreadable)} core(s) could not be read"] if unreadable else [])
    if result.get("error"):
        gaps.append(f"stopped early: {result['error']}")
    if gaps:
        return WARN, details + "; not read: " + "; ".join(gaps), data
    return PASS, details, data


def run_erisc_debug(
    tt_metal: Path,
    phase: Phase,
    dry_run: bool,
    logs_dir: Path,
    elf_dir: Path = DEFAULT_ERISC_ELF_DIR,
    *,
    sys_root: Path = Path("/sys"),
) -> None:
    """Dump every ETH core's FW state into ``<logs_dir>/erisc_debug.txt``.

    * **SKIP** — nothing read: tt-exalens is not importable, a chip has dropped
      off PCIe (so, as for the ETH status read, no chip is opened), no chip's ELF
      is in ``elf_dir`` (the details name each ELF needed), or the dump timed out
      or wrote no result.
    * **WARN** — some ETH cores read, not all: a chip's ELF is missing, or a core
      could not be read.
    * **PASS** — every ETH core of every chip read.

    What the cores were found doing is recorded in ``data`` and the text dump,
    not judged.
    """
    python = resolve_triage_python(tt_metal)
    text_path = logs_dir / "erisc_debug.txt"
    json_path = logs_dir / "erisc_debug.json"
    log_path = logs_dir / "erisc_debug.log"
    # -u: a dump killed at the timeout keeps what it had printed.
    cmd = [python, "-u", "-m", "diag.erisc_debug", f"--elf-dir={elf_dir}", f"--json={json_path}"]

    log("--- erisc debug ---")
    log(f"  cmd:  {shlex.join(cmd)}")

    if dry_run:
        print(f"  {'erisc_debug':30} (dry-run)")
        phase.add(Check(name="erisc_debug", status=SKIP, details=f"(dry) {shlex.join(cmd)}", ip="other"))
        return

    gate = pcie_gate_check(read_pcie_functions(sys_root))
    if gate.status != PASS:
        phase.add(Check(name="erisc_debug", status=SKIP, details=f"chips not opened: {gate.details}", ip="other"))
        return

    logs_dir.mkdir(parents=True, exist_ok=True)
    # A result left by an earlier run would be read as this one's.
    json_path.unlink(missing_ok=True)
    env = os.environ.copy()
    # Every chip is read, whatever the caller left visible to metal.
    env.pop("TT_METAL_VISIBLE_DEVICES", None)

    emit_running("erisc_debug")
    t0 = time.time()
    try:
        rc, out, err, timed_out = run_with_timeout(cmd, ERISC_DEBUG_TIMEOUT_S, cwd=str(SUITE_DIR), env=env)
    except OSError as e:
        rc, out, err, timed_out = 127, "", repr(e), False
    dt = time.time() - t0
    try:
        text_path.write_text(out)
    except OSError as e:
        log(f"  could not write {text_path}: {e!r}")
    write_command_log(log_path, shlex.join(cmd), rc, dt, f"(in {text_path.name})" if out else "", err)

    try:
        result = json.loads(json_path.read_text())
    except (OSError, ValueError):
        result = None

    data: dict = {}
    if timed_out:
        status, details = SKIP, f"timed out after {ERISC_DEBUG_TIMEOUT_S}s; what it printed is in {text_path.name}"
    elif not isinstance(result, dict):
        first = next((ln.strip() for ln in err.splitlines() if ln.strip()), "")
        status, details = SKIP, f"rc={rc} but no result written: {first[:200]}"
    else:
        status, details, data = summarize_erisc_debug(result)
    details += f" dur={dt:.1f}s log={text_path}"

    emit_result("erisc_debug", status, suffix=f"({dt:.1f}s)")
    data.update(command=shlex.join(cmd), rc=rc, duration_s=dt, report_file=str(text_path), stderr_tail=err[-2000:])
    phase.add(Check(name="erisc_debug", status=status, details=details, data=data, ip="other"))


# ─────────────────────────────────────────────────────────────────────────────
# The dump, in the child process. tt-exalens is imported only here.
# ─────────────────────────────────────────────────────────────────────────────

# Compared against the ELF. An eth_init ELF has all of them.
TEST_SECTIONS = (".init", ".fw_version", ".text", ".rodata")

# Postcode kind -> (boot_results member, the ELF constants that name its values)
POSTCODES = {
    "eth": (
        "eth_status",
        [f"POSTCODE_ETH_INIT_{s}" for s in "SKIP SERDES ETH_CTRL MACPCS PACKET PASS FAIL CODE_NOT_FOUND".split()],
    ),
    "serdes": (
        "serdes_results",
        [
            f"POSTCODE_SERDES_INIT_{s}"
            for s in (
                "NOT_STARTED SKIP UNCONNECTED RESET LANE_RESET MAN_EQ_START MAN_EQ_TX MAN_EQ_RX MAN_EQ_EQUALIZE "
                "MAN_EQ_SIGDET MAN_EQ_EQ_ASSERT MAN_EQ_EQ_DEASSERT WAIT_HOST ANLT_START ANLT_AN ANLT_LT ANLT_RETRAIN "
                "CDR BIST BIST_TIMER RETRAIN PASS RETRAIN_PASS FAIL RETRAIN_FAIL PRBS CODE_NOT_FOUND"
            ).split()
        ],
    ),
    "macpcs": (
        "macpcs_results",
        [
            f"POSTCODE_MACPCS_INIT_{s}"
            for s in (
                "NOT_STARTED START CONFIG_MAC CONFIG_PCS CONFIG_PMA RELEASE_RESET LINK_WAIT RETRAIN PASS FAIL"
            ).split()
        ],
    ),
}

# Where the ETH FW keeps its version in L1, for a chip whose firmware info has none.
ETH_FW_VERSION_ADDR = 0x7CFBC


def format_fw_version(version: int | None) -> str:
    """major << 16 | minor << 8 | patch -> vX.Y.Z"""
    return "unknown" if version is None else f"v{version >> 16}.{(version >> 8) & 0xFF}.{version & 0xFF}"


def find_elf(elf_dir: Path, tag: str) -> Path | None:
    """The release ELF of ``tag`` in ``elf_dir``, under the release's name or the underscore one."""
    return next((p for p in (elf_dir / f"eth_init-{tag}.elf", elf_dir / f"eth_init_{tag}.elf") if p.is_file()), None)


def eth_fw_version(device) -> int | None:
    try:
        v = device._umd_device.get_local_tt_device().get_firmware_info_provider().get_eth_fw_version()
    except Exception:
        v = None
    if v is not None:
        return v.major << 16 | v.minor << 8 | v.patch
    for location in device.get_block_locations(block_type="eth"):
        raw = device.noc_read32(location, ETH_FW_VERSION_ADDR)
        if raw:
            return raw & 0xFFFFFF
    return None


def load_elf(path: Path, version: int):
    """(ElfFile, None), or (None, why it cannot be used for ``version``)."""
    from ttexalens.elf import ElfFile

    try:
        elf = ElfFile(str(path))
    except Exception as e:
        return None, f"{path} could not be read: {e}"
    if any(elf.get_section_by_name(s) is None for s in TEST_SECTIONS):
        return None, f"{path} is not an eth_init ELF"
    section = elf.get_section_by_name(".fw_version")
    inside = int.from_bytes(bytes(section.data)[:4], "little") if section.size >= 4 else None
    if inside != version:
        return None, f"{path} is {format_fw_version(inside)} inside, not {format_fw_version(version)}"
    return elf, None


def postcode_names(elf) -> dict[str, dict[int, str]]:
    names: dict[str, dict[int, str]] = {}
    for kind, (_, constants) in POSTCODES.items():
        names[kind] = {}
        for constant in constants:
            try:
                names[kind][elf.get_constant(constant)] = constant
            except Exception:
                pass
    return names


def read_core(device, block, elf, names: dict[str, dict[int, str]]) -> dict:
    from ttexalens.elf import get_frame_callstack
    from ttexalens.memory_access import create_l1_memory_access

    location = block.location.to_user_str()
    print(f"Device: {device.id}, Location: {location}")
    core: dict = {"device": device.id, "location": location, "error": None}
    try:
        failed = []
        for name in TEST_SECTIONS:
            section = elf.get_section_by_name(name)
            buffer = bytearray(section.size)
            block.location.noc_read(section.address, buffer)
            if section.data != memoryview(buffer):
                failed.append(name)
        core["failed_sections"] = failed
        print(f"  Failed sections: {', '.join(failed)}" if failed else "  All sections correctly loaded into L1")

        pc = block.get_risc_debug("erisc0").get_pc()
        frames = [
            f"{f.function_name} ({f.file_info.file}:{f.file_info.line})" if f.file_info else f"{f.function_name}"
            for f in get_frame_callstack([elf], pc, extract_variables=False)
        ]
        core["pc"], core["callstack"] = hex(pc), frames
        print(f"  Top callstack (pc={hex(pc)}):")
        for frame in frames or ["<empty>"]:
            print(f"    {frame}")

        boot_results = elf.get_global("boot_results", create_l1_memory_access(block.location))
        core["postcodes"] = {}
        for kind, (member, _) in POSTCODES.items():
            value = int(getattr(boot_results, member).postcode)
            name = names[kind].get(value, "UNKNOWN")
            print(f"  {kind.upper()} status postcode: {hex(value)} - {name}")
            core["postcodes"][kind] = {"value": hex(value), "name": name}
    except Exception as e:
        # One core that cannot be read must not cost the rest of the dump.
        core["error"] = f"{type(e).__name__}: {e}"
        print(f"  ERROR: {core['error']}")
    return core


def dump(elf_dir: Path, result: dict) -> None:
    """Read every ETH core of every chip whose ELF is in ``elf_dir`` into ``result``,
    saying for each ETH FW version which ELF it needs and whether it was found."""
    from ttexalens import init_ttexalens

    context = init_ttexalens()
    by_version: dict = {}
    for device_id in sorted(context.devices):
        device = context.devices[device_id]
        by_version.setdefault(eth_fw_version(device), []).append(device)

    for version, devices in sorted(by_version.items(), key=lambda kv: (kv[0] is None, kv[0] or 0)):
        tag = format_fw_version(version)
        entry = {"eth_fw": None, "devices": [d.id for d in devices], "needed": None, "path": None, "reason": None}
        result["elfs"].append(entry)
        on = f"on {len(devices)} device(s)"
        if version is None:
            entry["reason"] = "the ETH FW version could not be read"
            print(f"ETH FW unknown {on}: not inspected, {entry['reason']}")
            continue
        entry["eth_fw"], entry["needed"] = tag, f"eth_init-{tag}.elf"
        path = find_elf(elf_dir, tag)
        if path is None:
            elf, entry["reason"] = None, f"not found in {elf_dir}" + (
                "" if elf_dir.is_dir() else " (no such directory)"
            )
        else:
            elf, entry["reason"] = load_elf(path, version)
        if elf is None:
            print(f"ETH FW {tag} {on}: needs {entry['needed']}, {entry['reason']}; not inspected")
            continue
        entry["path"] = str(path)
        print(f"ETH FW {tag} {on}: ETH ELF {path}")
        names = postcode_names(elf)
        for device in devices:
            for block in device.get_blocks("eth"):
                result["cores"].append(read_core(device, block, elf, names))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Dump the ETH base firmware's state on every ETH core.")
    ap.add_argument("--elf-dir", type=Path, default=DEFAULT_ERISC_ELF_DIR, help="ETH FW release ELFs")
    ap.add_argument("--json", type=Path, required=True, help="Where to write what was found")
    args = ap.parse_args(argv)
    try:
        import ttexalens  # noqa: F401
    except ImportError as e:
        print(f"ERROR: {e}; install tools/triage/requirements.txt (create_venv.sh does)", file=sys.stderr)
        return 1

    result: dict = {"elf_dir": str(args.elf_dir), "elfs": [], "cores": [], "error": None}
    try:
        dump(args.elf_dir, result)
    except Exception as e:
        traceback.print_exc()
        result["error"] = f"{type(e).__name__}: {e}"
    args.json.write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
