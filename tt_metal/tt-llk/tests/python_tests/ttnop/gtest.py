#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Run one native Metal gtest filter through the ttnop sweep."""

import argparse
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
PRELOAD = HERE / "libttnop_gtest.so"

arch = os.environ.get("CHIP_ARCH") or os.environ.get("ARCH_NAME") or "wormhole"
os.environ["CHIP_ARCH"] = {
    "wormhole_b0": "wormhole",
    "wh": "wormhole",
    "bh": "blackhole",
}.get(arch.strip().lower(), arch.strip().lower())

import scanner
import sweep
from cave import DetourError, Injector

TRISC_THREAD = {0: "unpack", 1: "math", 2: "pack"}
TRISC_PATH = re.compile(r"(?:^|/)trisc([0-3])/trisc\1\.elf$")


@dataclass(frozen=True)
class Target:
    raw_elf: str
    xip_elf: Path
    thread: str
    kernel: str
    scan: object


def _env(**values) -> dict:
    env = os.environ.copy()
    env.setdefault("TT_METAL_HOME", str(REPO))
    env.pop("TT_METAL_DISABLE_XIP_DUMP", None)
    for name in ("TTNOP_GTEST_TRACE", "TTNOP_GTEST_PATCH", "TTNOP_GTEST_ACK"):
        env.pop(name, None)
    env.update({name: str(value) for name, value in values.items()})
    env["LD_PRELOAD"] = (
        f"{PRELOAD}{':' + env['LD_PRELOAD'] if env.get('LD_PRELOAD') else ''}"
    )
    return env


def _run(binary: Path, test_filter: str, env: dict, timeout: float) -> tuple:
    command = [str(binary), f"--gtest_filter={test_filter}", "--gtest_color=no"]
    if "DISABLED_" in test_filter:
        command.append("--gtest_also_run_disabled_tests")
    try:
        result = subprocess.run(
            command,
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr, False
    except subprocess.TimeoutExpired as error:

        def text(value):
            return (
                value.decode(errors="replace")
                if isinstance(value, bytes)
                else value or ""
            )

        return -1, text(error.stdout), text(error.stderr), True


def _show_process(result: tuple) -> None:
    _, stdout, stderr, _ = result
    if stdout.strip():
        print(stdout.rstrip(), flush=True)
    if stderr.strip():
        print(stderr.rstrip(), file=sys.stderr, flush=True)


def _targets(trace: Path, config: sweep.Config) -> list:
    selected = os.environ.get("TTNOP_METAL_KERNEL", "").strip()
    targets, seen = [], set()
    for line in trace.read_text().splitlines() if trace.is_file() else ():
        raw_elf, separator, loading = line.rpartition("\t")
        if not separator or loading != "2":
            continue
        match = TRISC_PATH.search(raw_elf)
        if match is None or int(match.group(1)) not in TRISC_THREAD:
            continue
        # Quasar's fourth, isolate-SFPU thread is intentionally ignored until it
        # can be validated on Quasar hardware.
        if raw_elf in seen or (selected and re.search(selected, raw_elf) is None):
            continue
        seen.add(raw_elf)
        xip = Path(raw_elf + ".xip.elf")
        if not xip.is_file():
            raise DetourError(f"{xip} missing; XIP dumps must be enabled")
        scanned = scanner.scan(str(xip), config.site_mode)
        if not scanned.cave_start:
            raise DetourError(f"{xip} has no cave; run `make metal_cave` in {HERE}")
        elf = Path(raw_elf)
        targets.append(
            Target(
                raw_elf,
                xip.resolve(),
                TRISC_THREAD[int(match.group(1))],
                f"{elf.parents[2].name}/{elf.parents[1].name}",
                scanned,
            )
        )
    return targets


def _patch_words(variant: sweep.Variant, scan, max_delay: int) -> list:
    words = {}

    def read(address, count):
        return [
            words.get(address + index * 4, variant.site.word) for index in range(count)
        ]

    def write(address, values):
        words.update(
            (address + index * 4, word & 0xFFFFFFFF)
            for index, word in enumerate(values)
        )

    Injector(read, write, max_delay).arm(
        variant.thread,
        scan,
        variant.site,
        variant.delay,
        variant.filler_word,
    )
    return sorted(words.items())


def _text_section(data: bytes) -> tuple:
    if data[:6] != b"\x7fELF\x01\x01":
        raise DetourError("temp dump expects a little-endian ELF32")
    header = struct.unpack_from("<HHIIIIIHHHHHH", data, 16)
    offset, size, names_index = header[5], header[10], header[12]
    sections = [
        struct.unpack_from("<IIIIIIIIII", data, offset + index * size)
        for index in range(header[11])
    ]
    names_offset = sections[names_index][4]
    for section in sections:
        start = names_offset + section[0]
        if data[start : data.index(b"\0", start)] == b".text":
            return section[4], section[5], section[3]
    raise DetourError("temp dump source has no .text")


def _dump_elf(root: Path, test_filter: str, target: Target, variant, words) -> Path:
    data = bytearray(target.xip_elf.read_bytes())
    file_offset, size, address = _text_section(data)
    for patch_address, word in words:
        offset = patch_address - address
        if offset < 0 or offset + 4 > size:
            raise DetourError("temp dump patch falls outside .text")
        struct.pack_into("<I", data, file_offset + offset, word)

    safe = lambda value: re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    directory = root / safe(test_filter) / safe(target.kernel)
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / (
        f"{variant.seq:06d}_{variant.thread}_site{variant.site.index}_"
        f"n{variant.delay}_{safe(variant.filler)}.elf"
    )
    output.write_bytes(data)
    with open(root / "MANIFEST.txt", "a") as manifest:
        manifest.write(
            f"{output.relative_to(root)}\t{variant.label()}\tkernel={target.kernel}\n"
        )
    return output


class Runtime:
    def __init__(
        self,
        binary: Path,
        test_filter: str,
        targets: dict,
        config: sweep.Config,
        work: Path,
        dump_root: Path,
        verbose: bool,
        timeout: float,
    ):
        self.binary = binary
        self.test_filter = test_filter
        self.targets = targets
        self.config = config
        self.work = work
        self.dump_root = dump_root
        self.verbose = verbose
        self.timeout = timeout

    def run(self, variant):
        target = self.targets[variant.thread]
        words = _patch_words(variant, target.scan, self.config.max_delay)
        patch, ack = self.work / "patch.txt", self.work / "ack.txt"
        ack.unlink(missing_ok=True)
        with open(patch, "w") as output:
            output.write(f"{target.raw_elf}\n0x{target.scan.text_start:x}\n")
            output.writelines(
                f"0x{address:x} 0x{word:08x}\n" for address, word in words
            )

        if self.dump_root:
            dumped = _dump_elf(self.dump_root, self.test_filter, target, variant, words)
            if self.verbose:
                print(f">> elf={dumped}", flush=True)
        if self.verbose:
            print(
                f">> {self.test_filter}: kernel={target.kernel} {variant.label()}",
                flush=True,
            )

        result = _run(
            self.binary,
            self.test_filter,
            _env(TTNOP_GTEST_PATCH=patch, TTNOP_GTEST_ACK=ack),
            self.timeout,
        )
        if not ack.is_file() or target.raw_elf not in ack.read_text():
            _show_process(result)
            raise DetourError(f"patch was not applied to {target.raw_elf}")
        if result[3]:
            return "hang", f"gtest exceeded {self.timeout:g}s"
        if result[0]:
            lines = [
                line.strip()
                for line in (result[2] + "\n" + result[1]).splitlines()
                if line.strip()
            ]
            error = next(
                (line for line in lines if "[  FAILED  ]" in line),
                lines[-1] if lines else f"gtest exited {result[0]}",
            )
            print(f">> failed: {variant.label()} ({error})", flush=True)
            return "mismatch", error
        return None, ""


def run(argv: list) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--filter", required=True)
    args = parser.parse_args(argv[1:])
    if any(character in args.filter for character in "*?:"):
        raise DetourError("--gtest-filter must name exactly one Suite.Test")

    binary = Path(args.binary)
    binary = binary if binary.is_absolute() else REPO / binary
    if not binary.is_file():
        raise DetourError(f"gtest binary not found: {binary}")
    subprocess.run(["make", "--silent", "-C", str(HERE), "gtest_shim"], check=True)

    config = sweep.Config.from_env()
    verbose = os.environ.get("TTNOP_VERBOSE", "") not in ("", "0")
    timeout = float(os.environ.get("TTNOP_GTEST_TIMEOUT", "600"))
    dump_root = (
        HERE / "temp_elf"
        if os.environ.get("TTNOP_DUMP_ELFS", "") not in ("", "0")
        else None
    )
    if dump_root:
        shutil.rmtree(dump_root, ignore_errors=True)
        dump_root.mkdir()

    with tempfile.TemporaryDirectory(prefix="ttnop-gtest-") as directory:
        work = Path(directory)
        trace = work / "trace.txt"
        print(f">> baseline: {args.filter}", flush=True)
        baseline = _run(binary, args.filter, _env(TTNOP_GTEST_TRACE=trace), timeout)
        if baseline[3] or baseline[0]:
            _show_process(baseline)
            raise DetourError("clean gtest baseline failed")
        targets = _targets(trace, config)
        if not targets:
            raise DetourError("gtest loaded no selected compute kernel")

        failures, sequence, wedged = 0, 0, False
        for kernel in dict.fromkeys(target.kernel for target in targets):
            group = [target for target in targets if target.kernel == kernel]
            by_thread = {target.thread: target for target in group}
            planned = sweep.plan(
                config, {thread: target.scan for thread, target in by_thread.items()}
            )
            planned = [
                replace(variant, seq=sequence + index)
                for index, variant in enumerate(planned)
            ]
            sequence += len(planned)
            runtime = Runtime(
                binary,
                args.filter,
                by_thread,
                config,
                work,
                dump_root,
                verbose,
                timeout,
            )
            print(f">> kernel={kernel} variants={len(planned)}", flush=True)
            try:
                failures += len(sweep.run(config, planned, runtime, lambda *_: None))
            except sweep.DeviceWedged as error:
                print(f">> hang: {error}", file=sys.stderr)
                failures += 1
                wedged = True
                break

    if dump_root:
        print(f">> perturbed ELFs -> {dump_root}")
    return 75 if wedged else (1 if failures else 0)


if __name__ == "__main__":
    try:
        sys.exit(run(sys.argv))
    except (DetourError, ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"ttnop: {error}", file=sys.stderr)
        sys.exit(4)
