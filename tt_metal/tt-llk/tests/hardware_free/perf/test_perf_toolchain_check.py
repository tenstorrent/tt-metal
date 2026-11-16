# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import datetime as dt
import pathlib
import shutil
import struct
import subprocess
import sys
import types

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parents[3] / "perf"))
import toolchain_check
from window_hash import _hash16

MARKER = b"LLK_PROFILER:sources/x.cpp:42:TILE_LOOP"
ZONE = _hash16(MARKER)


def write_elf(path, secs):
    """A minimal ELF32 with the given (name, type, flags, addr, bytes) sections."""
    secs = [*secs, (".shstrtab", 3, 0, 0, None)]
    names = b"\0"
    offsets = []
    for name, *_ in secs:
        offsets.append(len(names))
        names += name.encode() + b"\0"
    body, heads = b"", [b"\0" * 40]
    for (name, kind, flags, addr, data), off in zip(secs, offsets):
        data = names if data is None else data
        heads.append(
            struct.pack(
                "<IIIIIIIIII",
                off,
                kind,
                flags,
                addr,
                52 + len(body),
                len(data),
                0,
                0,
                4,
                0,
            )
        )
        body += data
    header = b"\x7fELF\x01\x01\x01" + b"\0" * 9
    header += struct.pack(
        "<HHIIIIIHHHHHH",
        2,
        243,
        1,
        0,
        0,
        52 + len(body),
        0,
        52,
        0,
        0,
        40,
        len(heads),
        len(secs),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + body + b"".join(heads))


def kernel_elf(
    path, text=b"\x13\x00\x00\x00" * 4, debug=b"/build/a", loop=0x100, dump=None
):
    write_elf(
        path,
        [
            (".text", 1, 0x6, 0x100, text),
            (".rodata", 1, 0x2, 0x200, MARKER + b"\0"),
            (".data", 1, 0x3, 0x300, b"\1\2\3\4"),
            (".debug_str", 1, 0, 0, debug),
        ],
    )
    pad = (loop - 0x100) // 4
    lines = ["00000100 <run_kernel>:"]
    insns = [("addi", "sp,sp,-16")] * pad + [
        ("lui", f"a5,{hex(0xA0000 | ZONE)}"),
        ("addi", "a4,a4,1"),
        ("bne", f"a4,a3,{loop + 4:x} <run_kernel+{hex(loop + 4 - 0x100)}>"),
        ("lui", f"a5,{hex(0xB0000 | ZONE)}"),
        ("ret", ""),
    ]
    if dump is not None:
        insns[pad + 1] = dump
    lines += [f" {0x100 + 4 * k:x}:\t{m}\t{o}" for k, (m, o) in enumerate(insns)]
    pathlib.Path(f"{path}.dump").write_text("\n".join(lines) + "\n")


@pytest.fixture
def objdump(tmp_path):
    tool = tmp_path / "objdump"
    tool.write_text('#!/bin/sh\nexec cat "$3.dump"\n')
    tool.chmod(0o755)
    return str(tool)


def test_sections_and_text_size(tmp_path):
    kernel_elf(tmp_path / "k.elf")
    secs = toolchain_check.sections(tmp_path / "k.elf")
    assert secs[".text"][2] == 0x100
    assert toolchain_check.text_size(secs) == 16 + len(MARKER) + 1


def test_debug_sections_do_not_count(tmp_path):
    kernel_elf(tmp_path / "a.elf", debug=b"/tmp/a/tt-llk-build")
    kernel_elf(tmp_path / "b.elf", debug=b"/tmp/longer-b/tt-llk-build")
    kernel_elf(tmp_path / "c.elf", text=b"\x13\x00\x00\x00" * 5)
    a, b, c = (toolchain_check.sections(tmp_path / f"{x}.elf") for x in "abc")
    assert toolchain_check.differs(a, b) == []
    assert toolchain_check.differs(a, c) == [".text"]


def test_host_compiler_objects_match_across_build_paths(tmp_path):
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("no host compiler")
    (tmp_path / "k.c").write_text(
        "int f(int n){int s=0;for(int i=0;i<n;i++)s+=i*3;return s;}\n"
    )
    for side in ("a", "longer-b"):
        (tmp_path / side).mkdir()
        subprocess.run(
            [cc, "-O2", "-g", "-c", "../k.c", "-o", "k.o"],
            cwd=tmp_path / side,
            check=True,
        )
    if (tmp_path / "a/k.o").read_bytes()[:4] != b"\x7fELF":
        pytest.skip("the host compiler does not write ELF")
    a, b = (toolchain_check.sections(tmp_path / f"{s}/k.o") for s in ("a", "longer-b"))
    assert toolchain_check.differs(a, b) == []


def test_elfs_compares_two_builds(tmp_path, objdump):
    for side, debug in (("a", b"/a"), ("b", b"/longer/b")):
        kernel_elf(tmp_path / side / "t/v1/elf/math.elf", debug=debug, loop=0x108)
    kernel_elf(tmp_path / "a/t/v2/elf/math.elf")
    kernel_elf(tmp_path / "b/t/v2/elf/math.elf", text=b"\x13\x00\x00\x00" * 5)
    kernel_elf(tmp_path / "a/t/v3/elf/pack.elf")

    out = toolchain_check.elfs(tmp_path / "a", tmp_path / "b", objdump)

    assert out["t/v1/elf/math.elf"]["reproducible"]
    assert out["t/v1/elf/math.elf"]["loop"] == 0x108
    assert len(out["t/v1/elf/math.elf"]["hash"]) == 12
    assert out["t/v2/elf/math.elf"]["differs"] == [".text"]
    assert out["t/v3/elf/pack.elf"] == {
        "reproducible": False,
        "differs": ["only in one build"],
    }


def test_loop_moves_keep_the_hash_and_code_changes_do_not(tmp_path, objdump):
    kernel_elf(tmp_path / "a.elf", loop=0x100)
    kernel_elf(tmp_path / "b.elf", loop=0x110)
    kernel_elf(tmp_path / "c.elf", text=b"\x13" * 20, dump=("addi", "a4,a4,2"))
    fp = [
        toolchain_check.window_hash.fingerprint(objdump, tmp_path / f"{x}.elf")
        for x in "abc"
    ]
    assert toolchain_check.loop_start(objdump, tmp_path / "b.elf") == 0x110
    assert fp[0] == fp[1] != fp[2]


def _tools(installed="7.93.0", pin="7.93.0", **packages):
    return {
        "pin": pin,
        "installed": installed,
        "compiler": "riscv-tt-elf-g++ 15.1.0",
        "packages": {"numpy": "1.26.4", "tt-umd": "0.9.16", "rich": "13.0", **packages},
    }


def test_toolchain_reads_pin_install_and_pip(tmp_path, monkeypatch):
    (tmp_path / "tests/sfpi/compiler/bin").mkdir(parents=True)
    (tmp_path / "tests/sfpi-version").write_text(
        "sfpi_version='7.93.0'\nsfpi_build='1'\n"
    )
    (tmp_path / "tests/sfpi/sfpi.version").write_text("7.69.0\n")
    (tmp_path / "tests/sfpi/compiler/bin/riscv-tt-elf-g++").touch()
    outputs = {
        "--version": "riscv-tt-elf-g++ (sfpi:7.69.0) 15.1.0\nCopyright\n",
        "freeze": "numpy==1.26.4\ntt_umd==0.9.17\nfoo @ file:///x\n",
    }

    def run(args, **_):
        return types.SimpleNamespace(stdout=outputs[args[-1]])

    monkeypatch.setattr(toolchain_check.subprocess, "run", run)
    out = toolchain_check.toolchain(tmp_path)
    assert out["pin"] == "7.93.0"
    assert out["installed"] == "7.69.0"
    assert out["compiler"] == "riscv-tt-elf-g++ (sfpi:7.69.0) 15.1.0"
    assert out["packages"] == {
        "numpy": "1.26.4",
        "tt-umd": "0.9.17",
        "foo": "file:///x",
    }


def test_toolchain_report():
    today = {
        "llk-toolchain-blackhole-1": _tools(
            installed="7.69.0", **{"tt-umd": "0.9.17", "rich": "14.0"}
        ),
        "llk-toolchain-blackhole-2": _tools(
            installed="7.69.0", **{"tt-umd": "0.9.17", "rich": "14.0"}
        ),
        "llk-build-check-blackhole": {},
    }
    earlier = {
        "llk-toolchain-blackhole-1": _tools(),
        "llk-toolchain-blackhole-2": _tools(),
    }

    problems, lines = toolchain_check.toolchain_report(today, earlier)

    assert problems == [
        "SFPI 7.69.0 installed, 7.93.0 pinned",
        "pip tt-umd 0.9.16 → 0.9.17",
    ]
    assert lines == ["pip rich 13.0 → 14.0"]
    assert toolchain_check.toolchain_report(earlier, earlier) == ([], [])


def _elf(loop, hash_="h1", size=100, ok=True):
    return {
        "reproducible": ok,
        "differs": [] if ok else [".text"],
        "text_size": size,
        "loop": loop,
        "hash": hash_,
    }


def test_build_report_labels_each_kernel():
    earlier = {
        "llk-build-check-wormhole": {
            "t/same/elf/math.elf": _elf(0x100),
            "t/moved/elf/math.elf": _elf(0x100),
            "t/moved/elf/pack.elf": _elf(0x200),
            "t/code/elf/math.elf": _elf(0x100),
        }
    }
    today = {
        "llk-build-check-wormhole": {
            "t/same/elf/math.elf": _elf(0x100),
            "t/moved/elf/math.elf": _elf(0x110, size=116),
            "t/moved/elf/pack.elf": _elf(0x200),
            "t/code/elf/math.elf": _elf(0x100, hash_="h2", ok=False),
        }
    }

    problems, lines = toolchain_check.build_report(today, earlier)

    assert problems == [
        "wormhole: 1 of 4 sentinel ELFs not reproducible, e.g. t/code/elf/math.elf (.text)"
    ]
    assert lines[0] == (
        "wormhole: 3/4 ELFs reproducible; kernels: 1 loop code changed, "
        "1 same loop, moved, 1 unchanged (1 moved to another 64-byte offset)"
    )
    assert lines[1].startswith("  loop code changed: t/code/elf/math.elf")
    assert lines[2] == (
        "  same loop, moved: t/moved/elf/math.elf TILE_LOOP 0x100 (mod 16 0, mod 64 0)"
        " → 0x110 (mod 16 0, mod 64 16), text 100 → 116 B; "
        "t/moved/elf/pack.elf TILE_LOOP 0x200 (mod 16 0, mod 64 0)"
    )
    assert toolchain_check.build_report(today, {})[1] == [
        "wormhole: 4 sentinel ELFs, no earlier build to compare"
    ]


def test_report_finds_the_last_earlier_artifacts(tmp_path, monkeypatch):
    runs = {"2026-11-08": {"id": 8}, "2026-11-06": {"id": 6}, "2026-11-05": {"id": 5}}
    artifacts = {
        8: {"llk-toolchain-wormhole-1": _tools(pin="7.94.0")},
        6: {},
        5: {"llk-toolchain-wormhole-1": _tools()},
    }
    monkeypatch.setattr(
        toolchain_check, "scheduled_run", lambda repo, wf, day: runs.get(day)
    )
    monkeypatch.setattr(
        toolchain_check, "download", lambda repo, run, dest: artifacts[run["id"]]
    )

    out = toolchain_check.report("o/r", dt.date(2026, 11, 8), tmp_path)

    assert out["problems"] == ["SFPI 7.93.0 installed, 7.94.0 pinned"]
    assert out["lines"] == [
        "compared with the 2026-11-05 nightly",
        "SFPI pin: 7.93.0 → 7.94.0",
    ]
    assert toolchain_check.report("o/r", dt.date(2026, 11, 7), tmp_path)["lines"] == [
        "no build or toolchain artifacts from the 2026-11-07 nightly"
    ]


def test_download_reads_one_json_per_artifact(tmp_path, monkeypatch):
    def gh(*args):
        dest = pathlib.Path(args[args.index("-D") + 1])
        (dest / "llk-toolchain-blackhole-1").mkdir(parents=True)
        (dest / "llk-toolchain-blackhole-1/toolchain.json").write_text('{"pin": "1"}')

    monkeypatch.setattr(toolchain_check, "_gh", gh)
    assert toolchain_check.download("o/r", {"id": 1}, tmp_path) == {
        "llk-toolchain-blackhole-1": {"pin": "1"}
    }
