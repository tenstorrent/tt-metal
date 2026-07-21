# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
pytest --compile-producer python_tests/test_risc_compute.py
That writes:
    /tmp/tt-llk-build/shared/elf/brisc.elf
    /tmp/tt-llk-build/sources/risc_compute_test.cpp/<variant>/elf/{unpack,math,pack}.elf
The runtime args blob is built in this script.

python repro.py --cores 15 --iterations 500
"""

import argparse
import multiprocessing as mp
import os
import struct
import sys
import time
from pathlib import Path

# Informed by helpers/device.py and helpers/llk_params.py.
SOFT_RESET_REG = "RISCV_DEBUG_REG_SOFT_RESET_0"

# Informed by helpers/device.py:RiscCore.
BRISC_BIT = 11
TRISC_BITS = [12, 13, 14]  # TRISC0/1/2
ALL_CORE_BITS = [BRISC_BIT] + TRISC_BITS

# Informed by helpers/llk_params.py:Mailboxes.
MB_BASE = 0x1FFB8
MB_UNPACKER = MB_BASE + 0
MB_MATH = MB_BASE + 4
MB_PACKER = MB_BASE + 8
MB_BRISC_CMD0 = MB_BASE + 12
MB_BRISC_CMD1 = MB_BASE + 16
MB_BRISC_COUNTER = MB_BASE + 20
TRISC_MAILBOXES = [MB_UNPACKER, MB_MATH, MB_PACKER]

BRISC_BOOT_READY_SENTINEL = 0xB001CAFE
KERNEL_COMPLETE = 0xFF

# Informed by helpers/llk_params.py:BriscCmd.
CMD_IDLE = 0
CMD_START_TRISCS = 1
CMD_RESET_TRISCS = 2
CMD_UPDATE_START_ADDR_CACHE_AND_START = 3

# helpers/test_config.py
TRISC_START_ADDRS = [0x16DFF0, 0x16DFF4, 0x16DFF8]
RUNTIME_ARGS_ADDR = 0x20000

# TestConfig.DEFAULT_ARTEFACTS_PATH / SHARED_ELF_DIR.
ARTEFACTS = Path(os.environ.get("RUNNER_TEMP", "/tmp")) / "tt-llk-build"
SHARED_ELF_DIR = ARTEFACTS / "shared" / "elf"
VARIANT_ELF_ROOT = ARTEFACTS / "sources" / "risc_compute_test.cpp"


def build_runtime_args() -> bytes:
    """
    Build the runtime args struct for the pinned Int32 [32,96] variant.
    Informed by TestConfig.write_runtimes_to_L1 and StimuliConfig.generate_runtime_operands_values.
    """

    TILE = 128
    INT32_ENUM = 8
    TILE_BYTES = 0x1000
    buf_a, buf_b, buf_res = 0x21000, 0x24000, 0x27000
    args = struct.pack(
        "@" + "III" + "I" * 12 + "IIIIII",
        TILE,
        TILE,
        TILE,
        *([INT32_ENUM] * 12),
        buf_a,
        TILE_BYTES,
        buf_b,
        TILE_BYTES,
        buf_res,
        TILE_BYTES,
    )
    assert len(args) == 84, len(args)
    return args


def find_variant_elf_dir() -> Path:
    if not VARIANT_ELF_ROOT.is_dir():
        raise SystemExit(
            f"Missing ELFs under {VARIANT_ELF_ROOT}.\n"
            "First run pytest --compile-producer test_risc_compute.py.\n"
        )
    elf_dirs = sorted(VARIANT_ELF_ROOT.glob("*/elf"))
    if len(elf_dirs) != 1:
        raise SystemExit(
            f"Expected exactly one variant ELF dir under {VARIANT_ELF_ROOT}, "
            f"found {len(elf_dirs)}: {elf_dirs}. Clean {ARTEFACTS} and re-run the producer."
        )
    return elf_dirs[0]


def _make_helpers():
    from ttexalens import check_context
    from ttexalens.coordinate import OnChipCoordinate
    from ttexalens.tt_exalens_lib import (
        load_elf,
        read_word_from_device,
        write_to_device,
        write_words_to_device,
    )

    def register_store(location, device_id=0):
        ctx = check_context()
        dev = ctx.devices[device_id]
        coord = OnChipCoordinate.create(location, device=dev)
        return dev.get_block(coord).get_register_store()

    def soft_reset_mask(bits):
        return sum(1 << b for b in bits)

    def set_soft_reset(value, bits, location):
        rs = register_store(location)
        val = rs.read_register(SOFT_RESET_REG)
        if value:
            val |= soft_reset_mask(bits)
        else:
            val &= ~soft_reset_mask(bits)
        rs.write_register(SOFT_RESET_REG, val)

    def commit_soft_reset(value, bits, location, timeout=0.1):
        set_soft_reset(value, bits, location)
        rs = register_store(location)
        target = rs.read_register(SOFT_RESET_REG)
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if rs.read_register(SOFT_RESET_REG) == target:
                return
        raise TimeoutError("soft reset not committed")

    return dict(
        load_elf=load_elf,
        read_word=read_word_from_device,
        write_to=write_to_device,
        write_words=write_words_to_device,
        set_soft_reset=set_soft_reset,
        commit_soft_reset=commit_soft_reset,
    )


def worker(location: str, iterations: int, elf_dir: str, result_q: mp.Queue):
    """One process works on one physical Tensix core."""
    from ttexalens import tt_exalens_init

    tt_exalens_init.init_ttexalens()
    h = _make_helpers()
    runtime_args = build_runtime_args()

    brisc_elf = str(SHARED_ELF_DIR / "brisc.elf")
    trisc_elfs = [str(Path(elf_dir) / f"{n}.elf") for n in ("unpack", "math", "pack")]

    cmd_counter = 0

    def commit_brisc_command(command, timeout=1.0):
        nonlocal cmd_counter
        slot = MB_BRISC_CMD1 if (cmd_counter & 1) else MB_BRISC_CMD0
        h["write_words"](location, slot, [command])
        cmd_counter += 1
        end = time.time() + timeout
        while time.time() < end:
            if h["read_word"](location, MB_BRISC_COUNTER) == cmd_counter:
                return
        raise TimeoutError(f"brisc command {command} timed out")

    # run_elf_files first load path.
    h["commit_soft_reset"](1, ALL_CORE_BITS, location)
    h["load_elf"](brisc_elf, location, "brisc", verify_write=True)
    h["write_words"](location, MB_BRISC_COUNTER, [0])
    h["commit_soft_reset"](0, [BRISC_BIT], location)
    end = time.time() + 1.0
    while time.time() < end:
        if h["read_word"](location, MB_BRISC_COUNTER) == BRISC_BOOT_READY_SENTINEL:
            break
    else:
        result_q.put((location, "boot-fail", -1, "BRISC never signalled boot-ready"))
        return

    # Reload ELFs every iteration.
    for i in range(iterations):
        try:
            h["write_to"](location, RUNTIME_ARGS_ADDR, runtime_args)
            commit_brisc_command(CMD_RESET_TRISCS)  # BRISC puts TRISCs in reset
            for idx, elf in enumerate(trisc_elfs):
                start = h["load_elf"](  # bug fires here
                    elf,
                    location,
                    f"trisc{idx}",
                    return_start_address=True,
                    verify_write=False,
                )
                h["write_words"](location, TRISC_START_ADDRS[idx], [start])
            commit_brisc_command(CMD_UPDATE_START_ADDR_CACHE_AND_START)

            # Poll for completion.
            completed = set()
            deadline = time.time() + 2.0
            while time.time() < deadline:
                for mb in TRISC_MAILBOXES:
                    if (
                        mb not in completed
                        and h["read_word"](location, mb) == KERNEL_COMPLETE
                    ):
                        completed.add(mb)
                if len(completed) == len(TRISC_MAILBOXES):
                    break
            else:
                result_q.put((location, "timeout", i, "mailbox poll timeout"))
                return
        except Exception as e:
            result_q.put(
                (location, type(e).__name__, i, str(e).strip().splitlines()[0])
            )
            return

    result_q.put((location, "clean", iterations, ""))


def core_locations(n: int) -> list[str]:
    """TestConfig.setup_mode worker->location mapping."""
    return [f"{k // 8},{k % 8}" for k in range(n)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cores", type=int, default=15, help="concurrent cores (pytest -n)"
    )
    ap.add_argument("--iterations", type=int, default=500, help="kernel runs per core")
    args = ap.parse_args()

    elf_dir = str(find_variant_elf_dir())
    locations = core_locations(args.cores)
    print(
        f"{args.cores} cores x {args.iterations} iterations | elf_dir={elf_dir}",
        flush=True,
    )

    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    procs = [
        ctx.Process(target=worker, args=(loc, args.iterations, elf_dir, q))
        for loc in locations
    ]
    for p in procs:
        p.start()

    results = [q.get() for _ in procs]
    for p in procs:
        p.join()

    failures = [r for r in results if r[1] not in ("clean",)]
    for loc, kind, i, msg in sorted(results):
        tag = "OK  " if kind == "clean" else "FAIL"
        print(f"{tag} core={loc:>4} kind={kind:<14} iter={i} {msg}", flush=True)

    reset_fires = [r for r in failures if "not in reset" in r[3].lower()]
    print(
        f"\n{len(failures)}/{len(results)} cores failed, "
        f"{len(reset_fires)} with 'not in reset'",
        flush=True,
    )
    sys.exit(1 if reset_fires else 0)


if __name__ == "__main__":
    main()
