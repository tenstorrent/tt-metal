# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Whole-DRAM, all-bank read/write screen for silent wrong-address reads, with an optional co-located matmul load.

Every 4 B word holds a hash of its own bank, bank-local byte address and a per-launch seed, so the data is unique per
address and differs from launch to launch. A read that returns another row's data, stale data from an earlier
launch, or flipped bits all miscompare, and no fixed-period pattern can hide an alias the way walking-1s or
checkerboard does. A miscompared page is classified once: stale (every wrong byte equals the previous launch's data),
alias of bit k (every wrong byte equals the data of the page at A ^ (1 << k); for a source below the buffer, the data
read back from there), or other. A command/address fault shows
up as one hot bit in the alias histogram plus one hot (bank, byte lane) cell; random corruption lands in "other".

Each launch is a fill program followed by a check program on every selected chip at once; launch boundaries are the
only global barrier. An optional final "retention" launch leaves the chips idle between its fill and its check, so
every word is read back that long after it was written. Arms alternate per launch inside one device open, so "on" (matmul on the TRISCs of every reading core) and "off" share the
same training. GDDR self-refresh is entered only while no process holds the device, and CA is retrained on its exit,
so sweeping trainings means running this test as separate processes with an idle gap between them.
"""

import json
import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn

BANKS = 8
PAGE_BYTES = 2048
PAGE_WORDS = PAGE_BYTES // 4
ROW_WORDS = 128
LANE_OFF = 16
ALIAS_OFF = LANE_OFF + 4 * BANKS
MAGIC = 0xC0FFEE
NONE = 0xFFFFFFFF
KT = 8
CB_IN0, CB_IN1, CB_OUT = 0, 1, 16
CB_SCRATCH = (2, 3)

REPO = Path(__file__).resolve().parents[5]
OUT_DIR = Path(os.environ.get("STRESS_OUT_DIR", REPO / "repro_logs" / "sweep"))

KERNEL_DIR = "ttnn/cpp/ttnn/operations/experimental/diagnostics/dram/kernels"
DM_KERNEL = f"{KERNEL_DIR}/sweep_dm.cpp"
COMPUTE_KERNEL = f"{KERNEL_DIR}/matmul_compute.cpp"
# The kernels read these layout constants as defines, so host and device share one definition.
MM_DEFINES = [("CB_IN0", str(CB_IN0)), ("CB_IN1", str(CB_IN1)), ("CB_OUT", str(CB_OUT)), ("MM_KT", str(KT))]
DM_DEFINES = MM_DEFINES + [
    ("ALIAS_OFF", str(ALIAS_OFF)),
    ("LANE_OFF", str(LANE_OFF)),
    ("NO_ADDR", f"{NONE:#x}u"),
    ("OUT_MAGIC", str(MAGIC)),
    ("PAGE_BYTES", str(PAGE_BYTES)),
    ("PAGE_WORDS", str(PAGE_WORDS)),
    ("ROW_WORDS", str(ROW_WORDS)),
    ("SWEEP_BANKS", str(BANKS)),
]



def env_int(name, default):
    return int(os.environ.get(name, str(default)), 0)


def allocate_buffer(device, bank_mb):
    while bank_mb >= 64:
        rows = bank_mb * (1 << 20) // PAGE_BYTES * BANKS
        try:
            t = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, rows, PAGE_WORDS]),
                ttnn.uint32,
                ttnn.ROW_MAJOR_LAYOUT,
                device,
                ttnn.DRAM_MEMORY_CONFIG,
            )
            return t, rows, bank_mb
        except Exception as e:
            logger.warning(f"{bank_mb} MB per bank does not fit ({str(e).splitlines()[0]}), trying less")
            bank_mb -= 8
    raise RuntimeError("could not allocate a DRAM sweep buffer")


def build_programs(buf, tt_out, cores, grid, rows, chunk, writeback=0):
    slots = 2 * len(cores)
    bounds = [(s * rows // slots, (s + 1) * rows // slots) for s in range(slots)]
    scratch_bytes = ROW_WORDS * 4 + (2 * chunk + 1) * PAGE_BYTES
    cbs = [
        ttnn.CBDescriptor(
            total_size=scratch_bytes,
            core_ranges=grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.uint32, page_size=scratch_bytes)
            ],
        )
        for i in CB_SCRATCH
    ]
    for idx, tiles in ((CB_IN0, KT), (CB_IN1, KT), (CB_OUT, 1)):
        cbs.append(
            ttnn.CBDescriptor(
                total_size=tiles * 2048,
                core_ranges=grid,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.bfloat16, page_size=2048)
                ],
            )
        )
    accessor_args = (
        ttnn.TensorAccessorArgs(buf).get_compile_time_args() + ttnn.TensorAccessorArgs(tt_out).get_compile_time_args()
    )

    def dm_kernels(fill, args_fn):
        kernels = []
        for risc, (processor, noc) in enumerate(
            ((ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0), (ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1))
        ):
            rt = ttnn.RuntimeArgs()
            for i, (x, y) in enumerate(cores):
                s = 2 * i + risc
                rt[x][y] = args_fn(s, *bounds[s])
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=DM_KERNEL,
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    defines=DM_DEFINES,
                    core_ranges=grid,
                    compile_time_args=[CB_SCRATCH[risc], chunk, int(fill), int(risc == 0)] + accessor_args,
                    runtime_args=rt,
                    config=ttnn.DataMovementConfigDescriptor(processor=processor, noc=noc),
                )
            )
        return kernels

    def make(fill, compute_blocks, seed, passes, reps, inj, decode_max):
        inj_page, inj_bit, inj_mask = inj

        def args(s, p0, p1):
            return [
                buf.buffer_address(),
                tt_out.buffer_address(),
                s,
                p0,
                p1,
                seed,
                passes,
                reps,
                int(compute_blocks > 0),
                inj_page,
                inj_bit,
                inj_mask,
                decode_max,
                writeback,
            ]

        kernels = dm_kernels(fill, args)
        if compute_blocks > 0:
            rt = ttnn.RuntimeArgs()
            for x, y in cores:
                rt[x][y] = [compute_blocks]
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=COMPUTE_KERNEL,
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    defines=MM_DEFINES,
                    core_ranges=grid,
                    compile_time_args=[],
                    runtime_args=rt,
                    config=ttnn.ComputeConfigDescriptor(),
                )
            )
        return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)

    return make, bounds


def read_out(tt_out, slots, seed):
    out = ttnn.to_torch(tt_out).reshape(slots, ROW_WORDS).to(torch.int64) & 0xFFFFFFFF
    assert (out[:, 15] == MAGIC).all(), "a RISC did not report"
    assert (out[:, 0] == seed).all(), f"stale runtime args: expected seed {seed}, got {out[:, 0].unique().tolist()}"
    return out


def summarize(out):
    return {
        "bad_pages": int(out[:, 2].sum()),
        "bad_words": int(out[:, 3].sum()),
        "stale": int(out[:, 4].sum()),
        "other": int(out[:, 5].sum()),
        "undecoded": int(out[:, 6].sum()),
        "alias_src_outside": int(out[:, 13].sum()),
        "lanes": out[:, LANE_OFF:ALIAS_OFF].sum(0).reshape(BANKS, 4).tolist(),
        "alias_bits": {k: int(v) for k, v in enumerate(out[:, ALIAS_OFF : ALIAS_OFF + 32].sum(0).tolist()) if v},
    }


class ChipRun:
    def __init__(self, chip, device, cfg):
        gx, gy = cfg["grid"]
        self.chip, self.cfg = chip, cfg
        self.cores = [(x, y) for y in range(gy) for x in range(gx)]
        self.slots = 2 * len(self.cores)
        grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
        self.blocks = cfg["blocks"]
        arms = cfg["arms"] + (["retention"] if cfg["retention_s"] else [])
        self.per_arm = {a: {"launches": 0, "word_reads": 0, "bad_words": 0, "first": None} for a in arms}
        self.tt_out = ttnn.from_torch(
            torch.zeros(1, 1, self.slots, ROW_WORDS, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.buf, rows, self.bank_mb = allocate_buffer(device, cfg["bank_mb"])
        self.base = self.buf.buffer_address()
        self.make, self.bounds = build_programs(
            self.buf, self.tt_out, self.cores, grid, rows, cfg["chunk"], cfg["writeback"]
        )
        logger.info(
            f"chip {chip}: {self.bank_mb} MB/bank x {BANKS} banks at bank offset "
            f"{self.base:#x}..{self.base + rows // BANKS * PAGE_BYTES:#x}, {len(self.cores)} cores x 2 RISCs, "
            f"passes {cfg['passes']} reps {cfg['reps']}, arms {cfg['arms']}"
        )

    def run(self, fill, blocks, seed, passes, reps, inj=(NONE, 0, 0)):
        ttnn.generic_op([self.buf, self.tt_out], self.make(fill, blocks, seed, passes, reps, inj, self.cfg["decode_max"]))

    def self_test(self, seed):
        inj_page = 3
        self.run(True, 0, seed, 1, 1, (inj_page, 26, 0xFF000000))
        read_out(self.tt_out, self.slots, seed)
        self.run(False, 0, seed, 1, 1)
        s = summarize(read_out(self.tt_out, self.slots, seed))
        # The top byte of a word and of its alias match for a whole 64 B line at once, so lines of 16 words may be missing.
        assert (
            s["bad_pages"] == 1 and s["bad_words"] % 16 == 0 and s["bad_words"] >= PAGE_WORDS // 2
        ), f"chip {self.chip} self-test: {s}"
        want = [[0] * 4 for _ in range(BANKS)]
        want[inj_page % BANKS][3] = s["bad_words"]
        assert s["lanes"] == want, f"chip {self.chip} self-test lanes: {s['lanes']}"
        assert s["alias_bits"] == {26: 1} and s["stale"] == s["other"] == 0, f"chip {self.chip} self-test: {s}"

    def check(self, seed, arm):
        self.run(False, self.blocks if arm == "on" else 0, seed, self.cfg["passes"], self.cfg["reps"])

    def record(self, out, launch, arm):
        cfg, slots = self.cfg, self.slots
        done = out[:, 1]
        a = self.per_arm[arm]
        a["launches"] += 1
        a["word_reads"] += sum(
            int(done[s]) * cfg["reps"] * (self.bounds[s][1] - self.bounds[s][0]) * PAGE_WORDS for s in range(slots)
        )
        s = summarize(out)
        a["bad_words"] += s["bad_words"]
        for k in ("stale", "other", "undecoded", "alias_src_outside"):
            a[k] = a.get(k, 0) + s[k]
        for k, v in s["alias_bits"].items():
            a.setdefault("alias_bits", {})[k] = a.get("alias_bits", {}).get(k, 0) + v
        lanes = a.setdefault("lanes", [[0] * 4 for _ in range(BANKS)])
        for b in range(BANKS):
            for l in range(4):
                lanes[b][l] += s["lanes"][b][l]
        if s["bad_words"]:
            bad = [r for r in range(slots) if out[r, 3]]
            r0 = out[bad[0]]
            page = int(r0[7])
            first = {
                "launch": launch,
                "slot": bad[0],
                "core": self.cores[bad[0] // 2],
                "bank": page % BANKS,
                "addr": hex(self.base + page // BANKS * PAGE_BYTES + 4 * int(r0[8])),
                "got": hex(int(r0[9])),
                "exp": hex(int(r0[10])),
                "pass": int(r0[11]),
                "rep": int(r0[12]),
                "slots_bad": len(bad),
            }
            a["first"] = a["first"] or first
            logger.error(f"chip {self.chip} launch {launch} arm {arm}: {s} first {first}")
        if arm == "on":
            if int(done.min()) <= cfg["passes"]:
                # The kernel takes blocks as a uint32 runtime arg.
                self.blocks = min(self.blocks * 4, 1 << 30)
                logger.info(f"chip {self.chip}: matmul ended before the reads on some core, blocks -> {self.blocks}")
            elif int(done.min()) > 4 * cfg["passes"]:
                self.blocks = max(1, self.blocks // 2)
        return s["bad_words"], int(done.min()), int(done.max())

    def result(self):
        for arm, a in self.per_arm.items():
            logger.info(
                f"chip {self.chip} arm {arm}: {a['launches']} launches, {a['word_reads']:.3e} word reads, "
                f"bad words {a['bad_words']} stale {a.get('stale', 0)} other {a.get('other', 0)} "
                f"alias bits {a.get('alias_bits', {})}"
            )
        return {"chip": self.chip, "bank_mb": self.bank_mb, "base": hex(self.base), "blocks": self.blocks, **self.per_arm}


def run_chips(chips, cfg):
    """All chips run each fill and each check together, so the whole machine is under load for the full run."""
    arms = cfg["arms"]
    seed = cfg["seed0"]
    devices = ttnn.CreateDevices(chips)
    try:
        runs = [ChipRun(c, devices[c], cfg) for c in chips]
        if cfg["self_test"]:
            seed += 2
            for r in runs:
                r.self_test(seed)
            logger.info(f"self-test: planted bank 3 lane 3 alias of bit 26 decoded exactly on {len(runs)} chips")

        t_end = time.monotonic() + cfg["seconds"]
        launch = 0
        while launch < cfg["launches"] or time.monotonic() < t_end:
            arm = arms[launch % len(arms)]
            seed += 1
            t0 = time.perf_counter()
            # generic_op only enqueues; reading back one chip's output leaves the others running.
            for r in runs:
                r.run(True, 0, seed, 1, 1)
            for r in runs:
                read_out(r.tt_out, r.slots, seed)
            t1 = time.perf_counter()
            for r in runs:
                r.check(seed, arm)
            outs = [read_out(r.tt_out, r.slots, seed) for r in runs]
            t2 = time.perf_counter()
            stats = [r.record(out, launch, arm) for r, out in zip(runs, outs)]
            if launch < 2 * len(arms) or launch % 10 == 0:
                logger.info(
                    f"launch {launch} arm {arm} on {len(runs)} chips: fill {t1 - t0:.2f}s check {t2 - t1:.2f}s, "
                    f"passes/RISC {min(s[1] for s in stats)}..{max(s[2] for s in stats)}, "
                    f"bad words {sum(s[0] for s in stats)}"
                )
            launch += 1

        if cfg["retention_s"]:
            seed += 1
            for r in runs:
                r.run(True, 0, seed, 1, 1)
            for r in runs:
                read_out(r.tt_out, r.slots, seed)
            time.sleep(cfg["retention_s"])
            t0 = time.perf_counter()
            for r in runs:
                r.check(seed, "retention")
            outs = [read_out(r.tt_out, r.slots, seed) for r in runs]
            stats = [r.record(out, launch, "retention") for r, out in zip(runs, outs)]
            logger.info(
                f"retention launch {launch} on {len(runs)} chips: idle {cfg['retention_s']}s after fill, "
                f"check {time.perf_counter() - t0:.2f}s, bad words {sum(s[0] for s in stats)}"
            )
        for r in runs:
            ttnn.deallocate(r.buf)
    finally:
        ttnn.CloseDevices(devices)
    return [r.result() for r in runs]


@pytest.mark.timeout(0)
def test_dram_sweep_stress():
    chips_env = os.environ.get("STRESS_CHIPS", "all")
    chips = list(range(ttnn.GetNumAvailableDevices())) if chips_env == "all" else [int(c) for c in chips_env.split(",")]
    gx, gy = (int(v) for v in os.environ.get("STRESS_GRID", "12x10").split("x"))
    cfg = {
        "grid": (gx, gy),
        # The allocator steps down from here to the largest size that fits.
        "bank_mb": env_int("STRESS_BANK_MB", 4096),
        "arms": os.environ.get("STRESS_ARMS", "off,on").split(","),
        "launches": env_int("STRESS_LAUNCHES", 4),
        "seconds": env_int("STRESS_SECONDS", 0),
        "passes": env_int("STRESS_PASSES", 1),
        "reps": env_int("STRESS_REPS", 4),
        "chunk": env_int("STRESS_CHUNK", 16),
        "blocks": env_int("STRESS_BLOCKS", 4000000),
        "decode_max": env_int("STRESS_DECODE_MAX", 4096),
        "seed0": env_int("STRESS_SEED", int(time.time()) & 0xFFFF) * 1024,
        "self_test": env_int("STRESS_SELF_TEST", 1),
        "retention_s": env_int("STRESS_RETENTION_S", 0),
        # 1: the check phase writes every page back right after verifying it, so reads and writes interleave.
        "writeback": env_int("STRESS_WRITEBACK", 0),
    }
    assert all(a in ("on", "off") for a in cfg["arms"]), cfg["arms"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"chips {chips}, config {cfg}")
    results = run_chips(chips, cfg)
    (OUT_DIR / f"sweep_{os.environ.get('SLURM_JOB_ID', 'local')}.json").write_text(
        json.dumps({"config": cfg, "results": results}, indent=1, default=str)
    )
    bad = [(r["chip"], arm) for r in results for arm in r if isinstance(r[arm], dict) and r[arm]["bad_words"]]
    assert not bad, f"miscompares on (chip, arm): {bad}"
