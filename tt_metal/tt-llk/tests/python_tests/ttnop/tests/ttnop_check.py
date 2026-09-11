#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Check ttnop by arming a real cave detour on a live LLK kernel.

The baseline must pass. Each finding is then planted through `Injector.arm`
(the same L1 jal that a sweep writes) and classified the way Perturber.run
would: hang if the wait times out, mismatch if `passed_test` fails, drift if
the golden still passes but the tensor is not bit-exact vs the clean run.

    ./tests/ttnop_check.py
"""

import fcntl
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PYTHON_TESTS = HERE.parents[1]
sys.path[:0] = [str(HERE.parent), str(PYTHON_TESTS)]
os.environ.setdefault("LLK_HOME", str(PYTHON_TESTS.parents[1]))

import torch
from cave import Injector
from helpers.chip_architecture import get_chip_architecture
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, Tilize, format_dict
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_INDEX,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    TILE_COUNT,
    TILIZE,
    generate_input_dim,
)
from helpers.utils import passed_test
from loguru import logger
from sweep import describe_drift
from ttexalens import tt_exalens_init
from ttexalens.tt_exalens_lib import (
    read_word_from_device,
    write_words_to_device,
)

FORMAT = DataFormat.Float16_b
FORMATS = InputOutputFormat(FORMAT, FORMAT)
TILE = [32, 32]
TTI_NOP = 0x08000000
# .ttinsn ZEROSRC(..., src_mask=SrcA|SrcB): wipes unpacked operands.
ZEROSRC = 0x4400000C
# RV32 `jal x0, 0` makes the cave filler loop forever.
JAL_SELF = 0x0000006F


def setup():
    logger.disable("helpers")
    TestConfig.setup_build(Path(os.environ["LLK_HOME"]))
    TestConfig.setup_mode("master", False, False)
    TestConfig.create_build_directories()

    src_a, tiles_a, src_b, tiles_b = generate_stimuli(
        stimuli_format_A=FORMAT,
        input_dimensions_A=TILE,
        stimuli_format_B=FORMAT,
        input_dimensions_B=TILE,
    )
    return TestConfig(
        "sources/eltwise_unary_datacopy_test.cpp",
        FORMATS,
        templates=[generate_input_dim(TILE, TILE), TILIZE(Tilize.No)],
        runtimes=[
            DEST_INDEX(0),
            TILE_COUNT(tiles_a),
            NUM_FACES(4),
            NUM_BLOCKS(1),
            NUM_TILES_IN_BLOCK(1),
        ],
        variant_stimuli=StimuliConfig(
            src_a,
            FORMAT,
            src_b,
            FORMAT,
            FORMAT,
            tile_count_A=tiles_a,
            tile_count_B=tiles_b,
            tile_count_res=tiles_a,
            num_faces=4,
        ),
        dest_acc=DestAccumulation.No,
        unpack_to_dest=False,
    )


def as_tensor(values):
    return torch.as_tensor(values, dtype=format_dict[FORMAT])


def run(config):
    return as_tensor(config.run().result)


def retrigger(config):
    """Re-go the image already in L1. Does not rewrite stimuli, so a source poke sticks."""
    config.run_elf_files()
    config.wait_for_tensix_operations_finished()
    return as_tensor(config.variant_stimuli.collect_results(TestConfig.TENSIX_LOCATION))


def classify_result(baseline, result):
    if not passed_test(baseline, result, FORMAT):
        return "mismatch", ""
    drift, _ = describe_drift([baseline], [result])
    return ("drift", drift) if drift else (None, "")


def classify(config, baseline):
    try:
        return classify_result(baseline, run(config))
    except TimeoutError as error:
        return "hang", str(error)


def make_injector():
    location = TestConfig.TENSIX_LOCATION
    return Injector(
        read_words=lambda address, count: [
            read_word_from_device(location, address + 4 * i) for i in range(count)
        ],
        write_words=lambda address, words: write_words_to_device(
            location, address, words
        ),
    )


def inject(config, baseline, injector, thread, scan, site, payload):
    injector.arm(thread, scan, site, 1, payload)
    try:
        finding = classify(config, baseline)
    finally:
        injector.restore()
    if finding[0] != "hang":
        dirty, _ = describe_drift([baseline], [run(config)])
        assert not dirty, f"restore left {thread} {site.label()} dirty: {dirty}"
    return finding


def check_nops(config, baseline, injector, scans):
    for thread, scan in scans.items():
        for site in scan.sites:
            tag, detail = inject(
                config, baseline, injector, thread, scan, site, TTI_NOP
            )
            assert tag is None, f"{thread} {site.label()} -> {tag}: {detail}"
    return sum(len(scan.sites) for scan in scans.values())


def check_mismatch(config, baseline, injector, scans):
    """ZEROSRC wipes SrcA and SrcB, so the kernel must fail the golden."""
    scan = scans["unpack"]
    caught = [
        site
        for site in scan.sites
        if inject(config, baseline, injector, "unpack", scan, site, ZEROSRC)[0]
        == "mismatch"
    ]
    assert caught, "ZEROSRC in the cave was not reported as mismatch"
    return caught[0]


def check_drift(config, baseline, injector, scans):
    """A tti_nop detour plus a 1-bit source poke should report drift.

    A pure tti_nop cannot 1-ULP-corrupt this datacopy kernel. The cave is still
    armed, so the jal and filler still run. The 1-bit poke is what moves output.
    `config.run()` would rewrite stimuli from the host tensor, so this retriggers
    without that write.
    """
    scan = scans["unpack"]
    site = scan.sites[0]
    location = TestConfig.TENSIX_LOCATION
    addr = config.variant_stimuli.buf_a_addr
    original = read_word_from_device(location, addr)
    injector.arm("unpack", scan, site, 1, TTI_NOP)
    write_words_to_device(location, addr, [original ^ 1])
    try:
        result = retrigger(config)
    finally:
        write_words_to_device(location, addr, [original])
        injector.restore()
    tag, _ = classify_result(baseline, result)
    assert tag == "drift", f"expected drift, got {tag}"
    dirty, _ = describe_drift([baseline], [run(config)])
    assert not dirty, f"restore left drift dirty: {dirty}"
    moved, pcc = describe_drift([baseline], [result])
    return int((baseline != result).sum()), pcc, site


def check_hang(config, baseline, injector, scans):
    """`jal x0, 0` as the cave filler never returns. Expect the 2s TimeoutError.

    Restoring the instruction does not unwedge the core on this card, which
    is why the plugin abandons a hung core. This check needs --hang and runs last.
    """
    scan = scans["math"]
    site = scan.sites[0]
    injector.arm("math", scan, site, 1, JAL_SELF)
    try:
        tag, detail = classify(config, baseline)
    finally:
        injector.restore()
    assert tag == "hang", f"self-jump produced {tag}: {detail}"
    return site, detail


def reset_after_hang():
    """`tt-smi -r` is what actually clears a jal-self hang on this card."""
    print("running tt-smi -r (that is what clears the hang)")
    try:
        subprocess.run(["tt-smi", "-r"], check=True)
        print("    tt-smi -r done. Check complete.")
    except Exception as err:
        print(f"    tt-smi -r failed ({err}). Run it by hand")


def main():
    tt_exalens_init.init_ttexalens()
    if "CHIP_ARCH" not in os.environ:
        get_chip_architecture()
    lock = open(f"/tmp/tt-llk-test-{os.environ['CHIP_ARCH']}.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX)

    config = setup()
    try:
        baseline = run(config)
    except TimeoutError:
        print("PLEASE DO tt-smi -r  (core is hung from a previous run)")
        raise
    import scanner

    scans = {
        thread: scanner.scan(elf, "sync")
        for thread, elf in zip(TestConfig.KERNEL_COMPONENTS, config.temp_elfs)
    }
    assert all(scan.sites for scan in scans.values())
    print(f"ok  baseline  {baseline.numel()} elements")
    print(f"ok  scan      {sum(len(scan.sites) for scan in scans.values())} sites")

    injector = make_injector()
    print(
        f"ok  nops      {check_nops(config, baseline, injector, scans)} clean restores"
    )
    site = check_mismatch(config, baseline, injector, scans)
    print(f"ok  mismatch  ZEROSRC at unpack {site.label()}")
    changed, pcc, site = check_drift(config, baseline, injector, scans)
    print(
        f"ok  drift     tti_nop at unpack {site.label()}, "
        f"{changed} element(s), pcc {pcc:.9f}"
    )
    if "--hang" not in sys.argv:
        print("skip hang      ./tests/ttnop_check.py --hang  (jal-self wedges a core)")
        return 0
    try:
        site, detail = check_hang(config, baseline, injector, scans)
        print(f"ok  hang      jal-self at math {site.label()}: {detail}")
    finally:
        reset_after_hang()
    return 0


if __name__ == "__main__":
    sys.exit(main())
