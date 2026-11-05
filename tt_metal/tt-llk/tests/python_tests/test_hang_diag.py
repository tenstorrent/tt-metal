# Experiment only: step by step hang/recovery sequence, SEQ (comma list of stall/spin/ok) and REPEAT recovery applications.
import os

import pytest
from helpers import device as dev
from helpers.test_config import BuildMode, TestConfig


def _launch(source, timeout=2):
    cfg = TestConfig(f"sources/hang_{source}_test.cpp")
    cfg.generate_variant_hash()
    cfg.build_elfs()
    cfg.run_elf_files()
    cfg.wait_for_tensix_operations_finished(timeout=timeout)


def test_hang_diag():
    if TestConfig.BUILD_MODE == BuildMode.PRODUCE:
        pytest.skip()
    orig = dev.recover_tensix_core
    repeat = int(os.environ.get("REPEAT", "1"))

    def rec(location="0,0", device_id=0):
        for _ in range(repeat):
            orig(location, device_id)

    dev.recover_tensix_core = rec
    try:
        for i, step in enumerate(os.environ["SEQ"].split(",")):
            try:
                _launch(step)
                print(f"DIAG repeat={repeat} step{i} {step}: finished", flush=True)
            except TimeoutError:
                print(f"DIAG repeat={repeat} step{i} {step}: TIMEOUT", flush=True)
    finally:
        dev.recover_tensix_core = orig
