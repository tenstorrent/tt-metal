# Experiment only: which recovery step makes a core usable after a Tensix stall hang (HANG_VARIANTS, tried in order).
import os

import pytest
from helpers import device as dev
from helpers.test_config import BuildMode, TestConfig
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.debug_tensix import TensixDebug
from ttexalens.tt_exalens_lib import check_context


def _launch(source, timeout=2):
    cfg = TestConfig(f"sources/hang_{source}_test.cpp")
    cfg.generate_variant_hash()
    cfg.build_elfs()
    cfg.run_elf_files()
    cfg.wait_for_tensix_operations_finished(timeout=timeout)


def variant(name):
    def f(location="0,0", device_id=0):
        store = dev.get_register_store(location, device_id)
        dt = TensixDebug(OnChipCoordinate.create(location, device=check_context().devices[device_id]))
        ops = dt.device.instructions
        reg = "RISCV_DEBUG_REG_SOFT_RESET_0"
        held = store.read_register(reg) | dev.get_soft_reset_mask(dev.ALL_CORES)
        store.write_register(reg, held)

        def pulse(mask):
            store.write_register(reg, held | mask)
            for _ in range(16):
                store.read_register(reg)
            store.write_register(reg, held)
            for _ in range(16):
                store.read_register(reg)

        pipe = dev._PIPELINE_RESET_MASK
        if name == "dvalid":
            dt.inject_instruction(ops.TT_OP_SETDVALID(0b11), 0)
        elif name == "dvalid_clear":
            dt.inject_instruction(ops.TT_OP_SETDVALID(0b11), 0)
            dt.inject_instruction(ops.TT_OP_CLEARDVALID(0b11, 0), 0)
        elif name == "pulse_only":
            pulse(pipe)
        elif name == "pulse_full":
            dt.inject_instruction(ops.TT_OP_SEMINIT(15, 0, 0xFF), 0)
            pulse(pipe | (1 << 23))
        elif name == "dvalid_pulse_full":
            dt.inject_instruction(ops.TT_OP_SEMINIT(15, 0, 0xFF), 0)
            dt.inject_instruction(ops.TT_OP_SETDVALID(0b11), 0)
            dt.inject_instruction(ops.TT_OP_CLEARDVALID(0b11, 0), 0)
            pulse(pipe | (1 << 23))
        else:
            raise ValueError(name)

    return f


def test_hang_ladder():
    if TestConfig.BUILD_MODE == BuildMode.PRODUCE:
        pytest.skip()
    orig = dev.recover_tensix_core
    try:
        for name in os.environ["HANG_VARIANTS"].split(","):
            dev.recover_tensix_core = orig if name == "prod" else variant(name)
            try:
                _launch("stall")
            except TimeoutError:
                pass
            try:
                _launch("ok")
                _launch("ok")
                print(f"LADDER {name}: OK", flush=True)
                break
            except TimeoutError:
                print(f"LADDER {name}: HANG", flush=True)
    finally:
        dev.recover_tensix_core = orig
