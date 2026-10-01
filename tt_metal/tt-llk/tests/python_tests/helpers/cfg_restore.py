# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Host-side Tensix CFG (THCON) register snapshot/restore for the reconfig-escape pair sweep.

The premise: a correct kernel's init must (re)write every config register it depends on. To
test that, we capture a real op X's actual post-execution CFG residue, then replay that exact
residue in-kernel before a victim op K runs. Anything K rewrites is harmless; anything it
silently relies on (a reset/leftover value it never sets) stays as X left it and K either
miscomputes (PCC fail) or hangs.

This works because the CFG space is NOT reset at kernel launch: firmware boot only flips the
shadow id (`reset_cfg_state_id`) and zeroes PRNG_SEED, so values written while the TRISCs are
held in reset persist into kernel execution. See `run_elf_files()` for the injection point.

Access mechanism — important:
  The CFG register file at TENSIX_CFG_BASE (0xFFEF0000) is *core-private* address
  space, NOT directly NOC-addressable from the host. A raw read/write to
  0xFFEF0000 over the NOC times out. The only host path is the RISC debug
  private-memory interface (`risc_debug.read_memory`/`write_memory`) — the same
  path ttexalens' `get_tensix_state` uses to read CFG. We drive it through one
  TRISC; CFG is shared Tensix config, so a single core's private view reaches the
  whole register file. `ensure_private_memory_access()` transparently handles a
  core held in reset (it briefly releases it into an injected `JAL x0,0` loop,
  halts it, performs the accesses, then restores the saved code word and puts the
  core back in reset). All reads/writes happen inside one such session, so the
  core is released/halted/restored exactly once — not per register.

Memory map (Blackhole; Wormhole analogous with CFG_STATE_SIZE=47):
  - TENSIX_CFG_BASE = 0xFFEF0000.
  - The thread-config region is double-buffered into two shadow "states", each
    spanning CFG_STATE_SIZE 128-bit entries == CFG_STATE_SIZE*4 32-bit words
    (`stride`). Register `addr32` lives at word `addr32` in state 0 and
    `addr32 + stride` in state 1. `reset_cfg_state_id()` forces state 0 active
    before a kernel runs, so this module only ever reads/restores state 0.
  - Every CFG register defined in cfg_defines.h fits within a single state
    (highest addr32 is 222 on BH / 186 on WH, both below CFG_STATE_SIZE*4), so the
    "thread" group below genuinely covers the whole config register space.
"""

import json
import os
import sys

from ttexalens.tt_exalens_lib import check_context, convert_coordinate, get_tensix_state

from .chip_architecture import ChipArchitecture, get_chip_architecture
from .device_io import write_words_to_device
from .logger import logger

TENSIX_CFG_BASE = 0xFFEF0000

# CFG_STATE_SIZE counts 128-bit entries (from cfg_defines.h). Shadow-state stride
# in 32-bit words is CFG_STATE_SIZE * 4.
_CFG_STATE_SIZE = {
    ChipArchitecture.BLACKHOLE: 56,
    ChipArchitecture.WORMHOLE: 47,
}

# addr32 words that boot configures (NOT the kernel) and that kernels depend on —
# polluting them is over-reach (it breaks boot, not kernel compute-init), so the
# whole-space "thread" sweep excludes them. Matches boot.h::device_setup per arch:
#   - Blackhole: device_setup writes NO CFG-space register (only the 0xFFB12xxx debug
#     block + TTI instructions), so nothing is excluded.
#   - Wormhole: device_setup writes the TRISC reset-PC vectors — TRISC_RESET_PC_SEC0/1/2
#     (addr32 158/159/160) and RESET_PC_OVERRIDE (161). Trampling those wedges the boot PC.
_BOOT_OWNED_ADDR32 = {
    ChipArchitecture.BLACKHOLE: set(),
    ChipArchitecture.WORMHOLE: {158, 159, 160, 161},
    ChipArchitecture.QUASAR: set(),
}


def _thread_words(arch: ChipArchitecture) -> list[int]:
    excluded = _BOOT_OWNED_ADDR32.get(arch, set())
    return [a for a in range(0, _CFG_STATE_SIZE[arch] * 4) if a not in excluded]


# TRISC whose debug private-memory interface we drive to reach CFG. CFG is shared
# Tensix config, so any TRISC's private view works.
_ACCESS_RISC = "trisc0"


def _state_stride_words(arch: ChipArchitecture) -> int:
    return _CFG_STATE_SIZE[arch] * 4


def _word_core_addr(addr32: int, state: int, arch: ChipArchitecture) -> int:
    """Core-private CFG address of `addr32` in the given shadow state."""
    return TENSIX_CFG_BASE + (addr32 + state * _state_stride_words(arch)) * 4


def _get_risc_debug(location: str, arch: ChipArchitecture, device_id: int, context):
    context = context or check_context()
    coordinate = convert_coordinate(location, device_id, context)
    block = coordinate.device.get_block(coordinate)
    return block.get_risc_debug(
        _ACCESS_RISC, neo_id=0 if arch == ChipArchitecture.QUASAR else None
    )


def snapshot_cfg(location: str, items, *, device_id: int = 0, context=None) -> dict:
    """Read the given (state, addr32) words and return a {(state, addr32): value} map."""
    arch = get_chip_architecture()
    risc_debug = _get_risc_debug(location, arch, device_id, context)
    snap = {}
    with risc_debug.ensure_private_memory_access():
        for state, addr32 in items:
            snap[(state, addr32)] = risc_debug.read_memory(
                _word_core_addr(addr32, state, arch)
            )
    return snap


def thread_items(arch: ChipArchitecture) -> list:
    """Every kernel-owned (0, addr32) word. Restore only ever targets shadow state 0."""
    return [(0, a) for a in _thread_words(arch)]


# L1 base/magic for the restore-mode plan trisc.cpp applies -- must match apply_plan_at() there.
_INKERNEL_RESTORE_BASE = 0x1A000
_INKERNEL_RESTORE_MAGIC = 0x52535431  # 'RST1'
# L1 base/magic for the per-thread addr-mod plan -- must match trisc.cpp's apply_addrmod_restore().
# Applied AFTER the restore plan above, so a real captured addr-mod value overwrites that plan's
# reset-default-0 guess for the same addresses.
_INKERNEL_ADDRMOD_RESTORE_BASE = 0x1C000
_INKERNEL_ADDRMOD_RESTORE_MAGIC = 0x41525431  # 'ART1', must match trisc.cpp

# addr32 written via SETC16 (thread-private ThreadConfig) in real LLK code: the addr-mod section
# regs and CFG_STATE_ID. Everything else defaults to the shared cfg_write port. (BH addr32.)
_SETC16_ADDR32 = set(range(12, 55)) | {
    0
}  # ADDR_MOD_AB/DST/BIAS SEC0-7 span ~12..54; refine as needed


def _port_for(addr32: int) -> int:
    return 1 if addr32 in _SETC16_ADDR32 else 0


def write_inkernel_restore(
    location: str, entries, *, device_id: int = 0, context=None
) -> int:
    """Write a restore plan [(addr32, value[, port[, mask]]), ...] to L1 0x1A000.

    Replayed by trisc.cpp before the victim's own init runs, so a trial starts from the captured
    residue without a per-trial tt-smi -r. port omitted -> inferred from _port_for; mask omitted
    -> 0xFFFFFFFF, RMW'd on the shared port so unmasked firmware-owned bits are preserved.
    """
    words = [_INKERNEL_RESTORE_MAGIC, len(entries)]
    for e in entries:
        addr32, value = e[0], e[1]
        port = e[2] if len(e) > 2 else _port_for(addr32)
        mask = e[3] if len(e) > 3 else 0xFFFFFFFF
        words += [addr32, value, port, mask]
    write_words_to_device(
        location, _INKERNEL_RESTORE_BASE, words, device_id=device_id, context=context
    )
    return len(entries)


def write_inkernel_addrmod_restore(
    location: str, entries, *, ch1x=None, device_id: int = 0, context=None
) -> int:
    """Write a per-thread addr-mod restore plan [(addr32, v0, v1, v2), ...] to L1 0x1C000.

    Each compiled TRISC applies only its own v_thread (see apply_addrmod_restore() in trisc.cpp).
    ch1x, if given, is (unpacker0_val, packer_val) -- address_counters' channel1-X residue (see
    snapshot_adc_ch1x) -- appended as two trailing words gated by an explicit has_ch1x header:
    this L1 buffer is reused across trials, so trisc.cpp must never infer the trailer from
    leftover bytes of a longer prior plan.
    """
    words = [
        _INKERNEL_ADDRMOD_RESTORE_MAGIC,
        len(entries),
        1 if ch1x is not None else 0,
    ]
    for addr32, v0, v1, v2 in entries:
        words += [addr32, v0, v1, v2]
    if ch1x is not None:
        words += [ch1x[0] & 0x3FFFF, ch1x[1] & 0x3FFFF]
    write_words_to_device(
        location,
        _INKERNEL_ADDRMOD_RESTORE_BASE,
        words,
        device_id=device_id,
        context=context,
    )
    return len(entries)


# ThreadConfig (addr-mod, state id, ...) readout: a separate per-thread-banked array from
# Config[]/ConfigDualWrite (ordinary cfg_write()/cfg_read() can't reach it, and RISCV stores
# can't write it at all — SETC16 only). It sits immediately after Config[0]+Config[1]+
# ConfigDualWrite in the same core-private address space _word_core_addr() already reads, so
# risc_debug.read_memory() reaches it directly — no in-kernel capture needed.
_THREAD_CONFIG_BASE_WORDS = {
    ChipArchitecture.BLACKHOLE: 3 * _CFG_STATE_SIZE[ChipArchitecture.BLACKHOLE] * 4,
    ChipArchitecture.WORMHOLE: 3 * _CFG_STATE_SIZE[ChipArchitecture.WORMHOLE] * 4,
}
_THD_STATE_SIZE = {
    ChipArchitecture.BLACKHOLE: 68,
    ChipArchitecture.WORMHOLE: 57,
}
_THREAD_CFG_IDS = (0, 1, 2)  # THREAD_0_CFG/1/2 == UNPACK/MATH/PACK
# Blackhole ADDR_MOD_AB_SEC0-5 ThreadConfig-local indices (cfg_defines.h).
_ADDR_MOD_ADDR32_BH = sorted(
    set(range(12, 20)) | set(range(28, 36)) | set(range(37, 41)) | set(range(47, 55))
)


def _thread_config_addr(thread: int, local_idx: int, arch: ChipArchitecture) -> int:
    """Core-private address of ThreadConfig[thread][local_idx]."""
    return (
        TENSIX_CFG_BASE
        + (_THREAD_CONFIG_BASE_WORDS[arch] + thread * _THD_STATE_SIZE[arch] + local_idx)
        * 4
    )


def snapshot_addr_mod(location: str, *, device_id: int = 0, context=None) -> dict:
    """Read the real per-thread addr-mod residue left by whatever kernel last ran on this core.

    Returns {(thread, addr32): value}, one entry per thread per _ADDR_MOD_ADDR32_BH address.
    """
    arch = get_chip_architecture()
    risc_debug = _get_risc_debug(location, arch, device_id, context)
    out = {}
    with risc_debug.ensure_private_memory_access():
        for thread in _THREAD_CFG_IDS:
            for addr32 in _ADDR_MOD_ADDR32_BH:
                out[(thread, addr32)] = risc_debug.read_memory(
                    _thread_config_addr(thread, addr32, arch)
                )
    return out


def snapshot_adc_ch1x(location: str, *, device_id: int = 0, context=None) -> dict:
    """Read address_counters' channel1-X residue: hardware state entirely outside Config[]/
    ThreadConfig[] (the debug-bus ADC group, not the CFG bus this module otherwise captures).

    unpacker_addr_counter_init()/packer_addr_counter_init() deliberately skip this field (their
    own BitMask 0b1011 excludes bit 2) because a real hardware reset normally handles it; the
    comment at their call site names it "the tile X dimension" -- real, semantically meaningful
    state carried across calls within the untilize family, not scrubbable init residue. Restore
    mode never resets between trials, so a victim can otherwise inherit an unrelated prior
    kernel's channel1-X instead of the polluter's.

    Returns {"unpacker": v, "packer": v} (unpacker0 and packer channel1-X counters).
    """
    state = get_tensix_state(location, device_id=device_id, context=context)
    return {
        "unpacker": state.address_counters["adcs0_unpacker0_channel1_x_counter"],
        "packer": state.address_counters["adcs2_packers_channel1_x_counter"],
    }


def maybe_restore_cfg_from_env(location: str, *, device_id: int = 0, context=None):
    """Restore / snapshot CFG based on env vars. No-op (returns None) unless one is set.

    LLK_CFG_RESTORE=<path>          JSON {entries:[[addr32,value,port,mask]..]} ->
                                    write_inkernel_restore.
    LLK_CFG_ADDRMOD_RESTORE=<path>  JSON {entries:[[addr32,v0,v1,v2]..], ch1x:[unpacker,packer]}
                                    -> write_inkernel_addrmod_restore. ch1x is optional.
    LLK_CFG_SNAPSHOT=<path>         Dump every kernel-owned word to <path> as JSON; do not
                                    restore. Run once on a device where the kernel passes --
                                    that run's pre-kernel CFG is the pair-sweep baseline.
    """
    arch = get_chip_architecture()

    restore_path = os.environ.get("LLK_CFG_RESTORE")
    if restore_path:
        with open(restore_path) as f:
            rplan = json.load(f)
        rentries = [tuple(e) for e in rplan["entries"]]
        nr = write_inkernel_restore(
            location, rentries, device_id=device_id, context=context
        )
        msg = f"[CFG-RESTORE] restore entries={nr} -> L1 0x{_INKERNEL_RESTORE_BASE:X}"
        print(msg, file=sys.stderr, flush=True)
        logger.warning(msg)

    addrmod_restore_path = os.environ.get("LLK_CFG_ADDRMOD_RESTORE")
    if addrmod_restore_path:
        with open(addrmod_restore_path) as f:
            arplan = json.load(f)
        arentries = [tuple(e) for e in arplan["entries"]]
        ch1x = tuple(arplan["ch1x"]) if arplan.get("ch1x") is not None else None
        nar = write_inkernel_addrmod_restore(
            location, arentries, ch1x=ch1x, device_id=device_id, context=context
        )
        msg = f"[CFG-RESTORE] addrmod restore entries={nar} -> L1 0x{_INKERNEL_ADDRMOD_RESTORE_BASE:X}"
        print(msg, file=sys.stderr, flush=True)
        logger.warning(msg)

    snap_path = os.environ.get("LLK_CFG_SNAPSHOT")
    if snap_path:
        items = thread_items(arch)
        snap = snapshot_cfg(location, items, device_id=device_id, context=context)
        with open(snap_path, "w") as f:
            json.dump([[s, a, v] for (s, a), v in snap.items()], f)
        msg = (
            f"[CFG-RESTORE] snapshot arch={arch.value} words={len(snap)} -> {snap_path}"
        )
        print(msg, file=sys.stderr, flush=True)
        logger.warning(msg)
        return None

    return None
