# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Host-side Tensix CFG (THCON) register pollution for init-completeness testing.

The premise: a correct kernel's init must (re)write every config register it
depends on. To test that, we scribble garbage into the CFG space *before* the
kernel runs. Anything the kernel rewrites is harmless; anything it silently
relies on (a reset/leftover value it never sets) stays polluted and the kernel
either miscomputes (PCC fail) or hangs.

This works because the CFG space is NOT reset at kernel launch: firmware boot
only flips the shadow id (`reset_cfg_state_id`) and zeroes PRNG_SEED, so values
written while the TRISCs are held in reset persist into kernel execution. See
`run_elf_files()` for the injection point.

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
  - The thread-config region is double-buffered into two shadow "states". Each
    state spans CFG_STATE_SIZE 128-bit entries == CFG_STATE_SIZE*4 32-bit words.
    Register `addr32` lives at word `addr32` in state 0 and `addr32 + stride` in
    state 1, where `stride = CFG_STATE_SIZE*4` (224 words on BH). A kernel may
    flip between states mid-run, so we pollute both by default.
  - Every CFG register defined in cfg_defines.h fits within a single state
    (highest addr32 is 222 on BH / 186 on WH, both below CFG_STATE_SIZE*4), so the
    "thread" group below genuinely covers the whole config register space.
  - Word addr32 0 packs CFG_STATE_ID.StateID (bit 0) alongside legacy ALU format
    fields. Bit 0 selects the active shadow, so we preserve it (polluting it
    would just flip shadows — a confound, not a finding).
"""

import json
import os
import random
import sys
from functools import lru_cache

from ttexalens.tt_exalens_lib import (
    check_context,
    convert_coordinate,
    write_words_to_device,
)

from .chip_architecture import ChipArchitecture, get_chip_architecture
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
# whole-space "thread" sweep excludes them. Verified per arch against the boot path
# (tests/helpers/include/boot.h::device_setup):
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


# The "live" config-write surface, BIT-GRANULAR: {addr32: written-bit mask} where the mask is
# the union of bits some *reachable* LLK/Metal op actually writes (per the static sweep in
# cfg_state_map.md). Polluting only these bits targets the *reconfigurable* state — what a prior
# op can leave dirty — and crucially leaves never-written FIELDS at their reset default, so a
# failure is a candidate actionable reconfig-escape rather than reliance on a register/field
# nothing dirties. E.g. word 71 = 0xFFC80000 excludes Downsample (bits 0-18, never written) but
# keeps Pack_L1_Acc/LF8/Exp_threshold; word 6 (DEST_REGW_BASE) and the dead-writer PCK0_ADDR_BASE
# (word 16 high bits) are absent. DISABLE_RISC_BP (word 2 bits 22-31) is still masked by
# _PRESERVE_BITS. Regenerate (Blackhole) by parsing CFG-write sites for the target+mask of each
# write primitive (WRCFG/cfg[]=->whole word, SETC16/addr_mod->0xFFFF, cfg_reg_rmw_tensix->field
# mask), excluding comments and dead code. WH set differs (different addr32 numbering) — TODO.
_LIVE_MASK = {
    ChipArchitecture.BLACKHOLE: {
        0: 0x0000FFFF, 1: 0xFFFFFFFF, 2: 0xFFFFFFFF, 5: 0x0000FFFF, 7: 0x0000FFFF, 12: 0xFFFFFFFF,
        13: 0xFFFFFFFF, 14: 0xFFFFFFFF, 15: 0xFFFFFFFF, 16: 0x0000FFFF, 17: 0xFFFFFFFF, 18: 0xFFFFFFFF,
        19: 0x0000FFFF, 20: 0xFFFFFFFF, 21: 0xFFFFFFFF, 24: 0xFFFFFFFF,
        25: 0xFFFFFFFF, 28: 0x0000FFFF, 29: 0x0000FFFF, 30: 0x0000FFFF,
        31: 0x0000FFFF, 32: 0x0000FFFF, 33: 0x0000FFFF, 34: 0x0000FFFF, 35: 0x0000FFFF, 37: 0x0000FFFF,
        38: 0x0000FFFF, 39: 0x0000FFFF, 40: 0x0000FFFF, 41: 0x0000FFFF, 47: 0x0000FFFF, 48: 0x0000FFFF,
        49: 0x0000FFFF, 50: 0xFFFFFFFF, 51: 0x0000FFFF, 52: 0x0000FFFF, 53: 0x0000FFFF, 54: 0x0000FFFF,
        55: 0x0000FFFF, 56: 0xFFFFFFFF, 57: 0xFFFFFFFF, 59: 0xFFFFFFFF, 64: 0xFFFF000F, 65: 0xFFFFFFFF,
        68: 0xFFFFFFFF, 69: 0xFFFFFFFF, 70: 0xFFFFFFFF, 71: 0xFFC80000, 72: 0xFFFFFFFF, 73: 0x00000030,
        76: 0xFFFFFFFF, 77: 0xFFFFFFFF, 84: 0xFFFFFFFF, 86: 0xFFFFFFFF, 92: 0xFFFFFFFF, 93: 0xFFFFFFFF,
        112: 0xFFFF000F, 113: 0xFFFF0000, 119: 0x00400000, 120: 0x0000000F, 124: 0xFFFFFFFF, 125: 0xFFFFFFFF,
        140: 0xFFFFFFFF, 141: 0xFFFFFFFF, 180: 0xFFFFFFFF, 181: 0xFFFFFFFF, 182: 0xFFFFFFFF, 183: 0xFFFFFFFF,
        186: 0xFFFFFFFF, 209: 0xFFFFFFFF, 211: 0xFFFFFFFF, 220: 0x0000000B,
    },
}


def live_items(arch: ChipArchitecture, states=(0, 1)) -> list:
    """Bit-granular live items [(state, addr32, mask), ...] — pollute only written bits."""
    if arch not in _LIVE_MASK:
        raise ValueError(f"no live CFG mask for {arch.value} (only Blackhole so far)")
    return [(s, a, m) for s in states for a, m in _LIVE_MASK[arch].items()]


# Named addr32 groups within the per-state thread-config region.
# "alu" is the 3-word ALU config block (addr32 0,1,2) — a small, targeted probe.
# "thread" is the whole shadowed config register space (minus boot-owned words): every
# register defined in cfg_defines.h fits inside one state (see module docstring), so
# this is the complete kernel-owned config-register sweep.
# ("live" is handled separately — it is bit-granular, see _LIVE_MASK / live_items.)
_GROUPS = {
    ChipArchitecture.BLACKHOLE: {
        "alu": [0, 1, 2],
        "thread": _thread_words(ChipArchitecture.BLACKHOLE),
    },
    ChipArchitecture.WORMHOLE: {
        "alu": [0, 1, 2],
        "thread": _thread_words(ChipArchitecture.WORMHOLE),
    },
}

# addr32 -> bit mask of bits that must NOT be polluted: they are firmware/hardware-owned,
# not LLK-compute config, so trampling them is over-reach (breaks boot/debug, not kernel
# init). The rest of each such word stays poisonable. Same addr32/bit layout on BH and WH.
_PRESERVE_BITS = {
    0: 0x1,  # CFG_STATE_ID.StateID — selects the active shadow.
    # DISABLE_RISC_BP (bits 22-31): RISC branch-prediction enable. Set by firmware (brisc
    # disable_branch_prediction) and toggled by the debug halt path (ttexalens
    # set_branch_prediction); no compute kernel owns it. Poisoning it hangs the cores
    # (bisected to this exact field on BH, 2026-06-16). Bits 0-21 (STACC_RELU / ALU_ACC_CTRL)
    # remain kernel-owned and are still poisoned.
    2: 0xFFC00000,
}

# TRISC whose debug private-memory interface we drive to reach CFG. CFG is shared
# Tensix config, so any TRISC's private view works.
_ACCESS_RISC = "trisc0"


def _state_stride_words(arch: ChipArchitecture) -> int:
    return _CFG_STATE_SIZE[arch] * 4


def _word_core_addr(addr32: int, state: int, arch: ChipArchitecture) -> int:
    """Core-private CFG address of `addr32` in the given shadow state."""
    return TENSIX_CFG_BASE + (addr32 + state * _state_stride_words(arch)) * 4


@lru_cache(maxsize=8)
def _full_value_map(seed: int, arch: ChipArchitecture) -> dict:
    """The value every (state, addr32) word receives in the canonical whole-space sweep.

    Built by drawing one RNG sequence over the fixed ordering `for state in (0,1): for
    addr32 in 0..CFG_STATE_SIZE*4` — i.e. exactly what a sequential whole-space pollution
    would write. Cached per (seed, arch). Spans the full range (boot-owned words included)
    so a word's value never shifts when those words are excluded from the *pollute* set.
    """
    rng = random.Random(seed)
    stride = _state_stride_words(arch)
    return {
        (state, addr32): rng.getrandbits(32)
        for state in (0, 1)
        for addr32 in range(stride)
    }


def word_value(seed: int, state: int, addr32: int) -> int:
    """Deterministic 32-bit poison value for one word — the value it gets in the canonical
    whole-space sweep for this seed. Pure function of (seed, state, addr32), independent of
    which/how many words a given run actually pollutes. That subset-independence is what makes
    bisection sound: a culprit word carries an identical value whether tested alone or with
    everything, and a seed that fails the whole-space sweep is faithfully reproduced by subsets.
    """
    return _full_value_map(seed, get_chip_architecture())[(state, addr32)]


def resolve_words(group_or_words, arch: ChipArchitecture) -> list[int]:
    """Resolve a group name (e.g. "alu") or an explicit addr32 iterable to a word list."""
    if isinstance(group_or_words, str):
        groups = _GROUPS[arch]
        if group_or_words not in groups:
            raise ValueError(
                f"unknown CFG pollution group {group_or_words!r} for {arch.value}; "
                f"known: {sorted(groups)}"
            )
        return list(groups[group_or_words])
    return list(group_or_words)


def _get_risc_debug(location: str, arch: ChipArchitecture, device_id: int, context):
    context = context or check_context()
    coordinate = convert_coordinate(location, device_id, context)
    block = coordinate.device.get_block(coordinate)
    return block.get_risc_debug(
        _ACCESS_RISC, neo_id=0 if arch == ChipArchitecture.QUASAR else None
    )


def pollute_items(
    location: str,
    items,
    *,
    seed: int,
    restore=None,
    extra_preserve=None,
    device_id: int = 0,
    context=None,
    verify: bool = True,
) -> list[dict]:
    """Poison an explicit list of (state, addr32) words with deterministic values.

    `restore`, if given, is a {(state, addr32): value} map written FIRST (before any
    poisoning) so a bisection trial starts from a known-clean CFG regardless of what a
    prior trial left behind. Each item's poison value is `word_value(seed, state, addr32)`
    — subset-independent (see `word_value`). Returns a per-write log.

    `extra_preserve` is a {addr32: mask} of bits to leave untouched ON TOP OF the built-in
    `_PRESERVE_BITS`. The two are distinct classes: `_PRESERVE_BITS` is *over-reach* (firmware/
    hardware-owned, never a kernel-init finding — always masked); `extra_preserve` is *acknowledged
    kernel dependencies* — real findings a caller has already noted and wants to set aside so a
    sweep/bisection surfaces NEW ones (e.g. mask DEST_REGW_BASE to find the next datacopy gap).

    Call only while the target TRISCs are held in soft reset — the access core is
    released/halted/restored automatically around the accesses (one session for all).
    """
    arch = get_chip_architecture()
    extra_preserve = extra_preserve or {}
    log: list[dict] = []
    risc_debug = _get_risc_debug(location, arch, device_id, context)
    with risc_debug.ensure_private_memory_access():
        if restore:
            for (state, addr32), value in restore.items():
                risc_debug.write_memory(_word_core_addr(addr32, state, arch), value)
        for item in items:
            # item is (state, addr32[, mask[, value]]). mask selects which BITS to poison
            # (default all 32) — lets a caller poison one field of a mixed word. value, if given,
            # is the EXACT source to write into the masked bits (instead of the random word_value)
            # — used to plant a specific realistic value a real op would leave (e.g. Pack_L1_Acc=1)
            # to demonstrate an actionable reconfig-escape rather than random garbage.
            state, addr32 = item[0], item[1]
            mask = item[2] if len(item) > 2 else 0xFFFFFFFF
            src = item[3] if len(item) > 3 else word_value(seed, state, addr32)
            addr = _word_core_addr(addr32, state, arch)
            original = risc_debug.read_memory(addr)
            value = (original & ~mask) | (src & mask)
            preserve = _PRESERVE_BITS.get(addr32, 0) | extra_preserve.get(addr32, 0)
            if preserve:
                value = (value & ~preserve) | (original & preserve)
            risc_debug.write_memory(addr, value)
            readback = risc_debug.read_memory(addr) if verify else None
            log.append(
                {
                    "state": state,
                    "addr32": addr32,
                    "original": original,
                    "poisoned": value,
                    "readback": readback,
                }
            )
    return log


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


def restore_snapshot(
    location: str, snapshot: dict, *, device_id: int = 0, context=None
) -> None:
    """Write a {(state, addr32): value} snapshot back to CFG (full clean restore)."""
    arch = get_chip_architecture()
    risc_debug = _get_risc_debug(location, arch, device_id, context)
    with risc_debug.ensure_private_memory_access():
        for (state, addr32), value in snapshot.items():
            risc_debug.write_memory(_word_core_addr(addr32, state, arch), value)


def thread_items(arch: ChipArchitecture, states=(0, 1)) -> list:
    """All kernel-owned (state, addr32) words — the bisection candidate universe."""
    return [(s, a) for s in states for a in _thread_words(arch)]


def parse_preserve(spec) -> dict:
    """Parse an "addr32:mask,addr32:mask" string into {addr32: mask}. Empty/None -> {}."""
    out = {}
    for tok in (spec or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        a, _, m = tok.partition(":")
        out[int(a, 0)] = int(m, 0) if m else 0xFFFFFFFF
    return out


def pollute_cfg(
    location: str,
    words,
    *,
    seed: int,
    states=(0, 1),
    extra_preserve=None,
    device_id: int = 0,
    context=None,
    verify: bool = True,
) -> list[dict]:
    """Poison CFG words (group name / addr32 iterable) across the given shadow states.

    Thin wrapper over `pollute_items` preserving the original group/addr32 interface.
    """
    arch = get_chip_architecture()
    addr32_list = resolve_words(words, arch)
    items = [(state, addr32) for state in states for addr32 in addr32_list]
    return pollute_items(
        location,
        items,
        seed=seed,
        extra_preserve=extra_preserve,
        device_id=device_id,
        context=context,
        verify=verify,
    )


def restore_cfg(
    location: str, log: list[dict], *, device_id: int = 0, context=None
) -> None:
    """Restore the original values captured by a pollute log (best-effort cleanup)."""
    snapshot = {(e["state"], e["addr32"]): e["original"] for e in log}
    restore_snapshot(location, snapshot, device_id=device_id, context=context)


def _log_landed(log: list[dict], summary_prefix: str) -> None:
    # A write "landed" if the register's value changed. Many CFG words are sparse
    # (narrow fields / read-only bits), so readback rarely equals the full 32-bit poison
    # value — that is NOT a failure. The real failure mode is a write with *no* visible
    # effect (readback == original) across the board → the access mechanism missed CFG.
    verified = [e for e in log if e["readback"] is not None]
    landed = [e for e in verified if e["readback"] != e["original"]]
    summary = f"{summary_prefix} landed={len(landed)}/{len(verified)}"
    # Also print (flushed) so the reproducing seed survives even if the loguru file sink
    # is lost when a hung run is killed / the device is reset mid-test.
    print(summary, file=sys.stderr, flush=True)
    logger.warning(summary)
    if verified and not landed:
        logger.error(
            "[CFG-POLLUTE] no polluted word changed value — the access mechanism did "
            "not reach CFG (cores not in reset / wrong access core?)."
        )


# In-kernel pollution plan: host writes a magic-tagged plan to the device-print L1 region
# (0x15000), which the trisc.cpp prologue reads and applies via SETC16 (thread-private) /
# cfg_write (shared) BEFORE the kernel's init. This thrashes config through the same ports the
# kernel reads — reaching thread-private addr-mod that host CFG writes cannot.
_INKERNEL_PLAN_BASE = 0x15000
_INKERNEL_PLAN_MAGIC = 0x504F4C31  # 'POL1', must match trisc.cpp
# Restore plan (restore-mode): pristine CFG replayed before the poison so each trial starts clean
# WITHOUT a per-trial tt-smi -r. Same quad format, different L1 base/magic (must match trisc.cpp).
_INKERNEL_RESTORE_BASE = 0x1A000
_INKERNEL_RESTORE_MAGIC = 0x52535431  # 'RST1'

# addr32 written via SETC16 (thread-private ThreadConfig) in real LLK code: the addr-mod section
# regs and CFG_STATE_ID. Everything else defaults to the shared cfg_write port. (BH addr32.)
_SETC16_ADDR32 = set(range(12, 55)) | {0}  # ADDR_MOD_AB/DST/BIAS SEC0-7 span ~12..54; refine as needed


def _port_for(addr32: int) -> int:
    return 1 if addr32 in _SETC16_ADDR32 else 0


def write_inkernel_plan(
    location: str, entries, *, device_id: int = 0, context=None
) -> int:
    """Write a pollution plan [(addr32, value[, port[, mask]]), ...] to L1 for the prologue.

    port omitted -> inferred from _port_for (SETC16 for addr-mod/state, cfg_write otherwise).
    mask omitted -> 0xFFFFFFFF (whole word). On the shared port the prologue RMWs so unmasked
    bits are preserved (used to keep firmware-owned bits like DISABLE_RISC_BP intact).
    Returns the number of entries written.
    """
    words = [_INKERNEL_PLAN_MAGIC, len(entries)]
    for e in entries:
        addr32, value = e[0], e[1]
        port = e[2] if len(e) > 2 else _port_for(addr32)
        mask = e[3] if len(e) > 3 else 0xFFFFFFFF
        words += [addr32, value, port, mask]
    write_words_to_device(
        location, _INKERNEL_PLAN_BASE, words, device_id=device_id, context=context
    )
    return len(entries)


def write_inkernel_restore(
    location: str, entries, *, device_id: int = 0, context=None
) -> int:
    """Write a pristine-restore plan [(addr32, value[, port[, mask]]), ...] to L1 0x1A000.

    Replayed by the prologue BEFORE the poison plan (restore-mode): re-establishes the clean
    post-reset CFG so a trial isn't contaminated by a prior trial's poison in never-written
    fields. Same quad encoding as the poison plan, distinct base/magic. Constant across a
    catalog run (the captured pristine baseline), so a driver writes it once and reuses it.
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


def maybe_pollute_cfg_from_env(location: str, *, device_id: int = 0, context=None):
    """Pollute / snapshot CFG based on env. No-op (returns None) unless one is set.

    Modes (checked in order):
      LLK_POLLUTE_SNAPSHOT=<path>  Read every kernel-owned word (both shadows) and dump a
                                   JSON clean reference to <path>; do NOT pollute. Run this
                                   once on a device where the kernel passes — the snapshot is
                                   that passing run's pre-kernel CFG, the bisection baseline.
      LLK_POLLUTE_PLAN=<path>      Bisection trial: JSON {seed, snapshot:[[s,a,v]..],
                                   pollute:[[s,a]..]}. Restore the snapshot, then poison only
                                   the `pollute` subset. Lets a driver test arbitrary subsets.
      LLK_POLLUTE_CAPTURE=<path>   Realizable-palette capture: at each kernel launch read the live
                                   words (the prior kernel's leftover) and merge their values into
                                   a per-addr32 palette JSON. Run a suite with this to accumulate
                                   the set of values kernels actually leave. Does NOT pollute.
      LLK_POLLUTE_CFG=<spec>       Whole-space / group ("alu"/"thread"/"live") / addr32-list sweep.
                                   spec="live": bit-granular over the reconfigurable surface.
                                   spec="realizable": bit-granular AND poison each live word with a
                                   value from LLK_POLLUTE_PALETTE (a state real ops produce) — a
                                   failure is a candidate actionable escape, not random garbage.
                                   LLK_POLLUTE_SEED optional (random + logged if unset).

    LLK_POLLUTE_PRESERVE="a:m,a:m" (optional, all modes): extra {addr32: mask} bits to leave
    unpoisoned ON TOP OF the built-in over-reach set — used to mask already-found dependencies
    (e.g. "6:0xffff" for DEST_REGW_BASE) so a sweep/bisection surfaces the NEXT gap. The plan
    mode also accepts a "preserve":[[addr32,mask]..] field for the same purpose.
    """
    arch = get_chip_architecture()
    env_preserve = parse_preserve(os.environ.get("LLK_POLLUTE_PRESERVE"))

    # Restore-mode: replay the pristine baseline first (so the prologue's poison overlays a clean
    # CFG without a per-trial reset). Written alongside — NOT instead of — the poison plan below.
    restore_path = os.environ.get("LLK_POLLUTE_INKERNEL_RESTORE")
    if restore_path:
        with open(restore_path) as f:
            rplan = json.load(f)
        rentries = [tuple(e) for e in rplan["entries"]]
        nr = write_inkernel_restore(
            location, rentries, device_id=device_id, context=context
        )
        msg = f"[CFG-POLLUTE] restore entries={nr} -> L1 0x{_INKERNEL_RESTORE_BASE:X}"
        print(msg, file=sys.stderr, flush=True)
        logger.warning(msg)

    inkernel_path = os.environ.get("LLK_POLLUTE_INKERNEL")
    if inkernel_path:
        with open(inkernel_path) as f:
            plan = json.load(f)
        entries = [tuple(e) for e in plan["entries"]]
        n = write_inkernel_plan(
            location, entries, device_id=device_id, context=context
        )
        msg = f"[CFG-POLLUTE] inkernel plan entries={n} -> L1 0x{_INKERNEL_PLAN_BASE:X}"
        print(msg, file=sys.stderr, flush=True)
        logger.warning(msg)
        return None

    snap_path = os.environ.get("LLK_POLLUTE_SNAPSHOT")
    if snap_path:
        items = thread_items(arch)
        snap = snapshot_cfg(location, items, device_id=device_id, context=context)
        with open(snap_path, "w") as f:
            json.dump([[s, a, v] for (s, a), v in snap.items()], f)
        msg = (
            f"[CFG-POLLUTE] snapshot arch={arch.value} words={len(snap)} -> {snap_path}"
        )
        print(msg, file=sys.stderr, flush=True)
        logger.warning(msg)
        return None

    cap_path = os.environ.get("LLK_POLLUTE_CAPTURE")
    if cap_path:
        # Realizable-palette capture, NET-LEFTOVER ("set but not wiped"): read the live words at THIS
        # launch (= the prior kernel's post-state) and DIFF against the previous launch's read. A word
        # that CHANGED is one the just-finished kernel set and left changed (transients it wiped on
        # uninit read back unchanged -> excluded). Record the changed word's new value into the palette.
        # File holds {"palette": {addr32: sorted values}, "prev": {"s,a": value}}. Run a suite to
        # accumulate. Read-only (no pollution).
        items = [(s, a) for s in (0, 1) for a in _LIVE_MASK.get(arch, {})]
        now = snapshot_cfg(location, items, device_id=device_id, context=context)
        state = {"palette": {}, "prev": {}}
        if os.path.exists(cap_path):
            with open(cap_path) as f:
                state = json.load(f)
        pal, prev = state["palette"], state["prev"]
        for (s, a), v in now.items():
            key = f"{s},{a}"
            if key in prev and prev[key] != v:  # kernel changed this word and left it at v
                pal[str(a)] = sorted(set(pal.get(str(a), [])) | {v})
            prev[key] = v
        with open(cap_path, "w") as f:
            json.dump({"palette": pal, "prev": prev}, f)
        return None

    plan_path = os.environ.get("LLK_POLLUTE_PLAN")
    if plan_path:
        with open(plan_path) as f:
            plan = json.load(f)
        seed = int(plan["seed"])
        restore = {(s, a): v for s, a, v in plan.get("snapshot", [])}
        # pollute entries: [state, addr32] or [state, addr32, mask] (bit-granular).
        items = [tuple(e) for e in plan["pollute"]]
        preserve = {**{a: m for a, m in plan.get("preserve", [])}, **env_preserve}
        log = pollute_items(
            location,
            items,
            seed=seed,
            restore=restore,
            extra_preserve=preserve,
            device_id=device_id,
            context=context,
        )
        pres = f" preserve={sorted(preserve)}" if preserve else ""
        _log_landed(
            log,
            f"[CFG-POLLUTE] plan arch={arch.value} seed=0x{seed:08X} pollute={len(items)}{pres}",
        )
        return seed, log

    spec = os.environ.get("LLK_POLLUTE_CFG")
    if not spec:
        return None

    seed_env = os.environ.get("LLK_POLLUTE_SEED")
    seed = int(seed_env, 0) if seed_env else random.getrandbits(32)

    if spec.strip() == "realizable":
        # Bit-granular AND realizable: poison each live word with a VALUE some kernel actually
        # leaves (from the LLK_POLLUTE_CAPTURE palette), not random garbage. A failure here is a
        # candidate ACTIONABLE escape on a state real ops produce. Per word, pick a palette value
        # by seed (sweep seeds for coverage). Words with no palette entry are skipped.
        pal_path = os.environ["LLK_POLLUTE_PALETTE"]
        with open(pal_path) as f:
            raw = json.load(f)
        raw = raw.get("palette", raw)  # accept capture file {"palette":..,"prev":..} or static {addr32:[..]}
        palette = {int(a): v for a, v in raw.items()}
        rng = random.Random(seed)
        items = []
        for s in (0, 1):
            for a, mask in _LIVE_MASK[arch].items():
                vals = palette.get(a)
                if vals:
                    items.append((s, a, mask, vals[rng.randrange(len(vals))]))
        log = pollute_items(
            location, items, seed=seed, extra_preserve=env_preserve, device_id=device_id, context=context
        )
    elif spec.strip() == "live":
        # Bit-granular: pollute only the written bits of each live word (see _LIVE_MASK).
        log = pollute_items(
            location,
            live_items(arch),
            seed=seed,
            extra_preserve=env_preserve,
            device_id=device_id,
            context=context,
        )
    else:
        if "," in spec or spec.strip().isdigit():
            words = [int(tok, 0) for tok in spec.split(",") if tok.strip()]
        else:
            words = spec
        log = pollute_cfg(
            location,
            words,
            seed=seed,
            extra_preserve=env_preserve,
            device_id=device_id,
            context=context,
        )
    addr32_set = sorted({e["addr32"] for e in log})
    pres = f" preserve={sorted(env_preserve)}" if env_preserve else ""
    _log_landed(
        log,
        f"[CFG-POLLUTE] arch={arch.value} spec={spec!r} seed=0x{seed:08X} states=(0,1) words={addr32_set}{pres}",
    )
    return seed, log
