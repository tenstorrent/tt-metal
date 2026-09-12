# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Cave layout, detour arithmetic, and filler selection.

The cave is scratch space laid out once per site. The ELF only reserves the
range [start, limit), from `_etext` to `__loader_init_start` or the unused L1
tail. The injector writes this layout inside that range:

    start                  ->  NOP       \\
                               NOP        | max delay NOPs
                               ...        /
    clear                  ->  pacr's cfg-clear
    displaced_instruction  ->  <the instruction that used to live at the site>
    ret                    ->  jal x0, site+4
    set                    ->  pacr cfg-set + jal to the filler run   (unused otherwise)
    end                    ->  first byte past this layout (must be <= limit)

Delaying by n aims the site's jump at the first of n fillers, so switching
delay costs a single word write and no rebuild. On Quasar, PACR first jumps to
the cfg-set, then backward through n fillers and clear.
"""

from dataclasses import dataclass

# 100 fillers + set (for PACR) + clear (for PACR) + displaced + return.
DEFAULT_MAX_DELAY = 100

# RISC-V jal (J-type): signed 21-bit offset, 2-byte aligned.
# Instruction layout: imm[20] | imm[10:1] | imm[11] | imm[19:12] | rd=x0 | opcode
RISCV_JAL_OPCODE = 0x6F
JAL_IMM_BITS = 21
JAL_REACH = 1 << (JAL_IMM_BITS - 1)
JAL_IMM_MASK = (1 << JAL_IMM_BITS) - 1
JAL_IMM20_SHIFT, JAL_IMM20_POS = 20, 31
JAL_IMM10_1_SHIFT, JAL_IMM10_1_MASK, JAL_IMM10_1_POS = 1, 0x3FF, 21
JAL_IMM11_SHIFT, JAL_IMM11_POS = 11, 20
# POS is the bit the field starts at in the instruction, not the field width.
JAL_IMM19_12_SHIFT, JAL_IMM19_12_MASK, JAL_IMM19_12_POS = 12, 0xFF, 12


class DetourError(RuntimeError):
    """The tool could not patch safely. Never a finding about the kernel."""


def encode_jal(at: int, target: int) -> int:
    """`jal x0, target` as executed from `at`. rd=x0 so nothing is clobbered."""
    offset = target - at
    if offset % 2 or not -JAL_REACH <= offset < JAL_REACH:
        raise DetourError(f"jal from 0x{at:x} cannot reach 0x{target:x}")
    imm = offset & JAL_IMM_MASK
    return (
        (((imm >> JAL_IMM20_SHIFT) & 0x1) << JAL_IMM20_POS)
        | (((imm >> JAL_IMM10_1_SHIFT) & JAL_IMM10_1_MASK) << JAL_IMM10_1_POS)
        | (((imm >> JAL_IMM11_SHIFT) & 0x1) << JAL_IMM11_POS)
        | (((imm >> JAL_IMM19_12_SHIFT) & JAL_IMM19_12_MASK) << JAL_IMM19_12_POS)
        | RISCV_JAL_OPCODE
    )


@dataclass(frozen=True)
class Cave:
    start: int
    limit: int
    max_delay: int = DEFAULT_MAX_DELAY

    def __post_init__(self):
        if self.end > self.limit:
            raise DetourError(
                f"cave needs 0x{self.end - self.start:x} bytes but only "
                f"0x{self.limit - self.start:x} are reserved. Lower max_delay"
            )

    @property
    def clear(self) -> int:
        """One word past the filler bank: cfg-clear for PACR, or one more filler."""
        return self.start + self.max_delay * 4

    @property
    def displaced_instruction(self) -> int:
        """Address of the stashed site instruction, one word past clear."""
        return self.clear + 4

    @property
    def ret(self) -> int:
        """Address of the jump back to the instruction after the site."""
        return self.displaced_instruction + 4

    @property
    def set_addr(self) -> int:
        """Address of PACR's cfg-set and jump trampoline."""
        return self.ret + 4

    @property
    def end(self) -> int:
        """First byte past the cave layout, including the PACR trampoline."""
        return self.set_addr + 8

    def filler_start(self, delay: int) -> int:
        """Address of the first of `delay` fillers."""
        assert (
            0 <= delay <= self.max_delay
        ), f"delay {delay} exceeds cave capacity"  # max capacity is 100
        return self.displaced_instruction - delay * 4

    def tail(self, displaced_word: int, return_to: int) -> list:
        """Return the displaced instruction and the jump back."""
        return [displaced_word, encode_jal(self.ret, return_to)]


def filler_choices(
    thread: str, site, scan, forced: str = None, enable_unpacr_nop: bool = False
) -> list:
    """Which filler(s) to try at a site, as (name, word) pairs.

    - `tti_nop`: delays the issuing thread by one cycle.
    - `unpacr0` / `unpacr1`: delays one unpacker by one cycle using pure
      UNP_NOP mode.
    - `pacr`: delays Quasar packer 0 without writing L1.
    - `sfpnop`: delays the next SFPU instruction by one cycle.
    - `risc_nop`: delays the RISC alone

    TDMA NOP (`DMANOP`) can be added later.
    """
    unpacr_fillers = ("unpacr0", "unpacr1")
    if forced:
        # A forced filler can be a known name or a raw instruction word.
        word = scan.fillers.get(forced)
        if word is None:
            word = int(forced, 0)
        elif not word:
            raise DetourError(f"{forced} is not available on this architecture")
        if word in (scan.fillers[name] for name in unpacr_fillers):
            raise ValueError(
                "an UNPACR NOP word cannot be selected with TTNOP_FILLER; "
                "use TTNOP_ENABLE_UNPACR_NOP=1"
            )
        return [(forced, word)]
    names = ["tti_nop"]
    if enable_unpacr_nop and thread == "unpack" and scan.mode == "sync":
        names += [f"unpacr{unit}" for unit in (scan.unpackers() or (0, 1))]
    elif thread == "pack" and scan.fillers.get("pacr"):
        names.append("pacr")
    elif site.sfpu:
        # SFPU belongs to the math thread.
        names.append("sfpnop")
    names.append("risc_nop")
    return [(name, scan.fillers[name]) for name in names]


class Injector:
    """Applies and undoes detours through a caller-supplied word reader/writer.

    Same-site delay steps only rewrite the jump. Bookkeeping is per thread.
    Call forget() after something else reloads the kernel.
    """

    def __init__(self, read_words, write_words, max_delay: int = DEFAULT_MAX_DELAY):
        self._read = read_words
        self._write = write_words
        self.max_delay = max_delay
        self._filler = {}  # thread -> word packed into its filler run
        self._displaced = {}  # thread -> the Site whose instruction sits in its cave

    def cave_for(self, scan) -> Cave:
        if not scan.cave_start:
            raise DetourError(f"no cave found in {scan.elf}")
        return Cave(scan.cave_start, scan.cave_limit, self.max_delay)

    def forget(self) -> None:
        """Drop the bookkeeping after something else rewrote the kernel in place."""
        self._filler.clear()
        self._displaced.clear()

    def arm(self, thread: str, scan, site, delay: int, filler_word: int) -> None:
        cave = self.cave_for(scan)
        displaced = self._displaced.get(thread)
        new_site = displaced is None or displaced.addr != site.addr
        if new_site:
            # Check before any cave write: a different kernel in L1 would make
            # the filler run land on someone else's code.
            live = self._read(site.addr, 1)[0]
            if live != site.word:
                raise DetourError(
                    f"{thread} 0x{site.addr:05x} holds 0x{live:08x}, "
                    f"scan expected 0x{site.word:08x}"
                )
        pacr = bool(filler_word) and filler_word == scan.fillers.get("pacr")
        # Rewrite the filler run when the NOP type changes.
        if self._filler.get(thread) != filler_word:
            run = [filler_word] * cave.max_delay
            run.append(scan.fillers["pacr_cfg_clr"] if pacr else filler_word)
            self._write(cave.start, run)
            self._filler[thread] = filler_word

        # Restore the old site before preparing the new one.
        if new_site:
            self._restore(thread)
            self._write(cave.displaced_instruction, cave.tail(site.word, site.addr + 4))
            self._displaced[thread] = site

        # PACR needs no-write config around its run. Delay 0 bypasses the bracket.
        if pacr and delay:
            # Start one word earlier so clear follows the requested fillers.
            target = cave.filler_start(delay) - 4
            self._write(
                cave.set_addr,
                [scan.fillers["pacr_cfg_set"], encode_jal(cave.set_addr + 4, target)],
            )
            detour_at = cave.set_addr
        else:
            detour_at = cave.filler_start(delay)

        # Site jal into the cave, written at site.addr. Later delays are a different
        # jal at that same address and only the first write is read back.
        detour = encode_jal(site.addr, detour_at)
        self._write(site.addr, [detour])

        # Check the first write. Later delays only retarget this jal.
        if new_site and self._read(site.addr, 1)[0] != detour:
            raise DetourError(
                f"{thread} 0x{site.addr:05x} did not take the detour write"
            )

    def _restore(self, thread: str) -> None:
        site = self._displaced.pop(thread, None)
        if site is not None:
            self._write(site.addr, [site.word])

    def restore(self) -> None:
        """Restore every patched thread."""
        for thread in list(self._displaced):
            self._restore(thread)
