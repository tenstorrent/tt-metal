# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Cave layout, detour arithmetic and the filler policy.

The cave is scratch space laid out once per site:

    start ->  filler       \\
              filler        |  max_delay words
              ...          /
    parked -> <the instruction that used to live at the site>
    ret    -> jal x0, site+4

Delaying by n is nothing more than aiming the site's jump n words short of the
parked instruction, so switching delay costs a single word write and no rebuild.
Everything here is arithmetic on addresses, which is why the same code drives
both a device-memory write and a host-image write.
"""

from dataclasses import dataclass

# 100 fillers + the displaced instruction + the jump back = 102 words.
DEFAULT_MAX_DELAY = 100

RISCV_JAL_OPCODE = 0x6F
JAL_REACH = 1 << 20


class DetourError(RuntimeError):
    """The tool could not patch safely. Never a finding about the kernel."""


def encode_jal(at: int, target: int) -> int:
    """`jal x0, target` as executed from `at`. rd=x0 so nothing is clobbered."""
    offset = target - at
    assert (
        -JAL_REACH <= offset < JAL_REACH
    ), f"jal 0x{at:x}->0x{target:x} is out of reach"
    assert offset % 2 == 0, f"jal 0x{at:x}->0x{target:x} is misaligned"
    imm = offset & 0x1FFFFF
    return (
        (((imm >> 20) & 0x1) << 31)
        | (((imm >> 1) & 0x3FF) << 21)
        | (((imm >> 11) & 0x1) << 20)
        | (((imm >> 12) & 0xFF) << 12)
        | RISCV_JAL_OPCODE
    )


@dataclass(frozen=True)
class Cave:
    start: int
    limit: int
    max_delay: int = DEFAULT_MAX_DELAY

    def __post_init__(self):
        assert self.start % 4 == 0, "cave must be word aligned"
        assert self.end <= self.limit, (
            f"cave needs 0x{self.end - self.start:x} bytes but only "
            f"0x{self.limit - self.start:x} are reserved; lower max_delay"
        )

    @property
    def parked(self) -> int:
        return self.start + self.max_delay * 4

    @property
    def ret(self) -> int:
        return self.parked + 4

    @property
    def end(self) -> int:
        return self.ret + 4

    def entry(self, delay: int) -> int:
        """Where the site jumps to in order to execute `delay` fillers."""
        assert 0 <= delay <= self.max_delay, f"delay {delay} exceeds cave capacity"
        return self.parked - delay * 4

    def tail(self, displaced_word: int, return_to: int) -> list:
        """The two words behind the filler run: the displaced op, then the way back."""
        return [displaced_word, encode_jal(self.ret, return_to)]


def filler_choices(thread: str, site, scan, forced: str = None) -> list:
    """Which filler(s) to try at a site, as (name, word) pairs.

    A nop only widens a timing window if it retires on the unit whose timing is
    in question, so unpack sync sites get an unpacker-retired nop and SFPU sites
    get an SFPU nop. More than one pair comes back when the SETADCXX census
    found nothing and both unpackers are worth trying.
    """
    if forced:
        # A raw word is allowed so a one-off experiment can pin an encoding the
        # named table does not carry.
        return [(forced, scan.fillers.get(forced) or int(forced, 0))]
    if thread == "unpack" and scan.mode == "sync":
        units = scan.unpackers() or (0, 1)
        return [(f"unpacr{unit}", scan.fillers[f"unpacr{unit}"]) for unit in units]
    if site.sfpu:
        return [("sfpnop", scan.fillers["sfpnop"])]
    return [("tti_nop", scan.fillers["tti_nop"])]


class Injector:
    """Applies and undoes detours through a caller-supplied word reader/writer.

    A sweep walks the delays of one site back to back, so the cave contents barely
    change between variants. Whatever is already correct is left alone and the
    common case costs a single word: the jump at the site, aimed one filler further
    in. What is currently written is tracked per thread rather than re-read,
    because the tracking is only ever wrong if something outside reloaded the
    kernel — which is what forget() is for.
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
        if self._filler.get(thread) != filler_word:
            self._write(cave.start, [filler_word] * cave.max_delay)
            self._filler[thread] = filler_word

        displaced = self._displaced.get(thread)
        new_site = displaced is None or displaced.addr != site.addr
        if new_site:
            self._restore(thread)
            # The scan happened against the ELF on disk; refuse to patch if what is
            # actually loaded disagrees, rather than corrupting an unrelated instruction.
            live = self._read(site.addr, 1)[0]
            if live != site.word:
                raise DetourError(
                    f"{thread} 0x{site.addr:05x} holds 0x{live:08x}, "
                    f"scan expected 0x{site.word:08x}"
                )
            self._write(cave.parked, cave.tail(site.word, site.addr + 4))
            self._displaced[thread] = site

        detour = encode_jal(site.addr, cave.entry(delay))
        self._write(site.addr, [detour])
        # One read-back per site is enough; same-site delay changes are just a new jal.
        if new_site and self._read(site.addr, 1)[0] != detour:
            raise DetourError(
                f"{thread} 0x{site.addr:05x} did not take the detour write"
            )

    def _restore(self, thread: str) -> None:
        site = self._displaced.pop(thread, None)
        if site is not None:
            self._write(site.addr, [site.word])

    def restore(self) -> None:
        """Undo every live detour, not just the last one, so no thread is left half-patched."""
        for thread in list(self._displaced):
            self._restore(thread)
