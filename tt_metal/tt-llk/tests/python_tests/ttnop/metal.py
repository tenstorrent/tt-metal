# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""tt-metal backend: detours land in tt-metal's host-side kernel image, not in L1.

The LLK harness loads a kernel ELF once, so a 4-byte L1 poke survives thousands of
re-runs. metal re-writes the whole binary into L1 from its in-process
`ll_api::memory` on *every* launch with no "already configured" guard, and offers
no host-visible seam between "binaries written" and GO — so an L1 poke is erased
before the kernel runs. Poking the host image inverts that: metal re-applies the
perturbation for us on every launch, which is what keeps a variant to a few word
writes instead of a JIT recompile.

Everything about the cave and the detour is inherited from cave.py; only the ELF
lookup and the word reader/writer differ. See the README for the operator view
(slow dispatch, kernel selection, why the cave lives inside .text).
"""

import ctypes
import os
import re
import shutil
import struct
from contextlib import nullcontext
from pathlib import Path

import scanner
from cave import Cave, DetourError, Injector

HERE = Path(__file__).resolve().parent
SHIM = HERE / "libttnop_metal.so"

# TEMP: dump host images as ELFs after each arm so NOP insertion is inspectable.
# Set TTNOP_TEMP_DUMP_ELFS=0 to disable. Remove this block when done.
_TEMP_DUMP_ELFS = os.environ.get("TTNOP_TEMP_DUMP_ELFS", "1") not in ("", "0")
_TEMP_DUMP_DIR = HERE / "temp_elf_dumps"

# tt-metal names TRISCs by index; ttnop names them by role. Same order as the LLK
# harness's TestConfig.KERNEL_COMPONENTS and metal's -DCOMPILE_FOR_TRISC=N.
THREAD_TRISC = {"unpack": 0, "math": 1, "pack": 2}

# ll_api::memory::Loading. TRISC compute is CONTIGUOUS_XIP on both Wormhole and
# Blackhole (wh_hal_tensix.cpp / bh_hal_tensix.cpp). Passed to the shim explicitly
# so a future reader/writer actor — Wormhole NCRISC is plain CONTIGUOUS(1) —
# cannot silently create a second, wrongly-loaded cache entry.
LOADING_CONTIGUOUS_XIP = 2

# metal surfaces a wedge as a timeout/watcher RuntimeError rather than the LLK
# harness's TimeoutError, so the message has to be matched as well as the type.
_HANG_TEXT = re.compile(
    r"\b(?:timed?\s*out|timeout|hang(?:s|ing|ed)?|wedg(?:e|ed)|watchdog)\b",
    re.IGNORECASE,
)


def _cache_root() -> Path:
    """Where jit_build puts compiled kernels (build.cpp get_default_root_path)."""
    override = os.environ.get("TT_METAL_CACHE", "").strip()
    if override:
        return Path(override)
    home = os.environ.get("HOME", "").strip()
    if home and Path(home).exists():
        return Path(home) / ".cache" / "tt-metal-cache"
    return Path("/tmp/tt-metal-cache")


def discover_kernel_dir() -> Path:
    """The <build_key>/kernels/<name>/<hash>/ directory of the compute kernel to perturb.

    Layout is `<cache>/<build_key>/kernels/<kernel_name>/<hash>/trisc<N>/trisc<N>.elf`
    (build.cpp out_kernel_root_ + build_env_manager.cpp get_kernel_binary_path).

    The default is the compute kernel with the most recently written XIP dump,
    which is normally the op that just ran. That is only unambiguous when the test
    drives one op, so set TTNOP_METAL_KERNEL whenever it drives more than one.
    """
    root = _cache_root()
    kernel_filter = os.environ.get("TTNOP_METAL_KERNEL", "").strip()
    candidates = sorted(
        # The XIP dump is written when metal first loads an image, including on a
        # JIT-cache hit; the plain ELF's mtime only says when it was compiled.
        root.glob("*/kernels/*/*/trisc1/trisc1.elf.xip.elf"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if kernel_filter:
        candidates = [p for p in candidates if re.search(kernel_filter, str(p))]
    if not candidates:
        raise DetourError(
            f"no compiled compute kernel under {root}"
            + (f" matching {kernel_filter!r}" if kernel_filter else "")
            + " — run the test once first, and check TT_METAL_CACHE"
        )
    # .../<hash>/trisc1/trisc1.elf.xip.elf -> .../<hash>/
    return candidates[0].parent.parent


def _temp_elf32_text_range(elf_path: Path) -> tuple:
    """TEMP: (file_offset, byte_size, sh_addr) of the .text section in an ELF32."""
    data = elf_path.read_bytes()
    if data[:4] != b"\x7fELF":
        raise DetourError(f"TEMP dump: {elf_path} is not ELF")
    (
        _e_type,
        _e_machine,
        _e_version,
        _e_entry,
        _e_phoff,
        e_shoff,
        _e_flags,
        _e_ehsize,
        _e_phentsize,
        _e_phnum,
        e_shentsize,
        e_shnum,
        e_shstrndx,
    ) = struct.unpack_from("<HHIIIIIHHHHHH", data, 16)
    shstr = e_shoff + e_shstrndx * e_shentsize
    str_off = struct.unpack_from("<I", data, shstr + 16)[0]
    for i in range(e_shnum):
        sh = e_shoff + i * e_shentsize
        name_off, _sh_type, _sh_flags, sh_addr, sh_offset, sh_size = struct.unpack_from(
            "<IIIIII", data, sh
        )
        end = data.index(b"\0", str_off + name_off)
        if data[str_off + name_off : end] == b".text":
            return sh_offset, sh_size, sh_addr
    raise DetourError(f"TEMP dump: no .text in {elf_path}")


def _temp_write_image_into_elf(src_elf: Path, dst_elf: Path, image: "_Image") -> None:
    """TEMP: copy src ELF and overwrite .text with the live packed host image words."""
    shutil.copy2(src_elf, dst_elf)
    file_off, file_size, _sh_addr = _temp_elf32_text_range(src_elf)
    blob = b"".join(
        struct.pack("<I", image.view[i] & 0xFFFFFFFF) for i in range(image.text_words)
    )
    if len(blob) > file_size:
        raise DetourError(
            f"TEMP dump: image text {len(blob)}B exceeds ELF .text {file_size}B"
        )
    with open(dst_elf, "r+b") as fh:
        fh.seek(file_off)
        fh.write(blob)


class _Image:
    """A writable view of the packed image metal pushes to L1 for one thread."""

    def __init__(self, view, text_words: int, text_start: int):
        self.view = view
        self.text_words = text_words
        self.text_start = text_start

    def index(self, vaddr: int) -> int:
        """Word index of a scanned vaddr in the packed image.

        CONTIGUOUS_XIP packs the text segment first and .text is the first section
        in that segment, so text word 0 is image word 0.
        """
        offset = (vaddr - self.text_start) // 4
        if not 0 <= offset < self.text_words:
            raise DetourError(
                f"0x{vaddr:08x} is outside the image's {self.text_words} text words"
            )
        return offset

    def read(self, addr: int, count: int) -> list:
        base = self.index(addr)
        self.index(addr + (count - 1) * 4)
        return [self.view[base + i] for i in range(count)]

    def write(self, addr: int, words: list) -> None:
        base = self.index(addr)
        # Bound the tail too: a cave that ran off the end of .text would corrupt .data.
        self.index(addr + (len(words) - 1) * 4)
        for offset, word in enumerate(words):
            self.view[base + offset] = word & 0xFFFFFFFF


class MetalBackend:
    name = "metal"

    def __init__(self, max_delay: int):
        self.max_delay = max_delay
        self._shim = None
        self._elf_dir = None
        self._scans = {}
        self._images = {}
        self._injectors = {}

    # -- ELF lookup --------------------------------------------------------

    @property
    def kernel(self) -> str:
        """The JIT kernel name being perturbed, e.g. "eltwise_sfpu". Recorded so a
        report's reproduce line can re-select the same kernel."""
        return self._elf_dir.parent.name if self._elf_dir else ""

    def _elf_for(self, thread: str) -> Path:
        """On-disk kernel ELF for a thread (plain ELF, not the XIP dump).

        pathlib collapses the `//` that metal's cache key contains; use
        `_elf_cache_key` when talking to get_risc_binary. The plain ELF still
        carries the DWARF for addr2line.
        """
        trisc = THREAD_TRISC[thread]
        return self._elf_dir / f"trisc{trisc}" / f"trisc{trisc}.elf"

    def _elf_cache_key(self, thread: str) -> str:
        """Path string metal keyed its get_risc_binary image cache on.

        `Kernel::set_full_name` stores `name/hash/` (trailing slash).
        `BuildEnvManager::get_kernel_binary_path` then does `path += "/triscN/..."`,
        so the cache key has a `//` before `triscN`. A normalized spelling misses
        the cache, builds a second image nobody launches, and every poke is void.
        """
        trisc = THREAD_TRISC[thread]
        # Keep as a plain str: Path() collapses '//' to '/'.
        return f"{self._elf_dir}//trisc{trisc}/trisc{trisc}.elf"

    def _xip_elf_for(self, thread: str) -> Path:
        """The post-XIP dump metal writes beside each ELF as it loads it.

        Scanning this and not the plain ELF is load-bearing: XIPify rewrites
        text-targeting LUI into AUIPC, so the plain ELF's words would both
        mis-report what is at a site and let a now-PC-relative instruction slip past
        the relocatability filter and be moved into the cave.
        """
        return Path(str(self._elf_for(thread)) + ".xip.elf")

    # -- baseline ----------------------------------------------------------

    def watch_baseline(self):
        """Nothing to wrap. A ttnn test's own asserts are the golden, so the clean
        pass simply passing is the whole baseline, and the plugin already skips a
        case whose clean pass raised."""
        return lambda: None

    def ready(self) -> bool:
        return True

    # -- host image binding ------------------------------------------------

    def _load_shim(self):
        if self._shim is not None:
            return self._shim
        if not SHIM.is_file():
            raise DetourError(f"{SHIM} missing — run `make metal_shim` in ttnop/")
        shim = ctypes.CDLL(str(SHIM))
        shim.ttnop_image_words.restype = ctypes.POINTER(ctypes.c_uint32)
        shim.ttnop_image_words.argtypes = [ctypes.c_char_p, ctypes.c_uint32] + [
            ctypes.POINTER(ctypes.c_uint32)
        ] * 4
        self._shim = shim
        return shim

    def _bind_image(self, thread: str) -> _Image:
        """Take a writable view of the image metal will push to L1 for this thread.

        The pointer is into tt-metal's permanent per-path image cache, so it stays
        valid for the whole sweep and every write through it is picked up by the next
        launch. scans() clears bindings between cases because the selected ELF path
        may change.
        """
        shim = self._load_shim()
        elf_key = self._elf_cache_key(thread)
        xip = self._xip_elf_for(thread)
        xip_stamp = xip.stat().st_mtime_ns
        # total/text/loading/text_addr are out-params; the last two report what the
        # cache really holds (unused here — the mtime check is the live one).
        total, text, loading, text_addr = (ctypes.c_uint32(0) for _ in range(4))
        words = shim.ttnop_image_words(
            elf_key.encode(),
            LOADING_CONTIGUOUS_XIP,
            ctypes.byref(total),
            ctypes.byref(text),
            ctypes.byref(loading),
            ctypes.byref(text_addr),
        )
        if not words:
            raise DetourError(
                f"tt-metal could not hand back an image for {elf_key} "
                "(is the device open yet?)"
            )
        # get_risc_binary keys its cache on the path *string*. Had metal spelled this
        # path differently, try_emplace would have missed and constructed a second
        # image — one nobody launches, so every poke would land in the void and the
        # sweep would read 0% at every count. Constructing a CONTIGUOUS_XIP image
        # rewrites <elf>.xip.elf (tt_memory.cpp), and a cache *hit* constructs
        # nothing, so the dump's mtime says exactly which one happened.
        if xip.stat().st_mtime_ns != xip_stamp:
            raise DetourError(
                f"{thread}: the shim built its own image for {elf_key} instead of "
                "reusing metal's, so metal must have cached this kernel under a "
                "different path spelling. Pokes would not reach the device; check "
                "TT_METAL_CACHE for symlinks or a non-canonical path."
            )
        view = ctypes.cast(words, ctypes.POINTER(ctypes.c_uint32 * total.value))[0]
        return _Image(view, text.value, self._scans[thread].text_start)

    # -- the sweep ---------------------------------------------------------

    def scans(self, site_mode: str) -> dict:
        if "TT_METAL_SLOW_DISPATCH_MODE" not in os.environ:
            raise DetourError(
                "TTNOP_METAL=1 requires TT_METAL_SLOW_DISPATCH_MODE=1; fast dispatch "
                "snapshots the image into a DRAM buffer before ttnop mutates it"
            )
        self._elf_dir = discover_kernel_dir()
        # One backend instance serves the whole pytest process. A later case may
        # select a different JIT kernel, so thread name alone cannot identify an image.
        self._images.clear()
        self._injectors.clear()
        self._scans = {}
        for thread in THREAD_TRISC:
            xip = self._xip_elf_for(thread)
            if not xip.is_file():
                raise DetourError(
                    f"{xip} missing — the XIP dump is what carries the post-XIP "
                    "words; unset TT_METAL_DISABLE_XIP_DUMP and re-run"
                )
            self._scans[thread] = scanner.scan(str(xip), site_mode)
            self._images[thread] = self._bind_image(thread)
        # TEMP: clean baseline ELFs (no detour) for before/after compare.
        if _TEMP_DUMP_ELFS:
            self._temp_dump_all("00_baseline", note="clean image before any arm")
        return self._scans

    def _temp_dump_all(self, label: str, note: str = "") -> None:
        """TEMP: write every thread's live host image into a copy of its XIP ELF."""
        out = _TEMP_DUMP_DIR / self.kernel
        out.mkdir(parents=True, exist_ok=True)
        # Fresh manifest at baseline so a re-run does not append onto an old sweep.
        if label.startswith("00_"):
            (out / "TEMP_MANIFEST.txt").write_text(
                f"TEMP dumps for kernel={self.kernel}\n", encoding="utf-8"
            )
        with open(out / "TEMP_MANIFEST.txt", "a", encoding="utf-8") as log:
            log.write(f"\n=== {label} === {note}\n")
            for thread, image in self._images.items():
                trisc = THREAD_TRISC[thread]
                src = self._xip_elf_for(thread)
                dst = out / f"{label}__{thread}_trisc{trisc}.elf"
                _temp_write_image_into_elf(src, dst, image)
                log.write(f"  {thread}: {dst.name}  text_words={image.text_words}\n")
        print(f"TEMP: dumped ELFs -> {out}/ ({label})", flush=True)

    def _temp_dump_after_arm(
        self, thread: str, site, delay: int, filler_word: int
    ) -> None:
        """TEMP: snapshot all thread ELFs after a detour is armed."""
        label = (
            f"{thread}_{site.op}_0x{site.addr:05x}_n{delay}_" f"fill{filler_word:08x}"
        )
        # Sanitize for filesystems.
        label = re.sub(r"[^\w.\-]+", "_", label)
        out = _TEMP_DUMP_DIR / self.kernel
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "TEMP_MANIFEST.txt", "a", encoding="utf-8") as log:
            log.write(
                f"\n=== {label} === armed {thread} {site.op}@0x{site.addr:05x} "
                f"delay={delay} filler=0x{filler_word:08x}\n"
            )
            for thr, image in self._images.items():
                trisc = THREAD_TRISC[thr]
                src = self._xip_elf_for(thr)
                dst = out / f"{label}__{thr}_trisc{trisc}.elf"
                _temp_write_image_into_elf(src, dst, image)
                log.write(f"  {thr}: {dst.name}\n")
                if thr == thread:
                    site_word = image.read(site.addr, 1)[0]
                    cave_start = self._scans[thr].cave_start
                    n = min(max(delay, 1), 32)
                    fillers = image.read(cave_start, n)
                    log.write(
                        f"    SITE 0x{site.addr:05x} = 0x{site_word:08x} "
                        f"(expect JAL into cave)\n"
                        f"    CAVE 0x{cave_start:05x} first {n} words = "
                        + " ".join(f"0x{w:08x}" for w in fillers)
                        + f"  (expect filler 0x{filler_word:08x})\n"
                    )
        print(f"TEMP: dumped ELFs -> {out}/ ({label})", flush=True)

    def injector_for(self, thread: str) -> Injector:
        # Each thread is a separate host buffer, so each needs its own injector:
        # a single (read, write) pair cannot tell two images apart.
        if thread not in self._injectors:
            image = self._images[thread]
            injector = Injector(
                read_words=image.read,
                write_words=image.write,
                max_delay=self.max_delay,
            )
            # TEMP: wrap arm to snapshot ELFs after NOP/detour writes land.
            if _TEMP_DUMP_ELFS:
                _orig_arm = injector.arm

                def _temp_arm(thread, scan, site, delay, filler_word, _arm=_orig_arm):
                    _arm(thread, scan, site, delay, filler_word)
                    self._temp_dump_after_arm(thread, site, delay, filler_word)

                injector.arm = _temp_arm  # TEMP
            self._injectors[thread] = injector
        return self._injectors[thread]

    def restore(self) -> None:
        for injector in self._injectors.values():
            injector.restore()

    def quiet(self):
        # ttnn logs through loguru at levels the harness does not silence, and a
        # failing variant here is an ordinary assert rather than a tile dump, so
        # there is nothing worth muting.
        return nullcontext()

    def classify(self, err) -> tuple:
        # AssertionError before the hang regex: a golden mismatch that mentions
        # "timeout" must not abort the sweep as an unrecoverable hang.
        if isinstance(err, AssertionError):
            return "mismatch", str(err)
        if isinstance(err, TimeoutError) or _HANG_TEXT.search(str(err)):
            return "hang", str(err)
        return "error", f"{type(err).__name__}: {err}"

    def recover(self, replay) -> bool:
        # A metal hang takes the dispatcher with it, not just the Tensix, so there is
        # no in-process recovery to attempt (the LLK backend's soft-reset + reload has
        # no equivalent). Every later variant would report garbage: stop and let the
        # operator `tt-smi -r`.
        return False

    def finish(self) -> None:
        # Nothing to invalidate: the images are host-side and the injectors have
        # already been restored, so the next case re-discovers from a clean image.
        pass


def _offline_check() -> None:
    """Prove the image offset maths and the detour it writes, with no device.

    Everything the metal backend adds over the LLK one is address arithmetic: a
    scanned vaddr has to land on the right word of a packed image whose text base
    is not the vaddr base. Get that wrong and the sweep corrupts an unrelated
    instruction or silently patches nothing, so it is worth being able to check it
    from a laptop. Run with `python3 metal.py`.
    """

    def decode_jal(word: int, at: int) -> int:
        """Inverse of cave.encode_jal, so a bad encoding cannot pass by symmetry."""
        imm = (
            (((word >> 31) & 0x1) << 20)
            | (((word >> 21) & 0x3FF) << 1)
            | (((word >> 20) & 0x1) << 11)
            | (((word >> 12) & 0xFF) << 12)
        )
        if imm & (1 << 20):
            imm -= 1 << 21
        return at + imm

    # A kernel whose .text is 512 words at vaddr 0x6000, with the last 128 the cave.
    text_start, text_words, max_delay = 0x6000, 512, 100
    cave_start = text_start + (text_words - 128) * 4
    site_addr, site_word = text_start + 40 * 4, 0xDEADBEEF

    view = (ctypes.c_uint32 * text_words)()
    view[(site_addr - text_start) // 4] = site_word
    image = _Image(view, text_words, text_start)

    assert image.index(text_start) == 0, "text base must map to image word 0"
    assert image.index(site_addr) == 40
    for bad in (text_start - 4, text_start + text_words * 4):
        try:
            image.index(bad)
        except DetourError:
            pass
        else:
            raise AssertionError(f"0x{bad:x} should be out of bounds")

    cave = Cave(cave_start, cave_start + 128 * 4, max_delay)
    site = scanner.Site(index=0, addr=site_addr, word=site_word, op="TEST", sfpu=False)

    class _Scan:
        cave_start, cave_limit, elf = cave.start, cave.limit, "<offline>"

    injector = Injector(image.read, image.write, max_delay=max_delay)
    for delay in (0, 1, 37, max_delay):
        injector.arm("unpack", _Scan, site, delay, 0x08000000)

        landed = decode_jal(view[image.index(site_addr)], site_addr)
        assert landed == cave.entry(
            delay
        ), f"delay {delay}: site jumps to 0x{landed:x}, want 0x{cave.entry(delay):x}"
        # Exactly `delay` fillers are executed: the entry point sits that many words
        # short of the parked instruction, and every word from there on is a filler.
        executed = (cave.parked - landed) // 4
        assert executed == delay, f"delay {delay}: {executed} fillers would run"
        for word in range(delay):
            assert view[image.index(landed + word * 4)] == 0x08000000
        assert view[image.index(cave.parked)] == site_word, "displaced word not parked"
        back = decode_jal(view[image.index(cave.ret)], cave.ret)
        assert back == site_addr + 4, f"returns to 0x{back:x}, want 0x{site_addr + 4:x}"

    injector.restore()
    assert view[image.index(site_addr)] == site_word, "restore did not undo the detour"

    print(f"offline check passed (cave 0x{cave.start:x}..0x{cave.end:x})")


if __name__ == "__main__":
    _offline_check()
