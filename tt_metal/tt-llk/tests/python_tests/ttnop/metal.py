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
from contextlib import nullcontext
from pathlib import Path

import scanner
from cave import DetourError, Injector

HERE = Path(__file__).resolve().parent
SHIM = HERE / "libttnop_metal.so"

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

    def reset_case(self) -> None:
        pass

    def watch_baseline(self, perturber=None):
        """Nothing to wrap. A ttnn test's own asserts are the golden, so the clean
        pass simply passing is the whole baseline, and the plugin already skips a
        case whose clean pass raised."""
        return lambda: None

    def ready(self, nodeid: str = "") -> bool:
        return True

    def prepare_arm(self) -> None:
        pass

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
        return self._scans

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
