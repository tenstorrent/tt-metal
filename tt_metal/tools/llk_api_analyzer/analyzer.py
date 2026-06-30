"""Top-level orchestration: turn a run directory into a :class:`RunAnalysis`.

Ties together discovery (find compute kernels), DWARF loading, per-thread
extraction (LLK API calls) and descriptor parsing (data formats / tile sizes /
sync / dest-accumulate config).

DWARF note: the compute-kernel ELFs are linked ``ET_EXEC`` files that retain
RISC-V relocations against their ``.debug_*`` sections (linker-relaxation
bookkeeping). ``pyelftools`` has no RISC-V relocation handler, so we load DWARF
with ``relocate_dwarf_sections=False``. This is correct here because the linker
has already baked final values into the debug-section bytes and nothing the
analyzer reads is address-dependent (DIE tree, names, template/runtime constants
and location *expressions* are all relocation-independent); the relocations only
patch code addresses and loclist/line PC-range bounds, which we never use.
"""

from __future__ import annotations

from pathlib import Path

from elftools.elf.elffile import ELFFile

from .descriptors import parse_descriptors
from .discovery import KernelArtifacts, discover_compute_kernels
from .extractor import ExtractorConfig, LlkApiExtractor
from .model import ApiCall, ComputeThread, KernelAnalysis, RunAnalysis


class LlkAnalyzer:
    """Analyzes compute kernels to recover the LLK APIs they invoke."""

    def __init__(self, extractor_config: ExtractorConfig | None = None):
        self._extractor_config = extractor_config or ExtractorConfig()

    def analyze_run(self, root: str | Path) -> RunAnalysis:
        """Analyze every compute kernel found beneath ``root``."""
        kernels = discover_compute_kernels(root)
        analysis = RunAnalysis(root=str(root))
        for artifacts in kernels:
            analysis.kernels.append(self.analyze_kernel(artifacts))
        return analysis

    def analyze_kernel(self, artifacts: KernelArtifacts) -> KernelAnalysis:
        """Analyze a single compute kernel (all available TRISC threads)."""
        result = KernelAnalysis(name=artifacts.name, path=str(artifacts.directory))

        if artifacts.descriptors_header is not None:
            try:
                result.descriptors = parse_descriptors(artifacts.descriptors_header)
            except Exception as exc:  # noqa: BLE001 - keep going, just record it
                result.errors.append(f"descriptor parse failed: {exc}")

        for trisc_id, elf_path in sorted(artifacts.trisc_elfs.items()):
            thread = ComputeThread.from_trisc(trisc_id)
            try:
                result.api_calls.extend(self._analyze_elf(elf_path, thread))
                result.threads_analyzed.append(thread)
            except Exception as exc:  # noqa: BLE001 - one bad ELF must not abort the run
                result.errors.append(f"{thread.value} ({elf_path.name}): {exc}")

        return result

    def _analyze_elf(self, elf_path: Path, thread: ComputeThread) -> list[ApiCall]:
        with open(elf_path, "rb") as handle:
            elffile = ELFFile(handle)
            if not elffile.has_dwarf_info():
                raise RuntimeError("no DWARF debug info (rebuild with TT_METAL_RISCV_DEBUG_INFO=1)")
            # relocate_dwarf_sections=False: skip the (unsupported) RISC-V debug
            # relocations; see the module docstring for why this is correct.
            dwarf = elffile.get_dwarf_info(relocate_dwarf_sections=False)
            return LlkApiExtractor(dwarf, thread, self._extractor_config).extract()
