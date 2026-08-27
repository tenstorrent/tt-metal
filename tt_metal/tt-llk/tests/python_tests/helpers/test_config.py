# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import fcntl
import glob
import os
import re
import shlex
import shutil
import struct
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, fields
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Any, ClassVar, List

import numpy as np
import pytest
from filelock import FileLock
from ttexalens.tt_exalens_lib import (
    TTException,
    load_elf,
    parse_elf,
    read_word_from_device,
)

from . import device as device_module
from . import golden_generators as golden_generators_module
from .chip_architecture import ChipArchitecture, get_chip_architecture
from .data_format_inference import data_formats, is_format_combination_outlier
from .device import (
    CHIP_DEFAULT_BOOT_MODES,
    KERNEL_COMPLETE,
    TRISC_CORES,
    BootMode,
    RiscCore,
    commit_brisc_command,
    commit_tensix_soft_reset,
    exalens_device_setup,
    handle_if_assert_hit,
    reset_mailboxes,
    set_tensix_soft_reset,
    wait_brisc_boot_ready,
)
from .device_io import read_from_device, write_to_device, write_words_to_device
from .device_print import aux_size_for
from .format_config import (
    BLACKHOLE_DATA_FORMAT_ENUM_VALUES,
    FORMATS_CONFIG_STRUCT_COMPILETIME,
    FORMATS_CONFIG_STRUCT_RUNTIME,
    QUASAR_DATA_FORMAT_ENUM_VALUES,
    WORMHOLE_DATA_FORMAT_ENUM_VALUES,
    DataFormat,
    InputOutputFormat,
)
from .golden_generators import (
    GeneratorProxy,
    ProxyMode,
    dummy_golden_generator,
    get_golden_proxied,
)
from .llk_params import (
    BriscCmd,
    DestAccumulation,
    L1Accumulation,
    Mailboxes,
    MailboxesCoverage,
    MailboxesCoverageQuasar,
    MailboxesQuasar,
)
from .logger import logger
from .stimuli_config import StimuliConfig
from .target_config import TestTargetConfig
from .test_variant_parameters import (
    IN_TILE_DIMS,
    NUM_FACES,
    RuntimeParameter,
    TemplateParameter,
)
from .utils import create_directories, run_shell_command

TEMP_DIR = Path(tempfile.gettempdir())


class ProfilerBuild(Enum):
    Yes = "true"
    No = "false"


class CoverageBuild(Enum):
    Yes = "true"
    No = "false"


class BuildMode(Enum):
    DEFAULT = 0  # compile + execute
    PRODUCE = 1  # compile only
    CONSUME = 2  # execute only


class StimuliMode(Enum):
    INLINE = 0  # compute during test (default)
    GENERATE_ONLY = 1  # compute + save to disk, skip execution
    LOAD_CACHED = 2  # load from disk, skip computation


@dataclass
class TestOutcome:
    result: Any = None
    # Lines emitted by DEVICE_PRINT() during this run.
    # Empty if it's disabled.
    device_print_lines: list = field(default_factory=list)


class TestConfig:

    # === STATIC VARIABLES ===

    # Architecture Selection
    ARCH_NON_COMPUTE: ClassVar[str]
    ARCH_COMPUTE: ClassVar[str]
    ARCH_DEFINE: ClassVar[str]
    ARCH_LLK_ROOT: ClassVar[str]
    ARCH: ClassVar[str]
    ARCH_SPECIFIC_OPTIONS: ClassVar[str] = ""
    CHIP_ARCH: ClassVar[ChipArchitecture]
    DATA_FORMAT_ENUM: ClassVar[dict]

    # Artefact directories. Prefer GHA RUNNER_TEMP (disk) over the system temp
    # directory (often tmpfs) so compile artefacts do not accumulate in RAM and
    # OOM the runner (exit 137).
    DEFAULT_ARTEFACTS_PATH: ClassVar[Path] = TEMP_DIR / "tt-llk-build"
    ARTEFACTS_DIR: ClassVar[Path]
    SHARED_DIR: ClassVar[str]
    SHARED_OBJ_DIR: ClassVar[str]
    SHARED_ELF_DIR: ClassVar[str]
    COVERAGE_INFO_DIR: ClassVar[str]
    SYNC_DIR: ClassVar[Path]
    PERF_DATA_DIR: ClassVar[Path]
    DEFAULT_STIMULI_CACHE_FOLDER: ClassVar[Path]

    # Sources directories
    LLK_ROOT: ClassVar[Path]
    TESTS_WORKING_DIR: ClassVar[Path]
    TOOL_PATH: ClassVar[Path]

    HELPERS: ClassVar[Path]
    RISCV_SOURCES: ClassVar[Path]
    LINKER_SCRIPTS: ClassVar[Path]

    # Toolchain paths
    GXX: ClassVar[str]
    OBJDUMP: ClassVar[str]
    OBJCOPY: ClassVar[str]
    ELF_SIZE: ClassVar[str]
    GCOV: ClassVar[str]
    GCOV_TOOL: ClassVar[str]

    # Compilation options
    OPTIONS_ALL: ClassVar[str] = None
    OPTIONS_LINK: ClassVar[str] = None
    INITIAL_OPTIONS_COMPILE: ClassVar[str] = None
    INCLUDES: ClassVar[List[str]] = []
    # Out-of-tree -I header dirs from add_include_dirs(). Prepend shadows
    # in-tree copies; append sits after them.
    EXTRA_INCLUDE_PREPEND: ClassVar[List[str]] = []
    EXTRA_INCLUDE_APPEND: ClassVar[List[str]] = []
    # Extra -I dirs for #include <foo.cpp> (tests/helpers/src style).
    EXTRA_SRC_INCLUDE_PREPEND: ClassVar[List[str]] = []
    EXTRA_SRC_INCLUDE_APPEND: ClassVar[List[str]] = []
    # Truncated sha256 in _safe_artefact_key — on-disk name only, not a variant id.
    ARTEFACT_KEY_HASH_CHARS: ClassVar[int] = 12
    WITH_COVERAGE: ClassVar[bool] = False

    OPTIONS_COMPILE: ClassVar[str] = None
    MEMORY_LAYOUT_LD_SCRIPT: ClassVar[str] = None
    NON_COVERAGE_OPTIONS_COMPILE: ClassVar[str] = None

    SHARED_ARTEFACTS_AVAILABLE: ClassVar[bool] = False
    PROFILER_SHARED_ARTEFACTS_AVAILABLE: ClassVar[bool] = False
    KERNEL_COMPONENTS: ClassVar[list[str]] = ["unpack", "math", "pack"]

    # === Runtime static variables, for keeping context of multiple test runs
    CURRENT_LOADED_CONFIG: ClassVar[str] = "uninitialised"
    BUILD_MODE: ClassVar[BuildMode] = BuildMode.DEFAULT
    STIMULI_MODE: ClassVar[StimuliMode] = StimuliMode.INLINE
    SKIP_JUST_FOR_COMPILE_MARKER: ClassVar[str] = "SKIPPED_JUST_FOR_COMPILE"
    SKIP_JUST_FOR_STIMULI_MARKER: ClassVar[str] = "SKIPPED_JUST_FOR_STIMULI"
    _BUILD_DIRS_CREATED: ClassVar[bool] = False
    SPEED_OF_LIGHT: ClassVar[bool] = (
        False  # Should everything be converted to compile-time arguments?
    )

    TEST_TARGET: ClassVar[TestTargetConfig] = TestTargetConfig()

    WORKER_ID: ClassVar[str] = "master"
    TENSIX_LOCATION: ClassVar[str] = "0,0"
    # xdist worker index waiting for Exalens. setup_mode cannot ask the card yet:
    # silicon init_ttexalens and the RTL remote connect both happen after it.
    _PENDING_WORKER_INDEX: ClassVar[int | None] = None
    STIMULI_ADDRESS_MAP: ClassVar[dict[str, int]] = {}
    SIMULATOR_TIMEOUT: ClassVar[int] = 600

    # When the infrastructure itself needs to be tested, some functionality like compiling the artefacts and writing them
    # to tmpfs can be skipped (eg. object, elf and coverage data files etc.). This flag is used to skip such code to enable fast execution of infra tests.
    INFRA_TESTING: ClassVar[bool] = False

    # Determinism check: number of times each variant is executed on device.
    # When > 1, run() re-runs the kernel and asserts every run produces a
    # bit-identical result buffer (see --bit-exact-runs).
    BIT_EXACT_RUNS: ClassVar[int] = 1

    # CLI perf counter flags
    ENABLE_PERF_COUNTERS: ClassVar[bool] = False
    DUMP_RAW_COUNTERS: ClassVar[bool] = False
    DUMP_RAW_METRICS: ClassVar[bool] = False
    DUMP_CSV_COUNTERS: ClassVar[bool] = False

    # === Addresses ===
    RUNTIME_ADDRESS_NON_COVERAGE: ClassVar[int] = 0x20000
    RUNTIME_ADDRESS_COVERAGE: ClassVar[int] = 0x6E000
    TRISC_PROFILER_BARRIER_ADDRESS: ClassVar[int] = 0x16AFF4
    TRISC_START_ADDRS: ClassVar[list[int]] = [0x16DFF0, 0x16DFF4, 0x16DFF8]
    THREAD_PERFORMANCE_DATA_BUFFER_LENGTH = 0x400
    THREAD_PERFORMANCE_DATA_BUFFER = [
        0x16B000,  # Unpack
        0x16C000,  # Math
        0x16D000,  # Pack
    ]

    # Performance counter L1 memory addresses
    # NOTE: These addresses must match the values in tests/helpers/include/counters.h
    # Shared config + per-zone data layout (must match counters.h).
    # Shared config (200 words = 800 B) at base; per-zone data (5 bank-cycle
    # words + 200 counter-count words + sync = 860 B) follows.
    # 8 zones × 860 + 800 = 7680 B, fits below profiler region at 0x16AFF4.
    PERF_COUNTERS_BASE_ADDR: ClassVar[int] = 0x169000
    PERF_COUNTERS_MAX_ZONES: ClassVar[int] = 8  # Max zones (must match counters.h)
    _PERF_COUNTERS_CONFIG_WORDS: ClassVar[int] = 200
    _PERF_COUNTERS_DATA_WORDS: ClassVar[int] = 200  # per-zone counter-count slots
    _PERF_COUNTERS_BANK_CYCLES_WORDS: ClassVar[int] = 5  # OUT_L per bank (5 banks)

    # Shared config region
    PERF_COUNTERS_CONFIG_ADDR: ClassVar[int] = PERF_COUNTERS_BASE_ADDR
    PERF_COUNTERS_ZONES_BASE: ClassVar[int] = (
        PERF_COUNTERS_BASE_ADDR + _PERF_COUNTERS_CONFIG_WORDS * 4
    )

    # Per-zone data layout: [bank_cycles (5)][counter_counts (DATA_WORDS)][sync (1) + pad]
    _PERF_COUNTERS_ZONE_DATA_BYTES: ClassVar[int] = (
        _PERF_COUNTERS_BANK_CYCLES_WORDS + _PERF_COUNTERS_DATA_WORDS
    ) * 4  # 820 B = 20 (cycles) + 800 (counts)

    # Size of one full zone block (data + sync/pad)
    PERF_COUNTERS_ZONE_SIZE: ClassVar[int] = _PERF_COUNTERS_ZONE_DATA_BYTES + 40

    # Zone-0 flat addresses (kept for legacy callers; prefer zone_*_addr helpers below).
    PERF_COUNTERS_DATA_ADDR: ClassVar[int] = PERF_COUNTERS_ZONES_BASE
    PERF_COUNTERS_SYNC_CTRL_ADDR: ClassVar[int] = (
        PERF_COUNTERS_ZONES_BASE + _PERF_COUNTERS_ZONE_DATA_BYTES
    )

    # Trailing metadata written by PerfCounterManager (must match counters.h):
    # enabled_flag (4 B) + bank_mask (4 B) + valid_count[MAX_ZONES] (4 B each).
    _PERF_COUNTERS_TRAILING_METADATA_BYTES: ClassVar[int] = (
        4 + 4 + PERF_COUNTERS_MAX_ZONES * 4
    )

    # Total L1 reservation: shared config + per-zone blocks + trailing metadata.
    PERF_COUNTERS_SIZE: ClassVar[int] = (
        _PERF_COUNTERS_CONFIG_WORDS * 4
        + PERF_COUNTERS_MAX_ZONES * PERF_COUNTERS_ZONE_SIZE
        + _PERF_COUNTERS_TRAILING_METADATA_BYTES
    )

    # Legacy alias — sums per-zone bytes for back-compat with old callers
    _PERF_COUNTERS_BUFFER_SIZE: ClassVar[int] = _PERF_COUNTERS_ZONE_DATA_BYTES

    # Device print buffer. It sits above loaders, and under RUNTIME_ARGS_START.
    # Coverage builds extend TRISC sections past this address; device print
    # is disabled under coverage so the conflict doesn't matter.
    DEVICE_PRINT_BUFFER_BASE: ClassVar[int] = 0x15000
    # Matches RUNTIME_ARGS_START in the non-coverage linker scripts
    # (memory.{wormhole,blackhole,quasar}.ld). Passed to the build as
    # -DLLK_RUNTIME_ARGS_START so dprint.h can static_assert that the
    # device print buffer doesn't overlap RUNTIME_ARGS.
    DEVICE_PRINT_RUNTIME_ARGS_START: ClassVar[int] = 0x20000
    PROCESSOR_COUNT: ClassVar[int] = 0
    DEVICE_PRINT_BUFFER_SIZE: ClassVar[int] = 0x4000  # WH/BH/Quasar TRISC
    DEVICE_PRINT_BUFFER_SIZE2: ClassVar[int] = 0x2000  # Quasar DM
    DEVICE_PRINT_ENABLED: ClassVar[bool] = False

    # Single source of truth that maps component, risc_id and display name.
    # Passed to dprint.h through -DPROCESSOR_INDEX at build time, and
    # _risc_names_tensix and make_device_print_parser in device_print.py.
    # The kernel needs it to tell the host who it is when it prints, and
    # the host needs it to map it into a string and find the ELF on disk.
    # Quasar overrides this in setup_arch.
    RISC_INFO: ClassVar[dict[str, tuple[int, str]]] = {
        "unpack": (2, "UNPACK"),
        "math": (3, "MATH"),
        "pack": (4, "PACK"),
    }

    @staticmethod
    def device_print_buffers() -> list[tuple[int, int, int]]:
        """Per-buffer (base_address, size, processor_count) the host parser reads.

        Mirrors DevicePrintMemoryLayout (see dprint_buffer.h) and the dprint server's
        get_core_buffers(): WH/BH have a single buffer; Quasar has a TRISC/compute
        buffer (16 processors) immediately followed by a DM buffer (8 processors).
        processor_count drives the Aux header size, so it must match the device-side
        DevicePrintBuffer template arguments.
        """
        base = TestConfig.DEVICE_PRINT_BUFFER_BASE
        if TestConfig.ARCH == ChipArchitecture.QUASAR:
            return [
                (
                    base,
                    TestConfig.DEVICE_PRINT_BUFFER_SIZE,
                    16,
                ),  # TRISC, processor_offset 8
                (
                    base + TestConfig.DEVICE_PRINT_BUFFER_SIZE,
                    TestConfig.DEVICE_PRINT_BUFFER_SIZE2,
                    8,
                ),  # DM, processor_offset 0
            ]
        return [(base, TestConfig.DEVICE_PRINT_BUFFER_SIZE, TestConfig.PROCESSOR_COUNT)]

    @staticmethod
    def setup_arch():
        TestConfig.CHIP_ARCH = get_chip_architecture()
        match TestConfig.CHIP_ARCH:
            case ChipArchitecture.WORMHOLE:
                TestConfig.ARCH_NON_COMPUTE = "-mcpu=tt-wh"
                TestConfig.ARCH_COMPUTE = "-mcpu=tt-wh-tensix"
                TestConfig.ARCH_DEFINE = "-DARCH_WORMHOLE"
                TestConfig.ARCH_LLK_ROOT = "tt_llk_wormhole_b0"
                TestConfig.ARCH = ChipArchitecture.WORMHOLE
                TestConfig.DATA_FORMAT_ENUM = WORMHOLE_DATA_FORMAT_ENUM_VALUES
                TestConfig.PROCESSOR_COUNT = 5
            case ChipArchitecture.BLACKHOLE:
                TestConfig.ARCH_NON_COMPUTE = "-mcpu=tt-bh"
                TestConfig.ARCH_COMPUTE = "-mcpu=tt-bh-tensix"
                TestConfig.ARCH_DEFINE = "-DARCH_BLACKHOLE"
                TestConfig.ARCH_LLK_ROOT = "tt_llk_blackhole"
                TestConfig.ARCH = ChipArchitecture.BLACKHOLE
                TestConfig.DATA_FORMAT_ENUM = BLACKHOLE_DATA_FORMAT_ENUM_VALUES
                TestConfig.PROCESSOR_COUNT = 5
            case ChipArchitecture.QUASAR:
                TestConfig.ARCH_NON_COMPUTE = "-mcpu=tt-qsr32"
                TestConfig.ARCH_COMPUTE = "-mcpu=tt-qsr32-tensix"
                TestConfig.ARCH_DEFINE = "-DARCH_QUASAR"
                TestConfig.ARCH_LLK_ROOT = "tt_llk_quasar"
                TestConfig.ARCH = ChipArchitecture.QUASAR
                TestConfig.DATA_FORMAT_ENUM = QUASAR_DATA_FORMAT_ENUM_VALUES
                TestConfig.KERNEL_COMPONENTS = ["unpack", "math", "pack", "sfpu"]
                TestConfig.RISC_INFO = {
                    "unpack": (8, "UNPACK"),
                    "math": (9, "MATH"),
                    "pack": (10, "PACK"),
                    "sfpu": (11, "SFPU"),
                }
                TestConfig.PROCESSOR_COUNT = 24
                TestConfig.TRISC_START_ADDRS = [
                    0x16DFF0,
                    0x16DFF4,
                    0x16DFF8,
                    0x16DFFC,
                ]
                TestConfig.THREAD_PERFORMANCE_DATA_BUFFER = [
                    0x16B000,  # Unpack
                    0x16C000,  # Math
                    0x16D000,  # Pack
                    0x16E000,  # SFPU
                ]
                TestConfig.TRISC_PROFILER_BARRIER_ADDRESS = (
                    0x16AFF0  # BARRIER_START for 4 cores
                )
            case _:
                raise ValueError(
                    "Must provide CHIP_ARCH environment variable (wormhole / blackhole / quasar)"
                )

    @staticmethod
    def resolve_artefacts_path() -> Path:
        """Use $RUNNER_TEMP/tt-llk-build in GHA, else tempfile.gettempdir()/tt-llk-build."""
        runner_temp = os.environ.get("RUNNER_TEMP")
        if runner_temp:
            return Path(runner_temp) / "tt-llk-build"
        return TestConfig.DEFAULT_ARTEFACTS_PATH

    @staticmethod
    def setup_paths(sources_path: Path):
        TestConfig.ARTEFACTS_DIR = TestConfig.resolve_artefacts_path()

        TestConfig.LLK_ROOT = sources_path
        TestConfig.TESTS_WORKING_DIR = TestConfig.LLK_ROOT / "tests"
        TestConfig.TOOL_PATH = TestConfig.LLK_ROOT / "tests/sfpi/compiler/bin"

        TestConfig.HELPERS = TestConfig.TESTS_WORKING_DIR / "helpers"
        TestConfig.RISCV_SOURCES = TestConfig.TESTS_WORKING_DIR / "helpers/src"
        TestConfig.LINKER_SCRIPTS = TestConfig.TESTS_WORKING_DIR / "helpers/ld"

        # Toolchain paths
        TestConfig.GXX = str((TestConfig.TOOL_PATH / "riscv-tt-elf-g++").absolute())
        TestConfig.OBJDUMP = str(
            (TestConfig.TOOL_PATH / "riscv-tt-elf-objdump").absolute()
        )
        TestConfig.OBJCOPY = str(
            (TestConfig.TOOL_PATH / "riscv-tt-elf-objcopy").absolute()
        )
        TestConfig.ELF_SIZE = str(
            (TestConfig.TOOL_PATH / "riscv-tt-elf-size").absolute()
        )
        TestConfig.GCOV = str((TestConfig.TOOL_PATH / "riscv-tt-elf-gcov").absolute())
        TestConfig.GCOV_TOOL = str(
            (TestConfig.TOOL_PATH / "riscv-tt-elf-gcov-tool").absolute()
        )

        TestConfig.SHARED_DIR = TestConfig.ARTEFACTS_DIR / "shared"
        TestConfig.SHARED_OBJ_DIR = TestConfig.SHARED_DIR / "obj"
        TestConfig.SHARED_ELF_DIR = TestConfig.SHARED_DIR / "elf"
        # Profiler builds need separate shared artefacts (trisc.cpp compiles differently with -DLLK_PROFILER)
        TestConfig.PROFILER_SHARED_DIR = TestConfig.ARTEFACTS_DIR / "shared-profiler"
        TestConfig.PROFILER_SHARED_OBJ_DIR = TestConfig.PROFILER_SHARED_DIR / "obj"
        TestConfig.PROFILER_SHARED_ELF_DIR = TestConfig.PROFILER_SHARED_DIR / "elf"
        TestConfig.COVERAGE_INFO_DIR = TestConfig.ARTEFACTS_DIR / "coverage_info"
        TestConfig.PROFILER_META = TestConfig.ARTEFACTS_DIR / "profiler_meta"
        TestConfig.SYNC_DIR = TestConfig.ARTEFACTS_DIR / "sync_primitives"
        TestConfig.PERF_DATA_DIR = TestConfig.ARTEFACTS_DIR / "temp_perf_data"
        TestConfig.DEFAULT_STIMULI_CACHE_FOLDER = (
            TestConfig.ARTEFACTS_DIR / "temp_stimuli"
        )

    @staticmethod
    def perf_run_tag() -> str:
        """Directory name for this run's reports. Unique per invocation.

        Purely a filesystem concern: it never reaches the published table. The
        Parquet's ``run_id`` cannot serve here because every shard of one CI
        workflow shares it by design (it is a ROW_KEY column, and the data team's
        notion of "one run" spans all shards) — naming directories after it would
        make two shards collide the moment their artefacts are unzipped together.

        Seeded into the environment on first use so xdist workers and the
        controller agree; the pytest plugin sets it before workers spawn.

        CI sets ``PERF_RUN_TAG`` itself, because only the workflow can see the
        shard index: ``GITHUB_RUN_ID`` and ``CHIP_ARCH`` are shared by every shard
        of one architecture, so a tag built from them here would collide. The
        fallback below therefore only has to keep successive invocations apart,
        which a UTC timestamp does on its own.
        """
        tag = os.environ.get("PERF_RUN_TAG", "").strip()
        if not tag:
            run = os.environ.get("GITHUB_RUN_ID", "").strip()
            stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            tag = f"{run}-{stamp}" if run else f"local-{stamp}"
            os.environ["PERF_RUN_TAG"] = tag
        return tag

    @staticmethod
    def perf_run_dir() -> Path:
        """This run's report directory, ``perf_data/runs/<tag>``.

        One directory per invocation is what makes a report trustworthy: a shared
        mutable directory lets a narrower second run leave the first run's test
        directories in place, so the tree reads as complete while holding a blend
        of two runs. Nothing here is ever written by a second invocation.
        """
        return TestConfig.LLK_ROOT / "perf_data" / "runs" / TestConfig.perf_run_tag()

    @staticmethod
    def create_build_directories():
        """Create build directories. Uses class flag to skip redundant filesystem checks."""
        if TestConfig._BUILD_DIRS_CREATED:
            return

        create_directories(
            [
                TestConfig.ARTEFACTS_DIR,  # Parent first
                TestConfig.SYNC_DIR,
                TestConfig.SHARED_DIR,
                TestConfig.SHARED_OBJ_DIR,
                TestConfig.SHARED_ELF_DIR,
                TestConfig.PROFILER_SHARED_DIR,
                TestConfig.PROFILER_SHARED_OBJ_DIR,
                TestConfig.PROFILER_SHARED_ELF_DIR,
                TestConfig.COVERAGE_INFO_DIR,
            ]
        )
        TestConfig._BUILD_DIRS_CREATED = True

    @staticmethod
    def setup_compilation_options(
        with_coverage: bool = False,
        detailed_artefacts: bool = False,
        no_debug_symbols: bool = False,
        speed_of_light: bool = False,
    ):
        debug_flag = "" if no_debug_symbols else "-g "
        TestConfig.OPTIONS_ALL = (
            f"{debug_flag}-O3 "
            "-std=c++17 -ftt-nttp -ftt-constinit -ftt-consteval -ftt-no-dyninit "
            "-ffast-math "
            "-fno-finite-math-only -fsigned-zeros -fno-associative-math "
            "-fno-exceptions -fno-rtti -fno-use-cxa-atexit "
        )
        TestConfig.WITH_COVERAGE = with_coverage
        StimuliConfig.WITH_COVERAGE = with_coverage
        TestConfig.SPEED_OF_LIGHT = speed_of_light

        hw_specific_includes = []
        if TestConfig.ARCH == ChipArchitecture.WORMHOLE:
            hw_specific_includes = [
                "-I../../hw/inc/internal/tt-1xx/wormhole",
                "-I../../hw/inc/internal/tt-1xx/wormhole/wormhole_b0_defines",
                "-I../../hw/ckernels/wormhole_b0/metal/llk_api",
                "-I../../hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu",
            ]
        if TestConfig.ARCH == ChipArchitecture.BLACKHOLE:
            hw_specific_includes = [
                "-I../../hw/inc/internal/tt-1xx/blackhole",
                "-I../../hw/ckernels/blackhole/metal/llk_api",
                # Some SFPU kernels include their neighbours unqualified
                # ("ckernel_sfpu_exp.h") rather than as "llk_sfpu/<name>.h", which only
                # resolves with this on the path. Listed last so the tt-llk copy still
                # wins the basenames that exist in both trees.
                "-I../../hw/ckernels/blackhole/metal/llk_api/llk_sfpu",
                # Keep this list to include roots every Blackhole test needs: INCLUDES is a session-wide
                # ClassVar, so anything added here lands in the compile command for every Blackhole test. A
                # root only some tests need belongs in a per-test fixture instead.
            ]
        if TestConfig.ARCH == ChipArchitecture.QUASAR:
            hw_specific_includes = [
                "-I../../hw/inc/internal/tt-2xx/quasar",
                "-I../../hw/ckernels/quasar/metal/llk_api",
                "-I../../hw/ckernels/quasar/metal/llk_api/llk_sfpu",
            ]

        if detailed_artefacts:
            TestConfig.OPTIONS_ALL += (
                "-save-temps=obj -fdump-tree-all -fdump-rtl-all -v "
            )

        TestConfig.OPTIONS_LINK = (
            "-nostdlib -nostartfiles "
            "-Wl,-z,max-page-size=16 -Wl,-z,common-page-size=16 -Wl,--trace "
        )
        # LLK_ASSERT uses ebreak under ENV_LLK_INFRA (see common/llk_assert.h). Match Hal tensix cflags
        # (wh_hal.cpp / bh_hal.cpp): -mno-tt-fix-whbhebreak avoids 8 NOPs after ebreak.
        no_wh_ebreak_fixup = (
            "-mno-tt-fix-whbhebreak "
            if TestConfig.CHIP_ARCH
            in (ChipArchitecture.WORMHOLE, ChipArchitecture.BLACKHOLE)
            else ""
        )
        # Allow disabling LLK_ASSERT via env var for shape-coverage discovery runs:
        # with asserts off and DEVICE_PRINT_ENABLED on, LLK_VALIDATE_TENSOR_SHAPE_*
        # emits newly-seen TensorShapes via DPRINT instead of ebreaking the kernel,
        # so a single run can enumerate every (fn_name, shape) pair exercised.
        llk_assert_define = (
            ""
            if os.environ.get("TT_LLK_DISABLE_ASSERTS") == "1"
            else "-DENABLE_LLK_ASSERT "
        )
        TestConfig.INITIAL_OPTIONS_COMPILE = (
            "-Wall -Werror -Wno-error=deprecated-declarations "
            "-Wunused-parameter "
            "-Wfloat-equal -Wpointer-arith -Wnull-dereference -Wredundant-decls "
            "-Wuninitialized -Wmaybe-uninitialized "
            f"{no_wh_ebreak_fixup}"
            f"-DTENSIX_FIRMWARE -DENV_LLK_INFRA -DKERNEL_BUILD {llk_assert_define}{TestConfig.ARCH_DEFINE} "
            f"{'-DSPEED_OF_LIGHT' if TestConfig.SPEED_OF_LIGHT else ''}"
        )
        TestConfig.INCLUDES = (
            [
                "-Isfpi/include",
                # Relative to tests/ (compile cwd), not pytest's cwd.
                *[
                    f"-I{p}"
                    for p in TestConfig.llk_tree_include_roots(
                        Path("..") / TestConfig.ARCH_LLK_ROOT
                    )
                ],
                "-I../common",
                "-I../../hw/inc",
                "-Ifirmware/riscv/common",
                "-Ihelpers/include",
                "-I../../hostdevcommon/api",
            ]
            + hw_specific_includes
            + [
                # TODO: remove this after kernels get moved into Metal experimental (#52837)
                "-I../../../ttnn/cpp/ttnn/operations/experimental",
            ]
        )
        TestConfig._apply_extra_includes()

    @staticmethod
    def _as_include_flags(roots) -> List[str]:
        """Turn roots into unquoted ``-I`` flags (one list entry per dir)."""
        flags: List[str] = []
        for root in roots:
            text = str(root)
            flags.append(
                text
                if text.startswith("-I")
                else f"-I{Path(text).expanduser().resolve()}"
            )
        return flags

    @staticmethod
    def _argv(*parts) -> list:
        """Flatten compile-flag strings and raw path tokens into one argv.

        Strings are ``shlex.split`` (so already-quoted ``-I`` flags stay one
        token). Lists/tuples are appended as-is — use those for paths that
        may contain whitespace.
        """
        argv: list = []
        for part in parts:
            if part is None or part == "":
                continue
            if isinstance(part, (list, tuple)):
                argv.extend(
                    str(item) for item in part if item is not None and item != ""
                )
            else:
                argv.extend(shlex.split(str(part)))
        return argv

    @staticmethod
    def _safe_artefact_key(source_path: str) -> str:
        """Artefact dir under ``ARTEFACTS_DIR`` for an absolute driver path.

        The basename is used when it is a single safe token; otherwise it is
        sanitized and disambiguated so ``VARIANT_DIR`` never carries spaces
        or shell metacharacters.
        """
        name = Path(source_path).name
        if re.fullmatch(r"[A-Za-z0-9._+-]+", name):
            return f"sources/{name}"
        digest = sha256(os.path.abspath(source_path).encode()).hexdigest()[
            : TestConfig.ARTEFACT_KEY_HASH_CHARS
        ]
        safe = re.sub(r"[^A-Za-z0-9._+-]+", "_", name) or "driver.cpp"
        return f"sources/{safe}.{digest}"

    @staticmethod
    def _reject_unsafe_cpp_include_path(path: str) -> None:
        """``#include "..."`` is a q-char-sequence — no C-string escapes."""
        if any(char in path for char in '"\\\n'):
            raise ValueError(
                'C++ driver path cannot contain ", backslash, or newline '
                f'(not valid in #include "..."): {path!r}'
            )

    @staticmethod
    def _resolve_flag_roots(roots) -> list:
        resolved = []
        for root in roots:
            if str(root).startswith("-I"):
                resolved.append(str(root))
            else:
                resolved.append(Path(root).expanduser().resolve())
        return resolved

    def _extra_src_include_flag_lists(self) -> tuple[list, list]:
        """Instance + process extras split for before/after in-tree ``helpers/src``."""
        instance = TestConfig._as_include_flags(self.src_include_dirs)
        prepend = instance + [
            flag
            for flag in TestConfig.EXTRA_SRC_INCLUDE_PREPEND
            if flag not in instance
        ]
        seen = set(prepend)
        append = [
            flag for flag in TestConfig.EXTRA_SRC_INCLUDE_APPEND if flag not in seen
        ]
        return prepend, append

    @staticmethod
    def _merge_flag_list(existing: list, flags: list, prepend: bool) -> list:
        rest = [flag for flag in existing if flag not in flags]
        return flags + rest if prepend else rest + flags

    @staticmethod
    def _register_search_dirs(
        prepend_list: list, append_list: list, flags: list, prepend: bool
    ) -> tuple[list, list]:
        """Move ``flags`` onto the prepend or append side; never both."""
        if prepend:
            append_list = [flag for flag in append_list if flag not in flags]
            prepend_list = TestConfig._merge_flag_list(
                prepend_list, flags, prepend=True
            )
        else:
            prepend_list = [flag for flag in prepend_list if flag not in flags]
            append_list = TestConfig._merge_flag_list(append_list, flags, prepend=False)
        return prepend_list, append_list

    @staticmethod
    def _apply_extra_includes() -> None:
        prepend = list(TestConfig.EXTRA_INCLUDE_PREPEND)
        append = list(TestConfig.EXTRA_INCLUDE_APPEND)
        extras = set(prepend + append)
        rest = [flag for flag in TestConfig.INCLUDES if flag not in extras]
        TestConfig.INCLUDES = prepend + rest + append

    @staticmethod
    def llk_tree_include_roots(arch_root) -> List[Path]:
        """``-I`` dirs for one ``tt_llk_<arch>`` tree. ``-I`` is not recursive.

        Headers are spelled ``"ckernel.h"``, ``"experimental/foo.h"``,
        ``"sfpu/..."`` — the same three roots ``setup_compilation_options``
        already adds for the in-tree copy.
        """
        root = Path(arch_root)
        return [
            root / "llk_lib",
            root / "common" / "inc",
            root / "common" / "inc" / "sfpu",
        ]

    @staticmethod
    def add_include_dirs(*dirs, prepend: bool = True) -> None:
        """Add header search dirs (``-I``) for this process.

        Use for ``#include "foo.h"`` / ``"experimental/foo.h"``. Safe before or
        after ``setup_build``. ``prepend=True`` (default) places dirs before
        in-tree ``INCLUDES`` so they can shadow ``experimental/`` copies.
        ``prepend=False`` places them after in-tree dirs (no shadowing).
        Call from the external ``conftest`` after ``helpers`` is on ``sys.path``.

        For one variant only, pass ``include_dirs=[...]`` to the constructor.
        """
        flags = TestConfig._as_include_flags(dirs)
        (
            TestConfig.EXTRA_INCLUDE_PREPEND,
            TestConfig.EXTRA_INCLUDE_APPEND,
        ) = TestConfig._register_search_dirs(
            TestConfig.EXTRA_INCLUDE_PREPEND,
            TestConfig.EXTRA_INCLUDE_APPEND,
            flags,
            prepend,
        )
        if TestConfig.INCLUDES:
            TestConfig._apply_extra_includes()

    @staticmethod
    def add_src_include_dirs(*dirs, prepend: bool = True) -> None:
        """Add search dirs for ``#include <foo.cpp>`` on the kernel compile.

        This is the ``tests/helpers/src`` role — not where the test driver
        lives (that is ``test_name`` / an absolute path). ``prepend=True``
        (default) places dirs ahead of in-tree ``helpers/src`` so an
        out-of-tree ``trisc.cpp`` can shadow it. ``prepend=False`` places
        them after. Safe before or after ``setup_build``.

        For one variant only, pass ``src_include_dirs=[...]`` to the constructor.
        """
        flags = TestConfig._as_include_flags(dirs)
        (
            TestConfig.EXTRA_SRC_INCLUDE_PREPEND,
            TestConfig.EXTRA_SRC_INCLUDE_APPEND,
        ) = TestConfig._register_search_dirs(
            TestConfig.EXTRA_SRC_INCLUDE_PREPEND,
            TestConfig.EXTRA_SRC_INCLUDE_APPEND,
            flags,
            prepend,
        )

    @staticmethod
    def add_helpers_tree(*trees, prepend: bool = True) -> None:
        """Add a ``tests/helpers``-layout tree: ``<tree>/include`` + ``<tree>/src``.

        Shorthand for ``add_include_dirs(<tree>/include)`` plus
        ``add_src_include_dirs(<tree>/src)``. For one variant only, pass
        ``helpers_trees=[...]`` to the constructor.
        """
        includes = []
        sources = []
        for tree in trees:
            tree = Path(tree)
            includes.append(tree / "include")
            sources.append(tree / "src")
        TestConfig.add_include_dirs(*includes, prepend=prepend)
        TestConfig.add_src_include_dirs(*sources, prepend=prepend)

    @staticmethod
    def setup_build(
        sources_path: Path,
        with_coverage: bool = False,
        detailed_artefacts: bool = False,
        no_debug_symbols: bool = False,
        speed_of_light: bool = False,
    ):
        TestConfig.setup_arch()
        TestConfig.setup_paths(sources_path)
        TestConfig.setup_compilation_options(
            with_coverage, detailed_artefacts, no_debug_symbols, speed_of_light
        )
        device_module.Mailboxes = (
            (MailboxesCoverageQuasar if with_coverage else MailboxesQuasar)
            if TestConfig.CHIP_ARCH == ChipArchitecture.QUASAR
            else (MailboxesCoverage if with_coverage else Mailboxes)
        )

    @staticmethod
    def setup_mode(
        worker_id: str,
        compile_consumer: bool,
        compile_producer: bool,
        stimuli_only: str = None,
        use_stimuli: str = None,
        collect_only: bool = False,
    ):
        TestConfig.WORKER_ID = worker_id

        TestConfig._PENDING_WORKER_INDEX = None
        if worker_id == "master":
            TestConfig.TENSIX_LOCATION = "0,0"
        elif compile_producer:
            # Builds ELFs on the CPU and never reads the device, so it is not worth
            # opening a context per worker to answer a question it does not ask.
            row, col = divmod(int(worker_id[2:]), 8)
            TestConfig.TENSIX_LOCATION = f"{row},{col}"
        else:
            # Silicon and RTL do not have an Exalens context until later in
            # pytest_configure / pytest_runtest_setup. Asking now would miss,
            # and a cached miss used to pin the session to the 8-wide fallback.
            TestConfig._PENDING_WORKER_INDEX = int(worker_id[2:])

        if compile_consumer and compile_producer:
            raise RuntimeError(
                "Pytest can be configured to be either compilation producer, compilation consumer, or both by not setting any arguments. Both arguments at the same time are invalid."
            )

        if stimuli_only and use_stimuli:
            raise RuntimeError(
                "Pytest can be configured to compute stimuli only (and store it to a file), consume pre-computed stimuli (from a file), or to lazily calculate stimuli during execution (without any files in between). Both arguments at the same time are invalid."
            )

        if compile_producer:
            TestConfig.BUILD_MODE = BuildMode.PRODUCE
            golden_generators_module.get_golden_generator = dummy_golden_generator

        if compile_consumer:
            TestConfig.BUILD_MODE = BuildMode.CONSUME

        if stimuli_only:
            TestConfig.STIMULI_MODE = StimuliMode.GENERATE_ONLY
            GeneratorProxy.MODE = ProxyMode.CACHE_GOLDEN
            StimuliConfig.initialize_cache(
                (
                    stimuli_only
                    if stimuli_only != "_USE_DEFAULT_PATH"
                    else TestConfig.DEFAULT_STIMULI_CACHE_FOLDER
                )
            )
            golden_generators_module.get_golden_generator = get_golden_proxied

        if use_stimuli:
            TestConfig.STIMULI_MODE = StimuliMode.LOAD_CACHED
            GeneratorProxy.MODE = ProxyMode.LOAD_GOLDEN
            StimuliConfig.initialize_cache(
                (
                    use_stimuli
                    if use_stimuli != "_USE_DEFAULT_PATH"
                    else TestConfig.DEFAULT_STIMULI_CACHE_FOLDER
                )
            )
            golden_generators_module.get_golden_generator = get_golden_proxied

        # Start compilation from a clean artifact directory. With xdist, only
        # the controller can safely remove shared artifacts because workers may
        # already be writing to them. Skip cleanup during test collection so a
        # subsequent consumer run can reuse the existing build.
        if (
            TestConfig.BUILD_MODE in [BuildMode.PRODUCE, BuildMode.DEFAULT]
            and worker_id == "master"
            and not collect_only
        ):
            shutil.rmtree(TestConfig.ARTEFACTS_DIR.absolute(), ignore_errors=True)

    @staticmethod
    def resolve_worker_tensix_location():
        """Bind TENSIX_LOCATION from the card once Exalens has a context."""
        index = TestConfig._PENDING_WORKER_INDEX
        if index is None:
            return
        TestConfig.TENSIX_LOCATION = device_module.tensix_location_for_worker(index)
        TestConfig._PENDING_WORKER_INDEX = None

    # === Instance fields and methods ===
    def __init__(
        self,
        test_name: str,
        formats: InputOutputFormat = None,
        templates: list[TemplateParameter] = None,
        runtimes: list[RuntimeParameter] = None,
        variant_stimuli: StimuliConfig = None,
        boot_mode: BootMode = BootMode.DEFAULT,
        profiler_build: ProfilerBuild = ProfilerBuild.No,
        L1_to_L1_iterations: int = 1,
        unpack_to_dest: bool = False,
        unpack_to_srcs: bool = False,
        disable_format_inference: bool = False,
        dest_acc: DestAccumulation = DestAccumulation.No,
        l1_acc: L1Accumulation = L1Accumulation.No,
        skip_build_header: bool = False,
        compile_time_formats: bool = False,
        requires_device_print: bool = False,
        expected_nondeterministic: bool = False,
        include_dirs: list = None,
        src_include_dirs: list = None,
        helpers_trees: list = None,
    ):
        self.coverage_build = (
            CoverageBuild.Yes if TestConfig.WITH_COVERAGE else CoverageBuild.No
        )

        if test_name is None:
            raise RuntimeError(
                "test_name argument needs to be passed in order to resolve which C++ file is compiled"
            )

        self._prepared = False

        # This instance owns its parameter lists: copy on the way in, and never
        # mutate a caller's list or a default in place. The speed-of-light
        # branch below rebinds rather than appending, for the same reason.
        # (These used to default to a shared ``[]`` that the fold mutated, so a
        # variant constructed without explicit ``templates`` inherited the
        # previous variant's runtimes -- invisible until it changed what got
        # compiled. Regression coverage: test_test_config.py.)
        templates = list(templates or [])
        runtimes = list(runtimes or [])

        if TestConfig.SPEED_OF_LIGHT:
            templates = templates + runtimes
            runtimes = []
            compile_time_formats = True

        # Artefact directory is always relative to ARTEFACTS_DIR. An absolute
        # test_name is the C++ driver path; keep a sources/<basename> artefact
        # key so we don't mkdir through a .cpp file.
        if os.path.isabs(test_name):
            TestConfig._reject_unsafe_cpp_include_path(test_name)
            self.test_source_path = test_name
            self.test_name = TestConfig._safe_artefact_key(test_name)
        else:
            self.test_source_path = test_name
            self.test_name = test_name
        self.templates = templates
        self.runtimes = runtimes
        self.variant_stimuli = variant_stimuli
        self.boot_mode = boot_mode
        self.profiler_build = profiler_build
        self.L1_to_L1_iterations = L1_to_L1_iterations
        self.unpack_to_dest = unpack_to_dest
        self.unpack_to_srcs = unpack_to_srcs
        self.disable_format_inference = disable_format_inference
        self.l1_acc = l1_acc
        self.skip_build_header = skip_build_header
        self.compile_time_formats = compile_time_formats
        self.dest_acc = dest_acc
        self.requires_device_print = requires_device_print
        self.expected_nondeterministic = expected_nondeterministic
        # Per-variant header ``-I`` dirs land in ``local_options_compile`` (last
        # ``-I`` group), so they win over ``add_include_dirs`` and in-tree
        # headers but not over ``add_src_include_dirs``. Per-variant
        # ``src_include_dirs`` are emitted first and do win over
        # ``add_src_include_dirs``. Class methods are for suite-wide dirs.
        include_list = list(include_dirs or [])
        src_include_list = list(src_include_dirs or [])
        for helpers_tree in helpers_trees or []:
            helpers_tree = Path(helpers_tree)
            include_list.append(helpers_tree / "include")
            src_include_list.append(helpers_tree / "src")
        self.include_dirs = TestConfig._resolve_flag_roots(include_list)
        self.src_include_dirs = TestConfig._resolve_flag_roots(src_include_list)

        TILE_SIZES = {
            DataFormat.Bfp8_b: 68,
            DataFormat.Bfp4_b: 36,
            DataFormat.Float32: 256,
        }

        if formats:
            # Check if this is an outlier format combination that requires dest_acc to be enabled
            # Automatically enable dest_acc for outlier combinations
            if (
                is_format_combination_outlier(
                    formats.input_format,
                    formats.output_format,
                    dest_acc,
                )
                and TestConfig.CHIP_ARCH != ChipArchitecture.QUASAR
            ):
                self.dest_acc = DestAccumulation.Yes

            self.formats_config = data_formats(
                input_format=formats.input_format,
                input_format_B=formats.input_format_B,
                output_format=formats.output_format,
                is_fp32_dest_acc_en=dest_acc,
                num_iterations=self.L1_to_L1_iterations,
                unpacking_to_dest=self.unpack_to_dest,
                chip_arch=TestConfig.CHIP_ARCH,
                disable_format_inference=self.disable_format_inference,
                unpacking_to_srcs=self.unpack_to_srcs,
                # `formats` may be an InputOutputFormat (carries the hint) or a
                # FormatConfig (doesn't); fall back to None for the latter.
                register_format_hint=getattr(formats, "register_format_hint", None),
            )
            self.pack_size = TILE_SIZES.get(self.formats_config[0].output_format, 128)
            self.unpack_size_a = TILE_SIZES.get(
                self.formats_config[0].input_format, 128
            )
            self.unpack_size_b = TILE_SIZES.get(
                self.formats_config[0].input_format_B, 128
            )
        else:
            self.formats_config = None
            self.pack_size, self.unpack_size_a, self.unpack_size_b = 128, 128, 128

        # SrcS MX slice geometry follows unpack_S_dst width (same as _is_srcs_32bit_mode_), not dest_acc.
        if self.variant_stimuli:
            self.variant_stimuli.set_use_srcs(self.unpack_to_srcs)
            srcs_32bit_mode = (
                self.unpack_to_srcs
                and self.formats_config is not None
                and self.formats_config[0].unpack_S_dst.is_32_bit()
            )
            self.variant_stimuli.set_srcs_32bit_mode(srcs_32bit_mode)

        if (len(self.runtimes) > 0 or len(self.templates) > 0) and self.variant_stimuli:
            itd_param = next(
                (
                    param
                    for param in self.runtimes + self.templates
                    if isinstance(param, IN_TILE_DIMS)
                ),
                None,
            )
            faces_param = next(
                (
                    param
                    for param in self.runtimes + self.templates
                    if isinstance(param, NUM_FACES)
                ),
                None,
            )
            if itd_param and faces_param:
                temp_num_faces_A = (
                    faces_param.num_faces_A
                    if faces_param.num_faces_A
                    else faces_param.num_faces
                )
                if itd_param.in0_r_dim <= 16:
                    self.pack_size = (self.pack_size // faces_param.num_faces) * (
                        itd_param.in0_r_dim // self.variant_stimuli.face_r_dim
                    )
                    self.unpack_size_a = (self.unpack_size_a // temp_num_faces_A) * (
                        itd_param.in0_r_dim // self.variant_stimuli.face_r_dim
                    )

        # We need to call this here because this function generates serialisation format need for writing RTs to L1,
        # Which is needed by execution part of test infra
        if not TestConfig.SPEED_OF_LIGHT:
            self.generate_runtime_args_struct()

        if (
            self.coverage_build == CoverageBuild.Yes
            and self.profiler_build == ProfilerBuild.Yes
        ):
            raise RuntimeError(
                "You can't build profiler and coverage build at the same time, profiling tests will fail."
            )

    def generate_runtime_args_struct(self):
        # Generate runtime parameter struct
        lines = [
            "// Struct containing runtime parameter layout",
            "struct RuntimeParams {",
            "std::uint32_t TILE_SIZE_PACK;",
            "std::uint32_t TILE_SIZE_UNPACK_A;",
            "std::uint32_t TILE_SIZE_UNPACK_B;",
        ]

        self.runtime_format = "@III"  # tile size types for formatter

        if not self.compile_time_formats:
            # Append struct.pack format for each FormatConfig to L1. Each "I" encodes one
            # uint32_t DataFormat enum. Thirteen I's = thirteen fields appended in
            # write_runtimes_to_L1 (same order as argument_data). struct.pack encodes
            # those values using runtime_format into bytes for RuntimeParams on device.
            if self.L1_to_L1_iterations == 1:
                lines.append("FormatConfig formats;")
                self.runtime_format += "IIIIIIIIIIIII"
            else:
                lines.append(f"FormatConfig formats[{self.L1_to_L1_iterations}];")
                self.runtime_format += self.L1_to_L1_iterations * "IIIIIIIIIIIII"

        if self.variant_stimuli:
            stimuli_fields, stimuli_pack_format = (
                self.variant_stimuli.generate_runtime_struct_fields()
            )
            lines.extend(stimuli_fields)
            self.runtime_format += stimuli_pack_format

        for parameter in self.runtimes:
            field_str, param_field_types = parameter.convert_to_struct_fields()
            lines.append(field_str)
            self.runtime_format += param_field_types

        lines.append("};")

        self.runtime_arguments_struct = lines

    def write_runtimes_to_L1(self):
        if TestConfig.SPEED_OF_LIGHT:
            return

        argument_data = [
            self.pack_size,  # uint32_t TILE_SIZE_PACK;
            self.unpack_size_a,  # uint32_t TILE_SIZE_UNPACK_A;
            self.unpack_size_b,  # uint32_t TILE_SIZE_UNPACK_B;
        ]

        if not self.compile_time_formats:
            for format_tuple in self.formats_config:
                argument_data.extend(
                    [
                        TestConfig.DATA_FORMAT_ENUM[format_tuple.unpack_A_src],
                        TestConfig.DATA_FORMAT_ENUM[format_tuple.unpack_B_src],
                        TestConfig.DATA_FORMAT_ENUM[format_tuple.unpack_S_src],
                        TestConfig.DATA_FORMAT_ENUM[format_tuple.unpack_A_dst],
                        TestConfig.DATA_FORMAT_ENUM[format_tuple.unpack_B_dst],
                        TestConfig.DATA_FORMAT_ENUM[format_tuple.unpack_S_dst],
                        TestConfig.DATA_FORMAT_ENUM[format_tuple.math],
                        TestConfig.DATA_FORMAT_ENUM[format_tuple.sfpu_src],
                        TestConfig.DATA_FORMAT_ENUM[format_tuple.sfpu_dst],
                        TestConfig.DATA_FORMAT_ENUM[format_tuple.pack_src],
                        TestConfig.DATA_FORMAT_ENUM[format_tuple.pack_dst],
                        TestConfig.DATA_FORMAT_ENUM[format_tuple.pack_S_src],
                        TestConfig.DATA_FORMAT_ENUM[format_tuple.pack_S_dst],
                    ]
                )

        if self.variant_stimuli:
            argument_data.extend(
                self.variant_stimuli.generate_runtime_operands_values()
            )

        for param in self.runtimes:
            argument_data.extend(
                [
                    (
                        getattr(param, f.name).value
                        if issubclass(f.type, Enum)
                        else getattr(param, f.name)
                    )
                    for f in fields(param)
                ]
            )

        serialised_data = struct.pack(self.runtime_format, *argument_data)

        if len(serialised_data) != 0:
            if TestConfig.WITH_COVERAGE:
                write_to_device(
                    TestConfig.TENSIX_LOCATION,
                    TestConfig.RUNTIME_ADDRESS_COVERAGE,
                    serialised_data,
                )
            else:
                write_to_device(
                    TestConfig.TENSIX_LOCATION,
                    TestConfig.RUNTIME_ADDRESS_NON_COVERAGE,
                    serialised_data,
                )

    def collect_hash(self):
        lock_file = TEMP_DIR / "tt-llk-build-print.lock"
        lock_file.touch(exist_ok=True)

        with open(lock_file, "w") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                logger.debug("Variant hash: {}", self.variant_id)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

        pytest.skip()

    def _kernel_source_include(self) -> str:
        """C++ snippet that pulls in this variant's driver.

        Relative names keep the historical ``#include <sources/foo.cpp>`` form
        (resolved from ``tests/``). Absolute paths are quote-included so an
        out-of-tree driver does not have to live under ``tests/sources/``.

        ``test_source_path`` is set in ``__init__``. Some in-tree callers
        (the fuser) assign ``test_name`` later and leave ``test_source_path``
        empty — fall back to ``test_name`` so ``#include <>`` is never emitted.
        """
        source = str(self.test_source_path or self.test_name)
        if os.path.isabs(source):
            TestConfig._reject_unsafe_cpp_include_path(source)
            return f'#include "{source}"\n'
        return f"#include  <{source}>\n"

    def generate_variant_hash(self):
        NON_COMPILATION_ARGUMENTS = [
            "run_configs",
            "variant_id",
            "runtime_arguments_struct",
            "runtime_format",
            "passed_templates",
            "passed_runtimes",
            "current_run_type",
            "temp_elfs",
            # Host-side determinism-check opt-out; does not affect the compiled kernel.
            "expected_nondeterministic",
        ]

        if not TestConfig.SPEED_OF_LIGHT:
            NON_COMPILATION_ARGUMENTS += [
                "variant_stimuli",
                "pack_size",
                "unpack_size_a",
                "unpack_size_b",
                "runtimes",
                "formats_config" if not self.compile_time_formats else "",
            ]

        temp_str = [
            str(value)
            for field_name, value in self.__dict__.items()
            if field_name not in NON_COMPILATION_ARGUMENTS
        ]

        # Header and source search dirs decide which copy of a given relative
        # include a driver compiles against, so they are compilation inputs.
        # Most of them live in class state (``add_include_dirs`` /
        # ``add_src_include_dirs`` / ``add_helpers_tree``, and the in-tree
        # ``INCLUDES`` built by ``setup_compilation_options``), which
        # ``self.__dict__`` cannot see — so hashing instance fields alone gave
        # two variants the same id while they compiled against different
        # headers. That matters because ``prepare`` does not rebuild in CONSUME
        # mode: it trusts this id to locate the ELF the producer pass built.
        # Order is part of the value: these lists are precedence-ordered, and
        # reordering them changes which header wins.
        # ``INCLUDES`` is the merged view, and only exists once ``setup_build``
        # has run; before that the registered extras live solely in
        # ``EXTRA_INCLUDE_*``. Hash both, so a variant built before the merge
        # and one built after cannot collide.
        #
        # Each group is fenced by a label rather than concatenated, because the
        # *role* of a dir is as much a compilation input as the dir itself. A
        # flat list loses that: the same dir registered with ``prepend=True``
        # and ``prepend=False`` yields the same token sequence while deciding
        # opposite precedence against the in-tree ``helpers/src``. Labels cannot
        # be confused with flags, which all start with ``-I``.
        header_tokens = self._header_include_tokens()
        header_prepend = [
            flag
            for flag in TestConfig.EXTRA_INCLUDE_PREPEND
            if flag not in header_tokens
        ]
        header_append = [
            flag
            for flag in TestConfig.EXTRA_INCLUDE_APPEND
            if flag not in header_tokens
        ]
        src_include_prepend, src_include_append = self._extra_src_include_flag_lists()
        search_dirs = [
            "<<headers>>",
            *header_tokens,
            "<<headers-prepend>>",
            *header_prepend,
            "<<headers-append>>",
            *header_append,
            "<<src-prepend>>",
            *src_include_prepend,
            "<<src-append>>",
            *src_include_append,
        ]

        self.variant_id = sha256(
            str(" | ".join(temp_str + ["<<search-dirs>>"] + search_dirs)).encode()
        ).hexdigest()

    def resolve_shared_compile_options(self) -> tuple[str, str, str]:
        """Flags for brisc/coverage. Process-wide ``INCLUDES`` only.

        Shared artefacts are keyed by ``SHARED_DIR`` / ``.shared_complete``,
        not ``variant_id``. Per-variant ``include_dirs`` must not leak in.
        """
        return self._compose_compile_options(list(TestConfig.INCLUDES))

    def _header_include_tokens(self) -> List[str]:
        """Header ``-I`` tokens this variant compiles with, in search order.

        Per-variant dirs first, then the process-wide ``INCLUDES``.

        ``generate_variant_hash`` reads the same helper on purpose. The variant
        id has to describe what actually reaches the compiler, so if these two
        ever disagree the cache key stops matching the binary — which is silent,
        because ``prepare`` does not rebuild in CONSUME mode. Keep them sharing
        one definition rather than two copies of the expression.
        """
        return TestConfig._as_include_flags(self.include_dirs) + list(
            TestConfig.INCLUDES
        )

    def resolve_compile_options(self) -> tuple[str, str, str]:
        return self._compose_compile_options(self._header_include_tokens())

    def _compose_compile_options(self, include_tokens: list) -> tuple[str, str, str]:
        if (
            TestConfig.OPTIONS_COMPILE is not None
            and TestConfig.MEMORY_LAYOUT_LD_SCRIPT is not None
            and TestConfig.NON_COVERAGE_OPTIONS_COMPILE is not None
        ):
            return (
                TestConfig.OPTIONS_COMPILE,
                MEMORY_LAYOUT_LD_SCRIPT,
                NON_COVERAGE_OPTIONS_COMPILE,
            )

        MEMORY_LAYOUT_LD_SCRIPT = (
            f"{TestConfig.LINKER_SCRIPTS}/memory.{TestConfig.ARCH.value}.ld"
        )
        include_flags = " ".join(shlex.quote(flag) for flag in include_tokens)
        OPTIONS_COMPILE = f"{include_flags} {TestConfig.INITIAL_OPTIONS_COMPILE} "

        OPTIONS_COMPILE += (
            "-DLLK_BOOT_MODE_TRISC "
            if TestConfig.CHIP_ARCH == ChipArchitecture.QUASAR
            else "-DLLK_BOOT_MODE_BRISC "
        )

        NON_COVERAGE_OPTIONS_COMPILE = OPTIONS_COMPILE

        if self.coverage_build == CoverageBuild.Yes:
            NON_COVERAGE_OPTIONS_COMPILE = OPTIONS_COMPILE
            OPTIONS_COMPILE += (
                "-fprofile-arcs -ftest-coverage -fprofile-info-section -DCOVERAGE "
            )
            MEMORY_LAYOUT_LD_SCRIPT = (
                f"{TestConfig.LINKER_SCRIPTS}/memory.{TestConfig.ARCH.value}.debug.ld"
            )

        if self.profiler_build == ProfilerBuild.Yes:
            OPTIONS_COMPILE += "-DLLK_PROFILER "

        if os.environ.get("TT_METAL_DISABLE_SFPLOADMACRO") == "1":
            OPTIONS_COMPILE += "-DDISABLE_SFPLOADMACRO "

        return (OPTIONS_COMPILE, MEMORY_LAYOUT_LD_SCRIPT, NON_COVERAGE_OPTIONS_COMPILE)

    def build_shared_artefacts(self):
        if TestConfig.SHARED_ARTEFACTS_AVAILABLE:
            return

        shared_obj_dir = TestConfig.SHARED_OBJ_DIR
        shared_elf_dir = TestConfig.SHARED_ELF_DIR
        lock_file = TEMP_DIR / "tt-llk-build-shared.lock"

        done_marker = shared_obj_dir / ".shared_complete"

        # Fast path: if shared artefacts are already built
        if done_marker.exists():
            TestConfig.SHARED_ARTEFACTS_AVAILABLE = True
            return

        # Acquire lock for building shared artefacts
        lock = FileLock(lock_file)

        with lock:
            # Check again inside lock
            if done_marker.exists():
                TestConfig.SHARED_ARTEFACTS_AVAILABLE = True
                return

            _, local_memory_layout_ld, local_non_coverage = (
                self.resolve_shared_compile_options()
            )

            if TestConfig.WITH_COVERAGE:
                compile_command = (  # coverage.o : coverage.cpp
                    f"{TestConfig.GXX} {TestConfig.ARCH_NON_COMPUTE} {TestConfig.OPTIONS_ALL} {local_non_coverage} "
                    f'-fno-strict-aliasing -c -o {shared_obj_dir / "coverage.o"} {TestConfig.RISCV_SOURCES / "coverage.cpp"}'
                )
                logger.trace(compile_command)
                run_shell_command(compile_command, TestConfig.TESTS_WORKING_DIR)

            if TestConfig.CHIP_ARCH != ChipArchitecture.QUASAR:
                # Only compile BRISC with counter support when counters are enabled,
                # otherwise BRISC arms counter hardware which adds monitoring overhead.
                perf_cnt_flag = (
                    "-DPERF_COUNTERS_COMPILED "
                    if TestConfig.ENABLE_PERF_COUNTERS
                    else ""
                )
                compile_command = (  # brisc.elf : brisc.cpp
                    f"{TestConfig.GXX} {TestConfig.ARCH_NON_COMPUTE} {TestConfig.OPTIONS_ALL} {TestConfig.OPTIONS_LINK} {local_non_coverage} "
                    f'{"-DCOVERAGE " if TestConfig.WITH_COVERAGE else ""}'
                    f"{perf_cnt_flag}"
                    f'-T{local_memory_layout_ld} -T{TestConfig.LINKER_SCRIPTS / "brisc.ld"} -T{TestConfig.LINKER_SCRIPTS / "sections.ld"} '
                    f'-o {shared_elf_dir / "brisc.elf"} {TestConfig.RISCV_SOURCES / "brisc.cpp"}'
                )
                logger.trace(compile_command)
                run_shell_command(compile_command, TestConfig.TESTS_WORKING_DIR)

            # Mark shared artefacts as complete
            done_marker.touch()
            TestConfig.SHARED_ARTEFACTS_AVAILABLE = True

    def generate_compile_time_data_formats(self) -> list[str]:
        header_content: list[str] = [
            "// Data formats inferred by Python inference model"
        ]

        # Fused Test L1 to L1 : Input of first run is used as input for the second run ...
        # Not fusing: single L1-to-L1 iteration, so we retrieve one format configuration
        # L1_to_L1_iterations is the number of times we perform llk operations from L1 input tensor to L1 output tensor
        # If L1_to_L1_ITERATIONS is 1, we take input tensor from L1 -> unpack -> math -> pack -> L1
        # If L1_to_L1_ITERATIONS is greater than 1, we perform multiple iterations of unpack -> math -> pack, by taking results tensor in L1 to be input tensor of next iteration
        if self.L1_to_L1_iterations > 1:
            # Generate format data as arrays that params.h can use to construct FormatConfig objects
            header_content.extend(
                [
                    "// Format data for multiple L1-to-L1 iterations",
                    f"constexpr std::uint32_t L1_to_L1_ITERATIONS = {self.L1_to_L1_iterations};",
                    "#define FUSED_MULTIPLE_RUNS true",
                ]
            )

            # Create array of format configurations for multiple L1-to-L1 iterations
            unpack_a_in_values = [
                f"ckernel::to_underlying(DataFormat::{fmt.unpack_A_src.name})"
                for fmt in self.formats_config
            ]
            unpack_b_in_values = [
                f"ckernel::to_underlying(DataFormat::{fmt.unpack_B_src.name})"
                for fmt in self.formats_config
            ]
            unpack_a_out_values = [
                f"ckernel::to_underlying(DataFormat::{fmt.unpack_A_dst.name})"
                for fmt in self.formats_config
            ]
            unpack_b_out_values = [
                f"ckernel::to_underlying(DataFormat::{fmt.unpack_B_dst.name})"
                for fmt in self.formats_config
            ]
            unpack_s_in_values = [
                f"ckernel::to_underlying(DataFormat::{fmt.unpack_S_src.name})"
                for fmt in self.formats_config
            ]
            unpack_s_out_values = [
                f"ckernel::to_underlying(DataFormat::{fmt.unpack_S_dst.name})"
                for fmt in self.formats_config
            ]
            math_values = [
                f"ckernel::to_underlying(DataFormat::{fmt.math.name})"
                for fmt in self.formats_config
            ]
            sfpu_src_values = [
                f"ckernel::to_underlying(DataFormat::{fmt.sfpu_src.name})"
                for fmt in self.formats_config
            ]
            sfpu_dst_values = [
                f"ckernel::to_underlying(DataFormat::{fmt.sfpu_dst.name})"
                for fmt in self.formats_config
            ]
            pack_in_values = [
                f"ckernel::to_underlying(DataFormat::{fmt.pack_src.name})"
                for fmt in self.formats_config
            ]
            pack_out_values = [
                f"ckernel::to_underlying(DataFormat::{fmt.pack_dst.name})"
                for fmt in self.formats_config
            ]
            pack_s_in_values = [
                f"ckernel::to_underlying(DataFormat::{fmt.pack_S_src.name})"
                for fmt in self.formats_config
            ]
            pack_s_out_values = [
                f"ckernel::to_underlying(DataFormat::{fmt.pack_S_dst.name})"
                for fmt in self.formats_config
            ]

            header_content.extend(
                [
                    f"constexpr std::array<std::underlying_type_t<DataFormat>, L1_to_L1_ITERATIONS> UNPACK_A_IN_LIST = {{{', '.join(unpack_a_in_values)}}};",
                    f"constexpr std::array<std::underlying_type_t<DataFormat>, L1_to_L1_ITERATIONS> UNPACK_B_IN_LIST = {{{', '.join(unpack_b_in_values)}}};",
                    f"constexpr std::array<std::underlying_type_t<DataFormat>, L1_to_L1_ITERATIONS> UNPACK_S_IN_LIST = {{{', '.join(unpack_s_in_values)}}};",
                    f"constexpr std::array<std::underlying_type_t<DataFormat>, L1_to_L1_ITERATIONS> UNPACK_A_OUT_LIST = {{{', '.join(unpack_a_out_values)}}};",
                    f"constexpr std::array<std::underlying_type_t<DataFormat>, L1_to_L1_ITERATIONS> UNPACK_B_OUT_LIST = {{{', '.join(unpack_b_out_values)}}};",
                    f"constexpr std::array<std::underlying_type_t<DataFormat>, L1_to_L1_ITERATIONS> UNPACK_S_OUT_LIST = {{{', '.join(unpack_s_out_values)}}};",
                    f"constexpr std::array<std::underlying_type_t<DataFormat>, L1_to_L1_ITERATIONS> MATH_FORMAT_LIST = {{{', '.join(math_values)}}};",
                    f"constexpr std::array<std::underlying_type_t<DataFormat>, L1_to_L1_ITERATIONS> SFPU_IN_LIST = {{{', '.join(sfpu_src_values)}}};",
                    f"constexpr std::array<std::underlying_type_t<DataFormat>, L1_to_L1_ITERATIONS> SFPU_OUT_LIST = {{{', '.join(sfpu_dst_values)}}};",
                    f"constexpr std::array<std::underlying_type_t<DataFormat>, L1_to_L1_ITERATIONS> PACK_IN_LIST = {{{', '.join(pack_in_values)}}};",
                    f"constexpr std::array<std::underlying_type_t<DataFormat>, L1_to_L1_ITERATIONS> PACK_OUT_LIST = {{{', '.join(pack_out_values)}}};",
                    f"constexpr std::array<std::underlying_type_t<DataFormat>, L1_to_L1_ITERATIONS> PACK_S_IN_LIST = {{{', '.join(pack_s_in_values)}}};",
                    f"constexpr std::array<std::underlying_type_t<DataFormat>, L1_to_L1_ITERATIONS> PACK_S_OUT_LIST = {{{', '.join(pack_s_out_values)}}};",
                    "constexpr std::array<FormatConfig, L1_to_L1_ITERATIONS> formats_array = {",
                    "{FormatConfig(UNPACK_A_IN_LIST[0], UNPACK_B_IN_LIST[0], UNPACK_S_IN_LIST[0], UNPACK_A_OUT_LIST[0], UNPACK_B_OUT_LIST[0], UNPACK_S_OUT_LIST[0], MATH_FORMAT_LIST[0], SFPU_IN_LIST[0], SFPU_OUT_LIST[0], PACK_IN_LIST[0], PACK_OUT_LIST[0], PACK_S_IN_LIST[0], PACK_S_OUT_LIST[0]),",
                    "FormatConfig(",
                    "UNPACK_A_IN_LIST[1], UNPACK_B_IN_LIST[1], UNPACK_S_IN_LIST[1], UNPACK_A_OUT_LIST[1], UNPACK_B_OUT_LIST[1], UNPACK_S_OUT_LIST[1], MATH_FORMAT_LIST[1], SFPU_IN_LIST[1], SFPU_OUT_LIST[1], PACK_IN_LIST[1], PACK_OUT_LIST[1], PACK_S_IN_LIST[1], PACK_S_OUT_LIST[1])}};",
                ]
            )

        else:
            # Single iteration - use simple format inference
            # Generate format data as individual constants for single iteration
            formats_config = self.formats_config[0]
            header_content.extend(
                [
                    "// Format data for single L1-to-L1 iteration",
                    f"constexpr auto UNPACK_A_IN = ckernel::to_underlying(DataFormat::{formats_config.unpack_A_src.name});",
                    f"constexpr auto UNPACK_B_IN = ckernel::to_underlying(DataFormat::{formats_config.unpack_B_src.name});",
                    f"constexpr auto UNPACK_S_IN = ckernel::to_underlying(DataFormat::{formats_config.unpack_S_src.name});",
                    f"constexpr auto UNPACK_A_OUT = ckernel::to_underlying(DataFormat::{formats_config.unpack_A_dst.name});",
                    f"constexpr auto UNPACK_B_OUT = ckernel::to_underlying(DataFormat::{formats_config.unpack_B_dst.name});",
                    f"constexpr auto UNPACK_S_OUT = ckernel::to_underlying(DataFormat::{formats_config.unpack_S_dst.name});",
                    f"constexpr auto MATH_FORMAT = ckernel::to_underlying(DataFormat::{formats_config.math.name});",
                    f"constexpr auto SFPU_IN = ckernel::to_underlying(DataFormat::{formats_config.sfpu_src.name});",
                    f"constexpr auto SFPU_OUT = ckernel::to_underlying(DataFormat::{formats_config.sfpu_dst.name});",
                    f"constexpr auto PACK_IN = ckernel::to_underlying(DataFormat::{formats_config.pack_src.name});",
                    f"constexpr auto PACK_OUT = ckernel::to_underlying(DataFormat::{formats_config.pack_dst.name});",
                    f"constexpr auto PACK_S_IN = ckernel::to_underlying(DataFormat::{formats_config.pack_S_src.name});",
                    f"constexpr auto PACK_S_OUT = ckernel::to_underlying(DataFormat::{formats_config.pack_S_dst.name});",
                    "constexpr FormatConfig formats = FormatConfig(UNPACK_A_IN, UNPACK_B_IN, UNPACK_S_IN, UNPACK_A_OUT, UNPACK_B_OUT, UNPACK_S_OUT, MATH_FORMAT, SFPU_IN, SFPU_OUT, PACK_IN, PACK_OUT, PACK_S_IN, PACK_S_OUT);",
                ]
            )

        return header_content

    def generate_build_header(self) -> str:
        if TestConfig.ARCH == ChipArchitecture.QUASAR:
            sfpu_types_include = ""
        else:
            sfpu_types_include = '#include "llk_sfpu_types.h"'

        header_content: list[str] = [
            "// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC",
            "//",
            "// SPDX-License-Identifier: Apache-2.0",
            "// AUTO-GENERATED CONFIGURATION HEADER. DO NOT EDIT MANUALLY!",
            "",
            "#pragma once",
            "",
            "#include <array>",
            "#include <type_traits>",
            "",
            '#include "operand.h"',
            '#include "llk_defs.h"',
            f"{sfpu_types_include}",
            (
                # perf.h provides PerfRunType (needed for the PERF_RUN_TYPE declaration below).
                # Test sources that use MEASURE_PERF_COUNTERS get counters.h via params.h.
                '#include "perf.h"'
                if TestConfig.CHIP_ARCH != ChipArchitecture.QUASAR
                else ""
            ),
            '#include "tensix_types.h"',
            "#define RUNTIME_PARAMETERS  [[maybe_unused]] const struct RuntimeParams&",
            f"constexpr bool l1_acc_en = {self.l1_acc.value};",
            f"constexpr bool unpack_to_dest = {str(self.unpack_to_dest).lower()};",
        ] + (
            FORMATS_CONFIG_STRUCT_COMPILETIME
            if self.compile_time_formats
            else FORMATS_CONFIG_STRUCT_RUNTIME
        )

        if self.formats_config is None:
            header_content.append(
                f"constexpr bool is_fp32_dest_acc_en = {self.dest_acc.cpp_enum_value};"
            )
        else:
            header_content.append(
                f"constexpr bool is_fp32_dest_acc_en = {self.dest_acc.cpp_enum_value};"
            )

        if TestConfig.SPEED_OF_LIGHT:
            header_content.extend(
                [
                    f"constexpr std::uint32_t TILE_SIZE_PACK = {self.pack_size};",
                    f"constexpr std::uint32_t TILE_SIZE_UNPACK_A = {self.unpack_size_a};",
                    f"constexpr std::uint32_t TILE_SIZE_UNPACK_B = {self.unpack_size_b};",
                ]
            )

            if self.variant_stimuli:
                header_content.extend(
                    self.variant_stimuli.generate_stimuli_header_addresses()
                )

        for parameter in self.templates:
            header_content.append(parameter.convert_to_cpp())

        if self.compile_time_formats:
            header_content.extend(self.generate_compile_time_data_formats())

        if TestConfig.SPEED_OF_LIGHT:
            header_content.append("struct RuntimeParams {};")
        else:
            header_content.extend(self.runtime_arguments_struct)

        return "\n".join(header_content)

    @staticmethod
    def get_elf_text_size(elf_path: Path) -> int:
        """Returns the text section size (code+rodata) of an ELF in bytes."""
        result = subprocess.run(
            [TestConfig.ELF_SIZE, str(elf_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"riscv-tt-elf-size failed on {elf_path}:\n{result.stderr}"
            )
        # BSD format: header line + data line
        #    text    data     bss     dec     hex filename
        #    4096      32       0    4128    1020 unpack.elf
        lines = result.stdout.strip().splitlines()
        if len(lines) < 2:
            raise RuntimeError(
                f"Unexpected riscv-tt-elf-size output for {elf_path}:\n{result.stdout}"
            )
        try:
            return int(lines[1].split()[0])
        except (IndexError, ValueError) as e:
            raise RuntimeError(
                f"Failed to parse text size from riscv-tt-elf-size output for {elf_path}:\n{result.stdout}"
            ) from e

    def build_elfs(self):

        VARIANT_DIR = TestConfig.ARTEFACTS_DIR / self.test_name / self.variant_id
        if not self.skip_build_header:
            header_content = self.generate_build_header()
        done_marker = VARIANT_DIR / ".build_complete"

        if TestConfig.INFRA_TESTING:
            return

        self.build_shared_artefacts()

        # Fast path: if build is already complete, skip entirely
        if done_marker.exists():
            logger.debug("Build already complete for {}", self.variant_id[:12])
            return

        # Acquire lock for this variant to prevent concurrent builds
        lock_file = TestConfig.SYNC_DIR / f"{self.variant_id}.lock"
        lock = FileLock(lock_file)

        with lock:
            # Check again inside lock in case another process just finished
            if done_marker.exists():
                return

            VARIANT_OBJ_DIR = VARIANT_DIR / "obj"
            VARIANT_ELF_DIR = VARIANT_DIR / "elf"

            create_directories([VARIANT_OBJ_DIR, VARIANT_ELF_DIR])

            local_options_compile, local_memory_layout_ld, _ = (
                self.resolve_compile_options()
            )

            if not self.skip_build_header:
                with open(VARIANT_DIR / "build.h", "w") as f:
                    f.write(header_content)

            # Use correct shared artefact directory based on profiler build
            shared_obj_dir = (
                TestConfig.PROFILER_SHARED_OBJ_DIR
                if self.profiler_build == ProfilerBuild.Yes
                else TestConfig.SHARED_OBJ_DIR
            )

            src_include_prepend, src_include_append = (
                self._extra_src_include_flag_lists()
            )

            def build_kernel_part(name: str):
                # COMPILE_FOR_TRISC is the single source of truth for the compute thread id on every
                # arch (unpack=0/math=1/pack=2/sfpu=3). Quasar also gets -DLLK_TRISC_<NAME> below, but the
                # LLK headers now require COMPILE_FOR_TRISC (see ckernel_addrmod.h), so pass it for Quasar too.
                optional_kernel_flags = "-DCOMPILE_FOR_TRISC=" + str(
                    TestConfig.KERNEL_COMPONENTS.index(name)
                )

                if not self.compile_time_formats:
                    optional_kernel_flags += " -DRUNTIME_FORMATS"

                # EXPERIMENT: enable -DPERF_COUNTERS_COMPILED on TRISC.
                # Quasar is intentionally excluded: it adds a 4th compute thread
                # (SFPU) and the entry/exit barrier in `counters.h` posts a fixed
                # number of tokens for 3 threads, so enabling perf counters on
                # Quasar would deadlock the SFPU thread (it would spinwait on a
                # semaphore that never gets the extra post). A static_assert in
                # `counters.h` enforces this at compile time as a safety net.
                if (
                    TestConfig.ENABLE_PERF_COUNTERS
                    and TestConfig.CHIP_ARCH != ChipArchitecture.QUASAR
                ):
                    optional_kernel_flags += " -DPERF_COUNTERS_COMPILED"

                coverage_args = (
                    [
                        "-Wl,--start-group",
                        str(shared_obj_dir / "coverage.o"),
                        "-lgcov",
                        "-Wl,--end-group",
                    ]
                    if self.coverage_build == CoverageBuild.Yes
                    else []
                )
                trisc_define = "ISOLATE_SFPU" if name == "sfpu" else name.upper()
                device_print_flags = ""
                if TestConfig.DEVICE_PRINT_ENABLED or self.requires_device_print:
                    risc_id, _ = TestConfig.RISC_INFO[name]
                    # Quasar: kernel addresses the buffer through the uncached alias
                    # (see device_print.h:get_lock_atomic).
                    kernel_buffer_base = TestConfig.DEVICE_PRINT_BUFFER_BASE + (
                        0x400000 if TestConfig.ARCH == ChipArchitecture.QUASAR else 0
                    )
                    device_print_flags = (
                        "-DDEBUG_PRINT_ENABLED "
                        f"-DLLK_DEVICE_PRINT_BUFFER_BASE={kernel_buffer_base:#x} "
                        f"-DLLK_RUNTIME_ARGS_START={TestConfig.DEVICE_PRINT_RUNTIME_ARGS_START:#x} "
                        f"-DDEVICE_PRINT_BUFFER_SIZE={TestConfig.DEVICE_PRINT_BUFFER_SIZE} "
                        f"-DDEVICE_PRINT_BUFFER_SIZE2={TestConfig.DEVICE_PRINT_BUFFER_SIZE2} "
                        f"-DPROCESSOR_INDEX={risc_id} "
                    )
                compile_command = TestConfig._argv(
                    [TestConfig.GXX],
                    TestConfig.ARCH_COMPUTE,
                    TestConfig.ARCH_SPECIFIC_OPTIONS,
                    TestConfig.OPTIONS_ALL,
                    [f"-I{TestConfig.TESTS_WORKING_DIR}"],
                    src_include_prepend,
                    [f"-I{TestConfig.RISCV_SOURCES}"],
                    src_include_append,
                    [f"-I{VARIANT_DIR}"],
                    local_options_compile,
                    optional_kernel_flags,
                    f"-DLLK_TRISC_{trisc_define}",
                    device_print_flags,
                    TestConfig.OPTIONS_LINK,
                    coverage_args,
                    [
                        f"-T{local_memory_layout_ld}",
                        f"-T{TestConfig.LINKER_SCRIPTS / name}.ld",
                        f"-T{TestConfig.LINKER_SCRIPTS / 'sections.ld'}",
                    ],
                    # -lgcc pulls in libgcc soft-float/integer helpers (e.g. __mulsf3)
                    # that -nostdlib drops; only referenced helpers are linked.
                    ["-x", "c++", "-", "-lc", "-lgcc", "-o"],
                    [str(VARIANT_ELF_DIR / f"{name}.elf")],
                )

                logger.trace(" ".join(shlex.quote(part) for part in compile_command))

                run_shell_command(  # %.elf : path/to/kernel/test.cpp trisc.cpp [coverage.o libgcov.a]
                    compile_command,
                    TestConfig.TESTS_WORKING_DIR,
                    (f"{self._kernel_source_include()}#include  <trisc.cpp>\n"),
                )

            with ThreadPoolExecutor(
                max_workers=len(TestConfig.KERNEL_COMPONENTS)
            ) as executor:
                futures = [
                    executor.submit(build_kernel_part, name)
                    for name in TestConfig.KERNEL_COMPONENTS
                ]
                for fut in futures:
                    fut.result()

            if self.profiler_build == ProfilerBuild.Yes:
                # Extract profiler metadata
                PROFILER_VARIANT_META_DIR = Path(
                    TestConfig.PROFILER_META / self.test_name / self.variant_id
                )

                PROFILER_VARIANT_META_DIR.mkdir(exist_ok=True, parents=True)

                for component in TestConfig.KERNEL_COMPONENTS:
                    elf_path = VARIANT_ELF_DIR / f"{component}.elf"
                    meta_bin_path = PROFILER_VARIANT_META_DIR / f"{component}.meta.bin"
                    run_shell_command(
                        [
                            TestConfig.OBJCOPY,
                            "-O",
                            "binary",
                            "-j",
                            ".profiler_meta",
                            str(elf_path),
                            str(meta_bin_path),
                        ],
                        TestConfig.TESTS_WORKING_DIR,
                    )

            # Mark build as complete so other processes know they can use the artefacts
            done_marker.touch()

    def read_coverage_data_from_device(self):
        VARIANT_DIR = TestConfig.ARTEFACTS_DIR / self.test_name / self.variant_id
        # Extracting coverage stream from device, for all kernel parts, for all their compilation units
        coverage_stream = b""
        for trisc_name in TestConfig.KERNEL_COMPONENTS:
            temp_elf = parse_elf(VARIANT_DIR / f"elf/{trisc_name}.elf")
            coverage_start = temp_elf.symbols["__coverage_start"].value
            if not coverage_start:
                raise TTException(
                    f"__coverage_start not found in variant's {trisc_name}.elf"
                )
            length = read_word_from_device(
                TestConfig.TENSIX_LOCATION, addr=coverage_start
            )
            coverage_stream += read_from_device(
                TestConfig.TENSIX_LOCATION, coverage_start + 4, num_bytes=length - 4
            )

        if len(self.runtimes) == 0:
            stream_name = "deafult_stream_name.stream"
        else:
            stream_name = f"{sha256(str(' | '.join([str(run_arg) for run_arg in self.runtimes])).encode()).hexdigest()}.stream"

        logger.trace(stream_name)

        with open(
            VARIANT_DIR / stream_name,
            "wb",
        ) as fd:
            fd.write(coverage_stream)

    BRISC_ELF_LOADED: ClassVar[bool] = False
    LAST_LOADED_ELFS: ClassVar[Path] = Path()
    # Max BRISC bring-up attempts after a reset. A board-wide `tt-smi -r 0`
    # can leave a core slow-to-boot or wedged; each attempt re-issues the
    # soft-reset kick (re-polling alone cannot recover a wedged core).
    BRISC_BOOT_MAX_ATTEMPTS: ClassVar[int] = 3

    def run_elf_files(self) -> list:
        boot_mode = (
            CHIP_DEFAULT_BOOT_MODES[TestConfig.CHIP_ARCH]
            if self.boot_mode == BootMode.DEFAULT
            else self.boot_mode
        )

        # Zero the device print buffer header before each kernel run so the
        # first DEVICE_PRINT() observes wpos=rpos=0 and a free lock.
        if TestConfig.DEVICE_PRINT_ENABLED or self.requires_device_print:
            write_words_to_device(
                TestConfig.TENSIX_LOCATION,
                TestConfig.DEVICE_PRINT_BUFFER_BASE,
                [0] * (aux_size_for(TestConfig.PROCESSOR_COUNT) // 4),
            )

        if (
            TestConfig.CHIP_ARCH == ChipArchitecture.QUASAR
            and boot_mode != BootMode.TRISC
        ):
            raise ValueError("Quasar only supports TRISC boot mode")

        brisc_cmd_timeout = (
            TestConfig.SIMULATOR_TIMEOUT if TestConfig.TEST_TARGET.run_simulator else 1
        )

        if boot_mode == BootMode.BRISC:
            if not TestConfig.BRISC_ELF_LOADED:
                commit_tensix_soft_reset(1, location=TestConfig.TENSIX_LOCATION)
                load_elf(
                    elf_file=str((TestConfig.SHARED_ELF_DIR / "brisc.elf").absolute()),
                    location=TestConfig.TENSIX_LOCATION,
                    risc_name="brisc",
                    verify_write=True,
                )
                # Bring BRISC up, retrying the soft-reset kick until it reaches
                # its polling loop. A board-wide `tt-smi -r 0` can leave a core
                # slow-to-boot or wedged; re-polling alone never recovers a
                # wedged core, so each attempt re-asserts then de-asserts the
                # BRISC soft reset. BRISC_ELF_LOADED is latched only after
                # boot-ready succeeds, so a failed bring-up is retried on the
                # next test instead of poisoning the rest of this worker's run.
                last_err = None
                for attempt in range(TestConfig.BRISC_BOOT_MAX_ATTEMPTS):
                    if attempt:
                        commit_tensix_soft_reset(1, location=TestConfig.TENSIX_LOCATION)
                    # Pre-clear BriscCounter so we cannot latch onto a stale
                    # boot-ready sentinel left in L1 by a prior pytest process —
                    # mailboxes live at fixed L1 addresses outside any ELF
                    # section, so they survive ELF reload.
                    write_words_to_device(
                        TestConfig.TENSIX_LOCATION,
                        device_module.Mailboxes.BriscCounter.value,
                        [0],
                    )
                    commit_tensix_soft_reset(
                        0, [RiscCore.BRISC], TestConfig.TENSIX_LOCATION
                    )
                    try:
                        wait_brisc_boot_ready(
                            TestConfig.TENSIX_LOCATION, timeout=brisc_cmd_timeout
                        )
                    except TimeoutError as err:
                        last_err = err
                        continue
                    TestConfig.BRISC_ELF_LOADED = True
                    break
                else:
                    raise TimeoutError(
                        f"BRISC bring-up did not become ready after "
                        f"{TestConfig.BRISC_BOOT_MAX_ATTEMPTS} attempts"
                    ) from last_err

            # Reset only TRISCs, BRISC stays alive in its polling loop
            commit_brisc_command(
                TestConfig.TENSIX_LOCATION,
                BriscCmd.RESET_TRISCS,
                timeout=brisc_cmd_timeout,
            )
        else:
            commit_tensix_soft_reset(1, location=TestConfig.TENSIX_LOCATION)

        VARIANT_ELF_DIR = (
            TestConfig.ARTEFACTS_DIR / self.test_name / self.variant_id / "elf"
        )

        self.temp_elfs = [
            str((VARIANT_ELF_DIR / f"{trisc_name}.elf").absolute())
            for trisc_name in TestConfig.KERNEL_COMPONENTS
        ]

        if TestConfig.LAST_LOADED_ELFS != VARIANT_ELF_DIR:
            TestConfig.LAST_LOADED_ELFS = VARIANT_ELF_DIR

            for i, elf_file_path in enumerate(self.temp_elfs):
                if TestConfig.CHIP_ARCH == ChipArchitecture.WORMHOLE:
                    start_address = load_elf(
                        elf_file=elf_file_path,
                        location=TestConfig.TENSIX_LOCATION,
                        risc_name=f"trisc{i}",
                        return_start_address=True,
                        verify_write=False,
                    )
                    write_words_to_device(
                        TestConfig.TENSIX_LOCATION,
                        TestConfig.TRISC_START_ADDRS[i],
                        [start_address],
                    )
                else:
                    load_elf(
                        elf_file=elf_file_path,
                        location=TestConfig.TENSIX_LOCATION,
                        risc_name=f"trisc{i}",
                        neo_id=(
                            0
                            if TestConfig.CHIP_ARCH == ChipArchitecture.QUASAR
                            else None
                        ),
                        verify_write=False,
                    )

            if (
                boot_mode == BootMode.BRISC
                and TestConfig.CHIP_ARCH == ChipArchitecture.WORMHOLE
            ):
                commit_brisc_command(
                    TestConfig.TENSIX_LOCATION,
                    BriscCmd.UPDATE_START_ADDR_CACHE_AND_START,
                    timeout=brisc_cmd_timeout,
                )
                return

        match boot_mode:
            case BootMode.BRISC:
                commit_brisc_command(
                    TestConfig.TENSIX_LOCATION,
                    BriscCmd.START_TRISCS,
                    timeout=brisc_cmd_timeout,
                )
            case BootMode.TRISC:
                reset_mailboxes(TestConfig.TENSIX_LOCATION)
                set_tensix_soft_reset(0, [RiscCore.TRISC0], TestConfig.TENSIX_LOCATION)
            case BootMode.EXALENS:
                exalens_device_setup(TestConfig.CHIP_ARCH, TestConfig.TENSIX_LOCATION)
                set_tensix_soft_reset(0, TRISC_CORES, TestConfig.TENSIX_LOCATION)

        return

    def wait_for_tensix_operations_finished(self, timeout=2, poll_callback=None):
        """
        Args:
            elfs: List of ELF file paths (used for assert diagnostics).
            location: The location of the core to poll.
            timeout: Maximum time to wait (in seconds) before timing out.
            poll_callback: Optional callable invoked each iteration (used for device print drain).
        """

        mailboxes = {core for core in device_module.Mailboxes}
        if self.CHIP_ARCH != ChipArchitecture.QUASAR:
            mailboxes -= {
                device_module.Mailboxes.BriscCommand0,
                device_module.Mailboxes.BriscCommand1,
                device_module.Mailboxes.BriscCounter,
                device_module.Mailboxes.BriscBread0,
                device_module.Mailboxes.BriscBread1,
            }
        timeout = (
            TestConfig.SIMULATOR_TIMEOUT
            if TestConfig.TEST_TARGET.run_simulator
            else timeout
        )

        # Poll every mailbox in a single NoC transaction. They occupy one
        # contiguous block (Unpacker, +4, +8, plus +12 on Quasar), so reading the
        # whole span costs one round trip per iteration instead of one per TRISC.
        # Taking min/max rather than assuming adjacency keeps this correct even
        # if the layout gains a gap; it would just read a slightly wider span.
        base = min(mailbox.value for mailbox in mailboxes)
        span = max(mailbox.value for mailbox in mailboxes) + 4 - base
        word_index = {mailbox: (mailbox.value - base) // 4 for mailbox in mailboxes}

        completed = set()
        end_time = time.time() + timeout
        while time.time() < end_time:
            words = np.frombuffer(
                read_from_device(TestConfig.TENSIX_LOCATION, base, num_bytes=span),
                dtype=np.uint32,
            )
            for mailbox in mailboxes - completed:
                if words[word_index[mailbox]] == KERNEL_COMPLETE:
                    completed.add(mailbox)

            if poll_callback is not None:
                poll_callback()

            if completed == mailboxes:
                return

        handle_if_assert_hit(
            self.temp_elfs,
            core_loc=TestConfig.TENSIX_LOCATION,
        )

        trisc_hangs = [mailbox.name for mailbox in (mailboxes - completed)]
        raise TimeoutError(
            f"Timeout reached: waited {timeout} seconds for {', '.join(trisc_hangs)}"
        )

    def prepare(self):
        """Hash + build_elfs once. Safe to call from run() or earlier."""
        if self._prepared:
            return
        self.generate_variant_hash()
        if TestConfig.BUILD_MODE in [BuildMode.PRODUCE, BuildMode.DEFAULT]:
            self.build_elfs()
        self._prepared = True

    def run(self, poll_callback=None):
        self.prepare()

        logger.debug(
            "Running variant={} | location={}",
            self.variant_id[:12],
            TestConfig.TENSIX_LOCATION,
        )

        logger.debug(
            "ELF directory: {}",
            TestConfig.ARTEFACTS_DIR / self.test_name / self.variant_id / "elf",
        )

        if TestConfig.BUILD_MODE == BuildMode.PRODUCE:
            pytest.skip(TestConfig.SKIP_JUST_FOR_COMPILE_MARKER)

        self.write_runtimes_to_L1()

        if self.variant_stimuli:
            if TestConfig.STIMULI_MODE == StimuliMode.GENERATE_ONLY:
                self.variant_stimuli.save_to_cache()
                pytest.skip(TestConfig.SKIP_JUST_FOR_STIMULI_MARKER)
            elif TestConfig.STIMULI_MODE == StimuliMode.LOAD_CACHED:
                self.variant_stimuli.load_from_cache()

            self.variant_stimuli.write(TestConfig.TENSIX_LOCATION)

            # Run 0's share of the per-run clobber the bit-exactness check does
            # (_assert_bit_exact_repeats clears before each re-run). Without it
            # a kernel that writes fewer tiles than tile_count_res declares
            # leaves run 0 reading whatever the previous test left in L1 while
            # every re-run reads the sentinel there, and all of them get
            # reported as diverging.
            if self._bit_exact_check_applies():
                self.variant_stimuli.clear_result_buffer(TestConfig.TENSIX_LOCATION)

        # When device print is enabled, build a parser,
        # collect into dprint_lines, and return in TestOutcome.
        dprint_parser = None
        dprint_lines: list[str] = []
        wrapped_poll_callback = poll_callback
        if TestConfig.DEVICE_PRINT_ENABLED or self.requires_device_print:
            from .device_print import make_device_print_parser

            dprint_parser = make_device_print_parser(self)

            def _drain():
                batch = dprint_parser.poll(TestConfig.TENSIX_LOCATION)
                dprint_lines.extend(batch)
                for line in batch:
                    logger.debug(line)
                if poll_callback is not None:
                    poll_callback()

            wrapped_poll_callback = _drain

        self.run_elf_files()
        self.wait_for_tensix_operations_finished(poll_callback=wrapped_poll_callback)

        if dprint_parser is not None:
            final = dprint_parser.final_drain(TestConfig.TENSIX_LOCATION)
            dprint_lines.extend(final)
            for line in final:
                logger.debug(line)

        if self.coverage_build == CoverageBuild.Yes:
            self.read_coverage_data_from_device()

        # Repeat the on-device execution and assert every run is bit-identical.
        # Done before collect_results so the returned result is still the value
        # produced by the last (verified) run. Re-runs drain the device print
        # buffer without recording it, so repeats can't stall on a full buffer
        # nor duplicate run 0's output in the returned TestOutcome.
        self._assert_bit_exact_repeats(
            poll_callback=(
                None
                if dprint_parser is None
                else lambda: dprint_parser.poll(TestConfig.TENSIX_LOCATION)
            )
        )

        return TestOutcome(
            result=(
                self.variant_stimuli.collect_results(TestConfig.TENSIX_LOCATION)
                if self.variant_stimuli
                else None
            ),
            device_print_lines=dprint_lines,
        )

    def _bit_exact_unsupported_reason(self) -> str | None:
        """Why this variant cannot be checked for bit-exactness, or None if it can."""
        if self.coverage_build == CoverageBuild.Yes:
            return "not supported with coverage builds"
        if self.l1_acc == L1Accumulation.Yes:
            # The packer adds into the existing L1 destination, so each run
            # accumulates onto the previous one. Re-runs are legitimately
            # expected to differ and comparing them would be meaningless.
            return "L1 accumulation makes every run add onto the previous result"
        if self.expected_nondeterministic:
            # Negative controls that deliberately run with a corrupt HW config
            # (e.g. a zeroed addrmod) leave DEST addressing undefined, so the
            # result is not bit-reproducible by contract. The functional check
            # (expect_mismatch) still validates the single run; only the
            # bit-exact re-run comparison is meaningless here.
            return "variant intentionally exercises undefined hardware state"
        return None

    def _bit_exact_check_applies(self) -> bool:
        """Whether run() will compare repeats of this variant.

        Consulted before the first execution as well, so the result buffer can
        be cleared up front; it must therefore agree exactly with the guards in
        _assert_bit_exact_repeats.
        """
        return (
            TestConfig.BIT_EXACT_RUNS > 1
            and self.variant_stimuli is not None
            and self._bit_exact_unsupported_reason() is None
        )

    def _assert_bit_exact_repeats(self, poll_callback=None):
        """Re-run this variant and assert the result buffer is bit-identical.

        Only active when ``--bit-exact-runs`` (TestConfig.BIT_EXACT_RUNS) is
        greater than 1. Assumes the kernel has already been executed once (run()
        does the first execution), then re-runs it another ``BIT_EXACT_RUNS - 1``
        times, comparing the raw packed result bytes in L1 each time.

        The stimuli and runtime args run() wrote are untouched by the kernel, so
        they stay in L1 and are reused as-is. That keeps every run driven by
        byte-for-byte identical input and avoids re-packing the tensors on each
        iteration. The result region is clobbered with the same sentinel before
        every run, run 0 included (that one happens in run()). Uniform clobbering
        matters in both directions: it stops a run that writes fewer bytes than
        the last one from inheriting bytes that compare equal, and it keeps every
        run's untouched padding identical, so a kernel that writes less than its
        declared tile_count_res is not reported as non-deterministic.

        Every re-run is executed even if an earlier one already diverged, so the
        failure reports the full picture (how many runs differed and where)
        rather than stopping at the first mismatch.

        ``poll_callback`` is invoked while waiting for each re-run to finish; run()
        passes a device-print drain so a chatty kernel cannot fill the print
        buffer and stall.

        Skipped for coverage builds (re-runs would corrupt the coverage stream)
        and for tests without a stimuli/result buffer to read back.
        """
        runs = TestConfig.BIT_EXACT_RUNS
        if runs <= 1 or self.variant_stimuli is None:
            return
        unsupported = self._bit_exact_unsupported_reason()
        if unsupported is not None:
            logger.warning(
                "Bit-exactness check skipped for {}: {}.",
                self.variant_id[:12],
                unsupported,
            )
            return

        reference = self._read_output_regions()

        # Per differing run: (run_idx, region, first_offset, ref_byte, run_byte, num_diff).
        mismatches = []
        # Byte offsets that differed in any run, to tell a single flaky byte
        # apart from output that scatters differently every time.
        unstable_offsets = {region: set() for region in reference}

        for run_idx in range(1, runs):
            # Every run starts from the sentinel, so a run that writes fewer
            # bytes than the last one shows the sentinel in the tail instead of
            # inheriting the previous run's bytes and comparing equal. A varying
            # write extent is itself non-determinism, and this is the only thing
            # that catches it: the mailbox handshake proves the kernel finished,
            # not how much of the buffer it touched.
            self.variant_stimuli.clear_result_buffer(TestConfig.TENSIX_LOCATION)
            self.run_elf_files()
            self.wait_for_tensix_operations_finished(poll_callback=poll_callback)

            for region, current in self._read_output_regions().items():
                diff_offsets = np.flatnonzero(reference[region] != current)
                if diff_offsets.size == 0:
                    continue

                unstable_offsets[region].update(diff_offsets.tolist())
                first = int(diff_offsets[0])
                mismatches.append(
                    (
                        run_idx,
                        region,
                        first,
                        int(reference[region][first]),
                        int(current[first]),
                        int(diff_offsets.size),
                    )
                )

        if not mismatches:
            logger.debug(
                "Bit-exactness check passed for {}: {} runs bit-identical across {}.",
                self.variant_id[:12],
                runs,
                ", ".join(reference),
            )
            return

        affected = sorted({region for _, region, *_ in mismatches})
        diverging_runs = len({run_idx for run_idx, *_ in mismatches})
        lines = [
            f"Non-deterministic hardware output for variant {self.variant_id[:12]}: "
            f"{diverging_runs} of {runs - 1} re-runs differed from run 0 "
            f"in {', '.join(affected)}."
        ]
        lines.extend(
            f"  {region}: {len(unstable_offsets[region])} of {reference[region].size} "
            "byte(s) ever differed."
            for region in affected
        )
        lines.extend(
            f"  run {run_idx} [{region}]: {num_diff} byte(s) differ; "
            f"first at offset {first}: "
            f"run 0 = 0x{ref_byte:02X} vs run {run_idx} = 0x{run_byte:02X}"
            for run_idx, region, first, ref_byte, run_byte, num_diff in mismatches
        )
        lines.append(self._describe_input_integrity())
        raise AssertionError("\n".join(lines))

    def _read_output_regions(self) -> dict:
        """Raw bytes of every L1 region the kernel writes, keyed by region name.

        buffer_C is included because some tests use it as a second output (see
        test_sfpu_exp_parallel_matmul_quasar). When it is only an input it never
        changes between runs, so comparing it is harmless. It is deliberately not
        cleared between runs, unlike the result buffer, precisely because it may
        be an input.
        """
        stimuli = self.variant_stimuli
        location = TestConfig.TENSIX_LOCATION
        regions = {
            "result buffer": np.frombuffer(
                stimuli.collect_raw_result_bytes(location), dtype=np.uint8
            )
        }
        if stimuli.buffer_C is not None:
            regions["buffer_C"] = np.frombuffer(
                stimuli.collect_raw_buffer_c_bytes(location), dtype=np.uint8
            )
        return regions

    def _describe_input_integrity(self) -> str:
        """Report whether the input operands in L1 still match the stimuli.

        Re-runs reuse the input already in L1, so a kernel that writes to its own
        input would make later runs compute on different data and look like
        non-deterministic hardware. This distinguishes the two. Expected bytes
        come from re-writing the stimuli and reading them back, which reuses the
        real pack/write path instead of duplicating it.

        Only call this on a failure path: it costs two L1 reads plus a re-pack,
        and it restores the stimuli as a side effect.
        """
        stimuli = self.variant_stimuli
        actual = np.frombuffer(
            stimuli.read_input_region(TestConfig.TENSIX_LOCATION), dtype=np.uint8
        )
        stimuli.write(TestConfig.TENSIX_LOCATION)
        expected = np.frombuffer(
            stimuli.read_input_region(TestConfig.TENSIX_LOCATION), dtype=np.uint8
        )

        modified = int(np.count_nonzero(actual != expected))
        if not modified:
            return (
                "  Input operands in L1 were unchanged, so every run saw identical "
                "input: the divergence is in the hardware/kernel output itself."
            )
        return (
            f"  WARNING: {modified} of {actual.size} input byte(s) in L1 no longer "
            "match the stimuli, so the kernel writes to its own input and later runs "
            "did not see the same input. Fix that before suspecting the hardware."
        )


def process_coverage_run_artefacts() -> bool:
    start = time.time()
    sources = Path(TestConfig.ARTEFACTS_DIR) / "sources"

    compiled_variants = []
    for test_names in sources.iterdir():
        compiled_variants.extend(variant for variant in test_names.iterdir())

    def process_variants(compiled_variants: Path):
        for variant in compiled_variants:
            stream_runs = glob.glob(os.path.join(variant, "*.stream"))

            if not stream_runs:
                continue

            stream_parts = []
            for stream in stream_runs:
                with open(stream, "rb") as fd:
                    stream_parts.append(fd.read())
            merged_stream = b"".join(stream_parts)

            if merged_stream:
                run_shell_command(
                    f"{TestConfig.GCOV_TOOL} merge-stream",
                    TestConfig.TESTS_WORKING_DIR,
                    merged_stream,
                    text=False,
                )

                # Generate single .info file per variant
                info_hash = sha256(str(variant).encode()).hexdigest()
                command = (
                    f"lcov --gcov-tool {TestConfig.GCOV} --capture "
                    f"--directory {variant}/elf/ "
                    f"--output-file {TestConfig.COVERAGE_INFO_DIR}/{info_hash}.info "
                    "--rc lcov_branch_coverage=1"
                )
                run_shell_command(command, TestConfig.TESTS_WORKING_DIR)

    worker_num = 20

    logger.info("Processing code coverage data")
    with ThreadPoolExecutor(max_workers=worker_num) as executor:
        futures = [
            executor.submit(process_variants, work)
            for work in np.array_split(compiled_variants, worker_num)
        ]
        for fut in futures:
            fut.result()

    end = time.time()

    if not Path(TestConfig.COVERAGE_INFO_DIR).is_dir():
        logger.warning("{} does not exist. Early exit.", TestConfig.COVERAGE_INFO_DIR)
        return

    info_files = glob.glob(os.path.join(TestConfig.COVERAGE_INFO_DIR, "*.info"))
    logger.info(
        "Generated {} coverage .info files from streams in {:.2f}s, unifying",
        len(info_files),
        end - start,
    )

    # Reduce worker count to avoid workers having no files to process
    if len(info_files) < 2 * worker_num:
        worker_num = 1

    start = time.time()

    for i in range(worker_num):
        merged_path = TestConfig.ARTEFACTS_DIR / f"merged_coverage_{i}.info"
        try:
            shutil.copyfile(str(info_files[0]), merged_path)
        except IndexError:
            logger.warning("No worker files to be merged, exiting")
            return
        info_files.pop(0)

    def combine_files(index, info_files):
        merged_path = TestConfig.ARTEFACTS_DIR / f"merged_coverage_{index}.info"
        for info_file in info_files:
            cmd = f"lcov -a {merged_path} -a {info_file} -o {merged_path}"
            result = run_shell_command(cmd, TestConfig.ARTEFACTS_DIR)

            if result.returncode:
                logger.warning(
                    "Failed to merge {}, skipping: {}", info_file, result.stderr
                )

    with ThreadPoolExecutor(max_workers=worker_num) as executor:
        futures = [
            executor.submit(combine_files, i, work)
            for i, work in enumerate(np.array_split(info_files, worker_num))
        ]
        for fut in futures:
            fut.result()

    merged_path = TestConfig.ARTEFACTS_DIR / f"merged_coverage.info"
    shutil.copyfile(TestConfig.ARTEFACTS_DIR / f"merged_coverage_0.info", merged_path)

    for i in range(1, worker_num):
        info_file = TestConfig.ARTEFACTS_DIR / f"merged_coverage_{i}.info"
        cmd = f"lcov -a {merged_path} -a {info_file} -o {merged_path}"
        result = run_shell_command(cmd, TestConfig.ARTEFACTS_DIR)

        if result.returncode:
            logger.warning(
                "Failed to merge {}, skipping. Error: {}", info_file, result.stderr
            )

    end = time.time()
    logger.info("Combined {} coverage files in {:.2f}s", len(info_files), end - start)
