# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import fcntl
import glob
import os
import shutil
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import ClassVar, List

import numpy as np
import pytest
from filelock import FileLock
from ttexalens.tt_exalens_lib import (
    TTException,
    load_elf,
    parse_elf,
    read_from_device,
    read_word_from_device,
    write_to_device,
    write_words_to_device,
)

from . import device as device_module
from .chip_architecture import ChipArchitecture, get_chip_architecture
from .data_format_inference import data_formats, is_format_combination_outlier
from .device import (
    CHIP_DEFAULT_BOOT_MODES,
    BootMode,
    RiscCore,
    exalens_device_setup,
    reset_mailboxes,
    set_tensix_soft_reset,
    wait_for_tensix_operations_finished,
)
from .format_config import DataFormat, FormatConfig
from .llk_params import DestAccumulation, L1Accumulation, MailboxesDebug, MailboxesPerf
from .stimuli_config import StimuliConfig
from .test_variant_parameters import RuntimeParameter, TemplateParameter


class ProfilerBuild(Enum):
    Yes = "true"
    No = "false"


class CoverageBuild(Enum):
    Yes = "true"
    No = "false"


from .test_variant_parameters import (
    IN_TILE_DIMS,
    NUM_FACES,
    RuntimeParameter,
    TemplateParameter,
)
from .utils import create_directories, run_shell_command


class TestMode(Enum):
    DEFAULT = "Compile and consume sequentially"
    PRODUCE = "Just compile tests without executing them"
    CONSUME = "Just execute pre-compiled elfs"


class TestConfig:

    # === STATIC VARIABLES ===

    # Architecture Selection
    ARCH_NON_COMPUTE: ClassVar[str]
    ARCH_COMPUTE: ClassVar[str]
    ARCH_DEFINE: ClassVar[str]
    ARCH_LLK_ROOT: ClassVar[str]
    ARCH: ClassVar[str]
    CHIP_ARCH: ClassVar[ChipArchitecture]

    # Artefact directories
    DEFAULT_ARTEFACTS_PATH: ClassVar[Path] = Path("/tmp/tt-llk-build/")
    ARTEFACTS_DIR: ClassVar[Path]
    SHARED_DIR: ClassVar[str]
    SHARED_OBJ_DIR: ClassVar[str]
    SHARED_ELF_DIR: ClassVar[str]
    COVERAGE_INFO_DIR: ClassVar[str]
    SYNC_DIR: ClassVar[Path]
    PERF_DATA_DIR: ClassVar[Path]

    # Sources directories
    LLK_ROOT: ClassVar[Path]
    TESTS_WORKING_DIR: ClassVar[Path]
    TOOL_PATH: ClassVar[Path]
    HEADER_DIR: ClassVar[Path]

    HELPERS: ClassVar[Path]
    RISCV_SOURCES: ClassVar[Path]
    LINKER_SCRIPTS: ClassVar[Path]

    # Toolchain paths
    GXX: ClassVar[str]
    OBJDUMP: ClassVar[str]
    OBJCOPY: ClassVar[str]
    GCOV: ClassVar[str]
    GCOV_TOOL: ClassVar[str]

    # Compilation options
    OPTIONS_ALL: ClassVar[str] = None
    OPTIONS_LINK: ClassVar[str] = None
    INITIAL_OPTIONS_COMPILE: ClassVar[str] = None
    INCLUDES: ClassVar[List[str]] = []
    WITH_COVERAGE: ClassVar[bool] = False

    OPTIONS_COMPILE: ClassVar[str] = None
    MEMORY_LAYOUT_LD_SCRIPT: ClassVar[str] = None
    NON_COVERAGE_OPTIONS_COMPILE: ClassVar[str] = None

    SHARED_ARTEFACTS_AVAILABLE: ClassVar[bool] = False
    PROFILER_SHARED_ARTEFACTS_AVAILABLE: ClassVar[bool] = False
    KERNEL_COMPONENTS: ClassVar[list[str]] = ["unpack", "math", "pack"]

    # === Runtime static variables, for keeping context of multiple test runs
    CURRENT_LOADED_CONFIG: ClassVar[str] = "uninitialised"
    MODE: ClassVar[TestMode] = TestMode.DEFAULT
    SKIP_JUST_FOR_COMPILE_MARKER: ClassVar[str] = "SKIPPED_JUST_FOR_COMPILE"
    _BUILD_DIRS_CREATED: ClassVar[bool] = False

    # When the infrastructure itself needs to be tested, some functionality like compiling the artefacts and writing them
    # to tmpfs can be skipped (eg. object, elf and coverage data files etc.). This flag is used to skip such code to enable fast execution of infra tests.
    INFRA_TESTING: ClassVar[bool] = False

    # === Addresses ===
    RUNTIME_ADDRESS_NON_COVERAGE: ClassVar[int] = 0x20000
    RUNTIME_ADDRESS_COVERAGE: ClassVar[int] = 0x64000
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
    # Layout: 86 config words (344 bytes) + 172 data words (688 bytes) = 1032 (0x408) bytes per thread
    PERF_COUNTERS_BASE_ADDR: ClassVar[int] = 0x16A000
    PERF_COUNTERS_SIZE: ClassVar[int] = 0xC18  # 3096 bytes for all 3 threads
    _PERF_COUNTERS_CONFIG_WORDS: ClassVar[int] = 86
    _PERF_COUNTERS_DATA_WORDS: ClassVar[int] = 172
    _PERF_COUNTERS_THREAD_SIZE: ClassVar[int] = (
        _PERF_COUNTERS_CONFIG_WORDS + _PERF_COUNTERS_DATA_WORDS
    ) * 4  # 1032 bytes
    # Computed addresses (UNPACK=thread 0, MATH=thread 1, PACK=thread 2)
    PERF_COUNTER_UNPACK_CONFIG_ADDR: ClassVar[int] = PERF_COUNTERS_BASE_ADDR
    PERF_COUNTER_UNPACK_DATA_ADDR: ClassVar[int] = (
        PERF_COUNTERS_BASE_ADDR + _PERF_COUNTERS_CONFIG_WORDS * 4
    )
    PERF_COUNTER_MATH_CONFIG_ADDR: ClassVar[int] = (
        PERF_COUNTERS_BASE_ADDR + _PERF_COUNTERS_THREAD_SIZE
    )
    PERF_COUNTER_MATH_DATA_ADDR: ClassVar[int] = (
        PERF_COUNTERS_BASE_ADDR
        + _PERF_COUNTERS_THREAD_SIZE
        + _PERF_COUNTERS_CONFIG_WORDS * 4
    )
    PERF_COUNTER_PACK_CONFIG_ADDR: ClassVar[int] = (
        PERF_COUNTERS_BASE_ADDR + 2 * _PERF_COUNTERS_THREAD_SIZE
    )
    PERF_COUNTER_PACK_DATA_ADDR: ClassVar[int] = (
        PERF_COUNTERS_BASE_ADDR
        + 2 * _PERF_COUNTERS_THREAD_SIZE
        + _PERF_COUNTERS_CONFIG_WORDS * 4
    )

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
            case ChipArchitecture.BLACKHOLE:
                TestConfig.ARCH_NON_COMPUTE = "-mcpu=tt-bh"
                TestConfig.ARCH_COMPUTE = "-mcpu=tt-bh-tensix"
                TestConfig.ARCH_DEFINE = "-DARCH_BLACKHOLE"
                TestConfig.ARCH_LLK_ROOT = "tt_llk_blackhole"
                TestConfig.ARCH = ChipArchitecture.BLACKHOLE
            case ChipArchitecture.QUASAR:
                # until there is official support for quasar in SFPI fallback to BH
                TestConfig.ARCH_NON_COMPUTE = "-mcpu=tt-bh"
                TestConfig.ARCH_COMPUTE = "-mcpu=tt-bh-tensix"
                TestConfig.ARCH_DEFINE = "-DARCH_QUASAR"
                TestConfig.ARCH_LLK_ROOT = "tt_llk_quasar"
                TestConfig.ARCH = ChipArchitecture.QUASAR
            case _:
                raise ValueError(
                    "Must provide CHIP_ARCH environment variable (wormhole / blackhole / quasar)"
                )

    @staticmethod
    def setup_paths(sources_path: Path):
        TestConfig.LLK_ROOT = sources_path
        TestConfig.TESTS_WORKING_DIR = TestConfig.LLK_ROOT / "tests"
        TestConfig.TOOL_PATH = TestConfig.LLK_ROOT / "tests/sfpi/compiler/bin"
        TestConfig.HEADER_DIR = (
            TestConfig.TESTS_WORKING_DIR / f"hw_specific/{TestConfig.ARCH.value}/inc"
        )

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
    ):
        debug_flag = "" if no_debug_symbols else "-g "
        TestConfig.OPTIONS_ALL = f"{debug_flag}-O3 -std=c++17 -ffast-math"
        TestConfig.WITH_COVERAGE = with_coverage
        StimuliConfig.WITH_COVERAGE = with_coverage

        if detailed_artefacts:
            TestConfig.OPTIONS_ALL += (
                "-save-temps=obj -fdump-tree-all -fdump-rtl-all -v"
            )

        TestConfig.OPTIONS_LINK = "-fexceptions -Wl,-z,max-page-size=16 -Wl,-z,common-page-size=16 -nostartfiles -Wl,--trace"
        TestConfig.INITIAL_OPTIONS_COMPILE = f"-nostdlib -fno-use-cxa-atexit -Wall -fno-exceptions -fno-rtti -Wunused-parameter -Wfloat-equal -Wpointer-arith -Wnull-dereference -Wredundant-decls -Wuninitialized -Wmaybe-uninitialized -DTENSIX_FIRMWARE -DENV_LLK_INFRA -DENABLE_LLK_ASSERT {TestConfig.ARCH_DEFINE}"
        TestConfig.INCLUDES = [
            "-Isfpi/include",
            f"-I../{TestConfig.ARCH_LLK_ROOT}/llk_lib",
            f"-I../{TestConfig.ARCH_LLK_ROOT}/common/inc",
            f"-I../{TestConfig.ARCH_LLK_ROOT}/common/inc/sfpu",
            "-I../common",
            f"-I{TestConfig.HEADER_DIR}",
            f"-Ihw_specific/{TestConfig.ARCH.value}",
            f"-Ihw_specific/{TestConfig.ARCH.value}/metal_sfpu",
            "-Ifirmware/riscv/common",
            "-Ihelpers/include",
        ]

    @staticmethod
    def setup_build(
        sources_path: Path,
        with_coverage: bool = False,
        detailed_artefacts: bool = False,
        no_debug_symbols: bool = False,
    ):
        device_module.Mailbox = MailboxesDebug if with_coverage else MailboxesPerf

        TestConfig.setup_arch()
        TestConfig.setup_paths(sources_path)
        TestConfig.setup_compilation_options(
            with_coverage, detailed_artefacts, no_debug_symbols
        )

    @staticmethod
    def setup_mode(compile_consumer: bool = False, compile_producer: bool = False):

        if compile_consumer and compile_producer:
            raise ValueError(
                "Pytest can be configured to be either compilation producer or compilation consumer, not both"
            )

        TestConfig.ARTEFACTS_DIR = TestConfig.DEFAULT_ARTEFACTS_PATH
        TestConfig.MODE = TestMode.DEFAULT

        if compile_producer:
            TestConfig.MODE = TestMode.PRODUCE

        if compile_consumer:
            TestConfig.MODE = TestMode.CONSUME

        # Always have a fresh build when compiling
        if TestConfig.MODE != TestMode.CONSUME:
            shutil.rmtree(TestConfig.ARTEFACTS_DIR.absolute(), ignore_errors=True)

    # === Instance fields and methods ===
    def __init__(
        self,
        test_name: str,
        formats: FormatConfig = None,
        templates: list[TemplateParameter] = [],
        runtimes: list[RuntimeParameter] = [],
        variant_stimuli: StimuliConfig = None,
        boot_mode: BootMode = BootMode.DEFAULT,
        profiler_build: ProfilerBuild = ProfilerBuild.No,
        L1_to_L1_iterations: int = 1,
        unpack_to_dest: bool = False,
        disable_format_inference: bool = False,
        dest_acc: DestAccumulation = DestAccumulation.No,
        l1_acc: L1Accumulation = L1Accumulation.No,
        skip_build_header: bool = False,
    ):
        self.coverage_build = (
            CoverageBuild.Yes if TestConfig.WITH_COVERAGE else CoverageBuild.No
        )

        if test_name is None:
            raise RuntimeError(
                "test_name argument needs to be passed in order to resolve which C++ file is compiled"
            )

        self.test_name = test_name
        self.formats = formats
        self.templates = templates
        self.runtimes = runtimes
        self.variant_stimuli = variant_stimuli
        self.boot_mode = boot_mode
        self.profiler_build = profiler_build
        self.L1_to_L1_iterations = L1_to_L1_iterations
        self.unpack_to_dest = unpack_to_dest
        self.disable_format_inference = disable_format_inference
        self.dest_acc = dest_acc
        self.l1_acc = l1_acc
        self.skip_build_header = skip_build_header

        # We need to call this here because this function generates serialisation format need for writing RTs to L1,
        # Which is needed by execution part of test infra
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

        if self.variant_stimuli:
            if TestConfig.WITH_COVERAGE:
                self.variant_stimuli.coverage_addresses = True
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

    def write_runtimes_to_L1(self, location: str = "0,0"):
        TILE_SIZES = {
            DataFormat.Bfp8_b: 68,
            DataFormat.Float32: 256,
        }

        if self.formats is None:
            pack_size, unpack_size_a, unpack_size_b = 128, 128, 128
        else:
            pack_size = TILE_SIZES.get(self.formats.output_format, 128)
            unpack_size_a = TILE_SIZES.get(self.formats.input_format, 128)
            unpack_size_b = TILE_SIZES.get(self.formats.input_format, 128)

        if len(self.runtimes) > 0:
            itd_param = next(
                (param for param in self.runtimes if isinstance(param, IN_TILE_DIMS)),
                None,
            )
            faces_param = next(
                (param for param in self.runtimes if isinstance(param, NUM_FACES)), None
            )
            if itd_param and faces_param:
                temp_num_faces_A = (
                    faces_param.num_faces_A
                    if faces_param.num_faces_A
                    else faces_param.num_faces
                )
                if itd_param.in0_r_dim <= 16:
                    pack_size = (pack_size // faces_param.num_faces) * (
                        itd_param.in0_r_dim // self.variant_stimuli.face_r_dim
                    )
                    unpack_size_a = (unpack_size_a // temp_num_faces_A) * (
                        itd_param.in0_r_dim // self.variant_stimuli.face_r_dim
                    )

        argument_data = [
            pack_size,  # uint32_t TILE_SIZE_PACK;
            unpack_size_a,  # uint32_t TILE_SIZE_UNPACK_A;
            unpack_size_b,  # uint32_t TILE_SIZE_UNPACK_B;
        ]

        if self.variant_stimuli:
            argument_data.extend(
                self.variant_stimuli.generate_runtime_operands_values(self.formats)
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
                    location, TestConfig.RUNTIME_ADDRESS_COVERAGE, serialised_data
                )
            else:
                write_to_device(
                    location, TestConfig.RUNTIME_ADDRESS_NON_COVERAGE, serialised_data
                )

    def collect_hash(self):
        lock_file = Path("/tmp/tt-llk-build-print.lock")
        lock_file.touch(exist_ok=True)

        with open(lock_file, "w") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                print(self.variant_id, file=sys.stderr)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

        pytest.skip()

    def generate_variant_hash(self):
        NON_COMPILATION_ARGUMENTS = [
            "variant_stimuli",
            "run_configs",
            "variant_id",
            "runtime_arguments_struct",
            "runtime_format",
            "runtimes",
        ]

        temp_str = [
            str(value)
            for field_name, value in self.__dict__.items()
            if field_name not in NON_COMPILATION_ARGUMENTS
        ]

        self.variant_id = sha256(str(" | ".join(temp_str)).encode()).hexdigest()

    def resolve_compile_options(self) -> tuple[str, str, str]:

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
        OPTIONS_COMPILE = (
            f"{' '.join(TestConfig.INCLUDES)} {TestConfig.INITIAL_OPTIONS_COMPILE} "
        )

        if TestConfig.CHIP_ARCH == ChipArchitecture.QUASAR:
            OPTIONS_COMPILE += "-DLLK_BOOT_MODE_TRISC "
        else:
            OPTIONS_COMPILE += "-DLLK_BOOT_MODE_BRISC "

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

        return (OPTIONS_COMPILE, MEMORY_LAYOUT_LD_SCRIPT, NON_COVERAGE_OPTIONS_COMPILE)

    def build_shared_artefacts(self):
        # Profiler builds require different shared artefacts (trisc.cpp compiles with -DLLK_PROFILER)
        is_profiler = self.profiler_build == ProfilerBuild.Yes

        # Select appropriate directories, flags, and lock based on build type
        if is_profiler:
            if TestConfig.PROFILER_SHARED_ARTEFACTS_AVAILABLE:
                return
            shared_obj_dir = TestConfig.PROFILER_SHARED_OBJ_DIR
            shared_elf_dir = TestConfig.PROFILER_SHARED_ELF_DIR
            lock_file = "/tmp/tt-llk-build-shared-profiler.lock"
        else:
            if TestConfig.SHARED_ARTEFACTS_AVAILABLE:
                return
            shared_obj_dir = TestConfig.SHARED_OBJ_DIR
            shared_elf_dir = TestConfig.SHARED_ELF_DIR
            lock_file = "/tmp/tt-llk-build-shared.lock"

        done_marker = shared_obj_dir / ".shared_complete"

        # Fast path: if shared artefacts are already built
        if done_marker.exists():
            if is_profiler:
                TestConfig.PROFILER_SHARED_ARTEFACTS_AVAILABLE = True
            else:
                TestConfig.SHARED_ARTEFACTS_AVAILABLE = True
            return

        # Acquire lock for building shared artefacts
        lock = FileLock(lock_file)

        with lock:
            # Check again inside lock
            if done_marker.exists():
                if is_profiler:
                    TestConfig.PROFILER_SHARED_ARTEFACTS_AVAILABLE = True
                else:
                    TestConfig.SHARED_ARTEFACTS_AVAILABLE = True
                return

            local_options_compile, local_memory_layout_ld, local_non_coverage = (
                self.resolve_compile_options()
            )

            # tmu-crt0.o : tmu-crt0.S
            run_shell_command(
                f"""{TestConfig.GXX} {TestConfig.ARCH_NON_COMPUTE} {TestConfig.OPTIONS_ALL} {local_options_compile} -c -o {shared_obj_dir / "tmu-crt0.o"} {TestConfig.HELPERS / "tmu-crt0.S"}""",
                TestConfig.TESTS_WORKING_DIR,
            )

            # brisc.o : brisc.cpp
            if TestConfig.CHIP_ARCH != ChipArchitecture.QUASAR:
                brisc_define_coverage = "-DCOVERAGE" if TestConfig.WITH_COVERAGE else ""
                run_shell_command(
                    f"""{TestConfig.GXX} {TestConfig.ARCH_NON_COMPUTE} {brisc_define_coverage} {TestConfig.OPTIONS_ALL} {local_non_coverage} -c -o {shared_obj_dir / "brisc.o"} {TestConfig.RISCV_SOURCES / "brisc.cpp"}""",
                    TestConfig.TESTS_WORKING_DIR,
                )

            COVERAGE_DEPS = ""
            if self.coverage_build == CoverageBuild.Yes:
                COVERAGE_DEPS = f"{shared_obj_dir}/coverage.o -lgcov"
                # coverage.o : coverage.cpp
                run_shell_command(
                    f"""{TestConfig.GXX} {TestConfig.ARCH_NON_COMPUTE} {TestConfig.OPTIONS_ALL} {local_non_coverage} -fno-strict-aliasing -c -o {shared_obj_dir / "coverage.o"} {TestConfig.RISCV_SOURCES / "coverage.cpp"}""",
                    TestConfig.TESTS_WORKING_DIR,
                )

            def build_kernel_part_main(name: str):
                kernel_trisc_flag = ""
                if TestConfig.CHIP_ARCH != ChipArchitecture.QUASAR:
                    kernel_trisc_flag = f"-DCOMPILE_FOR_TRISC={TestConfig.KERNEL_COMPONENTS.index(name)}"

                run_shell_command(  # main_%.o
                    f"""{TestConfig.GXX} {TestConfig.ARCH_COMPUTE} {TestConfig.OPTIONS_ALL} {local_options_compile} {kernel_trisc_flag} -DLLK_TRISC_{name.upper()} -c -o {shared_obj_dir / f"main_{name}.o"} {TestConfig.RISCV_SOURCES / "trisc.cpp"}""",
                    TestConfig.TESTS_WORKING_DIR,
                )

            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [
                    executor.submit(build_kernel_part_main, name)
                    for name in TestConfig.KERNEL_COMPONENTS
                ]
                for fut in futures:
                    fut.result()

            if TestConfig.CHIP_ARCH != ChipArchitecture.QUASAR:
                # brisc.elf : tmu-crt0.o brisc.o
                run_shell_command(
                    f"""{TestConfig.GXX} {TestConfig.ARCH_NON_COMPUTE} {TestConfig.OPTIONS_ALL} {TestConfig.OPTIONS_LINK} {shared_obj_dir / "tmu-crt0.o"} {shared_obj_dir / "brisc.o"} {COVERAGE_DEPS} -T{local_memory_layout_ld} -T{TestConfig.LINKER_SCRIPTS / "brisc.ld"} -T{TestConfig.LINKER_SCRIPTS / "sections.ld"} -o {shared_elf_dir / "brisc.elf"}""",
                    TestConfig.TESTS_WORKING_DIR,
                )

            # Mark shared artefacts as complete
            done_marker.touch()
            if is_profiler:
                TestConfig.PROFILER_SHARED_ARTEFACTS_AVAILABLE = True
            else:
                TestConfig.SHARED_ARTEFACTS_AVAILABLE = True

    def infer_data_formats(self) -> list[str]:
        header_content: list[str] = [
            "// Data formats inferred by Python inference model"
        ]

        dest_acc = self.dest_acc
        l1_acc = self.l1_acc

        if self.formats is None:
            header_content.extend(
                [
                    f"constexpr bool is_fp32_dest_acc_en = {dest_acc.value};",
                    f"constexpr bool l1_acc_en = {l1_acc.value};",
                    f"constexpr bool unpack_to_dest = {str(self.unpack_to_dest).lower()};",
                    "",
                ]
            )

            return header_content

        # Check if this is an outlier format combination that requires dest_acc to be enabled
        # Automatically enable dest_acc for outlier combinations
        if (
            is_format_combination_outlier(
                self.formats.input_format, self.formats.output_format, self.dest_acc
            )
            and TestConfig.CHIP_ARCH != ChipArchitecture.QUASAR
        ):
            dest_acc = DestAccumulation.Yes

        # Dest accumulation
        header_content.append(f"constexpr bool is_fp32_dest_acc_en = {dest_acc.value};")

        # L1 accumulation
        header_content.append(f"constexpr bool l1_acc_en = {l1_acc.value};")

        # Fused Test L1 to L1 : Input of first run is used as input for the second run ...
        # Not fusing: single L1-to-L1 iteration, so we retrieve one format configuration
        # L1_to_L1_iterations is the number of times we perform llk operations from L1 input tensor to L1 output tensor
        # If L1_to_L1_ITERATIONS is 1, we take input tensor from L1 -> unpack -> math -> pack -> L1
        # If L1_to_L1_ITERATIONS is greater than 1, we perform multiple iterations of unpack -> math -> pack, by taking results tensor in L1 to be input tensor of next iteration

        formats_config = data_formats(
            input_format=self.formats.input_format,
            output_format=self.formats.output_format,
            is_fp32_dest_acc_en=dest_acc,
            num_iterations=self.L1_to_L1_iterations,
            unpacking_to_dest=self.unpack_to_dest,
            chip_arch=TestConfig.CHIP_ARCH,
            disable_format_inference=self.disable_format_inference,
        )

        header_content.append(
            f"constexpr bool unpack_to_dest = {str(self.unpack_to_dest).lower()};"
        )

        # Check if we need to generate multiple format configurations

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
                for fmt in formats_config
            ]
            unpack_a_out_values = [
                f"ckernel::to_underlying(DataFormat::{fmt.unpack_A_dst.name})"
                for fmt in formats_config
            ]
            math_values = [
                f"ckernel::to_underlying(DataFormat::{fmt.math.name})"
                for fmt in formats_config
            ]
            pack_in_values = [
                f"ckernel::to_underlying(DataFormat::{fmt.pack_src.name})"
                for fmt in formats_config
            ]
            pack_out_values = [
                f"ckernel::to_underlying(DataFormat::{fmt.pack_dst.name})"
                for fmt in formats_config
            ]

            header_content.extend(
                [
                    f"constexpr std::array<std::underlying_type_t<DataFormat>, L1_to_L1_ITERATIONS> UNPACK_A_IN_LIST = {{{', '.join(unpack_a_in_values)}}};",
                    f"constexpr std::array<std::underlying_type_t<DataFormat>, L1_to_L1_ITERATIONS> UNPACK_A_OUT_LIST = {{{', '.join(unpack_a_out_values)}}};",
                    f"constexpr std::array<std::underlying_type_t<DataFormat>, L1_to_L1_ITERATIONS> MATH_FORMAT_LIST = {{{', '.join(math_values)}}};",
                    f"constexpr std::array<std::underlying_type_t<DataFormat>, L1_to_L1_ITERATIONS> PACK_IN_LIST = {{{', '.join(pack_in_values)}}};",
                    f"constexpr std::array<std::underlying_type_t<DataFormat>, L1_to_L1_ITERATIONS> PACK_OUT_LIST = {{{', '.join(pack_out_values)}}};",
                ]
            )

        else:
            # Single iteration - use simple format inference
            # Generate format data as individual constants for single iteration
            formats_config = formats_config[0]
            header_content.extend(
                [
                    "// Format data for single L1-to-L1 iteration",
                    f"constexpr auto UNPACK_A_IN = ckernel::to_underlying(DataFormat::{formats_config.unpack_A_src.name});",
                    f"constexpr auto UNPACK_A_OUT = ckernel::to_underlying(DataFormat::{formats_config.unpack_A_dst.name});",
                    f"constexpr auto MATH_FORMAT = ckernel::to_underlying(DataFormat::{formats_config.math.name});",
                    f"constexpr auto PACK_IN = ckernel::to_underlying(DataFormat::{formats_config.pack_src.name});",
                    f"constexpr auto PACK_OUT = ckernel::to_underlying(DataFormat::{formats_config.pack_dst.name});",
                ]
            )

        return header_content

    def generate_build_header(self) -> str:
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
            '#include "llk_sfpu_types.h"',
            (
                '#include "perf.h"'
                if TestConfig.CHIP_ARCH != ChipArchitecture.QUASAR
                else ""
            ),
            '#include "tensix_types.h"',
            "",
            "// Basic configuration",
            "constexpr std::uint32_t TILE_SIZE_CNT = 0x1000;",
        ]

        TILE_SIZES = {
            DataFormat.Bfp8_b: 68,
            DataFormat.Float32: 256,
        }

        if self.formats is None:
            pack_size, unpack_size_a, unpack_size_b = 128, 128, 128
        else:
            pack_size = TILE_SIZES.get(self.formats.output_format, 128)
            unpack_size_a = TILE_SIZES.get(self.formats.input_format, 128)
            unpack_size_b = TILE_SIZES.get(self.formats.input_format, 128)

        if len(self.runtimes) > 0:
            itd_param = next(
                (param for param in self.runtimes if isinstance(param, IN_TILE_DIMS)),
                None,
            )
            faces_param = next(
                (param for param in self.runtimes if isinstance(param, NUM_FACES)), None
            )
            if itd_param and faces_param:
                temp_num_faces_A = (
                    faces_param.num_faces_A
                    if faces_param.num_faces_A
                    else faces_param.num_faces
                )
                if itd_param.in0_r_dim <= 16:
                    pack_size = (pack_size // faces_param.num_faces) * (
                        itd_param.in0_r_dim // self.variant_stimuli.face_r_dim
                    )
                    unpack_size_a = (unpack_size_a // temp_num_faces_A) * (
                        itd_param.in0_r_dim // self.variant_stimuli.face_r_dim
                    )

        header_content.extend(
            [
                f"constexpr std::uint32_t TILE_SIZE_PACK = {pack_size};",
                f"constexpr std::uint32_t TILE_SIZE_UNPACK_A = {unpack_size_a};",
                f"constexpr std::uint32_t TILE_SIZE_UNPACK_B = {unpack_size_b};",
            ]
        )

        for parameter in self.templates:
            header_content.append(parameter.covert_to_cpp())

        header_content.extend(self.infer_data_formats())
        header_content.extend(self.runtime_arguments_struct)

        return "\n".join(header_content)

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

            SFPI_DEPS = ""
            COVERAGE_DEPS = ""
            if self.coverage_build == CoverageBuild.Yes:
                SFPI_DEPS = "-lgcov"
                COVERAGE_DEPS = shared_obj_dir / "coverage.o"

            def build_kernel_part(name: str):
                kernel_trisc_flag = ""
                if TestConfig.CHIP_ARCH != ChipArchitecture.QUASAR:
                    kernel_trisc_flag = f"-DCOMPILE_FOR_TRISC={TestConfig.KERNEL_COMPONENTS.index(name)}"
                run_shell_command(  # kernel_%.o
                    f"""{TestConfig.GXX} {TestConfig.ARCH_COMPUTE} {TestConfig.OPTIONS_ALL} -I{VARIANT_DIR} {local_options_compile} {kernel_trisc_flag} -DLLK_TRISC_{name.upper()} -c -o {VARIANT_OBJ_DIR / f"kernel_{name}.o"} {TestConfig.TESTS_WORKING_DIR / self.test_name}""",
                    TestConfig.TESTS_WORKING_DIR,
                )

                run_shell_command(  # %.elf : main_%.o kernel_%.o [coverage.o] tmu-crt0.o
                    f"""{TestConfig.GXX} {TestConfig.ARCH_COMPUTE} {TestConfig.OPTIONS_ALL} {TestConfig.OPTIONS_LINK} {shared_obj_dir / f"main_{name}.o"} {VARIANT_OBJ_DIR / f"kernel_{name}.o"} {COVERAGE_DEPS} {shared_obj_dir / "tmu-crt0.o"} {SFPI_DEPS} -T{local_memory_layout_ld} -T{TestConfig.LINKER_SCRIPTS / f"{name}.ld"} -T{TestConfig.LINKER_SCRIPTS / "sections.ld"} -o {VARIANT_ELF_DIR / f"{name}.elf"}""",
                    TestConfig.TESTS_WORKING_DIR,
                )

            with ThreadPoolExecutor(max_workers=3) as executor:
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
                        f"{TestConfig.OBJCOPY} -O binary -j .profiler_meta {elf_path} {meta_bin_path}",
                        TestConfig.TESTS_WORKING_DIR,
                    )

            # Mark build as complete so other processes know they can use the artefacts
            done_marker.touch()

    def read_coverage_data_from_device(self, location="0,0"):
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
            length = read_word_from_device(location, addr=coverage_start)
            coverage_stream += read_from_device(
                location, coverage_start + 4, num_bytes=length - 4
            )

        if len(self.runtimes) == 0:
            stream_name = "deafult_stream_name.stream"
        else:
            stream_name = f"{sha256(str(' | '.join([str(run_arg) for run_arg in self.runtimes])).encode()).hexdigest()}.stream"

        with open(
            VARIANT_DIR / stream_name,
            "wb",
        ) as fd:
            fd.write(coverage_stream)

    BRISC_ELF_LOADED: ClassVar[bool] = False
    PROFILER_BRISC_ELF_LOADED: ClassVar[bool] = False

    def run_elf_files(self, location="0,0") -> list:
        boot_mode = (
            CHIP_DEFAULT_BOOT_MODES[TestConfig.CHIP_ARCH]
            if self.boot_mode == BootMode.DEFAULT
            else self.boot_mode
        )

        if (
            TestConfig.CHIP_ARCH == ChipArchitecture.QUASAR
            and boot_mode != BootMode.TRISC
        ):
            raise ValueError("Quasar only supports TRISC boot mode")

        reset_mailboxes(location)

        set_tensix_soft_reset(1, location=location)

        VARIANT_ELF_DIR = (
            TestConfig.ARTEFACTS_DIR / self.test_name / self.variant_id / "elf"
        )

        elfs = [
            str((VARIANT_ELF_DIR / f"{trisc_name}.elf").absolute())
            for trisc_name in TestConfig.KERNEL_COMPONENTS
        ]

        for i, elf in enumerate(elfs):
            if TestConfig.CHIP_ARCH == ChipArchitecture.WORMHOLE:
                start_address = load_elf(
                    elf_file=elf,
                    location=location,
                    risc_name=f"trisc{i}",
                    neo_id=(
                        0 if TestConfig.CHIP_ARCH == ChipArchitecture.QUASAR else None
                    ),
                    return_start_address=True,
                    verify_write=False,
                )
                write_words_to_device(
                    location, TestConfig.TRISC_START_ADDRS[i], [start_address]
                )
            else:
                load_elf(
                    elf_file=elf,
                    location=location,
                    risc_name=f"trisc{i}",
                    neo_id=(
                        0 if TestConfig.CHIP_ARCH == ChipArchitecture.QUASAR else None
                    ),
                    verify_write=False,
                )

        match boot_mode:
            case BootMode.BRISC:
                # Use correct shared ELF directory and loading flag based on profiler build
                is_profiler = self.profiler_build == ProfilerBuild.Yes
                if is_profiler:
                    if not TestConfig.PROFILER_BRISC_ELF_LOADED:
                        TestConfig.PROFILER_BRISC_ELF_LOADED = True
                        load_elf(
                            elf_file=str(
                                (
                                    TestConfig.PROFILER_SHARED_ELF_DIR / "brisc.elf"
                                ).absolute()
                            ),
                            location=location,
                            risc_name="brisc",
                            verify_write=False,
                        )
                else:
                    if not TestConfig.BRISC_ELF_LOADED:
                        TestConfig.BRISC_ELF_LOADED = True
                        load_elf(
                            elf_file=str(
                                (TestConfig.SHARED_ELF_DIR / "brisc.elf").absolute()
                            ),
                            location=location,
                            risc_name="brisc",
                            verify_write=False,
                        )
                set_tensix_soft_reset(0, [RiscCore.BRISC], location)
            case BootMode.TRISC:
                set_tensix_soft_reset(
                    0, [RiscCore.TRISC0, RiscCore.TRISC1, RiscCore.TRISC2], location
                )
            case BootMode.EXALENS:
                exalens_device_setup(TestConfig.CHIP_ARCH, location)
                set_tensix_soft_reset(
                    0, [RiscCore.TRISC0, RiscCore.TRISC1, RiscCore.TRISC2], location
                )

        return elfs

    def run(self, location="0,0", delete_artefacts: bool = False):
        self.generate_variant_hash()
        if TestConfig.MODE in [TestMode.PRODUCE, TestMode.DEFAULT]:
            self.build_elfs()

        if TestConfig.MODE == TestMode.PRODUCE:
            pytest.skip(TestConfig.SKIP_JUST_FOR_COMPILE_MARKER)

        self.write_runtimes_to_L1(location)
        self.variant_stimuli.write(location)
        elfs = self.run_elf_files(location)
        wait_for_tensix_operations_finished(elfs, location)

        if self.coverage_build == CoverageBuild.Yes:
            self.read_coverage_data_from_device(location)

        if delete_artefacts:
            shutil.rmtree(TestConfig.ARTEFACTS_DIR / self.test_name / self.variant_id)

        return self.variant_stimuli.collect_results(location)


def process_coverage_run_artefacts() -> bool:
    start = time.time()
    sources = Path(TestConfig.ARTEFACTS_DIR) / "sources"

    compiled_variants = []
    for test_names in sources.iterdir():
        compiled_variants.extend(variant for variant in test_names.iterdir())

    def process_variants(compiled_variants: Path):
        for variant in compiled_variants:
            stream_runs = glob.glob(os.path.join(variant, "*.stream"))

            for stream in stream_runs:

                with open(stream, "rb") as fd:
                    coverage_stream = fd.read()
                run_shell_command(
                    f"{TestConfig.GCOV_TOOL} merge-stream",
                    TestConfig.TESTS_WORKING_DIR,
                    coverage_stream,
                    text=False,
                )

                info_hash = sha256(str(stream).encode()).hexdigest()
                command = (
                    f"lcov --gcov-tool {TestConfig.GCOV} --capture "
                    f"--directory {variant / 'obj/'} "
                    f"--output-file {TestConfig.COVERAGE_INFO_DIR}/{info_hash}.info "
                    "--rc lcov_branch_coverage=1"
                )
                run_shell_command(command, TestConfig.TESTS_WORKING_DIR)

    worker_num = 20

    print(f"Processing code coverage data")
    with ThreadPoolExecutor(max_workers=worker_num) as executor:
        futures = [
            executor.submit(process_variants, work)
            for work in np.array_split(compiled_variants, worker_num)
        ]
        for fut in futures:
            fut.result()

    end = time.time()

    if not Path(TestConfig.COVERAGE_INFO_DIR).is_dir():
        print(f"{TestConfig.COVERAGE_INFO_DIR} does not exist. Early exit.")
        return

    info_files = glob.glob(os.path.join(TestConfig.COVERAGE_INFO_DIR, "*.info"))
    print(
        f"Generated {len(info_files)} coverage .info files from streams in {end - start:.2f}s, unifying"
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
            print("No worker files to be merged, exiting")
            return
        info_files.pop(0)

    def combine_files(index, info_files):
        merged_path = TestConfig.ARTEFACTS_DIR / f"merged_coverage_{index}.info"
        for info_file in info_files:
            cmd = f"lcov -a {merged_path} -a {info_file} -o {merged_path}"
            result = run_shell_command(cmd, TestConfig.ARTEFACTS_DIR)

            if result.returncode:
                print(f"Warning: Failed to merge {info_file}, skipping")
                print(f"Error: {result.stderr}")

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
            print(f"Warning: Failed to merge {info_file}, skipping")
            print(f"Error: {result.stderr}")

    end = time.time()
    print(f"Combined {len(info_files)} in {end - start:.2f}s")
