# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from functools import reduce
from hashlib import sha256
from typing import List

import pandas as pd
import pytest
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.data_format_inference import data_formats, is_format_combination_outlier
from helpers.llk_params import DestAccumulation, DestSync, PerfRunType
from helpers.logger import logger
from helpers.perf import PerfReport
from helpers.profiler import Profiler, ProfilerData
from helpers.test_config import BuildMode, ProfilerBuild, StimuliMode, TestConfig
from ttexalens.tt_exalens_lib import read_words_from_device

from .fused_operand import OperandRegistry
from .fused_operation import FusedOperation


@dataclass
class GlobalConfig:
    test_name: str = "fused_test"
    architecture: ChipArchitecture = None
    dest_acc: DestAccumulation = DestAccumulation.No
    regenerate_cpp: bool = False
    profiler_enabled: bool = False
    perf_run_type: PerfRunType = None
    loop_factor: int = 16


class FuserConfig(TestConfig):
    pipeline: List[FusedOperation]
    global_config: GlobalConfig
    operand_registry: OperandRegistry

    def __init__(
        self,
        pipeline: List[FusedOperation],
        global_config: GlobalConfig,
        operand_registry: OperandRegistry,
    ):
        super().__init__(test_name="", skip_build_header=True)

        self.pipeline = pipeline
        self.global_config = global_config
        self.operand_registry = operand_registry

        if self.global_config.architecture is None:
            self.global_config.architecture = self.CHIP_ARCH

        for operation in self.pipeline:
            if is_format_combination_outlier(
                operation.math.operations[0].src_a.data_format,
                operation.output.data_format,
                self.global_config.dest_acc,
            ):
                raise ValueError(
                    f"Dest Accumulation must be enabled for {operation.math.operations[0].src_a.data_format} input and {operation.output.data_format} output"
                )

        num_stages = len(self.pipeline)

        for i, operation in enumerate(self.pipeline, start=1):
            formats_config = data_formats(
                input_format=operation.math.operations[0].src_a.data_format,
                output_format=operation.output.data_format,
                is_fp32_dest_acc_en=self.global_config.dest_acc,
                num_iterations=1,
                unpacking_to_dest=operation.unpack_to_dest,
                chip_arch=get_chip_architecture(),
                disable_format_inference=False,
            )[0]

            operation.unpack_a_in = formats_config.unpack_A_src
            operation.unpack_a_out = formats_config.unpack_A_dst
            operation.unpack_b_in = formats_config.unpack_B_src
            operation.unpack_b_out = formats_config.unpack_B_dst
            operation.math_format = formats_config.math
            operation.pack_in = formats_config.pack_src
            operation.pack_out = formats_config.pack_dst
            operation.stage_id = i
            operation.num_stages = num_stages

            if operation.dest_sync == DestSync.Half:
                dest_capacity = (
                    4 if self.global_config.dest_acc == DestAccumulation.Yes else 8
                )
            else:
                dest_capacity = (
                    8 if self.global_config.dest_acc == DestAccumulation.Yes else 16
                )

            if operation.block_tiles_x * operation.block_tiles_y > dest_capacity:
                raise ValueError(
                    f"Block size ({operation.block_size}) is bigger than dest capacity ({dest_capacity})"
                )

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
            "pipeline",
            "global_config",
            "operand_registry",
        ]

        temp_str = [
            str(value)
            for field_name, value in self.__dict__.items()
            if field_name not in NON_COMPILATION_ARGUMENTS
        ]

        self.variant_id = sha256(str(" | ".join(temp_str)).encode()).hexdigest()

    def generate_and_build_test(self):
        from .fused_generator import FusedKernelGenerator

        code_generator = FusedKernelGenerator(self)
        code_generator.write_kernel(self.test_name)
        self.build_elfs()

    def run_perf_test(self, worker_id: str, run_count: int = 2):
        """Run performance tests for different isolation levels (L1, unpack, math, pack, congestion) and collect profiling data."""

        from .fused_generator import FUSED_TESTS_DIR

        self.global_config.profiler_enabled = True
        self.profiler_build = ProfilerBuild.Yes

        run_types = [
            PerfRunType.L1_TO_L1,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.MATH_ISOLATE,
            PerfRunType.PACK_ISOLATE,
            PerfRunType.L1_CONGESTION,
        ]

        perf_report = PerfReport()
        all_results = []

        for run_type in run_types:
            runs = []
            self.global_config.perf_run_type = run_type

            self.test_name = (
                FUSED_TESTS_DIR / f"{self.global_config.test_name}_{run_type.name}.cpp"
            )

            self.generate_variant_hash()

            self.operand_registry.allocate_l1_addresses()

            if self.BUILD_MODE in [BuildMode.PRODUCE, BuildMode.DEFAULT]:
                self.generate_and_build_test()

            if self.BUILD_MODE == BuildMode.PRODUCE:
                continue

            logger.info("Running perf test for run type: {}", run_type.name)
            for run_index in range(run_count):
                self.run_elf_files()
                self.wait_for_tensix_operations_finished()

                meta = Profiler._get_meta(self.test_name, self.variant_id)
                buffer_data = [
                    read_words_from_device(
                        self.TENSIX_LOCATION,
                        addr,
                        word_count=self.THREAD_PERFORMANCE_DATA_BUFFER_LENGTH,
                    )
                    for addr in self.THREAD_PERFORMANCE_DATA_BUFFER
                ]
                profiler_data = Profiler._parse_buffers(buffer_data, meta)
                profiler_data.df["run_index"] = run_index
                runs.append(profiler_data)

            get_stats = Profiler.STATS_FUNCTION[run_type]
            all_results.append(get_stats(ProfilerData.concat(runs)))

        if self.BUILD_MODE != BuildMode.PRODUCE and all_results:
            results = reduce(
                lambda left, right: pd.merge(
                    left, right, on="marker", how="outer", validate="1:1"
                ),
                all_results,
            )
            results["test_name"] = self.global_config.test_name
            results["loop_factor"] = self.global_config.loop_factor
            perf_report.append(results)
            logger.info("Perf results:\n{}", results)

            csv_prefix = f"{self.global_config.test_name}_fused_test"
            perf_report.dump_csv(f"{csv_prefix}.{worker_id}.csv")
            perf_report.post_process()
            perf_report.dump_csv(f"{csv_prefix}.{worker_id}.post.csv")

    def run_regular_test(self):
        """Run functional test: generate, build, write inputs to L1, execute kernel, read outputs and verify against golden."""

        from .fused_generator import FUSED_TESTS_DIR
        from .fused_golden import FusedGolden

        if self.STIMULI_MODE == StimuliMode.GENERATE_ONLY:
            pytest.skip(self.SKIP_JUST_FOR_STIMULI_MARKER)

        self.test_name = FUSED_TESTS_DIR / f"{self.global_config.test_name}.cpp"

        self.generate_variant_hash()

        self.operand_registry.allocate_l1_addresses()

        if self.BUILD_MODE in [BuildMode.PRODUCE, BuildMode.DEFAULT]:
            self.generate_and_build_test()

        if self.BUILD_MODE == BuildMode.PRODUCE:
            pytest.skip(self.SKIP_JUST_FOR_COMPILE_MARKER)

        self.operand_registry.write_inputs_to_l1(self.TENSIX_LOCATION)

        self.run_elf_files()
        self.wait_for_tensix_operations_finished()
        self.operand_registry.read_outputs_from_l1(self.TENSIX_LOCATION)
        golden = FusedGolden()
        assert golden.check_pipeline(self)
