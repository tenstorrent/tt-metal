# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from functools import reduce
from typing import List

import pandas as pd
from helpers.device import (
    collect_pipeline_results,
    write_pipeline_operands_to_l1,
)
from ttexalens.tt_exalens_lib import read_words_from_device

from .chip_architecture import ChipArchitecture, get_chip_architecture
from .data_format_inference import data_formats, is_format_combination_outlier
from .device import wait_for_tensix_operations_finished
from .fused_operation import FusedOperation
from .llk_params import DestAccumulation, DestSync, PerfRunType
from .logger import logger
from .perf import PerfReport
from .profiler import Profiler, ProfilerData
from .test_config import ProfilerBuild, TestConfig, TestMode


@dataclass
class GlobalConfig:
    test_name: str = "fused_test"
    architecture: ChipArchitecture = None
    dest_acc: DestAccumulation = DestAccumulation.No
    regenerate_cpp: bool = False
    profiler_enabled: bool = False
    perf_run_type: PerfRunType = None
    loop_factor: int = 16


@dataclass
class FuserConfig:
    pipeline: List[FusedOperation]
    global_config: GlobalConfig

    def __post_init__(self):
        if self.global_config.architecture is None:
            self.global_config.architecture = get_chip_architecture()

        for operation in self.pipeline:
            if is_format_combination_outlier(
                operation.src_a.data_format,
                operation.output.data_format,
                self.global_config.dest_acc,
                None,  # No src_b in fuser operations yet
            ):
                raise ValueError(
                    f"Dest Accumulation must be enabled for {operation.src_a.data_format} input and {operation.output.data_format} output"
                )

        num_stages = len(self.pipeline)

        for i, operation in enumerate(self.pipeline, start=1):
            formats_config = data_formats(
                input_format=operation.src_a.data_format,
                output_format=operation.output.data_format,
                is_fp32_dest_acc_en=self.global_config.dest_acc,
                num_iterations=1,
                unpacking_to_dest=operation.unpack_to_dest,
                chip_arch=get_chip_architecture(),
                disable_format_inference=False,
            )[0]

            operation.unpack_a_in = formats_config.unpack_A_src
            operation.unpack_a_out = formats_config.unpack_A_dst
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

            if (
                self.global_config.architecture == ChipArchitecture.BLACKHOLE
                and operation.math.bh_unpack_tilize_check()
            ):
                raise ValueError(
                    "Cannot fuse UnpackerTilizeA and other unpackers inside one l1-to-l1 run on Blackhole"
                )

    def create_test_config(self, cpp_path, profiler_enabled: bool) -> TestConfig:
        return TestConfig(
            test_name=cpp_path,
            profiler_build=ProfilerBuild.Yes if profiler_enabled else ProfilerBuild.No,
            skip_build_header=True,
        )

    def generate_and_build_test(self, cpp_path, test_config: TestConfig):
        from .fused_generator import FusedKernelGenerator

        if TestConfig.MODE != TestMode.CONSUME:
            code_generator = FusedKernelGenerator(self)
            code_generator.write_kernel(cpp_path, self.global_config.regenerate_cpp)

        test_config.generate_variant_hash()

        if TestConfig.MODE != TestMode.CONSUME:
            test_config.build_elfs()

    def run_perf_test(self, worker_id: str, location: str, run_count: int = 2):
        from .fused_generator import FUSED_TESTS_DIR

        self.global_config.profiler_enabled = True
        write_pipeline_operands_to_l1(self.pipeline)

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

            cpp_path = (
                FUSED_TESTS_DIR / f"{self.global_config.test_name}_{run_type.name}.cpp"
            )

            test_config = self.create_test_config(cpp_path, profiler_enabled=True)
            self.generate_and_build_test(cpp_path, test_config)

            if TestConfig.MODE == TestMode.PRODUCE:
                continue

            logger.info("Running perf test for run type: {}", run_type.name)
            for run_index in range(run_count):
                elfs = test_config.run_elf_files(location)
                wait_for_tensix_operations_finished(elfs, location)

                meta = Profiler._get_meta(test_config.test_name, test_config.variant_id)
                buffer_data = [
                    read_words_from_device(
                        addr=addr,
                        word_count=TestConfig.THREAD_PERFORMANCE_DATA_BUFFER_LENGTH,
                        location=location,
                    )
                    for addr in TestConfig.THREAD_PERFORMANCE_DATA_BUFFER
                ]
                profiler_data = Profiler._parse_buffers(buffer_data, meta)
                profiler_data.df["run_index"] = run_index
                runs.append(profiler_data)

            get_stats = Profiler.STATS_FUNCTION[run_type]
            all_results.append(get_stats(ProfilerData.concat(runs)))

        if TestConfig.MODE != TestMode.PRODUCE and all_results:
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

    def run_regular_test(self, location: str):
        from .fused_generator import FUSED_TESTS_DIR
        from .fused_golden import FusedGolden

        write_pipeline_operands_to_l1(self.pipeline, location)

        cpp_path = FUSED_TESTS_DIR / f"{self.global_config.test_name}.cpp"

        test_config = self.create_test_config(cpp_path, profiler_enabled=False)
        self.generate_and_build_test(cpp_path, test_config)

        if TestConfig.MODE == TestMode.PRODUCE:
            return

        elfs = test_config.run_elf_files(location)
        wait_for_tensix_operations_finished(elfs, location)
        collect_pipeline_results(self.pipeline)
        golden = FusedGolden()
        assert golden.check_pipeline(self)
