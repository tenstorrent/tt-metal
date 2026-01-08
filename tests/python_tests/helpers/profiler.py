# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import re
import shutil
from dataclasses import dataclass, fields
from enum import Enum
from hashlib import sha256
from typing import ClassVar

import pandas as pd
import pytest
from ttexalens.tt_exalens_lib import read_words_from_device

from .device import BootMode, wait_for_tensix_operations_finished
from .format_config import FormatConfig
from .llk_params import DestAccumulation, PerfRunType
from .perf import PerfReport
from .stimuli_config import StimuliConfig
from .test_config import ProfilerBuild, TestConfig, TestMode
from .test_variant_parameters import PERF_RUN_TYPE, RuntimeParameter, TemplateParameter


@dataclass
class ProfilerFullMarker:
    marker: str
    file: str
    line: int
    id: int


class ProfilerData:
    """
    Used to query the underlying Pandas DataFrame

    All queries are applied to the underlying DataFrame lazily, when the user requests the DataFrame

    There are two views of the underlying DataFrame:
    - The raw event view (self.raw()), which is meant to make manipulation in code easier
    - The profiler view (self.frame()), which is meant to improve readability when manually reviewing the data

    The underlying data is stored in the raw event view, so requesting the profiler view has slight overhead

    The raw event view:
    - Has three entry types: TIMESTAMP, ZONE_START, ZONE_END
    - Data from each thead is concatenated together (all UNPACK -> MATH -> PACK)
    - Each ZONE_START entry is immediately followed by its corresponding ZONE_END entry.
    - There is no "duration" column included.

    The profiler view:
    - Has two entry types: TIMESTAMP, ZONE
    - Entries are ordered by timestamp, even across threads
    - There is a "duration" column included.
    - The TIMESTAMP entries have duration = pd.NA
    - The ZONE entries have duration = ZONE_END - ZONE_START
    - The ZONE entries have timestamp = ZONE_START
    """

    @staticmethod
    def concat(runs: list["ProfilerData"]) -> "ProfilerData":
        raw_data = [run.raw() for run in runs]
        return ProfilerData(pd.concat(raw_data, ignore_index=True))

    def __init__(self, df: pd.DataFrame, mask: pd.Series | None = None):
        self.df = df
        self.mask = mask if mask is not None else pd.Series(True, index=df.index)

    def _apply_mask(self):
        self.df = self.df[self.mask]
        self.mask = pd.Series(True, index=self.df.index)

    def _assert_zones_valid(
        self, start_entries: pd.DataFrame, end_entries: pd.DataFrame
    ) -> None:
        """
        Verify that the start and end entries are in the correct order
        """

        if len(start_entries) != len(end_entries):
            raise AssertionError("Number of start and end entries do not match")

        start_entries = start_entries[["thread", "marker_id"]].reset_index(drop=True)
        end_entries = end_entries[["thread", "marker_id"]].reset_index(drop=True)

        if not start_entries.equals(end_entries):
            raise AssertionError("Zone START and END entries don't match")

    def raw(self) -> pd.DataFrame:
        """Returns the raw event view of the underlying data"""

        # Apply the mask to the underlying DataFrame
        self._apply_mask()

        start_entries = self.df[self.df["type"] == "ZONE_START"]
        end_entries = self.df[self.df["type"] == "ZONE_END"]

        self._assert_zones_valid(start_entries, end_entries)

        return self.df

    def _post_profiler_view(self) -> pd.DataFrame:
        """
        Returns the profiler view of the underlying data

        This view consists of TIMESTAMP and ZONE entries
        """

        timestamp_entries = self.df[self.df["type"] == "TIMESTAMP"].copy()
        timestamp_entries["duration"] = pd.NA

        start_entries = self.df[self.df["type"] == "ZONE_START"].reset_index(drop=True)
        end_entries = self.df[self.df["type"] == "ZONE_END"].reset_index(drop=True)

        self._assert_zones_valid(start_entries, end_entries)

        zone_entries = start_entries.copy()

        zone_entries["type"] = "ZONE"
        zone_entries["duration"] = end_entries["timestamp"] - start_entries["timestamp"]

        result = (
            pd.concat([timestamp_entries, zone_entries], ignore_index=True)
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        return result[
            [
                "thread",
                "type",
                "marker",
                "timestamp",
                "duration",
                "data",
                "marker_id",
                "file",
                "line",
            ]
        ]

    def frame(self) -> pd.DataFrame:
        """Return the underlying DataFrame while"""

        # Apply the mask to the underlying DataFrame
        self._apply_mask()

        return self._post_profiler_view()

    # Filter by thread
    def unpack(self) -> "ProfilerData":
        """Filter: Unpack thread data"""
        return ProfilerData(self.df, self.mask & (self.df["thread"] == "unpack"))

    def math(self) -> "ProfilerData":
        """Filter: Math thread data"""
        return ProfilerData(self.df, self.mask & (self.df["thread"] == "math"))

    def pack(self) -> "ProfilerData":
        """Filter: Pack thread data"""
        return ProfilerData(self.df, self.mask & (self.df["thread"] == "pack"))

    # Filter by type
    def zones(self) -> "ProfilerData":
        """Filter: Profiler zones"""
        zone_filter = (self.df["type"] == "ZONE_START") | (
            self.df["type"] == "ZONE_END"
        )
        return ProfilerData(self.df, self.mask & zone_filter)

    def timestamps(self) -> "ProfilerData":
        """Filter: Profiler timestamps"""
        return ProfilerData(self.df, self.mask & (self.df["type"] == "TIMESTAMP"))

    # Filter by marker
    def marker(self, marker: str) -> "ProfilerData":
        """Filter: Marker"""
        return ProfilerData(self.df, self.mask & (self.df["marker"] == marker))

    def __str__(self):
        return f"{self.raw()}"


def _stats_timings(perf_data: pd.DataFrame) -> pd.DataFrame:
    # dont aggregate marker column
    timings = perf_data.columns.drop("marker")
    result = perf_data.groupby("marker", as_index=False)[timings].agg(["mean", "std"])

    columns = ["marker"]
    columns += [f"{stat}({col})" for col in timings for stat in ["mean", "std"]]

    result.columns = columns
    return result


def _stats_l1_to_l1(data: ProfilerData) -> pd.Series:
    groups = data.zones().raw().groupby(["marker"])

    timings = []
    for (marker,), group in groups:
        unpack_start = group[
            (group["thread"] == "unpack") & (group["type"] == "ZONE_START")
        ].reset_index(drop=True)

        pack_end = group[
            (group["thread"] == "pack") & (group["type"] == "ZONE_END")
        ].reset_index(drop=True)

        if len(unpack_start) == 0 or len(pack_end) == 0:
            raise ValueError(
                "Zone must be captured on both unpack and pack for L1_TO_L1 to work properly"
            )

        if len(unpack_start) != len(pack_end):
            raise ValueError(
                f"Unpack and pack must be paired properly for L1_TO_L1 to work properly"
            )

        durations = pack_end["timestamp"] - unpack_start["timestamp"]

        marker_timings = pd.DataFrame(
            {
                "marker": marker,
                PerfRunType.L1_TO_L1.name: durations,
            }
        )
        timings.append(marker_timings)

    return _stats_timings(pd.concat(timings, ignore_index=True))


def _stats_thread(stat: str, raw_thread: pd.DataFrame) -> pd.DataFrame:
    start_entries = raw_thread[(raw_thread["type"] == "ZONE_START")].reset_index(
        drop=True
    )

    end_entries = raw_thread[(raw_thread["type"] == "ZONE_END")].reset_index(drop=True)

    if len(start_entries) != len(end_entries):
        raise ValueError(
            f"Mismatched start/end zones: {len(start_entries)} != {len(end_entries)}"
        )

    timings = pd.DataFrame(
        {
            "marker": start_entries["marker"],
            stat: end_entries["timestamp"] - start_entries["timestamp"],
        }
    )

    return _stats_timings(timings)


def _stats_unpack_isolate(data: ProfilerData) -> pd.DataFrame:
    return _stats_thread(PerfRunType.UNPACK_ISOLATE.name, data.unpack().raw())


def _stats_math_isolate(data: ProfilerData) -> pd.DataFrame:
    return _stats_thread(PerfRunType.MATH_ISOLATE.name, data.math().raw())


def _stats_pack_isolate(data: ProfilerData) -> pd.DataFrame:
    return _stats_thread(PerfRunType.PACK_ISOLATE.name, data.pack().raw())


def _stats_l1_congestion(data: ProfilerData) -> pd.DataFrame:
    stats = [
        _stats_thread(f"{PerfRunType.L1_CONGESTION.name}[UNPACK]", data.unpack().raw()),
        _stats_thread(f"{PerfRunType.L1_CONGESTION.name}[PACK]", data.pack().raw()),
    ]

    return pd.concat(stats, ignore_index=True)


class EntryType(Enum):
    TIMESTAMP = 0b1000
    TIMESTAMP_DATA = 0b1001
    ZONE_START = 0b1010
    ZONE_END = 0b1011


class ProfilerConfig(TestConfig):
    # === STATIC VARIABLES ===

    TEST_COUNTER: ClassVar[int] = 0

    META_PATTERN = re.compile(
        r"(?P<full_marker>LLK_PROFILER:(?P<file>[^:]+):(?P<line>\d+):(?P<marker>[^']+))",
    )

    # === Addresses ===
    BUFFER_LENGTH = 0x400
    THREAD_BUFFER = [
        0x16B000,  # Unpack
        0x16C000,  # Math
        0x16D000,  # Pack
    ]

    # === Bit masks ===
    ENTRY_TYPE_SHAMT = 28
    ENTRY_ID_SHAMT = ENTRY_TYPE_SHAMT - 16

    ENTRY_TYPE_MASK = 0xF << ENTRY_TYPE_SHAMT
    ENTRY_ID_MASK = 0xFFFF << ENTRY_ID_SHAMT
    ENTRY_TIME_HIGH_MASK = 0xFFF

    ENTRY_EXISTS_BIT = 0b1000 << ENTRY_TYPE_SHAMT

    # === Stats functions ===
    STATS_FUNCTION = {
        PerfRunType.L1_TO_L1: _stats_l1_to_l1,
        PerfRunType.UNPACK_ISOLATE: _stats_unpack_isolate,
        PerfRunType.MATH_ISOLATE: _stats_math_isolate,
        PerfRunType.PACK_ISOLATE: _stats_pack_isolate,
        PerfRunType.L1_CONGESTION: _stats_l1_congestion,
    }
    SUPPORTED_RUNS = STATS_FUNCTION.keys()

    @staticmethod
    def _hash_meta(s: str) -> int:
        hash32 = 2166136261
        for c in s.encode("ascii"):
            hash32 ^= c
            hash32 = (
                hash32 * 16777619
            ) & 0xFFFFFFFF  # simulate 32-bit unsigned overflow
        return (hash32 ^ (hash32 >> 16)) & 0xFFFF  # fold to 16 bits

    @staticmethod
    def _assert_no_collision(messages, message):
        existing = messages.get(message.id)
        if existing is not None and existing != message:
            raise AssertionError(f'Hash collision between "{message}" and "{existing}"')

    @staticmethod
    def _parse_meta(meta: str):
        if expr := ProfilerConfig.META_PATTERN.search(meta):
            groups = expr.groupdict()
            return ProfilerFullMarker(
                marker=groups["marker"],
                file=groups["file"],
                line=int(groups["line"]),
                id=ProfilerConfig._hash_meta(groups["full_marker"]),
            )
        else:
            return None

    @staticmethod
    def _get_meta(testname: str, variant_id: str) -> dict[id, ProfilerFullMarker]:
        profiler_data_dir = ProfilerConfig.PROFILER_META / testname / variant_id
        metadata = {}
        for thread in ProfilerConfig.KERNEL_COMPONENTS:
            file = profiler_data_dir / f"{thread}.meta.bin"
            if not file.exists():
                continue
            with open(file, "rb") as f:
                binary = f.read()
                strings = [s.decode("ascii") for s in binary.split(b"\0")]
                for s in strings:
                    if marker := ProfilerConfig._parse_meta(s):
                        ProfilerConfig._assert_no_collision(metadata, marker)
                        metadata[marker.id] = marker

        return metadata

    @staticmethod
    def _get_marker_id(
        metadata: dict, marker_name: str, file_suffix: str, line: int
    ) -> int:
        """Look up marker ID from metadata by marker name, file suffix, and line number.
        This provides stable marker ID lookup regardless of build environment paths."""
        for marker in metadata.values():
            if (
                marker.marker == marker_name
                and marker.file.endswith(file_suffix)
                and marker.line == line
            ):
                return marker.id
        raise ValueError(
            f"Marker '{marker_name}' not found in metadata (file ending with '{file_suffix}', line {line})"
        )

    @staticmethod
    def _dataframe(rows: list[dict] | None = None) -> pd.DataFrame:
        # Define the schema
        schema = {
            "thread": pd.CategoricalDtype(categories=ProfilerConfig.KERNEL_COMPONENTS),
            "type": pd.CategoricalDtype(
                categories=["TIMESTAMP", "ZONE_START", "ZONE_END"]
            ),
            "marker": "string",
            "timestamp": "int64",
            "data": "Int64",  # nullable
            "marker_id": "int32",
            "file": "string",
            "line": "int32",
        }

        return pd.DataFrame(rows or [], columns=schema.keys()).astype(schema)

    @staticmethod
    def _parse_buffers(buffers: list, profiler_meta: dict) -> pd.DataFrame:
        marker_rows = []
        # Parse each thread and append to the DataFrame
        for thread, buffer in zip(ProfilerConfig.KERNEL_COMPONENTS, buffers):
            marker_rows.extend(
                ProfilerConfig._parse_thread(thread, buffer, profiler_meta)
            )

        df = ProfilerConfig._dataframe(marker_rows)
        return ProfilerData(df)

    @staticmethod
    def _parse_thread(thread, words, profiler_meta) -> list[dict]:
        rows = []
        zone_stack = []

        word_stream = iter(words)
        for word in word_stream:
            if not (word & ProfilerConfig.ENTRY_EXISTS_BIT):
                break

            type = (
                word & ProfilerConfig.ENTRY_TYPE_MASK
            ) >> ProfilerConfig.ENTRY_TYPE_SHAMT
            marker_id = (
                word & ProfilerConfig.ENTRY_ID_MASK
            ) >> ProfilerConfig.ENTRY_ID_SHAMT

            try:
                marker = profiler_meta[marker_id]
            except KeyError:
                raise AssertionError(
                    f"Marker with ID {marker_id} not found in profiler metadata"
                )

            timestamp_high = word & ProfilerConfig.ENTRY_TIME_HIGH_MASK
            timestamp_low = next(word_stream)
            timestamp = (timestamp_high << 32) | timestamp_low

            match EntryType(type):
                case EntryType.TIMESTAMP:
                    rows.append(
                        ProfilerConfig._row(
                            thread, "TIMESTAMP", marker, timestamp, pd.NA
                        )
                    )

                case EntryType.TIMESTAMP_DATA:
                    data_high = next(word_stream)
                    data_low = next(word_stream)
                    data = (data_high << 32) | data_low
                    rows.append(
                        ProfilerConfig._row(
                            thread, "TIMESTAMP", marker, timestamp, data
                        )
                    )

                case EntryType.ZONE_START:
                    zone_stack.append(
                        ProfilerConfig._row(
                            thread, "ZONE_START", marker, timestamp, pd.NA
                        )
                    )

                case EntryType.ZONE_END:
                    rows.append(zone_stack.pop())  # Pop the ZONE_START pair
                    rows.append(
                        ProfilerConfig._row(
                            thread, "ZONE_END", marker, timestamp, pd.NA
                        )
                    )

        return rows

    @staticmethod
    def _row(thread, type, marker, timestamp, data) -> dict:
        return {
            "thread": thread,
            "type": type,
            "marker": marker.marker,
            "timestamp": timestamp,
            "data": data,
            "marker_id": marker.id,
            "file": marker.file,
            "line": marker.line,
        }

    def __init__(
        self,
        test_name: str,
        formats: FormatConfig = None,
        run_types: list[PerfRunType] = [],
        templates: list[TemplateParameter] = [],
        runtimes: list[RuntimeParameter] = [],
        variant_stimuli: StimuliConfig = None,
        unpack_to_dest=False,
        disable_format_inference=False,
        dest_acc=DestAccumulation.No,
    ):
        super().__init__(
            test_name,
            formats,
            templates,
            runtimes,
            variant_stimuli,
            BootMode.DEFAULT,
            ProfilerBuild.Yes,
            1,  # L1_2_L1s
            unpack_to_dest,
            disable_format_inference,
            dest_acc,
        )

        self.passed_templates = templates
        self.passed_runtimes = runtimes
        self.current_run_type = None

        unsupported = set(run_types) - ProfilerConfig.SUPPORTED_RUNS
        assert not unsupported, (
            f"ERROR: run_types {unsupported} not implemented. "
            f"Supported: {set(ProfilerConfig.SUPPORTED_RUNS)}"
        )

        self.run_configs = [
            (
                templates.copy() + [PERF_RUN_TYPE(run_type)],
                runtimes.copy(),
                run_type,
            )
            for run_type in run_types
        ]

    def get_data(self, location: str = "0,0") -> pd.DataFrame:
        meta = ProfilerConfig._get_meta(self.test_name, self.variant_id)
        buffer_data = [
            read_words_from_device(
                addr=buffer_address,
                word_count=ProfilerConfig.BUFFER_LENGTH,
                location=location,
            )
            for buffer_address in ProfilerConfig.THREAD_BUFFER
        ]

        return ProfilerConfig._parse_buffers(buffer_data, meta)

    def generate_variant_hash(self):
        NON_COMPILATION_ARGUMENTS = [
            "variant_stimuli",
            "run_configs",
            "variant_id",
            "runtime_params_struct",
            "runtime_format",
            "runtimes",
        ]
        temp_str = [
            str(value)
            for field_name, value in self.__dict__.items()
            if field_name not in NON_COMPILATION_ARGUMENTS
        ]

        self.variant_id = sha256(str(" | ".join(temp_str)).encode()).hexdigest()

    @staticmethod
    def _dataclass_names(parent, obj):
        """Provides the **names** of the columns for the report"""
        return [f.name for f in fields(obj)]

    @staticmethod
    def _dataclass_values(obj):
        """Provides the **values** of the columns for the report"""
        return [getattr(obj, f.name) for f in fields(obj)]

    def run(
        self,
        perf_report: PerfReport,
        run_count=2,
        location="0,0",
        delete_artefacts: bool = False,
    ):
        ProfilerConfig.TEST_COUNTER += 1
        results = []

        if TestConfig.MODE in [TestMode.PRODUCE, TestMode.DEFAULT]:
            for templates, runtimes, run_type in self.run_configs:
                self.current_run_type = run_type
                self.templates = templates
                self.runtimes = runtimes
                self.generate_variant_hash()
                self.build_elfs()

        if TestConfig.MODE == TestMode.PRODUCE:
            pytest.skip(TestConfig.SKIP_JUST_FOR_COMPILE_MARKER)

        for templates, runtimes, run_type in self.run_configs:
            self.current_run_type = run_type
            self.templates = templates
            self.runtimes = runtimes
            self.generate_variant_hash()
            runs = []
            for _ in range(run_count):
                self.write_runtimes_to_L1(location)
                elfs = self.run_elf_files(location)
                wait_for_tensix_operations_finished(elfs, location)

                profiler_data = self.get_data(location)

                runs.append(profiler_data)

            get_stats = ProfilerConfig.STATS_FUNCTION[run_type]
            results.append(get_stats(ProfilerData.concat(runs)))

            if delete_artefacts:
                shutil.rmtree(
                    TestConfig.ARTEFACTS_DIR / self.test_name / self.variant_id
                )

        results = pd.concat(results, ignore_index=True)
        run_results = results.groupby("marker").first().reset_index()

        names, values = [], []
        for param in self.passed_templates:
            names.extend(ProfilerConfig._dataclass_names(type(param).__name__, param))
            values.extend(ProfilerConfig._dataclass_values(param))

        for param in self.passed_runtimes:
            names.extend(ProfilerConfig._dataclass_names(type(param).__name__, param))
            values.extend(ProfilerConfig._dataclass_values(param))

        sweep = pd.DataFrame([values], columns=names)
        combined = sweep.merge(run_results, how="cross")

        perf_report.append(combined)
