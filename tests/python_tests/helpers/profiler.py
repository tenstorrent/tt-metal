# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pandas as pd
from helpers.chip_architecture import get_chip_architecture
from ttexalens.tt_exalens_lib import read_words_from_device


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
        return ProfilerData(self.df, self.mask & (self.df["thread"] == "UNPACK"))

    def math(self) -> "ProfilerData":
        """Filter: Math thread data"""
        return ProfilerData(self.df, self.mask & (self.df["thread"] == "MATH"))

    def pack(self) -> "ProfilerData":
        """Filter: Pack thread data"""
        return ProfilerData(self.df, self.mask & (self.df["thread"] == "PACK"))

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


class Profiler:

    META_PATTERN = re.compile(
        r"(?P<full_marker>LLK_PROFILER:(?P<file>[^:]+):(?P<line>\d+):(?P<marker>[^']+))",
    )

    BUFFER_LENGTH = 0x400
    THREAD_BUFFER = [
        0x16B000,  # Unpack
        0x16C000,  # Math
        0x16D000,  # Pack
    ]

    ENTRY_TYPE_SHAMT = 28
    ENTRY_ID_SHAMT = ENTRY_TYPE_SHAMT - 16

    ENTRY_TYPE_MASK = 0xF << ENTRY_TYPE_SHAMT
    ENTRY_ID_MASK = 0xFFFF << ENTRY_ID_SHAMT
    ENTRY_TIME_HIGH_MASK = 0xFFF

    ENTRY_EXISTS_BIT = 0b1000 << ENTRY_TYPE_SHAMT

    class EntryType(Enum):
        TIMESTAMP = 0b1000
        TIMESTAMP_DATA = 0b1001
        ZONE_START = 0b1010
        ZONE_END = 0b1011

    @staticmethod
    def dump_csv(profiler_data, filename: str = "profiler_data.csv") -> None:
        llk_home = Path(os.environ.get("LLK_HOME"))
        output_path = llk_home / "tests" / "build" / filename

        output_path.parent.mkdir(parents=True, exist_ok=True)
        profiler_data.to_csv(output_path, index=False)

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
        expr = Profiler.META_PATTERN.search(meta)
        if expr is None:
            return None

        groups = expr.groupdict()
        return ProfilerFullMarker(
            marker=groups["marker"],
            file=groups["file"],
            line=int(groups["line"]),
            id=Profiler._hash_meta(groups["full_marker"]),
        )

    @staticmethod
    def _get_meta(testname: str) -> dict[id, ProfilerFullMarker]:
        chip_arch = get_chip_architecture()
        llk_home = Path(os.environ.get("LLK_HOME"))

        profiler_dir = (
            llk_home
            / "tests"
            / "build"
            / chip_arch.value
            / "tests"
            / testname
            / "profiler"
        )

        files = [
            profiler_dir / "unpack.meta.bin",
            profiler_dir / "math.meta.bin",
            profiler_dir / "pack.meta.bin",
        ]

        meta = {}
        for file in files:
            if not file.exists():
                continue
            with open(file, "rb") as f:
                binary = f.read()
                strings = [s.decode("ascii") for s in binary.split(b"\0")]
                for s in strings:
                    marker = Profiler._parse_meta(s)
                    if marker:
                        Profiler._assert_no_collision(meta, marker)
                        meta[marker.id] = marker
        return meta

    @staticmethod
    def get_data(testname: str) -> pd.DataFrame:
        meta = Profiler._get_meta(testname)
        return Profiler._parse_buffers(Profiler._load_buffers(), meta)

    @staticmethod
    def _load_buffers(location="0,0", word_count=BUFFER_LENGTH) -> list[list[int]]:
        """Load profiler buffers from device memory for each thread."""
        return [
            read_words_from_device(
                location=location, addr=buffer_address, word_count=word_count
            )
            for buffer_address in Profiler.THREAD_BUFFER
        ]

    @staticmethod
    def _dataframe(rows: list[dict] | None = None) -> pd.DataFrame:
        # Define the schema
        schema = {
            "thread": pd.CategoricalDtype(categories=["UNPACK", "MATH", "PACK"]),
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
    def _parse_buffers(buffers, profiler_meta) -> pd.DataFrame:
        THREADS = ["UNPACK", "MATH", "PACK"]

        # Parse each thread and append to the DataFrame
        threads = [
            parsed_thread
            for thread, buffer in zip(THREADS, buffers)
            for parsed_thread in Profiler._parse_thread(thread, buffer, profiler_meta)
        ]

        df = Profiler._dataframe(threads)
        return ProfilerData(df)

    @staticmethod
    def _parse_thread(thread, words, profiler_meta) -> list[dict]:
        rows = []
        zone_stack = []

        word_stream = iter(words)
        for word in word_stream:
            if not (word & Profiler.ENTRY_EXISTS_BIT):
                break

            type = (word & Profiler.ENTRY_TYPE_MASK) >> Profiler.ENTRY_TYPE_SHAMT
            marker_id = (word & Profiler.ENTRY_ID_MASK) >> Profiler.ENTRY_ID_SHAMT

            try:
                marker = profiler_meta[marker_id]
            except KeyError:
                raise AssertionError(
                    f"Marker with ID {marker_id} not found in profiler metadata"
                )

            timestamp_high = word & Profiler.ENTRY_TIME_HIGH_MASK
            timestamp_low = next(word_stream)
            timestamp = (timestamp_high << 32) | timestamp_low

            entry_type = Profiler.EntryType(type)
            match entry_type:
                case Profiler.EntryType.TIMESTAMP:
                    rows.append(
                        Profiler._row(thread, "TIMESTAMP", marker, timestamp, pd.NA)
                    )

                case Profiler.EntryType.TIMESTAMP_DATA:
                    data_high = next(word_stream)
                    data_low = next(word_stream)
                    data = (data_high << 32) | data_low
                    rows.append(
                        Profiler._row(thread, "TIMESTAMP", marker, timestamp, data)
                    )

                case Profiler.EntryType.ZONE_START:
                    zone_stack.append(
                        Profiler._row(thread, "ZONE_START", marker, timestamp, pd.NA)
                    )

                case Profiler.EntryType.ZONE_END:
                    rows.append(zone_stack.pop())  # Pop the ZONE_START pair
                    rows.append(
                        Profiler._row(thread, "ZONE_END", marker, timestamp, pd.NA)
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
