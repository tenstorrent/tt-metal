# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import csv
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from ttexalens.tt_exalens_lib import read_words_from_device

from helpers.test_config import ProfilerBuild, generate_make_command
from helpers.utils import run_shell_command


def _hash_profiler_message(s: str) -> int:
    hash32 = 2166136261
    for c in s.encode("ascii"):
        hash32 ^= c
        hash32 = (hash32 * 16777619) & 0xFFFFFFFF  # simulate 32-bit unsigned overflow
    return (hash32 ^ (hash32 >> 16)) & 0xFFFF  # fold to 16 bits


_PROFILER_PATTERN = re.compile(
    r"'#pragma message: (?P<full_marker>LLK_PROFILER:(?P<file>[^:]+):(?P<line>\d+):(?P<marker>[^']+))'",
)


@dataclass
class ProfilerFullMarker:
    marker: str
    file: str
    line: int
    id: int


def _parse_profiler_message(line: str):
    # ex. '#pragma message: LLK_PROFILER:sources/example.cpp:1337:MARKER'
    expr = _PROFILER_PATTERN.search(line)
    if expr is None:
        return None

    groups = expr.groupdict()
    return ProfilerFullMarker(
        marker=groups["marker"],
        file=groups["file"],
        line=int(groups["line"]),
        id=_hash_profiler_message(groups["full_marker"]),
    )


def _assert_no_collision(messages, message):
    """Inserts message if no collisions, raises otherwise"""
    existing = messages.get(message.id)
    if existing is not None and existing != message:
        raise AssertionError(f'Hash collision between "{message}" and "{existing}"')


def build_perf_test(test_config):
    make_cmd = generate_make_command(test_config, ProfilerBuild.Yes)
    result = run_shell_command(f"cd .. && {make_cmd}")

    lines = result.stderr.splitlines()
    messages = {}
    for line in lines:
        if message := _parse_profiler_message(line):
            _assert_no_collision(messages, message)
            messages[message.id] = message
    return messages


@dataclass
class ProfilerTimestamp:
    full_marker: ProfilerFullMarker
    timestamp: int
    data: Optional[int] = None


@dataclass
class ProfilerZoneScoped:
    full_marker: ProfilerFullMarker
    start: int
    end: int
    duration: int


class ProfilerData:

    BUFFER_LENGTH = 0x400
    THREAD_BUFFER = {
        "UNPACK": 0x16B000,
        "MATH": 0x16C000,
        "PACK": 0x16D000,
    }

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
        rows = []
        rows.append(
            [
                "thread",
                "type",
                "marker",
                "timestamp",
                "data",
                "marker_id",
                "file",
                "line",
            ]
        )

        for thread, entries in profiler_data.items():
            for entry in entries:
                full_marker = entry.full_marker

                if isinstance(entry, ProfilerTimestamp):
                    rows.append(
                        [
                            thread,
                            "TIMESTAMP",
                            full_marker.marker,
                            entry.timestamp,
                            entry.data,
                            full_marker.id,
                            full_marker.file,
                            full_marker.line,
                        ]
                    )
                elif isinstance(entry, ProfilerZoneScoped):
                    # Add both start and end rows
                    rows.append(
                        [
                            thread,
                            "ZONE_START",
                            full_marker.marker,
                            entry.start,
                            "",
                            full_marker.id,
                            full_marker.file,
                            full_marker.line,
                        ]
                    )
                    rows.append(
                        [
                            thread,
                            "ZONE_END",
                            full_marker.marker,
                            entry.end,
                            "",
                            full_marker.id,
                            full_marker.file,
                            full_marker.line,
                        ]
                    )

        output_path = Path("../build") / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

    @staticmethod
    def get(profiler_meta):
        return ProfilerData._parse_buffers(ProfilerData._load_buffers(), profiler_meta)

    @staticmethod
    def _load_buffers(core_loc="0,0", word_count=BUFFER_LENGTH):
        """Load profiler buffers from device memory for each thread."""
        return {
            thread: read_words_from_device(
                core_loc=core_loc, addr=buffer_address, word_count=word_count
            )
            for thread, buffer_address in ProfilerData.THREAD_BUFFER.items()
        }

    @staticmethod
    def _parse_buffers(buffers, profiler_meta):
        return {
            thread: ProfilerData._parse_thread(words, profiler_meta)
            for thread, words in buffers.items()
        }

    @staticmethod
    def _parse_thread(words, profiler_meta):
        thread = []
        zone_stack = []
        word_stream = iter(words)
        for word in word_stream:
            if not (word & ProfilerData.ENTRY_EXISTS_BIT):
                break

            type = (
                word & ProfilerData.ENTRY_TYPE_MASK
            ) >> ProfilerData.ENTRY_TYPE_SHAMT

            marker_id = (
                word & ProfilerData.ENTRY_ID_MASK
            ) >> ProfilerData.ENTRY_ID_SHAMT

            try:
                marker = profiler_meta[marker_id]
            except KeyError:
                raise AssertionError(
                    f"Marker with ID {marker_id} not found in profiler metadata"
                )

            timestamp_high = word & ProfilerData.ENTRY_TIME_HIGH_MASK
            timestamp_low = next(word_stream)
            timestamp = (timestamp_high << 32) | timestamp_low

            entry_type = ProfilerData.EntryType(type)
            match entry_type:
                case ProfilerData.EntryType.TIMESTAMP:
                    thread.append(ProfilerTimestamp(marker, timestamp))

                case ProfilerData.EntryType.TIMESTAMP_DATA:
                    data_high = next(word_stream)
                    data_low = next(word_stream)
                    data = (data_high << 32) | data_low
                    thread.append(ProfilerTimestamp(marker, timestamp, data))

                case ProfilerData.EntryType.ZONE_START:
                    zone_stack.append(timestamp)

                case ProfilerData.EntryType.ZONE_END:
                    end = timestamp
                    start = zone_stack.pop()
                    duration = end - start
                    thread.append(ProfilerZoneScoped(marker, start, end, duration))

        return thread
