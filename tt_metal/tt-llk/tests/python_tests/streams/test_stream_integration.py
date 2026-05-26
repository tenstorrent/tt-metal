# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.format_config import DataFormat
from helpers.param_config import input_output_formats
from helpers.stream import Stream
from helpers.test_config import BuildMode, TestConfig
from helpers.test_variant_parameters import (
    HOST_IS_STREAM_CONSUMER,
    HOST_IS_STREAM_PRODUCER,
)
from ttexalens.tt_exalens_lib import read_word_from_device


def test_stream_risc_to_risc():
    execute_stream_workload(is_host_producer=False, is_host_consumer=False)


def test_stream_host_to_risc():
    execute_stream_workload(is_host_producer=True, is_host_consumer=False)


def test_stream_risc_to_host():
    execute_stream_workload(is_host_producer=False, is_host_consumer=True)


class PRNG:
    def __init__(self, seed: int):
        self.state = seed if seed else 1

    def next(self):
        self.state ^= (self.state << 13) & 0xFFFFFFFF
        self.state ^= self.state >> 17
        self.state ^= (self.state << 5) & 0xFFFFFFFF
        self.state &= 0xFFFFFFFF
        return self.state

    def byte(self):
        return self.next() & 0xFF


def host_producer(stream: Stream, packet_count: int, seed: int):
    random = PRNG(seed)
    for _ in range(packet_count):
        size = random.next() % 100 + 1
        packet = bytes(random.byte() for _ in range(size))
        stream.push(packet)


def host_consumer(stream: Stream, packet_count: int, seed: int):
    random = PRNG(seed)
    for _ in range(packet_count):
        size = random.next() % 100 + 1
        expected = bytes(random.byte() for _ in range(size))
        first = stream.peek()
        packet = stream.pop(size)

        if first is not None and first != expected[0]:
            raise ValueError("First byte of packet does not match expected")

        assert packet == expected


def execute_stream_workload(is_host_producer: bool, is_host_consumer: bool):

    STREAM_ADDRESS = 0x70000
    STREAM_DEPTH = 128
    DATA_SEED = 0xDEADBEEF
    PACKET_COUNT = 10

    if is_host_producer and is_host_consumer:
        raise ValueError("Host cannot be consumer and producer at the same time")

    formats = input_output_formats([DataFormat.Float16_b])[0]

    configuration = TestConfig(
        "sources/streams/stream_integration_test.cpp",
        formats,
        runtimes=[
            HOST_IS_STREAM_PRODUCER(is_host_producer),
            HOST_IS_STREAM_CONSUMER(is_host_consumer),
        ],
    )

    configuration.generate_variant_hash()

    if TestConfig.BUILD_MODE in [BuildMode.PRODUCE, BuildMode.DEFAULT]:
        configuration.build_elfs()

    if TestConfig.BUILD_MODE == BuildMode.PRODUCE:
        pytest.skip(TestConfig.SKIP_JUST_FOR_COMPILE_MARKER)

    stream = Stream(STREAM_ADDRESS, STREAM_DEPTH, TestConfig.TENSIX_LOCATION)
    stream.init()

    configuration.write_runtimes_to_L1()
    configuration.run_elf_files()

    if is_host_producer:
        host_producer(stream, PACKET_COUNT, DATA_SEED)
    if is_host_consumer:
        host_consumer(stream, PACKET_COUNT, DATA_SEED)

    configuration.wait_for_tensix_operations_finished()

    # read write and read pointer from the device memory using exalens and compare that they are equal (stream is empty)
    write_idx = read_word_from_device(TestConfig.TENSIX_LOCATION, STREAM_ADDRESS + 0)
    read_idx = read_word_from_device(TestConfig.TENSIX_LOCATION, STREAM_ADDRESS + 4)
    assert read_idx == write_idx, "Stream is not empty after the test is finished"
