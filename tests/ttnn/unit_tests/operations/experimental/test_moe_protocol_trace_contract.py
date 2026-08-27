# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
PRODUCER_HOST = ROOT / "ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/device/moe_compute_program_factory.cpp"
CONSUMER_HOST = ROOT / (
    "ttnn/cpp/ttnn/operations/experimental/ccl/moe/selective_reduce_combine/device/"
    "selective_reduce_combine_program_factory.cpp"
)
PRODUCER_KERNEL = ROOT / "ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/device/kernels/dm1.cpp"
CONSUMER_KERNEL = ROOT / (
    "ttnn/cpp/ttnn/operations/experimental/ccl/moe/selective_reduce_combine/device/kernels/dataflow/writer.cpp"
)


def test_protocol_trace_is_exact_opt_in_on_both_participants():
    for path in (PRODUCER_HOST, CONSUMER_HOST):
        source = path.read_text()
        assert 'std::getenv("TT_MOE_PROTOCOL_TRACE")' in source
        assert 'std::string_view(trace) == "1"' in source
        assert '["TT_MOE_PROTOCOL_TRACE"] = "1"' in source


def test_protocol_trace_calls_are_compiled_behind_define():
    for path in (PRODUCER_KERNEL, CONSUMER_KERNEL):
        source = path.read_text()
        assert "#ifdef TT_MOE_PROTOCOL_TRACE" in source
        assert "#define TT_MOE_PROTOCOL_DPRINT(...) DPRINT(__VA_ARGS__)" in source
        assert "#define TT_MOE_PROTOCOL_DPRINT(...)" in source


def test_protocol_trace_covers_semaphore_and_fabric_boundaries():
    producer = PRODUCER_KERNEL.read_text()
    consumer = CONSUMER_KERNEL.read_text()
    for phase in ("QZP WAIT", "QZP ACK", "QZP START", "QZP FINAL_WAIT", "QZP FINAL_ACK"):
        assert phase in producer
    for phase in (
        "QZC WAIT",
        "QZC ACK",
        "QZC CREDIT_BEGIN",
        "QZC CREDIT_DONE",
        "QZC MUX_WAIT",
        "QZC MUX_ACK",
        "QZC GLOBAL_WAIT",
        "QZC GLOBAL_ACK",
    ):
        assert phase in consumer
