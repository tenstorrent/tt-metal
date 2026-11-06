# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn

SHAPE = [1, 1, 32, 32]
# Each unrolled store adds at least one 4-byte instruction. 12,000 stores exceed the kernel-config buffer on every
# architecture (Blackhole's is 70,656 bytes), as in test_generic_op_preparation.py.
STORES_EXCEEDING_KERNEL_CONFIG_BUFFER = 12000


def _oversized_program():
    core = ttnn.CoreCoord(0, 0)
    runtime_args = ttnn.RuntimeArgs()
    runtime_args[0][0] = [0]
    kernel_source = (
        "#include <cstdint>\n"
        "void kernel_main() {\n"
        "    volatile uint32_t* sink = reinterpret_cast<volatile uint32_t*>(get_arg_val<uint32_t>(0));\n"
        f"#pragma GCC unroll {STORES_EXCEEDING_KERNEL_CONFIG_BUFFER}\n"
        f"    for (uint32_t i = 0; i < {STORES_EXCEEDING_KERNEL_CONFIG_BUFFER}; ++i) {{ sink[0] = i * 2654435761u; }}\n"
        "}\n"
    )
    return ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=kernel_source,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(core, core)]),
                compile_time_args=[],
                runtime_args=runtime_args,
                # BRISC avoids Wormhole's 16 KiB NCRISC code limit, so the program reaches the size check.
                config=ttnn.WriterConfigDescriptor(),
            )
        ],
        semaphores=[],
        cbs=[],
    )


def test_capture_closes_when_the_body_raises(expect_error, monkeypatch):
    begin_capture = ttnn._ttnn.jit_telemetry.begin_capture
    opened = []

    def record_begin():
        opened.append(begin_capture())
        return opened[-1]

    monkeypatch.setattr(ttnn._ttnn.jit_telemetry, "begin_capture", record_begin)

    with expect_error(ValueError, "boom"):
        with ttnn.jit_telemetry.capture():
            raise ValueError("boom")
    with expect_error(RuntimeError, "is not open"):
        ttnn._ttnn.jit_telemetry.end_capture(opened[0])


def test_capture_records_a_program_that_fails_the_size_check(device, expect_error):
    io_tensors = [
        ttnn.from_torch(
            torch.zeros(SHAPE, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
        ttnn.allocate_tensor_on_device(
            ttnn.Shape(SHAPE), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
        ),
    ]
    program = _oversized_program()
    with ttnn.jit_telemetry.capture() as cap:
        with expect_error(RuntimeError, "too large for kernel config buffer"):
            ttnn.generic_op(io_tensors, program)

    kernel_text = cap.stats["program_config_size.kernel_text.TENSIX"]
    total = cap.stats["program_config_size.total.TENSIX"]
    assert total["count"] == 1
    assert kernel_text["max"] > 4 * STORES_EXCEEDING_KERNEL_CONFIG_BUFFER
    assert total["max"] >= kernel_text["max"]
    # Headroom is recorded only for programs that fit.
    assert "program_config_size.headroom.TENSIX" not in cap.stats
