# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F201: the fork-call capture records a generic program runner's program by what it is, never by where it is."""

import pytest

from models.demos.common.bringup.testing import fork_capture as FC

ttnn = pytest.importorskip("ttnn")


def _desc(addr: int, define: str = "1", ct_base: int = 0x40000):
    """One chip's program as a builder makes it: a reader taking an io tensor's address as a runtime arg, an inline
    compute kernel with a baked buffer address in its source, compile-time args with a baked address."""
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))])
    rt = ttnn.RuntimeArgs()
    rt.append(ttnn.CoreCoord(0, 0), [addr, 7])
    rt.append(ttnn.CoreCoord(1, 0), [addr + 2048, 7])
    reader = ttnn.KernelDescriptor(
        kernel_source="ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=[32, ct_base],
        defines=[("MODE", define)],
        runtime_args=rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=f"#define BUF {addr}u\n#define TILES 4\nvoid kernel_main() {{}}\n",
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cores,
        compile_time_args=[4],
        defines=[],
        runtime_args=ttnn.RuntimeArgs(),
        config=ttnn.ComputeConfigDescriptor(),
    )
    cb = ttnn.CBDescriptor(
        total_size=4096,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=2048)],
    )
    pd = ttnn.ProgramDescriptor(kernels=[reader, compute], semaphores=[], cbs=[cb])
    md = ttnn.MeshProgramDescriptor()
    md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(0, 0))] = pd
    return md


def test_the_same_program_at_two_addresses_has_one_summary():
    a = FC.program_summary(_desc(0x1A2B00))
    b = FC.program_summary(_desc(0x2C4E00, ct_base=0x60000))
    assert a == b and a["digest"] != "unknown"
    assert a["programs"] == 1 and any(k.startswith("inline:") for k in a["kernels"])


def test_a_different_define_is_another_program():
    assert (
        FC.program_summary(_desc(0x1A2B00, define="1"))["digest"]
        != FC.program_summary(_desc(0x1A2B00, define="2"))["digest"]
    )


def test_runner_calls_get_stable_signatures(monkeypatch):
    """prepare_program at two addresses -> one signature; run_prepared of it carries the same program summary."""
    monkeypatch.setattr(FC, "_CALLS", {})
    monkeypatch.setattr(FC, "_DEVICE", {"fabric_config": "x"})
    monkeypatch.setattr(FC, "_device", lambda: None)
    FC._record_runner("prepare_program", (_desc(0x1A2B00), []), {})
    FC._record_runner("prepare_program", (_desc(0x2C4E00), []), {})
    calls = list(FC._CALLS.values())
    assert len(calls) == 1 and calls[0]["count"] == 2
    p = object()
    FC._PREPARED[id(p)] = (p, FC.program_summary(_desc(0x1A2B00)))
    FC._record_runner("run_prepared", (p, []), {})
    run = [c for c in FC._CALLS.values() if c["op"] == "ttnn.bringup.run_prepared"]
    assert run and run[0]["args"][0]["digest"] == calls[0]["args"][0]["digest"]


def test_masking_keeps_shapes_and_drops_addresses():
    assert FC._mask("#define SE_EW 1442944u\nn = 5120; m = 0x20000; k = 65535") == (
        "#define SE_EW A\nn = 5120; m = A; k = 65535"
    )


def test_every_capture_file_the_gate_names_is_a_gate_output(tmp_path):
    from models.demos.common.bringup.core.gate import outputs_in

    cmd = "BRINGUP_CAPTURE_FORKS=r/fork_calls.json x && DS41_MOE_IMPL=block BRINGUP_CAPTURE_FORKS=r/fork_calls_block.json y"
    got = {p.name for p in outputs_in(tmp_path, {"id": "O.1", "step": "optests", "gate": {"cmd": cmd}})}
    assert {"fork_calls.json", "fork_calls_block.json"} <= got
