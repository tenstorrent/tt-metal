# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import pytest

from models.demos.qwen38_27b_qb2.tests.gdn_phase_profile import instrument, required_markers, validate_coverage
from models.demos.qwen38_27b_qb2.tt.gdn_epilogue.op import source as epilogue_source
from models.demos.qwen38_27b_qb2.tt.gdn_step.op import kernel_source


def test_instrumentation_preserves_cpp_tokens_except_scopes_and_markers():
    import re

    sources = {
        name: kernel_source(name) for name in ("reader.cpp", "writer.cpp", "compute.cpp", "compute_resident.cpp")
    }
    sources.update({"epilogue/" + name: epilogue_source(name) for name in ("reader.cpp", "writer.cpp", "compute.cpp")})
    for name, original in sources.items():
        annotated = instrument(name, original)
        restored = annotated.removeprefix('#include "tools/profiler/kernel_profiler.hpp"\n')
        restored = re.sub(r'\{\nDeviceZoneScopedN\("GDN_[A-Z0-9_]+"\);\n', "", restored)
        # Removing all braces here verifies token preservation while allowing
        # the extra local scopes; hardware compilation checks their lifetime.
        tokens = lambda text: re.sub(r"[{}\s]", "", text)
        assert tokens(restored) == tokens(original)


def test_changed_anchor_fails_closed(expect_error):
    with expect_error(ValueError, "anchor changed"):
        instrument("reader.cpp", kernel_source("reader.cpp").replace("cb_reserve_back(0, 4)", "cb_reserve_back(0, 8)"))


def pipeline_report():
    hashes = {name: [str(rank) * 64 for rank in range(4)] for name in ("state", "raw_output", "output")}
    checks = [
        dict(rank=rank, state=dict(passed=True), raw_output=dict(passed=True), finite_output=True) for rank in range(4)
    ]
    return dict(
        state="completed",
        passed=True,
        cleanup_completed=True,
        device_ids=[3, 4, 5, 6],
        kernel_calls=48,
        cases=[
            dict(batch=batch, padding=padding, profiled=profiled, checks=deepcopy(checks), hashes=deepcopy(hashes))
            for batch in (16, 32)
            for padding in ("zero", "skip")
            for profiled in (False, True)
        ],
    )


def test_complete_pipeline_receipt_and_marker_coverage():
    validate_coverage(pipeline_report(), pipeline=True)
    markers = required_markers(pipeline=True)
    assert "GDN_RESIDENT_DELTA" in markers and "GDN_EP_READER_PADDING" in markers
    assert "GDN_COMPUTE_DELTA" not in markers
    assert len(required_markers()) == 10


@pytest.mark.parametrize(
    "damage", ["missing_case", "missing_rank", "failed_rank", "changed_output", "missing_hash", "unclean"]
)
def test_reject_incomplete_or_incorrect_pipeline_receipt(damage, expect_error):
    report = pipeline_report()
    if damage == "missing_case":
        report["cases"].pop()
    elif damage == "missing_rank":
        report["cases"][0]["checks"].pop()
    elif damage == "failed_rank":
        report["cases"][0]["checks"][0]["raw_output"]["passed"] = False
    elif damage == "changed_output":
        report["cases"][1]["hashes"]["output"][2] = "f" * 64
    elif damage == "missing_hash":
        report["cases"][0]["hashes"]["state"].pop()
    else:
        report["cleanup_completed"] = False
    with expect_error(ValueError, "profile|zones"):
        validate_coverage(report, pipeline=True)
