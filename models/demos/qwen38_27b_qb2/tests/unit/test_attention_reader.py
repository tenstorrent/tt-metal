# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import copy
from unittest.mock import patch

import pytest

from models.demos.qwen38_27b_qb2.tests.attention_reader import (
    CASES,
    COMMON,
    FACTORY,
    READER,
    VARIANTS,
    build_overlay,
    compare_readers,
    compilation_evidence,
    render_reader,
    sha,
    verify_overlay,
)

MODULE = "models.demos.qwen38_27b_qb2.tests.attention_reader"
SOURCE = """#include "dataflow_common.hpp"
    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<q_tile_bytes, num_cores>();
    uint32_t barrier_count = 0;
    read_q<cb_q_in, cb_q_rm, q_tile_bytes, q_chunk_tiles, is_q_sharded, tilize_q, use_half_tile, barrier_threshold>(
    // Paged template argument anchors; Q, mask and unpaged arguments must stay unchanged.
                    k_tile_bytes,
                    barrier_threshold,
                    v_tile_bytes,
                    barrier_threshold,
    read_mask_chunk<cb_mask_in, mask_tile_bytes, barrier_threshold, PNHt>(
    read_kv_mask_chunks<DHt, vDHt, barrier_threshold>(
"""


@pytest.mark.parametrize("variant", VARIANTS)
def test_override_keeps_installed_sources_immutable_and_detects_changes(tmp_path, variant, expect_error):
    native = tmp_path / "native"
    for relative, contents in (
        (READER, SOURCE),
        (COMMON, "// Original barriers and CB logic\n"),
        (FACTORY, "// Original CB sizes\n"),
    ):
        path = native / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    hashes = {str(relative): sha(native / relative) for relative in (READER, COMMON, FACTORY)}
    with patch(MODULE + ".PINNED", hashes):
        manifest = build_overlay(native, tmp_path / "overlay", variant)
        assert hashes == {relative: sha(native / relative) for relative in hashes}
        verify_overlay(manifest, tmp_path / "overlay", tmp_path / "empty-cwd")
        with expect_error(FileExistsError, "File exists"):
            build_overlay(native, tmp_path / "overlay", variant)
        (native / COMMON).write_text("// changed\n")
        with expect_error(ValueError, "Native dependency changed"):
            verify_overlay(manifest, tmp_path / "overlay", tmp_path / "empty-cwd")
        with expect_error(ValueError, "Pinned reader dependency changed"):
            build_overlay(native, tmp_path / "another-overlay", variant)


@pytest.mark.parametrize("variant", ["kv4", "kv8", "kv16"])
def test_only_paged_kv_intermediate_barriers_change(variant):
    text = render_reader(SOURCE, "/native/common.hpp", variant)
    assert text.count("                    kv_barrier_threshold,") == 2
    for line in SOURCE.splitlines():
        if "barrier_threshold" in line and not line.startswith("                    "):
            assert line in text
    assert "static_assert(qwen_reader_geometry" in text
    assert "!use_half_tile && !reuse_k && !use_k_mcast" in text
    assert f"constexpr uint32_t kv_barrier_threshold = {int(variant[2:])};" in text
    baseline = render_reader(SOURCE, "/native/common.hpp", "native")
    assert baseline.replace("/native/common.hpp", "dataflow_common.hpp") == SOURCE


def test_changed_or_ambiguous_source_stops_before_build(expect_error):
    for source in (SOURCE.replace("uint32_t barrier_count = 0;", "uint32_t counter = 0;"), SOURCE + SOURCE):
        with expect_error(ValueError, "anchor is missing or ambiguous"):
            render_reader(source, "/native/common.hpp", "kv8")
    with expect_error(ValueError, "Unsupported reader variant"):
        render_reader(SOURCE, "/native/common.hpp", "kv64")
    with expect_error(ValueError, "safe absolute path"):
        render_reader(SOURCE, '/native/unsafe"header.hpp', "native")


def test_wrong_override_or_cwd_shadow_rejected(tmp_path, expect_error):
    manifest = dict(overlay=str(tmp_path / "overlay"))
    with expect_error(ValueError, "does not match"):
        verify_overlay(manifest, tmp_path / "other", tmp_path)
    shadow = tmp_path / READER
    shadow.parent.mkdir(parents=True)
    shadow.write_text(SOURCE)
    with expect_error(ValueError, "shadows the reader"):
        verify_overlay(manifest, tmp_path / "overlay", tmp_path)


def test_jit_must_include_exact_override(tmp_path, expect_error):
    manifest = dict(reader="/override/" + str(READER))
    with expect_error(ValueError, "No JIT evidence"):
        compilation_evidence(tmp_path, manifest)
    includes = tmp_path / "kernel_includes.hpp"
    includes.write_text(f'#include "{manifest["reader"]}"\n')
    assert compilation_evidence(tmp_path, manifest) == [dict(path=str(includes), sha256=sha(includes))]
    includes.write_text(f'#include "/native/{READER}"\n')
    with expect_error(ValueError, "different attention reader"):
        compilation_evidence(tmp_path, manifest)


def reports():
    return [
        dict(
            variant=variant,
            passed=True,
            cleanup_completed=True,
            cases=[
                dict(
                    input_tokens=length,
                    batch=batch,
                    passed=True,
                    candidates=[dict(median_traced_call_us=time)],
                    selection=dict(timing_comparison_qualified=True),
                )
                for length, batch in CASES
            ],
        )
        for variant, time in zip([*VARIANTS, "native"], [100, 90, 80, 95, 101])
    ]


def test_cross_process_comparison_requires_clean_correct_stable_measurements(expect_error):
    runs = reports()
    comparisons = compare_readers(runs)
    assert all(c["fastest_passing_variant"] == "kv8" and c["timing_comparison_qualified"] for c in comparisons)
    runs[-1]["cases"][0]["selection"]["timing_comparison_qualified"] = False
    assert not compare_readers(runs)[0]["timing_comparison_qualified"]
    runs[-1]["cases"][0]["selection"]["timing_comparison_qualified"] = True
    runs[2]["cases"][0]["selection"]["timing_comparison_qualified"] = False
    assert compare_readers(runs)[0]["fastest_passing_variant"] == "kv4"
    runs[-1]["cases"][0]["candidates"][0]["median_traced_call_us"] = 110
    assert not compare_readers(runs)[0]["timing_comparison_qualified"]
    for key in ("passed", "cleanup_completed"):
        invalid = copy.deepcopy(runs)
        invalid[2][key] = False
        with expect_error(ValueError, "passing clean-device receipts"):
            compare_readers(invalid)
    runs[0]["cases"][0]["batch"] = 1
    with expect_error(ValueError, "geometry mismatch"):
        compare_readers(runs)
