# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""produce-data test-report artifact consolidation (API-cost POC).

Every test job uploads its JUnit XML as its own ``test_reports_<uuid>`` artifact, so a run
produces dozens and produce-data spends one GitHub REST ZIP download per artifact. The POC
(sanity-tests.yaml ``merge-test-reports`` job) packs them into a single ``test_reports_merged``
artifact so produce-data downloads once, keeping each job's reports under its own
``test_reports_<uuid>/`` subdir.

These tests pin the invariant that matters: the uuid -> xml mapping produce-data builds is
identical for the legacy (one artifact per job) and consolidated (one merged artifact) layouts,
so nothing downstream of the packaging changes.
"""

import pathlib

from infra.data_collection.github.workflows import get_workflow_run_uuids_to_test_reports_paths_


def _write_reports(artifacts_dir: pathlib.Path, template: str, uuids):
    for uuid in uuids:
        xml_path = artifacts_dir / template.format(uuid=uuid)
        xml_path.parent.mkdir(parents=True, exist_ok=True)
        xml_path.write_text("<testsuite/>")


def _uuids_with_reports(mapping):
    return sorted(uuid for uuid, xmls in mapping.items() if xmls)


def test_legacy_flat_layout_maps_each_uuid(tmp_path):
    run_id = 999
    artifacts_dir = tmp_path / str(run_id) / "artifacts"
    _write_reports(artifacts_dir, "test_reports_{uuid}/report.xml", ("aaa", "bbb", "ccc"))

    mapping = get_workflow_run_uuids_to_test_reports_paths_(tmp_path, run_id)

    assert _uuids_with_reports(mapping) == ["aaa", "bbb", "ccc"]


def test_consolidated_merged_layout_maps_each_uuid(tmp_path):
    # One merged artifact holding every job's reports under its own uuid subdir.
    run_id = 999
    artifacts_dir = tmp_path / str(run_id) / "artifacts"
    _write_reports(artifacts_dir, "test_reports_merged/test_reports_{uuid}/report.xml", ("aaa", "bbb", "ccc"))

    mapping = get_workflow_run_uuids_to_test_reports_paths_(tmp_path, run_id)

    # The merged container itself has no *.xml, so it never becomes a usable report source;
    # only the per-job uuids carry reports, exactly as in the flat layout.
    assert _uuids_with_reports(mapping) == ["aaa", "bbb", "ccc"]


def test_both_layouts_are_equivalent(tmp_path):
    flat = tmp_path / "flat" / "999" / "artifacts"
    merged = tmp_path / "merged" / "999" / "artifacts"
    _write_reports(flat, "test_reports_{uuid}/report.xml", ("u1", "u2"))
    _write_reports(merged, "test_reports_merged/test_reports_{uuid}/report.xml", ("u1", "u2"))

    flat_map = get_workflow_run_uuids_to_test_reports_paths_(tmp_path / "flat", 999)
    merged_map = get_workflow_run_uuids_to_test_reports_paths_(tmp_path / "merged", 999)

    # Same uuids resolve, each with the same number of xml files.
    assert _uuids_with_reports(flat_map) == _uuids_with_reports(merged_map)
    assert {u: len(x) for u, x in flat_map.items() if x} == {u: len(x) for u, x in merged_map.items() if x}
