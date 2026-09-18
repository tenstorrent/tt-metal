#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Sample wrapper logs, descriptors, and records used by the cluster health tests.

Kept inline rather than as checked-in data files: the samples are a few lines
each, and the tests that stat or rewrite them (mtime windows, backfill runs)
need a writable tree per test rather than a shared one.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

# Opaque caller-supplied path recorded as artifact_uri; never read from disk.
ARTIFACT_DIR = "/tmp/exabox/physical_validation/example"

CABLING_TWO_HOST = """\
graph_templates {
  key: "two_host"
  value {
    children {
      name: "node_0"
      node_ref { node_descriptor: "BH_GALAXY" }
    }
    children {
      name: "node_1"
      node_ref { node_descriptor: "BH_GALAXY" }
    }
  }
}

root_instance {
  template_name: "two_host"
  child_mappings {
    key: "bh_galaxy_sp_0"
    value {
      sub_instance {
        child_mappings {
          key: "node_0"
          value { host_id: 0 }
        }
        child_mappings {
          key: "node_1"
          value { host_id: 1 }
        }
      }
    }
  }
}
"""

DEPLOYMENT_TWO_HOST = """\
hosts: {
  aisle: "C"
  rack: 1
  shelf_u: 2
  host: "bh-glx-110-c01u02"
}
hosts: {
  aisle: "C"
  rack: 1
  shelf_u: 8
  host: "bh-glx-110-c01u08"
}
"""

FSD_TWO_HOST = """\
hosts { hostname: "bh-glx-110-c01u02" aisle: "C" rack: 1 shelf_u: 2 }
hosts { hostname: "bh-glx-110-c01u08" aisle: "C" rack: 1 shelf_u: 8 }
"""

GSD_TWO_HOST = """\
compute_node_specs:
  bh-glx-110-c01u02:
    motherboard: TEST
  bh-glx-110-c01u08:
    motherboard: TEST
ethernet_connections:
  - ignored: true
"""

RANKFILE_TWO_HOST = """\
rank 0=bh-glx-110-c01u02 slot=0
rank 1=bh-glx-110-c01u08 slot=0
"""

RANK_BINDINGS_TWO_HOST = """\
rank_bindings:
  - rank: 0
    mesh_id: 0
    mesh_host_rank: 0

  - rank: 1
    mesh_id: 0
    mesh_host_rank: 1
"""

# Shape of the checked-in Exabox bindings: mesh_host_rank is optional in tt-run.
RANK_BINDINGS_NO_HOST_RANK = """\
rank_bindings:
  - rank: 0
    mesh_id: 0
  - rank: 1
    mesh_id: 1

mesh_graph_desc_path: "tests/tt_metal/tt_fabric/custom_mesh_descriptors/wh_exabox_1x32_mesh_graph_descriptor.textproto"
"""

DIAG_REPORT_PASS = """\
{
  "tool_version": "0.3.0-draft",
  "tier": "light",
  "host": "bh-glx-110-c01u02",
  "started_utc": "2026-08-19T03:12:00.000000+00:00",
  "dry_run": false,
  "tt_smi_version": "5.1.1",
  "detected_board_rev": "RevC",
  "phases": {
    "snapshot": {"name": "snapshot", "status": "PASS", "duration_s": 1.0, "checks": [], "error": ""},
    "reset_loop": {"name": "reset_loop", "status": "PASS", "duration_s": 10.0, "checks": [], "error": ""},
    "tests": {"name": "tests", "status": "PASS", "duration_s": 25.0, "checks": [], "error": ""}
  },
  "ended_utc": "2026-08-19T03:12:36.500000+00:00",
  "total_duration_s": 36.5,
  "overall_status": "PASS"
}
"""

DIAG_REPORT_WARN = """\
{
  "tool_version": "0.3.0-draft",
  "tier": "light",
  "host": "bh-glx-110-c01u02",
  "started_utc": "2026-08-19T03:12:00+00:00",
  "dry_run": false,
  "detected_board_rev": "RevA/B",
  "phases": {},
  "ended_utc": "2026-08-19T03:12:10+00:00",
  "total_duration_s": 10.0,
  "overall_status": "WARN"
}
"""

DIAG_REPORT_FAIL = """\
{
  "tool_version": "0.3.0-draft",
  "tier": "medium",
  "host": "bh-glx-110-c01u08",
  "started_utc": "2026-08-19T04:00:00+00:00",
  "dry_run": false,
  "phases": {},
  "ended_utc": "2026-08-19T04:05:00+00:00",
  "total_duration_s": 300.0,
  "overall_status": "FAIL"
}
"""

DIAG_REPORT_DRY_RUN = """\
{
  "tool_version": "0.3.0-draft",
  "tier": "light",
  "host": "bh-glx-110-c01u02",
  "started_utc": "2026-08-19T03:12:00+00:00",
  "dry_run": true,
  "phases": {},
  "ended_utc": "2026-08-19T03:12:01+00:00",
  "total_duration_s": 1.0,
  "overall_status": "PASS"
}
"""

CANONICAL_WITH_STORE = """\
{
  "schema": "exabox.cluster_health.v1",
  "ts": "2026-08-19T03:12:00Z",
  "test_type": "physical",
  "status": "failed",
  "analyzer_code": 1,
  "hosts": [
    "bh-glx-110-c01u02",
    "bh-glx-110-c01u08",
    "bh-glx-110-c02u02",
    "bh-glx-110-c02u08"
  ],
  "topology": {
    "instance_paths": ["bh_galaxy_sp_0/node_0"]
  },
  "cluster": "exabox",
  "artifact_uri": "/tmp/exabox/physical_validation/example",
  "record_uri": "/tmp/cluster-health/2026-08-19/example-record-id.json",
  "record_id": "example-record-id",
  "source": "cli",
  "triggered_by": "mnijjar",
  "trigger_kind": "user",
  "orchestrator_id": "696e1a2b3c4d5e6f7a8b9c0d",
  "labels": {
    "quad": "110-C-Q1",
    "superpod": "SC36_3"
  }
}
"""

LAPTOP_DRY_RUN = """\
{
  "schema": "exabox.cluster_health.v1",
  "ts": "2026-08-19T03:12:00Z",
  "test_type": "physical",
  "status": "passed",
  "hosts": [
    "bh-glx-110-c01u02",
    "bh-glx-110-c01u08"
  ]
}
"""

HOST_FROM_DIAG = """\
{
  "schema": "exabox.cluster_health.v1",
  "ts": "2026-08-19T03:12:36Z",
  "test_type": "host",
  "status": "passed",
  "analyzer_code": 0,
  "hosts": ["bh-glx-110-c01u02"],
  "duration_s": 36.5,
  "labels": {
    "tier": "light",
    "board_rev": "RevC"
  }
}
"""

RECOVER_NO_ANALYZER_CODE = """\
{
  "schema": "exabox.cluster_health.v1",
  "ts": "2026-08-19T03:12:00+00:00",
  "test_type": "recover",
  "status": "passed",
  "hosts": ["bh-glx-110-c01u02"]
}
"""

# One complete run per test type, plus a physical run with no HOSTS= line that
# discovery is expected to skip. The .json sidecar overrides the .log footer.
PHYSICAL_LOG = "physical_validation-20260819T031200Z.log"
PHYSICAL_SIDECAR = "physical_validation-20260819T031200Z.json"
PHYSICAL_NO_HOSTS_LOG = "physical_validation-20260819T040000Z.log"
FABRIC_LOG = "fabric_tests-20260819T031300Z.log"
DISPATCH_LOG = "dispatch_tests-20260819T031400Z.log"
RECOVER_LOG = "recover-20260819T031000Z.log"

ARTIFACT_TREE_LOGS: dict[str, str] = {
    PHYSICAL_LOG: """\
=== Physical Validation - 20260819T031200Z ===
EXECUTION_DIR=/tmp/artifact_tree
OUTPUT_DIR=physical_validation/label/abcd1234_20260819T031200Z
HOSTS=bh-glx-110-c01u02,bh-glx-110-c01u08

--- Analyzing validation results ---

Analysis exit code: 1
""",
    PHYSICAL_SIDECAR: """\
{
  "status": "success",
  "analysis_exit_code": 0,
  "output_dir": "physical_validation/label/abcd1234_20260819T031200Z",
  "checked_at": "20260819T031200Z"
}
""",
    PHYSICAL_NO_HOSTS_LOG: """\
=== Physical Validation - 20260819T040000Z ===
OUTPUT_DIR=physical_validation/orphan/missing_hosts

Analysis exit code: 2
""",
    FABRIC_LOG: """\
=== Fabric Tests - 20260819T031300Z ===
OUTPUT_DIR=fabric_tests/110-C-Q1-all-20260819T031300Z
HOSTS=bh-glx-110-c01u02,bh-glx-110-c01u08

Analysis exit code: 3
""",
    DISPATCH_LOG: """\
=== Dispatch Tests - 20260819T031400Z ===
OUTPUT_DIR=dispatch_tests
HOSTS=bh-glx-110-c01u02

Analysis exit code: 1
""",
    RECOVER_LOG: """\
=== Recover - 20260819T031000Z ===
HOSTS=bh-glx-110-c01u02,bh-glx-110-c01u08

Recover completed successfully
""",
}

# Wrapper endings taken from real runs, where the terminal event is ambiguous.
PHYSICAL_TRUNCATED_LOG = """\
=== Physical Validation - 20260819T050000Z ===
HOSTS=bh-glx-110-c01u02,bh-glx-110-c01u08
OUTPUT_DIR=physical_validation/truncated/run
ITERATIONS=1

MPI body still running; wrapper never wrote Analysis exit code.
"""

RECOVER_FAILED_LOG = """\
=== Recover - 20260819T051000Z ===
HOSTS=bh-glx-110-d10u02,bh-glx-110-d10u08
OUTPUT_DIR=/tmp/recover-failed

[bh-glx-110-d10u02] Recovery attempt 1 of 1
[bh-glx-110-d10u02] Recovery attempt 1 of 1 failed (exit code 1).
[bh-glx-110-d10u02] Recovery completed at Wed Aug 19 05:04:23 UTC 2026
"""

RECOVER_SUCCEEDED_LOG = """\
=== Recover - 20260819T051100Z ===
HOSTS=bh-glx-110-d10u02,bh-glx-110-d10u08
OUTPUT_DIR=/tmp/recover-ok

[bh-glx-110-d10u02] Recovery attempt 1 of 2 failed (exit code 134).
[bh-glx-110-d10u02] Recovery succeeded on attempt 2
[bh-glx-110-d10u02] Recovery completed at Wed Aug 19 05:25:36 UTC 2026
"""

RECOVER_INCOMPLETE_LOG = """\
=== Recover - 20260819T051200Z ===
HOSTS=bh-glx-110-d10u02
OUTPUT_DIR=/tmp/recover-incomplete

Running recover.sh:
  ./tools/scaleout/exabox/recover.sh --hosts bh-glx-110-d10u02
"""


def temp_dir(test: unittest.TestCase) -> Path:
    """Return a fresh directory removed when ``test`` finishes."""
    holder = tempfile.TemporaryDirectory()
    test.addCleanup(holder.cleanup)
    return Path(holder.name)


def write(directory: Path, name: str, text: str) -> Path:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def temp_file(test: unittest.TestCase, name: str, text: str) -> Path:
    return write(temp_dir(test), name, text)


def log_tree(test: unittest.TestCase, name: str, text: str) -> Path:
    """Return an artifact root holding a single wrapper log under ``logs/``."""
    root = temp_dir(test)
    write(root / "logs", name, text)
    return root


def artifact_tree(test: unittest.TestCase) -> Path:
    """Return an artifact root holding one wrapper log per test type."""
    root = temp_dir(test)
    for name, text in ARTIFACT_TREE_LOGS.items():
        write(root / "logs", name, text)
    return root
