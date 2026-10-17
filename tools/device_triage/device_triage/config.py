# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = Path(os.environ.get("DEVICE_TRIAGE_CACHE", PROJECT_DIR / "cache"))
INDEX_DB = CACHE_DIR / "index.sqlite"
JIRA_CACHE = CACHE_DIR / "jira"
# Machine HC runs rebuilt from DCAMP bot-ticket attachments (history older than hc_logs keeps).
DCAMP_HC_ROOT = CACHE_DIR / "dcamp_hc"

CANVAS_PATH = Path(os.environ.get("DEVICE_TRIAGE_CANVAS", PROJECT_DIR / "docs" / "overview.canvas.tsx"))
# Category JQL ships with the code; jira-sync stores which tickets match each query.
JQL_QUERIES = Path(__file__).with_name("jql_categories.json")
JQL_MEMBERSHIP = JIRA_CACHE / "jql_membership.json"

JIRA_ENV = Path.home() / ".config" / "device_triage" / "jira.env"
JIRA_SITE = "https://tenstorrent.atlassian.net"
JIRA_PROJECTS = ("DIIM", "DCAMP")

MACHINE_HC_ROOT = Path("/data/ci-provisioner/workflows/hc_logs")
FUNTEST_ROOT = Path("/data/ci-provisioner/funtest_test_logs")
CLUSTER_REPORT_ROOTS = (
    Path("/data/dcamp/cluster-health-check/reports"),
    Path("/data/dcamp/cluster-health-check/reports-archive"),
)
RECOVER_LOG_ROOT = Path("/data/dcamp/cluster-health-check/logs/recover")
STACKSTORM_ROOT = Path("/data/stackstorm")
BMC_REBOOT_ROOT = Path("/data/dcamp/bmc-reboot-data-collection")

CHIPS_PER_GALAXY = 32
CHIPS_PER_UBB = 8

# Log lines beyond this are not scanned; a run that logs more than this is
# looping on one error and the first part carries the signal.
MAX_LINES_PER_FILE = 400_000
MAX_EVIDENCE_CHARS = 400
