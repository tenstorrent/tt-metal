# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

# Category JQL over DIIM summary + description; triage runs it against the host's tickets.
JQL_QUERIES = Path(__file__).with_name("jql_categories.json")

JIRA_ENV = Path.home() / ".config" / "device_triage" / "jira.env"
JIRA_SITE = "https://tenstorrent.atlassian.net"

MACHINE_HC_ROOT = Path("/data/ci-provisioner/workflows/hc_logs")
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
