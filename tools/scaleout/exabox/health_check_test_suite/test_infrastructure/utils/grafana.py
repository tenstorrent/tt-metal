# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Grafana deep links to the per-device tt-telemetry dashboard."""

from __future__ import annotations

from datetime import datetime, timedelta
from urllib.parse import urlencode

DASHBOARD_UID = "tt-telemetry-overview"
DASHBOARD_SLUG = "tt-telemetry-e28094-device-overview"
DATASOURCE_UID = "ffkl22q5tp3pcd"
CLUSTER = "exabox"

WINDOW_BEFORE = timedelta(minutes=60)
WINDOW_AFTER = timedelta(minutes=15)


def telemetry_dashboard_url(*, base_url: str, node: str, fail_time: datetime) -> str:
    """Per-node tt-telemetry dashboard URL, time-boxed around ``fail_time``.

    Bounds are absolute epoch-ms: tickets get read days later, when a relative
    range would show a node that has since rebooted.
    """
    query = urlencode(
        [
            ("orgId", "1"),
            ("from", str(int((fail_time - WINDOW_BEFORE).timestamp() * 1000))),
            ("to", str(int((fail_time + WINDOW_AFTER).timestamp() * 1000))),
            ("timezone", "utc"),
            ("var-DS_PROMETHEUS", DATASOURCE_UID),
            ("var-cluster", CLUSTER),
            ("var-hostname", node),
        ]
    )
    return f"{base_url.rstrip('/')}/d/{DASHBOARD_UID}/{DASHBOARD_SLUG}?{query}"
