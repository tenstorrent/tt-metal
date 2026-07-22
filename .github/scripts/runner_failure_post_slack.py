#!/usr/bin/env python3
"""Post runner-failure reports to Slack."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request

from runner_failure_common import (
    JobScanResult,
    RecentJob,
    UNKNOWN_RUNNER,
    format_signature_summary,
)


SLACK_API_URL = "https://slack.com/api/chat.postMessage"
MAX_SLACK_TEXT_LENGTH = 35000
MAX_SLACK_TABLE_ROWS = 100
MAX_SLACK_TABLE_CHARS = 9500
SLACK_TABLE_DATA_ROWS_PER_MESSAGE = MAX_SLACK_TABLE_ROWS - 1


@dataclass(frozen=True)
class SlackConfig:
    token: str
    channel: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post a completed runner-failure report to Slack.")
    parser.add_argument(
        "--runner-report-json",
        type=Path,
        required=True,
        help="Path to a completed single-runner report JSON.",
    )
    parser.add_argument(
        "--triggering-failures-json",
        type=Path,
        help="Optional JSON array of triggering failures from the scan workflow.",
    )
    parser.add_argument(
        "--runner-name",
        help="Runner/host name to use in Slack messages.",
    )
    parser.add_argument(
        "--slack-channel",
        default=os.environ.get("RUNNER_FAILURE_SCAN_SLACK_CHANNEL"),
        help="Slack channel ID for runner-failure summaries.",
    )
    return parser.parse_args()


def slack_config_from_channel(slack_channel: str | None) -> SlackConfig | None:
    if not slack_channel:
        return None
    token = os.environ.get("SLACK_BOT_TOKEN") or os.environ.get("RUNNER_FAILURE_SCAN_SLACK_BOT_TOKEN")
    if not token:
        raise RuntimeError("Slack channel is configured, but SLACK_BOT_TOKEN is not set.")
    return SlackConfig(token=token, channel=slack_channel)


def job_from_dict(value: dict[str, Any]) -> RecentJob:
    return RecentJob(
        owner_repo=str(value.get("owner_repo") or ""),
        workflow=str(value.get("workflow") or ""),
        workflow_id=str(value.get("workflow_id") or ""),
        run_id=str(value.get("run_id") or ""),
        run_attempt=str(value.get("run_attempt") or ""),
        run_url=str(value.get("run_url") or ""),
        job_id=str(value.get("job_id") or ""),
        name=str(value.get("name") or ""),
        runner_name=str(value.get("runner_name") or ""),
        status=str(value.get("status") or ""),
        conclusion=str(value.get("conclusion") or ""),
        html_url=str(value.get("html_url") or ""),
        started_at=str(value.get("started_at") or ""),
        completed_at=str(value.get("completed_at") or ""),
    )


def scan_result_from_dict(value: dict[str, Any]) -> JobScanResult:
    raw_signatures = value.get("signatures")
    signatures: tuple[str, ...] = ()
    if isinstance(raw_signatures, list):
        signatures = tuple(str(item) for item in raw_signatures if item)
    return JobScanResult(
        job=job_from_dict(value),
        log_status=str(value.get("log_status") or ""),
        log_checked=bool(value.get("log_checked")),
        signature_labels=signatures,
        fabric_missing_links=str(value.get("fabric_missing_links") or ""),
    )


def load_report_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(f"Unable to read report JSON {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Unable to parse report JSON {path}: {exc}") from exc

    if not isinstance(value, dict):
        raise RuntimeError(f"Report JSON {path} must contain an object.")
    return value


def scan_results_from_report(report: dict[str, Any]) -> list[JobScanResult]:
    values = report.get("scan_results")
    if not isinstance(values, list):
        return []
    return [scan_result_from_dict(value) for value in values if isinstance(value, dict)]


def int_from_report(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def runner_job_count_from_report(report: dict[str, Any], scan_results: list[JobScanResult]) -> int:
    counts = report.get("counts")
    if isinstance(counts, dict):
        return int_from_report(counts.get("runner_jobs"), default=len(scan_results))
    return len(scan_results)


def load_triggering_failures_json(path: Path | None) -> list[JobScanResult]:
    if path is None:
        return []

    try:
        raw_values = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(f"Unable to read triggering failures JSON {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Unable to parse triggering failures JSON {path}: {exc}") from exc

    if not isinstance(raw_values, list):
        raise RuntimeError(f"Triggering failures JSON {path} must contain a list.")

    return [scan_result_from_dict(value) for value in raw_values if isinstance(value, dict)]


def slack_escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def slack_link(url: str, label: str) -> str:
    if not url:
        return slack_escape(label)
    return f"<{url}|{slack_escape(label)}>"


def truncate_for_slack(value: str, max_length: int = MAX_SLACK_TEXT_LENGTH) -> str:
    if len(value) <= max_length:
        return value
    suffix = "\n... truncated ..."
    return value[: max_length - len(suffix)] + suffix


def post_slack_message(
    slack_config: SlackConfig,
    text: str,
    *,
    thread_ts: str | None = None,
    blocks: list[dict[str, Any]] | None = None,
) -> str:
    payload: dict[str, Any] = {
        "channel": slack_config.channel,
        "text": truncate_for_slack(text),
        "unfurl_links": False,
        "unfurl_media": False,
    }
    if blocks is not None:
        payload["blocks"] = blocks
    if thread_ts:
        payload["thread_ts"] = thread_ts

    slack_request = request.Request(
        SLACK_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {slack_config.token}",
            "Content-Type": "application/json; charset=utf-8",
        },
        method="POST",
    )
    try:
        with request.urlopen(slack_request, timeout=30) as response:
            response_body = response.read().decode("utf-8", errors="replace")
    except error.URLError as exc:
        raise RuntimeError(f"Slack post failed: {exc}") from exc

    try:
        response_json = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Slack returned non-JSON response: {response_body}") from exc

    if not response_json.get("ok"):
        slack_error = response_json.get("error") or "unknown_error"
        raise RuntimeError(f"Slack post failed: {slack_error}")

    ts = response_json.get("ts")
    if not isinstance(ts, str) or not ts:
        raise RuntimeError("Slack post succeeded but response did not include ts.")
    return ts


def slack_raw_text_cell(value: str) -> dict[str, str]:
    return {"type": "raw_text", "text": value}


def slack_blank_cell() -> dict[str, str]:
    return slack_raw_text_cell(" ")


def slack_link_cell(url: str, label: str) -> dict[str, Any]:
    if not url:
        return slack_raw_text_cell(label)
    return {
        "type": "rich_text",
        "elements": [
            {
                "type": "rich_text_section",
                "elements": [
                    {
                        "type": "link",
                        "url": url,
                        "text": label,
                    }
                ],
            }
        ],
    }


def slack_table_cell_text(cell: dict[str, Any]) -> str:
    if cell.get("type") == "raw_text":
        return str(cell.get("text") or "")

    pieces: list[str] = []
    for element in cell.get("elements") or []:
        for child in element.get("elements") or []:
            pieces.append(str(child.get("text") or child.get("url") or ""))
    return "".join(pieces)


def slack_table_rows_text_length(rows: list[list[dict[str, Any]]]) -> int:
    return sum(len(slack_table_cell_text(cell)) for row in rows for cell in row)


def slack_flag_cell(value: str) -> dict[str, str]:
    return slack_raw_text_cell(value or "-")


def compact_started_at(value: str) -> str:
    if len(value) >= len("YYYY-MM-DDTHH:MM"):
        return value[:16].replace("T", " ")
    return value or "unknown"


def workflow_job_label(job: RecentJob) -> str:
    if job.workflow:
        return f"{job.workflow} / {job.name}"
    return job.name


def slack_runner_table_header() -> list[dict[str, str]]:
    return [
        slack_raw_text_cell("Job"),
        slack_raw_text_cell("Status"),
        slack_raw_text_cell("Started"),
        slack_raw_text_cell("Scanned?"),
        slack_raw_text_cell("Failures"),
        slack_raw_text_cell("Workflow / job"),
    ]


def failure_summary_for_slack_cell(result: JobScanResult) -> str:
    if not result.log_checked or not result.signature_labels:
        # Slack table raw_text cells reject truly empty strings.
        return " "

    failures = list(result.signature_labels)
    if result.fabric_missing_links:
        failures = [
            (f"{label}: {result.fabric_missing_links}" if label == "Fabric link down (MGD topology)" else label)
            for label in failures
        ]
    return "; ".join(failures)


def has_runner_related_failure(result: JobScanResult) -> bool:
    return bool(result.signature_labels)


def slack_runner_table_row(result: JobScanResult) -> list[dict[str, Any]]:
    job = result.job
    return [
        slack_link_cell(job.html_url, job.job_id or "open"),
        slack_raw_text_cell(job.conclusion or job.status or "unknown"),
        slack_raw_text_cell(compact_started_at(job.started_at)),
        slack_flag_cell("Y" if result.log_checked else "N"),
        slack_raw_text_cell(failure_summary_for_slack_cell(result)),
        slack_raw_text_cell(workflow_job_label(job)),
    ]


def slack_skipped_jobs_row(skipped_count: int) -> list[dict[str, Any]]:
    job_word = "job" if skipped_count == 1 else "jobs"
    message = f"{skipped_count} {job_word} without runner-related failures skipped " "(see web for full table)"
    return [
        slack_raw_text_cell("..."),
        slack_blank_cell(),
        slack_blank_cell(),
        slack_blank_cell(),
        slack_blank_cell(),
        slack_raw_text_cell(message),
    ]


def compact_slack_runner_table_rows(
    results: list[JobScanResult],
) -> tuple[list[list[dict[str, Any]]], int]:
    rows: list[list[dict[str, Any]]] = []
    skipped_rows = 0
    index = 0

    while index < len(results):
        result = results[index]
        if has_runner_related_failure(result):
            rows.append(slack_runner_table_row(result))
            index += 1
            continue

        streak_start = index
        while index < len(results) and not has_runner_related_failure(results[index]):
            index += 1
        streak = results[streak_start:index]

        if len(streak) <= 2:
            rows.extend(slack_runner_table_row(item) for item in streak)
            continue

        skipped_count = len(streak) - 2
        skipped_rows += skipped_count
        rows.append(slack_runner_table_row(streak[0]))
        rows.append(slack_skipped_jobs_row(skipped_count))
        rows.append(slack_runner_table_row(streak[-1]))

    return rows, skipped_rows


def chunk_slack_table_rows(
    rows: list[list[dict[str, Any]]],
) -> list[list[list[dict[str, Any]]]]:
    chunks: list[list[list[dict[str, Any]]]] = []
    current: list[list[dict[str, Any]]] = []
    header = slack_runner_table_header()

    for row in rows:
        candidate = [header, *current, row]
        would_exceed_rows = len(candidate) > MAX_SLACK_TABLE_ROWS
        would_exceed_chunk_rows = len(current) >= SLACK_TABLE_DATA_ROWS_PER_MESSAGE
        would_exceed_chars = slack_table_rows_text_length(candidate) > MAX_SLACK_TABLE_CHARS
        if current and (would_exceed_rows or would_exceed_chunk_rows or would_exceed_chars):
            chunks.append(current)
            current = []
        current.append(row)

    if current:
        chunks.append(current)
    return chunks


def slack_runner_table_block(rows: list[list[dict[str, Any]]]) -> dict[str, Any]:
    return {
        "type": "table",
        "column_settings": [
            {"is_wrapped": False},
            {"is_wrapped": False},
            {"is_wrapped": False},
            {"align": "center"},
            {"is_wrapped": True},
            {"is_wrapped": True},
        ],
        "rows": [slack_runner_table_header(), *rows],
    }


def format_triggering_jobs_for_slack(runner: str, results: list[JobScanResult]) -> str:
    lines = [f"Triggering job failures for `{slack_escape(runner)}`:"]
    for result in results:
        signature_text = ", ".join(result.signature_labels)
        if result.fabric_missing_links:
            signature_text += f"; missing links: {result.fabric_missing_links}"
        lines.append(
            "* "
            f"{slack_link(result.job.html_url, result.job.job_id)} "
            f"{slack_escape(result.job.workflow)} / "
            f"{slack_escape(result.job.name)} "
            f"({slack_escape(signature_text)})"
        )
    return truncate_for_slack("\n".join(lines))


def format_runner_log_table_for_slack(runner: str, results: list[JobScanResult]) -> str:
    lines = [
        f"Full log signature report for `{slack_escape(runner)}` " "All runner jobs returned by the runner jobs API.",
        "The failures field is empty when the log was not checked or no known signature was found.",
    ]
    index = 0
    while index < len(results):
        result = results[index]
        if not has_runner_related_failure(result):
            streak_start = index
            while index < len(results) and not has_runner_related_failure(results[index]):
                index += 1
            streak = results[streak_start:index]
            if len(streak) > 2:
                skipped_count = len(streak) - 2
                lines.append(format_runner_log_line_for_slack(streak[0]))
                job_word = "job" if skipped_count == 1 else "jobs"
                lines.append(
                    "* "
                    f"{skipped_count} {job_word} without runner-related failures skipped "
                    "(see web for full table)"
                )
                lines.append(format_runner_log_line_for_slack(streak[-1]))
                continue
            else:
                streak_to_emit = streak
        else:
            index += 1
            streak_to_emit = [result]

        for result_to_emit in streak_to_emit:
            lines.append(format_runner_log_line_for_slack(result_to_emit))
    return truncate_for_slack("\n".join(lines))


def format_runner_log_line_for_slack(result: JobScanResult) -> str:
    job = result.job
    status = job.conclusion or job.status or "unknown"
    started_at = job.started_at or "unknown start"
    job_label = job.job_id or "unknown-job"
    failures = failure_summary_for_slack_cell(result)
    failures = failures.strip()
    failures_text = f"failures:`{slack_escape(failures)}` " if failures else ""
    return (
        "* "
        f"{slack_link(job.html_url, job_label)} "
        f"`{slack_escape(status)}` "
        f"`{slack_escape(started_at)}` "
        f"log:`{'Y' if result.log_checked else 'N'}` "
        f"{failures_text}"
        f"{slack_escape(job.workflow)} / {slack_escape(job.name)}"
    )


def post_runner_log_table_for_slack(
    slack_config: SlackConfig,
    *,
    runner: str,
    results: list[JobScanResult],
    thread_ts: str,
) -> None:
    table_rows, skipped_rows = compact_slack_runner_table_rows(results)
    compaction_note = ""
    if skipped_rows:
        job_word = "job" if skipped_rows == 1 else "jobs"
        compaction_note = (
            f" Slack collapsed {skipped_rows} no-hit {job_word}; " "the full table is in the workflow summary/artifact."
        )
    heading = (
        f"Full log signature table for `{slack_escape(runner)}`: "
        f"{len(results)} recent job(s) found for this runner. "
        "The Failures column is blank when the log was not checked or no known signature was found."
        f"{compaction_note}"
    )
    post_slack_message(slack_config, heading, thread_ts=thread_ts)

    if not table_rows:
        return

    chunks = chunk_slack_table_rows(table_rows)
    row_offset = 0
    for chunk in chunks:
        start_row = row_offset + 1
        end_row = start_row + len(chunk) - 1
        row_offset += len(chunk)
        fallback_text = (
            f"Runner log signature table for {runner}, "
            f"rows {start_row}-{end_row} of {len(table_rows)} displayed row(s)."
        )
        try:
            post_slack_message(
                slack_config,
                fallback_text,
                thread_ts=thread_ts,
                blocks=[slack_runner_table_block(chunk)],
            )
        except RuntimeError as exc:
            if "invalid_blocks" not in str(exc):
                raise
            print(
                "warning: Slack rejected table blocks; posting plain-text " f"fallback for {runner}: {exc}",
                file=sys.stderr,
            )
            post_slack_message(
                slack_config,
                format_runner_log_table_for_slack(runner, results),
                thread_ts=thread_ts,
            )
            return


def post_runner_report_from_report(
    slack_config: SlackConfig,
    report: dict[str, Any],
    *,
    triggering_failures: list[JobScanResult] | None = None,
    runner_name_override: str | None = None,
) -> None:
    runner_name = str(runner_name_override or report.get("runner_name") or UNKNOWN_RUNNER)
    scan_results = scan_results_from_report(report)
    hours = int_from_report(report.get("hours"), default=24)
    runner_job_count = runner_job_count_from_report(report, scan_results)
    report_failures = [result for result in scan_results if result.signature_labels]
    triggering_results = triggering_failures or []
    triggering_report_failures = [result for result in triggering_results if result.signature_labels]

    if report_failures:
        parent_text = (
            f"Detected runner failure on `{slack_escape(runner_name)}`: "
            f"Out of {runner_job_count} jobs in the last {hours}h "
            f"{slack_escape(format_signature_summary(report_failures))}. "
            "Details in thread \U0001F9F5"
        )
    elif triggering_report_failures:
        parent_text = (
            f"Detected runner failure on `{slack_escape(runner_name)}`: "
            f"triggering scan found {slack_escape(format_signature_summary(triggering_report_failures))}. "
            f"The in-depth report covered {runner_job_count} jobs in the last {hours}h. "
            "Details in thread \U0001F9F5"
        )
    else:
        parent_text = (
            f"Runner report for `{slack_escape(runner_name)}`: "
            f"Out of {runner_job_count} jobs in the last {hours}h "
            "no known runner-failure signatures found. "
            "Details in thread \U0001F9F5"
        )

    thread_ts = post_slack_message(slack_config, parent_text)
    if triggering_results:
        post_slack_message(
            slack_config,
            format_triggering_jobs_for_slack(runner_name, triggering_results),
            thread_ts=thread_ts,
        )
    post_runner_log_table_for_slack(
        slack_config,
        runner=runner_name,
        results=scan_results,
        thread_ts=thread_ts,
    )
    print(f"Posted Slack runner report for {runner_name}.")


def main() -> int:
    args = parse_args()
    try:
        slack_config = slack_config_from_channel(args.slack_channel)
        if slack_config is None:
            return 0
        report = load_report_json(args.runner_report_json)
        triggering_failures = load_triggering_failures_json(args.triggering_failures_json)
        post_runner_report_from_report(
            slack_config,
            report,
            triggering_failures=triggering_failures,
            runner_name_override=args.runner_name,
        )
        return 0
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
