"""Private cumulative-log observer. Never dispatches, fetches, or cancels."""
import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ISO = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?Z)\s*")
LOCAL = re.compile(r"\b(?:INFO|WARNING|ERROR|DEBUG) (\d\d-\d\d \d\d:\d\d:\d\d)\b")
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
RECEIVED = re.compile(r"Received request ([^\s:]+):")
PROGRESS = re.compile(r"Requesting API:.*?\b(\d+)/(\d+)\b")
ERROR = re.compile(
    r"Could not parse generations|API request failed|Retry attempt|Replacing with error sentinels|padding with error sentinels|__INFERENCE_ERROR__|__PARTIAL_OUTPUT__|Traceback|TimeoutError"
)
SCORE = re.compile(
    r'Running postprocessing|Processing results|Total evaluation time|Evaluation completed|Writing results|Saving results|"exact_match(?:,strict-match)?"\s*:'
)
TRUNC = re.compile(r"(?i)(?:log|output|line).{0,30}truncat|truncat.{0,30}(?:log|output|line)")


def date(s):
    # Python 3.10 accepts only 3/6 fractional digits; CI emits 7 or more.
    # Normalize solely for arithmetic; all recorded timestamps retain source text.
    normalized = re.sub(r"(?<=:\d{2})\.(\d+)(?=Z|[+-]\d{2}:\d{2}$|$)", lambda m: "." + (m[1] + "000000")[:6], s)
    return datetime.fromisoformat(normalized.replace("Z", "+00:00"))


def digest(b):
    return hashlib.sha256(b).hexdigest()


def fresh(clients, marker, year):
    return dict(
        version=1,
        clients=clients,
        marker=marker,
        year=year,
        offset=0,
        prefix_sha256=digest(b""),
        tail="",
        utf8_tail="",
        line=0,
        boundary=None,
        initial=[],
        responses={},
        latest_event=None,
        pending=None,
        http_status_counts={},
        progress=None,
        errors=0,
        score_events=0,
        truncation_events=0,
        incomplete_responses=0,
        unsupported_bodies=0,
    )


def line_info(line, year):
    line = ANSI.sub("", line)
    match = ISO.match(line)
    if match:
        stamp = match[1]
        line = line[match.end() :]
        basis = "CI timestamp UTC"
    else:
        match = LOCAL.search(line)
        stamp = f'{year}-{match[1].replace(" ","T")}Z' if match else None
        basis = "server timestamp assumed UTC, supplied year" if match else None
    return line, stamp, basis


def consume(s, raw):
    s["line"] += 1
    line, stamp, basis = line_info(raw, s["year"])
    # Decode only the explicit versioned native envelope; never evaluate extras.
    if "TTNATIVEHEX1 " in line:
        encoded = line.split("TTNATIVEHEX1 ", 1)[1].strip()
        try:
            if not re.fullmatch(r"[0-9a-f]+", encoded) or len(encoded) % 2:
                raise ValueError("Invalid native hex envelope")
            envelope = json.loads(bytes.fromhex(encoded).decode("utf-8"))
            if (
                not isinstance(envelope, dict)
                or envelope.get("version") != 1
                or not isinstance(envelope.get("message"), str)
                or envelope.get("name")
                not in ("vllm.entrypoints.serve.utils.request_logger", "vllm.entrypoints.serve.middleware.log_response")
            ):
                raise ValueError("Invalid native record schema")
            line = envelope["message"]
            if stamp is None:
                stamp = datetime.fromtimestamp(envelope["created"], timezone.utc).isoformat()
                basis = "native LogRecord created UTC; no outer CI timestamp"
        except (ValueError, TypeError, KeyError, OverflowError):
            s["truncation_events"] += 1
            return
    if stamp:
        s["latest_event"] = {"timestamp": stamp, "basis": basis, "line": s["line"]}
    if s["boundary"] is None:
        if s["marker"] not in line:
            return
        s["boundary"] = {"line": s["line"], "timestamp": stamp}
    if TRUNC.search(line):
        s["truncation_events"] += 1
        if s["pending"]:
            s["incomplete_responses"] += 1
            s["pending"] = None
        return
    m = RECEIVED.search(line)
    if m and m[1] not in [x["id"] for x in s["initial"]] and len(s["initial"]) < s["clients"]:
        s["initial"].append({"id": m[1], "timestamp": stamp, "line": s["line"]})
    m = PROGRESS.search(line)
    if m:
        s["progress"] = {"completed": int(m[1]), "total": int(m[2]), "timestamp": stamp, "line": s["line"]}
    m = re.search(r'"(?:GET|POST) [^"\r\n]+ HTTP/[^"]+" (\d{3})\b', line)
    if m:
        s["http_status_counts"][m[1]] = s["http_status_counts"].get(m[1], 0) + 1
    if ERROR.search(line):
        s["errors"] += 1
    # These are observations, not proof that a command containing a marker scored.
    if SCORE.search(line):
        s["score_events"] += 1
    marker = "response_body={"
    if marker in line:
        if s["pending"]:
            s["incomplete_responses"] += 1
        text = line.split(marker, 1)[1]
        if not text.lstrip().startswith("{"):
            s["unsupported_bodies"] += 1
            s["pending"] = None
            return
        s["pending"] = {"text": text, "timestamp": stamp, "line": s["line"]}
    elif s["pending"]:
        # A new logger event is not a continuation. CI timestamps alone are OK.
        if re.search(r"\b(?:INFO|WARNING|ERROR|DEBUG) \d\d-\d\d ", line):
            s["incomplete_responses"] += 1
            s["pending"] = None
            return
        s["pending"]["text"] += "\n" + line
    if not s["pending"]:
        return
    pending = s["pending"]
    text = pending["text"].lstrip()
    if len(text) > 16 * 1024 * 1024:
        s["incomplete_responses"] += 1
        s["pending"] = None
        return
    try:
        body, end = json.JSONDecoder().raw_decode(text)
    except ValueError:
        return
    # Stock logger wraps the serialized JSON with an extra pair of braces.
    if text[end:].strip() != "}":
        return
    rid = body.get("id") if isinstance(body, dict) else None
    if rid in [x["id"] for x in s["initial"]]:
        choices = body.get("choices")
        valid = isinstance(choices, list) and len(choices) == 1
        choice = choices[0] if valid and isinstance(choices[0], dict) else {}
        valid = valid and isinstance(choice.get("message"), dict) and choice.get("finish_reason") is not None
        record = {
            "timestamp": pending["timestamp"],
            "line_start": pending["line"],
            "line_end": s["line"],
            "json_sha256": digest(text[:end].encode()),
            "json_characters": end,
            "complete_json_wrapper": True,
            "single_finished_chat_choice": valid,
            "finish_reason": choice.get("finish_reason"),
            "visible_empty": not bool(choice.get("message", {}).get("content")) if valid else None,
        }
        s["responses"].setdefault(rid, []).append(record)
    s["pending"] = None


def observe(s, blob, start, end):
    if len(blob) < s["offset"] or digest(blob[: s["offset"]]) != s["prefix_sha256"]:
        raise ValueError("Snapshot is not append-only relative to state; preserve state and start a separate observer.")
    chunk = bytes.fromhex(s.get("utf8_tail", "")) + blob[s["offset"] :]
    s["utf8_tail"] = ""
    try:
        decoded = chunk.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        if exc.reason != "unexpected end of data":
            raise
        decoded = chunk[: exc.start].decode("utf-8", errors="strict")
        s["utf8_tail"] = chunk[exc.start :].hex()
    text = s["tail"] + decoded
    # Keep incomplete final line; never accept a partially downloaded response.
    lines = re.split(r"\r\n|\n|\r", text)
    s["tail"] = lines.pop()
    for line in lines:
        consume(s, line)
    s["offset"] = len(blob)
    s["prefix_sha256"] = digest(blob)
    latest = s["latest_event"]
    lag = (date(end) - date(latest["timestamp"])).total_seconds() if latest else None
    replies = []
    for request in s["initial"]:
        hits = s["responses"].get(request["id"], [])
        replies.append(
            {
                **request,
                "server_response_records": hits,
                "complete_server_reply_observed": any(h["single_finished_chat_choice"] for h in hits),
                "client_acknowledgement": "unknown",
            }
        )
    return dict(
        boundary=s["boundary"],
        planned_clients=s["clients"],
        initial_requests=replies,
        initial_c_identified=len(replies) == s["clients"],
        all_initial_server_replies_observed=len(replies) == s["clients"]
        and all(x["complete_server_reply_observed"] for x in replies),
        source_coverage_compromised=bool(s["truncation_events"]),
        errors_observed=s["errors"],
        http_status_counts=s["http_status_counts"],
        progress=s["progress"],
        score_markers_observed=s["score_events"],
        no_score_currently="unknown",
        no_score_reason="Absence in an asynchronous snapshot cannot establish current absence of aggregation; observed markers require coordinator context review.",
        truncation_events=s["truncation_events"],
        incomplete_response_records=s["incomplete_responses"],
        pending_response=bool(s["pending"]),
        unterminated_line=bool(s["tail"] or s.get("utf8_tail")),
        fetch_started_at=start,
        fetch_finished_at=end,
        fetch_seconds=(date(end) - date(start)).total_seconds(),
        latest_log_event=latest,
        last_event_lag_seconds=lag,
        snapshot_bytes=len(blob),
        snapshot_sha256=digest(blob),
        scope="Server response construction only; no client acknowledgement, dry pass, cancellation or score decision. Initial IDs require isolated evaluator traffic after the operator-selected boundary.",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot", "state", "output", "boundary-marker", "fetch-start", "fetch-end"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--clients", type=int, default=4)
    p.add_argument("--year", type=int, default=2026)
    a = p.parse_args()
    state = Path(a.state)
    s = json.loads(state.read_text()) if state.exists() else fresh(a.clients, a.boundary_marker, a.year)
    if (s["clients"], s["marker"], s["year"]) != (a.clients, a.boundary_marker, a.year):
        raise ValueError("Observer parameters changed")
    if a.clients < 1 or date(a.fetch_end) < date(a.fetch_start):
        raise ValueError("Invalid clients or fetch interval")
    result = observe(s, Path(a.snapshot).read_bytes(), a.fetch_start, a.fetch_end)
    with Path(a.output).open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    temporary = state.with_suffix(state.suffix + ".tmp")
    temporary.write_text(json.dumps(s))
    temporary.replace(state)
    print(
        json.dumps(
            {
                "output": a.output,
                "initial_identified": result["initial_c_identified"],
                "server_replies": result["all_initial_server_replies_observed"],
                "no_score_currently": "unknown",
            }
        )
    )


if __name__ == "__main__":
    main()
