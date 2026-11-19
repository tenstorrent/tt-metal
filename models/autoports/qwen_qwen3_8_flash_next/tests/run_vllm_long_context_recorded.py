# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Invoke shared P5 accuracy unchanged, retaining requests and streamed responses.

Pass this wrapper's options before ``--`` and the unchanged shared CLI after it.
This CPU-only logger sends no requests of its own and never retries. Its two
temporary hooks record the parsed stream result and tee the underlying response
bytes as they arrive; the shared probe still owns requests, timeouts and scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=repr) + "\n")
    temporary.replace(path)


def source_record(module):
    path = Path(module.__file__).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


class TeeResponse:
    """Forward the exact bytes and response lifetime while retaining a copy."""

    def __init__(self, response, sink, on_bytes):
        self.context = self.response = response
        self.sink = sink
        self.on_bytes = on_bytes

    def __enter__(self):
        self.response = self.context.__enter__()
        return self

    def __exit__(self, *args):
        return self.context.__exit__(*args)

    def __getattr__(self, name):
        return getattr(self.response, name)

    def record(self, data):
        self.sink.write(data)
        self.sink.flush()
        self.on_bytes(len(data))
        return data

    def __iter__(self):
        for raw in self.response:
            yield self.record(raw)

    def read(self, *args, **kwargs):
        return self.record(self.response.read(*args, **kwargs))


def run_probe(probe, shared_args, record_dir):
    """Delegate to a shared module; dependency injection supports CPU fake tests."""
    parsed = probe._build_parser().parse_args(shared_args)
    if parsed.part != "accuracy":
        raise ValueError("This serial evidence wrapper is only for --part accuracy")
    record_dir = Path(record_dir)
    # Exclusive directory creation protects existing attempts, even on failures.
    record_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = record_dir / "manifest.json"
    manifest = dict(
        status="running",
        started_utc=utc_now(),
        pid=os.getpid(),
        wrapper_source=source_record(sys.modules[__name__]),
        shared_sources={"run_long_context": source_record(probe), "served_probe": source_record(probe.sp)},
        shared_arguments=list(shared_args),
        request_count=0,
        responses_recorded=0,
        request_or_scoring_changes=False,
        automatic_retries=False,
    )
    write_json(manifest_path, manifest)
    original_stream = probe.sp.stream_request
    url_module = probe.sp.urllib.request
    original_open = url_module.urlopen
    started = time.monotonic()

    def progress(event):
        event = dict(utc=utc_now(), monotonic_seconds=time.monotonic(), **event)
        with (record_dir / "progress.jsonl").open("a") as handle:
            handle.write(json.dumps(event) + "\n")
        if event["event"] in ("request_started", "request_finished", "request_exception"):
            print("P5_RECORD " + json.dumps(event), flush=True)

    def recorded_stream(url, body, **kwargs):
        manifest["request_count"] += 1
        index = manifest["request_count"]
        prefix = record_dir / f"request_{index:02d}"
        request_start = time.monotonic()
        state = dict(index=index, status="request_started", response_bytes=0, started_utc=utc_now())
        write_json(prefix.with_suffix(".request.json"), dict(url=url, body=body, call_kwargs=kwargs))
        write_json(prefix.with_suffix(".state.json"), state)
        write_json(manifest_path, manifest)
        progress(dict(event="request_started", index=index, request=str(prefix.with_suffix(".request.json"))))
        last_progress = request_start

        def on_bytes(count):
            nonlocal last_progress
            first = state["response_bytes"] == 0
            state["response_bytes"] += count
            now = time.monotonic()
            if first or now - last_progress >= 10:
                if first:
                    state["first_response_byte_seconds"] = now - request_start
                state["elapsed_seconds"] = now - request_start
                write_json(prefix.with_suffix(".state.json"), state)
                progress(dict(event="response_progress", **state))
                last_progress = now

        with prefix.with_suffix(".response.body").open("xb") as sink:

            def logged_open(request, *args, **open_kwargs):
                # This hook is installed only during the delegated generation
                # call; P5's /tokenize, /health and reset calls are untouched.
                wire = dict(
                    url=request.full_url,
                    method=request.get_method(),
                    open_args=args,
                    open_kwargs=open_kwargs,
                )
                data = request.data
                if data is not None:
                    prefix.with_suffix(".request.body").write_bytes(data)
                write_json(prefix.with_suffix(".http.json"), wire)
                try:
                    response = original_open(request, *args, **open_kwargs)
                except HTTPError as error:
                    wire.update(status=error.code, headers=dict(error.headers or {}), http_error=True)
                    write_json(prefix.with_suffix(".http.json"), wire)
                    original_read = error.read

                    def error_read(*read_args, **read_kwargs):
                        raw = original_read(*read_args, **read_kwargs)
                        sink.write(raw)
                        sink.flush()
                        on_bytes(len(raw))
                        return raw

                    # Preserve the same exception and bytes. The shared probe
                    # reads the full body before truncating its error summary.
                    error.read = error_read
                    raise
                wire.update(status=getattr(response, "status", None), headers=dict(response.headers or {}))
                write_json(prefix.with_suffix(".http.json"), wire)
                return TeeResponse(response, sink, on_bytes)

            url_module.urlopen = logged_open
            try:
                result = original_stream(url, body, **kwargs)
                write_json(prefix.with_suffix(".response.json"), result)
                state.update(status="response_recorded", response_error=result.get("error"))
                manifest["responses_recorded"] += 1
                return result
            except BaseException:
                state.update(status="request_exception", exception=traceback.format_exc())
                raise
            finally:
                url_module.urlopen = original_open
                state["elapsed_seconds"] = time.monotonic() - request_start
                write_json(prefix.with_suffix(".state.json"), state)
                write_json(manifest_path, manifest)
                progress(
                    dict(
                        event="request_exception" if state["status"] == "request_exception" else "request_finished",
                        **state,
                    )
                )

    probe.sp.stream_request = recorded_stream
    try:
        exit_code = probe.main(shared_args)
        manifest.update(status="completed", shared_exit_code=exit_code)
        return exit_code
    except BaseException:
        manifest.update(status="exception", exception=traceback.format_exc())
        raise
    finally:
        probe.sp.stream_request = original_stream
        url_module.urlopen = original_open
        manifest.update(
            finished_utc=utc_now(),
            seconds=time.monotonic() - started,
            ttnn_imported=any(name == "ttnn" or name.startswith("ttnn.") for name in sys.modules),
        )
        write_json(manifest_path, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--record-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "readiness_vllm/P5_accuracy_raw",
    )
    parser.add_argument("shared_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    shared_args = args.shared_args[1:] if args.shared_args[:1] == ["--"] else args.shared_args
    from readiness_check import run_long_context

    return run_probe(run_long_context, shared_args, args.record_dir)


if __name__ == "__main__":
    raise SystemExit(main())
