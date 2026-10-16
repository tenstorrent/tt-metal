"""Download the Qwen3.8-Flash-Next checkpoint from ModelScope at a pinned revision, verified by SHA-256.

    python -m models.demos.blackhole.qwen38_flash_next.tools.download_checkpoint --out /data/Qwen3.8-Flash-Next

The file listing of the revision (path, size, SHA-256 per file) comes from the ModelScope API and is saved next to
the download (``<out>/.download/files.json``; ``tools/verify_checkpoint_files.py`` takes it).  Every file is fetched
as 128 MiB range segments by a pool of threads, written in place into a pre-sized file; a marker per finished
segment makes a restart skip what is done.  A complete file is hashed and compared with the listing; a mismatch
discards its markers so it is fetched again.  Resumable, parallel, 360 GB.  ``--verify-only`` hashes a copy
that is already in place (fetching nothing) and writes the markers, so the next plain run fetches only what is
absent or differs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from models.demos.blackhole.qwen38_flash_next.checkpoint import PINNED_CHECKPOINT_REVISION

MODEL = "Qwen/Qwen3.8-Flash-Next"
# The ModelScope commit of the release (2026-09-04): 145 files, 360,023,351,829 bytes; its safetensors, config,
# tokenizer and chat template are byte-identical to the revision the port pins (checkpoint.PINNED_CHECKPOINT_REVISION,
# which ModelScope no longer lists).  The server checks the files by digest, not by revision label.
RELEASE_REVISION = "2741eec155d03a8ce151b993ccce1a7b1e398d6b"
SEGMENT_BYTES = 128 << 20
LISTING_URL = "https://modelscope.cn/api/v1/models/{model}/repo/files?Revision={revision}&Recursive=true"
FILE_URL = "https://modelscope.cn/models/{model}/resolve/{revision}/{path}"


def fetch_listing(revision: str) -> list[dict]:
    with urllib.request.urlopen(LISTING_URL.format(model=MODEL, revision=revision), timeout=60) as response:
        document = json.load(response)
    files = [item for item in ((document.get("Data") or {}).get("Files") or []) if item.get("Type") == "blob"]
    if not files:
        raise SystemExit(f"ModelScope lists no files for {MODEL} at {revision}: {document.get('Message')}")
    return files


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Download:
    def __init__(self, revision: str, files: list[dict], out: Path, work: Path, threads: int) -> None:
        self.revision, self.out, self.work, self.threads = revision, out, work, threads
        self.files = {item["Path"]: item for item in files}
        self.lock = threading.Lock()
        self.remaining: dict[str, int] = {}
        self.failed: set[str] = set()
        self.done_bytes = 0
        self.inflight = 0
        self.log = (work / "download.log").open("a", buffering=1)
        self.pool = ThreadPoolExecutor(threads)

    def say(self, message: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {message}"
        self.log.write(line + "\n")
        print(line, flush=True)

    def segments(self, size: int) -> int:
        return max(1, (size + SEGMENT_BYTES - 1) // SEGMENT_BYTES)

    def marker(self, name: str, index: int) -> Path:
        return self.work / "segments" / name / f"{index}.done"

    def verified(self, name: str) -> Path:
        return self.work / "verified" / (name + ".ok")

    def fetch_segment(self, name: str, index: int, size: int) -> None:
        start = index * SEGMENT_BYTES
        end = min(size, start + SEGMENT_BYTES) - 1
        want = end - start + 1
        marker = self.marker(name, index)
        if marker.exists():
            return
        path = self.out / name
        url = FILE_URL.format(model=MODEL, revision=self.revision, path=name)
        for attempt in range(8):
            got = 0
            try:
                request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(request, timeout=120) as response:
                    if response.status != 206 and not (response.status == 200 and start == 0 and want == size):
                        raise RuntimeError(f"http {response.status}")
                    descriptor = os.open(path, os.O_WRONLY)
                    try:
                        while chunk := response.read(4 << 20):
                            os.pwrite(descriptor, chunk, start + got)
                            got += len(chunk)
                            with self.lock:
                                self.done_bytes += len(chunk)
                    finally:
                        os.close(descriptor)
                if got != want:
                    raise RuntimeError(f"short read {got} != {want}")
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.touch()
                return
            except Exception as error:  # noqa: BLE001 - network faults are retried
                with self.lock:
                    self.done_bytes -= got
                self.say(f"retry {name} segment {index} attempt {attempt}: {error}")
                time.sleep(min(60, 2**attempt))
        raise RuntimeError(f"segment failed {name} {index}")

    def check_file(self, name: str) -> bool:
        """Hash a complete file against the listing; a match writes its marker."""
        path = self.out / name
        item = self.files[name]
        size = path.stat().st_size
        digest = sha256_file(path) if size == item["Size"] else None
        if digest == item["Sha256"]:
            self.verified(name).parent.mkdir(parents=True, exist_ok=True)
            self.verified(name).write_text(f"{digest}  {name}\n")
            self.say(f"VERIFIED {name} {item['Size']}")
            return True
        self.say(f"BAD {name} size={size} expected {item['Size']} sha256={digest} expected {item['Sha256']}")
        return False

    def finish_file(self, name: str) -> bool:
        if self.check_file(name):
            return True
        self.say(f"{name}: fetching again")
        for index in range(self.segments(self.files[name]["Size"])):
            self.marker(name, index).unlink(missing_ok=True)
        return False

    def verify_only(self) -> int:
        """Hash the files already in place against the listing and write their markers; fetch nothing.

        For a copy that was not made by this tool: a following run without
        ``--verify-only`` then fetches only the files that are absent or differ.
        """
        started = time.time()
        present = [name for name in self.files if not self.verified(name).exists() and (self.out / name).is_file()]
        absent = sorted(name for name in self.files if not (self.out / name).is_file())
        self.say(
            f"verify-only revision={self.revision} files={len(present)} to hash, absent={absent}, threads={self.threads}"
        )
        outcomes = list(self.pool.map(self.check_file, present))
        self.pool.shutdown(wait=True)
        verified = sum(1 for name in self.files if self.verified(name).exists())
        bad = sorted(name for name, ok in zip(present, outcomes) if not ok)
        self.say(
            f"end verified {verified}/{len(self.files)} in {time.time() - started:.0f} s; bad {bad}; absent {absent}"
        )
        return 0 if verified == len(self.files) else 1

    def submit(self, name: str, index: int, size: int) -> None:
        with self.lock:
            self.inflight += 1
        self.pool.submit(self.run_segment, name, index, size)

    def run_segment(self, name: str, index: int, size: int) -> None:
        try:
            try:
                self.fetch_segment(name, index, size)
            except Exception as error:  # noqa: BLE001
                self.say(f"FAILED {name} segment {index}: {error}")
                with self.lock:
                    self.failed.add(name)
                return
            with self.lock:
                self.remaining[name] -= 1
                last = self.remaining[name] == 0
            if last and not self.finish_file(name):
                with self.lock:
                    self.remaining[name] = self.segments(size)
                for again in range(self.segments(size)):
                    self.submit(name, again, size)
        finally:
            with self.lock:
                self.inflight -= 1

    def run(self) -> int:
        started = time.time()
        todo = []
        for name, item in self.files.items():
            if self.verified(name).exists():
                continue
            path = self.out / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists() or path.stat().st_size != item["Size"]:
                with path.open("ab"):
                    pass
                os.truncate(path, item["Size"])
            self.remaining[name] = self.segments(item["Size"])
            todo.append((name, item["Size"]))
        total = sum(item["Size"] for item in self.files.values())
        self.say(
            f"start revision={self.revision} files={len(todo)} of {len(self.files)} threads={self.threads} bytes={total}"
        )
        for name, size in todo:
            for index in range(self.segments(size)):
                self.submit(name, index, size)
        while True:
            time.sleep(15)
            with self.lock:
                inflight, done = self.inflight, self.done_bytes
            elapsed = time.time() - started
            verified = sum(1 for name in self.files if self.verified(name).exists())
            self.say(
                f"progress {done / 1e9:.2f} GB in {elapsed:.0f} s = {done / max(elapsed, 1) / 1e6:.1f} MB/s; verified {verified}/{len(self.files)}"
            )
            if inflight == 0:
                break
        self.pool.shutdown(wait=True)
        verified = sum(1 for name in self.files if self.verified(name).exists())
        self.say(f"end verified {verified}/{len(self.files)}; failed {sorted(self.failed)}")
        return 0 if verified == len(self.files) and not self.failed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="the checkpoint directory to fill")
    parser.add_argument(
        "--revision",
        default=RELEASE_REVISION,
        help=f"the ModelScope commit (default {RELEASE_REVISION}; the port pins {PINNED_CHECKPOINT_REVISION})",
    )
    parser.add_argument("--threads", type=int, default=32, help="parallel range requests (default 32)")
    parser.add_argument("--work", type=Path, default=None, help="markers and log (default <out>/.download)")
    parser.add_argument("--listing-only", action="store_true", help="save the file listing and stop")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="hash the files already in --out against the listing and write their markers; fetch nothing",
    )
    args = parser.parse_args()
    out = args.out.resolve()
    work = (args.work or out / ".download").resolve()
    work.mkdir(parents=True, exist_ok=True)
    files = fetch_listing(args.revision)
    (work / "files.json").write_text(json.dumps({"Data": {"Files": files}}, indent=2, sort_keys=True) + "\n")
    print(
        f"{len(files)} files, {sum(item['Size'] for item in files):,} bytes at {args.revision}; listing {work / 'files.json'}"
    )
    if args.listing_only:
        return 0
    download = Download(args.revision, files, out, work, args.threads)
    return download.verify_only() if args.verify_only else download.run()


if __name__ == "__main__":
    sys.exit(main())
