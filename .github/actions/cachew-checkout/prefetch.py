# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Best-effort object seeding. Final checkout and credential cleanup belong to actions/checkout."""

import base64
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import subprocess
import tempfile
import urllib.parse


ACTION = Path(__file__).resolve().parent
TRUSTED_CACHES = tuple(json.loads((ACTION / "trusted_caches.json").read_text())["origins"])
STAGE_PREFIX = ".cachew-prefetch-"
# Git messages that mean the cache endpoint itself is unusable, as opposed to a
# per-repository or per-ref answer. Only these turn caching off for the job.
TRANSPORT_ERRORS = re.compile(
    rb"could not resolve|failed to connect|connection (?:refused|reset|timed out)|"
    rb"ssl certificate|ssl_connect|ssl_read|handshake|certificate verif|server certificate|rpc failed|early eof|remote end hung up|operation too slow|timed out|"
    rb"returned error: (?:3|5)\d\d",
    re.IGNORECASE,
)
# A slow cache is fine; a stalled one is not. The wall-clock cap only guards
# against a transfer that trickles forever just above the low-speed limit.
HARD_DEADLINE_SECONDS = 3600


class Bypass(Exception):
    """Stop the cache attempt and proceed directly to upstream checkout."""

    def __init__(self, message, transport=False):
        super().__init__(message)
        self.transport = transport


def origin(value):
    url = urllib.parse.urlsplit(value)
    # HTTP is useful for local integration fixtures only. Never weaken TLS for cachew.
    local_http = url.scheme == "http" and url.hostname in ("127.0.0.1", "localhost", "::1")
    if (
        (url.scheme != "https" and not local_http)
        or not url.netloc
        or url.username
        or url.password
        or url.path not in ("", "/")
        or url.query
        or url.fragment
    ):
        raise Bypass("unsupported server URL")
    return f"{url.scheme}://{url.netloc}"


def relative_path(value):
    path = PurePosixPath(value)
    if not value or path.is_absolute() or any(p in ("", ".", "..", ".git") for p in value.split("/")):
        raise Bypass("unsafe repository or submodule path")
    if "\\" in value or any(ord(c) < 32 for c in value):
        raise Bypass("unsafe repository or submodule path")
    return path


def duration(value, maximum):
    seconds = float(value)
    if not math.isfinite(seconds) or not 0 < seconds <= maximum:
        raise Bypass("unsupported cache timeout")
    return seconds


def cache_candidates(value, upstream):
    """Return the trusted cache origins to probe, in order. 'auto' tries every
    trusted origin; only the runner's own cluster answers with a valid
    certificate, and off-VLAN runners fail DNS immediately."""
    if (value or "auto").strip().lower() == "auto":
        return list(TRUSTED_CACHES)
    cache = origin(value)
    loopback_fixture = upstream.startswith("http://") and cache.startswith("http://")
    if cache not in TRUSTED_CACHES and not loopback_fixture:
        raise Bypass("untrusted cache origin")
    return [cache]


def git_version_at_least(executable, minimum):
    try:
        output = subprocess.run(
            [executable, "version"], capture_output=True, timeout=10, env={"PATH": os.environ.get("PATH", "")}
        ).stdout.decode()
    except (OSError, subprocess.SubprocessError):
        return False
    match = re.search(r"(\d+)\.(\d+)", output)
    return bool(match) and (int(match[1]), int(match[2])) >= minimum


def remove_stale_stages(target):
    """A cancelled earlier attempt can leave its private staging directory behind."""
    if not target.is_dir():
        return
    for entry in target.iterdir():
        if entry.name.startswith(STAGE_PREFIX) and entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry, ignore_errors=True)


class Prefetch:
    def __init__(self, environ):
        self.git_executable = environ.get("CACHEW_REAL_GIT", "git")
        for key in ("SSH_KEY", "SSH_KEY_SET", "LFS", "FILTER", "SPARSE_CHECKOUT"):
            value = environ.get(f"CACHEW_{key}", "")
            if value and value.lower() != "false":
                raise Bypass("checkout mode uses upstream directly")
        # GIT_CONFIG_COUNT (2.31) carries the transport settings below; older Git
        # would silently ignore them and use inherited helpers and rewrites.
        if not git_version_at_least(self.git_executable, (2, 31)):
            raise Bypass("Git 2.31 or newer is required")
        self.upstream = origin(environ.get("CACHEW_GITHUB_SERVER_URL") or environ["GITHUB_SERVER_URL"])
        self.caches = cache_candidates(environ.get("CACHEW_CACHEW_URL", "auto"), self.upstream)
        self.cache = None
        self.health_timeout = duration(environ["CACHEW_CACHEW_HEALTH_TIMEOUT"], 30)
        # Seconds without progress before a cache transfer is abandoned.
        self.fetch_timeout = duration(environ["CACHEW_CACHEW_FETCH_TIMEOUT"], 600)
        self.hard_deadline = duration(environ.get("CACHEW_HARD_DEADLINE", str(HARD_DEADLINE_SECONDS)), 86400)
        repository = environ.get("CACHEW_REPOSITORY") or environ["GITHUB_REPOSITORY"]
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", repository, re.ASCII):
            raise Bypass("unsupported repository name")
        relative_path(repository)
        self.url = f"{self.upstream}/{repository}"
        self.ref = environ.get("CACHEW_REF", "").strip()
        if not self.ref:
            self.ref = environ["GITHUB_SHA"] if repository.lower() == environ["GITHUB_REPOSITORY"].lower() else "HEAD"
        if self.ref.startswith(("-", "+")) or any(c.isspace() or c in ":*?[^~\\" for c in self.ref):
            raise Bypass("unsupported prefetch ref")
        self.depth = int(environ.get("CACHEW_FETCH_DEPTH", "1") or "1")
        if self.depth < 0:
            raise Bypass("unsupported fetch depth")
        self.tags = environ.get("CACHEW_FETCH_TAGS", "false").lower() == "true"
        self.submodules = environ.get("CACHEW_SUBMODULES", "false").lower()
        workspace = Path(environ["GITHUB_WORKSPACE"]).resolve()
        target = workspace / environ.get("CACHEW_PATH", "")
        # Resolve existing ancestors before creating anything, including symlinks.
        self.target = target.resolve()
        if not self.target.is_relative_to(workspace):
            raise Bypass("checkout path is outside the workspace")
        if self.target.is_dir() and not self.target.is_symlink():
            remove_stale_stages(self.target)
        if self.target.exists() and (not self.target.is_dir() or any(self.target.iterdir())):
            raise Bypass("existing checkout uses upstream directly")
        self.env = dict(environ)
        for key in list(self.env):
            if key.startswith(("GIT_CONFIG_", "GIT_TRACE")) or key in (
                "GIT_DIR",
                "GIT_WORK_TREE",
                "GIT_INDEX_FILE",
                "GIT_CURL_VERBOSE",
            ):
                del self.env[key]
        self.env.update(
            GIT_CONFIG_NOSYSTEM="1",
            GIT_CONFIG_GLOBAL=os.devnull,
            GIT_TERMINAL_PROMPT="0",
            GIT_ASKPASS="/bin/false",
            GIT_LFS_SKIP_SMUDGE="1",
        )
        self.token = environ.get("CACHEW_TOKEN", "")
        self.modules = 0

    def configure_transport(self):
        """Transport settings for the selected cache only. The token is never
        attached before a trusted origin has answered the liveness probe."""
        config = [
            ("http.followRedirects", "false"),
            ("credential.helper", ""),
            ("fetch.recurseSubmodules", "false"),
            ("http.lowSpeedLimit", "1000"),
            ("http.lowSpeedTime", str(max(1, math.ceil(self.fetch_timeout)))),
        ]
        if self.token:
            auth = base64.b64encode(f"x-access-token:{self.token}".encode()).decode()
            config.append((f"http.{self.cache}/.extraHeader", f"AUTHORIZATION: basic {auth}"))
        self.env["GIT_CONFIG_COUNT"] = str(len(config))
        for index, (key, value) in enumerate(config):
            self.env[f"GIT_CONFIG_KEY_{index}"] = key
            self.env[f"GIT_CONFIG_VALUE_{index}"] = value

    def git(self, gitdir, *arguments):
        command = [self.git_executable, f"--git-dir={gitdir}", *arguments]
        with subprocess.Popen(
            command, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True
        ) as process:
            try:
                output, errors = process.communicate(timeout=self.hard_deadline)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
                raise Bypass("cache operation timed out", transport=True) from None
        if process.returncode:
            # Git's diagnostics may contain credentials or untrusted server text,
            # so they are only classified, never printed.
            raise Bypass("cache Git operation failed", transport=bool(TRANSPORT_ERRORS.search(errors or b"")))
        return output

    def seed(self, gitdir, url, ref, main=False):
        self.git(gitdir, "init", "--bare", str(gitdir))
        self.git(gitdir, "remote", "add", "origin", url)
        cache_url = f"{self.cache}/git/{urllib.parse.urlsplit(self.upstream).netloc}/{url[len(self.upstream) + 1:]}"
        options = ["--force", "--no-recurse-submodules"]
        refs = [ref]
        if main and self.depth == 0:
            refs.extend(("+refs/heads/*:refs/remotes/origin/*", "+refs/tags/*:refs/tags/*"))
        elif main:
            options.append("--tags" if self.tags else "--no-tags")
        else:
            # A raw SHA fetch does not auto-follow tags without a destination.
            refs = [f"{ref}:refs/cachew/gitlink"]
        if self.depth:
            options.append(f"--depth={self.depth}")
        # Explicit URL, never a persisted rewrite. Exactly one fetch invocation.
        self.git(gitdir, "fetch", *options, "--", cache_url, *refs)
        commit = self.git(gitdir, "rev-parse", "FETCH_HEAD^{commit}").decode().strip()
        self.git(gitdir, "update-ref", "--no-deref", "HEAD", commit)
        if not main:
            self.git(gitdir, "update-ref", "-d", "refs/cachew/gitlink")
        self.git(gitdir, "config", "core.bare", "false")
        return commit

    def seed_children(self, gitdir, url, commit, level=0):
        if self.submodules not in ("true", "recursive") or (level and self.submodules != "recursive"):
            return
        blob = f"{commit}:.gitmodules"
        if not self.git(gitdir, "ls-tree", commit, "--", ".gitmodules"):
            return
        entries = self.git(gitdir, "config", f"--blob={blob}", "-z", "--list")
        for entry in entries.split(b"\0"):
            if not entry:
                continue
            key, _, path = entry.decode().partition("\n")
            if not key.startswith("submodule.") or not key.endswith(".path"):
                continue
            name = key[len("submodule.") : -len(".path")]
            relative_path(name)
            relative_path(path)
            tree = self.git(gitdir, "ls-tree", "-z", commit, "--", path)
            metadata, tree_path = tree.rstrip(b"\0").split(b"\t", 1)
            mode, kind, sha = metadata.split()
            if mode != b"160000" or kind != b"commit" or tree_path.decode() != path:
                raise Bypass("unsupported submodule gitlink")
            child_url = self.git(gitdir, "config", f"--blob={blob}", "--get", f"submodule.{name}.url").decode().strip()
            if child_url.startswith(("./", "../")):
                child_url = urllib.parse.urljoin(url + "/", child_url)
            if child_url.startswith("git@github.com:") and self.upstream == "https://github.com":
                child_url = "https://github.com/" + child_url[len("git@github.com:") :]
            if not child_url.startswith(self.upstream + "/"):
                raise Bypass("submodule uses another upstream")
            relative_path(child_url[len(self.upstream) + 1 :])
            self.modules += 1
            if level >= 10 or self.modules > 100:
                raise Bypass("submodule traversal limit reached")
            child_dir = gitdir / "modules" / name
            if child_dir.exists() or any(
                (parent / "config").is_file()
                for parent in child_dir.parents
                if parent.is_relative_to(gitdir / "modules")
            ):
                raise Bypass("overlapping submodule names")
            child_commit = self.seed(child_dir, child_url, sha.decode())
            self.seed_children(child_dir, child_url, child_commit, level + 1)

    def healthy(self, cache):
        # --disable ignores ~/.curlrc, which could otherwise add auth or retries.
        # The process deadline also bounds DNS resolution, unlike a socket timeout.
        try:
            health = subprocess.run(
                [
                    "curl",
                    "--disable",
                    "--silent",
                    "--fail",
                    "--retry",
                    "0",
                    "--max-time",
                    str(self.health_timeout),
                    "--output",
                    os.devnull,
                    "--write-out",
                    "%{http_code}",
                    cache + "/_liveness",
                ],
                env=self.env,
                capture_output=True,
                timeout=self.health_timeout + 0.5,
            )
        except subprocess.TimeoutExpired:
            return False
        return health.returncode == 0 and health.stdout == b"200"

    def probe(self):
        for cache in self.caches:
            if self.healthy(cache):
                self.cache = cache
                self.configure_transport()
                return
        raise Bypass("cache liveness failed")

    def run(self):
        self.target.mkdir(parents=True, exist_ok=True)
        # Stage on the target filesystem; publish only a complete object seed.
        with tempfile.TemporaryDirectory(prefix=STAGE_PREFIX, dir=self.target) as stage:
            gitdir = Path(stage) / ".git"
            commit = self.seed(gitdir, self.url, self.ref, main=True)
            self.seed_children(gitdir, self.url, commit)
            gitdir.rename(self.target / ".git")


def main():
    prefetched = False
    # unknown: no probe happened, so activation probes for itself.
    # true: a trusted cache answered; later per-command failures fall back individually.
    # false: the cache endpoint itself failed, so later dependencies skip it.
    available = "unknown"
    selected = ""
    try:
        session = os.environ.get("CACHEW_SESSION_DIR")
        if session and (Path(session) / "cache-disabled").exists():
            available = "false"
            raise Bypass("cache disabled after an earlier failure")
        attempt = Prefetch(os.environ)
        available = "false"
        attempt.probe()
        available = "true"
        selected = attempt.cache
        attempt.run()
        prefetched = True
        print("Cachew object prefetch complete; continuing with upstream checkout.")
    except Bypass as error:
        if error.transport:
            available = "false"
        print(f"Cachew prefetch bypassed: {error}. Continuing with upstream checkout.")
    except Exception:
        available = "false"
        print("Cachew prefetch unavailable. Continuing immediately with upstream checkout.")
    if output := os.environ.get("GITHUB_OUTPUT"):
        with open(output, "a") as stream:
            stream.write(f"prefetched={str(prefetched).lower()}\n")
            stream.write(f"available={available}\n")
            stream.write(f"cache-url={selected}\n")


if __name__ == "__main__":
    main()
