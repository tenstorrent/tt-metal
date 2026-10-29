# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Best-effort object seeding. Final checkout and credential cleanup belong to actions/checkout."""

import base64
import math
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import tempfile
import urllib.parse


class Bypass(Exception):
    """Stop the cache attempt and proceed directly to upstream checkout."""


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


class Prefetch:
    def __init__(self, environ):
        for key in ("SSH_KEY", "LFS", "FILTER", "SPARSE_CHECKOUT"):
            value = environ.get(f"CACHEW_{key}", "")
            if value and value.lower() != "false":
                raise Bypass("checkout mode uses upstream directly")
        self.upstream = origin(environ.get("CACHEW_GITHUB_SERVER_URL") or environ["GITHUB_SERVER_URL"])
        self.cache = origin(environ["CACHEW_CACHEW_URL"])
        trusted_caches = {
            "https://cachew.dev.tenstorrent.net",
            "https://cachew.ci.tenstorrent.net",
            "https://cachew.yyz2.ci.tenstorrent.net",
        }
        loopback_fixture = self.upstream.startswith("http://") and self.cache.startswith("http://")
        if self.cache not in trusted_caches and not loopback_fixture:
            raise Bypass("untrusted cache origin")
        self.health_timeout = duration(environ["CACHEW_CACHEW_HEALTH_TIMEOUT"], 30)
        self.fetch_timeout = duration(environ["CACHEW_CACHEW_FETCH_TIMEOUT"], 600)
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
        config = [("http.followRedirects", "false"), ("credential.helper", ""), ("fetch.recurseSubmodules", "false")]
        token = environ.get("CACHEW_TOKEN", "")
        if token:
            auth = base64.b64encode(f"x-access-token:{token}".encode()).decode()
            config.append((f"http.{self.cache}/.extraHeader", f"AUTHORIZATION: basic {auth}"))
        self.env["GIT_CONFIG_COUNT"] = str(len(config))
        for index, (key, value) in enumerate(config):
            self.env[f"GIT_CONFIG_KEY_{index}"] = key
            self.env[f"GIT_CONFIG_VALUE_{index}"] = value
        self.modules = 0

    def git(self, gitdir, *arguments):
        command = ["git", f"--git-dir={gitdir}", *arguments]
        with subprocess.Popen(
            command, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True
        ) as process:
            try:
                output, _ = process.communicate(timeout=self.fetch_timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
                raise Bypass("cache operation timed out") from None
        if process.returncode:
            # Git's diagnostics may contain credentials or untrusted server text.
            raise Bypass("cache Git operation failed")
        return output

    def seed(self, gitdir, url, ref):
        self.git(gitdir, "init", "--bare", str(gitdir))
        self.git(gitdir, "config", "remote.origin.url", url)
        cache_url = f"{self.cache}/git/{urllib.parse.urlsplit(self.upstream).netloc}/{url[len(self.upstream) + 1:]}"
        options = ["--force", "--no-recurse-submodules", "--tags" if self.tags else "--no-tags"]
        if self.depth:
            options.append(f"--depth={self.depth}")
        # Explicit URL, never a persisted rewrite. Exactly one fetch invocation.
        self.git(gitdir, "fetch", *options, "--", cache_url, ref)
        commit = self.git(gitdir, "rev-parse", "FETCH_HEAD^{commit}").decode().strip()
        self.git(gitdir, "update-ref", "--no-deref", "HEAD", commit)
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

    def run(self):
        # --disable ignores ~/.curlrc, which could otherwise add auth or retries.
        # The process deadline also bounds DNS resolution, unlike a socket timeout.
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
                self.cache + "/_liveness",
            ],
            env=self.env,
            capture_output=True,
            timeout=self.health_timeout + 0.5,
        )
        if health.returncode or health.stdout != b"200":
            raise Bypass("cache liveness failed")
        self.target.mkdir(parents=True, exist_ok=True)
        # Stage on the target filesystem; publish only a complete object seed.
        with tempfile.TemporaryDirectory(prefix=".cachew-prefetch-", dir=self.target) as stage:
            gitdir = Path(stage) / ".git"
            commit = self.seed(gitdir, self.url, self.ref)
            self.seed_children(gitdir, self.url, commit)
            gitdir.rename(self.target / ".git")


def main():
    prefetched = False
    try:
        Prefetch(os.environ).run()
        prefetched = True
        print("Cachew object prefetch complete; continuing with upstream checkout.")
    except Bypass as error:
        print(f"Cachew prefetch bypassed: {error}. Continuing with upstream checkout.")
    except Exception:
        print("Cachew prefetch unavailable. Continuing immediately with upstream checkout.")
    if output := os.environ.get("GITHUB_OUTPUT"):
        with open(output, "a") as stream:
            stream.write(f"prefetched={str(prefetched).lower()}\n")


if __name__ == "__main__":
    main()
