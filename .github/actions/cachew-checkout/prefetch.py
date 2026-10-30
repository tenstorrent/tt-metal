# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Complete checkout from cachew, leaving the repository actions/checkout would create.

Any failure removes everything this attempt created, so actions/checkout (which
runs only then) starts from the same empty target."""

import base64
import json
import math
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import shutil
import signal
import subprocess
import tempfile
import urllib.parse
import uuid


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
SHA = re.compile(r"[0-9a-fA-F]{40}")
HEADS_REFSPEC = "+refs/heads/*:refs/remotes/origin/*"
TAGS_REFSPEC = "+refs/tags/*:refs/tags/*"


class Bypass(Exception):
    """Stop the cache attempt and proceed directly to upstream checkout."""

    def __init__(self, message, transport=False):
        super().__init__(message)
        self.transport = transport


def loopback_http_allowed(environ):
    """Plain HTTP is only for the loopback fixtures in tests/. action.yml clears
    this test-only variable, so no workflow input can select plain HTTP."""
    return environ.get("CACHEW_TEST_ALLOW_LOOPBACK_HTTP") == "1"


def origin(value, environ):
    url = urllib.parse.urlsplit(value)
    # Never weaken TLS for cachew or upstream outside the test fixtures.
    local_http = (
        loopback_http_allowed(environ) and url.scheme == "http" and url.hostname in ("127.0.0.1", "localhost", "::1")
    )
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


def container_path(*parts):
    """Node's path.posix.join, which also collapses '..' segments."""
    return posixpath.normpath(posixpath.join("/github/workspace", *parts))


def duration(value, maximum):
    seconds = float(value)
    if not math.isfinite(seconds) or not 0 < seconds <= maximum:
        raise Bypass("unsupported cache timeout")
    return seconds


def cache_candidates(value, upstream, environ):
    """Return the trusted cache origins to probe, in order. 'auto' tries every
    trusted origin; only the runner's own cluster answers with a valid
    certificate, and off-VLAN runners fail DNS immediately."""
    if (value or "auto").strip().lower() == "auto":
        return list(TRUSTED_CACHES)
    cache = origin(value, environ)
    loopback_fixture = loopback_http_allowed(environ) and upstream.startswith("http://") and cache.startswith("http://")
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
        for key in ("SSH_KEY", "SSH_KEY_SET", "LFS", "SPARSE_CHECKOUT"):
            value = environ.get(f"CACHEW_{key}", "")
            if value and value.lower() != "false":
                raise Bypass("checkout mode uses upstream directly")
        self.filter = environ.get("CACHEW_FILTER", "").strip()
        if self.filter and not re.fullmatch(r"[\w:=+.%-]+", self.filter, re.ASCII):
            raise Bypass("unsupported partial-clone filter")
        # GIT_CONFIG_COUNT (2.31) carries the transport settings below; older Git
        # would silently ignore them and use inherited helpers and rewrites.
        if not git_version_at_least(self.git_executable, (2, 31)):
            raise Bypass("Git 2.31 or newer is required")
        self.upstream = origin(environ.get("CACHEW_GITHUB_SERVER_URL") or environ["GITHUB_SERVER_URL"], environ)
        self.caches = cache_candidates(environ.get("CACHEW_CACHEW_URL", "auto"), self.upstream, environ)
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
        # The same ref and commit as actions/checkout's input helper.
        self.ref = environ.get("CACHEW_REF", "").strip()
        self.commit = ""
        if not self.ref:
            if repository.lower() == environ["GITHUB_REPOSITORY"].lower():
                self.ref = environ.get("GITHUB_REF", "")
                self.commit = environ.get("GITHUB_SHA", "")
                if self.commit and self.ref and not self.ref.startswith("refs/"):
                    self.ref = f"refs/heads/{self.ref}"
        elif SHA.fullmatch(self.ref):
            self.commit, self.ref = self.ref, ""
        if self.commit and not SHA.fullmatch(self.commit):
            raise Bypass("unsupported checkout commit")
        self.commit = self.commit.lower()
        if self.ref.startswith(("-", "+")) or any(c.isspace() or c in ":*?[^~\\" for c in self.ref):
            raise Bypass("unsupported checkout ref")
        self.depth = int(environ.get("CACHEW_FETCH_DEPTH", "1") or "1")
        if self.depth < 0:
            raise Bypass("unsupported fetch depth")
        self.tags = environ.get("CACHEW_FETCH_TAGS", "false").lower() == "true"
        self.submodules = environ.get("CACHEW_SUBMODULES", "false").lower()
        self.persist = environ.get("CACHEW_PERSIST_CREDENTIALS", "true").lower() == "true"
        self.safe_directory = environ.get("CACHEW_SET_SAFE_DIRECTORY", "true").lower() == "true"
        self.runner_temp = environ.get("RUNNER_TEMP", "")
        if self.persist and not self.runner_temp:
            raise Bypass("RUNNER_TEMP is not defined")
        self.organization = None
        try:
            with open(environ["GITHUB_EVENT_PATH"]) as stream:
                owner_id = json.load(stream)["repository"]["owner"]["id"]
            if isinstance(owner_id, int) and not isinstance(owner_id, bool):
                self.organization = owner_id
        except (KeyError, TypeError, OSError, ValueError):
            pass
        # Unresolved, as actions/checkout writes them into persisted configuration.
        self.workspace_path = os.path.abspath(environ["GITHUB_WORKSPACE"])
        self.repository_path = os.path.abspath(os.path.join(self.workspace_path, environ.get("CACHEW_PATH", "")))
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
        # Worktree commands see the caller's Git configuration, as actions/checkout's do,
        # but can never reach the network: a missing object fails the attempt instead.
        self.local_env = {
            key: value for key, value in environ.items() if key not in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE")
        }
        self.local_env.update(GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="Never", GIT_LFS_SKIP_SMUDGE="1")
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
        self.auth = base64.b64encode(f"x-access-token:{self.token}".encode()).decode()
        self.upstream_env = dict(self.env)
        upstream_config = [
            ("credential.helper", ""),
            ("http.lowSpeedLimit", "1000"),
            ("http.lowSpeedTime", str(max(1, math.ceil(self.fetch_timeout)))),
        ]
        if self.token:
            upstream_config.append((f"http.{self.upstream}/.extraHeader", f"AUTHORIZATION: basic {self.auth}"))
        self.upstream_env["GIT_CONFIG_COUNT"] = str(len(upstream_config))
        for index, (key, value) in enumerate(upstream_config):
            self.upstream_env[f"GIT_CONFIG_KEY_{index}"] = key
            self.upstream_env[f"GIT_CONFIG_VALUE_{index}"] = value
        self.modules = 0
        self.refs = {}
        self.want = ""
        self.checkout_target = None
        self.credentials = None

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

    def run_git(self, command, env, message, upstream=False):
        with subprocess.Popen(
            command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True
        ) as process:
            try:
                output, errors = process.communicate(timeout=self.hard_deadline)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
                raise Bypass(f"{message} timed out", transport=not upstream) from None
        if process.returncode:
            # Git's diagnostics may contain credentials or untrusted server text,
            # so they are only classified, never printed.
            transport = not upstream and bool(TRANSPORT_ERRORS.search(errors or b""))
            raise Bypass(f"{message} failed", transport=transport)
        return output

    def git(self, gitdir, *arguments):
        return self.run_git([self.git_executable, f"--git-dir={gitdir}", *arguments], self.env, "cache Git operation")

    def local(self, *arguments, cwd=None):
        command = [self.git_executable, "-c", "protocol.allow=never", "-C", str(cwd or self.target), *arguments]
        return self.run_git(command, self.local_env, "local checkout")

    def ls_remote(self, *arguments):
        """The only upstream contact: names resolve there, objects come from the cache."""
        output = self.run_git(
            [self.git_executable, "ls-remote", *arguments], self.upstream_env, "upstream ref lookup", upstream=True
        )
        refs, symrefs = {}, {}
        for line in output.decode().splitlines():
            value, _, name = line.partition("\t")
            if value.startswith("ref: "):
                symrefs[name] = value[len("ref: ") :]
            elif SHA.fullmatch(value):
                refs[name] = value
        return refs, symrefs

    def lookup(self, ref, *options):
        sha = self.ls_remote("--refs", *options, self.url, ref)[0].get(ref)
        if not sha:
            raise Bypass("ref not found upstream")
        return sha

    def resolve(self):
        """The local refs and checkout target that actions/checkout's fetch and
        getCheckoutInfo produce for the same input, keyed by object name."""
        ref, commit = self.ref, self.commit
        if not ref and not commit:
            refs, symrefs = self.ls_remote("--symref", self.url, "HEAD")
            ref = symrefs.get("HEAD", "")
            if not ref.startswith("refs/heads/") or "HEAD" not in refs:
                raise Bypass("default branch not found upstream")
            self.ref, commit = ref, refs["HEAD"]
        lower = ref.lower()
        if lower.startswith("refs/heads/"):
            branch = ref[len("refs/heads/") :]
            commit = commit or self.lookup(ref, "--heads")
            self.refs[f"refs/remotes/origin/{branch}"] = commit
            self.want, self.checkout_target = commit, (branch, f"refs/remotes/origin/{branch}")
        elif lower.startswith("refs/pull/"):
            local = f"refs/remotes/pull/{ref[len('refs/pull/'):]}"
            self.refs[local] = commit or self.lookup(ref)
            self.want, self.checkout_target = self.refs[local], (local, None)
        elif lower.startswith("refs/tags/"):
            # Tags are fetched by name; the commit only verifies them.
            self.refs[ref] = self.lookup(ref, "--tags")
            self.want, self.checkout_target = self.refs[ref], (ref, None)
        elif lower.startswith("refs/"):
            if commit:
                self.want, self.checkout_target = commit, (commit, None)
            else:
                self.refs[ref] = self.lookup(ref)
                self.want, self.checkout_target = self.refs[ref], (ref, None)
        elif ref:
            # The refspecs refs/heads/<ref>* and refs/tags/<ref>*, then a branch before a tag.
            listed = self.ls_remote("--refs", "--heads", "--tags", self.url)[0]
            for name, sha in listed.items():
                if name.startswith(f"refs/heads/{ref}"):
                    self.refs[f"refs/remotes/origin/{name[len('refs/heads/'):]}"] = sha
                elif name.startswith(f"refs/tags/{ref}") and (not self.tags or name == f"refs/tags/{ref}"):
                    self.refs[name] = sha
            if f"refs/remotes/origin/{ref}" in self.refs:
                self.want = self.refs[f"refs/remotes/origin/{ref}"]
                self.checkout_target = (ref, f"refs/remotes/origin/{ref}")
            elif f"refs/tags/{ref}" in self.refs:
                self.want, self.checkout_target = self.refs[f"refs/tags/{ref}"], (f"refs/tags/{ref}", None)
            else:
                raise Bypass("branch or tag not found upstream")
        else:
            self.want, self.checkout_target = commit, (commit, None)
        self.commit = commit

    def cache_url(self, url):
        return f"{self.cache}/git/{urllib.parse.urlsplit(self.upstream).netloc}/{url[len(self.upstream) + 1:]}"

    def seed_main(self, gitdir):
        self.git(gitdir, "init", "--bare", str(gitdir))
        self.git(gitdir, "remote", "add", "origin", self.url)
        cache_url = self.cache_url(self.url)
        options = ["--force", "--no-recurse-submodules", "--no-tags"]
        refspecs = [HEADS_REFSPEC, TAGS_REFSPEC] if self.depth == 0 else [TAGS_REFSPEC] if self.tags else []
        if self.depth:
            options.append(f"--depth={self.depth}")
        # Explicit URL, never a persisted rewrite.
        if self.filter:
            # Standard checkout passes the filter to the main repository only. The
            # worktree checkout cannot fetch lazily, so seed the complete tip first;
            # the filtered fetch then adds only history.
            self.git(
                gitdir,
                "fetch",
                "--force",
                "--no-recurse-submodules",
                "--no-tags",
                "--depth=1",
                "--",
                cache_url,
                self.want,
            )
            # Lazy fetches belong to upstream, exactly as standard checkout configures them.
            self.git(gitdir, "config", "remote.origin.promisor", "true")
            self.git(gitdir, "config", "remote.origin.partialclonefilter", self.filter)
            options.append(f"--filter={self.filter}")
            if not self.depth:
                options.append("--unshallow")
        wants = sorted({self.want, *self.refs.values()})
        self.git(gitdir, "fetch", *options, "--", cache_url, *refspecs, *wants)
        if self.filter:
            # A filtered fetch from a URL registers that URL as a promisor remote.
            section = f"remote.{cache_url}."
            if any(
                line.startswith(section)
                for line in self.git(gitdir, "config", "--local", "--list").decode().splitlines()
            ):
                self.git(gitdir, "config", "--remove-section", section[:-1])
        # Upstream's answer wins over the cache's copy of the same name.
        for name, sha in self.refs.items():
            self.git(gitdir, "update-ref", name, sha)
        commit = self.git(gitdir, "rev-parse", f"{self.want}^{{commit}}").decode().strip()
        if self.commit and commit != self.commit:
            raise Bypass("ref does not point to the expected commit")
        self.git(gitdir, "config", "core.bare", "false")
        self.git(gitdir, "config", "core.logallrefupdates", "true")
        return commit

    def seed(self, gitdir, url, sha):
        self.git(gitdir, "init", "--bare", str(gitdir))
        self.git(gitdir, "remote", "add", "origin", url)
        options = ["--force", "--no-recurse-submodules"]
        if self.depth:
            options.append(f"--depth={self.depth}")
        # A raw SHA fetch does not auto-follow tags without a destination.
        self.git(gitdir, "fetch", *options, "--", self.cache_url(url), f"{sha}:refs/cachew/gitlink")
        commit = self.git(gitdir, "rev-parse", "refs/cachew/gitlink^{commit}").decode().strip()
        self.git(gitdir, "update-ref", "--no-deref", "HEAD", commit)
        self.git(gitdir, "update-ref", "-d", "refs/cachew/gitlink")
        self.git(gitdir, "config", "core.bare", "false")
        self.git(gitdir, "config", "core.logallrefupdates", "true")
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

    def persist_credentials(self):
        """actions/checkout's persisted token layout: a RUNNER_TEMP credentials file
        included only for this repository's Git directories, host and container paths."""
        self.credentials = os.path.join(self.runner_temp, f"git-credentials-{uuid.uuid4()}.config")
        placeholder = "AUTHORIZATION: basic ***"
        # The token never appears in a command line.
        self.local("config", "--file", self.credentials, f"http.{self.upstream}/.extraheader", placeholder)
        content = Path(self.credentials).read_text()
        if content.count(placeholder) != 1:
            raise Bypass("unable to write checkout credentials")
        Path(self.credentials).write_text(content.replace(placeholder, f"AUTHORIZATION: basic {self.auth}"))
        container_credentials = f"/github/runner_temp/{os.path.basename(self.credentials)}"
        gitdir = f"{self.repository_path}/.git"
        relative = os.path.relpath(self.repository_path, self.workspace_path)
        container_gitdir = container_path(relative, ".git")
        for key, value in (
            (f"includeIf.gitdir:{gitdir}.path", self.credentials),
            (f"includeIf.gitdir:{gitdir}/worktrees/*.path", self.credentials),
            (f"includeIf.gitdir:{container_gitdir}.path", container_credentials),
            (f"includeIf.gitdir:{container_gitdir}/worktrees/*.path", container_credentials),
        ):
            self.local("config", "--local", key, value)

    def persist_submodule_credentials(self, recursive):
        container_credentials = f"/github/runner_temp/{os.path.basename(self.credentials)}"
        listed = self.local(
            "submodule",
            "foreach",
            *recursive,
            "git config --local --show-origin --name-only --get-regexp remote.origin.url",
        ).decode()
        for config in re.findall(r"^file:([^\t]+)\tremote\.origin\.url", listed, re.M):
            gitdir = os.path.dirname(config)
            container = container_path(os.path.relpath(gitdir, self.workspace_path))
            self.local("config", "--file", config, f"includeIf.gitdir:{gitdir}.path", self.credentials)
            self.local("config", "--file", config, f"includeIf.gitdir:{container}.path", container_credentials)
        rewrites = [f"git@{urllib.parse.urlsplit(self.upstream).hostname}:"]
        if self.organization is not None:
            rewrites.append(f"org-{self.organization}@github.com:")
        for rewrite in rewrites:
            command = f"git config --local --add 'url.{self.upstream}/.insteadOf' '{rewrite}'"
            self.local("submodule", "foreach", *recursive, command)

    def finish(self):
        """actions/checkout's steps after its fetch, offline."""
        if self.safe_directory:
            # GIT_CONFIG_GLOBAL is the session's checkout.config; activation keeps the entry.
            self.local("config", "--global", "--add", "safe.directory", self.repository_path)
        self.local("config", "--local", "gc.auto", "0")
        if self.persist:
            self.persist_credentials()
        self.local("sparse-checkout", "disable")
        # Disabling leaves this behind even in a pristine repository; checkout removes it too.
        if self.local("config", "--local", "--get", "--default=", "extensions.worktreeConfig").strip():
            self.local("config", "--local", "--unset-all", "extensions.worktreeConfig")
        branch, start = self.checkout_target
        self.local("checkout", "--force", *(("-B", branch, start) if start else (branch,)))
        if self.submodules in ("true", "recursive"):
            recursive = ["--recursive"] if self.submodules == "recursive" else []
            depth = [f"--depth={self.depth}"] if self.depth else []
            self.local("submodule", "sync", *recursive)
            self.local("submodule", "update", "--init", "--force", *depth, *recursive)
            self.local("submodule", "foreach", *recursive, "git config --local gc.auto 0")
            if self.persist:
                self.persist_submodule_credentials(recursive)
        return self.local("log", "-1", "--format=%H").decode().strip()

    def discard(self):
        """Leave the empty target actions/checkout would have found."""
        if self.credentials:
            Path(self.credentials).unlink(missing_ok=True)
        if not self.target.is_dir():
            return
        for entry in self.target.iterdir():
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink(missing_ok=True)

    def run(self):
        self.target.mkdir(parents=True, exist_ok=True)
        try:
            self.resolve()
            # Stage on the target filesystem; publish only a complete object seed.
            with tempfile.TemporaryDirectory(prefix=STAGE_PREFIX, dir=self.target) as stage:
                gitdir = Path(stage) / ".git"
                commit = self.seed_main(gitdir)
                self.seed_children(gitdir, self.url, commit)
                gitdir.rename(self.target / ".git")
            return self.finish()
        except BaseException:
            self.discard()
            raise


def main():
    prefetched = False
    outputs = {}
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
        commit = attempt.run()
        prefetched = True
        # The same outputs as actions/checkout, which does not run.
        outputs = {"ref": attempt.ref, "commit": commit}
        print(f"Checked out {commit} from cachew; upstream checkout is not needed.")
    except Bypass as error:
        if error.transport:
            available = "false"
        print(f"Cachew checkout bypassed: {error}. Continuing with upstream checkout.")
    except Exception:
        available = "false"
        print("Cachew checkout unavailable. Continuing immediately with upstream checkout.")
    if output := os.environ.get("GITHUB_OUTPUT"):
        with open(output, "a") as stream:
            stream.write(f"prefetched={str(prefetched).lower()}\n")
            stream.write(f"available={available}\n")
            stream.write(f"cache-url={selected}\n")
            for key, value in outputs.items():
                stream.write(f"{key}={value}\n")


if __name__ == "__main__":
    main()
