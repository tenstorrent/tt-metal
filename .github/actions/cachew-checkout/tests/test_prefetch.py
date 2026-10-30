# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Real smart-HTTP repositories, fault injection, and the pinned checkout main/post bundle."""

import base64
from contextlib import redirect_stdout
import importlib.util
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import re
import socket
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import urllib.parse


ACTION = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prefetch", ACTION / "prefetch.py")
prefetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prefetch)
CHECKOUT_JS = os.environ.get("CHECKOUT_ACTION_JS")
TOKEN = "cachew-test-token-do-not-persist"


def git(path, *arguments, env=None):
    executable = (env or os.environ).get("CACHEW_REAL_GIT", "git")
    return (
        subprocess.check_output([executable, "-C", str(path), *arguments], env=env, stderr=subprocess.PIPE)
        .decode()
        .strip()
    )


def runner_file(path):
    values = {}
    lines = iter(path.read_text().splitlines())
    for line in lines:
        if "<<" in line:
            key, delimiter = line.split("<<", 1)
            content = []
            for item in lines:
                if item == delimiter:
                    break
                content.append(item)
            values[key] = "\n".join(content)
        elif "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


class GitServer:
    def __init__(self, project_root, upstream_host=None, real_git=None):
        self.root = project_root
        self.real_git = real_git or os.environ.get("CACHEW_REAL_GIT") or shutil.which("git")
        self.upstream_host = upstream_host
        self.requests = []
        self.health = 200
        self.health_delay = 0
        self.fail_repo = None
        self.fail_post = False
        self.fail_status = 503
        self.disconnect = False
        self.fetch_delay = 0
        self.block_submodules = False
        self.required_auth = {}
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                self.serve()

            def do_POST(self):
                self.serve()

            def serve(self):
                url = urllib.parse.urlsplit(self.path)
                owner.requests.append((self.command, url.path, self.headers.get("Authorization")))
                if url.path == "/_liveness":
                    time.sleep(owner.health_delay)
                    self.send_response(owner.health)
                    if owner.health == 302:
                        self.send_header("Location", "/redirected-health")
                    self.end_headers()
                    return
                path = url.path
                if owner.upstream_host:
                    prefix = f"/git/{owner.upstream_host}/"
                    if not path.startswith(prefix):
                        self.send_error(404)
                        return
                    path = "/" + path[len(prefix) :]
                repo = path.split("/")[2]
                if (
                    repo in owner.required_auth
                    and self.headers.get("Authorization", "").lower() != owner.required_auth[repo].lower()
                ):
                    self.send_error(401)
                    return
                if (owner.fail_repo == repo and (not owner.fail_post or self.command == "POST")) or (
                    owner.block_submodules and repo != "main"
                ):
                    time.sleep(owner.fetch_delay)
                    if owner.disconnect:
                        self.connection.shutdown(socket.SHUT_RDWR)
                        self.connection.close()
                    else:
                        self.send_response(owner.fail_status)
                        if owner.fail_status == 302:
                            self.send_header("Location", "/redirected-fetch")
                        self.end_headers()
                    return
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                env = dict(
                    os.environ,
                    GIT_PROJECT_ROOT=str(owner.root),
                    GIT_HTTP_EXPORT_ALL="1",
                    PATH_INFO=path,
                    REQUEST_METHOD=self.command,
                    QUERY_STRING=url.query,
                    CONTENT_TYPE=self.headers.get("Content-Type", ""),
                    CONTENT_LENGTH=str(len(body)),
                    SERVER_PROTOCOL="HTTP/1.1",
                    REMOTE_USER="fixture",
                    GIT_CONFIG_GLOBAL=os.devnull,
                    GIT_CONFIG_NOSYSTEM="1",
                )
                response = subprocess.check_output(
                    [owner.real_git, "http-backend"], input=body, env=env, stderr=subprocess.PIPE
                )
                headers, payload = response.split(b"\r\n\r\n", 1)
                status = 200
                parsed = []
                for line in headers.decode().splitlines():
                    key, value = line.split(":", 1)
                    if key.lower() == "status":
                        status = int(value.strip().split()[0])
                    else:
                        parsed.append((key, value.strip()))
                self.send_response(status)
                for key, value in parsed:
                    self.send_header(key, value)
                self.end_headers()
                try:
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def discoveries(self):
        return [path for method, path, _ in self.requests if method == "GET" and path.endswith("/info/refs")]


class PrefetchTests(unittest.TestCase):
    def setUp(self):
        inherited = dict(os.environ)
        real_git = inherited.get("CACHEW_REAL_GIT") or shutil.which("git")
        real_git = str(Path(real_git).resolve())
        for key in list(inherited):
            if key.startswith("CACHEW_"):
                del inherited[key]
        self.temp = tempfile.TemporaryDirectory(prefix="cachew-checkout-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.home = self.root / "home"
        self.home.mkdir()
        self.runtime = self.root / "runtime"
        self.runtime.mkdir()
        binaries = self.root / "bin"
        binaries.mkdir()
        (binaries / "git").symlink_to(real_git)
        self.project_root = self.root / "repositories"
        self.project_root.mkdir()
        self.upstream = GitServer(self.project_root, real_git=real_git)
        self.addCleanup(self.upstream.close)
        self.cache = GitServer(self.project_root, urllib.parse.urlsplit(self.upstream.url).netloc, real_git=real_git)
        self.addCleanup(self.cache.close)
        self.env = dict(
            inherited,
            HOME=str(self.home),
            XDG_CONFIG_HOME=str(self.home / ".config"),
            PATH=str(binaries) + os.pathsep + inherited.get("PATH", ""),
            GIT_CONFIG_NOSYSTEM="1",
            GITHUB_WORKSPACE=str(self.workspace),
            GITHUB_SERVER_URL=self.upstream.url,
            GITHUB_REPOSITORY="org/main",
            GITHUB_REF="refs/heads/main",
            GITHUB_EVENT_NAME="push",
            RUNNER_TEMP=str(self.runtime),
            GITHUB_OUTPUT=str(self.root / "output"),
            GITHUB_STATE=str(self.root / "state"),
            GITHUB_ENV=str(self.root / "env"),
            GITHUB_PATH=str(self.root / "path"),
            GITHUB_EVENT_PATH=str(self.root / "event.json"),
            CACHEW_CACHEW_URL=self.cache.url,
            CACHEW_CACHEW_HEALTH_TIMEOUT="0.1",
            CACHEW_CACHEW_FETCH_TIMEOUT="5",
            CACHEW_REPOSITORY="org/main",
            CACHEW_REF="main",
            CACHEW_PATH="checkout",
            CACHEW_FETCH_DEPTH="1",
            CACHEW_FETCH_TAGS="true",
            CACHEW_SUBMODULES="recursive",
            CACHEW_TOKEN=TOKEN,
            CACHEW_REAL_GIT=real_git,
        )
        self.env.pop("GIT_CONFIG_GLOBAL", None)
        (self.home / ".gitconfig").write_text("[user]\n\tname = Fixture\n\temail = fixture@example.com\n")
        (self.root / "event.json").write_text(json.dumps({"repository": {"default_branch": "main"}}))
        for name in ("output", "state", "env", "path"):
            (self.root / name).touch()
        grand = self.repository("grand")
        child = self.repository("child", [("nested-name", "deep/grand", "../grand", grand)])
        self.sha = self.repository("main", [("child-name", "deps/child", "../child", child)])
        self.env["GITHUB_SHA"] = self.sha
        self.target = self.workspace / "checkout"

    def repository(self, name, children=()):
        path = self.project_root / "org" / name
        path.mkdir(parents=True)
        git(path, "init", "--initial-branch=main", env=self.env)
        (path / "content.txt").write_text(name)
        git(path, "add", "content.txt", env=self.env)
        git(path, "commit", "-m", "initial", env=self.env)
        if children:
            modules = []
            for module_name, module_path, url, sha in children:
                modules.append(f'[submodule "{module_name}"]\n\tpath = {module_path}\n\turl = {url}\n')
                git(path, "update-index", "--add", "--cacheinfo", f"160000,{sha},{module_path}", env=self.env)
            (path / ".gitmodules").write_text("".join(modules))
            git(path, "add", ".gitmodules", env=self.env)
            git(path, "commit", "-m", "submodules", env=self.env)
        git(path, "tag", "v1.0.0", env=self.env)
        return git(path, "rev-parse", "HEAD", env=self.env)

    def attempt(self):
        capture = io.StringIO()
        with patch.dict(os.environ, self.env, clear=True), redirect_stdout(capture):
            prefetch.main()
        self.assertNotIn(TOKEN, capture.getvalue())
        self.assertNotIn(base64.b64encode(f"x-access-token:{TOKEN}".encode()).decode(), capture.getvalue())
        return runner_file(self.root / "output")["prefetched"] == "true"

    def assert_single_cache_attempts(self, repositories):
        paths = self.cache.discoveries()
        self.assertEqual([path.split("/")[-3] for path in paths], repositories)
        self.assertEqual(sum(path == "/_liveness" for _, path, _ in self.cache.requests), 1)
        for _, path, auth in self.cache.requests:
            if path == "/_liveness":
                self.assertIsNone(auth)
            else:
                expected = "basic " + base64.b64encode(f"x-access-token:{TOKEN}".encode()).decode()
                self.assertEqual(auth.lower(), expected.lower())

    def assert_no_cache_credentials(self):
        encoded = base64.b64encode(f"x-access-token:{TOKEN}".encode())
        configs = [self.home / ".gitconfig"]
        if self.target.exists():
            configs.extend(self.target.rglob("config"))
        for path in configs:
            data = path.read_bytes()
            self.assertNotIn(TOKEN.encode(), data)
            self.assertNotIn(encoded, data)
            self.assertNotIn(self.cache.url.encode(), data)

    def checkout(self, **inputs):
        env = dict(self.env)
        for name, value in {
            "repository": "org/main",
            "ref": self.env["CACHEW_REF"],
            "token": TOKEN,
            "path": "checkout",
            "fetch-depth": self.env["CACHEW_FETCH_DEPTH"],
            "fetch-tags": self.env["CACHEW_FETCH_TAGS"],
            "submodules": self.env["CACHEW_SUBMODULES"],
            "clean": "true",
            "persist-credentials": "true",
            "set-safe-directory": "false",
            **inputs,
        }.items():
            env["INPUT_" + name.upper()] = value
        before = list(self.cache.requests)
        result = subprocess.run(
            ["node", CHECKOUT_JS], cwd=self.workspace, env=env, capture_output=True, text=True, timeout=30
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.cache.requests, before, "official checkout must never use the cache")
        self.assertEqual(git(self.target, "rev-parse", "HEAD"), self.sha)
        if self.env["CACHEW_SUBMODULES"] == "recursive":
            self.assertEqual((self.target / "deps/child/deep/grand/content.txt").read_text(), "grand")
        self.assertEqual(runner_file(self.root / "output")["commit"], self.sha)
        # Run the real action's registered post handler using its saved runner state.
        env.update({"STATE_" + key: value for key, value in runner_file(self.root / "state").items()})
        post = subprocess.run(
            ["node", CHECKOUT_JS], cwd=self.workspace, env=env, capture_output=True, text=True, timeout=30
        )
        self.assertEqual(post.returncode, 0, post.stdout + post.stderr)
        self.assert_no_cache_credentials()
        for path in self.runtime.rglob("*"):
            if path.is_file():
                self.assertNotIn(TOKEN.encode(), path.read_bytes())
                self.assertNotIn(base64.b64encode(f"x-access-token:{TOKEN}".encode()), path.read_bytes())

    def test_success_seeds_exact_recursive_commits_without_persisting_transport(self):
        self.assertTrue(self.attempt())
        self.assert_single_cache_attempts(["main", "child", "grand"])
        self.assertEqual(git(self.target, "rev-parse", "HEAD"), self.sha)
        self.assertTrue((self.target / ".git/modules/child-name/modules/nested-name/objects").is_dir())
        self.assertEqual(git(self.target, "remote", "get-url", "origin"), self.upstream.url + "/org/main")
        self.assertEqual(
            git(self.target, "config", "--get", "remote.origin.fetch"), "+refs/heads/*:refs/remotes/origin/*"
        )
        self.assertEqual(
            git(self.target, "--git-dir=.git/modules/child-name", "config", "--get", "remote.origin.fetch"),
            "+refs/heads/*:refs/remotes/origin/*",
        )
        self.assert_no_cache_credentials()

    def test_submodules_auto_follow_tags_independently_of_main_fetch_tags(self):
        child = self.project_root / "org/child"
        git(child, "checkout", "-b", "unrelated", env=self.env)
        (child / "other.txt").write_text("unrelated tagged object")
        git(child, "add", "other.txt", env=self.env)
        git(child, "commit", "-m", "unrelated", env=self.env)
        git(child, "tag", "unrelated-tag", env=self.env)
        git(child, "branch", "-f", "main", "HEAD", env=self.env)
        git(child, "checkout", "main", env=self.env)
        for setting in ("false", "true"):
            with self.subTest(main_fetch_tags=setting):
                self.env["CACHEW_FETCH_TAGS"] = setting
                self.assertTrue(self.attempt())
                self.assertEqual(git(self.target, "--git-dir=.git/modules/child-name", "tag", "--list"), "v1.0.0")
                self.assertEqual(git(self.target, "tag", "--list"), "v1.0.0" if setting == "true" else "")
                self.assert_single_cache_attempts(["main", "child", "grand"])
                shutil.rmtree(self.target)
                self.cache.requests.clear()

    def test_depth_zero_seeds_other_branches_and_tags_in_the_one_main_fetch(self):
        source = self.project_root / "org/main"
        git(source, "checkout", "-b", "another-branch", env=self.env)
        (source / "other.txt").write_text("unique branch object")
        git(source, "add", "other.txt", env=self.env)
        git(source, "commit", "-m", "other branch", env=self.env)
        other = git(source, "rev-parse", "HEAD")
        git(source, "tag", "other-tag", env=self.env)
        git(source, "checkout", "main", env=self.env)
        self.env.update(CACHEW_FETCH_DEPTH="0", CACHEW_FETCH_TAGS="false")
        self.assertTrue(self.attempt())
        self.assertEqual(git(self.target, "rev-parse", "HEAD"), self.sha)
        self.assertEqual(git(self.target, "rev-parse", "refs/remotes/origin/another-branch"), other)
        self.assertEqual(git(self.target, "rev-parse", "refs/tags/other-tag"), other)
        self.assert_single_cache_attempts(["main", "child", "grand"])

    @unittest.skipUnless(CHECKOUT_JS, "set CHECKOUT_ACTION_JS to the pinned checkout dist/index.js")
    def test_pinned_checkout_reuses_all_objects_and_cleans_up_auth(self):
        self.assertTrue(self.attempt())

        def objects():
            return {
                p: (p.stat().st_ino, p.stat().st_size)
                for p in self.target.rglob("*")
                if p.is_file() and "objects" in p.parts
            }

        seeded_objects = objects()
        self.assertTrue(seeded_objects)
        self.assertEqual(len(list(self.target.rglob("objects"))), 3)
        self.upstream.block_submodules = True
        self.checkout()
        self.assertEqual(seeded_objects, objects())
        self.assertTrue(all("/org/main/" in path for _, path, _ in self.upstream.requests))
        self.assert_single_cache_attempts(["main", "child", "grand"])

    @unittest.skipUnless(CHECKOUT_JS, "set CHECKOUT_ACTION_JS to the pinned checkout dist/index.js")
    def test_health_failure_immediately_uses_upstream(self):
        self.cache.health = 503
        self.assertFalse(self.attempt())
        self.assert_single_cache_attempts([])
        self.assertFalse(self.target.exists())
        self.checkout()

    def test_health_redirect_is_not_followed(self):
        self.cache.health = 302
        self.assertFalse(self.attempt())
        self.assertEqual([p for _, p, _ in self.cache.requests], ["/_liveness"])

    def test_curl_configuration_cannot_add_health_auth_or_retry(self):
        (self.home / ".curlrc").write_text('header = "Authorization: unsafe-header"\nretry = 3\nlocation\n')
        self.cache.health = 503
        self.assertFalse(self.attempt())
        self.assert_single_cache_attempts([])

    def test_git_auth_failure_and_redirect_are_not_retried(self):
        self.cache.fail_repo = "main"
        for status in (401, 302):
            with self.subTest(status=status):
                self.cache.requests.clear()
                self.cache.fail_status = status
                self.assertFalse(self.attempt())
                self.assert_single_cache_attempts(["main"])
                self.assertEqual(len(self.cache.requests), 2)
                self.assertEqual(self.upstream.requests, [])

    def test_only_http_200_is_live(self):
        self.cache.health = 204
        self.assertFalse(self.attempt())
        self.assert_single_cache_attempts([])

    def test_connection_refusal_is_not_retried(self):
        self.env["CACHEW_CACHEW_URL"] = "http://127.0.0.1:1"
        self.assertFalse(self.attempt())
        self.assertFalse(self.target.exists())

    def test_untrusted_cache_origins_cannot_receive_checkout_token(self):
        for url in (
            "https://example.com",
            "https://cachew.dev.tenstorrent.net:444",
            "https://cachew.dev.tenstorrent.net.evil.test",
        ):
            self.env["CACHEW_CACHEW_URL"] = url
            self.assertFalse(self.attempt())
        self.env.update(GITHUB_SERVER_URL="https://github.com", CACHEW_CACHEW_URL=self.cache.url)
        self.assertFalse(self.attempt())
        self.assertEqual(self.cache.requests, [])
        self.assertFalse(self.target.exists())

    def test_health_timeout_is_not_retried(self):
        self.cache.health_delay = 0.4
        started = time.monotonic()
        self.assertFalse(self.attempt())
        self.assertLess(time.monotonic() - started, 0.35)
        self.assert_single_cache_attempts([])

    @unittest.skipUnless(CHECKOUT_JS, "set CHECKOUT_ACTION_JS to the pinned checkout dist/index.js")
    def test_main_failure_immediately_uses_upstream(self):
        self.cache.fail_repo = "main"
        self.assertFalse(self.attempt())
        self.assert_single_cache_attempts(["main"])
        self.assertEqual(list(self.target.iterdir()), [])
        self.checkout()

    @unittest.skipUnless(CHECKOUT_JS, "set CHECKOUT_ACTION_JS to the pinned checkout dist/index.js")
    def test_nested_failure_stops_cache_traversal_and_uses_upstream(self):
        repository = self.project_root / "org/main"
        sibling_sha = git(self.project_root / "org/grand", "rev-parse", "HEAD")
        with (repository / ".gitmodules").open("a") as modules:
            modules.write('[submodule "later-sibling"]\n\tpath = deps/sibling\n\turl = ../grand\n')
        git(repository, "update-index", "--add", "--cacheinfo", f"160000,{sibling_sha},deps/sibling", env=self.env)
        git(repository, "add", ".gitmodules", env=self.env)
        git(repository, "commit", "-m", "later sibling", env=self.env)
        self.sha = git(repository, "rev-parse", "HEAD")
        self.env["GITHUB_SHA"] = self.sha
        self.cache.fail_repo = "grand"
        self.assertFalse(self.attempt())
        self.assert_single_cache_attempts(["main", "child", "grand"])
        self.assertEqual(list(self.target.iterdir()), [])
        self.checkout()
        self.assertEqual((self.target / "deps/sibling/content.txt").read_text(), "grand")

    def test_mid_fetch_disconnect_is_not_retried(self):
        self.cache.fail_repo = "child"
        self.cache.fail_post = True
        self.cache.disconnect = True
        self.assertFalse(self.attempt())
        self.assert_single_cache_attempts(["main", "child"])
        posts = [p for method, p, _ in self.cache.requests if method == "POST" and "/child/" in p]
        self.assertEqual(len(posts), 1)
        self.assertEqual(list(self.target.iterdir()), [])

    def test_fetch_timeout_kills_transport_and_is_not_retried(self):
        self.env["CACHEW_CACHEW_FETCH_TIMEOUT"] = "0.3"
        self.cache.fail_repo = "main"
        self.cache.fetch_delay = 0.6
        self.assertFalse(self.attempt())
        self.assert_single_cache_attempts(["main"])
        self.assertEqual(list(self.target.iterdir()), [])

    def test_unsupported_modes_and_existing_checkout_skip_without_health_probe(self):
        for name, value in [
            ("SSH_KEY", "private key"),
            ("LFS", "true"),
            ("FILTER", "blob:none"),
            ("SPARSE_CHECKOUT", ".github"),
        ]:
            with self.subTest(name=name):
                self.env["CACHEW_" + name] = value
                self.assertFalse(self.attempt())
                del self.env["CACHEW_" + name]
        self.target.mkdir()
        (self.target / "keep.txt").write_text("keep")
        self.assertFalse(self.attempt())
        self.assertEqual((self.target / "keep.txt").read_text(), "keep")
        self.assertEqual(self.cache.requests, [])

    def test_path_escape_and_symlink_outside_workspace_skip_before_mutation(self):
        for path in ("../outside", str(self.root / "absolute-outside")):
            self.env["CACHEW_PATH"] = path
            self.assertFalse(self.attempt())
        (self.workspace / "link").symlink_to(self.home, target_is_directory=True)
        self.env["CACHEW_PATH"] = "link/escape"
        self.assertFalse(self.attempt())
        self.assertFalse((self.home / "escape").exists())
        self.assertEqual(self.cache.requests, [])

    @unittest.skipUnless(CHECKOUT_JS, "set CHECKOUT_ACTION_JS to the pinned checkout dist/index.js")
    def test_full_history_tag_keeps_official_semantics(self):
        self.env["CACHEW_REF"] = "refs/tags/v1.0.0"
        self.env["CACHEW_FETCH_DEPTH"] = "0"
        self.assertTrue(self.attempt())
        self.checkout()
        self.assertEqual(git(self.target, "rev-parse", "--is-shallow-repository"), "false")
        self.assertEqual(git(self.target, "describe", "--tags"), "v1.0.0")

    @unittest.skipUnless(CHECKOUT_JS, "set CHECKOUT_ACTION_JS to the pinned checkout dist/index.js")
    def test_explicit_sha_and_direct_submodules_at_depth_500(self):
        self.env.update(CACHEW_REF=self.sha, CACHEW_SUBMODULES="true", CACHEW_FETCH_DEPTH="500")
        self.assertTrue(self.attempt())
        self.assert_single_cache_attempts(["main", "child"])
        self.checkout(**{"persist-credentials": "false"})
        self.assertEqual((self.target / "deps/child/content.txt").read_text(), "child")
        self.assertFalse((self.target / "deps/child/deep/grand/content.txt").exists())

    @unittest.skipUnless(CHECKOUT_JS, "set CHECKOUT_ACTION_JS to the pinned checkout dist/index.js")
    def test_pull_request_merge_ref_keeps_official_semantics(self):
        git(self.project_root / "org/main", "update-ref", "refs/pull/7/merge", self.sha)
        self.env.update(CACHEW_REF="refs/pull/7/merge", GITHUB_REF="refs/pull/7/merge")
        self.assertTrue(self.attempt())
        self.checkout()

    def test_default_event_sha_without_submodules(self):
        self.env.update(CACHEW_REF="", CACHEW_SUBMODULES="false", CACHEW_FETCH_TAGS="false")
        self.assertTrue(self.attempt())
        self.assert_single_cache_attempts(["main"])
        self.assertEqual(git(self.target, "rev-parse", "HEAD"), self.sha)
        self.assertFalse((self.target / ".git/modules").exists())

    def test_unsafe_module_path_discards_seed_and_never_escapes_target(self):
        repository = self.project_root / "org/main"
        git(repository, "config", "-f", ".gitmodules", "submodule.child-name.path", "../escape")
        git(repository, "add", ".gitmodules", env=self.env)
        git(repository, "commit", "-m", "unsafe path", env=self.env)
        self.assertFalse(self.attempt())
        self.assert_single_cache_attempts(["main"])
        self.assertEqual(list(self.target.iterdir()), [])
        self.assertFalse((self.workspace / "escape").exists())

    def test_composite_forwards_every_pinned_checkout_input(self):
        names = "repository ref token ssh-key ssh-known-hosts ssh-strict ssh-user persist-credentials path clean filter sparse-checkout sparse-checkout-cone-mode fetch-depth fetch-tags show-progress lfs submodules set-safe-directory github-server-url".split()
        action = (ACTION / "action.yml").read_text()
        forwarding = action.split("      with:\n", 1)[1]
        self.assertEqual(set(re.findall(r"^        ([a-z-]+):", forwarding, re.M)), set(names))
        for name in names:
            self.assertIn(f"{name}: ${{{{ inputs.{name} }}}}", forwarding)
        self.assertIn("continue-on-error: true", action)
        self.assertIn("value: ${{ steps.checkout.outputs.ref }}", action)
        self.assertIn("value: ${{ steps.checkout.outputs.commit }}", action)
        self.assertIn("actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd", action)


if __name__ == "__main__":
    unittest.main()
