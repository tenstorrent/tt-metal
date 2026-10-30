# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import base64
import json
import os
from pathlib import Path
import subprocess
import shutil
import unittest
from unittest.mock import patch
import urllib.parse

import test_checkout as fixtures


HELPER = fixtures.ACTION / "session" / "index.js"
PREPARE = fixtures.ACTION / "session" / "prepare.sh"
NODE = shutil.which("node")


@unittest.skipUnless(fixtures.CHECKOUT_JS, "set CHECKOUT_ACTION_JS to the pinned checkout bundle")
class SessionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.CheckoutTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.env = self.fixture.env
        # Checkout v6 scopes its credential include to the absolute gitdir.
        # macOS /var aliases must agree with Git's canonical /private/var path.
        self.env["GITHUB_WORKSPACE"] = str(self.fixture.workspace.resolve())
        self.env["CACHEW_ACTION_PATH"] = str(fixtures.ACTION)
        self.env["CACHEW_PERSIST_CREDENTIALS"] = "true"
        self.posts = []
        self.prepared = False
        self.sequence = 0
        self.original_global = (self.fixture.home / ".gitconfig").read_bytes()

    def run_node(self, script, *args, env=None):
        result = subprocess.run(
            [NODE, str(script), *args],
            cwd=self.fixture.workspace,
            env=env or self.env,
            text=True,
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def prepare(self):
        if not self.prepared:
            self.prepared = True
            self.original_selection = self.env.get("GIT_CONFIG_GLOBAL")
            original = subprocess.run(
                ["git", "config", "--get", "user.name"],
                cwd=self.fixture.workspace,
                env=self.env,
                text=True,
                capture_output=True,
            )
            self.original_name = original.stdout.strip() if original.returncode == 0 else None
        # Exactly what action.yml runs: Node discovery, then `index.js prepare`.
        result = subprocess.run(
            ["bash", str(PREPARE)], cwd=self.fixture.workspace, env=self.env, text=True, capture_output=True, timeout=30
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(fixtures.runner_file(self.fixture.root / "output").get("ready"), "true", result.stdout)
        self.env.update(fixtures.runner_file(self.fixture.root / "env"))

    def checkout(self):
        state = self.fixture.root / f"checkout-state-{self.sequence}"
        self.sequence += 1
        state.touch()
        env = dict(self.env, GITHUB_STATE=str(state))
        for name, value in {
            "repository": self.env["CACHEW_REPOSITORY"],
            "ref": self.env["CACHEW_REF"],
            "token": self.env["CACHEW_TOKEN"],
            "path": self.env["CACHEW_PATH"],
            "fetch-depth": self.env["CACHEW_FETCH_DEPTH"],
            "fetch-tags": self.env["CACHEW_FETCH_TAGS"],
            "submodules": self.env["CACHEW_SUBMODULES"],
            "clean": "true",
            "persist-credentials": self.env["CACHEW_PERSIST_CREDENTIALS"],
            "set-safe-directory": self.env.get("CACHEW_SET_SAFE_DIRECTORY", "true"),
        }.items():
            env["INPUT_" + name.upper()] = value
        before = list(self.fixture.cache.requests)
        self.run_node(fixtures.CHECKOUT_JS, env=env)
        self.assertEqual(self.fixture.cache.requests, before, "official checkout must bypass every cache rewrite")
        self.posts.append((fixtures.CHECKOUT_JS, fixtures.runner_file(state)))

    def activate(self, available=None):
        self.env["CACHEW_AVAILABLE"] = available or fixtures.runner_file(self.fixture.root / "output").get(
            "available", "unknown"
        )
        self.run_node(HELPER, "activate")
        self.env.update(fixtures.runner_file(self.fixture.root / "env"))
        bins = (self.fixture.root / "path").read_text().splitlines()
        self.env["PATH"] = os.pathsep.join(dict.fromkeys([*reversed(bins), *self.env["PATH"].split(os.pathsep)]))
        return fixtures.runner_file(self.fixture.root / "output")["configured"] == "true"

    def finish(self):
        self.end_job()
        self.fixture.assert_no_cache_credentials()

    def end_job(self):
        """Run checkout's post steps with the session still in place, then do what
        the runner does when the job completes: empty RUNNER_TEMP."""
        directory = Path(self.env["CACHEW_SESSION_DIR"])
        self.assert_caller_settings_visible()
        for script, state in reversed(self.posts):
            env = dict(self.env)
            env.update({"STATE_" + key: value for key, value in state.items()})
            self.run_node(script, env=env)
            self.env.update(fixtures.runner_file(self.fixture.root / "env"))
        self.posts.clear()
        self.assertTrue(directory.exists(), "the session lives until the runner empties RUNNER_TEMP")
        self.assertEqual((self.fixture.home / ".gitconfig").read_bytes(), self.original_global)
        if self.original_selection:
            self.assertTrue(Path(self.original_selection).exists())
        for entry in self.fixture.runtime.iterdir():
            shutil.rmtree(entry) if entry.is_dir() and not entry.is_symlink() else entry.unlink()
        self.assertEqual((self.fixture.home / ".gitconfig").read_bytes(), self.original_global)

    def assert_caller_settings_visible(self):
        result = subprocess.run(
            [self.env["CACHEW_REAL_GIT"], "config", "--get", "user.name"],
            cwd=self.fixture.workspace,
            env=self.env,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.stdout.strip() if result.returncode == 0 else None, self.original_name)

    def flow(self):
        self.prepare()
        seeded = self.fixture.attempt()
        # As in action.yml: the official checkout runs only when cachew did not check out.
        if not seeded:
            self.checkout()
        active = self.activate()
        return seeded, active

    def real_git(self, *args):
        result = subprocess.run(
            [self.env["CACHEW_REAL_GIT"], *args],
            cwd=self.fixture.workspace,
            env=self.env,
            text=True,
            capture_output=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout.strip()

    def safe_directories(self, file=None):
        selection = ["--file", str(file)] if file else ["--global"]
        result = subprocess.run(
            [self.env["CACHEW_REAL_GIT"], "config", *selection, "--includes", "--null", "--get-all", "safe.directory"],
            cwd=self.fixture.workspace,
            env=self.env,
            text=True,
            capture_output=True,
        )
        self.assertIn(result.returncode, (0, 1), result.stderr)
        return result.stdout.split("\0")[:-1] if result.returncode == 0 else []

    def test_default_safe_directory_survives_activation_and_later_prepare(self):
        self.prepare()
        self.assertTrue(self.fixture.attempt())
        before = self.safe_directories()
        self.assertIn(str(self.fixture.target.resolve()), [str(Path(value).resolve()) for value in before if value])
        self.assertTrue(self.activate())
        self.assertEqual(self.safe_directories(), before)
        self.prepare()
        self.assertEqual(self.safe_directories(), before)
        self.assertTrue(self.activate("true"))
        self.assertEqual(self.safe_directories(), before)
        self.finish()

    def test_repeated_default_checkouts_keep_both_safe_directories_in_fallback(self):
        self.assertEqual(self.flow(), (True, True))
        directory = Path(self.env["CACHEW_SESSION_DIR"])
        wrapper = load_wrapper()
        state = json.loads((directory / "state.json").read_text())
        wrapper.disable_cache(self.env["CACHEW_REAL_GIT"], directory, state, self.env)
        self.env["CACHEW_PATH"] = "second-checkout"
        self.assertEqual(self.flow(), (False, False))
        expected = {str(self.fixture.target.resolve()), str((self.fixture.workspace / "second-checkout").resolve())}
        config = directory / "global.config"
        self.assertEqual({str(Path(value).resolve()) for value in self.safe_directories(config) if value}, expected)
        self.finish()

    def test_safe_directory_false_does_not_add_checkout_path(self):
        self.env["CACHEW_SET_SAFE_DIRECTORY"] = "false"
        self.assertEqual(self.flow(), (True, True))
        self.assertEqual(self.safe_directories(), [])
        self.finish()

    def test_safe_directory_empty_reset_and_deduplication_survive_rebuild(self):
        self.prepare()
        self.assertTrue(self.fixture.attempt())
        trusted = str(self.fixture.workspace / "explicitly-trusted")
        self.real_git("config", "--global", "--add", "safe.directory", "")
        self.real_git("config", "--global", "--add", "safe.directory", trusted)
        self.real_git("config", "--global", "--add", "safe.directory", trusted)
        self.assertTrue(self.activate())
        self.assertEqual(self.safe_directories(), ["", trusted])
        self.prepare()
        self.assertEqual(self.safe_directories(), ["", trusted])
        self.assertTrue(self.activate("true"))
        self.assertEqual(self.safe_directories(), ["", trusted])
        self.finish()

    def test_downstream_new_repository_uses_global_cache_and_post_cleans_auth(self):
        self.fixture.repository("dependency")
        expected = "basic " + base64.b64encode(f"x-access-token:{fixtures.TOKEN}".encode()).decode()
        self.fixture.cache.required_auth["dependency"] = expected
        self.assertEqual(self.flow(), (True, True))
        self.assertEqual(self.real_git("config", "--global", "--get", "checkout.workers"), "0")
        self.assertEqual(self.real_git("config", "--global", "--get", "checkout.thresholdForParallelism"), "50")
        self.assertEqual(self.real_git("config", "--global", "--get", "submodule.fetchJobs"), "0")
        self.assertEqual(self.real_git("config", "--global", "--get", "gc.auto"), "0")
        upstream_before = list(self.fixture.upstream.requests)
        self.real_git("clone", self.fixture.upstream.url + "/org/dependency", "dependency")
        self.assertEqual(self.fixture.upstream.requests, upstream_before)
        requests = [request for request in self.fixture.cache.requests if "/dependency/" in request[1]]
        self.assertTrue(requests)
        self.assertTrue(all(auth.lower() == expected.lower() for _, _, auth in requests))
        self.finish()

    @unittest.skipUnless(shutil.which("cmake"), "CMake is required for the downstream FetchContent regression")
    def test_real_cmake_fetchcontent_uses_the_post_checkout_global_configuration(self):
        self.fixture.repository("cmake-dependency")
        dependency = self.fixture.project_root / "org/cmake-dependency"
        (dependency / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.24)\nproject(dependency NONE)\n")
        fixtures.git(dependency, "add", "CMakeLists.txt", env=self.env)
        fixtures.git(dependency, "commit", "-m", "CMake dependency", env=self.env)
        commit = fixtures.git(dependency, "rev-parse", "HEAD")
        self.assertEqual(self.flow(), (True, True))
        self.fixture.upstream.block_submodules = True
        source = self.fixture.root / "cmake-source"
        source.mkdir()
        (source / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.24)\nproject(consumer NONE)\ninclude(FetchContent)\n"
            f"FetchContent_Declare(downstream GIT_REPOSITORY {self.fixture.upstream.url}/org/cmake-dependency GIT_TAG {commit})\n"
            "FetchContent_MakeAvailable(downstream)\n"
        )
        result = subprocess.run(
            ["cmake", "-S", str(source), "-B", str(self.fixture.root / "cmake-build")],
            env=self.env,
            text=True,
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue([url for _, url, _ in self.fixture.cache.requests if "/cmake-dependency/" in url])
        self.assertFalse([url for _, url, _ in self.fixture.upstream.requests if "/cmake-dependency/" in url])
        self.finish()

    def test_inherited_rewrite_cannot_redirect_unhealthy_fallback(self):
        key = f"url.{self.fixture.cache.url}/git/{self.fixture.upstream.server.server_address[0]}:{self.fixture.upstream.server.server_port}/.insteadOf"
        fixtures.git(self.fixture.workspace, "config", "--global", key, self.fixture.upstream.url + "/", env=self.env)
        self.original_global = (self.fixture.home / ".gitconfig").read_bytes()
        self.fixture.cache.health = 503
        self.assertEqual(self.flow(), (False, False))
        self.fixture.assert_single_cache_attempts([])
        self.assertTrue(self.fixture.upstream.requests)
        # The caller's inherited rewrite survives unchanged, outside this job's configuration.
        self.finish_without_credential_assertion()

    def finish_without_credential_assertion(self):
        self.end_job()

    def test_persist_false_does_not_leave_current_token_in_globals(self):
        self.env["CACHEW_PERSIST_CREDENTIALS"] = "false"
        self.assertEqual(self.flow(), (True, True))
        directory = Path(self.env["CACHEW_SESSION_DIR"])
        encoded = base64.b64encode(f"x-access-token:{fixtures.TOKEN}".encode())
        for file in directory.rglob("*"):
            if file.is_file():
                self.assertNotIn(fixtures.TOKEN.encode(), file.read_bytes())
                self.assertNotIn(encoded, file.read_bytes())
        self.finish()

    def test_secondary_checkout_keeps_primary_credentials_and_uses_its_own_scope(self):
        self.fixture.repository("secondary")
        self.fixture.repository("dependency")
        self.assertEqual(self.flow(), (True, True))
        self.env.update(
            CACHEW_REPOSITORY="org/secondary",
            CACHEW_PATH="secondary",
            CACHEW_REF="main",
            CACHEW_TOKEN="secondary-private-token",
        )
        self.assertEqual(self.flow(), (True, True))
        self.real_git("ls-remote", self.fixture.upstream.url + "/org/dependency")
        self.real_git("ls-remote", self.fixture.upstream.url + "/org/secondary")
        expected = {
            "dependency": "basic " + base64.b64encode(f"x-access-token:{fixtures.TOKEN}".encode()).decode(),
            "secondary": "basic " + base64.b64encode(b"x-access-token:secondary-private-token").decode(),
        }
        for name, header in expected.items():
            last = [auth for _, url, auth in self.fixture.cache.requests if f"/{name}/" in url][-1]
            self.assertEqual(last.lower(), header.lower())
        self.finish()

    def test_secondary_persist_false_does_not_erase_primary_token(self):
        self.fixture.repository("secondary")
        self.assertEqual(self.flow(), (True, True))
        self.env.update(
            CACHEW_REPOSITORY="org/secondary",
            CACHEW_PATH="secondary",
            CACHEW_REF="main",
            CACHEW_TOKEN="do-not-persist-secondary",
            CACHEW_PERSIST_CREDENTIALS="false",
        )
        self.assertEqual(self.flow(), (True, True))
        state = json.loads((Path(self.env["CACHEW_SESSION_DIR"]) / "state.json").read_text())
        self.assertNotIn("do-not-persist-secondary", json.dumps(state))
        self.assertNotIn(base64.b64encode(b"x-access-token:do-not-persist-secondary").decode(), json.dumps(state))
        self.real_git("ls-remote", self.fixture.upstream.url + "/org/main")
        expected = "basic " + base64.b64encode(f"x-access-token:{fixtures.TOKEN}".encode()).decode()
        last = [auth for _, url, auth in self.fixture.cache.requests if "/main/" in url][-1]
        self.assertEqual(last.lower(), expected.lower())
        self.finish()

    def test_disabled_cache_secondary_private_checkout_uses_its_current_token(self):
        self.fixture.repository("secondary")
        secondary_token = "secondary-private-token"
        self.fixture.upstream.required_auth.update(
            main="basic " + base64.b64encode(f"x-access-token:{fixtures.TOKEN}".encode()).decode(),
            secondary="basic " + base64.b64encode(f"x-access-token:{secondary_token}".encode()).decode(),
        )
        self.fixture.cache.fail_repo = "main"
        self.assertEqual(self.flow(), (False, False))
        cache_before = list(self.fixture.cache.requests)
        self.env.update(
            CACHEW_REPOSITORY="org/secondary", CACHEW_PATH="secondary", CACHEW_REF="main", CACHEW_TOKEN=secondary_token
        )
        self.assertEqual(self.flow(), (False, False))
        # Upstream credentials are repository-scoped by actions/checkout, exactly
        # as without cachew; they are never promoted to job-wide headers.
        for checkout in (self.fixture.target, self.fixture.workspace / "secondary"):
            result = subprocess.run(
                ["git", "-C", str(checkout), "ls-remote", "origin"],
                cwd=self.fixture.workspace,
                env=self.env,
                text=True,
                capture_output=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.fixture.cache.requests, cache_before)
        self.finish()

    def test_unsupported_mode_still_configures_later_dependencies(self):
        self.prepare()
        self.env["CACHEW_LFS"] = "true"
        self.assertFalse(self.fixture.attempt())
        self.assertTrue(self.activate())
        self.fixture.assert_single_cache_attempts([])
        self.finish()

    def test_existing_checkout_still_configures_later_dependencies(self):
        self.assertEqual(self.flow(), (True, True))
        before = len(self.fixture.cache.discoveries())
        self.assertEqual(self.flow(), (False, True))
        self.assertEqual(len(self.fixture.cache.discoveries()), before)
        self.finish()

    def test_global_include_files_are_copied_and_originals_stay_unchanged(self):
        include = self.fixture.home / "preferences.config"
        include.write_text(
            '[user]\n\tname = Included user\n[http "https://unrelated.example/"]\n\textraHeader = existing-unrelated-header\n'
        )
        with (self.fixture.home / ".gitconfig").open("a") as stream:
            stream.write("[include]\n\tpath = preferences.config\n")
        self.original_global = (self.fixture.home / ".gitconfig").read_bytes()
        original_include = include.read_bytes()
        self.assertEqual(self.flow(), (True, True))
        self.assertEqual(self.real_git("config", "--global", "--includes", "--get", "user.name"), "Included user")
        self.assertEqual(
            self.real_git("config", "--global", "--includes", "--get", "http.https://unrelated.example/.extraHeader"),
            "existing-unrelated-header",
        )
        self.finish()
        self.assertEqual(include.read_bytes(), original_include)

    def plain_checkout_without_cache(self, env, cachew=True):
        """The official checkout (and the cachew step, if given a rejected cache) without a session."""
        self.fixture.env = env
        if cachew:
            self.assertFalse(self.fixture.attempt())
        self.assertEqual(self.fixture.cache.requests, [])
        state = self.fixture.root / "plain-checkout-state"
        state.write_text("")
        checkout = dict(env, GITHUB_STATE=str(state))
        for name, value in {
            "repository": "org/main",
            "ref": "main",
            "token": fixtures.TOKEN,
            "path": "checkout",
            "fetch-depth": "1",
            "submodules": "recursive",
            "clean": "true",
            "persist-credentials": "true",
            "set-safe-directory": "true",
        }.items():
            checkout["INPUT_" + name.upper()] = value
        self.run_node(fixtures.CHECKOUT_JS, env=checkout)
        self.assertEqual(fixtures.git(self.fixture.target, "rev-parse", "HEAD"), self.fixture.sha)
        self.assertEqual(self.fixture.cache.requests, [], "no request, so no token, reaches any cache")
        self.assertEqual((self.fixture.root / "path").read_text(), "", "no Git wrapper on PATH")
        self.assertEqual((self.fixture.root / "env").read_text(), "", "no job environment change")
        self.assertEqual(list(self.fixture.runtime.glob("cachew-*")), [])
        self.fixture.assert_no_cache_credentials()

    def test_untrusted_or_plain_http_cache_origins_leave_plain_checkout(self):
        for url in (
            "http://cachew.cachew.svc.cluster.local",
            "https://cachew.cachew.svc.cluster.local",
            "http://10.43.0.100:8080",
            "http://cachew.ci.tenstorrent.net",
            "https://evil.example",
            "https://untrusted.example",
        ):
            with self.subTest(url=url):
                env = dict(self.env, CACHEW_CACHEW_URL=url)
                self.run_prepare_expecting_fallback(env)
        shutil.rmtree(self.fixture.target, ignore_errors=True)
        self.plain_checkout_without_cache(dict(self.env, CACHEW_CACHEW_URL="https://evil.example"))

    def test_loopback_http_without_the_test_variable_leaves_plain_checkout(self):
        env = dict(self.env)
        del env["CACHEW_TEST_ALLOW_LOOPBACK_HTTP"]
        # Both a loopback cache and a loopback upstream are refused in production.
        self.run_prepare_expecting_fallback(env)
        self.run_prepare_expecting_fallback(dict(env, CACHEW_GITHUB_SERVER_URL="https://github.com"))
        # The plain checkout itself is unaffected (actions/checkout has its own rules).
        self.plain_checkout_without_cache(env)

    def test_action_clears_the_test_only_loopback_variable(self):
        action = (fixtures.ACTION / "action.yml").read_text()
        self.assertNotIn("inputs.cachew-test", action)
        self.assertNotIn("${{ env.CACHEW_TEST_ALLOW_LOOPBACK_HTTP", action)
        cleared = action.count("CACHEW_TEST_ALLOW_LOOPBACK_HTTP: ''")
        self.assertEqual(cleared, action.count("CACHEW_TEST_ALLOW_LOOPBACK_HTTP"))
        self.assertEqual(cleared, 3, "session, cachew and configure steps each clear it")

    def test_original_explicit_global_file_is_restored_for_earlier_post_actions(self):
        custom = self.fixture.root / "caller-global.config"
        custom.write_text("[user]\n\tname = Caller global user\n")
        self.env["GIT_CONFIG_GLOBAL"] = str(custom)
        self.assertEqual(self.flow(), (True, True))
        self.assertEqual(self.real_git("config", "--global", "--includes", "--get", "user.name"), "Caller global user")
        self.finish()
        self.assertEqual(custom.read_text(), "[user]\n\tname = Caller global user\n")

    def test_original_empty_global_selection_remains_intentionally_disabled(self):
        self.env["GIT_CONFIG_GLOBAL"] = ""
        self.assertEqual(self.flow(), (True, True))
        self.assertIsNone(self.original_name)
        self.finish()

    def test_missing_python_keeps_upstream_usable_without_installing_shim(self):
        binaries = self.fixture.root / "without-python"
        binaries.mkdir()
        python = binaries / "python3"
        python.write_text("#!/bin/sh\nexit 127\n")
        python.chmod(0o700)
        self.env["PATH"] = str(binaries) + os.pathsep + self.env["PATH"]
        self.prepare()
        self.checkout()
        self.assertFalse(self.activate("unknown"))
        self.assertEqual((self.fixture.root / "path").read_text(), "")
        self.real_git("ls-remote", self.fixture.upstream.url + "/org/main")
        self.finish()

    def test_fixture_isolates_an_outer_action_session_and_path_shim(self):
        outer = self.fixture.root / "outer-action"
        outer.mkdir()
        (outer / "cache-disabled").touch()
        outer_global = self.fixture.root / "outer-global.config"
        outer_global.write_text("[user]\n\tname = Outer action user\n")
        shim = self.fixture.root / "outer-bin"
        shim.mkdir()
        marker = self.fixture.root / "outer-shim-invoked"
        executable = shim / "git"
        executable.write_text(f'#!/bin/sh\ntouch "{marker}"\nexit 99\n')
        executable.chmod(0o700)
        with patch.dict(
            os.environ,
            CACHEW_SESSION_DIR=str(outer),
            CACHEW_REAL_GIT=self.env["CACHEW_REAL_GIT"],
            GIT_CONFIG_GLOBAL=str(outer_global),
            PATH=str(shim) + os.pathsep + os.environ["PATH"],
        ):
            nested = fixtures.CheckoutTests()
            try:
                nested.setUp()
                self.assertNotIn("CACHEW_SESSION_DIR", nested.env)
                self.assertTrue(nested.attempt())
                self.assertFalse(marker.exists())
                self.assertEqual(outer_global.read_text(), "[user]\n\tname = Outer action user\n")
            finally:
                nested.doCleanups()

    def test_job_wide_upstream_header_never_overrides_other_credentials(self):
        # H2: actions/checkout scopes persisted credentials to the checked-out
        # repository. A job-wide github.com Authorization header would replace the
        # credentials of every later upstream request, e.g. a PAT in a clone URL
        # for another private repository (curl drops URL credentials when an
        # Authorization header is configured), or be sent twice.
        self.fixture.repository("private")
        pat = "separate-private-pat"
        self.fixture.upstream.required_auth["private"] = (
            "basic " + base64.b64encode(f"x-access-token:{pat}".encode()).decode()
        )
        primary = "basic " + base64.b64encode(f"x-access-token:{fixtures.TOKEN}".encode()).decode()
        self.fixture.upstream.required_auth["main"] = primary
        self.assertEqual(self.flow(), (True, True))
        host = urllib.parse.urlsplit(self.fixture.upstream.url).netloc
        result = subprocess.run(
            ["git", "clone", f"http://x-access-token:{pat}@{host}/org/private", str(self.fixture.root / "private")],
            env=self.env,
            text=True,
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        # The primary checkout keeps working upstream with exactly one header.
        target = str(self.fixture.target)
        self.real_git("-C", target, "push", "origin", "HEAD:refs/heads/pushed-after-activation")
        directory = Path(self.env["CACHEW_SESSION_DIR"])
        state = json.loads((directory / "state.json").read_text())
        load_wrapper().disable_cache(self.env["CACHEW_REAL_GIT"], directory, state, self.env)
        result = subprocess.run(
            ["git", "-C", target, "fetch", "origin"], env=self.env, text=True, capture_output=True, timeout=30
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.fixture.upstream.duplicate_auth, [])
        upstream_auth = [auth for _, url, auth in self.fixture.upstream.requests if "/org/main/" in url]
        self.assertTrue(upstream_auth and all(auth and auth.lower() == primary.lower() for auth in upstream_auth))
        for name in ("global.config", "checkout.config"):
            text = (directory / name).read_text()
            self.assertNotIn(f'[http "{self.fixture.upstream.url}', text)
        self.finish()

    def run_prepare_expecting_fallback(self, env):
        (self.fixture.root / "env").write_text("")
        (self.fixture.root / "output").write_text("")
        (self.fixture.root / "path").write_text("")
        result = subprocess.run(
            ["bash", str(PREPARE)],
            cwd=self.fixture.workspace,
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(fixtures.runner_file(self.fixture.root / "output").get("ready"), "false")
        self.assertEqual((self.fixture.root / "env").read_text(), "", "no job environment change")
        self.assertEqual((self.fixture.root / "path").read_text(), "", "no Git wrapper on PATH")
        self.assertEqual(list(self.fixture.runtime.glob("cachew-*")), [], "no session left behind")
        return result

    def node_free_path(self):
        """The test PATH with every `node` hidden (other tools stay available)."""
        entries = []
        for index, directory in enumerate(self.env["PATH"].split(os.pathsep)):
            source = Path(directory)
            if not (source / "node").exists():
                entries.append(directory)
                continue
            shadow = self.fixture.root / f"no-node-{index}"
            shadow.mkdir(exist_ok=True)
            for item in source.iterdir():
                if item.name != "node" and not (shadow / item.name).exists():
                    (shadow / item.name).symlink_to(item)
            entries.append(str(shadow))
        return os.pathsep.join(entries)

    def test_missing_node_falls_back_to_plain_checkout(self):
        # Composite run steps do not get the runner's bundled Node on PATH.
        env = dict(self.env, PATH=self.node_free_path())
        result = self.run_prepare_expecting_fallback(env)
        self.assertIn("Node.js 22+ unavailable", result.stdout)
        # action.yml skips the cachew and configure steps when the session is not ready.
        self.plain_checkout_without_cache(self.env, cachew=False)

    def runner_externals(self, version):
        """A runner root whose externals/<version> Node records that it ran."""
        root = self.fixture.root / "runner-root"
        temp = root / "_work" / "_temp"
        temp.mkdir(parents=True)
        externals = root / "externals" / version / "bin"
        externals.mkdir(parents=True)
        marker = self.fixture.root / "externals-node-used"
        (externals / "node").write_text(f'#!/bin/sh\ntouch "{marker}"\nexec "{shutil.which("node")}" "$@"\n')
        (externals / "node").chmod(0o755)
        return temp, marker

    def end_of_life_node_dir(self):
        """A directory whose `node` reports Node.js 20."""
        directory = self.fixture.root / "node20-bin"
        directory.mkdir()
        preload = directory / "v20.cjs"
        preload.write_text('Object.defineProperty(process.versions, "node", {value: "20.19.0"});\n')
        (directory / "node").write_text(f'#!/bin/sh\nexec "{shutil.which("node")}" --require "{preload}" "$@"\n')
        (directory / "node").chmod(0o755)
        return directory

    def test_runner_externals_node_is_used_when_node_is_not_on_path(self):
        # tt CI VMs: runner in /opt/runner, RUNNER_TEMP=/opt/runner/_work/_temp,
        # Node only in /opt/runner/externals/node24.
        temp, marker = self.runner_externals("node24")
        self.fixture.runtime = temp
        self.env.update(RUNNER_TEMP=str(temp), PATH=self.node_free_path())
        self.assertEqual(self.flow(), (True, True))
        self.assertTrue(marker.exists())
        self.assertTrue(Path(self.env["CACHEW_SESSION_DIR"]).is_relative_to(temp))
        self.finish()

    def test_end_of_life_node_on_path_is_skipped_for_runner_node24(self):
        temp, marker = self.runner_externals("node24")
        self.fixture.runtime = temp
        path = os.pathsep.join([str(self.end_of_life_node_dir()), self.node_free_path()])
        self.env.update(RUNNER_TEMP=str(temp), PATH=path)
        self.assertEqual(self.flow(), (True, True))
        self.assertTrue(marker.exists())
        self.finish()

    def test_end_of_life_node_is_never_used(self):
        # Neither an old `node` on PATH nor the runner's externals/node20.
        temp, marker = self.runner_externals("node20")
        self.fixture.runtime = temp
        path = os.pathsep.join([str(self.end_of_life_node_dir()), self.node_free_path()])
        result = self.run_prepare_expecting_fallback(dict(self.env, RUNNER_TEMP=str(temp), PATH=path))
        self.assertIn("Node.js 22+ unavailable", result.stdout)
        self.assertFalse(marker.exists())

    def test_prepare_errors_fall_back_to_plain_checkout(self):
        # H3: whatever prevents cachew must never fail a checkout that plain
        # actions/checkout performs.
        no_git = self.fixture.root / "no-git-bin"
        no_git.mkdir()
        for tool in ("node", "bash", "dirname"):
            (no_git / tool).symlink_to(shutil.which(tool))
        self.run_prepare_expecting_fallback(dict(self.env, PATH=str(no_git)))

        old_git = self.fixture.root / "old-git-bin"
        old_git.mkdir()
        (old_git / "git").write_text(
            '#!/bin/sh\nif [ "$1" = version ]; then echo "git version 2.25.1"; exit 0; fi\nexec '
            + self.env["CACHEW_REAL_GIT"]
            + ' "$@"\n'
        )
        (old_git / "git").chmod(0o700)
        self.run_prepare_expecting_fallback(dict(self.env, PATH=str(old_git) + os.pathsep + self.env["PATH"]))

        broken = self.fixture.home / ".gitconfig"
        original = broken.read_bytes()
        broken.write_text("[user\n\tname = broken\n")
        try:
            self.run_prepare_expecting_fallback(self.env)
        finally:
            broken.write_bytes(original)
        # The plain checkout still succeeds in the untouched environment.
        self.posts = []
        state = self.fixture.root / "plain-checkout-state"
        state.touch()
        env = dict(self.env, GITHUB_STATE=str(state))
        for name, value in {
            "repository": "org/main",
            "ref": "main",
            "token": fixtures.TOKEN,
            "path": "checkout",
            "fetch-depth": "1",
            "submodules": "recursive",
            "clean": "true",
            "persist-credentials": "true",
            "set-safe-directory": "true",
        }.items():
            env["INPUT_" + name.upper()] = value
        self.run_node(fixtures.CHECKOUT_JS, env=env)
        self.assertEqual(self.fixture.cache.requests, [])

    def test_wrapper_shim_only_starts_python_for_network_commands(self):
        self.assertEqual(self.flow(), (True, True))
        shim = Path(self.env["CACHEW_SESSION_DIR"]) / "bin/git"
        marker = self.fixture.root / "python-started"
        fake = self.fixture.root / "fake-python"
        fake.write_text(f'#!/bin/sh\ntouch "{marker}"\nexit 99\n')
        fake.chmod(0o700)
        text = shim.read_text()
        python = text.split("exec '", 1)[1].split("'", 1)[0]
        shim.write_text(text.replace(f"'{python}'", f"'{fake}'"))
        for arguments in (["rev-parse", "--is-inside-work-tree"], ["-c", "a.b=c", "-C", ".", "status", "--short"]):
            result = subprocess.run([str(shim), *arguments], cwd=self.fixture.target, env=self.env, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(marker.exists(), arguments)
        for arguments in (["fetch", "origin"], ["-C", ".", "ls-remote", "origin"], ["-c", "a.b=c", "clone", "x"]):
            result = subprocess.run([str(shim), *arguments], cwd=self.fixture.target, env=self.env, capture_output=True)
            self.assertEqual(result.returncode, 99, arguments)
            self.assertTrue(marker.exists(), arguments)
            marker.unlink()
        shim.write_text(text)
        self.finish()

    def test_failed_cache_attempt_stays_disabled_for_later_actions(self):
        self.fixture.cache.fail_repo = "main"
        self.assertEqual(self.flow(), (False, False))
        before = list(self.fixture.cache.requests)
        self.fixture.cache.fail_repo = None
        self.prepare()
        self.assertFalse(self.fixture.attempt())
        self.assertFalse(self.activate())
        self.assertEqual(self.fixture.cache.requests, before)
        self.finish()


def load_wrapper():
    import importlib.util

    spec = importlib.util.spec_from_file_location("git_wrapper", fixtures.ACTION / "git_wrapper.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    unittest.main()
