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

import test_prefetch as fixtures


HELPER = fixtures.ACTION / "session" / "index.js"


@unittest.skipUnless(fixtures.CHECKOUT_JS, "set CHECKOUT_ACTION_JS to the pinned checkout bundle")
class SessionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.PrefetchTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.env = self.fixture.env
        # Checkout v6 scopes its credential include to the absolute gitdir.
        # macOS /var aliases must agree with Git's canonical /private/var path.
        self.env["GITHUB_WORKSPACE"] = str(self.fixture.workspace.resolve())
        self.env["CACHEW_ACTION_PATH"] = str(fixtures.ACTION)
        self.env["CACHEW_PERSIST_CREDENTIALS"] = "true"
        self.posts = []
        self.sequence = 0
        self.original_global = (self.fixture.home / ".gitconfig").read_bytes()

    def run_node(self, script, *args, env=None):
        result = subprocess.run(
            ["node", str(script), *args],
            cwd=self.fixture.workspace,
            env=env or self.env,
            text=True,
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def prepare(self):
        if not self.posts:
            self.original_selection = self.env.get("GIT_CONFIG_GLOBAL")
            original = subprocess.run(
                ["git", "config", "--get", "user.name"],
                cwd=self.fixture.workspace,
                env=self.env,
                text=True,
                capture_output=True,
            )
            self.original_name = original.stdout.strip() if original.returncode == 0 else None
        state = self.fixture.root / f"helper-state-{self.sequence}"
        self.sequence += 1
        state.touch()
        env = dict(self.env, GITHUB_STATE=str(state))
        self.run_node(HELPER, env=env)
        self.env.update(fixtures.runner_file(self.fixture.root / "env"))
        self.posts.append((HELPER, fixtures.runner_file(state)))

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
        directory = Path(self.env["CACHEW_SESSION_DIR"])
        for script, state in reversed(self.posts):
            env = dict(self.env)
            env.update({"STATE_" + key: value for key, value in state.items()})
            self.run_node(script, env=env)
            self.env.update(fixtures.runner_file(self.fixture.root / "env"))
        self.posts.clear()
        self.assertFalse(directory.exists())
        self.assertEqual((self.fixture.home / ".gitconfig").read_bytes(), self.original_global)
        self.assert_restored_globals()
        self.fixture.assert_no_cache_credentials()

    def assert_restored_globals(self):
        if self.original_selection is not None:
            self.assertEqual(self.env["GIT_CONFIG_GLOBAL"], self.original_selection)
        else:
            proxy = Path(self.env["GIT_CONFIG_GLOBAL"])
            self.assertTrue(proxy.is_file())
            self.assertNotIn(fixtures.TOKEN, proxy.read_text())
            self.assertNotIn(base64.b64encode(f"x-access-token:{fixtures.TOKEN}".encode()).decode(), proxy.read_text())
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
        self.checkout()
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
        (directory / "cache-disabled").touch()
        shutil.copyfile(directory / "upstream.config", directory / "global.config")
        self.env["CACHEW_PATH"] = "second-checkout"
        self.assertEqual(self.flow(), (False, False))
        expected = {str(self.fixture.target.resolve()), str((self.fixture.workspace / "second-checkout").resolve())}
        for config in (directory / "global.config", directory / "upstream.config"):
            self.assertEqual({str(Path(value).resolve()) for value in self.safe_directories(config) if value}, expected)
        self.finish()

    def test_safe_directory_false_does_not_add_checkout_path(self):
        self.env["CACHEW_SET_SAFE_DIRECTORY"] = "false"
        self.assertEqual(self.flow(), (True, True))
        directory = Path(self.env["CACHEW_SESSION_DIR"])
        self.assertEqual(self.safe_directories(), [])
        self.assertEqual(self.safe_directories(directory / "upstream.config"), [])
        self.finish()

    def test_safe_directory_empty_reset_and_deduplication_survive_rebuild(self):
        self.prepare()
        self.assertTrue(self.fixture.attempt())
        self.checkout()
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
        directory = Path(self.env["CACHEW_SESSION_DIR"])
        for script, state in reversed(self.posts):
            self.run_node(script, env=dict(self.env, **{"STATE_" + key: value for key, value in state.items()}))
            self.env.update(fixtures.runner_file(self.fixture.root / "env"))
        self.posts.clear()
        self.assertFalse(directory.exists())
        self.assertEqual((self.fixture.home / ".gitconfig").read_bytes(), self.original_global)
        self.assert_restored_globals()

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
        for repository in ("main", "secondary"):
            result = subprocess.run(
                ["git", "ls-remote", self.fixture.upstream.url + "/org/" + repository],
                cwd=self.fixture.workspace,
                env=self.env,
                text=True,
                capture_output=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.fixture.cache.requests, cache_before)
        self.finish()

    def test_sparse_or_existing_target_still_configures_later_dependencies(self):
        self.prepare()
        self.env["CACHEW_SPARSE_CHECKOUT"] = ".github"
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

    def test_untrusted_optional_cache_configuration_keeps_checkout_upstream(self):
        self.env["CACHEW_CACHEW_URL"] = "https://untrusted.example"
        self.assertEqual(self.flow(), (False, False))
        self.assertEqual(self.fixture.cache.requests, [])
        self.assertTrue(self.fixture.upstream.requests)
        self.finish()

    def test_original_explicit_global_file_is_restored_for_earlier_post_actions(self):
        custom = self.fixture.root / "caller-global.config"
        custom.write_text("[user]\n\tname = Caller global user\n")
        self.env["GIT_CONFIG_GLOBAL"] = str(custom)
        self.assertEqual(self.flow(), (True, True))
        self.finish()
        self.assertEqual(self.env["GIT_CONFIG_GLOBAL"], str(custom))

    def test_original_empty_global_selection_remains_intentionally_disabled(self):
        self.env["GIT_CONFIG_GLOBAL"] = ""
        self.assertEqual(self.flow(), (True, True))
        self.finish()
        self.assertEqual(self.env["GIT_CONFIG_GLOBAL"], "")

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
            nested = fixtures.PrefetchTests()
            try:
                nested.setUp()
                self.assertNotIn("CACHEW_SESSION_DIR", nested.env)
                self.assertTrue(nested.attempt())
                self.assertFalse(marker.exists())
                self.assertEqual(outer_global.read_text(), "[user]\n\tname = Outer action user\n")
            finally:
                nested.doCleanups()

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


if __name__ == "__main__":
    unittest.main()
