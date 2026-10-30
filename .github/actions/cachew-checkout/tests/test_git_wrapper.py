# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Real downstream CMake/Git reads, one cache attempt, and job-wide rewrite removal."""

import base64
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import unittest
import urllib.parse

import test_prefetch as fixture


@unittest.skipUnless(shutil.which("cmake"), "CMake is required for downstream dependency integration tests")
class DownstreamGitTests(unittest.TestCase):
    def setUp(self):
        self.git_fixture = fixture.PrefetchTests()
        self.git_fixture.setUp()
        self.addCleanup(self.git_fixture.doCleanups)
        self.root = self.git_fixture.root
        self.env = dict(self.git_fixture.env)
        self.real_git = self.git_fixture.env.get("CACHEW_REAL_GIT") or shutil.which("git")
        self.session = self.root / "session"
        self.session.mkdir(mode=0o700)
        self.upstream = self.git_fixture.upstream
        self.cache = self.git_fixture.cache
        self.baseline = (self.git_fixture.home / ".gitconfig").read_bytes()
        upstream_config = self.session / "upstream.config"
        upstream_config.write_bytes(self.baseline)
        self.config(upstream_config, f"http.{self.upstream.url}/.extraHeader", self.authorization())
        for name, value in (
            ("checkout.workers", "0"),
            ("checkout.thresholdForParallelism", "50"),
            ("submodule.fetchJobs", "0"),
            ("gc.auto", "0"),
        ):
            self.config(upstream_config, name, value)
        active = self.session / "global.config"
        active.write_bytes(upstream_config.read_bytes())
        host = urllib.parse.urlsplit(self.upstream.url).netloc
        self.config(active, f"url.{self.cache.url}/git/{host}/.insteadOf", self.upstream.url + "/")
        self.config(active, f"http.{self.cache.url}/.extraHeader", self.authorization())
        self.config(active, "http.followRedirects", "false")
        (self.session / "state.json").write_text(json.dumps({"active": True}))
        shim = self.session / "bin/git"
        shim.parent.mkdir()
        shim.write_text(
            "#!/bin/sh\nexec "
            + shlex.quote(sys.executable)
            + " "
            + shlex.quote(str(fixture.ACTION / "git_wrapper.py"))
            + ' "$@"\n'
        )
        shim.chmod(0o700)
        self.env.update(
            CACHEW_REAL_GIT=self.real_git,
            CACHEW_SESSION_DIR=str(self.session),
            CACHEW_GIT_FETCH_TIMEOUT="5",
            GIT_CONFIG_GLOBAL=str(active),
            PATH=str(shim.parent) + os.pathsep + self.env["PATH"],
        )

    def authorization(self):
        return "AUTHORIZATION: basic " + base64.b64encode(f"x-access-token:{fixture.TOKEN}".encode()).decode()

    def config(self, path, key, value):
        subprocess.run([self.real_git, "config", "--file", str(path), key, value], check=True, capture_output=True)

    def dependency(self, name):
        self.git_fixture.repository(name)
        repository = self.git_fixture.project_root / "org" / name
        (repository / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.24)\nproject(cached_dependency NONE)\n"
        )
        fixture.git(repository, "add", "CMakeLists.txt", env=self.env)
        fixture.git(repository, "commit", "-m", "CMake dependency", env=self.env)
        return fixture.git(repository, "rev-parse", "HEAD", env=self.env)

    def configure(self, dependencies):
        source = self.root / "cmake-source"
        source.mkdir()
        content = "cmake_minimum_required(VERSION 3.24)\nproject(cachew_consumer NONE)\ninclude(FetchContent)\n"
        for name, sha in dependencies:
            content += f"FetchContent_Declare({name} GIT_REPOSITORY {self.upstream.url}/org/{name} GIT_TAG {sha})\n"
            content += f"FetchContent_MakeAvailable({name})\n"
        (source / "CMakeLists.txt").write_text(content)
        result = subprocess.run(
            ["cmake", "-S", str(source), "-B", str(self.root / "cmake-build")],
            env=self.env,
            text=True,
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def run_git(self, *arguments, cwd=None):
        result = subprocess.run(["git", *arguments], cwd=cwd, env=self.env, text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout.strip()

    def assert_disabled(self):
        self.assertTrue((self.session / "cache-disabled").exists())
        result = subprocess.run(
            [self.real_git, "config", "--global", "--get-regexp", r"^url\..*\.insteadof$"],
            env=self.env,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertEqual((self.git_fixture.home / ".gitconfig").read_bytes(), self.baseline)
        self.assertNotIn(self.cache.url.encode(), (self.session / "global.config").read_bytes())
        self.assertEqual(self.run_git("config", "--global", "checkout.workers"), "0")

    def test_cmake_dependency_uses_global_cache_and_authorization(self):
        sha = self.dependency("cmake-dependency")
        self.upstream.block_submodules = True
        self.configure([("cmake-dependency", sha)])
        self.assertEqual(len(self.cache.discoveries()), 1)
        self.assertEqual(self.upstream.requests, [])
        for _, _, auth in self.cache.requests:
            self.assertEqual(auth, self.authorization().removeprefix("AUTHORIZATION: "))
        repository = self.root / "cmake-build/_deps/cmake-dependency-src"
        self.assertEqual(
            self.run_git("-C", str(repository), "config", "--get", "remote.origin.url"),
            self.upstream.url + "/org/cmake-dependency",
        )

    def test_cmake_clone_error_removes_rewrite_and_later_dependency_stays_upstream(self):
        first = self.dependency("cmake-dependency")
        second = self.dependency("next-dependency")
        self.cache.fail_repo = "cmake-dependency"
        self.configure([("cmake-dependency", first), ("next-dependency", second)])
        self.assertEqual(len(self.cache.discoveries()), 1)
        self.assertEqual(len(self.upstream.discoveries()), 2)
        self.assertTrue(all("cmake-dependency" in path for _, path, _ in self.cache.requests))
        self.assert_disabled()

    def test_later_fetch_error_updates_origin_refs_upstream_without_cache_retry(self):
        self.dependency("cmake-dependency")
        target = self.root / "cloned"
        self.run_git("clone", self.upstream.url + "/org/cmake-dependency", str(target))
        repository = self.git_fixture.project_root / "org/cmake-dependency"
        (repository / "new.txt").write_text("new commit")
        fixture.git(repository, "add", "new.txt", env=self.env)
        fixture.git(repository, "commit", "-m", "update", env=self.env)
        sha = fixture.git(repository, "rev-parse", "HEAD", env=self.env)
        self.cache.fail_repo = "cmake-dependency"
        self.run_git("--git-dir=.git", "fetch", "--tags", "--force", "origin", cwd=target)
        self.assertEqual(len(self.cache.discoveries()), 2)
        self.assertEqual(len(self.upstream.discoveries()), 1)
        self.assertEqual(self.run_git("-C", str(target), "rev-parse", "origin/main"), sha)
        self.assert_disabled()

    def test_stalled_cache_clone_falls_back_after_deadline(self):
        sha = self.dependency("cmake-dependency")
        self.env["CACHEW_GIT_FETCH_TIMEOUT"] = "0.4"
        self.cache.fail_repo = "cmake-dependency"
        self.cache.fetch_delay = 0.8
        self.configure([("cmake-dependency", sha)])
        self.assertEqual(len(self.cache.discoveries()), 1)
        self.assertEqual(len(self.upstream.discoveries()), 1)
        self.assert_disabled()

    def test_cache_redirect_disables_rewrite_without_following(self):
        sha = self.dependency("cmake-dependency")
        self.cache.fail_repo = "cmake-dependency"
        self.cache.fail_status = 302
        self.configure([("cmake-dependency", sha)])
        self.assertEqual(len(self.cache.requests), 1)
        self.assertEqual(len(self.upstream.discoveries()), 1)
        self.assert_disabled()

    def test_cache_401_does_not_retry_with_inherited_credential_helper(self):
        sha = self.dependency("cmake-dependency")
        self.cache.fail_repo = "cmake-dependency"
        self.cache.fail_status = 401
        handler = self.cache.server.RequestHandlerClass
        original = handler.send_response

        def unauthorized(response, status, message=None):
            original(response, status, message)
            if status == 401:
                response.send_header("WWW-Authenticate", 'Basic realm="cachew-fixture"')

        handler.send_response = unauthorized
        marker = self.root / "credential-helper-called"
        helper = f"!touch {shlex.quote(str(marker))}; printf 'username=other\\npassword=other\\n'"
        self.config(self.session / "global.config", "credential.helper", helper)
        self.configure([("cmake-dependency", sha)])
        self.assertEqual(len(self.cache.requests), 1)
        self.assertFalse(marker.exists())
        self.assert_disabled()

    def test_native_recursive_clone_goes_upstream_without_hidden_cache_retries(self):
        target = self.root / "recursive"
        self.run_git("clone", "--recurse-submodules", self.upstream.url + "/org/main", str(target))
        self.assertEqual(self.cache.requests, [])
        self.assertEqual((target / "deps/child/deep/grand/content.txt").read_text(), "grand")
        self.assertFalse((self.session / "cache-disabled").exists())

    def test_push_bypasses_read_only_cache(self):
        self.dependency("cmake-dependency")
        target = self.root / "push-source"
        self.run_git("clone", self.upstream.url + "/org/cmake-dependency", str(target))
        cache_requests = list(self.cache.requests)
        self.run_git("-C", str(target), "push", "origin", "HEAD:refs/heads/proof")
        self.assertEqual(self.cache.requests, cache_requests)
        remote = self.git_fixture.project_root / "org/cmake-dependency"
        self.assertEqual(fixture.git(remote, "rev-parse", "proof"), fixture.git(remote, "rev-parse", "main"))

    def test_disabled_session_respects_explicit_step_global_configuration(self):
        (self.session / "cache-disabled").touch()
        explicit = self.root / "explicit.config"
        explicit.write_text("[user]\n name = Explicit\n")
        self.env["GIT_CONFIG_GLOBAL"] = str(explicit)
        self.assertEqual(self.run_git("config", "--get", "user.name"), "Explicit")


if __name__ == "__main__":
    unittest.main()
