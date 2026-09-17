# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Portable benchmark configuration must reject bad inputs before importing tensor libraries."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from models.demos.llama_3p1_8b_d_p.tests.performance import performance_config as config


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value))


class ConfigurationTests(unittest.TestCase):
    def fixture(self, root):
        repo = root / "repo"
        repo.mkdir()
        sources = {}
        for path in config.required_sources(repo):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic pinned source\n")
            sources[str(path)] = sha(path)
        pins = root / "pins.json"
        write(pins, sources)
        checkpoint = root / "checkpoint"
        checkpoint.mkdir()
        manifest = root / "books.json"
        manifest.write_text("{}")
        value = dict(
            schema_version=1,
            context_length=16384,
            repository=str(repo),
            checkpoint=str(checkpoint),
            source_pins=dict(path=str(pins), sha256=sha(pins)),
            book_fixture=dict(manifest_path=str(manifest), manifest_sha256=sha(manifest)),
        )
        path = root / "config.json"
        write(path, value)
        return path, value, repo, sources

    # A direct benchmark request needs exact inputs, not a hardcoded machine, job or root authorization flag.
    def test_direct_request_is_portable_and_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, value, repo, _ = self.fixture(Path(tmp))
            with mock.patch.object(config, "REPOSITORY", repo), mock.patch.object(
                config, "load_book_fixture"
            ) as fixture:
                actual = config.load_config(path)
                self.assertEqual(actual["context_length"], 16384)
                self.assertIsNone(actual["resource_review"])
                self.assertNotIn("node", actual)
                self.assertNotIn("job_id", actual)
                fixture.assert_called_once_with(
                    value["book_fixture"]["manifest_path"],
                    value["book_fixture"]["manifest_sha256"],
                    16384,
                    value["checkpoint"],
                )

    # Source bytes must be verified before a device import, including the benchmark and production modules.
    def test_missing_changed_and_unpinned_sources_refuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, value, repo, sources = self.fixture(Path(tmp))
            with mock.patch.object(config, "REPOSITORY", repo), mock.patch.object(config, "load_book_fixture"):
                target = next(iter(sources))
                Path(target).write_text("changed")
                with self.assertRaisesRegex(RuntimeError, "source"):
                    config.load_config(path)
                Path(target).write_text("synthetic pinned source\n")
                pins = Path(value["source_pins"]["path"])
                changed = dict(sources)
                del changed[target]
                write(pins, changed)
                value["source_pins"]["sha256"] = sha(pins)
                write(path, value)
                with self.assertRaisesRegex(ValueError, "omits"):
                    config.load_config(path)

    # The context, executable checkout, checkpoint and fixture digest must remain exact and finite.
    def test_invalid_geometry_paths_and_fixture_binding_refuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, value, repo, _ = self.fixture(Path(tmp))
            with mock.patch.object(config, "REPOSITORY", repo), mock.patch.object(config, "load_book_fixture"):
                for patch in (
                    dict(context_length=2048),
                    dict(context_length=16385),
                    dict(context_length=True),
                    dict(context_length=16384.0),
                    dict(repository="/other"),
                    dict(checkpoint="relative"),
                    dict(book_fixture={"manifest_path": "/other", "manifest_sha256": "bad"}),
                ):
                    write(path, dict(value, **patch))
                    with self.subTest(patch=patch), self.assertRaises((ValueError, RuntimeError, FileNotFoundError)):
                        config.load_config(path)

    # Existing site metadata may be retained, but an unbound resource receipt must not become report evidence.
    def test_optional_site_receipt_is_integrity_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, value, repo, _ = self.fixture(Path(tmp))
            receipt = Path(tmp) / "resource.json"
            receipt.write_text("{}")
            value.update(
                schema_version=3,
                resource_review=dict(path=str(receipt), sha256=sha(receipt)),
                node="example",
                job_id="example",
            )
            write(path, value)
            with mock.patch.object(config, "REPOSITORY", repo), mock.patch.object(config, "load_book_fixture"):
                self.assertEqual(config.load_config(path), value)
                receipt.write_text("changed")
                with self.assertRaisesRegex(RuntimeError, "resource_review"):
                    config.load_config(path)

    # The new executing helper must be required even when an otherwise valid map is rehashed after omission.
    def test_progress_helper_must_be_pinned(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, value, repo, sources = self.fixture(Path(tmp))
            helper = repo / "models/demos/llama_3p1_8b_d_p/tests/performance/request_progress.py"
            helper.parent.mkdir(parents=True, exist_ok=True)
            helper.write_text("independent helper fixture\n")
            sources[str(helper)] = sha(helper)
            pins = Path(value["source_pins"]["path"])
            write(pins, sources)
            value["source_pins"]["sha256"] = sha(pins)
            write(path, value)
            with mock.patch.object(config, "REPOSITORY", repo), mock.patch.object(config, "load_book_fixture") as book:
                config.load_config(path)
                book.reset_mock()
                del sources[str(helper)]
                write(pins, sources)
                value["source_pins"]["sha256"] = sha(pins)
                write(path, value)
                with self.assertRaisesRegex(ValueError, "omits"):
                    config.load_config(path)
                book.assert_not_called()

    # Changing the helper after binding must fail before fixture loading, with no tensor imports involved.
    def test_changed_progress_helper_bytes_refuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, value, repo, sources = self.fixture(Path(tmp))
            helper = repo / "models/demos/llama_3p1_8b_d_p/tests/performance/request_progress.py"
            helper.parent.mkdir(parents=True, exist_ok=True)
            helper.write_text("independent helper fixture\n")
            sources[str(helper)] = sha(helper)
            pins = Path(value["source_pins"]["path"])
            write(pins, sources)
            value["source_pins"]["sha256"] = sha(pins)
            write(path, value)
            with mock.patch.object(config, "REPOSITORY", repo), mock.patch.object(config, "load_book_fixture") as book:
                config.load_config(path)
                book.reset_mock()
                helper.write_text("mutated helper\n")
                with self.assertRaisesRegex(RuntimeError, "Pinned source bytes changed"):
                    config.load_config(path)
                book.assert_not_called()
