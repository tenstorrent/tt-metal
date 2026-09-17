# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Portable benchmark inputs; allocation and lease policy belongs to the external runner."""
import json
import re
from pathlib import Path

from models.demos.llama_3p1_8b_d_p.tests.performance.book_fixture_loader import load_book_fixture
from models.demos.llama_3p1_8b_d_p.tests.performance.source_inventory import file_sha, hash_files

REPOSITORY = Path(__file__).resolve().parents[5]
CONTEXTS = (4096, 8192, 16384, 32768, 65536, 131072)


def required_sources(repository):
    """Require the executing benchmark and model modules; the caller may pin additional native/build inputs."""
    package = Path(repository) / "models/demos/llama_3p1_8b_d_p"
    benchmark = package / "tests/performance"
    names = (
        "performance_config",
        "source_inventory",
        "request_progress",
        "book_fixture_loader",
        "book_selection",
        "book_observation",
        "long_context_performance_helpers",
        "long_context_performance_validation",
        "test_long_context_performance",
    )
    result = [benchmark / (name + ".py") for name in names]
    result += [
        package / "tt" / (name + ".py")
        for name in (
            "input",
            "weights",
            "model",
            "kv_cache",
            "attention",
            "decoder",
            "prefill_geometry",
            "rope",
            "qkv",
            "mlp",
            "rms_norm",
            "config",
        )
    ]
    return result


def validate_record(record, name):
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        raise ValueError(name + " must identify an absolute path and SHA256")
    if not Path(record["path"]).is_absolute() or not re.fullmatch("[0-9a-f]{64}", str(record.get("sha256", ""))):
        raise ValueError(name + " must identify an absolute path and SHA256")
    if file_sha(record["path"]) != record["sha256"]:
        raise RuntimeError(name + " provenance changed")


def load_config(path):
    """Check every supplied source byte and frozen fixture before pytest creates a mesh."""
    if not path:
        raise RuntimeError("An explicit performance config is required")
    config = json.loads(Path(path).read_text())
    if type(config.get("schema_version")) is not int or config["schema_version"] not in (1, 3):
        raise ValueError("Benchmark config schema1 or existing site schema3 is required")
    context = config.get("context_length")
    if type(context) is not int or context not in CONTEXTS:
        raise ValueError("Benchmark context must be4K/8K/16K/32K/64K/128K")
    # Respect a caller's closed site contract, without requiring that policy for a direct benchmark.
    for name in ("authorized", "resource_review_accepted"):
        if name in config and config[name] is not True:
            raise RuntimeError("Caller-provided site contract remains closed: " + name)
    if config.get("execution_verified", False) is not False:
        raise ValueError("Input config cannot claim this execution has passed")
    if config.get("execution_scope", "error_free_full32_prefill") != "error_free_full32_prefill":
        raise ValueError("A benchmark config cannot imply golden accuracy acceptance")
    for name in ("repository", "checkpoint"):
        if not isinstance(config.get(name), str) or not Path(config[name]).is_absolute():
            raise ValueError(name + " must be an absolute path")
    if Path(config["repository"]).resolve() != REPOSITORY.resolve():
        raise ValueError("Repository must be the checkout containing this benchmark")
    validate_record(config.get("source_pins"), "source_pins")
    pins = json.loads(Path(config["source_pins"]["path"]).read_text())
    if (
        not isinstance(pins, dict)
        or not pins
        or any(
            not isinstance(name, str)
            or not Path(name).is_absolute()
            or not isinstance(digest, str)
            or not re.fullmatch("[0-9a-f]{64}", digest)
            for name, digest in pins.items()
        )
    ):
        raise ValueError("Source map must contain absolute paths and exact SHA256 strings")
    if not all(str(source) in pins for source in required_sources(REPOSITORY)):
        raise ValueError("Source map omits an executing benchmark or model dependency")
    if hash_files(pins) != pins:
        raise RuntimeError("Pinned source bytes changed")
    fixture = config["book_fixture"]
    validate_record(dict(path=fixture.get("manifest_path"), sha256=fixture.get("manifest_sha256")), "book_fixture")
    load_book_fixture(fixture["manifest_path"], fixture["manifest_sha256"], context, config["checkpoint"])
    config.setdefault("resource_review", None)
    if config["resource_review"] is not None:
        validate_record(config["resource_review"], "resource_review")
    return config
