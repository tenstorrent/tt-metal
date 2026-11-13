# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Runtime admission: the server runs on the ttnn built from this checkout.

The interpreter's ``ttnn`` package and compiled extension must come from this repository; the checkout's git head and
tree and the extension's SHA-256 are the runtime identity the run records.  ``prepare_cpu`` then binds the checkpoint
and the caches.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any

import ttnn
from models.demos.blackhole.qwen38_flash_next.tools.evidence_records import Marker
from models.demos.blackhole.qwen38_flash_next.tools.hardware_profiles import ResidentHardwareProfile
from models.demos.blackhole.qwen38_flash_next.tools.live_decode_construction import prepare_live_decode
from models.demos.blackhole.qwen38_flash_next.tools.resident_decode import ResidentDecodeError
from models.demos.blackhole.qwen38_flash_next.ttnn.builder import RESIDENT_MAX_QSA_CACHE_CAPACITY

REPO_ROOT = Path(__file__).resolve().parents[5]
# The head of a tree without git history (a container image).
DECLARED_HEAD_VARIABLE = "QWEN38_TT_METAL_SHA"
SCHEMA = "qwen38-runtime-identity/v1"
# Expected values a launcher may pass; the admitted identity must match each one given.
EXPECTATIONS = (
    ("runtime-extension", "extension", Path),
    ("runtime-sha256", "extension_sha256", str),
    ("tt-metal-sha", "tt_metal_sha", str),
    ("source-head", "head", str),
    ("source-tree", "tree", str),
)


class RuntimeAdmissionError(ResidentDecodeError):
    pass


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def add_arguments(parser: argparse.ArgumentParser) -> None:
    for name, _key, kind in EXPECTATIONS:
        parser.add_argument(
            f"--{name}", type=kind, default=None, help="expected value; the admitted runtime must match"
        )


def git_identity(root: Path = REPO_ROOT) -> dict[str, Any]:
    """Head, tree and cleanliness of the checkout.  A tree without git history is admitted only when
    ``QWEN38_TT_METAL_SHA`` names its 40-hex head; with history, the variable, if set, must agree with the head."""

    declared = os.environ.get(DECLARED_HEAD_VARIABLE)
    if declared is not None and (len(declared) != 40 or any(digit not in "0123456789abcdef" for digit in declared)):
        raise RuntimeAdmissionError(f"{DECLARED_HEAD_VARIABLE} must be lowercase 40-hex, got {declared!r}")
    if not (root / ".git").exists():
        if declared is None:
            raise RuntimeAdmissionError(
                f"{root} is not a git checkout and {DECLARED_HEAD_VARIABLE} is unset: a tree without git history "
                "serves only with the commit it was built from declared in that variable"
            )
        return {"repo": str(root), "head": declared, "tree": None, "dirty": False, "source": "declared"}

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ("git", "-C", str(root), *arguments), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if completed.returncode != 0:
            raise RuntimeAdmissionError(f"{root} is not a git checkout: git {arguments[0]}: {completed.stderr.strip()}")
        return completed.stdout.strip()

    dirty = git("status", "--porcelain=v1", "--untracked-files=no")
    identity = {
        "repo": str(root),
        "head": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "dirty": bool(dirty),
    }
    if declared is not None and declared != identity["head"]:
        raise RuntimeAdmissionError(
            f"{DECLARED_HEAD_VARIABLE} {declared} differs from the checkout head {identity['head']}"
        )
    return identity


def source_proof(expected_head: str, expected_tree: str) -> dict[str, Any]:
    """The development source contract: this checkout is clean at exactly the expected head and tree."""

    for value, label in ((expected_head, "source head"), (expected_tree, "source tree")):
        if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"{label} must be lowercase 40-hex")
    identity = git_identity()
    if (identity["head"], identity["tree"]) != (expected_head, expected_tree) or identity["dirty"]:
        raise RuntimeAdmissionError(
            f"source provenance differs: head={identity['head']} tree={identity['tree']} dirty={identity['dirty']}"
        )
    return {"worktree": identity["repo"], "head": identity["head"], "tree": identity["tree"], "clean": True}


def checkout_identity() -> dict[str, Any]:
    """The loaded ttnn is this checkout's build: package and extension under the repository, tt-metal = the head."""

    module = Path(ttnn.__file__).resolve(strict=True)
    extension = Path(ttnn._ttnn.__file__).resolve(strict=True)
    outside = [str(path) for path in (module, extension) if not path.is_relative_to(REPO_ROOT)]
    if outside:
        raise RuntimeAdmissionError(
            f"ttnn is not this checkout's build: {outside} are outside {REPO_ROOT}; run the launcher of the checkout "
            "whose python_env imports it, or build this one (build_metal.sh, create_venv.sh)"
        )
    git = git_identity()
    return {
        "schema": SCHEMA,
        "source": "checkout",
        **git,
        "tt_metal_sha": git["head"],
        "python_module": str(module),
        "extension": str(extension),
        "extension_sha256": sha256_of(extension),
    }


def admit_runtime(args: argparse.Namespace) -> dict[str, Any]:
    """The runtime identity this run records: the checkout's build.  Every expected value given must match."""

    identity = checkout_identity()
    differences = []
    for name, key, kind in EXPECTATIONS:
        expected = getattr(args, name.replace("-", "_"), None)
        if expected is None:
            continue
        actual = identity[key]
        if kind is Path:
            expected, actual = str(expected.resolve(strict=True)), str(Path(actual).resolve(strict=True))
        if expected != actual:
            differences.append(f"--{name} {expected} vs admitted {actual}")
    if getattr(args, "source_head", None) is not None and identity["dirty"]:
        differences.append("--source-head given, checkout dirty")
    if differences:
        raise RuntimeAdmissionError("runtime identity differs: " + "; ".join(differences))
    return identity


def declared_runtime(args: argparse.Namespace) -> dict[str, Any]:
    """A runtime a development runner names outright (``--tt-metal-sha``, ``--runtime-extension``, ``--runtime-sha256``):
    the loaded extension is proven to be that file with that digest; the checkout is the model source."""

    proof = runtime_proof(args.runtime_extension, args.runtime_sha256)
    if len(args.tt_metal_sha) != 40 or any(character not in "0123456789abcdef" for character in args.tt_metal_sha):
        raise ValueError("tt_metal_sha must be lowercase 40-hex")
    return {
        "schema": SCHEMA,
        "source": "declared",
        **git_identity(),
        "tt_metal_sha": args.tt_metal_sha,
        "python_module": proof["python_module"],
        "extension": proof["extension"],
        "extension_sha256": proof["sha256"],
    }


def runtime_proof(extension: Path, expected_sha256: str) -> dict[str, str]:
    """The loaded extension is the named file with the expected digest."""

    loaded = Path(ttnn._ttnn.__file__).resolve(strict=True)
    digest = sha256_of(loaded)
    if loaded != extension.resolve(strict=True) or digest != expected_sha256:
        raise RuntimeAdmissionError(
            f"loaded TTNN extension differs: {loaded}/{digest} != {extension}/{expected_sha256}"
        )
    return {"python_module": str(Path(ttnn.__file__).resolve()), "extension": str(loaded), "sha256": digest}


def prepare_cpu(
    args: argparse.Namespace,
    *,
    marker: Marker,
    hardware_profile: ResidentHardwareProfile,
    identity: dict[str, Any] | None = None,
) -> Any:
    """Bind the checkpoint and the cache roots.  The resident build follows ``--allocated-context`` (its default when
    the caller has none); the profile's route must be resolved by now."""

    if hardware_profile.route is None:
        raise RuntimeAdmissionError(f"profile {hardware_profile.lane} has no route yet; resolve it before preparing")
    if identity is None:
        identity = declared_runtime(args) if getattr(args, "tt_metal_sha", None) is not None else admit_runtime(args)
    marker("before-cpu-preparation")
    prepared = prepare_live_decode(
        checkpoint_root=args.checkpoint,
        component_cache_root=args.component_cache_root,
        routed_bf4_scratch_root=args.routed_bf4_scratch_root,
        model_io_cache_root=args.model_io_cache_root,
        tt_metal_sha=identity["tt_metal_sha"],
        runtime_extension=Path(identity["extension"]),
        runtime_sha256=identity["extension_sha256"],
        allocated_context=getattr(args, "allocated_context", RESIDENT_MAX_QSA_CACHE_CAPACITY),
        physical_ids=hardware_profile.route,
        marker=marker,
    )
    marker("after-cpu-preparation")
    return prepared
