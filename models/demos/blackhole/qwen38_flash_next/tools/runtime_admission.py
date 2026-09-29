"""Runtime admission: the server runs on the ttnn built from this checkout.

The interpreter's ``ttnn`` must import from this repository (the ``ttnn`` package and its compiled extension under
the checkout); the checkout's git head and tree and the extension's SHA-256 are the runtime identity the run
records (the result document, ``READY``, ``/health``).  "Built from this checkout" replaces the development team's sealed
archives.  The development launchers keep their archive seal through ``QWEN38_RUNTIME_SEAL_MODULE`` (a module with
``add_arguments`` and ``identity``; its code lives outside the public tree).  ``prepare_cpu`` then binds the
checkpoint, the caches and the optional CPU-staged BF4 corpus through ``live_decode_diagnostic``.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import ttnn
from models.demos.blackhole.qwen38_flash_next.tools.evidence_records import Marker
from models.demos.blackhole.qwen38_flash_next.tools.hardware_profiles import ResidentHardwareProfile
from models.demos.blackhole.qwen38_flash_next.tools.live_decode_diagnostic import (
    DEFAULT_CPU_ORACLE,
    Qwen38RetainedCPUTokenOracle,
    load_retained_cpu_token_oracle,
    prepare_live_decode_diagnostic,
)
from models.demos.blackhole.qwen38_flash_next.tools.resident_decode import ResidentDecodeError
from models.demos.blackhole.qwen38_flash_next.ttnn.builder import RESIDENT_MAX_QSA_CACHE_CAPACITY

REPO_ROOT = Path(__file__).resolve().parents[5]
SEAL_MODULE_VARIABLE = "QWEN38_RUNTIME_SEAL_MODULE"
# The head of a tree without git history (a container image); the vLLM adapter reads the same variable first.
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


def _seal_module():
    name = os.environ.get(SEAL_MODULE_VARIABLE)
    return None if not name else importlib.import_module(name)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    for name, _key, kind in EXPECTATIONS:
        parser.add_argument(
            f"--{name}", type=kind, default=None, help="expected value; the admitted runtime must match"
        )
    seal = _seal_module()
    if seal is not None:
        seal.add_arguments(parser)


def git_identity(root: Path = REPO_ROOT) -> dict[str, Any]:
    """Head, tree and cleanliness of the checkout.  A tree without git history (a container image ships the tree
    without ``.git``) is admitted only when ``QWEN38_TT_METAL_SHA`` names its head, the 40-hex commit the image was
    built from (recorded as ``source`` ``declared``, no tree, not dirty); a checkout with history is its own identity,
    and the variable, if set there, must agree with the head (a disagreement is refused, never overridden)."""

    declared = os.environ.get(DECLARED_HEAD_VARIABLE)
    if declared is not None and (len(declared) != 40 or any(c not in "0123456789abcdef" for c in declared)):
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
    """The runtime identity this run records: the checkout's build, or (a configured seal module) the development team's sealed
    archive with the checkout as the model source.  Every expected value given must match."""

    seal = _seal_module()
    if seal is None:
        identity = checkout_identity()
    else:
        extension = Path(ttnn._ttnn.__file__).resolve(strict=True)
        identity = {
            "schema": SCHEMA,
            "source": "sealed",
            **git_identity(),
            "tt_metal_sha": args.tt_metal_sha,
            "python_module": str(Path(ttnn.__file__).resolve(strict=True)),
            "extension": str(extension),
            "extension_sha256": sha256_of(extension),
            "bundle": seal.identity(args),
        }
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
) -> tuple[Any, Qwen38RetainedCPUTokenOracle]:
    """Bind the checkpoint, the cache roots, the optional CPU-staged BF4 corpus and the CPU oracle; the resident
    build follows ``--allocated-context`` (the default when the caller has no such option); the mesh contract takes
    the profile's route, which must be resolved by now."""

    if hardware_profile.route is None:
        raise RuntimeAdmissionError(f"profile {hardware_profile.lane} has no route yet; resolve it before preparing")
    if identity is None:
        identity = declared_runtime(args) if getattr(args, "tt_metal_sha", None) is not None else admit_runtime(args)
    producer_argument = getattr(args, "bf4_producer_identity", None)
    producer = None if producer_argument is None else json.loads(Path(producer_argument).read_text(encoding="utf-8"))
    marker("before-cpu-preparation")
    prepared = prepare_live_decode_diagnostic(
        checkpoint_root=args.checkpoint,
        component_cache_root=args.component_cache_root,
        routed_bf4_scratch_root=args.routed_bf4_scratch_root,
        model_io_cache_root=args.model_io_cache_root,
        tt_metal_sha=identity["tt_metal_sha"],
        runtime_extension=Path(identity["extension"]),
        runtime_sha256=identity["extension_sha256"],
        allocated_context=getattr(args, "allocated_context", RESIDENT_MAX_QSA_CACHE_CAPACITY),
        physical_ids=hardware_profile.route,
        bf4_corpus_root=getattr(args, "bf4_corpus", None),
        bf4_corpus_verification=getattr(args, "bf4_corpus_verification", None),
        bf4_producer=producer,
        marker=marker,
    )
    oracle = load_retained_cpu_token_oracle(getattr(args, "cpu_oracle", None) or DEFAULT_CPU_ORACLE)
    marker("after-cpu-preparation")
    return prepared, oracle
