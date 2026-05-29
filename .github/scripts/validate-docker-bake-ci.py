#!/usr/bin/env python3
"""Validate CI-facing Docker Bake wiring without building images."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys

BAKE_FILE = "dockerfile/docker-bake.hcl"
HARBOR_PREFIX = "harbor.ci.tenstorrent.net/"
REPO = "ghcr.io/tenstorrent/tt-metal"
# zstd level 22 (max) — matches the BAKE_OUTPUT env in build-docker-artifact.yaml.
# Update here whenever the workflow env constant changes.
BAKE_OUTPUT = "type=image,push=true,compression=zstd,compression-level=22,force-compression=true,oci-mediatypes=true"


TOOL_TAGS = {
    "ccache": f"{REPO}/tt-metalium/tools/ccache:test",
    "mold": f"{REPO}/tt-metalium/tools/mold:test",
    "doxygen": f"{REPO}/tt-metalium/tools/doxygen:test",
    "cba": f"{REPO}/tt-metalium/tools/cba:test",
    "gdb": f"{REPO}/tt-metalium/tools/gdb:test",
    "cmake": f"{REPO}/tt-metalium/tools/cmake:test",
    "yq": f"{REPO}/tt-metalium/tools/yq:test",
    "zstd": f"{REPO}/tt-metalium/tools/zstd:test",
    "sfpi": f"{REPO}/tt-metalium/tools/sfpi:test",
    "openmpi": f"{REPO}/tt-metalium/tools/openmpi:test",
}

VENV_TAGS = {
    "ci-build-venv": f"{REPO}/tt-metalium/python-venv/ci-build:test",
    "ci-test-venv": f"{REPO}/tt-metalium/python-venv/ci-test:test",
}


def _parse_output_kvs(spec: str) -> dict[str, str]:
    """Parse 'type=image,push=true,...' into a dict of lowercase string values."""
    return {k: v.lower() for k, v in (kv.split("=", 1) for kv in spec.split(","))}


def output_matches(actual: list | None, expected: str) -> bool:
    """Return True if *actual* from --print JSON matches *expected*.

    Docker Buildx may render the output field in several ways depending on version:
      - String list  : ["type=image,push=true,..."]          (older Buildx)
      - CSV-split    : ["type=image", "push=true", ...]      (--set splits on commas)
      - Object list  : [{"type": "image", "push": True, ...}] (newer Buildx)
    """
    if not actual:
        return False
    # Exact string match (oldest format)
    if actual == [expected]:
        return True
    # CSV-split: --set may split the comma-separated value into individual list items
    if actual == expected.split(","):
        return True
    # Parsed-object format: newer Buildx emits a dict per output spec
    if len(actual) == 1 and isinstance(actual[0], dict):
        expected_kv = _parse_output_kvs(expected)
        actual_kv = {k: str(v).lower() for k, v in actual[0].items()}
        return actual_kv == expected_kv
    return False


def harbor_prefixed(image: str, prefix: str = HARBOR_PREFIX) -> str:
    if not image:
        return ""
    if image.startswith(prefix):
        return image
    return f"{prefix}{image}"


def run_bake_print(*args: str) -> dict:
    completed = subprocess.run(
        ["docker", "buildx", "bake", "-f", BAKE_FILE, "--print", *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return json.loads(completed.stdout)


def main() -> int:
    if shutil.which("docker") is None:
        print("docker is required for Bake validation", file=sys.stderr)
        return 1

    sets: list[str] = []
    for target in ("ci-build", "ci-test", "dev"):
        for tool, tag in TOOL_TAGS.items():
            sets.extend(
                [
                    "--set",
                    f"{target}.contexts.{tool}-layer=docker-image://{harbor_prefixed(tag)}",
                ]
            )
        sets.extend(
            [
                "--set",
                f"{target}.contexts.ci-build-venv-layer=docker-image://{harbor_prefixed(VENV_TAGS['ci-build-venv'])}",
                "--set",
                f"{target}.contexts.ci-test-venv-layer=docker-image://{harbor_prefixed(VENV_TAGS['ci-test-venv'])}",
                "--set",
                f"{target}.tags={REPO}/tt-metalium/{target}:test",
                "--set",
                f"{target}.output={BAKE_OUTPUT}",
            ]
        )

    for target in ("basic-dev", "basic-ttnn-runtime"):
        for tool in ("cmake", "sfpi", "openmpi", "ccache"):
            sets.extend(
                [
                    "--set",
                    f"{target}.contexts.{tool}-layer=docker-image://{harbor_prefixed(TOOL_TAGS[tool])}",
                ]
            )
        sets.extend(
            [
                "--set",
                f"{target}.tags={REPO}/tt-metalium/{target}:test",
                "--set",
                f"{target}.output={BAKE_OUTPUT}",
            ]
        )

    for tool in ("ccache", "mold", "zstd", "sfpi", "openmpi"):
        sets.extend(
            [
                "--set",
                f"manylinux.contexts.{tool}-layer=docker-image://{harbor_prefixed(TOOL_TAGS[tool])}",
            ]
        )
    sets.extend(
        [
            "--set",
            f"manylinux.tags={REPO}/tt-metalium/manylinux:test",
            "--set",
            f"manylinux.output={BAKE_OUTPUT}",
        ]
    )

    for tool in ("ccache", "sfpi"):
        sets.extend(
            [
                "--set",
                f"evaluation.contexts.{tool}-layer=docker-image://{harbor_prefixed(TOOL_TAGS[tool])}",
            ]
        )
    sets.extend(
        [
            "--set",
            f"evaluation.tags={REPO}/tt-metalium/evaluation:test",
            "--set",
            f"evaluation.output={BAKE_OUTPUT}",
        ]
    )

    rendered = run_bake_print(
        *sets,
        "ci-build",
        "ci-test",
        "dev",
        "basic-dev",
        "basic-ttnn-runtime",
        "manylinux",
        "evaluation",
    )
    targets = rendered["target"]

    for name in (
        "ci-build",
        "ci-test",
        "dev",
        "basic-dev",
        "basic-ttnn-runtime",
        "manylinux",
        "evaluation",
    ):
        target = targets[name]
        if not target.get("tags"):
            raise AssertionError(f"{name} has no tag after CI overrides")
        if not output_matches(target.get("output"), BAKE_OUTPUT):
            raise AssertionError(f"{name} output override was not preserved " f"(got {target.get('output')!r})")
        for context in target.get("contexts", {}).values():
            if context.startswith("docker-image://") and HARBOR_PREFIX not in context:
                raise AssertionError(f"{name} context did not use Harbor prefix: {context}")

    if harbor_prefixed(f"{REPO}/image:tag") != f"{HARBOR_PREFIX}{REPO}/image:tag":
        raise AssertionError("Harbor prefix composition failed")
    if harbor_prefixed(f"{HARBOR_PREFIX}{REPO}/image:tag") != f"{HARBOR_PREFIX}{REPO}/image:tag":
        raise AssertionError("Harbor prefix composition double-prefixed an image")
    if harbor_prefixed(f"{REPO}/image:tag", "") != f"{REPO}/image:tag":
        raise AssertionError("Fallback prefix composition failed")

    print("Docker Bake CI validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
