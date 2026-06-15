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


def get_tools_from_group(group_name: str = "tools") -> list[str]:
    """Derive all tool names from the bake 'tools' group targets."""
    bake = run_bake_print(group_name)
    return sorted(bake["group"][group_name]["targets"])


def get_tools_for_target(target: str) -> list[str]:
    """Derive tool names consumed by a target from its '-layer' contexts."""
    bake = run_bake_print(target)
    contexts = bake["target"][target].get("contexts", {})
    return sorted(
        key.removesuffix("-layer") for key in contexts if key.endswith("-layer") and not key.startswith("ci-")
    )


def main() -> int:
    if shutil.which("docker") is None:
        print("docker is required for Bake validation", file=sys.stderr)
        return 1

    # Derive TOOL_TAGS from bake's tools group — no hardcoded tool list
    all_tools = get_tools_from_group("tools")
    TOOL_TAGS = {t: f"{REPO}/tt-metalium/tools/{t}:test" for t in all_tools}

    # Derive per-target tool lists from bake contexts
    main_tools = get_tools_for_target("ci-build")
    basic_tools = get_tools_for_target("basic-dev")
    manylinux_tools = get_tools_for_target("manylinux")
    evaluation_tools = get_tools_for_target("evaluation")

    sets: list[str] = []
    venv_contexts_by_target = {
        "ci-build-light": {},
        "ci-build": {"ci-build-venv-layer": VENV_TAGS["ci-build-venv"]},
        "ci-test-light": {},
        "ci-test": {"ci-test-venv-layer": VENV_TAGS["ci-test-venv"]},
        "dev-light": {},
        "dev": {"ci-test-venv-layer": VENV_TAGS["ci-test-venv"]},
    }

    for target, venv_contexts in venv_contexts_by_target.items():
        for tool in main_tools:
            sets.extend(
                [
                    "--set",
                    f"{target}.contexts.{tool}-layer=docker-image://{harbor_prefixed(TOOL_TAGS[tool])}",
                ]
            )
        for context_name, tag in venv_contexts.items():
            sets.extend(
                [
                    "--set",
                    f"{target}.contexts.{context_name}=docker-image://{harbor_prefixed(tag)}",
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

    for target in ("basic-dev", "basic-ttnn-runtime"):
        for tool in basic_tools:
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

    for tool in manylinux_tools:
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

    for tool in evaluation_tools:
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
        "ci-build-light",
        "ci-build",
        "ci-test-light",
        "ci-test",
        "dev-light",
        "dev",
        "basic-dev",
        "basic-ttnn-runtime",
        "manylinux",
        "evaluation",
    )
    targets = rendered["target"]

    for name in (
        "ci-build-light",
        "ci-build",
        "ci-test-light",
        "ci-test",
        "dev-light",
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

    for name, expected_venv_contexts in venv_contexts_by_target.items():
        actual_venv_contexts = {
            key for key in targets[name].get("contexts", {}) if key in {"ci-build-venv-layer", "ci-test-venv-layer"}
        }
        if actual_venv_contexts != set(expected_venv_contexts):
            raise AssertionError(
                f"{name} venv contexts mismatch: expected {sorted(expected_venv_contexts)}, "
                f"got {sorted(actual_venv_contexts)}"
            )

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
