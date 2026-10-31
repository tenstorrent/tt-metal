#!/usr/bin/env python3
"""Check the autonomous bringup context-length contract.

The contract is intentionally simple:

* target context is the HF-advertised context length;
* supported context must be the target, unless device DRAM proves otherwise;
* later stages must not use a smaller max_model_len-style cap than the
  supported context recorded by the model.

This script is a runner-side guardrail. It avoids broad eval-parameter
inference because names such as "max_length" are overloaded across harnesses.
Stage-review and tti-release instructions handle deliberate eval weakening.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


CONTRACT_PATH = Path("doc/context_contract.json")
CONTEXT_KEYS = {
    "max_model_len",
    "max_model_length",
    "model_max_len",
    "model_max_length",
    "max_context",
    "max_context_length",
    "context_length",
    "max_seq_len",
    "max_sequence_length",
    "max_position_embeddings",
}
DRAM_REASONS = {
    "device_dram",
    "dram",
    "device_dram_capacity",
    "hardware_dram_capacity",
}
TEXT_PATTERNS = [
    re.compile(r"--max-model-len(?:=|\s+)(\d+)"),
    re.compile(r"\bmax_model_len\b\s*[:=]\s*(\d+)"),
    re.compile(r"\bmodel_max_len\b\s*[:=]\s*(\d+)"),
    re.compile(r"\bmax_context(?:_length)?\b\s*[:=]\s*(\d+)"),
]


def slugify(hf_model: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", hf_model.lower()).strip("_")
    return slug


def parse_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"\d+", value.strip()):
        return int(value.strip())
    return None


def first_int(mapping: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = parse_int(mapping.get(key))
        if value is not None:
            return value
    return None


def load_contract(model_dir: Path) -> tuple[dict[str, Any] | None, Path]:
    path = model_dir / CONTRACT_PATH
    if not path.is_file():
        return None, path
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        raise SystemExit(f"Could not parse {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"{path} must contain a JSON object.")
    return data, path


def hf_context_from_local_config(hf_model: str | None) -> int | None:
    if not hf_model:
        return None
    try:
        from transformers import AutoConfig  # type: ignore

        config = AutoConfig.from_pretrained(hf_model, trust_remote_code=True, local_files_only=True)
    except Exception:
        return None

    candidates = []
    for obj in (config, getattr(config, "text_config", None)):
        if obj is None:
            continue
        for attr in ("max_position_embeddings", "max_sequence_length", "max_seq_len", "seq_length", "n_positions"):
            value = parse_int(getattr(obj, attr, None))
            if value is not None:
                candidates.append(value)
    return max(candidates) if candidates else None


def has_dram_limit_evidence(contract: dict[str, Any]) -> bool:
    reason = str(contract.get("limiting_reason") or contract.get("reduction_reason") or "").strip().lower()
    evidence = contract.get("capacity_evidence") or contract.get("dram_evidence") or contract.get("capacity_probe")
    return reason in DRAM_REASONS and bool(evidence)


def iter_json_values(value: Any, path: str = ""):
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            yield from iter_json_values(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from iter_json_values(child, f"{path}[{index}]")
    else:
        yield path, value


def checked_files(model_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for rel in (
        "tt",
        "readiness_vllm",
        "doc/vllm_integration",
        "doc/optimized_vllm",
        "doc/tti_release",
    ):
        root = model_dir / rel
        if root.is_file():
            paths.append(root)
        elif root.is_dir():
            paths.extend(p for p in root.rglob("*") if p.is_file())
    return sorted(set(paths))


def scan_caps(model_dir: Path, minimum_context: int) -> tuple[list[str], list[str]]:
    critical: list[str] = []
    advisory: list[str] = []

    for path in checked_files(model_dir):
        try:
            if path.stat().st_size > 2_000_000:
                continue
        except OSError:
            continue

        suffix = path.suffix.lower()
        rel = path.relative_to(model_dir)

        if suffix == ".json":
            try:
                data = json.loads(path.read_text())
            except Exception:
                continue
            for json_path, value in iter_json_values(data):
                key = json_path.rsplit(".", 1)[-1].split("[", 1)[0].lower()
                amount = parse_int(value)
                if key in CONTEXT_KEYS and amount is not None and amount < minimum_context:
                    critical.append(f"{rel}:{json_path} sets {key}={amount} below {minimum_context}")
            continue

        if suffix not in {".md", ".txt", ".log", ".py", ".yaml", ".yml", ".sh", ".toml"}:
            continue
        try:
            text = path.read_text(errors="ignore")
        except Exception:
            continue
        for line_no, line in enumerate(text.splitlines(), 1):
            for pattern in TEXT_PATTERNS:
                match = pattern.search(line)
                if not match:
                    continue
                amount = int(match.group(1))
                if amount < minimum_context:
                    advisory.append(
                        f"{rel}:{line_no} mentions cap {amount} below {minimum_context}: {line.strip()[:160]}"
                    )

    return critical, advisory


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--hf-model", default="")
    parser.add_argument("--stage", default="")
    parser.add_argument("--require-contract", action="store_true")
    parser.add_argument("--strict-caps", action="store_true")
    args = parser.parse_args()

    if args.model_dir:
        model_dir = Path(args.model_dir)
    elif args.hf_model:
        model_dir = Path("models/autoports") / slugify(args.hf_model)
    else:
        print("Neither --model-dir nor --hf-model was provided.", file=sys.stderr)
        return 3

    contract, contract_path = load_contract(model_dir)
    if contract is None:
        message = f"Missing context contract: {contract_path}"
        if args.require_contract:
            print(message, file=sys.stderr)
            return 2
        print(f"ADVISORY: {message}", file=sys.stderr)
        return 1

    contract_target = first_int(
        contract,
        ("hf_advertised_context", "target_context", "hf_context", "advertised_context"),
    )
    hf_target = hf_context_from_local_config(args.hf_model)
    target = max(v for v in (contract_target, hf_target) if v is not None) if (contract_target or hf_target) else None
    supported = first_int(
        contract,
        ("current_supported_context", "supported_context", "max_supported_context", "served_context", "max_model_len"),
    )

    if target is None:
        print(f"{contract_path} does not record an HF-advertised target context.", file=sys.stderr)
        return 2
    if supported is None:
        print(f"{contract_path} does not record the current supported context.", file=sys.stderr)
        return 2

    if supported < target and not has_dram_limit_evidence(contract):
        print(
            f"{contract_path} supports context {supported}, below HF-advertised {target}, "
            "without device-DRAM capacity evidence.",
            file=sys.stderr,
        )
        return 2

    critical, advisory = scan_caps(model_dir, supported)
    for finding in critical:
        print(f"CONTEXT CAP: {finding}", file=sys.stderr)
    for finding in advisory:
        print(f"ADVISORY CONTEXT CAP: {finding}", file=sys.stderr)

    if critical:
        return 2
    if advisory and args.strict_caps:
        return 2

    detail = "DRAM-limited" if supported < target else "full HF context"
    print(f"Context contract OK for {model_dir}: target={target}, supported={supported} ({detail}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
