# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Write an unselected configurable-weight BFP8 policy and its exact diff."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original = args.source.read_bytes()
    policy = json.loads(original)
    changes = []

    def visit(node, path=()):
        for key, value in node.items():
            if key == "fixed":
                continue
            if isinstance(value, dict):
                visit(value, (*path, key))
            elif value == "bfloat4_b":
                if not any(part in key for part in ("weight_dtype", "gate_dtype", "down_dtype")):
                    raise ValueError(f"Unexpected nonweight BFP4 field: {path}/{key}")
                node[key] = "bfloat8_b"
                changes.append({"path": ".".join((*path, key)), "before": value, "after": node[key]})

    visit(policy)
    policy["config_id"] = "diagnostic_unselected_configurable_weight_bfp8"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(policy, indent=2) + "\n")
    manifest = {
        "status": "diagnostic_unselected_not_accuracy_validated",
        "source_sha256": hashlib.sha256(original).hexdigest(),
        "candidate_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "changes": changes,
        "unchanged": "Fixed prefill policy, activations, KV cache, collectives, fidelity, context and sampling",
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"changed_weight_fields": len(changes), "candidate_sha256": manifest["candidate_sha256"]}))


if __name__ == "__main__":
    main()
