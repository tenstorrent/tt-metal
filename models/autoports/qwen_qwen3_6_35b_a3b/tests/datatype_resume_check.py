# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Run the installed resume gate unchanged, retaining failed logit pairs."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import torch
from readiness_check import run_prefill_resume_check as runner

from models.autoports.qwen_qwen3_6_35b_a3b.tests.datatype_runtime_identity import (
    runtime_identity,
    source_input_identity,
)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--diagnostics-dir", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    args.diagnostics_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = args.diagnostics_dir / "run_provenance.json"
    if provenance_path.exists():
        raise FileExistsError(provenance_path)
    original_case = runner.run_case
    original_compare = runner.compare_logits
    original_run = runner.run_prefill_resume_check
    original_import = runner._import_build_generator

    def diagnostic_run(**kwargs):
        model_root = Path(kwargs["model_dir"])
        selected = model_root / "doc/datatype_sweep/selected_precision_config.json"

        def selected_sha():
            return hashlib.sha256(selected.read_bytes()).hexdigest() if selected.is_file() else None

        inputs = [Path(__file__), Path(runner.__file__)]
        if selected.is_file():
            inputs.append(selected)
        for value in (
            kwargs.get("text_file"),
            kwargs.get("prompts_file"),
            runner.BOOK_RELPATH,
            runner.LEGACY_BOOK_RELPATH,
        ):
            if value is not None and Path(value).is_file():
                inputs.append(Path(value))
        for value in (kwargs.get("build_kwargs") or {}).values():
            if isinstance(value, str) and value and len(value) < 4096:
                path = next((p for p in (model_root / value, Path(value)) if p.is_file()), None)
                if path is not None:
                    inputs.append(path)
        provenance = {
            "status": "running",
            "command": sys.argv,
            "gate_modified": False,
            "build_kwargs": kwargs.get("build_kwargs") or {},
            "selected_precision_artifact": str(selected) if selected.is_file() else None,
            "selected_precision_sha256": selected_sha(),
            "native_runtime_identity": runtime_identity(),
            "source_input_identity": source_input_identity(model_root=model_root, explicit_paths=inputs),
            "native_identity_capture": "Actual resume runner process, before generator construction and after generator teardown",
        }

        def save_provenance():
            provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")

        def audited_import(root):
            build = original_import(root)

            def audited_build(*build_args, **build_kwargs):
                generator = build(*build_args, **build_kwargs)
                provenance["runtime_precision"] = generator.model.precision_summary()
                provenance["actual_precision_policy"] = generator.model.precision_config
                save_provenance()
                return generator

            return audited_build

        save_provenance()
        try:
            with patch.object(runner, "_import_build_generator", audited_import):
                record = original_run(**kwargs)
            provenance["status"] = "completed"
            record["runtime_provenance"] = provenance
            record["runtime_provenance_artifact"] = str(provenance_path)
            return record
        except BaseException:
            provenance["status"] = "runtime_error"
            raise
        finally:
            errors = []
            try:
                provenance["native_runtime_identity_after"] = runtime_identity()
                provenance["source_input_identity_after"] = source_input_identity(
                    model_root=model_root, explicit_paths=inputs
                )
                provenance["selected_precision_sha256_after"] = selected_sha()
                for name in ("native_runtime_identity", "source_input_identity", "selected_precision_sha256"):
                    if provenance[name + "_after"] != provenance[name]:
                        errors.append(name + " changed")
            except Exception as exc:
                errors.append(str(exc))
            if errors:
                provenance["status"] = "provenance_failed"
                provenance["errors"] = errors
            save_provenance()
            if errors and sys.exc_info()[0] is None:
                raise RuntimeError("Resume provenance failed; see run_provenance.json")

    def diagnostic_case(**kwargs):
        pair = []

        def compare(reference, candidate):
            if not pair:
                pair.extend((reference, candidate))
            return original_compare(reference, candidate)

        with patch.object(runner, "compare_logits", compare):
            result = original_case(**kwargs)
        if not result["pass"]:
            stem = f"L{len(kwargs['tokens'])}_s{kwargs['start']}"
            tensor_path = args.diagnostics_dir / f"{stem}.pt"
            if tensor_path.exists():
                raise FileExistsError(tensor_path)
            torch.save({"fresh_logits": pair[0], "resumed_logits": pair[1]}, tensor_path)
            metadata = {
                "gate_result": result,
                "tokens": kwargs["tokens"],
                "positions": runner.window_positions(kwargs["start"], len(kwargs["tokens"]), kwargs["window"]),
                "runtime_precision": kwargs["generator"].model.precision_summary(),
                "runtime_provenance_artifact": str(provenance_path),
                "tensor_path": str(tensor_path),
                "gate_modified": False,
            }
            (args.diagnostics_dir / f"{stem}.json").write_text(json.dumps(metadata, indent=2) + "\n")
        return result

    with (
        patch.object(runner, "run_case", diagnostic_case),
        patch.object(runner, "run_prefill_resume_check", diagnostic_run),
    ):
        return runner._main(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
