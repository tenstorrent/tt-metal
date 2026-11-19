# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Run installed P1 unchanged, adding immutable, actual-process cache provenance.

Usage: python -m ...tests.datatype_p1_runner --evidence RUN.process.json -- <runner args>
Existing unbound/mismatched caches fail closed; preserve them before deliberately
rescoring. Imports and --help are CPU-only. Only main's execution imports TTNN.
"""

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

from models.autoports.qwen_qwen3_6_35b_a3b.tests.datatype_runtime_identity import (
    digest,
    runtime_identity,
    source_input_identity,
)


def file_sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def jsonable(value):
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def file_inputs(model, kwargs):
    result = {}
    for key, value in sorted(kwargs.items()):
        if not isinstance(value, str) or not value or len(value) > 4096:
            continue
        for path in (model / value, Path(value)):
            if path.is_file():
                result[key] = dict(path=str(path.resolve()), sha256=file_sha(path))
                break
    return result


def binding_path(score_path, cid, sha):
    if Path(cid).name != cid or cid in (".", ".."):
        raise ValueError("P1 candidate IDs must be safe file names")
    return score_path.parent / "bindings" / cid / (sha + ".json")


def immutable_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert json.loads(path.read_text()) == value, f"Immutable evidence collision: {path}"
        return
    with path.open("x") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


class CacheGuard:
    """Wrap only persistence; all numerical computation remains in the runner."""

    def __init__(self, runner, args, evidence, native, sources):
        self.runner, self.args, self.evidence = runner, args, evidence
        self.model = Path(args.model_dir).resolve()
        self.native, self.sources = native, sources
        self.original_load, self.original_save = runner.load_scores, runner.save_scores
        self.used = {}
        # Subset/selection changes intentionally reuse identical numerical scores.
        excluded = {
            "candidates",
            "reference",
            "selected",
            "only",
            "higher_precision",
            "max_candidates",
            "no_score_cache",
        }
        self.settings = jsonable({key: value for key, value in vars(args).items() if key not in excluded})
        self.run_inputs = source_input_identity(
            model_root=self.model,
            explicit_paths=self.explicit_paths(),
        )

    def explicit_paths(self):
        paths = [self.args.candidates]
        paths.extend(item["path"] for item in file_inputs(self.model, self.settings).values())
        if self.args.text_file:
            paths.append(self.args.text_file)
        candidates = self.runner.load_candidates(self.args.candidates, self.args.max_candidates)
        for kwargs in candidates.values():
            paths.extend(item["path"] for item in file_inputs(self.model, kwargs).values())
        return sorted(set(map(str, paths)))

    def stable(self):
        assert runtime_identity() == self.native, "Actual P1 process native identity changed"
        assert source_input_identity(model_root=self.model) == self.sources, "P1 Python source identity changed"
        assert (
            source_input_identity(model_root=self.model, explicit_paths=self.explicit_paths()) == self.run_inputs
        ), "P1 config/text/candidate inputs changed during execution"

    def expected_context(self, cid, sha, kwargs, against):
        ref = None
        if against is not None:
            ref_id = against["id"]
            assert ref_id in self.used, "Reference must have a validated creation binding before candidate scoring"
            used = self.used[ref_id]
            ref_path = self.runner.scores_path(self.model, ref_id)
            assert file_sha(ref_path) == used["score_sha256"], "Reference score changed after it was consumed"
            ref = dict(
                id=ref_id,
                score_sha256=used["score_sha256"],
                binding_sha256=used["binding_sha256"],
                build_kwargs=jsonable(against["kwargs"]),
                input_files=file_inputs(self.model, against["kwargs"]),
            )
        return dict(
            schema_version=1,
            candidate_id=cid,
            text_sha=sha,
            text_file_sha256=file_sha(self.args.text_file) if self.args.text_file else None,
            build_kwargs=jsonable(kwargs),
            input_files=file_inputs(self.model, kwargs),
            scored_against=ref,
            runtime_identity=self.native,
            source_identity=self.sources,
            settings=self.settings,
            settings_input_files=file_inputs(self.model, self.settings),
        )

    def record(self, cid, path, binding, kind):
        record = dict(
            score_artifact=str(self.runner.scores_path(self.model, cid)),
            score_sha256=binding["score_sha256"],
            binding_artifact=str(path),
            binding_sha256=file_sha(path),
            disposition=kind,
        )
        self.used[cid] = record
        self.evidence["score_bindings"] = dict(self.used)

    def load(self, model_dir, cid, sha, kwargs, against):
        score_path = self.runner.scores_path(model_dir, cid)
        if not score_path.exists():
            return None
        self.stable()
        score_sha = file_sha(score_path)
        path = binding_path(score_path, cid, score_sha)
        assert path.is_file(), f"Unbound score cache {score_path}; preserve/remove it explicitly before rescoring"
        binding = json.loads(path.read_text())
        context = self.expected_context(cid, sha, kwargs, against)
        assert binding["context"] == context, f"Stale P1 cache binding for {cid}; preserve/remove it before rescoring"
        assert binding["context_digest"] == digest(context) and binding["score_sha256"] == score_sha
        rows = self.original_load(model_dir, cid, sha, kwargs, against)
        assert rows is not None, f"Installed runner rejected bound score cache {cid}"
        self.record(cid, path, binding, "reused")
        return rows

    def save(self, model_dir, cid, sha, kwargs, against, rows):
        self.stable()
        context = self.expected_context(cid, sha, kwargs, against)
        score_path = self.runner.scores_path(model_dir, cid)
        # Never overwrite historical score bytes, even with --no-score-cache.
        assert not score_path.exists(), f"Refusing to overwrite existing score cache: {score_path}"
        self.original_save(model_dir, cid, sha, kwargs, against, rows)
        self.stable()
        score_sha = file_sha(score_path)
        binding = dict(score_sha256=score_sha, context=context, context_digest=digest(context))
        path = binding_path(score_path, cid, score_sha)
        immutable_json(path, binding)
        self.record(cid, path, binding, "created")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("runner_args", nargs=argparse.REMAINDER)
    options = parser.parse_args(argv)
    forwarded = options.runner_args
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--model-dir", type=Path, required=True)
    model = probe.parse_known_args(forwarded)[0].model_dir.resolve()
    assert not options.evidence.exists(), "Refusing to overwrite process provenance"
    options.evidence.parent.mkdir(parents=True, exist_ok=True)
    sources = source_input_identity(model_root=model)
    from readiness_check import run_precision_reasoning as runner

    import ttnn  # noqa: F401 -- actual scoring process, never imported by CPU-only checks

    native = runtime_identity()
    assert source_input_identity(model_root=model) == sources, "Source changed while importing P1"
    evidence = dict(
        schema_version=1,
        command=[sys.executable, *sys.argv],
        pid=os.getpid(),
        runtime_identity_before=native,
        source_identity_before=sources,
        status="running",
        score_bindings={},
    )
    original_run, original_load, original_save = runner.run, runner.load_scores, runner.save_scores
    guard = None

    def guarded_run(args, **kwargs):
        nonlocal guard
        guard = CacheGuard(runner, args, evidence, native, sources)
        evidence["run_input_identity"] = guard.run_inputs
        evidence["settings"] = guard.settings
        runner.load_scores, runner.save_scores = guard.load, guard.save
        return original_run(args, **kwargs)

    runner.run = guarded_run
    status = 1
    try:
        options.evidence.write_text(json.dumps(evidence, indent=2) + "\n")
        status = runner.main(forwarded)
        evidence["runner_exit_code"] = status
        if guard is not None:
            guard.stable()
        evidence["status"] = "complete"
        return status
    except BaseException:
        evidence["status"] = "error"
        evidence["error"] = traceback.format_exc()
        raise
    finally:
        runner.run, runner.load_scores, runner.save_scores = original_run, original_load, original_save
        try:
            evidence["runtime_identity_after"] = runtime_identity()
            evidence["source_identity_after"] = source_input_identity(model_root=model)
            evidence["identity_unchanged"] = (
                evidence["runtime_identity_after"] == native and evidence["source_identity_after"] == sources
            )
            if not evidence["identity_unchanged"]:
                evidence["status"] = "error"
        finally:
            options.evidence.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
        assert evidence.get("identity_unchanged"), "P1 process identity changed; evidence is invalid"


if __name__ == "__main__":
    sys.exit(main())
