"""Opt-in submission ordering; upstream request construction and scoring stay intact."""

import argparse
import copy
import hashlib
import json
import os
import shutil
import tempfile
import time
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def read_jsonl(path):
    with Path(path).open() as source:
        return [json.loads(line) for line in source if line.strip()]


def preserved_requests(config_path, manifest_path, inputs_path):
    from lm_eval.models.openai_completions import LocalChatCompletion

    config = json.loads(Path(config_path).read_text())
    manifest = json.loads(Path(manifest_path).read_text())
    if digest({k: v for k, v in manifest.items() if k != "manifest_sha256"}) != manifest["manifest_sha256"]:
        raise ValueError("Frozen manifest checksum mismatch")
    expected = {
        (task, doc): sha
        for task, spec in manifest["tasks"].items()
        for doc, sha in zip(spec["indices"], spec["document_sha256"], strict=True)
    }
    membership = {task: group for group, spec in manifest["groups"].items() for task in spec["tasks"]}
    backend = SimpleNamespace(model=config["model"], _max_gen_toks=2048)
    rows = []
    for row in read_jsonl(inputs_path):
        if digest(row["doc"]) != row["doc_sha256"] or row["doc_sha256"] != expected[(row["task"], row["doc_id"])]:
            raise ValueError("Preserved document differs from frozen manifest")
        if digest(row["arguments"]) != row["request_sha256"]:
            raise ValueError("Preserved request arguments differ from recorded hash")
        payload = LocalChatCompletion._create_payload(
            backend, row["messages"], generate=True, gen_kwargs=copy.deepcopy(row["arguments"][1]), seed=1234
        )
        rows.append(
            dict(
                task=row["task"],
                group=membership[row["task"]],
                doc_id=row["doc_id"],
                doc_sha256=row["doc_sha256"],
                request_sha256=row["request_sha256"],
                wire_payload_sha256=digest(payload),
                seed=payload["seed"],
            )
        )
    if len(rows) != len(expected) or {(row["task"], row["doc_id"]) for row in rows} != set(expected):
        raise ValueError("Preserved inputs must cover every frozen document exactly once")
    return config, manifest, rows


def reconstruct(config_path, manifest_path):
    """Rebuild all upstream requests and intercept before any transport call."""
    from unittest.mock import patch

    from benchmark_stage.evaluate import evaluate_groups
    from lm_eval.models.openai_completions import LocalChatCompletion

    config = json.loads(Path(config_path).read_text())
    manifest = json.loads(Path(manifest_path).read_text())
    expected = {
        (task, doc): sha
        for task, spec in manifest["tasks"].items()
        for doc, sha in zip(spec["indices"], spec["document_sha256"], strict=True)
    }
    rows = []

    class Captured(Exception):
        pass

    def capture(self, requests, **kwargs):
        for req in requests:
            doc_id = manifest["tasks"][req.task_name]["indices"][req.doc_id]
            key = digest([req.args[0], req.args[1]])
            metadata = self.request_metadata[key]
            document_hash = digest(req.doc)
            if document_hash != expected[(req.task_name, doc_id)]:
                raise ValueError("Reconstructed document differs from frozen manifest")
            payload = self._create_payload(
                self.create_message([req.args[0]]),
                generate=True,
                gen_kwargs=copy.deepcopy(req.args[1]),
                seed=self._seed,
            )
            rows.append(
                dict(**metadata, doc_sha256=document_hash, wire_payload_sha256=digest(payload), seed=payload["seed"])
            )
        raise Captured

    async def no_transport(*args, **kwargs):
        raise AssertionError("Request reconstruction must never call transport")

    generation = config["generation"][config["tasks"][0]]
    if any(config["generation"][task] != generation for task in config["tasks"]):
        raise ValueError("Ordering requires the existing identical-generation shared pool")
    with tempfile.TemporaryDirectory(prefix="k2-order-reconstruction-") as temporary:
        with (
            patch.object(LocalChatCompletion, "generate_until", capture),
            patch.object(LocalChatCompletion, "amodel_call", no_transport),
            patch.dict(os.environ, {"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}),
        ):
            try:
                evaluate_groups(
                    model=config["model"],
                    base_url=config["base_url"],
                    manifest_path=manifest_path,
                    groups=config["tasks"],
                    output=Path(temporary),
                    generation=generation,
                    shared=True,
                )
            except Captured:
                pass
    if len(rows) != len(expected) or {(row["task"], row["doc_id"]) for row in rows} != set(expected):
        raise ValueError("Reconstructed request set does not cover every frozen document exactly once")
    if len({row["request_sha256"] for row in rows}) != len(rows):
        raise ValueError("Duplicate reconstructed API requests")
    if {row["seed"] for row in rows} != {1234}:
        raise ValueError("Unexpected upstream request seed")
    return config, manifest, rows


def make_plan(config, manifest, requests, observation_dir):
    """Read runtime evidence only: no answers, references, metrics, or scores."""
    observation_dir = Path(observation_dir).resolve()
    observed_manifest = json.loads((observation_dir / "manifest.json").read_text())
    if observed_manifest["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("Runtime observations must use the exact same frozen manifest")
    by_request = {}
    evidence_files = []
    for group in config["tasks"]:
        links_path = observation_dir / group / "request_links.jsonl"
        responses_path = observation_dir / group / "responses.jsonl"
        links, responses = read_jsonl(links_path), read_jsonl(responses_path)
        if len(links) != len(responses):
            raise ValueError("Close the observed run before freezing its response/link evidence")
        response_map = {row["id"]: row for row in responses}
        if len(response_map) != len(responses):
            raise ValueError("Duplicate response IDs in observation evidence")
        for link in links:
            row = response_map[link["response_id"]]
            key = link["request_sha256"]
            if key in by_request:
                raise ValueError("Duplicate observed request")
            by_request[key] = dict(
                task=link["task"],
                doc_id=link["doc_id"],
                observed_output_tokens=int(row["usage"]["completion_tokens"]),
                observed_response_id=row["id"],
            )
        for path in (links_path, responses_path):
            evidence_files.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    for name in ("manifest.json", "run_config.json", "benchmark-inputs.jsonl"):
        path = observation_dir / name
        if not path.exists() and name == "benchmark-inputs.jsonl":
            continue
        evidence_files.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    if not set(by_request) <= {row["request_sha256"] for row in requests}:
        raise ValueError("Observation contains requests outside the reconstructed frozen set")
    entries = []
    for request in requests:
        observed = by_request.get(request["request_sha256"])
        if observed and (observed["task"], observed["doc_id"]) != (request["task"], request["doc_id"]):
            raise ValueError("Observation request/document mapping differs")
        entries.append(
            dict(
                request,
                unfinished=observed is None,
                observed_output_tokens=observed["observed_output_tokens"] if observed else None,
                observed_response_id=observed["observed_response_id"] if observed else None,
            )
        )
    entries.sort(
        key=lambda row: (not row["unfinished"], -(row["observed_output_tokens"] or 0), row["task"], row["doc_id"])
    )
    for rank, entry in enumerate(entries):
        entry["rank"] = rank
    plan = dict(
        schema_version=1,
        policy="unfinished_first_then_descending_observed_output_tokens",
        tie_breaker="task_then_document_id",
        model=config["model"],
        manifest_sha256=manifest["manifest_sha256"],
        concurrency=32,
        batch_size=1,
        seed=1234,
        count=len(entries),
        entries=entries,
        source_evidence=evidence_files,
        request_set_sha256=digest(sorted(row["request_sha256"] for row in entries)),
        scoring_information_used=False,
        responses_reused=False,
    )
    paths = [Path(__file__).resolve(), Path(__file__).parent / "benchmark_order_hook/sitecustomize.py"]
    from benchmark_stage import evaluate, responses, subsets
    from lm_eval.models import api_models, openai_completions

    paths.extend(
        Path(module.__file__).resolve() for module in (evaluate, responses, subsets, api_models, openai_completions)
    )
    paths.append(Path(api_models.__file__).resolve().parents[1] / "evaluator.py")
    plan["implementation_sources"] = [
        dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in paths
    ]
    plan["plan_sha256"] = digest(plan)
    return plan


def validate_plan(plan):
    if plan.get("plan_sha256") != digest({k: v for k, v in plan.items() if k != "plan_sha256"}):
        raise ValueError("Request-order plan checksum mismatch")
    entries = plan["entries"]
    if plan["schema_version"] != 1 or plan["concurrency"] != 32 or plan["batch_size"] != 1 or plan["seed"] != 1234:
        raise ValueError("Unsupported request-order contract")
    if len(entries) != plan["count"] or [row["rank"] for row in entries] != list(range(len(entries))):
        raise ValueError("Invalid request-order ranks")
    keys = [row["request_sha256"] for row in entries]
    if len(set(keys)) != len(keys) or len({(row["task"], row["doc_id"]) for row in entries}) != len(entries):
        raise ValueError("Request-order plan has duplicate requests or documents")
    if digest(sorted(keys)) != plan["request_set_sha256"]:
        raise ValueError("Request-order set checksum mismatch")
    return {row["request_sha256"]: row for row in entries}


def ordering_wrapper(original, plan, audit_path=None, observer=None):
    indexed = validate_plan(plan)

    async def ordered(self, requests, cache_keys, *, generate=True, ctxlens=None, **kwargs):
        if not generate:
            return await original(self, requests, cache_keys, generate=generate, ctxlens=ctxlens, **kwargs)
        if self._concurrent != 32 or self._batch_size != 1 or self._seed != plan["seed"] or self.model != plan["model"]:
            raise ValueError("Running backend differs from frozen scheduling contract")
        requests, cache_keys = list(requests), list(cache_keys)
        if len(requests) != len(cache_keys) or (ctxlens is not None and len(ctxlens) != len(requests)):
            raise ValueError("Request/cache/context tuple lengths differ")
        keys = [digest(key) for key in cache_keys]
        if len(keys) != len(set(keys)) or set(keys) != set(indexed):
            raise ValueError("Live request set differs from frozen order; no requests submitted")
        for key in keys:
            expected, observed = indexed[key], self.request_metadata[key]
            if any(expected[field] != observed[field] for field in ("request_sha256", "task", "group", "doc_id")):
                raise ValueError("Live upstream request/document mapping differs from order plan")
        for request, cache_key, key in zip(requests, cache_keys, keys, strict=True):
            payload = self._create_payload(
                self.create_message([request]), generate=True, gen_kwargs=copy.deepcopy(cache_key[1]), seed=self._seed
            )
            if digest(payload) != indexed[key]["wire_payload_sha256"]:
                raise ValueError("Live wire payload differs from frozen upstream payload; no requests submitted")
        permutation = sorted(range(len(keys)), key=lambda index: indexed[keys[index]]["rank"])
        audit = dict(
            plan_sha256=plan["plan_sha256"],
            manifest_sha256=plan["manifest_sha256"],
            complete_request_set_verified=True,
            requests=len(keys),
            concurrency=32,
            seed=self._seed,
            request_order=[keys[index] for index in permutation],
            original_request_order=keys,
            scheduled_to_original=permutation,
            original_to_scheduled=[permutation.index(index) for index in range(len(permutation))],
            wire_payloads_verified=True,
            completed=False,
            responses_reused=False,
        )

        def save():
            if audit_path is not None:
                Path(audit_path).write_text(json.dumps(audit, indent=2) + "\n")

        save()
        try:
            result = await original(
                self,
                [requests[index] for index in permutation],
                [cache_keys[index] for index in permutation],
                generate=generate,
                ctxlens=None if ctxlens is None else [ctxlens[index] for index in permutation],
                **kwargs,
            )
            if len(result) != len(permutation):
                raise ValueError("Upstream returned an incomplete request result set")
            restored = [None] * len(result)
            for scheduled_index, original_index in enumerate(permutation):
                restored[original_index] = result[scheduled_index]
            audit.update(completed=True, results_restored_to_upstream_order=True)
            if observer is not None:
                audit["observed_admission_order"] = observer.admissions
                audit["observed_api_ids"] = observer.responses
            save()
            return restored
        except BaseException as error:
            audit["error"] = dict(type=type(error).__name__, message=str(error))
            save()
            raise

    return ordered


def install(plan_path, audit_path, activation=None):
    from lm_eval.models.openai_completions import LocalChatCompletion

    plan = json.loads(Path(plan_path).read_text())
    validate_plan(plan)
    for item in plan.get("implementation_sources", []) + plan.get("source_evidence", []):
        if hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Frozen ordering provenance changed: " + item["path"])
    active_key = ContextVar("k2_order_request", default=None)
    journal_path = Path(audit_path).with_name("request-order-api-links.jsonl")
    observer = SimpleNamespace(admissions=[], responses=[])

    def observe(kind, key, **fields):
        row = dict(kind=kind, request_sha256=key, observed_ns=time.perf_counter_ns(), **fields)
        (observer.admissions if kind == "admitted" else observer.responses).append(row)
        with journal_path.open("a") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    class ObservedSemaphore:
        def __init__(self, inner, key):
            self.inner, self.key = inner, key

        async def acquire(self):
            acquired = await self.inner.acquire()
            try:
                observe("admitted", self.key, admission_index=len(observer.admissions))
            except BaseException:
                if acquired:
                    self.inner.release()
                raise
            return acquired

        def release(self):
            self.inner.release()

    original_call = LocalChatCompletion.amodel_call
    original_parse = LocalChatCompletion.parse_generations

    async def observed_call(self, *args, cache_keys=None, **kwargs):
        key = digest(cache_keys[0])
        token = active_key.set(key)
        kwargs["sem"] = ObservedSemaphore(kwargs["sem"], key)
        try:
            return await original_call(self, *args, cache_keys=cache_keys, **kwargs)
        finally:
            active_key.reset(token)

    def observed_parse(self, outputs, **kwargs):
        for row in outputs if isinstance(outputs, list) else [outputs]:
            observe("response", active_key.get(), response_id=row["id"], created=row.get("created"))
        return original_parse(self, outputs, **kwargs)

    LocalChatCompletion.amodel_call = observed_call
    LocalChatCompletion.parse_generations = observed_parse
    LocalChatCompletion.get_batched_requests = ordering_wrapper(
        LocalChatCompletion.get_batched_requests, plan, audit_path, observer
    )
    Path(audit_path).parent.mkdir(parents=True, exist_ok=True)
    Path(audit_path).with_name("request-order-activation.json").write_text(
        json.dumps(dict(plan_sha256=plan["plan_sha256"], pid=os.getpid(), activation=activation), indent=2) + "\n"
    )
    shutil.copyfile(plan_path, Path(audit_path).with_name("request-order-plan.json"))


def verify_run(run, plan_path):
    """Check complete fresh request coverage and retain scheduling provenance."""
    from benchmark_stage.responses import scoring_response
    from lm_eval.models.openai_completions import LocalChatCompletion

    run, plan_path = Path(run), Path(plan_path)
    plan = json.loads(plan_path.read_text())
    indexed = validate_plan(plan)
    audit = json.loads((run / "request-order-observed.json").read_text())
    activation = json.loads((run / "request-order-activation.json").read_text())
    arguments = activation["activation"]["orig_argv"]
    if not any(arguments[i : i + 3] == ["-m", "benchmark_stage", "evaluate"] for i in range(len(arguments) - 2)):
        raise ValueError("Scheduling hook was not activated in the intended evaluate child")
    if activation["plan_sha256"] != plan["plan_sha256"] or audit["plan_sha256"] != plan["plan_sha256"]:
        raise ValueError("Run used a different scheduling plan")
    if not all(
        audit.get(key)
        for key in (
            "completed",
            "results_restored_to_upstream_order",
            "complete_request_set_verified",
            "wire_payloads_verified",
        )
    ):
        raise ValueError("Scheduling run has incomplete request/result verification")
    expected = [row["request_sha256"] for row in plan["entries"]]
    if audit["request_order"] != expected or len(audit["original_request_order"]) != len(expected):
        raise ValueError("Run order differs from the frozen permutation")
    forward, inverse = audit["scheduled_to_original"], audit["original_to_scheduled"]
    if sorted(forward) != list(range(len(expected))) or len(inverse) != len(expected):
        raise ValueError("Invalid recorded permutation")
    if any(
        inverse[original] != scheduled or audit["original_request_order"][original] != expected[scheduled]
        for scheduled, original in enumerate(forward)
    ):
        raise ValueError("Recorded inverse permutation differs")
    journal = read_jsonl(run / "request-order-api-links.jsonl")
    admissions = [row for row in journal if row["kind"] == "admitted"]
    observed = [row for row in journal if row["kind"] == "response"]
    if [row["request_sha256"] for row in admissions] != expected:
        raise ValueError("Actual semaphore admission order differs from frozen order")
    if admissions != audit["observed_admission_order"] or observed != audit["observed_api_ids"]:
        raise ValueError("Live API journal differs from final audit")
    observed_by_key = {row["request_sha256"]: row for row in observed}
    if len(observed) != len(expected) or set(observed_by_key) != set(expected):
        raise ValueError("Actual API response coverage is incomplete or duplicated")
    response_rows = {}
    expected_parsed = {}
    for group in sorted({row["group"] for row in plan["entries"]}):
        links = read_jsonl(run / group / "request_links.jsonl")
        raw = read_jsonl(run / group / "responses.jsonl")
        by_id = {row["id"]: row for row in raw}
        if len(raw) != len(by_id) or len(links) != len(raw):
            raise ValueError("Raw response/link coverage differs")
        for link in links:
            key = link["request_sha256"]
            if key in response_rows or key not in indexed:
                raise ValueError("Duplicate or unplanned raw request")
            entry = indexed[key]
            if any(entry[field] != link[field] for field in ("group", "task", "doc_id")) or entry["group"] != group:
                raise ValueError("Raw document/request mapping differs")
            raw_row = by_id[link["response_id"]]
            observation = observed_by_key[key]
            if observation["response_id"] != raw_row["id"] or observation["created"] != raw_row.get("created"):
                raise ValueError("Observed API identity differs from raw response")
            if raw_row["id"] == entry["observed_response_id"]:
                raise ValueError("Prior response identity was reused")
            response_rows[key] = dict(
                rank=entry["rank"],
                task=entry["task"],
                doc_id=entry["doc_id"],
                request_sha256=key,
                response_id=raw_row["id"],
                created=raw_row.get("created"),
            )
            expected_parsed[key] = LocalChatCompletion.parse_generations(
                SimpleNamespace(think_end_token=None), scoring_response(raw_row)[0]
            )
    if set(response_rows) != set(expected) or len({row["response_id"] for row in response_rows.values()}) != len(
        expected
    ):
        raise ValueError("Raw response coverage or response identity uniqueness failed")
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest["manifest_sha256"] != plan["manifest_sha256"]:
        raise ValueError("Run frozen manifest differs from ordering plan")
    _, _, final_inputs = preserved_requests(
        run / "run_config.json", run / "manifest.json", run / "benchmark-inputs.jsonl"
    )
    if {row["request_sha256"]: row["wire_payload_sha256"] for row in final_inputs} != {
        row["request_sha256"]: row["wire_payload_sha256"] for row in plan["entries"]
    }:
        raise ValueError("Final retained inputs differ from frozen request/wire-payload set")
    by_document = {(row["task"], row["doc_id"]): row for row in plan["entries"]}
    sample_documents, sample_filters = set(), set()
    for task in sorted(manifest["tasks"]):
        group = next(row["group"] for row in plan["entries"] if row["task"] == task)
        for sample in read_jsonl(run / group / f"samples_{task}.jsonl"):
            # lm-eval evaluator.py logs doc_id_true = indices[doc_id]. Filters
            # can produce multiple sample rows for one original document.
            document_key = (task, sample["doc_id"])
            entry = by_document[document_key]
            filter_key = (*document_key, sample["filter"])
            if filter_key in sample_filters or digest(sample["doc"]) != entry["doc_sha256"]:
                raise ValueError("Upstream sample document/hash/filter mapping differs")
            if len(sample["arguments"]) != 1 or digest(sample["arguments"][0]) != entry["request_sha256"]:
                raise ValueError("Upstream sample request arguments differ")
            if sample["resps"] != [expected_parsed[entry["request_sha256"]]]:
                raise ValueError("Upstream scoring received a response belonging to a different request")
            sample_documents.add(document_key)
            sample_filters.add(filter_key)
    if sample_documents != set(by_document):
        raise ValueError("Upstream scoring sample coverage is incomplete")
    provenance = run / "request-order-sources"
    provenance.mkdir(exist_ok=True)
    retained = []
    for index, item in enumerate(plan["implementation_sources"]):
        source = Path(item["path"])
        if hashlib.sha256(source.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Ordering implementation changed during run")
        target = provenance / f"{index:02d}-{source.name}"
        shutil.copyfile(source, target)
        retained.append(dict(item, retained_path=str(target.relative_to(run))))
    shutil.copyfile(plan_path, run / "request-order-plan.json")
    proof = dict(
        valid=True,
        count=len(expected),
        manifest_sha256=plan["manifest_sha256"],
        plan_sha256=plan["plan_sha256"],
        wire_payloads_verified=True,
        complete_response_coverage=True,
        results_restored_to_upstream_order=True,
        actual_admission_order_verified=True,
        upstream_sample_assignment_verified=True,
        upstream_sample_documents=len(sample_documents),
        upstream_sample_filter_rows=len(sample_filters),
        responses_reused=False,
        scoring_information_used=False,
        activation=activation,
        implementation_sources=retained,
        api_ids_by_planned_rank=[response_rows[key] for key in expected],
    )
    (run / "request-order-verification.json").write_text(json.dumps(proof, indent=2) + "\n")
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--observations", type=Path)
    parser.add_argument("--verify-run", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.verify_run:
        if args.plan is None:
            parser.error("--verify-run requires --plan")
        proof = verify_run(args.verify_run, args.plan)
        print(json.dumps({key: proof[key] for key in ("valid", "count", "plan_sha256")}))
        return
    if any(value is None for value in (args.config, args.manifest, args.observations)):
        parser.error("Preparation requires --config, --manifest and --observations")
    inputs = args.observations / "benchmark-inputs.jsonl"
    config, manifest, requests = (
        preserved_requests(args.config, args.manifest, inputs)
        if inputs.exists()
        else reconstruct(args.config, args.manifest)
    )
    plan = make_plan(config, manifest, requests, args.observations)
    validate_plan(plan)
    if not args.check_only:
        if args.output is None or args.output.exists():
            raise ValueError("Specify a new output path for a frozen ordering plan")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(plan, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                requests=plan["count"],
                manifest_sha256=plan["manifest_sha256"],
                plan_sha256=plan["plan_sha256"],
                unfinished=sum(row["unfinished"] for row in plan["entries"]),
                inference_calls=0,
                scoring_information_used=False,
                check_only=args.check_only,
            )
        )
    )


if __name__ == "__main__":
    main()
