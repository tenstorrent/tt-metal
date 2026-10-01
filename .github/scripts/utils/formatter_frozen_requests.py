# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic observer: capture baseline requests and replay them unchanged."""

import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys


def freeze(requests, request_class, fixture):
    requests = list(requests)
    if os.environ["FORMATTER_PAIR_LEG"] == "baseline":
        assert not fixture.exists(), "Never overwrite a frozen request set"
        data = [dataclasses.asdict(request) for request in requests]
        encoded = json.dumps(data, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
        fixture.write_bytes(encoded)
        fixture.with_suffix(".sha256").write_text(hashlib.sha256(encoded).hexdigest() + "\n")
        return requests
    encoded = fixture.read_bytes()
    assert hashlib.sha256(encoded).hexdigest() == fixture.with_suffix(".sha256").read_text().strip()
    data = json.loads(encoded)
    assert len(data) == len(requests)
    return [request_class(**request) for request in data]


def main():
    mode, *args = sys.argv[1:]
    fixture = Path(os.environ["FORMATTER_PAIR_FIXTURE"])
    if mode == "assertions":
        import pytest

        manifest = json.loads(Path("/pair/control/.github/scripts/utils/formatter_pair_manifest.json").read_text())
        proof = manifest["assertion_payload_proof"]

        class Provenance:
            def __init__(self):
                self.collected = []
                self.outcomes = {}

            def pytest_collection_finish(self, session):
                assert len(session.items) == 3
                module = session.items[0].module
                assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == proof["tests_source_sha256"]
                assert (
                    hashlib.sha256(Path(module.run_concurrent_batch.__code__.co_filename).read_bytes()).hexdigest()
                    == proof["utils_source_sha256"]
                )
                self.collected = [item.name for item in session.items]
                assert self.collected == proof["test_names"]

            def pytest_runtest_logreport(self, report):
                name = report.nodeid.rsplit("::", 1)[1]
                assert name in self.collected
                self.outcomes.setdefault(name, {})[report.when] = report.outcome

            def pytest_sessionfinish(self, session, exitstatus):
                receipt = fixture.parent.parent / os.environ["FORMATTER_PAIR_LEG"] / "assertion-provenance.json"
                receipt.write_text(
                    json.dumps(
                        {
                            "collected": self.collected,
                            "sealed_payload_proof": proof,
                            "exit_status": int(exitstatus),
                            "outcomes": self.outcomes,
                        },
                        indent=2,
                    )
                    + "\n"
                )

        raise SystemExit(pytest.main(args, plugins=[Provenance()]))
    if mode == "plain":
        import vllm.benchmarks.serve as serve

        assert (
            hashlib.sha256(Path(serve.__file__).read_bytes()).hexdigest() == os.environ["FORMATTER_PLAIN_SOURCE_SHA256"]
        )
        # Freeze AFTER the stock server-tokenizer alignment, before any probe or
        # measured request. The candidate reuses exact strings, IDs and lengths.
        original = serve._align_prompts_to_server_tokenizer

        async def aligned(*positional, **keywords):
            requests = await original(*positional, **keywords)
            return freeze(requests, serve.SampleRequest, fixture)

        serve._align_prompts_to_server_tokenizer = aligned
        sys.argv = ["vllm", "bench", "serve", *args]
        from vllm.entrypoints.cli.main import main as cli

        cli()
        return

    assert mode == "structured"
    source = Path("/pair/vllm-source/benchmarks/benchmark_serving_structured_output.py")
    text = source.read_text()
    assert hashlib.sha256(text.encode()).hexdigest() == os.environ["FORMATTER_STRUCTURED_SOURCE_SHA256"]
    target = "input_requests = sample_requests(tokenizer, args)"
    assert text.count(target) == 1
    # Retain the original schema/prompt generator and every original assertion.
    # Fresh UUID4 schemas are captured once, then replaced by that exact fixture.
    text = text.replace(target, "input_requests = _formatter_frozen(sample_requests(tokenizer, args), SampleRequest)")
    scope = {
        "__name__": "__main__",
        "__file__": str(source),
        "_formatter_frozen": lambda requests, cls: freeze(requests, cls, fixture),
    }
    sys.path.insert(0, str(source.parent))
    sys.argv = [str(source), *args]
    exec(compile(text, str(source), "exec"), scope)


if __name__ == "__main__":
    main()
