# Diagnosis: stock custom weights dispatch parser mismatch

Observed actual failure: local-tti-benchmark-preflight/client.log shows run.py constructs launchers/run_llm_bench.py with --model set to an absolute pinned snapshot and supplies runtime-model-spec-json; child exits2 before requests because run_workflows.py validates --model against catalog choices.

Cause: run.py's custom weights resolver accepts arbitrary identity, but run_workflows.py:144 validates catalog-only identity before loading runtime spec. The later command_factory.py:147-150 intentionally gives runtime spec identity precedence, so rejection happens too early. This explains successful top-level resolution, correct emitted runtime JSON, and child argparse failure without any model request.

Supported unchanged-source route: invoke the documented benchmark launcher directly with canonical catalog --model plus that existing runtime-model-spec-json. Parser accepts the catalog name; _canonicalize_cli_model then uses the pinned snapshot from runtime JSON, and _build_context prefers that same runtime spec. No selector, wrapper, altered model source, mutable Hub main, or copied implementation is needed.
