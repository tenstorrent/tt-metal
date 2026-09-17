# Full-prefill performance and book observation

This benchmark runs all 32 Llama-3.1-8B-Instruct layers, final normalization and the full vocabulary head. It supports configured capacities of 4K, 8K, 16K, 32K, 64K and 128K. A supported value is not a device-fit or speed claim.

## Workload and measurements

Use one SP4/TP8 Blackhole Galaxy, BF16 weights/activations and BFP8_B KV storage. Two distinct book prompts use two slots in sequence. Each slot has one warmup and three measured requests. Every request uses contiguous 1,024-token chunks and runs the full model.

Prompt wall time starts before the first token upload and ends after the last model synchronization. Chunk forward time starts after token-upload synchronization and ends after model synchronization. Throughput is prompt tokens divided by prompt wall time for that sequential user. Model/cache construction, tokenizer loading, output readback, hashes, book ranking and cleanup are outside prompt timing. This is eager host wall time, not kernel time or concurrent serving throughput.

The test retains each chunk's device logits until all chunks complete. It then checks all 32 decoded shards for finite values and exact equality to the same-slot/chunk warmup. Measured calls must not compile new programs. It assembles the last prompt row in TP vocabulary order and reports the top five tokens and the actual book word's rank. The prompts are raw book text with one BOS token; no chat template, token append or decode loop is used. Book ranks have no semantic pass threshold. No golden model or KV comparison is performed.

Report completion means this workload completed its execution checks. The report keeps full_model_accepted=false and execution_verified=false: the external runner must separately verify process success, exact source identity, JUnit and clean device teardown. The test does not establish full-model accuracy.

## Inputs and source identity

Set LLAMA_LONG_CONTEXT_PERF_CONFIG to an explicit JSON file before collecting the device test. See config.example.json. Config paths are absolute, but no machine name, job ID, private evidence import or allocation policy is embedded in this package.

The source map is an absolute-path-to-SHA256 object. It must include every path returned by performance_config.required_sources(repository). Add the native libraries, kernel sources, checkpoint files and runtime inputs required by your external provenance policy. The loader verifies every supplied byte before tensor-library import; the test checks the map again before and after its work. This minimum Python dependency inventory alone does not prove complete native-build provenance.

The book loader consumes the frozen fixture manifest format, its 12 exact-length fixtures, tokenizer receipt and original book-source binding. It verifies their hashes and the selected checkpoint metadata. These fixture payloads are external inputs, not bundled books or generated model results. Prepare them before device execution, and keep all manifest-referenced paths available on the target machine.

The portable schema1 config has no Slurm or approval requirement. An existing site schema3 config can also be supplied; explicit closed authorization fields remain closed, and a supplied resource receipt is hash-checked. The external runner retains owner, lease, physical-node lock, active-step, process/FD, instrumentation and resource policy. This package does not replace those guards.

## Run

Use the existing tt-metal Python/native environment and a separately authorized idle Galaxy. From the repository root, create a fresh evidence directory and set:

    export LLAMA_LONG_CONTEXT_PERF_CONFIG=/absolute/path/to/config.json
    export LLAMA_PERFORMANCE_EVIDENCE_DIR=/absolute/path/to/fresh-evidence
    export TT_MESH_GRAPH_DESC_PATH="$PWD/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto"
    mkdir -p "$LLAMA_PERFORMANCE_EVIDENCE_DIR"

Run the exact test through the site's guarded runner, or after equivalent external health/ownership checks:

    python -m pytest models/demos/llama_3p1_8b_d_p/tests/performance/test_long_context_performance.py::test_full_prefill_long_context_performance -q -s --junitxml="$LLAMA_PERFORMANCE_EVIDENCE_DIR/pytest.xml"

The test creates a fresh captures-C child. Reusing that child refuses. Use a caller-supplied four-thread Torch setup for comparable measurements; thread policy stays outside reusable test imports.

## CPU checks

These host tests import no tensor or device libraries. The collection checks invoke pytest in isolated subprocesses:

    python -m unittest -v models.demos.llama_3p1_8b_d_p.tests.performance.test_configuration_cpu models.demos.llama_3p1_8b_d_p.tests.performance.test_timing_cpu models.demos.llama_3p1_8b_d_p.tests.performance.test_book_cpu models.demos.llama_3p1_8b_d_p.tests.performance.test_report_cpu models.demos.llama_3p1_8b_d_p.tests.performance.test_source_inventory_cpu models.demos.llama_3p1_8b_d_p.tests.performance.test_optional_collection_cpu

They exercise actual timing boundaries with deferred fake work, exact-once retained-output cleanup, continued cache/model/synchronization cleanup after individual faults, primary-error preservation and final-report persistence, complete ordered chunk/request coverage, shorter-report denominator rejection, full-chip output inventory, final SP/TP mapping, finite/repeat checks, fixture/config/source binding and deterministic bounded source hashing. The saved 2K timing fixture contains only existing clocks and summaries; it checks arithmetic without rerunning 2K.

## Interpretation

Configured cache capacity equals prompt context in the reported measurements. Attention currently gathers/reorders the configured cache before slicing the logical prefix. A larger configured capacity can change latency even for the same prompt. Do not infer 128K fit or speed from smaller measurements. Preserve existing results after documentation or import-only changes; remeasure only when execution changes materially.

## Optional collection

Without LLAMA_LONG_CONTEXT_PERF_CONFIG, pytest skips only this benchmark module before importing tensor libraries. An explicitly supplied empty, malformed or closed configuration remains an error before tensor imports. Ordinary unrelated tests still collect and run. The import gate does not change the benchmark function, timing intervals or workload.
