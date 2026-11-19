# Benchmark adapter accounting diagnosis

Scope: CPU-only inspection and execution of the adapter class extracted from the current source AST. The test excludes all module imports, injects CPU torch and a fake generator/recorder, and neither imports TTNN nor opens devices. Parent owns implementation changes and AutoFix validation.

The failing contract is that every submitted decode phase retains the correct request identity and sequence position across slot migration, and reaches one completion boundary after its output becomes host-consumable. The benchmark collector needs this to match measured requests/steps and complete elapsed phases.

Three causes were supported by source and a CPU reproduction before repair:

1. `decode_forward` iterated `slot_remap` directly in an integer-keyed dictionary lookup. The pinned runner's `_decode_state_slot_remap` returns a torch tensor, whose scalar elements have tensor identity hashes. Membership checks against integer keys failed and the comprehension replaced all request ownership with an empty dictionary. A two-slot `[1,0]` CPU submission produced metadata containing only slot/position and no request. Normalize remap values to Python integers before indexing.
2. With `reload_inputs=False`, the adapter sampled `generator.caches[slot].position` before the underlying generator applied slot migration. The canonical generator instead uses `caches[remap[slot]].position` at this same point. For positions 100/200 and permutation `[1,0]`, the correct recorded positions are 200/100. Read the source slots before dispatch.
3. The canonical generator's default `read_from_device=True` path performs its readback and host processing internally, returning a CPU torch tensor. The adapter only registered a pending ticket and never ended it. The pinned plugin recognizes that host output and skips `process_decode_output_host`, so no later callback closes the ticket. CPU reproduction showed one `begin`, no end/completion, and one outstanding ticket after direct return. Complete the direct host-result path immediately; preserve event-based completion for deferred readback. Empty/inactive decode also returns a torch tensor even with read_from_device=False; an optional async read of that host output must not produce a second end or KeyError.

Submission-row `page_table[row]` indexing is correct. Canonical `_table_plan` explicitly uses submitted row order rather than physical slot indices, including padded rows. No physical-slot reindexing should be introduced.

Regression file: `tests/test_benchmark_adapter_cpu.py` covers tensor remap identity, resident source positions, direct host completion, deferred async completion at the existing read event followed by host processing, and an empty host result with optional async read. Run without repository conftest:

```bash
/home/ttuser/qwen38-flash/benchmark-env/bin/python -m pytest --noconftest -o addopts= -q models/autoports/qwen_qwen3_8_flash_next/tests/test_benchmark_adapter_cpu.py
```

Initial regression-suite evidence: `doc/benchmark/setup/adapter-regression-initial.log` reports 5 passed in 0.53 seconds; the parent's repairs had landed before this suite executed. The earlier pre-repair extracted-class reproduction was observed separately and is described above. Actual serving/hardware validation remains the parent's task; these host tests establish control-flow bookkeeping only and do not prove device timing or model performance.
