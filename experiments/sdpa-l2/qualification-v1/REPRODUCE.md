# Reproducing numerical qualification

Use the retained improved Blackhole source snapshot, not unmodified main.
Numerical-source hashes and the qualification host patch are recorded here.
Do not apply the patch over unrelated factory edits without reviewing them.

On the allocated Blackhole checkout, activate its existing Python/toolchain
environment, with TT_METAL_HOME, ARCH_NAME=blackhole, and PYTHONPATH configured.
The completed run used Python 3.10.19 and torch 2.11.0+cpu for input generation
and FP64 reference evaluation.

```bash
git apply --check experiments/sdpa-l2/qualification-v1/qualification-host.patch
git apply experiments/sdpa-l2/qualification-v1/qualification-host.patch
touch ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp
CMAKE_BUILD_PARALLEL_LEVEL=12 cmake --build build_Release --target install

python_env/bin/python experiments/sdpa-l2/qualification-v1/qualify.py --self-test --output /tmp/unused-qualification.jsonl
python_env/bin/python experiments/sdpa-l2/qualification-v1/test_gates.py
python_env/bin/python experiments/sdpa-l2/qualification-v1/probe_guards.py
python_env/bin/python experiments/sdpa-l2/qualification-v1/qualify.py --output fresh-results.jsonl
python_env/bin/python experiments/sdpa-l2/qualification-v1/probe_mask_guards.py
python_env/bin/python experiments/sdpa-l2/qualification-v1/adjudicate.py fresh-results.jsonl fresh-accepted-results.jsonl
```

The runner resumes an existing raw output by case ID/mode, without repeating
completed cases. Use a fresh filename for independent replication. Historical
ERROR records are retained but retried. The scorer uses the final record for
each case/mode; it refuses to overwrite its destination.

The collector exits successfully when data collection completes, even when
numerical gates fail. Its exit code is NOT a qualification pass. Acceptance
is recorded per case/head in the scored JSONL and summarized in REPORT.md.
Likewise, the completeness audit checks collection/provenance, not numerical
acceptance. Unsupported cases and missing hardware/data do not count as passes.

Two focused reproductions, while the qualification guards are installed:

```bash
# Narrow normal-input FP32 per-head L2 miss.
python_env/bin/python experiments/sdpa-l2/qualification-v1/qualify.py --group normal --lengths 131072 --heads 5 --seeds 1235 --modes accurate --output fresh-normal-miss.jsonl
# BF16 constant-V normalization exceeds the one-ULP structural limit.
python_env/bin/python experiments/sdpa-l2/qualification-v1/qualify.py --group structural --lengths 32768 --heads 5 --seeds 1234 --distributions constant_v --modes fast --output fresh-constant-v.jsonl
```

Never run two device-owning scripts concurrently on this card. The rounding
oracle is CPU-only and can run separately:

```bash
python_env/bin/python experiments/sdpa-l2/qualification-v1/audit_rounding_floor.py --output fresh-oracle.jsonl
python_env/bin/python experiments/sdpa-l2/qualification-v1/adjudicate.py fresh-oracle.jsonl fresh-accepted-oracle.jsonl
```

The acceptance-scoring step only implements the proposal's explicit exception
for undefined PCC on constant outputs; it does not waive defined PCC failures
or modify any thresholds. Read SPEC.md for all gates and known coverage gaps.

Restore the retained factory after qualification, forcing a rebuild despite
potentially preserved source timestamps:

```bash
git apply -R --check experiments/sdpa-l2/qualification-v1/qualification-host.patch
git apply -R experiments/sdpa-l2/qualification-v1/qualification-host.patch
touch ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp
CMAKE_BUILD_PARALLEL_LEVEL=12 cmake --build build_Release --target install
```

The default short-context BF16 guard is restored by this step. Qualification
results for those lengths describe compensation explicitly enabled, not the
default build's dispatch behavior. FP32 unsupported cases must never be
substituted with fallback measurements.
