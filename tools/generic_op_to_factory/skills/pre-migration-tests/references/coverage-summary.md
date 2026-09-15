# What we test and which bugs we hope to catch

Intended coverage, selected for the operation's actual contract—not a claim that
these tests have been authored or passed. See [cases.md](cases.md) for implementation
details and limitations.

| What we test | Bugs we hope to catch |
| --- | --- |
| Arguments and defaults | C++ interprets a call differently from Python. |
| Invalid inputs | Bad inputs slip through or valid inputs get rejected. |
| Output properties | Wrong shape, dtype, layout or memory placement. |
| Numerical results | Wrong calculations, indexing or copied values. |
| Boundary cases | Missing tail elements or incorrect work distribution. |
| Cache transitions | Reusing an incompatible program, unnecessary misses, stale addresses or arguments. |
| Aliases and ownership | Unexpected buffer sharing or broken in-place behavior. |
| Protected memory | Corrupting inputs, earlier outputs or untouched regions. |
| Watcher checks | Invalid device transfers and supported buffer overruns. |
| Poisoned padding/guards | Reading irrelevant memory that contaminates the result. |
| Dirty outputs and prior state | Missing writes, missing zero-outs or stale accumulators. |
| Downstream use | Output works alone but breaks the next operation. |
| Native routing | Tests accidentally call Python instead of C++. |
| Existing golden parity | Migration changes previously recorded behavior, including failures. |

The skill authors and verifies the shared acceptance tests on the source before
porting. During migration, those tests run on C++ only. Native routing is a flow
check; existing Python/C++ golden parity is separate and does not require every
source golden to pass.

## Limits and pending integration

- Poisoning cannot detect reads whose values never affect observable results.
- Protected-memory checks cover only the regions actually observed; true guards
  require verified physical placement. Watcher is not a tensor-ownership checker.
- Watcher execution/evidence integration and safe hooks for direct internal
  scratch/CB poisoning remain pending. Ordinary pytest success does not verify them.
- The focused `--dev` safety pass is separate from ordinary correctness evidence;
  simulator runs do not provide the same NoC-sanitizer coverage.
- TTNN comparison mode is not used as acceptance evidence. Operation tests carry
  explicit independent references, case parameters, assertions and tolerances.
- NoC Debug Dump, memory reporting and Emule ASAN are risk-triggered diagnostics;
  host sanitizer, descriptor-parity and profiling checks normally belong after
  native C++ exists. See [advanced-checks.md](advanced-checks.md).
- Record actual per-operation coverage and results in the
  [handoff](handoff.md); this list is not execution evidence.
