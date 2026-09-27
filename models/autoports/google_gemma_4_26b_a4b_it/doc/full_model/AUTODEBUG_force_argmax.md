# AutoDebug: force-argmax changed-input index failure

2026-09-27. Source-only; no implementation edits, hardware runs, or new
correctness/performance measurements by this investigation.

## Verdict

**Reject force-argmax for the delivered generator configuration.** The existing
exact-token oracle is already sufficient evidence that this alternative is
incorrect for the tested B3 changed-input workload. Its internal failing
boundary remains unlocalized. Repairing this slower, unselected alternative is
not required to establish the selected canonical split path, provided the
delivered path continues to disable force-argmax and does not advertise the
alternative as validated.

`sampler_padded.json` reports canonical split greedy and sampled cases passing
B1 and B3. Its B1 warmed host-wall replay means are 508.87 us for canonical
greedy and 3302.49 us for force-argmax; these are isolated sampler measurements,
not full-model performance. This measured speed independently rejects
force-argmax as the selection even without attributing the B3 anomaly. The
coordinator subsequently reports B32 selected-path tests also pass; those runs
were performed independently, not by this investigation.

## Exact failure evidence

The probe uses BF16 TILE logits, local shape `[1,1,32,65536]`, TP4, 262144 global
vocabulary, persistent replicated `[1,1,1,32]` output, and two host input updates
against the same sampling trace. `known_logits` places a unique `+100` peak in
each active row and `-100` everywhere else in that row. These numbers are exact
in BF16, so there is no near-tie or model-precision ambiguity.

| B3 force-greedy replay | Expected first three tokens | Actual on every chip |
| --- | --- | --- |
| Round 0 | `[4,65577,131150]` | `[4,65577,131150]` |
| Round 1 | `[131077,196650,79]` | `[131077,163882,79]` |

For failing slot 1:

```text
expected = 196650 = 3*65536 + 42 = 0x3002a
actual   = 163882 = 2*65536 + 32768 + 42 = 0x2802a
difference = 32768 elements = 65536 BF16 bytes
previous round's token = 65577
```

The error is not simply a stale previous sampled token. The surviving local
offset 42 and exact half-shard displacement are useful localization clues, but
one observed offset does not prove a 16-bit truncation, bit-width limit, fixed
address displacement, or race. All four replicas agreeing does not establish
correct gather contents: they can agree on the same incorrectly assembled
full row.

## Relevant path and hypotheses

`models/common/sampling/tt_sampling.py:760` chooses force-argmax only for semantic
greedy params when allowed. For this configuration its path is:

1. `all_gather_async` of the full vocabulary, yielding replicated
   `[1,1,32,262144]` TILE logits. It supplies a barrier semaphore and explicitly
   uses no persistent gather output.
2. `_untilize_chunk_count(262144)` returns four. Split into 65536-wide tiled
   chunks; untilize each; concatenate the four ROW_MAJOR chunks along vocab.
3. `ttnn.argmax(..., dim=-1, keepdim=False, output_tensor=out)` uses the
   ROW_MAJOR multicore argmax path.

The canonical path instead gathers top-k candidate values/indices and applies
the sampling kernel. Passing canonical cases therefore isolate the defect to
the force-specific chain, not to the host winner calculation or generic
persistent output interface.

Hypotheses, none yet verified:

- **Gather or layout assembly relocates/duplicates the +100 peak.** Inspect
  full gathered TILE logits, each untilized chunk, and the concatenated
  ROW_MAJOR input. A duplicate peak at the smaller returned index would make
  argmax select the wrong token correctly for its corrupted input.
- **Argmax reads the wrong chunk or returns a wrong partial index.** Its
  multicore factory supplies separate byte offsets and reduction-index
  offsets. Compare an argmax-only trace against exact replicated input to
  remove all gather/layout operations.
- **Trace replay state differs from eager state.** Use A/B/A inputs in the
  smallest failing component. The initial passing round alone cannot prove
  repeated replay correctness.

The inspected argmax kernel uses `uint32_t` for `src_offset`, `red_dim_offset`,
loop index, `max_idx`, partial indices, and output. BF16 **values** use uint16;
that is not evidence that indices are limited to 65535. Relevant source:
`argmax_multi_core_program_factory.cpp:444`,
`reader_argmax_interleaved_multicore.cpp:42`, and
`argmax_common.hpp:108` under `ttnn/cpp/ttnn/operations/reduction/argmax/device/`.

There is a known prior replay-state mechanism documented directly in the
current kernel around line 491: leftover `done_sem` counts can allow stale
partials on replay. The current source already resets both semaphores at
kernel exit. Do not claim this existing repair is absent or that the present
failure is the same defect without an isolated reproducer. Likewise, the
force-gather source already avoids persistent buffers and supplies its barrier;
the documented missing-barrier mechanism is not established here.

## Smallest discriminating probes

The existing standalone command can reproduce/recheck the candidate without
loading model weights (**proposed, not run here**):

```bash
OMP_NUM_THREADS=8 timeout 120 python -m models.autoports.google_gemma_4_26b_a4b_it.tests.probe_full_sampler --batch-sizes 3 --pad-batch --modes greedy --force-mode force --iterations 0 --output models/autoports/google_gemma_4_26b_a4b_it/doc/full_model/force_argmax_repro.json
```

For localization, adapt that small probe in this order:

1. Upload `known_logits(3,0,physical_batch=32)` **replicated**, BF16 ROW_MAJOR,
   shape `[1,1,32,262144]`. Allocate the same UINT32 ROW_MAJOR output
   `[1,1,1,32]`. Warm and capture **only** `ttnn.argmax(input, dim=-1,
   keepdim=False, output_tensor=out)`. Copy A/B/A into the same input and replay,
   checking every active row on every chip. Also check eager argmax before
   capture. Failure localizes to argmax/runtime; pass removes the simplest
   argmax-only explanation.
2. If argmax-only passes, upload the same replicated input in TILE layout and
   capture **only** split-four/untilize/ROW_MAJOR-concat. Read its result after
   replay and compare exact BF16 values at both 196650 and 163882, all peak
   indices, and the full three active rows. This removes CCL and argmax.
3. If layout assembly passes, capture **only** the force path's exact async
   gather over sharded input, with the same topology, barrier/semaphore calls,
   and changed A/B/A inputs. Compare each device's gathered tensor to the
   original global CPU tensor. Then combine gather plus layout assembly if
   individual components pass; cross-op lifetime/order can matter.

Preallocate debug buffers before capture, and read them only after replay.
Keeping previously temporary intermediates alive or inserting copies can
change aliasing/timing and mask a defect; label such instrumented controls and
retain the uninstrumented reproducer. If an isolated component fails, place
the unique peak around 32767/32768, 65535/65536, 163839/163840, and
196607/196608 on rows 0/1/2 to distinguish position from row dependence.

A quick override to one large untilize could be a later controlled A/B, but it
is not a justified fix now: the common chunked path exists to avoid wide-row
L1/circular-buffer limits. Similarly, do not apply a +32768 correction to
returned indices or add unconditional synchronizations based on one failure.

## Delivery implication

Keep canonical split sampling selected (`force_argmax=False`) and retain this
failed alternative in the comparison ledger. The passed canonical exact-token
oracle and its own multi-batch/feedback tests establish that path separately.
The existing force failure already proves rejection; the adapted probes are
needed only to attribute or repair the underlying force-specific defect.
If force-argmax remains an exposed experimental switch, clearly mark it
unvalidated/rejected for this model configuration rather than a supported
performance mode.
