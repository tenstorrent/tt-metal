# Stage review

Reviewer: `/root/stage_review_qwen25_coder32` (fresh read-only subagent)

Verdict: **clean-pass**

## Required work

None.

## Review coverage

The reviewer independently reproduced graph classification, Python compilation, JSON validation, the context checker, Black's Python 3.12 check, test collection, runtime fallback source inspection, HF config/checkpoint identity, IR/emit semantics, constant-eval weight mapping, TP collective collapse, and context-memory arithmetic. It inspected both selected raw MLIR graphs, the two lowered emits, the implementation, tests, all evidence logs, and the documentation. By contract, it did not open TT devices or rerun hardware tests.

## Non-blocking concerns

- The three `.log` evidence files match the repository-wide ignore rule and must be force-added to the checkpoint. The main agent does so explicitly.
- The capacity summaries predate the final decoder file by roughly 22 seconds because the last implementation correction was decode-only. The current combined post-change functional run covers both prefill and decode, so the reviewer did not classify the prefill capacity evidence as stale.
- Evidence logs are concise result captures rather than untouched full pytest stdout. Values are internally consistent with the executable tests and current implementation; teardown and device-health observations are separately documented.

## Anomaly disposition

- Single-function p300 visibility failure: controlled by exposing paired functions `2,3` while opening a 1x1 mesh; later runs and final inventory passed.
- Nanobind teardown reference warnings: controlled; the reviewer reproduced them during collect-only without device execution, localizing them to the Python binding lifecycle.
- Decode 40-head padding/PCC failure: fixed by restoring the emitted post-RoPE logical Q/K slices; final synthetic and real decode PCCs pass.
- Initial 3,969 padding assumption: fixed by direct adjacent-length probes.
- Sequence 4,000 OOM: controlled and honestly represented by the 3,999 DRAM-limited context contract.

## Residual risk

Hardware results were reviewed from recorded evidence rather than independently rerun. Evidence covers the emitted batch, required prefill lengths, layer-32 real weights, and one decode position; broader decode positions, performance, BF8 selection, full-model composition, multichip, tracing, generation, and serving remain outside this stage.
