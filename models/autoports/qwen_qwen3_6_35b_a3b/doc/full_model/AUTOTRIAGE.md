# AUTOTRIAGE

## Diagnosis
The initial generator-owned sampling capture calls `_run_sampling` without its
required keyword-only `grammar_on`. Python necessarily raises TypeError after
begin_trace_capture and before end_trace_capture. The probe's finally block then
tries to close a mesh with an unfinished capture. This is a wrapper lifecycle
bug, not evidence of a sampler kernel or device fault.

## Triage evidence
`logs/reduced_generate.stack.txt`: live Python stack is close_mesh_device.
`triage/tt-triage.txt`: all four devices are idle in dump_op_mesh, no active op
reported. Dispatch readers wait for commands; no active kernel assertion. ETH
NoC counter mismatches are reported in the full text (the summary's script exit
statuses are not semantic health verdicts). They do not identify a stuck model
producer. The original exception is hidden by finally/close blocking.

## Source evidence and lifecycle ledger
`models/common/sampling/generator.py:_run_sampling` requires grammar_on.
Initial model-local `prepare` supplied only penalties_on and tt_out_tok.
Model capture begin/end: balanced. Sampling begin: 1; body raises before any
sampling device op; sampling end: 0. Cleanup attempts release/close while capture
remains open. This explains an idle device with a blocked host cleanup.

## Proposed fix
Supply grammar_on=False, and pair every capture begin/end in try/finally so a
Python failure cannot strand capture state. Print exceptions before mesh close.

## Uncertainty
A fresh bounded replay must confirm repair; triage alone does not prove sampler
correctness, allocation safety or coherent feedback. Preserve the original run
until its timeout expires. Then check holders, recover and smoke the mesh.
