# AUTOTRIAGE

## Diagnosis
No device hang established. The representative server continued preparation from four-row length288 to320 while evidence was collected. Long intervals between progress prints include kernel compilation; there is no timeout or device assertion.

## Triage Evidence
`triage-summary.txt` passes DRAM, Ethernet, ARC and basic component checks. Operation/callstack reports failed because inspector `getPrograms()` returned None; their absence is not evidence of idle or healthy kernels. `tt-triage.txt` and console.log preserve the capture.

## Source Evidence
`tt/bounded_prefill.py:prepare` prepares all512 physical lengths for both B1 and B4, printing only every32 lengths. Each shape executes representative layer kernels, terminal projection and variable-length concat. Sparse progress logs cannot distinguish compilation from a device wait.

## Downstream Effects
None demonstrated. No reset or process termination performed.

## Proposed Fix
No source fix supported by this capture. Continue the bounded startup and use the next progress/health result. If it stalls again, gather evidence with inspector enabled on an isolated reproducer.

## Uncertainty
This capture does not establish a per-operation stop site. Server startup and allocation-tracked lifecycle remain unvalidated until completion.
