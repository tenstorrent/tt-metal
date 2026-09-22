<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->

# Shared-runner capacity follow-up

The 2K/two-slot runner, producer and address-table gates have passed. The next
capacities are listed below. These are planned shared-runner checks, not claims
that the larger cases have run on this branch.

| Capacity per slot | 1K chunks per slot | Two-slot H2D run | Table/boundary readback | Shutdown | Status |
|---|---:|---|---|---|---|
| 4K / 4,096 tokens | 4 | Required | Required | Required | To do |
| 8K / 8,192 tokens | 8 | Required | Required | Required | To do |
| 16K / 16,384 tokens | 16 | Required | Required | Required | To do |
| 32K / 32,768 tokens | 32 | Required | Required | Required | To do |
| 64K / 65,536 tokens | 64 | Required | Required | Required | To do |

## One focused scenario at each capacity

1. Allocate two slots with the requested capacity. Keep SP4/TP8, 32 layers,
   BFP8 cache pages and 1,024-token compute chunks.
2. Publish the table and import its protobuf. Check its capacity, 16 configs,
   32 layers, two slots, 32-token pages and 4,352-byte page size.
3. Send two distinct prompts through the shared producer and runner. Interleave
   their chunks. Require both requests to reach the full requested length.
4. Read source pages through the published table and compare their bytes with
   independently indexed live cache tensors. Cover every K/V head, layer and
   slot at the start, SP-row boundaries, a middle chunk and the final chunk.
   Include the last valid page. Check earlier pages again after continuation.
5. Confirm slot isolation: each slot must retain its own prompt's cache. Check
   full-prefix completion before any external reader consumes the data.
6. Send the shutdown sentinel. Require successful runner and producer exits and
   a verdict that records both completed lengths.
7. Record total run time and synchronized chunk times. Keep startup, reference
   generation and readback time separate from prefill execution.

## Numerical checks and test cost

The completed 2K gate compared every layer's K/V with independent FP32 Hugging
Face output. Keep that detailed numerical gate at 2K. Do not generate full
4K–64K golden K/V by default. An optional check can compare the unchanged first
2K prefix with its saved golden, provided the larger prompt starts with exactly
the same tokens. Such a prefix check does not establish numerical accuracy of
the later cache positions.

The exhaustive small table test and compact host regressions need not be repeated
as a full edge-case sweep for every capacity. The larger cases target changed
allocation strides, continuation, end-of-capacity addresses and two-slot isolation.

The initial `llama31 sc1` launcher is deliberately fixed to the accepted 2K preset.
The first follow-up step is to parameterize that preset and its completion verdict
for the requested capacity. The table above is the execution checklist for that
extension. 128K remains deferred.
