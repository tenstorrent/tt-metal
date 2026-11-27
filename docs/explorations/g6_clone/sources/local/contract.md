# Interface contract: g6_clone

Campaign: group 6 (data movement, layout, creation), op library. Revision 2, 2026-11-25, written by the lead from the traced production runs of three models.
Revision 2 differs from revision 1 only in section 7 (the gate for block formats and the content of tile padding); a revision is a new file and a contract that was handed in is never edited.

## 1. The unit

- Name: g6_clone
- Kind: op
- One call computes: out = x.clone(): a copy of the input into a new tensor, with the dtype and memory placement the call asks for.
- Replaces today: 29 case(s) with 1 device program(s) per call (the incumbent op of the same function; one call of the new op is exactly one device program).
- Calls per model step: the traced forward passes make the calls listed in section 4 (FLUX: 249 calls in 13 distinct cases, Qwen: 3559 calls in 16 distinct cases).
- Tags: multi-core,mesh,traced,fp32

## 2. Boundary

- Before the unit runs: any op that produced the input tensor(s) in the layout and placement of section 3; the unit changes nothing else.
- After the unit: any op that consumes the output in the layout and placement of section 3.
- Programs between the unit and each neighbour after integration: 0.
- Fixed: this boundary. One call is one device program. The unit is never merged with a neighbouring unit and never split by its generator. Free: everything inside it.

## 3. Tensors at the boundary

Shapes are per device (FLUX.2-dev on a (2,2) mesh of 4 Blackhole chips (shapes are per device); Qwen3.8-27B text-only on a (1,4) mesh of 4 Blackhole chips (shapes are per device)). The classes below are every combination of dtype, layout and memory placement that production calls the op with; `Count` is the number of distinct cases (section 4). Every tensor is on device; none is persistent; no tensor is updated in place unless the semantics above say so.

| Name | Role | Example shape | dtype | Layout | Memory | Rank | Count |
|---|---|---|---|---|---|---|---|
| in0 | input | [1, 1, 1, 62080] | bfloat16 | TILE | INTERLEAVED/DRAM | 2,3,4 | 14 |
| in0 | input | [16] | uint32 | ROW_MAJOR | INTERLEAVED/DRAM | 1,4 | 5 |
| in0 | input | [16, 12, 128, 128] | float32 | TILE | INTERLEAVED/DRAM | 2,4 | 4 |
| in0 | input | [16] | int32 | ROW_MAJOR | INTERLEAVED/DRAM | 1 | 3 |
| in0 | input | [16, 3, 2560] | bfloat16 | ROW_MAJOR | INTERLEAVED/DRAM | 3 | 3 |
| out | output | [1, 1, 1, 62080] | bfloat16 | TILE | INTERLEAVED/DRAM | 2,3,4 | 14 |
| out | output | [16] | uint32 | ROW_MAJOR | INTERLEAVED/DRAM | 1,4 | 5 |
| out | output | [16, 12, 128, 128] | float32 | TILE | INTERLEAVED/DRAM | 2,4 | 4 |
| out | output | [16] | int32 | ROW_MAJOR | INTERLEAVED/DRAM | 1 | 3 |
| out | output | [16, 3, 2560] | bfloat16 | ROW_MAJOR | INTERLEAVED/DRAM | 3 | 3 |

Other arguments of the call: none besides the tensors; 0 distinct sets of non-tensor positional values (begins, ends, dims, shapes, padding, memory configs), all listed per case in the complete case list.

## 4. Call matrix

Every case production calls the unit with becomes a case of the golden suite. The complete list is the file `scenarios.json` of this unit (every case: tensors with shape, dtype, layout, memory; the other arguments; the output). Summary by model:

| Model | Distinct cases | Calls in the traced passes | Mesh |
|---|---|---|---|
| FLUX.2-dev | 13 | 249 | a (2,2) mesh of 4 Blackhole chips (shapes are per device) |
| Qwen3.8-27B text-only | 16 | 3559 | a (1,4) mesh of 4 Blackhole chips (shapes are per device) |

Flagged performance cases (the cases the incumbent spends the most device time on; the pipeline tunes these):

| Case | Model | Input shape(s) | dtype | Layout | Memory | Perf |
|---|---|---|---|---|---|---|
| p1 | Qwen | [16, 12, 128, 128] | float32 | TILE | INTERLEAVED/DRAM | yes |
| p2 | Qwen | [8, 12, 128, 128] | float32 | TILE | INTERLEAVED/DRAM | yes |
| p3 | FLUX | [1, 512, 5120] | bfloat16 | TILE | INTERLEAVED/DRAM | yes |
| p4 | Qwen | [1, 1, 32, 62080] | bfloat16 | TILE | INTERLEAVED/DRAM | yes |

## 5. Environment (parameters of the black rules)

- l1_free_bytes: 1347776 (the free L1 per core measured in a traced FLUX.2-dev step; assumed for all three models, not measured separately)
- Trace: the call runs inside a captured trace; buffers allocated before capture: the output buffers the caller passes or the unit allocates before capture
- Program cache: on; distinct call signatures per request: up to the number of distinct cases of section 4 that one model pass makes
- Mesh: the unit runs on every chip of the mesh with the same per-device shape

## 6. State

Not stateful.

## 7. Accuracy

- Reference: the PyTorch function named in section 1 applied to the logical values; weights: not applicable (random inputs).
- Gate: bit-exact equality of the logical values for every output dtype except the block formats; for an output in a block format (bfloat8_b, bfloat4_b) the device result equals the host rendering of the reference values in that format within one quantization step of the format (the step of the group's shared exponent), because moving values regroups the shared-exponent groups and the result is not bit-comparable to the logical reference
- Padding: positions of a tiled tensor beyond its logical shape are not part of the result; their content is unspecified (the unit need not write zeros), except for the pad op, whose padding value is its result. (Whether any consumer in the three models reads padding positions as values was not checked; the integration tests will show it.)
- Model-level check after integration: the models' own accuracy tests at their thresholds.
