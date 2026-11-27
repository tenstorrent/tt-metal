# g6_clone: the reference implementation

The contract names the reference: "out = x.clone(): a copy of the input into a new tensor"
(`sources/local/contract.md:10`), checked against "the PyTorch function named in section 1
applied to the logical values" (`sources/local/contract.md:72`). The PyTorch function is pinned
at the commit of the torch build installed on this host (torch 2.11.0).

## Call path

1. The caller holds a device tensor `x` and calls the op once with that tensor and nothing else:
   every one of the 29 production cases has one tensor argument, no positional literal and no
   keyword (`sources/local/scenarios.json:7-32`, the same fields in every case; the oracle's probe
   "call matrix: 29 cases ..." checks all of them; `sources/local/contract.md:39`).
2. PyTorch: `clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor`, function and
   method variants, dispatched for dense tensors to the composite `clone`
   (`sources/github/pytorch/pytorch@70d99e998b49/aten/src/ATen/native/native_functions.yaml:7085-7088`).
   Default filled silently: `memory_format=None` becomes `MemoryFormat::Preserve`
   (`sources/github/pytorch/pytorch@70d99e998b49/aten/src/ATen/native/TensorFactories.cpp:2273`).
3. `clone` allocates a new tensor with the sizes, strides and options (dtype, device) of the
   source and copies the source into it
   (`sources/github/pytorch/pytorch@70d99e998b49/aten/src/ATen/native/TensorFactories.cpp:2275-2292`).
4. The documented contract: "Returns a copy of :attr:`input`."
   (`sources/github/pytorch/pytorch@70d99e998b49/torch/_torch_docs.py:2777-2779`).

## Inputs, outputs and state as the reference sees them

| Tensor | Shape (symbols) | Dtype at the call site | Memory format | Role |
|---|---|---|---|---|
| x | any rank R in 1..4, sizes (d_0, ..., d_{R-1}) | bfloat16, float32, int32, uint32 | the device tensor's layout (TILE or ROW_MAJOR), DRAM interleaved | input |
| out | the shape of x | the dtype of x | the layout and placement of x | output, new storage |

No state: nothing is kept between calls (`sources/local/contract.md:68`).

## Semantic statements

| Id | Statement | Source | Note |
|---|---|---|---|
| S1 | out is a new tensor with the shape and the dtype of x; with `memory_format` preserved it also keeps the source's memory arrangement | `sources/github/pytorch/pytorch@70d99e998b49/aten/src/ATen/native/TensorFactories.cpp:2273-2285` | on the device the arrangement is layout and placement, which the contract fixes equal to the input's (`sources/local/contract.md:26-37`) |
| S2 | out[i] = x[i] for every logical index i: the value is copied, not converted | `sources/github/pytorch/pytorch@70d99e998b49/aten/src/ATen/native/TensorFactories.cpp:2290` | same dtype on both sides, so the copy is bit for bit (oracle probes "bit patterns ...", "special values ...") |
| S3 | a zero tensor (the internal `_is_zerotensor` marker) is filled with zeros instead | `sources/github/pytorch/pytorch@70d99e998b49/aten/src/ATen/native/TensorFactories.cpp:2287-2288` | a PyTorch autograd artifact with the same values as S2; no device tensor has it |

## Semantics versus GPU artifacts

| Construct in the reference | Verdict | Why |
|---|---|---|
| `empty_strided` with the source's strides for a dense source, `empty_like` otherwise | artifact: dropped | host stride bookkeeping; a device tensor's arrangement is its layout and placement, fixed by the contract |
| `memory_format` keyword | artifact: dropped (DOSSIER.md D4) | no device tensor has strides to preserve or change; no production call passes it (`sources/local/contract.md:39`) |
| `_is_zerotensor` branch (S3) | artifact: dropped | gives the same values as S2 |
| copy without conversion (S2) | semantics: kept | changes nothing in a value: every bit pattern, -0.0, inf, NaN payloads and subnormals included |
| new storage (S1) | semantics: kept | the result must not alias the input: a later write to one must not show in the other (oracle probe "new storage ...") |

## Traps

- A float32 detour is not a copy: an oracle in the usual form (compute in float32, cast back)
  loses int32 and uint32 values above 2^24 in magnitude; 2^24 + 1 comes back as 2^24 (oracle
  probe "float32 round trip loses int32 2^24 + 1"). The oracle keeps the input dtype (D6).
- A path through an arithmetic engine is not a copy either: float32 operands of the matrix
  engine are cut to TF32 (the oracle's float32 precision row shows that cut as relative RMS
  0.0004 on randn data, emulated, not confirmed on a device), and any unpack and pack may treat
  subnormals, -0.0 and NaN payloads otherwise than a byte copy (INFERENCE; risks.md R1).
- The result must be new storage: returning the input tensor itself, or a view of it, has the
  right values and breaks every caller that writes either tensor later (S1).
- Padding of a TILE tensor is not part of the result (`sources/local/contract.md:74`); the
  logical elements are (S2).

## Paper versus code

No paper: the reference is a library function. The documentation and the code agree (one copy,
new storage, same dtype): `sources/github/pytorch/pytorch@70d99e998b49/torch/_torch_docs.py:2779`
and `sources/github/pytorch/pytorch@70d99e998b49/aten/src/ATen/native/TensorFactories.cpp:2270-2293`.

| Topic | Paper | Code | Effect and evidence | The spec follows |
|---|---|---|---|---|
| none | - | - | - | code |

## Op boundary

- Inside: one call copies one input tensor into one newly allocated output tensor of the same
  shape, dtype, layout and memory placement (`sources/local/contract.md:10`, `:26-37`).
- Outside: whatever produced the input and whatever consumes the output; nothing else changes
  (`sources/local/contract.md:17-18`).
- Why: decided by the lead. One call is one device program, never merged with a neighbour and
  never split (`sources/local/contract.md:20`).

## Around the op in the model

Shapes are per device; FLUX.2-dev runs on a (2,2) mesh and Qwen3.8-27B text-only on a (1,4)
mesh of Blackhole chips, every chip with the same per-device shape
(`sources/local/contract.md:24`, `:64`). Each line below is a case of
`sources/local/scenarios.json` (its `id` in brackets; FLUX ids start at 4); calls are per traced
forward pass.

| Mode | Shapes | Dtype | Called how often |
|---|---|---|---|
| Qwen, TILE, DRAM | [16,12,128,128] (2, p1), [8,12,128,128] (6, p2), [1,12,128,128] (13) | float32 | 336, 480, 816 |
| Qwen, TILE, DRAM | [1,1,32,62080] (15, p4), [1,1,1,62080] (8) | bfloat16 | 15, 148 |
| Qwen, ROW_MAJOR, DRAM | [16,3,2560] (3), [8,3,2560] (7), [1,3,2560] (14) | bfloat16 | 336, 480, 816 |
| Qwen, ROW_MAJOR, DRAM | [16] (0), [8] (4), [32] (9), [1] (10), [1,1,1,32] (12) | uint32 | 10, 9, 32, 17, 32 |
| Qwen, ROW_MAJOR, DRAM | [16] (1), [8] (5), [1] (11) | int32 | 10, 9, 13 |
| FLUX, TILE, DRAM | [1,512,5120] (4, p3), [1,256,5120] (12), [1,256,3072] (6), [1,5120] (13) | bfloat16 | 42, 3, 18, 6 |
| FLUX, TILE, DRAM | [1,2048,128] (5), [1,8192,128] (14), [1,1,2048,128] (9), [1,1,256,128] (10), [1,1,2304,128] (11), [1,1,8192,128] (15), [1,1,8448,128] (16) | bfloat16 | 15, 3, 30, 36, 30, 6, 6 |
| FLUX, TILE, DRAM | [1,1] (8) | bfloat16 | 18 |
| FLUX, TILE, DRAM | [1,1] (7) | float32 | 36 |

Totals: Qwen 16 cases and 3559 calls, FLUX 13 cases and 249 calls
(`sources/local/contract.md:45-48`). The flagged performance cases p1 to p4
(`sources/local/contract.md:50-57`) are marked above.

## Upstream tests and expected values

- Not pinned: PyTorch's own tests of `clone` were not read. The expected values are the
  definition itself (S2), checked bit for bit by the oracle on every production dtype and shape.
