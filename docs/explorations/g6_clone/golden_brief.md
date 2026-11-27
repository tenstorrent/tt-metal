# g6_clone: golden brief

Marks: [spec] fixed by the spec, [recommend] a recommendation for the user to confirm,
[candidate] a suggestion the suite author decides. Under the lead's contract the suite holds the
29 production cases of `sources/local/scenarios.json` and the cases the mandated rules name,
each once; no further axis value is proposed for generality (BB-22).

## 0. Identity

- op_name: `g6_clone` (the "# golden:" header, the suite directory, the op module and the
  stem of the spec file)
- Signature: see `spec/g6_clone.txt`; position of `*`: none, the op takes `input_tensor` alone
- Symbols the suite imports from the op module: `g6_clone`, `INPUT_TAGGERS`, `SUPPORTED`,
  `EXCLUSIONS`; beyond these: nothing (no `fp32_dest_acc_en` axis, no compute configuration)
- Keyword arguments the runner passes on every call: none [spec]

## 1. Oracle  [A1, A2]

- Math: `math.md#definition`
- `pytorch_g6_clone(...)`: `oracle/pytorch_g6_clone.py`, pure torch, the op's single parameter;
  the built-in `torch.clone`, with no float32 round trip: the input dtype in and out, because
  int32 and uint32 values above 2^24 are not exact in float32 (D6) [spec]
- Provenance: pytorch/pytorch@70d99e998b4955e0049d13a98d77ae1b14db1f45 (torch 2.11.0),
  `sources/github/pytorch/pytorch@70d99e998b49/aten/src/ATen/native/TensorFactories.cpp:2270-2293`;
  deliberate differences: `memory_format` is not a parameter (D4); no float32 round trip (D6)
- Equivalence: E1. `upstream()` is `torch.clone` of the installed torch 2.11.0, the build whose
  source is pinned; compared in float64 on seeded shapes (1,1,32,32), (2,3,64,96), (1,2,17,50),
  (16,), (1,1), (3,5,7), seeds 0 to 5, largest relative difference 0.0. Probes, all passed: random
  bit patterns of bfloat16, float32, int32 and uint32 bit-identical through reference, oracle and
  upstream; -0.0, +-inf, NaN payloads, subnormals and integer extremes kept; a float32 round trip
  loses int32 2^24 + 1; new storage; all 29 production shapes bit-identical
  (`oracle/oracle_report.json`). No comparison at another level.
- Weight and state tensors as the op receives them: none
- Inputs whose values have a domain: none restricts the values, but the generator matters. Every
  product cell draws random bit patterns of its dtype, uniform over all patterns (bfloat16: int16
  patterns viewed as bfloat16; float32: int32 patterns viewed as float32; int32 and uint32: the
  full 32-bit range), seeded, instead of the template's `torch.randn` line: randn values survive a
  TF32 or bfloat16 detour often enough to hide R1, while random patterns hold NaN payloads,
  subnormals and integers above 2^24 (`math.md#numerics`) [spec: the spec's exact, bit-pattern
  comparison needs it]. If the host-to-device path cannot carry NaN payloads unchanged, keep the
  NaN payload case (0x7FC00001) as an expected failure with that reason rather than drop it (Q3).
- Output contract, one line per returned tensor: out, shape, dtype and layout of the input, DRAM
  interleaved; the oracle returns `input.clone()`; graded exactly, bit for bit, because the
  tensor is moved and not computed; the runner also asserts on every cell that the output's
  dtype, layout and memory configuration equal the input's [spec]  [C6]

## 2. Axes  [B3, B4, C5, D7, E1-E7]

| Axis | TARGET values (Python literals) | Derived from | Phase 0 | What a missing value changes | Note |
|---|---|---|---|---|---|
| dtype | [ttnn.float32, ttnn.bfloat16, ttnn.int32, ttnn.uint32] | tagger tag_dtype(inputs, axes): inputs[0]["dtype"], the dtype of input_tensor; the output shares it | ttnn.float32 | the element width: 4 B or 2 B, so tile bytes (4096 or 2048) and row bytes; no value is converted | [recommend] |
| layout | [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT] | tagger tag_layout(inputs, axes): inputs[0]["layout"]; the output shares it | ttnn.TILE_LAYOUT | what a page is: one 32x32 tile, or one row of the last dimension (4 B to 5120 B in production) | [recommend] |

Both axes are projected from a scenario dict (`inputs` = `(scenario,)`, scenario =
`{"input_shape": ..., "dtype": ..., "layout": ...}`), as shipped data-movement suites project
their layout, so that only the five (dtype, layout) pairs production uses are built: float32 and
bfloat16 in TILE, bfloat16, int32 and uint32 in ROW_MAJOR [recommend]. Projecting `dtype` from a
scenario dict has no precedent in a shipped suite (INFERENCE that it works the same way as
`layout`) [candidate].

Cardinality: no finite axis is iterated: one cell per INPUTS entry, 29 cells.
Discretionary choices [E1-E6]: no `alignment` axis and no `rank` axis: a copy has no kernel
path that differs by alignment or rank (math.md, axes table: no element depends on another), so
a tagger would bucket shapes only; the non-aligned and rank-1 to rank-4 shapes are in INPUTS
anyway [recommend]. No `memory_layout` axis: every case is DRAM interleaved
(`sources/local/contract.md:26-37`); the runner creates DRAM-interleaved tensors [spec]. No
`output_dtype`: the output dtype is the input's [spec].
Compute config [E7]: not exposed; the op does no arithmetic.

## 3. Cells to keep out, by mechanism

- Not in TARGET: bfloat8_b and other dtypes, sharded and L1 placements, rank 0 and above 4: no
  production call uses them (`sources/local/contract.md:26-37`) and BB-22 limits the suite to
  production cases [recommend].
- INVALID candidates: none. Every (dtype, layout) pair of TARGET can be stored; INPUTS builds the
  production pairs only.
- EXCLUSIONS the spec instructs: none.
- INVALID_FOR_ARCH candidates: none.

## 4. INPUTS

- Entry form: scenario dict, one per production case of `sources/local/scenarios.json`.
- Shapes, each with its reason (case id in brackets; Q = Qwen, F = FLUX):

  | Shape | Why it is in |
  |---|---|
  | (16, 12, 128, 128) float32 TILE | Q2, flagged p1; 3072 tiles, 768 x 4 |
  | (8, 12, 128, 128) float32 TILE | Q6, flagged p2; 1536 tiles |
  | (1, 12, 128, 128) float32 TILE | Q13; 192 tiles |
  | (1, 1) float32 TILE | F7; single tile, both dims non-aligned; pairs with F8 (R6) |
  | (1, 512, 5120) bfloat16 TILE | F4, flagged p3; 2560 tiles, 16 x 160 |
  | (1, 1, 32, 62080) bfloat16 TILE | Q15, flagged p4; one tile row of 1940 tiles (R4) |
  | (1, 1, 1, 62080) bfloat16 TILE | Q8; H non-aligned, 1 logical row per tile (R3) |
  | (1, 5120) bfloat16 TILE | F13; rank 2, H non-aligned |
  | (1, 256, 5120), (1, 256, 3072) bfloat16 TILE | F12, F6; short-wide |
  | (1, 2048, 128), (1, 8192, 128) bfloat16 TILE | F5, F14; rank 3, tall-narrow |
  | (1, 1, 2048, 128), (1, 1, 256, 128), (1, 1, 2304, 128), (1, 1, 8192, 128), (1, 1, 8448, 128) bfloat16 TILE | F9, F10, F11, F15, F16; rank 4, tall-narrow, H at least 512 |
  | (1, 1) bfloat16 TILE | F8; single tile; pairs with F7 (R6) |
  | (16, 3, 2560), (8, 3, 2560), (1, 3, 2560) bfloat16 ROW_MAJOR | Q3, Q7, Q14; rows of 5120 B; 48, 24, 3 pages |
  | (16,), (8,), (32,), (1,) uint32 ROW_MAJOR | Q0, Q4, Q9, Q10; rank 1, pages of 64, 32, 128, 4 B (R5) |
  | (1, 1, 1, 32) uint32 ROW_MAJOR | Q12; rank 4, one page of 128 B |
  | (16,), (8,), (1,) int32 ROW_MAJOR | Q1, Q5, Q11; pair with the uint32 shapes (R6) |

  Standard buckets: ranks 1 to 4 are covered; W non-aligned only and both non-aligned: (1, 1)
  and the rank-1 ROW_MAJOR shapes; H non-aligned only: (1, 1, 1, 62080), (1, 5120); wide: 62080
  and 5120; H at least 512: yes. A bucket production does not call is not added (BB-22).
- Witness table: dtype ttnn.float32 (16, 12, 128, 128); ttnn.bfloat16 (1, 512, 5120); ttnn.int32
  (16,); ttnn.uint32 (16,). layout ttnn.TILE_LAYOUT (16, 12, 128, 128); ttnn.ROW_MAJOR_LAYOUT
  (16, 3, 2560).
- Shapes that stress per-core memory: none. The largest tensor, (16, 12, 128, 128) float32, is a
  grid of 768 x 4 tiles; one unit of work of the page-staging regime holds a few pages, 4096 B
  each (tensix.md).

## 5. LOOSE_CASES candidates

- Model profile: the four flagged cases are the model profile; each stands in INPUTS already, at
  the model's dtype, layout and DRAM-interleaved placement; no fidelity applies.
- Perf focus (the contract flags these, `sources/local/contract.md:50-57`), `# attention: PERF
  FOCUS` on [recommend]:
  - `{"inputs": ({"input_shape": (16, 12, 128, 128), "dtype": ttnn.float32, "layout": ttnn.TILE_LAYOUT},)}`
    p1: 3072 tiles, 12582912 B read and 12582912 B written, no arithmetic: the goal is the DRAM
    traffic of one read and one write of the tensor.
  - p2 (8, 12, 128, 128) float32 TILE: 1536 tiles, 6291456 B each way; same form of goal.
  - p3 (1, 512, 5120) bfloat16 TILE: 2560 tiles, 5242880 B each way.
  - p4 (1, 1, 32, 62080) bfloat16 TILE: 1940 tiles in one tile row, 3973120 B each way.
  All four exceed 32 tiles, so the perf passes measure them. Being INPUTS entries already, they
  need only the comment, not a second test [candidate].
- Design-forcing: none beyond p4, whose single tile row of 1940 tiles is the shape a split by
  tile rows leaves on one core; the outcome that must hold is the cores rule of the spec.
- Adversarial numerics: random bit patterns are the product draw already (brief 1).
- Poisoned padding:
  `{"inputs": ({"input_shape": (1, 1), "dtype": ttnn.float32, "layout": ttnn.TILE_LAYOUT},), "extras": {"poison_padding": float("nan")}}`
  and the same for (1, 1, 1, 62080) bfloat16 TILE (an expected xfail until bfloat16 is
  supported): the input's tile padding filled with a loud value (NaN, or 1e30 if NaN cannot be
  placed there); the logical output must be unchanged. A copy has no reduction, so folded padding
  makes no scale error; what fails is a misaddressed row or column of a partial tile [recommend].

## 6. Regression tests

Inside Phase 0 (float32, TILE). Graded bit for bit.

| Test | Input generator and seed | Shape (inside Phase 0) | Hazard it targets |
|---|---|---|---|
| bit patterns | float32 specials: -0.0, +-inf, NaN with payload 0x7FC00001, subnormals 0x00000001 and 0x807FFFFF, 0x7F7FFFFF, tiled over the tensor; plus uniform random patterns, seed 0 | (1, 12, 128, 128) | R1: an engine or conversion path |
| new storage | randn, seed 1 | (1, 1) | R2: the output must not share a buffer with the input (buffer addresses differ; the input read back after the call equals its original bit for bit; writing the input afterwards leaves the output unchanged) |
| repeat | random patterns, seed 2, called twice | (16, 12, 128, 128) | BB-06: bit-identical results on every run |

## 7. Tolerances  [D8]

- Exact for every dtype: bit-identical output, the contract's gate
  (`sources/local/contract.md:73`). The suite copies the data-movement form of the reshard suite's
  `helpers.py` of a shipped data-movement suite (`("exact", None)` per dtype, compared outside `check_output`), and compares bit
  patterns (int16 or int32 views) rather than values, so that NaN equals NaN with its payload and
  -0.0 differs from 0.0 [recommend]. The block-format clause of the gate does not apply: no
  bfloat8_b or bfloat4_b case exists.
- Outputs graded by different rules: none: one output.
- Metric caveats for this op: PCC and relative RMS are not used; a tolerance would pass the TF32
  cut (relative RMS 0.0004 on randn, `oracle/oracle_report.json`, emulated (storage and operands
  only), not confirmed on a device).
- Phases that `math_approx_mode=True` makes coarse: none.

## 8. Rules  [F]

The op's own rules (the mandated BB rules follow them in the spec, word for word):

- When the op runs (every call, Phase 0 included): every logical element of
  the output MUST be bit-identical to its input element, for every bit
  pattern of the dtype -- MUST NOT pass the values through a conversion or an
  arithmetic engine (a float32 detour, a matrix-engine operand, a 16-bit
  register), which changes integers above 2^24, low float32 mantissa bits,
  subnormals or NaN payloads while randn inputs still look right.
- When the op returns (every call): the output MUST be a newly allocated
  device tensor that shares no buffer with input_tensor, and input_tensor
  MUST be unchanged by the call -- MUST NOT return input_tensor, a view of it
  or any tensor on its buffer, nor use the input's buffer as scratch; the
  values would be right and a later write through either tensor would change
  the other.
- When the entry point runs (every call): it MUST dispatch the op's own
  program, built from its own program descriptor and kernels -- MUST NOT
  delegate the copy to any existing device op, a memory-configuration or
  layout conversion included; one wrapped op is one program with right
  values, which no golden cell can tell apart.
- When the op runs (every call): input_tensor MUST cross DRAM at most once as
  a read and the output at most once as a write -- MUST NOT stage the data
  through an intermediate device tensor or a second pass over DRAM; what
  stays inside the cores' own L1 is allowed, and no golden cell counts DRAM
  passes.
- When ROW_MAJOR_LAYOUT is in SUPPORTED or an input is not tile-aligned
  (Phase 0 included): kernels MUST handle that data natively -- the entry
  point MUST NOT call ttnn.to_layout, ttnn.tilize, ttnn.untilize, ttnn.pad,
  ttnn.slice or any other host-side tensor transformation; the values would
  be right and the kernel unchanged.
- When a TILE input is not tile-aligned: the logical elements of the output
  MUST NOT depend on the content of the input's tile padding -- a producer
  in the model owes no zeros there, while a host-made test tensor holds
  zeros and hides the defect.
- When a shape in the golden INPUTS runs out of memory: the op MUST NOT
  refuse the shape, list it in EXCLUSIONS or tag it by size -- the cell stays
  a failing cell until the design runs it; an out-of-memory there, or a
  refusal under the per-core L1 bound of the model environment, is a defect
  of the block, never a support statement.
- When declaring SUPPORTED, INPUT_TAGGERS or EXCLUSIONS (every phase): the op
  MUST NOT add an alignment, rank or size axis, a tagger beyond tag_dtype and
  tag_layout, or an EXCLUSIONS entry that refuses a production case -- every
  case of TARGET is a plain copy, and a refused cell is an expected failure
  that hides missing work.
- When an input holds more than one tile or row: the number of cores used
  MUST be the blocking decision's choice, not a limit imposed by the tensor's
  shape -- no golden cell sees how many cores ran.

## 9. Open items for the suite author

- Axis names: `dtype` and `layout` exist already; their projection from a scenario dict through
  `tag_dtype` and `tag_layout` is new for `dtype` [candidate].
- The runner: random-bit-pattern inputs per dtype (brief 1); DRAM-interleaved placement for
  every case; bitwise exact comparison (brief 7).
- Extras keys the runner must recognize: `poison_padding` (brief 5).
- The per-core L1 bound of the model (BB-05) is not exercised by a suite that runs the op alone; a
  case that holds the rest of L1 allocated while p1 and p4 run would see it [candidate].
- LOOSE_CASES and their test are added by hand; so are the PERF FOCUS comments on p1 to p4.
- uint32 tensors: the host side of torch holds `torch.uint32` with few operations; generate
  int32 patterns and view them as uint32 (as the oracle does).
- Keep `eval/prompts/g6_clone.txt` as installed: the spec exists and is authoritative; the
  skill must not write or change it.
