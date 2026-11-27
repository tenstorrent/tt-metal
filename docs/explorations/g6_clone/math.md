# g6_clone: mathematical design

## Notation

| Symbol | Meaning | Shape or range | Fixed by the model, or per call |
|---|---|---|---|
| R | rank of the input | 1, 2, 3 or 4 in the production cases | per call (per case) |
| d_0 .. d_{R-1} | sizes of the input; d_{R-1} is the last (row) dimension W, d_{R-2} the height H | 1 to 62080 in the production cases | per call (per case) |
| i | a logical index (i_0, ..., i_{R-1}), 0 <= i_k < d_k | - | - |
| N | number of logical elements, prod_k d_k | 1 to 3145728 in the production cases | per call |
| x | the input tensor | (d_0, ..., d_{R-1}) | per call |
| out | the output tensor | (d_0, ..., d_{R-1}) | per call |
| dtype | bfloat16, float32, int32 or uint32; the same for x and out | - | per call |

## Definition

    M1  out = new tensor, shape(out) = shape(x), dtype(out) = dtype(x), layout(out) = layout(x), placement(out) = placement(x)
    M2  bits(out[i]) = bits(x[i])   for every logical index i

Edge cases:
- Special values: -0.0, +-inf, NaN with any payload and subnormals of bfloat16 and float32 are
  copied as their bit patterns (M2); integers of every value, int32 -2^31 and uint32 2^32 - 1
  included, likewise.
- Tile padding: positions of a TILE tensor beyond its logical shape are not part of the result;
  their content in out is unspecified (`sources/local/contract.md:74`). They never enter a value
  of M2, since no logical element depends on another.
- Size 1 and single-tile tensors ([1], [1,1], [16]): the definition is the same; nothing is
  reduced, so no degenerate case exists.
- Aliasing: out shares no storage with x (M1, "new tensor").

## Equivalence to the reference

| Statement of the definition | Reference statements |
|---|---|
| M1 | S1 |
| M2 | S2 (S3 gives the same values) |

## Axes and dependence

No result reads more than one input position: out[i] depends on x[i] alone, for every axis.

| Axis | Extent in the model | Does a result span it? Why | Partial state and combine function | Operands that vary along it |
|---|---|---|---|---|
| d_0 (leading, batch-like) | 1 to 32 | no: element i reads x[i] only | none | x, out |
| d_1 (heads, rows) | 1 to 8192 | no: as above | none | x, out |
| H = d_{R-2} | 1 to 8448 | no: as above | none | x, out |
| W = d_{R-1} (last) | 1 to 62080 | no: as above | none | x, out |

Every axis can be cut anywhere into independent parts; there is nothing to combine. The only
coupling is the storage layout: a TILE tensor stores 32 x 32 tiles, a ROW_MAJOR tensor one page
per row (tensix.md, Geometry).

## Reformulations

None that changes a value: every evaluation order of M2 gives the same bits. Moving bytes in
any grouping (whole pages, parts of pages, several pages at once) is the same function as long
as every logical element lands at its own position in the output's layout. That is a fact of
storage, not an F-form, and needs no check of its own beyond the probes of
`oracle/check_oracle.py`.

## Counts

| Quantity | Formula | p1 [16,12,128,128] float32 | p3 [1,512,5120] bfloat16 | p4 [1,1,32,62080] bfloat16 | [1,1,1,62080] bfloat16 TILE | [16,3,2560] bfloat16 ROW_MAJOR |
|---|---|---|---|---|---|---|
| Arithmetic operations | 0 | 0 | 0 | 0 | 0 | 0 |
| Logical elements N | prod d_k | 3145728 | 2621440 | 1986560 | 62080 | 122880 |
| Logical bytes read and written, each | N x bytes(dtype) | 12582912 | 5242880 | 3973120 | 124160 | 245760 |
| Passes over x | 1 read | 1 | 1 | 1 | 1 | 1 |
| Passes over out | 1 write | 1 | 1 | 1 | 1 | 1 |
| State carried | 0 | 0 | 0 | 0 | 0 | 0 |

The stored size can exceed the logical size: tensix.md, Geometry, gives the tiled and paged
bytes of every production tensor.

## Numerics

- Value ranges: any bit pattern of the dtype; the op computes nothing, so no range matters.
- Accumulation depth: none.
- Hazards: any conversion on the way. A float32 detour loses integers above 2^24 (oracle probe);
  an engine that reads float32 as TF32 cuts 13 mantissa bits; a pass through a 16-bit register
  rounds float32 to bfloat16 (INFERENCE from the formats). All three break M2.
- Stability steps that are part of the definition: none.
- Order sensitivity: none.
- Host precision checks: `oracle/oracle_report.json`, float32 row: relative RMS 0.0004 if the
  inputs were cut to TF32 (emulated (storage and operands only), not confirmed on a device): the
  size of the error a matrix-engine path would make, within the default float32 tolerance and
  still not a copy. bfloat16 row: storage floor 0. bfloat8_b is not in the call matrix: not run.
- Inputs that stress it: random bit patterns of each dtype (NaN payloads, subnormals, -0.0,
  integers above 2^24), not randn alone, because randn values survive a TF32 or bfloat16 detour
  more often than not at the default tolerance.

## Definition versus choice

| Must be preserved | Free |
|---|---|
| every logical element's bit pattern (M2) | how bytes are grouped into transfers, how the work is split, which cores do it |
| shape, dtype, layout and placement of out equal to x's (M1) | the content of tile padding in out |
| out is new storage, x unchanged | the order in which elements are moved |
