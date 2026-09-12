# AutoDebug: batch-32 linear-attention L1 collision

Status: allocation diagnosis confirmed by the main agent's narrow serialized
device experiment. This investigator inspected source and completed artifacts,
and wrote this report; it did not execute TTNN/device code or edit implementation.

## Verified follow-through

The main agent first tested the broader `public_dram_batch=32` control, which
passed (`batch32_dram_boundary.json/log`). It then restricted the repair to
attention projection outputs: `_linear` passes `force_dram` to `_public_rows`
when `role == "attention"` and batch reaches `attention_dram_batch`, configured
as 8. Conversion still occurs **before** public-row repacking. Public norm and
final residual tensors retain their previous L1 placement.

I independently inspected `batch32_attention_boundary.json/log`, its source
hash, and the changed placement predicate:

- Source SHA256: `be53b731687a023766dc019ddfa6e753efa10adf74f4330692bcfb12c7fb13bf`.
- Layer 0, batch 32, length 257: prefill PCC `0.9997835287111836`, continuation
  PCC `0.99978339493263`, traced decode PCC `0.9998478293418884`.
- Changed-input trace PCC `0.9998071789741516`; four sequential trace PCCs
  range from `0.9997844696044922` to `0.9998758435249329`.
- `repeat_bitwise_equal=true`, `runtime_fallback_audit=passed`; clean mesh close.
- Per-user decode PCCs are at least `0.9998401999473572` in the saved result.

This narrow pass verifies that moving the attention public projection boundary
and its inherited intermediates is sufficient for the original B32 GDN failure.
It does not require norm/final-residual placement changes or speculative
deallocation. B1 does not enter the new placement branch. The threshold of 8
extends the repair to B8/B16; those batches and the complete both-kind suite are
being tested by the main agent and are not claimed verified by this report.

## Starting evidence

`stress_initial_failure.log:473–479` identifies
`test_optimized_default[32-257,31-0]`: layer 0, batch 32, initial length 257.
Prefill PCC is `0.9997835287111836`, continuation PCC is
`0.99978339493263`, then decode warmup throws:

```text
ChunkGdnPrepOperation
L1 buffer allocated at 618496
static circular buffer region ends at 1115136
core range [0-0 - 0-0]
```

The Python call is `_delta` -> `chunk_gated_delta_rule`; this is a host-side
allocation-region validation failure, not an observed device hang. Subsequent
`cq_id 0 is out of range` errors occur during failure cleanup. The same stress
log contains separate full-attention continuation alignment failures; those do
not explain this layer-0 allocation error.

## Ranked diagnosis

### 1. Expanded public attention tensors occupy the GDN scratch region — verified

The DRAM-sharded matmul executes with compact `[1,1,B,K]` rows and writes an L1
width-sharded result. `_linear` then calls `_public_rows(..., keep_sharded=False)`
for `linear_attn.packed` (`optimized_decoder.py:341–357`). The original default
uses an L1 interleaved conversion before restoring `[B,1,N]`
(`:427–437`). That public layout requires a full tile-height allocation **per
batch element**. It is necessary data repacking, not a free shape view.

For B=32 and N=16512, compact tile payload is 1,056,768 bytes
(`[1,1,32,16512]`), while the public tiled result has padded shape
`[32,32,16512]` and is 33,816,576 bytes (32.25 MiB). The matmul's sharded
allocation can also include a few tail tiles; that does not change this 32x
public-row expansion.

`_delta` keeps `packed` and its extracted values in scope through GDN prep.
Slicing, padding, and untilizing normally inherit source placement:

- `operations/data_movement/slice/slice.cpp:123–124`;
- `operations/data_movement/pad/pad.cpp:208`;
- `operations/data_movement/untilize/untilize.cpp:74`.

Thus public packed L1 also leads to large L1 QKV and Z intermediates. The table
shows individual physical payloads at the decode boundary; do not sum aliased
objects such as `qkv`/`padded_qkv` without inspecting their actual buffers.

| Live Python value | Shape/layout contributing storage | Payload at B32 | Last necessary use |
|---|---|---:|---|
| `decode_forward.n`, `_delta.x` | public `[32,1,5120]`, TILE padded `[32,32,5120]` | 10 MiB | packed projection input |
| `packed` | public `[32,1,16512]`, TILE padded `[32,32,16512]` | 32.25 MiB | QKV/Z/B/A extraction |
| `qkv` | `[32,1,10240]`, TILE padded `[32,32,10240]` | 20 MiB | temporal padding / row conversion |
| `padded_qkv` | `[32,32,10240]`, TILE | 20 MiB if separately backed | row conversion |
| `row_qkv` | `[32,32,10240]`, row major | 20 MiB | per-user convolutions and history copy |
| `z` | `[32,1,6144]`, TILE padded `[32,32,6144]` | 12 MiB | gated norm and following multiply, after GDN |
| `beta` | `[32,1,48]`, TILE padded width 64 | 0.125 MiB before time padding | all GDN batch slices |
| `a` / `g` | `[32,1,48]`, FP32 TILE padded width 64 | 0.25 MiB each before time padding | `a`: compute g; `g`: GDN batch slices |

The large matrices are interleaved across L1 banks rather than residing wholly
on one core, but their combined per-bank footprint can force the lowest dynamic
allocation beneath the GDN static region. The error alone does not identify
which named tensor owns address 618496, so exact owner attribution needs an
allocation snapshot; the lifetime and size evidence supports the boundary
placement experiment without inventing that attribution.

### 2. Avoidable Python lifetimes amplify the pressure — supported, secondary

At the GDN call, Python still retains `packed`, `qkv`, `padded_qkv`, `row_qkv`,
`history_tail`, `a`, and `chunks`, despite their earlier final uses. Even when
some values alias, retaining a view can retain its underlying allocation.
`z` is genuinely needed after GDN and cannot simply be discarded.

The normalized input has **two caller references**: `decode_forward.n` and the
`_delta.x` argument. Deleting `_delta.x` alone cannot release that allocation.
A public norm placement change is safer than force-deallocating a tensor owned
by the calling frame. The existing `_public_rows` switch applies to this norm
boundary too.

### 3. The GDN scratch requirement itself is unsupported at B32 — refuted as a first explanation

`chunk_gdn_phased_program_factory.cpp:172–241` sizes prep circular buffers from
chunk and head dimensions, not the overall batch count. Here chunk=32,
K=V=128 yields Ct=1, Kt=Vt=4: 20 BF16 tiles plus 235 FP32 tiles =
**1,003,520 bytes per used core**. The observed end at 1,115,136 corresponds to
a 111,616-byte base. It is a substantial, fixed per-core reservation.

The model already splits the independent batch axis using
`scan_batch=floor(110/48)=2`, so each GDN invocation covers at most 96 value
heads. Lowering this split to one would still leave the same per-core CB sizes
and would not remove the large public L1 matrices. It is not the focused fix.

Convolution outputs already default to DRAM
(`qkv_causal_conv1d_silu.cpp:25`), as do GDN intermediate/output tensors
(`chunk_gated_delta_rule.cpp:263` and phased output specs). The persistent
recurrent state and convolution history are allocated in DRAM by the model.
There is no source basis for changing recurrent precision, switching native
GDN variants, disabling trace, or falling back to another decoder.

## Smallest focused experiment

At `_linear`'s **packed attention output** boundary, for B32 only, convert the
compact L1-sharded output to DRAM **before** reshaping to `[B,1,N]`:

```python
# After the same optimized DRAM-sharded matmul, before _public_rows.
if name == "linear_attn.packed" and batch >= 32:
    return ttnn.reshape(ttnn.to_memory_config(output, ttnn.DRAM_MEMORY_CONFIG), [batch, 1, n])
```

This changes only activation placement at that boundary. It preserves the
optimized projection, packed ordering, dtype/fidelity, convolution, recurrent
state, padded identity steps, GDN batch splitting, trace, and public shape.
The subsequent QKV/Z slices and row-major QKV should inherit DRAM placement,
removing the dominant L1 consumers before GDN starts. Do not first produce the
expanded public L1 matrix and then copy it; that retains the peak allocation.
Do not route the DRAM result through a helper that immediately copies it back
to L1.

The main agent initially added a convenient alternative switch:
`public_dram_batch=32` in `_public_rows`. Its one-knob A/B also moves the B32
public normalized-input and final residual boundaries to DRAM. That is slightly
broader than the packed-only experiment but retains B1 behavior and directly
tests the same L1-placement hypothesis. Record its broader scope accurately.
The experiment is listed in `commands.log` as `batch32_dram_boundary`, using
`batch32_dram_policy.json`. It passed, and the narrower attention-only experiment
subsequently passed as documented above. No lifetime cleanup was necessary to
resolve this reproduced failure.

Suggested focused regression, run by the main agent under exclusive device use:

```bash
python_env/bin/python -m pytest -q \
  'models/autoports/qwen_qwen3_8_27b/tests/test_optimized_decoder.py::test_optimized_default[32-257,31-0]'
```

The original test patches both FunctionalDecoder and FusedDecoder constructors
to reject fallback. Its real-activation runner checks prefill/continuation,
eager and traced decode, per-user PCC, sequential replay, changed inputs and
repeat equality. For a policy-switch experiment, use the experiment wrapper
with layer 0, batch 32, length 257, continuation and benchmark enabled, then
rerun the exact default-policy regression after retaining the fix.

At the first GDN call, the discriminating observation is unchanged q/k/v/g/beta
shape/dtype/layout and unchanged static CB requirement, accompanied by reduced
live L1 allocation so its lowest address no longer overlaps the static region.
If the placement change passes, check B1 length128/257 with the unchanged branch
and existing B32 length31 coverage before final acceptance. Use per-user PCC,
not just a flattened batch PCC, to detect accidental row-layout changes.

## If placement alone is insufficient: release in dependency order

These are follow-up experiments, not changes to bundle before the first A/B:

1. After extracting qkv/z/beta/a, remove the local `packed` reference. After
   creating `row_qkv`, remove `qkv` and `padded_qkv` references. Account for aliasing
   so a buffer is not force-deallocated while a needed view still uses it.
2. After concatenating convolution outputs and copying `history_tail` into
   `state.conv`, release `row_qkv`, `history_tail`, and the `chunks` container.
   The Q/K/V concatenations needed by GDN remain live; convolution outputs are
   DRAM by default, so `chunks` is mainly a DRAM lifetime improvement.
3. After computing `g`, release `a`. Keep g/beta until all batch slices finish.
4. Keep `z` in DRAM through GDN and gated norm. It is needed twice afterward,
   so deleting it before GDN would require unnecessary recomputation.
5. If normalized public input still materially limits L1, use the B32 DRAM
   `_public_rows` boundary for `_norm`, or restructure caller ownership so its
   last-use release occurs before GDN. Do not mutate/free caller input/state.

Prefer normal reference lifetime cleanup where possible. Explicit TTNN
deallocation must respect aliases, queued operations, and capture/replay
lifetimes, and must be checked through the existing trace regression.
