# AutoFix diagnosis: fused CCL datatype contract

Source-only investigation of the experimental AGMM candidate. No hardware
commands or implementation changes were made by this agent. Runtime controls
and any repair are owned by the parent; results are not assumed here.

## Evidence and finding

Failing command recorded in `commands.log`:

```bash
bash models/autoports/qwen_qwen3_8_27b/tests/run_multichip_experiment.sh combined_agmm_ccl8_adapted_l0 --policy-file models/autoports/qwen_qwen3_8_27b/doc/multichip_decoder/agmm_ccl8_adapted_policy.json --repeats 20
```

The log reaches the baseline PCC finite-value assertion after the trace checks.
It does not identify the failing output; the parent is adding that evidence.
The policy combines Ring AGMM, sharded residuals, packed MLP and BFP8 CCL,
with `carry_residual=false`. The earlier BF16 `agmm_adapted_l0.json` passes.
That earlier run also differs in packing/carry flags, so BF16 versus BFP8 is
not yet a fully isolated runtime contrast.

**Concrete native contract mismatch:** at inspection, the model casts AGMM
input to BFP8 while requesting BF16 output. Without an explicit gather buffer,
the native op uses the requested output dtype for its gather buffer too, then
reads raw BFP8 input using BF16 tile size and interpretation. This is a
byte-layout mismatch, not evidence of inherent BFP8 numerical instability.

Paths below under `ttnn/cpp/ttnn/operations/experimental/ccl/` refer to the
named op's directory.

1. `multichip_decoder.py:251-279`: output/down projection input becomes BFP8;
   `all_gather_minimal_matmul_async(..., dtype=ttnn.bfloat16)` receives it.
2. AGMM `device/all_gather_minimal_matmul_async_device_operation.cpp:367-375`
   uses `output_dtype` for the activation-gather intermediate; line 393 uses
   the same dtype for matmul output. No persistent gather buffer is supplied.
3. AGMM `device/all_gather_minimal_matmul_async_program_factory.cpp:228-229`
   derives activation CB format/size from that BF16 gather tensor. It enables
   `READ_FROM_LOCAL_INPUT` at line 699; lines 805-807 separately identify the
   original input as BFP8.
4. AGMM `device/kernels/dm_in0_sender.cpp:426-436` passes the gather-derived
   `in0_tile_size` into `read_in0_block_sync`, together with the original
   input accessor. `device/kernels/matmul_dataflow_common.hpp:443-447` issues
   a raw `noc.async_read` of that size from the original input. No cast occurs.
5. A BF16 tile is 2048 bytes; BFP8 is 1088 bytes
   (`tt_metal/api/tt-metalium/tt_backend_api_types.hpp:128,138`). The local
   read therefore requests 2048 bytes from a 1088-byte tiled source and the
   consumer unpacks them as BF16. The factory's CB allocation at lines 430-431
   confirms the unpack format. This can produce invalid/nonfinite activation
   values; exact first affected tensor still needs the parent's probe.

## Shape/weight and packing adjudication

AGMM intentionally changes output/down weights to output-axis shards:
`multichip_decoder.py:109-112,149-154`. Linear-attention output has local input
K1536, gathered K6144, local weight `[6144,1280]`; MLP down has local K4352,
gathered K17408, local weight `[17408,1280]`. Ring K-block 8 divides both
local tile counts, 48 and 136. This ordering matches head/intermediate shards.

Packed MLP builds each local weight as `[gate4352,up4352]` columns
(`multichip_decoder.py:174-176`), and the inherited `_finish` splits at the
local `intermediate_size=4352` (`optimized_decoder.py:576-581`). It applies
ordinary SiLU/multiply; it does not enable the native tile-interleaved
`fuse_swiglu` contract. No packing contradiction was found. The first AGMM
output projection occurs before this packed MLP, making it a useful boundary
for the parent's finite-value probe.

## Minimum experiment and prospective fix

- Keep the combined policy fixed and make AGMM output dtype equal to its
  input dtype (`dtype=xx.dtype`), then typecast its returned projection to
  BF16 before residual arithmetic. This makes input, gather buffer and
  activation CB all BFP8. It also quantizes matmul output to BFP8, so require
  final PCC/state/cache and refreshed-input trace checks; do not assume this
  extra rounding meets the accuracy gate.
- Parent's planned BF16 AGMM packed control isolates the interaction with
  packing/carry flags. Compare finite values before AGMM, after AGMM, after
  the residual add, and after packed MLP. Record dtype/shape at each boundary.
- A more precise model-local candidate supplies a BFP8 gather-shaped
  `persistent_output_buffer` while retaining BF16 matmul output. AGMM
  `create_output_tensors` at lines 407-411 uses that tensor verbatim; factory
  activation CBs then derive BFP8 from it while output CBs remain BF16.
  The Python API documents this parameter as the gather buffer (binding
  lines 125-128). Allocate correct per-shape/per-call buffers before capture
  and preserve lifetime/ordering; this candidate needs focused eager/trace
  verification and has not been proved by this source investigation.

## MMRS is a different datatype path

`minimal_matmul_strided_reduce_scatter_async` does **not** share the observed
AGMM mismatch. Its op at `device/minimal_matmul_strided_reduce_scatter_async_op.cpp:243-269`
creates matmul output from the requested dtype and gives RS intermediate and
final output that same dtype. Its program at lines 256-263 passes the matmul
output as RS input; lines 301-309 invoke the regular minimal-matmul factory.
That factory independently derives input/output formats
(`experimental/minimal_matmul/device/minimal_matmul_program_factory.cpp:161-166`).

Thus the queued MMRS configuration with BFP8 input and BF16 output is
source-consistent, but **its communication payload remains BF16**. The model's
`ccl_dtype` cast currently quantizes the matmul input on this path. A real
BFP8 MMRS communication candidate must request BFP8 matmul output/RS and cast
the returned RS result to BF16; that changes partial-sum/reduction precision
and needs separate accuracy validation. Do not report the queued BF16-output
MMRS run as measured BFP8 CCL performance.

## Status

AGMM mismatch verified by source; runtime causality/repair checks pending with
the parent. No implementation fix or performance result is claimed here.
The core native repair would decouple activation-gather dtype from matmul
output dtype and validate raw-transfer consistency, outside this stage's
authorized native-edit scope. The model-local candidates above avoid it.

## Parent hardware follow-up

Resolution: fixed for the experimental family. `agmm_bfp8_dtype_fixed_l0.json`
passes with matching BFP8 fused intermediate/output dtype and explicit BF16
result conversion (minimum PCC0.999942, decode0.8469ms).
`mmrs_bfp8_payload_l0.json` passes with actual BFP8 RS payload (decode0.7572ms).
Both are slower than the selected BF16 row-parallel default. No dtype-family
rejection relies on the original NaN failure.
