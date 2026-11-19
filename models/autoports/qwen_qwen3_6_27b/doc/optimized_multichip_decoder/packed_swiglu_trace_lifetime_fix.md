# Packed SwiGLU probe lifetime correction

`packed_swiglu_smoke.log` failed allocation tracking before the first baseline
replay. Its three live allocations match the harness order: candidate output
allocation and GenericOp warmup happened after baseline capture; the first
replay-input copy also compiled CopyDeviceOperation after that capture. This
was a harness lifetime failure, not a numerical or kernel result. The log
records normal device closure.

Only `tests/probe_packed_swiglu.py` changed. For each rows/rounding group it now:

1. Builds both input references and explicitly deallocates their eager outputs.
2. Warms both replay copies, restoring the original input afterwards.
3. Allocates/retains the baseline output and warms the native chain with its
   supported `output_tensor` parameter.
4. Allocates/retains outputs and warms **all** selected implementations before
   capturing the baseline or any candidate.
5. Forbids program-cache misses across capture/replay, then releases every group
   trace before freeing its outputs. Input banks are freed after both rounding
   groups. The cache-miss setting is restored on normal and exception exits.

No allocation acknowledgement, corruptible scope, program-cache tracking skip,
helper change or model change was added. Exact comparison, changed-input replay,
conversion guards and matched timing remain in place. Output identity and
warmed cache counts are recorded in each case. Device confirmation is pending.

CPU checks: the existing 350 tile-mapping/source cases pass; Python compile and
formatting pass. Executing the **actual probe loop AST** against CPU lifetime
stand-ins covered eight cases (rows1/32, both rounding modes, both implementations):
26 allocations/frees, 12 captures, 18 cache signatures, 36 copies and 80 replays.
The stand-ins reject allocations/cache misses after a live trace or freeing a
trace-owned tensor; all checks passed and every allocation was freed. No TTNN
import or device operation was performed.

Parent rerun under the existing serialized wrapper with a fresh artifact tag:

```bash
TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
python -m models.autoports.qwen_qwen3_6_27b.tests.probe_packed_swiglu \
  --device --chips 4 --rows 1,32 --implementation both --replays 4 --repeats 100 \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/packed_swiglu_smoke_lifetime2.json
```
