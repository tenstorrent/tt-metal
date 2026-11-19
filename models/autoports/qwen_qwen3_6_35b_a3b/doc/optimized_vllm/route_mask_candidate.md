# Opt-in inactive grouped-MoE route candidate

Implemented from `AUTODEBUG_FLOOR.md`, 2026-11-17. CPU/source work only;
**no kernel compilation, hardware correctness or performance claim**.

Toggle: `QwenModel(..., suppress_inactive_moe_routes=True)`. Default isFalse.
The adapter is unchanged and therefore still constructs the control unless
the stage owner explicitly enables the candidate there after validation.
The toggle affects masked multirow decode only; prefill and singleton decode
retain the original execution. No native shared kernel or ACK drain changed.

Changed model-owned files:

- `tt/model.py`: validates/stores the construction flag and passes the
  persistent active vector plus physical row-to-slot mapping to each layer.
- `tt/generator.py`: supplies its existing active_mask to model decode.
- `tt/multichip_decoder.py`: propagates route activity through MoE padding;
  masks IDs once before count scatter and reuses them for dispatch and
  post-combine. Requires local expert dispatch and deterministic grouped decode.
- New `tt/moe_route_mask.py`: one-reader byte-preserving route mask helper.

Callable helper:

```python
mask_inactive_routes(ids, active_mask, dispatch_table, slots, mesh)
```

`ids` is UINT32 TILE logical[1,1,32,8], padded[1,1,32,32]. `active_mask` is the
existing UINT32 ROW_MAJOR vector32. `dispatch_table` is the model's rank-local
INT32 ROW_MAJOR[1,256] `ep_reduce_table`; the tensor carries each rank's
different local ownership. `slots` contains2..32 unique physical indices
mapping the unpadded rows into the activity vector. Inputs require interleaved
DRAM. Output has identical geometry/dtype/layout; no cast or layout conversion
occurs inside the helper.

The kernel reads the whole route tile, activity vector and rank-local table
from device during replay. Rank0 chooses valid off-rank expert64; ranks1–3
choose expert0. Both have `table[id]==-1` on the respective rank. It changes
only the eight logical route words of inactive/padded rows. Tile padding and
every active ID bit remain untouched. A row beyond `len(slots)` is padding
even if a different live physical slot has the same numerical row index.
All32-active therefore copies the complete tile identically. No invalid
negative expert index is passed to scatter or native indexing.

Count scatter observes the masked IDs; the existing rank-local expert mask
zeros off-rank counts. Dispatch and post-combine receive the very same IDs,
so native `table==-1` skip semantics agree with counts. Scores, top-k ordering
of active rows, precision, expert kernel/grid/output buffers, shared experts,
residuals, and conditional cache/state commits are unchanged. The native
unconditional reader ACK drain remains untouched.

The helper allocates only its temporary output tile while preparing/capturing
the model graph. No new persistent per-layer tensor, host refresh or per-token
Python call is added to replay. Existing generator two-phase preparation warms
every configured bucket before first capture, and existing activity refresh
updates the same tensor read by the trace. The transient output is consumed by
the model trace; it needs no broad allocation exemption. Tracking1/skip0 must
verify this reasoning on actual hardware.

Validation performed: source `pre-commit run --files` for the four source
files exited0; all applicable hooks passed. No TTNN import or device job was
run. Hardware validation must establish:

1. Helper all-rank exactness for all-active32, all-inactive, holes, non-prefix
   row-to-slot mapping, low/high slots and padded B2/B4; inactive IDs are
   in0..255 and off-rank, while active IDs and padding bits are identical.
   Change persistent activity and input IDs between replays with no recapture.
2. Real layers0/3 candidate versus unchanged control, exact active logits,
   tokens, all-chip recurrent/conv/KV state, dormant owners, B2/B4/B32 and
   transitions, greedy/native32/native64 sampling, two output leases, changed
   and unchanged tables. Retain the tracker positive control.
3. All40 exact source-matched controls and real async plugin lifecycle after
   promotion; check zero host fallback, default precision/context and trace
   retention. Same-harness primary/CI timing must determine actual benefit.

Remaining numerical uncertainty: removing inactive rows changes expert counts,
packed row offsets and potentially grouped intermediate representation even
with stochastic rounding disabled. Preserving active IDs and kernel arguments
does not itself prove exact active logits. An exactness failure needs a focused
first-divergence investigation; it is not acceptable to relax the reference or
claim the candidate preserves behavior solely from source inspection.

## Actual router dtype correction

The owner's first direct helper probe passed its six UINT16 fixtures, but real
generator preparation failed the dtype guard before test cases ran. The first
failure artifact remains `readiness_vllm/optimized_route_probe/result.json`.
Native TopK explicitly selects UINT32 indices for FP32 router input
(`ttnn/cpp/ttnn/operations/reduction/topk/device/topk_device_operation.cpp`,
FP32 validation/`is_uint32_index_required` contract). The helper now requires
the actual UINT32 contract, preserving its dtype without a cast. Its tile
transfer is4096bytes; activity begins at4096, ownership at4224 and scratch is
5248+64bytes including alignment allowance. Tile face indices remain in
elements. The parent owns rerunning actual UINT32 direct fixtures and the
generator discriminator. No corrected hardware result is claimed here.
