# Exact decode recurrent adapter: source contract and minimal proposal

This is a source/CPU investigation, not an implemented or measured fusion. The proposed boundary is the complete public-decode `_recurrent_step`: original-layout Q/K/V/beta/decay plus the persistent FP32 state enter; FP32 core output and the updated same state allocation leave. Prefill and its chunk64 graph remain outside the change. No canonical/HF state transplant is proposed.

## Established arithmetic and storage contract

Frozen ab28's `_recurrent_step` uses Q/K FP32 `[1,1,16,128]`, V BF16 `[1,1,32,128]`, beta and exact table decay FP32 `[1,1,32]`, and state FP32 `[1,32,128,128]`. The state is2,097,152 bytes and public decode passes `round_state=True`, `output_state=state`. A logical value head h uses key/query head `h//2`. Output is FP32 `[1,1,32,128]`. State stays FP32 in storage; the next public call applies BF16 round/widen at its input boundary. Rounding the outgoing state early would change the contract.

The exact stages are:

1. Native state FP32→BF16→FP32 activation preprocessing, packed to FP32, then SFPU multiply by that head's stored decay. Pack FP32 decayed state.
2. Separate SFPU multiply `decayed*K`, pack FP32 products, then the ordered reduction below to obtain memory.
3. Native V BF16→FP32 conversion. Separate SFPU subtraction `V-memory`, pack FP32, multiply beta, pack FP32 delta.
4. Separate SFPU `K*delta`, pack FP32, then SFPU add to decayed state, pack FP32 into the persistent output-state allocation. No addcmul/FMA contraction is equivalent by assumption.
5. Native `MUL_UNARY_SFPU` scales Q by the exact FP32 scalar derived from `128**(-0.5)` and packs the preprocessed operand. Separate SFPU `new*scaledQ`, pack FP32 products, then the same ordered reduction to core output.

`binary_ng_device_operation.cpp` routes equal FP32 ADD/SUB/MUL inputs to SFPU even when their public default fast flag is true. It explicitly rejects false for FP32 ADD/SUB; the proposal does not introduce that invalid flag. `eltwise_utils_sfpu.hpp` shows that per-input activation preprocessing is a real copy→activation→FP32 pack pass consumed by the downstream binary operation. Both state rounding and Q scaling require that boundary. V widening must use native BF16 unpack/typecast/FP32 pack, not manual integer widening; the recent convolution finite-code failure demonstrated why those can differ at tiny values.

## Exact reduction association, confirmed on saved operands

The existing product layout is `[1,head32,key128,value128]`. Permuting to `[1,key128,head32,value128]` and reshaping yields `[8,16,32,128]`. `fast_reduce_nc.cpp` sorts dimensions and reduces the larger axis first: dim1 length16, then dim0 length8. Each FP32 native reduction copies its first term into destination, then applies `add_binary_tile` left-to-right. It does **not** start with zero. The16-term result is packed to FP32 between reduction stages; the final8-term result is packed again. There is no pack/unpack between terms within either native left fold.

`tests/v6_recurrent_fusion_contract_cpu.py` and `v6_recurrent_fusion_contract_cpu.json` record raw native source hashes, method AST hashes, actual capture hashes and a bounded CPU control. For both saved memory products and saved output products at positions227/738/5799/3271, the16-then8 CPU schedule matches all4,096 output FP32 codes. A single128-term left fold differs in3,352–3,515 codes per output. This validates the index/association interpretation on those operands; it does not prove general SFPU-versus-IEEE equivalence, FTZ behavior, or an unimplemented kernel.

The3271 capture originates from an earlier prefix-state diagnostic and is supplemental same-operand evidence only. The227/738/5799 captures are available without that transplant. A new acceptance test must initialize its own ordinary state and must never inject HF state. Saved states are diagnostic operands, not a production initialization policy.

## Minimal coherent full-adapter candidate

Start with **two model-local arithmetic/layout kernels around the unchanged two native `fast_reduce_nc(dims=[0,1])` calls**. Each native call contains two reductions, so this is six device operations before optional debugging, not a proposed single-dispatch claim. It changes the producers' physical placement and fuses adjacent SFPU operations while retaining the already-proven reduction implementation and association.

Use group-major product shape `[8,16,32,128]` from the beginning. A work partition `(key-group0..7,value-tile0..3)` gives32 workers. Each worker's SFPU tile has32 head rows and32 value columns. It processes its16 consecutive key indices in order. The reader gathers the original state's per-head row segments, expands K/Q/decay/beta into the required tile lanes, and uses the original BF16 V tile directly with native conversion. All32×32 independent lanes are useful; a one-head-per-worker design would waste most lanes or require another unproven within-tile reduction.

- First generic operation emits FP32 decayed state in group-major order and FP32 memory products in that same order. Its internal native round/widen, multiply, and FP32 pack/unpack boundaries match the explicit graph. It does not retain reduction accumulators across arithmetic passes.
- Unchanged native16-then8 `fast_reduce_nc` consumes the already grouped memory products. The head/key permutation and reshape materialization disappear. Its result is already `[1,1,32,128]`, suitable as one memory tile per value block.
- Second generic operation forms the native widened V, subtraction and beta product, then for each key emits the separately multiplied/added new state in the original persistent state tile layout and output products directly in group-major layout. Q scaling keeps its native FP32 preprocessing pack. The persistent state is written only after the first operation has consumed all original state, making in-place aliasing explicit.
- Unchanged native16-then8 reduction of output products produces the final `[1,1,32,128]` core directly. The existing reduction output permutation and the immediate inverse caller permutation cancel in this physical arrangement.

This retains large FP32 product tensors in the first candidate; it does not falsely claim their elimination. It removes their explicit transpose materializations, head-layout adapters and separate arithmetic dispatches. Decayed/product buffers retain the existing L1 placement and2MiB logical sizes; persistent state stays DRAM. Exact transient CB sizing and L1 coexistence must be calculated during implementation and checked on the full decoder; no capacity reduction is authorized.

A subsequent single-program design could keep16 key rows' decayed values locally, reduce16 terms, exchange8 partials, broadcast memory, update state and reduce output. That requires inter-core synchronization plus precise destination-register lifetime across arithmetic and reduction. Packing a running sum after every term is **not** automatically equivalent to the native no-pack left fold. This more complex design is deferred until the coherent six-operation adapter is correct and timed; it is not a prerequisite or claimed result.

## Component gate and timing plan

The primary oracle must AST-extract the complete native `_recurrent_step` and `_recurrent_sum` from the frozen qualified decoder and use the same input/output allocations. Compare every FP32 core and state word, with debug comparisons for rounded state, decayed values, memory products, memory, subtraction, delta, new state, scaled Q/products and output. First use the three non-transplanted actual fixtures plus zero/random states, signed zeros, BF16-midpoint and near-underflow controls. Exercise both state aliasing and separate output allocations; public integration remains restricted to `decode and cache_boundary` and the available exact decay table value. No exp fallback or cache dtype change is introduced. Each kernel must use native SFPU APIs and unpack modes, with no FPU/matmul/native-GDN substitute.

Repeated inputs evolve recurrent state; unlike the convolution history, it does not become a fixed state after three calls. Timed native/fused runs must start from identical ordinary state and execute the same warmup/capture/replay counts, comparing outputs and states at matching steps. First-versus-last values are not expected to be equal. Determinism means restarting from the same setup state and repeating the same trajectory. Resets or backup copies belong outside measured groups and are diagnostic-only. Traced groups should contain at least100 replays, count all reader/writer/adaptation costs, and retain in-place state mutation. A changing-input sequence must separately compare both trajectories.

Prior rejected native FDR/GDN and recurrent-matmul controls narrow FP32 state through FPU/TF32 or alter reduction order. Output-only native controls also lost complete-adapter latency. This proposal reuses the selected native SFPU arithmetic and reductions instead; those historical losses do not measure this new layout/dispatch design. Any implementation still needs component exactness, whole-decoder every-token PCC, repeated-run and traced speedup before selection. No device execution, production edit, frozen-helper edit, or performance claim accompanies this memo.

## Implemented isolated component, CPU preparation complete

The parent approved the six-dispatch proposal. `tests/v6_recurrent_six_stage_component.py`, SHA256 `eb7f8e37d4bfe49f338e27216472d39aece2181775d13f1d451c6bf962cec43f`, now contains two static reader/compute/writer triples and `RecurrentSixStage`. It uses32 workers, one for each16-key group/value-column tile. All FP32 arithmetic uses the corresponding native SFPU APIs. State rounding, V native BF16 unpack/widen, subtraction, beta product, update product, state add, Q scalar preprocessing and both product outputs each keep their required FP32 packing boundaries. ADD/SUB use the native default `DstRoundingMode` template, and Q uses `binop_with_scalar_tile_init`/`mul_unary_tile` with scalar bits0x3db504f3. Both native `fast_reduce_nc(dims=[0,1])` calls are unchanged. The component file has no Torch or host-conversion imports/calls and allocates no persistent tensors.

The measured/oracle harness to use is `tests/v6_recurrent_six_stage_cached_probe.py`, SHA256 `00bec4eb9de9d64a25922a2946beaedab00aa81d8860106d9c64ad150528b4c6`. CPU setup passed in `v6_recurrent_six_stage_cached_setup.json`, with source/component/kernel snapshots. It proves that the reader gather and writer scatter cover every524,288 state word exactly once and are inverses, and verifies every GQA coefficient address. It prepares actual non-transplanted227/738/5799 fixtures, ordinary random and zero states, and a signed-zero/subnormal/BF16-midpoint boundary fixture. No device execution is implied by this CPU result.

The initial unmeasured `v6_recurrent_six_stage_probe.py`/`v6_recurrent_six_stage_setup.json` are retained but superseded: that harness reconstructed the AST oracle inside every native call. The cached version constructs the uninstrumented native oracle once before all measured calls, using a preexisting native object, so AST parsing/compilation cannot inflate the native eager time. The candidate runtime/kernel is identical between harnesses.

The complete gate checks alias and separately allocated state outputs, debug and ordinary runtime paths, and optionally both `round_state` values. With the full six fixtures and `--both-round-modes`, this is48 configurations. Strict comparisons include core/state, input-state immutability for separate output, rounded and decayed state, both product tensors, memory, V widening, subtraction, delta, update, and scaled Q. All cases are recorded before failing the aggregate gate; differing tensors are preserved. A separate12-step changing-input sequence begins from an ordinary generated state and compares both live trajectories after every step.

The timing protocol compares native/fused checkpoints after identical five-call warmup, three groups of20 eager updates, trace capture, five trace warmups and three groups of100 timed replays. It checks core/state at matching checkpoints instead of expecting an evolving state to remain unchanged. After capture, it copies state into an already allocated diagnostic device backup; after timing it restores that backup outside timing and executes305 replays, checking the entire resulting core/state for deterministic restart. The backup is test-only and is not part of the component's runtime allocation contract.

### Temporary storage accounting

| Allocation | Normal | Debug |
|---|---:|---:|
| Phase1 local circular buffers per active core | 45,056 bytes (44KiB) | 49,152 bytes (48KiB) |
| Phase2 local circular buffers per active core | 63,488 bytes (62KiB) | 83,968 bytes (82KiB) |
| Active workers | 32 | 32 |
| New persistent component storage | 0 | 0 |

Normal decayed and product arrays each contain2,097,152 FP32 bytes in interleaved L1. Memory/core outputs are16,384 bytes each; the unchanged intermediate8-group native reduction is131,072 bytes. The memory-product handle is released before allocating the output-product array in normal mode. Component-local live data at the reduction boundary is bounded by two2MiB arrays plus the16KiB/128KiB reduction tensors; the decoder's other live allocations must still be included in integration capacity testing. The preexisting2MiB DRAM state is updated in place for public usage. Debug additionally retains observability buffers and is deliberately excluded from timing. No model context capacity is lowered or inferred solely from these byte counts.

Parent-owned first device command:

```bash
PYTHONPATH=. python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/tests/v6_recurrent_six_stage_cached_probe.py --run-device --cases actual227 --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_recurrent_six_stage_actual_device.json
```

After that passes, full gate plus matched timing:

```bash
PYTHONPATH=. python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/tests/v6_recurrent_six_stage_cached_probe.py --run-device --both-round-modes --time --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_recurrent_six_stage_all_device.json
```

All new files were formatted with pinned autoflake/isort/Black before hashes and setup snapshots were frozen. No production or prior frozen source was edited. Compilation, device exactness, complete-adapter speed, and integration memory fit remain explicit unverified gates.
