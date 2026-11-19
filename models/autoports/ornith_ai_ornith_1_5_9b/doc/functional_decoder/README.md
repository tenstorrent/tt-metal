# Ornith-1.5-9B functional decoder

Stage 1 implements both text decoder kinds from `ornith-ai/Ornith-1.5-9B@489cb97981b8654bcfcf30ce1f94ed1b62e07b53` on a single Blackhole chip. Accuracy, advertised-context, allocation, determinism, watcher and profiler gates pass. Independent [stage review](stage_review_round1.md) returned **clean-pass** in round 1. Local checkpoint details are recorded in the work log.

The pinned Qwen3.5 text configuration has width 4096, MLP width 12288 and 32 layers. Layers 0, 1 and 2 modulo 4 use a four-tap causal convolution and gated delta recurrence. Layers 3 modulo 4 use full attention: 16 query heads, 4 KV heads, head width 256, Q/K RMSNorm, 64-dimensional partial RoPE (theta 10000000), and a sigmoid output gate. Both kinds include offset RMSNorm, SwiGLU and both residual additions. Weights/activations are BF16, recurrence is FP32, and matmuls use HiFi4 with FP32 accumulation.

## Public contract

```python
FunctionalDecoder.from_state_dict(state_dict, *, hf_config, layer_idx,
                                  mesh_device, chunk_size=1024)
decoder.allocate_state(*, context=None)
decoder.prefill_forward(x, *, state, start_position=0, page_table=None)
decoder.decode_forward(x, *, state, page_table=None, current_position=None)
decoder.prefill_batch(rows, *, states, start_positions, page_tables)
decoder.decode_batch(rows, *, states, current_positions, page_tables)
```

`FunctionalDecoder` subclasses `LightweightModule`. `state_dict` accepts layer-local HF keys or the canonical `model.language_model.layers.<index>.` prefix. Conversion and constants are constructed only at setup. Inputs and outputs are device BF16 tensors `[1,1,S,4096]`; decode has S=1. Open a 1x1 mesh with `physical_device_ids=[0], trace_region_size=0`. Retain topology discovery on P300; do not isolate half the board through `TT_VISIBLE_DEVICES`.

Each request owns its state. Full-attention K/V caches are BF16 `[physical_blocks,4,32,256]`; the device ROW_MAJOR INT32 page table is `[1,logical_blocks]`. Its entries must address allocated physical blocks, and its unpadded logical extent must cover the request, including internal 32-token padding. State allocation context is a multiple of 32; this is a physical allocation constraint, not a logical prompt-length restriction. The caller must validate device-valued decode positions against the allocated cache/page map and the advertised context before replay. The layer does not read device metadata back to validate it. Prefill `start_position` is a nonnegative absolute host integer; decode `current_position` is a persistent device INT32 `[1]` tensor. Position denotes the next token slot, not the length after writing it. Full attention consumes both inputs for cache addressing and RoPE.

Linear-attention state is FP32 `[1,32,128,128]` recurrence plus BF16 `[1,1,3,8192]` convolution history. Position and page-table inputs are unused by this recurrent kind: history is represented by that state, and the caller advances it in order. Fresh requests need zero state; continuation requires the exact preceding prefix state. Full-attention padding writes only future unused slots and cannot be shared with another active request. Subsequent real tokens overwrite those slots before attention reads them.

Prefill chunks at 1024 tokens and returns every logical output. It accepts non-aligned lengths; full attention pads internally and trims outputs. Non-tile-aligned continuation first decodes individual tokens to reach an aligned start. Ragged batch APIs accept independent row tensors, states, starts and page maps. They run a per-row functional baseline, tested through batch 32; dense batch fusion belongs to a later stage.

## Trace and runtime contract

The entire decode forward, including cache update, recurrence, attention, MLP and residuals, is recorded and replayed. Before first capture, the caller prepares all physical shapes and prefill/continuation paths that will coexist with the trace. Input, position, page-table and state buffers stay at stable addresses. Update their contents before replay; restore state after warmup/capture. No tensor values cross to the host inside either forward.

`tests/validate.py` demonstrates one retained single-row trace across nine changed requests, 27 replays, zero request-time releases, changed activations/page permutations/positions, then a short request after longer ones. It separately captures all 32 batch rows in one trace and repeats three times. Output is acknowledged by exact tensor only: its allocation is an intentionally overwritten trace output, read before the next replay, with no other live trace consumer. No broad allocation scope and no skipped program-cache tracking are used. A positive control proves an unrelated live allocation is detected. Trace teardown at the end of a test is distinct from request-time eviction.

The runtime audit guards Torch tensor operations and TTNN host upload/readback entrypoints during each measured pass. Inspection also covers GDN constants and device slice helpers. Setup, input refresh and final PCC readback are explicit harness boundaries. Python metadata/chunk planning is allowed; host tensor computation is not.

## Accuracy and capability evidence

The acceptance threshold is PCC >=0.995 throughout; it has not been relaxed. Tests use real layer 0 and layer 3 weights with the unchanged target config. Offline pytest generates deterministic weights from saved per-tensor real-weight statistics.

| Claim | Evidence | Coverage / remaining risk |
|---|---|---|
| Both layer kinds, real weights | `linear_final.json`, `full_final.json` | Every output compared at lengths 1,31,32,33,1023,1024,1025,2051,17; includes batch 32 |
| Paged prefill and decode | `full_final.json`, `tests/validate.py` | Permuted physical pages, reconstructed K/V checked against HF, device current positions |
| Prefix continuation | `*_final.json` resume rows | Starts 32,33,1024,1025,2050; per-row residual-subtracted PCC; dropped-history controls fail the same gate |
| Chunked parity | `*_final.json` unchunked controls | Independent eager unchunked run compared with HF and chunked results |
| Advertised 262144-token prefill and decode | `linear_long.json`, `full_long_fixed.json`, `../context_contract.json` | Complete TTNN forwards at 262139,262143,262144; traced decode at zero-based position 262143; no capability reduction |
| Long-context numerical reference | `tests/long_context.py` and long JSON records | Sampled first/middle/terminal windows, not whole-output PCC. Linear HF processes every token; full HF windows attend to exact HF K/V for all preceding tokens |
| Repeated-input determinism | `*_final.json`, long JSON records | Three bit-equal replay outputs after identical state restoration; repeated prefill bit-equal before/after capture |
| Allocation safety | `trace_alloc_tracking/positive_control.json`, final run logs | Tracking enabled, program-cache exclusions disabled; deliberate unsafe allocation caught |

Minimum whole-output PCC across the final short/medium/batch batteries is 0.99968177 (linear) and 0.99966115 (full). Minimum positive continuation per-row contribution PCC is 0.99890006 and 0.99961668 respectively. Long-context sampled prefill minima are 0.99866915 and 0.99820769; final-position traced decode PCC is 0.99865657 and 0.99977314. The lowest long contribution PCC is 0.99812716. Both watcher batteries pass with `TT_METAL_WATCHER=10` and noinline instrumentation; see `watcher_audit.json`. Warmed device-kernel sums (prefill / traced decode) are 13.240 / 1.655 ms for linear attention and 8.194 / 1.841 ms for full attention at S=1024, batch 1; see `functional_decoder.md` for units, gaps and limitations. Negative-control failures are intentional and excluded from positive minima.

`full_long_fixed.json` predates a linear-attention-only GDN chunk correction; its full-attention implementation is unchanged. `linear_long.json` and `linear_final.json` are rerun after that correction. Historical 64-token GDN results have the `_gdn64` suffix. See `AUTODEBUG.md` and `AUTOFIX.md` for source identity and the correction's validation.

## Reproduction and limits

Run commands from the tt-metal root in its active `python_env`; see [work_log.md](work_log.md), `tests/run_evidence.sh`, and [functional_decoder.md](functional_decoder.md) for exact commands and performance units. Tests serialize hardware use on chip 0. Do not run watcher and profiler together.

This stage covers decoder layers, not a model stack or generated text. Long-context PCC is sampled with full history; it cannot establish numerical agreement at every long output row. The batch implementation favors clear independent state ownership over throughput. Callers must honor cache capacity, state ownership and trace preparation contracts. No optimized-decoder, multichip, full-model or serving implementation is included.

The review certifies layer behavior. Repeated replay restores prefix state; a future generator must separately validate sustained decode trajectories and its actual interleaving of resume/batch paths with retained traces. These are handoff risks, not claims of completed generation or serving validation.
