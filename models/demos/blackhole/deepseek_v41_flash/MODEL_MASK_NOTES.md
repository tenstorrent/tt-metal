# Grammar masks in the speculative verify round (model side, M1 / M2)

Switch: `DSV41_STRUCTURED_SPEC=1` (alias `DSV41_STRUCT_MASK=1`), default OFF. Off = no buffer, no extra trace, no behaviour change.
Plugin contract: `STRUCTURED_SPEC_INTERFACE.md` (`grammar_bitmask` int32 `[W, n_plugin, 4040]` on the verify call, `permute_to_runner` in `tt/grammar_mask.py`).

## Where the mask acts
Only the sampled verify round (traces A / S / B, `tt/spec_decoder.py`). Trace A (verify -> fp32 logits `[1,1,T,16160]` per device, vocab sharded over the 8 mesh columns) is untouched, so
the hidden taps, the KV / compressor commit inputs and the drafter inputs do not depend on the mask. Trace S (sampler) adds the bias to the logits ONCE, before the sampler:
`logits + bias` then `samp.run` (shared top-32-per-column sampler, or its exact fp32 greedy helper for greedy rows). So top-k / top-p / greedy argmax only ever see allowed tokens
(an allowed token far outside the unmasked top-32 wins when it is the best allowed one). Trace B (accept / commit / draft) reads `a_in` as today.
Rule "draft accepted iff it equals the sample" makes the mask exact for M2 as well: a draft that violates the grammar can never equal a masked sample.

A round that carries a mask always runs the split A / S / B round (greedy rows: greedy parameters, `sample_cfg` of all greedy rows = exact masked argmax). A round without a mask is
byte-for-byte today's path (single trace when no sampled row, otherwise A / S / B with the unmasked S).

## Buffers (allocated in `SpecDecoder.alloc_sampled`, i.e. at runner build, before ANY trace capture; one set per SpecRunner: adaptive k siblings and decode buckets each own theirs)
* `mask_buf`: int32 ROW_MAJOR `[rows * T, 4040]`, `ShardTensor2dMesh(dims=(0, 1))`, T = U * n block rows of a mesh row, row `r * n + j` = user `r`, block position `j` (the layout of `a_in` / `ctrl`).
  Device (mesh row i, column c) holds `[T, 505]` words: the 16160 tokens of its logits shard (129,280 = 8 * 16,160 = 8 * 505 * 32, no word straddles a shard). Initial value all ones.
* `mask_shifts`: int32 `[32]` = arange(32), replicated.
Memory: 2 KB per block row per device, negligible. Upload per masked round: `rows * T * 16,160 B` (about 1.6 MB at B=16 / n=6) via `copy_host_to_device_tensor` into the SAME buffer (trace-safe).

## Ops (`SpecDecoder.mask_bias`, composed from existing ttnn ops, no C++)
`reshape [T,505,1]` -> `bitwise_right_shift(arange(32))` -> `bitwise_and(1)` -> `reshape [1,1,T,16160]` -> TILE -> fp32 -> `* 1e30 - 1e30` (allowed: exactly +0.0, so allowed logits are
bit-identical; banned: -1e30, finite, so the sampler's bf16 candidates / exp never see inf or NaN) -> `add` to the fp32 logits. The upstream `TTSampling.enable_device_grammar` is not used
(refuses `sampling_dp != 1`; our adapter uses `sampling_dp = rows`).

## Traces
Per position bucket a second sampler trace `tid_sm` (the same body behind the bias) is captured next to `tid_s`, right after it (it reads the same bucket's trace-A logits). Unmasked
rounds replay `tid_s` and never pay the unpack. The compile pass (`prepare`) runs the masked sampler eagerly once before any capture. The mask buffer is only read by `tid_sm`;
an unmasked round leaves the (stale) contents alone, a masked round overwrites all of it.

## Host (`generator_vllm._spec_verify` / `SpecRunner._round`)
`grammar_bitmask` -> `_grammar_for_runner` -> `permute_to_runner(block, [(logical, runner row)], B, n, kv=n-1, column0_only=not M2)` (same `rr` pairs as `X[rr] = tokens[ii]`, bucket runners and
the adaptive slice to the first k+1 columns included) -> `runner.grammar_packed` -> `_round`: upload, replay A, `tid_sm`, B. No restricted row (all column-0 words all ones) = None = unmasked round.
M1: a restricted row must have `num_valid_drafts == 0` (checked, ValueError otherwise); the existing nv = 0 path forces m = 0, so the committed token is the masked sample of column 0.
M2 (`DSV41_STRUCTURED_M2=1`): per-position columns (grammar state after j drafts), restricted rows may carry drafts; the drafter stays unconstrained.
Capability: `model_capabilities["supports_verify_grammar_bitmask"]`, instance `verify_grammar_enabled` (every runner has a mask buffer).

## Other paths
* Plain (non-spec) decode and prefill's first token: unchanged. The plugin (M0) samples those on the host from full logits with `apply_grammar_bitmask`. The shared plain sampler is never touched.
* Prefix cache / APC: untouched.

## Tests
CPU: `tests/test_struct_mask_model_cpu.py` (shard layout vs logits columns, bias arithmetic, masked argmax, row layout, adaptive slice), `tests/test_grammar_mask_cpu.py` (plugin agent).
Device: `/mnt/tt-data/ssinghal/dsv4-logs/structured_model/cmds/` (c10 greedy: all-ones = unmasked, random masks vs host reference computed from trace A's logits, allowed token outside the
unmasked top-32, non-structured rows unchanged, k=5 / k=3; c20 sampled: membership + law vs softmax over the allowed set).

## Device results (rc2 image, 2026-10-31, hosts .34)
* Model level (adapter harness, B=16 / k=5): all-ones mask == unmasked; random masks == host argmax over the allowed tokens of trace A's fp32 logits; an allowed token far outside the unmasked top-32 is chosen;
  non-structured rows bit-identical; sampled rows never leave the allowed set and follow softmax over the allowed set (192 draws vs the unmasked top_k=6 control); M2: per-position masks == host reference, and the
  speculative masked decode (drafts, accept / commit on device) emits exactly the plain masked decode stream (48 tokens x 16 rows identical, 29 vs 47 rounds).
* Cost: masked sampler trace +0.50 ms (1.03 -> 1.52 ms), packed mask upload 3.4 ms (1.6 MB at B=16), host `permute_to_runner` 2.8 ms (M1) / 6.3 ms (M2) before the helper is made cheap. Rounds without a mask are unchanged.
* Server level (plugin d9d8ca1, VLLM_TT_STRUCTURED_SPEC=1 + DSV41_STRUCTURED_SPEC=1): B=16/262144 prod spec: matrix 28/28, mixed 55/56 (the failure is a max_tokens truncation of an unbounded array, not a grammar violation),
  tool_checks 28 PASS / 0 FAIL, tool_protocol_probe rc 0, hang canary 30 x (8 x 8k) with structured cells in the background clean; B=8/1M catalog default (adaptive k): 28/28, 56/56, 28 PASS, probe 0, canary clean.
