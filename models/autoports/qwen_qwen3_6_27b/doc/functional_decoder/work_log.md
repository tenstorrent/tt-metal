# Functional-decoder work log

## 2026-07-29 — architecture and hardware inventory

- Checkout base: `b9e6c242a34011e3daeebab9207fbb5b79750f39`
- Branch: `mvasiljevic/qb2/skillexp/base`
- `AutoConfig.from_pretrained("Qwen/Qwen3.6-27B",
  trust_remote_code=True)` resolved to the installed
  `transformers.models.qwen3_5` implementation.
- Official checkpoint revision:
  `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.
- Read the HF decoder, full attention, gated delta net, MLP, RMSNorm, gated
  RMSNorm, MRoPE, causal convolution, chunk rule, recurrent rule, and cache
  paths line by line.
- `timeout 60 tt-smi -ls --local`: pass; four Blackhole p300c devices visible.
- `timeout 120 python <1x1 open/close smoke>`: pass; mesh `(1, 1)`,
  `Arch.BLACKHOLE`, `trace_region_size=0`.
- No reset was required. The mesh command closed normally. Nanobind printed
  shutdown reference-leak diagnostics after `MESH_SMOKE_OK`; this did not
  affect device open/close.
- The pre-existing untracked `.agents/prompts/alchemy_later/` and
  `models/autoports/openai_gpt_oss_20b/doc/optimized_decoder/...` artifacts
  are unrelated and will not be modified.

### Canonical real-weight shard mapping

Layer 0 (`linear_attention`) tensors occupy official checkpoint shards 1, 6,
and 8. Layer 3 (`full_attention`) tensors occupy shards 4, 6, 7, and 8.
The exact mapping was read from the official
`model.safetensors.index.json`; real-weight evidence must load only the
required layer keys from those shards rather than materializing the full
causal LM.

### Open gates

All PCC, trace, capacity, watcher, determinism, fallback-audit, performance,
and stage-review gates remain open. No capability reduction has been made.

### Initial contract test

Command:

```text
pytest -q models/autoports/qwen_qwen3_6_27b/tests/test_functional_decoder.py
```

Result: 3 passed in 1.23s. This proves only the immutable real HF config,
layer-kind counts, advertised-context metadata, and canonical checkpoint
prefix. It is not decoder correctness evidence.

## 2026-07-29 — traced decode token-mixer bring-up

Implemented both real-shape decode token mixers with TTNN-only runtime
operations:

- full attention: gated Q projection, Q/K per-head RMSNorm, device-indexed
  partial MRoPE, paged KV update, paged decode SDPA, decode head concat, gate,
  and output projection;
- linear attention: depthwise causal-convolution state update, Q/K L2 norm,
  recurrent gated-delta update, gated RMSNorm, and output projection.

The only sharded layout is the minimal layout required by paged cache update,
decode SDPA, and decode head concat: one height shard per batch row. Blackhole's
11x10 grid required a rectangular workload factorization for head concat:
1x1 at batch 1 and 8x4 at batch 32. No matmul program config, tuned per-core
grid, or compute-kernel override was added.

Commands:

```text
python models/autoports/qwen_qwen3_6_27b/tests/full_attention_decode_smoke.py --batch 1
python models/autoports/qwen_qwen3_6_27b/tests/full_attention_decode_smoke.py --batch 32
python models/autoports/qwen_qwen3_6_27b/tests/linear_attention_decode_smoke.py --batch 1
python models/autoports/qwen_qwen3_6_27b/tests/linear_attention_decode_smoke.py --batch 32
```

All four commands passed eager execution, trace capture, one trace replay,
output shape, and exact repeated zero-output checks using target dimensions.
The full-attention runs used distinct paged-cache blocks for all batch rows.
These are zero-weight token-mixer shape/trace smokes only: they do not prove
nonzero numerical correctness, the common MLP/residual path, latency, or the
final traced-decode acceptance gate.

## 2026-07-29 — full-layer trace, prefill, PCC, and real weights

Full-layer trace capture/replay passed through both residuals, both RMSNorms,
the token mixer, and the 17,408-wide SwiGLU for both layer kinds at batch 1 and
batch 32:

```text
python models/autoports/qwen_qwen3_6_27b/tests/full_attention_decode_smoke.py --batch 1 --full-layer
python models/autoports/qwen_qwen3_6_27b/tests/full_attention_decode_smoke.py --batch 32 --full-layer
python models/autoports/qwen_qwen3_6_27b/tests/linear_attention_decode_smoke.py --batch 1 --full-layer
python models/autoports/qwen_qwen3_6_27b/tests/linear_attention_decode_smoke.py --batch 32 --full-layer
```

These use full-shape zero weights and prove structural traceability only.

Prefill smokes:

```text
python models/autoports/qwen_qwen3_6_27b/tests/full_attention_decode_smoke.py --mode prefill --batch 1 --sequence 32
python models/autoports/qwen_qwen3_6_27b/tests/full_attention_decode_smoke.py --mode prefill --batch 1 --sequence 33
python models/autoports/qwen_qwen3_6_27b/tests/linear_attention_decode_smoke.py --mode prefill --batch 1 --sequence 4
```

All passed and returned the requested logical sequence. Full attention filled
the paged KV cache. Linear attention used the exact recurrent token rule.

Nonzero full-shape numerical results through `from_state_dict`:

| Kind/mode | Sequence | PCC |
|---|---:|---:|
| full attention decode | 1 | 0.9999608149363718 |
| full attention paged prefill | 33 | 0.9997293846372911 |
| linear attention decode | 1 | 0.9999979564622733 |
| linear attention prefill | 5 | 0.9999980573642917 |

Commands were the corresponding `*_synthetic_pcc.py` scripts with
`--mode decode` or the prefill sequence shown above.

Downloaded only official checkpoint shards 1, 6, and 8 for representative
linear-attention layer 0:

```text
snapshot_download("Qwen/Qwen3.6-27B", revision="6a9e13...", allow_patterns=[1,6,8])
```

Real-weight command:

```text
python models/autoports/qwen_qwen3_6_27b/tests/linear_attention_real_pcc.py
```

Result: PCC `0.9999218584509479`, passing the default 0.995 bar. See
`real_weight_decode.log`. The test loads only layer-0 keys using
`safetensors.safe_open`, constructs HF on meta, and transfers the same canonical
state through TTNN `from_state_dict`.

## 2026-07-29 — trace, profiler, watcher, and context closure

The full-layer trace harness now uses deterministic nonzero BF16 inputs.
Replay output exactly equals eager output and has PCC 0.99999994–1.0 for both
layer kinds at batch 1 and 32. Ten-replay median host timings were:

| Kind | Batch 1 | Batch 32 |
|---|---:|---:|
| full attention | 2.450561 ms | 2.652966 ms |
| linear attention | 3.141318 ms | 21.486593 ms |

Tracy commands used `python -m tracy -r -p -v -o <artifact-dir>` with the
corresponding smoke script. `tt-perf-report` filtered `PERF_DECODE` /
`PERF_DECODE_END` or `PERF_PREFILL` / `PERF_PREFILL_END`; every directory under
`tracy/` retains `*_ops.csv`, filtered report CSV, human-readable report, and
console provenance. Warmed full-layer prefill was 3.729979 ms at full-attention
sequence 33 and 11.629245 ms at linear-attention sequence 5.

Independent watcher commands used:

```text
TT_METAL_WATCHER=2 TT_METAL_WATCHER_NOINLINE=1 \
TT_METAL_WATCHER_DISABLE_ETH=1 TT_METAL_LOGS_PATH=<unique-dir> \
python <decode-smoke> --batch 32 --full-layer --iterations 2
```

Both representative kinds passed traced replay. Their watcher logs contain no
fatal, exception, assert, invalid-NoC, CB/L1 out-of-bounds, stack-overflow, or
sanitize signature.

Full attention now selects framework-default paged chunked SDPA above 32,768
tokens. No program config, compute-kernel config, or grid was added. Device
capacity probes:

| Logical prefill | Result |
|---:|---|
| 32,769 | pass (first non-aligned chunked length) |
| 131,071 | pass |
| 163,839 | pass |
| 180,223 | pass |
| 188,415 | pass |
| 192,511 | pass |
| 194,559 | hard DRAM OOM |
| 196,607 | hard DRAM OOM |
| 262,143 | hard DRAM OOM |

At 194,559 the allocator requested 2,390,753,280 bytes, or 298,844,160 bytes
per bank, while the largest free block was 283,529,088 bytes per bank. Decode
at position 262,143 passes with the full 262,144-token batch-1 paged cache.
Linear decode also passes at position 262,143; its state size is independent of
position. Linear prefill passed 32/33 and 64/65 boundaries. A reversed two-page
page table at full-attention decode position 65 passed traced replay.

## 2026-07-29 — AutoFix after independent review

The first independent stage review returned `more-work-needed`: zero-weight
trace smokes reduced the full layer to an identity residual and could not prove
cache/state semantics. `$autofix` used `AUTODEBUG.md` to replace that evidence.

`tests/traced_synthetic_pcc.py` captures a nonzero synthetic decoder after an
eager compile, restores mutable cache/state, then copies new hidden and
current-position values into stable device buffers for two sequential replays.
HF advances the matching cache/state. Results:

| Kind | Batch | Step 1 PCC | Step 2 PCC |
|---|---:|---:|---:|
| full attention | 1 | 0.998809274 | 0.999285760 |
| full attention | 32 | 0.999506372 | 0.999703475 |
| linear attention | 1 | 0.999987516 | 0.999988071 |
| linear attention | 32 | 0.999967999 | 0.999990846 |

Batch-32 tests use row-distinct inputs and assert no row aliasing. The same
meaningful paths produced `tracy_nonzero/` reports and watcher-clean runs with
`TT_METAL_WATCHER=10`.

`tests/full_attention_cache_pcc.py` uses one decoder instance for nonzero
sequence-65 prefill and position-65 decode with page table `[[1,0]]`. Decode
PCC is 0.999905286 and physical-cache assertions prove logical pages landed in
the permuted physical blocks.

This exposed and fixed a real cache-allocation bug: the old
`ceil(batch*context/page)` formula underallocated non-aligned contexts.
Allocation is now `batch*ceil(context/page)` (batch32/context65 is 64 blocks,
not 33).

Linear prefill formerly retained one TTNN output per token. The balanced
binary concat reducer bounded references but did not remove the one-decode-
graph-per-token dispatch bottleneck: target-shape sequence 32,769 took
120.358 seconds and sequence 262,143 remained CPU-bound after roughly 36
minutes.

`$autofix` verified that TTNN's SSM `prefix_scan` cannot represent gated
delta's dense matrix transition. A focused alternative expresses each token
as the affine transform `R' = A R + B`, then composes all transforms in a
64-token chunk with a six-level Hillis-Steele batched-matmul scan. Focused
probe results were PCC 0.999918401 at 2 groups / width 32 and PCC 0.999763906
at the target 48 groups / width 128. The runtime now also vectorizes the
four-tap causal convolution over each chunk and carries only the bounded
convolution/recurrent states between chunks.

Full-layer HF-vs-TTNN prefill passes at logical sequence 5 with PCC
0.999998050 and at the 64+1 boundary with PCC 0.999997842. Both set
`ttnn.CONFIG.throw_exception_on_fallback = True`. The same target-shape
sequence-32,769 capacity probe now passes in 85.899 seconds (versus 120.358
seconds), with nonzero output and recurrent state. Non-aligned tails remain
logical TTNN shapes rather than artificial tokens, so persistent state advances
by exactly the requested sequence length.

The public full-layer target-shape sequence-192,511 probe passes in 474.957
seconds with 985,656,229 nonzero output elements and all 131,072 recurrent
state elements nonzero. Sequence 262,143 reaches the hard MLP DRAM boundary
after 652.147 seconds: a 9,126,805,504-byte allocation needs 1,140,850,688
bytes per bank, while the largest free block is 856,953,216 bytes per bank.
These artifacts are `linear_prefill_target_192511_chunked.log` and
`linear_prefill_target_262143_chunked.log`.

The final page-routing oracle checks both key and value caches at tile-aligned
discriminating slots: physical block 1 slot 63 is populated and physical block
0 slot 63 is zero for page table `[[1,0]]`. Identity/ignored routing fails this
control. See `autofix_trace_cache/full_cache_prefill_decode_discriminating.log`.

Final watcher evidence uses `TT_METAL_WATCHER=10` under `watcher10/`; the
earlier interval-2 runs above are retained only as historical bring-up logs.

## 2026-07-29 — final independent review

A fresh read-only `$stage-review` inspected the implementation, raw
correctness/capacity/watcher/profiler artifacts, capability contract, and both
AutoFix rounds. Verdict: `clean-pass`, with no required work or hard-check
gaps. Residual risk is long-prefill speed, explicitly deferred to the optimize
stage. The recorded review is `stage_review.md`.

Local checkpoint (never pushed):

- repo: `/home/mvasiljevic/tt-metal`
- branch: `mvasiljevic/qb2/skillexp/base`
- implementation commit: `8e09249fae8012e98f437446f9a5a8f48174ede6`
- evidence commit: `ef4ae2d1b43`

Two fallback-audit runs set `ttnn.CONFIG.throw_exception_on_fallback = True`;
both nonzero traced layer kinds pass. Hardware recovery during AutoFix reset
only the exact failing devices 3 then 2 after a stale fatal and ERISC heartbeat
failure; a subsequent 1x1 mesh smoke passed on device 2.
