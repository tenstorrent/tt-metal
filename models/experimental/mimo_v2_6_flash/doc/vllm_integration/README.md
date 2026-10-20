# MiMo-V2.6-Flash vLLM serving on Blackhole Galaxy

`tt/generator_vllm.py` (`MiMoV2ForCausalLM`) serves XiaomiMiMo/MiMo-V2.6-Flash-MOPD through
[vllm-tt-plugin](https://github.com/tenstorrent/vllm-tt-plugin) (tracking issue tenstorrent/vllm-tt-plugin#216).
`--data-parallel-size 4` runs as four in-process lanes, one TP=8 replica per Galaxy row (`MiMoGalaxyGenerator`).
Decode replays one trace per replica; greedy rows sample on device with async decode, other sampling runs on the host.
Short first chunks prefill in traced batches; longer chunks and later chunks run eagerly.

The tables below up to "Not done" were measured at 52f85d31b3e; the section "Throughput at concurrency 16" describes
the branch `mimo-v2.6/vllm-c16` built on it.

Measured 2026-10-20 on one BH Galaxy, tt-metal `mimo-v2.6/vllm` at 52f85d31b3e (the merge of
`mimo-v2.6/vllm-prefill-trace`, `mimo-v2.6/slot-parity` and `mimo-v2.6/moe-prefill`, tenstorrent/tt-metal#61462),
plugin `mimo-v2.6-flash/vllm-plugin-dev` at 11dc5a9, vLLM 0.30.0, MOPD weights (48 layers, BFP4 experts, BFP8 KV).
The branch changes `chunked_flash_mla_prefill` (C++), so it needs its own ttnn build. Rows marked (ee6895badd4) were
measured before the merges, at that commit with plugin 47cc860.

| Check | Result |
|---|---|
| Plugin HTTP suite `tests/tt` (no spec decode), 32K, 4 lanes x 4 users | 77 passed, 1 skipped, 1 failed: `test_bad_words` (the model stops inside its reasoning when the banned word is the natural next token). The 13 seed / batch reproducibility tests that failed at ee6895badd4 (64 passed, 14 failed) pass |
| Chunked prefill (`--max-num-batched-tokens 2048`) | the three needle-recall tests pass: one split prompt, a 3x-budget prompt, and four prompts sharing steps with ragged chunk boundaries |
| Prefix caching (ee6895badd4) | a second request on a cached 5.3K-token prefix prefills in 2.1 s instead of 31.8 s and recalls a passphrase from inside the prefix |
| Greedy vs the standalone generator, 16 chat prompts (15-552 tokens), 32 tokens | 16/16 identical (32K server; at ee6895badd4: 15/16, and 14/16 on the 1M server) |
| Decode-trace recaptures | 0 over startup, the HTTP suite, the greedy run and the benchmarks (36 at ee6895badd4) |
| `test_model_tp8_dp4` (48 layers, fp32 golden) | top-1 0.925 (37/40), logits PCC min 0.9581 / mean 0.9922; each merged change alone gives 0.950 |
| `test_model_traced_tp8_dp4` | traced == eager (PCC 1.000000) |
| Qualitative, 6 readiness prompts, greedy and sampled (t=1.0, top_p=0.95) (ee6895badd4) | coherent and correct; the `mimo` reasoning parser separates reasoning from content |
| Tool calling (`--tool-call-parser mimo --enable-auto-tool-choice`) | `get_weather({"city": "Paris", "unit": "celsius"})`, `finish_reason=tool_calls` |
| `--max-model-len 1048576` (ee6895badd4) | serves; KV pool 1.27M tokens per lane (1.21x one 1M request); a 20,936-token needle prompt recalls correctly (87 s, mostly prefill) |

## Serving performance (host sampling, greedy, `vllm bench serve`, random 128 in / 128 out, `--ignore-eos`)

| Concurrency (prompts) | TTFT median / p99 | TPOT mean / p99 | ITL median / p99 | t/s/user | Output tok/s |
|---|---|---|---|---|---|
| 1 (4) | 274 / 281 ms | 23.6 / 24.0 ms | 23.6 / 28.9 ms | 42.4 | 39.1 |
| 16 (64) | 3608 / 3654 ms | 34.4 / 59.6 ms | 32.4 / 42.9 ms | 29.1 | 263.4 |
| 1 (4), ee6895badd4 | 506 / 1795 ms | 23.9 / 24.2 ms | 23.8 / 27.4 ms | 41.9 | 33.1 |
| 16 (64), ee6895badd4 | 4217 / 5762 ms | 34.6 / 66.5 ms | 32.0 / 42.0 ms | 28.9 | 233.7 |

32K server, after a 16-prompt warmup run. At ee6895badd4 the 1M server's TPOT was 23.3 ms at concurrency 1 and
49.4 ms at 16 (wider page tables, below). The traced device step is 13.5 ms (1 row) / 15.1 ms (4 rows) with the
on-device greedy loop (`tests/perf/decode_bandwidth.py`); serving adds host sampling (logit rows read back every
step), per-step input copies and page-table bookkeeping. The expert-major MoE prefill halves TTFT at concurrency 1;
the p99 drop is the recapture that the first request used to pay. At concurrency 16, TTFT is still about 13x the
single-request value (not investigated).

## Throughput at concurrency 16 (branch `mimo-v2.6/vllm-c16`)

Same benchmark (random 128 in / 128 out, `--ignore-eos`, greedy, 64 prompts at concurrency 16 and 4 at concurrency 1,
DP=4 x 4 seqs, 32K server), plugin `mimo-v2.6-flash/c16-throughput`. Each step below adds to the one before it.

| Step | c=16 output tok/s | c=16 TTFT median | c=16 TPOT mean / ITL median | c=1 TTFT / TPOT |
|---|---|---|---|---|
| merged branch (52f85d31b3e) | 263.4 | 3608 ms | 34.4 / 32.4 ms | 274 / 23.6 ms |
| device greedy + async decode, batched first-chunk prefill | 644.5 | 1206 ms | 15.1 / 13.2 ms | 196 / 13.1 ms |
| traced prefill batches, decode RoPE from device tables | 897.5 | 613 ms | 13.2 / 13.0 ms | 162 / 13.0 ms |
| one logits row per user read back, plugin `prefill_gather_ms` 50 | 992.1 | 323 ms | 13.3 / 13.0 ms | 202 / 13.0 ms |
| `prefill_gather_ms` 200 + `prefill_gather_quiet_ms` 25, four runs on two servers | 1009.6, 1024.2, 1012.0, 1001.6 | 306-324 ms | 13.2-13.6 / 13.0 ms | 175 / 13.0 ms |

The rows from the second on ran with the other sessions' host-only CPU jobs paused (`vllm bench` processes trigger
the pause); with them running, the host side of every step slows down and c=16 fell to 351-561 tok/s.

What changed:

- **Decode** (`decode_input_update_contract` 1, `supports_async_decode`, `supports_sample_on_device` with
  `max_device_top_k` 1, no device penalties or logprobs): the vLLM-mode trace takes its greedy token from GreedyHead
  (the lowest id on ties, like the host's argmax) and writes it into the next step's token input, so a steady step is
  one trace replay per lane with no copies (the RoPE cos / sin come from device tables at the trace's own positions,
  `setup_decode_rope_tables`, up to 128K context) and a non-blocking token read. Any non-greedy row, penalties,
  logprobs (plugin capability `supports_device_logprobs`, new) or structured output send that step to host sampling,
  which reloads the inputs.
- **Prefill batches**: a lane's first chunks of one padded length (<= 512) run as one pass
  (`TtMiMoModel.prefill_enqueue_batch`), replaying a trace per (users, padded length) captured after the decode trace
  (28 per replica, `trace_region_size` 1 GB), and read back one logits row per user. A short prompt with a prefix-cache
  hit is recomputed from position 0 in such a pass without writing its cached blocks. A pass of 4 x 128 tokens on all
  4 lanes takes ~245 ms (device bound) against ~880 ms eagerly.
- **Row invariance**: a user's prefill is bitwise the same alone or in any batch, which vLLM's seed / batch tests
  need. `ttnn.all_reduce` and the auto-configured router linear are not row-invariant, so a batch reduces as
  [users, 1, padded, hidden] (equal to each user's own reduce for padded >= 96, checked on device) or with a
  chip-ordered fp32 sum (padded 32 / 64), the prefill router uses a fixed-K-block program config, and prefill never
  takes the 32-row decode kernels (RMSNorm, layer 0's MLP, router).
- **Plugin `prefill_gather_ms`** (lane-DP, default off): while no lane runs a request, a prefill step waits until the
  oldest waiting request is that old, every slot has a request, or no request arrived for `prefill_gather_quiet_ms`
  (default 10). A wave of 16 requests reaching an idle engine prefills in one step instead of two.

Remaining time at c=16 (8.00-8.18 s for 64 x 128 tokens): 4 waves of 127 decode steps at 13.0 ms (6.6 s), one ~245 ms
prefill step per wave, and ~0.1 s per wave between a wave's last token and the next requests reaching the engine.
The first wave of a run can still split when the client ramps up more than 25 ms apart.

Validation on this branch (48 layers, with the launch settings below): HTTP suite 78 passed, 1 skipped, 0 failed; greedy vs the standalone generator
16/16 (each prompt with its own `cache_salt`: the four passage prompts share a prefix, and a prefix-cache hit on a
long prompt computes a continuation chunk instead of the standalone's one-shot prefill); 0 decode-trace recaptures. In the smoke driver
(tracked allocations, program-cache misses forbidden after the capture): batched and traced prefill bitwise equal to
per-user eager prefill, device greedy equal to host greedy over 5 exact bf16 top-2 ties, device RoPE equal to host
RoPE. The prefill numerics differ from 52f85d31b3e by the prefill router and the 32-row kernels (standalone greedy
11/16 prompts identical over 32 tokens to the 52f85d31b3e standalone); decode numerics are unchanged.
`test_model_tp8_dp4` on this branch: top-1 0.950, logits PCC min 0.9611 / mean 0.9923 (52f85d31b3e: 0.925, 0.9581 /
0.9922).

## Launch

```bash
export TT_METAL_HOME=<tt-metal build> PYTHONPATH=<this tt-metal checkout> MESH_DEVICE="(4,8)" HF_HUB_OFFLINE=1
cd <this tt-metal checkout>   # the model's kernel sources resolve against the working directory first
vllm serve /path/to/MiMo-V2.6-Flash-MOPD --served-model-name XiaomiMiMo/MiMo-V2.6-Flash-MOPD --trust-remote-code \
  --data-parallel-size 4 --max-num-seqs 4 --max-model-len 32768 --block-size 64 \
  --enable-chunked-prefill --max-num-batched-tokens 2048 --enable-prefix-caching \
  --reasoning-parser mimo --tool-call-parser mimo --enable-auto-tool-choice \
  --additional-config '{"tt": {"trace_mode": "decode_only", "trace_region_size": 1000000000, "l1_small_size": 65536, "dispatch_core_axis": "col", "sample_on_device_mode": "decode_only", "prefill_gather_ms": 200, "prefill_gather_quiet_ms": 25}}'
```

- `trace_region_size` 1 GB holds the decode trace and the 28 prefill-batch traces per replica (150 MB is enough with
  `MIMO_PREFILL_TRACE=0`). `sample_on_device_mode` and the prefill gather need plugin `mimo-v2.6-flash/c16-throughput`.

- `--max-num-seqs` is per lane (16 users in total). `--block-size 64` is required.
- vLLM starts one API server per DP rank (4) by default; that needs plugin 11dc5a9 or later (before it, every
  extra API server rejected the plugin's structured-output setting and `--api-server-count 1` was required).
  Checked: 24 concurrent completions, a chat and 4 JSON-schema requests over the 4 API servers all succeed.
- `MIMO_KV_GB_PER_CHIP` (default 6) caps the KV pool per lane; `MIMO_MODEL_PATH` / `MIMO_TT_CACHE` as for the tests.
- Clients on this host should use `127.0.0.1`: it has no `localhost` entry in /etc/hosts, and the async OpenAI
  client times out resolving it.

## KV cache

`get_kv_cache_spec` reports a FullAttentionSpec for the 9 global layers and a SlidingWindowSpec (window 128) for the
39 SWA layers, both K 192 / V 128, with the global page padded to the SWA page so vLLM keeps block size 64. vLLM
groups them into 6 KV cache groups (1 global, 5 SWA of up to 9 layers), each with its own block table, all drawing
block ids from one pool per lane. The model keeps one device page table per group
(`TtMiMoModel.setup_vllm_kv_cache` / `set_vllm_page_tables`, copied only when a row changes) and reads a later SWA
chunk's window from vLLM's SlidingWindowSpec table. Buffers are shared the way vLLM shares them (the i-th layer of
every group): K for all layers, V among layers of the same width; 4.7 KB per token per chip.

Under vLLM each global layer keeps both copies of its KV head on a chip pair (the standalone model splits users
between the pair by slot parity): a prefix-cached block can be reused by a request in a slot of either parity.
Decode still splits the attention by slot. The cost is 1M x 4 users no longer fitting at once; the 6 GB pool
holds about 1.27M tokens per lane.

## Trace safety

Prefill runs between decode replays, so anything it leaves allocated after a decode trace capture can be
overwritten by the replay (tech_reports/AdvancedPerformanceOptimizationsForModels/TraceCorrectness.md):

- the MoE routing zeros are filled on device per call (a per-length cache made while serving was corrupted);
- the prefill RMSNorm reads a tiled weight, so it can fall back to its streaming kernel once decode's persistent L1
  buffers exist (otherwise its circular buffers clash with them);
- `warmup_model_prefill` compiles every prefill program a served chunk can run before the decode capture, with an
  idle eager decode step first so decode's persistent L1 buffers are in place (some program keys depend on the free
  L1). The programs key on the chunk's padded length and start alignment only (`chunked_flash_mla_prefill` takes the
  chunk start as a device tensor), so the warmup runs `prefill_warmup_chunks` through one layer of each kind: 62
  chunks and 902 programs per replica at `--max-num-batched-tokens 2048`, including the expert-major MoE prefill's
  `generic_op` programs. It takes 84 s at startup with the kernel cache warm (396 s cold).
- a prefill that still compiles a new program makes `generator_vllm.py` recapture that replica's decode trace
  (`decode_recaptures`): 0 during the HTTP suite and benchmarks.

With `TT_METAL_TRACE_ALLOC_TRACKING=1` (`ttnn.execute_trace` then refuses to replay over a live buffer allocated after
the capture) and program-cache misses forbidden after the capture, a driver that serves 16 prompts (15-2154 tokens)
over 4 lanes twice, with blocks freed and reallocated in between, ran with no unsafe buffer, no new program, zero
recaptures and identical tokens in both rounds, for 2048-token chunks and for ragged 448-token chunks. The MoE
prefill's grouped-experts program (about 966 KB of static circular buffers on 120 cores) runs after decode without
clashing with decode's persistent L1 buffers.

## Decode-slot parity

A user's decode logits used to depend on whether its decode slot was even or odd (max |logit diff| 0.4-1.5 over 10
steps from the same prefill; slots of one parity were bitwise identical), which vLLM's 13 seed / batch
reproducibility tests catch since a request can land in either parity. The cause was layer 0, the dense global
layer: its decode attention ended in `ttnn.all_reduce`. A global layer's chip pair splits the users by slot parity,
so only chips 0/2/4/6 (even slots) or 1/3/5/7 (odd slots) contribute a user's row, and the ring reduce-scatter adds
those rows with bf16 partial sums in a parity-dependent order. The compact layers 1-47 were not affected: ReduceExpand
sums the chips in chip order in fp32, so the other chip's zero row does not change the order.

Layer 0's attention now reduces like theirs (`TtMiMoDecoderLayer.set_compact_attention`, at most 4 decode users):
slots 0-3 give bitwise-identical logits (48 layers, 10 traced steps; 1 layer before: up to 0.125, after: 0.0). Decode
time is unchanged (device loop 13.48 / 15.06 ms on 1 / 4 rows, 13.47 / 15.07 before). `test_model_tp8_dp4`: PCC min
0.9608, mean 0.9925 as before; top-1 39/40 -> 38/40, the changed row being golden 1's third decode step, a reference
near-tie (fp32 top-1 / top-2 margin 0.089) that is now an exact bf16 tie (margin 0.125 before), broken toward the
other token.

A global layer's prefill chunk after the first (`_prefill_owner_chunk`) zeroes the non-owner chip's rows before an
`all_reduce` in the same way, so prompts longer than one chunk likely still prefill slightly differently per parity
(not measured).

## Not done

- Slot parity of chunked prefill (above).
- Device sampling beyond greedy (top-k / top-p / temperature, penalties, logprobs run on the host, synchronously).
- A 1M-token prompt, and the 1M server on the c16 branch (decode RoPE tables are off past 128K context).
- Prefill and decode in the same step for different lanes (the plugin runs a step as all-prefill or all-decode).
- Speculative decoding (lane DP refuses it); tracing later prefill chunks and chunks longer than 512 tokens.
