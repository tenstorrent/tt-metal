# MiMo-V2.6-Flash vLLM serving on Blackhole Galaxy

`tt/generator_vllm.py` (`MiMoV2ForCausalLM`) serves XiaomiMiMo/MiMo-V2.6-Flash-MOPD through
[vllm-tt-plugin](https://github.com/tenstorrent/vllm-tt-plugin) (tracking issue tenstorrent/vllm-tt-plugin#216).
`--data-parallel-size 4` runs as four in-process lanes, one TP=8 replica per Galaxy row (`MiMoGalaxyGenerator`).
Sampling runs on the host; decode replays one trace per replica; prefill runs eagerly in chunks.

Measured 2026-10-19 on one BH Galaxy, tt-metal `mimo-v2.6/vllm` at ee6895badd4, plugin
`mimo-v2.6-flash/vllm-plugin-dev` at 47cc860, vLLM 0.30.0, MOPD weights (48 layers, BFP4 experts, BFP8 KV).

| Check | Result |
|---|---|
| Plugin HTTP suite `tests/tt` (no spec decode), 32K, 4 lanes x 4 users | 64 passed, 1 skipped, 14 failed: 13 seed / batch reproducibility tests (decode-slot parity, below) and `test_bad_words` (the model stops inside its reasoning when the banned word is the natural next token) |
| Chunked prefill (`--max-num-batched-tokens 2048`) | the three needle-recall tests pass: one split prompt, a 3x-budget prompt, and four prompts sharing steps with ragged chunk boundaries |
| Prefix caching | a second request on a cached 5.3K-token prefix prefills in 2.1 s instead of 31.8 s and recalls a passphrase from inside the prefix |
| Greedy vs the standalone generator, 16 chat prompts (15-552 tokens), 32 tokens | 15/16 identical (32K server), 14/16 (1M server); the others diverge at a near-tie after 13+ tokens |
| Qualitative, 6 readiness prompts, greedy and sampled (t=1.0, top_p=0.95) | coherent and correct; the `mimo` reasoning parser separates reasoning from content |
| Tool calling (`--tool-call-parser mimo --enable-auto-tool-choice`) | `get_weather({"city": "Paris", "unit": "celsius"})`, `finish_reason=tool_calls` |
| `--max-model-len 1048576` | serves; KV pool 1.27M tokens per lane (1.21x one 1M request); a 20,936-token needle prompt recalls correctly (87 s, mostly prefill) |

## Serving performance (host sampling, greedy, `vllm bench serve`, random 128 in / 128 out, `--ignore-eos`)

| Concurrency (prompts) | TTFT median / p99 | TPOT mean / p99 | ITL median / p99 | t/s/user | Output tok/s |
|---|---|---|---|---|---|
| 1 (4) | 506 / 1795 ms | 23.9 / 24.2 ms | 23.8 / 27.4 ms | 41.9 | 33.1 |
| 16 (64) | 4217 / 5762 ms | 34.6 / 66.5 ms | 32.0 / 42.0 ms | 28.9 | 233.7 |

32K server. On the 1M server, TPOT is 23.3 ms at concurrency 1 and 49.4 ms at 16 (wider page tables, below).
The traced device step is 13.5 ms (1 row) / 15.1 ms (4 rows) with the on-device greedy loop
(`tests/perf/decode_bandwidth.py`); serving adds host sampling (logit rows read back every step), per-step input
copies and page-table bookkeeping. TTFT is dominated by the per-32-token `sparse_matmul` MoE prefill.

## Launch

```bash
export TT_METAL_HOME=<tt-metal build> PYTHONPATH=<this tt-metal checkout> MESH_DEVICE="(4,8)" HF_HUB_OFFLINE=1
cd <this tt-metal checkout>   # the model's kernel sources resolve against the working directory first
vllm serve /path/to/MiMo-V2.6-Flash-MOPD --served-model-name XiaomiMiMo/MiMo-V2.6-Flash-MOPD --trust-remote-code \
  --data-parallel-size 4 --api-server-count 1 --max-num-seqs 4 --max-model-len 32768 --block-size 64 \
  --enable-chunked-prefill --max-num-batched-tokens 2048 --enable-prefix-caching \
  --reasoning-parser mimo --tool-call-parser mimo --enable-auto-tool-choice \
  --additional-config '{"tt": {"trace_mode": "decode_only", "trace_region_size": 150000000, "l1_small_size": 65536, "dispatch_core_axis": "col"}}'
```

- `--max-num-seqs` is per lane (16 users in total). `--block-size 64` is required.
- `--api-server-count 1`: without it vLLM starts one API server per DP rank, and each extra one rejects the
  plugin's structured-output setting (plugin fix in progress).
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
- a prefill that compiles new programs makes `generator_vllm.py` recapture that replica's decode trace
  (`decode_recaptures`; 36 during the HTTP suite and benchmarks, ~0.4 s each). The next step is warming every
  reachable prefill program before the capture (later chunks need a start-position-independent chunked MLA call).

With `TT_METAL_TRACE_ALLOC_TRACKING=1`, a driver that serves 16 prompts (15-2154 tokens) over 4 lanes twice, with
blocks freed and reallocated in between, found no unsafe buffer at any replay and identical tokens in both rounds.

## Decode-slot parity

A user's decode logits depend on whether its decode slot is even or odd: slots 0 and 2 give bitwise-identical
logits, slots 1 and 3 too, but the two parities differ (max |logit diff| 0.4-1.5 over 10 steps from the same
prefill). Companions, lanes and interleaved prefills change nothing. The standalone generator shows the same
numbers, so it is a property of the model's TP=8 global-layer user split, not of serving. vLLM places a request in
either parity, which is what the 13 reproducibility tests catch.

## Not done

- Decode-slot parity (above), and prefill recaptures down to zero.
- Device sampling and async decode (host sampling reads every user's logit rows each step).
- A 1M-token prompt: at ~250 prefill tokens/s per lane it would take ~70 min; the expert-major MoE prefill
  (tenstorrent/tt-metal#61462) is the fix.
- Speculative decoding (lane DP refuses it), prefill tracing.
