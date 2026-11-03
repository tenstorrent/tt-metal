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
  `setup_decode_rope_tables`; see "Device RoPE tables") and a non-blocking token read. Any non-greedy row, penalties,
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
  --enable-chunked-prefill --max-num-batched-tokens 2048 --enable-prefix-caching --prefix-cache-retention-interval 1024 \
  --reasoning-parser mimo --tool-call-parser mimo --enable-auto-tool-choice \
  --additional-config '{"tt": {"trace_mode": "decode_only", "trace_region_size": 1000000000, "l1_small_size": 65536, "dispatch_core_axis": "col", "sample_on_device_mode": "decode_only", "prefill_gather_ms": 200, "prefill_gather_quiet_ms": 25}}'
```

- `MIMO_EXPERT_QUANT=gptq` serves the GPTQ-prepared bfloat4_b experts (same speed, KL about half the stock
  packer's; see doc/tp8_dp4_plan.md "Accuracy"). Building their cache the first time needs
  `MIMO_EXPERT_PREP_DIR` with `tests/perf/gptq_experts.py` output. The default is the MSE-prepared experts.
- `trace_region_size` 1 GB holds the decode trace, the 28 prefill-batch traces and the traced later prefill chunks per
  replica (150 MB is enough with `MIMO_PREFILL_TRACE=0` and `MIMO_PREFILL_CHUNK_TRACE=0`). `sample_on_device_mode` and the prefill gather need plugin `mimo-v2.6-flash/c16-throughput`.

- `--max-num-seqs` is per lane (16 users in total). `--block-size 64` is required.
- `--prefix-cache-retention-interval 1024`: the SWA layers' prefix-cache checkpoints. vLLM 0.30's default (0) keeps
  a SWA group's cached blocks only for the window at the end of each prompt (and at a prefix two requests were seen
  to share), so a request that shares a long prefix with an earlier one but not its whole prompt (same document,
  another question) hits nothing: the global group hits, the SWA groups miss, and the hit is the shortest of the
  groups. 1024 keeps a window's blocks every 1024 tokens, so such a hit lands at most 1023 tokens short of the
  shared prefix; a multiple of 128 is required (the scheduler block), and 128 caches every SWA block.
- vLLM starts one API server per DP rank (4) by default; that needs plugin 11dc5a9 or later (before it, every
  extra API server rejected the plugin's structured-output setting and `--api-server-count 1` was required).
  Checked: 24 concurrent completions, a chat and 4 JSON-schema requests over the 4 API servers all succeed.
- `MIMO_KV_GB_PER_CHIP` caps the KV pool per lane (default 6.7 with the pair split, else 6); the adapter warns when
  the cap holds fewer than `--max-num-seqs` requests of `--max-model-len` (vLLM then queues the rest).
  `MIMO_MODEL_PATH` / `MIMO_TT_CACHE` as for the tests.
- `MIMO_PREFILL_CHUNK_TRACE` (default 1; `MIMO_PREFILL_CHUNK_TRACE=0` opts out) traces a whole later prefill chunk
  (start > 0) of the step's full token budget and replays it (`TtMiMoModel.setup_prefill_chunk_traces`): two traces per
  replica (the global layers' legacy and grouped / K-split SDPA programs; `MIMO_PREFILL_CHUNK_TRACE_SDPA=256,128` adds
  the program of a start at an odd multiple of 128). Warmup sets them up and `warmup_model_decode` captures them last,
  after the decode, prefill-batch and MTP traces, so it works with `trace_mode` `decode_only`. Each eager chunk costs
  ~0.23 s of host dispatch, so one host thread serving 4 busy lanes was host bound; replays make it device bound (256K
  per lane, budget 4096, pair split: 4 lanes 109 -> 53 s, 9.6K -> 20.0K tok/s for the Galaxy; one lane unchanged;
  logits bitwise equal). A lane's last, partial chunk and chunks of other lengths still run eagerly. The traces are
  optional: when their setup fails, or the trace region has no room for the next one (checked against the allocator
  before each capture) or a capture fails, the replica drops them with a warning ("traced later prefill chunks
  disabled, later chunks run eagerly") and serves with eager later chunks instead of failing startup; the same holds
  for a build whose `chunked_flash_mla_prefill` takes no runtime start. The log line "later-chunk prefill traces
  captured; trace region X MB of Y MB used per device" gives the region's headroom.
- `MIMO_KV_PAIR_SPLIT` (default 1; "Global KV pair split" below) halves the global layers' KV: 4 x 1M per lane fits at
  `MIMO_KV_GB_PER_CHIP=6.7` (6.69 GB: 4 x 546,368 pool tokens; 6.6 holds 3.95), the default. It needs a ttnn build whose MLA attention ops return their LSE; `MIMO_KV_PAIR_SPLIT=0`
  restores the duplicated layout (and serves on builds without those ops).
- Clients on this host should use `127.0.0.1`: it has no `localhost` entry in /etc/hosts, and the async OpenAI
  client times out resolving it.

## One engine per Galaxy row

`--data-parallel-size 4` runs the four rows as lanes of one engine, and a step is all-prefill or all-decode for every
lane: a prefill on one lane stops the other lanes' decode (on SPEED-Bench coding at c=16, prefill steps take 19-26% of
the wall time). `serving/serve_rows.sh` runs four engines instead, each with `--data-parallel-size 1` on one row's 8
chips (`TT_VISIBLE_DEVICES`, `MESH_DEVICE=(1,8)`, ports 8001-8004), behind `serving/lb.py` on port 8000. A row
prefills while the others decode, and each engine's host work runs in its own process.

The proxy routes by prefix, because each engine has its own prefix cache.
- **Prefix key.** It hashes each request's prompt in 256-character chunks: the chat messages and tools, or a
  completion prompt or its token ids, plus `cache_salt`.
- **Index.** An LRU index of the chunk hashes records which engines served each prefix.
- **Routing.** A request whose longest known prefix is at least 1024 characters goes to the least loaded engine
  holding it. If that engine has more than 2 requests in flight above the least loaded engine, the request goes to
  the least loaded engine instead, which then caches the prefix too. Any other request goes to the least loaded
  engine.
- **Health and metrics.** `/health` is 200 only when every engine is healthy. `/metrics` concatenates the engines'
  metrics. `/lb/stats` shows the routing counters.
- **Failover.** A request whose engine refuses the connection goes to another engine.

```bash
TT_METAL_HOME=<tt-metal build> PLUGIN_SRC=<vllm-tt-plugin at mimo-v2.6-flash/single-lane>/src VLLM=<venv>/bin/vllm \
  MODEL=/path/to/MiMo-V2.6-Flash-MOPD SAMPLE_ON_DEVICE=decode_only MIMO_EXPERT_QUANT=gptq \
  models/experimental/mimo_v2_6_flash/serving/serve_rows.sh <log dir> 32768
```

- **Plugin.** Branch `mimo-v2.6-flash/single-lane` of vllm-tt-plugin runs a model that declares
  `folds_dp_into_lanes` as one lane at `--data-parallel-size 1`. Each engine then takes the lane path: stable slots,
  chained verifies, sampled verifies accepted on device.
  - Without it, an engine takes the plugin's single-engine path. Plain decode runs at the same speed there.
  - MTP does not: verifies are not chained, and a sampled verify is accepted on the host in `logits` mode. MTP at
    c=16 then served 992 tok/s greedy and 93 tok/s sampled.
- **`OMP_NUM_THREADS=8` per engine** (the script's default). Torch's default pool has one thread per core. With four
  engines, every small torch op on the step path waited on the other engines' spinning pools.
  - A one-lane MTP verify step then took 61 ms instead of 23, with a 50 ms gap between verifies, and MTP at c=16
    served 439 tok/s.
  - Plain decode was not affected: a steady step replays its trace and runs almost no torch ops.
- **Rows.** Chip ids depend on the machine, so `serve_rows.sh` finds each row's chips before it starts the engines,
  with `serving/discover_rows.py` (it opens the whole mesh; set `ROW_DEVICES` to skip it). vllm-tt-plugin's standard DP
  discovers its ranks' groups the same way.
  - Four processes can each open one row at the same time: a ring all_gather and an all_reduce on every row are exact.
  - tt-metal pins each chip's dispatch threads to core (local chip id) % N, which is the same set of cores in every
    engine. Spreading them with a per-engine `LD_PRELOAD` offset made no difference.
- **Startup** takes about 1.5 minutes, because each engine warms up 4 users of one replica. The 4-lane engine takes
  about 4.5 minutes.
- **Stop.** Each engine runs in its own process group (`setsid`). Stopping sends SIGTERM to every group, then SIGKILL
  to what is left after `STOP_TIMEOUT` seconds (default 120), or 10 s after the last EngineCore is gone. When an engine
  or the proxy exits, the script stops the rest and exits with that process's status.
  - Not SIGINT: an engine started in the background inherits SIGINT ignored, and an EngineCore installs its handler
    only once the model is loaded. A SIGINT 60 s into startup was ignored until the SIGKILL.
  - On SIGTERM a loading EngineCore, the process holding the chips, exits at once. Its API server, still waiting for
    the EngineCore's handshake in a blocking poll, cannot run its SIGTERM handler, and multiprocessing's resource
    tracker ignores SIGTERM; both go with the SIGKILL.
- **Cold TTNN cache.** The four engines share one `MIMO_TT_CACHE`, and `ttnn.as_tensor` writes a cache file in place.
  Each cache tag (`top`, every layer, each MTP layer, DFlash) is therefore filled under a file lock
  (`tt.common.cache_fill`): one engine converts and writes it, and the others wait, then load it. A complete tag takes
  no lock.
- **vLLM's own DP** (`--data-parallel-size 4` without the lane fold) is no alternative. For an MoE model vLLM runs the
  DP ranks in lockstep (`DPEngineCoreProc`), and the plugin refuses standard DP for MoE.

Measured 2026-10-28 with ttnn and model code at ae0a9130ab7 (the packed grouped verify, #63006), GPTQ BFP4 experts and
`sample_on_device_mode` `decode_only`. The plugin was `mimo-v2.6-flash/single-lane` for the row engines and its base
ee2b54f for the 4-lane engine.
- Both engines ran `--max-num-batched-tokens 2048` (the script's default then) and the default KV budget (6.7 GB per
  chip with the pair split). tt-inference-server's spec ships 4096-token chunks and sets 6.7. `serve_rows.sh` now
  defaults to 4096 (`MNBT`); the numbers below are not re-measured at 4096.

SPEED-Bench coding, 64 requests, 512 output tokens, c=16, 32K server:

| | 4 lanes, one engine | 4 engines, one per row | |
|---|---|---|---|
| plain, greedy: output tok/s | 835.4 | 983.8 | +18% |
| plain, T=1.0, top-p 0.95 | 785.5 | 1002.1 | +28% |
| plain, TTFT mean | 577 ms | 421-453 ms | |
| MTP (`MIMO_SPEC_SLOTS=8 MIMO_SPEC_GROUPS=4`, K=1, 8 seqs per engine), greedy | 931.2 | 1178.6 | +27% |
| MTP, T=1.0, top-p 0.95 | 883.8 | 1146.2 | +30% |
| MTP, tokens per verify (greedy) | 1.91 | 1.92 | |
| MTP, TTFT mean | 422 ms | 323 ms | |

The 4-lane plain numbers are from the build before #63006. That change touches only the verify.

16 prompts of 256K random token ids, 2048 output tokens each, c=16, MTP, 1M server:

| | 4 lanes | 4 engines |
|---|---|---|
| all 16 done | 685 s | 320 s |
| TTFT median / max | 494 / 661 s | 224 / 293 s |
| prefill over the run | 6,122 tok/s | 13,122 tok/s |
| decode window (all 16 decoding) | 1291.8 tok/s | 1301.0 tok/s |

**tt-inference-server.** The release flow starts this deployment itself, on branch
`mimo-v2.6-flash/row-engines` of tt-inference-server. A device spec's `submesh_engines: true` makes
`vllm-tt-metal/src/run_vllm_api_server.py` split the mesh into `data_parallel_size` device groups, discovered at launch
as the plugin's standard DP does (the four rows on a Galaxy), start one engine per group, and serve the same proxy
(`engine_proxy.py`) on the service port, so every workflow talks to one endpoint.
- **Checked on the host:** the launcher with the dev catalog's MiMo runtime spec (1M context) came up with all 4
  engines healthy in 250 s. SPEED-Bench c=16 through it gave 958 tok/s greedy and 938 sampled, and SIGTERM stopped
  every engine in 8 s. The MTP entry gave 1122 greedy and 1038 sampled. Every verify was accepted on device and 96%
  were chained.
- **Image:** it needs the vllm-tt-plugin branch above.

Shared prefixes, through the proxy with and without prefix routing (`--min-match-chars` 1e12 turns it off):
- Workload: 8 documents of 6000 random words (about 8K tokens), 4 questions each, sent question by question (a
  document's requests are not adjacent), c=16, 32 output tokens. Each run used its own `cache_salt`, and the two
  modes alternated, two runs each.
- The plugin reports no prefix-cache counters on TT; the tokens actually prefilled come from the engines' MiMo
  prefill timing lines.

| | load only | prefix routing |
|---|---|---|
| wall time | 9.4 / 8.3 s | 7.7 / 6.5 s |
| TTFT mean / median | 2.50 / 2.57 s, 2.27 / 2.29 s | 1.90 / 1.37 s, 1.94 / 1.62 s |
| prompt tokens prefilled | 186,788 / 157,346 | 109,992 / 103,825 |

Prefilling each document once would be about 70K tokens. The rest is requests that arrived while their document
was still prefilling on the same engine, plus spills.

Checks:
- c=1 greedy on 16 prompts (63 to 60,000 tokens, 64 output tokens), served by rows 0 and 1, is identical to the
  4-lane engine on the same build (16/16). Two 4-lane runs at c=16 agree on 7/16, from batch composition.
- `MIMO_SPEC_CHAIN_CHECK=1` over 32 requests: 7552 chained drafts checked across the 4 engines, 0 differ.
- Startup, a chat request through the proxy, the merged `/metrics`, and shutdown on SIGTERM work with the committed
  script.

## KV cache

`get_kv_cache_spec` reports a FullAttentionSpec for the 9 global layers and a SlidingWindowSpec for the 39 SWA
layers, both K 192 / V 128. The SWA layers attend a window of 128; their spec declares 128 + the chunk start alignment
- 1 (255 with the pair split, 191 without; `swa_spec_window`), because vLLM frees sliding-window blocks by the window it
is told and `prefill_forward` recomputes up to alignment - 1 positions in front of an unaligned chunk start (see the
prefill bullet below). With the pair split (below) the global spec has block size 128, whose page is the
SWA page (4 KV heads x 128 tokens against 8 x 64); with `MIMO_KV_PAIR_SPLIT=0` it keeps block size 64 with its page
padded to the SWA page. vLLM groups them into 6 KV cache groups (1 global, 5 SWA of up to 9 layers), each with its own
block table, all drawing block ids from one pool per lane. The model keeps one device page table per group
(`TtMiMoModel.setup_vllm_kv_cache` / `set_vllm_page_tables`, copied only when a row changes) and reads a later SWA
chunk's window from vLLM's SlidingWindowSpec table. Buffers are shared the way vLLM shares them (the i-th layer of
every group): K for all layers, V among layers of the same width; 4.7 KB per token per chip.

A global layer's KV head is served by a chip pair (4 KV heads over 8 chips). The standalone model splits the users
between the pair by slot parity, which vLLM cannot use: a prefix-cached block can be reused by a request in a slot of
either parity. With `MIMO_KV_PAIR_SPLIT=0` each global layer keeps both copies of its KV head on the pair and decode
still splits the attention by slot; the 6 GB pool then holds about 1.27M tokens per lane (one 1M request).

## Global KV pair split

`MIMO_KV_PAIR_SPLIT=1` (the default, vLLM mode only; `reference/kv_pair_split.py`, `tt/pair_split.py`) stores each
global KV head once per pair by splitting every block instead of the users:

- **Layout.** A block j of vLLM's global group (block id b) covers global positions [128 j, 128 j + 128). Chip
  c = chip % 2 of the pair keeps positions [128 j + 64 c, 128 j + 64 c + 64) in rows 0..63 of the same physical page
  b (`[1, 64, dim]` per chip, shared with the SWA layers as before): global position p sits at local index
  `64 (p >> 7) + (p & 63)` on the chip with `((p >> 6) & 1) == c`. Both chips use vLLM's table unchanged, one entry
  per 64 local positions, so the paged ops read and write local positions with no change.
- **Capacity.** A global token costs half a pool token: `get_max_tokens_all_users` counts a request's global group as
  `64 ceil(max_model_len / 128)` pool tokens. At `MIMO_KV_GB_PER_CHIP=6.7` the pool holds 4 x 1M requests per lane,
  c=16 at 1M on the Galaxy: the model asks for 2,185,472 pool tokens (6.69 GB), and the plugin's sizing adds its
  output and sliding-window headroom for 34,216 blocks (6.70 GB per chip), where vLLM accounts 8,537 blocks per 1M
  request (8,192 global and 5 x 69 sliding-window blocks with a 2048-token step budget and two steps in flight).
- **Decode.** Every chip runs its KV group's 16 Q heads over its own keys for every decode user (no owner mask), with
  `paged_flash_multi_latent_attention_decode(..., return_lse=True)`: the normalized partial output o and its LSE
  (`m + log l`). `PairSplit.merge_heads` then runs one launch: each chip's exchange core sends its 16 x B LSEs to the
  pair partner over the fabric (one hop, `kernels/attn/pair_lse_send.cpp`, a fused scatter write and semaphore
  increment into one of two receive slots), and 64 merge cores (`kernels/attn/pair_merge_heads.cpp`) wait for the
  partner's packet, compute `alpha = 1 / (1 + exp(clamp(lse_p - lse, -80, 80)))` in fixed point (~1e-8) and write
  `alpha * o` with the heads concatenated (in place of ConcatHeads), each bf16 product rounded once. The receive slot
  and the expected arrival come from per-core call counts in persistent global semaphores that every launch advances
  on every chip. The row reduction after `wo_decode` then sums `W (alpha_A o_A + alpha_B o_B)`, which is full
  attention (prefill keeps the ttnn-op `PairSplit.merge`: an `all_gather` of the LSE and eltwise ops). A user with no
  key on a chip has o = 0 and LSE -1e30, so the other chip carries it. The per-chip write positions (-1 where the chip
  does not keep the token) and SDPA positions (the last local index at or before the user's position) are derived
  inside the trace from the step's global positions, which advance with `plus_one`, by one one-core program
  (`kernels/attn/pair_positions.cpp`, mirrored by `positions_step`); it also writes a verify's serial (one slot each)
  write positions.
- **Prefill.** A first chunk is unchanged (8 heads per chip over the chunk's own K / V) but for its fill table: one
  entry per 64 positions, the block on the chip that keeps them and the null block on the other (`kv.pair_fill_rows`,
  also for batched first chunks). A later chunk runs the group's 16 Q heads through pair-split
  `chunked_flash_mla_prefill(..., kv_pair_parity_tensor=<the chip's parity>, return_lse=True)` (it reads the chip's
  local keys [0, (start + length) / 2) with an unsplit later chunk's program: from 64K the grouped / K split one, G 2
  S 15, whose LSE the K split's merge forms from the pieces' statistics) and the same merge, in place of the head
  split that `MIMO_KV_PAIR_SPLIT=0` runs (8 heads per chip over both copies: the same work). Chunk starts and later chunks' lengths are multiples of 128: a later chunk pads to at
  least 128 tokens, and `prefill_forward` moves a chunk start back to a 128-token block. vLLM's prefix-cache hits
  land on its scheduler block, lcm(64, 128) = 128, and the plugin's TTScheduler ends partial chunks on its 128-token
  grid (`prefill_chunk_alignment`), but a running request whose leftover step budget cannot reach the grid gets the
  shorter chunk, so its next chunk starts up to 127 positions past a block (two long prompts chunk-prefilled in one
  step do this). The moved-back chunk recomputes those positions, whose sliding windows reach up to 127 positions
  further back than vLLM keeps for the scheduled start; the SWA specs' longer window (`swa_spec_window`) keeps them
  (vLLM overwrites a spec's `extra_retained_tokens` from the speculative config, so the margin cannot go there).
  `prefill_warmup_chunks(..., pair_split=True)` warms the later chunks at those starts and lengths, and each length
  that takes the grouped / K split program once at 64K.
- **Slot parity.** Every chip of the pair computes every slot, so a global layer no longer depends on a slot's parity
  in decode or in later prefill chunks.
- The SWA layers' attention, the MTP and DFlash caches, the MTP history and the ring backfill are unchanged.

The two attention ops' new arguments are used only in `TtMiMoAttention._paged_mla_decode_lse` and
`_chunked_mla_prefill_lse` (`PAIR_PARITY_KWARG`). `tests/test_kv_pair_split_host.py` checks the mappings (0, 63, 64,
127, 128 and random positions to 1M), the fill tables and decode positions through a torch mirror of the paged ops,
the merge against full attention, the warmup's coverage, the capacity and the adapter's per-group block sizes and
table widths, that `prefill_forward` moving a chunk start back reads only sliding-window blocks vLLM still holds (at
every offset, with vLLM's own freeing rule; the model's 128 window as vLLM's trips at offsets 63 and up), and, where
vLLM and the plugin are installed, that vLLM 0.30 groups the specs with block sizes 128 / 64 on one page size and that
the plugin's pool at 6.7 GB holds 4 x 1M by vLLM's accounting and passes its planner. `tests/test_kv_pair_split_tp8.py`
(device, one Galaxy row): the positions program and the merge against their host mirrors (merge PCC 0.999999), a
global layer's prefill chunks and decode steps with the split on vs off (positions straddling 64 / 128: prefill PCC
0.99999, decode 0.99998; 32K context: 0.99999 / 0.99998), and the serving decode configurations (the compact SDPA
over 4 table rows, and verifies writing K / V slot by slot at 62..65 and 126..129: 0.99999 / 0.99998); the split cache
holds the unsplit one's K / V at each local index (PCC 1.0).

Served (vLLM, DP=4 lanes x `--max-num-seqs 4`, `--max-model-len 1048576`, `MIMO_KV_GB_PER_CHIP=6.7`, greedy device
sampling, no speculation; the same build with `MIMO_KV_PAIR_SPLIT=0` as the baseline):

| | split | `MIMO_KV_PAIR_SPLIT=0` |
|---|---|---|
| KV pool per lane (vLLM) | 4,202,656 tokens, 4.01x a 1M request | 2,149,317 tokens, 2.05x |
| c=16, 128 in / 128 out | 912 tok/s, TPOT 15.16 ms, median ITL 13.51 ms | 967 tok/s, 14.29 ms, 12.97 ms |
| c=1, 128 / 128 | TPOT 9.76 ms, median ITL 9.69 ms | 9.32 ms, 9.22 ms |
| c=16, 32K in / 256 out | median TTFT 43.9 s, median ITL 14.41 ms | 43.9 s, 14.01 ms |
| APC check (16K document, 16 questions) | 48/48 correct, warm == cold 32/32 | the same |
| plugin HTTP suite (tests/tt) | 76 passed, 2 failed | 74 passed, 4 failed |

The table's split column ran the first merge (the LSE all_gather and seven ttnn ops, 41.1 us per global layer, about
0.5 ms per step). The fused merge (`tests/perf/kagent_pair_merge_bench.py`: 8.4-8.6 us per call at 4 users, found with
tt-kernel-agent) brings the split back to the baseline's speed: c=16 128 / 128 955.6 tok/s, TPOT 14.50 ms, median ITL
12.99 ms; c=1 TPOT 9.33 ms, median ITL 9.25 ms; c=16 32K / 256 median ITL 13.84 ms; APC check exact (48/48, warm ==
cold 32/32); HTTP suite 76 passed, the same 2 seed tests failed. At long context the step reads the same KV bytes as
before, since decode already split the users. Both HTTP
runs fail the same two seed-reproducibility tests (`test_seeding`, `test_same_seeds_reproduce_across_batches`); the
baseline also fails `test_specific_seed_reproducible[42]` and `test_mixed_params_batch`. Greedy over 16 prompts of 63
to 60K tokens (64 tokens each): split and baseline agree on 3 of 16 served alone (c=1), and the continuations stay
coherent; for scale, each mode agrees with itself on 6 of 16 between two c=16 runs and on 8 of 16 between c=1 and
c=16, so these prompts flip on rounding near-ties, and the split changes the attention's rounding in every global
layer.

16 users x 1M, every request a full 1M-token prompt (`long_context_16x1m.md`; client `tests/perf/vllm_long_prompts.py`):
- At the default 6.7 GB pool all 16 complete in 57 minutes.
- Each lane prefills its four requests one after another, ~855 s each, so the TTFTs are 14, 29, 43 and 57 minutes.
- One 1M prompt alone takes 356 s (2.95K tok/s). Four lanes prefilling together reach only 1.7x that throughput:
  host dispatch is the bound.
- Decode at 1M for one request alone: 12.3 ms per token.

## Device RoPE tables

Decode, the verify and the MTP draft read their RoPE cos / sin as rows of device tables at their own position inputs
(`TtMiMoModel.setup_decode_rope_tables` / `rope_table_rows`, `ttnn.embedding` over a uint32 position row), so no step
uploads cos / sin, at any context length. MiMo needs four tables: cos and sin for the SWA theta (the 39 SWA layers and
the MTP layer) and for the global theta (9 global layers), RoPE on 64 of the 192 head dims. Each row is the host's
`decode_host_inputs` value of that position (`rope.partial_rope_table`, the same fp32 products and cos / sin, the same
bf16 conversion), so device and host cos / sin are bitwise equal and greedy output does not change.

Until this change the tables were made only up to 128K context: whole 192-wide rows, which at 1M cost
1M x 4 x 192 x 2 bytes = 1.61 GB per chip, against ~34.0 GB per chip of which the weights take 23.7-24.2 GB, the KV
pool 6 GB (`MIMO_KV_GB_PER_CHIP`; one 1M request is 3.2 GB of it) and the trace region 1 GB. Nothing else limits the
row count (`ttnn.embedding` reads one 128 / 384-byte row-major page per position). Past 128K the tables now keep only
the 64 rotated dims (`compact`, `ROPE_TABLE_FULL_WIDTH_MAX_LEN` in generator_vllm.py) and each read appends the
constant 1 / 0 tail of the other 128 dims with one `ttnn.concat` (a persistent tile per value): 537 MB per chip at 1M
(67 MB at 128K, where the whole-row tables of 201 MB are kept). The tables run `ROPE_TABLE_MARGIN` (32) rows past
`max_seq_len - 1`, so a position derived on device a few steps ahead of the host never indexes past them.
`MIMO_DEVICE_ROPE=0` uploads the cos / sin every step instead. `tests/test_rope_tables_host.py` checks the rows
against the host cos / sin bit for bit (every position to 1M, both thetas, whole and compact);
`tests/test_spec_inputs_tp8.py` checks the device rows against the host-uploaded cos / sin.

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
- with the KV pair split, the per-chip positions program and the merge's ops (LSE all_gather, the partner slice, the
  fp32 eltwise ops) run in the decode warm-up before the decode capture, on the shapes the verify uses too
  (generic_op keys on its runtime arguments' count, not their addresses, so the verify trace's run of the positions
  program is a cache hit); their constants and output buffers are made in `setup_vllm_kv_cache`. The later chunks'
  pair-split programs compile in `warmup_model_prefill` (`prefill_warmup_chunks(..., pair_split=True)`).

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
(not measured). The KV pair split (the default) removes that: both chips of a pair contribute to every slot.

## Speculative decoding (MTP)

Not yet run on hardware. The 3 MTP layers (`model.mtp.layers.{0,1,2}`) draft 3 tokens through the plugin's
model-owned drafter contract (`docs/SPEC_DECODE_CONTRACT.md`), draft i + 1 from layer i (below; `MIMO_MTP_LAYERS=1`
drafts all 3 with layer `MIMO_MTP_LAYER`, default 0, as upstream vLLM does for MiMo); lane DP serves it with the plugin
branch `mimo-v2.6-flash/vllm-plugin-dev-lane-spec-decode`. Launch as above with:

```bash
export MIMO_SPEC_DECODE=mtp   # MIMO_MTP_LAYERS=3 (default; needs MIMO_DEVICE_ROPE=1) or 1
vllm serve ... \
  --speculative-config '{"method": "custom_class", "model": "vllm_tt_plugin.model_owned_drafter", "num_speculative_tokens": 3}'
```

- **Who speculates.** A lane drafts after a step that ran one request in it; a lane with several requests declines
  (`num_valid` 0) and decodes as before. A verify answers in the accept mode the plugin asks for: `argmax_ids` when
  every request of the step is greedy; `fused_sample` when a request samples with only temperature / top_k / top_p
  inside the device sampler's bounds (acceptance on device, below); `logits` otherwise (the plugin's host walk, for
  penalties, logprobs, grammar, token filters).
- **Verify** (`reference/spec_decode.py`, `TtMiMoModel.capture_verify_trace`): the decode step with the request in
  every decode slot of its replica, column j of the candidate block in slot j at position p + j, each slot with the
  request's page tables. It stays on the 4-user fast path (streaming MoE, SwaDecodeQkv, GreedyHead). The slots share
  cache tiles that the per-slot cache writers read, patch and write back concurrently, so the verify step writes K / V
  one slot at a time (`serial` positions in `TtMiMoAttention.forward_decode`); the query of the fused SWA kernel does
  not depend on the positions, so it is repeated with one slot's position each. A block that carries drafts beside
  another request (one joined after the proposal) is packed into passes of 4 slots.
- **Draft** (`tt/mtp.py`): one traced program per replica, each MTP layer with its own KV cache, a ring of 256
  positions addressed at 256 + t % 256 with RoPE at t. With every layer (the default, below) pass i runs layer i over
  the draft's rows. With one layer the fill pass runs the MTP layer over the committed tokens, each slot taking the
  target's final normed hidden there (the persistent buffer of the last layer's fused norm) and the embedding of the
  committed token; two steps on the last slot feed the layer its own output (vLLM's MTP proposer with one layer).
- **Inputs** (`reference/spec_inputs.py`, `tt/spec_inputs.py`): a verify and a draft each copy one packed int32
  tensor (tokens and positions, the verify's chain row; the draft's ring cache positions and RoPE indices) instead of
  13-14 tiny tensors; a short program at the start of each trace writes the trace's input buffers from it (owned
  positions, owner and token masks, serial positions, RoPE index). With `MIMO_DEVICE_ROPE=0` (no device RoPE tables)
  the verify also copies its 4 cos / sin and the draft its 6.
- **Acceptance on device** (`fused_sample`, `tt/verify_sampler.py`, `kernels/sample_verify.cpp`, host mirror
  `reference/sample_verify.py`): after each verify pass a small trace runs DeviceSampler's SampleScan and
  SampleChipMerge over the pass's slots (each with its request's temperature / top_k / top_p), the row all_gather,
  and SampleVerify on one core: per request, the plugin's point-mass accept walk with the uniforms the plugin drew
  from the request's generator (`spec_sampling`), greedy requests on the verify's argmax ids. The host reads each
  request's committed tokens and count instead of 4 x 152k fp32 logits, and walks nothing. Exact where the slot's 256
  candidates hold the nucleus (the decode sampler's condition); otherwise the request is accepted on its candidates
  and counted (`MiMo vLLM fused accept` log line, every 1024 fused verifies). The sampled slots' parameters are copied
  only when they change (`VerifySampler.stage`), so a greedy pass copies its 256-byte control page alone.
- **Draft chained behind the verify** (`tt/spec_chain.py`, `kernels/spec_chain.cpp`, host mirror
  `reference/spec_chain.py`). An `argmax_ids` verify runs the same accept trace, every request greedy: its committed
  tokens are exactly the plugin's greedy walk over the ids. The accept trace ends with the chain step (one core):
  from SampleVerify's page and the verify upload's chain row (the source's slots and the position limit, packed by
  the host with the verify's tokens and positions) it writes the MTP draft's packed upload as `pack_mtp(plan_mtp(...))`
  would from the plugin's commit (the fill over the committed tokens, the anchor at the last one, the ring cache
  positions and RoPE indices), and the header of the pass's result page (the accept page, the argmax ids, the
  anchor). On a lane that speculates the host enqueues, without waiting, verify, accept and the MTP draft trace, which
  ends with the drafts step (the anchor's three drafts into the result page), then reads that page: 320 bytes per
  pass, one read, where the step used to read the verify, walk, plan and upload the draft, and read the drafts.
  `propose_draft_tokens` returns the drafts already read when the plugin's commit (count, tokens, positions) is the
  device's; it drafts from the host plan as before after a plain decode step, a `logits` verify, without the decode
  RoPE tables (contexts past 128K: the draft's cos / sin are host uploads), at the context end (the chain declines when
  the bonus reaches max_seq_len - K; a narrower propose that still fits drafts from the host plan), or on a differing
  commit. The anchor needs no gather: the fill pass runs the MTP layer on every committed slot's own hidden row (the
  verify's final-norm buffer), and the anchor is the one slot the chain gives step positions. `MIMO_SPEC_CHAIN=0`
  restores the unchained flow (an `argmax_ids` verify then runs no accept step); `MIMO_SPEC_CHAIN_CHECK=1` drafts every
  chained draft again from the host plan and logs differences (a deferred verify, below, waits for its pages and checks
  its drafts before it returns). `MIMO_VLLM_TIMING=1` splits the verify's wait at the accept trace's end (the
  verify's device share, then the chained draft's) and counts chained and host-planned drafts.
  Every buffer the chain reads or writes is persistent (accept page, `mtp_packed`, the MTP heads' token rows, the
  result page), so the relay step of a chained verify (below) reads them.
- **Async scheduling** (`supports_async_spec_decode`; `MIMO_ASYNC_SPEC=0` declares none, and the launch then needs
  `--no-async-scheduling`). Ordinary steps of a speculating launch overlap the step in flight as without
  speculation (at concurrency 16 every step is one: every lane holds several requests, none drafts, and a propose
  touches no device). The plugin proposes for an ordinary step when it commits it, a step after its submission; when
  the lane's next step is already enqueued, the resident hidden the MTP would read is that step's, so the propose
  enqueues nothing and offers placeholder drafts, which make the plugin drain the newer step and propose for it (that
  propose drafts on device). A fused_sample verify asked to defer its read (`defer_read`, every fused verify of a
  launch that chains; `MIMO_SPEC_DEFER=0` answers host tensors instead) returns a handle (`VerifyHandle`, with a
  `tip_id`): each pass's result page travels to the host behind it without blocking (`SpecChain.read_async`), the
  plugin reads the handle a step later, possibly on another thread (`read_decode_output(handle)`: the committed
  tokens, counts and p0), and its propose returns the drafts the pages carry, enqueueing nothing. Every verify now
  enqueues all its passes, each followed by its page copy, before the host waits.
- **Chained verifies** (the plugin's section 4j: `chain_rows`, `chain_tip`). A verify built over the deferred one in
  flight names the rows whose block only the device knows: the last committed token at the pending step's p0 plus
  its count, then the drafts chained behind it. The host lays out the row's pass at the plugin's provisional
  positions (an upper bound: they only pick the page tables, which vLLM indexes by absolute position) with
  placeholder tokens and the chain row's relay word; the verify trace now starts with the relay step
  (`kernels/spec_chain.cpp` mode 2, mirror `reference.spec_chain.relay_step`), which writes the row's tokens and
  positions into `verify_packed` and its drafts into the accept step's control page (staged before the verify now)
  from the result page the lane's last pass left; `derive_verify`, the decode step, the accept trace and the chained
  draft then run as for a block the host built. The chain step writes the slots' positions into the page, and the
  readout reports them as p0. So a speculating lane runs relay, verify, accept, draft, relay, ... with the host one
  step behind, copying one packed upload and one 256-byte control page and reading one 320-byte page per step. The
  adapter checks the contract (the tip, a chain row is a lane source whose draft chained, placeholder tokens, count
  -1, 1..3 drafts, a provisional position at or above the highest the device can have reached, nothing else in its
  lane); a chain onto a step whose draft the device declined (the context limit, which the plugin's 2 (1 + K) margin
  keeps away) idles the pass and raises at the read. A chain row whose page tables hold vLLM's null block inside the
  window of its lowest possible position is counted (`chain_stats["uncovered"]`) and logged: vLLM frees
  sliding-window blocks by the scheduler's optimistic length.
- **A prefill over a verify in flight** (`supports_prefill_over_pending_spec`, with `MIMO_ASYNC_SPEC`). A request the
  plugin leaves out because the step in flight reaches its `max_tokens` needs nothing from the model: a verify sends
  its slot as a gap, and a steady decode's resident inputs decode it once more into its own blocks until the next
  reload. A prefill may run while a
  deferred verify is in flight: it writes only its slots' page-table rows and K / V, behind the verify on the device
  queue, and the verify's pages are already on their way to the host, so its read and propose follow as usual;
  nothing chains onto it afterwards, and the lane's MTP ring restarts.
- **Tests.** `tests/test_spec_decode_host.py` (no device): the planning, the ring addressing, and speculative greedy
  equal to plain greedy on a small fp32 MiMo with an MTP, an oracle and a partially wrong drafter.
  `tests/test_spec_inputs_host.py` (no device): the packed inputs' torch mirrors against the host-built inputs;
  `tests/test_spec_inputs_tp8.py` (one Galaxy row): the device programs against the same values, eager and traced,
  and the chain and drafts steps bit for bit against their mirror. `tests/test_spec_chain_host.py` (no device): the
  chain's mirror against the host plan (every accepted count, the ring's wrap, the context limit) and after the accept
  kernel's greedy mirror; `tests/test_spec_chain_adapter_host.py` (no device, needs vllm-tt-plugin): the adapter's
  verify / propose on fake lanes made of the mirrors (chained and host-planned drafts, the fallbacks);
  `tests/test_spec_async_adapter_host.py` (no device, needs vllm-tt-plugin): the same lanes driven through the
  plugin's async calls (overlapped ordinary steps and their proposes, deferred verifies read on another thread,
  chained verifies equal to the synchronous path token for token, the contract's refusals).
  `tests/test_sample_verify_host.py` (no device): the accept kernel's integer mirror against the plugin's accept-walk
  semantics (accept boundaries, the residual without the draft, the bonus, greedy ties, top_k and top_p, truncation),
  and against `vllm_tt_plugin.spec_accept` itself when the plugin is installed.
  `tests/test_spec_decode_tp8_dp4.py` (Galaxy, needs vllm-tt-plugin): the adapter's speculative tokens equal its
  greedy tokens, and its first verifies agree across modes (logits argmax, greedy and sampled fused_sample against the
  plugin's walk); every propose after a verify returns the chained drafts, the first ones equal to the host plan's;
  logs the mean accepted length and ms per token. `tests/test_verify_sampler_tp8.py` (Galaxy): the
  kernel bit for bit against the mirror on the device's own candidate pages, eager and traced.
- **Open.** The chained verifies on hardware with every MTP layer and the backfill (the relay, deferred reads and
  chained verifies ran on BH Galaxy with one layer); the accept step's SampleScan on all-greedy passes (only
  SampleVerify's argmax walk is needed there); more than one speculating request per lane (16 / 32
  verify rows exceed the 4-user kernels); the trace region size with both extra traces.

## Speculative decoding (DFlash)

The checkpoint's DFlash draft model (`dflash/`: 5 dense Qwen3-style layers, block of 8, non-causal, sliding window
1024, attention sinks; ~1.47B parameters) drafts through the same contract, verify, accept and chain path as MTP
(`tt/dflash.py`, `reference/dflash.py`). Its whole block (K = 7) needs the 8-slot verify:

```bash
export MIMO_SPEC_DECODE=dflash MIMO_SPEC_SLOTS=8  # needs MIMO_DEVICE_ROPE=1 (the draft reads the decode RoPE tables)
vllm serve ... --max-num-seqs 8 \
  --speculative-config '{"method": "custom_class", "model": "vllm_tt_plugin.model_owned_drafter", "num_speculative_tokens": 7}'
```

- **Context.** The decode and verify traces copy the outputs of target layers 0, 11, 23, 35, 47 into five tap buffers;
  the draft program projects the committed rows (fc, hidden_norm, each layer's k_proj / v_proj / k_norm, RoPE) into a
  per-lane ring of the last 1024 positions (a one-hot matmul scatter), and each prefill chunk writes its last 1024
  rows. A request's first chunk, or a draft that does not continue the ring, resets it.
- **Draft.** One traced program per replica: [anchor, 7 x mask] through the 5 layers (DRAM-sharded projections, the
  block's attention as one decode-SDPA user whose 64 query heads are the (row, head) pairs), the target LM head and a
  GreedyHead over rows 1-7. 2.5 ms per draft on one row; drafts equal the fp32 reference's on 99.6% of 480
  (`tests/test_dflash_tp8.py`).
- **Chain.** As MTP's (Speculative decoding above): the accept trace's chain step writes the DFlash draft's packed
  upload from what the device committed (`reference/spec_chain.py` `dflash_chain_step`: the committed columns' slots
  as context rows at their positions, the last committed token at the bonus as the anchor), and the draft trace ends
  with the drafts step, which copies the GreedyHead row into the pass's result page; the relay, the deferred reads and
  the chained verifies are unchanged. `MIMO_SPEC_CHAIN_CHECK=1` on hardware: 320 chained drafts, all equal to the
  host plan's.
- **8-slot verify** (`MIMO_SPEC_SLOTS=8`, `reference/spec_decode.py`; 4 by default, MTP's): a pass of up to 8 slots,
  so up to 7 drafts. The decode kernels serve 8 rows: the MoE streams load each expert the slots selected once
  (`MIMO_MOE_DEDUP=1`, the default; the gate|up and down streams take the distinct experts in user order, with
  per-row routing weights), the 8-row compact qkv / o streams and row reductions, SwaDecodeQkv (one launch writes
  every slot's K / V), GreedyHead, and the DeviceSampler / VerifySampler up to 8 users (launch 8 sequences per lane).
  A verify with more slots than the device sampler's users accepts greedy requests only (`GreedyVerify`,
  `kernels/greedy_verify.cpp`) and raises on a sampled one.
- **Limits.** Prompts prefill one row at a time (no batched prefill); one speculating request per lane.

c=1, 16 SPEED-Bench coding prompts, 512 output tokens (BH Galaxy, 2026-10-23; tokens / step, TPOT):

| drafter | verify slots | greedy | sampled (T 1.0, top_p 0.95) |
|---|---|---|---|
| 3 MTP layers, chained (default) | 4 | 3.47, 4.35 ms | 3.33, 4.53 ms |
| DFlash, K = 3, chained | 4 | 3.33, 4.54 ms | 3.20, 4.73 ms |
| DFlash, K = 5, chained | 8 | 4.34, 4.44 ms | 4.00, 4.82 ms |
| DFlash, K = 7, chained | 8 | 5.00, 4.20 ms | 4.45, 4.72 ms |
| DFlash, K = 7, host-planned (the build before the chain) | 8 | 5.00, 5.82 ms | |

A K = 7 step takes ~21 ms of device time (the 8-slot verify ~17.5 ms, its accept step 0.23 ms, the draft ~2.6 ms)
against ~15 ms for MTP's (the 4-slot verify ~12 ms, three MTP passes ~2.65 ms). Of the 8-slot verify's extra
~5.5 ms, ~2.8 ms is the MoE: 8 consecutive tokens of these prompts select 37.5 distinct experts per layer, 4 select 23,
each streamed in ~4 us; the rest is the 8-row attention and reductions. That eats most of DFlash's 1.5 extra tokens
per step: K = 7 is 3.5% faster than MTP greedy and ~4% slower sampled, where it accepts 4.45 tokens (sampled runs vary
by a few percent: another MTP run gave 3.25, 4.65 ms). K = 5 pays the 8-row kernels with fewer tokens; at K = 3 DFlash
accepts slightly less than the MTP layers for the same step. Open: an accept trace without SampleScan for all-greedy
passes (0.23 ms per 8-slot step, 0.15 ms per 4-slot step, for both drafters); the draft's 10 row all-reduces.

### Two requests per lane (draft groups)

With more than one request in a lane the drafters drafted nothing: they kept one request's state per lane. A pass of
8 slots holds two requests of 4 slots each, so the drafters now draft for both, each request in a **draft group** with
its own state (`MIMO_SPEC_SLOTS=8 MIMO_SPEC_GROUPS=2`, `reference/spec_decode.py`), K = 3 each:

```bash
export MIMO_SPEC_DECODE=mtp MIMO_SPEC_SLOTS=8 MIMO_SPEC_GROUPS=2   # or dflash
vllm serve ... --max-num-seqs 8 \
  --speculative-config '{"method": "custom_class", "model": "vllm_tt_plugin.model_owned_drafter", "num_speculative_tokens": 3}'
```

- **Groups.** A request keeps its group while it runs: the adapter assigns one at its prompt (a free group, else the
  one whose request ran least recently, `generator_vllm._draft_groups`) and places its verify columns there
  (`plan_verify_passes`); the accept page's groups follow (`pass_rows`). A lane of more requests than groups drafts
  for none.
- **Layouts** (`reference/spec_chain.py`, `kernels/spec_chain.cpp`): the chain row, the result page's status and
  drafts sections, and the MTP descriptions have a block per group; every chain step runs per group (one group: the
  layouts of before).
- **MTP**: one ring per group (the page table maps a group's slots to its blocks) and the group's stash slots; one
  program drafts both groups, each from its own description; a request's first draft backfills its group's ring from
  the history (prompts reset nothing).
- **DFlash**: one context ring per group (the prompt writes its request's), the block's attention one decode-SDPA user
  per group, the drafts in GreedyHead rows g * 3 ...
- **Draft programs compile before any trace capture** (`TtMiMoMTP.warm`, `TtDFlash.warm`, in
  `TtMiMoModel.capture_decode_trace`). The tiled reshape keeps a device mapping table, allocated when its program
  compiles; compiled after the decode and verify captures, it sat where those traces write on replay, and the
  two-group DFlash draft hung on its first replay in serving (tt-triage: every reshape reader in `cb_reserve_back`).

c=8 (four lanes of two requests), 32 SPEED-Bench coding prompts, 512 output tokens (BH Galaxy, 2026-10-24; output
tokens/s, TPOT, tokens/step):

| configuration | greedy | sampled (T 1.0, top_p 0.95) |
|---|---|---|
| no speculation | 554, 12.44 ms | 571, 12.05 ms |
| 3 MTP layers, 4-slot verify (default) | 418, 16.50 ms, 3.44 | 351, 20.08 ms, 3.18 |
| 3 MTP layers, draft groups (K = 3) | **807, 8.65 ms, 3.42** | **733, 9.29 ms, 3.18** |
| DFlash, draft groups (K = 3) | 707, 9.31 ms, 3.34 | 670, 9.74 ms, 3.17 |

With the 4-slot verify a lane of two requests verifies without drafts (only lanes left with one request draft), and
the speculative path costs more than plain decode: 25% fewer tokens/s than no speculation. With draft groups every
request drafts, at its c=1 acceptance: +46% greedy, +28% sampled over no speculation. DFlash accepts slightly less than
the MTP layers at K = 3 and prefills one row at a time (TTFT 695 vs 391 ms). The steps are host-bound: the verify's
host side takes ~14 ms per step for the four lanes (page tables 2.4 ms) against ~20 ms of device work, so that host
work is the next lever. At c=1 a draft-group launch runs the 8-row kernels for one request (use the 4-slot launch).

### One MTP layer per draft

The checkpoint has 3 MTP layers (`num_nextn_predict_layers`). Upstream vLLM drafts all 3 tokens with layer 0; the GPU
references come from vLLM 0.31.0 patched to expose all 3 layers, which selects `MultiModuleMTPSpeculator`: draft i + 1
from layer i. The layers are not chained: each takes the target's hidden (SGLang's multi-layer EAGLE runs MiMoV2 the
same way, `chain_mtp_hidden_states` off). The device drafts that way by default (`MIMO_MTP_LAYERS=3`); `1` restores the
one-layer drafter unchanged (same programs and uploads), for A/B. The host reference (`reference/spec_decode.py`
`plan_mtp_layers`, `reference/mtp.py` `ReferenceMTPDrafter`) is checked against a recomputation of every layer over the
whole sequence after steps of every accepted count (`tests/test_spec_decode_host.py`).

- **Equations.** Committed tokens x (the last at B), drafts d, x~ = x then d_1, d_2. Layer i at position q takes the
  target's final normed hidden h_q and the embedding of x~[q + i + 1], RoPE at q, K / V at q in its own cache; draft
  i + 1 is its greedy token at row B - 1. Every layer's cache holds, at each q <= B - 1, the entry of
  (h_q, x~[q + i + 1]).
- **Rows.** Each draft runs the 3 layers over the same rows: the committed rows A .. B - 1 (A the verify's column-0
  position) and, before them, up to max(0, 3 - m) re-prefill rows: layer i's rows A - i .. A - 1 took the last
  draft's d_1 .. d_i, stale once rejected. 3 rows, 4 when all 3 drafts were accepted: one 4-slot pass per layer,
  right-aligned, the anchor (the row at B - 1) in slot 3. Layer i's tokens are the layer-0 tokens shifted by i, the
  drafts so far appended. The re-prefill rows come from the stash: every row of the last draft's pass (target
  hidden, layer-0 token, position), found by position, consecutively from A - 1 down. Keeping all 4 rows (not only
  the last 2) makes re-planning a commit idempotent: a draft planned again from the commit it ran on (the chain
  check, or a host plan after the device chained a different commit) finds the same rows.
- **Device program** (`tt/mtp.py` `_body_layers`, kernel modes 3 and 4 of `kernels/spec_chain.cpp`, integer mirror
  `reference/spec_chain.py` `plan_step`, `tokens_step`, whole-program mirror `reference/mtp.py` `MtpProgramMirror`).
  Which rows a draft runs depends on the stash, which only the device keeps, so the packed upload describes the commit
  instead of the passes (`pack_mtp_layers`: drafting, m, A, the source's slots, the committed tokens), written by the
  chain step behind a verify or by the host after a plain step, a logits verify or a differing commit. The draft trace
  then runs:
  1. the plan step (one core, ~10 us): plan_mtp_layers from the description and the stash page; writes the passes'
     persistent inputs (`LayerInputs`: ring cache positions, 4 serial positions, RoPE indices, the hidden table's
     rows, the layer-0 tokens) and the new stash page;
  2. one RoPE gather (the SWA tables at the RoPE indices), shared by the 3 passes;
  3. the hidden gather: the table [final norm buffer rows, stash rows] (row-major, 64 x 4096) indexed by the plan's
     rows (`ttnn.embedding`, an exact copy); the gathered rows become the stash's hidden rows;
  4. pass i on layer i over the 4 slots (eh_proj, the SWA block writing K / V slot by slot, final norm, LM head,
     GreedyHead), before passes 1 and 2 the tokens step (one core) writing that pass's tokens from the layer-0
     tokens and the anchor's greedy token of the passes before;
  5. the drafts step (the anchor is slot 3).
  An idle description (warm-up, capture, a declined chain) writes no K / V and leaves the stash as it was: idle slots
  gather their own stash row. A prompt resets the rings and the stash page. Every buffer the kernel steps touch is
  persistent and made in `TtMiMoMTP.setup_cache`, so they compile in `SpecChain.warm` before the decode capture.
- **Cost.** One layer: a fill pass (<= 4 slots) and 2 one-slot passes, all on layer 0, ~2.7 ms on device. Every layer:
  3 passes of <= 4 slots on layers 0, 1, 2: the same op graph and weight bytes per pass (each layer ~46 MB per chip
  at bfloat8_b), plus the serial K / V writes of up to 4 slots in passes 2 and 3, the plan, tokens and gather ops
  (~0.1 ms), minus the one-layer derive program's ~16 small ops: about 2.8-3.0 ms (to measure; `MIMO_VLLM_TIMING=1`
  reports it as the chained draft wait). Memory per replica: 2 more layers, ~0.74 GB (92 MB per chip; eh_proj 8192 x 4096, SWA
  q / k / v / o, dense MLP 4096 -> 16384 -> 4096), 2 more ring caches (~0.2 MB per chip each), the stash (256 KB of
  hidden rows per chip, one 128-byte page) and the inputs (9 pages of 128 bytes).
- **Tests.** `tests/test_spec_chain_host.py`: the chain step's description against propose's, the plan step against
  plan_mtp_layers for every commit count, a plain step's slot, stash hits, misses and holes, the ring's wrap and the
  context end, the tokens step against mtp_layer_tokens, and the whole program's mirror against the reference
  drafter (exactly equal drafts and outputs, equal to the recomputation) over every pair of accepted counts and plain
  steps. `tests/test_spec_chain_adapter_host.py`: the adapter on fake lanes running the program's mirror: every
  chained draft equals the host reference drafter's for the plugin's commit, the chain check finds no difference, a
  differing commit drafts the reference's tokens from the host description. `tests/test_spec_inputs_tp8.py`
  `test_spec_chain_layers_tp8` (Galaxy row): the chain, drafts, plan and tokens modes bit for bit against the mirror,
  eager and traced.
  `tests/test_spec_decode_tp8_dp4.py` (Galaxy): speculative greedy == plain greedy, and chained == host-planned
  drafts, with 3 layers and with 1.
- **Plugin.** No change: still 3 drafts per step.

### MTP ring backfill (target-hidden history)

The MTP rings only hold what the lane's drafts wrote, while vLLM's MTP layers see every position. Two gaps: a new
request's rings start empty (the prompt's rows are never drafted), and a lane serving several requests decodes
without drafting, so when it drops back to one request the rings miss the positions in between; in both cases the
first ~128 drafts attend a window with missing or stale entries. `MIMO_MTP_BACKFILL=1` (default, with the decode
RoPE tables; `tt/mtp_backfill.py`, `kernels/mtp_history.cpp`, host mirror `reference/mtp_backfill.py`) closes both
with one mechanism, for 1 and 3 MTP layers:

- **History.** Each decode slot that may speculate (4 per lane) keeps a ring of 132 rows (the MTP window and a
  verify pass: a backfill right after a verify at A reads A - 127 .. A - 1 while the pass's rows A .. A + 3 are in
  the ring), indexed by position % 132: the target's final normed hidden (bf16, from the final norm's persistent
  buffer), the position and the token the step took there. Every plain decode step writes its slots' rows (in the
  decode trace: a copy of the step's input tokens at its start, an untilize of the final norm tile and one one-core
  kernel at its end); every verify pass writes its slots' rows into their requests' rings (in the verify trace; the
  chain row carries each slot's request slot), rows past the committed ones being rewritten by the next step, which
  starts there; every prefill chunk writes the final norm of its last 132 rows (eager chunks, and the traced
  first-chunk batches through a persistent write list) and invalidates the slot's other entries unless it continues
  its request.
- **Backfill.** Before a host-planned draft whose request's rows the rings do not continue (`mtp_ring`: the lane's
  rings hold that slot's rows up to A - 1; after a prompt, after steps without drafts) the adapter replays the
  backfill trace: a plan step (the rows of the whole ring, 256 positions from the block holding A - 127, real where
  the history holds h_q, layer i taking the token the history holds at q + i + 1), one gather of the history rows and
  their cos / sin, then per MTP layer its input over the 256 rows (enorm / hnorm, eh_proj, the block's input norm)
  and its K / V written into the whole ring by one `paged_fill_cache` (`TtMiMoAttention.fill_kv_rows`: no attention,
  MLP or LM head). The other rows write garbage where no query reads before a draft rewrites them (below the window,
  at A and past it). With 3 layers the draft right after commits m tokens x[A + 1 .. A + m] and re-runs the
  max(0, 3 - m) rows before A in every layer, from the stash, where the plan puts rows A - 2, A - 1 (hidden, layer-0
  token, position); every other row takes tokens at positions <= A + m - 1, which the history holds committed (after
  a verify at A, its rows A .. A + m - 1), so the plan needs nothing of the commit. A draft chained behind a verify
  the host built checks the rings the same way (the backfill then runs between the accept trace and the draft: a
  prompt over a deferred verify resets the rings while the request goes on); one chained behind a relayed verify
  continues the draft before it. Under async scheduling the ring a chained draft left is known once its page is read
  (`mtp_ring` holds the verify's handle until then), and a propose the plugin made stale drafts nothing, so the
  backfill comes with the propose that does.
- **Exactness** (host, `tests/test_mtp_backfill_host.py`): a lane serving request X (its prompt in chunks, a plain
  step, speculative steps with every accepted count), then X beside a second request for 3, 12 or 40 plain steps
  (its prompt resets the rings), then X alone again: every draft equals the drafts of an MTP that saw every position
  (a recomputation over X's whole sequence), for 1 and 3 layers; without the backfill the first drafts differ. A
  backfill right after a verify (a prompt reset the rings while X speculates) leaves every layer's K / V over the
  window the draft reads equal to that recomputation's, for m = 1 .. 4 and 1 and 3 layers, with the history sized as
  production sizes it (`history_rows`: the window and a verify pass).
  `tests/test_spec_inputs_tp8.py` `test_mtp_history_tp8` checks the kernel's four modes bit for bit on a Galaxy row;
  `tests/test_spec_decode_tp8_dp4.py` runs with and without it.
- **Cost.** Per decode step ~10-20 us (one copy, one untilize of [32, 4096], one kernel moving 4 rows of 8 KB and a
  4 KB page) against a ~13 ms step; per verify pass one kernel; per prefill chunk a final norm and untilize of its
  rows and one kernel; per backfill (once per transition: a new request's first draft, a lane back to one request)
  up to 127 real rows x 1 or 3 layers, as 256-row passes: about 1 ms on device (to measure). Memory per chip: 4.3 MB
  of history rows (4 slots x 132 x 4096 bf16), a 4 KB page, ~6 KB of backfill inputs (and 2 x 256 x 128 bf16 cos /
  sin tails with compact RoPE tables).

## Not done

- The KV pair split: a 4 x 1M per lane run (16 prompts of 1M tokens; a 1M prefill takes about an hour per lane), an
  accuracy eval with the split on (only greedy agreement and the APC check so far).
- Slot parity of chunked prefill with `MIMO_KV_PAIR_SPLIT=0` (above).
- Device sampling beyond greedy (top-k / top-p / temperature, penalties, logprobs run on the host, synchronously).
- A 1M-token prompt, and the 1M server with the compact device RoPE tables (not yet run on hardware).
- Prefill and decode in the same step for different lanes (the plugin runs a step as all-prefill or all-decode); one
  engine per row ("One engine per Galaxy row") runs them at the same time on different rows.
- Row engines:
  - the plugin HTTP suite and an accuracy eval through the proxy (or a full TTI release run);
  - DFlash;
  - c=16 at 1M.
- Tracing later prefill chunks and chunks longer than 512 tokens.
