# embeddinggemma-2 serving study scripts

The scripts behind the 2026-10-29 serving-path comparison of `google/embeddinggemma-2` (bare tt-metal, a FastAPI
server, tt-media-server Python, tt-media-server C++). Everything ran on one Blackhole chip of a Galaxy (86:00.0,
PCIe x8, 12x10 grid), 160 W, `TT_MM_THROTTLE_PERF=0`, with this commit's tt-metal (`a12d330bb3c`, Gio's
`gtobarTT/embedding_gemma2_bringup`) and tt-inference-server `gtobarTT/embedding_gemma2` @ `e1fcaa29`.
Absolute numbers from a Galaxy chip do not transfer to a P150 (13x10 grid, x16); the per-path deltas do.

These files are a study harness, not a product. Paths default to the original host; override them with env vars
(see `env_emb.sh`): `EMB_ROOT` (caches, HF home, logs, results), `EMB_TT_METAL` (tt-metal checkout with
`python_env`), `EMB_TTIS` (tt-inference-server checkout), `EMB_CHIP` (`TT_VISIBLE_DEVICES` for bare and FastAPI),
`DEVICE_IDS` (media servers, e.g. `"(0)"`), `EMB_DEV` (device node the chain scripts wait on), `UV_PYTHON_LIB`.

## Files

| File | What it does |
|---|---|
| `env_emb.sh` | sourced by every script: venv, chip, 160 W, caches, HF offline, model revision `914f7f89`, throttle 0 |
| `emb_common.py` | model id, revision, the (batch, tokens) points, `make_texts` (texts that tokenize to exactly N ids incl. BOS/EOS) |
| `emb_bare.py` | bare path: HF tokenizer call + `EmbeddingGemma2ForEmbedding.forward` + `ttnn.synchronize_device`, per-call p50s, writes golden embeddings |
| `emb_client.py` | one-at-a-time client: `POST /v1/embeddings`, one string per request, 20 reps after 3 warmups, p50/p90, min cosine vs the bare JSON |
| `emb_load.py` | closed-loop load: N clients, each sends the next single-text request when its reply lands; 5 s warmup + 30 s |
| `emb_serve_fastapi.py` | the FastAPI server (Ashai's template via Gio's `bge_m3/demo/serve.py`, worker swapped to embeddinggemma-2, plus `--max-batch`) |
| `run_fastapi_dp.sh` / `emb_load_mp.py` / `emb_hf_ref.py` / `emb_check_dp.py` / `emb_shape_soak.py` | DP=32: launcher, multi-process load client, HF fp32 reference, per-replica check, model-only shape soak (see DP=32 below) |
| `run_fastapi.sh` / `run_media_py.sh` / `run_media_cpp.sh` | start each server (ports 8101 / 8102 / 8103) with the exact env used; print pid + log |
| `build_cpp_server.sh` | cpp_server build (private rustup >= 1.80, empty HF token so gated tokenizers are skipped, `CARGO_EXECUTABLE` pinned) |
| `compare_chain.sh` | bare (timing off, on), then each server in `SRVS` (timing off, on) with the same client and cosine gate |
| `fanout_chain.sh` | batched points as B concurrent single-text requests per server |
| `load_chain.sh` | load 1/8/32 in flight x 128/512 tokens per server |
| `fastapi_bs_rerun.sh` / `fastapi_bs32_repro.sh` / `fastapi_bs32_gapfill.sh` | the FastAPI batch-32 runs: same-window controls, one load point per fresh server |
| `emb_hang_repro.py` / `hang_repro_watch.sh` | model-only repro of the intermittent readback hang (mixed batch-1 / batch-8 trace replays) |
| `text_encoder_timing.patch` | optional env-gated per-phase timing hooks for `tt/text_encoder.py` (`EMB_TIMING_LOG`); off for every reported number |

## How each number was produced

```bash
source env_emb.sh                       # after setting the overrides above
SRVS="fastapi media_py media_cpp" ./compare_chain.sh                 # one at a time, all paths + cosine gate
EMB_FASTAPI_ARGS="--max-batch 32" SRVS=fastapi ./compare_chain.sh    # FastAPI b32 one at a time
SRVS="fastapi media_py media_cpp" ./load_chain.sh $EMB_ROOT/results/<dir>   # load 1/8/32 x 128/512
CONCS="8" TOK=128 MAXB=32 ./fastapi_bs32_repro.sh                    # one FastAPI b32 load point, fresh server
```

## Reading the numbers

- **"FastAPI b32"** is the server setting `--max-batch 32`: the most queued texts one forward may take. With one
  request in flight every forward carries one text. `--batch-wait-ms` defaults to 0 (the worker never waits to fill a
  batch). b1 is the template default (one text per forward).
- **Bare includes host work.** Bare e2e = HF tokenizer `__call__` (`padding=True, truncation=True,
  return_tensors="pt"`) + `forward` (host pad, two H2D copies, trace replay, D2H) + device sync. Both parts are
  reported separately (`tokenize_ms_p50`, `forward_ms_p50`).
- **Tokenizers differ by path.** The FastAPI server tokenizes with the Rust backend directly
  (`backend_tokenizer.encode(text).ids`), which is much cheaper than the HF `__call__` bare uses (~36 ms of bare's
  ~141 ms at 8,192 tokens). That is why FastAPI ties bare at 8,192 tokens: its tokenizer saving cancels its ~4 ms
  serving overhead. At short inputs the tokenizer cost is small and the overhead shows.
- **C++ server batch delay.** `MAX_BATCH_DELAY_TIME_MS` was not set (default 2 ms); every batch, including a lone
  request, waited ~2.07 ms to collect (`Dispatch summary` lines in the server log). `USE_DYNAMIC_BATCHER` is read only by
  the Python backend.
- **Clients** are Python on the same host as the server, keep-alive sessions; the same client reached 402 req/s on
  FastAPI b32, so it does not limit the C++ (247 req/s) or media Python (288 req/s) points.

## Versions

The first commit on this branch is the exact code that produced the 2026-10-29 report (paths made overridable, same
defaults). The next commit replaces `emb_serve_fastapi.py` with a hardened version (non-blocking start-up, worker
watchdog, validation, `emb_batcher.py`, CPU tests in `test_emb_serve.py`); it has passed CPU tests only and has not
been measured on a chip.

## DP=32 on a Blackhole Galaxy (2026-11-17)

`run_fastapi_dp.sh` serves one replica per chip on all 32 chips of a BH Galaxy (8x4), host u20-40 (10.82.97.40),
same model commit as above, 160 W, `TT_MM_THROTTLE_PERF=0`:

```bash
source env_emb.sh                    # EMB_ROOT etc. as above
DP=32 FRONTENDS=8 ./run_fastapi_dp.sh  # 8 server processes on port 8132, 4 chips each; ~25 s to ready with warm caches
```

What changed in `emb_serve_fastapi.py`:

- **Least-loaded dispatch** (`emb_batcher.pick_chip`). The old pool queued `max_batch` slots per chip in a row, so
  with `--max-batch 32` the first 32 requests all went to chip 0 as one batch while the other chips idled. A chip now
  gets a second text only when every ready chip holds one; requests beyond `dp x max_batch` wait in arrival order.
- **`--max-batch 1` for DP, and the trace ladder follows it.** Traced multi-row forwards hang intermittently on Galaxy
  chips: `emb_hang_repro.py` with batch-8 only (one trace, no switching) hung on 1 of 2 chips; mixed b1/b8 hung on
  3 of 4; batch 1 passed 8000 forwards on 2/2 chips, `emb_shape_soak.py` batch 1 at random lengths 2-2048 passed
  6000 forwards on 8/8 chips, eager b8 passed 2/2. The hang is inside the trace (host waits in `to_torch`), not in the
  server, and is not caused by trace switching, the 160 W cap or throttle 0. The worker now builds the model with
  `max_batch_size = --max-batch`, so a b1 server captures only b1 traces.
- **One event loop serves ~700 req/s**; 32 chips at b1 can do ~5,200 texts/s at 128 tokens (6.1 ms device each).
  `--reuse-port` lets several servers, each with its own `--chips`, share the port (`FRONTENDS=K` in the launcher);
  responses are encoded with `orjson` + numpy (the FastAPI default built 768 Python floats per row).
- `--chips` defaults to `0..dp-1`; chip 0 builds first to fill the weight/kernel caches (`--no-warm-first` to skip);
  `/metrics` has `texts_per_chip`; `--debug-chip` adds the serving device to each row (validation only).

Load: `emb_load_mp.py` (16 aiohttp client processes, closed loop, 5 s warmup + 20 s, same host). req/s (p50 ms),
single-text requests, 128 tokens:

| Server | 1 in flight | 8 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|---|
| original, 1 chip, `--max-batch 1` | 105 (9.5) | 121 (67) | 120 (264) | | | | |
| original, 1 chip, `--max-batch 32` | 88 (11.4) | hang | | | | | |
| original code, `--dp 32` | 101 (10.4) | | 673 (46) | | 633 (192) | | |
| DP=32, 1 front-end, before orjson | 97 (10.4) | 614 (12.6) | 683 (45) | 664 (92) | 644 (189) | 628 (382) | 610 (775) |
| DP=32, 1 front-end, orjson | 115 (8.7) | | 1,001 (34) | | 778 (154) | | |
| **DP=32, 8 front-ends** | **121 (8.3)** | **947 (8.4)** | **3,648 (8.0)** | **4,322 (14)** | **4,351 (29)** | **4,347 (58)** | **4,297 (118)** |

512 tokens: original 1 chip b1 76 req/s (13.3 ms) at 1 in flight, 88 (357 ms) at 32; DP=32 x 8 front-ends 81 (12.4)
at 1, 2,543 (11.9) at 32, 3,024 (19) at 64, 3,095 (43) at 128 = 1.58 M tokens/s. Peak DP=32 / original 1 chip:
36x at 128 tokens, 35x at 512. List requests of 32 x 128-token texts: one request takes ~62 ms (it reaches one
front-end, 4 chips, 8 b1 forwards each); 16 such requests in flight give 4,302 texts/s. Soak: 120 s at 256 in flight,
517,781 requests, 4,315 req/s, 0 errors.

Correctness (`emb_hf_ref.py` + `emb_check_dp.py`, server with `--debug-chip`): 22 texts (4-2048 tokens) on every one
of the 32 chips, cosine vs HF fp32 (transformers 5.20.2 overlay) min 0.99943, mean 0.99965, the 32 chips
bit-identical; 2,816 rows of concurrent list requests, same min. Gate 0.99.
