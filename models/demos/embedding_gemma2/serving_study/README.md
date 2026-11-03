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
