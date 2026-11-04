# DeepSeek V4.1 Flash Server: C++ Port (Master Plan)

> **For agentic workers:** this is the umbrella plan. Each phase below is its own sub-project and gets its own
> bite-sized TDD plan (`docs/superpowers/plans/2026-11-04-dsv41-cpp-phase<N>-*.md`) before implementation starts.
> Use superpowers:subagent-driven-development or superpowers:executing-plans per phase.

**Goal:** Serve DeepSeek V4.1 Flash's OpenAI/DeepSeek-compatible HTTP API from a C++ process with the same wire
behaviour as `models/experimental/deepseek_v41_flash/demo/server.py`, removing the Python GIL from the
token-to-packet hot path.

**Architecture:** Port in layers, outermost first, behind one narrow `ModelBackend` interface. The C++ server
(HTTP, scheduler, sessions, streaming, files, image bookkeeping) first drives the existing Python model through
embedded CPython; then the decode hot path (packet write, Engram lookup, D2H read, sampling, trace replay) moves
to C++ against ttnn's C++ socket/trace APIs, while Python still builds the model and captures traces at startup.
A full C++ model (layers, prefill, vision) is an optional final phase.

**Tech Stack:** C++20, tt-metal/ttnn C++ API, nlohmann_json, Boost.Asio (already in `third_party/`),
cpp-httplib or Boost.Beast (new dependency, see Decision D2), HF `tokenizers` via tokenizers-cpp (new
dependency), spdlog/tt-logger, nanobind embedding (already in `third_party/`), GoogleTest.

**Spec:** the behaviour of the Python server itself: the docstrings of
`models/experimental/deepseek_v41_flash/demo/server.py` and `models/experimental/deepseek_v4_flash/demo/server.py`
are the spec. The Python server stays in the tree as the reference implementation and test oracle.

## Global Constraints

- Build only through `./build_metal.sh`; the new target is added to the CMake tree it drives.
- Any test that opens the device (server against real model, ttnn socket tests) is run by the user, never by the
  agent; the agent supplies the exact command.
- No thin wrapper helpers (`.cursor/rules/no-thin-wrappers.mdc`): call real constructors/APIs at call sites.
- Wire compatibility: same routes, JSON shapes, SSE framing, error envelope, status codes and `usage` fields as the
  Python server. `models/experimental/deepseek_v4_flash/demo/client.py` must work unchanged.
- Defaults copied verbatim: `--num-users 64`, `--max-context 1048576`, `--total-context num_users * 204800`,
  `--max-running 8`, `--decode-ahead 32`, `--image-rows 40960` (multiple of 32), `L1_SMALL 10240`,
  `--prefill-kv-only` on, `--vision` on, vision stage 5, 64 images/request, 64 MiB/upload, 4 GiB uploads total,
  file lifetime `[3600, 2592000]` s, model id `deepseek-v4.1-flash`, device sampling on.

---

## 1. What `server.py` (V4.1) actually is

`deepseek_v41_flash/demo/server.py` (1343 lines) is a set of subclasses over the V4 server; porting it means
porting the whole inherited stack:

| Source | Lines | Responsibility |
|---|---|---|
| `deepseek_v41_flash/demo/server.py` | 1343 | V4.1 engine/prefiller, images, Files API, `/models`, DeepSeek `usage` |
| `deepseek_v4_flash/demo/server.py` | 3774 | HTTP handler, SSE, scheduler, slot pool, output reader, turns, prefill policy, args |
| `deepseek_v4_flash/demo/chat_cli.py` | 1065 | `ChatEngine`, `UserSession`, `ChatFormat`, mesh open, decode step plumbing |
| `deepseek_v41_flash/encoding_dsv41.py` | 983 | Chat template rendering, DSML tool-call tags, image placeholder layout |
| `deepseek_v4_flash/demo/tui.py` | 1027 | Live console |
| `deepseek_v41_flash/tt/engram_lookup.py` | 259 | Host n-gram hash + gather over ~101 GB tables per Engram layer |
| `deepseek_v41_flash/tt/model.py` (+ V4 base) | 3135 + 4775 | Device model; the server uses ~45 of its methods |

### 1.1 Components in the V4.1 file

- **`_V41Handler`** (`_Handler` subclass): adds `GET /models`, `/v1/models`, Files API (`GET/POST/DELETE /files[/<id>]`,
  also `/v1/files`), chunked upload read with progress, DeepSeek `usage` (cache hit/miss split, reasoning tokens) on
  the last SSE chunk.
- **`V41GenerationServer`** (`GenerationServer` subclass): image parts in user messages (`image`, `image_url`,
  `file`; base64 or `file_id` only, never URLs/paths), placeholder blocks tagged with image sha1, per-request
  image/row limits, `_render_ids` expanding each placeholder into its span of token ids past the vocab,
  `_pinned_images` (images any conversation holds), `upload_file` (encode on upload via a scheduler chore).
- **`_FileStore`**: uploads by id, expiry, optional on-disk persistence (`<id>.bin`, then `<id>.json` written
  atomically last), byte budget, paginated listing.
- **`_ImageBook`**: sha1 -> `_Image` (token types, prepared ViT patches or cached rows, embedding-table slots), LRU
  eviction of unpinned images, free-slot deque over `image_rows - FIRST_IMAGE_ROW`, ViT encode + embedding-row
  write under the model's CQ lock (only with no decode step in flight), `vision_rows/<sha1>.pt` cache keyed by
  processor settings.
- **`_ImageStatus`**: live/done image pipeline states for the console.
- **`V41ChatEngine`** (`ChatEngine` subclass): builds config, tokenizer, RoPE, `DeepSeekV41Model`, `lm_head`,
  vision tower; `prepare_static_decode` (paged pools, sessions, device sampler); **packet-writer thread**
  (`write_traced` / `repeat_traced` / `write_prefed` keep packets in posting order; V4.1 packet writes also do the
  Engram host lookup for layers 1 and 14); `capture_vision` as the last device allocation of startup.
- **`_V41Prefiller`**: builds the traced prefill model (`build_prefill`, KV-only = layers 0-20), alignment rules
  (last 128 tokens always left to decode replay), continuation from session caches (`seed_prefill`,
  `commit_prefill_state`, bounded replay).
- **`parse_args` / `main`**: V4 parser + V4.1 defaults, then `serve(...)`.

### 1.2 Inherited V4 machinery (where the complexity is)

- **`_Scheduler`** (~1110 lines): the single device-driving thread. Admission, chores (`run_on_scheduler`), rounds,
  `_inflight` dispatch order, decode-ahead posting (`_post_ahead`, interleaved across turns), prefed steps,
  `_settle` before any session rewrite, traced-prefill vs decode-replay choice (`_choose_feed`, using a measured
  replay rate), resume/restore/offload of sessions to host RAM, cancellation (`_stop_prompt`, `_cut_short`),
  abort/drain on failure.
- **`_OutputReader`**: D2H reader thread; samples and writes the next packet itself when armed (`PREFED`) so the
  scheduler is off the token path; strict posting-order invariants.
- **`_SlotPool`**: slot states (running/queued/parked/empty/host), conversation matching (`_conversation_key`,
  `_continues`), idle-timeout eviction, host-RAM budget, 429 on exhaustion.
- **`_Turn`**: per-request state, timing, rates, cancellation event, restore info.
- **`_Streamer`**: incremental detokenization, `<think>` split into `reasoning_content`, DSML tool-call block
  hold-back and parse into OpenAI `tool_calls`.
- **`GenerationServer`**: `generate`, `_relay` (client-disconnect polling), `_sync_messages`, `_render_ids`,
  session reset/cancel APIs, stats.
- **`_Handler`**: `/v1/chat/completions`, `/v1/completions`, cancel routes, `/v1/sessions`, `/health`, CORS
  `OPTIONS`, SSE, OpenAI error envelope, `X-Request-Id`.
- **`UserSession` / `ChatEngine`** (chat_cli): per-slot conversation, render cursor (`_next_render`), feed, post/
  write/read traced steps.

### 1.3 Thread model to preserve

HTTP threads (never touch device) -> `_Scheduler` thread (only thread that seats/posts/rewrites sessions) ->
model replay thread (issues `execute_trace`) -> packet-writer thread (H2D + Engram) -> output-reader thread (D2H,
sample, prefed packet). Plus TUI thread. Invariants: packets written in posting order; outputs read in dispatch
order; session rewrites only after `_settle`; image encode only with `_replay_q` empty, under `_cq_lock`.

### 1.4 Model surface used by the server (the `ModelBackend` contract)

Startup: constructor, `prepare_static_decode`, `build_prefill`, `compile_traces`, `compile_traced_prefill`,
`capture_traced_prefill`, `snapshot_resident_state`, `prefetcher_session`, `shutdown`, `vision.capture`,
`EmbeddingRowWriter`, `write_embedding_rows`.
Sessions: `open_session`, `activate_session(s)`, `reset_session`, `truncate_session`, `end_session_at`,
`export_session`, `import_session`, `session_usage`, `session_tokens_left`, `ensure_session_capacity`,
`context_limit`.
Decode hot path: `replay_traced_ahead` / post, `write_step_packet`, `read_decoded_output`, `decode_traced`,
`synchronize`, `device_sampler.set_sampling`.
Prefill: `prefill_traced`, `seed_prefill`, `commit_prefill_state`, `continuation_tokens`, `continue_alignment`,
`release_traced_prefill`, `release_prefetch_buffers`, `restore_prefetch_buffers`, `reload_resident_state`.
Vision: `vision.encode`, `row_writer.write`.

---

## 2. Decisions to make before Phase 1

Decided 2026-11-04: D1 = Phases 0-3, D2 = cpp-httplib via CPM.

- **D1 - Scope / boundary.** Recommended: Phases 1-3 (C++ serving + C++ decode hot path, Python still builds the
  model and captures traces). A full C++ model (Phase 5) is ~8k lines of ttnn Python plus prefill and vision and
  is a separate project.
- **D2 - HTTP library.** Boost is vendored with `asio` but not `beast`. Options: add `beast` to
  `BOOST_INCLUDE_LIBRARIES` in `third_party/CMakeLists.txt`, or add header-only cpp-httplib via CPM. Recommended:
  cpp-httplib (thread-per-connection matches the Python `ThreadingHTTPServer`, has chunked/SSE writers and
  multipart parsing).
- **D3 - Tokenizer.** tokenizers-cpp (Rust `tokenizers` behind a C API; needs a Rust toolchain in the build) vs a
  hand-written BPE over `tokenizer.json`. Recommended: tokenizers-cpp, with a golden test over a corpus against
  `transformers.AutoTokenizer`.
- **D4 - Image preprocessing.** Keep `image_processor` in embedded Python (Phase 1-3) vs port decode/resize
  (libjpeg-turbo, libpng, libwebp, giflib) bit-exactly. Recommended: keep in Python until Phase 4.
- **D5 - Live console.** Drop for v1 (plain spdlog) vs port `tui.py` with FTXUI. Recommended: drop for v1,
  expose the same data at `GET /v1/stats` JSON.
- **D6 - Location.** `models/experimental/deepseek_v41_flash/cpp/` with its own `CMakeLists.txt`, added from the
  top-level tree behind `option(TT_BUILD_DSV41_SERVER OFF)` so default builds are unaffected.

---

## 3. Target layout

```
models/experimental/deepseek_v41_flash/cpp/
  CMakeLists.txt
  server/
    main.cpp                 # args, startup order, signal handling
    args.{hpp,cpp}           # every CLI flag + env var of build_parser/parse_args, same defaults
    http_handler.{hpp,cpp}   # routes, SSE, error envelope, CORS, multipart, client-gone detection
    generation_server.{hpp,cpp}  # generate/relay/sync_messages/render_ids/cancel/stats (+ V4.1 images)
    messages.{hpp,cpp}       # normalized_messages, content/text/tool normalization, conversation_key, continues
    streamer.{hpp,cpp}       # incremental detok, think split, DSML hold-back and tool-call parse
    sampler.{hpp,cpp}        # temperature/top_p host sampler, argmax
    slot_pool.{hpp,cpp}      # slot states, matching, host-RAM conversations, idle timeout
    turn.{hpp,cpp}           # per-request state and rates
    scheduler.{hpp,cpp}      # the device-driving thread (port of _Scheduler)
    output_reader.{hpp,cpp}  # D2H reader thread with prefed packets
    packet_writer.{hpp,cpp}  # ordered H2D packet thread (V41ChatEngine._write_packets)
    prefill_policy.{hpp,cpp} # _Prefiller/_V41Prefiller alignment + choose_feed math
    file_store.{hpp,cpp}     # Files API store
    image_book.{hpp,cpp}     # sha1 -> slots, LRU, pinned, rows cache
    model_record.{hpp,cpp}   # /models body, usage
  chat/
    encoding_dsv41.{hpp,cpp} # port of encoding_dsv41.py (render, merge tool msgs, sort results, tags)
    tokenizer.{hpp,cpp}      # tokenizers-cpp wrapper with incremental decode state
  backend/
    model_backend.hpp        # the abstract contract of section 1.4
    python_backend.{hpp,cpp} # embedded CPython calling DeepSeekV41Model (Phase 1)
    native_decode.{hpp,cpp}  # C++ hot path over ttnn sockets/traces (Phase 3)
    engram_lookup.{hpp,cpp}  # C++ hash + gather over the host tables (Phase 3)
    fake_backend.{hpp,cpp}   # host-only fake honouring the ordering contracts, for tests
  tests/
    test_encoding_golden.cpp, test_tokenizer_golden.cpp, test_streamer.cpp, test_messages.cpp,
    test_slot_pool.cpp, test_scheduler_fake.cpp, test_output_reader.cpp, test_file_store.cpp,
    test_image_book.cpp, test_engram_golden.cpp, test_http_conformance.py
  tools/
    dump_goldens.py          # runs the Python reference to emit golden JSON for the C++ tests
```

---

## 4. Phases

### Phase 0 - Baseline and oracle (host-only plus one user-run device session)

Deliverable: numbers to beat and goldens to match.

- Instrument nothing new; capture from the existing server's debug lines and live console: per-user tok/s at 1, 8,
  32 users; host gap from output read to next packet (`_window_gap`); dispatch seconds per step; TTFT for 1k/16k/
  128k prompts. User runs the Python server; agent provides the commands and a load script.
- `tools/dump_goldens.py` (host-only, no device): for a corpus of requests (plain chat, multi-turn, tools, tool
  results, thinking on/off, every reasoning effort, images as placeholders, `/v1/completions`) dump
  `normalized_messages`, rendered prompt text, token ids, `_conversation_key`, `_continues` decisions, and
  `_Streamer` output chunks for recorded token streams. Also dump Engram `hash_session` / `gather` results for
  random token streams (reads the staged tables; host-only).
- Exit criterion: baseline table committed next to this plan; goldens generated reproducibly.

### Phase 1 - C++ serving layer over the Python model

Deliverable: `dsv41_server` binary, wire-compatible, driving `DeepSeekV41Model` through embedded Python.

Tasks (each gets TDD steps in the phase plan):
1. CMake target, `option(TT_BUILD_DSV41_SERVER)`, links ttnn, nlohmann_json, spdlog, nanobind embed, GTest.
2. `encoding_dsv41` port + golden test (render, merge tool messages, sort results, DSML tags, effort aliases
   `medium -> high`, image placeholder).
3. Tokenizer wrapper + golden test (encode, decode, incremental decode with partial UTF-8 hold-back).
4. `messages` (normalization, `_conversation_key`, `_continues`, tool call/tool normalization) + golden tests.
5. `streamer` (think split, DSML hold-back, `_parse_tool_calls`, `reasoning_tokens`) + golden tests.
6. `sampler` (temperature/top_p semantics of `_make_sampler`, argmax; device-sampled int32 page passthrough).
7. `file_store` (expiry, persistence order, torn-upload skip, pagination, byte budget) + unit tests.
8. `fake_backend` honouring: posting-order packets, dispatch-order outputs, session block accounting,
   `PagedCacheFull`, configurable latency per stage.
9. `slot_pool` + tests (running/queued/parked/empty/host, idle timeout, 429).
10. `turn`, `output_reader` (armed/prefed path), `packet_writer` + concurrency tests against the fake backend
    (TSAN build).
11. `scheduler` port + tests on the fake backend: admission, chores, settle-before-rewrite, decode-ahead (lone
    and interleaved), prefed accounting, `repeat_traced` on early stop, cancellation mid-prefill and mid-decode,
    failure -> drain -> 503.
12. `prefill_policy` (`aligned` with KV-only 128-token tail, continuation alignment, bounded replay, choose-feed
    estimate from replay rate).
13. `image_book` bookkeeping in C++ (sha1, LRU, pinned, free slots, `rows_of`, limits), calling Python for
    `load_image`, `vision.encode`, `row_writer.write`, and `.pt` row cache I/O.
14. `generation_server` + `http_handler` + `model_record` (all routes of both servers, SSE, DeepSeek `usage`,
    `stream_options.include_usage`, `X-Request-Id`, cancel routes, client-gone polling, Files API, CORS).
15. `python_backend`: embeds CPython, imports `V41ChatEngine` / `_V41Prefiller` construction code path for
    startup (reusing it verbatim keeps startup ordering: warmup, prefill snapshot, `capture_vision` last), then
    exposes the section 1.4 methods. GIL acquired per call; C++ threads mirror the Python ones.
16. `args` + `main`: every flag/env var of `build_parser` + V4.1 `parse_args`, `check_args` validation.
17. HTTP conformance test (`test_http_conformance.py`): runs the same request corpus against the Python server and
    the C++ server, both on the fake backend (Python side via a host-only fake engine), diffs responses modulo ids
    and timestamps.
18. User-run device acceptance: C++ server on the Galaxy, `client.py` chat, tools, images, file upload, 8
    concurrent users, cancel; compare output tokens to the Python server under greedy decoding.

Exit criterion: conformance diff empty; device acceptance passes; throughput no worse than Phase 0.

### Phase 2 - Remove the GIL from the scheduler loop

Deliverable: scheduler/reader/writer threads call the backend without the GIL on every step.

- Expose `write_step_packet`, `read_decoded_output`, `activate_sessions`, post/replay through nanobind functions
  that release the GIL around the ttnn calls (`nb::gil_scoped_release`), or call the underlying C++ ttnn objects
  directly by extracting them from the Python model (`ttnn.Tensor`, socket handles, trace ids are C++ objects
  behind nanobind).
- Re-measure `_window_gap` and per-user rate at 8 and 32 users.

### Phase 3 - Native decode hot path

Deliverable: `native_decode` + `engram_lookup` in C++; Python only at startup and for prefill/vision.

- After startup, export from Python: submesh devices, per-variant trace ids, H2D/D2H socket objects, packet layout
  (`_step_context` fields: token(s), sliding pos, compress pos, page row, ring offset, window offset), session ->
  slot map, `_variant_key` rules (pos tier, image), Engram table handles.
- Port `_write_packet` (packet fields per session slot) and `_write_engram_rows` (hash per session history,
  gather per layer, zero rows for image-span tokens), `_execute_traces` (`ttnn::execute_trace` per submesh,
  non-blocking, under the CQ lock), `ensure_session_capacity` host bookkeeping, `read_decoded_output` incl.
  `drain_stale_outputs`.
- Engram tables: mmap the staged tables; golden test against Phase 0 dumps.
- Session state ownership moves to C++ for decode (`_session_pos`, `_slot_of`, page tables); prefill and
  export/import still go through Python with an explicit sync point under `_settle`.
- Exit: greedy token-for-token match with Python on the device; host gap and tok/s improved vs Phase 0.

### Phase 4 (optional) - Native images and console

- Port image decode/resize (bit-exact vs `image_processor`, golden test on a corpus of JPEG/PNG/GIF/WebP), replace
  `.pt` row cache with a versioned raw format (migration reads old `.pt` once via Python tool).
- FTXUI console reproducing `tui.py` keys (`d`, `v`, `p`, `c`, `k`, `q`).

### Phase 5 (optional, separate project) - C++ model

- Port `DeepSeekV41Model` build, trace capture, prefill model and vision tower to ttnn C++. Out of scope here;
  needs its own spec.

---

## 5. Testing strategy

- **Golden tests** for every pure-host component (encoding, tokenizer, messages, streamer, Engram), produced by
  the Python reference via `tools/dump_goldens.py`.
- **Fake backend** reproduces the ordering contracts so the scheduler, reader and writer are tested host-only,
  including under ThreadSanitizer.
- **Conformance**: Python vs C++ HTTP diff on the same corpus and fake backend.
- **Device acceptance** (user-run only): greedy token equality vs Python, multi-user throughput, cancel, images,
  Files API, restart with `--cache-dir` persistence.

## 6. Risks

- Scheduler semantics are subtle (prefed steps, interleaved decode-ahead, `repeat_traced`, `_cut_short`); a
  mismatch shows up as corrupted sliding windows rather than errors. Mitigation: port line-by-line with the fake
  backend asserting packet/output order and window positions.
- Embedded CPython + C++ threads: deadlocks between the GIL and the model's `_cq_lock` / `packet_lock`. Rule: never
  hold a C++ lock while acquiring the GIL; take the GIL first.
- Startup order is load-bearing (vision capture last, L1_SMALL sizing, prefetcher session LIFO). Phase 1 keeps
  Python's startup path verbatim.
- Tokenizer/template drift between C++ and Python silently changes prompts. Goldens run in CI.
- Engram lookup touches ~200 GB of host RAM; the C++ path must share the already staged tables rather than load a
  second copy.
- New dependencies (cpp-httplib, tokenizers-cpp + Rust) need approval for the tt-metal build.
