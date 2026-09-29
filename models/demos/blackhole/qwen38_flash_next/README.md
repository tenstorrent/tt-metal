# Qwen3.8-Flash-Next on Blackhole

A TTNN port of Qwen3.8-Flash-Next (48 hybrid layers: gated delta-net attention, sparse attention with an indexer,
a 256-expert MoE routed from a BF4 corpus; one multi-token-prediction layer) served from one 1x4 mesh of Blackhole
chips: an OpenAI-compatible chat server with chunked prefill, contexts to 262,144 tokens, greedy or sampled decode,
thinking and tool calls in the Qwen chat template.

This repository is a tt-metal checkout that carries the runtime fixes the model needs (section 1).  Clone it, build it
the standard way, download the checkpoint and start the server; the first start converts the weights into its caches.
No prebuilt archive, no pinned binary, no host-specific configuration.

| hardware | profile | status |
|---|---|---|
| QuietBox, 4x p150c (fw 19.4.1.0) | `tt-quietbox` | verified 2026-09-04 (startup acceptance 96/96 against the CPU, 19.6 tokens/s at 32k) and 2026-09-06 from a fresh clone; `docs/PROOFS.md` |
| 4x p150 in one host, ethernet line | `p150-line` | the numbers below were measured on it; no fresh-clone run is recorded (section 9, `docs/PROOFS.md`) |
| QuietBox 2, 2x p300c (4 dies) | `qb2` | verified 2026-09-19 (acceptance identical to the 4x p150 boxes), 2026-09-23 (the compact expert layout) and 2026-09-29 (two DRAM readers per bank on its mixed-harvest dies: the 4x p150 rows, 68.7 tok/s on the 560-token chat prompt with `--mtp 4`); section 7, `docs/PROOFS.md` |

## Performance (4x p150)

| path | measured | notes |
|---|---|---|
| prompt prefill | 750-790 tok/s (128-row chunks, the default since 2026-09-29; 410 in 32-row chunks alone); 3,003 tok/s with `--prefill-slab 2048`; with `--mtp 4` 2,645 tok/s (2026-09-27, tolerance-class; pinned `A3-slab2048-mtp4-32k`) | 2026-09-26 (block-shared attention + one-pass combine + three-ring expert stream + GDN row programs defaults: 31,716 tokens in 10.56 s; the chunk rates 2026-09-25); the slab is tolerance-class against the chunk bodies (`docs/PREFILL.md`: the ms per prompt token, the TTFTs, the one-call expert stream; on long chat prompts the two forms' greedy streams part at token 30 / 0 of a 2k / 20k document, `docs/NUMERICS.md`) |
| decode, one stream | 38.6 tok/s greedy, 36.5 tok/s sampled | 25.9 ms per token greedy (2026-09-26: the router tail's live-row exp and the MoE dense composite, bitwise: `docs/NUMERICS.md`), flat with depth; the sampled figure is the 2026-09-25 measurement; the defaults are listed below the table |
| decode, 4 / 8 streams | 28.9 / 23.4 tok/s per user (116 / 187 aggregate) | the batched-decode lane body measured directly, 34.6 / 42.7 ms per step; the chat server serves one stream (2026-09-25, component class: `docs/NUMERICS.md`) |
| decode with MTP (`--mtp 4`) | greedy 68.06 tok/s on a 560-token chat prompt, 64.16 on a 177-token multi-turn chat, 104.32 code (256-token answers), 111.40 json and 50.06 prose over their whole answers (153 / 218 tokens, EOS honoured; 2.94 / 4.75 / 2.68 / 2.09 / 4.45 tokens per pass; the stream leaves the CPU reference where plain decode does, 12 of 12 records) against 38.6 plain greedy; `--mtp 3` (2026-09-25) 45.5 / 55.4 / 60.6 / 30.7; `--mtp 5` opt-in (`docs/NUMERICS.md`); sampled 17.5 / 16.2 ms per token (the card profiles, thinking / non-thinking) | answers measured client-side with EOS honoured on the 4-chip p150 line (2026-09-28, the fork runtime built at `5eac9c778edb`; the served default set: the GDN verify-rows fold `gdn_rows_scan` with its gate scalars in one SFPU pass, the MoE rows programs on two rings, the gated-residual read's re-associated norm `gr_recip_last`; the pass p50 41.9 ms pipelined on the 560-token chat prompt; a 256-token row whose answer runs past 256 tokens is measured to 256, a row over a shorter answer stops at its end); speculative drafting with exact acceptance for greedy requests and, by default on an `--mtp --sampling` server, sampled ones; the fold against the wrap over 43 EOS-honoured prompts +2.44 % +- 1.09 % tokens per second (paired mean +1.36 +- 0.61, 2026-09-26); the fold program's gate scalars in one SFPU pass (2026-09-27) 111.0 -> 85.8 us per GDN layer and -1.16 ms per verify pass in the census, pins identical; the wrap stays the opt-out (`QWEN38_FUSED_OFF=gdn_rows_scan`: with the rings and `gr_recip_last` the wrap read 64.35 / 56.77 / 102.2 / 103.89 / 50.18 at 2.98 / 2.56 / 4.65 / 4.78 / 2.24 tokens per pass, sampled 17.6 / 19.3 ms, on 2026-09-27 on the same line class; `docs/NUMERICS.md`) |
| decode, 4 concurrent requests with MTP (`--lanes 4 --mtp 4`, opt-in, greedy only) | 132.2 tok/s aggregate on 4 x the 560-token chat prompt at once (36.5 / 39.4 / 42.9 / 47.1 per request) against 57.6 for the same four served in turn; the mixed acceptance batch 37.7 / 46.6 / 74.1 / 28.1 per user | 2026-09-29, a 1x4 p150 line at 32k, greedy, EOS honoured, byte-identical to the single stream's (13/13 rows); TTFT 1.10 / 2.27 / 3.47 / 4.65 s in a four-request burst against 0.91; an admission runs in segments between the other lanes' passes (`--lanes-stall-budget` 0.5 s: their longest pause 0.62 s, where the 2026-09-28 form froze them 1.35 s per admission at 134.0 tok/s aggregate); host load 0.9-1.6; `docs/SERVER.md`, `docs/NUMERICS.md` |
| image prompt (512 x 512 PNG, 256 image tokens, 282 prompt tokens) | TTFT 0.59 s end to end, streamed (server-side 0.52 s: prefill 0.50 s; tower 0.048 s of which 0.030 s device) against 0.52 s for a 282-token text prompt on the same server | 2026-09-29, a 1x4 p150 line at 32k, greedy, thinking off, host load 2-5; the tower's eight row buckets add 42 s to READY with the host's kernel cache warm (both hosts 42-44 s, MEASURED); `docs/SERVER.md` |
| contexts | 32k, 64k, 128k, 256k | 256k is single-user; MTP fits at 32k, 64k and 128k |
| correctness | bitwise repeatable; 96/96 greedy token match against the CPU reference on the acceptance prompt | chunked prefill is tolerance-class against the CPU reference on all 48 layers (`docs/NUMERICS.md`) |

The decode defaults since 2026-09-25 (each one's measurement and switch: `docs/NUMERICS.md`): BF8 dense weights, the
fused GDN step, the compact expert layout, the MoE post program's one-read routing, the MoE dense composite (the router
top-k beside the shared expert's eltwise, down linear and dispatch untilize as one program); 9 decode chains as fused programs,
the gated-residual read as two programs with its gathers inside them, the router tail's top-k on one core per token
group, two cores reading each DRAM bank in the decode linears.  A sampled request draws on the device after the top-32
candidate row (27.4 ms per token on both model cards; the host sampler's law, gated in `docs/NUMERICS.md`) on one-stream
`--sampling` servers; an `--mtp` server samples on the host; `--host-sampler` restores host sampling (`docs/SERVER.md`).

## 1. What you need

- Four Blackhole chips in one host as one 1x4 mesh: a QuietBox (4x p150c in an ethernet ring), four p150 cards in one
  host as an ethernet line, or four chips of a larger host (`--devices`); tt-kmd and the firmware bundle the chips
  shipped with (19.4.1.0 or later).
- This repository, built (section 2): tt-metal main `28238f903b` (2026-09-06; the device token streams are bitwise
  those of the previous base `d04395ed86` of 2026-08-29) plus the runtime fixes the model needs, not on main, without an upstream
  equivalent:
  - moe_compute: `Fix empty-rank moe compute metadata ownership` (the tilize writer; routed experts at batch 1),
    `skip idle-expert combine sync in moe_compute B=1` (-1.2 ms per token), `Fix fused MoE source buffer double
    counting`, `Add exact TP4 TTNN component path` (`moe_compute(..., local_combine=True)`, the DRAM-bank-to-worker
    query; the DRAM-sharded matmul's second reader per bank placed on a 1x4 mesh, 2026-09-16)
  - the all-gather and fabric guards: `Guard all-gather scatter state initialization`, `Preserve ring connections in
    all-gather endpoint guard`, `Fix all-gather endpoint no-target connection access`, `Clear fabric router packet
    tags on teardown`
  - the four `#23023` commits (`Bind BF4 cache loads to verified file descriptors`, `Reject lexical aliases for
    descriptor loads`, `Fail closed BF4 cache shape and cleanup`, `Bind BF4 tensorbin payload and exact types`):
    `ttnn.load_tensor` on `/proc/self/fd` paths, used by `ttnn/bf4.py`

  The server admits only a `ttnn` imported from this checkout's own build and records the checkout's commit, tree and
  extension digest with every run (`tools/runtime_admission.py`; `docs/SERVER.md`).
- The checkpoint: 360 GB (131 safetensors shards, the tokenizer, the chat template; section 3).
- Disk under `--cache-root`: 69 GB for the BF4 expert cache (built once, shared by every context, kept across runtime
  rebuilds), about 23 GB for the 32k context and 10 GB for each other allocated context (the converted non-expert
  weights and the model I/O cache), about 1.3 GB of JIT kernel cache (`docs/SERVER.md`, disk and memory).
- Host memory: 64 GB is comfortable (the first start converts one MoE layer at a time, about 10 GB of host tensors in
  flight).  The optional CPU reference (`tools/run_full_cpu_oracle.py`) needs 170-240 GB and is not a user step.
- Build tooling: what tt-metal's `install_dependencies.sh` installs (clang-20 or gcc-12, cmake, ninja, python 3.10, `uv`).

## 2. Build

The standard tt-metal flow from a fresh clone (about 10 minutes on 64 cores; on the QuietBox's 32 cores `build_metal.sh`
took 684 s and `create_venv.sh` 95 s, 2026-09-06):

    git clone https://github.com/sjettTT/tt-qwen-3.8-flash-next.git
    cd tt-qwen-3.8-flash-next
    git submodule update --init tt_metal/third_party/umd tt_metal/third_party/tracy tt_metal/third_party/tt-cluster-descriptors
    ./build_metal.sh
    ./create_venv.sh

`build_metal.sh` takes its defaults (Release, clang-20 with libstdc++; `--toolchain-path cmake/x86_64-linux-gcc-12-toolchain.cmake`
for gcc).  `create_venv.sh` makes `python_env/` with torch and installs `ttnn` from this tree in editable mode (on a
later update it asks before reusing an existing `python_env/`; answer `y`), so `python_env/bin/python -c 'import ttnn'`
resolves inside the checkout; the launcher uses that interpreter and sets `TT_METAL_HOME` itself.

## 3. The checkpoint

    python_env/bin/python -m models.demos.blackhole.qwen38_flash_next.tools.download_checkpoint --out /data/Qwen3.8-Flash-Next

fetches `Qwen/Qwen3.8-Flash-Next` from ModelScope at the release commit `2741eec155d03a8ce151b993ccce1a7b1e398d6b`
(145 files, 360,023,351,829 bytes) as parallel 128 MiB range requests (`--threads`, default 32), verifies every file's
SHA-256 against the ModelScope listing (saved as `<out>/.download/files.json`) and resumes after an interruption.  The
port pins an earlier ModelScope revision label (`checkpoint.PINNED_CHECKPOINT_REVISION`) whose weights, config,
tokenizer and chat template are byte-identical to the release commit; the server checks the files by digest (every
shard's header, the index, the file and tensor manifests, `config.json`, the tokenizer, `chat_template.jinja`) and
refuses a checkpoint that differs, with the digests printed.

`--verify-only` checks a copy already on the host the same way (every file present hashed against the listing, a marker
per verified file, nothing fetched); a following run fetches only the files absent or different.  `tools/verify_checkpoint_files.py
--checkpoint DIR --modelscope-tree <out>/.download/files.json --output report.json` re-hashes a download later (standard
library only, any `python3`); `tools/checkpoint_budget.py` prints the per-device residency budget; `tools/safetensors_metadata.py`
lists tensors.

## 4. Start the server

Every decode token reads sixteen 320-byte rows of the PLE n-gram table, which stays in the checkpoint (104 GB in 33
shards of layer 1: 51.2B parameters, over 100 GB at BF16) and is read through the page cache; the port assumes a host
whose RAM holds it (performance with the table on disk is not measured).  Read it once before the start (13 s from a
warm cache on the QuietBox; `docs/SERVER.md`):

    python_env/bin/python -m models.demos.blackhole.qwen38_flash_next.tools.prewarm_ple_table --checkpoint /data/Qwen3.8-Flash-Next

Then the server; `--profile` is `tt-quietbox`, `p150-line` or `qb2` (section 7), and `tools/` in this README is
`models/demos/blackhole/qwen38_flash_next/tools/`:

    models/demos/blackhole/qwen38_flash_next/tools/run_qwen38_chat_server.sh --profile tt-quietbox \
        --checkpoint /data/Qwen3.8-Flash-Next --cache-root /data/qwen38-cache --acceptance

The launcher prints the checkout it runs from (commit, modified or not), the interpreter and the `ttnn` extension, the
profile, the device set, the context and the run directory (`<cache-root>/runs/<stamp>/`, which holds `READY`,
`phase-markers.jsonl`, `requests.jsonl`, `acceptance.json` and, at shutdown, `result.json` and `STOPPED`).

**The first start** converts the routed experts of all 49 MoE layers into the BF4 cache
(`<cache-root>/caches/bf4-experts/`, 69 GB in the compact expert layout: about 27 s per layer on a 4x p150 host, the
49 under half an hour; the QuietBox 2026-09-25: 1448 s), then builds the caches of the chosen context, compiles the
kernels and captures the traces.  `--prepare-only --bf4-stage-limit N` converts at most N missing layers and stops, for
a machine that bounds a job's wall time.  The cache is keyed by the checkpoint and the converter's sources, not by the
tt-metal revision: a rebuilt runtime keeps it, and every start checks one re-packed expert against it (`docs/SERVER.md`).

**Warm starts** reach `READY` in about five minutes.  `--acceptance` replays the twelve shipped CPU greedy records
(`tools/acceptance/greedy-prompts/`) against the CPU before the server listens; `--require-json-96` refuses to serve
unless the `json` record matches the CPU 96/96.  The other records leave the CPU stream after 2-75 tokens; a replay
that leaves earlier than the pinned table is a regression (`docs/NUMERICS.md`).

| launcher flag | meaning |
|---|---|
| `--profile tt-quietbox\|p150-line\|qb2` | the hardware profile: the mesh graph descriptor, the device set, the route |
| `--allocated-context 32768\|65536\|131072\|262144` | the resident build (KV caches, RoPE tables and the context limit; the limit is the context minus 64 for the consumed EOS step); default 32768 |
| `--acceptance`, `--require-json-96` | replay the twelve CPU greedy records at start; refuse to serve unless `json` matches 96/96 |
| `--prepare-only --bf4-stage-limit N` | convert at most N missing expert layers into the BF4 cache and stop |
| `--long-chunks` | accepted for compatibility: the 128-row prefill chunks where the prompt allows are the served default since 2026-09-29 (section 6, bitwise the 32-row chunks); with `--mtp` the MTP layer's rows run inside the 128-row chunks too |
| `--prefill-slab 2048` | prefill slabs of 2048 rows ahead of the 128-row chunks: one matmul per dense linear, the routed experts in one `moe_compute` call on three rings, tolerance-class against the chunk bodies (`docs/PREFILL.md`); off by default; combines with `--mtp 4` since 2026-09-27 (+99 ms per traced slab, 2,645 tok/s at 31.7k prompt tokens; pinned `A3-slab2048-mtp4-32k`; the TTFT table in `docs/PREFILL.md`) |
| `--mtp 3\|4\|5` | speculative drafting on greedy requests (section 6); off by default; `QWEN38_MTP_DRAFTS_PER_REQUEST=1` opens the k = 4 and k = 5 chains together and a request picks with `extra_body.mtp_drafts` (`docs/SERVER.md`) |
| `--port`, `--host` | the listening port; `--host` default `0.0.0.0`: the QuietBox and p150-line profiles serve the LAN |
| `--serve-seconds N` | stop after N seconds (a drain: the request in flight gets its reply) |
| `--sampling` / `--no-sampling` | the launcher passes `--sampling`: sampled requests are served, a request naming no sampling field is still the bitwise greedy stream; `--no-sampling` refuses sampling fields with HTTP 400 (+0.3 ms per token saved) |
| `--stall-seconds N` | the watchdog on zero device progress, 300 through the launcher; 0 disables it (`docs/SERVER.md`) |
| `--validate-only` | run the checks and the CPU preparation without opening the mesh (a profile whose route is derived at start, `p150-line`, still needs the chips present) |
| `--devices A,B,C,D` | run `p150-line` on four other KMD device nodes (four chips of a larger host) |
| `--python` | another interpreter of this checkout |

## 5. Talk to it

OpenAI-compatible HTTP on the port you chose:

    curl -s http://<host>:8000/health
    curl -s http://<host>:8000/v1/models
    curl -s http://<host>:8000/v1/chat/completions -H 'content-type: application/json' -d '{
      "model": "Qwen/Qwen3.8-Flash-Next",
      "messages": [{"role": "user", "content": "Why does ice float?"}],
      "max_tokens": 256, "stream": true}'

The command-line client:

    python_env/bin/python -m models.demos.blackhole.qwen38_flash_next.tools.qwen38_chat_cli --url http://<host>:8000/v1 --thinking --tools

`POST /v1/chat/completions` (streaming or one document), `GET /v1/models` and `GET /v1/models/<id>`, `GET /health`.  The request rules in short
(`docs/SERVER.md` has them in full):

| request | rule |
|---|---|
| `messages` | the prompt, exactly: the server adds no system prompt, and the device prompt is the reference render (`tokenizer.apply_chat_template`), so `usage.prompt_tokens` is the count the client computes itself |
| `messages` content parts | `text` and `image_url` (`data:` URLs only; `detail` low / auto / high; 64 .. 16,384 image tokens per image, counted in `usage.prompt_tokens`; also under `--lanes`); video parts get HTTP 400; on a server whose tower is not resident, image parts get HTTP 400 `vision_unavailable` naming the reason (`docs/SERVER.md`) |
| `GET /v1/models/<id>` | the model object (`context_length`, `max_model_len`, the limits), the same document as the list's entry; another id is 404 `model_not_found` |
| `max_tokens` / `max_completion_tokens` | default and limit: the remaining context (the context limit less the prompt); a prompt over the limit gets HTTP 400 `context_length_exceeded` |
| `stream`, `stop`, `ignore_eos`, `seed`, `logprobs` | as in OpenAI; `logprobs` are relative to the read candidate row, not the vocabulary |
| `tools` / `tool_choice` | OpenAI shape, `tool_calls` finish reason; arguments are typed by the tool's parameter schema |
| `enable_thinking` (default true), `reasoning_effort`, `thinking_budget` | reasoning streams as `reasoning_content`; `chat_template_kwargs` means the same as the top-level fields |
| sampling | no sampling field: greedy, the argmax stream bitwise equal to the greedy loop the acceptance replay measures (`temperature 0` and `greedy: true` are the same path); `temperature > 0` samples with it (`top_p` 1.0, `top_k` 20, no penalties unless given); another field alone (`top_p`, `top_k`, `min_p`, a penalty, `seed`) samples with the model card's profile for the thinking mode (`/health.sampling_defaults`), so `seed` alone is a reproducible sampled stream |
| `response_format` other than `text`, `logit_bias`, `parallel_tool_calls: false` | refused with HTTP 400 rather than dropped; unknown fields are logged |
| follow-up turns | the device keeps its committed prefix and prefills only the new turn; with thinking on it restores the prompt-end snapshot instead (`docs/SERVER.md`) |
| concurrency | one request decodes at a time; up to four wait in the queue (`queue_wait_seconds` in `usage`), the fifth gets HTTP 503 |

A client that hangs up is noticed at the next device step; a streaming request gets an SSE keepalive every 30 s
through the queue wait and the prefill; the stop signal (`--serve-seconds`, SIGTERM) drains.  Hang-ups, stalled
readers, deadlines, the stall watchdog and the `/health` fields: `docs/SERVER.md`.

## 6. Long context and MTP

- `--allocated-context 65536` serves 65,472 tokens; `131072` and `262144` the same minus 64.  Each context has its own
  component and model I/O caches under `--cache-root` (the BF4 expert cache is shared); 256k leaves about 750 MB per
  device free and is single-user (`docs/PREFILL.md` has the long prompts' times to the first token).
- The chunked prefill runs 128-row chunks where the prompt allows (the remainder in 32-row chunks) by default since
  2026-09-29; `--long-chunks` is accepted and changes nothing (`--prefill-mode teacher_forced` has no chunks): 1.55 ms per
  prompt token through the server (a 6942-token prompt in 10.8 s) against 3.3 with 32-row chunks alone, the same
  tokens bitwise on all 48 layers.  With `--mtp` the MTP layer's rows run inside the 128-row chunks
  as they do inside the 32-row ones (the remainder keeps its 32-row chunks); the gate is the MTP stream after either
  prefill being the same (`docs/NUMERICS.md`).  Measured 2026-09-26 with `--mtp 4` on the 4-chip p150 line: a cold
  560-token chat prompt 0.92 s to the first token (1.60 ms per prompt token) against 1.34 s (2.34) with 32-row chunks
  alone, a 513-token prompt 0.79 s against 1.20 s; decode unchanged (58.2 tok/s greedy on the 560-token prompt,
  the acceptance table 12/12, the served classes within 0.2 tok/s of the 32-row form's).
- `--mtp 3|4|5` drafts K tokens per pass with exact acceptance; off by default.  It fits at 32k, 64k and 128k at any k,
  not at 256k (94 MB free per bank against the pair's 128 MiB contiguous).  The committed stream is not bitwise with plain decode
  on 3 of the 12 acceptance prompts: near-ties within one bf16 step, not a defect (`docs/NUMERICS.md` has the indices
  and the pinned table).  On an `--mtp --sampling` server sampled requests draft too, by exact speculative sampling
  (plain sampling's law kept; greedy requests run the fused verify, the pinned greedy stream, sampled ones the split form;
  `docs/SERVER.md` has the seed, fingerprint and `--mtp-gdn-anchor` rules); `QWEN38_MTP_SAMPLED=0` restores the 1-row sampled loop.

## 7. QuietBox 2

A p300 card is two Blackhole dies joined on the card; a QuietBox 2 (2x p300c) has four dies in one ring (the two
on-card links and the two Warp400 links), so it is one 1x4 instance, served with `--profile qb2` (section 4).
The profile exports `tools/qb2_p300_1x4_line_mesh_graph_descriptor.textproto` (a 1x4 LINE over three of the four ring
links, two channels per link as in tt-metal's `p300_x2` descriptor): tt-metal classifies a p300 cluster that is not
exactly two or four dies as CUSTOM and refuses to open without one.  The route is derived at start from the fabric's
chip order and recorded in `READY` beside the ring walk (our box: `(1, 0, 3, 2)` where the walk gives `(0, 1, 2, 3)`).
Its dies are harvested differently (one serves four of its banks from another worker column), so until 2026-09-29 the
server read each DRAM bank with one core there.  The two-reader form is now stock on such a mesh: the first die's
bank-to-worker placement is shared by every die (a reader addresses its bank by id, so the odd die reads those banks from
a worker one to three NoC hops further), recorded in `READY` (`dram_workers_per_bank`, `dram_workers_placement`) and warned
once by tt-metal; one reader remains only where the placement cannot be validated (a die with another worker or DRAM
grid, or a bank count other than eight), the reason in `READY` (`dram_workers_fallback`).  Measured on our box 2026-09-29
with the release container's arguments (`--mtp 4 --long-chunks --prefill-slab 2048`, host load 1-2.4): one reader 63.9
tok/s on the 560-token chat prompt, two readers 68.7 (the 4x p150 line's 68.3); json 105.2 -> 113.1, multi-turn 59.9 ->
64.8, prose 46.6 -> 50.3, code 97.6 -> 104.9; plain decode 38.5 -> 40.5; the 13-record startup replay identical
(`docs/NUMERICS.md`).
Verified on our box on 2026-09-19 (`READY` 194 s after launch, `json` 96/96, the divergence table identical to the 4x
p150 boxes) and 2026-09-23 (the compact expert layout: 9.17 GB more free per device), and on a contributor's box from
a fresh clone on 2026-09-07 (`docs/PROOFS.md`).

## 8. Layout

| where | what |
|---|---|
| `chat.py` `checkpoint.py` `config.py` `reference.py` | the checkpoint, the chat template, the torch reference model |
| `diagnostic_bf4.py` | the binder of a CPU-staged BF4 expert corpus |
| `tt/` | torch reference components (the CPU oracle every test compares against) |
| `ttnn/` | the device model: builder, layers, GDN, QSA, MoE/BF4, embedding, sampling, MTP |
| `tools/run_qwen38_chat_server.sh`, `qwen38_chat_cli.py`, `qb_mesh_smoke.py` | the launcher; the client; open the mesh and check the route without the model |
| `tools/qwen38_chat_server.py`, `qwen38_chat_session.py`, `qwen38_chat_protocol.py`, `qwen38_sampling_step.py` | the HTTP server; the traced decode chain and the mesh open; the request/reply protocol; the sampling step |
| `tools/qwen38_asgi.py` | the server behind uvicorn: the container form the tt-model package serves (`docs/SERVER.md`) |
| `tools/runtime_admission.py`, `live_decode_diagnostic.py` | the runtime identity (this checkout's build); the CPU preparation and the live construction (the BF4 conversion on the first start) |
| `tools/hardware_profiles.py`, `physical_route.py`, `resident_decode.py`, `evidence_records.py` | the profiles (tt-quietbox, p150-line, tt-quietbox-2); the route derivation; the chain's fixed points; the run records |
| `tools/download_checkpoint.py`, `prewarm_ple_table.py`, `verify_checkpoint_files.py`, `checkpoint_budget.py`, `safetensors_metadata.py` | the ModelScope download with SHA-256 verification; the n-gram pre-warm and the checkpoint tools of sections 3 and 4 |
| `tools/acceptance/` | the CPU greedy records the startup replay compares against, the two-step CPU oracle |
| `tools/qwen38_reference_corpus.py` | the teacher-forced reference corpus Q38-REF-v1 (`tools/reference/`) and its columns: HF on the CPU, the `tt/` oracle, a served chain's agreement records; `score` |
| `tools/stage_full_bf4_cpu.py`, `verify_full_bf4_cpu.py`, `bind_full_bf4_corpus_cpu.py`, `probe_full_bf4_binding_cpu.py` | a BF4 expert corpus staged on the CPU (produce, verify, bind, probe) |
| `tools/ci/q38_ci.py` | the regression harness: pins.json, baselines/, job runner, verdicts, seeding |
| `tests/` | the no-device tests (set `QWEN38_CHECKPOINT` for the checkpoint-reading ones) |
| `docs/` | PROOFS, NUMERICS, SERVER, TESTING, PREFILL (the "More" section below) |

Run the tests from the repository root (`docs/TESTING.md` has the regression harness and the reference corpus):

    QWEN38_CHECKPOINT=/data/Qwen3.8-Flash-Next python_env/bin/python -m pytest models/demos/blackhole/qwen38_flash_next/tests

## 9. What is verified, and the known limits

- Verified runs: the hardware table above and `docs/PROOFS.md` (the QuietBox 2026-09-04 and 2026-09-06, the QuietBox 2
  2026-09-07, 2026-09-19 and 2026-09-23).  No fresh-clone run is recorded for a p150 line; this README's numbers were
  measured on 4x p150 hosts.
- Numerics (`docs/NUMERICS.md`, the 2026-09-25 pinned table): `json` 96/96 against the CPU, the other eleven records
  leave the CPU greedy stream between token 2 and 75; MTP is not bitwise with plain decode on 3 of 12 prompts (near-ties).
- 256k context is single-user; MTP is not available at 256k.
- One request decodes at a time by default (the traced chain is single-stream); the queue holds four more.  `--lanes B`
  (2..8, needs `--mtp`, greedy only) serves B at once through the MTP lane chain; an admission's prefill runs in segments
  between the other lanes' passes under `--lanes-stall-budget` (`docs/SERVER.md`).
- Images: `image_url` parts of at most 16,384 image tokens each (the stock 4096 x 4096 px maximum); served under `--lanes` too (2026-09-29, byte-identical to the single stream); every
  start runs the tower's eight row buckets once (about 44 s of READY, about 100 s more on a host whose kernel cache has none of the
  tower's programs); a DRAM shortfall starts the server text-only and says so in READY and `/health` (`docs/SERVER.md`).
- The first start is long (the 69 GB expert conversion); a host that kills long jobs needs `--prepare-only
  --bf4-stage-limit N` runs first.
- Python 3.10 (`create_venv.sh` default); Linux x86_64.

## More

- `docs/PROOFS.md`: the fresh-clone release proof of 2026-09-06 on the QuietBox (build, checkpoint, first start,
  acceptance, `--mtp 4`, `--long-chunks`, 64k) with its run logs; the QuietBox 2 runs.
- `docs/NUMERICS.md`: the acceptance mechanism, the pinned divergence tables (plain, `--long-chunks`, MTP, teacher-forced),
  the fused decode kernels, their switches and throughput, the device sampler's law gate, what "bitwise" means here.
- `docs/SERVER.md`: the request rules in full, sampling, thinking, the prompt-end snapshot, the stall watchdog,
  `/health`, runtime admission, the BF4 cache identity, the n-gram table pre-warm, disk and memory.
- `docs/TESTING.md`: the no-device tests, the reference corpus Q38-REF-v1 and its scorer, the regression harness
  (`tools/ci/q38_ci.py`), how to run the acceptance gate.
- `docs/PREFILL.md`: the prefill chunk bodies and the opt-in 2048-row slab (`--prefill-slab`): what runs as one
  matmul, its numerics class, what it costs and saves, the routed experts in one call, the served rates.
