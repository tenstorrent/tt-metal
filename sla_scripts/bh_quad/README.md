# Llama-3.3-70B on 4 BH Galaxies: 128 concurrent users

One vLLM server per Galaxy (the single-Galaxy SLA configuration from `../bh_tp32`, 32 slots each:
DP4 x `max_num_seqs` 8) behind `lb.py`, an OpenAI-compatible load balancer. Clients use the balancer exactly like
one vLLM server; it adds no model-side change, so accuracy is the single-Galaxy accuracy.

```
clients ──> lb.py :9000 ──┬─> Galaxy A vLLM :8000  (32 slots)
                          ├─> Galaxy B vLLM :8000  (32 slots)
                          ├─> Galaxy C vLLM :8000  (32 slots)
                          └─> Galaxy D vLLM :8000  (32 slots)
```

## Files

| File | What |
|---|---|
| `lb.py` | the balancer (aiohttp + uvloop, one process) |
| `lb_selftest.py` | 24 checks against 4 fake backends, CPU only, ~1 min |
| `serve.sh` | one Galaxy's server in the shipped config, on 0.0.0.0:8000 |
| `quad_bench.sh` | the SLA lines through the balancer: 128 x 4K, 128 x 16K, 4 x 240 RPM, needle, sanity |

## Run

On each Galaxy (work dir `J` with `tt-metal/`, `weights/`, `cache/llama70b/` (222 GB tensor cache),
`vllm-tt-plugin-gio/`):

```bash
J=/mnt/nvme0/jerrywang bash sla_scripts/bh_quad/serve.sh
```

On any host that reaches all four:

```bash
python sla_scripts/bh_quad/lb.py --backends http://10.82.97.43:8000,http://10.82.97.44:8000,http://10.82.97.45:8000,http://10.82.97.46:8000
```

```bash
HF_MODEL=/path/to/Llama-3.3-70B-Instruct OUT=runs/quad bash sla_scripts/bh_quad/quad_bench.sh
```

`GET /lb/stats` shows per-backend in-flight, prefill backlog, served, errors, TTFT EWMA.
`POST /lb/backend {"url": "...", "enabled": false}` drains a Galaxy (no new requests; running ones finish).

## How the balancer routes

- **Slots:** at most `--slots` (32) requests in flight per backend. Requests past the total wait in one FIFO
  queue and take the first slot that frees on any backend. No backend is ever over-subscribed, so each Galaxy
  sees at most the load the single-Galaxy SLA numbers were measured at.
- **Cost:** a request goes to the backend with the lowest
  `prefill backlog (prompt tokens waiting for their first output token) + 256 x running requests + the prompt
  tokens this backend would have to prefill`. A burst spreads evenly; a backend busy with long prompts gets the
  short ones last; prompts whose prefix a backend probably still caches (hashed at 2K/8K/32K/128K chars) count
  only their uncached part there. That is how conversations stay with their history without piling a burst of
  identical prompts onto one Galaxy.
- **Failures:** connection refused, a reset, or a 5xx before the first byte: the backend is marked down and the
  request retried elsewhere (2 retries), invisible to the client. A backend dying mid-stream cuts the client
  connection (the client sees an error, never a clean but short answer). Health checks (`/health` every 2 s)
  bring a backend back. A client that disconnects closes the upstream connection, so vLLM aborts the request.
- **Not routed:** `GET /v1/models`, `/tokenize`, `/version`... go to any healthy backend without a slot.

## Self-test (lb_selftest.py, Nov 23, all 24 pass)

128 concurrent streams split exactly 32/32/32/32 with every byte in order; 300 concurrent queue and never exceed
32 per backend; pending-prefill accounting returns to 0; chat, non-streaming and 4xx pass through (4xx not
retried); short requests avoid a backend prefilling 100K tokens; shared-prefix requests stick while a burst of
identical prompts still spreads 2/2/2/2; a backend returning 503 or killed: all requests succeed elsewhere, and it
rejoins (32 of the next 128) when healthy; a backend dying mid-stream gives those clients an error and leaves the
rest intact; 16 abandoned client streams abort upstream and free their slots; a drained backend gets nothing.
Overhead: 1000 tiny requests take the same time through the balancer as direct.

## First live run (Nov 26-27, 2026)

Staged from .43 to .40, .41 and .46: weights 132 GB, `cache/llama70b` 222 GB, tt-metal without `generated/`
(rsync `.../tt-metal/ host:.../tt-metal/` with `--exclude=/generated/`; without the trailing slash the exclude
anchors one level up and copies 841 GB of profiler output). Each host links `/mnt/nvme0` to its copy so every path
matches .43. One server per Galaxy (`serve.sh`), balancer on .43 (`lb.py`, port 9000). Bring-up script:
`/dev/shm/jw_quad_up.sh` (`START=1`, `RESET=1`).

Results with 2 Galaxies (.43 + .46). .40 served garbage (bad chip state from earlier jobs) and was drained; .41 was
taken by another user.

| Line | Result | Target |
|---|---|---|
| 32 x 16K at once | 3.84 s mean (P99 6.09 s) | 5 s |
| 32 x 4K closed loop | 0.77 s mean | 1 s |
| Concurrent needle | 0 misses | |
| Quick MMLU via the balancer | 82.02% (single Galaxy 81.84%) | |
| Balancer | 194 requests, 0 failed, 98 / 96 split | |

Failover: SIGINT to .46's server in the middle of 128 x 4K at concurrency 32. The balancer marked it down within
seconds and .43 took all 32 users: 112 requests succeeded, exactly the 16 streaming from .46 failed (a started stream
cannot be replayed). The restarted server rejoined by itself after ~10 min and passed sanity cold and cached. A clean
SIGINT shutdown leaves the chips healthy.

Resets on these hosts:
- `tt-smi -glx_reset_auto` (IPMI tray reset) re-probes every NVMe drive on PCIe root complex 00. Mounts on it
  return I/O errors until `umount -l` + mount, and the drive comes back renamed (nvme3n1 -> nvme3n2). On .40
  `/big_home` (LVM) and on .41 another user's drive sit on that complex.
- `tt-smi -r` (chip-level PCI reset) on .46: rc 0, all 32 boards re-initialized, zero NVMe kernel messages, the
  drive kept its name and stayed mounted. Use it instead of `-glx_reset_auto` where shared mounts are at risk.
- Run `sanity_check.py` (cold and cached long prompt) on every server before it joins the balancer.

### Balancer tuning (Nov 27, .43 + .46, 32 users per Galaxy as in the full quad)

`--decode-weight` sweep, one run each (256 repeated for noise). Script: `/dev/shm/jw_lb_sweep.sh`, results in
`runs/quad/sweep/`.

| decode-weight | 64 x 4K closed loop | 64 x 16K at once | 480 RPM mix mean (P99) |
|---|---|---|---|
| 256 (default) | 0.953 s | 6.84 s | 0.398 s (1.03 s) |
| 64 | 1.002 s | 6.79 s | 0.387 s (0.96 s) |
| 1024 | 0.976 s | 6.85 s | 0.408 s (1.05 s) |
| 256, repeat | 0.948 s | 6.86 s | 0.394 s (1.06 s) |

All within noise; the default stays. At full per-Galaxy load each Galaxy behaves exactly as it does alone (single
Galaxy: 32 x 4K 0.95-0.98 s, 32 x 16K 6.7-6.8 s), so the balancer adds nothing and costs nothing. What a quad buys is
fewer users per Galaxy: 32 users over 2 Galaxies gave 32 x 16K 3.84 s and 32 x 4K 0.77 s. Lines with 32 users per
Galaxy (128 over 4) inherit the single-Galaxy numbers, including 16K at ~6.8 s.

## Reserving the Galaxies (claim.sh)

Other users share these hosts as the same `ttuser` account, and a note in `/tmp` did not stop them: on Nov 26 two
quad Galaxies were reset under running servers. `claim.sh claim <who> <until> <what>`, run on each Galaxy, makes the
reservation visible where people and agents actually act:

| Check | What it does |
|---|---|
| tt-smi reset guard | `tt-smi -r` / `-glx_reset*` print the reservation and exit 1 while claimed, unless `TT_CLAIM_OVERRIDE=1`. Other tt-smi uses run normally. The original launcher is kept as `<path>.preclaim` (`/usr/local/bin/tt-smi`, or `~/.local/bin/tt-smi` on .41, which is a Python entry point and gets a bash wrapper). `galaxy_up.sh` sets the override for its own reset. |
| Login banner | `/etc/profile.d/zz-galaxy-claim.sh` prints the reservation in red on every login shell. |
| Lock | holds `/tmp/dsv4_dev.lock`, which the DeepSeek job chains queue on. |
| Notes + broadcast | `~/CLAIMED_BY_<who>.txt`, one `wall` message. |

`claim.sh status` shows each check; `claim.sh release` undoes everything and restores tt-smi byte for byte.
A reboot wipes `/tmp`: the guards installed on Nov 27 read `/tmp/GALAXY_CLAIM`, so after a reboot copy
`~/CLAIMED_BY_<who>.txt` back to it and re-take the lock (the repo version of claim.sh now uses `/etc/galaxy_claim`,
which survives reboots). Test the guard only with a harmless target (`tt-smi -r 999`), never with a real reset.

## Bring-up notes (Nov 27)

- `tt-smi -r` on .41 once brought back only 24 of 32 chips ("Requested mesh grid shape (8, 4) is larger than number
  of available devices 24"); the host was then power-cycled, which restored all 32. After a reboot, start without a
  reset (`RESET=0`).
- `python: not found` on .42: the venv points at uv's `~/.local/share/uv/python/cpython-3.10-linux-x86_64-gnu`, a
  symlink to `cpython-3.10.19-linux-x86_64-gnu` that .42 lacked. `add_galaxy.sh` now checks `import ttnn` on the
  target before starting, and its "server alive" check uses an anchored pgrep (the unanchored one matched its own ssh
  command and never noticed a dead server).
- A tt-smi reset re-probes NVMe drives on PCIe complex 00 only for `-glx_reset*`; after any such reset elsewhere,
  remount the drive by label (`sudo mount LABEL=jw_nvme /mnt/nvme0` on .42).

## Full quad: 4 Galaxies (.40 .41 .42 .43), Nov 27, 2026

One server per Galaxy, balancer on .43 (`lb.py`, port 9000, `--stall-timeout 300`). Every server passed
`sanity_check.py` (cold and cached long prompt) before joining. Bench: `/dev/shm/jw_q4.sh`, results in `runs/quad/q4/`.

| Line | 4 Galaxies | 1 Galaxy |
|---|---|---|
| 32 x 16K at once | 2.60 s / 2.78 s mean (P99 4.20 / 4.65 s) | 6.8 s |
| 32 x 4K closed loop | 0.47 s / 0.53 s mean (P99 1.10 / 0.88 s) | 0.95 s |
| 128 x 4K closed loop | 0.97 s mean (P99 3.60 s) | (32 x 4K: 0.95 s) |
| 128 x 16K at once | 6.81 s mean (P99 12.2 s) | (32 x 16K: 6.8 s) |
| 960 RPM mix | 0.42 s mean (P99 1.03 s) | (240 RPM: 0.39 s) |
| Concurrent needle | 0 misses | 0 |
| Full MMLU (14,042 questions) | 83.01% +/- 0.30 | 82.9-83.1% |
| Balancer | 16,654 requests, 0 failed, 4,196 / 4,097 / 4,187 / 4,174 | |

32-user lines get faster with every Galaxy added (8 users per Galaxy). 128-user lines carry 32 users per Galaxy, so
they match the single-Galaxy numbers: the balancer adds no overhead, and each Galaxy is the limit.

Stopping a server: `kill -TERM <serve.sh pid>` (serve.sh forwards it to vLLM as a clean shutdown). SIGINT does
nothing to a serve.sh started in the background by `galaxy_up.sh`: a non-interactive shell's background jobs ignore
SIGINT, and an ignored signal cannot be trapped. The first quad failover test sent SIGINT and so tested nothing.

### Hung server (Nov 27, .41)

.41's server hung on the device mid-prefill (engine at 100% CPU, no output, health still OK), holding 8 requests for
7 minutes. Stopping its processes reset those connections before any output, so the balancer retried all 8
elsewhere and the run completed with 0 failures. `lb.py --stall-timeout` (default 300 s) now catches this without a
human: a backend that sends nothing for that long is marked down, and requests that have no output yet are retried
on another Galaxy (`lb_selftest.py`: 28/28, including a hung fake backend).

.41 then came back with 24 of 32 chips after `tt-smi -r` and after `-glx_reset_auto` ("Didn't find all 32 chips");
only `ipmitool chassis power cycle` restored all 32 (twice on Nov 27). After a power cycle start without a reset and
restore `/tmp/GALAXY_CLAIM` from `~/CLAIMED_BY_<who>.txt`.

### Failover on the full quad (Nov 27, `/dev/shm/jw_failover.sh`)

SIGTERM to .42's serve.sh 40 s into 128 x 4K (concurrency 128). The balancer marked .42 down within 15 s; the 32
requests streaming from it failed (a started stream cannot be replayed) and the other 352 succeeded on .40/.41/.43
(TTFT mean 2.35 s, P99 12.0 s while 128 users shared 96 slots). .42 released its chips at once, restarted with
`galaxy_up.sh START=1` (warm, ~10 min), rejoined by itself, passed sanity cold and cached, and 128 x 4K on all four
was back to 0.98 s mean (P99 3.62 s), 384/384 succeeded. Balancer totals: 17,422 requests, 32 failed (exactly the
ones on the killed server).

## Decode in the 16K bursts (Nov 27): KV fix and the per-Galaxy limit

Note: the Coredge SLA gives 16K bursts only a TTFT target (32 x cold 16K at once, 5 s). The 25 / 15 tok/s decode figures below reuse the general decode target for reference; they are not SLA lines.

`LLAMA_BH_KV_TOKENS_PER_LANE` (generator_vllm.py `get_max_tokens_all_users`, default 163,840 in env.sh; was the plugin
fallback 131,072). The old budget held exactly 8 x 16K prompts per lane, so in 32 x 16K at once the first 4 users
stalled 6.4-6.7 s for KV blocks after the last prefill. Now every wave sees only the later prefill waves.

| Line | Before | After | Target |
|---|---|---|---|
| 1 Galaxy 32 x 16K decode (mean / P99 tok/s) | 16.5 / 9.4 | 18.5 / 12.6 | 25 / 15 |
| Quad 32 x 16K decode | 36.7 / 29.0 | 37.6 / 32.9 (TTFT 2.33 s) | 25 / 15 |
| Quad 128 x 16K decode | 16.4 / 9.3 | 18.7 / 12.6 | 25 / 15 |

TTFT and 32 x 4K unchanged; sanity passes. Per-wave breakdown (`/dev/shm/tpot_by_wave.py`): per-token time = 28.4 ms
decode step + 1.48 s for each prefill wave after the user's first token, spread over 199 tokens. Passing 25 / 15 needs
the average user to sit through <= 1.5 later waves, i.e. <= 16 users per Galaxy: 32 x 16K passes on 2+ Galaxies,
128 x 16K needs 8. Prefilling in fewer, bigger waves only moves first tokens later (same last-token time), so it is not
used.

### Decode step at 16K context (device profile, Nov 27, `/dev/shm/decode_layer_breakdown.py`)

`test_decoder_device_perf.py` with `DECODE_START_POS=16384`, batch 32, current env.sh flags, last iteration of 10
layers, kernel time averaged over 32 devices:

| Op | us / layer | Share |
|---|---|---|
| SdpaDecode (attention over 16K of KV) | 149.7 | 45% |
| Matmul | 40.7 | 12% |
| AllReduceAsync | 27.9 | 8% |
| ReduceScatterMinimalAsync | 27.1 | 8% |
| AllGather(Async) | 33.3 | 10% |
| LayerNorm, KV update, QKV heads, rotary, reshard, concat, eltwise | ~51 | 16% |
| Kernel total / wall incl. dispatch gaps | 330 / 353 | x80 = 26.4 / 28.2 ms (served: 28.4 ms) |

Attention runs at ~238 GB/s per chip (8 users x 1 KV head x 16K tokens, bfp8 K+V = 35.6 MB per layer), ~46% of
DRAM peak. Bound: with attention at zero cost the step is still ~16.2 ms, so the 16K-burst decode lines would reach at
most 23.7 tok/s mean (target 25) and 14.7 tok/s P99 (target 15) at 32 users per Galaxy. Decode-kernel work cannot pass
them; realistic attention work (bandwidth to ~80%, bfp4 KV) gives a ~20 ms step, ~21.7 / 13.9 tok/s.

After this on-device profile, `tt-smi -r` left .43 unable to enumerate its topology (`GetNumAvailableDevices`:
`unordered_map::at`), as on .42; `-glx_reset_auto` fixed it. Remount `/mnt/nvme0` a few seconds after the reset (by
UUID via fstab): a remount right after the reset raced the NVMe re-probe and still returned I/O errors.

### Prefill/decode disaggregation: feasibility test (Nov 27)

Idea: some Galaxies only prefill and ship each user's KV (~2.84 GB for 16K tokens, bfp8, 80 layers) to Galaxies that
only decode, so decoding users never stall behind prefill.

- Network (`/dev/shm/net_tput.py`, .43 -> .41, 100 Gb/s links, MTU 1450): 1.68 GB/s one TCP stream, 4.56 GB/s four,
  5.69 GB/s eight. One 16K user's KV takes ~0.5 s (~0.3 s if tuned toward line rate). Workable.
- Capacity is the blocker: Galaxy decode runs batch 32 (TG supports batch 1 and 32), and the KV pool holds ~39 users
  of 16.6K per Galaxy. 128 x 16K needs 128 decode slots = all 4 Galaxies decoding, so no Galaxy is free to only
  prefill: disaggregation cannot run 128 users on 4 Galaxies.
- 32 x 16K: impossible on one Galaxy (needs separate prefill hardware). On 2 Galaxies (1 prefill + 1 decode) decode
  would pass (~35 tok/s, no stalls) with TTFT ~7 s, but simply splitting the 32 users over the same 2 Galaxies already
  passes decode (28.1 / 21.4 tok/s) with TTFT 3.84 s. Disaggregation is dominated; not built.

### Optional: admission cap for 16K bursts (off by default; `lb.py --slots 16`)

The balancer admits 32 users per Galaxy by default. To turn the cap on, start it with `--slots 16`:

```bash
python sla_scripts/bh_quad/lb.py --backends <galaxy urls> --port 9000 --slots 16
```

Capping each Galaxy at 16 concurrent users (the rest queue in the balancer) keeps every admitted user behind at most
3 later prefill waves. Decode (reference 25 / 15 tok/s, not an SLA line) passes; TTFT rises because queued users wait:

| Line | Uncapped | `--slots 16` |
|---|---|---|
| 1 Galaxy 32 x 16K (single-backend balancer) | decode 18.5 / 12.6, TTFT 6.8 s | decode 28.0 / 21.4, TTFT 9.3 s (P99 16.9) |
| Quad 128 x 16K | decode 18.7 / 12.6, TTFT 6.8 s | decode 28.2 / 21.4, TTFT 9.2 s (P99 17.0), 128/128 ok |

The capped decode step is ~24.6 ms (28.4 uncapped). Use `--slots 16` only when long-prompt decode speed matters more
than time to first token; the default stays 32.
