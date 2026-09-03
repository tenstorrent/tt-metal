# Gemma4 31B MTP on Wormhole T3K — session handoff

Use this in a **new Cursor session**. Production path is fused greedy MTP (it-assistant EAGLE drafter, **not** DFlash) on **Wormhole B0 T3K 1×8**.

## Identity

| | |
|---|---|
| Tree | `/home/user/rtp/tt-metal` |
| Branch | `ign/gemma4_31B_MTP_Dflash` |
| Python | `/home/user/rtp/tt-metal/python_env/bin/python` |
| Target | `HF_MODEL=google/gemma-4-31B-it` (hidden 5376, 60 layers, 32 Q / 16 KV) |
| Drafter | `GEMMA4_ASSISTANT_MODEL=google/gemma-4-31B-it-assistant` (~453M, 4 layers, hidden 1024, same 32 Q / 16 KV, all KV-shared) |
| Mesh | `MESH_DEVICE=T3K` → 1×8, **decode TP=8 DP=1** (4 Q / 2 KV heads per chip) |
| HF cache | `HF_HOME=/home/user/.cache/huggingface`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` |
| Logs | `/tmp/mtp-long-gen/` |
| Main MTP A/B worktree (old numbers) | `/home/user/rtp/tt-metal-main-a2a` @ `3a33073484f` |

**HEAD at handoff:** `f132f2b1e1f` — Ring CCL on WH 31B TP-8 + fused all-reduce for tiny (drafter-width) decode activations.

**Uncommitted (do not mix into the perf story unless you mean to):**

- `models/demos/gemma4/tt/spec_decode.py` — fused **trace reuse** (graph key, restage seed+I/O, no recapture on new prompt) **plus** host draft/verify timers. Trace reuse is **not device-tested** (T3K was busy with Tracy).
- `models/demos/gemma4/demo/text_demo_v2.py` — log split for untraced draft vs verify time (fused trace cannot split those on the host).

## Shared device

T3K is shared. **Do not kill** other `proj_sdk` / Tracy jobs. If `CHIP_IN_USE` or ETH heartbeat `0xaabb0094`, wait or **warm-reset only when the user allows**:

```bash
/home/user/tt/tt-metal/python_env/bin/tt-smi -r
```

(`~/.tenstorrent-venv/bin/tt-smi` has a broken python symlink.)

One **CCL-bearing capture per process**. Pytest collection calling `get_num_devices()` can collide with a live job.

**Always** `unset GEMMA4_SPEC_TRACE` then set it explicitly. Leftover `=1` hung `test_verify_batchsize_invariance`.

## How MTP works here

Prefill the 31B once (unbounded sliding KV). Each decode iter (fused greedy, K=3):

1. Drafter: K steps at a **fixed** position, Q-only, SDPA into the **target** last sliding/full KV.
2. Packed verify: one 31B forward of **K+1** candidates (query-head / batch-SDPA pack, not K+1 fake users).
3. Greedy accept until first mismatch + **bonus** token (`committed = drafts[:m] + [target_ids[m]]`).
4. Next seed is a verify hidden row (`h_rows[m] → tr["h"]`), not a full reseed (unless `GEMMA4_SPEC_FUSED_RESEED=1`, slower).

Production loop: **K drafts + verify in one Metal trace** (`generate_fused` / `_generate_fused_traced`). Replay is ~70 ms; **first capture is ~4 s** (compile run + `begin_trace_capture`). That 4 s is recording the fused **command graph**, not reloading weights. Both models stay resident after load.

**Weight stream:** each 31B verify still reads ~all 31B weights from DRAM (`M` is 1 or packed 4). That is why 1-token decode is ~45.5 ms and packed verify cannot beat that by much.

If **all K drafts match**: tokens/iter = **K+1 = 4**. At ~70 ms/iter that is **~57 tok/s**, not 3× no-MTP. Current mean is ~**2.56 tok/iter** (~1.56 accepted drafts + bonus).

Do **not** retune K, DFlash, or fused reseed unless asked.

## Apples-to-apples scoreboard

**Same row for with and without MTP:**

`models/demos/gemma4/tests/e2e/test_isl_sweep.py::test_demo_text -k batch-1`

Same condiment JSON, **87 prompt tokens**, **200 new tokens**, `max_seq_len=1024`, greedy, trace on, page table. Switch is only `--speculative`.

On **main** that row lives in `text_demo_v2.py::test_demo_text -k batch-1` (no `test_isl_sweep.py`).

**Do not mix:**

- `text_demo.py::test_demo` (padded 128, always 200 tok) vs default `test_demo_spec_decode` (computing prompt, EOS ~175, acc luck → fake 42 tok/s).

### Warm T3K 1×8 (greedy, K=3)

| | Steady tok/s | ms/iter | tok/iter | Wall @ 200 (incl ~4 s capture) | vs no-MTP |
|---|---:|---:|---:|---:|---:|
| No-MTP (this branch, batch-1) | **21.98** | 45.50 | 1.00 | 21.98 | 1.00× |
| MTP **main** `3a33073484f` | **23.71** | 112.49 | 2.67 | 16.52 | 1.08× |
| MTP after fused-KV (pre L1) | 33.36 | 75.89 | 2.53 | 19.65 | 1.52× |
| MTP after decode-QKV (L1) | 34.56 | ~72.3 | 2.51 | ~20.4 | 1.57× |
| MTP after Ring + tiny fused AR | **36.33–36.58** | **70.1–70.6** | ~2.56 | **~20.8–20.9** | **~1.66×** |

Steady already beats no-MTP. **Wall at 200 tokens still loses** because of capture. At 500–2000 tokens, or a process that **reuses** the fused trace, wall should track steady.

Isolated (slightly older page size): packed P=4 ~53 ms; skip-KV ~50; 1-token decode ~45.5; K=3 draft ~**10.8 ms** (24 tiny ARs); host ~2.3; fused replay tax vs sum ~10 ms. After L1+Ring, iter ~70 ms. Physical floor ≈ decode + cheap drafts ≈ **~2.3×** at current accept, **not 3×**.

## Eval steps (copy-paste)

Shared env:

```bash
cd /home/user/rtp/tt-metal
source python_env/bin/activate   # or use python_env/bin/python directly

export HF_HOME=/home/user/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MESH_DEVICE=T3K
export HF_MODEL=google/gemma-4-31B-it

export GEMMA4_MAX_SEQ_LEN=1024
export GEMMA4_MAX_NEW_TOKENS=200
export GEMMA4_TEMPERATURE=0
export PYTHONUNBUFFERED=1

unset CI GEMMA4_PREFILL_SDPA_FIDELITY TT_CACHE_PATH GEMMA4_SPEC_PROMPT GEMMA4_NUM_LAYERS
unset GEMMA4_SPEC_TRACE
```

Confirm T3K is idle (`ps` for `tracy` / `test_mtp_layer0` / other `pytest models/demos/gemma4`). Do not start if `CHIP_IN_USE`.

### 1. No MTP

```bash
unset GEMMA4_ASSISTANT_MODEL
unset GEMMA4_SPEC_TRACE
export GEMMA4_SPEC_TRACE=0

python -m pytest \
  models/demos/gemma4/tests/e2e/test_isl_sweep.py::test_demo_text \
  -k 'batch-1' -s --timeout 3600 -q
```

Look for `Decode: … tok/s/user` (warm **~21.98**).

### 2. MTP

```bash
export GEMMA4_ASSISTANT_MODEL=google/gemma-4-31B-it-assistant
unset GEMMA4_SPEC_TRACE
export GEMMA4_SPEC_TRACE=1
export GEMMA4_SPEC_DRAFT_LEN=3

python -m pytest \
  models/demos/gemma4/tests/e2e/test_isl_sweep.py::test_demo_text \
  -k 'batch-1' -s --timeout 3600 \
  --speculative --spec-draft-len 3
```

Look for:

- `Gemma4 parallel: mesh=(1, 8) … TP=8 DP=1 … local_heads Q=4 KV=2`
- `CCLManager: … num_links=1 topology=Ring`
- `Spec setup/trace capture: ~4.x s (wall decode incl. setup: ~21 tok/s)`
- `Verify iterations: … (~70 ms/iter)`
- `Decode: … @ ~36 tok/s/user`  ← **steady**; compare this to no-MTP
- Wall line includes capture; do not claim 2× until **both** steady ≥2× **and** wall at 200–500 still beats ~22 tok/s

Run twice; quote the **warm** row.

### 3. Correctness (required after behavioral changes)

```bash
export GEMMA4_ASSISTANT_MODEL=google/gemma-4-31B-it-assistant
unset GEMMA4_SPEC_TRACE
export GEMMA4_SPEC_TRACE=1

python -m pytest \
  models/demos/gemma4/tests/e2e/test_mtp_wh_correctness.py::test_spec_decode_traced_1x8 \
  -s --timeout 1800
```

Must be **TOKEN-IDENTICAL** to untraced greedy (`trace_region_size=192e6`).

Also: `test_spec_decode_matches_greedy_1x8`.

**Do not** run `test_verify_batchsize_invariance` with `GEMMA4_SPEC_TRACE=1`.

### 4. Isolated draft / packed / full breakdown

One CCL capture per process. Set **one** component:

```bash
export GEMMA4_RUN_ASSISTANT_PROBES=1
export GEMMA4_SPEC_DRAFT_LEN=3
export GEMMA4_SPEC_PERF_REPS=10
export GEMMA4_SPEC_PROFILE_COMPONENT=draft   # or packed | target | full
unset GEMMA4_SPEC_TRACE
export GEMMA4_SPEC_TRACE=1

python -m pytest \
  models/demos/gemma4/tests/e2e/test_mtp_wh_correctness.py::test_mtp_current_path_breakdown_1x8 \
  -s --timeout 1800
```

Draft gate from the plan: K=3 chain **&lt; 5 ms**. Last measured **10.77 ms**.

## TP-8 — are we using full potential?

**31B target: yes for TP degree.** `MeshConfig(decode tp=mesh.shape[1])` → TP=8 on 1×8. Heads divide evenly. All 8 chips on the 31B; DP=1 is correct for batch-1 latency.

**Not fully used on Wormhole (vs Blackhole):**

| Lever | Status |
|---|---|
| `num_links=2` | Hangs (event-order). Stuck at **1 link**. |
| DRAM-sharded decode matmul | `can_dram_shard` is **BH-only**; WH historically garbage PCC. 31B decode stays interleaved ~45.5 ms. |
| CCL topology | **Ring** is now default for dense 31B WH ≥8 (`f132f2b`). `GEMMA4_CCL_TOPOLOGY=linear` A/B. MoE stays Linear. |
| Drafter | Same **TP=8** so Q matches sharded target KV. 4 layers × (attn AR + MLP AR) × K=3 ≈ **24 tiny ARs** (~11 ms). Replicated-MLP-only was **slower**. Do not drop ARs while weights stay sharded. Full replicate or TP=1 on one chip needs KV all-gather / submesh. |

Tiny fused AR (`N≤2048`, `M≤32`) is default for drafter-width ARs; 31B `N=5376` stays RS+AG split. `GEMMA4_CCL_TINY_FUSED=0` restores split everywhere.

## Flags (defaults on unless noted)

| Flag | Default | Notes |
|---|---|---|
| `GEMMA4_SPEC_TRACE` | demo follows `enable_trace`; **unset then set** | `=1` fused speed |
| `GEMMA4_SPEC_DRAFT_LEN` | 3 | K |
| `GEMMA4_SPEC_SHARD_ARGMAX` | on | skip 262k vocab AG |
| `GEMMA4_SPEC_FAST_HOST` | on | one H2D + one D2H |
| `GEMMA4_PACKED_VERIFY_BATCH_SDPA` | on | |
| `GEMMA4_PACKED_VERIFY_SEQ_KV` | on | |
| `GEMMA4_PACKED_FUSED_KV` | on | |
| `GEMMA4_SPEC_DEVICE_SEED` | on | |
| `GEMMA4_PACKED_DECODE_QKV` | on | packed P≤32 uses decode QKV; padding to 32 then slice **regressed** |
| `GEMMA4_CCL_TINY_FUSED` | on | fused AR only for tiny decode widths |
| `GEMMA4_SPEC_SHAPED_IO` | **off** | A/B; did not beat in-graph unpack |
| `GEMMA4_SPEC_FUSED_RESEED` | **off** | slower reference |
| `GEMMA4_PACKED_VERIFY_SKIP_KV_WRITE` | **off** | probe only |
| `GEMMA4_SPEC_TRACE_EAGER_COMPILE` | **on** | `=0` skips compile run before capture — **untested** |
| `GEMMA4_ASSISTANT_REPLICATE_MLP` | off | already lost (~13.5 vs ~11 ms) |

## What already landed (this branch)

Packed fused verify; shard-argmax; seed-row on device; batch-SDPA; seq-KV; fast host; fused K+V; decode QKV for packed P; `prepare_fused_trace` before decode timer (does **not** hide 4 s from the user); Ring + tiny fused AR.

L2 shaped I/O: no win. L4 full drafter replicate: not done. L3 wall: metric/idempotent only until trace **reuse** is tested.

## Next work (priority)

1. **Finish trace reuse** (uncommitted in `spec_decode.py`): key on graph `(K,P,…)` not `(token,pos)`; restage seed+I/O. First capture still ~4 s; **second generate in the same process** should be seed-only (~45 ms). Then wall @ 200 can match ~36 tok/s. Device-test: traced 1×8 + two generates one process + batch-1. Then commit **without** mixing the host timer logs unless wanted.
2. **Packed verify → 1-token decode** (~57 ms of the 70 ms iter is verify+tax vs 45.5 ms decode). Biggest remaining tok/s lever.
3. **Drafter without 24 TP ARs** (full replicate + KV gather, or TP=1 chip). Gate: draft &lt; 5 ms.
4. Serving-shaped eval: wall at 500–2000 tokens, reuse one trace.

Dead ends unless a research spike: `num_links=2`, WH DRAM-shard, retune K/DFlash/reseed, more host packing (~2 ms already).

## Key files

- `models/demos/gemma4/tt/spec_decode.py` — fused MTP, io_pack, prepare/capture, shard-argmax
- `models/demos/gemma4/tt/attention/decode.py` — packed_decode_forward
- `models/demos/gemma4/tt/assistant/model.py` — drafter (shares mesh/CCL)
- `models/demos/gemma4/tt/common.py` — `create_tt_model` / `create_assistant_model`; TP log line
- `models/demos/gemma4/tt/ccl.py` — Ring default, tiny fused AR
- `models/demos/gemma4/demo/text_demo_v2.py` — spec demo
- `models/demos/gemma4/tests/e2e/test_isl_sweep.py` — batch-1 A/B
- `models/demos/gemma4/tests/e2e/test_mtp_wh_correctness.py` — 1×8 wrappers

Load assistant **after** target prefill warmup (comment in `text_demo_v2.py`): earlier load broke prefill trace / profiler sync.

## New-session first actions

1. Confirm branch + uncommitted files.
2. Confirm T3K idle.
3. If continuing trace reuse: correctness then batch-1 then a **second** generate in-process; commit if TOKEN-IDENTICAL and reuse setup ≪ 4 s.
4. Do not claim 2× until steady **and** wall beat no-MTP ~22 tok/s.
