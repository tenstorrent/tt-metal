# Cold-prefill regression: bounded chunked prefill (e3aa655) — root cause + env-gated fix

## Symptom (measured)

Cold single-user prefill (prefix caching OFF, one request from scratch) got **~2.4× slower**:

| date | source | ISL | cold TTFT | prefill tok/s |
|------|--------|-----|-----------|---------------|
| 2026-07-24 22:43 | `doc/vllm_integration/sweep_4chip_128k/sweep_table.tsv` | 131072 | 126,376 ms | ~1037 |
| 2026-08-03 | `doc/vllm_integration/stage_d/sweep.tsv` | 130048 | 298,740 ms | ~435 |

Same P150x4, same `TT_LAGUNA_PIPE_CHUNK=2048` in the launch env, single user, `--ignore-eos`, APC off. So it is a **code change**, not config/measurement.

## Root cause: e3aa655 flipped the effective outer prefill chunk from 8192 → 2048

`e3aa655acd5` ("bounded chunked prefill for long-context serving") landed **46 min after** the fast 22:43 sweep. Its *entire* model-code change is to start **honouring** the `TT_LAGUNA_PIPE_CHUNK` env knob in `OptimizedDecoder.__init__` (plus a warmup page-table-width fix):

```python
# added by e3aa655 (tt/optimized_decoder.py)
self.PIPE_CHUNK = int(os.environ.get("TT_LAGUNA_PIPE_CHUNK", OptimizedDecoder.PIPE_CHUNK))
```

Before this commit the knob **did not exist** — `PIPE_CHUNK` was the hard-coded class attribute `8192` and the env var was silently ignored:

```
$ git show e3aa655^:.../optimized_decoder.py | grep -n PIPE_CHUNK
364:    PIPE_CHUNK = 8192            # class default, NOT read from env pre-commit
```

So the two sweeps did **not** run the same chunk size, despite both setting `TT_LAGUNA_PIPE_CHUNK=2048`:

- **22:43 fast sweep (pre-commit):** env ignored → outer chunk **CH = 8192** → 131072 / 8192 = **16 chunks**.
- **08-03 sweep (post-commit):** env honoured → outer chunk **CH = 2048** → 131072 / 2048 = **64 chunks**.

### Why 4× smaller chunks is ~2.4× slower: redundant KV-prefix re-read

For every global (non-sliding) layer, `_prefill_pipelined` attends each outer chunk against the **full paged KV cache** with `chunk_start_idx=gpos` (`tt/multichip_decoder.py` / `tt/optimized_decoder.py`):

```python
CH = (self.PIPE_CHUNK // bs) * bs
for c in range(0, seq, CH):
    ...
    attn = ttnn.transformer.chunked_scaled_dot_product_attention(
        q, kv_cache["k"], kv_cache["v"], user_pt,
        chunk_start_idx=gpos,              # chunk c reads cache K/V[0 : gpos+ch]
        compute_kernel_config=self._sdpa_compute,
    )
```

Chunk *i* (0-indexed) re-reads the prefix `K/V[0 : (i+1)·CH]` from the paged DRAM cache. Summing the DRAM bytes read across all `n = seq/CH` chunks:

```
bytes ≈ kv_bytes · Σ_{i=0..n-1} (i+1)·CH
      = kv_bytes · CH · n(n+1)/2
      = kv_bytes · (seq² / CH + seq) / 2       # dominant term ∝ 1 / CH
```

The redundant-prefix DRAM traffic is **inversely proportional to the outer chunk CH**. Shrinking CH from 8192 → 2048 (×¼) **quadruples** the dominant `seq²/CH` re-read term. At `seq = 131072` this prefix re-read dominates the cold-prefill wall clock, so quartering the chunk drives the ~2.4× TTFT regression (also ×4 the per-chunk op-dispatch count: rms/qkv/rope/2×paged_fill/concat_heads/gate/wo/all_reduce/mlp per chunk).

Note the SDPA *compute* (≈ `seq²/2` causal FLOPs) and the per-token projection/MLP work are **chunk-count-independent** — accuracy and total matmul work are unchanged; only the redundant KV-prefix bandwidth and fixed per-op overhead scale with the chunk count. The commit message's claim that "chunked == single-shot (~79 ms/layer)" was profiled at short ISL where the `seq²/CH` term is negligible; it does not hold at 128k.

## Confound check: the uncommitted session edits are NOT a contributor

The working tree has uncommitted item-2.5 edits (fused `nlp_create_qkv_heads` / `nlp_concat_heads` replacing the manual reshape/permute head packing) and decode-only k64 SDPA sweep knobs. `git diff` confirms these edits **do not touch** the outer-chunk structure:

- `git diff -- tt/optimized_decoder.py | grep -E "PIPE_CHUNK|CH = \(self"` → only the branch-condition **context** line appears; the `CH = (self.PIPE_CHUNK // bs) * bs` line and the `PIPE_CHUNK` env read are **unchanged**.
- item 2.5 is a per-op replacement of the head reshape — same chunk count, same KV-read pattern, same tensor shapes. It cannot produce a 2.4× prefill delta (it was landed as an efficiency-neutral micro-op).

**Verdict: the regression is solely e3aa655** (activating `PIPE_CHUNK=2048`). The session edits are not implicated.

## The fix (env-gated, default = current bounded behavior)

New env flag **`TT_LAGUNA_PREFILL_FAST`** (default `0` = unchanged). When `=1`, the **outer** `_prefill_pipelined` chunk uses **`TT_LAGUNA_PREFILL_FAST_CHUNK`** (default `8192`, the proven-safe pre-regression size), cutting the `seq²/CH` prefix re-read ~4×. It does **not** change the single-shot branch threshold (`PIPE_CHUNK`), the SDPA k-chunk, or any per-op numerics — only how a `> PIPE_CHUNK` prefill is sub-chunked. Per-chunk activation stays bounded to `FAST_CHUNK` (8192), so there is **no full-length allocation**.

Code (minimal, localized to the chunk-bounding logic):

```python
# tt/optimized_decoder.py  __init__  (inherited by MultichipDecoder via super().__init__)
self.PREFILL_FAST = os.environ.get("TT_LAGUNA_PREFILL_FAST", "0") == "1"
self.PREFILL_FAST_CHUNK = int(os.environ.get("TT_LAGUNA_PREFILL_FAST_CHUNK", 8192))

@property
def _prefill_pipe_chunk(self):
    # default == live PIPE_CHUNK (byte-identical bounded path; honours a runtime-mutated PIPE_CHUNK,
    # which test_prefill_pipelined_matches_hf relies on). FAST=1 swaps in the larger chunk.
    return self.PREFILL_FAST_CHUNK if self.PREFILL_FAST else self.PIPE_CHUNK
```

and in both `_prefill_pipelined` (optimized + multichip):

```python
CH = (self._prefill_pipe_chunk // bs) * bs   # env-gated outer chunk (TT_LAGUNA_PREFILL_FAST)
```

**Default off ⇒ `_prefill_pipe_chunk == self.PIPE_CHUNK` ⇒ byte-identical to today.** Nothing changes until the flag is flipped.

### Warmup / trace safety

`_prefill_bucket_lens()` warms the full power-of-two ladder up to the servable context by calling `prefill_forward` on the **same decoder instance / same env**, so with `PREFILL_FAST=1` the warmup naturally compiles the 8192-wide chunked-SDPA and the 16-chunk reassembly programs *before* the decode trace. At 131072 all chunks are a clean 8192 (16×8192), and a 130048 request rounds **up** to the 131072 bucket (`_bucket_len`), so serving runs only pre-compiled programs — no under-trace recompile. **The flag must be set at server launch (before warmup)**, which is the natural place for an env var.

### Memory / context-cap guidance

The user is fine capping served context at **131072**. The 8192-wide per-chunk activation was proven memory-safe at 131072 in the 22:43 sweep. **Do not enable the fast path for `max_model_len > 131072`** without a fresh OOM check — the fast path bounds activation to 8192 (same as it did pre-regression), but the KV pool / page-table footprint at >128k has not been re-validated for this chunking.

## Validation plan (run AFTER the current bench finishes; mesh must be free)

All commands assume the Stage-D launch env. The A/B is a single cold-prefill point at the top bucket, flag OFF vs ON. APC is OFF so every request is a true cold prefill (no prefix-cache hit even though the warm sends the same prompt).

### 1. A/B cold prefill @ 131072 (expect ~299s → ~126s TTFT)

Reuse `scripts/stage_d_latency_sweep.sh`'s server + `vbench`, adding `TT_LAGUNA_PREFILL_FAST` to the server env (the `export ... TT_LAGUNA_PIPE_CHUNK=2048 ...` line in `start_server`). Minimal standalone form:

```bash
LOCAL=/home/ttuser/.local/lib/model-bringup/tt-metal
BASE=/home/ttuser/dev/tt-metal/models/autoports/poolside_laguna_xs_2_1
BV=/home/ttuser/.local/lib/tt-inference-server/.workflow_venvs/.venv_benchmarks_vllm/bin
VBIN=/home/ttuser/.tenstorrent-venv/bin
SERVE_PP=/home/ttuser/dev/tt-metal:$LOCAL/vllm:$LOCAL/vllm/plugins/vllm-tt-plugin/src

run_point () {  # $1 = FAST (0|1) ; $2 = tag
  cd /tmp
  tt-smi -r all >/dev/null 2>&1; sleep 8   # clean eth cores before opening the mesh
  setsid bash -c "
    export TT_METAL_HOME=$LOCAL PYTHONPATH=$SERVE_PP \
           TT_LAGUNA_PIPE_CHUNK=2048 TT_LAGUNA_PREFILL_FAST=$1 \
           TT_LAGUNA_WEIGHT_CACHE_DISABLE=1 TT_LAGUNA_DECODE_SDPA_PC=1
    exec $VBIN/python -u -m models.common.readiness_check.run_vllm_server \
      --model-dir $BASE --hf-model poolside/Laguna-XS-2.1 --mesh-device P150x4 --stages serve \
      --max-num-seqs 16 --block-size 64 --max-model-len 131072 \
      --tt-config '{\"trace_region_size\":1500000000,\"fabric_config\":\"FABRIC_1D_RING\"}' \
      --additional-server-args='--trust-remote-code --max-num-batched-tokens 131072 --no-enable-prefix-caching'
  " > /tmp/laguna_fast_$2.log 2>&1 &
  echo $! > /tmp/laguna_fast_pgid_$2
  for i in $(seq 1 360); do sleep 5; curl -sf -m3 http://localhost:8000/health >/dev/null 2>&1 && break; done
  # warm (compiles bucket programs) then the measured cold point
  $BV/vllm bench serve --backend vllm --base-url http://localhost:8000 --endpoint /v1/completions \
    --model poolside/Laguna-XS-2.1 --dataset-name random --num-prompts 1 \
    --random-input-len 130048 --random-output-len 8 --max-concurrency 1 --ignore-eos >/dev/null 2>&1
  $BV/vllm bench serve --backend vllm --base-url http://localhost:8000 --endpoint /v1/completions \
    --model poolside/Laguna-XS-2.1 --dataset-name random --num-prompts 1 \
    --random-input-len 130048 --random-output-len 1024 --max-concurrency 1 --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el \
    --save-result --result-filename /tmp/laguna_fast_$2.json 2>&1 | tee -a /tmp/laguna_fast_$2.log
  g=$(cat /tmp/laguna_fast_pgid_$2); kill -TERM -"$g" 2>/dev/null; sleep 12; kill -KILL -"$g" 2>/dev/null
  sleep 5; tt-smi -r all >/dev/null 2>&1; sleep 10
}

run_point 0 off   # expect mean_ttft_ms ~298,000  (~435 tok/s)  == current bounded baseline
run_point 1 on    # expect mean_ttft_ms ~126,000  (~1037 tok/s) == recovered pre-regression speed
python3 -c 'import json;print("OFF ttft",json.load(open("/tmp/laguna_fast_off.json"))["mean_ttft_ms"]);print("ON  ttft",json.load(open("/tmp/laguna_fast_on.json"))["mean_ttft_ms"])'
```

Report ISL/OSL/E2EL and prefill tok/s (`ISL / (TTFT_ms/1000)`), never ms/tok. **Pass = ON TTFT back near ~126s (~1000 tok/s); OFF unchanged near ~299s.**

### 2. Prefill accuracy unchanged (offline, tiny — but opens a device; run only when mesh free)

The fix changes only chunk *size*, not math, so PCC must be identical. Run with the flag both off and on:

```bash
cd /home/ttuser/dev/tt-metal
export TT_METAL_HOME=/home/ttuser/.local/lib/model-bringup/tt-metal
export PYTHONPATH=/home/ttuser/dev/tt-metal
for F in 0 1; do
  TT_LAGUNA_PREFILL_FAST=$F $VBIN/python -m pytest -q \
    models/autoports/poolside_laguna_xs_2_1/tests/test_multichip_decoder.py::test_prefill_pcc \
    models/autoports/poolside_laguna_xs_2_1/tests/test_multichip_decoder.py::test_prefill_chunked_matches_hf \
    models/autoports/poolside_laguna_xs_2_1/tests/test_multichip_decoder.py::test_prefill_pipelined_matches_hf
done
```

Both flag values must PASS with identical PCC (these tests mutate `dec.PIPE_CHUNK`/`PREFILL_SDPA_CHUNK` at runtime; the fix's live-reading property preserves that, and FAST=1 only enlarges the outer chunk).

### 3. OOM sanity

`run_point 1 on` at `--max-model-len 131072` reaching the healthy state and completing the 131072 bench with no `out of memory` in `/tmp/laguna_fast_on.log` confirms ≤128k fits with the 8192-wide chunk. **Do NOT flip the flag with `--max-model-len > 131072`** (e.g. the 262144 tool-calling server) until a separate OOM check at that context is done.

## Summary

- **Root cause:** solely `e3aa655` — it started honouring `TT_LAGUNA_PIPE_CHUNK=2048`, quartering the outer prefill chunk (8192→2048) and thus ~4×-ing the redundant `seq²/CH` KV-prefix DRAM re-read in `_prefill_pipelined`, giving ~2.4× slower cold TTFT at 131072. Session item-2.5 edits are **not** a contributor.
- **Fix:** `TT_LAGUNA_PREFILL_FAST=1` (+ `TT_LAGUNA_PREFILL_FAST_CHUNK`, default 8192) restores the larger outer chunk for `≤131072` serving. Default `0` = byte-identical to current bounded behavior. Accuracy-neutral (no math change).
- **Expected:** cold prefill @131072 ~299s → ~126s TTFT, ~435 → ~1037 tok/s.

## Validation result (2026-08-03, on device)

Ran `scripts/validate_prefill_fast.sh` (mesh free after the bench run):
- **Accuracy:** `test_prefill_pcc` + `test_prefill_chunked_matches_hf` + `test_prefill_pipelined_matches_hf`
  with `TT_LAGUNA_PREFILL_FAST=1` → **33 passed** (PCC ≥ 0.995). Fix is accuracy-neutral, as predicted.
- **OOM:** none at `--max-model-len 131072` with the 8192 chunk — the fast path fits ≤128k.
- **Speed:** cold TTFT @130048 (APC off): FAST=0 **298.7 s** → FAST=1 **208.3 s** = **1.43×**.

**Caveat — partial recovery.** The `seq²/CH` model projected ~2.4× (→~126 s) assuming the outer
`_prefill_pipelined` chunk was the sole driver. Measured 1.43× means a residual ~1.65× gap remains. Leading
suspect: the **inner** per-chunk SDPA read is still bounded by `TT_LAGUNA_PREFILL_SDPA_CHUNK` (default 8192,
untouched by this fix), and/or other prefill-path changes since the 2026-07-24 22:43 sweep. Follow-up: A/B
`TT_LAGUNA_PREFILL_SDPA_CHUNK` and/or bisect the prefill path between 07-24 and today to close the rest.
