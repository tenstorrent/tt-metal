# MiniMax-M3 prefill zone profiling

Per-zone device-kernel time for one prefill chunk attending an existing KV cache — the
"5k attended to 25k / 55k" case — split into the parts we care about: `ring_joint_sdpa` and the dense
MLP on the dense layers (0-2), and the full MSA + MoE breakdown on the sparse layers (3-59).

## Run it

```bash
./run_prefill_profile.sh                        # both 5k@25k and 5k@55k, bf4 experts
PROFILE_CACHE=25600 ./run_prefill_profile.sh    # one depth only
PROFILE_NUM_LAYERS=6 ./run_prefill_profile.sh   # fast bring-up: 3 dense + 3 sparse layers
NOC_TRACES=1 ./run_prefill_profile.sh           # + measured DRAM/NOC util per op (needs tt-npe)
EXPERT_DTYPE=bf8 ./run_prefill_profile.sh
```

The wrapper follows the same conventions as `run_prefill_perf.sh`: venv activate, `tt-smi -glx_reset`
per run, real tokens tiled out of a long golden trace, `LOGURU_LEVEL=INFO` + a DEBUG grep filter, logs
under `prefill_profile_logs/`. It then parses each run's CSV and writes `zones_*.html`.

## How it works

| piece | what it does |
|---|---|
| [utils/profiler_utils.py](../../utils/profiler_utils.py) | `zone(name)` context manager: emits `M3_ZONE_START/END <name>` Tracy signposts (+ a host Tracy zone). No-op unless `M3_PROFILE_ZONES=1`. |
| [profile_prefill.py](profile_prefill.py) | warmup → fill cache to N tokens (un-profiled) → run ONE chunk inside a `profiled_chunk` zone, reading the device profiler after every layer. |
| [parse_zone_perf.py](parse_zone_perf.py) | streams the ops CSV, rebuilds the zone hierarchy from the signpost rows, rolls up ns / ops / bytes / GB/s per zone per device. |

Attribution: CSV rows are in host-enqueue order, so the ops between a zone's START and END signposts
are exactly the ops that zone enqueued. Each op is charged to the innermost open zone and every
enclosing one, so a parent's total always covers its children. Only zones under `profiled_chunk` are
reported — that is what excludes warmup and the cache-prefix chunks, whose ops share the same CSV.

Same mechanism deepseek_v3_d_p uses (`forward_layer_{i}_start` in `tt/tt_prefill_transformer.py`,
`MLA_START`/`MLA_END` in `tt/mla/mla.py`), extended to a nested hierarchy.

## Reading the report

- **`ms` is the worst device's sum.** With 32 chips the mesh waits for the slowest, so the max is the
  wall-clock-relevant number. `skew ms` (max − min) is what separates a genuinely slow CCL from one
  that is merely waiting on a peer.
- **`GB/s` is bytes-moved ÷ that zone's device time**, with bytes computed from each op's input+output
  shapes and dtypes (block-float formats include their block scales). Compare against the chip's DRAM
  ceiling to judge whether a zone is bandwidth-bound.
- **`DRAM%` / NOC util** only appear with `NOC_TRACES=1` (tt-npe simulates the traffic and the profiler
  fills `DRAM BW UTIL (%)` / `NOC UTIL (%)` per op).
- `ops/layer` on a parent zone counts its children's ops too.

## Two gotchas that will bite

**The 1000-op profiler buffer.** Only ~1000 ops per device are buffered; one M3 chunk enqueues
~50-60 ops × 60 layers. `profile_prefill.py` therefore calls `ttnn.ReadDeviceProfiler` after every
layer (via the model's `on_layer_complete` seam) in *every* phase — warmup and prefix included, because
an un-drained phase overflows the buffer and the profiled chunk then comes back empty. This inflates
host wall-clock; take latency numbers from `run_prefill_perf.sh`, not from here.

**`PROFILE_SKIP_PREFIX=1` is approximate.** It skips the prefix fill and attends a zeroed cache. Op
shapes and therefore costs are identical, but the attention outputs are garbage, so the hidden states
reaching the MoE router are unrealistic and the expert load imbalance (`dispatch`, `experts_mm`,
`combine`) is not representative. Bring-up only. For the same reason the harness uses real tiled tokens
rather than random ids.

## The `cache_read/deshard` hypothesis

The packed KV cache is one tensor per K/V/index_k of shape
`[num_users*num_layers, 1, seq_local, head_dim]` ([attention/kv_cache.py](../../tt/attention/kv_cache.py)).
The MSA cache-read path converts the **whole** tensor from NdShard to DRAM-interleaved on **every**
sparse layer — the round-robin bank mapping is only intact for the full tensor, so it cannot slice one
layer's slot first ([attention/prefill.py](../../tt/attention/prefill.py)). At 61440 tokens that is
~63 MiB per tensor, read+write, ×3 tensors, ×57 layers ≈ 20+ GiB of DRAM traffic per chunk — plausibly
more than every expert weight read combined.

`profile_prefill.py` logs the expected byte count at startup, and the report separates
`attn/cache_read/deshard` from `attn/cache_read/slice`, so the measured cost and GB/s land right next
to the prediction.
