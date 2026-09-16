# Handoff — state as of 2026-09-16 20:05Z

**Read order:** [`README.md`](README.md) (the seeding doc: what this is, what to do next, what NOT
to do) → this file (state and commands) → [`MULTIHOP_SWA_HALO.md`](MULTIHOP_SWA_HALO.md) §11-12
(full record and reproduction).

## State

* Branch **`kmabee/gemma4-swa-multihop-halo`**, pushed to `tenstorrent/tt-metal`, based on
  `main @ df15dfd17d5`. Two commits, working tree clean, **no PR opened yet**:
  * `21df7f2031e` — implementation + these docs (18 files)
  * `e4c0df6a76e` — tests (2 files)
* Verified on device against exactly what is committed: accuracy gate **3 passed**; chunk-0 times
  **131.2 / 173.9 / 242.6 ms** for chunks 2048 / 4096 / 8192 on 8x4.
* A `/simplify` pass ran after the first commit — 10 cleanups applied, see "What the cleanup pass
  changed" below. Perf and correctness confirmed unchanged by it.
* Working dir `/data/kmabee/tt-metal-2`. Board released, nothing running.
* **Rebuild before the next device run**: pre-commit reformatted a kernel header after the last
  `--target install`, so `build_Release/lib/_ttnncpp.so` is one formatting pass stale. Kernel
  headers are JIT-compiled so results are still valid, but do not trust a mixed state.

## What the cleanup pass changed (context for a reviewer)

Four parallel review agents (reuse / simplification / efficiency / altitude). Applied: dropped a
provably-dead `link_base` ternary; replaced two derivable config fields with accessors; swapped a
hand-rolled semaphore-id scan for `find_available_semaphore_id` (restoring the missing
`NUM_SEMAPHORES` bound); routed all three hop-count derivations through one layout call; removed a
`hop_count()` + `TT_FATAL` the validator already enforced; made the device header call the host
`chunked_sliding_halo_hop_rows` instead of copying it; named `max_source_ranges`' ring-size
assumption **and added a validator check** (an over-wide ring previously produced an empty work plan,
i.e. silently no attention); collapsed a 4x-duplicated Python guard into one helper; parametrized
two pairs of copy-pasted tests; deleted a `GEMMA4_PREFILL_DUMP_DIR` debug block whose premise this
work disproved.

Deliberately NOT applied, with reasons: `halo_links_per_hop = max(1, links/hops)` (a real
generalization that would use all links when `links > hops`, but it is a no-op on this 2-link box so
it could not be tested); restructuring the halo helper to take a hop vector and return its cores;
promoting the link hand-off to the CCL layer or the fabric mux (mux overhead only amortizes around
1.5 MB/link, a hop here is ~0.5 MB); several micro-optimisations rated negligible.

## Both original tasks are DONE

**Task 1 — hops share links.** Implemented as sequential EDM channel hand-off (~50 lines), not the
kernel restructuring the old handoff planned. Gated: 1-hop, 2-hop and **4-hop** accuracy tests all
pass. Unlocked chunk 2048 at CP=8. Deployed 1-hop path unchanged (242.7 vs 243.1 ms).

**Task 2 — single-setting tradeoff.** Measured, 8x4, `gemma4_runs/mh_sweep`:

| chunk | hops | TTFT | T(256k) | worst case as sole setting |
|---:|---:|---:|---:|---:|
| 2048 | 4 | 131.3 ms | 28.66 s | 2.61x |
| **4096** | 2 | 174.2 ms | 17.19 s | **1.61x — best single setting** |
| 8192 (today) | 1 | 242.7 ms | 13.71 s | 1.85x |
| 16384 | 1 | 443.7 ms | 11.55 s | 3.38x |
| 32768 | 1 | 928.2 ms | 10.96 s | 7.07x |

## There is no big lever left in the halo — measured, not assumed

The multi-hop halo protocol costs **~4-5 ms at chunk 4096** (~0.09 ms per sliding layer), upper bound
8.9 ms. Measured by holding the chunk fixed and halving the sliding window so the hop count drops to
1, with chunk 8192 as the control (hop count 1 either way, so its gap is pure attention compute):

| chunk | window | hops | chunk-0 |
|---:|---:|---:|---:|
| 4096 | 1024 | 2 | 174.2 ms |
| 4096 | 512 | 1 | 165.3 ms |
| 8192 | 1024 | 1 | 242.7 ms |
| 8192 | 512 | 1 | 232.9 ms |

So do **not** spend time on: more fabric links, a fabric mux, a line-multicast halo, or optimising
the rendezvous protocol. The halo payload is invariant at ~2 MiB/device/layer and the protocol is
already cheap. Chunk 2048's 131.3 ms is near the floor for that chunk size.

What is left is the **~69 ms fixed per-chunk floor** (~1.15 ms per layer over 60 layers), which is
what makes small chunks inefficient. Attacking that means profiling where a minimum-size chunk
spends its time — `tools/tracy/process_ops_logs.py` gives per-op `PM IDEAL [ns]`, `PM FPU UTIL (%)`,
`NOC UTIL (%)`, `ETH BW UTIL (%)`, `DRAM BW UTIL (%)` (never pass `--device-trace-profiler`). That is
a fresh investigation, not a continuation of this one.

## Method warning

Both wrong conclusions in this investigation came from **extrapolating a parametric fit outside its
fitted range** — first "K grows 13.4 ms/hop" from two points differing in two variables, then "a flat
21.5 ms multi-hop adder" from a quadratic fitted on 8192-32768 and evaluated at 2048-4096. The
measured per-chunk values are cheap: one 9m53s sweep gives all five chunk sizes. Use them, and treat
any fit as interpolation only.

## Traps — do not re-pay

* **`ninja <target>` does NOT install.** It links `build_Release/ttnn/_ttnncpp.so`, but Python loads
  `build/lib/_ttnncpp.so` via `ttnn/ttnn/_ttnn.so`'s DT_NEEDED, and nothing copies between them.
  Cost ~40 min and three bogus device results this session. Use
  `cmake --build build_Release --target install`, then verify with
  `ls -la build_Release/lib/_ttnncpp.so`. Kernel `.cpp` edits ARE JIT-compiled and do take effect —
  which is what makes a mixed host+kernel change so confusing.
* This box wanted a `tt-smi -glx_reset` between runs at one point; a hang shows as ~12x CPU-time to
  elapsed-time on the pytest child with no new files under `~/.cache/tt-metal-cache`.
* `timeout --signal=INT` does not kill a device-hung pytest; use `timeout -k 10 <s>`.
* `--collect-only` opens all 32 chips — never run it alongside a live job.
* Cross-chunk-size hidden-state PCC has no resolution at 60 layers (two known-good sizes differ by
  0.992 / worst row 0.456). Always run that control before believing a diff.

## Commands — reproducing the 2k and 4k runs

Full environment, build and expected output in
[`MULTIHOP_SWA_HALO.md`](MULTIHOP_SWA_HALO.md) §12. The short version:

```bash
source /data/kmabee/gemma4_runs/env.sh   # TT_METAL_HOME, HF paths, mesh-8x4 tt_cache
cd $TT_METAL_HOME

# 4k chunk, 8x4 -> chunk-0 174.0 ms (2 halo hops)
timeout -k 10 900 ./python_env/bin/python3 -m pytest \
  models/demos/gemma4_d_p/demo/text_demo_prefill.py \
  -k "readback_final-ctx_32k-chunk4096-text-8x4" -sv

# 2k chunk, 8x4 -> chunk-0 131.2 ms (4 halo hops time-sharing 2 links)
timeout -k 10 900 ./python_env/bin/python3 -m pytest \
  models/demos/gemma4_d_p/demo/text_demo_prefill.py \
  -k "readback_final-ctx_32k-chunk2048-text-8x4" -sv

# all five chunk sizes, one device session, ~10 min
timeout -k 10 2400 ./python_env/bin/python3 -m pytest \
  models/demos/gemma4_d_p/demo/text_demo_prefill.py \
  -k "readback_final and ctx_32k and 8x4 and (chunk2048 or chunk4096 or chunk8192 or chunk16384 or chunk32768)" -sv

# correctness gate: 1-, 2- and 4-hop, ~70 s, must be `3 passed`
timeout -k 10 900 ./python_env/bin/python3 -m pytest \
  tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py \
  -k "chunked_sliding_linear_topology_accuracy or multi_hop_sliding_halo_linear_topology_accuracy" -sv
```

Chunk 0's device time IS the TTFT for a prompt that fits one chunk, so `ctx_32k` is enough; the
later chunks give the per-prefix slope. Analysis scripts (not in-tree) live in
`~/debug-docs/gemma4_swa_multihop_halo-noissue/scripts/`: `chunk_tradeoff_measured.py` builds the
tables above, `op_perf_model.py` evaluates ttnn's own perf model, `probe_links.py` counts fabric
links in ~40 s.

## Original command notes


```bash
source /data/kmabee/gemma4_runs/env.sh
# accuracy (69 s, all three hop counts)
timeout -k 10 900 ./python_env/bin/python3 -m pytest tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py \
  -k "chunked_sliding_linear_topology_accuracy or multi_hop_sliding_halo_linear_topology_accuracy" -sv
# full chunk sweep (9m53s, gives a(C) and slope(C) for every chunk size)
timeout -k 10 2400 ./python_env/bin/python3 -m pytest models/demos/gemma4_d_p/demo/text_demo_prefill.py \
  -k "readback_final and ctx_32k and 8x4 and (chunk2048 or chunk4096 or chunk8192 or chunk16384 or chunk32768)" -sv
```
Analysis: `scripts/chunk_tradeoff_measured.py` (the tables above), `scripts/op_perf_model.py`
(ttnn's own perf model evaluated for gemma4), `scripts/probe_links.py` (link probe, ~40 s).
