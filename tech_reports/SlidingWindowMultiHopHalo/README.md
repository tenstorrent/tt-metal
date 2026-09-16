# Multi-hop sliding-window halo for Gemma4 chunked prefill

**Start here.** This directory is the complete record of making Gemma4's sliding-window attention
work at prefill chunk sizes below `window * CP`, and of measuring what chunk size actually costs.
It is written to be picked up cold by a new session.

---

## 1. What the work was

Gemma4's sliding window is 1024 tokens. Under context parallelism each rank holds a `chunk / CP`
slab of the sequence, so a rank whose slab is narrower than the window must pull the missing history
from its cyclic predecessors — one **hop** per predecessor:

| CP | chunk | per-rank slab | hops needed |
|---:|---:|---:|---:|
| 8 | 8192 | 1024 | 1 |
| 8 | 4096 | 512 | 2 |
| 8 | 2048 | 256 | 4 |

`RingJointSDPA` supported exactly one hop, which put a hard floor of `chunk >= 8192` at CP8 and
therefore a floor under TTFT. This work generalises the halo to any hop count, and — because an
ERISC exposes only one worker sender channel per direction — makes hops **time-share** the fabric
links rather than requiring one link each.

## 2. Status: done, gated, and on a branch

| | |
|---|---|
| Multi-hop halo, any `h <= ring_size` | **works** |
| Hops sharing links (4 hops on 2 links) | **works** |
| Correctness, 1 / 2 / 4 hops, SP8 + linear fabric | **3 passed** (~70 s) |
| Deployed 1-hop path | **unchanged** — 242.6 ms vs 243.1 ms before |
| Branch | `kmabee/gemma4-swa-multihop-halo`, pushed, **no PR opened yet** |

Two commits: `21df7f2031e` (implementation + these docs) and `e4c0df6a76e` (tests).
[`NEXT_SESSION.md`](NEXT_SESSION.md) has the exact state, the cleanup-pass record and the commands.

## 3. The numbers that matter

Measured on a BH Galaxy, mesh 8x4 (CP8/TP4). Chunk 0's device time **is** the TTFT for any prompt
that fits in one chunk.

| chunk | hops | TTFT | 256k prefill | worst case as sole setting |
|---:|---:|---:|---:|---:|
| 2048 | 4 | 131.3 ms | 28.66 s | 2.61x |
| **4096** | 2 | 174.2 ms | 17.19 s | **1.61x — best single setting** |
| 8192 (today's default) | 1 | 242.7 ms | 13.71 s | 1.85x |
| 16384 | 1 | 443.7 ms | 11.55 s | 3.38x |
| 32768 | 1 | 928.2 ms | 10.96 s | 7.07x |

Three conclusions a follow-up session should not re-derive:

1. **4096 is the best single chunk setting** if one has to serve all traffic — 39% better TTFT below
   8k tokens for 25% at long context, and a lower worst case than today's 8192. 2048 is ~2x worse
   overall and is a hard pass. 32768 is the throughput optimum at 256k.
2. **The halo is not the problem.** Its payload is invariant at ~2 MiB per device per layer whatever
   the hop count, and the multi-hop protocol costs only ~4-5 ms at chunk 4096. Fabric links were a
   *connection-slot* limit, never a bandwidth one.
3. **What makes small chunks expensive is a ~69 ms fixed per-chunk floor** (~1.15 ms per layer over
   60 layers), paid no matter how few tokens the chunk holds. That is the only real lever left.

## 4. The documents

| file | what is in it |
|---|---|
| **[MULTIHOP_SWA_HALO.md](MULTIHOP_SWA_HALO.md)** | The full record. §1-3 the constraint, how the halo works, the design. §4-7 modelled estimates, code scope, test ladder, risks. §8-10 sessions 1-2 including every bug and its root cause. **§11 the link-sharing design and all measurements.** **§12 how to reproduce every run.** |
| [TLDR_FABRIC_LINKS.md](TLDR_FABRIC_LINKS.md) | Superseded by §11, kept for the fabric-link probe results, which still stand: this box physically has 2 chip-to-chip links, and no fabric-type or reset knob changes that. |
| [NEXT_SESSION.md](NEXT_SESSION.md) | Handoff: state, the measured tables, reproduction commands, and the traps. |

Analysis scripts are **not** in-tree; they live in
`~/debug-docs/gemma4_swa_multihop_halo-noissue/scripts/` — `chunk_tradeoff_measured.py` (the tables
above), `op_perf_model.py` (ttnn's own SDPA perf model evaluated for Gemma4), `probe_links.py`
(counts fabric links in ~40 s).

## 5. Reproduce in two commands

Full environment and expected output in [MULTIHOP_SWA_HALO.md §12](MULTIHOP_SWA_HALO.md).

```bash
# 2k chunk on 8x4 -- 4 halo hops time-sharing 2 links -> chunk-0 ~131 ms
timeout -k 10 900 ./python_env/bin/python3 -m pytest \
  models/demos/gemma4_d_p/demo/text_demo_prefill.py \
  -k "readback_final-ctx_32k-chunk2048-text-8x4" -sv

# correctness gate, ~70 s, must print `3 passed`
timeout -k 10 900 ./python_env/bin/python3 -m pytest \
  tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py \
  -k "chunked_sliding_linear_topology_accuracy or multi_hop_sliding_halo_linear_topology_accuracy" -sv
```

## 6. If you are picking this up for follow-up work

**Worth doing.** Attack the ~69 ms fixed per-chunk floor. Profile a minimum-size chunk and find what
a chunk pays regardless of token count; `tools/tracy/process_ops_logs.py` reports per-op
`PM IDEAL [ns]`, `PM FPU UTIL (%)`, `NOC UTIL (%)`, `ETH BW UTIL (%)`, `DRAM BW UTIL (%)`. Halving
that floor is roughly what turns 4k from "arguable premium" into "obvious win". Never pass
`--device-trace-profiler` — it destroys op attribution.

**Measured to be worth ~nothing — do not spend time here.** More fabric links; a fabric mux core;
a line-multicast halo (slot indexed by `sender_ring_id mod h` so one multicast replaces `h`
unicasts); optimising the rendezvous protocol. All target per-hop or halo-bandwidth cost, and both
are measurably near zero (§11.3).

**Out of scope for this work.** Variable/dynamic chunk size selected per request by ISL — a separate
effort owns that.

**Two method warnings, each of which produced a confidently wrong conclusion here.**
1. *Do not extrapolate a fitted cost model outside its fitted range.* A quadratic fitted on chunks
   8192-32768 mispredicted chunk 4096 by ~13 ms and made a ~4 ms hop cost look like ~21 ms. Measuring
   all five chunk sizes takes one 10-minute sweep; do that instead.
2. *`ninja <target>` does not install.* It links `build_Release/ttnn/_ttnncpp.so`, but Python loads
   `build/lib/_ttnncpp.so`, and nothing copies between them. Use
   `cmake --build build_Release --target install` and check the mtime of
   `build_Release/lib/_ttnncpp.so` before trusting any device result. Kernel `.cpp` edits *are*
   JIT-compiled and do take effect, which makes a mixed host+kernel change especially confusing.

## 7. Invariants worth knowing before touching the halo

* `halo_tokens = ceil((W-1) / k_chunk) * k_chunk` — 1024 here, and **independent of chunk size**.
  Per-device Q slab is `chunk / CP`, so `hops = ceil(halo_tokens / (chunk / CP))`.
* Every device ships exactly `halo_tokens` of KV per layer, split across its hops. More hops means
  more, smaller, differently-addressed transfers — not more bytes.
* One worker core per (fabric link, direction): two *concurrent* workers on one EDM sender channel
  deadlock. *Sequential* reuse is supported and is what this branch does — `close_start()` persists
  the producer cursor "for the next connection on this channel" and `open_start()` adopts it.
* One incrementer per semaphore. `Semaphore::up` can lose updates on WH/BH with several
  incrementers, which is why all hops funnel their ready-increment through one rendezvous core.
* Cross-chunk-size hidden-state PCC has **no resolution** at 60 layers — two known-good chunk sizes
  differ by PCC 0.992 / worst row 0.456. Always run that control before believing a diff.
