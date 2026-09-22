# Qwen3.5-2B prefill on the 32-chip Blackhole Galaxy: measured roofline and where the time goes

Companion to `SP_PREFILL_README.md`, which covers the SP x TP wavefront itself. This file
records what the hardware actually delivers, why the under-2 ms target is out of reach, and the
per-op cost ranking that any further optimisation should be driven by.

Everything below is **measured on this box**, traced (not eager), best-of-N. Reproduce with the
scripts in `tests/perf/roofline/`. Numbers dated 2026-09-22.

## Headline: 2 ms is not reachable; ~6.5 ms is the floor

| | per layer | 24 layers |
|---|---|---|
| 2 ms target | 83 us | 2.00 ms |
| measured compute floor (best split, MLP only) | 180 us | **4.31 ms** |
| measured compute floor (all ops, MLP ~= 66% of layer FLOPs) | ~270 us | **~6.5 ms** |
| **current, verified** | ~1600 us | **42.79 ms** |

The floor above already assumes zero collectives, zero data movement and perfect overlap. 2 ms
would need ~90% MFU on shapes that measure 17-28%.

### Why: the model is too small to fill the machine

One Blackhole chip's peak, swept over shapes (`bench_peak.py`):

| shape (M x K x N) | TFLOPS |
|---|---|
| 2048 x 4096 x 4096 | **193** |
| 1024 x 2048 x 6144 | 143 |
| 2048 x 2048 x 2048 | 116 |
| 512 x 2048 x 6144 | 86 |

193 TFLOPS is the practical ceiling, **not** the ~300 on the spec sheet. Qwen3.5-2B split 32
ways never presents a shape that big: 2B params / 32 chips = 64M params per chip, and 4096
tokens / 8 spans = 512 tokens. Per-chip MLP wall-clock for one layer, for every split of 32
chips (`bench_split.py`):

| split | tok/chip | MLP/layer | MFU | x24 layers |
|---|---|---|---|---|
| SP4 x TP8 | 1024 | 211 us | 24% | 5.05 ms |
| **SP8 x TP4** | 512 | **180 us** | **28%** | **4.31 ms** |
| SP16 x TP2 | 256 | 229 us | 22% | 5.50 ms |
| SP32 x TP1 | 128 | 306 us | 16% | 7.35 ms |

**Adding chips to a layer makes it worse.** 32-way token split runs at 16% MFU, 8-way at 28%:
more parallelism shrinks each matmul below what fills 120 Tensix cores. This is the single most
important fact for anyone planning a "use all 32 chips on one layer" scheme.

### L1 residency buys nothing (this refutes the SRAM / weight-reload thesis)

Same shape, weights in L1 vs DRAM, traced (`bench_l1_vs_dram.py`):

```
weights DRAM:  289.7us   33.4 TFLOPS
weights L1:    290.0us   33.3 TFLOPS
```

0.1% apart. bfloat4_b vs bfloat8_b is likewise a wash. **These matmuls are utilisation-bound,
not memory-bound**, so keeping the whole 2B model in SRAM, and the tt-blaze reload pipeline as a
*latency* lever, do not apply. (Reload solves kernel-binary staging, which is not what we pay
for here.) One layer's weights are 48 MB against 189 MB of L1, so residency is *possible* -- it
just does not help.

## Where the 42.79 ms actually goes

Device-kernel time by op, from a Tracy capture of the SP x TP run, **excluding
Send/RecvDirectAsync** (those are wavefront *wait*, not work -- they read as 96% of raw total
and will mislead you):

| op | share of work | per call | cores used |
|---|---|---|---|
| **AllGather + ReduceScatter** | **45%** | 79 / 155 us | 10-12 |
| Matmul | 15% | 41 us | 77 |
| ChunkGdnScan + ChunkGdnPrep | 11% | 146 / 67 us | **8** / 64 |
| SDPA | 9% | 494 us | 120 |
| Slice / BinaryNg / Tilize / Untilize / Concat | 10% | 4-15 us | 113-120 |

We spend **3x more on collectives than on matmul.** Per layer that is 4 all-gathers + 2
reduce-scatters.

### Collectives are a fixed ~120 us each, NOT bandwidth

Isolated, traced, 8 chips, Linear topology (`bench_ccl.py`):

| collective | bytes | time |
|---|---|---|
| all_gather bf16 (1024x256) | 0.52 MB | 123.0 us |
| reduce_scatter bf16 (1024x2048) | **4.19 MB** | **117.5 us** |
| all_gather fp32 | 1.05 MB | 207.6 us |
| reduce_scatter fp32 | 8.39 MB | 201.3 us |
| all_gather, tiny (1024x32) | 0.07 MB | 47.8 us |
| **fused ttnn.all_reduce** (1024x2048) | 4.19 MB | **223.3 us** |

8x the payload for the same time. There is a large fixed cost and only a weak per-byte term, so
**collective COUNT is the lever, not message size or tensor layout.** Note the isolated 123 us
exceeds the in-model average of 78.8 us, so the in-model figure is already partly overlapped --
the 79 us is not idle skew waiting to be scheduled away.

fp32 costs ~1.7x bf16. 1152 of the reduce-scatters (one per GDN layer) run in fp32. That is
deliberate, not an oversight: bf16 partial sums on the GDN out-proj took PCC to ~0.69. See
`ccl_cast()` in `tp_common.py`.

## Change in flight: replicated residual (`QWEN36_REPL_RESIDUAL=1`)

**Status: implemented, verification run not yet complete. Do not trust until the A/B lands.**

Today each half-layer pays three collectives:

```
reduce_scatter  ->  norm stats all_gather  ->  norm output all_gather
(residual left fractured on hidden)   (forced DistributedNorm)   (re-materialise full hidden)
```

The fracture is what forces the distributed norm, and the norm's output gather rebuilds exactly
the full hidden dim the next column-parallel matmul wanted anyway. Classic Megatron TP instead
keeps the residual **replicated**: one fused all-reduce, both norm gathers local.

Arithmetic from the table above, per half-layer: `117 + 48 + 123 = 288 us` today vs `223 us`
fused. Using in-model rather than isolated rates the saving is ~355 us/layer, i.e. **~8.5 ms of
a hoped-for 42.8 -> ~34 ms**. The activation-memory cost of replicating is the reason the
fractured layout exists at all, and it does not bind on a 2B model with 189 MB of L1.

Implemented as `tp_common.residual_all_reduce()` behind one predicate,
`tp_common.repl_residual_enabled()`, which also forces `is_distributed_norm() -> False` and
`agmm_disabled() -> True`. These three gates MUST move together; they have drifted apart twice
before and the failure mode is a silent K mismatch in the next matmul.

One subtlety worth knowing: `tt_all_reduce` **ignores** `cluster_axis` on a flat mesh (it has a
`1 in mesh.shape` branch), which is why every call site passes `cluster_axis=0` even on a (1,N)
submesh where axis 0 has extent 1. `ttnn.all_reduce` does *not* ignore it and would silently
reduce over a single device, returning the unreduced partial. `residual_all_reduce` passes
`cluster_axis=None` on a flat mesh.

## Things that do not work, and why

- **All 32 chips on one layer, one layer resident at a time (tt-blaze "partial unroll" shape).**
  Killed by the split table: 32-way is 16% MFU vs 28% at 8-way. Fewer, bigger shards win.
- **Replicating MLP weights and splitting tokens to avoid the MLP all-reduce.** The replacement
  token-dim all_gather costs 123.8 us -- the same fixed cost as the collective it removes --
  while the MLP compute goes 211 -> 306 us. Net loss.
- **Keeping the model in L1 / tt-blaze runtime binary reload as a latency lever.** See above:
  L1 vs DRAM is 0.1%.
- **More ethernet links.** `TT_CCL_LINKS=3` and `=4` both fail with
  `fabric.cpp:184 link_idx < candidate_eth_chans.size()`. Only 2 channels exist on this mesh.
  The `tt-multichip-ccl-review` skill's "Galaxy = 4 links" is Wormhole TG, not BH Galaxy.
- **Wider core grids for collectives.** Grid 72 -> 108 cores changed nothing; they are not
  core-bound.
- **AGMM (`all_gather_minimal_matmul_async`).** A pessimisation here: 44.19 ms with vs 42.79 ms
  without. Keep `QWEN36_NO_AGMM=1`.
- **SP=16.** The mesh caps at 8 columns. SP16 x TP2 also measured 51.76 ms e2e -- the 16-hop
  wavefront costs more than the cheaper collectives save.

## Pipeline parallel: why it was dropped

Cost model fitted to measured per-layer times on this box:

    T_layer(span) = 0.82 ms + 1.67 us/token

With `S` stages, `L=24` layers and `M` microbatches over 4096 tokens:

    total = (M + S - 1) * (L/S) * T_layer(4096/M)

| stages | chips | best M | tok/microbatch | optimum |
|---|---|---|---|---|
| 4 | 4 x TP8 = 32 | 5 | 819 | 105.0 ms |
| 8 | 8 x TP4 = 32 | 8 | 512 | 75.4 ms |
| 16 | 16 x TP2 = 32 | 11 | 372 | 56.2 ms |
| **24** | **24 x TP1 = 24 (8 chips idle)** | 14 | 293 | **48.4 ms** |

The best case, 48.4 ms, still loses to the current 42.79 ms, and it needs the *maximum*
possible stage count -- one layer per device, which caps at 24 because there are only 24
layers, leaving 8 of the 32 chips idle.

The reason PP loses is that **the 0.82 ms fixed cost is per (layer x microbatch)**. Splitting
into M microbatches to fill the pipeline multiplies that overhead by M, so you pay it 14 times
over. PP only wins when the fixed per-layer cost is small next to the per-token term; here it
is 0.82 ms against 0.49 ms of actual token work at the optimum.

## Open leads, ranked by measured size

1. **Collective count.** 45% of work. The replicated-residual change above is the first cut;
   after it, the floor is 2 collectives/layer at ~120-220 us each.
2. **GDN plumbing, ~18% of work.** `ChunkGdnScan` runs on **8 of 120 cores** (146 us/call), and
   Slice/BinaryNg/Tilize/Untilize/Concat add another ~10% in pure data movement. Owner for the
   GDN layer work is Izajasz Wrosz.
3. **Fabric packet size.** The CCL layer logs
   `Fabric packet size 4352 B is suboptimal for transporting 2048 B pages. Configure 8192 B` --
   i.e. 2 tiles per packet where 4 would fit. There is no env knob (`TT_METAL_FABRIC_*` has no
   packet-size override); it needs a fabric-builder change. Untried, and it bears directly on
   the single largest cost.
4. **Conv2d in the GDN path is completely flat in sequence length** (0.173 ms at span 512 vs
   0.174 ms at span 1024) -- ~4.2 ms across 24 layers of pure overhead. A tap-based rewrite (4
   shifted multiply-adds instead of Conv2d + Halo + sharded conversions) is blocked on shifts of
   1-3 rows not being tile-aligned in TILE layout.

## Reproducing

```bash
cd tt-metal && source python_env/bin/activate
export PYTHONPATH=$PWD LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH TT_METAL_HOME=$PWD TT_METAL_RUNTIME_ROOT=$PWD
export TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0 HF_MODEL=Qwen/Qwen3.5-2B
unset TT_METAL_CACHE
R=models/demos/blackhole/qwen36/tests/perf/roofline

python $R/bench_peak.py        # per-chip peak matmul TFLOPS
python $R/bench_split.py       # MLP wall-clock floor for each SP x TP split of 32 chips
python $R/bench_l1_vs_dram.py  # L1 vs DRAM weight residency A/B
python $R/bench_ccl.py         # collective cost vs bytes and dtype
python $R/bench_allreduce.py   # fused all_reduce vs reduce_scatter + all_gather
```

Gotchas that cost real time here:

- Set `ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)` **before** opening the mesh, or every
  collective dies with "Trying to get un-initialized fabric context".
- Open the full `MeshShape(4, 8)` and carve with `create_submeshes` (plural). Partial opens
  fail the fabric handshake.
- On a `(1, N)` submesh the live cluster axis is **1**, not 0.
- Creating a second submesh in the same process after using the first **hangs**
  (`bench_ccl_hops.py` is the known-bad example; run one TP width per process).
- Do not `pkill` a run mid-fabric-op; it can wedge the device. `tt-smi -r` recovers it.
  **Never `tt-smi -glx_reset`** -- it bricks this box.
