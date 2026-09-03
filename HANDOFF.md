# mHC fused post-mix — handoff

**Not for merge.** This file is the tip commit of `nmilicevic/ds-mhc-fused-post` and must be
dropped before that branch ever becomes a PR.

## What this branch is

`nmilicevic/ds-mhc-fused-post` branches off `nmilicevic/ds-mhc-explore` at `36b85c9ee66`, the tip
of PR #49039 (mHC parametrization kernel). It adds one thing: a fused kernel for the **post** half
of mHC, `X' = H_res·X + H_post^T·F(H_pre·X)` — the part applied *after* the sublayer F.

PR #49039 and its branch are untouched. This work becomes a separate PR later; the umbrella issue
is #55173, under #40703.

## The op

`ttnn.experimental.deepseek_prefill.mhc_post(y, residual, post, comb, consts, n)`

Computes, per work unit `(token-tile, column-tile)`:

    out[t, j*C+c] = post[t,j] * y[t,c] + sum_i comb[t, i*n+j] * residual[t, i*C+c]

Flat layout `[1,1,T,n*C]`, stream *i* at columns `[i*C, (i+1)*C)`. Production shape is
T = 5120/8 = **640** tokens/device, C = 7168, **n = 4 always**, fp32.

Two non-obvious implementation facts, both load-bearing:

1. **Coefficient broadcast is a matmul, not a broadcast LLK.** `post`/`comb` vary per token but not
   per column. `unary_bcast<ROW>` can only address column 0, so instead each coefficient column *k*
   is expanded across a full tile by `coeff_tile @ E_k`, where the host-built const `E_k`
   (`[n*n,32,32]`, row *k* all ones, else zero) is resident in `c_2`. Exact — the dropped terms are
   products with a true zero. Built by `build_bcast_consts(n)` in `tt_mhc.py`.
2. **The whole mix accumulates in DEST slot 0 on the FPU.** `tt_metal/hw/inc/api/compute/eltwise_binary.h:95`
   — the *two-argument* `mul_tiles_init(icb0, icb1)` passes `acc_to_dest = true`, and on
   Wormhole/Blackhole accumulation is the default. So five successive `mul_tiles(..., idst=0)` sum
   into the same slot. No SFPU fold, no scratch slots. DEST is cleared at `tile_regs_acquire()`.

   This cost a session to find. An earlier version folded terms with SFPU `add_binary_tile` inside
   the same `tile_regs_acquire` block; mixing FPU `mul_tiles` accumulation with SFPU adds in one
   block silently discards the running accumulation. Symptom was pcc 0.967 with a purely
   **batch-order-dependent** error — forward order lost the first batch, reversed order lost the
   last. Do not reintroduce an SFPU pass here.

DEST capacity, verified from `tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel.h:839`
(`get_dest_max_tiles`): SyncHalf + fp32 accum + 32x32 tiles = **4 tiles**, so dst 0..3 are legal.

## Measured (2026-09-02, Blackhole p150b, T=640 C=7168 n=4 fp32)

Host wall-clock, 10 iters, no profiler attached — **the same basis as the numbers published on
PR #49039**:

| ms/iter | composite | fused | |
|---|---|---|---|
| `hc_post` alone | 3.764 | **0.447** | 8.4x |
| whole wrapped sublayer (`forward`) | 6.149 | **2.961** | 2.08x |

The composite arm reproduces the published 6.138 ms baseline, so the A/B is anchored.

Tracy confirms the traffic model: **165.3 MB** moved per fused call vs **1547.7 MB** composite
(9.4x less), in **1 dispatch instead of 81**. That is **370 GB/s against this machine's measured
445 GB/s ceiling — 83% of peak**. Hard floor is 0.371 ms; ~0.08 ms of headroom remains and it is
pure bandwidth, not algorithm. Target was "0.6 ms" — met.

**Do not compare a Tracy device-FW number against the published wall-clock figures.** The profiler
charges marker writes to each of the composite's ~106 small ops per call, so the same code sums to
~9.2 ms under Tracy. Both are correct; they are different bases.

## Correctness

`pcc = 1.0000000` against the composite. `max|Δ| ≈ 1.6e-2` on values of O(10), i.e. ~4e-4 relative
— that is tf32-level FPU srcA/srcB precision, not a bug.

    pytest models/demos/deepseek_v3_d_p/tests/pcc/test_mhc_post_op.py   # 8 passed
    pytest models/demos/deepseek_v3_d_p/tests/pcc/test_mhc.py           # 37 passed, 1 skipped

The skip is pre-existing ("attention over a length-1 sequence is trivial"). `test_mhc.py` covers the
wrapped-sublayer path, which now routes through the fused op via `TtMHCWrap.forward`.

The composite `TtMHCWrap.hc_post` is deliberately **kept** in `tt_mhc.py` as the reference arm for
both the PCC test and the perf A/B. Do not delete it.

## Environment (this worktree)

`/localdev/nmilicevic/tt-metal` is a **different worktree with another active session** — do not
read, write, build, or run git against it. It shares the object store, so no bare `git stash` and
no checkout of other branches.

Every shell needs:

    cd /localdev/nmilicevic/tt-metal-ds-mhc-explore
    source python_env/bin/activate
    export TT_METAL_HOME=/localdev/nmilicevic/tt-metal-ds-mhc-explore
    export PYTHONPATH=$TT_METAL_HOME

Verify `echo $TT_METAL_HOME && which python` both contain `-ds-mhc-explore` before proceeding.

Build with `bash build_metal.sh` (incremental). **Kernels are JIT-compiled at runtime — editing
only `device/kernels/**` needs no rebuild.** Host-side op files do.

Run everything on device through tt-device-mcp (`owner: "[claude]nmilicevic"`), never bare pytest.

## Perf measurement

`models/demos/deepseek_v3_d_p/tests/perf/test_mhc_perf.py` now has `mhc-hc-post-fused` and
`mhc-e2e-composite` arms.

    python -m tracy -r -p -v -m pytest <file>::test_mhc_block_perf -k T640

with `MHC_PERF_ITERS=4`. **At the default 10 the profiler's 12000-marker DRAM buffer overflows** and
tracy dies in post-processing with `AssertionError: Device data missing: Op ... not present in
cpp_device_perf_report.csv`. That assertion means dropped markers, nothing else.

`-k` arguments must not contain spaces — the MCP job runner strips quoting, so
`-k 'block_perf and T640'` becomes a file-not-found on `and`. Use a `::` node id plus a single-token
filter.

Region attribution script: `scratchpad/analyze.py <ops_perf_results_*.csv>` — segments by signpost,
sums device-FW and computes traffic per region. Wall-clock A/B script: `scratchpad/ab_post.py`.
Both live in the session scratchpad, not the repo; re-create from this branch's perf test if lost.

## Next steps

- **bf16.** Explicitly deferred so far. The op is fp32-only today (`TtMHCWrap` asserts it, because
  the parametrization kernel requires it). Halving the element size halves the 165 MB and should
  take the op toward ~0.22 ms. This is the single largest remaining win.
- Open a PR for this branch — separate from #49039, and drop this file first.
- Scope reminder: only **5k chunks** perf matters. 1M total context runs the same 640 tokens.
- Perf guardrail carried from the PR: never commit anything worse than 20.5 ms total.

## Loose ends on PR #49039 (not this branch)

- CI runs 33637074885 (blackhole-e2e) and 33637078130 (l2-nightly) on `36b85c9ee66` were never
  checked.
- `tests/pipeline_reorg/` CODEOWNERS (`@roseli-TT`, `@tdowdallTT`) may need a manual reviewer
  request.
