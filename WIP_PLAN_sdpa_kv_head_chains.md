# WIP plan: SDPA K/V chains keyed by KV head, then Q-head crossing as its own optimization

Temporary file: delete before the PR merges. Answers pavlejosipovic's first review comment on PR #59695
(`pack_gqa_heads` vs. ring joint SDPA's KV-head-grouped transport).

## Background

- **Today's chains** (`sdpa_program_factory.cpp:1016-1440`, `reader_interleaved.cpp:142-180, 330-700`). They are built
  only for non-causal, unchunked, unwindowed calls without sliding window or `reuse_kv`. Chains are keyed by
  `(batch, Q head)`: `head_segments[nb*NQH + nh]`, plus the kernel test `nq == chain_head`. With GQA, each of a KV head's
  `NQH/NKH` Q heads gets its own chain, and each chain's injector reads the same K/V from DRAM.
- **`pack_gqa_heads`** relabels Q as `[B, NKH, pack*Sq, d]` (factory `:381-382`; the memory doesn't change). That one
  relabeling produces two effects:
  1. **Transport:** one packed head is one KV head, so the chains become per KV head.
  2. **Scheduling:** a Q chunk can run past the end of one Q head into the next. This only matters when `q_chunk` does
     not divide `Sq`. With `Sq % q_chunk == 0` the packed and unpacked flat Q-chunk orders are identical.
  The relabeling is the reason for all of its restrictions: no mask, no sink, no causal, tile-aligned unpadded `Sq`, and
  the writer remap/wrap for `output_concat_heads`.
- **Reviewer's ask:** get (1) directly by keying chains on the KV head, automatically, with Q scheduling and output
  layout unchanged. Keep (2) as a separate optimization if it earns its place. Causal is out of scope here: ring joint
  can chain causal calls only because its K range is uniform per ring iteration, while standard SDPA's reader stops
  each Q chunk at its own diagonal.

## Step 1: key the K/V chains by KV head (default on)

### Code

Host, `sdpa_program_factory.cpp`:
- [ ] Segment key at `:1050`: `nb*NKH + nh/q_heads_per_kv` instead of `nb*NQH + nh`, and size `head_segments` to match.
- [ ] **K and V heads can differ** (MLA prefill: `NKH != NVH`, comment at `:384`). Chain members must share both the K
      head and the V head. Key on the pair `(k_head, v_head)`, or on the finer grouping when one nests in the other.
      Fall back to the Q-head key otherwise. GQA (`NKH == NVH`) reduces to the KV head.
- [ ] `head = head_id % NQH` at `:1233` and the batch/head match in the multicast pass at `:1327-1328` switch to the new
      key. `chain.head` (runtime arg at `:1585`) now carries the key.
- [ ] Keep unchanged: start selection, the no-conflict wrap, the K/V CB slot-parity filter (`:1160-1175`), the
      heaviest-first sort for mixed counts, and the multicast eligibility checks. They run on the new segments as is.
- [ ] Debug logs: the per-chain message and the chains built/skipped and multicast summaries should name the key.

Kernel, `reader_interleaved.cpp`:
- [ ] `should_forward` / `should_receive` (`:482-484`) compare `chain_head` with the key computed from `nq`
      (`k_head = nq / q_heads_per_k`, `v_head` likewise), not with `nq`.
- [ ] `per_head_q_iter` resets on a key change instead of an `nq` change (`:363-366`). `q_iter` is used only for the
      `q_iter < next_core_q_chunks` gate (`:483`), so a count across the whole group is safe. Update the comment at
      `:332`.
- [ ] `ring_distributed_sdpa_program_factory.cpp` uses the same reader but is always causal, so the chain args are
      compiled out. No change needed; confirm it still builds.

Multicast policy. Longer chains are more likely to span grid rows, and multicast is all-or-nothing. Add a
**temporary** env knob, removed before merge, so the options can be A/B'd without rebuilding:
- `A`: always key by KV head. Accept losing multicast.
- `B`: key by KV head only if every resulting chain is still multicast-eligible whenever the Q-head-keyed build was.
  Otherwise build with the Q-head key.
- `C`: key by KV head, but split a group's segments at physical row boundaries so each piece can be multicast.
  This is more code; implement it only if A vs. B leaves a gap.
- `Q`: today's Q-head key, as the baseline in the same binary.

### Tests

- [ ] Bit-identical output vs. the Q-head key (knob `Q`) for GQA 32/8 non-causal at several grids and q chunks. Cover
      mixed q counts (8×7), whole KV heads per core (8×4) and a 12×10 case. `reuse_kv_test_utils.check_reuse_kv` shows
      the grid-skip pattern.
- [ ] Shapes `pack_gqa_heads` cannot take, which now get the shared transport: an attention mask (pplx serving pad
      mask), an attention sink, padded `Sq` (e.g. 500), and MHA (must be unchanged).
- [ ] MLA prefill shape with `NKH != NVH`: still correct, which exercises the K/V key choice.
- [ ] Existing `test_sdpa_prefill.py`, `test_sdpa_pack_gqa_heads.py`, `test_sdpa_reuse_kv.py`,
      `test_sdpa_single_k_chunk.py` and `test_mla_prefill.py` still pass.

## Benchmarks

Tooling is on `amorrison/high-batch-optim` under `models/demos/blackhole/pplx_embed_4b/perf_tools/`:
`bench_common_traced.py` (`make_traced`, traced device time), `bench_sdpa_bs1.py`, `bench_sdpa_batched.py`. Check them
out into the working tree uncommitted (`git checkout origin/amorrison/high-batch-optim -- <path>`) and point them at
this branch's build. For every configuration, record:
- latency (µs/call, traced, same clock/chip);
- chain topology: chains built, chain lengths, multicast on/off (debug logs);
- K/V DRAM reads per call: injectors plus non-participating cores, times K+V bytes per chunk sequence;
- bit-identity vs. the `Q` baseline.

### pplx-embed-4b shapes (the motivating workload)

Common to all calls: Qwen3-4B attention, Q `[B, 32, 512, 128]`, K/V `[B, 8, 512, 128]`, bfp8, non-causal,
LoFi, `fp32_dest_acc_en=False` (streaming kernel), `exp_approx_mode=True`, `scale=0.088388346`,
`output_concat_heads=True`. Blackhole P150.

| Batch | Placement | Shipped config (from `demo/_common.py`) | Arms |
|---|---|---|---|
| 1 | Q/K/V in L1 | unpacked: q256/k256 on 8×8; packed (`pack_gqa_heads`): q192/k512 on 11×8 | `Q` / `A` / `B` (`C` if it exists) at q256/k256 8×8, q256/k512 8×8, q256/k256 12×10 (`3fb93863a13`'s 49.7 → 38.5 µs case); plus current `pack_gqa_heads` at the same chunks, which must match `A` exactly since q256 divides 512 |
| 1, masked | same, plus the serving pad mask | unpacked only (pack rejects masks) | `Q` vs. `A`/`B`: new coverage, the transport win the mask used to block |
| 8 | Q DRAM, K/V L1 (`QWEN_HEADS_KV_L1`) | 12×8, q512/k512; reuse_kv q128 | `Q` / `A` / `B` at q512/k512 (chains on); reuse_kv q128 as the reference (chains off) |
| 16 | same | 12×10, q512/k512; reuse_kv q128 | same as bs8 |
| 32 | same | 12×10, q512/k512; reuse_kv q128 | same as bs8 |

Batched `reuse_kv` turns chains off, so step 1 doesn't change those calls. They're in the table as the reference that
the `reuse_kv` follow-up (retention plus forwarding) must beat. Step 1's batched number is the chains-on q512 arm.

### Multicast-policy examples

Run the same `Q` / `A` / `B` arms on at least:
- [ ] 32/8, S=512, q128, B=1, 12×10: the worked example. `Q` keeps multicast with 4-core row-local chains; `A` gets
      8 chains, mostly 16 cores crossing rows, so multicast is lost.
- [ ] 32/8, S=512, q192, B=1, 11×8 (packed in step 2): one KV head per grid row by construction, so multicast survives.
- [ ] A ratio that gives long chains across many rows, e.g. 64/8 or 28/4 at S=1024–2048.
- [ ] One MHA shape: all arms identical.

Decide the default from these. Write down the rule that picks it.

### In-model (optional, after standalone numbers)

Cherry-pick step 1 onto `amorrison/high-batch-optim` and run `perf_tools/sustained_run.sh` cold/sustained at bs1 and
bs16 (3 alternating rounds, the methodology in `doc/POSITIVE_RESULTS.md`). Check STS-B accuracy is unchanged (output is
expected to be bit-identical).

## Step 2: Q-head crossing as its own optimization

The code exists already: the `pack_gqa_heads` relabeling (`gqa_pack_factor`), writer args 23/24, the `packed_concat`
writer branch, and `head_wrap_tile_skip` in both `write_block` drains. The reader and compute kernels need no changes.
Work:
- [ ] Rebase onto step 1. With the relabeling, `q_heads_per_k == 1`, so the new key reduces to the packed head. Confirm
      the chains match step 1's at `Sq % q_chunk == 0`.
- [ ] Benchmark **crossing vs. padded at the same q chunk and grid, both with step-1 transport**:
      - q192 on 11×8 and on 12×10 (crossing: 11 chunks per KV group vs. padded: 4×3 = 12 with 64 rows of padding
        each);
      - q160/k256;
      - bs1 and bs8.
      `7519cd8f0a8`'s 35.9 → 29.3 µs also changed the grid from 12×10 to 11×8, so it doesn't isolate crossing.
- [ ] Decide:
      - **auto-enable** when non-causal, unmasked, no sink, tile-aligned `Sq`, and `Sq % q_chunk != 0` (output is
        bit-identical), and remove the flag;
      - keep a **narrow opt-in** flag;
      - or **drop** it (removes the writer wrap, the validation block and the API flag).
- [ ] Rewrite the docstring: K/V traffic wording moves to the automatic path. The flag (if kept) is described as
      "Q chunks may cross Q-head boundaries within a GQA group; useful when `q_chunk_size` does not divide `s`".
- [ ] Update `pplx_embed_4b/tt/attention.py` (`QWEN_SDPA_GQA_PACK*`) on `high-batch-optim` to match.

## Follow-ups (not this plan)

- `reuse_kv` review comment: plan same-core K/V retention together with forwarding instead of turning chains off.
  Builds on step 1's chain code.
- `math_thread_row_sums` review comment: matched PACK-vs-MATH profiles and the Blackhole-specific rationale.
- Causal GQA K/V sharing: key `(nb, kvh, q_chunk)` with group-major scheduling.

## PR logistics

- Step 1 changes default behavior for every non-causal GQA call, while the rest of the PR adds opt-in flags. Ask the
  reviewer whether step 1 should be its own PR or the first commit with `pack_gqa_heads` rebased on top.
- Reply to the review comment once the step-1 numbers are in. Point out that ring joint's causal support relies on a
  uniform K range per ring iteration, which is why causal isn't covered here.

## Results (fill in on hardware)

| Shape / config | Arm | µs/call | Chains (count × length) | Multicast | K/V DRAM reads | Bit-identical |
|---|---|---|---|---|---|---|
| | | | | | | |
