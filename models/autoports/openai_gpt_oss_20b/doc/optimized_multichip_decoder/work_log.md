# GPT-OSS 20B optimized multichip decoder work log

## 2026-07-23: initialization and topology audit

- Starting HEAD: `753f7207969`
- Starting worktree: clean
- Scope: optimize `tt/multichip_decoder.py` in place on the fixed four-card
  P300c mesh; no full-model or vLLM work
- Skills: `optimize`, `tt-device-usage`, `graph-fusing`, `shard-advise`
- Accepted baseline source: `doc/multichip_decoder/`

Read the complete skill contracts, the completed multichip README/work log,
the selected source path, current tests, the accepted profiler tables, and
the context contract.  The operation-topology audit is recorded in
`README.md` before tuning.

Hardware checks were serialized:

```bash
timeout 60 tt-smi -ls --local
python -c "import ttnn; mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,4), trace_region_size=0); ttnn.close_mesh_device(mesh); print('MESH_SMOKE_OK')"
```

Both passed and all four Blackhole P300c devices were visible.  The known
low `/dev/shm` and unknown `B850M-C` metadata warnings remain non-fatal.

The first implementation target is the dominant output/expert collective.
The accepted path converts a width-sharded L1 partial to interleaved L1 and
then invokes generic `ttnn.all_reduce`; the profile consequently reports
material `AllBroadcastDeviceOperation` rows.  The source contains a
Blackhole-compatible persistent-buffer minimal all-reduce for
width-sharded L1 tensors.  It will be validated in isolation before layer
integration.

## 2026-07-23: collective and topology families

Implemented direct width-sharded L1
`ttnn.experimental.all_reduce_async` for both decode collectives. Each
collective owns a persistent width-sharded output buffer and global
semaphore. The real-weight check passed before making it the default.
With the incoming precision/geometry policy, layer-12 S=17 decode moved
from 0.6317 ms to 0.6274 ms.

The following coherent movement/fusion controls were then adapted and run
on the target mesh:

- Carried EP residual: pad H2880 to H2944, persistent reduce-scatter to
  local H736, distributed RMSNorm, row-sharded router plus all-reduce, and
  persistent fused all-gather into the real packed QKV. This did not
  restore the old replicated residual before measurement. At 500 trace
  replays it measured 0.611822 versus 0.477278 ms (layer 12) and 0.608622
  versus 0.474047 ms (layer 13). All compared tensors had PCC at least
  0.999898 and top-4 routing was exact. Rejected.
- Fused attended all-gather plus rank-local O: retried with the selected
  BFP8 payload and a persistent gather buffer. Output PCC was about
  0.999924, but the measured boundary was 0.193203/0.193707 ms versus
  0.066065/0.066069 ms for local O plus minimal all-reduce. Rejected.
- Fused matmul plus reduce-scatter: the exact Blackhole source gate and
  issue 46181 race were captured by a static test. The carried and fused
  AG+O controls above are the adapted hardware retries for the same
  movement/fusion objective.
- Prefill async all-reduce: implemented temporarily with persistent
  semaphores/buffers and verified on both layer kinds. S=17 prefill was
  5.118337/5.131175 ms versus 5.115365/5.131078 ms; S=128 was
  21.823940/25.302397 ms versus 21.817186/25.305283 ms. The mixed
  noise-level result did not justify persistent prefill state, so the
  candidate implementation was removed.

The fused/packed projection check implemented one sparse gate-up matmul
followed by on-device bias, typecast, and stride-2 slices. It passed
layer-12 correctness but decoded at 0.853319 ms versus about 0.591 ms.
The packed QKV projection remains selected; gate and up remain separate.

## 2026-07-23: sharding, geometry, and datatype families

Ran the compiler sharding advisor against representative dense QKV/O
shapes. The QKV recommendation passed layer-12 PCC but decoded at roughly
0.651/0.653 ms for layers 12/13. The O recommendation matched the current
90-core, `in0_block_w=8` geometry. Artifacts are in `shard_advise/`.

Adapted rank-local QKV `[2880,1280]` and O `[1024,2880]` weights for
DRAM-sharded decode matmuls rather than rejecting the first incompatible
shape. Layer-12 PCC passed. Decode measured 0.662247/0.593453 ms for
layers 12/13 versus 0.591067/0.522615 ms selected, so it remains
off-default.

Sparse geometry results for layer 12:

| Gate/up / down geometry | Traced decode (ms) |
| --- | ---: |
| gate 90 cores, K90; prior down | 0.600778 |
| gate 30 cores, K90; prior down | 0.632532 |
| gate 90 cores, K45; prior down | 0.597747 |
| gate 90 cores, K45; down 90 cores, K90 | 0.591404 |

Selected the last row. Host inspection of the real router produced
rank-, token-, and input-dependent local active counts, including zero.
Fixed `nnz` is therefore invalid; runtime mask-derived active execution
stays selected. Exact profiler-input counts were captured later.

Precision/fidelity sweep:

- BFP4 attention plus LoFi failed real prefill PCC at 0.883036/0.793809.
- BFP8 attention with LoFi and HiFi2 passed; HiFi2 was selected for short
  decode.
- BFP4 whole-expert, gate/up-only, and down-only policies failed real
  full-layer PCC.
- BFP8 expert activations failed the full layer.
- HiFi2 expert compute passed but was slower than LoFi.
- BFP8 CCL passed but regressed layer-12 decode to 0.622805 ms because of
  conversions; BF16 remained selected.

The resulting policy is BFP8 short-decode attention weights, BF16
prefill/long-decode attention weights, BFP8 expert weights, BF16 expert
activations and CCL payloads, HiFi2 short attention, selected long
HiFi2/HiFi4, and LoFi experts.

## 2026-07-23: profiler and final-path correction

The first profiler attempt used two prefill iterations and twenty trace
replays and overflowed the device profiler buffer. It was discarded. The
final compact capture used one warmed S=17 prefill and one traced replay,
with watcher disabled:

- layer 12 raw CSV:
  `generated/profiler/reports/2026_07_23_08_43_51/ops_perf_results_2026_07_23_08_43_51.csv`,
  SHA-256
  `b0c57cdcf9c62c4213623e7ab35b5ddde2033dbc2b00474a1f2e75395ea06386`
- layer 13 raw CSV:
  `generated/profiler/reports/2026_07_23_08_44_51/ops_perf_results_2026_07_23_08_44_51.csv`,
  SHA-256
  `0005bb316fe8953070beaee4c48e6c56daf8900a370bfd1e0571903eee93da42`

`tt-perf-report` compact tables under `tracy/` show decode minimal
all-reduce at 205.96/138.38 us and active sparse matmuls at
246.37/176.39 us for layers 12/13. The incoming frozen profile's layer-12
generic all-reduces were about 298.49 us per replay, so the selected
collective reduces that component by about 31%.

An initial all-length run exposed one near-tied sliding routing change at
position 130 when the newly selected BFP8 attention copies were also used
in long decode. This was a real failed gate, not dismissed as noise.
Long decode was changed to use the existing BF16 QKV/O weights at and
above position 127; BFP8 copies remain short-decode-only. The corrected
default then passed:

- S=128/129/2048 real-weight precision for both layer kinds;
- positions 127-131 for both layer kinds;
- reverse-paged advertised context 131072 for both layer kinds; and
- exact mutable trace replay at positions 17-19, 128-131, and 191-193.

The authoritative correctness artifacts are:

```text
logs/final_precision_all_lengths_after_long_bf16_fix.junit.xml
logs/final_boundary_after_long_bf16_fix.junit.xml
logs/final_endpoint_after_long_bf16_fix.junit.xml
```

## 2026-07-23: pre-review default performance and safety gates

Recaptured the final default with 20 warmed prefill iterations and 500
traced decode replays:

| Sequence | Layer | Prefill before -> after (ms) | Decode before -> after (ms) |
| --- | --- | ---: | ---: |
| 17 | 12 sliding | 5.0955 -> 5.119264 | 0.6317 -> 0.591068 |
| 17 | 13 full | 5.1102 -> 5.129240 | 0.6347 -> 0.522615 |
| 128 | 12 sliding | 21.7227 -> 21.817896 | 1.0581 -> 1.039061 |
| 128 | 13 full | 25.2078 -> 25.288609 | 1.0417 -> 1.023655 |

These pre-review JSON filenames contain
`final_default_after_long_bf16_`. They were authoritative for the first
review, but are superseded by the later tile-bounded L1 default and are not
the stage's final reported numbers.

Watcher ran after profiling, in a separate process:

```bash
TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 \
  pytest -q -s \
  models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_real_weight_prefill_decode_matches_current_optimized \
  models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_warmed_trace_replay_mutates_hidden_position_and_local_cache
```

All four cases passed; the watcher log contains no error/assert/hang, and
the final `timeout 60 tt-smi -ls --local` showed all four P300c devices
visible and reset-capable. Static fallback/geometry/source-gate tests also
passed. `doc/context_contract.json` now records the unchanged 131072
context, fixed persistent memory additions, public arbitrary-length
contract, and replicated no-collective inter-layer residual contract.

The final whole-file default run completed in 319.13 seconds with 34
passes and 18 expected opt-in skips. The skipped endpoint, performance,
and topology candidates were all run separately with their enabling
environment variables and have dedicated artifacts.

The first independent `$stage-review` followed this checkpoint and returned
the remediation findings addressed below.

## 2026-07-23: first-review remediation

The first independent review returned `more-work-needed` for five evidence
gaps: an exact decode fused-matmul/collective trial, decode-only BFP4 on the
final topology, a policy-locked sparse sweep, profiler/roofline
reconciliation, and long watcher coverage. Each was addressed on the
serialized 1x4 target mesh.

### Corrected collective and fused-matmul controls

The earlier RS+AG harness accidentally compared two identical configs. It
was corrected to compare the selected minimal all-reduce with
`rs_ag_pad64`. Both layer kinds passed PCC (0.999866/0.999843); the
candidate decoded at 0.627794/0.559360 ms versus
0.591068/0.522615 ms and was rejected.

The exact decode O-projection shape was then run through
`minimal_matmul_strided_reduce_scatter_async`. The first 4x1-grid attempt
failed the primitive's >=2x2-grid requirement. A 4x2-grid retry with the
reduce-scatter workers offset to row 2 passed per-rank PCC >=0.99969. The
off-default whole-decoder implementation passed the accepted attention and
output gates:

| Layer | Attention PCC | Output PCC | Candidate / selected decode (ms) |
| --- | ---: | ---: | ---: |
| 12 sliding | 0.998186 | 0.994854 | 0.726307 / 0.591068 |
| 13 full | 0.997396 | 0.996687 | 0.658184 / 0.522615 |

This direct adapted retry is the material rejection. The upstream
Blackhole issue-46181 gate is documented only for prefill M-tiles=32 and
is no longer used to reject decode M=1.

### Final-topology datatype and sparse policy sweeps

A decode-only BFP4/LoFi policy retained BF16 prefill and long-decode
weights. Mutable trace/cache behavior remained exact, but short attention
PCC was 0.969133/0.968371, below the 0.99 gate. Decode was
0.600810/0.506508 ms; the layer-13 speedup cannot select a policy that
fails both layer kinds.

The policy-locked sparse matrix covered both layers:

- decode compact 45-core/subblock-2:
  0.613600/0.529952 ms;
- decode expert outputs in DRAM:
  0.619588/0.549213 ms;
- decode swapped gate/down K blocks:
  0.592720/0.524193 ms;
- prefill compact 45-core/subblock-2 at S=17:
  5.042155/5.099517 ms, but S=128 regressed to
  22.502769/26.532988 ms;
- prefill K90 at S=17:
  5.176125/5.231182 ms, and S=128:
  22.255763/26.392827 ms.

All non-OOM variants passed the real-weight PCC policy. The selected
90-core gate/up K45 and down K90 decode geometry remained fastest.

Prefill L1 output produced the one additional selected optimization.
S=17 passed both layers and measured 4.133678/4.146684 ms. Direct M=128
failed with a physical 188,743,680-byte allocation request, and an adapted
64-token chunk retried the family but failed at 94,371,840 bytes while only
a 353,472-byte largest free block remained per bank. Because M=32 is the
largest feasible tile, the default now places prefill expert outputs in L1
for internally padded M<=32 and in DRAM for M>=64. This keeps arbitrary
public sequence lengths and the 131072-token context unchanged.
`logs/sparse_policy_sweep.json` records the complete matrix and legality
constraints.

### Final authoritative measurements

The tile-bounded L1 default passed S=17/128/129/2048 real-weight
correctness for sliding and full layers. Final 20-iteration prefill and
500-replay traced decode measurements are:

| Sequence | Layer | Prefill before -> after (ms) | Decode before -> after (ms) |
| --- | --- | ---: | ---: |
| 17 | 12 sliding | 5.0955 -> 4.142142 | 0.6317 -> 0.590842 |
| 17 | 13 full | 5.1102 -> 4.158728 | 0.6347 -> 0.522351 |
| 128 | 12 sliding | 21.7227 -> 21.811075 | 1.0581 -> 1.039083 |
| 128 | 13 full | 25.2078 -> 25.297751 | 1.0417 -> 1.023668 |

At this checkpoint, `final_default_after_l1_policy_*` superseded all
earlier timing JSONs. The later final-review remediation recaptured the
unchanged default after its source refactor; `final_default_after_review2_*`
is the final authority for stage handoff.

### Long watcher, profile, and accounting

Worker/Tensix watcher ran separately from profiling:

```bash
TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 pytest -q \
  test_multichip_decoder.py::test_warmed_long_position_trace_replay_matches_eager \
  test_multichip_decoder.py::test_real_weight_boundary_positions_match_current_optimized
```

All four layer cases passed. This covers positions 127-131 and sequential
trace/cache banks at 128-131 and 191-193. Full ETH instrumentation remains
physically unavailable because the inherited incoming-stage attempt
requires 27,920 bytes on ACTIVE_ETH against a 25,600-byte limit before
model execution; the final CCL-heavy worker watcher and device/fabric
health are the documented compensating controls.

A separate S=128 one-iteration Tracy capture passed both layer kinds. The
canonical raw op CSV SHA-256 is
`ceba2dad739fcfd0b641d777d6cf6ba0dda25cc814972d5eebd5ad4c1051c76e`.
The first postprocessing used global top-4 as `--active-experts 4`. Final
review caught that this is wrong for EP4 rank-local sparse rows; the
corrected accounting is recorded below. Remaining advice corresponds to
candidates already tried: DRAM-sharded decode attention, L1 sparse output,
larger legal K blocks/subblocks, and fidelity reductions.

`logs/performance_accounting.json` records rank-local compulsory decode
bytes and 512-GB/s floors, short and long profile row/gap totals, signpost
walls, and authoritative uninstrumented latencies. It explicitly notes
that four-device merged asynchronous device-row sums overlap and are not a
wall clock. Duplicate 2-GB Tracy
intermediates were removed after retaining the canonical 7.6-MB op CSV,
compact CSV tables, summaries, PNGs, hashes, log, and JUnit.

To ensure the selected M<=32 L1 prefill branch itself has final-code
profile evidence, S=17 was recaptured after all implementation changes.
Both layer kinds passed. Its canonical op CSV SHA-256 is
`32e1a64bbb91d5f54bc569d577e6e15b8d66b8204d11d640bfdc601178952b35`;
four compact tables cover short prefill and decode.
`logs/final_short_profile_provenance.json` records signposts, hashes, and
the uninstrumented reconciliation.

The final review then found that global top-4 had been incorrectly passed
as four active experts on every EP rank. A dedicated exact-input run
captured decode rank counts `[1,3,0,0]`/`[2,2,0,0]` at S=17 and
`[1,1,0,2]`/`[2,1,0,1]` at S=128. Decode tables were regenerated with
the critical rank-local K=3/2/2/2. Prefill tables omit sparse modeling
because local activity varies by token and rank and cannot be represented
by the tool's single integer; exact histograms and unique rank-local
experts are in `profile_ep_activity_layer*_seq*.json`. The corrected
decode floors are 0.1701935 ms for S=17 layer 12 and 0.118556 ms for the
S=17 layer 13 input. The S=128 long-decode BF16 QKV/O copies raise the
common bytes, giving a 0.130706 ms floor for both layer kinds.

The final full default suite completed in 321.32 seconds with 34 passes and
20 expected opt-in skips. The static runtime/fallback and geometry audits
passed within it. A separate advertised-context run passed both layers at
131072 tokens. The post-run `tt-smi -ls --local` listed all four P300c
devices as available and reset-capable.

## 2026-07-23: final-review and AutoFix remediation

The next independent review found two remaining evidence defects. First,
global top-4 activity had been applied as four local experts per EP rank in
the profiler roofline. That was corrected with the exact-input activity run
and rank-local/manual accounting described above. Second, the fused
O-projection plus reduce-scatter timing still restored replication
immediately and therefore did not test the lower-movement family through the
current post-attention MoE.

The repo-local `$autodebug` runner was attempted first as required by
`$autofix`, but its nested Codex process could not start because this host
does not provide the bubblewrap sandbox helper; it did not produce an
`AUTODEBUG.md`. An authorized fresh-context source-review subagent then
performed the inspection-only diagnosis. It confirmed:

- fused O+RS produces a correct padded H2944 rank-local H736 shard;
- distributed RMSNorm and a row-sharded router can consume that shard;
- the selected EP4 whole-expert sparse weights require replicated H2880 at
  the gate/up boundary; and
- preserving EP4 ownership without a hidden redistribution is structurally
  impossible on the same 1x4 axis. A hidden-axis TP expert alternative would
  replace the hidden gather with a much larger packed gate/up reduction.

The focused verify/refute experiment was
`test_fused_o_rs_deferred_through_post_attention_moe`. It starts from the
real BFP8 local O inputs/weights, runs fused matmul+RS, shards and adds the
incoming BF16 residual, carries local H736 through distributed
post-attention RMSNorm and a row-sharded FP32 router, and delays its first
hidden gather until the selected EP4 sparse gate/up contract actually
requires H2880. It then runs the full gate-selected sparse MLP, gathers the
carried residual only for the final layer add, and compares that complete
boundary to the selected replicated family.

The first enabled attempt reached both complete graphs but the comparison
harness reused an alias intentionally consumed by the sparse helper. The
input was cloned and the identical topology was retried rather than
rejected on that API error. The adapted result passed both layer kinds:

| Layer | Min attention-residual PCC | Min post-norm PCC | Final PCC | Selected / deferred candidate (ms) |
| --- | ---: | ---: | ---: | ---: |
| 12 sliding | 0.999851 | 0.999771 | 0.999450 | 0.448808 / 0.659886 |
| 13 full | 0.999874 | 0.999844 | 0.999677 | 0.526294 / 0.737228 |

Top-4 routing agreed exactly. The candidate was rejected only after this
deferred-gather full-layer comparison. Artifacts are
`logs/candidate_fused_o_rs_deferred.{log,junit.xml}` and
`logs/candidate_fused_o_rs_deferred_layer{12,13}.json`.

The review also requested a real hardware fallback-throw check. Both
real-weight S=17 layer cases passed with
`throw_exception_on_fallback=true`; see
`logs/final_throw_on_fallback_hardware.*`. Exact selected minimal-AR
worker, buffer, semaphore, topology, and non-exposed API fields are recorded
in `logs/minimal_all_reduce_config.json`.

After factoring the off-default fused RS primitive into a helper used by
the focused experiment, the final default was recaptured with 20 warmed
prefill iterations and 500 trace replays:

| Sequence | Layer | Prefill before -> final (ms) | Decode before -> final (ms) |
| --- | --- | ---: | ---: |
| 17 | 12 sliding | 5.0955 -> 4.140304 | 0.6317 -> 0.590920 |
| 17 | 13 full | 5.1102 -> 4.153870 | 0.6347 -> 0.522363 |
| 128 | 12 sliding | 21.7227 -> 21.812672 | 1.0581 -> 1.039031 |
| 128 | 13 full | 25.2078 -> 25.292354 | 1.0417 -> 1.023719 |

Only `final_default_after_review2_*` JSON files are authoritative final
numbers.

The exact final source then passed:

- the complete default file: 34 passed and 26 expected opt-in skips in
  323.93 seconds (`final_default_full_suite_after_review2.*`);
- worker/Tensix watcher at positions 127-131 and trace/cache banks
  128-131/191-193: 4 passed, with ETH disabled only for the inherited
  27,920-byte versus 25,600-byte physical instrumentation limit
  (`final_watcher_after_review2.*`);
- both advertised-context endpoints at 131072 tokens
  (`final_endpoint_after_review2.*`); and
- post-gate health: all four Blackhole P300c devices available and
  reset-capable (`final_hardware_health_after_review2.log`).

## 2026-07-23: independent final review

The fresh `$stage-review` returned `clean-pass` with no required work,
concerns, or hard-check gaps. It independently verified the corrected
rank-local EP accounting and hashes, the deferred-gather fused O+RS
full-layer trial, final `after_review2` measurements, all correctness and
safety gates, gate-selected EP4 execution, and the no-collective
inter-layer contract. Its retained report is `stage_review.md`.

During review, obsolete `*_active4_summary.{csv,png}` derivatives were
found in the final profile directories. They were removed; no
`*active4*` artifact remains, and all corrected `epmax`/`ep_variable`
tables plus raw CSV hashes still match provenance.

The local checkpoint SHA is recorded below after commit creation. No push
is performed.
