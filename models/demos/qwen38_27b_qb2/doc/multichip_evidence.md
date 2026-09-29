# T3K TP=8 multichip evidence

Target: Wormhole B0 T3K (4x n300, 8 chips), `ClusterType.T3K`, `(1, 8)` mesh, TP=8.
Baseline: the measured Blackhole P300_X2 TP=4 implementation in this same tree, which is the
only prior art for these kernels; there is no single-chip TTNN baseline for this model because
the decoder's policy assumes a mesh.

## Strategy

1D tensor parallelism on the single mesh axis, which is what `tech_reports/LLMs/llms.md` §3.3
recommends for a 1D mesh of eight chips. Weights shard on the projection axis, the residual
stream stays replicated, and collectives run as Ring over `FABRIC_1D_RING`.

Ring is not a free choice. `all_reduce_async` at the model's shapes completes in 0.06 s on
`FABRIC_1D_RING` and never completes on `FABRIC_1D`, because a ring collective needs the hop
from the last device back to the first. The same op as a Linear collective runs fine on
`FABRIC_1D`. Only the mismatch deadlocks, so `validate_fabric_topology` rejects it at
construction rather than leaving it to a device timeout minutes into serving.

`num_links` is 1. Two links on Ring hang and wedge the devices; the second ethernet link of
each chip pair carries the dispatch datapath.

## Per-device geometry at TP=8

Every head count below is already per-device. Seven QB2 constants had to become
platform-derived; each was a real hardware difference, not a bug.

| quantity | QB2 TP=4 | T3K TP=8 | why it moves |
| --- | --- | --- | --- |
| fabric packet payload | 8192 B | 7616 B | Wormhole ceiling is 7 Bfp8_b tiles, Blackhole's is 14 |
| head / embedding shard | `vocab/4`, `hidden/4` | `/TP` | slice ran past the per-device tensor |
| sharded layernorm grid | `(10, 4)` | `(8, 5)` | `CoreCoord(9, ...)` does not exist on 8x8 |
| DRAM readers per bank | 2-3 | 1 | `num_workers_per_dram_bank > 1` is Blackhole-only |
| `down_cores` | 8 | 34 | `down_proj` K is `intermediate/TP`, 68 tiles; 8 does not divide it |
| LM head chunk width | 16384 | multiple of `banks*64` | an over-allocated bank shard returns wrong values silently |
| `prefill_1d_down_k` | 8 (GDN) / 17 | 17 | same 68 tiles, different matmul path |
| KV heads | 1 per device | 1 per device, replication 2 | 4 heads cannot shard 8 ways |
| `recurrent` state | `(B, 12, 128, 128)` | `(B, 6, 128, 128)` | `linear_num_value_heads/TP` |
| `conv` state | `(B, 3, 2560)` | `(B, 3, 1280)` | `2*k*128 + v*128` per device |

`down_cores` and `prefill_1d_down_k` are the same underlying fact reached by two paths, so
`tests/unit/test_policy_blocking.py` now checks every K block size and DRAM core count against
every role's K on both widths rather than relying on inspection.

## Context contract

`doc/context_contract.json`. The full advertised 262144-token window is kept: a single
full-context request costs 2.14 GiB of the 7.192 GiB left per device after weights and the
trace reserve. Only 16 of the 64 layers hold a paged KV cache; the other 48 hold 18.4 MiB of
recurrent state that does not grow with context.

## Correctness

Against an fp32 HuggingFace reference, with the bfloat4_b weight-quantization floor as the
control, because the precision policy costs far more than any absolute PCC threshold worth
writing down:

| check | result |
| --- | --- |
| per-layer prefill, `linear_attention` | 0.999784 against a 0.999325 floor |
| per-layer prefill, `full_attention` | 0.978978 against a 0.980890 floor |
| end-to-end logits, 7 steps | matches or beats the same floor at 3 of 7 |
| greedy agreement, teacher-forced | 7/7 |
| DRAM head vs interleaved head | `logit_pcc` 0.9999963, identical greedy tokens |
| replica agreement across 8 devices | bit-identical (`dev_spread` 0) |

Full-vocabulary PCC reads 0.82-0.98 and top-k PCC is no better, so the divergence is not
confined to an irrelevant tail. It is the precision policy: an fp32 reference carrying nothing
but bfloat4_b weights scores the same band. Softmax KL is useless here, reading 0.00000 beside
a top-1000 PCC of 0.822, because a wide top-1 gap makes the distribution a near-delta on both
sides.

`tests/test_layer_pcc.py` gates per-layer PCC against the floor rather than a fixed number, so
it self-calibrates if the precision policy changes. Its `FLOOR_MARGIN` of 0.01 is a hand-set
slack, not a derived bound: `full_attention` measures 0.978978 against a 0.980890 floor, so it
passes on that margin rather than on merit.

The head is not where the logit error lives. The shipping `config/precision.json` is
`head_bfp4_lofi`, putting the LM head at `bfloat4_b` and LoFi. Running the same prompt and the
same teacher-forced token sequence against the same fp32 reference under a policy that differs
only in the head group, `bfloat8_b` and HiFi2, moves mean top-1000 PCC from 0.8963 to 0.9024:

| step | bfp4/LoFi head | bfp8/HiFi2 head | delta |
| --- | ---: | ---: | ---: |
| prefill | 0.8461 | 0.8544 | +0.0083 |
| decode1 | 0.8220 | 0.8263 | +0.0043 |
| decode2 | 0.8846 | 0.8894 | +0.0047 |
| decode3 | 0.9078 | 0.9121 | +0.0042 |
| decode4 | 0.9639 | 0.9691 | +0.0052 |
| decode5 | 0.8974 | 0.9054 | +0.0080 |
| decode6 | 0.9519 | 0.9602 | +0.0083 |

Top-10 overlap against the reference is identical at every step under both policies, and greedy
agreement is 7/7 under both. So the head accounts for roughly 0.6 of the ten-point gap from
1.0; the remainder is the `bfloat4_b` projections compounding over 64 layers. That also earns
the shipping head choice: the cheaper head costs 0.006 PCC and changes no ranking.

## Behaviour

Full 64 layers: correct Rayleigh-scattering answer, `17 * 24 = 408` with the right stop token,
identical output across replays. Paged KV cache and warmed traced decode replay both exercised;
400 decode steps with 16 scheduler-style slot remaps held a flat 6.4 s per 50 steps with no
drift, and 40 cycles of interleaved prefill and decode were equally stable.

## Instrumented run

Waypoint-and-assert clean over a full prefill and traced decode: 21542 lines, zero trip
markers, clean detach, and `retraining events: 0` on all 40 ethernet rows.

This is **not** an unqualified watcher-clean run. A bare `TT_METAL_WATCHER=10` cannot link on
Wormhole: the instrumented fabric erisc router puts `.text` at `0xee08`, past the `0xEBE0` end
of `ERISC_APP_KERNEL_CODE`. The run above therefore has `NOC_SANITIZE`, `SANITIZE_NOC` and
`ETH` disabled, which is exactly the coverage that would catch an out-of-bounds transaction, so
that fault class remains unchecked. A failed watcher build also leaves the devices needing
`tt-smi -r`; a subsequent run reports an unexpected `run_mailbox` value until it is reset.

## Gaps

- No PCC against a single-chip TTNN baseline, which is what would separate sharding and
  collective error from HuggingFace-versus-TTNN numerics. The measurements above bundle both.
- No `tt-perf-report` for this path. The KDA blocker is real but narrow:
  `kda_performance_model.cpp` asserts Blackhole and `qkv_causal_conv1d_silu` and
  `sigmoid_gated_rms_norm` reach it from `create_op_performance_model`, so only the 48
  `linear_attention` layers cannot be profiled. The 16 `full_attention` layers, the LM head,
  the norms and every collective can be, and this tree already holds wormhole_b0 profiler
  results for the MLP and CCL sweeps. The missing report is a scoped exclusion, not an
  impossibility.
- No runtime fallback audit.
- Non-aligned sequence lengths are only covered where a bug forced it. `prefill_1d` selects on
  a 64..256 window and `chunk_size` is 4096, so the boundaries either side of both deserve
  explicit cases.
- 262144 tokens is a capacity result, not a latency or quality result; no run at that length
  has been executed on this mesh.
- No single-chip-versus-multichip speedup, which needs the baseline in the item above.
- The qualitative suite is in `readiness_qualitative/`, covering prompt format and answer
  quality on the shared six prompts at 256 tokens against a native-bfloat16 control. It is not
  an accuracy gate: no top-1/top-5/top-100 and no AIME24 reference exist yet.
- The residual and fused-CCL families are still unmeasured here. `fused_grid` and
  `rs_core_offset` hold Blackhole values that no 8x8 worker grid can satisfy, so `mmrs`,
  `agmm` and the sharded-residual path cannot run without T3K-legal values first. A shape
  error is not a rejection, so this family remains open.

## Batch-1 performance

Warmed, prompt 128 and generate 128, the shape the serving profile uses, at the shipping
`ccl_dtype` of bfloat16:

| metric | value |
| --- | ---: |
| TTFT | 247.0 ms |
| decode, token out | 72.321 ms, 13.83 t/s/u |
| decode, no readback | 71.522 ms, 13.98 t/s/u |

Both decode boundaries are reported because they answer different questions: the no-readback
figure is the logits-side comparison, and token out adds the final norm, LM head, sampling and
the caller-visible readback. The boundary between them costs 0.799 ms, 1.10% of the step, so
sampler work is not the dominant token-out cost.

Steady-state host work is one refresh each of token, position and RoPE per 148 trace replays,
not one per generated token, so decode state advances on device rather than from the host.

### Layer-stack lower bound

Decode latency is linear in layer count. Least squares over four depths, each measured the same
way, gives 0.9676 ms per layer with a 9.600 ms intercept and residuals within 0.010 ms:

| layers | no readback | token out |
| ---: | ---: | ---: |
| 4 | 13.468 ms | 14.383 ms |
| 16 | 25.078 ms | 25.966 ms |
| 32 | 40.573 ms | 41.350 ms |
| 64 | 71.522 ms | 72.321 ms |

The 64-layer stack is therefore 61.93 ms and the full-model-only cost is 10.40 ms, 14.4% of the
step. Only 0.799 ms of that is the token-out boundary, so the remainder is the embedding
all-gather, the final norm, the LM head and logits movement, all of which sit inside both decode
measurements. The LM head is the largest single candidate there.

This is a marginal per-layer cost measured inside the real traced loop, not a standalone
optimized per-layer latency, because no such per-layer measurement exists for this mesh.

### Against the Blackhole reference

QB2 publishes 39.4 t/s/u and 67.9 ms TTFT at the same shape, so this mesh reaches 35% of its
decode throughput and 3.6 times its TTFT. With the bfloat8_b collective measured below it would
be near 16.8 t/s/u, still 43%. That is consistent with the platform deltas recorded above:
64 worker cores against roughly 110, one usable ethernet link against two, and one DRAM reader
per bank because multiple readers are Blackhole-only. It is a hardware gap rather than a defect,
but it is larger than the phrase "slower on Wormhole" would suggest.

## Collective dtype

`num_links` is 1 here against QB2's 2, so the collective carries twice the hops on half the
links, and the dtype it moves was carried over from the two-link measurement. Measured warmed,
single variable through `precision_config`, 20 warm-up steps then 100 timed steps at ISL 128:

| batch | ccl bfloat16 | ccl bfloat8_b | saving |
| ---: | --- | --- | --- |
| 1 | 71.522 ms, 13.98 t/s/u | 59.543 ms, 16.79 t/s/u | 11.98 ms, 16.8% |
| 8 | 101.411 ms, 9.86 t/s/u | 89.462 ms, 11.18 t/s/u | 11.95 ms, 11.8% |

Every arm held a spread under 0.2% between p10 and p90. The saving is the same 12 ms at both
batches, not a larger fraction at higher concurrency: the all-reduce workspace is
`[1, 1, 32, 5120 * TP]`, tile-padded to 32 rows whatever the batch, so its payload is fixed per
decode step and halving the dtype halves a constant cost. That puts the bfloat16 collective at
roughly 24 ms of every step, about 0.37 ms per layer.

It costs nothing measurable. Against the same fp32 reference and the same teacher-forced
tokens, mean top-1000 logit PCC moves 0.8963 to 0.8989, two of seven steps negative and five
positive, with identical top-10 overlap at every step and 7/7 greedy agreement. Free-running
greedy output is byte-identical across all four arms above.

The value is not reachable from the platform overlay as the merge stands. `_TP_POLICY` is
applied before the caller policy, and `decoder_policy` always supplies `ccl_dtype` from the
precision policy, so a platform entry for it is overwritten. Moving this to bfloat8_b therefore
means either a T3K precision artifact selected at deployment, or giving the platform overlay
ownership of the keys precision also sets. Both reach QB2's shipping path, where this is
unmeasured, so the choice is deliberately left open rather than taken here.
