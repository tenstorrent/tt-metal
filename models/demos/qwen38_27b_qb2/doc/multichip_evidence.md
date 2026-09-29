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

The floor itself does not model the head. The shipping `config/precision.json` is
`head_bfp4_lofi`, putting the LM head at `bfloat4_b` and LoFi, while the floor control
quantized the head to `bfloat8_b` and the only recorded head comparison holds precision
constant between two TT programs. So head-precision error is unbounded by the evidence here,
and the 0.822 top-1000 logit PCC cannot yet be attributed between the projections and the head.
A control against `QWEN_PRECISION_CONFIG=baseline`, which differs only in the head group, is
the missing measurement.

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
- No performance measurement for this mesh: no warmed TTFT, no decode tokens/s/user, no
  single-chip-versus-multichip speedup, and no host-work counter dump from the traced decode
  loop. The one timing figure quoted above is a stability observation, not a benchmark.
- No qualitative suite. This is a chat checkpoint and the only generated evidence comes from a
  raw continuation prompt, so prompt-format coverage is missing entirely.
- Stage 5 optimization families were not measured on this mesh. `num_links` is 1 here against
  QB2's 2, which halves collective bandwidth and makes the collective families the dominant
  question, yet `ccl_dtype`, residual layout and the fused CCL paths are carried over from the
  two-link measurement. `fused_grid` and `rs_core_offset` still hold Blackhole values that no
  Wormhole worker grid can satisfy, so those families cannot run as written.
