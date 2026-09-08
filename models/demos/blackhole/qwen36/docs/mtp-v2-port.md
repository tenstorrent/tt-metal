<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
SPDX-License-Identifier: Apache-2.0
-->

# Porting `atupe/qwen36-mtp-v2` MTP spec decode onto `ign/qwen359b`

**Status: landed on `ign/mtp_qwen3.6` and validated on T3K silicon.** The port was developed on a
separate branch in its own worktree; it is now integrated into the in-tree model, where MTP spec
decode is the DEFAULT single-user TP decode path (`QWEN36_SPEC=0` opts out). Re-measured after
integration at ISL 128, both legs on one build: **16.93 -> 38.82 tok/s (2.29x)**. See
`README-T3K-27B.md` for the user-facing numbers and run commands.

The `ttnn` side adds a C++ op and changes `sdpa_decode`, so `./build_metal.sh` (or
`cmake --build build_Release --target install`) is required.

## What this replaces

`ign/qwen359b` already carried an MTP spec decode (`Qwen36MTPHead` /
`Qwen36SpeculativeDecoder`) whose verify was a **96-row masked chunk** replayed from
a 64-block-aligned GDN anchor. Two structural ceilings capped it:

* the verify pushed the whole bucket through all 64 layers while `pending` only
  cycled 1..64, so most rows were padding; and
* the drafter ran fully replicated, so every device read the whole drafter head
  per leg — a leg's weight traffic did not shrink with the mesh.

The v2 design removes both: verify is a **K+1-row forward** with per-token GDN state
buffered by the recurrence kernel, and commit just *points* the durable state at the
accepted slot — no rollback, no re-processing of committed tokens. v2's commit log
records **45 -> 61 tok/s at ISL 128, K=10** and a later **~73 vs ~28 plain** on ITS
config (Blackhole P150x4). Those numbers are not this branch's.

The old design's files are superseded and must be retired when this lands:
`tt/mtp.py`, `tt/spec_decode.py` (same paths, different API), plus
`tests/test_spec_decode.py`, `tests/test_spec_decode_host.py`,
`tests/unit/test_mtp.py`, `tests/test_mtp_weights_host.py`,
`reference/mtp_torch.py`, `docs/mtp.md`.

## How it was applied

All 13 v2 commits (`e4d4e6d1853..atupe/qwen36-mtp-v2`) cherry-picked in order. The
delta is overwhelmingly additive — ~7.6k insertions, ~140 deletions — so `model.py`
(+945), `layer.py`, `mtp.py`, `spec_decode.py` and the entire new ttnn C++ op merged
with no conflict at all. Conflicts appeared only where this branch has its own
Wormhole rewrites, and in every case this branch's tuned path was kept and v2's
addition grafted in beside it:

| File | Resolution |
| --- | --- |
| `tt/attention/tp.py` | Kept `_col_proj(prefill_progcfg_fn=)`, the asymmetric K-chunk, `_kv_fused_shard_cfgs`, the N300 KV-pad fork and its late `dealloc`. Added `_kv_update_shard_cfg`, the aliased per-row KV write as a new FIRST arm of the write branch, the SDPA batch assert, `decode_sdpa_max_cores`, and the fused spec-SDPA plan. |
| `tt/gdn/tp.py` | Kept the Wormhole conv/recurrent dispatch imports and the whole Wormhole-forked `_conv1d_prefill`. Added `fused_recurrent_gated_delta_rule_ttnn`, `use_fused_recurrent_decode` as the first arm of the decode recurrence, and the `T < K-1` shift-register tail. |
| `tt/model.py` | Kept `_rope_from_idx` / `_argmax_device` and the tuned vocab-AG knobs. Added `_lm_head(out_dtype=)` and the CCL dtype passthrough. |
| `demo/text_demo.py` | Took v2's spec paragraph but dropped its `QWEN_GDN_PHASED` / `QWEN_GDN_FLAT_QKV` lines — this branch retired both (the helpers in `gdn/fused_chunk.py` hardcode `True` and read no env), so importing them verbatim would document dead flags. |

Three changes were needed beyond mechanical conflict resolution.

**1. `tuned_vocab_all_gather` hardcoded a bf16 cast.** v2 feeds the drafter's LM head
`out_dtype=float32` so its argmax does not throw candidates away to bf16 ties, and
routes the gather at that dtype — but this branch's local `tuned_vocab_all_gather`
(a copy of `ccl.tt_all_gather`'s `cluster_axis=None` branch, which upstream
parameterises) cast unconditionally to bf16. v2's own comment says that cast both
reinstates the ties *and* returns garbage on this tensor (an out-of-range drafter
token id). Added the `dtype` parameter, defaulting to bf16 so every base/verify
caller is byte-identical.

**2. `_conv1d_prefill`'s conv was a closure.** v2 factors the conv call out so the
verify can reuse it over the window it already holds (saving a duplicate concat and a
dead `new_state` slice — ~7 ms/iteration over 48 GDN layers). This branch's
`_conv1d_prefill` is a different, Wormhole-forked function. Rather than take v2's
version and lose that work, the conv call was promoted to `_conv1d_raw` and
`_conv1d_window` / `_conv1d_verify` added beside it. The verify's geometry is the
concatenated form (`clen = K-1+T`, prep padding `(0,0)`), so it shares the existing
geometry-keyed weight-prep cache entry and the tuned prefill path is unchanged.

**3. The hybrid verify's decode-RoPE table ignored permuted RoPE.** v2's new
`_rope_tp_cos_sin_decode_torch` builds its per-row cos/sin at `args.rope_head_dim` with a
plain rotate-half, which is right on v2's own configs — permuted full-width RoPE is off
there. On this branch it is ON for `wh_9b_n300`, where the channel permutation is folded
into the target's q/k weights at load time and every other rope helper hands out
`rope.rope_width`-wide permuted rows. The verify would have rotated the wrong channels:
wrong output, no error. It now widens through the same `to_full_width_rot_mats` off the
same `rope.inv_freq` the prefill helper uses, so the two agree by construction. Verified
byte-identical on a non-permuted config (27B/T3K, this branch's target), where
`full_head_dim` is `None` and the widener short-circuits.

**4. `_SPEC_SDPA_L1_FIT` is arch-gated (`tpc.is_blackhole()`).** Its
`(groups, cores-per-head, k-chunk)` triples were fitted to Blackhole's 110-core grid
— `55` cores/head per group at T=8, `36` at T=12. Wormhole tops out at 8x8 and fewer
after harvesting, so those numbers do not describe the grid: the op would either
TT_FATAL on the L1 budget or silently run an unmeasured split. On Wormhole the plan is
`None`, which falls through to the legacy `B=T` SDPA call that the hybrid-verify commit
(`092f0c15da4`) shipped and measured at 61 tok/s. **Re-tune the table on a WH grid
before removing this gate** — that is the single biggest piece of v2 perf work this
port does not yet get on Wormhole.

## MEASURED on T3K / Qwen3.6-27B

Same build, same prompt, warm kernels, back to back at ISL 128 (`./mtp_bench.sh 128`):

| | plain (`QWEN36_SPEC=0`) | MTP spec | |
| --- | --- | --- | --- |
| decode | 16.96 tok/s | **36.81 tok/s** | **2.17x** |
| TTFT | 0.56 s | 5.20 s | drafter-KV warm + trace captures |

Per iteration (K=6, 5.00 committed tokens/iter), and how it got there:

| phase | as ported | traced | |
| --- | --- | --- | --- |
| draft | 127.4 ms | **33.3 ms** | 6 eager legs x ~19 ms of HOST enqueue each; the device drained all 6 in 2.8 ms |
| reseed | 34.5 ms | **5.8 ms** | one B=K+1 forward, ~35 ms host vs 0.8 ms device (`alias_kv_write` = a paged_update_cache pair PER ROW) |
| verify | 89.9 ms | 89.8 ms | untouched -- now 67% of the iteration, and the part that grows with context |
| commit + readback + accept + other | ~5 ms | ~5.7 ms | |
| **total** | **257 ms** | **134.5 ms** | |

The whole gap to v2's Blackhole numbers was HOST DISPATCH, not silicon. Both hot phases were
eager loops whose cost is invisible on Blackhole and dominant on Wormhole. Neither of the levers
this doc originally listed (sharding the drafter, the fused verify SDPA) was the problem: the
drafter is already TP-sharded, and the fused SDPA is an ISL-slope fix worth ~0.02 ms at ISL 128.

Everything above is ISL 128 only. Verify is now the dominant phase and it is the ISL-sensitive
one, so longer prompts need their own measurement -- and that is where the still-gated
`_SPEC_SDPA_L1_FIT` re-tune would pay.

### Correctness gates, on Wormhole T3K / 27B### Correctness gates, on Wormhole T3K / 27B

Both were `@run_for_blackhole()` in v2 -- incidentally, because that was the only silicon it
ran on. The tests are arch-neutral (run spec, run plain greedy, compare token ids), so they
are widened to `run_for_wormhole_b0_or_blackhole()`.

| test | result |
| --- | --- |
| `test_spec_lossless.py` | **PASS** -- lossless to token 11, then a legitimate near-tie flip (ref top-2 gap 0.375 < 2.0; 6/48 positions were near-ties, min gap 0.125). K=3: accept 2.69/3, per-depth 0.92/0.92/0.85 |
| `test_spec_determinism.py` | **PASS** x2 (prompt_len 128 and 130) -- 3 runs identical, accept 2.692 |

Losslessness is the property that makes the speedup meaningful: committed tokens come from
the target's verify rows, so spec must reproduce the plain greedy trajectory rather than
merely a plausible one. A high accept rate alone would not prove it.

### Remaining perf levers, largest first

1. **Verify, 89.8 ms (67%).** Untouched. At ISL 128 it is the 64-layer forward at T=K+1 rows, not
   KV scanning, so the Blackhole-gated `_SPEC_SDPA_L1_FIT` buys ~nothing here -- but it is the
   phase that grows with context, so re-tuning that table for an 8x8 grid is the long-ISL lever.
2. **Draft, 33.3 ms.** Now 6 trace replays + one window staging call. What is left is the staging
   (`rot_mats_decode` host trig + uploads, ~1.8 ms) and K separate id readbacks; batching those
   into one readback and hoisting the RoPE would take another few ms.
3. **TTFT, 5.20 s vs 0.56 s.** Drafter-KV prompt warming plus the verify/commit/draft captures.
   Worth attention before enabling spec by default for short requests.

## Verified so far (host only, no device)

* Every module imports; every symbol `spec_decode.py` calls on the model, the GDN
  layers, the MTP head and the attention layer exists.
* `tests/test_mtp_torch_ref.py` — 7/7 against the real Qwen3.6-27B `mtp.*` weights.
* `tests/test_weight_mapping.py` — 33/33 (`HF_MODEL=Qwen/Qwen3.5-9B`).
* `pre-commit` clean over all 49 touched files.

## What still needs a device

The ttnn side adds a new C++ op (`transformer/fused_recurrent_gated_delta_rule/`,
12 files) and modifies `sdpa_decode`, so **`./build_metal.sh` is required** before any
of this runs.

```bash
./build_metal.sh
export ARCH_NAME=wormhole_b0 WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen3.6-27B

# 1 chip -- the new C++ op vs FLA naive, at real 27B GDN dims (8 cases)
MESH_DEVICE=N150 pytest models/demos/blackhole/qwen36/tests/test_fused_recurrent_gdn.py -v -s
# 1 chip -- the sdpa_decode spec_multi_pos kernel change (21 cases, op level, not arch-gated)
MESH_DEVICE=N150 pytest tests/ttnn/unit_tests/operations/sdpa/test_sdpa_decode_spec_multi_pos.py -v
# T3K -- drafter head PCC vs the torch reference (2 cases)
pytest models/demos/blackhole/qwen36/tests/test_mtp_tp.py -v -s
# T3K -- THE gate: spec output == plain greedy, token for token
pytest models/demos/blackhole/qwen36/tests/test_spec_lossless.py -v -s
pytest models/demos/blackhole/qwen36/tests/test_spec_determinism.py -v -s

# A/B, in this order — the baseline first, so the comparison is on one build
QWEN36_SPEC=0 pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s -k traced_128
QWEN36_SPEC_TIMING=1 pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s -k traced_128
```

`test_spec_lossless.py` is the gate that matters: committed tokens come from the
target verify, so spec output must equal plain greedy token for token. If it fails,
suspect the three changes above before suspecting the design.
