# GPT-OSS true-prefill DFlash handoff

This path replaces target prefill-by-decode for DFlash feature generation. It
is opt-in and does not change ordinary GPT-OSS prefill.

## Producer contract

The target taps the MoE branch output immediately after `self.mlp(...)` and
before the residual add at layers `1, 9, 17, 25, 33`. For target hidden width
`H=2880`, drafter `fc.weight` is an `nn.Linear` weight with shape
`[H, 5*H]`. Prefill accumulates:

```text
reduced_hidden = Σ moe_output_i @ fc.weight[:, i*H:(i+1)*H].T
```

The result is pre-norm. The producer must not apply the drafter's
`hidden_norm`; the reload/DFlash KV-injection path owns that operation.

`DFlashPrefillResult` publishes:

- `slot_id` and the real absolute range `[actual_start, actual_end)`;
- `chunk_size` so a consumer can reject padded rows;
- device-resident `reduced_hidden`, sequence-sharded block-cyclically over SP
  and feature-width-sharded over TP;
- host fp32 logits and greedy `y0` for the final real prompt token;
- enqueue timing hooks for each target FC and the final head.

The result owns its device tensor after return. A `dflash_sink` takes that
ownership when supplied. Padded storage rows may exist in the tensor, but only
the declared real range may be migrated or used to prime the drafter.

## Enablement

The common adapter resolves deployment configuration:

```bash
PREFILL_DFLASH=1
TT_HF_DRAFT_MODEL=/path/to/gpt-oss-120b-DFlash
```

The path is passed as `TtPrefillRuntimeConfig.dflash_checkpoint_path`; model
code does not read environment variables. A caller requests the handoff on the
final prompt chunk with `dflash_handoff=True` or `dflash_sink=...`. Default
`prefill_chunk` return behavior is unchanged.

The current producer requires one runtime to own all 36 target layers. Passing
partial FC sums between pipeline ranks is intentionally left to the P/D owner;
the runtime fails instead of publishing an incomplete feature.

## Proof rungs

CPU contract:

```bash
pytest models/demos/gpt_oss_d_p/tests/unit/test_dflash_prefill.py
```

Synthetic SP×TP device accumulation:

```bash
pytest models/demos/gpt_oss_d_p/tests/unit/test_dflash_device.py
```

Real 36-layer 1k Galaxy proof:

```bash
PREFILL_TRACE_DIR=/path/to/matching/kv_golden \
BLAZE_DFLASH_SEED_REF=/path/to/seed.pt \
TT_HF_DRAFT_MODEL=/path/to/gpt-oss-120b-DFlash \
HF_MODEL=/path/to/gpt-oss-120b \
python models/demos/gpt_oss_d_p/tests/galaxy_prefill_dflash.py
```

The Galaxy rung preserves the existing target-KV PCC gate, checks aggregate
and per-position pre-norm feature PCC, checks final `y0`/top-k, reports
disabled/enabled latency and export cost, and writes a consumer fixture.
Use a 1k-position seed to gate the complete feature trajectory and final token.
`PREFILL_TPS_ITERS=N PREFILL_TSU_MIN=3000` makes the synchronized enabled-path
median an executable throughput gate rather than a one-run characterization.
`GPT_OSS_KV_PCC_MIN` and `GPT_OSS_DFLASH_PCC_MIN` set the corresponding
correctness floors; all failures are reported together.

The reload owner feeds that fixture through bulk drafter-KV priming. Set
`PREFILL_DFLASH_COMPAT_CONTROL` to the current prefill-by-decode result and
`PREFILL_DFLASH_COMPAT_RESULT` to the reload result to gate prompt-KV PCC, the
first proposal block, and a short accepted continuation.

The consumer fixture has `format_version: 1` and records
`feature_contract: fc_only_prenorm`. Its independent correctness source is the
CPU/HF teacher-forced seed; that seed is not generated from prefill-by-decode.
The latter remains a separate compatibility control so a shared target-path
error cannot validate itself.

## Kimi-equivalent CI gates

The pre-P/D ladder now mirrors Kimi's test semantics:

1. `gpt_oss_dflash_prefill` in the tt-metal Blaze prefill matrix runs the real
   36-layer producer on SC1, gates target KV and independent-HF features, emits
   the versioned handoff, and enforces the 3,000 token/s 1k-prefill floor.
2. The outer Blaze `gpt-oss / dflash-prefill` Galaxy leg consumes a staged
   true-prefill trace. It feeds identical fc-only features to HF and TT DFlash,
   reconstructs the block-offset/circular device caches, and requires K and V
   PCC `>=0.999` for all eight layers.
3. That consumer leg also compares the first proposal, accepted prefix, and
   short speculative sequence with a separately staged torch DFlash run over
   the same true-prefill features. Proposal tokens must be in the corresponding
   torch top-32 candidate set (the BF8-tolerant argmax criterion); the target
   continuation and accepted prefix remain exact. The prefill-by-decode result
   remains the additional producer-compatibility control described above.

The SC1 model-store paths used by CI are:

```text
/mnt/models/blaze/openai/gpt-oss-120b/cache/dflash/seed_1016.pt
/mnt/models/blaze/openai/gpt-oss-120b/cache/dflash/true_prefill_1k.trace.pt
/mnt/models/blaze/openai/gpt-oss-120b/cache/dflash/true_prefill_torch_control_1k.pt
```

For a local consumer run on one Galaxy/loudbox submesh:

```bash
GPT_OSS_DFLASH_PREFILL_TRACE=/path/to/handoff.trace.pt \
GPT_OSS_DFLASH_PREFILL_CONTROL=/path/to/torch-result.pt \
TT_DFLASH_DRAFT_CKPT=/path/to/gpt-oss-120b-DFlash \
TT_DFLASH_TARGET_CKPT=/path/to/gpt-oss-120b \
TT_METAL_SLOW_DISPATCH_MODE=1 \
pytest tests/blaze/dflash/temporal/test_gptoss_dflash_prefill_kv.py -sv
```

CPU-only cache-addressing and malformed-capture checks live in
`tests/blaze/dflash/temporal/test_kv_cache_export.py`.

## Downstream dependencies

This change does not implement generic P/D transport or reload consumption.
Those stacks must:

1. migrate target KV with the FC-target-aware migration layout;
2. transport the typed feature/y0 handoff without treating padded rows as real;
3. apply `hidden_norm`, bulk-prime drafter KV, and begin at position `S` from
   `y0` without replaying the prompt through the target.

The final system gate remains a 1k prompt from one prefill Galaxy into the 4×4
quad reload decoder, matching the non-disaggregated first proposal and fixed
continuation, followed by the 3,000 TSU target and AIME evaluation. Those are
consumer/system gates, not claims made by the producer tests.

Kimi's SC4 disaggregated runner is intentionally not copied yet: target-KV and
feature transport, slot/address tables across P/D, and the 4×4 reload consumer
belong to the pending P/D branch. The SC1 producer and DFlash K/V/proposal gates
above do not depend on that work.
