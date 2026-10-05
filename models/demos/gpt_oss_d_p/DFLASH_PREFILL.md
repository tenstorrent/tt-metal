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

The reload owner feeds that fixture through bulk drafter-KV priming. Set
`PREFILL_DFLASH_COMPAT_CONTROL` to the current prefill-by-decode result and
`PREFILL_DFLASH_COMPAT_RESULT` to the reload result to gate prompt-KV PCC, the
first proposal block, and a short accepted continuation.

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
