# GPT-OSS true-prefill DFlash handoff

DFlash is the speculative-decoding drafter used with GPT-OSS-120B. A small
drafter model proposes several tokens at once and the full ("target") model
verifies them. The drafter does not read the target's tokens directly. It is
conditioned on a feature vector per position, built from the outputs of five
target layers. Before decoding can start, every prompt position needs that
feature, and the drafter's own KV cache is primed from it.

Before this change the features came from **prefill-by-decode**: the prompt was
pushed through the decode pipeline one step at a time just to collect the
target-layer activations, which is slow for long prompts. **True prefill**
computes the same features inside the ordinary chunked prefill pass, so a
prefill Galaxy can hand them to the decode machines with the target KV. This is
opt-in and does not change ordinary GPT-OSS prefill.

Terms used below: *proof rungs* are the test levels, from CPU-only contracts to
the full model on a Galaxy. *TSU* is tokens per second per user during decode.
*AIME* is the math benchmark used as the end-to-end quality gate. *SC1* and
*SC4* are the one-Galaxy and four-Galaxy CI machine types.

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
- on the final chunk, host fp32 logits and greedy `y0` for the last real prompt
  token (None on earlier chunks);
- enqueue timing hooks for each target FC and the final head.

The result owns its device tensor after return. A `dflash_sink` takes that
ownership when supplied. Padded storage rows may exist in the tensor, but only
the declared real range may be migrated or used to prime the drafter.

## Enablement

The model and runtime read no environment variables. Set
`TtPrefillRuntimeConfig.dflash_checkpoint_path` to the drafter checkpoint and
call `TtPrefillRuntime.prefill_chunk` directly:

- one-shot prompt: `dflash_handoff=True` (or `dflash_sink=...`);
- chunked prompt: request the handoff on **every** chunk, because each chunk's
  features are only computed when requested, and pass `dflash_final_chunk=False`
  on all but the last. Only the last chunk then pays for the LM head, the logits
  readback and the host argmax.

Default `prefill_chunk` return behavior is unchanged.

The common prefill runner is not wired for GPT-OSS DFlash yet: it requires
trace capture and never passes `dflash_handoff`/`dflash_sink`.
`GptOssPrefillAdapter.supports_dflash` stays `False`, and the adapter raises
`NotImplementedError` if the runner asks for DFlash (`PREFILL_DFLASH=1` with
`DFLASH_HF_MODEL`).

The current producer requires one runtime to own all 36 target layers. Passing
partial FC sums between pipeline ranks is intentionally left to the P/D owner;
the runtime fails instead of publishing an incomplete feature.

## Proof rungs

CPU contract (checkpoint metadata, FC slicing, final-token location, adapter refusal):

```bash
pytest models/demos/gpt_oss_d_p/tests/unit/test_dflash_prefill.py
```

Device accumulation (single card; the SP×TP case needs a 2×2 mesh and skips otherwise):

```bash
pytest models/demos/gpt_oss_d_p/tests/unit/test_dflash_device.py
```

Real 36-layer 1k Galaxy proof, with the same weights cache and settings as the CI section:

```bash
export HF_MODEL=/mnt/models/blaze/openai/gpt-oss-120b
export TT_CACHE_PATH=/mnt/models/blaze/openai/gpt-oss-120b
export GPT_OSS_WEIGHTS_FROM_CACHE=1 EXPERT_DTYPE=bf4
export PREFILL_TRACE_DIR=/mnt/models/blaze/openai/gpt-oss-120b/golden/longbook_qa_eng_prefill_5000
export BLAZE_DFLASH_SEED_REF=/mnt/models/blaze/openai/gpt-oss-120b/cache/dflash/seed_1016.pt
export TT_HF_DRAFT_MODEL=/path/to/gpt-oss-120b-DFlash
# Linear needs no mesh descriptor. For ring, unset this and export
# TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto
export PREFILL_TOPOLOGY=linear
mpirun --pernode --tag-output bash -lc 'python3 models/demos/gpt_oss_d_p/tests/galaxy_prefill_dflash.py'
```

The Galaxy rung keeps the existing target-KV PCC gate, checks the pre-norm
features and the final-row logits against the independent seed, reports
disabled/enabled latency and export cost, and writes a consumer fixture. All
failures are reported together. Floors (CI sets the same values explicitly):

| variable | default | gate |
|---|---|---|
| `GPT_OSS_KV_PCC_MIN` | unset | min per-layer target KV PCC (0.85 in CI) |
| `GPT_OSS_DFLASH_PCC_MIN` | 0.88 | aggregate pre-norm feature PCC |
| `GPT_OSS_DFLASH_POS_PCC_MIN` | 0.20 | worst single-position feature PCC |
| `GPT_OSS_DFLASH_LOGITS_PCC_MIN` | 0.90 | final-row logits PCC over the seed's top-64 ids |
| `GPT_OSS_DFLASH_TOP10_OVERLAP_MIN` | 7 | ids shared by our and the seed's top-10 |
| `GPT_OSS_DFLASH_SEED_TOP1_MIN` | unset | seed sanity: its teacher-forced top-1 hit rate on the prompt |
| `PREFILL_TSU_MIN` | unset | synchronized enabled-path prefill tokens/s |

`y0` must also equal the seed's top-1 or lie in its top-5. The seed's
teacher-forced top-1 hit rate is always printed (`SEED_TOP1`). Use a 1k-position
seed to gate the complete feature trajectory and final token.
`PREFILL_TPS_ITERS=N` reports the synchronized enabled-path median.
`PREFILL_REQUIRE_GALAXY=1` makes a missing Galaxy fail instead of skip.

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

The pre-P/D ladder mirrors Kimi's test semantics:

1. `gpt_oss_dflash_prefill` in the tt-metal Blaze prefill matrix runs the real
   36-layer producer on SC1, gates target KV, independent-HF features and
   final-row logits, emits the versioned handoff, and reports DFlash-disabled/
   enabled prefill latency. It sets no `PREFILL_TSU_MIN`, so throughput is
   reported, not gated. It runs one 1k chunk on linear topology, so the ring
   path and `actual_start > 0` are not covered for the features.
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
Producers call `TtPrefillRuntime.prefill_chunk` with `dflash_handoff` or
`dflash_sink` directly. Those stacks must:

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
