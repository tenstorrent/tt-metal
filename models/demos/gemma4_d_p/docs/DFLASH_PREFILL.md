# DFlash KV-only prefill

Enable `create_tt_model(..., dflash_enabled=True)` or set `PREFILL_DFLASH=1`
for the service and canonical demo. Disabled mode allocates no draft weights or
caches. Asynchronous collectives are optional.

## Computation and storage

- Checkpoint: `z-lab/gemma-4-31B-it-DFlash`. Read dimensions and tap IDs from its
  config; load only FC, hidden norm, K/V projections, and K norms.
- Tap target decoder outputs after their layer scalars. This checkpoint uses
  zero-based layers `[1, 12, 23, 35, 46, 57]`. Gather the target's TP-split
  token rows before each projection, preserving the residual for the next layer.
- Accumulate one FC block projection per tap, with output columns sharded across
  TP. All-gather once, then apply the context RMS norm.
- Each draft layer computes `K = RoPE(k_norm(k_proj(context)))` and
  `V = v_proj(context)`. No draft attention, MLP, Q/O projection, or logits.
- RoPE uses the target's absolute positions, theta 1,000,000, and HF half-split
  channel order, matching Gemma's DFlash decoder.
- Each target user gets a matching draft slot. BFP8 K/V tensors have per-device
  shape `[users * draft_layers, kv_heads / TP, max_seq_len / CP, head_dim]`.
  Batch index is `user_id * draft_layers + draft_layer_idx`.
- All five draft layers retain full history, including sliding layers. At CP8/TP4,
  caches use 2.656 GiB per 256K user across the Galaxy. BF16 weights use about
  3.40 GiB; replicated 256K RoPE tables add 4 GiB, excluding scratch.
- Service acknowledgements follow target layers 0–59, then draft layers 60–64.
  Migration adds `dflash_k_h00`–`dflash_k_h07` and corresponding V entries, plus
  stages for the two packed tensors. Existing target config IDs remain unchanged.

## Prefill-to-decode migration

With `PREFILL_DFLASH=1`, the published table has 52 configs: the existing 36
target configs followed by `dflash_k_h00`–`dflash_k_h07` and
`dflash_v_h00`–`dflash_v_h07`. Target rows remain 0–59; draft rows are 60–64.
Each draft chunk is 32 tokens by 128 channels of BFP8, or 4352 bytes.

The standalone `migration_driver` derives the draft layer extent from this
published table and requests all 65 layers by default. It also waits for all 65
layer acknowledgements per prefill chunk. An explicit `PREFILL_MIGRATION_LAYERS`
subset is honored; a target-only subset does not migrate the draft. When
`PREFILL_DFLASH=1`, a missing table or a table without draft configs is rejected
before issuing migration. Direct `MigrationDriver.run` callers must pass
`kv_table` to include the draft rows.

The SC20/DFlash decode adapter in `tt-blaze` uses this same 52-config contract.
Its physical layout is different: TP8 shards one head per device, with a private
32-row proposal tile before committed history and padded, bank-striped slot
storage. Decode's table maps logical position `p` to row `p + 32`; prefill exports
logical position `p` directly from its CP8/TP4 packed caches. The worker copies
4352-byte chunks between the independently described addresses. No prefill-side
proposal padding or cache repacking is required. Both endpoints must enable the
drafter contract before pairing.

Host regressions cover the migration driver's complete layer range, explicit
subsets, missing-table rejection, and corruption in the final draft V head.
The sibling Blaze test `test_gemma4_dflash_prefill_contract.py` exercises both
production table builders, protobuf I/O, and byte copies into decode's physical
slot layout. These checks do not replace a multi-host hardware migration run.

## Weights

The prepared checkpoint lives at
`/mnt/models/huggingface/hub/models--z-lab--gemma-4-31B-it-DFlash/`.
Converted weights live at
`/mnt/models/huggingface/tt_cache/gemma4_d_p/z-lab--gemma-4-31B-it-DFlash/7c2c04e4905eeb67/bf16_mesh8x4_v1/`.

Both roots follow `HF_HOME`. Draft tensor caches are separate from `TT_CACHE_PATH`
(the target cache) and identified by consumed weight contents, config, mesh,
precision, and layout version. Direct callers can override `dflash_checkpoint_path`;
the service uses `DFLASH_HF_MODEL`.

## Run

From the repository root:

```bash
export HF_MODEL=google/gemma-4-31B-it
export HF_HOME=/mnt/models/huggingface
export TT_CACHE_PATH=/mnt/models/huggingface/tt_cache/gemma4_d_p/google--gemma-4-31B-it
export HF_HUB_OFFLINE=1
export PREFILL_DFLASH=1
pytest \
  'models/demos/gemma4_d_p/demo/text_demo_prefill.py::test_prefill_long_context_traced[blackhole-ctx_256k-chunk8192-text-8x4]' \
  -sv --timeout=1800
```

The same environment enables DFlash in the [prefill service](PREFILL_SERVICE.md).
Standalone callers can access the resulting caches through `model.dflash.kv_cache`.

## Tests

- `tests/unit/test_dflash.py`: selective loading, cache identity, validation,
  tap/ack ordering, accumulator reset, and draft migration addresses/stages.
- `tests/test_dflash.py`: independent PyTorch K/V reference, checkpoint weights,
  eager and trace execution, changing inputs/positions, interleaved users, and
  preservation of untouched slots and earlier cache rows. Includes chunk6656 and TP sequence-parallel, block-sharded target outputs.
- `tests/test_prefill_service.py -k dflash`: real producer/service, six full 256K
  slots, 8193-token reuse, preserved target/draft KV samples, and all 65 acks.

Run these with the environment above. Set `GEMMA4_CCL_ASYNC=1` separately to
exercise asynchronous collectives.
