# GPT-OSS-120B 1K prefill acceptance

This gate exercises the existing common prefill runner on one 4×8 Blackhole Galaxy, with SP=4,
TP=8, all 36 layers, and two allocated cache slots. Each slot receives one 1,024-token request.
It uses the original MXFP4 GPT-OSS checkpoint and the runtime's BF4 expert weights. A complete
compatible TTNN weight cache can be reused with `GPT_OSS_WEIGHTS_FROM_CACHE=1`.

## Reference contract

Set `PREFILL_TRACE_DIR` to a complete HF golden trace, or set `PREFILL_PRODUCER_SLOT_TRACES` to
one shared trace or two comma-separated traces. Each trace must contain at least 1,024 valid token
IDs and all 36 `kv_cache/layer_N.safetensors` files with K/V shape `[1,8,S,64]`. Keys are post-RoPE
in the HF half-split frame; values are unrotated. A longer causal trace can supply the first 1K rows.
The attention sinks and 128-token sliding window must match the checkpoint.

Use distinct prompts when compatible goldens are available. Reusing one prompt in both slots checks
that both runtime slots produce correct KV, while the table test below uses distinct exact patterns
and checks that filling slot 1 preserves slot 0. Shared prompts alone do not prove model-level
isolation between different prompts.

The runner preserves the full K/V history for every layer, including sliding-attention layers.
`GPT_OSS_BOUNDED_SLIDING_KV` must be `0`. The published table has configurations K0–K7 followed by
V0–V7, 32 tokens per page, and 2,176 bytes per BFP8_B page.

Host layer acknowledgements are published only after device synchronization completes the queued
KV writes. Native migration reads occur outside the producer's command queue, so a Python layer
return alone is insufficient. The synchronization applies only when a host completion sink is attached.

## Commands

From a source-consistent TTNN environment, set:

```bash
export TT_METAL_HOME="$PWD"
export PYTHONPATH="$PWD"
export PREFILL_MODEL=gpt_oss_d_p
export PREFILL_HF_MODEL=/path/to/original-mxfp4-checkpoint
export HF_MODEL="$PREFILL_HF_MODEL"
export PREFILL_TRACE_DIR=/path/to/compatible-hf-golden
export PREFILL_TTNN_CACHE=/path/to/ttnn-cache-root
export GPT_OSS_WEIGHTS_FROM_CACHE=1
export GPT_OSS_BOUNDED_SLIDING_KV=0
```

The adapter resolves weights below
`$PREFILL_TTNN_CACHE/gpt_oss_d_p_bh_32dev/4x8`. Cache-only execution requires all expert-bias sidecars
as well as the quantized tensor files. An absent cache should fail rather than silently running
without biases. Creating a cache from the original checkpoint may temporarily unpack MXFP4 on the
host before TT conversion; this does not require a separate dequantized checkpoint.

Run the table gate, then the common producer/runner gate, serially on the reserved Galaxy:

```bash
python -m pytest models/demos/gpt_oss_d_p/tests/test_kv_cache_table.py \
  -k 1k_all_layers_two_slots
python -m pytest models/demos/common/prefill/tests/test_producer_runner_e2e.py \
  -k gptoss120b_1k
```

For the common SC1 launcher, configure its hostfile/TCP interface as for the other models and run:

```bash
export PREFILL_SUMMARIES=/path/to/persistent-results
bash models/demos/common/prefill/runners/ci/run_multirank_pcc.sh gptoss120b sc1
```

## Required evidence

- Table: 36,864 unique page addresses; protobuf round-trip preserves every lookup; every page exactly
  matches its independently seeded input; all slot-0 bytes survive a subsequent slot-1 fill.
- Model: both slots complete, exactly 72 layer acknowledgements arrive (36 per request), and every layer's K and V compare to
  the golden through the exported table. Tensor shapes and finite values are checked before PCC.
- Numerical floor: the existing common CI floor of 0.85 is the minimum. An explicitly stronger
  `PREFILL_STANDALONE_CHUNKED_PCC` is supported; lowering the floor or using NaN/infinity is rejected.
  Preserve actual per-layer K/V scores and any failures. This KV gate alone is not a logits or
  generation-quality sign-off.
- Runtime: published H2D descriptor/table/device map; the runner drains its shutdown sentinel and exits
  naturally with code 0 within the bounded shutdown wait. Forced cleanup or a nonzero exit fails.
- Provenance: source commit, checkpoint/config/tokenizer identity, exact token IDs, golden source,
  cache dtype and commands. A missing golden capture revision remains a reported limitation.

Device-free configuration checks:

```bash
python -m unittest \
  models.demos.gpt_oss_d_p.tests.unit.test_prefill_acceptance_config \
  models.demos.gpt_oss_d_p.tests.unit.test_prefill_runtime_contract -v
```
