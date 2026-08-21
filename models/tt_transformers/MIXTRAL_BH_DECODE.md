# Mixtral decode on Blackhole: `in0_block_w` L1 overflow

Branch notes for the local changes on top of `micah/native-sim`. The Mixtral decode path is the functional fix; the test env-var knobs are only for simulator sweeps.

## Problem

Mixtral-8x7B decode uses DRAM-sharded matmuls for the expert MLP (`w1`/`w3`/`w2`). Those weights are sharded across the device's DRAM banks, one shard per bank, then streamed into L1 circular buffers on a small worker grid (`num_cores=8`).

The weight CB size is:

```
in0_block_w × (tiles per DRAM bank along N)
```

`dram_matmul_config` used to pick `in0_block_w` as the largest divisor of `k / (32 × num_cores)` that is at most 8. For Mixtral `w1`/`w3` that is:

| | `k` | `n` | DRAM banks | tiles per bank along N | default `in0_block_w` |
|---|---|---|---|---|---|
| Wormhole | 4096 | 14336 | 12 | `ceil(14336 / 32 / 12) = 38` | 8 |
| Blackhole | 4096 | 14336 | 8 | `ceil(14336 / 32 / 8) = 56` | 8 |

Wormhole fits: `8 × 38` tiles of bfp8 is well inside L1.

Blackhole does not. The same `in0_block_w=8` with a 56-tile-wide shard puts the weight CB **~209 KB over L1**. Decode then fails at program compile / CB allocation, not at numerical PCC.

`w2` (`n=4096`) is narrower, so its shards stay small enough on both architectures. The overflow is specific to the two wide up-projections.

## Fix

`ModelArgs.dram_matmul_config` now takes `max_in0_block_w` (default still 8, so every existing caller is unchanged).

`TtMixtralMLP` decode `w1`/`w3` pass `max_in0_block_w=4` on Blackhole and keep 8 on Wormhole:

```python
ff_block_w = 4 if is_blackhole() else 8
```

Halving the K-block doubles inner-loop iterations and halves the weight CB. That is enough to fit L1 on BH; WH keeps the already-tuned value.

Files:

- `models/tt_transformers/tt/model_config.py` — optional cap on `in0_block_w`
- `models/tt_transformers/tt/mixtral_mlp.py` — BH-only `w1`/`w3` cap in the decode path

Prefill is untouched. Prefill uses a different matmul config (`core_grid=8×8`, not DRAM-sharded decode).

## Why not change the default for everyone

`in0_block_w=8` is a throughput knob, not a correctness knob. Lowering it globally would slow Wormhole Mixtral decode (and any other DRAM-sharded matmul that currently uses the default) for no L1 benefit. The extra tiles per bank only show up on Blackhole, and only on the widest Mixtral experts.

## Test knobs (secondary)

The Mixtral and attention unit tests now read optional env vars so simulator runs can change context / sequence / step count without editing parametrization. Unset, behavior is identical to before.

| Variable | Where | Default | Meaning |
|---|---|---|---|
| `TTSIM_CTX` | `test_attention.py`, `test_mixtral_decoder.py` | 0 / existing start pos | decode start position |
| `TTSIM_MAXSEQ` | `test_attention.py`, `test_attention_prefill.py`, `test_mixtral_decoder.py` | 256 / existing | KV-cache / `max_seq_len` |
| `TTSIM_STEPS` | `test_attention.py`, `test_mixtral_decoder.py`, `test_mixtral_moe.py` | 10 / 1 | generation or iteration count |
| `TTSIM_SEQ` | `test_mixtral_moe.py` | 1 | seqlen |
| `TTSIM_BATCH` | `test_mixtral_moe.py` | 32 | batch |

Example:

```bash
TTSIM_CTX=128 TTSIM_MAXSEQ=2048 TTSIM_STEPS=4 \
  pytest models/tt_transformers/tests/mixtral/test_mixtral_decoder.py
```
