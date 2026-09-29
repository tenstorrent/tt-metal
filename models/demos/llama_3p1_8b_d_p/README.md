<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Llama-3.1-8B Prefill (`llama_3p1_8b_d_p`)

Disaggregated **prefill** for Llama-3.1-8B on **one 4×8 Blackhole Galaxy** (SP=4, TP=8), plugging
into the model-agnostic `models/demos/common/prefill` engine. Decode runs separately in tt-blaze
(the 40-stage ring, on SC4) and receives the KV through the migration data plane.

Umbrella: [tt-blaze#4137](https://github.com/tenstorrent/tt-blaze/issues/4137) ·
prefill: [#4138](https://github.com/tenstorrent/tt-blaze/issues/4138)

## Configuration

```
SP = 4 (mesh rows)   sequence sharded block-cyclic; weights IDENTICAL down a column
TP = 8 (mesh cols)   heads + MLP width sharded; weights DIFFER per column

PREFILL_CHUNK_SIZE    1024      -> S_loc = 256 tokens/chip;  must satisfy % (SP*32) == 0
PREFILL_MAX_SEQ_LEN   2048      -> cache depth 512/chip;     must satisfy % CHUNK_SIZE == 0
PREFILL_NUM_LAYERS    32        runner defaults to 61 — PIN IT
layers_per_chunk      32        engine defaults to 64 — PIN IT
KV cache dtype        bfloat8_b (matches decode; compute is bf16)
```

TP=8 is not arbitrary: `num_key_value_heads = 8` over 8 columns puts **exactly one KV head per
chip**, so a KV chunk lives on exactly one chip and the migration DeviceGroup degenerates to a
single node. It also divides every width cleanly — 4096/8 = 512 (16 tiles), 14336/8 = 1792
(56 tiles) — so none of the tile-alignment padding other models carry is needed here.

## Layout

```
reference/llama_3p1_8b_config.py   dim SSOT (Llama31_8BConfig, exposes FABRIC_PAYLOAD_SIZE)
tt/config.py                       MeshConfig — SP/TP validation + shard mappers
tt/model_config.py                 ModelArgs — weights path, HF config cross-check
tt/runners/adapters/llama_3p1_8b.py   PrefillModelAdapter subclass (the engine's only seam)
tt/runners/manifests/llama_3p1_8b.json  model manifest (keeps rank bindings model-agnostic)
reference/ scripts/ tests/ tests/unit/ utils/   placeholders for the stack above
```

## Correctness reference

Bottom-up PCC against a self-contained torch/HF reference (`reference/model.py`, lands with
[#4147](https://github.com/tenstorrent/tt-blaze/issues/4147)), plus CPU-generated golden KV for the
full-model check. Thresholds: norm/rope ≥ 0.999, attention/MLP ≥ 0.99. Goldens must round-trip
weights through the device dtype (bf8_b cache) before PCC, or a full-precision golden leaves a
spurious ~0.94–0.96 gap that reads as a real bug.
