<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Llama-3.1-8B Prefill (`llama_3p1_8b_d_p`)

Disaggregated **prefill** for Llama-3.1-8B on **one 4×8 Blackhole Galaxy** (SP=4, TP=8), plugging
into the model-agnostic `models/demos/common/prefill` engine. Decode runs separately in tt-blaze
(the 40-stage ring, on SC4) and receives the KV through the migration data plane.

Umbrella: [tt-blaze#4137](https://github.com/tenstorrent/tt-blaze/issues/4137) ·
prefill: [#4138](https://github.com/tenstorrent/tt-blaze/issues/4138)

Host/device RoPE, RMSNorm, MLP, QKV projection, and source-cache tests have recorded passes.
These modules are published through commit 4cf42fb. Attention accuracy remains under investigation.
The decoder block, full model, runtime, and migration have not passed their integration gates.
See the [implementation and verification roadmap](ROADMAP.md) for the staged plan.

The [native KV migration learning guide](docs/kv-migration-learning.md) explains the complete
request path, address layouts, completion signals, and required tests. Its
[standalone HTML edition](docs/kv-migration-learning.html) includes diagrams and expandable sections.

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
reference/ scripts/ tests/ tests/unit/ utils/   reference, test, and support areas for the stack above
```

## Correctness reference

Module tests compare against independent PyTorch/Hugging Face references. The full-model reference
is planned with [#4147](https://github.com/tenstorrent/tt-blaze/issues/4147).
The roadmap and individual tests record their numerical limits.

Report both correlation and magnitude-sensitive error. Keep device-input precision diagnostics
separate from source-reference acceptance. Exact cache-byte checks establish placement; they do
not establish complete model correctness.
