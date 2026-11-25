# GLM-5.3-Flash

The working implementation of `zai-org/GLM-5.3-Flash` (`glm5_next_text`), plus the CPU reference and
the unit tests. Moved here from `models/autoports/zai_org_glm_5_3_flash/` so it sits with the other
model ports under `models/experimental/`.

Provenance: `aperezvicente-TT/glm-5.3-flash` @ `18457222c2c` (Apache-2.0), which was written for
Blackhole. This tree is that code ported to **Wormhole B0**, with the fixes and the test suite
described in [`TESTS_AND_PORT.md`](TESTS_AND_PORT.md).

## Status on Wormhole Galaxy (32 chips, `j11glx01`)

```
PYTHONPATH=$PWD ./python_env/bin/python -m pytest models/experimental/glm53_flash/tests/ -q
→ 254 passed, 2 skipped, 7 xfailed
```

Six xfails are KDA **prefill**, blocked by a hardware limit: `prepare_chunk_recurrence` needs 33
dataflow buffers per core and Wormhole Gen1 allows 32. KDA decode works. See
[`TESTS_AND_PORT.md`](TESTS_AND_PORT.md) for the two approaches to lifting it. The seventh is the
upstream `Terminal` lm_head, which needs the DRAM prefetcher (`DramReaderPlacement`); `GLMHead` in
`tt/kda_block/terminal.py` is the Wormhole replacement.

## Layout

| path | what |
|---|---|
| `tt/` | the device implementation (~90 modules) |
| `tt/kernels/` | ~250 hand-written C++ kernels (KDA, MLA, MoE, mHC, fabric) |
| `tests/` | unit tests, one file per module, each graded against a torch reference |
| `reference/` | HF-based CPU reference: builds real `Glm5NextTextDecoderLayer`s from the checkpoint |
| `mesh/` | mesh graph descriptors (1x1, 2x2, 1x6, 3x2, 1x8 line/ring, 2x4) |
| `TESTS_AND_PORT.md` | the Wormhole port write-up and test inventory |
| `EXTERNAL_REPO.md` | survey of the upstream repo against this port |

## The model in one paragraph

45 layers. **34 are KDA** (Kimi Delta Attention — linear attention, constant-size recurrent state, no
positions) and **11 are DSA/MLA** (at 3, 7, 11 … 43). The first 3 layers have a dense SwiGLU FFN
(`I = 12288`), the rest a 288-expert MoE (`I = 2048`, top-8). Every layer is wrapped in **mHC**
(`hc_mult = 4`, Sinkhorn-constrained 4-stream residual). Hidden 4096, vocab 154880, fp8 weights in
128×128 blocks, **no RoPE at all** (`mla_use_nope`, `qk_rope_head_dim = 0`) — which is what makes
MLA's absorbed-query form exact.

## Device modules

| file | role |
|---|---|
| `tt/config.py` | `Glm5NextTextConfig`, `LayerSpec`, layer-kind dispatch |
| `tt/precision.py` | the dtype map: experts `bfloat4_b`, projections `bfloat8_b`, `A_log`/`dt_bias`/router bias/mHC constants fp32 |
| `tt/weights.py` | weight sources: pre-packed tile cache, or a host state dict |
| `tt/decoder_layer.py` | one layer: hc → norm → attn → hc → norm → FFN → hc |
| `tt/hyperconnection.py` | mHC pre/post, Sinkhorn via `mhc_split_sinkhorn` |
| `tt/kda.py` | KDA block (prefill + decode) |
| `tt/mla.py` | NoPE MLA + DSA, absorbed form, paged latent cache |
| `tt/moe.py`, `tt/mlp.py` | router, routed experts, shared expert, dense SwiGLU |
| `tt/functional_decoder.py` | the layer + cache orchestration, prefill/decode entry points |
| `tt/model.py`, `tt/generator.py` | full model, serving loop |
| `tt/prefill_tp.py`, `tt/stack_model.py` | TP prefill, multi-chip stack |
| `tt/pmk_*.py`, `tt/mk_*.py` | the megakernel path |

## The CPU reference

`reference/` builds real HF `Glm5NextTextDecoderLayer` modules from the checkpoint and runs them on
CPU, so the device can be graded against the model the checkpoint actually ships. It needs
`transformers==5.20.2` (`glm5_next` appeared in 5.16); `python_env` has it.

```bash
PYTHONPATH=$PWD ./python_env/bin/python -c "
from models.experimental.glm53_flash.reference.checkpoint import default_snapshot_dir, shard_headers
from models.experimental.glm53_flash.reference.hf_layer_reference import load_config, build_layer
snap = default_snapshot_dir(); hdr = shard_headers(snap); cfg, text = load_config(snap)
layer, sd = build_layer(snap, text, 0, hdr)   # real Glm5NextTextDecoderLayer, real weights
print(len(sd), 'tensors')"
```

Useful pieces: `checkpoint.py` (safetensors reader + 128×128 fp8 dequant),
`hf_layer_reference.py` (per-layer dumps, greedy generation), `logits_check.py` (the accuracy
metric that matters — per-position logits PCC, top-1/top-5, KL), `check_kda_reference.py` (proves the
in-tree KDA torch reference equals HF's `Glm5NextTextLinearAttention` on real weights).

## History of this directory

It briefly held a from-scratch skeleton written against `deepseek_v41_flash`'s architecture, while the
upstream implementation was still being located. That skeleton was superseded by this tree: its three
stub modules that collided by name (`tt/config.py`, `tt/kda.py`, `tt/model.py`) were replaced by the
working ones. `reference/` came from the skeleton and is kept because it is the CPU truth the device
tests grade against.
