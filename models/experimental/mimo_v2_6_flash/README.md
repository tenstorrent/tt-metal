# MiMo-V2.6-Flash on Blackhole Galaxy (TP=8 x DP=4)

TTNN implementation of [XiaomiMiMo/MiMo-V2.6-Flash-MOPD](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD)
(text only) for a 32-chip Blackhole Galaxy run as four independent TP=8 replicas, one per row of
the 4x8 mesh. Tracking issue: tenstorrent/tt-metal#60807. The parallel plan and memory budget are
in [doc/tp8_dp4_plan.md](doc/tp8_dp4_plan.md).

## Layout

| Path | What |
|---|---|
| `reference/config.py` | Config parsed from `config.json` (no remote code) |
| `reference/checkpoint.py` | Checkpoint reader: FP8 block dequant (incl. the per-chunk fused QKV layout), MXFP4 experts |
| `reference/model.py` | fp32 torch reference, bit-identical to HF `MiMoV2DecoderLayer` on real weights |
| `reference/make_golden.py` | Streams all 48 layers on the host and saves golden hidden states and logits |
| `tt/common.py` | `MeshPlan` (TP axis, mappers, all-reduce), weight cache helper, matmul configs |
| `tt/rope.py` | Partial RoPE via `rotary_embedding_llama` (interleaved first 64 of 192 dims) |
| `tt/attention.py` | SWA / global attention, paged KV, prefill and decode |
| `tt/moe.py` | fp32 routing + TP-sharded experts with `sparse_matmul` |
| `tt/decoder.py`, `tt/model.py` | Decoder layer and full model for one 1x8 replica |
| `tt/generator.py` | `MiMoGalaxyGenerator`: four replicas on the Galaxy rows (DP=4) |
| `tests/` | TP=8 x DP=4 tests on the 4x8 mesh; results in [doc/tp8_dp4_plan.md](doc/tp8_dp4_plan.md#validation-tests-measured-2026-10-15-on-one-bh-galaxy-branch-mimo-v26tp8-dp4) |

## Running

```bash
export MIMO_MODEL_PATH=/path/to/MiMo-V2.6-Flash-MOPD    # HF checkpoint
export MIMO_TT_CACHE=/path/to/tensor-cache              # converted TTNN weights (~180 GB for 48 layers), one subdir per checkpoint
python -m models.experimental.mimo_v2_6_flash.reference.make_golden --out golden_48l.pt
export MIMO_GOLDEN=golden_48l.pt

pytest models/experimental/mimo_v2_6_flash/tests/test_decoder_tp8_dp4.py   # one layer per type
pytest models/experimental/mimo_v2_6_flash/tests/test_model_tp8_dp4.py     # 48 layers
```

On a Galaxy whose CPLD firmware predates v1.16, reset with `tt-smi -glx_reset`: after `tt-smi -r`
some Ethernet links can stay untrained, and opening the mesh then fails in
`Cluster::initialize_ethernet_sockets` with `unordered_map::at`.

## Not done yet

- Chunked prefill (`start_pos > 0`) and prefix caching.
- Prefill tracing and performance tuning (decode is traced; everything is DRAM-interleaved with default program configs).
- Device sampling (the host reads full logits each step).
- vLLM integration (vllm-tt-plugin#216), sampling on device, speculative decoding.
