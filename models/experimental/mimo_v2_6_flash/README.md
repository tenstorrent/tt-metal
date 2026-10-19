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
| `tt/kv.py` | Paged-KV tables: global layers split each chip pair's users; SWA layers page by absolute position with only the in-window blocks backed (`SwaBlockTable`) |
| `tt/attention.py` | SWA / global attention: chunked prefill, decode (`SwaDecodeQkv`, MLA decode for global layers) |
| `tt/proj_stream.py` | Bank-local streaming decode projections (qkv, o) and the fused router + sigmoid + bias (`RouterScoreStream`) |
| `tt/moe.py`, `tt/moe_stream.py` | fp32 routing; decode streams the selected experts bank-locally (`GateUpStream`, `DownStream`) and reduces over the row with a fabric all-gather fused with the residual add and the next RMSNorm (`ReduceExpand`) |
| `tt/moe_prefill.py` | Prefill experts, expert-major: on-device plan, grouped expert kernel, k-sum (`DevicePlan`, `GroupedExperts`, `CombineK`) |
| `tt/greedy_head.py` | Greedy decode head (argmax across the row's vocab shards on device) |
| `tt/kernels/` | `generic_op` kernels: `moe_*` / `fabric_*` (decode MoE and reductions), `attn/`, `proj/`, `prefill/` |
| `tt/decoder.py`, `tt/model.py` | Decoder layer and full model for one 1x8 replica |
| `tt/generator.py` | `MiMoGalaxyGenerator`: four replicas on the Galaxy rows (DP=4), traced decode with a device-resident greedy loop |
| `tests/` | TP=8 x DP=4 tests on the 4x8 mesh; `tests/perf/` perf scripts and tt-kernel-agent check / bench commands. Results in [doc/tp8_dp4_plan.md](doc/tp8_dp4_plan.md) |

## Running

```bash
export MIMO_MODEL_PATH=/path/to/MiMo-V2.6-Flash-MOPD    # HF checkpoint
export MIMO_TT_CACHE=/path/to/tensor-cache              # converted TTNN weights (~186 GB for 48 layers), one subdir per checkpoint
python -m models.experimental.mimo_v2_6_flash.reference.make_golden --out golden_48l.pt
export MIMO_GOLDEN=golden_48l.pt

pytest models/experimental/mimo_v2_6_flash/tests/test_kv_tables.py               # host-side SWA block table (no device)
pytest models/experimental/mimo_v2_6_flash/tests/test_decoder_tp8_dp4.py          # one layer per type, prefill + decode
pytest models/experimental/mimo_v2_6_flash/tests/test_chunked_prefill_tp8_dp4.py  # chunked prefill per layer type
pytest models/experimental/mimo_v2_6_flash/tests/test_moe_decode_tp8.py           # streaming MoE decode vs prefill
pytest models/experimental/mimo_v2_6_flash/tests/test_model_tp8_dp4.py            # 48 layers vs the fp32 golden
pytest models/experimental/mimo_v2_6_flash/tests/test_model_chunked_tp8_dp4.py    # 48 layers: one shot vs chunks vs prefix hit
pytest models/experimental/mimo_v2_6_flash/tests/test_model_traced_tp8_dp4.py     # traced decode
python models/experimental/mimo_v2_6_flash/tests/perf/decode_bandwidth.py --steps 32 --repeats 5
python models/experimental/mimo_v2_6_flash/tests/perf/prefill_time.py
```

On a Galaxy whose CPLD firmware predates v1.16, reset with `tt-smi -glx_reset`: after `tt-smi -r`
some Ethernet links can stay untrained, and opening the mesh then fails in
`Cluster::initialize_ethernet_sockets` with `unordered_map::at`.

## Not done yet

- Prefill dispatch: prefill runs eagerly and the host (Python op dispatch, `generic_op` descriptors rebuilt per call)
  limits four concurrent rows to ~8.3k tokens/s per Galaxy against ~29k the devices could sustain; caching the
  descriptors or tracing fixed-size chunks would remove it.
- Non-greedy sampling on device (the device loop is greedy; the host loop reads per-chip top-k candidates).
- vLLM integration (vllm-tt-plugin#216), speculative decoding.
