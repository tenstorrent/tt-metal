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
| `tt/attention.py` | SWA / global attention: prefill, decode (`SwaDecodeQkv`, MLA decode for global layers) |
| `tt/proj_stream.py` | Bank-local streaming decode projections (qkv, o) and the fused router + sigmoid + bias (`RouterScoreStream`) |
| `tt/moe.py`, `tt/moe_stream.py` | fp32 routing; decode streams the selected experts bank-locally (`GateUpStream`, `DownStream`) and reduces over the row with a fabric all-gather fused with the residual add and the next RMSNorm (`ReduceExpand`) |
| `tt/moe_prefill.py` | Prefill experts, expert-major: on-device plan, grouped expert kernel, k-sum (`DevicePlan`, `GroupedExperts`, `CombineK`) |
| `tt/greedy_head.py` | Greedy decode head (argmax across the row's vocab shards on device) |
| `tt/kernels/` | `generic_op` kernels: `moe_*` / `fabric_*` (decode MoE and reductions), `attn/`, `proj/`, `prefill/` (MoE prefill) |
| `tt/decoder.py`, `tt/model.py` | Decoder layer and full model for one 1x8 replica |
| `tt/generator.py` | `MiMoGalaxyGenerator`: four replicas on the Galaxy rows (DP=4), traced decode with a device-resident greedy loop |
| `reference/mtp.py`, `reference/spec_decode.py` | fp32 MTP layer (vLLM's `MiMoV2MTP`), the drafters (one layer, or one layer per draft) and the device program's mirror; host bookkeeping of MTP speculative decoding (verify passes, draft rows, the stash, the MTP ring) |
| `tt/mtp.py` | `TtMiMoMTP`: the MTP layers as a traced 3-token drafter on one replica (draft i + 1 from layer i, or one layer) |
| `tt/mtp_backfill.py`, `reference/mtp_backfill.py` | The target-hidden history (decode, verify and prefill write it) and the MTP ring backfill from it (`kernels/mtp_history.cpp`); their host mirror |
| `tt/dflash.py`, `reference/dflash.py` | `TtDFlash`: the checkpoint's DFlash draft model (`dflash/`, 5 dense layers) as a traced one-pass drafter on one replica (`MIMO_SPEC_DECODE=dflash`): the decode / verify traces keep target layers 0/11/23/35/47's outputs, the draft program writes the committed rows into a context ring and drafts [anchor, 7 masks] non-causally; the fp32 reference as vLLM's `qwen3_dflash` runs it |
| `tt/verify_sampler.py`, `reference/sample_verify.py` | Speculative acceptance on device (the plugin's `fused_sample`): the verify's accept walk on the device sampler's candidates (`kernels/sample_verify.cpp`; greedy only, `kernels/greedy_verify.cpp`, past the sampler's users), and its host layouts and integer mirror |
| `tt/spec_inputs.py`, `tt/spec_chain.py`, `reference/spec_inputs.py`, `reference/spec_chain.py` | A verify's and a draft's packed inputs, derived on device; the MTP or DFlash draft chained behind the verify and its acceptance (`kernels/spec_chain.cpp`), with one result page per pass; their torch / integer mirrors |
| `tt/generator_vllm.py` | `MiMoV2ForCausalLM`, the vllm-tt-plugin model class: DP=4 as in-process lanes, vLLM-owned hybrid KV (1 global + 5 SWA KV cache groups). Launch and results in [doc/vllm_integration/README.md](doc/vllm_integration/README.md) |
| `reference/vision.py`, `reference/vision_inputs.py` | fp32 MiMo ViT (`visual.*`), bit-identical to HF's `MiMoVisionTransformer` except the merger norm (RMSNorm as vLLM, not HF's LayerNorm); vLLM's `MiMoV2OmniMultiModalProcessor` on the host for inputs as served |
| `tt/vision/` | The ViT on one 1x8 row, TP=8 with a sequence-parallel residual stream (`vit.py`), and `MiMoVisionEncoder` (`encoder.py`: `embed_image` / `embed_video` -> host bf16 [n, 4096]; `warm()` compiles the fixed program set: every pass is padded to one of 21 row buckets; passes up to 4096 rows replay traces, `capture_traces()`); `tests/perf/vision_latency.py` times it, `tests/perf/allreduce_fp32_race.py` checks the fp32 all-reduce race standalone |
| `reference/audio.py`, `tt/audio/host_backend.py` | Audio input: fp32 torch reference of mel -> audio tokenizer encoder -> 20 RVQ codes -> patch encoder -> [n, 4096] (bit-identical to HF), and `HostAudioBackend.embed_audio` on the host CPU; test clips from `reference/make_audio_clips.py` |
| `tt/multimodal.py`, `reference/make_golden_mm.py` | Multimodal input (doc/multimodal_plan.md): the plugin's items, the encoder backend interface (device ViT `tt/vision/`, host audio `tt/audio/`), the item embedding cache and the scatter into a prefill chunk's rows, which `TtMiMoModel.prefill_enqueue(embeds=)` merges on device; `MiMoV2OmniForCausalLM` in `tt/generator_vllm.py` serves it (`TT_MODEL_CLASS_OVERRIDES=TTMiMoV2OmniForCausalLM=models.experimental.mimo_v2_6_flash.tt.generator_vllm:MiMoV2OmniForCausalLM`); the multimodal golden |
| `tests/` | TP=8 x DP=4 tests on the 4x8 mesh; `tests/perf/` perf scripts and tt-kernel-agent check / bench commands. Results in [doc/tp8_dp4_plan.md](doc/tp8_dp4_plan.md) |

## Running

```bash
export MIMO_MODEL_PATH=/path/to/MiMo-V2.6-Flash-MOPD    # HF checkpoint
export MIMO_TT_CACHE=/path/to/tensor-cache              # converted TTNN weights (~186 GB for 48 layers), one subdir per checkpoint
python -m models.experimental.mimo_v2_6_flash.reference.make_golden --out golden_48l.pt
export MIMO_GOLDEN=golden_48l.pt

pytest models/experimental/mimo_v2_6_flash/tests/test_kv_tables.py               # page tables, host only
pytest models/experimental/mimo_v2_6_flash/tests/test_prefill_helpers.py          # prefill padding, host only
pytest models/experimental/mimo_v2_6_flash/tests/test_sample_verify_host.py       # device acceptance's mirror, host only
pytest models/experimental/mimo_v2_6_flash/tests/test_decoder_tp8_dp4.py          # one layer per type, prefill + decode
pytest models/experimental/mimo_v2_6_flash/tests/test_chunked_prefill_tp8_dp4.py  # chunked prefill, one layer per type
pytest models/experimental/mimo_v2_6_flash/tests/test_moe_decode_tp8.py           # streaming MoE decode vs prefill
pytest models/experimental/mimo_v2_6_flash/tests/test_model_tp8_dp4.py            # 48 layers vs the fp32 golden
pytest models/experimental/mimo_v2_6_flash/tests/test_model_chunked_tp8_dp4.py    # 48 layers: chunked, prefix hit
pytest models/experimental/mimo_v2_6_flash/tests/test_model_traced_tp8_dp4.py     # traced decode, device greedy loop
pytest models/experimental/mimo_v2_6_flash/tests/test_dflash_tp8.py               # DFlash drafter vs the fp32 reference
pytest models/experimental/mimo_v2_6_flash/tests/test_vision_reference_host.py    # ViT reference == HF, host only
TT_VISIBLE_DEVICES=<row chips> pytest models/experimental/mimo_v2_6_flash/tests/test_vision_tp8.py  # ViT on one row
pytest models/experimental/mimo_v2_6_flash/tests/test_multimodal_host.py          # multimodal items, cache, scatter, host only
pytest models/experimental/mimo_v2_6_flash/tests/test_prefill_embeds_tp8.py       # prefill from embeddings == from tokens
python -m models.experimental.mimo_v2_6_flash.reference.make_golden_mm --audio clip.wav --out golden_mm_48l.pt
MIMO_MM_GOLDEN=golden_mm_48l.pt pytest models/experimental/mimo_v2_6_flash/tests/test_model_mm_tp8.py  # image / audio
python models/experimental/mimo_v2_6_flash/tests/perf/decode_bandwidth.py --steps 32 --repeats 5
python models/experimental/mimo_v2_6_flash/tests/perf/prefill_time.py             # 4K prompt, 32K in 4K chunks, one row
```

The device tests can outlast pytest's default 300 s timeout on a cold weight cache; pass `--timeout 3000`.

On a Galaxy whose CPLD firmware predates v1.16, reset with `tt-smi -glx_reset`: after `tt-smi -r`
some Ethernet links can stay untrained, and opening the mesh then fails in
`Cluster::initialize_ethernet_sockets` with `unordered_map::at`.

## Not done yet

- Prefill dispatch: prefill runs eagerly, and one Python thread dispatches every op of all four rows (`generic_op`
  descriptors are rebuilt per call). A 4K prompt takes 0.73 s on one row (5.6k tokens/s), but four rows at once reach
  only 8.3k tokens/s per Galaxy. Caching the descriptors or tracing fixed-size chunks are the candidates.
- Non-greedy sampling on device (the device loop is greedy; the host loop reads per-chip top-k candidates).
- Serving gaps (doc/vllm_integration/README.md): device sampling is greedy only (other sampling runs on the host);
  chunked prefill's slot parity is not measured.
- Speculative decoding: MTP drafting through vLLM is implemented but not yet run on hardware
  (doc/vllm_integration/README.md, Speculative decoding); DFlash drafts its whole block (K = 7) on the 8-slot verify
  (`MIMO_SPEC_SLOTS=8`), chained on device, within a few percent of chained MTP at c=1; with two requests per lane
  (`MIMO_SPEC_GROUPS=2`, c=8) both drafters draft for both, MTP +46% tokens/s over no speculation
  (doc/vllm_integration/README.md, Speculative decoding (DFlash), Two requests per lane).
