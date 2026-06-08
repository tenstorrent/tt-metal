# dots.ocr TP4 tests (Wormhole, 4× n150)

```bash
source python_env/bin/activate
export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) ARCH_NAME=wormhole_b0 \
       WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
export MESH_DEVICE=N150x4
```

```bash
# sdpa decode config sweep
python -m tracy -r -p -m pytest models/experimental/tt_symbiote/tests/sdpa_sweep_test.py -s

# vision attention  (S=11264)
pytest models/experimental/tt_symbiote/tests/test_dots_ocr_vision_attention_tp.py -k s11264 -s

# vision mlp  (S=11264)
pytest models/experimental/tt_symbiote/tests/test_dots_ocr_vision_mlp_tp.py -k m11264 -s

# text decoder prefill attention
pytest models/experimental/tt_symbiote/tests/test_dots_ocr_text_attention_prefill_tp.py -s

# text decoder prefill mlp
pytest models/experimental/tt_symbiote/tests/test_dots_ocr_text_mlp_prefill_tp.py -s

# text decoder decode mlp
pytest models/experimental/tt_symbiote/tests/test_dots_ocr_text_mlp_decode_tp.py -s


```

Tracy (any test): replace `pytest` with `python -m tracy -v -r -p -m pytest`.

## SDPA-decode config sweep — device time

batch=32, heads=12, kv_heads=2 (GQA), head_dim=128, cur_pos=128, MAX_SEQ=1024 · per-device.

| Rank | KV cache | k_chunk | exp_approx | fidelity | Device time (µs) |
|---|---|---|---|---|---|
| 1  | bfp8 | 64 | ✓ | LoFi | 33 |
| 2 | bfp8 | 64 | ✗ | HiFi2 | 35 |
| 3 | bfp8 | 64 | ✓ | HiFi2 | 36 |
| 4 | bfp8 | 32 | ✗ | HiFi2 | 38 |
| 5 | bf16 | 32 | ✗ | HiFi2 | 43 |
| 6 | bfp8 | 128 | ✗ | HiFi2 | 45 |
| 6 | bfp8 | 128 | ✓ | HiFi2 | 45 |
| 6 | bfp8 | 128 | ✓ | LoFi | 45 |
| 9 | bf16 | 64 | ✗ | HiFi2 | 48 |
| 9 | bf16 | 64 | ✓ | LoFi | 48 |
| 11 | bf16 | 128 | ✗ | HiFi2 | 70  |
| 11 | bf16 | 128 | ✓ | HiFi2 | 70 |
