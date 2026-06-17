# Kimi K2.5 (kimi_k25)

## Platforms
    Galaxy (WH) — TG (32 chips), DUAL (64 chips), QUAD (128 chips)

## Introduction

This demo runs inference for [moonshotai/Kimi-K2.5](https://huggingface.co/moonshotai/Kimi-K2.5) — a 384-expert Mixture-of-Experts (MoE) language model with Multi-head Latent Attention (MLA).

Kimi K2.5's text backbone is architecturally near-identical to DeepSeek V3. This implementation reuses the `models/demos/deepseek_v3` runtime (MLA, MoE dispatch, CCL, RoPE, embeddings) and adds:

1. **`utils/config_adapter.py`** — maps Kimi K2.5 HF config (`kimi_k2`/`kimi_k25` model type) to DSV3-compatible parameter struct; validates all critical fields at startup
2. **`utils/weight_loader.py`** — `KimiLazyStateDict` that strips the `language_model.` checkpoint prefix and auto-dequantizes INT4 group-32 expert weights on access
3. **`tt/kimi_model.py`** — `KimiGenerator` subclass of `DeepseekGenerator`; injects weight loader + config validation
4. **`scripts/kimi_preflight.py`** — hardware readiness checker; validates environment, devices, imports, config before running inference

## Key Architecture Differences from DeepSeek V3

```
Parameter              Kimi K2.5         DeepSeek V3
---------------------  ---------------   -----------
Routed experts         384               256
Attention heads        64                128
n_group / topk_group   1 / 1 (flat)      8 / 4 (grouped)
first_k_dense_replace  1                 3
rope_theta             50,000            10,000
rms_norm_eps           1e-5              1e-6
vocab_size             163,840           ~129,280
routed_scaling_factor  2.827             2.5
Weight quantization    INT4 group-32     FP8
                       (experts only)
MTP layers             0                 1
```

## Hardware Requirements

- **Minimum**: 1× Galaxy 6U (TG, 32 Wormhole chips, 384 GB DRAM)
- Weights: ~220 GB (85 GB BF16 attention + ~120 GB INT4 experts + ~15 GB misc)
- KV cache (256K context, 1 seq): ~530 MB/device on TG

```
Topology              Chips   DRAM     Experts/device
--------------------  ------  -------  --------------
TG (1× Galaxy 6U)    32      384 GB   12
DUAL (2× Galaxy)      64      768 GB   6
QUAD (4× Galaxy)      128     1.5 TB   3
```

Single-chip topologies (N150, N300) are **not supported** — expert sharding requires ≥32 chips.

## Prerequisites

- `tenstorrent/tt-metal` cloned and built: see [INSTALLING.md](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
- `pip install transformers>=4.57.1 safetensors compressed-tensors`
- HF weights downloaded: `moonshotai/Kimi-K2.5` (~555 GB raw, 64 shards)

## Environment Variables

```bash
export KIMI_HF_MODEL=/path/to/Kimi-K2.5       # local weight directory (required for real weights)
export KIMI_CACHE=/localdev/$USER/kimi-cache   # TTNN weight cache
export MESH_DEVICE=TG                           # TG, T3K, DUAL, or QUAD
```

## Hardware Readiness Check

Before running inference, verify your environment with the preflight script:

```bash
MESH_DEVICE=TG python models/demos/kimi_k25/scripts/kimi_preflight.py
```

The script checks Python version, `/dev/tenstorrent` devices, `MESH_DEVICE` topology,
`KIMI_HF_MODEL` directory, required imports, and `KimiK25Config` validation. On success
it prints the exact `pytest` commands to use.

```
[PASS] Python 3.10.12 ≥ 3.9 ✓
[PASS] /dev/tenstorrent: 4 device(s) found ✓
[PASS] MESH_DEVICE=TG ✓ (TG topology selected)
[PASS] KIMI_HF_MODEL=/mnt/MLPerf/.../Kimi-K2.5 — index file found ✓
[PASS] kimi_k25 imports ok ✓
[PASS] KimiK25Config validates: 61 layers, 384 experts ✓
[PASS] ttnn importable ✓

ALL CHECKS PASSED — Kimi K2.5 ready to run
```

## Running Tests

### CPU-only (no hardware, no weights)

```bash
# All CPU tests (189 tests across 9 files)
pytest models/demos/kimi_k25/tests \
  -k "not test_forward_pass and not test_full_model and not Hardware" \
  --timeout 60

# Individual test files
pytest models/demos/kimi_k25/tests/test_kimi_preflight.py   # preflight meta-tests (33)
pytest models/demos/kimi_k25/tests/test_kimi_generate.py    # full-model CPU tests
pytest models/demos/kimi_k25/tests/test_moe_gate.py         # MoE gate config tests
pytest models/demos/kimi_k25/tests/test_moe.py              # MoE CPU sanity tests
pytest models/demos/kimi_k25/tests/test_mla.py              # MLA config + CPU tests
pytest models/demos/kimi_k25/tests/test_weight_loader.py    # INT4 weight loader tests
pytest models/demos/kimi_k25/tests/test_int4_dequantize.py  # INT4 dequantize unit tests
pytest models/demos/kimi_k25/tests/test_kimi_model.py       # model adapter unit tests
pytest models/demos/kimi_k25/tests/test_conftest_hooks.py   # conftest meta-tests
```

### Hardware tests (TG — random weights, no checkpoint needed)

```bash
# Hardware gate: forward pass smoke test (decode + prefill)
MESH_DEVICE=TG pytest models/demos/kimi_k25/tests/test_kimi_generate.py \
  -k "test_forward_pass" --timeout 600 -v \
  2>&1 | tee /tmp/kimi_hw.log

# PCC correctness test (CPU vs TT numerical comparison, PCC ≥ 0.95)
MESH_DEVICE=TG pytest models/demos/kimi_k25/tests/test_kimi_generate.py \
  -k "test_pcc_correctness_random_weights" --timeout 900 -v
```

### Hardware tests (T3K alternative, 8-chip)

```bash
MESH_DEVICE=T3K pytest models/demos/kimi_k25/tests/test_kimi_generate.py \
  -k "test_forward_pass[mode_decode_seq_1_batch_32]" --timeout 600 -v
```

### Module-level hardware tests (real weights required)

```bash
KIMI_HF_MODEL=/mnt/MLPerf/.../Kimi-K2.5 MESH_DEVICE=TG \
pytest models/demos/kimi_k25/tests/test_moe_gate.py \
       models/demos/kimi_k25/tests/test_moe.py \
       models/demos/kimi_k25/tests/test_mla.py \
  --timeout 600 -v
```

## Running the Demo

```bash
MESH_DEVICE=TG KIMI_HF_MODEL=/mnt/MLPerf/.../Kimi-K2.5 \
  python models/demos/kimi_k25/demo/demo.py \
    --model-path $KIMI_HF_MODEL \
    --cache-dir $KIMI_CACHE \
    "Hello, Kimi!" "What is the capital of France?"

# Smoke test with random weights (no real checkpoint needed)
MESH_DEVICE=TG python models/demos/kimi_k25/demo/demo.py --random-weights \
  "Test prompt for smoke testing"

# Batch inference from prompts file
MESH_DEVICE=TG KIMI_HF_MODEL=... \
  python models/demos/kimi_k25/demo/demo.py \
    --model-path $KIMI_HF_MODEL \
    --prompts-file models/demos/kimi_k25/demo/test_prompts.json \
    --output-path /tmp/kimi_output.json
```

## Milestone Status

```
M1  Scaffold + config_adapter               ✅ DONE
M2  INT4 weight loader                      ✅ DONE
M3  Model adapter + MoE gate validation     ✅ DONE
M4  Single-layer MLA + MoE accuracy        ✅ DONE (CPU tests; HW pending)
M5  Full model on Galaxy (forward pass)     🔲 HW GATE — awaiting TG run
M6  CI yaml + demo script                   ✅ DONE (HW activation pending)
```

**Current blocker**: `test_forward_pass` on TG or T3K. All 189 CPU tests pass.
Use `kimi_preflight.py` to verify readiness, then run the M5 hardware gate command above.

## File Structure

```
models/demos/kimi_k25/
├── __init__.py
├── README.md
├── conftest.py                    # pytest fixtures (hf_config, hf_config_short, mesh_device)
├── ci/
│   └── galaxy-kimi-tests-impl.yaml  # CI workflow (unit/smoke/module/full)
├── demo/
│   ├── demo.py                    # CLI inference demo
│   ├── test_demo.py               # demo CI tests (smoke, tg_light, tg_full, throughput)
│   └── test_prompts.json          # 8 sample prompts for CI/manual testing
├── scripts/
│   └── kimi_preflight.py          # hardware readiness checker
├── tests/
│   ├── test_conftest_hooks.py     # conftest meta-tests (13)
│   ├── test_int4_dequantize.py    # INT4 dequantize unit tests
│   ├── test_kimi_generate.py      # full-model tests (28 CPU + 2 HW)
│   ├── test_kimi_model.py         # model adapter unit tests (12)
│   ├── test_kimi_preflight.py     # preflight meta-tests (33)
│   ├── test_mla.py                # MLA single-layer tests (19 CPU + 1 HW)
│   ├── test_moe.py                # MoE tests (4 CPU + 1 HW)
│   ├── test_moe_gate.py           # MoE gate tests (7 CPU + 5 CPU + HW)
│   └── test_weight_loader.py      # weight loader tests (10 unit + integration)
├── tt/
│   └── kimi_model.py              # KimiGenerator + load_kimi_model()
└── utils/
    ├── config_adapter.py          # KimiK25Config dataclass + from_hf_config()
    ├── int4_dequantize.py         # INT4 group-32 dequantizer
    └── weight_loader.py           # KimiLazyStateDict
```

## References

- [Kimi K2.5 paper](https://arxiv.org/abs/2602.02276)
- [HuggingFace model](https://huggingface.co/moonshotai/Kimi-K2.5)
- [DeepSeek V3 tt-metal demo](../deepseek_v3/)
- Research doc: `/workspace/group/research/kimi-k2-5-galaxy-port.md`
- First-failure triage guide: `projects/kimi/agent/NOTES.md`
