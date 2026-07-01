# NVIDIA Nemotron-3-Nano-30B-A3B-BF16 on Tenstorrent Hardware

## Platforms

| Device | Status | Notes |
|---|---|---|
| BH (Blackhole), single chip | Supported | `ttnn.open_device(device_id=0)`, single device, `l1_small_size=24576`; per-layer weight streaming (30B does not fit on device at once); greedy text generation with compose (graduated child stubs) or monolith backbone |

This port targets a **single Blackhole device**. It is **not** a multi-chip / mesh
(TP) port: `num_key_value_heads = 2` is not divisible by TP 4/8/32, so a
tensor-parallel KV split across a mesh is a hard blocker (see Known Limitations /
`kernel_findings.json`). Tensor-parallel = 1 (single chip) is the supported, tested
configuration. There is no Wormhole, Grayskull, or mesh path in the
supported/tested configuration.

The model has ~30 B total parameters and does not fit on device at once, so
per-layer weights are streamed from host and evicted after each layer (peak
device residency ≈ one layer). The residual stream is carried in fp32; matmuls use
HiFi4 + `fp32_dest_acc`.

## Introduction

`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (`NemotronHForCausalLM`) is a 52-layer
hybrid **Mamba2 / GQA-attention / Mixture-of-Experts** causal language model:
~30 B total parameters with ~3 B active per token (MoE top-6 of 128 + 1 shared
expert). This port runs a greedy HuggingFace-style `generate()` pipeline (real
prompt → tokenizer → chained TTNN stubs → greedy decode → text) on Tenstorrent
hardware via TTNN, compared against the HF reference `NemotronHForCausalLM.generate()`.

Weights are loaded in bfloat16; the full backbone math runs on device. The decode
loop control (token selection, EOS check, sequence bookkeeping) runs on the host in
Python/PyTorch — see Known Limitations.

## Model Architecture

A single `NemotronHForCausalLM` backbone of 52 layers. Each layer is one of three
kinds, fixed by the config's `hybrid_override_pattern`:

```
input_ids ─► nemotron_h_model (backbone driver: embedding, RMSNorm, residual, attention helper)
              └─ per layer (52), pattern MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME:
                   M-layer (×23) ─► nemotron_h_block            (full Mamba2, 1st M-layer)        ─┐
                   M-layer       ─► nemotron_h_mamba2_mixer  ─► mamba_r_m_s_norm_gated             │ Mamba2
                   E-layer (×23) ─► nemotron_h_m_o_e         ─► nemotron_h_topk_router             │ MoE (128 experts, top-6 + shared)
                                                            └► re_l_u_squared_activation           │
                   *-layer (×6)  ─► GQA attention (REUSE, no RoPE)                                 ┘
            ─► final RMSNorm ─► lm_head (untied) ─► next-token logits ─► argmax
```

- **52 layers** total: **23 Mamba2** (`M`), **23 MoE** (`E`), **6 attention** (`*`).
- **Mamba2 layers** are state-space (SSD) mixers; the first M-layer uses the full
  `nemotron_h_block`, the rest use `nemotron_h_mamba2_mixer` (+ gated grouped RMSNorm).
- **MoE layers** route each token to the top-6 of 128 routed experts plus 1 shared
  expert, with a `relu(x)²` activation.
- **Attention layers** are GQA (32 query heads, 2 KV heads) and use **no RoPE**
  (`rope_theta` is present in config but vestigial — applying it drops PCC). Attention
  is REUSE (not a synthesized work product).

## Key Model Parameters (`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`)

| Parameter | Value |
|---|---|
| Architecture | `NemotronHForCausalLM` |
| Total layers | 52 (23 Mamba2 / 23 MoE / 6 attention) |
| Hidden size | 2688 |
| Attention heads (Q / KV) | 32 / 2 (GQA) |
| Mamba SSM heads / state size | 64 / 128 |
| MoE routed experts / shared / top-k | 128 / 1 / 6 |
| MoE intermediate size | 1856 |
| MLP activation | `relu2` (`relu(x)²`) |
| Vocabulary size | 131072 |
| Tied embeddings | No (untied `lm_head`) |
| RoPE | Not applied (vestigial `rope_theta=10000`) |
| EOS token id | 2 |
| Weight precision | bfloat16 |
| Total parameters | ~30 B (~3 B active per token) |

## Graduated Modules

All seven modules are native TTNN stubs, PCC-verified against captured HF golden,
and all are invoked in a composed run (Gate 2).

| Module | Role |
|---|---|
| `nemotron_h_model` | backbone driver (embedding, RMSNorm, attention helper, residual scaffold) |
| `nemotron_h_block` | full Mamba2 layer (used for the first M-layer) |
| `nemotron_h_mamba2_mixer` | Mamba2 SSM mixer (remaining M-layers) |
| `mamba_r_m_s_norm_gated` | gated grouped RMSNorm inside the Mamba mixer |
| `nemotron_h_m_o_e` | MoE mixer (E-layers) |
| `nemotron_h_topk_router` | top-6-of-128 router inside the MoE |
| `re_l_u_squared_activation` | `relu(x)²` expert activation inside the MoE |

## How to Run

Run from the tt-metal root directory.

### Demo (on device)

Real prompt → TTNN pipeline → greedy decode → generated text:

```bash
./python_env/bin/python -m models.demos.nvidia_nemotron_3_nano_30b_a3b_bf16.demo.demo_text_generation \
    --prompt "The capital of France is" --max-new-tokens 5
```

Flags: `--prompt`, `--max-new-tokens`, `--compose {0,1}` (1 = compose graduated
child stubs, default; 0 = monolith backbone), `--device-id`.

Example output:

```
PROMPT     : 'The capital of France is'
GENERATED  : ' Paris.' ...
NEW_IDS    : [6993, 2613, 3501, 7185, 34315]
```

### End-to-end test (Gate 1 / 2 / 3 vs HF golden)

```bash
TT_E2E_COMPOSE=1 TT_E2E_N=5 ./python_env/bin/python -m pytest \
    models/demos/nvidia_nemotron_3_nano_30b_a3b_bf16/tests/e2e/test_e2e_pipeline.py -s
```

Environment: `TT_E2E_COMPOSE=1` composes the graduated children (Gate 2); `0` runs
the monolith backbone. `TT_E2E_N` sets the generation horizon (both sides capped to
the same N).

### Regenerate the HF golden (only if the prompt or N changes)

```bash
TT_E2E_PROMPT="The capital of France is" TT_E2E_N=5 ./python_env/bin/python -m \
    models.demos.nvidia_nemotron_3_nano_30b_a3b_bf16.tests.e2e.make_golden
```

### Per-module PCC tests

```bash
./python_env/bin/python -m pytest models/demos/nvidia_nemotron_3_nano_30b_a3b_bf16/tests/pcc/ -v
```

## Correctness Gates (prompt = "The capital of France is", N = 5)

The e2e test enforces four checks; all must pass.

| Gate | Result |
|---|---|
| Gate 1 — no torch runtime fallback (everything ran on device) | PASS (`fallbacks=[]`) |
| Gate 2 — all 7 graduated modules invoked (compose) | PASS (`missing=[]`) |
| Gate 3 — mean per-step next-token logits PCC ≥ 0.95 vs HF | **0.9983** |
| Behavioral — greedy token match vs HF | exact: `[6993, 2613, 3501, 7185, 34315]` |

Per-component PCC (vs captured HF golden, target 0.99): `nemotron_h_block` 0.99999,
`nemotron_h_mamba2_mixer` 0.99999, `nemotron_h_m_o_e` 0.99998, `nemotron_h_topk_router`
1.0, `mamba_r_m_s_norm_gated` 0.99999, `re_l_u_squared_activation` 1.0; the backbone
path is validated through the e2e logits PCC (0.9983).

Gate 2 is proven by an execution registry (`tt/_invocation.py`) — each child is
recorded when it actually runs, not by the caller's optimism.

## Performance

**Not yet measured.** No throughput / latency numbers (TTFT, prefill, decode
tokens/s, end-to-end wall time) are certified for this port yet. A bounded
device-time perf workload exists at `tests/e2e/test_perf.py` (it drives
`tt/pipeline.py` so `perf_automation` / Tracy can attribute per-op device time), but
it has not been run to produce a published performance table. This section will be
filled with measured numbers once that sweep is run; until then, no figures are
claimed.

```bash
# device-time perf workload (measurement harness, not a published table yet)
./python_env/bin/python -m pytest models/demos/nvidia_nemotron_3_nano_30b_a3b_bf16/tests/e2e/test_perf.py -s
```

## Repository Layout

```
models/demos/nvidia_nemotron_3_nano_30b_a3b_bf16/
├── demo/
│   └── demo_text_generation.py    # runnable demo (argparse + __main__)
├── tt/
│   ├── pipeline.py                # the ONE shared chained forward (demo + test import this)
│   ├── _hf_compat.py              # mamba-ssm / cuda-stream shims for CPU HF load
│   └── _invocation.py             # execution registry proving which stubs ran (Gate 2)
├── tests/
│   ├── e2e/
│   │   ├── test_e2e_pipeline.py   # e2e gates (Gate 1/2/3) vs HF golden
│   │   ├── make_golden.py         # regenerates the HF reference golden
│   │   └── test_perf.py           # bounded device-time perf workload (Tracy/perf_automation)
│   └── pcc/                       # per-module PCC ≥ 0.99 tests
├── _stubs/                        # graduated native TTNN module implementations
├── e2e_plan.json                  # planner output (task head, gates, metric)
├── bringup_status.json            # per-component bring-up status
├── kernel_findings.json           # TTNN kernel constraints found during bring-up
└── README.md
```

## Known Limitations

### Hardware / deployment

- **Single Blackhole chip only (TP = 1).** A tensor-parallel mesh split is blocked:
  `num_key_value_heads = 2` is not divisible by TP 4/8/32, and KV replication is not
  supported in `tt_transformers` (`kernel_findings.json`). The supported config is a
  single device.
- **30 B does not fit on device.** Weights are streamed per layer and evicted after
  each layer; peak device residency is roughly one layer's weights.

### Kernel constraints (`kernel_findings.json`)

- **`ttnn.topk` runs single-core.** Multi-core top-k requires the vocab to be a
  power of two and < 65536; the vocab is 131072, so the kernel falls back to the
  single-core path (lower decode throughput). No action needed — correctness is
  unaffected.

### `generate()` host work

`generate()` is **not** a fully device-resident autoregressive loop. The backbone
(embedding, all 52 layers, final norm, `lm_head`) runs on device, but the decode
loop control runs on the host in Python/PyTorch:

| Step | Where | Notes |
|---|---|---|
| Decode loop control | Host (Python) | one iteration per output token; EOS checked on host |
| Greedy token pick | Host | `argmax` over the logits row read back from device |
| Sequence bookkeeping | Host (Python) | token ids accumulated in Python lists; tensors rebuilt per step |
| Tokenization / detokenization | Host | HF tokenizer before/after the device path |

### Bring-up / tooling note

This demo was produced by the `tt_hw_planner` `emit-e2e` composition step. A prior
committed "(working)" checkpoint silently failed Gate 2 because the compose pipeline
recorded child invocations into a local set while the test read the global
`_invocation` registry, and a later perf-optimization checkpoint committed the model
without re-running the compose Gate-2 test. The pipeline now records every child
into the global registry, and Gate 2 passes. When checkpointing this model, always
re-run the compose e2e gate (`TT_E2E_COMPOSE=1`) on the exact files being committed.

## References

- [NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 on HuggingFace](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)
- HuggingFace Transformers `modeling_nemotron_h.py` (remote code in the model repo)
- [Tenstorrent TT-Metalium / TT-NN](https://github.com/tenstorrent/tt-metal)
