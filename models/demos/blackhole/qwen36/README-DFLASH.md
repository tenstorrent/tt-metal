# Qwen3.6-27B DFlash speculative decoding

[DFlash](https://github.com/z-lab/dflash) ([arXiv:2602.06036](https://arxiv.org/abs/2602.06036))
speculative decoding for Qwen3.6-27B on a Wormhole T3K. A 5-layer block-diffusion drafter proposes
15 tokens in one forward from the target's residual stream; the target verifies all 16 slots in one
forward and commits the longest matching prefix plus one bonus token. Greedy verification accepts
only the target's own argmax, so the output is the target's own greedy output; the drafter only
changes speed. Companion to [README.md](README.md).

| Role | Checkpoint | Params | Env var |
| --- | --- | --- | --- |
| Target | `Qwen/Qwen3.6-27B` | 27B, 64 layers | `HF_MODEL` |
| Drafter | `z-lab/Qwen3.6-27B-DFlash` | 1.73B, 5 layers | `DFLASH_HF_MODEL` |

## Supported devices

| Device | Mesh (`MESH_DEVICE`) | Target | Drafter | Status |
| --- | --- | --- | --- | --- |
| Wormhole T3K (4x N300, 8 chips) | `T3K`, `(1, 8)` | 8-way tensor parallel | replicated on all 8 chips | validated |

## Architecture

```mermaid
flowchart LR
    P[prompt] --> T0[TtTarget chunked prefill<br/>taps kept per chunk]
    T0 -->|first token| L{speculative step}
    subgraph target [Qwen3.6-27B target - 64 layers, TP=8]
        V[verify trace, 16 rows<br/>decode-mode attention, recurrent GDN] --> H[LM head + argmax<br/>in the trace]
        V -->|residual taps<br/>layers 1,16,31,46,61| TAP[taps]
        C[commit trace<br/>keep accepted GDN slot]
    end
    subgraph drafter [DFlash drafter - 5 layers, replicated, traced]
        FC[all-gather taps<br/>fc + hidden_norm] --> DL[4 sliding + 1 full<br/>attention layers]
        DL --> DH[target LM head<br/>+ argmax in the trace]
    end
    L --> FC
    TAP --> FC
    DH -->|15 draft ids| V
    H -->|accepted prefix + bonus token| C
    C --> L
```

| Module | Class | File |
| --- | --- | --- |
| Speculative loop (backend-agnostic) | `dflash_generate` | [reference/dflash/generate.py](reference/dflash/generate.py) |
| Device target: chunked prefill, 16-row verify, GDN slot commit | `TtTarget` | [tt/dflash/target.py](tt/dflash/target.py) |
| Target additions: residual taps, verify buffers and trace | `Qwen36Model` | [tt/model.py](tt/model.py) |
| Device drafter: tap projection, 5 layers | `TtDFlashDrafter` | [tt/dflash/drafter.py](tt/dflash/drafter.py) |
| Drafter adapter for the loop, traced draft step | `TtDrafter` | [tt/dflash/speculative_drafter.py](tt/dflash/speculative_drafter.py) |
| Drafter weights, dtypes | `load_drafter_weights` | [tt/dflash/weights.py](tt/dflash/weights.py) |
| Config, checkpoint resolution, runtime sizing | `DFlashDrafterConfig`, `paged_blocks_for` | [tt/dflash/config.py](tt/dflash/config.py) |
| Host reference drafter (vendored from z-lab, MIT) | `DFlashDraftModel`, `HostDrafter` | [reference/dflash/dflash.py](reference/dflash/dflash.py), [reference/dflash/drafters.py](reference/dflash/drafters.py) |
| Host reference target | `HFTarget` | [reference/dflash/targets.py](reference/dflash/targets.py) |
| Demo | `test_demo_dflash` | [demo/dflash_demo.py](demo/dflash_demo.py) |

Per-op device mapping of one drafter step: [DFLASH_DRAFTER_OP_MAPPING.md](DFLASH_DRAFTER_OP_MAPPING.md).

### Deviations from the reference

None of these changes the greedy output (asserted by the tests).

| Area | Reference (z-lab / HF) | This implementation | Why |
| --- | --- | --- | --- |
| Drafter precision | bf16 | `BFLOAT8_B` projections, `BFLOAT4_B` MLP gate/up, bf16 norms | weight-bandwidth bound |
| Drafter parallelism | one device | replicated on all 8 chips | bound by collectives and dispatch |
| Drafter KV | growing cache | fixed-capacity buffer; sliding layers keep a 2080-row ring | stable addresses under the parked traces |
| Target rollback | `cache.crop()` (a no-op on GDN layers) | the verify stashes the GDN state after every slot; `commit` keeps the accepted one | GDN state cannot be truncated |
| Verify LM head | full logits | 16 rows, argmax in the trace | only the block's rows are needed |

## Running

```bash
export DFLASH_RUN_TARGET=1 MESH_DEVICE=T3K
export HF_MODEL=Qwen/Qwen3.6-27B DFLASH_HF_MODEL=z-lab/Qwen3.6-27B-DFlash
export TT_CACHE_PATH=$HOME/.cache/tt_cache/Qwen3.6-27B

pytest models/demos/blackhole/qwen36/demo/dflash_demo.py -v -s                          # all cases
pytest models/demos/blackhole/qwen36/demo/dflash_demo.py -v -s -k "spec_128 and not long"  # one case
```

| Parameter | Meaning | Cases |
| --- | --- | --- |
| `seqlen` | prompt length (ISL) | `spec_128`, `spec_128_long` (128); `isl_512` ... `isl_64k` (512 - 65536) |
| `max_generated_tokens` | tokens to generate | 100 (`spec_128_long`: 256) |
| `DFLASH_PROMPT` | literal prompt instead of the sample prompt | optional |
| `DFLASH_NUM_SPEC` | drafted tokens per step, 1 - 15 (block of n + 1 slots) | optional |
| `DFLASH_TRACE_DRAFTER=0` | run the draft step eagerly instead of traced | optional |

ISL 128 uses the shared question prompt `text_demo.py` uses; longer ISLs use a Frankenstein excerpt
with a request to summarise it. The demo runs one eager and one traced warm-up generation (verify, commit and draft step captured between them), then
times a traced generation and reports TTFT, decode TPS, acceptance and the output. It asserts the
traced run emits exactly the eager run's tokens and applies `text_demo.py`'s output-quality check.

Example (`spec_128`, greedy): the input is the ISL-128 condiment question, clipped to 128 tokens so it
ends mid-sentence ("... There are so many condiments to choose from, each"). Expected output begins:

> bringing its unique flavor and texture to enhance different dishes. Do you prefer the classic taste
> of ketchup, the creamy richness of mayonnaise, the perfect balance of sweet and spicy in sriracha,
> or perhaps something more exotic like hoisin sauce? [...] \<think\> Here's a thinking process: [...]

## Tests

| Test | What it checks |
| --- | --- |
| [tests/test_dflash_drafter_tp.py](tests/test_dflash_drafter_tp.py) | ttnn drafter vs host drafter PCC, per module, real weights |
| [tests/test_dflash_verify_pcc.py](tests/test_dflash_verify_pcc.py) | full-depth (64-layer) verify-path logits PCC vs HuggingFace, every row |
| [tests/reference/test_dflash_device.py](tests/reference/test_dflash_device.py) | device target taps and rollback; full 27B speculation == autoregressive (`DFLASH_RUN_TARGET=1`) |
| [tests/reference/test_dflash_host.py](tests/reference/test_dflash_host.py) | host reference: speculation == autoregressive greedy |
| [tests/perf/test_dflash_traced_throughput.py](tests/perf/test_dflash_traced_throughput.py) | traced vs eager verify: identical tokens, tok/s |
| [tests/perf/test_profile_dflash_drafter.py](tests/perf/test_profile_dflash_drafter.py) | one drafter step under Tracy (per-op CSV) |
| [tests/perf/test_profile_dflash_verify.py](tests/perf/test_profile_dflash_verify.py) | one verify forward (3 GDN + 1 attention layers) under Tracy (per-op CSV) |

```bash
export MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen3.6-27B DFLASH_HF_MODEL=z-lab/Qwen3.6-27B-DFlash
T=models/demos/blackhole/qwen36/tests
pytest -svq $T/test_dflash_drafter_tp.py
pytest -svq $T/test_dflash_verify_pcc.py                              # HF reference on CPU, ~60 GB RAM
DFLASH_RUN_TARGET=1 pytest -svq $T/reference/test_dflash_device.py
DFLASH_RUN_TARGET=1 pytest -svq $T/perf/test_dflash_traced_throughput.py
pytest $T/reference/test_dflash_host.py -sv                           # host only, no device
python -m tracy -p --op-support-count 100000 -r -v -m pytest $T/perf/test_profile_dflash_drafter.py
python -m tracy -p --op-support-count 100000 -r -v -m pytest $T/perf/test_profile_dflash_verify.py
```

## PCC

T3K, real weights.

| Test | Scope | PCC | Gate |
| --- | --- | --- | --- |
| `test_dflash_drafter_tp` | `fc` + `hidden_norm` tap projection | 0.99990 | 0.99 |
| | one sliding-window layer / one full-attention layer | 0.99608 / 0.99609 | 0.99 |
| | full 5-layer drafter / second step with carried context | 0.99433 / 0.99504 | 0.99 |
| `test_dflash_verify_pcc` | prompt, last row (position 149) | 0.9951 | 0.95 |
| | verify block, positions 150-165 | 0.9566 | 0.95 |

**Exception:** the verify test gates at 0.95, the full-depth decode test's gate, not the 0.98
prefill gate. A few rows where the reference puts almost all probability on one token score low on
full-vocab PCC while the argmax agrees (100% on the block), so they reflect the target's precision.

## Performance

T3K, batch 1, greedy, `demo/dflash_demo.py`, on `ign/mtp_qwen3.6` `5b85ed62c08`. TTFT: call to first
token. Decode TPS: rate after the first token. "vs production" compares with production traced
decode on the same build (`text_demo.py -k "traced_128 and not 128k"`, ISL 128, batch 1):
17.03 tok/s.

| ISL (new tokens) | TTFT | Decode TPS | vs production | Acceptance (tok/step) |
| --- | --- | --- | --- | --- |
| 128 (100) | 0.51 s | 51.22 | 3.01x | 6.19 |
| 128 (256) | 0.52 s | 48.27 | 2.83x | 5.80 |
| 512 (100) | 0.50 s | 36.92 | 2.17x | 4.50 |
| 1k (100) | 0.51 s | 42.73 | 2.51x | 5.21 |
| 2k (100) | 0.58 s | 40.45 | 2.38x | 5.82 |
| 3k (100) | 1.02 s | 43.82 | 2.57x | 5.82 |
| 4k (100) | 1.19 s | 38.22 | 2.24x | 5.21 |

Single runs; expect up to ~15% run-to-run variation in decode TPS. Decode TPS tracks acceptance,
which depends on the generated text. Compile and warm-up are excluded. ISLs up to 64k run; the
drafter's history capacity is sized per request.

## Dependencies

| Component | Version |
| --- | --- |
| tt-metal / TTNN | branch `dflash-on-mtp` on `ign/mtp_qwen3.6` `5b85ed62c08`; uses that branch's `sdpa_decode` spec-verify mode and fused recurrent GDN op; no C++ changes of its own |
| Target checkpoint | `Qwen/Qwen3.6-27B` @ `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` |
| Drafter checkpoint | `z-lab/Qwen3.6-27B-DFlash` @ `0919688658996800f86b895034249700e9481106` |
| transformers / torch | 5.12.1 / 2.11.0+cpu |
| Firmware / KMD | fw bundle 19.2.0.0 / tt-kmd 2.7.0-rc1 |
