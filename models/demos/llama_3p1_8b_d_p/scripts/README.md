<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Golden-KV scripts

Host-side tooling for the golden per-layer KV cache the device prefill is graded against
([tt-blaze#4147](https://github.com/tenstorrent/tt-blaze/issues/4147)), mirroring
`models/demos/gpt_oss_d_p/scripts/`.

| script | needs | what it does |
| --- | --- | --- |
| `generate_golden_kv_cache.py` | the checkpoint, CPU only | runs the torch reference and saves post-RoPE K and raw V per layer |
| `verify_golden_kv.py` | the trace dir only | sanity-checks a generated trace before anything is graded against it |

## Generating

```bash
python3 models/demos/llama_3p1_8b_d_p/scripts/generate_golden_kv_cache.py \
    --prompt "The capital of France is" --out /data/$USER/llama31_golden_short
python3 models/demos/llama_3p1_8b_d_p/scripts/verify_golden_kv.py /data/$USER/llama31_golden_short
```

A trace dir carries **both** the token IDs and the golden KV, so a per-slot trace gives a per-slot
prompt and a per-slot golden — which is what the Gate 2 cross-talk check needs.

## Two things that make a broken prefill look correct

**Frame.** The golden stores K in the **Meta-interleaved** frame (`--frame meta`, the default). The
reference computes in the HF frame because that is what makes it comparable to HF logits, but blaze
decode *writes* the Meta-interleaved frame, and the golden has to match what decode writes rather
than what prefill happens to do. An HF-frame golden agrees with a prefill that has the same frame
error — the two cancel — so it passes a device-vs-golden comparison while decode reads a permuted
cache. `--frame hf` exists only for debugging that divergence.

**Dtype.** The device cache is `bfloat8_b`, and the golden is stored in `bfloat16` (bf8_b has no
torch equivalent), so **the consumer must round-trip the golden through bf8_b before computing
PCC**. A full-precision golden leaves a spurious ~0.94–0.96 gap that reads as a real bug.
`verify_golden_kv.py` reports that bf16-vs-bf8 self-PCC up front, so the size of the effect is
visible before it gets mistaken for one.
