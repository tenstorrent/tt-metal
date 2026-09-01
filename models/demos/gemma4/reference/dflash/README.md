# Gemma4-31B DFlash reference (torch/HF, no ttnn)

Pure-torch reference for DFlash speculative decoding paired with Gemma4-31B, validated
against the real checkpoints before any ttnn porting starts on this branch
(`ign/gemma4_31B_MTP_Dflash`).

- Target (verifier): [`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it)
- Drafter: [`z-lab/gemma-4-31B-it-DFlash`](https://huggingface.co/z-lab/gemma-4-31B-it-DFlash)
- `dflash.py`: verbatim copy of `github.com/z-lab/dflash @ 07ebd93db9f472af339b644bb70221ad8428328a`,
  `dflash/model.py` (MIT). Generic — the same classes back Kimi-K2.6-DFlash in
  `models/demos/deepseek_v3_d_p/reference/dflash_prefill/` with different config values.
- `dflash_e2e_check.py`: our runner — loads both real checkpoints, runs the vendored
  `dflash_generate` draft/verify/accept loop on a real prompt, reports the generated text
  plus acceptance-length/timing stats.

## Required environment — separate venv, do NOT use the shared `tt-metal/python_env`

The upstream package pins `transformers==5.15.0` exactly (`dflash`'s own `pyproject.toml`).
This repo's shared `python_env` has `transformers==5.12.1`, which is missing
`DynamicCache.activate_past_recording()` — `dflash_generate` calls it unconditionally and
will crash with `AttributeError` on the older version. Rather than patch the vendored file
or the shared repo-wide `transformers` install (used by every other model here), this uses
an isolated venv pinned to the exact upstream-required version:

```bash
uv venv --python 3.10 /path/to/dflash_venv
uv pip install --python /path/to/dflash_venv/bin/python3 torch torchvision "transformers==5.15.0" accelerate
```

Install `torch`/`torchvision` together with `transformers` in the same command (as above) so
`uv` resolves a mutually compatible set — installing them separately, or inheriting a stray
`torchvision` from an unrelated global site-packages via `--system-site-packages`, produces a
torch/torchvision ABI mismatch (`RuntimeError: operator torchvision::nms does not exist`) at
import time, since `transformers` imports `torchvision` for its vision code paths (Gemma4 is
a VLM checkpoint) even for a text-only prompt.

No GPU exists in this environment (`torch.cuda.is_available() == False`) — the check runs on
CPU. The 31B target is already cached locally; only the drafter (1.5B params) downloads on
first run.

## Run

```bash
source /path/to/dflash_venv/bin/activate
cd /home/user/proj_sdk/tt-metal
python3 -m models.demos.gemma4.reference.dflash.dflash_e2e_check \
    --max-new-tokens 32 --prompt "Write a one-line Python function that reverses a string."
```

Expect prefill + each verify-block forward pass to take real wall-clock time (a dense
31B-parameter model on CPU) — this is a correctness smoke check, not a throughput benchmark.
Keep `--max-new-tokens` small.

## Verified result (2026-09-01, this environment)

```
Prompt (25 tokens): 'Write a one-line Python function that reverses a string.'
GENERATED (16 tokens in 34.7s, 0.46 tok/s overall):
```python
def reverse_string(s): return s[::-1]

time_to_first_token=20.33s  time_per_output_token=897.7ms
block_size=16  iterations=2  tokens/iteration: [10, 5]  avg=7.50
```

Correct output, and the measured 7.50 tokens/iteration average lines up with the checkpoint's
own published acceptance-length numbers (~8.0 on HumanEval-style prompts) — real evidence the
wiring (weights, context taps, sliding-window masking, logit softcapping, block verify/accept)
is faithful to the real checkpoint, not just "it didn't crash."
