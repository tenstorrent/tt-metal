# MiniMax-H3 VAE bringup — handoff

Paste the block below as the first message in a fresh session on the new machine.

---

## Handoff prompt

> Bring up the MiniMax-H3 VAEs in `models/tt_dit` on a 4x8 Blackhole Galaxy: the
> visual VAE (encode and decode) and the audio VAE (encode and decode), each with
> comprehensive unit tests and measured performance. VAEs only — the denoising
> transformer and text encoder are explicitly out of scope.
>
> The work lives on branch **`kevinmi/minimax-h3-vae`** in `tenstorrent/tt-metal`.
> **It is not in your local clone — fetch it first**, then check it out. Commit and
> push there only, never to `main` or another branch:
>
> ```bash
> git fetch origin kevinmi/minimax-h3-vae
> git checkout kevinmi/minimax-h3-vae
> ```
>
> It is based on `origin/cglagovich/minimax-h3` (`42ecb2e0339`), which owns the
> canonical folder structure — fetch that too if you need to diff against it. Any PR
> targets `kevinmi/minimax-h3-vae`.
>
> **Use the `gh` CLI for everything GitHub-side** — PRs, issues, reading the
> reference PRs (`gh pr view`, `gh pr diff`, `gh api`). Do not scrape the web UI.
>
> **The weights are already on this machine at `/data/cglavioch/minmax-h3`** — do not
> re-download the 135 GiB. If that exact path is not there, glob for it before
> downloading anything:
>
> ```bash
> ls /data/*/minmax-h3 /data/*/minimax-h3 2>/dev/null
> ```
>
> (The path was given from memory and its spelling differs from the branch owner's
> `cglagovich`, so it may be slightly off.) You need only `FL2VA/video_vae` and
> `FL2VA/audio_vae` for VAE work.
>
> Start by reading, in this order:
>
> 1. `models/tt_dit/models/MiniMaxH3_VAE_PLAN.md` — the plan. Source of truth for
>    scope, architecture, gates and the test plan.
> 2. `STATE.md` — execution state: what already passes, every measurement taken,
>    and the amendments where a measurement contradicted an assumption.
> 3. `models/tt_dit/models/MiniMaxH3.md` — the canonical folder structure and the
>    pinned diffusers reference commit. Conform to it; do not invent a layout.
>
> Re-read the plan and `STATE.md` before every iteration, and append to `STATE.md`
> as you go: current milestone, gate evidence, resets, failed attempts, next step.
> If a measurement contradicts the plan, append a dated amendment with the evidence
> rather than silently diverging.
>
> You are on a machine that drives the device directly from bash — no broker, no job
> queue. Run pytest directly. **You have standing permission to run
> `tt-smi -glx_reset` yourself at any time without asking**, including proactively
> when mesh state is in doubt. `tt-smi -r` is forbidden. Details in the plan's §6.
>
> Work one component at a time and do not move on until its gate passes with
> evidence saved. Correctness before performance.

---

## What is already on this branch

Base: `origin/cglagovich/minimax-h3` (`42ecb2e0339`), which carries the canonical
folder structure and `MiniMaxH3.md`.

**Green, VAE-relevant:**

- `pipelines/minimax_h3/packing.py` — packed-sequence geometry, bit-exact vs the
  diffusers reference (fp64 rotary grid, `17n+5 -> 5n+2`, canvas solver).
- `pipelines/minimax_h3/conditioning.py` — the keyframe encode recipe: ImageNet
  pixel normalization, seed-42 *sampled* posterior, the load-bearing float16 round
  trip, and the request-generator draw order.
- `pipelines/minimax_h3/scheduler.py` — rectified-flow Euler, bit-exact including
  full 49-evaluation rollouts at both shifts.
- `pipelines/minimax_h3/adaln_precompute.py` — parked (DiT-side) but green.
- Tests for all of the above: **84 passed, 3 skipped** on host, no device needed.

**WIP, the immediate next thing:**

- `models/vae/minimax_h3/vae_minimax_h3.py` — the T=1 keyframe encoder, built on
  `Conv2dViaConv3d`. Its host-side gate passes (5 tests prove the single-frame
  collapse and that it does not compound). **The device half is unvalidated** — the
  first run failed only because the single-device test asked for a fabric ring; that
  is fixed but the encoder itself has never run on device.

**Parked, green, not in scope:** the DiT module tree and its weight-load gate
(4 passed on the real 4x8 mesh) live on `kevinmi/minimax-h3`.

## First moves on the new machine

```bash
# 1. Install the pinned reference. Note the uv caveat in MiniMaxH3.md: a bare
#    pip install inside a uv venv silently installs to ~/.local and has no effect.
uv pip install --python <workspace>/tt-metal/python_env/bin/python --no-deps \
  "diffusers @ git+https://github.com/huggingface/diffusers@abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc"
python -c "from diffusers import AutoencoderKLMiniMaxH3; print('ok')"
python -c "import ttnn; print('ttnn still ok')"

# 2. Confirm the host suite still passes — no device required.
./python_env/bin/python -m pytest models/tt_dit/tests/models/minimax_h3/ -q --no-header

# 3. Then the first device gate: the T=1 encoder.
./python_env/bin/python -m pytest \
  models/tt_dit/tests/models/minimax_h3/test_vae_encoder_minimax_h3.py -q --no-header
```

## Weights

**Already on the machine at `/data/cglavioch/minmax-h3`.** Do not re-download the
135 GiB. Verify and point the tests at it:

```bash
ls /data/cglavioch/minmax-h3
# if absent, the spelling may be off -- glob before downloading anything
ls -d /data/*/min*max-h3 2>/dev/null
```

VAE work needs only `FL2VA/video_vae` (9.8 G) and `FL2VA/audio_vae` (578 M). The
62 G transformer and 63 G text encoder are for the parked DiT work.

The `source/` subdirectory is the one that matters: `FL2VA/video_vae/config.json`
hides the real architecture behind `source_path`, and
`FL2VA/video_vae/source/config.json` is the authority (see the plan's §1).

Only as a last resort, if no local copy exists:

```bash
hf download MiniMaxAI/MiniMax-H3 --include "FL2VA/video_vae/*" "FL2VA/audio_vae/*"
```

## Traps already paid for

Each of these cost a failed run or a wrong assumption; the plan and `STATE.md` have
the full detail.

| Trap | What happens |
|---|---|
| `device_params=ring_params` on a 1-device mesh | ethernet handshake times out before any kernel runs |
| piping a device run to `tail -N` | log stays empty until exit; looks like a hang |
| bare `python` | cannot import `ttnn` |
| `pre-commit` not on `PATH` | commit aborts; also black/isort reformat then abort the first attempt |
| tt_dit `RMSNorm` defaults `bias=True` | every H3 norm is weight-only, so all need `bias=False` |
| `Module` is an ABC | a parameter-only container still has to declare `forward` |
| comparing fp32 tensors to Python float literals | `0.7` is not representable; compare tensor to tensor |
| `neighbor_pad_async` has no `reflect` | H3 pads reflect; only the two global image edges differ from replicate |
