# Alpamayo 2 Edge on Blackhole P100

Functional bring-up of NVIDIA [Alpamayo 2 Edge](https://huggingface.co/nvidia/Alpamayo2-Edge) on one
Blackhole P100. The model takes 22 frames from six cameras, a navigation instruction and the ego
history. It writes a short chain-of-causation (CoT) text and predicts 64 future waypoints (6.4 s).

On the device:

- The SigLIP2 vision tower (27 blocks) and patch-merger projector, composed from tt_dit vision modules.
- The 28-layer Cosmos3 Edge LM: prefill and token-by-token CoT decode with a KV cache.
- The 28-layer action expert, which attends over the LM cache, and the action in/out projections.

On the host: input preparation, token sampling, the Fourier features and the Euler update of the
10-step flow, and the conversion of actions to a trajectory. [`reference/host_ops.md`](reference/host_ops.md)
lists each host step with its call site and the reason it stays on the host.

```
reference/   pure-torch reference (the golden for every test), input preparation, host-op inventory
tt/          vision.py, text.py (LM + cache), generate.py (CoT loop), expert.py, action.py, model.py
tests/       host tests, pcc/ per block, e2e/ end to end
demo/        demo.py
```

## Setup

1. Build tt-metal and create `python_env` as described in the top-level [INSTALLING.md](../../../INSTALLING.md).
2. Download the weights (OpenMDW-1.1 license, about 5 GB) into your Hugging Face cache:

   ```bash
   export HF_HOME=/path/to/hf_data
   python -c 'from huggingface_hub import snapshot_download; snapshot_download("nvidia/Alpamayo2-Edge")'
   ```

   Every entry point resolves the checkpoint with `snapshot_download("nvidia/Alpamayo2-Edge")`.
   After the download, set `HF_HUB_OFFLINE=1` to skip the network check. The tested snapshot is
   `b92b6aa326bd3e2235db28f248ad5a30b5890e64`.

## Demo

Put 22 RGB frames in a directory, named `cam{id}_{t_ms}.png`: 3 frames each from cameras 0, 2, 3
and 5, and 5 frames each from cameras 1 and 6. `t_ms` is the capture time in milliseconds. Only the
times relative to the latest frame enter the prompt. Use images you are allowed to use. No images
ship with this package.

```bash
cd $TT_METAL_HOME
HF_HOME=/path/to/hf_data HF_HUB_OFFLINE=1 python models/experimental/alpamayo2_edge/demo/demo.py \
    --frames /path/to/frames --nav "Continue straight." --seed 0
```

The demo prints the CoT and the 64 xyz waypoints. They are in meters, in the ego frame at the
latest frame, 0.1 s apart. The ego history defaults to driving straight at `--speed` (10 m/s). For a
real history, pass `--ego-history history.npz` with `xyz` [20, 3] and `rot` [20, 3, 3].

On 22 uniform-noise frames, with `--seed 0` and "Continue straight.", the demo prints the CoT
`<NOMINAL>\n<|traj_future_start|>`. That is the same CoT the official stack samples on the same kind
of input. The predicted path runs nearly straight for 62.4 m.

> On the shared lab machines, every device run, including the demo and every pytest, goes through
> the `tt-device-mcp` job runner instead of a shell.

## Reduced depth

`ALPAMAYO_NUM_LAYERS=N` builds the vision tower, the LM and the expert with N layers each. The LM and
the expert always have the same depth. Use it for fast iteration; at N=2 the e2e file runs in about
3 minutes. Results are only claimed at full depth (27/28/28). The CoT at reduced depth is
meaningless text.

## Tests

The CPU reference goldens are cached on disk in `$ALPAMAYO_CACHE_DIR`, by default
`${XDG_CACHE_HOME:-~/.cache}/alpamayo2_edge`. Keep it private to your user. Entries are keyed on the
inputs, the checkpoint's `config.json` and weight index, the depths, the torch version, and the
sources of `reference/*.py` and of the test file that computes the golden. The first full-depth run
of a golden takes a few minutes of CPU time.

| File | Device | What |
|---|---|---|
| `tests/test_config.py`, `test_inputs.py`, `test_forced_cot.py` | no | config, input preparation, forced CoT |
| `tests/test_reference_vision.py`, `test_reference_text.py`, `test_reference_model.py` | no | the reference against upstream code and fixtures |
| `tests/test_generate_host.py` | no | CoT loop stop logic and host sampling |
| `tests/pcc/test_vision.py` | P100 | block 0, projector, tower at 16:9 and 4:3 |
| `tests/pcc/test_text.py` | P100 | ReLU² FFN, layer 0, prefill logits and K/V, cache sizing |
| `tests/pcc/test_decode.py` | P100 | 128-step teacher-forced decode |
| `tests/pcc/test_expert.py` | P100 | expert layer 0 and the 28-layer stack over the reference prefix |
| `tests/pcc/test_action.py` | P100 | in-proj, one-step velocity, 10-step flow |
| `tests/e2e/test_e2e.py` | P100 | full pipeline: teacher-forced, lm_head dtype measurement, free-running, two samples |

```bash
HF_HOME=/path/to/hf_data HF_HUB_OFFLINE=1 pytest -s models/experimental/alpamayo2_edge/tests/pcc/test_decode.py
# full-depth e2e: one test per invocation keeps each run at 2-5 minutes
HF_HOME=/path/to/hf_data HF_HUB_OFFLINE=1 pytest -s \
    "models/experimental/alpamayo2_edge/tests/e2e/test_e2e.py::test_e2e_teacher_forced"
```

The teacher-forced tests force the same 128-token CoT (`reference/forced_cot.py`: 127 tokens of
driving text and the anchor) and the same flow noise through the TT path and the reference. The
input is `synthetic_sample(0)`: 22 uniform-noise frames and a straight 10 m/s history, with S = 3256
prompt tokens and a 3384-row prefix for the expert. NVIDIA PhysicalAI-AV data is not used.

`reference/README.md` shows that the reference matches the official transformers + NVlabs stack
(worst 1 − PCC 3.3e-12, xyz within 3.8e-6 m).

## Measured accuracy (full depth, PCC against the fp32 reference)

Thresholds: PCC ≥ 0.99 on every block and on the trajectory.

| Block | Test | PCC |
|---|---|---|
| Vision block 0 | random input, two windows | 99.9984 % |
| Projector | random input | 99.9988 % |
| Vision tower + projector, 16:9 cameras | `synthetic_sample(0)`, 11568 patches | 99.9153 % |
| Vision tower + projector, 4:3 cameras | 11904 patches | 99.8807 % |
| LM ReLU² FFN / layer 0 (S = 3256) | | 99.9989 % / 99.9997 % |
| LM prefill, last-token logits | reference vision embeddings | 99.9742 % (argmax equal) |
| LM prefill K/V, layers 0 and 27 | | ≥ 99.9866 % |
| Teacher-forced decode logits, 128 steps | reference vision embeddings | min 99.7604 %, mean 99.9490 % |
| Expert layer 0 / layer-0 update alone | reference prefix cache | 99.9997 % / 99.8095 % |
| Expert, 28 layers | reference prefix cache | 99.8868 % |
| Action in-proj / velocity at step 0 | reference prefix cache | 99.9996 % / 99.9992 % |
| Actions after 10 Euler steps | reference prefix cache | 99.9848 % (xyz error ≤ 4.92 cm) |
| **End to end**, vision embeddings | TT path throughout | 99.9153 % |
| End to end, prefill last-token logits | | 99.9702 % |
| End to end, decode logits, 128 steps | | min 99.7546 %, mean 99.9484 % |
| End to end, actions | | 99.9681 % |
| **End to end, xyz** | | 99.999974 % |
| Two samples (B = 2), xyz per sample | own noise per sample | 99.999970 % / 99.999997 % |

Forward motion dominates the xyz PCC, so it barely separates trajectories; the position error
below is the meaningful trajectory metric.

**Vision (Decision 1):** the tower stays at or above 0.99 at both camera aspect ratios: 99.915 % at
16:9 and 99.881 % at 4:3. The fallback to the HunyuanImage-3 SigLIP2 block was not needed.

**Trajectory position error.** The max Euclidean error over the 64 waypoints, against the reference
with the same CoT and noise, always occurs at the last waypoint:

| Run | Max error |
|---|---|
| End to end, B = 1 | 6.11 cm |
| End to end, B = 2, samples 0 / 1 | 6.41 cm / 2.25 cm |
| Expert + flow only, over the reference prefix | 4.92 cm |

The proposed bound is **0.13 m**, about twice the worst measured error (6.4 cm), rounded. Until the
bound is recorded, the tests log the error and do not assert it.

**Decode quality and the lm_head dtype.** These numbers are from the 128-step teacher-forced run, end
to end. KL is KL(softmax(ref / 0.6) ‖ softmax(tt / 0.6)), averaged over the steps. CoT time is warm
and untraced: prefill plus 128 decode steps.

| lm_head weights | argmax agreement | decode logits PCC min / mean | KL mean / max (nats) | CoT time |
|---|---|---|---|---|
| bf8 (default) | 96.9 % (4 of 128 steps differ) | 99.7546 % / 99.9484 % | 3.10e-3 / 4.53e-2 | 7.66 s |
| bf16 | 96.9 % (4 of 128 steps differ) | 99.9825 % / 99.9930 % | 2.18e-3 / 2.92e-2 | 7.79 s (+1.7 %), +0.26 GB DRAM |

The steps that differ are near-ties in the reference: its top-1 and top-2 logits are 0.04 to 0.05
apart (0.27 at one bf8 step). The logits come back from the device in bf16, and at these logit
magnitudes a bf16 step (0.03 to 0.06) is as large as those margins.

- With the bf16 head, three of the four differences are exact ties in the bf16 logits, which the argmax
  tie-break settles against the reference.
- With the bf8 head, all four are real wrong picks: the TT logit of the reference's token is 0.03 to
  0.19 below the TT argmax.

The bf16 head therefore improves the logits but not the agreement. Keeping bf8 is a cost/benefit call:
bf16 costs 1.7 % CoT time and 0.26 GB of DRAM. Free-running CoTs can drift from the reference's at
near-tie steps, and sampling at temperature 0.6 makes the CoT stochastic anyway. Reading the logits back
in fp32, or taking the argmax on the device, is the next lever to measure.

**Timing (not optimized).** Full depth, warm, one sample, teacher-forced 128-token CoT: vision 0.43 s,
CoT 7.66 s, flow 0.66 s, 8.7 s in total. Decode runs untraced, at about 60 ms per step including the
logits readback and host sampling.

## Known limits

- **Batch:** B = 1 is tested end to end, plus a two-sample smoke test (cache repeat, two-row decode,
  two-row expert). The tt_dit decode supports B < 8 or a multiple of 8. One prompt per call.
- **Host round trips:** one logits readback per decode step, and two transfers per flow step for the
  Fourier features and the velocity (`reference/host_ops.md`).
- **Decode is untraced.** No performance work has been done yet.
- **Inputs are synthetic only:** uniform-noise frames and a straight history. Driving quality (minADE)
  is not measured. Which permissively licensed driving dataset to use for it is still open.
- **Weights:** the vision tower and projector linears are bf16 (the tt_dit `Linear` default). The LM
  linears are bf8 (the tt_dit encoder default). The expert and the action projections are bf16.
