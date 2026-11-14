# MoE expert placement

Traffic-calibrated, lossless expert placement for expert-parallel MoE models on Tenstorrent meshes.

A placement moves routed experts between the chips of one dispatch group (mesh column) so that the busiest chip
gets less routed-expert work. Weights, top-k selection and outputs are unchanged; only which chip runs which
expert moves. This tool calibrates one placement per MoE layer on a diverse, multilingual request corpus,
scores it on held-out requests, and decides with an on-device A/B whether to ship it.

No model weights are trained or changed. "Calibration" here means choosing the permutation.

## When it helps

Placement removes per-chip imbalance of routed-expert work. It only matters when:

1. **Experts are expert-parallel:** each chip owns whole experts and tokens are dispatched to them. Models
   that split every expert across chips (tensor-parallel experts, e.g. on 1xN meshes) gain nothing.
2. **The original layout is uneven:** low *utilization* (mean chip load / busiest chip load). Fewer
   experts per chip means more imbalance.
3. **Routed-expert compute is a large share of prefill time,** and its time follows token load.

| Model, 8x4 Blackhole Galaxy prefill | Experts per chip | Original utilization | Modeled | Measured |
|---|---|---|---|---|
| GLM-5.3, 78 layers | 8 | 0.42 | −18.1% | **−4.07%** |
| Kimi-K3, 24 layers | 28 | 0.72 | −8.3% | **−0.87%** |

The modeled number (busiest-chip routed-expert tiles) ranks plans for one model but overstates the
on-device gain by a model-dependent factor (0.22x and 0.10x above). The on-device A/B decides.

## Pipeline

```
download -> pack -> capture -> calibrate -> heldout -> preset -> screen -> prepare -> ab [-> profile] -> report
  host      host*    device      host         host      host      host      device    device  device      host
```

| Stage | What it does | Output (in the work directory) |
|---|---|---|
| `download` | public corpus: 7 domains, 23 sources, ~20 languages | `raw/*.jsonl` (model-independent; reuse) |
| `pack` | tokenize with the model's tokenizer and chat format; mixes, edge cases, golden control | `sequences/<name>/` |
| `capture` | untraced prefill of every sequence on the original cache, dump routing | `captures/<name>/routing.npz` |
| `import-routing` | routing from any other source instead of `capture` | `captures/<name>/routing.npz` |
| `calibrate` | split requests 50/50 per source, train plans on TRAIN, score on TEST | `calibration/` (results.csv, plans_owners.npz, summary.json) |
| `heldout` | balanced sequence of TEST requests only, for the A/B | `sequences/heldout_mix_bal/`, `heldout/` |
| `preset` | plan -> validated `bijective_within_tp_group` preset | `presets/<model>_<plan>.json` |
| `screen` | original-layout imbalance, modeled gain, expected range, go/no-go | `screen.json` |
| `prepare` | placed weight cache (moved expert shards byte-verified, everything else linked) | the destination you give |
| `ab` | O,B,B,O,O,B per workload (golden control with PCC + KV digests; held-out mix), ship gate | `ab/summary.json` |
| `profile` | sync-bracketed MoE stage timing: MoE and routed-expert share of prefill | `profile/` |
| `report` | everything above as markdown | `report.md` |

All stages resume: finished captures and A/B runs are kept, failed ones are renamed and retried.

## Quick start (Kimi-K3 on one Galaxy)

```bash
export KIMI_K3_HF_MODEL=/mnt/weka/model-weights/llm/moonshotai/Kimi-K3-mxfp4-2496450e
export TT_KIMI_K3_PREFILL_TTNN_CACHE=/mnt/weka/model-cache/scratch/moonshotai/Kimi-K3-Cache/Kimi-K3-Cache-prefill
W=/data/$USER/moe-placement/kimi_k3
M="python -m models.common.moe_placement"

$M download  --workdir $W                       # once; any machine with internet + `datasets`
$M pack      --workdir $W --model kimi_k3 --cpp-root $TT_METAL_HOME
$M capture   --workdir $W --model kimi_k3        # Galaxy, ~5 min per sequence
$M calibrate --workdir $W --model kimi_k3        # CPU, minutes with many cores
$M heldout   --workdir $W --model kimi_k3
$M preset    --workdir $W --model kimi_k3 --plan bal128_s1 --source-main $(git rev-parse HEAD)
$M screen    --workdir $W --model kimi_k3 --preset $W/presets/kimi_k3_bal128_s1.json
$M prepare   --workdir $W --model kimi_k3 --preset $W/presets/kimi_k3_bal128_s1.json --destination /scratch/$USER/k3-placed
$M ab        --workdir $W --model kimi_k3 --placed-cache /scratch/$USER/k3-placed
$M profile   --workdir $W --model kimi_k3 --placed-cache /scratch/$USER/k3-placed   # optional
$M report    --workdir $W
```

Run device stages inside an allocation (`srun --jobid <job> --overlap ...`) with the repository's
`python_env`. Placed caches are large (GLM-5.3 ~375 GB, Kimi-K3 ~340 GB): delete them when done.

## Models

`python -m models.common.moe_placement models` lists registered adapters.

| Model | Experts / top-k | Transport model | Depth | Status |
|---|---|---|---|---|
| `glm_5_3` | 256 / 8 | ring | 78 | verified end to end |
| `kimi_k3` | 896 / 16 | linear | 24 | verified end to end |
| `kimi_k2_7` | 384 / 8 | ring | 61 | configured, needs a first capture |
| `deepseek_v3` | 256 / 8 | ring | 61 | configured, needs a first capture |
| `mistral_small_4` | 128 / 4 | ring | 36 | configured; its test has no single-iteration untraced case yet |

Every device run records the transport TtMoe actually used and fails if it differs from the cost model.

### Other MoE models

* **Host stages work for any model now:** write a `MoeSpec` JSON (`name, num_experts, top_k, rows, groups,
  transport, chunk_tokens, n_chunks`), bring routing as `expert_ids [layers, tokens, top_k]` arrays
  (`import-routing`), and run `calibrate`, `preset` and `screen` with `--spec spec.json`.
* **Device stages need an adapter** (`adapters/base.py`). It must provide:
  1. a routing dump;
  2. an on-device expert-id remap loaded from the placed cache;
  3. a placed-cache writer;
  4. timed and golden prefill runs.

  The deepseek_v3_d_p family has all four. Other expert-parallel stacks (`gpt_oss`, `deepseek_v3` decode,
  `gemma4` at 8x4, `minimax_m3`) need an adapter; `models/common/moe` decode configs already carry an
  `expert_mapping` hook that a placement could plug into.

## Method details

* **Split:** inside every source, requests (not chunks) are assigned 50/50 by tokens with a fixed per-source
  permutation. No held-out token is ever seen by a calibrated plan, including in the A/B's held-out sequence.
* **Balanced mix:** each draw goes to the domain, then source, furthest below an equal token share.
* **Optimizer:** per layer and dispatch group, simulated annealing over expert swaps between rows, scoring
  busiest-chip tiles (1.0), busiest dispatch and combine link (0.25 each) and total hops (0.05).
* **Constraints:** `aggregate` (default; training sums may not get worse) keeps most of the gain; `frame`
  (no training chunk may get worse) keeps little of it.
* **Defaults:** `bal128_s1` (128 balanced training chunks). 32 chunks already reach most of the gain; seeds
  agree within 0.1 point.
* **Chat preamble:** if the model adds a fixed system preamble to every request, check sensitivity with
  `calibrate --eval-from calibration --skip-prefix <N>`.
* **Screen:**
  * `verdict` needs original utilization ≤ 0.8 and a modeled gain ≤ −3%.
  * `expected` is modeled gain × routed-expert share (from `profile`, else 1) × [0.1, 1].

## Tests

```bash
pytest --noconftest models/common/moe_placement/tests -q   # host only, no device
```

The tool reproduces the original GLM-5.3 and Kimi-K3 studies bit for bit: identical plans for all 22
calibration sets, identical result tables, identical `bal128_s1` presets and identical held-out sequences.
Device smoke test through the Kimi-K3 adapter on an 8x4 Galaxy:
* the routing capture is identical to the study's capture;
* the timed held-out run takes 2.641 s, matching the study's original-cache runs;
* the profile gives MoE at 67.6% and routed experts at 20.0% of the profiled iteration, matching the study's
  hand-built profile.
