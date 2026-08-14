# Host runbook: Gemma 4 31B autoport on BH QuietBox 2 (p300x2)

Date: 2026-08-14 UTC
Host: `qb2-120-p02t03` — 2x Blackhole `p300c` (4 chips), 11x10 worker grid,
249 GB RAM, 16 physical cores, TT-KMD 2.10.0, no CUDA/ROCm, no `docker` CLI.

Every problem hit while moving this autoport to a fresh host, and the exact fix.
Written because most of these fail in ways that look like model or hardware
faults but are environment or contract issues, and several cost 10-15 minutes
per rediscovery because they surface only after a full weight load.

## Working environment recipe

```bash
cd /home/mvasiljevic/tt-metal
unset LD_LIBRARY_PATH TT_METAL_RUNTIME_ROOT      # see "ttop env vars" below
source python_env/bin/activate
export TT_METAL_HOME=$PWD
export LD_LIBRARY_PATH=$PWD/build/lib
export PYTHONPATH=/home/mvasiljevic/vllm:$PWD     # vLLM package MUST precede repo root
# serving only:
export TT_GEMMA4_TEXT_VER=gemma4_31b_autoport
export GEMMA4_31B_AUTOPORT_DIR=$PWD/models/autoports/google_gemma_4_31b
export GEMMA4_31B_TENSOR_CACHE=/home/mvasiljevic/models/tt_cache/gemma4_31b_full
# only for logprob/determinism compatibility checks, never for perf claims:
# export GEMMA4_31B_VLLM_HOST_SAMPLING_COMPAT=1
```

Local assets prepared on this host:

| Path | What |
| --- | --- |
| `/home/mvasiljevic/models/google/gemma-4-31B` | checksum-verified base checkpoint |
| `~/.cache/huggingface/hub/models--google--gemma-4-31B/snapshots/5bbc2fb1…` | symlink so the HF **id** resolves offline |
| `/home/mvasiljevic/models/tt_cache/gemma4_31b_full` | warmed 30 GB TTNN tensor cache |
| `/home/mvasiljevic/vllm` + `<tt-metal>/vllm` symlink | vLLM `dev` @ `bf98d55` + TT plugin |
| `/home/mvasiljevic/tt-inference-server` | TTI, `v0.18.0` = `d5913e816ac5` |

## Setup problems

**`ttop` env vars break everything.** `/etc/profile.d/ttop.sh` exports
`LD_LIBRARY_PATH=/home/user/tt-metal/build/lib` and
`TT_METAL_RUNTIME_ROOT=/home/user/tt-metal`, pointing at a checkout that does not
exist here. Symptoms range from segfaults to wrong-library loads. `unset` both in
every shell before doing anything.

**Git identity is unset.** `git commit` fails with "unable to auto-detect email
address (got 'mvasiljevic@qb2-120-p02t03.(none)')". Set repo-local
`user.email`/`user.name`.

**`openai-codex` is not installed** in `python_env`, so
`.agents/scripts/multigoal` cannot start. Fix:
`python -m pip install -r .agents/requirements.txt` (installs `openai-codex`;
`codex` CLI itself was already at `/home/mvasiljevic/.local/bin/codex`, v0.147.0).

**HF id vs path.** `tt/model.py::_resolve_checkpoint()` looks under
`~/.cache/huggingface/hub/models--google--gemma-4-31B/snapshots/*`. The weights
arrived as a plain directory at `/mnt/models/blaze/...`, so a snapshot symlink is
required for unmodified tooling to find them. The local revision is
`5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89`, not the recorded
`d77cb0be8ad40327cc1c6b70eff4b3f0be35bee3`, but the HF tree API shows identical
LFS oids for every file except `README.md`, and the staged copy was checksummed
against them. The recorded `.refpt` reference stays valid.

## vLLM install and the nested-checkout trap

Install per `plugins/vllm-tt-plugin/docs/install-vllm-tt.sh`, from the vLLM repo
root with tt-metal's `python_env` active:

```bash
VLLM_TARGET_DEVICE=empty uv pip install -e . \
  --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
uv pip install -e plugins/vllm-tt-plugin
```

Result `vllm 0.1.dev60+gbf98d556b.d20260814.empty`, `vllm-tt-plugin 0.0.0`,
`tblib 3.2.2`. torch stayed at 2.11.0+cpu and `ttnn` still imports, so the
validated model environment is undisturbed. Never let the PyPI CUDA `vllm` wheel
in; the plugin's `pyproject.toml` deliberately omits vLLM as a dependency for
exactly that reason.

**A nested `vllm` checkout at `<tt-metal>/vllm` shadows the installed package.**
`tests/test_vllm_adapter_contract.py` reads
`ROOT/"vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py"`, so the
checkout must be reachable there. But with the tt-metal root on `PYTHONPATH`, a
bare `vllm/` directory resolves as a **namespace package**: `import vllm` yields
`__file__ = None` and `vllm.plugins` points at the repo's *top-level* `plugins/`
directory. Symptom:

```text
ImportError: cannot import name 'PLATFORM_PLUGINS_GROUP' from 'vllm.plugins' (unknown location)
```

Fix: put the real package directory first —
`PYTHONPATH=/home/mvasiljevic/vllm:$TT_METAL_HOME`. The nested path here is a
symlink, excluded via `.git/info/exclude` (local only, so the tracked
`.gitignore` is untouched and it can never be committed into tt-metal).

## Serving problems, in the order they surface

Each of these appears only after a full model load, so fix them up front.

**1. Do not override `--served-model-name` when `--hf-model` is a filesystem
path.** `run_vllm_server.py` sends `"model": hf_model` verbatim in every probe.
Passing `--served-model-name google/gemma-4-31B` alongside
`--hf-model /path/to/weights` registers only the alias, so probes 404:

```text
RuntimeError: non-aligned 149-token request returned 404
The model `/home/mvasiljevic/models/google/gemma-4-31B` does not exist.
```

Fix: pass the **HF id** (`--hf-model google/gemma-4-31B`), which the cache
symlink resolves offline, so the request field and the served name agree. The
recorded evidence confirms this was the original shape: its saved response has
`"model": "google/gemma-4-31B"`.

**2. The context contract is enforced; a reduced context is refused.** Trying to
serve small to save time fails by design:

```text
ValueError: Gemma 4 31B serving must use context_contract max_model_len=113280, got 8192
```

Raised at `tt/generator_vllm.py:107`. Serve at 113280. Useful side effect: this
doubles as the capacity probe.

**3. Logprob and determinism checks need an opt-in flag.** `run_vllm_server`'s
`_run_logit_determinism_check` requests `"logprobs": 20`, which routes to the
host sampler, which the autoport refuses by default:

```text
ValueError: host sampling compatibility is disabled; set
GEMMA4_31B_VLLM_HOST_SAMPLING_COMPAT=1 only for shared sampling/logprob compatibility tests
```

The engine then dies with `EngineDeadError` and the server must be restarted.
Set `GEMMA4_31B_VLLM_HOST_SAMPLING_COMPAT=1`. It is safe for perf runs because
the guard fires only when `sampling_params is None` — per request, only for
requests that hand logits to the host. Normal traffic keeps the device path.
Do not treat it as a general-purpose switch: a logprob request mid-stream
releases the decode traces, so the next request pays trace re-capture.

**4. `run_vllm_server` writes into the recorded evidence tree.** `output_dir` is
hardcoded as `model_dir / "readiness_vllm"` with no CLI override, so any run
overwrites committed stage 09/10 artifacts. A failed run of mine replaced a
recorded passing 200 non-aligned check with a 404 and truncated `server.log`
from 13,981 lines to 874. All those artifacts are tracked, so
`git checkout -- models/autoports/<model>/readiness_vllm/` restores them —
copy your own artifacts elsewhere first, and check `git status` after every run.

## Hardware: ethernet heartbeat timeouts are recoverable

Seen twice on 2026-08-14, both on device 0, after repeated mesh open/close
cycles:

```text
RuntimeError: TT_THROW @ tt_metal/llrt/llrt.cpp:566: tt::exception
Device 0: Timed out while waiting for active ethernet core 29-25 to become
active again. Try resetting the board.
```

The recorded Stage 11 run hit the same class on the same device (core 31-25) and
its anomaly ledger resolved it as "infrastructure recovery; not a model result".
This is not a model failure. Recovery, per `$tt-device-usage`:

```bash
timeout 60 tt-smi -ls --local
timeout 180 tt-smi -r
timeout 60 tt-smi -ls --local
# then prove the mesh opens before resuming
python -c "import ttnn; md=ttnn.open_mesh_device(ttnn.MeshShape(1,4)); print(md.shape); ttnn.close_mesh_device(md)"
```

Both times a single bounded reset restored all 4 chips and a clean 1x4 mesh
open/close. Kill stale device holders before resetting, and reset after any
`EngineDeadError` rather than reloading onto dirty devices — a wasted load costs
10+ minutes.

Also note `HugePages_Total: 0` here; UMD warns and falls back to regular pages,
costing host-device DMA bandwidth. Worth fixing before quoting serving
throughput.

## Measurement traps

**Cold JIT inflates perf by ~30x while leaving accuracy identical.** First
teacher-forcing run on a fresh checkout: 9796.54 ms TTFT, 18.07 t/s/u decode,
`JIT cache stats: 333/448 hits`. Identical command warmed: 318.01 ms, 27.89
t/s/u, `448/448 hits`. Accuracy was bit-identical across both
(0.920/1.000/1.000), which is the tell that the gap was compilation, not
computation. Never quote a perf number taken before the JIT cache and weight
staging are warm.

**`--tensor-cache` helps, but less than the load time suggests.** Teacher forcing
932 s without, 680 s with: 252 s (27%) saved, identical results. Bounded because
`from_pretrained` always calls `_load_checkpoint_state`, so the 62 GB safetensors
read happens either way; only the torch->TTNN conversion and host tiling are
skipped. The flag was added to `run_prefill_check`, `run_teacher_forcing`, and
`run_autoregressive` in commit `840b8301c40`.

**Two different precision regimes are reachable by default.**
`run_prefill_check`/`run_teacher_forcing` go through `build_generator`, which
consults only `GEMMA4_31B_PRECISION_CONFIG` — unset means the **BF16 LM-head
default**. `tests/run_full_model_qualitative.py` instead falls back to
`doc/datatype_sweep/selected_precision_config.json`, so it runs the Stage 08
selected **`lm_head_bfp8_hifi2`** policy. Check
`runtime_precision.config_id` in the output before comparing anything.

## Results on this host

Correctness reproduces the recorded values; see
`full_model/revalidation_p300x4/README.md` for the full baseline.

| Evidence | This host | Recorded |
| --- | --- | --- |
| Prefill top1/top5/top100 (BF16 default) | 0.910 / 1.000 / 1.000 | identical |
| Teacher forcing top1/top5/top100 | 0.920 / 1.000 / 1.000 | 0.91 (+1 token of 100) |
| Token-out steady decode (selected BFP8) | 33.9225 t/s/u | 34.256 (-0.97%) |
| Static contract tests | 25/25 | 23/23 at remediation |
| vLLM adapter contract tests | 26/27 | 16 recorded |
| Serving KV pool | 9740 blocks, 8.62x concurrency at 113,280 | 9740 blocks |
| Non-aligned 149-token serving request | PASS | PASS |

No core-grid change was needed: every grid is either derived from
`compute_with_storage_grid_size()` or a fixed grid that fits inside 11x10.
`tt/fused_decoder.py:53` hardcodes `CoreCoord(11, 10)`, which exactly fits here
and is latent on any narrower part.

The one adapter-contract failure is
`test_plugin_registers_the_autoport_and_honors_greedy_only_policy`, which is the
lost vLLM plugin work, not a defect on this host. See
`tti_release/STAGE11_PREREQUISITES_p300x2.md`.
