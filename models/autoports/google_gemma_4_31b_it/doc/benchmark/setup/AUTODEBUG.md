# AutoDebug: inherited Stage 11 launcher paths

Inspection date: 2026-11-13. Checkout: `v5-extended-run`, HEAD
`a12a2675ccd670bb648b26c86760a44a2f822fc6`.

Revision scope: the two focus files matched that HEAD during the initial reads
and diagnosis. Concurrent edits to both appeared during final verification.
All source-line findings below refer to the inspected baseline, not those new
edits. This investigation did not make or validate the concurrent corrections.

## Diagnosis

The earliest defect is `tests/run_serving.sh:4–6`: it overwrites the valid enabled
skill roots with missing v3 paths, then asks Python to execute a nonexistent
`environment.py`. With `set -e`, the shell exits before the readiness runner.
Correcting that exposes a second definite bad input: the runner's model path is
hardcoded to a missing HF snapshot. After a successful launch, a third defect in
`tools/benchmark_server.py:199–204` prevents identity generation by querying Git
in a nonexistent plugin checkout.

These are host-side setup/provenance defects. No model, kernel, device timing,
accuracy, or performance conclusion follows from this investigation.

Paths below are relative to
`models/autoports/google_gemma_4_31b_it/`, unless qualified otherwise. `BRINGUP`
means the enabled `TT_MODEL_BRINGUP_ROOT`, not the obsolete root in the shell.

## Direct observations

Read `AGENTS.md`, `bringup/run_contract.json`, and
`bringup/agentic_evals/inputs.json`. The current operator instructions require
local execution, the new autoport, and the fixed model/tokenizer revision
`842da3794eaa0b77d5f08bae87a17459d91ff475`. Old operator entries in
`run_contract.json` do not supersede the later v5 instructions.

Filesystem and interpreter metadata checks found:

| Item | Observed value / status |
| --- | --- |
| Enabled bringup root | `/home/mvasiljevic/v5-agentic/codex-home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.27`; exists |
| Enabled AutoDebug root | `/home/mvasiljevic/v5-agentic/codex-home/plugins/cache/tenstorrent-skills/tt-autodebug/0.1.9`; exists |
| Shell's two `.codex-gemma4-v3` roots | Missing, including the requested `scripts/environment.py` |
| `HF_HOME` | `/home/mvasiljevic/.cache/huggingface` |
| `HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE` | Unset in the investigator environment |
| Hardcoded `/home/bringup/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/<revision>` | Missing |
| Same pinned snapshot under `$HF_HOME/hub/` | Exists; `config.json`, `tokenizer.json`, `tokenizer_config.json`, `generation_config.json`, and `model.safetensors.index.json` exist |
| Hardcoded `/home/mvasiljevic/gemma4-31b-pipeline-v3/vllm-tt-plugin-stage09` | Missing |
| Actual plugin origin, using `python_env/bin/python` and top-level `find_spec` | `<checkout>/bringup/artifacts/v4-serving-plugin/src/vllm_tt_plugin/__init__.py` |
| Plugin editable `.pth` | Points to that v4 checkout's `src` directory |
| Plugin `direct_url.json` | Still records the missing v3 checkout; it disagrees with actual import resolution |

Snapshot checks establish existence of those files only, not completeness or
correctness of every weight shard. The plugin was not imported or executed.

## Supported findings and minimum corrections

### 1. Unconditional skill exports defeat the enabled environment

**Evidence:** `tests/run_serving.sh:4–5` assigns both v3 roots unconditionally;
`:6` invokes the bringup root's `scripts/environment.py`. `:2` enables `set -e`.
`tools/benchmark_server.py:308,350–352` inherits the caller's environment and
launches this shell in the checkout, so a correct parent environment cannot
override those shell assignments. This affects all benchmark batch choices,
including the configured 1, 8, and 32.

**Consequence:** from the checkout, Python receives an absent script pathname;
the failed command substitution makes the assignment fail and the shell exits.
Changing only `TT_MODEL_BRINGUP_ROOT` is insufficient: the active environment
helper validates `TT_AUTODEBUG_ROOT` too.

**Minimum correction:** remove the two historical assignments. Require the
caller-selected roots to be nonempty and use the selected bringup helper, keeping
its dependency validation. For example, require
`${TT_MODEL_BRINGUP_ROOT:?}` and `${TT_AUTODEBUG_ROOT:?}` rather than selecting a
version by globbing installed packages or embedding a new v5 absolute path.
`BRINGUP/scripts/environment.py:18–33` validates the AutoDebug manifest/skills;
`:36–40` adds the selected bringup runtime to `PYTHONPATH`. An isolated host-only
check of that helper with the current roots passed. No launcher was invoked.

### 2. The explicit HF model path bypasses the working cache

**Evidence:** `tests/run_serving.sh:13` assigns the missing `/home/bringup/...`
snapshot, and `:16` passes it as `--hf-model`.
`BRINGUP/runtime/readiness_check/run_vllm_server.py:284–285` puts that value in
the server's `--model` argument; `:337–340` also exports it as `HF_MODEL`.
`HF_HOME` remains set correctly, but does not rewrite an explicit local path.

**Consequence:** after fixing the skill roots, the configured model location is
still nonexistent. The precise vLLM/Transformers exception was not reproduced;
the path cannot name the intended local checkpoint. Revision flags do not repair
the location of a local directory.

`tools/benchmark_server.py:326–344` independently embeds the same obsolete
snapshot in its additional-argument string. Its `--additional-server-args`
appears after the shell's default through `tests/run_serving.sh:22` (`"$@"`).
The readiness parser defines this as an ordinary string/store option
(`run_vllm_server.py:967–976`), so the last occurrence **replaces**, rather than
extends, the shell's whole string. A small argparse-only check confirmed that
behavior. Resolve the path consistently across both launchers.

**Do not overstate the trailing token:** shell `:21` puts its repeated snapshot
after `--served-model-name`; that option accepts a list, so the token can be a
second served alias. The benchmark puts its trailing path after
`--max-num-seqs N`, where it can be the optional `model_tag` positional argument.
The installed direct API-module entry point parses arguments and calls
`run_server(args)` without the CLI serve command's `model_tag`-to-`model`
assignment. The definite failing model input is the shell's `--hf-model`, not a
proven load from every repeated token. Correcting only the benchmark's trailing
path therefore cannot repair the shell's model input.

Parser evidence is in installed `vllm/engine/arg_utils.py:230–252,987–988`,
`vllm/entrypoints/launchers/cli_args.py:417–422`, and
`vllm/entrypoints/launchers/api_server/entry.py:199–222`; compare the separate
CLI conversion in `vllm/entrypoints/cli/serve.py:54–55`.

**Minimum correction:** derive one local snapshot from the HF hub cache and the
fixed repository/revision, then use it for the actual runner model input and
any retained duplicate argument. The installed
`huggingface_hub/constants.py:158–184` defines cache precedence:
`HF_HUB_CACHE`, then legacy `HUGGINGFACE_HUB_CACHE`, then `$HF_HOME/hub`
(with the library's normal home/XDG defaults). Reuse that resolver rather than
hardcoding a username. Validate the pinned directory before launching. An
existing explicit snapshot override, if supported by the correction, must be
validated against the same pinned revision. Neither launcher implements
`MODEL_SNAPSHOT`/`HF_SNAPSHOT` environment overrides; the shell does accept a
later caller CLI `--hf-model` through `"$@"`, which must retain its precedence.
Do not fall back to unpinned `main`, fetch a different revision,
copy a checkpoint, or change credentials to repair a path.

Keep the model and tokenizer revision flags, text-only mode, served model name,
262144 context, async scheduling, prefix caching, and authorized top-k 32 recipe.
The path correction does not require changing scheduler behavior or plugin runtime.

### 3. Benchmark provenance queries the wrong plugin checkout

**Evidence:** `tools/benchmark_server.py:199` fixes `plugin` to the nonexistent
v3 path. `:201–204` runs `git -C <that path> rev-parse HEAD` with
`subprocess.check_output`. It does not discover the installed package or use
`VLLM_TT_PLUGIN_ROOT`.

**Consequence:** after health and the preceding registration/configuration checks
pass, `observe()` raises before writing `configuration-bN.json` or the requested
identity (`:220–262`). This also affects the reuse path (`:284–294`). It does not
explain an initial server-load failure: it is a later, independent reporting
failure, and the server may already be alive when it occurs.

**Minimum correction:** discover `vllm_tt_plugin` using the same
`ROOT/python_env/bin/python` used to serve; validate its editable source layout;
derive the checkout from its actual `find_spec(...).origin`. Verify any explicit
configured root agrees with that origin, then collect provenance there. Do not
replace the v3 constant with a v4/v5 absolute constant, and do not trust the stale
`direct_url.json` as the source of the checkout. Retain actual origin alongside
commit evidence. The existing readiness resolver uses this origin-based approach
(`run_vllm_server.py:208–224`).

**Additional observed local prerequisite:** ordinary
`git -C bringup/artifacts/v4-serving-plugin rev-parse HEAD` fails with Git's
“dubious ownership” check in this session: investigator UID 6002, checkout and
`.git` owner UID 1000. Thus changing only the path still leaves this exact Git
command failing here. Reading `.git/HEAD` and its named loose ref found
`90f33d2e392eb8bfda6948df9dabe712a724cb1a`; that is a local ref observation, not
proof of a clean tree or unchanged runtime. For future Git collection, use a
properly owned/trusted checkout, or an exact-path, command-scoped trust exception
only for the verified intended checkout. No global Git configuration was changed;
do not use a wildcard trust exception or substitute a remembered commit.

### 4. The stale plugin environment variable breaks sampling discovery

**Evidence:** `tests/run_serving.sh:8` overwrites any caller-selected
`VLLM_TT_PLUGIN_ROOT` with the missing v3 path. The active readiness helper
compares this variable with the actual import origin and raises on mismatch
(`run_vllm_server.py:208–224`). Its default stages include sampling (`:143`), and
sampling triggers that helper through `_find_plugin_tests_dir()` (`:1181–1182`).

**Scope:** the benchmark explicitly requests `--stages serve`
(`tools/benchmark_server.py:315–317`). The serve-only source check at readiness
`:1177–1179` does not consult this variable. Therefore the variable is a definite
default/sampling readiness defect, **not an established additional serve-only
launch blocker**, and it does not itself redirect Python imports to v3.

**Minimum correction:** remove the forced assignment and let the readiness
helper discover the installed standalone plugin; preserve an explicit caller
value for validation. A host-only check executing just the filesystem resolver
functions extracted from the active helper produced a mismatch error for the v3
value and correctly returned `bringup/artifacts/v4-serving-plugin` when unset or
set to that matching root. No plugin/runtime changes or reinstall are needed to
correct this variable.

## Other potential issues / follow-up scope

- **Other benchmark clients retain the same absent tokenizer path.**
  `tools/benchmark_client.py:14,48–49` and
  `tools/benchmark_performance_partial.py:22,168–169` need the
  same resolution if those tools are selected for the new run. They are outside
  the two requested focus files; their use by the current orchestration was not
  established. Do not call the complete Stage 11 flow portable after checking
  only server startup.
- **Benchmark extra arguments drop the shell's parser flags.** The replacement
  described above omits `--reasoning-parser gemma4`, `--tool-call-parser gemma4`,
  and `--enable-auto-tool-choice`, present in shell `:21`. This is concrete
  command drift, but not evidence for the reported missing-path failure or a
  measured benchmark error. If this launcher is reused for agentic chat/tool
  traffic, preserve the intended parser recipe explicitly.
- **Direct shell invocation assumes the checkout is the working directory.**
  `python_env/bin/python`, `--model-dir`, and `$PWD` cache paths are relative in
  the shell. The benchmark supplies `cwd=ROOT`, so this is not the reported
  benchmark failure. Deriving/cd-ing to the checkout from `BASH_SOURCE[0]` is an
  optional portability improvement for invocation from other directories.
- Existing phase timing, asynchronous completion, device-owner checks, and
  hardware behavior were not evaluated. None is needed to explain the confirmed
  path failures, and no new synchronization or profiler is warranted by them.

## Verification and claim review

Performed only reads and cheap host checks: `bash -n` on the shell (pass), AST
parse of `benchmark_server.py` (pass), targeted path existence checks, package
metadata/top-level import-spec inspection, the environment helper validation,
isolated filesystem resolver checks, and argparse option precedence. No server,
target model, TTNN device operation, benchmark workload, or network request ran.
No old benchmark outputs were read or reused. This investigation did not edit
source files or installed packages; this report is its only deliverable.

The first draft received an independent inspection pass and was checked against
both the active readiness consumer and the installed vLLM parser. The headline
claims distinguish the earliest shell exit,
the later invalid model input, and the post-readiness provenance failure. The
plugin variable's serve-only effect and repeated snapshot tokens were narrowed
to their demonstrated roles rather than promoted as extra model-load failures.

Before a future hardware launch, host-side validation should establish that both
launchers retain the selected skill roots, resolve the same pinned local model,
identify the actual installed plugin, and can record its provenance. Capture
generated argv with process launch stubbed to verify these properties. A later
live readiness check is still required to establish successful serving and actual
autoport registration; this inspection does not claim either.
