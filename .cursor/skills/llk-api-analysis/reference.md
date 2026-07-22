# LLK API Analysis — Reference

Detailed setup, known gotchas, and the Quasar-gap method. Read the relevant
section only when the model needs it.

## The analyzer (`tt_metal.tools.llk_api_analyzer`)

**Source: PR [#48671](https://github.com/tenstorrent/tt-metal/pull/48671) — branch
`halgh/llk-api-analyzer`.** The tool lives at `tt_metal/tools/llk_api_analyzer/` and,
until #48671 merges to `main`, is not present on a fresh checkout. Bring it in with
either `git merge origin/halgh/llk-api-analyzer` (purely additive — new files only) or,
in a worktree that lacks it, by copying `tt_metal/tools/llk_api_analyzer/` from a checkout
that has it. This skill branch already carries it via that merge.

A TTNN/metal run JIT-compiles each compute kernel into three Tensix RISC-V ELFs
(trisc0/1/2 = unpack/math/pack). Built with `-O3 -flto`, they carry DWARF when
`TT_METAL_RISCV_DEBUG_INFO=1`. The analyzer walks the `DW_TAG_inlined_subroutine`
tree (all compute APIs are `ALWI`, so the surviving inlined calls ≈ the runtime
call graph), reads template args as configuration, and parses `chlkc_descriptors.h`
for data formats / tile sizes. `--run` executes your command with an isolated
`TT_METAL_CACHE` + debug info, then analyzes the kernels it produced.

Layers (`-l`): `llk_api` (default), `llk_core`, `compute_api`, `other`.
Keep the default `llk_api` for these reports.

## Environment

```bash
cd <tt_metal_root>
source python_env/bin/activate
export PYTHONPATH=<tt_metal_root> OMP_NUM_THREADS=24 MKL_NUM_THREADS=24
# analyzer deps:
python -c "import elftools, tabulate" 2>/dev/null || uv pip install pyelftools tabulate
```

## Branch-only models → worktree + build

If the model exists only on a feature branch that diverges in C++
(`tt_metal/`, `ttnn/cpp`, `tt-llk`, ...), do NOT switch the user's active branch.
Use an isolated worktree with its own build and venv:

```bash
cd <main_tt_metal_root>
git fetch origin <branch>
git worktree add -b <local_name> <worktree_path> origin/<branch>
cd <worktree_path>
git submodule update --init --recursive \
  tt_metal/third_party/tracy tt_metal/third_party/umd tt_metal/third_party/tt-cluster-descriptors
# tt-llk is a plain directory on most branches (already populated); init it only if it is a submodule.

# The analyzer tool (PR #48671) may be absent — pull it from its branch, or copy it in:
git merge --no-edit origin/halgh/llk-api-analyzer 2>/dev/null || \
  { [ -d tt_metal/tools/llk_api_analyzer ] || cp -r <main_tt_metal_root>/tt_metal/tools/llk_api_analyzer tt_metal/tools/; }

./build_metal.sh -c            # full build, ccache on
./create_venv.sh               # worktree-local python_env
source python_env/bin/activate
uv pip install pyelftools tabulate
```

**Critical — do NOT reuse the main repo's `python_env`.** Its `.pth`/editable-install
entries hardcode the main repo path, so it imports the *main* branch's `_ttnn.so`.
Verify the worktree venv resolves locally:

```bash
python -c "import ttnn,os;print(os.path.dirname(ttnn.__file__))"   # must be under <worktree_path>
```

Prefer building on fast local disk (e.g. `/localdev/...`) over networked FS.

Cleanup when done:
```bash
cd <main_tt_metal_root>
git worktree remove <worktree_path> --force
git branch -D <local_name>
```

## Device / grid

- Match the grid to the host. Some models hardcode their own sharding grid;
  others take `TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="X,Y"` (front of the run
  command). Blackhole p100a ≈ use a ~20-core (`"4,3"`) config; P150 has the full
  grid. Configs that need a larger grid than the host has will `pytest.skip`.
- If a prior run was killed mid-execution, reset the device: `tt-smi -r 0`.

## Known gotchas

- **CPU reference forward can time out.** Some PCC tests run a PyTorch reference
  in bf16 that hits a single-threaded conv fallback and blows past a hardcoded
  `@pytest.mark.timeout(...)` → the TTNN forward never runs → empty CSV. CLI
  `--timeout` does NOT override a marker. Fix by editing the test to run the
  reference in fp32 and bumping the marker; revert after. (Not needed for fast
  models, e.g. yolov8s @ 640x640.) Recognize this failure mode: CSV has only the
  header line.
- **PCC assert failure is usually fine.** Kernels compile *before* the PCC assert,
  so a `FAILED`/exit-1 run still yields a full CSV; the analyzer prints
  "analyzing whatever kernels were produced". Always sanity-check row count.
- **Arch-support guard → whole suite skipped.** A test/conftest may hard-skip on the
  host arch, e.g. TTTv2's `models/common/tests/conftest.py`:
  `if ttnn.device.is_blackhole(): pytest.skip("Blackhole device is not supported for this test yet")`.
  Every case then skips *before the device opens* — the CSV is empty and no bypass of
  the CPU-timeout kind helps. Grep for `is_blackhole`/`is_wormhole`/`pytest.skip` first.
  Fixes: run on a supported machine (reserve the right SKU), or — only if the goal is to
  see what *would* compile — temporarily gate the guard behind an env flag and revert
  after (expect it may still fail later in model build if the arch is genuinely unfinished).
- Weights: models load them differently (local pkl, NAS/CI generator, or
  auto-download via a library like `ultralytics`). Auto-download needs internet
  and the relevant pip package installed in the venv.
- **Gated HF model → ungated mirror.** For LLK analysis the compiled kernels depend on
  the model's *config* (shapes, #heads/layers, dtypes, rope), **not** on weight values.
  So if the canonical id is gated (`hf_hub_download` → 401 `GatedRepoError`) and no
  `HF_TOKEN`/cache/`/mnt/MLPerf` is available, substitute an ungated re-upload with an
  identical config and point `HF_MODEL` at it. Verify the config matches before using:

  ```python
  # e.g. meta-llama/Llama-3.2-1B-Instruct (gated) → unsloth/Llama-3.2-1B-Instruct (open)
  from huggingface_hub import snapshot_download
  d = snapshot_download("unsloth/Llama-3.2-1B-Instruct",
        allow_patterns=["*.json","*.safetensors","*.model","tokenizer*"])
  # confirm architectures / hidden_size / num_hidden_layers / num_attention_heads /
  # num_key_value_heads / head_dim / vocab_size / torch_dtype / rope_scaling all match.
  ```

  Then run with `HF_MODEL=<open-mirror>`. (The `.refpt` accuracy reference is only needed
  for a `token-accuracy` case; the `batch-*` perf cases don't use it.)

## Quasar-gap method (`scripts/quasar_gap.py`)

Per base API, count LLK source files mentioning the API name under each arch's
`tt_metal/hw/ckernels/<arch>` + `tt_metal/tt-llk/tt_llk_<arch>`. Flag names
present on a reference arch (blackhole/wormhole_b0) but absent on the target
(quasar) as candidate GAPs. Rows found on no arch are INCONCLUSIVE (shared or
higher-level API).

This is a name-substring heuristic and a **first pass**. Always confirm each
flagged GAP by reading the actual headers, because:
- an op may exist under a different name, or only for some format/param variants;
- a conversion may be routed through a datacopy / unpack-pack gasket rather than a
  dedicated op (e.g. block-float `Bfp8_b ↔ Float16_b` typecast is a pack/unpack
  reformat, not the SFPU typecast op — so typecast is NOT a gap even though a
  `_uninit`-style name variant may not match verbatim).

Recurring real gap across models: the `fast_tilize` / `fast_untilize` family
(`llk_{math,pack,unpack}_fast_tilize_*`, `llk_*_fast_untilize_*`) — present on
Blackhole/Wormhole, absent on Quasar.
