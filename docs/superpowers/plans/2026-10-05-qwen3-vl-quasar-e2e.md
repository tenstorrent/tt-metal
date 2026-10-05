# Qwen3-VL Quasar e2e Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A repeatable e2e harness that runs Qwen3-VL-4B-Instruct (V vision blocks, T text layers, K decode steps) on WH/BH (ttsim or hardware), craq-sim and the Quasar emu2x3 emulator, and gates on per-stage PCC against a truncated HF reference.

**Architecture:** Permanent Quasar adaptations (bf16, grid-derived configs, layer counts, vision padding) live in the Quasar model copy behind `QuasarModelArgs`/`QuasarVisionModelArgs`, selected on Quasar or by `--quasar-config`. A thin pytest harness under `tests/e2e/` builds inputs from a size preset, captures HF goldens, runs the TT pipeline with stage recorders and an op-override registry, and writes a per-run folder. Three bash wrappers set per-target environment.

**Tech Stack:** Python 3 / pytest, ttnn, transformers 5.12 (`Qwen3VLForConditionalGeneration`), torch 2.11 CPU, bash.

**Spec:** `docs/superpowers/specs/2026-10-05-qwen3-vl-quasar-e2e-design.md`

## Global Constraints

- Model: `Qwen/Qwen3-VL-4B-Instruct` only (HF_MODEL); dims fixed: text hidden 2560, 36 layers, 32/8 heads, head_dim 128, intermediate 9728, vocab 151936; vision depth 24, hidden 1024, 16 heads, deepstack 5/11/17.
- bf16 everywhere on the Quasar config: weights, activations, KV cache, `ccl_dtype`, `lm_head_dtype`; fidelity HiFi4. No bfloat8_b / bfloat4_b.
- Model shapes never change; only memory placement/layout may change, and only where the captured config does not fit the grid or is unsupported.
- Non-Quasar behavior of the model copy is unchanged unless `--quasar-config` is passed.
- Defaults: test V=2, T=2, K=1, deepstack at real taps; scripts V=2, T=2, K=1, `--deepstack-at 0`, `--size tiny`. Every default in scripts carries a ≤2-sentence comment; KV-blocks flag has a 1-line doc.
- No change to `tt_metal/llrt/llrt.cpp` is committed (local debugging patch only).
- Commits: plain messages, no AI attribution (repo CLAUDE.md). Commit only on branch `gchoudhary/59033/quasar/get-qwen3_vl-functional-end-to-end-on-emulator-and-craq-sim`. Never push without asking.
- Bash: `#!/usr/bin/env bash`, `set -euo pipefail`, `shopt -s nullglob`, arrays, quoted expansions, `shellcheck -o all` clean.
- Run one pytest file per invocation (user preference). Environment before any python: `source python_env/bin/activate && export TT_METAL_HOME=$(pwd) TT_METAL_RUNTIME_ROOT=$(pwd) PYTHONPATH=$(pwd):$PYTHONPATH`.
- Host fallbacks never produce PASS: any active host fallback ⇒ verdict `DIAGNOSTIC`.
- Emulator is a shared, slow resource: ask the user before every emulator run (needs their IRD reservation / `NNG_SOCKET_ADDR`).

## Review Focus

1. Padded rows leaking into comparisons (vision pad, prefill pad to 128, last-token tile slice) — expected: every stage is sliced to the real token count before PCC; a padded tensor compared unsliced must fail loudly with a shape error, not silently pass.
2. Stale TT weight cache after changing V/T, dtype policy, grid or deepstack remap — expected: cache directory key includes all of them, so a changed config never loads tensors built for another.
3. Op hooks silently disabled because `enable_fast_runtime_mode` is on — expected: the test detects it and states in `pcc.md` that `progress.log` is unavailable; scripts always disable fast runtime mode.
4. Grid mismatch between the WH/craq-sim override and the emulator — expected: `--qwen-expect-grid` makes the test fail at start if `compute_with_storage_grid_size()` differs.
5. Deepstack remap applied to only one side — expected: TT and HF configs are both derived from one function; a unit test asserts identical indexes/depth.

---

## File Structure

All new harness files under `models/experimental/ops/quasar/qwen3_vl/tests/e2e/` (abbrev. `E2E/`); model-copy files under `models/experimental/ops/quasar/qwen3_vl/tt/` (abbrev. `TT/`).

| File | Responsibility |
|---|---|
| `TT/quasar_config.py` (new) | `vision_padded_seq_len`, `truncate_hf_config`, `bf16_decoders_precision`, `QuasarModelArgs`, `QuasarVisionModelArgs`, `model_args_classes`, grid helpers |
| `TT/model.py`, `TT/vision_attention.py`, `TT/vision_mlp.py`, `TT/vision_layernorm.py`, `TT/model_config.py` (modify) | use args-provided dtypes/grids/padding instead of literals |
| `E2E/__init__.py` (new, empty) | package marker |
| `E2E/conftest.py` (new) | `--qwen-*` pytest options, `qwen_run_config` fixture |
| `E2E/config.py` (new) | `RunConfig` dataclass, option parsing, cache-key |
| `E2E/presets.py` (new) | `Preset`, `PRESETS`, `build_inputs` |
| `E2E/host_reference.py` (new) | truncated HF model + `Goldens` capture |
| `E2E/pcc.py` + `E2E/thresholds.json` (new) | per-stage compare, verdict, `pcc.md` |
| `E2E/progress.py` (new) | ttnn pre/post op hooks → `progress.log` |
| `E2E/recorder.py` (new) | `StageRecorder` wrapping TT module instances |
| `E2E/op_overrides.py` (new) | workarounds + host fallbacks registry, `OverrideSession` |
| `E2E/tt_runner.py` (new) | TT pipeline: vision → prefill → teacher-forced decode |
| `E2E/test_qwen3_vl_e2e.py` (new) | the single e2e test node |
| `E2E/test_harness_cpu.py` (new) | CPU-only unit tests of the harness |
| `E2E/test_fallbacks.py` (new) | fallback-vs-real-op certification (WH/BH only) |
| `E2E/probe_grid.py` (new) | prints arch + compute grid of the current target |
| `E2E/_common.sh`, `run_wh_bh.sh`, `run_craq.sh`, `run_emu.sh` (new) | flag parsing, env per target, run folder, exit codes |
| `E2E/README.md` (new) | recipes, flags, cherry-pick list, grid table |

Spec deviations (additive only): `tt_runner.py`, `recorder.py`, `progress.py`, `config.py`, `probe_grid.py`, `_common.sh`, `test_harness_cpu.py` split responsibilities the spec assigned to the test file.

Stage names used everywhere: `vision.block{i}`, `vision.deepstack{j}`, `vision.merger`, `text.layer{i}`, `text.norm`, `text.logits.prefill`, `text.logits.decode{k}`.

---

### Task 1: Download checkpoint, stage ttsim, probe grids

**Files:**
- Create: `E2E/__init__.py` (empty), `E2E/probe_grid.py`

**Interfaces:**
- Produces: a recorded grid table (in the task report, copied into README in Task 9): for each target, `compute_with_storage_grid_size()` (x, y) and the `TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE` value that reproduces the emulator's grid on ttsim WH/BH and craq-sim. Name the emulator grid `EMU_GRID="XxY"` (x = columns).

- [ ] **Step 1: Download the checkpoint** (≈9 GB, uses `HF_TOKEN` from env)

```bash
source python_env/bin/activate
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-VL-4B-Instruct'))"
```
Expected: prints a path under `~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/...`.

- [ ] **Step 2: Write `E2E/probe_grid.py`**

```python
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Print arch and compute grid of the device the current env targets."""
import ttnn


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        g = mesh.compute_with_storage_grid_size()
        print(f"PROBE arch={ttnn.get_arch_name()} grid={g.x}x{g.y} dram_grid={mesh.dram_grid_size()}")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Stage ttsim WH and BH**

```bash
for a in wh:wormhole_b0_80_arch bh:blackhole_140_arch; do
  n="${a%%:*}"; y="${a##*:}"; d="/localdev/$USER/ttsim/sim_${n}"
  mkdir -p "$d" && cp "/localdev/$USER/ttsim/src/_out/release_${n}/libttsim.so" "$d/" \
    && cp "tt_metal/soc_descriptors/${y}.yaml" "$d/soc_descriptor.yaml"
done
```

- [ ] **Step 4: Probe native grids** (one run each)

```bash
export TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_DISABLE_SFPLOADMACRO=1
TT_METAL_SIMULATOR=/localdev/$USER/ttsim/sim_wh/libttsim.so python models/experimental/ops/quasar/qwen3_vl/tests/e2e/probe_grid.py
TT_METAL_SIMULATOR=/localdev/$USER/ttsim/sim_bh/libttsim.so python models/experimental/ops/quasar/qwen3_vl/tests/e2e/probe_grid.py
TT_METAL_SIMULATOR=/localdev/$USER/sim/libttsim.so python models/experimental/ops/quasar/qwen3_vl/tests/e2e/probe_grid.py
```
Expected: three `PROBE ...` lines (WH 8x8, BH ~13x10 or as reported, craq-sim 8x4 or as reported).

- [ ] **Step 5: Probe the emulator grid — ASK THE USER FIRST**

Ask: "May I run a one-off grid probe on emu-quasar-2x3 (opens/closes the device once)? Please confirm NNG_SOCKET_ADDR / reservation." If approved:

```bash
TT_METAL_SIMULATOR=/proj_sw/user_dev/$USER/tt-umd-simulators/build/emu-quasar-2x3/ \
TT_METAL_SLOW_DISPATCH_MODE=1 python models/experimental/ops/quasar/qwen3_vl/tests/e2e/probe_grid.py
```
Record `EMU_GRID`. If not approved, ask the user to run it and paste the `PROBE` line.

- [ ] **Step 6: Find the override value reproducing EMU_GRID**

The override sets the grid *end coordinate* (`tt_metal/llrt/core_descriptor.cpp:199`). For `EMU_GRID=XxY` try `"$((X-1)),$((Y-1))"` first, then `"X,Y"`:

```bash
TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="<candidate>" TT_METAL_SIMULATOR=/localdev/$USER/ttsim/sim_wh/libttsim.so \
  python models/experimental/ops/quasar/qwen3_vl/tests/e2e/probe_grid.py
```
Repeat on craq-sim and ttsim BH. Expected: a value printing `grid=XxY` on all three. Record it as `GRID_OVERRIDE_2X3`.

- [ ] **Step 7: Commit**

```bash
git add models/experimental/ops/quasar/qwen3_vl/tests/e2e/__init__.py models/experimental/ops/quasar/qwen3_vl/tests/e2e/probe_grid.py
git commit -m "qwen3_vl quasar e2e: add grid probe"
```

---

### Task 2: RunConfig, presets, pytest options

**Files:**
- Create: `E2E/config.py`, `E2E/presets.py`, `E2E/conftest.py`, `E2E/test_harness_cpu.py`

**Interfaces:**
- Produces:
  - `RunConfig` (frozen dataclass) fields: `size: str`, `vision_layers: int`, `text_layers: int`, `decode_steps: int`, `deepstack_at: int | None`, `kv_blocks: int | None`, `host_ops: tuple[str, ...]`, `disable_wa: tuple[str, ...]`, `quasar_config: bool`, `expect_grid: tuple[int, int] | None`, `run_dir: Path`.
  - `RunConfig.from_options(getoption: Callable[[str], object]) -> RunConfig`
  - `RunConfig.cache_key(grid: tuple[int, int]) -> str`
  - `parse_grid(s: str | None) -> tuple[int, int] | None` (`"3x2"` → `(3, 2)`)
  - `Preset` (frozen): `name`, `max_seq_len`, `kv_blocks`, `block_size=32`, `make_image: Callable[[], PIL.Image.Image]`, `prompt: str`
  - `PRESETS: dict[str, Preset]` with `"tiny"`, `"demo"`
  - `build_inputs(preset, processor) -> transformers.BatchFeature` (keys `input_ids`, `attention_mask`, `pixel_values`, `image_grid_thw`)
  - fixture `qwen_run_config` → `RunConfig`
- HF model id constant: `HF_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"` in `config.py`.

- [ ] **Step 1: Write failing CPU tests** in `E2E/test_harness_cpu.py`

```python
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests of the e2e harness; no device needed."""
from pathlib import Path

import pytest

from models.experimental.ops.quasar.qwen3_vl.tests.e2e.config import HF_MODEL_ID, RunConfig, parse_grid
from models.experimental.ops.quasar.qwen3_vl.tests.e2e.presets import PRESETS, build_inputs


def _opts(**over):
    base = {
        "--qwen-size": "tiny",
        "--qwen-vision-layers": 2,
        "--qwen-text-layers": 2,
        "--qwen-decode-steps": 1,
        "--qwen-deepstack-at": None,
        "--qwen-kv-blocks": None,
        "--qwen-host-ops": "",
        "--qwen-disable-wa": "",
        "--qwen-quasar-config": False,
        "--qwen-expect-grid": None,
        "--qwen-run-dir": "/tmp/x",
    }
    base.update(over)
    return base.__getitem__


def test_run_config_defaults():
    cfg = RunConfig.from_options(_opts())
    assert (cfg.vision_layers, cfg.text_layers, cfg.decode_steps) == (2, 2, 1)
    assert cfg.deepstack_at is None and cfg.host_ops == () and cfg.run_dir == Path("/tmp/x")


def test_run_config_lists_and_grid():
    cfg = RunConfig.from_options(
        _opts(**{"--qwen-host-ops": "linear, rms_norm", "--qwen-expect-grid": "3x2", "--qwen-deepstack-at": 0})
    )
    assert cfg.host_ops == ("linear", "rms_norm")
    assert cfg.expect_grid == (3, 2) and cfg.deepstack_at == 0


def test_parse_grid_rejects_garbage():
    with pytest.raises(ValueError):
        parse_grid("3by2")


def test_cache_key_covers_everything():
    a = RunConfig.from_options(_opts())
    keys = {
        a.cache_key((3, 2)),
        a.cache_key((8, 4)),
        RunConfig.from_options(_opts(**{"--qwen-text-layers": 3})).cache_key((3, 2)),
        RunConfig.from_options(_opts(**{"--qwen-deepstack-at": 0})).cache_key((3, 2)),
        RunConfig.from_options(_opts(**{"--qwen-quasar-config": True})).cache_key((3, 2)),
    }
    assert len(keys) == 5


@pytest.mark.parametrize(
    "name, grid, image_tokens, seq",
    [("tiny", [1, 16, 16], 64, 78), ("demo", [1, 86, 128], 2752, 2766)],
)
def test_preset_token_counts(name, grid, image_tokens, seq):
    from transformers import AutoProcessor

    inputs = build_inputs(PRESETS[name], AutoProcessor.from_pretrained(HF_MODEL_ID))
    assert inputs["image_grid_thw"][0].tolist() == grid
    assert int(inputs["image_grid_thw"][0].prod()) // 4 == image_tokens
    assert inputs["input_ids"].shape[1] == seq


def test_preset_kv_capacity_covers_seq():
    for p in PRESETS.values():
        assert p.kv_blocks * p.block_size >= p.max_seq_len
```

- [ ] **Step 2: Run, expect import failure**

Run: `pytest models/experimental/ops/quasar/qwen3_vl/tests/e2e/test_harness_cpu.py -v`
Expected: FAIL (`ModuleNotFoundError: ...e2e.config`).

- [ ] **Step 3: Implement `E2E/config.py`**

```python
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Run configuration for the Qwen3-VL Quasar e2e test."""
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

HF_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"


def parse_grid(s):
    if s in (None, ""):
        return None
    m = re.fullmatch(r"(\d+)x(\d+)", str(s))
    if not m:
        raise ValueError(f"grid must look like 3x2, got {s!r}")
    return int(m.group(1)), int(m.group(2))


def _csv(s):
    return tuple(x.strip() for x in str(s or "").split(",") if x.strip())


@dataclass(frozen=True)
class RunConfig:
    size: str
    vision_layers: int
    text_layers: int
    decode_steps: int
    deepstack_at: int | None
    kv_blocks: int | None
    host_ops: tuple
    disable_wa: tuple
    quasar_config: bool
    expect_grid: tuple | None
    run_dir: Path

    @classmethod
    def from_options(cls, getoption: Callable[[str], object]) -> "RunConfig":
        ds = getoption("--qwen-deepstack-at")
        kv = getoption("--qwen-kv-blocks")
        return cls(
            size=str(getoption("--qwen-size")),
            vision_layers=int(getoption("--qwen-vision-layers")),
            text_layers=int(getoption("--qwen-text-layers")),
            decode_steps=int(getoption("--qwen-decode-steps")),
            deepstack_at=None if ds is None else int(ds),
            kv_blocks=None if kv is None else int(kv),
            host_ops=_csv(getoption("--qwen-host-ops")),
            disable_wa=_csv(getoption("--qwen-disable-wa")),
            quasar_config=bool(getoption("--qwen-quasar-config")),
            expect_grid=parse_grid(getoption("--qwen-expect-grid")),
            run_dir=Path(str(getoption("--qwen-run-dir"))),
        )

    def cache_key(self, grid) -> str:
        ds = "real" if self.deepstack_at is None else f"at{self.deepstack_at}"
        policy = "qcfg_bf16" if self.quasar_config else "native"
        return f"{policy}_g{grid[0]}x{grid[1]}_v{self.vision_layers}_t{self.text_layers}_ds{ds}"
```

- [ ] **Step 4: Implement `E2E/presets.py`**

```python
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Input size presets: tiny for fast iteration, demo for the captured graph sizes."""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image

_DEMO_JPEG = Path(__file__).resolve().parents[6] / "models" / "sample_data" / "demo.jpeg"
PROMPT = "Describe this image."


def _demo_image():
    return Image.open(_DEMO_JPEG).convert("RGB")


def _tiny_image():
    # 256x256 = the processor's min_pixels, so it is not resized: 256 patches -> 64 image tokens.
    return _demo_image().resize((256, 256), Image.BICUBIC)


@dataclass(frozen=True)
class Preset:
    name: str
    max_seq_len: int
    kv_blocks: int
    make_image: Callable
    prompt: str = PROMPT
    block_size: int = 32


PRESETS = {
    # 78 tokens pad to a 128 prefill; 8 blocks x 32 = 256 tokens of KV for prefill plus decode.
    "tiny": Preset("tiny", max_seq_len=256, kv_blocks=8, make_image=_tiny_image),
    # Captured demo sizes: 2766 tokens pad to 4096; 1024 blocks as in the graph capture.
    "demo": Preset("demo", max_seq_len=4096, kv_blocks=1024, make_image=_demo_image),
}


def build_inputs(preset: Preset, processor):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": preset.prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(text=[text], images=[preset.make_image()], padding=True, return_tensors="pt")
```

Verify the `parents[6]` depth: `python -c "from models.experimental.ops.quasar.qwen3_vl.tests.e2e.presets import _DEMO_JPEG; print(_DEMO_JPEG, _DEMO_JPEG.exists())"` must print `True`; adjust the index if not.

- [ ] **Step 5: Implement `E2E/conftest.py`**

```python
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest

from models.experimental.ops.quasar.qwen3_vl.tests.e2e.config import RunConfig


def pytest_addoption(parser):
    g = parser.getgroup("qwen3_vl_e2e")
    g.addoption("--qwen-size", default="tiny", choices=["tiny", "demo"])
    g.addoption("--qwen-vision-layers", type=int, default=2)
    g.addoption("--qwen-text-layers", type=int, default=2)
    g.addoption("--qwen-decode-steps", type=int, default=1)
    g.addoption("--qwen-deepstack-at", type=int, default=None)
    g.addoption("--qwen-kv-blocks", type=int, default=None, help="Paged KV-cache blocks of 32 tokens (default: preset).")
    g.addoption("--qwen-host-ops", default="")
    g.addoption("--qwen-disable-wa", default="")
    g.addoption("--qwen-quasar-config", action="store_true", default=False)
    g.addoption("--qwen-expect-grid", default=None)
    g.addoption("--qwen-run-dir", default="generated/qwen3_vl_quasar/adhoc")


@pytest.fixture
def qwen_run_config(request):
    return RunConfig.from_options(request.config.getoption)
```

- [ ] **Step 6: Run tests, expect PASS**

Run: `pytest models/experimental/ops/quasar/qwen3_vl/tests/e2e/test_harness_cpu.py -v`
Expected: all PASS (the preset test downloads only the processor files).

- [ ] **Step 7: Commit**

```bash
git add models/experimental/ops/quasar/qwen3_vl/tests/e2e/{config,presets,conftest,test_harness_cpu}.py
git commit -m "qwen3_vl quasar e2e: run config, size presets, pytest options"
```

---

### Task 3: Truncation, deepstack remap, vision padding (model copy)

**Files:**
- Create: `TT/quasar_config.py` (first part)
- Modify: `TT/model.py:270` and `TT/model.py:396` (`((x // 2048) + 1) * 2048`)
- Test: `E2E/test_harness_cpu.py`

**Interfaces:**
- Produces:
  - `vision_padded_seq_len(n: int) -> int` — ceil to 128 when `n <= 2048`, else ceil to 2048.
  - `truncate_hf_config(config, vision_layers: int, text_layers: int, deepstack_at: int | None) -> config` — mutates and returns an HF `Qwen3VLConfig`: `vision_config.depth`, `text_config.num_hidden_layers`, and if `deepstack_at` is not None `vision_config.deepstack_visual_indexes = [deepstack_at]`.

- [ ] **Step 1: Add failing tests** (append to `E2E/test_harness_cpu.py`)

```python
from models.experimental.ops.quasar.qwen3_vl.tt.quasar_config import truncate_hf_config, vision_padded_seq_len


@pytest.mark.parametrize("n, want", [(1, 128), (216, 256), (256, 256), (2048, 2048), (2049, 4096), (11008, 12288)])
def test_vision_padded_seq_len(n, want):
    assert vision_padded_seq_len(n) == want


def test_truncate_hf_config_same_for_tt_and_hf():
    from transformers import AutoConfig

    a = truncate_hf_config(AutoConfig.from_pretrained(HF_MODEL_ID), 2, 2, 0)
    b = truncate_hf_config(AutoConfig.from_pretrained(HF_MODEL_ID), 2, 2, 0)
    assert a.vision_config.depth == b.vision_config.depth == 2
    assert a.text_config.num_hidden_layers == 2
    assert a.vision_config.deepstack_visual_indexes == b.vision_config.deepstack_visual_indexes == [0]


def test_truncate_hf_config_keeps_real_taps():
    from transformers import AutoConfig

    c = truncate_hf_config(AutoConfig.from_pretrained(HF_MODEL_ID), 2, 2, None)
    assert c.vision_config.deepstack_visual_indexes == [5, 11, 17]
```

- [ ] **Step 2: Run, expect FAIL** (`ModuleNotFoundError: ...tt.quasar_config`)

Run: `pytest models/experimental/ops/quasar/qwen3_vl/tests/e2e/test_harness_cpu.py -v -k "padded or truncate"`

- [ ] **Step 3: Create `TT/quasar_config.py`**

```python
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Quasar bring-up configuration for the Qwen3-VL copy: bf16, device-derived grids, truncation."""
import math


def vision_padded_seq_len(n: int) -> int:
    # vision_attention needs a multiple of 128, and of 2048 once the sequence exceeds 2048.
    step = 128 if n <= 2048 else 2048
    return math.ceil(n / step) * step


def truncate_hf_config(config, vision_layers, text_layers, deepstack_at=None):
    config.vision_config.depth = vision_layers
    config.text_config.num_hidden_layers = text_layers
    if deepstack_at is not None:
        config.vision_config.deepstack_visual_indexes = [deepstack_at]
    return config
```

- [ ] **Step 4: Use it in `TT/model.py`**

At both sites (line ~270 in `forward`, ~396 in `forward_single_user`) replace
```python
            seq_len = ((unpadded_seq_len // 2048) + 1) * 2048
```
with
```python
            seq_len = vision_padded_seq_len(unpadded_seq_len)
```
(keep each site's indentation) and update the adjacent comment to `# Pad to what vision_attention accepts (see vision_padded_seq_len)`. Add the import at the top of `TT/model.py`:
```python
from models.experimental.ops.quasar.qwen3_vl.tt.quasar_config import vision_padded_seq_len
```

- [ ] **Step 5: Run tests, expect PASS**

Run: `pytest models/experimental/ops/quasar/qwen3_vl/tests/e2e/test_harness_cpu.py -v`

- [ ] **Step 6: Commit**

```bash
git add models/experimental/ops/quasar/qwen3_vl/tt/quasar_config.py models/experimental/ops/quasar/qwen3_vl/tt/model.py models/experimental/ops/quasar/qwen3_vl/tests/e2e/test_harness_cpu.py
git commit -m "qwen3_vl quasar: pad vision patches to what attention needs; truncation helper"
```

---

### Task 4: HF host reference and golden capture

**Files:**
- Create: `E2E/host_reference.py`
- Test: `E2E/test_harness_cpu.py`

**Interfaces:**
- Consumes: `truncate_hf_config` (Task 3), `HF_MODEL_ID`, `build_inputs` (Task 2).
- Produces:
  - `load_hf_model(vision_layers, text_layers, deepstack_at) -> Qwen3VLForConditionalGeneration` (float32, eval)
  - `@dataclass Goldens`: `tensors: dict[str, torch.Tensor]` (float32, unpadded), `teacher_tokens: list[int]` (len K), `prefill_len: int`, `num_patches: int`
  - `run_reference(model, inputs, decode_steps: int) -> Goldens`
- Golden shapes: `vision.block{i}` `[num_patches, 1024]`; `vision.deepstack{j}` and `vision.merger` `[num_patches/4, 2560]`; `text.layer{i}` `[prefill_len, 2560]` (before the deepstack add); `text.norm` `[2560]` (last real token); `text.logits.prefill` and `text.logits.decode{k}` `[151936]`.

- [ ] **Step 1: Add failing test**

```python
@pytest.mark.slow
def test_reference_tiny_stage_shapes():
    from transformers import AutoProcessor

    from models.experimental.ops.quasar.qwen3_vl.tests.e2e.host_reference import load_hf_model, run_reference

    inputs = build_inputs(PRESETS["tiny"], AutoProcessor.from_pretrained(HF_MODEL_ID))
    g = run_reference(load_hf_model(2, 2, 0), inputs, decode_steps=2)
    assert g.prefill_len == 78 and g.num_patches == 256 and len(g.teacher_tokens) == 2
    assert g.tensors["vision.block1"].shape == (256, 1024)
    assert g.tensors["vision.deepstack0"].shape == (64, 2560)
    assert g.tensors["vision.merger"].shape == (64, 2560)
    assert g.tensors["text.layer1"].shape == (78, 2560)
    assert g.tensors["text.norm"].shape == (2560,)
    assert g.tensors["text.logits.decode1"].shape == (151936,)
    assert set(g.tensors) == {
        "vision.block0", "vision.block1", "vision.deepstack0", "vision.merger",
        "text.layer0", "text.layer1", "text.norm",
        "text.logits.prefill", "text.logits.decode0", "text.logits.decode1",
    }
```

- [ ] **Step 2: Run, expect FAIL**

Run: `pytest models/experimental/ops/quasar/qwen3_vl/tests/e2e/test_harness_cpu.py -v -k reference`

- [ ] **Step 3: Implement `E2E/host_reference.py`**

```python
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Truncated HF Qwen3-VL reference: runs prefill plus teacher-forced decode and keeps per-stage goldens."""
from dataclasses import dataclass, field

import torch
from transformers import AutoConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

from models.experimental.ops.quasar.qwen3_vl.tests.e2e.config import HF_MODEL_ID
from models.experimental.ops.quasar.qwen3_vl.tt.quasar_config import truncate_hf_config


@dataclass
class Goldens:
    tensors: dict = field(default_factory=dict)
    teacher_tokens: list = field(default_factory=list)
    prefill_len: int = 0
    num_patches: int = 0


def load_hf_model(vision_layers, text_layers, deepstack_at):
    config = truncate_hf_config(AutoConfig.from_pretrained(HF_MODEL_ID), vision_layers, text_layers, deepstack_at)
    model = Qwen3VLForConditionalGeneration.from_pretrained(HF_MODEL_ID, config=config, torch_dtype=torch.float32)
    return model.eval()


def _first(out):
    return out[0] if isinstance(out, (tuple, list)) else out


@torch.no_grad()
def run_reference(model, inputs, decode_steps):
    g = Goldens(prefill_len=int(inputs["input_ids"].shape[1]), num_patches=int(inputs["image_grid_thw"][0].prod()))
    visual, lm = model.model.visual, model.model.language_model
    hooks = []

    def keep(name, fn=lambda t: t):
        def hook(_mod, _inp, out):
            g.tensors[name] = fn(_first(out)).detach().float().clone()

        return hook

    for i, blk in enumerate(visual.blocks):
        hooks.append(blk.register_forward_hook(keep(f"vision.block{i}", lambda t: t.reshape(-1, t.shape[-1]))))
    taps = [i for i in visual.deepstack_visual_indexes if i < len(visual.blocks)]
    for j, _ in enumerate(taps):
        hooks.append(visual.deepstack_merger_list[j].register_forward_hook(keep(f"vision.deepstack{j}")))
    hooks.append(visual.merger.register_forward_hook(keep("vision.merger")))
    for i, layer in enumerate(lm.layers):
        hooks.append(layer.register_forward_hook(keep(f"text.layer{i}", lambda t: t.reshape(-1, t.shape[-1]))))
    hooks.append(lm.norm.register_forward_hook(keep("text.norm", lambda t: t.reshape(-1, t.shape[-1])[-1])))
    try:
        out = model(**inputs)
    finally:
        for h in hooks:
            h.remove()
    g.tensors["text.logits.prefill"] = out.logits[0, -1].float().clone()

    ids = inputs["input_ids"]
    tok = int(out.logits[0, -1].argmax())
    for k in range(decode_steps):
        g.teacher_tokens.append(tok)
        ids = torch.cat([ids, torch.tensor([[tok]])], dim=1)
        step = model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            pixel_values=inputs["pixel_values"],
            image_grid_thw=inputs["image_grid_thw"],
        )
        g.tensors[f"text.logits.decode{k}"] = step.logits[0, -1].float().clone()
        tok = int(step.logits[0, -1].argmax())
    return g
```

- [ ] **Step 4: Run, expect PASS**

Run: `pytest models/experimental/ops/quasar/qwen3_vl/tests/e2e/test_harness_cpu.py -v -k reference`
If a hook output shape differs (e.g. HF 5.x vision blocks returning a tuple, or the merger returning `[1, N, D]`), fix the `fn` reshape so the asserted shapes hold. Do not change the asserted shapes.

- [ ] **Step 5: Commit**

```bash
git add models/experimental/ops/quasar/qwen3_vl/tests/e2e/host_reference.py models/experimental/ops/quasar/qwen3_vl/tests/e2e/test_harness_cpu.py
git commit -m "qwen3_vl quasar e2e: truncated HF reference with per-stage goldens"
```

---

### Task 5: PCC compare, thresholds, verdict

**Files:**
- Create: `E2E/pcc.py`, `E2E/thresholds.json`
- Test: `E2E/test_harness_cpu.py`

**Interfaces:**
- Produces:
  - `pcc(a: torch.Tensor, b: torch.Tensor) -> float`
  - `@dataclass StageResult`: `stage: str`, `pcc: float`, `max_abs: float`, `threshold: float`, `status: str` (`PASS`/`FAIL`/`NONFINITE`/`MISSING`/`SHAPE`), `seconds: float`
  - `thresholds_for(preset: str, path: Path = THRESHOLDS_JSON) -> Callable[[str], float]`
  - `compare(goldens: dict, actual: dict, threshold: Callable, seconds: dict) -> list[StageResult]`
  - `@dataclass Verdict`: `status` (`PASS`/`FAIL`/`DIAGNOSTIC`), `first_failure: str | None`, `markdown: str`
  - `verdict(results, host_ops: list[str], override_hits: dict[str, int], notes: list[str]) -> Verdict`

- [ ] **Step 1: Add failing tests**

```python
import torch

from models.experimental.ops.quasar.qwen3_vl.tests.e2e import pcc as P


def test_pcc_identical_and_constant():
    x = torch.randn(64, 32)
    assert P.pcc(x, x) == pytest.approx(1.0)
    assert P.pcc(torch.ones(4), torch.ones(4)) == 1.0
    assert P.pcc(torch.ones(4), torch.zeros(4)) == 0.0


def test_compare_statuses():
    g = {"a": torch.randn(10, 4), "b": torch.randn(10, 4), "c": torch.randn(3), "d": torch.randn(3)}
    act = {"a": g["a"].clone(), "b": torch.randn(10, 4), "c": torch.tensor([1.0, float("nan"), 0.0]), "d": torch.randn(4)}
    res = {r.stage: r.status for r in P.compare(g, act, lambda s: 0.99, {})}
    assert res == {"a": "PASS", "b": "FAIL", "c": "NONFINITE", "d": "SHAPE"}
    act.pop("a")
    assert {r.stage: r.status for r in P.compare(g, act, lambda s: 0.99, {})}["a"] == "MISSING"


def test_verdict_first_failure_in_stage_order():
    rs = [P.StageResult("vision.block0", 1, 0, 0.99, "PASS", 0), P.StageResult("text.layer0", 0.5, 1, 0.99, "FAIL", 0),
          P.StageResult("text.layer1", 0.4, 1, 0.99, "FAIL", 0)]
    v = P.verdict(rs, [], {}, [])
    assert v.status == "FAIL" and v.first_failure == "text.layer0" and "text.layer0" in v.markdown


def test_verdict_host_ops_is_diagnostic_even_when_all_pass():
    rs = [P.StageResult("vision.block0", 1, 0, 0.99, "PASS", 0)]
    v = P.verdict(rs, ["ttnn.linear"], {"host:ttnn.linear": 3}, [])
    assert v.status == "DIAGNOSTIC" and "ttnn.linear" in v.markdown


def test_thresholds_patterns_and_preset_override(tmp_path):
    f = tmp_path / "t.json"
    f.write_text('{"default": {"text.layer*": 0.99, "text.logits.*": 0.98}, "demo": {"text.layer*": 0.97}}')
    assert P.thresholds_for("tiny", f)("text.layer1") == 0.99
    assert P.thresholds_for("demo", f)("text.layer1") == 0.97
    assert P.thresholds_for("tiny", f)("text.logits.decode0") == 0.98
    with pytest.raises(KeyError):
        P.thresholds_for("tiny", f)("unknown.stage")
```

- [ ] **Step 2: Run, expect FAIL**

Run: `pytest models/experimental/ops/quasar/qwen3_vl/tests/e2e/test_harness_cpu.py -v -k "pcc or compare or verdict or thresholds"`

- [ ] **Step 3: Implement `E2E/thresholds.json`**

```json
{
  "default": {
    "vision.block*": 0.99,
    "vision.deepstack*": 0.99,
    "vision.merger": 0.99,
    "text.layer*": 0.99,
    "text.norm": 0.99,
    "text.logits.prefill": 0.98,
    "text.logits.decode*": 0.98
  },
  "tiny": {},
  "demo": {}
}
```

- [ ] **Step 4: Implement `E2E/pcc.py`**

```python
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Per-stage PCC comparison, thresholds and the run verdict."""
import fnmatch
import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch

THRESHOLDS_JSON = Path(__file__).with_name("thresholds.json")
_ORDER = ["vision.block", "vision.deepstack", "vision.merger", "text.layer", "text.norm", "text.logits.prefill", "text.logits.decode"]


def _stage_sort_key(stage):
    prefix = re.sub(r"\d+$", "", stage)
    idx = int(re.search(r"(\d+)$", stage).group(1)) if re.search(r"\d+$", stage) else 0
    return (_ORDER.index(prefix) if prefix in _ORDER else len(_ORDER), idx)


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    if a.std() == 0 or b.std() == 0:
        return 1.0 if torch.equal(a, b) else 0.0
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


@dataclass
class StageResult:
    stage: str
    pcc: float
    max_abs: float
    threshold: float
    status: str
    seconds: float


def thresholds_for(preset, path=THRESHOLDS_JSON):
    data = json.loads(Path(path).read_text())
    table = {**data["default"], **data.get(preset, {})}

    def lookup(stage):
        hits = [p for p in table if fnmatch.fnmatchcase(stage, p)]
        if not hits:
            raise KeyError(f"no threshold for stage {stage}")
        return table[max(hits, key=len)]

    return lookup


def compare(goldens, actual, threshold, seconds):
    out = []
    for stage in sorted(goldens, key=_stage_sort_key):
        g, t, thr, sec = goldens[stage], actual.get(stage), threshold(stage), seconds.get(stage, 0.0)
        if t is None:
            out.append(StageResult(stage, float("nan"), float("nan"), thr, "MISSING", sec))
        elif tuple(t.shape) != tuple(g.shape):
            out.append(StageResult(stage, float("nan"), float("nan"), thr, "SHAPE", sec))
        elif not torch.isfinite(t).all():
            out.append(StageResult(stage, float("nan"), float("nan"), thr, "NONFINITE", sec))
        else:
            p = pcc(g, t)
            mx = float((g.float() - t.float()).abs().max())
            out.append(StageResult(stage, p, mx, thr, "PASS" if p >= thr else "FAIL", sec))
    return out


@dataclass
class Verdict:
    status: str
    first_failure: str | None
    markdown: str


def verdict(results, host_ops, override_hits, notes):
    failed = [r for r in results if r.status != "PASS"]
    status = "DIAGNOSTIC" if host_ops else ("FAIL" if failed else "PASS")
    first = failed[0].stage if failed else None
    lines = [f"# Verdict: {status}", ""]
    if first:
        lines += [f"**First failing stage:** `{first}`", ""]
    if host_ops:
        lines += [f"**Ops on host (not a pass):** {', '.join(host_ops)}", ""]
    lines += ["| stage | status | pcc | threshold | max abs | seconds |", "|---|---|---|---|---|---|"]
    lines += [f"| {r.stage} | {r.status} | {r.pcc:.5f} | {r.threshold} | {r.max_abs:.4g} | {r.seconds:.1f} |" for r in results]
    if override_hits:
        lines += ["", "| override | hits |", "|---|---|"] + [f"| {k} | {v} |" for k, v in sorted(override_hits.items())]
    if notes:
        lines += [""] + [f"- {n}" for n in notes]
    return Verdict(status, first, "\n".join(lines) + "\n")
```

- [ ] **Step 5: Run, expect PASS**

Run: `pytest models/experimental/ops/quasar/qwen3_vl/tests/e2e/test_harness_cpu.py -v`

- [ ] **Step 6: Commit**

```bash
git add models/experimental/ops/quasar/qwen3_vl/tests/e2e/pcc.py models/experimental/ops/quasar/qwen3_vl/tests/e2e/thresholds.json models/experimental/ops/quasar/qwen3_vl/tests/e2e/test_harness_cpu.py
git commit -m "qwen3_vl quasar e2e: per-stage PCC, thresholds and verdict"
```

---

### Task 6: Progress log and stage recorder

**Files:**
- Create: `E2E/progress.py`, `E2E/recorder.py`
- Test: `E2E/test_harness_cpu.py`

**Interfaces:**
- Produces:
  - `class ProgressLog(path: Path)`: `.stage: str` (current model stage), `pre(operation, args, kwargs)`, `post(operation, args, kwargs, output)`, `installed()` context manager (registers ttnn hooks), `last_unfinished() -> str | None`, `hooks_active() -> bool` (False if `ttnn.CONFIG.enable_fast_runtime_mode`)
  - `class StageRecorder(progress: ProgressLog)`: `.tensors: dict[str, torch.Tensor]`, `.seconds: dict[str, float]`, `wrap(monkeypatch, obj, stage: str, transform: Callable[[torch.Tensor], torch.Tensor], when: Callable[[tuple, dict], bool] = always, append_dim: int | None = None)` — replaces `obj.forward` (instance attribute) so the first tensor output is copied to host, transformed and stored; with `append_dim` set, repeated calls concatenate (chunked prefill).
  - `to_host(t) -> torch.Tensor` (float32, first device of a 1x1 mesh)

- [ ] **Step 1: Add failing tests** (CPU, fake ops/modules; `to_host` is injectable)

```python
from models.experimental.ops.quasar.qwen3_vl.tests.e2e.progress import ProgressLog
from models.experimental.ops.quasar.qwen3_vl.tests.e2e.recorder import StageRecorder


class _Op:
    python_fully_qualified_name = "ttnn.fake"


def test_progress_log_unfinished(tmp_path):
    log = ProgressLog(tmp_path / "progress.log")
    log.stage = "text.layer0"
    log.pre(_Op(), (torch.zeros(2, 3),), {})
    log.post(_Op(), (torch.zeros(2, 3),), {}, None)
    log.pre(_Op(), (torch.zeros(4),), {})
    assert log.last_unfinished().startswith("ttnn.fake")
    text = (tmp_path / "progress.log").read_text()
    assert "PRE ttnn.fake stage=text.layer0" in text and "POST ttnn.fake" in text


def test_recorder_wrap_transform_append(tmp_path, monkeypatch):
    class Mod:
        def forward(self, x, mode="prefill"):
            return x * 2

    m = Mod()
    rec = StageRecorder(ProgressLog(tmp_path / "p.log"), to_host=lambda t: t.float())
    rec.wrap(monkeypatch, m, "text.layer0", lambda t: t[:3], when=lambda a, k: k.get("mode") == "prefill", append_dim=0)
    m.forward(torch.ones(4, 2))
    m.forward(torch.ones(4, 2), mode="decode")
    m.forward(torch.ones(4, 2))
    assert rec.tensors["text.layer0"].shape == (6, 2)
    assert rec.seconds["text.layer0"] >= 0
```

- [ ] **Step 2: Run, expect FAIL**

Run: `pytest models/experimental/ops/quasar/qwen3_vl/tests/e2e/test_harness_cpu.py -v -k "progress or recorder"`

- [ ] **Step 3: Implement `E2E/progress.py`**

```python
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Append one line per ttnn op start/end so a hang points at the op that never finished."""
import time
from contextlib import contextmanager


def _name(operation):
    return getattr(operation, "python_fully_qualified_name", None) or getattr(operation, "__name__", repr(operation))


def _describe(x):
    shape = getattr(x, "shape", None)
    if shape is None:
        return type(x).__name__
    parts = [f"shape={list(shape)}", f"dtype={getattr(x, 'dtype', '?')}"]
    mc = getattr(x, "memory_config", None)
    if callable(mc):
        try:
            m = mc()
            parts.append(f"mem={m.memory_layout}/{m.buffer_type}")
        except Exception:
            pass
    return "(" + ", ".join(parts) + ")"


class ProgressLog:
    def __init__(self, path):
        self.path = path
        self.stage = "-"
        self._open = []
        self._f = open(path, "a", buffering=1)

    def _write(self, kind, operation, args):
        args_s = " ".join(_describe(a) for a in args)
        self._f.write(f"{time.time():.3f} {kind} {_name(operation)} stage={self.stage} {args_s}\n")

    def pre(self, operation, args, kwargs):
        self._open.append(f"{_name(operation)} stage={self.stage} " + " ".join(_describe(a) for a in args))
        self._write("PRE", operation, args)

    def post(self, operation, args, kwargs, output):
        if self._open:
            self._open.pop()
        self._write("POST", operation, args)

    def last_unfinished(self):
        return self._open[-1] if self._open else None

    @staticmethod
    def hooks_active():
        import ttnn

        return not ttnn.CONFIG.enable_fast_runtime_mode

    @contextmanager
    def installed(self):
        import ttnn

        with ttnn.register_pre_operation_hook(self.pre), ttnn.register_post_operation_hook(self.post):
            yield self
```

- [ ] **Step 4: Implement `E2E/recorder.py`**

```python
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Capture per-stage TT outputs by wrapping module instances' forward."""
import time

import torch


def to_host(t):
    import ttnn

    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()


def _first(out):
    return out[0] if isinstance(out, (tuple, list)) else out


class StageRecorder:
    def __init__(self, progress, to_host=to_host):
        self.progress = progress
        self.to_host = to_host
        self.tensors = {}
        self.seconds = {}

    def wrap(self, monkeypatch, obj, stage, transform, when=lambda a, k: True, append_dim=None):
        orig = obj.forward

        def forward(*args, **kwargs):
            if not when(args, kwargs):
                return orig(*args, **kwargs)
            prev, self.progress.stage = self.progress.stage, stage
            t0 = time.time()
            try:
                out = orig(*args, **kwargs)
            finally:
                self.progress.stage = prev
            self.seconds[stage] = self.seconds.get(stage, 0.0) + time.time() - t0
            value = transform(self.to_host(_first(out)))
            if append_dim is not None and stage in self.tensors:
                value = torch.cat([self.tensors[stage], value], dim=append_dim)
            self.tensors[stage] = value
            return out

        monkeypatch.setattr(obj, "forward", forward)
```

- [ ] **Step 5: Run, expect PASS**

Run: `pytest models/experimental/ops/quasar/qwen3_vl/tests/e2e/test_harness_cpu.py -v`

- [ ] **Step 6: Commit**

```bash
git add models/experimental/ops/quasar/qwen3_vl/tests/e2e/progress.py models/experimental/ops/quasar/qwen3_vl/tests/e2e/recorder.py models/experimental/ops/quasar/qwen3_vl/tests/e2e/test_harness_cpu.py
git commit -m "qwen3_vl quasar e2e: op progress log and stage recorder"
```

---

### Task 7: Quasar model args — device name, bf16 policy, args-driven vision dtypes

**Files:**
- Modify: `TT/quasar_config.py` (append), `TT/model_config.py` (`VisionModelArgs.__init__`), `TT/vision_attention.py` (lines ~178, 231, 450, 456, 509), `TT/vision_mlp.py` (lines ~56, 59), `TT/vision_layernorm.py` (lines ~94, 108), `TT/model.py:198`
- Test: `E2E/test_harness_cpu.py`

**Interfaces:**
- Consumes: `models.tt_transformers.tt.model_config` (`ModelArgs`, `DecodersPrecision`, `ModelOptimizations`, `TensorGroup`, `OpGroup`, `PrecisionSetting`, `MathFidelitySetting`), `TT/model_config.VisionModelArgs`.
- Produces:
  - `bf16_decoders_precision(num_decoders: int, model_name: str) -> DecodersPrecision`
  - `class QuasarModelArgs(ModelArgs)` and `class QuasarVisionModelArgs(VisionModelArgs)`; both accept the parent's args, default `optimizations` to bf16, set `lm_head_dtype = ccl_dtype = ttnn.bfloat16`.
  - `model_args_classes(force: bool) -> tuple[type, type]` → `(QuasarModelArgs, QuasarVisionModelArgs)` if `force or is_quasar()` else `(ModelArgs, VisionModelArgs)`.
  - New `VisionModelArgs` attributes (defaults keep today's behavior): `vision_weight_dtype = ttnn.bfloat8_b`, `vision_mlp_fc1_dtype` (bf4 if `bfp4_mlp` else bf8), `vision_compute_kernel_config(math_fidelity, fp32_dest_acc_en, packer_l1_acc)` method.
  - `QUASAR_DEVICE_NAME = "N150"`.

- [ ] **Step 1: Add failing test**

```python
def test_bf16_precision_everywhere():
    from models.tt_transformers.tt.model_config import OpGroup, TensorGroup
    import ttnn

    from models.experimental.ops.quasar.qwen3_vl.tt.quasar_config import bf16_decoders_precision

    p = bf16_decoders_precision(2, "Qwen3-VL-4B-Instruct")
    for d in range(2):
        for g in TensorGroup:
            assert p.get_tensor_dtype(d, g) == ttnn.bfloat16, g
    conf = p.decoder_optimizations[0]
    assert all(v.value == "hifi4" for v in conf.op_fidelity_settings.values())
```

- [ ] **Step 2: Run, expect FAIL**

Run: `pytest models/experimental/ops/quasar/qwen3_vl/tests/e2e/test_harness_cpu.py -v -k bf16`

- [ ] **Step 3: Append to `TT/quasar_config.py`**

```python
from contextlib import contextmanager

import ttnn
from models.common.utility_functions import is_quasar
from models.tt_transformers.tt import model_config as _ttt_mc
from models.tt_transformers.tt.model_config import (
    DecodersPrecision,
    MathFidelitySetting,
    ModelArgs,
    ModelOptimizations,
    OpGroup,
    PrecisionSetting,
    TensorGroup,
)
from models.experimental.ops.quasar.qwen3_vl.tt.model_config import VisionModelArgs

# ModelArgs keys tuning tables by device name; Quasar has none, so it borrows the single-chip WH entries.
QUASAR_DEVICE_NAME = "N150"


def bf16_decoders_precision(num_decoders, model_name):
    settings = {
        "TensorPrecision": {g: PrecisionSetting.BF16 for g in TensorGroup},
        "OpFidelity": {g: MathFidelitySetting.HIFI4 for g in OpGroup},
    }
    return DecodersPrecision(num_decoders, model_name, ModelOptimizations(settings))


@contextmanager
def _quasar_device_name():
    orig = _ttt_mc.determine_device_name

    def name(mesh_device):
        try:
            return orig(mesh_device)
        except ValueError:
            return QUASAR_DEVICE_NAME

    _ttt_mc.determine_device_name = name
    try:
        yield
    finally:
        _ttt_mc.determine_device_name = orig


class _QuasarArgsMixin:
    def _quasar_init(self, parent_init, *args, **kwargs):
        kwargs.setdefault("optimizations", lambda a: bf16_decoders_precision(a.n_layers, a.model_name))
        with _quasar_device_name():
            parent_init(self, *args, **kwargs)
        self.lm_head_dtype = ttnn.bfloat16
        self.ccl_dtype = ttnn.bfloat16


class QuasarModelArgs(_QuasarArgsMixin, ModelArgs):
    def __init__(self, *args, **kwargs):
        self._quasar_init(ModelArgs.__init__, *args, **kwargs)


class QuasarVisionModelArgs(_QuasarArgsMixin, VisionModelArgs):
    def __init__(self, *args, **kwargs):
        self._quasar_init(VisionModelArgs.__init__, *args, **kwargs)
        self.vision_weight_dtype = ttnn.bfloat16
        self.vision_mlp_fc1_dtype = ttnn.bfloat16


def model_args_classes(force=False):
    if force or is_quasar():
        return QuasarModelArgs, QuasarVisionModelArgs
    return ModelArgs, VisionModelArgs
```

Check before moving on: `grep -n "lm_head_dtype\|ccl_dtype" models/tt_transformers/tt/model_config.py models/tt_transformers/tt/lm_head.py` — confirm both attribute names; if `ccl_dtype` is a property, set the backing attribute it reads instead.

- [ ] **Step 4: Add args-driven dtypes to `VisionModelArgs`** (`TT/model_config.py`, end of `__init__`)

```python
        # Weight dtypes for the vision tower; QuasarVisionModelArgs overrides both to bf16.
        self.vision_weight_dtype = ttnn.bfloat8_b
        self.vision_mlp_fc1_dtype = ttnn.bfloat4_b if self.optimizations.bfp4_mlp else ttnn.bfloat8_b

    def vision_compute_kernel_config(self, math_fidelity, fp32_dest_acc_en, packer_l1_acc):
        return ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=math_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=fp32_dest_acc_en,
            packer_l1_acc=packer_l1_acc,
        )
```

- [ ] **Step 5: Replace literals at the call sites**

Open each site and replace only the literal:
- `TT/vision_attention.py` ~178 (`wqkv_bias` dtype) and ~231 (`wqkv` dtype): `ttnn.bfloat8_b` → `self.args.vision_weight_dtype` (use whatever the module calls its args, e.g. `configuration`; check `__init__`).
- `TT/vision_attention.py` ~450, ~456 (q/v typecast) and ~509 (wo output fallback): `ttnn.bfloat8_b` → `self.args.vision_weight_dtype`.
- `TT/vision_mlp.py` ~56 (fc1): the `bfloat4_b if ... else bfloat8_b` expression → `args.vision_mlp_fc1_dtype`; ~59 (fc2): `ttnn.bfloat8_b` → `args.vision_weight_dtype`.
- `TT/vision_layernorm.py` ~94, ~108: `ttnn.WormholeComputeKernelConfig(math_fidelity=X, math_approx_mode=..., fp32_dest_acc_en=Y, packer_l1_acc=Z)` → `args.vision_compute_kernel_config(X, Y, Z)` (keep the original values).
- `TT/model.py:198`: `dtype=ttnn.bfloat8_b` default → `dtype=None`, and at the top of `DropInVisionTransformer.__init__` add `dtype = dtype if dtype is not None else model_args.vision_weight_dtype`.

Run `grep -n "bfloat8_b\|bfloat4_b\|WormholeComputeKernelConfig" models/experimental/ops/quasar/qwen3_vl/tt/*.py` afterwards. Expected: hits only in `model_config.py` defaults and in `attention.py` (dead copy, not imported).

- [ ] **Step 6: Run CPU tests, expect PASS**

Run: `pytest models/experimental/ops/quasar/qwen3_vl/tests/e2e/test_harness_cpu.py -v`

- [ ] **Step 7: Commit**

```bash
git add models/experimental/ops/quasar/qwen3_vl/tt/
git add models/experimental/ops/quasar/qwen3_vl/tests/e2e/test_harness_cpu.py
git commit -m "qwen3_vl quasar: Quasar model args with bf16 policy and args-driven vision dtypes"
```

---

### Task 8: TT runner and the e2e test (native grid on ttsim WH)

**Files:**
- Create: `E2E/tt_runner.py`, `E2E/test_qwen3_vl_e2e.py`

**Interfaces:**
- Consumes: `RunConfig`, `PRESETS`, `build_inputs`, `load_hf_model`, `run_reference`, `Goldens`, `thresholds_for`, `compare`, `verdict`, `ProgressLog`, `StageRecorder`, `model_args_classes`, `truncate_hf_config`. `OverrideSession` arrives in Task 11; until then the test passes `overrides=None`.
- Produces: `run_tt(cfg, preset, inputs, hf_model, goldens, mesh_device, recorder, monkeypatch) -> None` (fills `recorder.tensors` with all stage names that `goldens.tensors` has); `test_qwen3_vl_e2e`; writes `<run_dir>/pcc.md`, `<run_dir>/verdict.txt`, `<run_dir>/progress.log`.

- [ ] **Step 1: Confirm two facts the runner relies on**

```bash
grep -n "get_last_token" models/tt_transformers/tt/generator.py | head
grep -n "def decode_forward" -A30 models/experimental/ops/quasar/qwen3_vl/tt/generator.py | grep -n "return"
```
Record: (a) the formula for the 32-row slice start used for `last_token_idx` (expected `(last_token_idx // 32) * 32`); (b) what `decode_forward` returns when `sampling_params=None` (expected `(logits, log_probs)` with logits `[B, 1, vocab]` on host). If either differs, adapt `_norm_row` / `_logits_vec` below.

- [ ] **Step 2: Implement `E2E/tt_runner.py`**

```python
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TT pipeline for one user: vision -> merge -> prefill -> teacher-forced decode, mirroring demo/demo.py."""
import torch

import ttnn
from models.experimental.ops.quasar.qwen3_vl.tt.common import (
    PagedAttentionConfig,
    get_hf_visual,
    get_pad_embedding,
    merge_vision_tokens_single_user_ttnn,
    multimodal_rope_single_user_from_hf,
    preprocess_inputs_prefill_single_user_ttnn,
)
from models.experimental.ops.quasar.qwen3_vl.tt.generator import Generator
from models.experimental.ops.quasar.qwen3_vl.tt.model import DropInVisionTransformer, Transformer
from models.experimental.ops.quasar.qwen3_vl.tt.quasar_config import model_args_classes


def _page_table(cfg_blocks, batch):
    perm = torch.randperm(cfg_blocks)
    return torch.argsort(perm).reshape(batch, cfg_blocks // batch)


def _logits_vec(t, vocab):
    t = t if isinstance(t, torch.Tensor) else ttnn.to_torch(t)
    return t.float().reshape(-1, t.shape[-1])[0, :vocab]


def run_tt(cfg, preset, inputs, hf_model, goldens, mesh_device, recorder, monkeypatch):
    text_cls, vision_cls = model_args_classes(force=cfg.quasar_config)
    hf_cfg = hf_model.config
    n_patches, n_img, L = goldens.num_patches, goldens.num_patches // 4, goldens.prefill_len
    vocab = hf_cfg.text_config.vocab_size
    kv_blocks = cfg.kv_blocks or preset.kv_blocks

    # --- vision ---
    vargs = vision_cls(mesh_device, max_batch_size=1, max_seq_len=preset.max_seq_len)
    vargs.hf_config.vision_config.depth = hf_cfg.vision_config.depth
    vargs.hf_config.vision_config.deepstack_visual_indexes = list(hf_cfg.vision_config.deepstack_visual_indexes)
    visual = DropInVisionTransformer(get_hf_visual(hf_model), vargs)
    tv = visual.tt_model
    for i, blk in enumerate(tv.blocks):
        recorder.wrap(monkeypatch, blk, f"vision.block{i}", lambda t: t.reshape(-1, t.shape[-1])[:n_patches])
    taps = [i for i in tv.deepstack_visual_indices if i < len(tv.blocks)]
    for j, _ in enumerate(taps):
        recorder.wrap(monkeypatch, tv.deepstack_merger_list[j], f"vision.deepstack{j}",
                      lambda t: t.reshape(-1, t.shape[-1])[:n_img, :2560])
    recorder.wrap(monkeypatch, tv.patch_merger, "vision.merger", lambda t: t.reshape(-1, t.shape[-1])[:n_img, :2560])

    # --- text ---
    args = text_cls(mesh_device, instruct=True, max_batch_size=1, max_seq_len=preset.max_seq_len)
    args.n_layers = cfg.text_layers
    state_dict = args.load_state_dict()
    paged = PagedAttentionConfig(block_size=preset.block_size, max_num_blocks=kv_blocks)
    model = Transformer(
        args=args, mesh_device=mesh_device, dtype=ttnn.bfloat16 if cfg.quasar_config else ttnn.bfloat8_b,
        state_dict=state_dict, weight_cache_path=args.weight_cache_path(ttnn.bfloat16), paged_attention_config=paged,
    )
    kv_cache = [layer.attention.layer_past for layer in model.layers]
    prefill = lambda a, k: str(k.get("mode", "")).endswith("PREFILL")
    for i, layer in enumerate(model.layers):
        recorder.wrap(monkeypatch, layer, f"text.layer{i}", lambda t: t.reshape(-1, t.shape[-1]),
                      when=prefill, append_dim=0)
    row = (L - 1) % 32
    recorder.wrap(monkeypatch, model.norm, "text.norm", lambda t: t.reshape(-1, t.shape[-1])[row], when=prefill)
    args.use_qk_fused = False
    gen = Generator(model, args, mesh_device, processor=args.processor, tokenizer=args.tokenizer)
    page_table = _page_table(kv_blocks, 1)

    # --- one user, as in demo/demo.py ---
    ids, mask, thw = inputs["input_ids"][0], inputs["attention_mask"][0], inputs["image_grid_thw"][0]
    image_embeds, deepstack = visual.forward_single_user(inputs["pixel_values"], grid_thw=thw)
    text_embeds = hf_model.model.language_model.embed_tokens(ids.unsqueeze(0))
    text_tt = ttnn.from_torch(
        text_embeds.squeeze(0), device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 1), mesh_shape=args.cluster_shape),
    )
    embeds, deepstack = merge_vision_tokens_single_user_ttnn(ids, text_tt, image_embeds, hf_cfg, deepstack, args)
    pad = get_pad_embedding(hf_model, args.tokenizer.pad_token_id, args)
    x, deepstack, decoding_pos, _ = preprocess_inputs_prefill_single_user_ttnn(
        embeds, args, mask, pad_embedding=pad, deepstack_visual_embeds=deepstack
    )
    cos, sin, rope_deltas = multimodal_rope_single_user_from_hf(
        ids, thw.unsqueeze(0), hf_model, args, pad_token_id=args.tokenizer.pad_token_id
    )
    pt_user = gen._ttt_generator._get_prefill_user_page_table(page_table, kv_cache, decoding_pos)
    logits = gen.prefill_forward_single_user_text(
        ttnn.unsqueeze(x, 0), page_table=pt_user, user_id=0, last_token_idx=decoding_pos - 1,
        rot_mats=(cos, sin), kv_cache=kv_cache, deepstack_visual_embeds=deepstack,
    )
    recorder.tensors["text.logits.prefill"] = _logits_vec(logits, vocab)

    gen.update_rope_deltas([rope_deltas.squeeze(0).item()])
    pos = torch.tensor([decoding_pos])
    for k, tok in enumerate(goldens.teacher_tokens):
        recorder.progress.stage = f"text.decode{k}"
        out = gen.decode_forward(
            torch.tensor([[tok]]), pos, enable_trace=False, page_table=page_table, kv_cache=kv_cache,
            sampling_params=None, reload_inputs=True, reload_page_table=False,
        )
        recorder.tensors[f"text.logits.decode{k}"] = _logits_vec(out[0] if isinstance(out, tuple) else out, vocab)
        pos = pos + 1
```

Weight cache: the test (Step 3) sets `TT_CACHE_PATH` to a per-`cache_key` directory before `run_tt`, so `weight_cache_path` is config-specific.

- [ ] **Step 3: Implement `E2E/test_qwen3_vl_e2e.py`**

```python
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL-4B e2e on WH/BH/Quasar: per-stage PCC against a truncated HF reference."""
import os
from pathlib import Path

import pytest
import torch

from models.experimental.ops.quasar.qwen3_vl.tests.e2e import pcc as P
from models.experimental.ops.quasar.qwen3_vl.tests.e2e.config import HF_MODEL_ID
from models.experimental.ops.quasar.qwen3_vl.tests.e2e.host_reference import load_hf_model, run_reference
from models.experimental.ops.quasar.qwen3_vl.tests.e2e.presets import PRESETS, build_inputs
from models.experimental.ops.quasar.qwen3_vl.tests.e2e.progress import ProgressLog
from models.experimental.ops.quasar.qwen3_vl.tests.e2e.recorder import StageRecorder
from models.experimental.ops.quasar.qwen3_vl.tests.e2e.tt_runner import run_tt


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_qwen3_vl_e2e(mesh_device, qwen_run_config, monkeypatch):
    cfg = qwen_run_config
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    grid = mesh_device.compute_with_storage_grid_size()
    if cfg.expect_grid is not None:
        assert (grid.x, grid.y) == cfg.expect_grid, f"device grid {grid.x}x{grid.y} != expected {cfg.expect_grid}"

    monkeypatch.setenv("HF_MODEL", os.environ.get("HF_MODEL", HF_MODEL_ID))
    cache_root = Path(os.environ.get("TT_CACHE_PATH", Path.home() / ".cache" / "tt_qwen3_vl_quasar"))
    monkeypatch.setenv("TT_CACHE_PATH", str(cache_root / cfg.cache_key((grid.x, grid.y))))

    from transformers import AutoProcessor

    preset = PRESETS[cfg.size]
    inputs = build_inputs(preset, AutoProcessor.from_pretrained(HF_MODEL_ID))
    hf_model = load_hf_model(cfg.vision_layers, cfg.text_layers, cfg.deepstack_at)
    goldens = run_reference(hf_model, inputs, cfg.decode_steps)

    notes = []
    progress = ProgressLog(cfg.run_dir / "progress.log")
    if not progress.hooks_active():
        notes.append("progress.log unavailable: ttnn fast runtime mode is on (set TTNN_CONFIG_OVERRIDES).")
    recorder = StageRecorder(progress)
    host_ops, hits = [], {}
    with progress.installed():
        run_tt(cfg, preset, inputs, hf_model, goldens, mesh_device, recorder, monkeypatch)

    results = P.compare(goldens.tensors, recorder.tensors, P.thresholds_for(cfg.size), recorder.seconds)
    v = P.verdict(results, host_ops, hits, notes)
    (cfg.run_dir / "pcc.md").write_text(v.markdown)
    (cfg.run_dir / "verdict.txt").write_text(v.status + "\n")
    print(v.markdown)
    assert v.status == "PASS", f"{v.status}: first failing stage {v.first_failure}; see {cfg.run_dir}/pcc.md"
```

- [ ] **Step 4: Run on ttsim WH, native grid, Quasar config** (measures runtime too)

```bash
export TT_METAL_SIMULATOR=/localdev/$USER/ttsim/sim_wh/libttsim.so TT_METAL_SLOW_DISPATCH_MODE=1 \
  TT_METAL_DISABLE_SFPLOADMACRO=1 MESH_DEVICE=N150 \
  TTNN_CONFIG_OVERRIDES='{"enable_fast_runtime_mode": false}'
time pytest models/experimental/ops/quasar/qwen3_vl/tests/e2e/test_qwen3_vl_e2e.py -sv --timeout=14400 \
  --qwen-quasar-config --qwen-deepstack-at 0 --qwen-run-dir generated/qwen3_vl_quasar/wh_sim/first
```
Expected: PASS with every stage ≥ threshold. On failure follow superpowers:systematic-debugging: read `pcc.md`; a `SHAPE`/`MISSING` stage is a harness bug (fix the transform/slice in `tt_runner.py`, not the threshold); a `FAIL` stage on WH with HiFi4 bf16 is a config bug in this task's model-args changes. Do not lower thresholds to get green.

- [ ] **Step 5: Run once without `--qwen-quasar-config`** (native WH bf8 path) to confirm the harness also works on the unmodified config. Expected: completes; record its PCC table in the task report (thresholds may legitimately fail at bf8; report, do not tune).

- [ ] **Step 6: Commit**

```bash
git add models/experimental/ops/quasar/qwen3_vl/tests/e2e/tt_runner.py models/experimental/ops/quasar/qwen3_vl/tests/e2e/test_qwen3_vl_e2e.py
git commit -m "qwen3_vl quasar e2e: TT runner and per-stage PCC e2e test"
```

---

### Task 9: Run scripts and README

**Files:**
- Create: `E2E/_common.sh`, `E2E/run_wh_bh.sh`, `E2E/run_craq.sh`, `E2E/run_emu.sh`, `E2E/README.md`

**Interfaces:**
- Consumes: pytest options from Task 2; grid values from Task 1 (`EMU_GRID`, `GRID_OVERRIDE_2X3`).
- Produces: exit codes `0` PASS, `1` FAIL/error, `3` DIAGNOSTIC; run folder `generated/qwen3_vl_quasar/<target>/<UTC timestamp>/` with `run.log`, `command.txt`, `env.txt`, `git.txt`, `progress.log`, `pcc.md`, `verdict.txt`.

- [ ] **Step 1: Write `E2E/_common.sh`**

```bash
#!/usr/bin/env bash
# Shared flag parsing and run-folder handling for the Qwen3-VL e2e run scripts (sourced, not executed).
set -euo pipefail
shopt -s nullglob

E2E_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${E2E_DIR}/../../../../../.." && pwd)"
TEST_PATH="models/experimental/ops/quasar/qwen3_vl/tests/e2e/test_qwen3_vl_e2e.py"

# Two vision blocks and two text layers keep emulator runs near an hour while covering a block-to-block handoff.
VISION_LAYERS=2
TEXT_LAYERS=2
# One decode step exercises the paged-KV decode path once; more steps repeat the same ops.
DECODE_STEPS=1
# Tap deepstack at block 0 so two vision blocks still run the deepstack mergers and their add into the text layers.
DEEPSTACK_AT=0
# Tiny (256x256 image, 128-token prefill) is the fast iteration size; demo matches the graph captures.
SIZE=tiny
KV_BLOCKS="" # paged KV-cache blocks of 32 tokens; empty uses the preset's value.
HOST_OPS=""
DISABLE_WA=""
# The default profile catches LLK asserts via watcher without NoC sanitize, which is 20-30x slower.
DEBUG_PROFILE=default
NOC_SANITIZE=0
TIMEOUT=""
EXTRA_PYTEST=()

usage() {
  printf '%s\n' "Usage: $0 [--size tiny|demo] [--vision-layers N] [--text-layers N] [--decode-steps K]" \
    "  [--deepstack-at I|real] [--kv-blocks N] [--host-ops a,b|all] [--disable-wa a,b]" \
    "  [--debug fast|default|deep] [--noc-sanitize] [--timeout S] ${TARGET_USAGE:-} [-- <pytest args>]"
}

parse_common_flag() { # returns 0 and sets SHIFT_BY if consumed
  SHIFT_BY=2
  case "$1" in
    --size) SIZE="$2" ;;
    --vision-layers) VISION_LAYERS="$2" ;;
    --text-layers) TEXT_LAYERS="$2" ;;
    --decode-steps) DECODE_STEPS="$2" ;;
    --deepstack-at) DEEPSTACK_AT="$2" ;;
    --kv-blocks) KV_BLOCKS="$2" ;;
    --host-ops) HOST_OPS="$2" ;;
    --disable-wa) DISABLE_WA="$2" ;;
    --debug) DEBUG_PROFILE="$2" ;;
    --timeout) TIMEOUT="$2" ;;
    --noc-sanitize) NOC_SANITIZE=1; SHIFT_BY=1 ;;
    *) return 1 ;;
  esac
}

apply_debug_profile() {
  export TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 TT_METAL_DISABLE_SFPLOADMACRO=1
  export TTNN_CONFIG_OVERRIDES='{"enable_fast_runtime_mode": false}'
  case "${DEBUG_PROFILE}" in
    fast) ;;
    default | deep)
      export TT_METAL_WATCHER=1 TT_METAL_WATCHER_TEST_MODE=1 TT_METAL_LLK_ASSERTS=1
      if [[ "${NOC_SANITIZE}" -eq 0 ]]; then export TT_METAL_WATCHER_DISABLE_NOC_SANITIZE=1; fi
      if [[ "${DEBUG_PROFILE}" == deep ]]; then
        export TT_METAL_WATCHER_DUMP_ALL=1 TT_METAL_WATCHER_NOINLINE=1 TT_METAL_DPRINT_ONE_FILE_PER_RISC=1 \
          TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS=1 TT_METAL_WATCHER_DISABLE_PAUSE=1 TT_METAL_LOGGER_LEVEL=DEBUG
      fi
      ;;
    *) printf 'unknown --debug %s\n' "${DEBUG_PROFILE}" >&2; exit 1 ;;
  esac
}

run_pytest() { # $1 = target name; remaining args = target-specific pytest options
  local target="$1"; shift
  local run_dir; run_dir="${REPO_ROOT}/generated/qwen3_vl_quasar/${target}/$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p -- "${run_dir}"
  local cmd=(pytest "${TEST_PATH}" -sv "--timeout=${TIMEOUT}" --qwen-run-dir "${run_dir}"
    --qwen-size "${SIZE}" --qwen-vision-layers "${VISION_LAYERS}" --qwen-text-layers "${TEXT_LAYERS}"
    --qwen-decode-steps "${DECODE_STEPS}" --qwen-host-ops "${HOST_OPS}" --qwen-disable-wa "${DISABLE_WA}" "$@")
  if [[ "${DEEPSTACK_AT}" != real ]]; then cmd+=(--qwen-deepstack-at "${DEEPSTACK_AT}"); fi
  if [[ -n "${KV_BLOCKS}" ]]; then cmd+=(--qwen-kv-blocks "${KV_BLOCKS}"); fi
  cmd+=("${EXTRA_PYTEST[@]}")
  printf '%q ' "${cmd[@]}" >"${run_dir}/command.txt"
  env | grep -E '^(TT_|TTNN_|MESH_|HF_|NNG_|ARCH_)' | sort >"${run_dir}/env.txt" || true
  { git -C "${REPO_ROOT}" rev-parse HEAD; git -C "${REPO_ROOT}" log --oneline origin/main..HEAD; } >"${run_dir}/git.txt"
  local rc=0
  (cd -- "${REPO_ROOT}" && "${cmd[@]}") 2>&1 | tee "${run_dir}/run.log" || rc=$?
  printf '\nRun folder: %s\n' "${run_dir}"
  if [[ -f "${run_dir}/pcc.md" ]]; then cat -- "${run_dir}/pcc.md"; fi
  if [[ -f "${run_dir}/progress.log" ]]; then
    printf '\nLast ops:\n'; tail -n 5 -- "${run_dir}/progress.log"
  fi
  if [[ -f "${run_dir}/verdict.txt" ]] && grep -qx DIAGNOSTIC "${run_dir}/verdict.txt"; then exit 3; fi
  exit "${rc}"
}
```

- [ ] **Step 2: Write `E2E/run_wh_bh.sh`**

```bash
#!/usr/bin/env bash
# WH/BH baseline with the Quasar config: same flags as the Quasar scripts, on ttsim (--ttsim wh|bh) or hardware.
set -euo pipefail
shopt -s nullglob
TARGET_USAGE="[--ttsim wh|bh] [--grid 2x3|native]"
# shellcheck source=./_common.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

TTSIM=""
# 2x3 reproduces the emulator grid so the baseline runs the same configs; native uses the full chip.
GRID=2x3
TIMEOUT=14400
while (($#)); do
  if parse_common_flag "$@"; then shift "${SHIFT_BY}"; continue; fi
  case "$1" in
    --ttsim) TTSIM="$2"; shift 2 ;;
    --grid) GRID="$2"; shift 2 ;;
    --) shift; EXTRA_PYTEST=("$@"); break ;;
    -h | --help) usage; exit 0 ;;
    *) printf 'unknown flag %s\n' "$1" >&2; usage; exit 1 ;;
  esac
done

if [[ -n "${TTSIM}" ]]; then
  export TT_METAL_SIMULATOR="${QWEN_TTSIM_DIR:-/localdev/${USER}/ttsim}/sim_${TTSIM}/libttsim.so"
  [[ -f "${TT_METAL_SIMULATOR}" ]] || { printf 'missing %s (stage it, see README)\n' "${TT_METAL_SIMULATOR}" >&2; exit 1; }
else
  unset TT_METAL_SIMULATOR
fi
export MESH_DEVICE=N150
apply_debug_profile
GRID_ARGS=()
if [[ "${GRID}" == 2x3 ]]; then
  export TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="<GRID_OVERRIDE_2X3 from Task 1>"
  GRID_ARGS=(--qwen-expect-grid "<EMU_GRID from Task 1>")
fi
run_pytest "wh_bh_${TTSIM:-hw}_${GRID}" --qwen-quasar-config "${GRID_ARGS[@]}"
```

Replace the two `<... from Task 1>` strings with the measured literals before committing (e.g. `"2,1"` and `"3x2"`).

- [ ] **Step 3: Write `E2E/run_craq.sh`**

```bash
#!/usr/bin/env bash
# Qwen3-VL e2e on craq-sim (Quasar functional simulator); 2x3 grid by default, 8x4 on request.
set -euo pipefail
shopt -s nullglob
TARGET_USAGE="[--grid 2x3|8x4]"
# shellcheck source=./_common.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

# 2x3 matches the emulator so craq-sim is a fast pre-flight for emulator runs; 8x4 is craq-sim's full grid.
GRID=2x3
TIMEOUT=3600
while (($#)); do
  if parse_common_flag "$@"; then shift "${SHIFT_BY}"; continue; fi
  case "$1" in
    --grid) GRID="$2"; shift 2 ;;
    --) shift; EXTRA_PYTEST=("$@"); break ;;
    -h | --help) usage; exit 0 ;;
    *) printf 'unknown flag %s\n' "$1" >&2; usage; exit 1 ;;
  esac
done

export TT_METAL_SIMULATOR="${QWEN_CRAQ_SIM:-/localdev/${USER}/sim/libttsim.so}"
[[ -f "${TT_METAL_SIMULATOR}" ]] || { printf 'missing craq-sim %s\n' "${TT_METAL_SIMULATOR}" >&2; exit 1; }
export MESH_DEVICE=N150
apply_debug_profile
GRID_ARGS=()
if [[ "${GRID}" == 2x3 ]]; then
  export TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="<GRID_OVERRIDE_2X3 from Task 1>"
  GRID_ARGS=(--qwen-expect-grid "<EMU_GRID from Task 1>")
fi
run_pytest "craq_${GRID}" --qwen-quasar-config "${GRID_ARGS[@]}"
```

- [ ] **Step 4: Write `E2E/run_emu.sh`**

```bash
#!/usr/bin/env bash
# Qwen3-VL e2e on the Quasar emu-quasar-2x3 RTL emulator (shared Zebu farm; needs your IRD reservation).
set -euo pipefail
shopt -s nullglob
# shellcheck source=./_common.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

# Emulator runs at kHz; four hours bounds a 2+2 layer tiny run with margin.
TIMEOUT=14400
while (($#)); do
  if parse_common_flag "$@"; then shift "${SHIFT_BY}"; continue; fi
  case "$1" in
    --) shift; EXTRA_PYTEST=("$@"); break ;;
    -h | --help) usage; exit 0 ;;
    *) printf 'unknown flag %s\n' "$1" >&2; usage; exit 1 ;;
  esac
done

export TT_METAL_SIMULATOR="${QWEN_EMU_DIR:-/proj_sw/user_dev/${USER}/tt-umd-simulators/build/emu-quasar-2x3/}"
[[ -d "${TT_METAL_SIMULATOR}" ]] || { printf 'missing emulator dir %s\n' "${TT_METAL_SIMULATOR}" >&2; exit 1; }
[[ -n "${NNG_SOCKET_ADDR:-}" ]] || { printf 'NNG_SOCKET_ADDR is not set (see testing-with-quasar-emulator skill)\n' >&2; exit 1; }
export NNG_SOCKET_LOCAL_PORT="${NNG_SOCKET_LOCAL_PORT:-5555}"
if pgrep -u "${USER}" -f "pytest .*qwen3_vl.*e2e" >/dev/null; then
  printf 'another qwen3_vl e2e pytest of yours is running; refusing to share the NNG port\n' >&2; exit 1
fi
export MESH_DEVICE=N150
apply_debug_profile
trap 'printf "\nIf this run hung, check for leftover Zebu jobs (testing-with-quasar-emulator skill).\n"' EXIT
run_pytest "emu_2x3" --qwen-expect-grid "<EMU_GRID from Task 1>"
```

- [ ] **Step 5: Lint**

Run: `chmod +x models/experimental/ops/quasar/qwen3_vl/tests/e2e/*.sh && shellcheck -o all -x models/experimental/ops/quasar/qwen3_vl/tests/e2e/*.sh`
Expected: no findings. Fix any finding rather than disabling it, except `SC1091`/`SC2154`-style source-following issues which `-x` resolves.

- [ ] **Step 6: Smoke the wrapper**

Run: `models/experimental/ops/quasar/qwen3_vl/tests/e2e/run_wh_bh.sh --ttsim wh --grid native`
Expected: same result as Task 8 Step 4, exit 0, run folder printed with all seven files.

- [ ] **Step 7: Write `E2E/README.md`** covering: purpose (2 sentences); the three scripts with one example each; the flag table (from `usage`); exit codes; run-folder contents; the grid table from Task 1; ttsim staging commands (Task 1 Step 3); how to read `progress.log` on a hang; the optional uncommitted `llrt.cpp` op-timeout patch (spec §9); and a "Cherry-picks applied" table (PR, commit, why, drop-when) — initially empty.

- [ ] **Step 8: Commit**

```bash
git add models/experimental/ops/quasar/qwen3_vl/tests/e2e/*.sh models/experimental/ops/quasar/qwen3_vl/tests/e2e/README.md
git commit -m "qwen3_vl quasar e2e: run scripts for WH/BH, craq-sim and emulator"
```

---

### Task 10: Grid-derived configs (2x3 on ttsim WH)

**Files:**
- Modify: `TT/quasar_config.py` (`_QuasarArgsMixin`), `TT/vision_attention.py:337-351`, `TT/model_config.py:54-61`, `TT/rope.py:60-63`

**Interfaces:**
- Consumes: `self.max_grid_size` (`ttnn.CoreGrid`, set by `ModelArgs.__init__` from the device).
- Produces: `_QuasarArgsMixin` method overrides `find_grid(N)`, `find_grid_k_n(K, N)`, `find_prefill_grid(row_tiles, col_tiles)`, `get_attn_sdpa_decode_program_config(prefetcher=None)`, `get_attn_sdpa_prefill_program_config(seq_len=1, chunk_start_idx=None)`, `get_attn_sdpa_output_mem_config(mode, batch_size_per_device_group=1, prefetcher=None)`, plus post-init attribute fixes. Base-class behavior is unchanged when the device grid is ≥ 8x8.

- [ ] **Step 1: Run the failing baseline**

Run: `models/experimental/ops/quasar/qwen3_vl/tests/e2e/run_wh_bh.sh --ttsim wh --grid 2x3`
Expected: FAIL (an op rejecting an 8x8 program config / shard grid outside the device). Keep the run folder for the report.

- [ ] **Step 2: Add the grid overrides to `_QuasarArgsMixin`**

```python
    # Grid helpers bounded by the real device grid instead of WH's 8x8.
    def _grid_bounds(self):
        return self.max_grid_size.y, self.max_grid_size.x

    def find_grid(self, N):
        max_rows, max_cols = self._grid_bounds()
        cores = sorted((k for k in range(1, max_rows * max_cols + 1) if N % k == 0), key=lambda k: abs(k - 32))
        for c in cores:
            for rows in range(1, max_rows + 1):
                if c % rows == 0 and c // rows <= max_cols:
                    return rows, c // rows
        raise AssertionError(f"no grid for {N} tiles within {max_rows}x{max_cols}")

    def find_grid_k_n(self, K, N):
        max_rows, max_cols = self._grid_bounds()
        for c in sorted((c for c in range(1, max_rows * max_cols + 1) if K % c == 0 and N % c == 0), reverse=True):
            for rows in range(1, max_rows + 1):
                if c % rows == 0 and c // rows <= max_cols:
                    return rows, c // rows
        raise AssertionError(f"no grid for K={K}, N={N} within {max_rows}x{max_cols}")

    def find_prefill_grid(self, row_tiles, col_tiles):
        max_rows, max_cols = self._grid_bounds()
        cols = next(i for i in range(max_cols, 0, -1) if col_tiles % i == 0)
        rows = next(i for i in range(max_rows, 0, -1) if row_tiles % i == 0)
        return rows, cols

    def get_attn_sdpa_decode_program_config(self, prefetcher=None):
        cfg = super().get_attn_sdpa_decode_program_config(prefetcher)
        cfg.compute_with_storage_grid_size = (self.max_grid_size.x, self.max_grid_size.y)
        return cfg

    def get_attn_sdpa_prefill_program_config(self, seq_len=1, chunk_start_idx=None):
        cfg = super().get_attn_sdpa_prefill_program_config(seq_len, chunk_start_idx)
        cfg.compute_with_storage_grid_size = (self.max_grid_size.x, self.max_grid_size.y)
        return cfg

    def _quasar_fix_grid_attrs(self):
        g = self.max_grid_size
        self.dram_shard_grid_width = g.x
        self.prefill_rows = g.y
        self.attn_input_grid = self.dram_shard_core_grid_for_k(self.dim)
        rows, cols = self.find_grid_k_n(self.dim // 32, self.hidden_dim // 32)
        self.mlp_core_grid = ttnn.CoreGrid(y=rows, x=cols)
        rows, cols = self.find_grid_k_n(self.hidden_dim // 32, self.dim // 32)
        self.mlp2_core_grid = ttnn.CoreGrid(y=rows, x=cols)
        rows, cols = self.find_grid(self.dim // 32)
        self.lm_head_core_grid = ttnn.CoreGrid(y=rows, x=cols)
        per = 32 * self.lm_head_core_grid.num_cores
        self.max_columns_per_device_lm_head = max(per, (668 * self.lm_head_core_grid.num_cores) // per * per)
        self.min_kv_prefill_shard_seqlen = float("inf")  # never L1-shard K/V for fill_cache on small grids
        self.model_config["LM_HEAD_OUTPUT_MEMCFG"] = ttnn.DRAM_MEMORY_CONFIG
```

Call `self._quasar_fix_grid_attrs()` at the end of `QuasarModelArgs.__init__` only (text model). Do not call it from `QuasarVisionModelArgs`: its `dim`/`hidden_dim` are the vision tower's, and the vision path gets its grids from the bounded `find_prefill_grid` plus Step 4.

Then check how the base computes `mlp_core_grid`/`mlp2_core_grid` (`grep -n "mlp_core_grid =\|mlp2_core_grid =" models/tt_transformers/tt/model_config.py`) and mirror exactly the same `find_grid_k_n` arguments; and whether `SDPAProgramConfig.compute_with_storage_grid_size` is assignable (`python -c "import ttnn; c=ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(8,8)); c.compute_with_storage_grid_size=(2,3); print(c)"`). If it is read-only, construct a new `SDPAProgramConfig` copying `q_chunk_size`, `k_chunk_size`, `exp_approx_mode`.

- [ ] **Step 3: SDPA output memory config** (`num_to_corerange` defaults to 8x8)

```python
    def get_attn_sdpa_output_mem_config(self, mode, batch_size_per_device_group=1, prefetcher=None):
        import models.tt_transformers.tt.model_config as mc

        orig = mc.num_to_corerange
        g = self.max_grid_size
        mc.num_to_corerange = lambda x, start_core=ttnn.CoreCoord(0, 0), grid_x=8, grid_y=8: orig(x, start_core, g.x, g.y)
        try:
            return super().get_attn_sdpa_output_mem_config(mode, batch_size_per_device_group, prefetcher)
        finally:
            mc.num_to_corerange = orig
```

- [ ] **Step 4: Vision QKV program config and WO grid**

In `TT/vision_attention.py:337-351` replace `compute_with_storage_grid_size=(8, 8)` with `compute_with_storage_grid_size=(grid.x, grid.y)` where `grid = self.args.max_grid_size` (use the module's args attribute name), replace the literal `8` row count with `grid.y`, and replace `dram_shard_grid_width=8`-derived `per_core_N` with `math.ceil(n_tiles / grid.x)` where `n_tiles` is the output width in tiles already computed there. In `TT/model_config.py:54-61` `VISION_WO_PREFILL_PROGCFG` already calls `self.find_prefill_grid`, which the mixin bounds; change `in0_block_w=1 if self.is_galaxy else self.dim // 1024` only if the 2x3 run rejects it (then use `math.gcd(self.dim // 32, 4)`).

- [ ] **Step 5: Rope decode sharding** — `TT/rope.py:60-63`: keep the Blackhole branch only when the device grid is at least `CoreGrid(x=4, y=8)`; otherwise use the generic `num_cores_to_corerangeset(batch, device grid)` path.

- [ ] **Step 6: Iterate the 2x3 baseline until PASS**

Run: `models/experimental/ops/quasar/qwen3_vl/tests/e2e/run_wh_bh.sh --ttsim wh --grid 2x3`
For each new failure: identify the config from the traceback and `progress.log`, find its site in the agent map below, fix it in `_QuasarArgsMixin` (preferred) or the Qwen copy, rerun. Remaining expected sites (from the grid survey): `get_attn_create_head_output_mem_config` (BH `CoreGrid(4,8)`), `get_lm_head_input_mem_config` (PREFILL → return `ttnn.DRAM_MEMORY_CONFIG`), DRAM-sharded matmul `per_core_N` at `model_config.py` ~1472/1534/1860/2189 (derive from `max_grid_size.x`), `get_attn_qkv_program_config` PREFILL (`CoreCoord(8,8)` → max grid, `per_core_M = ceil(seq_tiles / grid.y)`). Each fix keeps the captured layout when it fits; fall back to DRAM interleaved only when it does not, and say so in a one-line comment.

- [ ] **Step 7: Regression on the native grid**

Run: `models/experimental/ops/quasar/qwen3_vl/tests/e2e/run_wh_bh.sh --ttsim wh --grid native`
Expected: PASS (overrides are no-ops at 8x8).

- [ ] **Step 8: Commit**

```bash
git add models/experimental/ops/quasar/qwen3_vl/tt/
git commit -m "qwen3_vl quasar: derive program configs and shard grids from the device grid"
```

---

### Task 11: Op-override registry, host fallbacks, DIAGNOSTIC

**Files:**
- Create: `E2E/op_overrides.py`
- Modify: `E2E/test_qwen3_vl_e2e.py` (install session, pass host ops/hits to verdict)
- Test: `E2E/test_harness_cpu.py`

**Interfaces:**
- Consumes: `to_host` (Task 6), `ttnn.get_golden_function`, `graph_case._ref_create_qkv_heads`, `_ref_create_qkv_heads_decode`, `_ref_concat_heads`, `_ref_concat_heads_decode`, `_ref_matmul` (from `models.experimental.ops.quasar.tests.qwen3_vl_ops.graph_case`).
- Produces:
  - `@dataclass(frozen=True) Workaround`: `name`, `target` (dotted path, e.g. `"ttnn.linear"`), `reason`, `remove_when`, `applies(args, kwargs) -> bool`, `rewrite(original, args, kwargs) -> object`
  - `@dataclass(frozen=True) HostFallback`: `target`, `source` (`"golden"`/`"graph_case"`/`"hand"`), `torch_fn(targs: list, tkwargs: dict) -> torch.Tensor | list | None`, `inplace_arg: int | None` (index of the tensor updated in place), `certified_by: str | None`
  - `WORKAROUNDS: list[Workaround]` (empty), `FALLBACKS: dict[str, HostFallback]`
  - `resolve(target) -> tuple[object, str]` (parent object, attribute name)
  - `class OverrideSession(mesh_device, host_ops: tuple, disable_wa: tuple, allow_uncertified: bool = False)`: `install(monkeypatch)`, `.hits: Counter`, `.host_ops_active: list[str]`
  - Fallback short names accepted by `--host-ops`: the last dotted component (`linear`, `scaled_dot_product_attention`, ...), or `all`.

- [ ] **Step 1: Add failing tests**

```python
from models.experimental.ops.quasar.qwen3_vl.tests.e2e import op_overrides as O


def test_resolve_dotted_target():
    import ttnn

    parent, attr = O.resolve("ttnn.experimental.paged_update_cache")
    assert parent is ttnn.experimental and attr == "paged_update_cache"


def test_uncertified_fallback_refused(monkeypatch):
    s = O.OverrideSession(mesh_device=None, host_ops=("linear",), disable_wa=())
    monkeypatch.setitem(O.FALLBACKS, "ttnn.linear", O.FALLBACKS["ttnn.linear"].__class__(
        **{**O.FALLBACKS["ttnn.linear"].__dict__, "certified_by": None}))
    with pytest.raises(RuntimeError, match="not certified"):
        s.install(monkeypatch)


def test_unknown_host_op_rejected(monkeypatch):
    s = O.OverrideSession(mesh_device=None, host_ops=("no_such_op",), disable_wa=())
    with pytest.raises(KeyError):
        s.install(monkeypatch)


def test_hand_rope_matches_rotate_half_definition():
    x = torch.randn(1, 2, 64, 128)
    cos, sin = torch.randn(1, 1, 64, 128), torch.randn(1, 1, 64, 128)
    t = torch.zeros(32, 32)
    for i in range(0, 32, 2):
        t[i, i + 1], t[i + 1, i] = 1.0, -1.0
    out = O.FALLBACKS["ttnn.experimental.rotary_embedding_llama"].torch_fn([x, cos, sin, t.reshape(1, 1, 32, 32)], {})
    rot = torch.stack([-x[..., 1::2], x[..., 0::2]], dim=-1).reshape(x.shape)
    assert torch.allclose(out, x * cos + rot * sin, atol=1e-5)


def test_hand_paged_update_cache():
    cache = torch.zeros(4, 2, 32, 8)
    upd = torch.randn(1, 1, 32, 8)  # [1, batch, kv_heads padded, hd]
    page_table = torch.tensor([[2, 0, 1, 3]])
    O.FALLBACKS["ttnn.experimental.paged_update_cache"].torch_fn(
        [cache, upd], {"update_idxs_tensor": torch.tensor([33]), "page_table": page_table}
    )
    assert torch.equal(cache[0, :, 1, :], upd[0, 0, :2, :])
```

(The paged-update expectation: position 33 → logical block 1 → physical `page_table[0,1]=0`, row `33 % 32 = 1`.)

- [ ] **Step 2: Run, expect FAIL**

Run: `pytest models/experimental/ops/quasar/qwen3_vl/tests/e2e/test_harness_cpu.py -v -k "resolve or certified or host_op or hand_"`

- [ ] **Step 3: Confirm call signatures used by the model**

```bash
grep -rn "paged_update_cache\|paged_fill_cache\|rotary_embedding_llama(" models/tt_transformers/tt/attention.py models/experimental/ops/quasar/qwen3_vl/tt/*.py | head
```
Record the positional/keyword names; the fallbacks below assume `paged_update_cache(cache, input, update_idxs_tensor=..., page_table=...)`, `paged_fill_cache(cache, input, page_table, batch_idx=...)`, `rotary_embedding_llama(x, cos, sin, trans_mat, is_decode_mode=...)`. Adapt the `torch_fn` argument handling if they differ.

- [ ] **Step 4: Implement `E2E/op_overrides.py`**

```python
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Named Quasar workarounds and host fallbacks for bisecting; host fallbacks never make a run pass."""
import inspect
from collections import Counter
from dataclasses import dataclass
from typing import Callable

import torch

from models.experimental.ops.quasar.qwen3_vl.tests.e2e.recorder import to_host


@dataclass(frozen=True)
class Workaround:
    name: str
    target: str
    reason: str
    remove_when: str
    applies: Callable
    rewrite: Callable


@dataclass(frozen=True)
class HostFallback:
    target: str
    source: str
    torch_fn: Callable
    inplace_arg: int | None = None
    certified_by: str | None = None


WORKAROUNDS: list = []


def resolve(target):
    import ttnn

    parts = target.split(".")
    assert parts[0] == "ttnn", target
    parent = ttnn
    for p in parts[1:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def _golden(target):
    def fn(targs, tkwargs):
        import ttnn

        g = ttnn.get_golden_function(getattr(*resolve(target)))
        params = inspect.signature(g).parameters
        return g(*targs, **{k: v for k, v in tkwargs.items() if k in params})

    return fn


def _graph_case(ref_name, out_shapes):
    def fn(targs, tkwargs):
        from models.experimental.ops.quasar.tests.qwen3_vl_ops import graph_case as G

        case = {"kwargs": {k: {"v": v} for k, v in tkwargs.items()}, "outs": [{"shape": s} for s in out_shapes(targs, tkwargs)]}
        out = getattr(G, ref_name)({str(i): t for i, t in enumerate(targs)}, tkwargs, case)
        if out is None:
            raise RuntimeError(f"{ref_name} does not model these shapes")
        return out

    return fn


def _qkv_shapes(targs, kw):
    x = targs[0]
    nh, nkv = kw["num_heads"], kw.get("num_kv_heads", kw["num_heads"])
    hd = x.shape[-1] // (nh + 2 * nkv)
    s = x.shape[-2]
    return [(1, nh, s, hd), (1, nkv, s, hd), (1, nkv, s, hd)]


def _qkv_decode_shapes(targs, kw):
    x = targs[0]
    nh, nkv = kw["num_heads"], kw.get("num_kv_heads", kw["num_heads"])
    hd = x.shape[-1] // (nh + 2 * nkv)
    b = kw.get("batch_size", 1)
    return [(1, b, nh, hd), (1, b, nkv, hd), (1, b, nkv, hd)]


def _rope_llama(targs, kw):
    x, cos, sin, trans = targs[:4]
    t = trans.reshape(-1, trans.shape[-2], trans.shape[-1])[0, :32, :32].float()
    big = torch.block_diag(*([t] * (x.shape[-1] // 32)))
    return x.float() * cos.float() + (x.float() @ big) * sin.float()


def _paged_update(targs, kw):
    cache, upd = targs[0], targs[1]
    idxs, pt = kw["update_idxs_tensor"], kw["page_table"]
    bs = cache.shape[2]
    for b, pos in enumerate(idxs.reshape(-1).tolist()):
        if pos < 0:
            continue
        blk = int(pt[b, pos // bs])
        cache[blk, :, pos % bs, :] = upd[0, b, : cache.shape[1], :]
    return None


def _paged_fill(targs, kw):
    cache, x, pt = targs[0], targs[1], targs[2] if len(targs) > 2 else kw["page_table"]
    b = kw.get("batch_idx", 0)
    bs = cache.shape[2]
    for s in range(x.shape[2]):
        cache[int(pt[b, s // bs]), :, s % bs, :] = x[0, :, s, :]
    return None


def _fb(target, source, fn, inplace_arg=None):
    return HostFallback(target, source, fn, inplace_arg, certified_by=None)


FALLBACKS = {
    f.target: f
    for f in [
        _fb("ttnn.linear", "golden", _golden("ttnn.linear")),
        _fb("ttnn.matmul", "golden", _golden("ttnn.matmul")),
        _fb("ttnn.rms_norm", "golden", _golden("ttnn.rms_norm")),
        _fb("ttnn.layer_norm", "golden", _golden("ttnn.layer_norm")),
        _fb("ttnn.add", "golden", _golden("ttnn.add")),
        _fb("ttnn.multiply", "golden", _golden("ttnn.multiply")),
        _fb("ttnn.transformer.scaled_dot_product_attention", "golden",
            _golden("ttnn.transformer.scaled_dot_product_attention")),
        _fb("ttnn.transformer.paged_scaled_dot_product_attention_decode", "golden",
            _golden("ttnn.transformer.paged_scaled_dot_product_attention_decode")),
        _fb("ttnn.experimental.minimal_matmul", "graph_case", _graph_case("_ref_matmul", lambda a, k: [()])),
        _fb("ttnn.experimental.nlp_create_qkv_heads", "graph_case", _graph_case("_ref_create_qkv_heads", _qkv_shapes)),
        _fb("ttnn.experimental.nlp_create_qkv_heads_decode", "graph_case",
            _graph_case("_ref_create_qkv_heads_decode", _qkv_decode_shapes)),
        _fb("ttnn.experimental.nlp_concat_heads", "graph_case", _graph_case("_ref_concat_heads", lambda a, k: [()])),
        _fb("ttnn.experimental.nlp_concat_heads_decode", "graph_case",
            _graph_case("_ref_concat_heads_decode", lambda a, k: [()])),
        _fb("ttnn.experimental.rotary_embedding_llama", "hand", _rope_llama),
        _fb("ttnn.experimental.paged_update_cache", "hand", _paged_update, inplace_arg=0),
        _fb("ttnn.experimental.paged_fill_cache", "hand", _paged_fill, inplace_arg=0),
    ]
}

# Filled in by Task 12 after test_fallbacks.py passes: target -> "test id @ arch commit".
CERTIFIED: dict = {}


def _short(target):
    return target.rsplit(".", 1)[-1]


class OverrideSession:
    def __init__(self, mesh_device, host_ops, disable_wa, allow_uncertified=False):
        self.mesh_device = mesh_device
        self.host_ops = tuple(host_ops)
        self.disable_wa = set(disable_wa)
        self.allow_uncertified = allow_uncertified
        self.hits = Counter()
        self.host_ops_active = []

    def _selected_fallbacks(self):
        if self.host_ops == ("all",):
            return list(FALLBACKS.values())
        by_short = {_short(t): f for t, f in FALLBACKS.items()}
        return [FALLBACKS.get(n) or by_short[n] for n in self.host_ops]

    def install(self, monkeypatch):
        for fb in self._selected_fallbacks():
            cert = fb.certified_by or CERTIFIED.get(fb.target)
            if cert is None and not self.allow_uncertified:
                raise RuntimeError(f"host fallback {fb.target} is not certified (run test_fallbacks.py)")
            self._install_fallback(monkeypatch, fb)
        for wa in WORKAROUNDS:
            if wa.name not in self.disable_wa:
                self._install_workaround(monkeypatch, wa)

    def _install_fallback(self, monkeypatch, fb):
        import ttnn

        parent, attr = resolve(fb.target)
        self.host_ops_active.append(fb.target)

        def wrapper(*args, **kwargs):
            self.hits[f"host:{fb.target}"] += 1
            conv = lambda v: to_host(v) if isinstance(v, ttnn.Tensor) else v
            targs = [conv(a) for a in args]
            tkw = {k: conv(v) for k, v in kwargs.items()}
            out = fb.torch_fn(targs, tkw)
            if fb.inplace_arg is not None:
                dev = args[fb.inplace_arg]
                ttnn.copy_host_to_device_tensor(
                    ttnn.from_torch(targs[fb.inplace_arg], dtype=dev.dtype, layout=dev.layout), dev
                )
                return None
            ref = next(a for a in args if isinstance(a, ttnn.Tensor))
            mc = kwargs.get("memory_config") or ttnn.DRAM_MEMORY_CONFIG
            up = lambda t: ttnn.from_torch(
                t.to(torch.bfloat16), dtype=kwargs.get("dtype") or ref.dtype, layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            res = [up(t) for t in out] if isinstance(out, (list, tuple)) else up(out)
            if mc.is_sharded():
                res = [ttnn.to_memory_config(r, mc) for r in res] if isinstance(res, list) else ttnn.to_memory_config(res, mc)
            return res

        monkeypatch.setattr(parent, attr, wrapper)

    def _install_workaround(self, monkeypatch, wa):
        parent, attr = resolve(wa.target)
        original = getattr(parent, attr)

        def wrapper(*args, **kwargs):
            if wa.applies(args, kwargs):
                self.hits[f"wa:{wa.name}"] += 1
                return wa.rewrite(original, args, kwargs)
            return original(*args, **kwargs)

        monkeypatch.setattr(parent, attr, wrapper)
```

Notes for the implementer:
- `_graph_case` for ops whose ref ignores `outs` passes a dummy shape list; the ref returns `None` only when it cannot model the shapes, which raises.
- `test_uncertified_fallback_refused` relies on every `FALLBACKS` entry starting with `certified_by=None` and `CERTIFIED` empty. Remove `monkeypatch.setitem` from that test if it proves redundant, but keep the assertion.

- [ ] **Step 5: Wire into the e2e test** (`E2E/test_qwen3_vl_e2e.py`)

Replace `host_ops, hits = [], {}` and the `with progress.installed():` block with:

```python
    from models.experimental.ops.quasar.qwen3_vl.tests.e2e.op_overrides import OverrideSession

    session = OverrideSession(mesh_device, cfg.host_ops, cfg.disable_wa)
    session.install(monkeypatch)
    with progress.installed():
        run_tt(cfg, preset, inputs, hf_model, goldens, mesh_device, recorder, monkeypatch)
    host_ops, hits = session.host_ops_active, dict(session.hits)
```

Add the pytest option `--qwen-allow-uncertified` (store_true) in `E2E/conftest.py`, a `RunConfig.allow_uncertified: bool` field (parse in `from_options`; update `_opts` in the CPU test with `"--qwen-allow-uncertified": False`), and pass it to `OverrideSession`.

- [ ] **Step 6: Run CPU tests, expect PASS**

Run: `pytest models/experimental/ops/quasar/qwen3_vl/tests/e2e/test_harness_cpu.py -v`

- [ ] **Step 7: DIAGNOSTIC smoke on ttsim WH**

Run: `models/experimental/ops/quasar/qwen3_vl/tests/e2e/run_wh_bh.sh --ttsim wh --grid native --host-ops rms_norm -- --qwen-allow-uncertified`
Expected: exit code 3, `pcc.md` says `Verdict: DIAGNOSTIC` and lists `ttnn.rms_norm` with a non-zero hit count.

- [ ] **Step 8: Commit**

```bash
git add models/experimental/ops/quasar/qwen3_vl/tests/e2e/
git commit -m "qwen3_vl quasar e2e: op-override registry; host fallbacks make a run DIAGNOSTIC"
```

---

### Task 12: Certify host fallbacks against the real ops (WH/BH)

**Files:**
- Create: `E2E/test_fallbacks.py`
- Modify: `E2E/op_overrides.py` (`CERTIFIED` entries)

**Interfaces:**
- Consumes: `FALLBACKS`, `resolve`, `graph_case` (`run_case` internals for building tensors from captured case specs: `_build_value(spec, mesh_device, case, op_name, key, torch_sink)`), the case lists in `models/experimental/ops/quasar/tests/qwen3_vl_ops/test_<op>.py` (`CASES`).
- Produces: one test per (fallback, captured case): fallback output vs real op output on WH/BH, PCC ≥ 0.999 and equal shapes; skipped with reason on Quasar.

- [ ] **Step 1: Read the case-building helpers**

```bash
sed -n 405,520p models/experimental/ops/quasar/tests/qwen3_vl_ops/graph_case.py
sed -n 1179,1260p models/experimental/ops/quasar/tests/qwen3_vl_ops/graph_case.py
```
Note how `run_case` turns a case into device tensors + matching torch tensors (`torch_sink`) and calls the op. The certification test reuses exactly that to build inputs, then calls both the real op and the fallback on the same inputs.

- [ ] **Step 2: Write `E2E/test_fallbacks.py`**

```python
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Certify each host fallback against the real ttnn op on the captured qwen3_vl_ops cases (WH/BH only)."""
import importlib

import pytest
import torch

import ttnn
from models.common.utility_functions import is_quasar
from models.experimental.ops.quasar.qwen3_vl.tests.e2e import op_overrides as O
from models.experimental.ops.quasar.qwen3_vl.tests.e2e.pcc import pcc
from models.experimental.ops.quasar.qwen3_vl.tests.e2e.recorder import to_host
from models.experimental.ops.quasar.tests.qwen3_vl_ops import graph_case as G

_CASE_MODULE = {  # fallback target -> generated qwen3_vl_ops test module holding CASES
    "ttnn.linear": "test_linear",
    "ttnn.rms_norm": "test_rms_norm",
    "ttnn.layer_norm": "test_layer_norm",
    "ttnn.add": "test_add",
    "ttnn.multiply": "test_multiply",
    "ttnn.transformer.scaled_dot_product_attention": "test_scaled_dot_product_attention",
    "ttnn.transformer.paged_scaled_dot_product_attention_decode": "test_paged_scaled_dot_product_attention_decode",
    "ttnn.experimental.minimal_matmul": "test_minimal_matmul",
    "ttnn.experimental.nlp_create_qkv_heads": "test_nlp_create_qkv_heads",
    "ttnn.experimental.nlp_create_qkv_heads_decode": "test_nlp_create_qkv_heads_decode",
    "ttnn.experimental.nlp_concat_heads": "test_nlp_concat_heads",
    "ttnn.experimental.nlp_concat_heads_decode": "test_nlp_concat_heads_decode",
    "ttnn.experimental.rotary_embedding_llama": "test_rotary_embedding_llama",
    "ttnn.experimental.paged_update_cache": "test_paged_update_cache",
    "ttnn.experimental.paged_fill_cache": "test_paged_fill_cache",
}


def _bf16(case):
    """Captured cases may be bfp8/bfp4; certify in bf16 like the Quasar config."""
    def fix(spec):
        if isinstance(spec, dict) and spec.get("k") == "t" and spec.get("dtype") in ("BFLOAT8_B", "BFLOAT4_B"):
            return {**spec, "dtype": "BFLOAT16"}
        return spec

    return {**case, "args": [fix(a) for a in case["args"]], "kwargs": {k: fix(v) for k, v in case["kwargs"].items()}}


def _params():
    out = []
    for target, mod in _CASE_MODULE.items():
        m = importlib.import_module(f"models.experimental.ops.quasar.tests.qwen3_vl_ops.{mod}")
        out += [pytest.param(target, _bf16(c), id=f"{mod}-{c['id']}") for c in m.CASES]
    return out


@pytest.mark.parametrize("target, case", _params())
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_fallback_matches_op(mesh_device, target, case):
    if is_quasar():
        pytest.skip("fallbacks are certified on WH/BH, never on Quasar")
    fb = O.FALLBACKS[target]
    args, kwargs = G.build_inputs_for_case(case, mesh_device)  # see Step 3
    real = getattr(*O.resolve(target))(*args, **kwargs)
    if fb.inplace_arg is not None:
        real_t = to_host(args[fb.inplace_arg])
        args2, kwargs2 = G.build_inputs_for_case(case, mesh_device)
        tav = [to_host(a) if isinstance(a, ttnn.Tensor) else a for a in args2]
        tkw = {k: to_host(v) if isinstance(v, ttnn.Tensor) else v for k, v in kwargs2.items()}
        fb.torch_fn(tav, tkw)
        mine = tav[fb.inplace_arg]
        pairs = [(real_t, mine)]
    else:
        tav = [to_host(a) if isinstance(a, ttnn.Tensor) else a for a in args]
        tkw = {k: to_host(v) if isinstance(v, ttnn.Tensor) else v for k, v in kwargs.items()}
        mine = fb.torch_fn(tav, tkw)
        reals = real if isinstance(real, (list, tuple)) else [real]
        mines = mine if isinstance(mine, (list, tuple)) else [mine]
        pairs = [(to_host(r), m.float()) for r, m in zip(reals, mines)]
    for r, m in pairs:
        r = r[tuple(slice(0, s) for s in m.shape)]  # drop tile padding on the device side only
        assert tuple(r.shape) == tuple(m.shape)
        assert pcc(r, m) >= 0.999, f"{target} {case['id']}: pcc {pcc(r, m):.5f}"
```

- [ ] **Step 3: Add `build_inputs_for_case` to `graph_case.py`** — a small public helper that runs the input-building half of `run_case` (the loop that calls `_build_value` for each arg/kwarg) and returns `(args, kwargs)` of device tensors/literals, with no op call and no checks. Refactor `run_case` to call it so the two cannot diverge. Run one existing op test to prove no regression:

Run (ttsim WH env from Task 8): `pytest models/experimental/ops/quasar/tests/qwen3_vl_ops/test_add.py -v`
Expected: same pass/skip set as before the refactor.

- [ ] **Step 4: Run certification on ttsim WH**

Run: `pytest models/experimental/ops/quasar/qwen3_vl/tests/e2e/test_fallbacks.py -v --timeout=7200` (ttsim WH env)
Expected: PASS for every case. A failure means the fallback is wrong (or the case is outside what the ref models): fix the fallback; do not loosen 0.999. Cases a fallback cannot model may be skipped with an explicit `pytest.skip(reason)` naming the shape, and that fallback's certificate must then state its covered shapes.

- [ ] **Step 5: Run on ttsim BH**, same command with the BH simulator. Expected: PASS.

- [ ] **Step 6: Record certificates** in `E2E/op_overrides.py`:

```python
CERTIFIED = {
    "ttnn.linear": "test_fallbacks.py::test_fallback_matches_op[test_linear-*] @ ttsim wh+bh <short sha>",
    # ... one line per target that passed
}
```

- [ ] **Step 7: Collective check against HF**

Run: `models/experimental/ops/quasar/qwen3_vl/tests/e2e/run_wh_bh.sh --ttsim wh --grid 2x3 --host-ops all`
Expected: exit 3 (DIAGNOSTIC by design) and every stage PCC ≥ 0.999 in `pcc.md`. Lower PCC on any stage means a fallback or the harness is wrong for real model tensors — fix before continuing.

- [ ] **Step 8: Commit**

```bash
git add models/experimental/ops/quasar/qwen3_vl/tests/e2e/ models/experimental/ops/quasar/tests/qwen3_vl_ops/graph_case.py
git commit -m "qwen3_vl quasar e2e: certify host fallbacks against real ops on WH/BH"
```

---

### Task 13: WH/BH baseline gate

**Files:**
- Modify: `E2E/README.md` (baseline results table)

- [ ] **Step 1: ttsim WH and BH, 2x3 and native, tiny**

```bash
for sim in wh bh; do for g in 2x3 native; do
  models/experimental/ops/quasar/qwen3_vl/tests/e2e/run_wh_bh.sh --ttsim "$sim" --grid "$g" || echo "FAILED $sim $g"
done; done
```
Expected: four PASS runs (exit 0). Any failure on BH only: fix in `_QuasarArgsMixin` (BH branches) and rerun all four.

- [ ] **Step 2: Ask the user for the hardware baseline**

Give the user: the branch name, commit sha, and the commands
```bash
models/experimental/ops/quasar/qwen3_vl/tests/e2e/run_wh_bh.sh --grid 2x3
models/experimental/ops/quasar/qwen3_vl/tests/e2e/run_wh_bh.sh --grid native
```
to run on their WH or BH machine, plus `pytest models/experimental/ops/quasar/qwen3_vl/tests/e2e/test_fallbacks.py` for hardware certification. Ask them to copy the run folders back (or paste `pcc.md`). Proceed to Task 14 meanwhile; add hardware results to `CERTIFIED` and the README when they arrive.

- [ ] **Step 3: Record results** in a README "Baselines" table (target, grid, size, V/T/K, verdict, wall time, run folder).

- [ ] **Step 4: Commit**

```bash
git add models/experimental/ops/quasar/qwen3_vl/tests/e2e/README.md
git commit -m "qwen3_vl quasar e2e: record WH/BH baselines"
```

---

### Task 14: craq-sim bring-up (2x3, then 8x4)

**Files:** model copy and `E2E/op_overrides.py` (`WORKAROUNDS`), as failures dictate.

This task is iterative. Each loop iteration is one failure → one fix → one commit.

- [ ] **Step 1: First run**

Run: `models/experimental/ops/quasar/qwen3_vl/tests/e2e/run_craq.sh`
Expected first outcomes, in rough order: (a) `determine_device_name` is already handled; (b) an op rejecting a config on Quasar; (c) a hang (pytest timeout → read `progress.log` "Last ops"); (d) PCC failures.

- [ ] **Step 2: Per-failure loop** (use superpowers:systematic-debugging)

1. Identify the failing op and its args from the traceback or the last `PRE` line without a `POST` in `progress.log`.
2. Check whether a candidate PR covers it (#58909 minimal_matmul, #58912 DRAM-sharded matmul, #58913 paged fused update_cache, #58914 reshape_view). If so, ask the user before cherry-picking: `git fetch origin pull/<N>/head:pr-<N> && git cherry-pick <commits>`. Then rebuild (`./build_metal.sh --development`), rerun, and add a row to the README cherry-pick table.
3. Otherwise reproduce in isolation: `TT_METAL_SIMULATOR=/localdev/$USER/sim/libttsim.so pytest models/experimental/ops/quasar/tests/qwen3_vl_ops/test_<op>.py -k <case>` (one file per invocation).
4. Bisect numerics with `--host-ops <op>` (DIAGNOSTIC runs): if routing one op to host makes the downstream stages pass, that op is the culprit.
5. Fix in priority order: (i) a config fix in `_QuasarArgsMixin`/model copy that keeps the captured layout if it fits; (ii) a narrowly scoped `Workaround` (op plus shape/memcfg predicate, `reason`, `remove_when`); (iii) if it is an op/kernel bug, stop and report to the user with the isolated repro. Do not edit kernels without the user's go-ahead.
6. For hangs that `progress.log` cannot localize, apply the uncommitted `llrt.cpp` op-timeout patch locally (spec §9), rerun with `TT_METAL_OPERATION_TIMEOUT_SECONDS=<s>`, then `git checkout -- tt_metal/llrt/llrt.cpp`.
7. Rerun `run_craq.sh`. When a new workaround or config change lands, rerun `run_wh_bh.sh --ttsim wh --grid 2x3` to keep the baseline green.
8. Commit: `git commit -m "qwen3_vl quasar: <what> for <op> on craq-sim"`.

If uploads via `ttnn.from_torch(..., layout=TILE)` hang on Quasar (as seen in the llama bring-up), add a workaround modelled on `_install_quasar_tilize_from_torch` in `models/experimental/llama32_1b_quasar/tests/demos/llama32_1b/test_llama_e2e.py:95-188` (upload row-major, then tilize on device).

- [ ] **Step 3: craq-sim 2x3 PASS**

Run: `models/experimental/ops/quasar/qwen3_vl/tests/e2e/run_craq.sh`
Expected: exit 0, record wall time.

- [ ] **Step 4: craq-sim 8x4**

Run: `models/experimental/ops/quasar/qwen3_vl/tests/e2e/run_craq.sh --grid 8x4`
Before the first 8x4 craq run, run the matching baseline `run_wh_bh.sh --ttsim wh --grid native` (8x8 ≥ 8x4; if any config differs at 8x4, add an 8x4 WH override run with `TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE` set for 8x4 and `--qwen-expect-grid 8x4`). Loop as in Step 2 until exit 0.

- [ ] **Step 5: Update README** (results table, workaround list with reasons, cherry-picks) and commit.

---

### Task 15: Emulator 2x3 tiny

- [ ] **Step 1: Ask the user** to confirm their reservation and `NNG_SOCKET_ADDR`, and that it is OK to start a (likely multi-hour) run.

- [ ] **Step 2: Run**

Run: `models/experimental/ops/quasar/qwen3_vl/tests/e2e/run_emu.sh`
Expected: exit 0. If the runtime exceeds the one-hour target, report per-stage seconds from `pcc.md` and propose reductions (e.g. `--debug fast` once craq-sim is stable, `--decode-steps 0`, or V=1/T=1) to the user instead of changing defaults unilaterally.

- [ ] **Step 3: On failure** — apply the Task 14 Step 2 loop, but reproduce on craq-sim 2x3 first whenever possible (minutes vs hours). Only emulator-specific failures (RTL vs functional sim differences) are debugged on the emulator, and each such run needs the user's go-ahead.

- [ ] **Step 4: Record and commit** README results.

---

### Task 16: Demo preset

- [ ] **Step 1: WH baseline at demo size**

Run: `models/experimental/ops/quasar/qwen3_vl/tests/e2e/run_wh_bh.sh --ttsim wh --grid 2x3 --size demo`
Expected: PASS (or per-stage numbers to set `thresholds.json` `"demo"` entries — only with the user's agreement and a stated reason). Note the vision tower pads 11008 → 12288 here; if vision stages fail only at demo size, check padding/masking before anything else.

- [ ] **Step 2: craq-sim 2x3 demo**: `run_craq.sh --size demo`; loop as in Task 14. Record wall time.

- [ ] **Step 3: Emulator demo** only if craq-sim timing suggests it fits the user's budget; ask first. Otherwise report the estimate and propose an intermediate preset (a new `PRESETS` entry with its own token counts added to `test_preset_token_counts`).

- [ ] **Step 4: Record and commit** README results.
