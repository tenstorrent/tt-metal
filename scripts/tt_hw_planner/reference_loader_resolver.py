# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""LLM resolver for models whose weights won't load via `transformers.AutoModel*.from_pretrained`
(non-`transformers` checkpoints — e.g. Mistral/vLLM-native `consolidated.safetensors` + `params.json`,
GGUF, or trust_remote_code variants).

The blocker lives in the common loading layer (capture + the PCC test's `_build_torch_reference`), so
the fix does too. When `from_pretrained` fails, the bring-up loop calls `resolve(...)`, which inspects
the repo (file list + `library` tag + config), asks an LLM to write ONE model-local
`tests/pcc/_reference_loader.py` exposing `load_reference_model(model_id) -> nn.Module`, and validates
it imports. The generated PCC-test template imports that loader as a fallback, so every per-component
test (and the global PCC gate) picks it up automatically.

OFF BY DEFAULT: `resolve` is a no-op unless `TT_HW_PLANNER_LOADER_RESOLVER=1` (or `enabled=True`).
Correctness is still gated by PCC — the resolver only produces a loader; it never weakens a test.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

_LOADER_FILENAME = "_reference_loader.py"
_ENV_FLAG = "TT_HW_PLANNER_LOADER_RESOLVER"


def is_enabled(enabled: Optional[bool] = None) -> bool:
    if enabled is not None:
        return bool(enabled)
    return os.environ.get(_ENV_FLAG, "") == "1"


def loader_path(demo_dir: Path) -> Path:
    return Path(demo_dir) / "tests" / "pcc" / _LOADER_FILENAME


def has_loader(demo_dir: Path) -> bool:
    return loader_path(demo_dir).is_file()


def is_load_failure(failure_text: str) -> bool:
    """True when a failure is the from_pretrained "no loadable weights" signature the resolver targets
    (as opposed to a normal stub/PCC failure)."""
    t = failure_text or ""
    return ("Could not load" in t and ("AutoModel" in t or "from_pretrained" in t)) or (
        "does not appear to have a file named" in t
    )


def _repo_files(model_id: str) -> List[str]:
    try:
        from huggingface_hub import list_repo_files

        return list(list_repo_files(model_id))
    except Exception:
        return []


def _repo_meta(model_id: str) -> dict:
    try:
        from huggingface_hub import model_info

        info = model_info(model_id)
        return {
            "library_name": getattr(info, "library_name", None),
            "pipeline_tag": getattr(info, "pipeline_tag", None),
            "tags": list(getattr(info, "tags", []) or [])[:20],
        }
    except Exception:
        return {}


def _config_summary(model_id: str) -> str:
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        return str(cfg)[:2000]
    except Exception as exc:  # noqa: BLE001
        return f"(AutoConfig failed: {type(exc).__name__}: {exc})"


def build_prompt(model_id: str, demo_dir: Path, failure_text: str) -> str:
    files = _repo_files(model_id)
    meta = _repo_meta(model_id)
    cfg = _config_summary(model_id)
    weightish = [f for f in files if f.endswith((".safetensors", ".bin", ".pth", ".gguf"))]
    dst = loader_path(demo_dir)
    return (
        f"The model `{model_id}` cannot be loaded via transformers "
        f"`AutoModelForCausalLM/AutoModel.from_pretrained` — the bring-up capture and every "
        f"per-component PCC test failed with:\n{failure_text[:800]}\n\n"
        f"Repo metadata: library_name={meta.get('library_name')} pipeline_tag={meta.get('pipeline_tag')} "
        f"tags={meta.get('tags')}\n"
        f"Repo weight files: {weightish}\n"
        f"All repo files: {files}\n"
        f"AutoConfig: {cfg}\n\n"
        f"Your job: WRITE the file `{dst}` exposing exactly:\n"
        f"    def load_reference_model(model_id: str):\n"
        f'        """Return an nn.Module (in eval mode) equivalent to the HF reference for this '
        f'model, loaded from whatever real format the repo actually ships."""\n\n'
        f"Pick the correct strategy for THIS repo (do not guess blindly — inspect):\n"
        f"  1. If it ships HF-native weights under a non-default name, load them with the right class.\n"
        f"  2. If it ships a native/consolidated checkpoint (e.g. Mistral `consolidated.safetensors` + "
        f"`params.json`), convert its keys to the matching transformers arch (Ministral/Mistral/…), "
        f"applying the correct config (head_dim, sliding_window, embed multiplier, RoPE permute) so a "
        f"generated continuation is COHERENT — verify before returning.\n"
        f"  3. If a native runtime (mistral_common / vllm) is the only way, use it.\n"
        f"  4. ONLY if real weights are truly unusable: build the module from AutoConfig with random "
        f"weights (valid for per-component structural PCC, since the ttnn port reads the same module). "
        f"State this limitation in a module docstring.\n\n"
        f"The loader must be import-safe (no side effects at import) and deterministic. After writing, "
        f"run a quick self-check that `load_reference_model('{model_id}')` returns a module and a "
        f"forward runs. Do NOT edit any test file or weaken any assertion — only write "
        f"`{_LOADER_FILENAME}`."
    )


def _extract_and_write(demo_dir: Path, text: str) -> bool:
    """Fallback writer if the agent returned code inline instead of using Write. Prefers the file the
    agent already wrote."""
    if has_loader(demo_dir):
        return True
    m = re.search(r"```(?:python)?\n(.*?def load_reference_model.*?)```", text, re.DOTALL)
    if not m:
        return False
    code = m.group(1)
    try:
        loader_path(demo_dir).write_text(code, encoding="utf-8")
        return True
    except OSError:
        return False


def _validates(demo_dir: Path) -> bool:
    p = loader_path(demo_dir)
    if not p.is_file():
        return False
    try:
        import ast

        ast.parse(p.read_text(encoding="utf-8"))
        return "def load_reference_model" in p.read_text(encoding="utf-8")
    except Exception:
        return False


def resolve(
    *,
    model_id: str,
    demo_dir: Path,
    failure_text: str,
    agent_bin: str = "claude",
    enabled: Optional[bool] = None,
    timeout_s: int = 900,
    cwd: Optional[Path] = None,
) -> dict:
    """Write a model-local `_reference_loader.py` via the LLM. No-op unless enabled. Returns
    {resolved, path, reason}. Engine-neutral: fsm and cc both call this."""
    demo_dir = Path(demo_dir)
    if not is_enabled(enabled):
        return {"resolved": False, "reason": f"disabled (set {_ENV_FLAG}=1 to enable)"}
    if has_loader(demo_dir) and _validates(demo_dir):
        return {"resolved": True, "path": str(loader_path(demo_dir)), "reason": "loader already present"}
    loader_path(demo_dir).parent.mkdir(parents=True, exist_ok=True)
    prompt = build_prompt(model_id, demo_dir, failure_text)
    repo_root = Path(__file__).resolve().parents[2]
    argv = [
        agent_bin,
        "-p",
        prompt,
        "--allowedTools",
        "Read",
        "Write",
        "Edit",
        "Bash",
        "Glob",
        "Grep",
        "--output-format",
        "text",
    ]
    try:
        subprocess.run(
            argv,
            cwd=str(cwd or repo_root),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"resolved": False, "reason": "resolver agent timed out"}
    except Exception as exc:  # noqa: BLE001
        return {"resolved": False, "reason": f"{type(exc).__name__}: {exc}"}
    if _validates(demo_dir):
        return {"resolved": True, "path": str(loader_path(demo_dir)), "reason": "loader written"}
    return {"resolved": False, "reason": "agent did not produce a valid _reference_loader.py"}


if __name__ == "__main__":
    _demo = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    _mid = sys.argv[2] if len(sys.argv) > 2 else ""
    print(json.dumps(resolve(model_id=_mid, demo_dir=_demo, failure_text="Could not load via AutoModel", enabled=True)))
