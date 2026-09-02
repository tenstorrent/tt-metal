# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""LLM resolver for models whose weights won't load via `transformers.AutoModel*.from_pretrained`
(non-`transformers` checkpoints — e.g. Mistral/vLLM-native `consolidated.safetensors` + `params.json`,
GGUF, or trust_remote_code variants).

The blocker lives in the common loading layer (capture + the PCC test's `_build_torch_reference`), so
the fix does too. When `from_pretrained` fails, the bring-up loop calls `resolve(...)`, which inspects
the repo (file list + `library` tag + config), asks an LLM to write ONE model-local
`tests/pcc/_reference_loader.py` exposing `load_reference_model(model_id) -> nn.Module`. The generated
PCC-test template imports that loader as a fallback, so every per-component test (and the global PCC
gate) picks it up automatically.

What validation here does and does NOT mean, in three layers:
  1. STRUCTURAL — it parses and really defines `load_reference_model(model_id)` (`_validates`).
  2. RUNTIME — it is executed, and must hand back an `nn.Module` that has parameters (`verify`).
  3. PROVENANCE — its parameters are sampled against the shipped checkpoint, which catches a
     reference built from config with weights never loaded (`weight_provenance`, advisory).
  4. Self-consistency, which catches a reference that loads the right weights but wires them up
     wrongly -- the one failure with no golden to be caught by (`check_invariants`).

What none of them establish is that the reference computes the RIGHT thing. For these checkpoints
there is by definition no golden to compare against — that is why the loader had to be written at
all — so a reference that loads real weights but wires them up wrongly (a transposed projection, a
wrong RoPE base) will still be the yardstick every later PCC score is measured against.

OFF BY DEFAULT: `resolve` is a no-op unless `TT_HW_PLANNER_LOADER_RESOLVER=1` (or `enabled=True`).
Correctness is still gated by PCC — the resolver only produces a loader; it never weakens a test.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

_LOADER_FILENAME = "_reference_loader.py"
_LOADER_FUNC = "load_reference_model"
_ENV_FLAG = "TT_HW_PLANNER_LOADER_RESOLVER"

# Set by a loader that had to fall back to random weights (prompt strategy 5). Machine-readable on
# purpose: the fallback used to be recorded in a module docstring, which nothing reads, so a run
# could be scored against a reference unrelated to the shipped weights with no trace in the result.
_RANDOM_WEIGHTS_FLAG = "REFERENCE_USES_RANDOM_WEIGHTS"

# Long enough that a causal model has a past to respect, short enough to stay cheap on CPU.
_PROBE_SEQ = 8
# The invariants are exact statements, so this is float noise from reordered reductions only.
_INVARIANT_ATOL = 1e-4

# Weight-provenance sampling. Small tensors (norms, biases) are skipped because their moments
# collide easily -- a vector of ones matches a vector of ones whatever checkpoint it came from.
# Sampling a handful of tensors is enough: a real load matches nearly all of them, random init
# essentially none, so the verdict does not get sharper by reading more of a multi-GB shard.
_FINGERPRINT_MIN_NUMEL = 4096
_PROVENANCE_SAMPLE = 12
_MOMENT_TOL = 1e-4


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
    if os.path.isdir(model_id):
        base = Path(model_id)
        return [str(p.relative_to(base)) for p in base.rglob("*") if p.is_file()]
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
        f"  4. If the repo has NO `model_type` / `auto_map` (AutoConfig itself raises 'Unrecognized "
        f"model' — the architecture is NOT in transformers), it lives OUTSIDE transformers: pip-install "
        f"and import the model's OWN package (infer it from library_name / tags / repo files — e.g. a "
        f"TTS/vocoder checkpoint) OR import the repo's trust_remote_code modeling module, then "
        f"instantiate its native nn.Module with the REAL weights the repo ships. This is the correct "
        f"path for config-less custom architectures.\n"
        f"  5. ONLY if real weights are truly unusable: build the module from AutoConfig with random "
        f"weights (valid for per-component structural PCC, since the ttnn port reads the same module). "
        f"If and ONLY if you take this path, set `{_RANDOM_WEIGHTS_FLAG} = True` at module level, so "
        f"the run can report that PCC was scored against structure and not against the real weights. "
        f"Do not set it on any other path.\n\n"
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


def _loader_ast(demo_dir: Path):
    """Parsed loader module, or None when it is missing or does not parse."""
    p = loader_path(demo_dir)
    if not p.is_file():
        return None
    try:
        return ast.parse(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 -- unparseable is just "not a valid loader"
        return None


def _defines_loader(tree) -> bool:
    """Is there a REAL module-level `def load_reference_model(model_id)`?

    Was `"def load_reference_model" in source`, which the name merely being MENTIONED satisfied: a
    file whose only occurrence was in a comment or a docstring -- so it defined nothing at all --
    passed the gate and was banked as a resolved loader. Require the actual definition, and require
    it to take the model id, so a zero-arg stub is not mistaken for the real thing either.
    """
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == _LOADER_FUNC:
            a = node.args
            return bool(a.posonlyargs or a.args or a.vararg)
    return False


def uses_random_weights(demo_dir: Path) -> bool:
    """Did the loader fall back to random weights instead of the shipped checkpoint?

    Such a reference validates STRUCTURE only: every per-component PCC score is then measured
    against a model whose weights mean nothing, so a port can score 1.0 while reproducing noise.
    That is a legitimate last resort (prompt strategy 5), but it has to travel with the result
    rather than sit in a docstring, so callers can surface it.
    """
    tree = _loader_ast(demo_dir)
    if tree is None:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == _RANDOM_WEIGHTS_FLAG for t in node.targets
        ):
            return bool(getattr(node.value, "value", False) is True)
    return False


def _validates(demo_dir: Path) -> bool:
    tree = _loader_ast(demo_dir)
    return tree is not None and _defines_loader(tree)


def load_reference(demo_dir: Path, model_id: str):
    """Execute the model-local loader and return whatever `load_reference_model` builds.

    Single home for "run the loader": module_tree and decompose each had their own copy of the
    spec_from_file_location dance, and decompose's copy rebuilt the path from string parts instead
    of calling loader_path(), so a change to the filename would have silently missed it. Raises
    whatever the loader raises -- callers already turn that into their own diagnostic.
    """
    import importlib.util as ilu

    p = loader_path(demo_dir)
    spec = ilu.spec_from_file_location("_tt_hw_planner_reference_loader", str(p))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {p}")
    mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, _LOADER_FUNC, None)
    if not callable(fn):
        raise AttributeError(f"{p} defines no callable {_LOADER_FUNC}")
    return fn(model_id)


def verify(demo_dir: Path, model_id: str) -> dict:
    """Actually RUN the loader and report what came back.

    The structural gate only proves a `def` is present; it cannot tell a loader that builds the
    real model from one that raises, returns None, or hands back something that is not a module.
    Those all used to be discovered much later, as an error far from its cause. Returns
    {ok, status, reason}: `broken` means the loader itself is at fault, while `unverified` means
    THIS environment could not run it (no weights cached, torch missing) and is not held against
    the loader -- an environment problem must not be reported as a bad loader.
    """
    try:
        import torch.nn as nn
    except Exception as exc:  # noqa: BLE001 -- no torch here is an environment fact, not a verdict
        return {"ok": True, "status": "unverified", "reason": f"torch unavailable: {exc}"}

    try:
        ref = load_reference(demo_dir, model_id)
    except Exception as exc:  # noqa: BLE001 -- any failure to produce a model is the loader's
        return {"ok": False, "status": "broken", "reason": f"{type(exc).__name__}: {exc}"}

    if ref is None:
        return {"ok": False, "status": "broken", "reason": f"{_LOADER_FUNC} returned None"}
    if not isinstance(ref, nn.Module):
        return {
            "ok": False,
            "status": "broken",
            "reason": f"{_LOADER_FUNC} returned {type(ref).__name__}, not nn.Module",
        }
    if not any(True for _ in ref.parameters()):
        return {"ok": False, "status": "broken", "reason": "reference module has no parameters"}
    invariants = check_invariants(ref)
    if invariants.get("status") == "violated":
        return {
            "ok": False,
            "status": "broken",
            "reason": invariants["reason"],
            "invariants": invariants,
            "provenance": weight_provenance(model_id, ref),
        }
    return {
        "ok": True,
        "status": "verified",
        "reason": "loader returned an nn.Module with parameters",
        "provenance": weight_provenance(model_id, ref),
        "invariants": invariants,
    }


def _is_autoregressive(ref) -> bool:
    """Does this model claim to generate left-to-right?

    Causality is only a law for decoders. An encoder -- a text embedder, or the audio tower in
    front of a speech model -- is bidirectional BY DESIGN, and every position legitimately sees
    every other. Applying the check to one would condemn a perfectly good reference, so it is asked
    rather than assumed: `is_decoder` and `can_generate()` are the model's own statements about
    itself, which is also why neither is a name to keep in step with anything.
    """
    cfg = getattr(ref, "config", None)
    if getattr(cfg, "is_decoder", False) or getattr(cfg, "is_causal", False):
        return True
    can_generate = getattr(ref, "can_generate", None)
    try:
        return bool(can_generate()) if callable(can_generate) else False
    except Exception:  # noqa: BLE001
        return False


def _close(a, b) -> bool:
    """Equal up to float noise, refusing to compare tensors of different shapes.

    Shape-strict on purpose: `torch.allclose` would happily broadcast a one-row slice against a
    many-row reference and call the result a mismatch, turning a comparison mistake into a verdict
    about the model. The tolerance scales with the values because logits run to tens or hundreds,
    where a fixed 1e-4 is below the noise of a reordered reduction.
    """
    import torch

    if a.shape != b.shape:
        return False
    return torch.allclose(a, b, rtol=0, atol=_INVARIANT_ATOL * max(1.0, float(b.abs().max())))


def _first_tensor(out):
    """The tensor a forward returned, whatever wrapper it came in.

    Asks the wrapper to unpack itself instead of reaching for field names: a HuggingFace output
    orders its own fields and drops the empty ones, so the first tensor out of `to_tuple()` is the
    result field whether the model calls it logits, hidden states, or something this has never
    heard of.

    Deliberately not `bringup_loop._normalize_out`, which unwraps one specific field name and so
    hands back the wrapper untouched for any model that does not use it -- and which lives in a
    module that pulls in ttnn, where this one must keep working on a box with no torch at all.
    """
    import torch

    if isinstance(out, torch.Tensor):
        return out
    unpack = getattr(out, "to_tuple", None)
    if callable(unpack):
        out = unpack()
    elif isinstance(out, dict):
        out = list(out.values())
    if isinstance(out, (list, tuple)):
        for item in out:
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def _probe_input(ref) -> Optional[dict]:
    """Kwargs that will drive this model's forward, asked of the model rather than assumed.

    Order matters: a model that publishes `dummy_inputs` has stated what it takes, so that is used
    verbatim. Otherwise the required parameters are read off the forward signature and sized from
    the model's own config, with the element type taken from the annotation the module carries.
    Nothing here names an argument -- a model that calls its tokens something unusual still works.
    """
    import torch

    published = getattr(ref, "dummy_inputs", None)
    if isinstance(published, dict) and published:
        return dict(published)

    cfg = getattr(ref, "config", None)
    if cfg is None:
        return None
    vocab = getattr(cfg, "vocab_size", None)
    hidden = getattr(cfg, "hidden_size", None)
    try:
        params = inspect.signature(ref.forward).parameters
    except (TypeError, ValueError):
        return None

    built = {}
    for p in params.values():
        if p.name == "self" or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if p.default is not inspect.Parameter.empty:
            continue  # the model defaults it; leave it to the model
        text = (p.annotation if isinstance(p.annotation, str) else str(p.annotation)).lower()
        if "long" in text or "int" in text:
            if not isinstance(vocab, int) or vocab < 2:
                return None
            built[p.name] = torch.randint(1, min(vocab, 1000), (1, _PROBE_SEQ), dtype=torch.long)
        elif isinstance(hidden, int):
            built[p.name] = torch.randn(1, _PROBE_SEQ, hidden)
        else:
            return None
    return built or None


def check_invariants(ref) -> dict:
    """Advisory: does this reference obey properties every correct implementation must obey?

    Provenance answers "are these the shipped weights"; it cannot answer "are they wired up
    right". A transposed projection or a wrong RoPE base loads real weights, returns real-looking
    numbers, and has no golden to be checked against -- that is the whole reason the loader had to
    be generated. These sidestep the missing golden: rather than asking whether an output is
    CORRECT, they ask whether the model contradicts itself, which needs nothing to compare to.

    A violation is positive evidence of a wiring bug and the gate treats it as fatal. Absence of
    evidence is not: each check reports `skipped` when its precondition is absent, and a model
    this cannot drive comes back `unverified` rather than broken.
    """
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return {"status": "unverified", "reason": f"torch unavailable: {exc}"}

    kwargs = _probe_input(ref)
    if not kwargs:
        return {"status": "unverified", "reason": "could not build an input the model would accept"}

    def _run(kw):
        with torch.no_grad():
            return _first_tensor(ref(**kw))

    try:
        base = _run(kwargs)
    except Exception as exc:  # noqa: BLE001 -- an un-runnable probe is not a verdict on the model
        return {"status": "unverified", "reason": f"probe forward raised {type(exc).__name__}: {exc}"}
    if base is None:
        return {"status": "unverified", "reason": "forward returned no tensor to inspect"}

    checks: dict = {}

    # Same input twice. A model that cannot reproduce its own output cannot be a stable reference.
    try:
        checks["determinism"] = "pass" if _close(_run(kwargs), base) else "FAIL"
    except Exception:
        checks["determinism"] = "skipped"

    # An input batched beside a DIFFERENT one must come back unchanged: samples do not interact.
    # The second row has to differ, which is the whole subtlety here -- batching an input against a
    # copy of itself leaves any averaging across the batch invisible, since the mean of identical
    # rows is the row. Rolling gives a neighbour that is genuinely different but equally valid.
    try:
        batched = {
            k: torch.cat([v, torch.roll(v, 1, dims=-1)], dim=0) if hasattr(v, "dim") and v.dim() > 0 else v
            for k, v in kwargs.items()
        }
        out = _run(batched)
        # The rolled rows were APPENDED, so the original input is the leading base.shape[0] of
        # them. Slicing a single row instead would broadcast against a multi-row base and report a
        # mismatch that is an artefact of the comparison rather than anything the model did.
        checks["batch_invariance"] = "pass" if out is not None and _close(out[: base.shape[0]], base) else "FAIL"
    except Exception:
        checks["batch_invariance"] = "skipped"

    # Editing a later position must not move an earlier one: a causal model cannot see its future.
    # Attention-mask and position/RoPE mistakes show up here, and only here, without a golden.
    try:
        edited = None
        for k, v in kwargs.items() if _is_autoregressive(ref) else ():
            if hasattr(v, "dim") and v.dim() == 2 and not v.is_floating_point() and v.shape[1] > 1:
                # Swap in a value the tensor already holds rather than incrementing: an id one past
                # the largest drawn can be one past the vocabulary, and the embedding lookup would
                # raise -- which this reads as "could not run", quietly losing the check.
                differs = (v[0, :-1] != v[0, -1]).nonzero()
                if differs.numel() == 0:
                    continue
                bumped = v.clone()
                bumped[0, -1] = v[0, int(differs[0])]
                edited = dict(kwargs, **{k: bumped})
                break
        if edited is None:
            checks["causality"] = "skipped"
        else:
            out = _run(edited)
            checks["causality"] = "pass" if out is not None and _close(out[:, :-1], base[:, :-1]) else "FAIL"
    except Exception:
        checks["causality"] = "skipped"

    failed = sorted(k for k, v in checks.items() if v == "FAIL")
    if failed:
        return {
            "status": "violated",
            "checks": checks,
            "reason": (
                f"the reference breaks {', '.join(failed)}, which every correct implementation "
                f"holds regardless of weights -- it is wired wrongly, and every PCC measured "
                f"against it inherits that"
            ),
        }
    if all(v == "skipped" for v in checks.values()):
        return {"status": "unverified", "checks": checks, "reason": "no invariant could be driven"}
    return {"status": "holds", "checks": checks, "reason": "self-consistent under the checks that ran"}


def _checkpoint_files(model_id: str) -> List[Path]:
    """Shipped safetensors shards for `model_id`, local dir or hub cache, newest-first by size."""
    if os.path.isdir(model_id):
        found = list(Path(model_id).rglob("*.safetensors"))
    else:
        try:
            from huggingface_hub import snapshot_download

            root = snapshot_download(model_id, allow_patterns=["*.safetensors"], local_files_only=True)
            found = list(Path(root).rglob("*.safetensors"))
        except Exception:
            return []
    return sorted(found, key=lambda p: p.stat().st_size, reverse=True)


def _fingerprint(t) -> Optional[Tuple[int, float, float]]:
    """(numel, mean, std) of a float tensor, or None if it is not one worth comparing.

    Deliberately order-insensitive: a correct loader is allowed to permute weights (RoPE layouts
    differ between checkpoint and `transformers` conventions), so comparing values positionally
    would flag correct conversions. Whole-tensor moments survive any permutation.
    """
    try:
        if not t.is_floating_point() or t.numel() < _FINGERPRINT_MIN_NUMEL:
            return None
        f = t.detach().float()
        return (int(t.numel()), float(f.mean()), float(f.std()))
    except Exception:
        return None


def weight_provenance(model_id: str, ref) -> dict:
    """Advisory: do this module's parameters actually come from the shipped checkpoint?

    This is the closest thing to numerical proof available without a golden output to compare
    against. It cannot confirm the maths is right, but it does catch the failure that is otherwise
    invisible -- a loader that builds the architecture from config and never loads the weights, so
    PCC is measured against a randomly-initialised "reference" and means nothing.

    Matching is on (numel, mean, std) because a real load copies most tensors through unchanged,
    while random init reproduces neither the moments nor the per-tensor spread of trained weights.
    Advisory only: a loader that legitimately transforms weights (dequantising, merging LoRA) can
    score low while being correct, so this reports and never blocks.
    """
    files = _checkpoint_files(model_id)
    if not files:
        return {"status": "unverified", "reason": "no local safetensors checkpoint to compare against"}
    try:
        from safetensors import safe_open
    except Exception as exc:  # noqa: BLE001 -- absence of the reader is an environment fact
        return {"status": "unverified", "reason": f"safetensors unavailable: {exc}"}

    have = {}
    for p in ref.parameters():
        fp = _fingerprint(p)
        if fp:
            have.setdefault(fp[0], []).append(fp)
    if not have:
        return {"status": "unverified", "reason": "no parameters large enough to fingerprint"}

    checked = matched = 0
    try:
        with safe_open(str(files[0]), framework="pt") as f:
            for key in list(f.keys()):
                if checked >= _PROVENANCE_SAMPLE:
                    break
                fp = _fingerprint(f.get_tensor(key))
                if not fp:
                    continue
                checked += 1
                matched += any(
                    abs(fp[1] - c[1]) <= _MOMENT_TOL and abs(fp[2] - c[2]) <= _MOMENT_TOL for c in have.get(fp[0], ())
                )
    except Exception as exc:  # noqa: BLE001 -- unreadable shard is not the loader's fault
        return {"status": "unverified", "reason": f"could not read checkpoint: {exc}"}

    if not checked:
        return {"status": "unverified", "reason": "no comparable tensors in checkpoint"}
    if matched:
        return {"status": "from_checkpoint", "reason": f"{matched}/{checked} sampled tensors match the checkpoint"}
    return {
        "status": "no_match",
        "reason": (
            f"0/{checked} sampled tensors match the shipped checkpoint -- the reference may be "
            f"randomly initialised, which would make any PCC against it meaningless"
        ),
    }


def _resolved(demo_dir: Path, reason: str) -> dict:
    """Success payload, carrying the random-weight caveat when one applies."""
    out = {"resolved": True, "path": str(loader_path(demo_dir)), "reason": reason}
    if uses_random_weights(demo_dir):
        out["random_weights"] = True
        out["caveat"] = (
            "reference built from RANDOM weights, not the shipped checkpoint: PCC against it "
            "verifies STRUCTURE only and does not bound numerical correctness"
        )
    return out


def _accept(demo_dir: Path, model_id: str, reason: str) -> dict:
    """Structural gate passed -- now run the thing before calling it resolved.

    Reporting a loader as resolved when it cannot produce a model is what turned a bad loader into
    a confusing failure somewhere downstream. A loader that is merely unverifiable here still
    counts as resolved; only one that demonstrably fails to build a model is rejected.
    """
    v = verify(demo_dir, model_id)
    if not v["ok"]:
        return {"resolved": False, "reason": f"{_LOADER_FILENAME} does not produce a model: {v['reason']}"}
    out = _resolved(demo_dir, reason)
    out["verification"] = v
    return out


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
        return _resolved(demo_dir, "loader already present")
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
        return _accept(demo_dir, model_id, "loader written")
    return {"resolved": False, "reason": f"agent did not produce a valid {_LOADER_FILENAME}"}


if __name__ == "__main__":
    _demo = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    _mid = sys.argv[2] if len(sys.argv) > 2 else ""
    print(json.dumps(resolve(model_id=_mid, demo_dir=_demo, failure_text="Could not load via AutoModel", enabled=True)))
