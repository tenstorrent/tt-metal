# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""emit-e2e — LLM-driven end-to-end pipeline builder (build agent + grader agent)."""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


def _pipeline_self_opens_device(demo_dir: Path):
    """Return [(rel_path, lineno)] for every ttnn.open_device / open_mesh_device
    call in the emitted tt/ package that is NOT inside an `if __name__ ==
    "__main__"` guard (fixes-plan Point 10).

    The pipeline must run on the single device passed into build_pipeline; the
    test fixture is the sole opener (with num_command_queues=2 + trace_region_size
    for the trace+2CQ lever). A second, ad-hoc open in the importable/callable path
    creates a competing device with a different command-queue count — the exact
    cause of the trace+2CQ `id < mesh_command_queues_.size()` fatal. A standalone
    `__main__` self-test opening its own device is fine and not flagged.
    """
    tt_pkg = demo_dir / "tt"
    hits = []
    if not tt_pkg.is_dir():
        return hits
    for p in sorted(tt_pkg.rglob("*.py")):
        try:
            src = p.read_text(errors="ignore")
            tree = ast.parse(src)
        except Exception:
            continue
        main_spans = [
            (n.lineno, n.end_lineno)
            for n in ast.walk(tree)
            if isinstance(n, ast.If)
            and isinstance(n.test, ast.Compare)
            and isinstance(n.test.left, ast.Name)
            and n.test.left.id == "__name__"
        ]
        for n in ast.walk(tree):
            if (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr in ("open_device", "open_mesh_device")
            ):
                ln = n.lineno
                if not any(a <= ln <= b for a, b in main_spans):
                    hits.append((str(p.relative_to(demo_dir)), ln))
    return hits


def _reset_device() -> str:
    chips = os.environ.get("TT_HW_PLANNER_RESET_CHIPS", "0,1,2,3")
    tt_smi = shutil.which("tt-smi") or "/home/ttuser/.tenstorrent-venv/bin/tt-smi"
    if not Path(tt_smi).exists():
        return "device reset SKIPPED (tt-smi not found)"
    try:
        r = subprocess.run([tt_smi, "-r", chips], capture_output=True, text=True, timeout=420)
        return "device reset (tt-smi -r %s) rc=%d" % (chips, r.returncode)
    except Exception as e:  # noqa: BLE001
        return "device reset FAILED (%s) — a hard boot may be required" % e


def _verbose() -> bool:
    """Screen-verbosity gate (matches the cli.py TT_HW_PLANNER_VERBOSE convention).
    Off by default: keep the terminal clean; the full agent stream always lands
    in the per-phase log file regardless."""
    return os.environ.get("TT_HW_PLANNER_VERBOSE", "") not in ("", "0", "false", "False")


def _e2e_cell(rel: str, sub: str, f) -> str:
    return f"`{rel}/{sub}/{f.name}`" if f else "(none)"


def _repoint_canonical_demo(demo_dir, demo_files) -> Optional[list]:
    """Make the advertised ``demo/demo.py`` lead to the real emitted pipeline(s)
    after emit-e2e passes, instead of leaving the CPU scaffold as a dead
    entrypoint (fixes-plan Point 8).

    Single task -> ``demo.py`` runs the one real ``demo_<task>.py``. Multi-task ->
    ``demo.py`` becomes a dispatcher that lists the per-task demos and runs one by
    name. Never raises; returns the demo filenames it pointed at, or None.
    """
    try:
        demo_root = Path(demo_dir) / "demo"
        names = [f.name for f in (demo_files or []) if f.name != "demo.py"]
        if not demo_root.is_dir() or not names:
            return None
        header = "# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.\n# SPDX-License-Identifier: Apache-2.0\n"
        if len(names) == 1:
            body = (
                header
                + '"""Canonical entrypoint: runs the real emitted pipeline demo (repointed by emit-e2e)."""\n'
                + "import os\nimport runpy\nimport sys\n\n"
                + "_HERE = os.path.dirname(os.path.abspath(__file__))\n"
                + f"_TARGET = os.path.join(_HERE, {names[0]!r})\n"
                + "sys.argv[0] = _TARGET\n"
                + 'runpy.run_path(_TARGET, run_name="__main__")\n'
            )
        else:
            body = (
                header
                + '"""Canonical dispatcher over the emitted per-task pipeline demos (repointed by emit-e2e)."""\n'
                + "import os\nimport runpy\nimport sys\n\n"
                + "_HERE = os.path.dirname(os.path.abspath(__file__))\n"
                + f"_DEMOS = {names!r}\n\n\n"
                + "def main() -> None:\n"
                + "    if len(sys.argv) > 1:\n"
                + "        pick = sys.argv[1]\n"
                + '        target = pick if pick.endswith(".py") else f"demo_{pick}.py"\n'
                + "        if target in _DEMOS:\n"
                + "            sys.argv = [os.path.join(_HERE, target)] + sys.argv[2:]\n"
                + '            runpy.run_path(os.path.join(_HERE, target), run_name="__main__")\n'
                + "            return\n"
                + '    print(f"This model has {len(_DEMOS)} task demo(s). Run one of:")\n'
                + "    for _n in _DEMOS:\n"
                + '        print(f"  python demo/{_n}")\n\n\n'
                + 'if __name__ == "__main__":\n    main()\n'
            )
        (demo_root / "demo.py").write_text(body)
        return names
    except Exception:
        return None


def emit_e2e_report(model_id: str, demo_dir, *, verdict: str = "PASS") -> None:
    """Consolidated end-of-emit-e2e report: the on-device vs CPU-fallback split, and
    per task/demo the real-input demo + full-model e2e PCC test + trace+2CQ perf test
    (with reproduce commands). Prints a terminal table AND writes <demo>/E2E_REPORT.md.
    Multimodal: one row per demo/task. Non-fatal — never raises."""
    import time as _time
    from pathlib import Path as _P

    try:
        demo_dir = _P(demo_dir)
        _dd = str(demo_dir)
        _mi = _dd.rfind("/models/")
        rel = _dd[_mi + 1 :] if _mi >= 0 else demo_dir.name

        split_lines = []
        try:
            from ..cli import _format_compute_split, _format_op_split

            split_lines += _format_compute_split(model_id, label="components")
            split_lines += _format_op_split(model_id, label="operations")
        except Exception:
            pass
        fallback = []
        try:
            from ..final_categorization import build_final_categorization

            _cat = build_final_categorization(model_id=model_id, demo_dir=demo_dir)
            fallback = sorted(
                list(getattr(_cat, "kernel_missing", []) or []) + list(getattr(_cat, "pending", []) or [])
            )
        except Exception:
            fallback = []

        demo_files = sorted((demo_dir / "demo").glob("demo_*.py")) if (demo_dir / "demo").is_dir() else []
        if str(verdict).upper() == "PASS":
            _repoint_canonical_demo(demo_dir, demo_files)
        e2e_dir = demo_dir / "tests" / "e2e"
        all_e2e = sorted(e2e_dir.glob("test_*.py")) if e2e_dir.is_dir() else []

        pcc_by_key = {}
        try:
            rep = json.loads((demo_dir / "grader_report.json").read_text())
            for c in rep.get("calls") or []:
                pccs = c.get("final_pcc") or []
                try:
                    pcc_by_key[str(c.get("call", "")).lower()] = min(float(x) for x in pccs) if pccs else None
                except Exception:
                    pcc_by_key[str(c.get("call", "")).lower()] = None
        except Exception:
            pass

        def _match(task, want_perf):
            for f in all_e2e:
                if task in f.name.lower() and (("perf" in f.name.lower()) == want_perf):
                    return f
            return None

        rows = []
        if demo_files:
            for dfp in demo_files:
                task = dfp.stem[len("demo_") :] if dfp.stem.startswith("demo_") else dfp.stem
                rows.append(
                    (task, dfp, pcc_by_key.get(task.lower()), _match(task.lower(), False), _match(task.lower(), True))
                )
        else:
            _pcc = None
            if pcc_by_key and all(v is not None for v in pcc_by_key.values()):
                _pcc = min(pcc_by_key.values())
            rows.append(
                (
                    "e2e",
                    None,
                    _pcc,
                    next((f for f in all_e2e if "perf" not in f.name.lower()), None),
                    next((f for f in all_e2e if "perf" in f.name.lower()), None),
                )
            )

        bar = "=" * 78
        print("\n" + bar)
        print(f"  E2E REPORT — {model_id}")
        print(f"  Verdict: {verdict}")
        print(bar)
        print("  PIPELINE PLACEMENT (on-device vs CPU fallback):")
        for ln in split_lines:
            print(ln)
        print(f"    CPU-fallback modules: {', '.join(fallback) if fallback else '(none — fully on device)'}")
        print("  " + "-" * 74)
        print(f"    {'task':<12} {'e2e PCC':<9} {'demo (real I/O)':<26} trace+2CQ perf test")
        for task, dfp, pcc, pcc_test, perf_test in rows:
            pcc_s = f"{pcc:.4f}" if isinstance(pcc, float) else "n/a"
            demo_s = (f"demo/{dfp.name}" if dfp else "(none)")[:26]
            print(f"    {task[:12]:<12} {pcc_s:<9} {demo_s:<26} {perf_test.name if perf_test else '(none)'}")
        print("  " + "-" * 74)
        print("  REPRODUCE (per task):")
        for task, dfp, pcc, pcc_test, perf_test in rows:
            print(f"    {task}:")
            if dfp:
                print(f"      demo   → python {rel}/demo/{dfp.name}")
            if pcc_test:
                print(f"      pcc    → pytest {rel}/tests/e2e/{pcc_test.name} -svv")
            if perf_test:
                print(f"      trace  → pytest {rel}/tests/e2e/{perf_test.name} -svv")
        print(f"  full report → {rel}/RUN_REPORT.md")
        print(bar)

        md = [
            f"# E2E report — `{model_id}`",
            "",
            f"_Generated: {_time.strftime('%Y-%m-%d %H:%M:%S %Z')}_",
            "",
            f"**Verdict: {verdict}**",
            "",
            "## Pipeline placement (on-device vs CPU fallback)",
            "",
        ]
        for ln in split_lines:
            md.append(f"- {ln.strip()}")
        md.append(
            "- CPU-fallback modules: "
            + (", ".join(f"`{m}`" for m in fallback) if fallback else "(none — fully on device)")
        )
        md += [
            "",
            "## Per task / demo",
            "",
            "| task | e2e PCC | demo (real input→output) | e2e PCC test | trace+2CQ perf test |",
            "|---|---|---|---|---|",
        ]
        for task, dfp, pcc, pcc_test, perf_test in rows:
            pcc_s = f"{pcc:.4f}" if isinstance(pcc, float) else "n/a"
            md.append(
                f"| `{task}` | {pcc_s} | {_e2e_cell(rel, 'demo', dfp)} | "
                f"{_e2e_cell(rel, 'tests/e2e', pcc_test)} | {_e2e_cell(rel, 'tests/e2e', perf_test)} |"
            )
        md += ["", "## Reproduce", ""]
        for task, dfp, pcc, pcc_test, perf_test in rows:
            md.append(f"### {task}")
            md.append("```bash")
            if dfp:
                md.append(f"python {rel}/demo/{dfp.name}")
            if pcc_test:
                md.append(f"pytest {rel}/tests/e2e/{pcc_test.name} -svv")
            if perf_test:
                md.append(f"pytest {rel}/tests/e2e/{perf_test.name} -svv")
            md.append("```")
            md.append("")
        from ..run_report import refresh_bringup_section, upsert_report_section

        refresh_bringup_section(demo_dir, model_id)
        upsert_report_section(demo_dir, "emit-e2e", "\n".join(md))
        try:
            (demo_dir / "E2E_REPORT.md").unlink()  # consolidated into RUN_REPORT.md — drop the standalone
        except OSError:
            pass
    except Exception as exc:
        print(f"  [e2e-report] skipped: {type(exc).__name__}: {exc}")


_G1_TORCH_DELEGATION = (
    r"self\._torch_module\s*\(",
    r"self\.torch_module\s*\(",
    r"_get_torch_submodule\s*\(",
)


def _class_to_task_slug(cls_name: str, model_prefix: str) -> str:
    tail = cls_name[len(model_prefix) :] if cls_name.startswith(model_prefix) else cls_name
    if tail in ("", "Model"):
        return "base"
    if tail.startswith("For"):
        tail = tail[3:]
    mapping = {
        "TextToText": "t2tt",
        "SpeechToText": "s2tt",
        "TextToSpeech": "t2st",
        "SpeechToSpeech": "s2st",
        "CausalLM": "lm",
        "ConditionalGeneration": "gen",
        "QuestionAnswering": "qa",
        "SequenceClassification": "seq_cls",
        "TokenClassification": "tok_cls",
        "MaskedLM": "mlm",
    }
    return mapping.get(tail, tail.lower())


def _enumerate_task_heads(model_id: str) -> list:
    try:
        import transformers
        from transformers import AutoConfig
    except Exception:
        return []
    try:
        cfg = AutoConfig.from_pretrained(model_id)
    except Exception:
        return []
    prefix_camel = ""
    archs = getattr(cfg, "architectures", None) or []
    if archs:
        base = archs[0]
        for suffix in ("Model", "ForCausalLM", "ForSeq2SeqLM", "ForConditionalGeneration"):
            if base.endswith(suffix):
                prefix_camel = base[: -len(suffix)]
                break
        if not prefix_camel:
            m = re.match(r"^(.+?)(For[A-Z]\w*|Model)$", base)
            prefix_camel = m.group(1) if m else base
    if not prefix_camel:
        model_type = getattr(cfg, "model_type", "") or ""
        prefix_camel = "".join(part.capitalize() for part in model_type.split("_"))
    if not prefix_camel:
        return []

    candidates = []
    for name in dir(transformers):
        if not name.startswith(prefix_camel):
            continue
        tail = name[len(prefix_camel) :]
        if tail and tail[0].islower():
            continue
        obj = getattr(transformers, name, None)
        if not isinstance(obj, type):
            continue
        if not hasattr(obj, "from_pretrained"):
            continue
        if name.endswith("PreTrainedModel") or name.endswith("Config"):
            continue
        if tail == "" or tail == "Model" or tail.startswith("For"):
            has_gen = hasattr(obj, "generate")
            if has_gen or tail == "Model":
                candidates.append((name, _class_to_task_slug(name, prefix_camel)))
    seen = set()
    unique = []
    for cls_name, slug in candidates:
        if cls_name in seen:
            continue
        seen.add(cls_name)
        unique.append({"class": cls_name, "task": slug})
    return unique


_G1B_HF_FALLBACK = (
    r"(?<!\w)hf_model\.text_decoder\s*\(",
    r"(?<!\w)hf_model\.t2u_model\s*\(",
    r"(?<!\w)hf_model\.vocoder\s*\(",
    r"(?<!\w)hf_model\.speech_encoder\s*\(",
    r"(?<!\w)hf_model\.text_encoder\s*\(",
    r"(?<!\w)hf_model\.lm_head\s*\(",
    r"(?<!\w)self\.hf_model\.text_decoder\s*\(",
    r"(?<!\w)self\.hf_model\.t2u_model\s*\(",
    r"(?<!\w)self\.hf_model\.vocoder\s*\(",
    r"(?<!\w)self\.hf_model\.speech_encoder\s*\(",
    r"(?<!\w)self\.hf_model\.text_encoder\s*\(",
    r"(?<!\w)self\.hf_model\.lm_head\s*\(",
    r"(?<!\w)hf_model\.(text_decoder|t2u_model|vocoder|speech_encoder|text_encoder|lm_head)\.\w+\s*\(",
    r"(?<!\w)self\.hf_model\.(text_decoder|t2u_model|vocoder|speech_encoder|text_encoder|lm_head)\.\w+\s*\(",
    r"(?<!\w)hf_model\.(text_decoder|t2u_model|vocoder|speech_encoder|text_encoder)\.[\w.\[\]0-9]+\s*\(",
    r"(?<!\w)self\.hf_model\.(text_decoder|t2u_model|vocoder|speech_encoder|text_encoder)\.[\w.\[\]0-9]+\s*\(",
)

_HF_ALIAS_ROOTS = ("text_decoder", "t2u_model", "vocoder", "speech_encoder", "text_encoder", "lm_head")

_TORCH_COMPUTE_BLOCKLIST = {
    "matmul",
    "mm",
    "bmm",
    "addmm",
    "einsum",
    "layer_norm",
    "batch_norm",
    "group_norm",
    "instance_norm",
    "rms_norm",
    "softmax",
    "log_softmax",
    "sigmoid",
    "tanh",
    "relu",
    "gelu",
    "silu",
    "leaky_relu",
    "elu",
    "hardswish",
    "hardsigmoid",
    "mish",
    "scaled_dot_product_attention",
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_transpose1d",
    "conv_transpose2d",
    "embedding",
    "embedding_bag",
    "dropout",
    "dropout1d",
    "dropout2d",
    "dropout3d",
    "argmax",
    "argmin",
    "topk",
    "multinomial",
}

_G5_HOST_SAMPLING = (
    r"torch\.argmax\s*\(",
    r"torch\.multinomial\s*\(",
    r"torch\.topk\s*\(",
)
_G5_HOST_XFER = ("from_torch", "to_torch", "from_device", "to_device")


def _host_op_gate_reason(probe_result):
    if not isinstance(probe_result, dict) or not probe_result.get("ran"):
        return (
            "on-device: fully-on-device NOT proven — the pipeline exposes no runnable "
            "host_op_selftest hook (emit it per COMMAND 3 so the observer can verify). "
            "Set E2E_ALLOW_HOST_DECODE=1 to waive for a genuinely host-bound model."
        )
    v = probe_result.get("verdict") or {}
    if v.get("on_device", True):
        return None
    ops = v.get("host_ops") or []
    return (
        "G5-observer: the forward executes host tensor ops (aten) on the compute path: "
        + ", ".join(ops[:8])
        + " — not fully on-device (observed via TorchDispatchMode; port those on-device, "
        "or waive with E2E_ALLOW_HOST_DECODE=1)"
    )


_G1B_HOT_FN_NAME = re.compile(
    r"^(run_[a-z0-9]+|__call__|forward|_forward|_apply_[a-z0-9_]+|decode_step|decode_prefill)$" r"|_trace_step$|_step$"
)
_G1B_SETUP_FN_NAME = re.compile(
    r"_trace_setup$|_write_inputs$|_reference_for_stage$|^_hf_reference_|"
    r"^_make_stage_inputs$|^_hf_capacities$|^load_hf_model$|^build_tt_stubs$|"
    r"^_make_arg_for$|^_captured_output_of$|^_shape_hidden$"
)


def _fn_is_hot(name: str) -> bool:
    """True iff a function name matches a HOT-path pattern (must not delegate
    to HF as computation). False for setup/reference functions that legitimately
    use HF to inject fixed reference tensors."""
    if _G1B_SETUP_FN_NAME.search(name):
        return False
    return bool(_G1B_HOT_FN_NAME.search(name))


def _check_hf_fallback(src: str) -> list:
    """AST pass: find HF-fallback / torch-compute patterns in HOT-path
    functions and any same-file helpers they call (transitively)."""
    import ast

    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []

    funcs_by_name: dict = {}
    for scope in ast.walk(tree):
        if not isinstance(scope, (ast.Module, ast.ClassDef)):
            continue
        for child in scope.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                funcs_by_name.setdefault(child.name, child)

    def _attr_root(node) -> str:
        """Return dotted string if the attribute chain contains `hf_model`
        (either as the root Name or nested like `self.hf_model.<X>`);
        else empty string."""
        parts = []
        cur = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        parts = list(reversed(parts))
        if "hf_model" in parts:
            return ".".join(parts)
        return ""

    def _dotted_call(node: "ast.Call") -> str:
        """Reconstruct `x.y.z(...)` string for a Call node's func chain."""
        parts = []
        cur = node.func
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        return ".".join(reversed(parts)) + "(...)"

    hits: list = []
    to_inspect: list = [(fn, None) for name, fn in funcs_by_name.items() if _fn_is_hot(name)]
    seen: set = set()

    while to_inspect:
        node, reached_via = to_inspect.pop()
        if node.name in seen:
            continue
        seen.add(node.name)
        if reached_via is not None and _G1B_SETUP_FN_NAME.search(node.name):
            continue

        aliases: dict = {}
        for sub in ast.walk(node):
            if isinstance(sub, ast.Assign):
                root = _attr_root(sub.value)
                if root and any(s in root for s in _HF_ALIAS_ROOTS):
                    for tgt in sub.targets:
                        if isinstance(tgt, ast.Name):
                            aliases[tgt.id] = root

        via_suffix = f" [via {reached_via}]" if reached_via else ""

        for sub in ast.walk(node):
            if not isinstance(sub, ast.Call):
                continue
            f = sub.func

            if (
                isinstance(f, ast.Attribute)
                and isinstance(f.value, ast.Name)
                and f.value.id == "self"
                and f.attr in funcs_by_name
                and f.attr not in seen
            ):
                caller_chain = f"{reached_via}->{node.name}" if reached_via else node.name
                to_inspect.append((funcs_by_name[f.attr], caller_chain))

            if isinstance(f, ast.Attribute):
                root = _attr_root(f)
                if root:
                    tail = root.split("hf_model.", 1)[-1] if "hf_model." in root else ""
                    top = tail.split(".", 1)[0]
                    if top in _HF_ALIAS_ROOTS:
                        hits.append(f"{node.name}(): {_dotted_call(sub)}{via_suffix}")
                        continue
            walker = f
            while isinstance(walker, ast.Attribute):
                walker = walker.value
            if isinstance(walker, ast.Name) and walker.id in aliases:
                hits.append(f"{node.name}(): {_dotted_call(sub)} [alias for {aliases[walker.id]}]{via_suffix}")
                continue
            if isinstance(f, ast.Name) and f.id in aliases:
                hits.append(f"{node.name}(): {f.id}(...) [alias for {aliases[f.id]}]{via_suffix}")
                continue
            if isinstance(f, ast.Attribute):
                chain = []
                walker = f
                while isinstance(walker, ast.Attribute):
                    chain.append(walker.attr)
                    walker = walker.value
                if isinstance(walker, ast.Name):
                    root_name = walker.id
                    leaf = chain[0] if chain else ""
                    if root_name == "torch" and "functional" in chain:
                        hits.append(f"{node.name}(): {_dotted_call(sub)} [torch.nn.functional in HOT path]{via_suffix}")
                        continue
                    if root_name == "torch" and leaf in _TORCH_COMPUTE_BLOCKLIST:
                        hits.append(f"{node.name}(): {_dotted_call(sub)} [torch compute in HOT path]{via_suffix}")
                        continue
                    if root_name == "ttnn" and leaf == "to_torch":
                        hits.append(
                            f"{node.name}(): {_dotted_call(sub)} [host-copy fingerprint: ttnn.to_torch in HOT path — data leaves device, host compute implied]{via_suffix}"
                        )
                        continue
                if isinstance(walker, ast.Name) and walker.id == "F":
                    hits.append(
                        f"{node.name}(): {_dotted_call(sub)} [F.* (torch.nn.functional) in HOT path]{via_suffix}"
                    )
                    continue
    return hits


def _run_deterministic_gates(demo_dir: Path, pcc: float, timeout_s: int):
    """Model-agnostic gate runner: G1 native, G2/G3 (run tests/e2e), G4 demo/ structure. Returns (ok, reasons)."""
    reasons = []
    e2e_dir = demo_dir / "tests" / "e2e"
    test_files = sorted(e2e_dir.glob("test_*.py")) if e2e_dir.is_dir() else []
    if not test_files:
        return False, ["G2/G3: no tests/e2e/test_*.py to run"]

    demo_subdir = demo_dir / "demo"
    demo_entrypoints = sorted(demo_subdir.glob("demo_*.py")) if demo_subdir.is_dir() else []
    if not demo_entrypoints:
        reasons.append("G4 structure: no runnable demo/demo_*.py entrypoint (standard layout requires per-Call demos)")
    else:
        no_main = [p.name for p in demo_entrypoints if "__main__" not in p.read_text(errors="ignore")]
        if no_main:
            reasons.append(f"G4 structure: demo entrypoint(s) missing `__main__` (not runnable): {', '.join(no_main)}")
    if not (demo_dir / "tt").is_dir():
        reasons.append("G4 structure: no tt/ package (standard demo layout)")
    if not (demo_dir / "README.md").is_file():
        reasons.append("G4 structure: no README.md (standard demo layout)")

    _self_opens = _pipeline_self_opens_device(demo_dir)
    if _self_opens:
        _where = ", ".join(f"{rel}:{ln}" for rel, ln in _self_opens)
        reasons.append(
            "G5 device-ownership: the pipeline opens its own device "
            f"(ttnn.open_device/open_mesh_device at {_where}). The pipeline MUST run on the `device` passed "
            "into build_pipeline — the test fixture is the SOLE device opener (it opens once with "
            "num_command_queues=2 + trace_region_size for the trace+2CQ lever). A second ad-hoc open creates a "
            "competing device with a different command-queue count, which is what breaks trace+2CQ with "
            "`id < mesh_command_queues_.size()`. Remove these open_device calls and thread the passed-in device through."
        )

    if os.environ.get("E2E_ALL_TASKS", "0") == "1":
        model_id = os.environ.get("E2E_MODEL_ID", "")
        if model_id:
            required_heads = _enumerate_task_heads(model_id)
            emitted_slugs = {p.stem.replace("demo_", "", 1) for p in demo_entrypoints}
            missing = [h for h in required_heads if h["task"] not in emitted_slugs]
            if missing:
                names = ", ".join(f"{h['task']} ({h['class']})" for h in missing)
                reasons.append(
                    f"G4 --all-tasks: {len(missing)}/{len(required_heads)} required task-head demo(s) missing: {names} "
                    f"(one demo/demo_<task>.py per HF-registered head required — do not collapse)"
                )

    if os.environ.get("E2E_REQUIRE_ON_DEVICE", "1") != "0" and os.environ.get("E2E_ALLOW_HOST_DECODE") != "1":
        repo = demo_dir
        for parent in demo_dir.parents:
            if (parent / "models").is_dir():
                repo = parent
                break
        hop = repo / "scripts" / "tt_hw_planner" / "_host_op_probe.py"
        if hop.is_file():
            penv3 = dict(os.environ)
            penv3["TT_METAL_HOME"] = str(repo)
            penv3["PYTHONPATH"] = str(repo) + os.pathsep + penv3.get("PYTHONPATH", "")
            penv3.pop("TT_METAL_DEVICE_PROFILER", None)
            _pb3 = repo / "python_env" / "bin" / "python"
            _pbin3 = str(_pb3) if _pb3.exists() else sys.executable
            try:
                pr3 = subprocess.run(
                    [_pbin3, str(hop), str(demo_dir)],
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                    cwd=str(repo),
                    env=penv3,
                )
                _res = None
                for line in ((pr3.stdout or "") + "\n" + (pr3.stderr or "")).splitlines():
                    if line.startswith("HOST_OP_PROBE="):
                        try:
                            _res = json.loads(line.split("=", 1)[1])
                        except Exception:  # noqa: BLE001
                            _res = None
                _r = _host_op_gate_reason(_res)
                if _r:
                    reasons.append(_r)
            except Exception as _e:  # noqa: BLE001
                reasons.append("on-device: host-op observer probe could not run (%s) — cannot prove on-device" % _e)

    py = sys.executable
    for parent in [Path.cwd(), *demo_dir.parents]:
        cand = parent / "python_env" / "bin" / "python"
        if cand.exists():
            py = str(cand)
            break
    demo_repo_root = demo_dir
    for parent in demo_dir.parents:
        if (parent / "models").is_dir():
            demo_repo_root = parent
            break
    gate_env = dict(os.environ)
    gate_env["PYTHONPATH"] = str(demo_repo_root) + os.pathsep + gate_env.get("PYTHONPATH", "")
    gate_env["TT_METAL_HOME"] = str(demo_repo_root)
    pytest_out = ""
    hang_timeout = min(int(timeout_s), int(os.environ.get("E2E_GATE_HANG_TIMEOUT", "2700")))
    gate_tests = [f for f in test_files if "perf" not in f.name] or test_files
    try:
        proc = subprocess.run(
            [py, "-m", "pytest", *[str(f) for f in gate_tests], "-p", "no:cacheprovider", "-rA", "-s"],
            capture_output=True,
            text=True,
            timeout=hang_timeout,
            cwd=str(demo_repo_root),
            env=gate_env,
        )
        pytest_out = proc.stdout or ""
        if proc.returncode != 0:
            tail = "\n".join(pytest_out.splitlines()[-15:])
            reasons.append(f"G2/G3: tests/e2e did not pass (pytest rc={proc.returncode}); tail:\n{tail}")
    except subprocess.TimeoutExpired:
        _rst = _reset_device()
        reasons.append(
            f"G2/G3: tests/e2e exceeded {hang_timeout}s with no verdict (likely device/fabric hang) — {_rst}"
        )

    for cnt, kind in re.findall(r"(\d+)\s+(xfailed|xpassed|skipped|errors?)\b", pytest_out):
        if int(cnt) > 0:
            reasons.append(
                f"G2/G3: tests/e2e reported {cnt} {kind} — a gate test may only PASS; "
                f"xfail/skip/error is not an accepted outcome (fix it or it stays a gate failure)"
            )

    for _val in re.findall(r"PCC[^=\n]*=\s*(-?\d+(?:\.\d+)?)", pytest_out):
        if float(_val) < pcc:
            reasons.append(f"G3: measured PCC {_val} < required {pcc} (tool-enforced threshold)")

    for p in test_files:
        try:
            src = p.read_text(errors="ignore")
        except Exception:
            continue
        if re.search(r"pytest\.xfail|mark\.xfail|pytest\.skip|assert\s+True\b", src):
            reasons.append(f"honesty: {p.name} contains pytest.xfail / pytest.skip / assert True")

    # --- Enumerate any stubs still on torch fallback (static + runtime) --------
    # Two complementary scans, both pure read-side, both silent when clean:
    #   G1-static  — grep _stubs/*.py for `self._torch_module(...)` / `_get_torch_submodule(...)`
    #                (the un-graduated scaffold shape). Wires up the previously-dead
    #                _G1_TORCH_DELEGATION patterns from line 282 so operators see WHICH
    #                stubs the LLM never rewrote, not just an aggregate count.
    #   G1-runtime — read _runtime_fallbacks.json for components that fell back during
    #                the pytest run just completed. Complements the static scan when a
    #                stub looks native but its runtime except-branch fired (e.g. conv2d
    #                CPU-fallback under an edge-case shape).
    # When both scans return empty (fully graduated pipeline), no reason is appended
    # and gate behavior is identical to before. Purely additive diagnostic on exit.
    _stubs_dir = demo_dir / "_stubs"
    _static_offenders: List[str] = []
    if _stubs_dir.is_dir():
        for _stub in sorted(_stubs_dir.glob("*.py")):
            if _stub.name.startswith("_"):
                continue
            try:
                _src = _stub.read_text(errors="ignore")
            except Exception:
                continue
            for _pat in _G1_TORCH_DELEGATION:
                if re.search(_pat, _src):
                    _static_offenders.append(_stub.stem)
                    break
    if _static_offenders:
        _shown = _static_offenders[:12]
        _tail = f", +{len(_static_offenders) - 12} more" if len(_static_offenders) > 12 else ""
        reasons.append(
            f"G1-static: {len(_static_offenders)} stub(s) still route __call__ through torch "
            f"(scaffold not yet graduated by LLM): {', '.join(_shown)}{_tail}"
        )

    _rt_offenders: List[str] = []
    _rtj = demo_dir / "_runtime_fallbacks.json"
    if _rtj.is_file():
        try:
            _rt = json.loads(_rtj.read_text())
            if isinstance(_rt, dict):
                for _comp, _info in sorted(_rt.items()):
                    if not isinstance(_info, dict):
                        continue
                    _kinds = _info.get("kinds") or []
                    _helpers = _info.get("helpers") or []
                    if _helpers or _kinds:
                        _kk = ",".join(sorted(set(_kinds))) if _kinds else "?"
                        _rt_offenders.append(f"{_comp} ({_kk}, {len(_helpers)} call(s))")
        except Exception:
            pass
    if _rt_offenders:
        _shown = _rt_offenders[:8]
        _tail = f"; +{len(_rt_offenders) - 8} more" if len(_rt_offenders) > 8 else ""
        reasons.append(
            f"G1-runtime: {len(_rt_offenders)} component(s) hit CPU fallback during pytest: "
            f"{'; '.join(_shown)}{_tail}"
        )

    _rt_off = os.environ.get("E2E_REQUIRE_TRACE", "1") == "0"
    _anno = os.environ.get("E2E_ALLOW_NO_TRACE") == "1"
    _ack = os.environ.get("E2E_I_KNOW_TRACE_IS_BROKEN") == "1"
    if _rt_off and not _ack:
        reasons.append(
            "G6 trace+2CQ: E2E_REQUIRE_TRACE=0 requires paired E2E_I_KNOW_TRACE_IS_BROKEN=1 "
            "acknowledgement — refusing to silently skip the trace-capture gate. Without this "
            "gate, a stub with host-side torch ops or mid-forward host->device writes can pass "
            "emit-e2e and then fail every trace+2CQ measurement downstream (scorecard N/A). "
            "Set BOTH env vars if you truly want to waive."
        )
    elif _rt_off and _ack:
        print("[emit-e2e] WARNING: G6 trace+2CQ gate DISABLED via E2E_I_KNOW_TRACE_IS_BROKEN=1")
        print("[emit-e2e]          emitted pipeline may not run trace+2CQ in optimize.")

    if not _rt_off and not _anno and not reasons:
        probe_py = Path(__file__).resolve().parent.parent / "_trace_capture_probe.py"
        if probe_py.is_file():
            tenv = dict(os.environ)
            tenv["TT_METAL_HOME"] = str(demo_repo_root)
            tenv["PYTHONPATH"] = str(demo_repo_root) + os.pathsep + tenv.get("PYTHONPATH", "")
            g6_hang = min(int(hang_timeout), int(os.environ.get("E2E_G6_HANG_TIMEOUT", "600")))
            proc = subprocess.Popen(
                [py, str(probe_py), str(demo_dir)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(demo_repo_root),
                env=tenv,
                start_new_session=True,
            )
            stdout, stderr = "", ""
            timed_out = False
            try:
                stdout, stderr = proc.communicate(timeout=g6_hang)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    pgid = os.getpgid(proc.pid)
                    os.killpg(pgid, signal.SIGTERM)
                    for _ in range(10):
                        if proc.poll() is not None:
                            break
                        time.sleep(0.5)
                    if proc.poll() is None:
                        os.killpg(pgid, signal.SIGKILL)
                except Exception:  # noqa: BLE001
                    pass
                try:
                    proc.wait(timeout=10)
                except Exception:  # noqa: BLE001
                    pass
            except Exception:  # noqa: BLE001
                pass

            if timed_out:
                _rst = _reset_device()
                reasons.append(
                    f"G6 trace+2CQ: trace-capture probe hung >{g6_hang}s "
                    f"(subprocess group killed, {_rst}); fix-loop should treat as failure and iterate"
                )
            else:
                tr = None
                for line in ((stdout or "") + "\n" + (stderr or "")).splitlines():
                    if line.startswith("TRACE_PROBE="):
                        try:
                            tr = json.loads(line.split("=", 1)[1])
                        except Exception:  # noqa: BLE001
                            tr = None
                if tr is None:
                    reasons.append("G6 trace+2CQ: trace-capture probe produced no verdict (could not run)")
                elif not tr.get("trace_ready"):
                    _b = "; ".join(x.get("guidance", x.get("rung", "")) for x in (tr.get("static_blockers") or []))
                    _cap = (tr.get("device_capture") or {}).get("reason", "")
                    reasons.append(
                        "G6 trace+2CQ: pipeline not trace+2CQ-validated per stage — "
                        + (_b or _cap or "capture failed")
                        + " (set E2E_ALLOW_NO_TRACE=1 to waive for a genuinely non-traceable model)"
                    )

    try:
        from ..trace_gate import build_fix_directive, evaluate_trace_gate, overflow_fix_loop, record_trace_verdict

        _proof = None
        _proof_raw = os.environ.get("E2E_TRACE_OVERFLOW_PROOF")
        if _proof_raw:
            try:
                _proof = json.loads(_proof_raw)
            except Exception:  # noqa: BLE001
                _proof = None
        _tg = evaluate_trace_gate(demo_dir, allow_no_trace=_anno, overflow_proof=_proof)
        if _tg.get("verdict") == "FAIL" and _is_overflow_detail(_tg.get("capture_detail")):
            _fix = overflow_fix_loop(demo_dir)
            print("[emit-e2e] trace-gate overflow fix-loop: %s" % _fix.get("detail"))
            if _fix.get("resolved"):
                _tg = evaluate_trace_gate(demo_dir, trace_caps=_fix.get("caps"), allow_no_trace=_anno)
            elif _fix.get("proof"):
                _tg = evaluate_trace_gate(demo_dir, allow_no_trace=True, overflow_proof=_fix.get("proof"))
        for _r in _tg.get("reasons") or []:
            if _r not in reasons:
                reasons.append(_r)
        print(
            "[emit-e2e] trace-gate verdict=%s (%d graduated, %d ungraduated): %s"
            % (
                _tg.get("verdict"),
                len(_tg.get("policy", {}).get("graduated_modules") or []),
                len(_tg.get("policy", {}).get("eager_eligible_modules") or []),
                _tg.get("reason"),
            )
        )
        _fd = build_fix_directive(_tg)
        if _fd:
            print("[emit-e2e] trace-gate fix directive: %s" % _fd)
        record_trace_verdict(demo_dir, _tg)
    except Exception as _tge:  # noqa: BLE001
        print("[emit-e2e] trace-gate evaluation skipped: %s" % _tge)

    return (len(reasons) == 0), reasons


def _is_overflow_detail(detail):
    d = (detail or "").lower()
    return any(m in d for m in ("trace region", "trace_region", "overflow", "out of memory", "oom", "not enough space"))


_TT_ONLY_CONTRACT = """
================ STRICT TT-ONLY CONTRACT — gate WILL auto-fail on violation ================

The TT pipeline MUST be a pure TTNN forward path. The following are FORBIDDEN in the
pipeline's HOT path (run_*, __call__, forward, _apply_*, decode_step, decode_prefill,
*_trace_step, *_step and any function you write that gets called from these):

  A. HF orchestration / HF wrappers:
       - model.generate(...), self.model.generate(...), hf_model.generate(...)
       - hf_model.<X>(...) or self.model.<X>(...) where <X> is a submodule
         (text_encoder, text_decoder, t2u_model, vocoder, speech_encoder, lm_head, etc.)
       - Nested / aliased forms of the above:
         hf_model.vocoder.unit_embedding(...), voc = hf_model.vocoder → voc.dur_predictor(...)
       - Monkey-patching HF submodule forwards:
         self.model.text_decoder.forward = <wrapper>, or ANY assignment to a
         self.model.<X>.forward attribute. Patched-HF-then-generate() IS Approach B
         and IS a shortcut — the pipeline must be an EXPLICIT chain, not HF orchestration.

  B. torch compute wrappers (any torch host-side compute op):
       - torch.matmul, torch.mm, torch.bmm, torch.einsum
       - torch.softmax, torch.log_softmax
       - torch.layer_norm, torch.rms_norm, torch.batch_norm, torch.group_norm
       - torch.embedding, torch.embedding_bag
       - torch.conv1d, torch.conv2d, torch.conv3d, torch.conv_transpose*
       - torch.scaled_dot_product_attention
       - torch.relu, torch.gelu, torch.silu, torch.tanh, torch.sigmoid, torch.leaky_relu
       - torch.argmax, torch.topk, torch.multinomial (host sampling)
       - torch.dropout / torch.nn.functional.<X> / F.<X>
     Shape and dtype ops (torch.zeros, torch.tensor, torch.arange, torch.cat,
     torch.reshape, torch.expand, torch.repeat_interleave, torch.full_like,
     torch.manual_seed, torch.no_grad, .to(dtype)) are ALLOWED — they're just prep.

  C. Coverage-sweep shortcut for Gate 2:
     Any function named coverage_step / coverage_sweep / invoke_all_stubs /
     _touch_all_graduated / etc. that iterates over graduated stubs and calls
     each once just to check the "invoked" counter — that DOES NOT count for
     Gate 2. Every graduated stub must be INSIDE THE REAL FORWARD PATH,
     with its output actually feeding downstream computation on the way to
     the final task output. The gate will require this.

REQUIRED SHAPE:
  Write an EXPLICIT chain as a Python function you author yourself. Example:
    def run_<task>(inputs, stubs, hf_model, ...):
        x = stubs["seamless_m4_t_encoder"](inputs["input_ids"])        # TT
        for step in range(N):
            y = stubs["seamless_m4_t_decoder"](x, prev_tokens, ...)    # TT
            prev_tokens = ttnn.argmax(y, ...)                          # TT (on device)
        units = stubs["seamless_m4_t_t2u_model"](prev_tokens)          # TT
        wave = stubs["seamless_m4_t_hifi_gan"](units)                  # TT
        return wave

ALLOWED HF USAGE (SETUP / REFERENCE ONLY — NOT the forward path):
  1. hf_model.config.<X> / hf_model.generation_config.<X> — pure attribute reads
  2. weight extraction at build time: hf_model.<X>.<Y>.weight / .bias
  3. HF calls inside a _hf_reference_<task>() helper — computing the GOLDEN
     output for PCC comparison against your TT pipeline. This is separate
     from the TT pipeline.
  4. HF calls inside <stage>_trace_setup(inputs) — seeding fixed-value
     persistent buffers BEFORE trace capture begins, so trace+2CQ has stable
     inputs. The trace_step itself must be pure TT.
============================================================================================
"""


def _build_cc_fix_prompt(*, model_id, demo_dir, pcc) -> str:
    """Per-round prompt for the emit-e2e cc engine. The gate is the sole authority; the agent works
    exactly the failing gates it names and never weakens them."""
    return (
        f"You are finishing the end-to-end TTNN pipeline for {model_id} in {demo_dir} (required e2e "
        f"PCC >= {pcc}).\n"
        "LOOP every iteration: call mcp__e2e-mcp__termination_check FIRST. It is the SOLE authority on "
        "whether you are done (can_stop=true) and returns next_target = the FAILING gates: G1 (stubs "
        "must be native ttnn, not torch-delegating), G2/G3 (tests/e2e must PASS on device with measured "
        "PCC >= threshold; xfail/skip/error is NOT acceptable), G4 (demo/ + tt/ + README structure).\n"
        "Fix EXACTLY the failing gates by editing tests/e2e/, _stubs/, demo/, tt/ as needed. NEVER "
        "weaken, xfail, skip, or assert-True a gate. Re-run termination_check after each fix. STOP only "
        "when can_stop=true; if it is already true, do nothing.\n"
        "The gate is COMBINED: beyond G1-G4 it ALSO requires the pipeline be everything-on-device / "
        "trace-capturable (no per-layer weight streaming, no host token loop, a real device trace "
        "captures) so trace + 2CQ can run. next_target may name a host op (residency / token-feed / KV / "
        "fixed-shape decode step) — fix it ON DEVICE the same way, WITHOUT regressing PCC (correctness is "
        "re-checked first every round, so a host-free edit that breaks PCC is rejected immediately).\n"
        + _TT_ONLY_CONTRACT
    )


def cmd_emit_e2e(args) -> int:
    from .optimize import invalid_trace_flag_error

    _tf = invalid_trace_flag_error()
    if _tf:
        print("error: " + _tf)
        return 1
    return _emit_e2e_phase_a(args)


def _run_emit_e2e_cc(*, model_id, demo_dir, pcc, timeout_s, agent_bin, max_rounds) -> int:
    """emit-e2e cc engine: after the builder runs, drive the fix loop through the shared cc harness
    against the e2e_mcp deterministic gate (which REUSES the same G1–G4 `_run_deterministic_gates` the
    legacy loop uses). The gate is the sole stop authority. Returns 0 iff the gate reports can_stop."""
    import json as _json
    import os as _os

    from .. import cc_harness

    repo_root = Path(__file__).resolve().parents[3]
    thp_dir = repo_root / "scripts" / "tt_hw_planner"
    server_path = thp_dir / "e2e_mcp.py"
    pybin = str(repo_root / "python_env" / "bin" / "python")
    if not Path(pybin).is_file():
        pybin = sys.executable
    mcp_env = {
        "E2E_MCP_DEMO_DIR": str(demo_dir),
        "E2E_MCP_PCC": str(pcc),
        "E2E_MCP_TIMEOUT": str(timeout_s),
        "E2E_MODEL_ID": model_id,
        "E2E_ALL_TASKS": _os.environ.get("E2E_ALL_TASKS", "0"),
        "E2E_REQUIRE_TRACE": _os.environ.get("E2E_REQUIRE_TRACE", "1"),
        "E2E_REQUIRE_ON_DEVICE": _os.environ.get("E2E_REQUIRE_ON_DEVICE", "1"),
        "TT_METAL_HOME": str(repo_root),
        "PYTHONPATH": str(repo_root),
        "PATH": f"{repo_root / 'python_env' / 'bin'}{_os.pathsep}/usr/bin:/bin",
    }
    cfg = cc_harness.build_mcp_config(pybin, server_path, mcp_env, "e2e-mcp")
    cfg_path = thp_dir / f".e2e_mcp_config_{re.sub(r'[^A-Za-z0-9._-]', '_', model_id)}.json"
    cfg_path.write_text(_json.dumps(cfg, indent=2))
    env = dict(_os.environ)
    env["TT_METAL_HOME"] = str(repo_root)
    env["PYTHONPATH"] = str(repo_root)

    def gate_fn():
        return cc_harness.gate_status(pybin, thp_dir, "e2e_mcp", mcp_env, repo_root)

    prompt = _build_cc_fix_prompt(model_id=model_id, demo_dir=demo_dir, pcc=pcc)
    allowed = ["mcp__e2e-mcp__termination_check", "Read", "Edit", "Write", "Bash", "Grep", "Glob"]
    print("\n  ===== PHASE 3 (cc engine): harness fix-loop on the e2e gate =====\n")
    res = cc_harness.run_cc_loop(
        prompt=prompt,
        mcp_config_path=cfg_path,
        allowed_tools=allowed,
        cwd=repo_root,
        env=env,
        gate_fn=gate_fn,
        max_rounds=max_rounds,
        claude_bin=agent_bin,
    )
    final = gate_fn()
    sep = "=" * 78
    print("\n" + sep)
    print(f"  cc engine: rounds={res['rounds']} can_stop={final.get('can_stop')} halted={res['halted']}")
    print(sep)
    if final.get("can_stop"):
        emit_e2e_report(model_id, demo_dir, verdict="PASS")
    return 0 if final.get("can_stop") else 1


def _source_phase2_shard_stubs(demo_dir: Path) -> list:
    stub_dir = demo_dir / "_stubs"
    if not stub_dir.is_dir():
        return []
    sourced = []
    for snap in sorted(stub_dir.glob("*.py.last_good_sharded")):
        live = snap.with_suffix("")
        if live.suffix != ".py":
            continue
        try:
            live.write_bytes(snap.read_bytes())
            sourced.append(live.stem)
        except OSError:
            pass
    return sourced


def _emit_e2e_phase_a(args) -> int:
    try:
        from ..cli import _quiet_framework_logging

        _quiet_framework_logging()
    except Exception:
        pass
    model_id = args.model_id
    demo_dir = _resolve_demo_dir(args)
    pcc = float(getattr(args, "pcc_target", 0.9) or 0.9)

    os.environ["E2E_MODEL_ID"] = model_id
    os.environ["E2E_ALL_TASKS"] = "1" if bool(getattr(args, "all_tasks", False)) else "0"
    agent_model = getattr(args, "model", None) or "opus"
    agent_bin = getattr(args, "agent_bin", "claude") or "claude"
    timeout_s = int(getattr(args, "agent_timeout_s", 0) or 0) or 14400
    max_grade_rounds = int(getattr(args, "max_grade_rounds", 0) or 0) or 20

    # One consolidated full log for the whole run (builder + grader + fix
    # appended in order). Clean screen, complete log, no per-phase scatter.
    import re as _re

    _safe = _re.sub(r"[^A-Za-z0-9._-]", "_", model_id)
    full_log = Path("generated") / f"emit_e2e_{_safe}_full.log"
    try:
        full_log.parent.mkdir(parents=True, exist_ok=True)
        full_log.write_text("")  # start fresh each run
    except Exception:
        full_log = None

    sep = "=" * 78
    print(sep)
    print(f"  EMIT-E2E (LLM agent)  {model_id}")
    print(f"  demo_dir={demo_dir}  pcc>={pcc}  model={agent_model}")
    if full_log is not None:
        print(f"  full log (complete transcript) → {full_log}")
    print(sep)

    _pc = _planned_parallelism(model_id, args)
    from ..parallelism import read_parallelism_manifest

    _manifest = read_parallelism_manifest(demo_dir)
    _mismatch = _topology_mismatch(_manifest, _pc, _mesh_chip_count(getattr(args, "mesh", None)))
    if _mismatch:
        print(sep)
        print(f"  ✗ EMIT-E2E ABORTED (topology guard) — {_mismatch}")
        print(sep)
        return 2
    if _manifest and int(_manifest.get("tp", 1)) > 1:
        if _pc is not None:
            print(
                f"  ✓ topology guard: --mesh reverified against graduated split "
                f"TP={_manifest.get('tp')} x DP={_manifest.get('dp')} on {_manifest.get('chips')} chips"
            )
        # (a tp<=1 manifest, or no manifest, is not enforced — see _topology_mismatch)
    _parallel_note = _parallelism_prompt_block(_pc)
    if _pc is not None and _pc.chips > 1:
        print(f"  chip placement: {_pc.chips}-chip mesh → TP={_pc.tp} x DP={_pc.dp} (kernel-viability selected)")
        print("  builder will open the mesh at this split; tt-metal auto-discovers the fabric topology.")
        if _pc.tp > 1:
            _sharded = _source_phase2_shard_stubs(demo_dir)
            if _sharded:
                print(
                    f"  [shard] sourced Phase-2 TP-sharded stubs (compose as-is, do NOT replicate): {', '.join(_sharded)}"
                )
                _parallel_note += (
                    f"\n\nPHASE-2 SHARD STUBS ({', '.join(_sharded)}): these _stubs ALREADY implement the "
                    f"proven TP={_pc.tp} split (ShardTensorToMesh + all_reduce/cluster_axis). Compose them "
                    f"AS-IS on the mesh — do NOT rewrite their sharding to replication. Only components with "
                    f"NO shard implementation may be replicated. The final pipeline MUST contain "
                    f"ShardTensorToMesh + a collective (all_reduce/all_gather); a pure-replication pipeline is "
                    f"NOT an acceptable TP={_pc.tp} result."
                )
                _parallel_note += (
                    "\n\nSCHEME EVALUATION (do this BEFORE composing — the tool has verified the chip "
                    "count/TP/DP match bring-up, but the per-component SCHEME is YOUR check and it must "
                    "generalize to THIS model): for EACH sharded stub above, read its forward + docstring "
                    "to identify its parallel scheme (TP column/row, expert-parallel EP, sequence/SP, "
                    "replicate) and the mesh AXIS + COLLECTIVE it uses, then validate that scheme against "
                    f"THIS model's config at TP={_pc.tp} x DP={_pc.dp} — e.g. expert-parallel requires "
                    "n_routed_experts % (expert-axis degree) == 0; attention TP requires degree <= "
                    "num_key_value_heads; sequence-parallel requires seq % degree == 0; every sharded "
                    "component must agree on which physical mesh axis is the sharded (cols/TP) axis. If any "
                    "component's scheme is INCOMPATIBLE with this mesh/config, STOP and report it as a hole — "
                    "do NOT silently replicate it. Gathered output MUST still equal the single-device HF "
                    "golden; the on-device PCC gate is the final arbiter."
                )
            else:
                print(
                    "  [shard] no Phase-2 (.last_good_sharded) stubs present — components will be REPLICATED "
                    "(TP=1). Run the shard bring-up (promote TT_HW_PLANNER_SHARD=1) to produce them for real TP."
                )
        print(sep)

    print("\n  ===== PHASE 1+2: BUILDER agent (plan → build → iterate) =====\n")
    _trace_note = (
        _TRACE_PROMPT_BLOCK
        if (os.environ.get("E2E_REQUIRE_TRACE", "1") != "0" or os.environ.get("E2E_REQUIRE_ON_DEVICE", "1") != "0")
        else ""
    )
    build_prompt = _build_agent_prompt(
        model_id=model_id,
        demo_dir=demo_dir,
        pcc=pcc,
        parallel_note=_parallel_note,
        trace_note=_trace_note,
        all_tasks=bool(getattr(args, "all_tasks", False)),
    )
    rc_build, build_final = _run_agent(
        prompt=build_prompt,
        agent_bin=agent_bin,
        agent_model=agent_model,
        timeout_s=timeout_s,
        label="builder",
        log_path=full_log,
    )
    if rc_build != 0:
        if (demo_dir / "tt").is_dir():
            print(
                f"\n  ⚠ builder agent exited rc={rc_build}, but {demo_dir}/tt exists — "
                f"entering CC fix-loop against the gate from current state"
            )
        else:
            print(f"\n  ✗ builder agent exited rc={rc_build}; skipping grade")
            return 1
    else:
        print("  ✓ builder finished (exit 0)")

    return _run_emit_e2e_cc(
        model_id=model_id,
        demo_dir=demo_dir,
        pcc=pcc,
        timeout_s=timeout_s,
        agent_bin=agent_bin,
        max_rounds=max_grade_rounds,
    )


def _run_agent(*, prompt: str, agent_bin: str, agent_model: str, timeout_s: int, label="agent", log_path: Path = None):
    """Run one agent. The SCREEN always stays clean — only a throttled
    `· <label> working…` heartbeat — while the COMPLETE agent stream (narration,
    tool calls, results) is appended to ``log_path`` (one consolidated file for
    the whole emit-e2e run). The structured grader report is rendered by the
    caller. This is how emit-e2e gets a clean screen + one full log without a
    regex filter (the agent's free-form narration can't be pattern-matched)."""
    cmd = [
        agent_bin,
        "-p",
        prompt,
        "--model",
        agent_model,
        "--dangerously-skip-permissions",
        "--add-dir",
        str(Path.cwd()),
        "--output-format",
        "stream-json",
        "--verbose",
    ]
    log_fh = None
    if log_path is not None:
        try:
            log_fh = open(log_path, "a", buffering=1, errors="ignore")
        except Exception:
            log_fh = None
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(Path.cwd()),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError:
        print(f"  ✗ agent binary not found: {agent_bin!r}")
        if log_fh:
            log_fh.close()
        return 2, ""

    final_text = ""
    start = time.monotonic()
    last_hb = start
    tool_calls = 0
    HB_EVERY_S = 45
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if log_fh is not None:  # COMPLETE stream → one consolidated log file
                try:
                    log_fh.write(line)
                except Exception:
                    pass
            _rendered, final, _atext, n_tool = _render_stream_event(line)
            if final:
                final_text = final
            tool_calls += n_tool
            now = time.monotonic()  # CLEAN screen: heartbeat only, never the transcript
            if now - last_hb >= HB_EVERY_S:
                sys.stdout.write(f"  · {label} working… {int(now - start)}s, {tool_calls} tool calls\n")
                sys.stdout.flush()
                last_hb = now
        rc = proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        print(f"\n  ✗ agent exceeded {timeout_s}s; killed")
        if log_fh:
            log_fh.close()
        return 1, final_text
    finally:
        if log_fh:
            try:
                log_fh.close()
            except Exception:
                pass
    return (0 if rc == 0 else 1), final_text


def _render_stream_event(line: str):
    """Render one stream-json event to a screen line.

    Returns ``(rendered, final, assistant_text, n_tool_use)``: ``rendered`` is
    what to print under verbose (or ``None``), ``final`` is the agent's terminal
    ``result`` text, ``assistant_text`` is the raw text of an assistant turn
    (used to dedup the verbose final summary), and ``n_tool_use`` is how many
    tool calls this event carried (for the non-verbose progress heartbeat)."""
    line = line.rstrip("\n")
    if not line.strip():
        return None, None, None, 0
    try:
        ev = json.loads(line)
    except Exception:
        # Non-JSON lines (framework log spill) are noise on screen; the full
        # raw stream is in the log file. Show only under verbose.
        return (("  · " + line) if (_verbose() and line.strip()) else None), None, None, 0

    etype = ev.get("type")
    if etype == "system":
        # init / thinking_tokens / task_started / task_notification / task_updated
        # carry no signal for the watcher and arrive dozens of times — drop them.
        return None, None, None, 0

    if etype == "assistant":
        out = []
        text_parts = []
        n_tool = 0
        for c in (ev.get("message", {}) or {}).get("content", []) or []:
            t = c.get("type")
            if t == "text":
                txt = (c.get("text") or "").strip()
                if txt:
                    out.append("  " + txt.replace("\n", "\n  "))
                    text_parts.append(txt)
            elif t == "tool_use":
                n_tool += 1
                out.append("  → " + _fmt_tool(c.get("name", "?"), c.get("input", {}) or {}))
        return (
            ("\n".join(out) if out else None),
            None,
            ("\n".join(text_parts) if text_parts else None),
            n_tool,
        )

    if etype == "user":
        # Tool-result previews (`↳`) are the bulk of the on-screen clutter: file
        # headers, ttnn DEBUG dumps leaking through Read/Bash output, and the
        # agent's own `<tool_use_error>` retries. The preceding `→` action line
        # already says what the agent did, and the full result is in the log.
        # Keep these only under verbose.
        if not _verbose():
            return None, None, None, 0
        for c in (ev.get("message", {}) or {}).get("content", []) or []:
            if c.get("type") == "tool_result":
                content = c.get("content")
                txt = content if isinstance(content, str) else json.dumps(content)
                first = (txt or "").strip().splitlines()[0] if (txt or "").strip() else ""
                if first:
                    return "      ↳ " + first[:160], None, None, 0
        return None, None, None, 0

    if etype == "result":
        return None, ev.get("result") or "", None, 0

    return None, None, None, 0


def _fmt_tool(name: str, inp: dict) -> str:
    try:
        if name == "Bash":
            return "Bash: " + str(inp.get("command", ""))[:150]
        if name in ("Read", "Edit", "Write", "NotebookEdit"):
            return f"{name} {inp.get('file_path', inp.get('path', ''))}"
        if name in ("Grep", "Glob"):
            return f"{name} {inp.get('pattern', '')} {inp.get('path', '')}".rstrip()
        if name in ("Task", "Agent"):
            return f"{name}: {str(inp.get('description', inp.get('prompt', '')))[:120]}"
        return f"{name} {json.dumps(inp)[:120]}"
    except Exception:
        return name


def _mesh_chip_count(mesh_arg) -> int:
    if not mesh_arg:
        return 1
    try:
        prod = 1
        for tok in str(mesh_arg).lower().replace(",", "x").split("x"):
            if tok.strip():
                prod *= int(tok.strip())
        return max(prod, 1)
    except Exception:
        return 1


def _planned_parallelism(model_id: str, args):
    from ..parallelism import plan_parallelism

    return plan_parallelism(model_id, _mesh_chip_count(getattr(args, "mesh", None)))


def _topology_mismatch(manifest, pc, given_chips: int):
    """Deterministic fail-safe: return an error string if the --mesh-derived split disagrees with the
    topology bring-up actually graduated at (recorded in parallelism_manifest.json), else None.

    Only enforced when bring-up graduated a real sharded split (tp>1) — a single-device / replicate-only
    graduation has nothing to reproduce. This is the decidable floor (pure chip/tp/dp equality); the
    per-component SCHEME compatibility is evaluated by the builder LLM, not here."""
    if not manifest:
        return None  # no recorded topology (older bring-up) — don't block, fall through to current behavior
    g_chips = int(manifest.get("chips", 1))
    g_tp = int(manifest.get("tp", 1))
    g_dp = int(manifest.get("dp", 1))
    if g_tp <= 1:
        return None  # bring-up graduated single-device / replicate-only — nothing to enforce
    hint = f"Pass --mesh {g_dp}x{g_tp} (or --mesh {g_chips}) to match, or re-run bring-up at the new mesh."
    if pc is None:
        return (
            f"bring-up graduated TP={g_tp} x DP={g_dp} on {g_chips} chips, but --mesh implies "
            f"{given_chips} chip(s) (single-device). {hint}"
        )
    if pc.chips != g_chips or pc.tp != g_tp or pc.dp != g_dp:
        return (
            f"topology mismatch — bring-up graduated TP={g_tp} x DP={g_dp} on {g_chips} chips, but "
            f"--mesh implies TP={pc.tp} x DP={pc.dp} on {pc.chips} chips. {hint}"
        )
    return None


def _parallelism_prompt_block(pc) -> str:
    if pc is None or pc.chips <= 1:
        return ""
    return f"""

================ CHIP PLACEMENT — {pc.chips}-CHIP MESH (TP={pc.tp} x DP={pc.dp}) ================
The tool has selected this parallelism split for `{pc.chips}` chips by checking per-TP kernel
viability (largest kernel-viable TP degree that divides the mesh; the remaining chips become
data-parallel replicas). Place the pipeline on the mesh accordingly:

  - BEFORE opening the mesh, enable the inter-chip fabric: `ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)`.
    Without this, any CCL (all_gather / all_reduce) raises `TT_FATAL ... fabric_context_ != nullptr`.
    tt-metal AUTO-DISCOVERS the cluster topology, so do NOT set TT_MESH_GRAPH_DESC_PATH for any mesh size.
  - Open a mesh device of {pc.chips} chips via `ttnn.open_mesh_device(ttnn.MeshShape({pc.dp}, {pc.tp}))`
    (rows = DP={pc.dp}, cols = TP={pc.tp}); close it at the end. If only a single device is available
    at runtime, fall back to it and note that in the run output.
  - DATA-PARALLEL axis (DP={pc.dp}): replicate the model across the {pc.dp} replica rows using
    `mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)` when moving tensors on-device, and compose
    back with the matching composer. DP replicates weights (no per-chip memory saving) and runs
    independent data copies.
  - TENSOR-PARALLEL axis (TP={pc.tp}): where the reused ttnn modules already support column/row
    sharding, shard the sharded weights along the TP axis with `ttnn.ShardTensorToMesh(mesh_device, dim=<shard_dim>)`
    on the {pc.tp} TP columns; keep embeddings / norms / lm_head replicated. If a module does not
    expose a shard dim, keep it replicated rather than guessing a split.
  - The e2e PCC gate is unchanged: parity is still measured against the same HF golden. Placing the
    pipeline on more chips must NOT change the numerical result — only where it runs.
"""


_TRACE_PROMPT_BLOCK = """
================ COMMAND 3 — TRACE+2CQ CONTRACT (host-free full pipeline) ================
AFTER Gates 1-3 pass (correct + on-device), make the pipeline trace+2CQ-capturable per
STAGE. Derive the stages from the HF reference config (Source A) — architectures /
is_encoder_decoder / sub-configs give the phases (ForCausalLM -> [prefill, decode];
encoder-decoder -> [encode, prefill, decode]; add [vocode] for speech output). Record them
as `PIPELINE_STAGES = [...]` in tt/pipeline.py.

For EACH stage expose, ON THE PIPELINE object, the generic contract the perf/2CQ engine binds:
  <stage>_trace_setup(inputs): pin the stage's VARIABLE dim (the sequence axis; bound =
    config max_position_embeddings) to a fixed capacity C, and PRE-UPLOAD the padded input +
    every shape-dependent constant (causal mask, RoPE sin/cos, KV / cross-attn pad) into
    PERSISTENT device buffers OUTSIDE the trace. Take the constant VALUES FROM THE HF REFERENCE
    itself (call rotary_emb / _update_causal_mask; KV shape = kv_heads x head_dim) so they match
    the golden exactly. Mask the padded positions so output on [0:real_len] is unchanged.
  <stage>_trace_step(): ONE host-op-free forward at the fixed shape reading ONLY those persistent
    buffers (NO from_torch / NO per-call ttnn.zeros/arange INSIDE the trace).
  <stage>_write_inputs(): stage the next input on command-queue 1 (per-token for an AR decode
    stage; the prompt / next chunk for a one-shot stage) -> flips on the 2CQ path.
AR stages ALSO keep the decode contract (decode_prefill seeds resident self- AND, for a seq2seq
decoder, cross-attn KV; decode_step reads them, never recomputes).

Expose a MODULE-LEVEL factory `build_pipeline(device, model=None, **kwargs)` in tt/pipeline.py that
CONSTRUCTS AND RETURNS the resident pipeline OBJECT — the one carrying PIPELINE_STAGES and the
per-stage <stage>_trace_setup/_trace_step/_write_inputs hooks (+ the AR decode contract). This is the
SINGLE entry the perf/2CQ harness (optimize's generated test) calls to OBTAIN that object for
measurement. It MUST return the object, NOT run it — no generate()/run_tts()/one-shot result, which
exposes none of the hooks and makes the trace engine skip. Accept and ignore any demo kwargs (text,
prompt, language, …) for call-signature compatibility; the resident build derives its shapes from the
config, not a prompt. `trace_capture_selftest` and the demo entry MUST build through this same factory
so there is ONE build surface.

Expose trace_capture_selftest(device): for EACH stage in PIPELINE_STAGES, capture ONE step in
ttnn.begin_trace_capture / end_trace_capture, execute_trace it, then RELEASE the trace before the
next stage (stage traces must NOT co-reside). Return True only if every stage captured host-free
AND its trace output matches the reference (PCC). Size trace_region_size from the LARGEST stage
(pinned C x layers); if a capture overflows the region, shrink C or degrade that stage to
single-CQ and PRINT the fallback (never silently drop). This recipe is identical for every model
— derive the specifics from the config; do NOT hardcode a per-model map.

Also expose host_op_selftest() — the AUTHORITATIVE fully-on-device check. Run the model's forward
under scripts.tt_hw_planner.host_op_observer.observe_host_ops(), with input-ENCODING (tokenize,
feature extraction / mel / FFT) and one-time weight build done OUTSIDE the observed region, and the
model math (encoded-inputs -> output, every stage incl. the prefix embedding) INSIDE it. Return
host_op_observer.verdict(ops). ttnn ops do not dispatch through torch, so a truly on-device forward
fires ZERO host aten ops; any aten op inside is host compute the ttnn-crossing checks cannot see
(e.g. a host-built prefix uploaded via ttnn.as_tensor / compute_embeddings on host). This is the
check that decides fully-on-device — do NOT leave host math (embedding, HF submodule forwards,
sampling) in the observed region. Multi-task: observe EACH task head's forward; fail if ANY fires
host aten ops.
"""


def _required_heads_block(model_id: str, all_tasks: bool) -> str:
    if not all_tasks:
        return ""
    heads = _enumerate_task_heads(model_id)
    if not heads:
        return ""
    lines = ["", "REQUIRED TASK HEADS (from HF transformers reference — enumerated automatically):"]
    for h in heads:
        lines.append(f"  - {h['class']:<50s} → demo/demo_{h['task']}.py + tests/e2e/test_e2e_{h['task']}.py")
    lines.append(
        f"You MUST emit ONE demo entrypoint + ONE e2e test PER head listed above "
        f'({len(heads)} in total). Do NOT collapse them into a smaller set ("one covers the others" '
        "is NOT acceptable — the gate counts demo_*.py files against this list). If two heads share "
        "internal stubs, factor the shared chain into tt/ and have each demo import + call it."
    )
    return "\n".join(lines) + "\n"


def _build_agent_prompt(
    *, model_id: str, demo_dir: Path, pcc: float, parallel_note: str = "", trace_note: str = "", all_tasks: bool = False
) -> str:
    heads_note = _required_heads_block(model_id, all_tasks)
    return f"""You are bringing up a REAL end-to-end TTNN pipeline for the model
`{model_id}`. Work in this repository with your tools (Read/Edit/Write/Bash).
{heads_note}

There are exactly TWO information sources. Use ONLY these — do NOT read any
sibling model under models/demos/<other-model>/:

  SOURCE A — HuggingFace hub for `{model_id}`:
    config.json, tokenizer/processor/feature_extractor, the AutoModel
    registry (which task heads this model supports), and the reference
    model + model.generate() as the golden output for parity.

  SOURCE B — the bring-up tool output for this model at:
    {demo_dir}
      - bringup_status.json   (components + status; GRADUATED = NEW with a
        `_stubs/<name>.py.last_good_native` OR `.py.last_good_sharded` snapshot.
        Bring-up is single-phase: TP=1 graduates a native single-device body,
        TP>1 graduates the shardable modules DIRECTLY sharded (the .last_good_sharded
        body already does ShardTensorToMesh + all_reduce). The LIVE `_stubs/<name>.py`
        IS the graduated body — compose it as-is. REUSE entries have no stub and are
        NOT graduated work products)
      - _stubs/*.py           (the graduated TTNN stubs; each exposes
                               build(device, torch_module) and a callable)
      - _captured/<name>/{{args,kwargs,output}}.pt   (HF golden tensors)
      - tests/pcc/            (per-component PCC tests)

================ COMMAND 1 — ACT AS PLANNER ================
Based on Group A and Group B information ONLY, act as a planner and create a
sketch plan (mental model) that produces a task_heads JSON with: what "pass"
means, which graduated stubs go where, the validation metric, behavioral
proof, and a self-validation plan. Make sure the pipeline uses ALL graduated
modules from Source B and does not leave any graduated module out. Correctly
VERIFY that the graduated modules are listed correctly so none are wasted.
Write the plan to {demo_dir}/e2e_plan.json.

================ COMMAND 2 — ORCHESTRATE THE BUILD ================
Based on that plan and only information from the plan, fire parallel agents
working on Call 1, Call 2, … Call N (the task heads) separately if there is no
dependency between them; if two calls share a graduated module, use only ONE
agent for them. Iterate using Gate 1, Gate 2, and Gate 3 until you have an
end-to-end pipeline ready:

  Gate 1 — every routed graduated stub is still real ttnn (not torch fallback);
           a sharded (TP>1) body counts as native — do NOT rewrite it to replication.
  Gate 2 — every graduated module is actually INVOKED in the pipeline run
           (no graduated module left out — this is critical).
  Gate 3 — the pipeline's FINAL output PCC vs the HF golden (Source A) is
           >= {pcc}.

CRITICAL REQUIREMENTS:
  - The pipeline must NOT be a smoke test. It must be a REAL pipeline that
    takes input exactly as collected from Sources A+B and emits output exactly
    as defined in Sources A+B (e.g. audio->text, text->text, text->audio).
    Input is constructed via the HF processor/tokenizer/feature_extractor;
    output is the real task output, compared to the HF reference (Source A).
  - It must chain the graduated stubs into the actual forward pass and produce
    real task output — not just pass tensors around. Each stage must be fed the
    previous TT stage's real output; NEVER inject a matched/reference tensor at
    a joint (that hides wiring bugs the e2e test exists to catch).
  - ALL graduated modules/components must be used in the pipeline.
  - The end-to-end pipeline must pass PCC >= {pcc}.
  - ALWAYS print the achieved end-to-end PCC on EVERY run, pass OR fail — e.g.
    `print(f"e2e PCC={{achieved_pcc}}")` on its own line immediately BEFORE the
    final assert — so the measured number is visible in the test output
    regardless of the verdict (not only surfaced in the assert message on fail).
  - GENERATIVE heads (reference is `model.generate()`): reproduce generate()'s
    real chain and compare the TT-generated output to it. To keep the on-device
    gate fast, CAP BOTH SIDES to the same small horizon N (e.g. 40): pass
    `max_new_tokens=N` to `model.generate()` AND stop the TT decode loop at N,
    then compare the first-N sequence (+ per-step PCC). Do NOT run full-length
    generation (too slow — the gate times out). Do NOT cap only the TT side
    while HF runs full length (lengths won't match → false fail, and HF is still
    slow). Both sides capped to the same N → fast, faithful, no false mismatch.

STRUCTURE — emit a complete, runnable package in the standard demo layout
(the same demo/ + tt/ + tests/ package style used by demos under models/demos/).
For ANY model, emit a complete, runnable package — not a lone test file:
  {demo_dir}/
    demo/         per-task runnable demo entrypoint(s) (one per Call) that load
                  real input, run the chained TTNN pipeline, emit real output.
                  EACH must have a `__main__` + argparse and be runnable as
                  `python -m ...demo.demo_<task>`.
    tt/           the ONE shared chained pipeline (the real forward pass over the
                  graduated stubs) that BOTH demo/ and tests/e2e/ import and call.
    tests/e2e/    the e2e pipeline test(s): real input -> chained stubs ->
                  real output, asserting Gate 1/2/3 (all stubs INVOKED + final
                  PCC >= {pcc} vs HF golden).
    README.md     what each Call does, how to run it, the PCC numbers.

  CRITICAL — DEMO AND TEST MUST SHARE ONE PIPELINE: the chained forward pass (the
  exact wiring of the graduated stubs) lives in `tt/` as a single function, and
  BOTH the demo entrypoint AND the e2e test import and call it. Do NOT write two
  separate copies of the wiring — if the demo has its own copy it WILL drift from
  the test and ship a broken pipeline while the test stays green. A passing test
  must GUARANTEE a working demo because they run identical code. emit-e2e's
  deliverable is a runnable demo; a green test with no/working demo is NOT done.
Match the conventions of existing demos under models/demos/ rather than
inventing a new layout. Keep iterating (fix the stub/wiring, re-run on the TT device) until the
gates pass. Use `./python_env/bin/python -m pytest <file> -s` to run on device.
Report a final summary: which calls are READY, the FINAL_PCC per call, and
confirm all graduated modules were invoked.
{parallel_note}{trace_note}
{_TT_ONLY_CONTRACT}
"""


def _resolve_demo_dir(args) -> Path:
    raw = getattr(args, "output", None)
    if raw:
        p = Path(raw)
        return p.parent if p.suffix == ".py" else p
    try:
        from ..scaffold_demo_folder import _slug

        slug = _slug(args.model_id.split("/")[-1])
    except Exception:
        slug = args.model_id.split("/")[-1].replace("-", "_").lower()
    demos_root = Path.cwd() / "models" / "demos"
    if demos_root.is_dir():
        for cand in demos_root.rglob(slug):
            if cand.is_dir() and (cand / "bringup_status.json").is_file():
                return cand
    return Path(f"models/demos/{slug}")
