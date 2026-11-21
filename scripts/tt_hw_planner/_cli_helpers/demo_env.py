"""Pre-flight check that the shared transformers stack's demo can run with the installed ``transformers``."""

from __future__ import annotations

from typing import List, Tuple

from ..discovery import BRINGUP_ROOT, safe_relative_to_root, ttt_dir


def _check_demo_environment_compat(
    *, demo_module_path: str = "models.tt_transformers.demo.simple_text_demo"
) -> Tuple[bool, List[str]]:
    """2026-05-23 environment pre-check: validate that the demo path
    the tool is about to invoke can actually run with the installed
    `transformers` version.

    Why this exists: `up --auto <supported-LLM>` ends up invoking
    `tt_transformers/simple_text_demo.py` via pytest. That demo was
    written against `transformers==4.x`. When `transformers==5.x` is
    installed (e.g. 5.8.1 on this box), the demo fails with errors
    like:
      - `ImportError: cannot import name 'AutoModelForVision2Seq'`
      - `TypeError: pow(NoneType, Tensor)` (rope_theta missing)
      - `RuntimeError: Could not infer dtype of tokenizers.Encoding`

    Each individual symptom is opaque. The user has to dig through
    repo code to figure out why their model crashed. The TOOL's
    contract is to catch this kind of thing upfront and abort with
    a clear message.

    Returns (ok, problems_list). On `not ok`, the caller is expected
    to print the problems and abort. Returns `(True, [])` when the
    environment looks clean and the demo should be able to run.

    Strategy: fast static checks only. No subprocess imports (those
    would take 5-10s and pull in ttnn). We grep the demo's import
    chain for known-broken symbols and version-check transformers."""
    problems: List[str] = []

    try:
        import transformers as _tf

        tf_version = getattr(_tf, "__version__", "(unknown)")
    except Exception as exc:
        problems.append(f"transformers is not importable ({type(exc).__name__}: {exc})")
        return False, problems

    try:
        major = int(str(tf_version).split(".")[0])
    except (ValueError, IndexError):
        major = 0

    if major >= 5:
        mc_path = BRINGUP_ROOT() / ttt_dir() / "tt" / "model_config.py"
        common_path = BRINGUP_ROOT() / ttt_dir() / "tt" / "common.py"
        repo_files_on_demo_path = [
            BRINGUP_ROOT() / "models" / "common" / "llama_models.py",
            mc_path,
            common_path,
        ]

        for p in repo_files_on_demo_path:
            if not p.is_file():
                continue
            try:
                src = p.read_text()
            except Exception:
                continue

            for i, line in enumerate(src.splitlines()):
                if "AutoModelForVision2Seq" not in line:
                    continue
                if "from transformers import" not in line:
                    continue
                window = "\n".join(src.splitlines()[max(0, i - 3) : i + 4])
                if "AutoModelForImageTextToText" not in window:
                    problems.append(
                        f"{safe_relative_to_root(p)}:{i + 1} "
                        f"imports `AutoModelForVision2Seq` from "
                        f"transformers (removed in 5.x; renamed "
                        f"to `AutoModelForImageTextToText`). The "
                        f"demo will ImportError before the model "
                        f"even loads."
                    )

        if mc_path.is_file():
            try:
                mc_src = mc_path.read_text()
            except Exception:
                mc_src = ""
            if 'text_config.get("rope_theta")' in mc_src and "rope_parameters" not in mc_src:
                problems.append(
                    f"{safe_relative_to_root(mc_path)} reads "
                    "`rope_theta` only; transformers 5.x migrates this "
                    "field to `rope_parameters['rope_theta']` for some "
                    "models (e.g. Phi-3.5). Runtime will crash with "
                    "`TypeError: pow(NoneType, Tensor)` at rope.py."
                )

        if common_path.is_file():
            try:
                common_src = common_path.read_text()
            except Exception:
                common_src = ""
            if "apply_chat_template" in common_src and (
                "_normalize_token_result_to_list" not in common_src
                and "_chat_template_ids" not in common_src
                and 'hasattr(result, "input_ids")' not in common_src
                and 'hasattr(encoded, "input_ids")' not in common_src
                and 'hasattr(encoded, "ids")' not in common_src
            ):
                problems.append(
                    f"{safe_relative_to_root(common_path)} calls "
                    "`apply_chat_template(tokenize=True)` but does "
                    "not normalize the return type. In transformers "
                    "5.x this returns `BatchEncoding` (or "
                    "`tokenizers.Encoding`), not `List[int]`. "
                    "Runtime will crash with `RuntimeError: Could "
                    "not infer dtype of tokenizers.Encoding`."
                )

    if problems:
        problems.insert(
            0,
            f"transformers=={tf_version} is installed; the "
            f"`{demo_module_path}` codepath assumes 4.x APIs in "
            f"several places. Detected issues:",
        )

    return (not problems), problems
