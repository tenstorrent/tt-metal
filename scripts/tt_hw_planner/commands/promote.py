from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def cmd_promote(args) -> int:
    from ..cli import (
        _API_KEY_ENV_VAR,
        _PROVIDER_LABEL,
        _auto_iteration_blockers,
        _check_agent_ready,
        _emit_and_verify_runnable_demo,
        _enforce_memory_fit_or_abort,
        _print_bringup_summary,
        _prompt_for_api_key,
        _resolve_tiered_model_aliases,
        _run_auto_iterate_loop,
        cmd_bringup,
    )

    MODEL = args.model_id
    BOX = args.box
    sep = "=" * 78

    def banner(title: str) -> None:
        print()
        print(sep)
        print(f"  {title}")
        print(sep)

    if getattr(args, "regen_demo_only", False):
        banner(f"REGEN-DEMO-ONLY for {MODEL}")
        ok, _ = _emit_and_verify_runnable_demo(MODEL, sep=sep)
        return 0 if ok else 1

    banner(f"PROMOTE  resume bring-up: replace CPU fallback with native TTNN for {MODEL}")

    from ..bringup_loop import find_demo_dir

    demo_dir = find_demo_dir(MODEL)
    if demo_dir is None:
        print(
            f"ERROR: no scaffolded demo for {MODEL}. Run bring-up first:\n"
            f"    python -m scripts.tt_hw_planner up {MODEL} --box {BOX} --execute\n",
            file=sys.stderr,
        )
        return 2

    mem_fit_rc = _enforce_memory_fit_or_abort(
        MODEL,
        box_name=BOX,
        mesh_str=getattr(args, "mesh", None),
        dtype_override=getattr(args, "dtype", None),
    )
    if mem_fit_rc is not None:
        return mem_fit_rc

    ungraduated, smoke_tests = _auto_iteration_blockers(MODEL)
    if not ungraduated and not smoke_tests:
        if not getattr(args, "auto", False):
            banner(f"PROMOTE  nothing to do — every NEW component is already native TTNN")
            _print_bringup_summary(MODEL, box=BOX, sep=sep)
            return 0

        all_new: List[str] = []
        try:
            status_path = demo_dir / "bringup_status.json"
            if status_path.is_file():
                _status_data = json.loads(status_path.read_text())
                for _c in _status_data.get("components", []):
                    if _c.get("status") == "NEW":
                        _name = str(_c.get("name", "")).strip()
                        if _name:
                            all_new.append(_name)
        except Exception:
            pass
        if not all_new:
            banner(f"PROMOTE  nothing to do — no NEW components tracked")
            _print_bringup_summary(MODEL, box=BOX, sep=sep)
            return 0
        ungraduated = sorted(set(all_new))
        smoke_tests = []
        banner(
            f"PROMOTE  --auto set: re-validating {len(ungraduated)} "
            f"structurally-native component(s) via pytest pre-flight "
            f"before declaring done"
        )

    targets = sorted(set(ungraduated + smoke_tests))
    print(f"  {len(targets)} component(s) targeted for native TTNN promotion:")
    for n in targets:
        print(f"    - {n}")
    print()
    print("  current state:")
    _print_bringup_summary(MODEL, box=BOX, sep=sep)

    if not getattr(args, "auto", False):
        print(
            "\n  --auto not set; run with --auto to drive native TTNN synthesis via LLM:\n"
            f"    python -m scripts.tt_hw_planner promote {MODEL} --box {BOX} --auto\n"
            "\n  Or hand-write replacements:\n"
            f"    python -m scripts.tt_hw_planner bringup {MODEL} --handoff-to-chat\n"
            f"    python -m scripts.tt_hw_planner bringup {MODEL} --apply-all-responses\n"
            f"    python -m scripts.tt_hw_planner up {MODEL} --box {BOX} --execute\n"
        )
        return 0

    provider = (getattr(args, "auto_agent", None) or "cursor").lower()
    if provider not in ("cursor", "claude"):
        banner(f"PROMOTE: unknown --auto-agent {provider!r}")
        print("  Supported: cursor, claude", file=sys.stderr)
        return 2

    ready, msg = _check_agent_ready(provider)
    if not ready:
        label = _PROVIDER_LABEL.get(provider, provider)
        env_var = _API_KEY_ENV_VAR.get(provider, "API_KEY")
        banner(f"--auto: {label} credentials not detected")
        print(msg)
        key = _prompt_for_api_key(provider)
        if key:
            ready, msg = _check_agent_ready(provider)

    if not ready:
        env_var = _API_KEY_ENV_VAR.get(provider, "API_KEY")
        label = _PROVIDER_LABEL.get(provider, provider)
        banner("LLM promotion not possible — no API keys provided")
        print(
            f"  Set the API key and re-run with --auto:\n"
            f"    export {env_var}=<your-key>\n"
            f"    python -m scripts.tt_hw_planner promote {MODEL} \\\n"
            f"        --box {BOX} --auto --auto-agent {provider}\n"
        )
        print(msg, file=sys.stderr)
        return 0

    agent_bin = msg
    model_alias = getattr(args, "auto_model", None)
    if not model_alias:
        model_alias = "opus" if provider == "claude" else "sonnet-4"
    model_light, model_heavy, model_super_heavy = _resolve_tiered_model_aliases(
        provider=provider,
        auto_model=model_alias,
        auto_model_light=getattr(args, "auto_model_light", None),
        auto_model_heavy=getattr(args, "auto_model_heavy", None),
        auto_model_super_heavy=getattr(args, "auto_model_super_heavy", None),
        auto_model_tiered=bool(getattr(args, "auto_model_tiered", False)),
    )
    if model_light or model_heavy or model_super_heavy:
        _super_label = f" → super_heavy={model_super_heavy}" if model_super_heavy else ""
        print(
            f"  [auto:{provider}] tiered model switching enabled: "
            f"light={model_light or model_alias}, heavy={model_heavy or model_alias}{_super_label}"
        )

    if bool(getattr(args, "op_synth", False)) and not bool(getattr(args, "no_op_synth", False)):
        banner(
            "PROMOTE  --op-synth: re-autofill CPU-fallback components with "
            "op-level partial stubs before entering auto-iterate"
        )
        try:
            promote_autofill_argv = argparse.Namespace(
                model_id=MODEL,
                next=False,
                component=None,
                autofill=True,
                overwrite_autofill=False,
                op_synth=True,
                run_tests=False,
                no_emit_tests=True,
                overwrite_tests=False,
                keep_passing_stubs=True,
                format="text",
                synthesize=False,
                synthesize_component=None,
                llm_provider=None,
                llm_model=None,
                llm_endpoint=None,
                llm_max_retries=2,
                llm_dry_run=False,
                no_fetch_upstream=False,
                emit_prompts=False,
                apply_response=None,
                handoff_to_chat=False,
                apply_all_responses=False,
                list_synth_targets=False,
            )
            cmd_bringup(promote_autofill_argv)
        except Exception as exc:
            print(
                f"  op-synth re-autofill failed (non-fatal, continuing with " f"existing stubs): {exc}",
                file=sys.stderr,
            )

    return _run_auto_iterate_loop(
        MODEL=MODEL,
        BOX=BOX,
        mesh=getattr(args, "mesh", None),
        dtype=getattr(args, "dtype", None),
        batch=getattr(args, "batch", 1),
        max_seq_len=getattr(args, "max_seq_len", 1024),
        max_generated_tokens=getattr(args, "max_generated_tokens", 200),
        accuracy=getattr(args, "accuracy", False),
        no_trace=getattr(args, "no_trace", False),
        no_paged_attention=getattr(args, "no_paged_attention", False),
        no_instruct=getattr(args, "no_instruct", False),
        download_first=getattr(args, "download_first", False),
        strict=getattr(args, "strict", False),
        demo_dir=demo_dir,
        provider=provider,
        agent_bin=agent_bin,
        model=model_alias,
        max_iters=getattr(args, "auto_max_iters", 5),
        sep=sep,
        target_components=targets,
        strict_native=True,
        agent_timeout_s=getattr(args, "auto_agent_timeout", 1500),
        allow_kill_stale=not getattr(args, "no_kill_stale", False),
        allow_device_reset=not getattr(args, "no_device_reset", False),
        max_attempts_per_component=getattr(args, "auto_max_attempts_per_component", 2),
        allow_partial_cpu=getattr(args, "allow_partial_cpu", False),
        model_light=model_light,
        model_heavy=model_heavy,
        model_super_heavy=model_super_heavy,
    )
