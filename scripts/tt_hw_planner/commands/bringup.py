from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def cmd_bringup(args) -> int:
    from ..cli import (
        LLMError,
        apply_all_responses,
        apply_response,
        autofill_stubs,
        build_handoff_master,
        emit_prompts,
        list_synth_targets,
        next_task,
        render_bringup_loop_json,
        render_bringup_loop_text,
        render_next,
        render_synth_json,
        render_synth_results,
        render_synth_targets,
        resolve_llm_config,
        run_bringup_loop,
        synthesize_all_new,
        synthesize_component,
    )

    if args.next:
        try:
            task = next_task(model_id=args.model_id, component=args.component)
        except (FileNotFoundError, KeyError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        print(render_next(task, model_id=args.model_id))
        return 0

    if args.autofill:
        try:
            actions = autofill_stubs(
                model_id=args.model_id,
                overwrite=args.overwrite_autofill,
                op_synth=bool(getattr(args, "op_synth", False)),
            )
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        print("Phase-1 torch-fallback autofill:")
        for name, action in actions:
            print(f"  [{action:25s}]  {name}")
        if not actions:
            print("  (no NEW stubs to autofill — model has nothing missing)")
        print(
            "\nThe demo now runs the HF reference module on host CPU for these "
            "components; the TT device path is exercised end-to-end. Replace "
            "each `_stubs/<component>.py` body with TTNN ops to move that "
            "component onto TT hardware (Phase 2)."
        )
        return 0

    if args.list_synth_targets:
        only = [args.synthesize_component] if args.synthesize_component else None
        try:
            targets = list_synth_targets(model_id=args.model_id, only=only)
        except LLMError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        print(render_synth_targets(targets))
        return 0

    if args.emit_prompts:
        only = [args.synthesize_component] if args.synthesize_component else None
        try:
            written = emit_prompts(
                model_id=args.model_id,
                only=only,
                fetch_upstream=not args.no_fetch_upstream,
            )
        except LLMError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        if not written:
            print("No NEW components to emit prompts for.")
            return 0
        print(f"Wrote {len(written)} self-contained prompt file(s):")
        for name, path in written:
            print(f"  {name:25s}  {path}")
        print()
        example_name = written[0][0]
        example_response = f"_synth_responses/{example_name}.py"
        print(
            "Next steps (no API key required):\n"
            "  1. Open each .prompt.md and paste the whole `## Prompt` block\n"
            "     into your chat assistant (e.g. the Cursor chat).\n"
            "  2. Save the assistant's response to a .py file under the demo's\n"
            "     `_synth_responses/` folder, e.g.:\n"
            f"       {example_response}\n"
            "  3. Apply it (concrete example using the first component):\n"
            f"       python -m scripts.tt_hw_planner bringup {args.model_id} \\\n"
            f"           --apply-response {example_name} {example_response}\n"
            "  4. Validate via PCC:\n"
            f"       python -m scripts.tt_hw_planner bringup {args.model_id} \\\n"
            f"           --run-tests --component {example_name}\n"
            "\n"
            "  (Names of all NEW components above are real values you can use\n"
            "  directly — no placeholder substitution needed.)\n"
        )
        return 0

    if args.handoff_to_chat:
        only = [args.synthesize_component] if args.synthesize_component else None
        try:
            master_path = build_handoff_master(
                model_id=args.model_id,
                only=only,
                fetch_upstream=not args.no_fetch_upstream,
            )
        except LLMError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        if master_path is None:
            print("No NEW components to hand off — nothing for the chat agent to write.")
            return 0

        if not getattr(args, "quiet_handoff", False):
            rel = master_path.relative_to(Path.cwd()) if master_path.is_absolute() else master_path
            print(f"Wrote master handoff prompt for `{args.model_id}`:")
            print(f"  {master_path}")
            print()
            print("Next steps (this is the WHOLE workflow — two commands total):")
            print()
            print("  1. Open Cursor chat (Ctrl/Cmd+L) and paste exactly this one line:")
            print()
            print(f"        @{rel} please write all the files this asks for")
            print()
            print("     The chat agent will read the file and write one Python file")
            print("     per NEW component into the demo's `_synth_responses/` folder.")
            print()
            print("  2. After the chat finishes, bulk-install everything it wrote:")
            print()
            print(f"        python -m scripts.tt_hw_planner bringup {args.model_id} --apply-all-responses")
            print()
        return 0

    if args.apply_all_responses:
        try:
            results = apply_all_responses(model_id=args.model_id)
        except LLMError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        print(f"apply-all-responses for `{args.model_id}`: {len(results)} file(s) processed")
        for r in results:
            print(f"  [{r.status:25s}]  {r.component}")
            if r.stub_path:
                print(f"      stub:    {r.stub_path}")
            if r.backup_path:
                print(f"      backup:  {r.backup_path}")
            if r.note:
                print(f"      note:    {r.note}")
        any_failed = any(
            r.status in ("syntax-error", "empty", "error", "recursion-trap", "signature-collision") for r in results
        )
        return 0 if not any_failed else 3

    if args.apply_response:
        component, response_file = args.apply_response
        try:
            result = apply_response(
                model_id=args.model_id,
                component_name=component,
                response_path=Path(response_file),
            )
        except LLMError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        print(f"apply-response: {result.status}")
        print(f"  component: {result.component}")
        if result.stub_path:
            print(f"  stub:      {result.stub_path}")
        if result.backup_path:
            print(f"  backup:    {result.backup_path}")
        if result.note:
            print(f"  note:      {result.note}")
        return 0 if result.status == "applied" else 3

    if args.synthesize:
        only = [args.synthesize_component] if args.synthesize_component else None

        try:
            targets = list_synth_targets(model_id=args.model_id, only=only)
        except LLMError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

        print(render_synth_targets(targets))
        print()

        if targets.refused and args.synthesize_component:
            print(
                "ERROR: the component you asked for is not NEW. The LLM is "
                "only allowed to write modules that scaffold marked NEW. "
                "Aborting before any API call.",
                file=sys.stderr,
            )
            return 2

        if not targets.new:
            print("Nothing to synthesize. Exiting before contacting the LLM.")
            return 0

        try:
            cfg = resolve_llm_config(
                provider=args.llm_provider,
                model=args.llm_model,
                endpoint=args.llm_endpoint,
            )
        except LLMError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

        print(
            f"Contacting LLM ({cfg.provider}/{cfg.model}) to synthesize "
            f"{len(targets.new)} NEW component(s)" + (" — DRY-RUN, no API call" if args.llm_dry_run else "") + "..."
        )
        print()

        try:
            if args.synthesize_component:
                res = synthesize_component(
                    model_id=args.model_id,
                    component_name=args.synthesize_component,
                    cfg=cfg,
                    run_tests=args.run_tests,
                    max_retries=args.llm_max_retries,
                    dry_run=args.llm_dry_run,
                    fetch_upstream=not args.no_fetch_upstream,
                )
                results = [res]
            else:
                results = synthesize_all_new(
                    model_id=args.model_id,
                    cfg=cfg,
                    run_tests=args.run_tests,
                    max_retries=args.llm_max_retries,
                    dry_run=args.llm_dry_run,
                    fetch_upstream=not args.no_fetch_upstream,
                )
        except LLMError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

        if args.format == "json":
            print(render_synth_json(results))
        else:
            print(render_synth_results(results, model_id=args.model_id))
        any_failed = any(r.final_status in ("failed", "syntax-error", "error") for r in results)
        return 0 if not any_failed else 3

    try:
        result = run_bringup_loop(
            model_id=args.model_id,
            emit_tests=not args.no_emit_tests,
            run_tests=args.run_tests,
            remove_passing_stubs=not args.keep_passing_stubs,
            overwrite_tests=args.overwrite_tests,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.format == "json":
        print(render_bringup_loop_json(result))
    else:
        print(render_bringup_loop_text(result))
    return 0
